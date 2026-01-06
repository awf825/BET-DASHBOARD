#!/usr/bin/env python3
"""
nhl_daily_edges.py

Daily production helper:
1) Pull today's NHL games + moneyline odds via Scoreboard API (user-provided).
2) Build a feature row for each matchup using latest rolling stats from games_table_20252026.
3) Score with a trained model (sklearn Pipeline recommended) and compute edges/kelly stakes
   using a "frozen" strategy (defaults match your current notebook).

Assumptions / design choices:
- games_table_YYYYYYYY is a *wide per-game* table with columns like:
    gamePk/game_id, gameDate/game_date, home_abbrev, away_abbrev, homeWin,
    plus feature columns prefixed with home_/away_.
- To create features for an *upcoming* game, we use each team's most recent available
  feature snapshot from their latest completed game in that same season table.

Usage example:
  python nhl_daily_edges.py \
    --db nhl_scrape.sqlite \
    --table games_table_20252026 \
    --model model_logit.joblib \
    --bankroll 5000 \
    --out picks_today.csv
"""

import argparse
import json
import sqlite3
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sbrscrape import Scoreboard
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None  # type: ignore

ABBREV_ALIASES = {
    # scoreboard -> DB
    "CLB": "CBJ",
    "TB":  "TBL",
    "LA":  "LAK",
    "SJ":  "SJS",
    "MON": "MTL",
    "CAL": "CGY",
    "NAS": "NSH",
    "VEG": "VGK",
    # sometimes seen:
    "NJ":  "NJD",
    "WAS": "WSH",
    "WIN": "WPG",
    "UTAH": "UTA"
}

def norm_abbrev(a: str) -> str:
    a = (a or "").strip().upper()
    return ABBREV_ALIASES.get(a, a)


# ----------------------------
# Odds helpers
# ----------------------------
def american_to_decimal(odds: float) -> float:
    if odds is None or not np.isfinite(odds):
        return np.nan
    odds = float(odds)
    if odds > 0:
        return 1.0 + odds / 100.0
    return 1.0 + 100.0 / abs(odds)


def implied_prob_from_american(odds: float) -> float:
    if odds is None or not np.isfinite(odds):
        return np.nan
    odds = float(odds)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def kelly_fraction(p: float, dec_odds: float) -> float:
    if p is None or dec_odds is None:
        return 0.0
    if not (np.isfinite(p) and np.isfinite(dec_odds)):
        return 0.0
    b = float(dec_odds) - 1.0
    if b <= 0:
        return 0.0
    f = (float(p) * b - (1.0 - float(p))) / b
    if not np.isfinite(f):
        return 0.0
    return float(max(0.0, min(1.0, f)))


# ----------------------------
# DB helpers
# ----------------------------
def _table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return [r[1] for r in rows]


def _pick_col(cols: List[str], *candidates: str) -> str:
    cols_l = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in cols_l:
            return cols_l[cand.lower()]
    raise KeyError(f"Missing expected column. Tried {candidates}. Found: {cols}")


def _safe_team_abbrev(s: Any) -> str:
    if s is None:
        return ""
    if isinstance(s, dict):
        s = s.get("default") or ""
    return str(s).strip().lower()


def _today_ymd() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def get_latest_team_feature_vector(conn, table: str, team_abbrev: str, asof_date: str | None = None):
    """
    Works with *wide* games_table_YYYYYYYY tables that have:
      - homeTeamAbbrev_x / awayTeamAbbrev_x (or _y variants)
      - lots of rolling features prefixed with 'home_' and 'away_'

    Returns a dict/Series-like object of the team's latest available rolling features
    (prefix stripped, e.g. 'last5_GF_mean' instead of 'home_last5_GF_mean').
    """
    team = (team_abbrev or "").lower().strip()
    if not team:
        return None

    # 1) discover abbrev columns (your table has *_x / *_y)
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    cols_set = set(cols)

    def first_present(*cands):
        for c in cands:
            if c in cols_set:
                return c
        return None

    # c_home = first_present("homeTeamAbbrev_x", "homeTeamAbbrev_y", "homeTeamAbbrev")
    # c_away = first_present("awayTeamAbbrev_x", "awayTeamAbbrev_y", "awayTeamAbbrev")
    # Some builders create duplicate abbrev cols like homeTeamAbbrev_x/homeTeamAbbrev_y after merges.
    c_home = _pick_col(
        cols,
        "home_abbrev",
        "homeTeamAbbrev",
        "home_team_abbrev",
        "homeTeamAbbrev_y",
        "homeTeamAbbrev_x",
    )
    c_away = _pick_col(
        cols,
        "away_abbrev",
        "awayTeamAbbrev",
        "away_team_abbrev",
        "awayTeamAbbrev_y",
        "awayTeamAbbrev_x",
    )

    c_date = first_present("gameDate", "game_date", "date")

    if not c_home or not c_away or not c_date:
        raise KeyError(
            f"{table}: missing required abbrev/date cols. "
            f"Need homeTeamAbbrev*, awayTeamAbbrev*, gameDate. Found: {cols}"
        )

    # 2) pick feature columns (prefer *_mean/std/rank blocks; ignore IDs/names/targets)
    ignore_prefixes = ("homeTeam", "awayTeam", "game", "season")
    feature_cols = [
        c for c in cols
        if (c.startswith("home_") or c.startswith("away_"))
        and not any(c.startswith(p) for p in ignore_prefixes)
    ]
    if not feature_cols:
        raise KeyError(f"{table}: no home_/away_ feature columns found. Found cols={len(cols)}")

    select_cols = ", ".join([c_date, c_home, c_away] + feature_cols)

    # 3) query latest game row for that team (as-of optional)
    params = [team, team]
    where = f"(LOWER({c_home})=? OR LOWER({c_away})=?)"
    if asof_date:
        where += f" AND {c_date} <= ?"
        params.append(asof_date)

    q = f"""
        SELECT {select_cols}
        FROM {table}
        WHERE {where}
        ORDER BY {c_date} DESC
        LIMIT 1
    """

    row = conn.execute(q, params).fetchone()
    if row is None:
        return None

    # 4) decide whether team was home or away in that latest row
    row_dict = dict(zip([c_date, c_home, c_away] + feature_cols, row))
    is_home = (str(row_dict[c_home]).lower().strip() == team)
    prefix = "home_" if is_home else "away_"

    # 5) return ONLY the team's side features, with prefix stripped
    out = {}
    for c in feature_cols:
        if c.startswith(prefix):
            out[c[len(prefix):]] = row_dict[c]

    return out



def build_matchup_features_from_latest_snapshots(
    conn: sqlite3.Connection,
    table: str,
    home_abbrev: str,
    away_abbrev: str,
    feature_cols: List[str],
    asof_date: Optional[str] = None,
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    h = get_latest_team_feature_vector(conn, table, home_abbrev, asof_date=asof_date)
    a = get_latest_team_feature_vector(conn, table, away_abbrev, asof_date=asof_date)

    print(h,a)

    dbg = {
        "home_abbrev": _safe_team_abbrev(home_abbrev),
        "away_abbrev": _safe_team_abbrev(away_abbrev),
        "home_snapshot_found": h is not None,
        "away_snapshot_found": a is not None,
        "home_snapshot_side": None if h is None else h.get("_side"),
        "away_snapshot_side": None if a is None else a.get("_side"),
        "home_snapshot_date": None if h is None else h.get("_asof_gameDate"),
        "away_snapshot_date": None if a is None else a.get("_asof_gameDate"),
    }

    if h is None or a is None:
        return None, dbg

    row: Dict[str, Any] = {}
    for c in feature_cols:
        if c.startswith("home_"):
            base = c[len("home_"):]
            row[c] = h.get(base, np.nan)
        elif c.startswith("away_"):
            base = c[len("away_"):]
            row[c] = a.get(base, np.nan)
        else:
            row[c] = np.nan

    return pd.DataFrame([row]), dbg


# ----------------------------
# Scoreboard odds integration
# ----------------------------
@dataclass
class TodayGame:
    game_id: str
    home_abbrev: str
    away_abbrev: str
    home_odds: float
    away_odds: float
    # keep any of your existing extra fields below if you have them
    # gameDate: str = ""   (or whatever you already had)

    @staticmethod
    def empty():
        return TodayGame(
            game_id="",
            home_abbrev="",
            away_abbrev="",
            home_odds=float("nan"),
            away_odds=float("nan"),
        )

    def is_valid(self) -> bool:
        return (
            bool(self.game_id)
            and bool(self.home_abbrev)
            and bool(self.away_abbrev)
            and np.isfinite(self.home_odds)
            and np.isfinite(self.away_odds)
        )


PREFERRED_BOOK = "fanduel"

def _extract_game_fields(g: dict) -> TodayGame:
    """
    Scoreboard schema (example):
      home_team_abbr, away_team_abbr
      home_ml: {book: odds}, away_ml: {book: odds}
      (no game_id in payload; we synthesize one)
    """
    try:
        home_abbrev = (g.get("home_team_abbr") or "").upper().strip()
        away_abbrev = (g.get("away_team_abbr") or "").upper().strip()

        if not home_abbrev or not away_abbrev:
            return TodayGame.empty()

        # pick ML odds from your preferred book
        home_ml_map = g.get("home_ml") or {}
        away_ml_map = g.get("away_ml") or {}

        home_ml = home_ml_map.get(PREFERRED_BOOK)
        away_ml = away_ml_map.get(PREFERRED_BOOK)

        # fallback: first available book with both sides
        if home_ml is None or away_ml is None:
            common_books = set(home_ml_map.keys()) & set(away_ml_map.keys())
            for bk in common_books:
                if home_ml_map.get(bk) is not None and away_ml_map.get(bk) is not None:
                    home_ml = home_ml_map[bk]
                    away_ml = away_ml_map[bk]
                    break

        if home_ml is None or away_ml is None:
            return TodayGame.empty()

        # synthesize a stable "game_id" for today (date + matchup)
        date_str = (g.get("date") or "")[:10]  # 'YYYY-MM-DD'
        if not date_str:
            date_str = "unknown-date"
        game_id = f"{date_str}_{away_abbrev}@{home_abbrev}"

        return TodayGame(
            game_id=str(game_id),
            home_abbrev=home_abbrev,
            away_abbrev=away_abbrev,
            home_odds=float(home_ml),
            away_odds=float(away_ml),
        )

    except Exception:
        return TodayGame.empty()



def get_today_games_from_scoreboard(yyyy_mm_dd: str) -> List[TodayGame]:
    sb = Scoreboard(sport="NHL", date=yyyy_mm_dd)
    if not hasattr(sb, "games"):
        return []

    raw = sb.games
    if not isinstance(raw, list):
        return []

    out: List[TodayGame] = []
    for g in raw:
        if not isinstance(g, dict):
            continue
        tg = _extract_game_fields(g)
        if not tg.is_valid():
            continue

        tg.home_abbrev = norm_abbrev(tg.home_abbrev)
        tg.away_abbrev = norm_abbrev(tg.away_abbrev)

        out.append(tg)

    return out


import numpy as np
import pandas as pd

def _ensure_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Add any missing columns as NaN (so imputer can handle them)."""
    missing = [c for c in cols if c not in df.columns]
    for c in missing:
        df[c] = np.nan
    return df

def apply_notebook_feature_pipeline(df_row: pd.DataFrame, bundle) -> pd.DataFrame:
    """
    Apply the SAME feature selection / ordering as the notebook.
    Expects df_row is a 1-row DataFrame.
    Returns a 1-row DataFrame ready for model.predict_proba.
    """
    def make_diff_features(df):
        diff_cols = []
        for c in df.columns:
            if c.startswith("home_") and c.replace("home_", "away_") in df.columns:
                diff_name = c.replace("home_", "diff_")
                df[diff_name] = df[c] - df[c.replace("home_", "away_")]
                diff_cols.append(diff_name)
        return df, diff_cols

    df_row, DIFF_COLS = make_diff_features(df_row)

    df_row = df_row[DIFF_COLS].replace([np.inf, -np.inf], np.nan)

    # imputer = SimpleImputer(strategy="median")
    # scaler = StandardScaler()
    final_feature_cols = bundle["final_feature_cols"]

    # df_row is a 1-row DataFrame
    df_row = df_row.reindex(columns=final_feature_cols)

    # df_row = imputer.transform(df_row)
    # df_row = scaler.transform(df_row)


    # print(df_row)
    # X_imp = pd.DataFrame(
    #     imputer.transform(df_row),
    #     columns=df_row.columns,
    #     index=df_row.index,
    # )

    # if scaler is not None:
    #     X_final = pd.DataFrame(
    #         scaler.transform(X_imp),
    #         columns=X_imp.columns,
    #         index=X_imp.index,
    #     )
    # else:
    #     X_final = X_imp

    return df_row


# ----------------------------
# Scoring + frozen strategy
# ----------------------------
def compute_edges_and_stakes(
    df_row: pd.DataFrame,
    home_odds: float,
    away_odds: float,
    bankroll: float,
    min_edge: float,
    max_edge: float,
    kelly_scale: float,
    max_stake_frac: float,
    bundle: dict,
    edge_strategy: str = "flat",  # flat|opening|closing
) -> Dict[str, Any]:

    # Defensive: unwrap model bundle if passed accidentally
    model = bundle["model"]
    if isinstance(model, dict):
        if "model" in model:
            model = model["model"]
        else:
            raise TypeError(f"Expected fitted model, got dict keys={model.keys()}")

    if not hasattr(model, "predict_proba"):
        raise TypeError(f"Model has no predict_proba(): {type(model)}")
    
    df_row = apply_notebook_feature_pipeline(df_row, bundle)
    
    p_home = float(model.predict_proba(df_row)[:, 1][0])
    print(p_home)
    p_away = 1.0 - p_home

    home_dec = american_to_decimal(home_odds)
    away_dec = american_to_decimal(away_odds)
    home_imp = implied_prob_from_american(home_odds)
    away_imp = implied_prob_from_american(away_odds)

    ev_home = p_home * home_dec - 1.0
    ev_away = p_away * away_dec - 1.0

    side = "home" if ev_home >= ev_away else "away"
    ev = ev_home if side == "home" else ev_away

    edge_home = p_home - home_imp
    edge_away = p_away - away_imp
    edge = edge_home if side == "home" else edge_away

    dec = home_dec if side == "home" else away_dec
    p = p_home if side == "home" else p_away

    # bet = (np.isfinite(ev) and (edge >= min_edge) and (edge <= max_edge))
    is_home = side == "home"
    is_favorite = dec < 2.0

    if is_home and is_favorite:
        segment = "Home Favorite"
    elif is_home and (not is_favorite):
        segment = "Home Underdog"
    elif (not is_home) and is_favorite:
        segment = "Road Favorite"
    else:
        segment = "Road Underdog"

    seg_bounds_opening = {
        "Road Underdog": (0.03, 0.05),
        "Road Favorite": (0.02, 0.03),
        "Home Underdog": (0.03, 0.05),
        "Home Favorite": (1.00, 1.00),  # effectively disable
    }
    seg_bounds_closing = {
        "Road Favorite": (0.05, 0.08),
        "Home Underdog": (0.05, 0.08),
        "Road Underdog": (0.02, 0.03),
        "Home Favorite": (1.00, 1.00),  # effectively disable
    }

    eff_min = min_edge
    eff_max = max_edge
    if edge_strategy == "opening":
        eff_min, eff_max = seg_bounds_opening.get(segment, (min_edge, max_edge))
    elif edge_strategy == "closing":
        eff_min, eff_max = seg_bounds_closing.get(segment, (min_edge, max_edge))

    # Always respect hard caps passed via CLI
    eff_min = max(0.0, float(eff_min))
    eff_max = min(float(max_edge), float(eff_max))

    bet = (np.isfinite(ev) and (edge >= eff_min) and (edge <= eff_max))

    kf = kelly_fraction(p, dec)
    stake_frac = float(min(max_stake_frac, kelly_scale * kf)) if bet else 0.0
    stake = float(bankroll * stake_frac)

    return {
        "p_home": p_home,
        "p_away": p_away,
        "home_imp": float(home_imp),
        "away_imp": float(away_imp),
        "ev_home": float(ev_home),
        "ev_away": float(ev_away),
        "side": side,
        "edge": float(edge),
        "kelly_f": float(kf),
        "stake_frac": float(stake_frac),
        "stake": float(stake),
        "bet": bool(bet),
        "segment": segment,
        "min_edge_used": eff_min,
        "max_edge_used": eff_max
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--table", default="games_table_20252026")
    ap.add_argument("--model", required=True, help="joblib model/pipeline with predict_proba")
    ap.add_argument("--features", default=None,
                    help="Optional JSON (.json) or newline list (.txt) of feature columns. "
                         "If omitted, infer all home_/away_ columns from the table.")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (default today)")
    ap.add_argument("--bankroll", type=float, default=5000.0)

    # frozen strategy defaults
    ap.add_argument("--kelly-scale", type=float, default=0.10)
    ap.add_argument("--min-edge", type=float, default=0.05)
    ap.add_argument("--max-edge", type=float, default=0.12)
    ap.add_argument("--max-stake-frac", type=float, default=0.02)

    ap.add_argument("--out", default="nhl_picks_today.csv")
    ap.add_argument("--include-debug", action="store_true")
    ap.add_argument(
        "--edge-strategy",
        choices=["flat", "opening", "closing"],
        default="closing",
        help=(
            "How to gate bets by edge. 'flat' uses --min-edge/--max-edge globally. "
            "'opening'/'closing' use segment-based min/max edge windows."
        ),
    )

    args = ap.parse_args()

    run_date = args.date or _today_ymd()
    try:
        datetime.strptime(run_date, "%Y-%m-%d")
    except ValueError:
        raise SystemExit("--date must be YYYY-MM-DD")

    if joblib is None:
        raise SystemExit("joblib not available; install with: pip install joblib")

    bundle = joblib.load(args.model)
    # model = bundle["model"]
    # print(bundle)
    # FEATURE_COLS = bundle["meta"]["features"]

    conn = sqlite3.connect(args.db)

    if args.features:
        if args.features.lower().endswith(".json"):
            with open(args.features, "r", encoding="utf-8") as f:
                feature_cols = json.load(f)
        else:
            with open(args.features, "r", encoding="utf-8") as f:
                feature_cols = [ln.strip() for ln in f if ln.strip()]
    else:
        cols = _table_columns(conn, args.table)
        feature_cols = [c for c in cols if c.startswith("home_") or c.startswith("away_")]
        if not feature_cols:
            raise SystemExit(f"Could not infer home_/away_ feature columns from {args.table}. Provide --features.")

    games = get_today_games_from_scoreboard(run_date)
    if not games:
        print(f"No available games for {run_date}.")
        return

    out_rows: List[Dict[str, Any]] = []
    for tg in games:
        df_row, dbg = build_matchup_features_from_latest_snapshots(
            conn, args.table, tg.home_abbrev, tg.away_abbrev, feature_cols, asof_date=run_date
        )
        # print('df_row: ', df_row)
        if df_row is None:
            row = {
                "date": run_date,
                "game_id": tg.game_id,
                "home": tg.home_abbrev,
                "away": tg.away_abbrev,
                "home_odds": tg.home_odds,
                "away_odds": tg.away_odds,
                "status": "MISSING_FEATURES",
            }
            if args.include_debug:
                row.update(dbg)
            out_rows.append(row)
            continue

        scored = compute_edges_and_stakes(
            df_row=df_row,
            bundle=bundle,
            home_odds=tg.home_odds,
            away_odds=tg.away_odds,
            bankroll=args.bankroll,
            min_edge=args.min_edge,
            max_edge=args.max_edge,
            kelly_scale=args.kelly_scale,
            max_stake_frac=args.max_stake_frac,
            edge_strategy=args.edge_strategy
        )

        row = {
            "date": run_date,
            "game_id": tg.game_id,
            # "start_time": tg.start_time,
            "home": tg.home_abbrev,
            "away": tg.away_abbrev,
            "home_odds": tg.home_odds,
            "away_odds": tg.away_odds,
            "p_home": scored["p_home"],
            "p_away": scored["p_away"],
            "home_imp": scored["home_imp"],
            "away_imp": scored["away_imp"],
            "side": scored["side"],
            "edge": scored["edge"],
            "ev_home": scored["ev_home"],
            "ev_away": scored["ev_away"],
            "kelly_f": scored["kelly_f"],
            "stake_frac": scored["stake_frac"],
            "stake": scored["stake"],
            "bet": scored["bet"],
            "status": "OK",
        }
        if args.include_debug:
            row.update(dbg)
        out_rows.append(row)

    conn.close()

    df_out = pd.DataFrame(out_rows)
    if "bet" in df_out.columns:
        df_out["bet"] = df_out["bet"].fillna(False).astype(bool)
        df_out["bet_int"] = df_out["bet"].astype(float)
        df_out = df_out.sort_values(["bet_int", "edge"], ascending=[False, False]).drop(columns=["bet_int"])

    df_out.to_csv(args.out, index=False)
    print(f"Wrote {len(df_out)} rows -> {args.out}")
    if "bet" in df_out.columns:
        n_bets = int(df_out["bet"].sum())
        print(f"Eligible bets: {n_bets}/{len(df_out)}")
        if n_bets:
            print(df_out[df_out["bet"]].head(10)[["home","away","side","edge","stake","stake_frac","p_home","home_odds","away_odds"]].to_string(index=False))


if __name__ == "__main__":
    main()

# python nhl_daily_edges.py \
#   --db nhl_scrape.sqlite \
#   --table games_table_20252026 \
#   --model model_logit.joblib \
#   --edge-strategy closing \
#   --date 2026-01-01 \
#   --bankroll 5000 \
#   --out nhl_picks_today.csv