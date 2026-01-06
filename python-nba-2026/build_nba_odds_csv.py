#!/usr/bin/env python3
import argparse
import json
import re
import sqlite3
from datetime import datetime, timezone

import pandas as pd


def norm_team(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def parse_date_to_ymd(dt_str: str) -> str:
    if not dt_str:
        return ""
    s = str(dt_str).strip()
    try:
        if s.endswith("Z"):
            s = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        dt = dt.astimezone(timezone.utc)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return s[:10]


def odds_team_to_abbrev(team_obj) -> str:
    """
    SBR team objects typically include shortName (e.g., BOS, LAL) and/or fullName.
    Prefer shortName when available.
    """
    if not team_obj:
        return ""
    if isinstance(team_obj, str):
        # already an abbrev or name
        t = team_obj.strip()
        if len(t) in (2, 3, 4):
            return t.lower()
        return norm_team(t)

    short = (team_obj.get("shortName") or team_obj.get("abbrev") or team_obj.get("abbr") or "").strip()
    if short:
        return short.lower()

    full = team_obj.get("fullName") or team_obj.get("name") or team_obj.get("displayName") or ""
    return norm_team(full)


def pick_book_line(odds_list, book_preference, which_line):
    """
    odds_list: list like
      [{'sportsbook': 'fanduel', 'openingLine': {...}, 'currentLine': {...}}, ...]
    Returns (book, homeOdds, awayOdds) using preference and line selection.
    """
    if not isinstance(odds_list, list):
        return (None, None, None)

    by_book = {}
    for entry in odds_list:
        b = (entry.get("sportsbook") or "").lower()
        if b:
            by_book[b] = entry

    # 1) preferred books
    for b in book_preference:
        entry = by_book.get(b)
        if not entry:
            continue
        line = entry.get(which_line) or {}
        home = line.get("homeOdds")
        away = line.get("awayOdds")
        if home is not None and away is not None:
            return (b, home, away)

    # 2) first complete line
    for entry in odds_list:
        b = (entry.get("sportsbook") or "").lower() or None
        line = entry.get(which_line) or {}
        home = line.get("homeOdds")
        away = line.get("awayOdds")
        if home is not None and away is not None:
            return (b, home, away)

    return (None, None, None)


def main():
    ap = argparse.ArgumentParser(description="Build NBA odds CSV from SBR-style JSON (NHL builder-style).")
    ap.add_argument("--db", required=True, help="Path to your nba sqlite DB (must contain a 'games' table).")
    ap.add_argument("--odds-json", required=True, help="Path to nba odds json produced by get_nba_odds_fixed.py")
    ap.add_argument("--out", default="nba_odds.csv", help="Output CSV path")
    ap.add_argument("--seasons", default=None, help="Optional season filter, e.g. 2023-2025 or 2024")
    ap.add_argument("--which-line", choices=["currentLine", "openingLine"], default="currentLine")
    ap.add_argument("--book", default="fanduel", help="Preferred sportsbook(s), comma-separated, lowercase")
    ap.add_argument("--table", default="games", help="Games table name (default: games)")
    args = ap.parse_args()

    book_pref = [b.strip().lower() for b in args.book.split(",") if b.strip()]

    # -------------------------------
    # 1) Load games table
    # -------------------------------
    conn = sqlite3.connect(args.db)
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({args.table})").fetchall()]
    if not cols:
        raise SystemExit(f"Table '{args.table}' not found or has no columns in {args.db}")

    cols_l = {c.lower(): c for c in cols}

    def need(*names):
        for n in names:
            if n.lower() in cols_l:
                return cols_l[n.lower()]
        raise SystemExit(f"{args.table} missing column (tried {names}). Found: {cols}")

    c_game_id = need("game_id", "gamepk", "id", "gameId")
    c_season  = cols_l.get("season")  # optional
    c_date    = need("game_date", "gamedate", "date", "start_date", "startdate")
    c_home_ab = None
    c_away_ab = None

    # Prefer abbrev columns if present; else fall back to name columns and normalize.
    for cand in ("home_abbrev", "homeabbrev", "home_abbr", "homeabbr"):
        if cand in cols_l:
            c_home_ab = cols_l[cand]
            break
    for cand in ("away_abbrev", "awayabbrev", "away_abbr", "awayabbr"):
        if cand in cols_l:
            c_away_ab = cols_l[cand]
            break

    c_home_nm = None
    c_away_nm = None
    for cand in ("home_name", "hometeamname", "home_team_name", "home", "homeTeam"):
        if cand in cols_l:
            c_home_nm = cols_l[cand]
            break
    for cand in ("away_name", "awayteamname", "away_team_name", "away", "awayTeam"):
        if cand in cols_l:
            c_away_nm = cols_l[cand]
            break

    select_cols = [f"{c_game_id} AS game_id", f"{c_date} AS game_date"]
    if c_season:
        select_cols.append(f"{c_season} AS season")
    if c_home_ab:
        select_cols.append(f"{c_home_ab} AS home_abbrev")
    if c_away_ab:
        select_cols.append(f"{c_away_ab} AS away_abbrev")
    if (not c_home_ab) and c_home_nm:
        select_cols.append(f"{c_home_nm} AS home_name")
    if (not c_away_ab) and c_away_nm:
        select_cols.append(f"{c_away_nm} AS away_name")

    df_games = pd.read_sql(f"SELECT {', '.join(select_cols)} FROM {args.table}", conn)
    conn.close()

    df_games["date_ymd"] = df_games["game_date"].map(parse_date_to_ymd)

    if "home_abbrev" in df_games.columns:
        df_games["home_key"] = df_games["home_abbrev"].astype(str).str.lower()
    else:
        df_games["home_key"] = df_games.get("home_name", "").map(norm_team)

    if "away_abbrev" in df_games.columns:
        df_games["away_key"] = df_games["away_abbrev"].astype(str).str.lower()
    else:
        df_games["away_key"] = df_games.get("away_name", "").map(norm_team)

    # Optional season filter
    if args.seasons and "season" in df_games.columns:
        if "-" in args.seasons:
            a, b = args.seasons.split("-", 1)
            seasons = list(range(int(a), int(b) + 1))
        else:
            seasons = [int(args.seasons)]
        df_games = df_games[df_games["season"].isin(seasons)].copy()
        print(f"[games] filtered to seasons {seasons[0]}–{seasons[-1]}: {len(df_games)} rows")
    else:
        print(f"[games] loaded {len(df_games)} rows")

    key_to_game_id = {
        (r.date_ymd, r.home_key, r.away_key): r.game_id
        for r in df_games.itertuples(index=False)
    }

    # -------------------------------
    # 2) Load odds JSON + match
    # -------------------------------
    with open(args.odds_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    out_rows = []
    hits = 0
    misses = 0

    for date_key, games in data.items():
        date_ymd = parse_date_to_ymd(date_key)

        for g in games:
            gv = g.get("gameView") or {}
            away_key = odds_team_to_abbrev(gv.get("awayTeam"))
            home_key = odds_team_to_abbrev(gv.get("homeTeam"))

            game_id = key_to_game_id.get((date_ymd, home_key, away_key))
            if game_id is None:
                misses += 1
                continue

            odds = (g.get("odds") or {}).get("moneyline")
            book, home_odds, away_odds = pick_book_line(odds, book_pref, args.which_line)
            if home_odds is None or away_odds is None:
                continue

            hits += 1
            out_rows.append({
                "game_id": game_id,
                "home_odds": float(home_odds),
                "away_odds": float(away_odds),
                "book": book,
                "which_line": args.which_line,
                "date": date_ymd,
            })

    df_out = pd.DataFrame(out_rows).drop_duplicates(subset=["game_id"], keep="last")
    df_out.to_csv(args.out, index=False)

    print(f"[match] hits={hits} unique_games={len(df_out)} misses={misses}")
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
