#!/usr/bin/env python3
"""
Build an NBA odds CSV from an SBR odds JSON file (same shape as nhl_odds.json),
mapping games to rows in an existing SQLite table that contains:

  - Date column (YYYY-MM-DD)
  - Home team column (e.g., TEAM_NAME)
  - Away team column (e.g., TEAM_NAME.1)

Unlike the NHL version, this does NOT require a gameId. It produces a CSV you can
merge back onto your dataset using (Date, home team, away team) or row_id.

Example:
  python build_nba_odds_csv_from_json.py \
    --db Date/dataset.sqlite \
    --table dataset_2012P \
    --date-col Date \
    --home-col TEAM_NAME \
    --away-col TEAM_NAME.1 \
    --odds-json nba_moneylines.json \
    --out nba_odds.csv \
    --book fanduel,betmgm \
    --which-line currentLine
"""

import argparse
import json
import re
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def norm_team(s: str) -> str:
    """Normalize team strings for robust matching."""
    if s is None:
        return ""
    s = str(s).strip().lower()
    # normalize common punctuation
    s = s.replace("&", "and")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# Optional aliases (add if you run into mismatches)
TEAM_ALIASES = {
    # "la clippers": "los angeles clippers",
    # "la lakers": "los angeles lakers",
    # "ny knicks": "new york knicks",
    # "okc thunder": "oklahoma city thunder",
}


def apply_alias(name: str) -> str:
    n = norm_team(name)
    return TEAM_ALIASES.get(n, n)


def team_name_from_gameview(team_obj: Any) -> str:
    """
    Extract best-available team name from SBR gameView team object.
    Prefer fullName (usually 'Boston Celtics'), fallback to name/displayName/shortName.
    """
    if not team_obj:
        return ""
    if isinstance(team_obj, str):
        return team_obj
    if isinstance(team_obj, dict):
        return (
            team_obj.get("fullName")
            or team_obj.get("name")
            or team_obj.get("displayName")
            or team_obj.get("shortName")
            or ""
        )
    return ""


def pick_book_line(
    odds_list: Optional[List[Dict[str, Any]]],
    book_pref: List[str],
    which_line: str,
) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    """
    Select a sportsbook entry based on preference order and return:
      (book, home_odds, away_odds)
    """
    if not odds_list:
        return (None, None, None)

    # index available entries by book
    by_book = {}
    for entry in odds_list:
        b = (entry.get("sportsbook") or "").strip().lower()
        if b:
            by_book[b] = entry

    # try preferred books in order
    for b in book_pref:
        entry = by_book.get(b)
        if not entry:
            continue
        line = entry.get(which_line) or {}
        home = line.get("homeOdds")
        away = line.get("awayOdds")
        if home is not None and away is not None:
            return (b, float(home), float(away))

    # fallback: first entry that has the requested line
    for entry in odds_list:
        b = (entry.get("sportsbook") or "").strip().lower() or None
        line = entry.get(which_line) or {}
        home = line.get("homeOdds")
        away = line.get("awayOdds")
        if home is not None and away is not None:
            return (b, float(home), float(away))

    return (None, None, None)


def qident(col: str) -> str:
    """Quote a SQLite identifier safely (handles dots like TEAM_NAME.1)."""
    # double-up embedded quotes
    col = col.replace('"', '""')
    return f'"{col}"'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="Path to your NBA SQLite database")
    ap.add_argument("--table", required=True, help="Table containing Date/home/away columns")
    ap.add_argument("--date-col", default="Date", help="Date column name (YYYY-MM-DD)")
    ap.add_argument("--home-col", default="TEAM_NAME", help="Home team column name")
    ap.add_argument("--away-col", default="TEAM_NAME.1", help="Away team column name")
    ap.add_argument("--odds-json", required=True, help="Path to NBA odds JSON (nhl_odds.json shape)")
    ap.add_argument("--out", default="nba_odds.csv", help="Output CSV")
    ap.add_argument("--which-line", choices=["currentLine", "openingLine"], default="currentLine")
    ap.add_argument(
        "--book",
        default="fanduel",
        help="Comma-separated preference list (e.g., 'fanduel,betmgm').",
    )
    ap.add_argument(
        "--keep-all-books",
        action="store_true",
        help="If set, output one row per book per game (ignores --book preference).",
    )

    args = ap.parse_args()
    book_pref = [b.strip().lower() for b in args.book.split(",") if b.strip()]

    # -------------------------------
    # 1) Load keys from DB table
    # -------------------------------
    con = sqlite3.connect(args.db)

    sql = f"""
    SELECT
      rowid AS row_id,
      {qident(args.date_col)} AS date_ymd,
      {qident(args.home_col)} AS home_team,
      {qident(args.away_col)} AS away_team
    FROM {qident(args.table)}
    WHERE {qident(args.date_col)} IS NOT NULL
      AND {qident(args.home_col)} IS NOT NULL
      AND {qident(args.away_col)} IS NOT NULL
    """
    df = pd.read_sql_query(sql, con)
    con.close()

    df["date_ymd"] = df["date_ymd"].astype(str).str.slice(0, 10)
    df["home_norm"] = df["home_team"].map(lambda x: apply_alias(x))
    df["away_norm"] = df["away_team"].map(lambda x: apply_alias(x))

    # (date, home_norm, away_norm) -> row_id (first seen)
    key_to_row = {}
    for r in df.itertuples(index=False):
        k = (r.date_ymd, r.home_norm, r.away_norm)
        if k not in key_to_row:
            key_to_row[k] = int(r.row_id)

    print(f"[db] loaded {len(df)} rows from {args.table} (unique keys={len(key_to_row)})")

    # -------------------------------
    # 2) Load odds JSON
    # -------------------------------
    with open(args.odds_json, "r", encoding="utf-8") as f:
        odds_by_date = json.load(f)

    out_rows: List[Dict[str, Any]] = []
    hits = 0
    misses = 0
    games_seen = 0

    for date_ymd, games in odds_by_date.items():
        if not isinstance(games, list):
            continue
        for g in games:
            games_seen += 1
            gv = g.get("gameView") or {}

            home_name = team_name_from_gameview(gv.get("homeTeam"))
            away_name = team_name_from_gameview(gv.get("awayTeam"))

            home_norm = apply_alias(home_name)
            away_norm = apply_alias(away_name)

            k = (date_ymd, home_norm, away_norm)
            row_id = key_to_row.get(k)

            if row_id is None:
                misses += 1
                continue

            moneyline = (g.get("odds") or {}).get("moneyline") or []
            if not moneyline:
                continue

            if args.keep_all_books:
                # One row per sportsbook entry
                for entry in moneyline:
                    book = (entry.get("sportsbook") or "").strip().lower()
                    line = entry.get(args.which_line) or {}
                    home_odds = line.get("homeOdds")
                    away_odds = line.get("awayOdds")
                    if home_odds is None or away_odds is None:
                        continue
                    hits += 1
                    out_rows.append(
                        {
                            "row_id": row_id,
                            "date": date_ymd,
                            "home_team": home_name,
                            "away_team": away_name,
                            "home_odds": float(home_odds),
                            "away_odds": float(away_odds),
                            "book": book,
                            "which_line": args.which_line,
                        }
                    )
            else:
                book, home_odds, away_odds = pick_book_line(moneyline, book_pref, args.which_line)
                if home_odds is None or away_odds is None:
                    continue
                hits += 1
                out_rows.append(
                    {
                        "row_id": row_id,
                        "date": date_ymd,
                        "home_team": home_name,
                        "away_team": away_name,
                        "home_odds": float(home_odds),
                        "away_odds": float(away_odds),
                        "book": book,
                        "which_line": args.which_line,
                    }
                )

    if not out_rows:
        print(f"[match] hits=0 misses={misses} games_seen={games_seen}")
        print("No rows written. If this is unexpected, you likely need TEAM_ALIASES tweaks.")
        # still write empty file with headers
        pd.DataFrame(columns=["row_id","date","home_team","away_team","home_odds","away_odds","book","which_line"]).to_csv(args.out, index=False)
        print(f"[done] wrote empty {args.out}")
        return

    df_out = pd.DataFrame(out_rows)

    # If not keeping all books, de-dupe by row_id (keep last)
    if not args.keep_all_books:
        df_out = df_out.drop_duplicates(subset=["row_id"], keep="last")

    df_out.to_csv(args.out, index=False)

    print(f"[match] hits={hits} rows_out={len(df_out)} misses={misses} games_seen={games_seen}")
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()

# python build_nba_odds_csv_from_json.py \
#     --db Data/dataset.sqlite \
#     --table dataset_2012P \
#     --date-col Date \
#     --home-col TEAM_NAME \
#     --away-col TEAM_NAME.1 \
#     --odds-json nba_moneylines_2023_2025.json \
#     --out nba_odds.csv \
#     --book fanduel,betmgm \
#     --which-line currentLine