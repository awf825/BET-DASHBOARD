#!/usr/bin/env python3
import argparse
import json
import re
import sqlite3
from datetime import datetime, timezone

import pandas as pd


def norm_team(s: str) -> str:
    """Normalize team name strings for matching."""
    if s is None:
        return ""
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


# Extend this if you hit mismatches (e.g., "D-backs", etc.)
TEAM_ALIASES = {
    "dbacks": "arizona diamondbacks",
    "diamondbacks": "arizona diamondbacks",
    "la dodgers": "los angeles dodgers",
    "la angels": "los angeles angels",
    "ny yankees": "new york yankees",
    "ny mets": "new york mets",
}


def apply_alias(name: str) -> str:
    n = norm_team(name)
    return TEAM_ALIASES.get(n, n)


def parse_date_to_ymd(dt_str: str) -> str:
    """
    Accepts:
      - '2024-04-02'
      - '2024-04-02T23:10:00Z'
      - '2024-04-02T23:10:00+00:00'
    Returns 'YYYY-MM-DD' in UTC.
    """
    if not dt_str:
        return ""
    s = dt_str.strip()
    try:
        if s.endswith("Z"):
            s = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        dt = dt.astimezone(timezone.utc)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        # fallback: take first 10 chars if it looks like YYYY-MM-DD...
        return s[:10]


def pick_book_line(odds_list, book_preference, which_line):
    """
    odds_list: list like [{'sportsbook': 'fanduel', 'openingLine': {...}, 'currentLine': {...}}, ...]
    which_line: 'currentLine' or 'openingLine'
    returns (book, homeOdds, awayOdds) or (None,None,None)
    """
    if not isinstance(odds_list, list):
        return (None, None, None)

    # build dict sportsbook -> entry
    by_book = {}
    for entry in odds_list:
        b = (entry.get("sportsbook") or "").lower()
        by_book[b] = entry

    for b in book_preference:
        entry = by_book.get(b)
        if not entry:
            continue
        line = entry.get(which_line) or {}
        home = line.get("homeOdds")
        away = line.get("awayOdds")
        if home is not None and away is not None:
            return (b, home, away)

    # fallback: first with odds
    for entry in odds_list:
        b = (entry.get("sportsbook") or "").lower()
        line = entry.get(which_line) or {}
        home = line.get("homeOdds")
        away = line.get("awayOdds")
        if home is not None and away is not None:
            return (b, home, away)

    return (None, None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="Path to mlb_scrape.sqlite")
    ap.add_argument("--odds-json", required=True, help="Path to mlb_odds.json (from odds scraper)")
    ap.add_argument("--out", default="odds_2024_2025.csv")
    ap.add_argument("--seasons", default="2024-2025")
    ap.add_argument("--which-line", choices=["currentLine", "openingLine"], default="currentLine")
    ap.add_argument("--book", default="fanduel",
                    help="Preferred sportsbook (lowercase), e.g. fanduel,draftkings,bet365. "
                         "You can pass comma-separated for priority order.")
    args = ap.parse_args()

    # seasons
    if "-" in args.seasons:
        a, b = args.seasons.split("-", 1)
        seasons = list(range(int(a), int(b) + 1))
    else:
        seasons = [int(args.seasons)]

    book_pref = [b.strip().lower() for b in args.book.split(",") if b.strip()]

    # 1) Load your schedule_games (only last two years)
    conn = sqlite3.connect(args.db)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(schedule_games)").fetchall()]
    cols_l = {c.lower(): c for c in cols}

    def need(*names):
        for n in names:
            if n.lower() in cols_l:
                return cols_l[n.lower()]
        raise SystemExit(f"schedule_games missing required column (tried {names}). Found: {cols}")

    c_gamepk = need("gamePk", "game_pk", "game_id")
    c_season  = need("season", "year")
    c_date    = need("gameDate", "game_date", "date")
    c_home_nm = need("homeTeamName", "home_team_name", "homeName")
    c_away_nm = need("awayTeamName", "away_team_name", "awayName")

    seasons_list = ",".join(str(s) for s in seasons)
    df_sched = pd.read_sql(f"""
        SELECT {c_gamepk} AS gamePk,
               {c_season} AS season,
               {c_date} AS gameDate,
               {c_home_nm} AS homeTeamName,
               {c_away_nm} AS awayTeamName
        FROM schedule_games
        WHERE {c_season} IN ({seasons_list})
    """, conn)
    conn.close()

    df_sched["date_ymd"] = df_sched["gameDate"].map(parse_date_to_ymd)
    df_sched["home_norm"] = df_sched["homeTeamName"].map(apply_alias)
    df_sched["away_norm"] = df_sched["awayTeamName"].map(apply_alias)

    # index for matching
    key_to_gamepk = {}
    for r in df_sched.itertuples(index=False):
        key_to_gamepk[(r.date_ymd, r.home_norm, r.away_norm)] = int(r.gamePk)

    print(f"[schedule] loaded {len(df_sched)} games for seasons {seasons[0]}–{seasons[-1]}")

    # 2) Load odds JSON (date -> list of games)
    with open(args.odds_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    out_rows = []
    misses = 0
    hits = 0

    for date_key, games in data.items():
        date_ymd = parse_date_to_ymd(date_key)

        for g in games:
            gv = g.get("gameView") or {}
            away = apply_alias(gv.get("awayTeam", {}).get("fullName", ""))
            home = apply_alias(gv.get("homeTeam", {}).get("fullName", ""))

            gamepk = key_to_gamepk.get((date_ymd, home, away))
            if gamepk is None:
                misses += 1
                continue

            odds = (g.get("odds") or {}).get("moneyline")
            book, home_odds, away_odds = pick_book_line(odds, book_pref, args.which_line)
            if home_odds is None or away_odds is None:
                continue

            hits += 1
            out_rows.append({
                "gamePk": gamepk,
                "home_odds": float(home_odds),
                "away_odds": float(away_odds),
                "book": book,
                "which_line": args.which_line,
                "date": date_ymd,
            })

    df_out = pd.DataFrame(out_rows).drop_duplicates(subset=["gamePk"], keep="last")
    df_out.to_csv(args.out, index=False)

    print(f"[match] hits={hits} unique_gamePk={len(df_out)} misses={misses}")
    print(f"[done] wrote {args.out}")

if __name__ == "__main__":
    main()
