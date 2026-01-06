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


# NHL aliases (extend if needed)
TEAM_ALIASES = {
    "ny rangers": "new york rangers",
    "ny islanders": "new york islanders",
    "nj devils": "new jersey devils",
    "la kings": "los angeles kings",
    "st louis blues": "st louis blues",
    "vegas knights": "vegas golden knights",
}

NHL_TEAM_ABBREV = {
    "anaheim ducks": "ana",
    "arizona coyotes": "ari",
    "boston bruins": "bos",
    "buffalo sabres": "buf",
    "calgary flames": "cgy",
    "carolina hurricanes": "car",
    "chicago blackhawks": "chi",
    "colorado avalanche": "col",
    "columbus blue jackets": "cbj",
    "dallas stars": "dal",
    "detroit red wings": "det",
    "edmonton oilers": "edm",
    "florida panthers": "fla",
    "los angeles kings": "lak",
    "minnesota wild": "min",
    "montreal canadiens": "mtl",
    "nashville predators": "nsh",
    "new jersey devils": "njd",
    "new york islanders": "nyi",
    "new york rangers": "nyr",
    "ottawa senators": "ott",
    "philadelphia flyers": "phi",
    "pittsburgh penguins": "pit",
    "san jose sharks": "sjs",
    "seattle kraken": "sea",
    "st louis blues": "stl",
    "tampa bay lightning": "tbl",
    "toronto maple leafs": "tor",
    "vancouver canucks": "van",
    "vegas golden knights": "vgk",
    "washington capitals": "wsh",
    "winnipeg jets": "wpg",
}

def odds_team_to_abbrev(team_obj):
    if not team_obj:
        return ""
    name = team_obj.get("fullName") or ""
    name = name.lower()
    name = re.sub(r"[^a-z ]+", "", name).strip()
    return NHL_TEAM_ABBREV.get(name, "")


def apply_alias(name: str) -> str:
    n = norm_team(name)
    return TEAM_ALIASES.get(n, n)


def parse_date_to_ymd(dt_str: str) -> str:
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
        return s[:10]


def pick_book_line(odds_list, book_preference, which_line):
    """
    odds_list: list like
      [{'sportsbook': 'fanduel', 'openingLine': {...}, 'currentLine': {...}}, ...]
    """
    if not isinstance(odds_list, list):
        return (None, None, None)

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
    ap.add_argument("--db", required=True, help="Path to nhl_scrape.sqlite")
    ap.add_argument("--odds-json", required=True, help="Path to nhl_odds.json")
    ap.add_argument("--out", default="nhl_odds.csv")
    ap.add_argument("--seasons", default="2023-2025")
    ap.add_argument("--which-line", choices=["currentLine", "openingLine"], default="currentLine")
    ap.add_argument("--book", default="fanduel",
                    help="Preferred sportsbook(s), comma-separated, lowercase")
    args = ap.parse_args()

    # seasons
    if "-" in args.seasons:
        a, b = args.seasons.split("-", 1)
        seasons = list(range(int(a), int(b) + 1))
    else:
        seasons = [int(args.seasons)]

    book_pref = [b.strip().lower() for b in args.book.split(",") if b.strip()]

    # -------------------------------
    # 1) Load games table
    # -------------------------------
    conn = sqlite3.connect(args.db)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(games)").fetchall()]
    cols_l = {c.lower(): c for c in cols}

    def need(*names):
        for n in names:
            if n.lower() in cols_l:
                return cols_l[n.lower()]
        raise SystemExit(f"games table missing column (tried {names}). Found: {cols}")

    c_game_id = need("game_id", "gamepk")
    c_season  = need("season")
    c_date    = need("game_date", "gamedate", "date")
    c_home_nm = need("home_name", "hometeamname", "home_team_name")
    c_away_nm = need("away_name", "awayteamname", "away_team_name")
    c_home_ab = need("home_abbrev", "homeAbbrev")
    c_away_ab = need("away_abbrev", "awayAbbrev")

    df_games = pd.read_sql(f"""
        SELECT {c_game_id} AS game_id,
            {c_date}    AS game_date,
            {c_home_ab} AS home_abbrev,
            {c_away_ab} AS away_abbrev
        FROM games
    """, conn)

    conn.close()

    df_games["date_ymd"] = df_games["game_date"].map(parse_date_to_ymd)
    df_games["home_abbrev"] = df_games["home_abbrev"].str.lower()
    df_games["away_abbrev"] = df_games["away_abbrev"].str.lower()

    print(df_games.head())
    print(df_games.columns.tolist())


    key_to_game_id = {
        (r.date_ymd, r.home_abbrev, r.away_abbrev): r.game_id
        for r in df_games.itertuples(index=False)
    }

    print(f"[games] loaded {len(df_games)} games for seasons {seasons[0]}–{seasons[-1]}")

    # -------------------------------
    # 2) Load odds JSON
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
            away = odds_team_to_abbrev(gv.get("awayTeam"))
            home = odds_team_to_abbrev(gv.get("homeTeam"))

            game_id = key_to_game_id.get((date_ymd, home, away))
            if game_id is None:
                misses += 1
                continue

            odds = (g.get("odds") or {}).get("moneyline")
            book, home_odds, away_odds = pick_book_line(
                odds, book_pref, args.which_line
            )
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
