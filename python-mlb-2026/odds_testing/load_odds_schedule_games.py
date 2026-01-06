#!/usr/bin/env python3
"""
Add moneyline odds columns to schedule_games and populate them from a CSV.

Why CSV?
- MLB StatsAPI doesn't include betting odds.
- You can source odds from anywhere (Action Network export, Sportsbookreview, Pinnacle, etc.)
  and drop them into a simple file.
- This script only touches seasons 2024–2025 (configurable).

Expected CSV columns (choose one of these modes):

MODE A (best): keyed by gamePk
    gamePk, home_odds, away_odds

MODE B: keyed by date + team IDs
    gameDate (or date), homeTeamId, awayTeamId, home_odds, away_odds
Where gameDate can be ISO (e.g., 2024-04-02T23:10:00Z) or just YYYY-MM-DD.

Odds format: American odds (e.g., -120, +105).
"""

import argparse
import sqlite3
import pandas as pd

DEFAULT_DB = "../mlb_scrape.sqlite"

ADD_COLS = [
    ("home_odds", "REAL"),
    ("away_odds", "REAL"),
    ("odds_source", "TEXT"),
    ("odds_updated_at_utc", "TEXT"),
]

def ensure_cols(conn: sqlite3.Connection, table: str, cols):
    existing = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
    for name, ctype in cols:
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ctype}")
    conn.commit()

def detect_col(cols, *candidates):
    cl = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in cl:
            return cl[cand.lower()]
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--csv", required=True, help="Path to odds CSV")
    ap.add_argument("--source", default="import", help="Stored into odds_source")
    ap.add_argument("--seasons", default="2024-2025", help="e.g. 2024-2025 or 2025")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # seasons parsing
    if "-" in args.seasons:
        a, b = args.seasons.split("-", 1)
        seasons = list(range(int(a), int(b) + 1))
    else:
        seasons = [int(args.seasons)]

    conn = sqlite3.connect(args.db)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")

    # Ensure columns exist
    ensure_cols(conn, "schedule_games", ADD_COLS)

    # Identify key cols in schedule_games
    sg_cols = [r[1] for r in conn.execute("PRAGMA table_info(schedule_games)")]
    sg_gamepk = detect_col(sg_cols, "gamePk", "game_pk", "game_id")
    sg_season = detect_col(sg_cols, "season", "year")
    sg_gamedate = detect_col(sg_cols, "gameDate", "game_date", "date")
    sg_home_id = detect_col(sg_cols, "homeTeamId", "home_team_id", "homeId")
    sg_away_id = detect_col(sg_cols, "awayTeamId", "away_team_id", "awayId")

    if not sg_season:
        raise SystemExit("schedule_games must have a season/year column.")
    if not sg_gamepk and not (sg_gamedate and sg_home_id and sg_away_id):
        raise SystemExit(
            "Need either gamePk in schedule_games, or (gameDate + homeTeamId + awayTeamId)."
        )

    df = pd.read_csv(args.csv)

    # Normalize odds column names from the CSV
    df_cols = {c.lower(): c for c in df.columns}
    csv_gamepk = df_cols.get("gamepk")
    csv_home_odds = df_cols.get("home_odds") or df_cols.get("home_ml") or df_cols.get("home_moneyline")
    csv_away_odds = df_cols.get("away_odds") or df_cols.get("away_ml") or df_cols.get("away_moneyline")

    if csv_home_odds is None or csv_away_odds is None:
        raise SystemExit("CSV must include home_odds and away_odds (or home_ml/away_ml).")

    # Filter to requested seasons if the CSV has a season column, else we filter via schedule_games join
    csv_season = df_cols.get("season") or df_cols.get("year")

    # Decide matching mode
    mode = None
    if csv_gamepk and sg_gamepk:
        mode = "gamePk"
    else:
        # Need date + home/away team IDs in CSV
        csv_date = df_cols.get("gamedate") or df_cols.get("date")
        csv_home_id = df_cols.get("hometeamid") or df_cols.get("home_team_id") or df_cols.get("homeid")
        csv_away_id = df_cols.get("awayteamid") or df_cols.get("away_team_id") or df_cols.get("awayid")
        if not (csv_date and csv_home_id and csv_away_id):
            raise SystemExit(
                "CSV must have gamePk (preferred) OR (date/gameDate + homeTeamId + awayTeamId)."
            )
        mode = "date_teamids"

        # Normalize date to YYYY-MM-DD for matching (we'll match on date portion)
        df[csv_date] = pd.to_datetime(df[csv_date], utc=True, errors="coerce").dt.strftime("%Y-%m-%d")

    # Prepare updates
    updated = 0
    skipped = 0

    # Create a temp table for fast updates
    conn.execute("DROP TABLE IF EXISTS _tmp_odds_import;")
    conn.execute("""
        CREATE TABLE _tmp_odds_import (
            gamePk INTEGER,
            gameDateDay TEXT,
            homeTeamId INTEGER,
            awayTeamId INTEGER,
            home_odds REAL,
            away_odds REAL
        );
    """)

    if mode == "gamePk":
        tmp = pd.DataFrame({
            "gamePk": df[csv_gamepk].astype("Int64"),
            "gameDateDay": None,
            "homeTeamId": None,
            "awayTeamId": None,
            "home_odds": pd.to_numeric(df[csv_home_odds], errors="coerce"),
            "away_odds": pd.to_numeric(df[csv_away_odds], errors="coerce"),
        })
    else:
        csv_date = df_cols.get("gamedate") or df_cols.get("date")
        csv_home_id = df_cols.get("hometeamid") or df_cols.get("home_team_id") or df_cols.get("homeid")
        csv_away_id = df_cols.get("awayteamid") or df_cols.get("away_team_id") or df_cols.get("awayid")
        tmp = pd.DataFrame({
            "gamePk": pd.to_numeric(df.get(csv_gamepk), errors="coerce").astype("Int64") if csv_gamepk else None,
            "gameDateDay": df[csv_date],
            "homeTeamId": pd.to_numeric(df[csv_home_id], errors="coerce").astype("Int64"),
            "awayTeamId": pd.to_numeric(df[csv_away_id], errors="coerce").astype("Int64"),
            "home_odds": pd.to_numeric(df[csv_home_odds], errors="coerce"),
            "away_odds": pd.to_numeric(df[csv_away_odds], errors="coerce"),
        })

    # Drop rows missing required odds
    tmp = tmp.dropna(subset=["home_odds", "away_odds"])
    tmp.to_sql("_tmp_odds_import", conn, if_exists="append", index=False)

    seasons_list = ",".join(str(s) for s in seasons)

    # Update using either gamePk join or date+team join
    if mode == "gamePk":
        sql = f"""
        UPDATE schedule_games
        SET home_odds = (SELECT t.home_odds FROM _tmp_odds_import t WHERE t.gamePk = schedule_games.{sg_gamepk}),
            away_odds = (SELECT t.away_odds FROM _tmp_odds_import t WHERE t.gamePk = schedule_games.{sg_gamepk}),
            odds_source = ?,
            odds_updated_at_utc = datetime('now')
        WHERE {sg_season} IN ({seasons_list})
          AND {sg_gamepk} IN (SELECT gamePk FROM _tmp_odds_import WHERE gamePk IS NOT NULL);
        """
        if args.dry_run:
            print("[dry-run] Would run update by gamePk.")
        else:
            conn.execute(sql, (args.source,))
            conn.commit()
    else:
        # Match on date portion + team IDs
        # schedule_games.gameDate may be full timestamp; normalize to date with substr(,1,10)
        sql = f"""
        UPDATE schedule_games
        SET home_odds = (
                SELECT t.home_odds
                FROM _tmp_odds_import t
                WHERE t.gameDateDay = substr(schedule_games.{sg_gamedate}, 1, 10)
                  AND t.homeTeamId = schedule_games.{sg_home_id}
                  AND t.awayTeamId = schedule_games.{sg_away_id}
            ),
            away_odds = (
                SELECT t.away_odds
                FROM _tmp_odds_import t
                WHERE t.gameDateDay = substr(schedule_games.{sg_gamedate}, 1, 10)
                  AND t.homeTeamId = schedule_games.{sg_home_id}
                  AND t.awayTeamId = schedule_games.{sg_away_id}
            ),
            odds_source = ?,
            odds_updated_at_utc = datetime('now')
        WHERE {sg_season} IN ({seasons_list})
          AND EXISTS (
                SELECT 1
                FROM _tmp_odds_import t
                WHERE t.gameDateDay = substr(schedule_games.{sg_gamedate}, 1, 10)
                  AND t.homeTeamId = schedule_games.{sg_home_id}
                  AND t.awayTeamId = schedule_games.{sg_away_id}
            );
        """
        if args.dry_run:
            print("[dry-run] Would run update by (date, homeTeamId, awayTeamId).")
        else:
            conn.execute(sql, (args.source,))
            conn.commit()

    # Report coverage
    q = f"""
    SELECT
      {sg_season} AS season,
      COUNT(*) AS n_games,
      SUM(home_odds IS NOT NULL AND away_odds IS NOT NULL) AS n_with_odds
    FROM schedule_games
    WHERE {sg_season} IN ({seasons_list})
    GROUP BY {sg_season}
    ORDER BY {sg_season};
    """
    rep = pd.read_sql(q, conn)
    print(rep)

    conn.execute("DROP TABLE IF EXISTS _tmp_odds_import;")
    conn.commit()
    conn.close()

if __name__ == "__main__":
    main()
