#!/usr/bin/env python3
"""
make_park_factors.py

Create a park factors CSV from your SQLite MLB DB (no re-scrape).

This version reads from your `schedule_games` table (seasons 2015–2025), since your
games_table_YYYY tables don't contain final scores.

Park Factor definition (runs)
----------------------------
For each (season, venueId):

PF_runs = 100 * (avg total_runs at this venue in this season) /
                (avg total_runs at all OTHER venues in this season)

where total_runs = home_score + away_score

Inputs required
--------------
From `schedule_games`:
- season (or year)
- gamePk (or game_pk or game_id)
- home score and away score columns (common names auto-detected)
- venueId OR (if missing) context_game.venueId can be joined via gamePk

Output
------
CSV columns:
season, venueId, parkFactorRuns, n_games, avg_runs_venue, avg_runs_other, season_avg_runs

Usage
-----
python make_park_factors.py --db mlb_scrape.sqlite --out park_factors.csv
python make_park_factors.py --db mlb_scrape.sqlite --seasons 2016-2025 --min-games 30
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
from typing import Dict, List, Optional, Tuple


def parse_seasons_arg(s: str) -> List[int]:
    out: List[int] = []
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        if "-" in p:
            a, b = p.split("-", 1)
            a, b = int(a), int(b)
            if b < a:
                a, b = b, a
            out.extend(list(range(a, b + 1)))
        else:
            out.append(int(p))
    return sorted(set(out))


def table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    return [r[1] for r in cur.fetchall()]


def pick_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def has_table(conn: sqlite3.Connection, name: str) -> bool:
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,))
    return cur.fetchone() is not None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="mlb_scrape.sqlite", help="SQLite DB path")
    ap.add_argument("--out", default="park_factors.csv", help="Output CSV path")
    ap.add_argument("--seasons", default=None, help="e.g. '2016-2025' or '2019,2021-2023'. Default: all in schedule_games.")
    ap.add_argument("--min-games", type=int, default=20, help="Minimum games at venue in season to emit PF")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    if not has_table(conn, "schedule_games"):
        raise SystemExit("Missing table: schedule_games")

    cols = table_columns(conn, "schedule_games")

    season_col = pick_col(cols, ["season", "year"])
    gamepk_col = pick_col(cols, ["gamePk", "game_pk", "game_id"])
    home_score_col = pick_col(cols, ["home_score", "homeScore", "home_runs", "homeRuns", "homeR", "home_team_score", "homeScoreFinal"])
    away_score_col = pick_col(cols, ["away_score", "awayScore", "away_runs", "awayRuns", "awayR", "away_team_score", "awayScoreFinal"])
    venue_col = pick_col(cols, ["venueId", "venue_id", "parkId", "venue"])

    if not season_col or not gamepk_col:
        raise SystemExit("schedule_games must contain season/year and gamePk/game_id columns.")

    if not home_score_col or not away_score_col:
        raise SystemExit(
            "Could not find home/away score columns in schedule_games. "
            "If your columns have different names, tell me the exact column names from PRAGMA table_info(schedule_games)."
        )

    # If schedule_games lacks venueId, we will join context_game on gamePk
    join_context = False
    if not venue_col:
        if not has_table(conn, "context_game"):
            raise SystemExit(
                "schedule_games has no venueId and context_game is missing. "
                "Run build_context_features.py --fill-starters first to populate context_game.venueId."
            )
        # verify context_game has venueId
        ccols = table_columns(conn, "context_game")
        if not pick_col(ccols, ["venueId", "venue_id"]):
            raise SystemExit("context_game exists but has no venueId column?")
        join_context = True

    # Seasons to process
    if args.seasons:
        seasons = parse_seasons_arg(args.seasons)
    else:
        q = f"SELECT DISTINCT {season_col} AS season FROM schedule_games ORDER BY season"
        seasons = [int(r["season"]) for r in conn.execute(q).fetchall() if r["season"] is not None]

    rows_out: List[dict] = []

    for season in seasons:
        if join_context:
            sql = f"""
            SELECT s.{season_col} AS season,
                   s.{gamepk_col} AS gamePk,
                   (s.{home_score_col} + s.{away_score_col}) AS total_runs,
                   c.venueId AS venueId
            FROM schedule_games s
            JOIN context_game c
              ON c.gamePk = s.{gamepk_col}
            WHERE s.{season_col} = ?
              AND c.venueId IS NOT NULL
              AND s.{home_score_col} IS NOT NULL AND s.{away_score_col} IS NOT NULL
            """
        else:
            sql = f"""
            SELECT {season_col} AS season,
                   {gamepk_col} AS gamePk,
                   ({home_score_col} + {away_score_col}) AS total_runs,
                   {venue_col} AS venueId
            FROM schedule_games
            WHERE {season_col} = ?
              AND {venue_col} IS NOT NULL
              AND {home_score_col} IS NOT NULL AND {away_score_col} IS NOT NULL
            """

        data = conn.execute(sql, (season,)).fetchall()
        if not data:
            print(f"[warn] season {season}: no rows with venueId+scores; skipping.")
            continue

        total_runs_all = [float(r["total_runs"]) for r in data if r["total_runs"] is not None]
        if not total_runs_all:
            print(f"[warn] season {season}: no total_runs; skipping.")
            continue

        sum_all = sum(total_runs_all)
        n_all = len(total_runs_all)
        season_avg = sum_all / n_all

        by_venue: Dict[int, List[float]] = {}
        for r in data:
            vid = r["venueId"]
            tr = r["total_runs"]
            if vid is None or tr is None:
                continue
            by_venue.setdefault(int(vid), []).append(float(tr))

        season_rows = 0
        for vid, trs in by_venue.items():
            n = len(trs)
            if n < args.min_games:
                continue
            sum_venue = sum(trs)
            if n_all - n <= 0:
                continue
            avg_venue = sum_venue / n
            avg_other = (sum_all - sum_venue) / (n_all - n)
            if avg_other <= 0:
                continue
            pf = 100.0 * (avg_venue / avg_other)
            rows_out.append({
                "season": int(season),
                "venueId": int(vid),
                "parkFactorRuns": round(pf, 3),
                "n_games": int(n),
                "avg_runs_venue": round(avg_venue, 3),
                "avg_runs_other": round(avg_other, 3),
                "season_avg_runs": round(season_avg, 3),
            })
            season_rows += 1

        print(f"[ok] season {season}: venues={len(by_venue)} emitted={season_rows}")

    rows_out.sort(key=lambda r: (r["season"], r["venueId"]))
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "season", "venueId", "parkFactorRuns",
            "n_games", "avg_runs_venue", "avg_runs_other", "season_avg_runs"
        ])
        w.writeheader()
        w.writerows(rows_out)

    print(f"[done] wrote {len(rows_out)} rows to {args.out}")


if __name__ == "__main__":
    main()
