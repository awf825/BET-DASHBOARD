#!/usr/bin/env python3
"""
make_park_factors.py

Create a park factors CSV from YOUR existing SQLite MLB game-level DB.

Why this approach?
- No re-scraping seasons
- No paywalls
- Produces a simple, defensible "Runs Park Factor" per (season, venueId)
  that you can plug directly into build_context_features.py via --park-csv

Definition used (simple, stable baseline)
----------------------------------------
For each (season, venueId):

PF_runs = 100 * (avg total_runs per game at this venue in this season) /
                (avg total_runs per game in all OTHER venues in this season)

Where total_runs = home_score + away_score

Notes:
- This is a *venue-centric* park factor (not team-centric). It will still be
  somewhat confounded by team strength / schedule, but it’s usually good enough
  for an “Option C” context feature baseline. You can refine later if you want
  a multi-year smoothing or a team-adjusted method.

Prerequisites
-------------
Your DB needs:
- context_game with venueId for games, OR games_table_YYYY already containing venueId.
- game scores (home_score, away_score) in your games tables.

If venueId isn't present yet:
- run build_context_features.py first with --fill-starters (it populates venueId/name
  from the StatsAPI feed) — you do NOT need to re-scrape seasons.

Output
------
CSV with columns: season, venueId, parkFactorRuns, n_games, avg_runs_venue, avg_runs_other

Usage
-----
python make_park_factors.py --db mlb_scrape.sqlite --out park_factors.csv
python make_park_factors.py --db mlb_scrape.sqlite --seasons 2016-2025 --min-games 30
"""

from __future__ import annotations

import argparse
import csv
import re
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


def find_games_tables(conn: sqlite3.Connection) -> List[Tuple[int, str]]:
    cur = conn.cursor()
    cur.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name LIKE 'games_table_%'
        ORDER BY name
    """)
    out: List[Tuple[int, str]] = []
    for (name,) in cur.fetchall():
        m = re.match(r"^games_table_(\d{4})$", name)
        if m:
            out.append((int(m.group(1)), name))
    return out


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="mlb_scrape.sqlite", help="SQLite DB path")
    ap.add_argument("--out", default="park_factors.csv", help="Output CSV path")
    ap.add_argument("--seasons", default=None, help="e.g. '2016-2025' or '2019,2021-2023'. Default: all discovered.")
    ap.add_argument("--min-games", type=int, default=20, help="Minimum games at venue in season to emit PF")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    discovered = dict(find_games_tables(conn))
    if not discovered:
        raise SystemExit("No games_table_YYYY tables found.")

    seasons = parse_seasons_arg(args.seasons) if args.seasons else sorted(discovered.keys())

    # We will try to use venueId from context_game if present; else from games_table_YYYY.
    has_context = False
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='context_game'")
    if cur.fetchone():
        has_context = True

    rows_out: List[dict] = []

    for season in seasons:
        t = discovered.get(season)
        if not t:
            continue

        cols = table_columns(conn, t)
        gamepk_col = pick_col(cols, ["gamePk", "game_pk", "game_id"])
        hscore_col = pick_col(cols, ["home_score", "homeScore", "home_runs", "homeRuns", "homeR"])
        ascore_col = pick_col(cols, ["away_score", "awayScore", "away_runs", "awayRuns", "awayR"])
        venue_col  = pick_col(cols, ["venueId", "venue_id", "parkId"])

        if not gamepk_col or not hscore_col or not ascore_col:
            print(f"[warn] {t}: missing gamePk/home_score/away_score; skipping.")
            continue

        # Build a temp table of (gamePk, total_runs, venueId)
        if has_context:
            sql = f"""
            SELECT g.{gamepk_col} AS gamePk,
                   (g.{hscore_col} + g.{ascore_col}) AS total_runs,
                   c.venueId AS venueId
            FROM {t} g
            JOIN context_game c ON c.gamePk = g.{gamepk_col}
            WHERE c.venueId IS NOT NULL
              AND g.{hscore_col} IS NOT NULL AND g.{ascore_col} IS NOT NULL
            """
        else:
            if not venue_col:
                print(f"[warn] {t}: no venueId and no context_game; skipping.")
                continue
            sql = f"""
            SELECT {gamepk_col} AS gamePk,
                   ({hscore_col} + {ascore_col}) AS total_runs,
                   {venue_col} AS venueId
            FROM {t}
            WHERE {venue_col} IS NOT NULL
              AND {hscore_col} IS NOT NULL AND {ascore_col} IS NOT NULL
            """

        data = conn.execute(sql).fetchall()
        if not data:
            print(f"[warn] {t}: no rows with venueId+scores; skipping.")
            continue

        # Overall season avg
        total_runs_all = [float(r["total_runs"]) for r in data if r["total_runs"] is not None]
        if not total_runs_all:
            continue
        season_avg = sum(total_runs_all) / len(total_runs_all)

        # Group by venue
        by_venue: Dict[int, List[float]] = {}
        for r in data:
            vid = r["venueId"]
            tr = r["total_runs"]
            if vid is None or tr is None:
                continue
            by_venue.setdefault(int(vid), []).append(float(tr))

        # Compute PF vs "all other venues" in the same season
        for vid, trs in by_venue.items():
            n = len(trs)
            if n < args.min_games:
                continue
            avg_venue = sum(trs) / n

            # avg_other = (sum_all - sum_venue) / (n_all - n_venue)
            sum_all = sum(total_runs_all)
            n_all = len(total_runs_all)
            sum_venue = sum(trs)
            n_venue = n
            if n_all - n_venue <= 0:
                continue
            avg_other = (sum_all - sum_venue) / (n_all - n_venue)

            if avg_other <= 0:
                continue

            pf = 100.0 * (avg_venue / avg_other)

            rows_out.append({
                "season": season,
                "venueId": vid,
                "parkFactorRuns": round(pf, 3),
                "n_games": n,
                "avg_runs_venue": round(avg_venue, 3),
                "avg_runs_other": round(avg_other, 3),
                "season_avg_runs": round(season_avg, 3),
            })

        print(f"[ok] season {season}: venues={len(by_venue)} rows_out_total={len(rows_out)}")

    # Write CSV
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
