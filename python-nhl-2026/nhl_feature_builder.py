#!/usr/bin/env python3
"""
nhl_feature_builder.py

Builds a wide, model-ready per-game table similar in spirit to your MLB `games_table_YYYY`:
- One row per game (home vs away)
- Target: homeWin
- Features: home_* and away_* season-to-date + last10 + last20 rolling means/stds
- Optional: league ranks for season-to-date features (computed pregame)
- Optional: goalie features (post-game goalie lines rolled forward), if you decide to keep them

ASSUMPTIONS
-----------
You have already populated an SQLite DB (default: nhl_scrape.sqlite) with:
- games(game_id PRIMARY KEY, game_date, season, home_team_id, away_team_id, home_score, away_score, ...)

and a per-team per-game boxscore table like:
- team_boxscore(game_id, team_id, is_home, goals, shots, hits, blocks, pim,
                faceoff_pct, giveaways, takeaways,
                pp_goals, pp_opps)

The companion scraper I provided (`nhl_scraper.py`) creates compatible tables:
- games
- team_boxscore
- goalie_lines (optional)

This script creates:
- games_table_{season}  (season is the starting year, e.g., 2015 => 2015-16 season)

Usage
-----
python nhl_feature_builder.py --db nhl_scrape.sqlite build --start-season 2015 --end-season 2025

"""

from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd


TEAM_METRICS = [
    # core score/volume
    ("GF", "goals_for"),
    ("GA", "goals_against"),
    ("SOGF", "shots_for"),
    ("SOGA", "shots_against"),
    ("SHOTDIFF", "shot_diff"),
    # special teams (pp/pk)
    # ("PPG", "pp_goals"),
    ("PPOPP", "pp_opps"),
    ("PPPCT", "pp_pct"),
    ("PKPCT", "pk_pct"),
    ("STI", "special_teams_index"),
    # puck / physical
    ("FO_PCT", "faceoff_pct"),
    ("HITS", "hits"),
    ("BLKS", "blocks"),
    ("PIM", "pim"),
    ("GIVE", "giveaways"),
    ("TAKE", "takeaways"),
    # derived luck-ish (approx, all strengths)
    ("SHPCT", "sh_pct"),
    ("SVPCT", "sv_pct"),
    ("PDO", "pdo_approx"),
]


def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def load_team_games(conn: sqlite3.Connection, season_start_year: int) -> pd.DataFrame:
    """
    Returns one row per TEAM per GAME for the given season_start_year.
    """
    # Identify season rows
    df = pd.read_sql(
        """
        SELECT
            g.game_id,
            g.game_date,
            g.season,

            -- team identity
            tb.team_abbrev,
            tb.team_side,

            -- game result
            g.home_score,
            g.away_score,

            -- team boxscore stats
            tb.goals,
            tb.shots,
            tb.hits,
            tb.blocks,
            tb.pim,
            tb.faceoff_pct,
            tb.giveaways,
            tb.takeaways,
            tb.pp_opps,
            tb.pp_pct,
            tb.pk_pct,
            tb.sh_goals_for
        FROM games g
        JOIN team_boxscore tb
            ON tb.game_id = g.game_id
        WHERE g.season = ?
        ORDER BY g.game_date, g.game_id, tb.team_side
        """,
        conn,
        params=(season_start_year,),
        parse_dates=["game_date"],
    )

    # add opponent stats
    # Each game has two rows (home, away). We can self-merge on game_id.
    opp = df[["game_id", "team_abbrev", "shots", "goals", "pp_pct", "pp_opps", "pk_pct"]].copy()
    opp = opp.rename(
        columns={
            "team_abbrev": "opp_team_abbrev",
            "shots": "opp_shots",
            "goals": "opp_goals",
            # "pp_goals": "opp_pp_goals",
            "pp_pct": "opp_pp_pct",
            "pp_opps": "opp_pp_opps",
            "pk_pct": "opp_pk_pct"
        }
    )

    df = df.merge(opp, on="game_id", how="left")
    df = df[df["team_abbrev"] != df["opp_team_abbrev"]].copy()

    # derived per-game team features
    df["goals_for"] = df["goals"].astype(float)
    df["goals_against"] = df["opp_goals"].astype(float)
    df["shots_for"] = df["shots"].astype(float)
    df["shots_against"] = df["opp_shots"].astype(float)
    df["shot_diff"] = df["shots_for"] - df["shots_against"]

    # PP / PK
    # df["pp_goals"] = df["pp_goals"].astype(float)
    df["pp_opps"] = df["pp_opps"].astype(float)
    # df["pp_pct"] = np.where(df["pp_opps"] > 0, df["pp_goals"] / df["pp_opps"], np.nan)
    df["pp_pct"] = df["pp_pct"].astype(float)

    # PK% from opponent PP
    # PK% = 1 - (opp_pp_goals / opp_pp_opps)
    # df["pk_pct"] = np.where(df["opp_pp_opps"] > 0, 1.0 - (df["opp_pp_goals"] / df["opp_pp_opps"]), np.nan)
    df["pk_pct"] = df["pk_pct"].astype(float)

    df["special_teams_index"] = df["pp_pct"] + df["pk_pct"]

    # shooting/save and PDO approx (all strengths, not 5v5)
    df["sh_pct"] = np.where(df["shots_for"] > 0, df["goals_for"] / df["shots_for"], np.nan)
    df["sv_pct"] = np.where(df["shots_against"] > 0, 1.0 - (df["goals_against"] / df["shots_against"]), np.nan)
    df["pdo_approx"] = 100.0 * (df["sh_pct"] + df["sv_pct"])

    # pass through simple stats
    for col in ["hits", "blocks", "pim", "giveaways", "takeaways", "faceoff_pct"]:
        df[col] = df[col].astype(float)

    # pregame ordering key: we must ensure features use only PRIOR games
    df = df.sort_values(["team_abbrev", "game_date", "game_id"]).reset_index(drop=True)
    return df


def rolling_features(
    df_team_games: pd.DataFrame,
    windows: Dict[str, Optional[int]] = None,
) -> pd.DataFrame:
    """
    Build rolling mean/std for each team and metric.
    windows:
      - {"season": None, "last10": 10, "last20": 20}
    Returns df with added columns like:
      season_GF_mean, season_GF_std, last10_GF_mean, ...
    All computed with shift(1) to prevent leakage.
    """
    if windows is None:
        windows = {"season": None, "last5": 5}

    out = df_team_games.copy()

    for prefix, w in windows.items():
        for code, metric in TEAM_METRICS:
            s = out.groupby("team_abbrev")[metric]
            prev = s.shift(1)

            if w is None:
                mean = prev.groupby(out["team_abbrev"]).expanding().mean().reset_index(level=0, drop=True)
                std  = prev.groupby(out["team_abbrev"]).expanding().std(ddof=0).reset_index(level=0, drop=True)
            else:
                mean = prev.groupby(out["team_abbrev"]).rolling(w, min_periods=max(3, w // 3)).mean().reset_index(level=0, drop=True)
                std  = prev.groupby(out["team_abbrev"]).rolling(w, min_periods=max(3, w // 3)).std(ddof=0).reset_index(level=0, drop=True)

            out[f"{prefix}_{code}_mean"] = mean
            out[f"{prefix}_{code}_std"] = std


    return out


def add_league_ranks(df_team_roll: pd.DataFrame, rank_prefix: str = "season") -> pd.DataFrame:
    """
    For each game date, rank teams by their season-to-date rolling means (pregame).
    Adds columns like season_GF_rank, season_SOGF_rank, ...
    Rank 1 = best (highest) for "good when high" metrics, and 1 = best (lowest) for GA/SOGA/PIM etc.

    This is optional because it can add many columns; it’s fast and purely derived from your own table.
    """
    out = df_team_roll.copy()

    # which metrics are "lower is better"
    lower_better = {"GA", "SOGA", "PIM"}
    for code, _metric in TEAM_METRICS:
        col = f"{rank_prefix}_{code}_mean"
        if col not in out.columns:
            continue

        # rank within each date across teams
        ascending = code in lower_better
        out[f"{rank_prefix}_{code}_rank"] = out.groupby("game_date")[col].rank(ascending=ascending, method="average")

    return out


def build_wide_games_table(conn: sqlite3.Connection, season_start_year: int, include_ranks: bool = True) -> pd.DataFrame:
    df_team = load_team_games(conn, season_start_year)
    df_roll = rolling_features(df_team)
    if include_ranks:
        df_roll = add_league_ranks(df_roll, rank_prefix="season")

    # Split back into home/away rows and pivot wide
    home = df_roll[df_roll["team_side"] == "home"].copy()
    away = df_roll[df_roll["team_side"] == "away"].copy()

    # Base game-level fields
    g = pd.read_sql(
        """
        SELECT
            game_id AS gamePk,
            season,
            game_date AS gameDate,
            home_abbrev AS homeTeamAbbrev,
            away_abbrev AS awayTeamAbbrev,
            home_name   AS homeTeamName,
            away_name   AS awayTeamName,
            CASE
                WHEN home_score > away_score THEN 1.0
                ELSE 0.0
            END AS homeWin
        FROM games
        WHERE season = ?
        ORDER BY game_date, game_id;
                """,
        conn,
        params=(season_start_year,),
        parse_dates=["gameDate"],
    )

    # Determine feature columns (all the rolling/rank columns)
    feature_cols = [c for c in df_roll.columns if re_is_feature_col(c)]

    home_small = home[["game_id", "team_abbrev"] + feature_cols].rename(columns={"game_id": "gamePk"})
    away_small = away[["game_id", "team_abbrev"] + feature_cols].rename(columns={"game_id": "gamePk"})

    home_small = home_small.add_prefix("home_")
    away_small = away_small.add_prefix("away_")

    # recover key columns renamed by prefix
    home_small = home_small.rename(columns={"home_gamePk": "gamePk", "home_team_abbrev": "homeTeamAbbrev"})
    away_small = away_small.rename(columns={"away_gamePk": "gamePk", "away_team_abbrev": "awayTeamAbbrev"})

    wide = g.merge(home_small, on="gamePk", how="left").merge(away_small, on="gamePk", how="left")

    return wide



def re_is_feature_col(c: str) -> bool:
    # rolling mean/std and ranks
    if c.startswith(("season_", "last5_")) and (c.endswith("_mean") or c.endswith("_std") or c.endswith("_rank")):
        return True
    return False


def write_table(conn: sqlite3.Connection, df: pd.DataFrame, table_name: str) -> None:
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_gameDate ON {table_name}(gameDate);")
    conn.commit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="nhl_scrape.sqlite")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Build games_table_{season} for a range")
    b.add_argument("--start-year", type=int, default=2015)
    b.add_argument("--end-year", type=int, default=2024)

    b.add_argument("--no-ranks", action="store_true", help="Skip league-rank features")

    args = ap.parse_args()
    conn = connect(args.db)

    if args.cmd == "build":
        def nhl_seasons(start_year: int, end_year: int):
            """
            Yields NHL season codes like 20152016, 20162017, ...
            """
            for y in range(start_year, end_year + 1):
                yield y * 10000 + (y + 1)

        for season in nhl_seasons(args.start_year, args.end_year):
            df = build_wide_games_table(conn, season, include_ranks=not args.no_ranks)

            table = f"games_table_{season}"
            write_table(conn, df, table)
            print(f"Wrote {table}: {len(df):,} rows, {df.shape[1]:,} cols")

    conn.close()


if __name__ == "__main__":
    main()
