#!/usr/bin/env python3
"""
MLB scraper (StatsAPI) — Step 1 + Step 2

Step 1:
  - Pull MLB schedule for a season via StatsAPI (gameType=R only; NO playoffs)
  - Write to SQLite table: schedule_games

Step 2:
  - Iterate final games in schedule_games for that season
  - Fetch boxscore JSON per gamePk
  - For each team (home/away), append/UPSERT into:
      {team}_batters_{year}   -> B1-B13 (+ placeholders B14-B17)
      {team}_starters_{year}  -> SP1-SP9 (+ derived WHIP)
      {team}_pitchers_{year}  -> P1-P9 + P17-P18 (bullpen only; starter excluded)

Notes:
- No Baseball-Reference scraping, so no 403 problems.
- SP10-SP16 and P10-P16 are removed as requested.
- Advanced B14-B17 are placeholders for later (external sources only if needed).
"""

from __future__ import annotations

import argparse
import random
import re
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import requests

STATSAPI_BASE = "https://statsapi.mlb.com"
SCHEDULE_ENDPOINT = STATSAPI_BASE + "/api/v1/schedule"
BOXSCORE_ENDPOINT = STATSAPI_BASE + "/api/v1/game/{gamePk}/boxscore"


# ----------------------------
# Config / session / utilities
# ----------------------------

@dataclass
class ScrapeConfig:
    season: int
    sqlite_path: str = "mlb_scrape.sqlite"

    connect_timeout_s: float = 10.0
    read_timeout_s: float = 30.0
    max_retries: int = 6
    base_backoff_s: float = 1.0

    min_delay_s: float = 0.10
    max_delay_s: float = 0.35
    jitter_s: float = 0.15


def polite_sleep(cfg: ScrapeConfig) -> None:
    time.sleep(random.uniform(cfg.min_delay_s, cfg.max_delay_s) + random.uniform(0, cfg.jitter_s))


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "python-mlb-scraper/1.0", "Accept": "application/json"})
    return s


def fetch_json(session: requests.Session, url: str, params: Optional[Dict[str, Any]], cfg: ScrapeConfig) -> Dict[str, Any]:
    last_exc: Optional[Exception] = None
    for attempt in range(cfg.max_retries):
        polite_sleep(cfg)
        try:
            r = session.get(url, params=params, timeout=(cfg.connect_timeout_s, cfg.read_timeout_s))
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                backoff = cfg.base_backoff_s * (2 ** attempt) + random.uniform(0, 0.5)
                time.sleep(backoff)
                continue
            raise RuntimeError(f"GET {r.url} -> {r.status_code}")
        except Exception as e:
            last_exc = e
            backoff = cfg.base_backoff_s * (2 ** attempt) + random.uniform(0, 0.5)
            time.sleep(backoff)

    raise RuntimeError(f"GET failed after retries; url={url}; params={params}; last_exc={last_exc}") from last_exc


def slug_team(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        if s == "":
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return None


def innings_to_float(ip: Any) -> Optional[float]:
    """
    StatsAPI inningsPitched is a string like "5.2" meaning 5 and 2/3 innings.
    Convert to float innings: 5.666...
    """
    if ip is None:
        return None
    if isinstance(ip, (int, float)):
        return float(ip)
    if not isinstance(ip, str):
        return None
    ip = ip.strip()
    if ip == "":
        return None
    if "." in ip:
        whole, frac = ip.split(".", 1)
        try:
            w = int(whole)
            f = int(frac)
        except ValueError:
            return None
        if f == 0:
            return float(w)
        if f == 1:
            return w + 1.0 / 3.0
        if f == 2:
            return w + 2.0 / 3.0
        return None
    try:
        return float(int(ip))
    except ValueError:
        return None


def compute_whip(h: Optional[int], bb: Optional[int], ip: Optional[float]) -> Optional[float]:
    if h is None or bb is None or ip is None or ip <= 0:
        return None
    return (h + bb) / ip


# ----------------------------
# SQLite schema
# ----------------------------

def init_sqlite(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL;")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS scrape_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at_utc TEXT DEFAULT (datetime('now')),
            note TEXT
        )
        """
    )

    # Step 1 target
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schedule_games (
            season INTEGER NOT NULL,
            gamePk INTEGER NOT NULL,
            gameDate TEXT,
            status TEXT,
            homeTeamId INTEGER,
            homeTeamName TEXT,
            awayTeamId INTEGER,
            awayTeamName TEXT,
            homeScore INTEGER,
            awayScore INTEGER,
            homeWin REAL,            -- 1.0 if home won, 0.0 if home lost, NULL if not final
            PRIMARY KEY (season, gamePk)
        )
        """
    )
    conn.commit()
    return conn


def ensure_table(conn: sqlite3.Connection, table_name: str, ddl: str) -> None:
    conn.execute(ddl.format(table=table_name))
    conn.commit()


def ensure_batters_table(conn: sqlite3.Connection, team_slug: str, year: int) -> str:
    table = f"{team_slug}_batters_{year}"
    ddl = """
    CREATE TABLE IF NOT EXISTS "{table}" (
        gamePk INTEGER PRIMARY KEY,
        season INTEGER,
        gameDate TEXT,
        isHome INTEGER,
        teamId INTEGER,
        teamName TEXT,
        opponentId INTEGER,
        opponentName TEXT,
        teamScore INTEGER,
        oppScore INTEGER,
        won INTEGER,

        -- Core B1-B13
        B1_AB INTEGER,
        B2_H INTEGER,
        B3_BB INTEGER,
        B4_SO INTEGER,
        B5_PA INTEGER,
        B6_BA REAL,
        B7_OBP REAL,
        B8_SLG REAL,
        B9_OPS REAL,
        B10_Pit INTEGER,     -- pitchesSeen if available
        B11_Str INTEGER,     -- strikes if available
        B12_PO INTEGER,
        B13_A INTEGER,

        -- Optional advanced placeholders B14-B17
        B14_wRCplus REAL,
        B15_HardHitPct REAL,
        B16_BarrelPct REAL,
        B17_OPSplus REAL
    )
    """
    ensure_table(conn, table, ddl)
    return table


def ensure_starters_table(conn: sqlite3.Connection, team_slug: str, year: int) -> str:
    table = f"{team_slug}_starters_{year}"
    ddl = """
    CREATE TABLE IF NOT EXISTS "{table}" (
        gamePk INTEGER PRIMARY KEY,
        season INTEGER,
        gameDate TEXT,
        isHome INTEGER,
        teamId INTEGER,
        teamName TEXT,
        opponentId INTEGER,
        opponentName TEXT,
        teamScore INTEGER,
        oppScore INTEGER,
        won INTEGER,

        pitcherId INTEGER,
        pitcherName TEXT,

        -- SP1-SP9 only
        SP1_IP REAL,
        SP2_H INTEGER,
        SP3_BB INTEGER,
        SP4_SO INTEGER,
        SP5_HR INTEGER,
        SP6_ERA REAL,
        SP7_BF INTEGER,
        SP8_Pit INTEGER,
        SP9_Str INTEGER,

        WHIP REAL
    )
    """
    ensure_table(conn, table, ddl)
    return table


def ensure_pitchers_table(conn: sqlite3.Connection, team_slug: str, year: int) -> str:
    table = f"{team_slug}_pitchers_{year}"
    ddl = """
    CREATE TABLE IF NOT EXISTS "{table}" (
        gamePk INTEGER PRIMARY KEY,
        season INTEGER,
        gameDate TEXT,
        isHome INTEGER,
        teamId INTEGER,
        teamName TEXT,
        opponentId INTEGER,
        opponentName TEXT,
        teamScore INTEGER,
        oppScore INTEGER,
        won INTEGER,

        -- Bullpen only: P1-P9 + P17-P18
        P1_IP REAL,
        P2_H INTEGER,
        P3_BB INTEGER,
        P4_SO INTEGER,
        P5_HR INTEGER,
        P6_ERA REAL,
        P7_BF INTEGER,
        P8_Pit INTEGER,
        P9_Str INTEGER,

        P17_IR INTEGER,
        P18_IS INTEGER,

        relieverCount INTEGER
    )
    """
    ensure_table(conn, table, ddl)
    return table


# ----------------------------
# Step 1: Schedule ingestion
# ----------------------------

def iter_regular_season_games_for_year(session: requests.Session, year: int, cfg: ScrapeConfig) -> List[Dict[str, Any]]:
    params = {
        "sportId": 1,
        "season": year,
        "gameType": "R",  # Regular season only (NO playoffs)
    }
    data = fetch_json(session, SCHEDULE_ENDPOINT, params=params, cfg=cfg)

    out: List[Dict[str, Any]] = []
    for date_block in data.get("dates", []):
        for g in date_block.get("games", []):
            gamePk = g.get("gamePk")
            gameDate = g.get("gameDate")
            status = (g.get("status") or {}).get("detailedState")

            teams = g.get("teams") or {}
            home = teams.get("home") or {}
            away = teams.get("away") or {}

            home_team = (home.get("team") or {})
            away_team = (away.get("team") or {})

            home_score = home.get("score")
            away_score = away.get("score")

            is_final = (g.get("status") or {}).get("abstractGameState") == "Final"
            home_win: Optional[float] = None
            if is_final and isinstance(home_score, int) and isinstance(away_score, int):
                home_win = 1.0 if home_score > away_score else 0.0

            out.append(
                {
                    "season": year,
                    "gamePk": int(gamePk),
                    "gameDate": gameDate,
                    "status": status,
                    "homeTeamId": int(home_team.get("id")) if home_team.get("id") else None,
                    "homeTeamName": home_team.get("name"),
                    "awayTeamId": int(away_team.get("id")) if away_team.get("id") else None,
                    "awayTeamName": away_team.get("name"),
                    "homeScore": home_score if isinstance(home_score, int) else None,
                    "awayScore": away_score if isinstance(away_score, int) else None,
                    "homeWin": home_win,
                }
            )

    return out


def upsert_schedule_games(conn: sqlite3.Connection, games: List[Dict[str, Any]]) -> None:
    rows = [
        (
            g["season"], g["gamePk"], g["gameDate"], g["status"],
            g["homeTeamId"], g["homeTeamName"], g["awayTeamId"], g["awayTeamName"],
            g["homeScore"], g["awayScore"], g["homeWin"],
        )
        for g in games
    ]
    conn.executemany(
        """
        INSERT INTO schedule_games (
            season, gamePk, gameDate, status,
            homeTeamId, homeTeamName, awayTeamId, awayTeamName,
            homeScore, awayScore, homeWin
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(season, gamePk) DO UPDATE SET
            gameDate=excluded.gameDate,
            status=excluded.status,
            homeTeamId=excluded.homeTeamId,
            homeTeamName=excluded.homeTeamName,
            awayTeamId=excluded.awayTeamId,
            awayTeamName=excluded.awayTeamName,
            homeScore=excluded.homeScore,
            awayScore=excluded.awayScore,
            homeWin=excluded.homeWin
        """,
        rows, 
    )

    conn.commit()


def run_step1(conn: sqlite3.Connection, session: requests.Session, cfg: ScrapeConfig) -> None:
    conn.execute("INSERT INTO scrape_runs(note) VALUES (?)", (f"step1-schedule season={cfg.season}",))
    conn.commit()

    print(f"\n[step1] Fetching MLB regular season schedule: season={cfg.season} gameType=R")
    games = iter_regular_season_games_for_year(session, cfg.season, cfg)
    finals = sum(1 for g in games if g["homeWin"] is not None)
    print(f"[step1] games returned: {len(games)} | finals with labels: {finals}")

    upsert_schedule_games(conn, games)

    shown = 0
    for g in games:
        if g["homeWin"] is None:
            continue
        print(
            f"  gamePk={g['gamePk']} {g['gameDate']} | "
            f"{g['awayTeamName']} @ {g['homeTeamName']} "
            f"{g['awayScore']}-{g['homeScore']} homeWin={g['homeWin']}"
        )
        shown += 1
        if shown >= 10:
            break


# ----------------------------
# Step 2: Boxscore ingestion
# ----------------------------

def get_game_context(conn: sqlite3.Connection, season: int, gamePk: int) -> Dict[str, Any]:
    row = conn.execute(
        """
        SELECT season, gamePk, gameDate, status,
               homeTeamId, homeTeamName, awayTeamId, awayTeamName,
               homeScore, awayScore, homeWin
        FROM schedule_games
        WHERE season=? AND gamePk=?
        """,
        (season, gamePk),
    ).fetchone()
    if not row:
        raise RuntimeError(f"schedule_games missing season={season} gamePk={gamePk}")

    (
        season, gamePk, gameDate, status,
        homeTeamId, homeTeamName, awayTeamId, awayTeamName,
        homeScore, awayScore, homeWin
    ) = row

    return {
        "season": season,
        "gamePk": gamePk,
        "gameDate": gameDate,
        "homeTeamId": homeTeamId,
        "homeTeamName": homeTeamName,
        "awayTeamId": awayTeamId,
        "awayTeamName": awayTeamName,
        "homeScore": homeScore,
        "awayScore": awayScore,
        "homeWin": homeWin,
    }


def parse_team_batting(team_stats: Dict[str, Any]) -> Dict[str, Any]:
    bat = (team_stats.get("batting") or {})
    fld = (team_stats.get("fielding") or {})

    return {
        "B1_AB": bat.get("atBats"),
        "B2_H": bat.get("hits"),
        "B3_BB": bat.get("baseOnBalls"),
        "B4_SO": bat.get("strikeOuts"),
        "B5_PA": bat.get("plateAppearances"),
        "B6_BA": _to_float(bat.get("avg")),
        "B7_OBP": _to_float(bat.get("obp")),
        "B8_SLG": _to_float(bat.get("slg")),
        "B9_OPS": _to_float(bat.get("ops")),
        "B10_Pit": bat.get("pitchesSeen"),   # may be None
        "B11_Str": bat.get("strikes"),       # may be None
        "B12_PO": fld.get("putOuts"),
        "B13_A": fld.get("assists"),
    }


def find_starting_pitcher_id(team_block: Dict[str, Any]) -> Optional[int]:
    pitchers = team_block.get("pitchers") or []
    if not pitchers:
        return None
    try:
        return int(pitchers[0])
    except Exception:
        return None


def pitcher_line(team_block: Dict[str, Any], pitcher_id: int) -> Dict[str, Any]:
    players = team_block.get("players") or {}
    pkey = f"ID{pitcher_id}"
    pdata = players.get(pkey) or {}
    stats = ((pdata.get("stats") or {}).get("pitching") or {})
    return stats

def opponent_pitches_and_strikes(box: Dict[str, Any], side: str) -> Tuple[Optional[int], Optional[int]]:
    """
    side = 'home' or 'away' for the batting team we are processing.
    We derive B10/B11 from the OPPONENT team's pitchersThrown/strikes.
    """
    opp_side = "away" if side == "home" else "home"
    opp_block = (box.get("teams") or {}).get(opp_side) or {}
    opp_players = opp_block.get("players") or {}
    opp_pitchers = [pid for pid in (opp_block.get("pitchers") or []) if isinstance(pid, int)]

    total_pit = 0
    total_str = 0
    got_any = False

    for pid in opp_pitchers:
        pdata = opp_players.get(f"ID{pid}") or {}
        pst = ((pdata.get("stats") or {}).get("pitching") or {})
        pit = pst.get("pitchesThrown")
        st = pst.get("strikes")

        if isinstance(pit, int):
            total_pit += pit
            got_any = True
        if isinstance(st, int):
            total_str += st
            got_any = True

    if not got_any:
        return None, None

    # If strikes are missing but pitches exist, we still return pitches.
    return total_pit if total_pit > 0 else None, total_str if total_str > 0 else None


def agg_reliever_totals(team_block: Dict[str, Any], starter_id: Optional[int]) -> Dict[str, Any]:
    players = team_block.get("players") or {}
    pitcher_ids = [pid for pid in (team_block.get("pitchers") or []) if isinstance(pid, int)]
    relievers = [pid for pid in pitcher_ids if starter_id is None or pid != starter_id]

    totals = {
        "ip": 0.0,
        "h": 0,
        "bb": 0,
        "so": 0,
        "hr": 0,
        "bf": 0,
        "pit": 0,
        "str": 0,
        "er": 0,
        "ir": 0,
        "is": 0,
        "count": len(relievers),
    }

    for pid in relievers:
        pkey = f"ID{pid}"
        pdata = players.get(pkey) or {}
        pst = ((pdata.get("stats") or {}).get("pitching") or {})

        ip = innings_to_float(pst.get("inningsPitched")) or 0.0
        totals["ip"] += ip
        totals["h"] += int(pst.get("hits") or 0)
        totals["bb"] += int(pst.get("baseOnBalls") or 0)
        totals["so"] += int(pst.get("strikeOuts") or 0)
        totals["hr"] += int(pst.get("homeRuns") or 0)
        totals["bf"] += int(pst.get("battersFaced") or 0)
        totals["pit"] += int(pst.get("pitchesThrown") or 0)
        totals["str"] += int(pst.get("strikes") or 0)
        totals["er"] += int(pst.get("earnedRuns") or 0)
        totals["ir"] += int(pst.get("inheritedRunners") or 0)
        totals["is"] += int(pst.get("inheritedRunnersScored") or 0)

    era = None
    if totals["ip"] > 0:
        era = 9.0 * totals["er"] / totals["ip"]

    return {
        "P1_IP": totals["ip"],
        "P2_H": totals["h"],
        "P3_BB": totals["bb"],
        "P4_SO": totals["so"],
        "P5_HR": totals["hr"],
        "P6_ERA": era,
        "P7_BF": totals["bf"],
        "P8_Pit": totals["pit"],
        "P9_Str": totals["str"],
        "P17_IR": totals["ir"],
        "P18_IS": totals["is"],
        "relieverCount": totals["count"],
    }


def upsert_row(conn: sqlite3.Connection, table: str, row: Dict[str, Any]) -> None:
    cols = list(row.keys())
    placeholders = ",".join(["?"] * len(cols))
    updates = ",".join([f"{c}=excluded.{c}" for c in cols if c != "gamePk"])
    sql = f"""
    INSERT INTO "{table}" ({",".join(cols)})
    VALUES ({placeholders})
    ON CONFLICT(gamePk) DO UPDATE SET {updates}
    """
    conn.execute(sql, [row[c] for c in cols])


def process_game(conn: sqlite3.Connection, session: requests.Session, cfg: ScrapeConfig, gamePk: int) -> None:
    ctx = get_game_context(conn, cfg.season, gamePk)

    url = BOXSCORE_ENDPOINT.format(gamePk=gamePk)
    box = fetch_json(session, url, params=None, cfg=cfg)

    teams = box.get("teams") or {}
    for side in ("home", "away"):
        tblock = teams.get(side) or {}
        team_info = tblock.get("team") or {}
        team_id = team_info.get("id")
        team_name = team_info.get("name") or side
        team_slug = slug_team(str(team_name))
        is_home = 1 if side == "home" else 0

        opponent_id = ctx["awayTeamId"] if is_home else ctx["homeTeamId"]
        opponent_name = ctx["awayTeamName"] if is_home else ctx["homeTeamName"]
        team_score = ctx["homeScore"] if is_home else ctx["awayScore"]
        opp_score = ctx["awayScore"] if is_home else ctx["homeScore"]

        won = None
        if ctx["homeWin"] is not None:
            won = int(ctx["homeWin"] == 1.0) if is_home else int(ctx["homeWin"] == 0.0)

        batters_table = ensure_batters_table(conn, team_slug, cfg.season)
        starters_table = ensure_starters_table(conn, team_slug, cfg.season)
        pitchers_table = ensure_pitchers_table(conn, team_slug, cfg.season)

        # 2a Batting
        team_stats = tblock.get("teamStats") or {}
        b = parse_team_batting(team_stats)

        # Derive B10/B11 from opponent pitching (more reliable than team batting totals)
        opp_pit, opp_str = opponent_pitches_and_strikes(box, side)

        batting_row = {
            "gamePk": gamePk,
            "season": cfg.season,
            "gameDate": ctx["gameDate"],
            "isHome": is_home,
            "teamId": team_id,
            "teamName": team_name,
            "opponentId": opponent_id,
            "opponentName": opponent_name,
            "teamScore": team_score,
            "oppScore": opp_score,
            "won": won,
            **b,
            "B10_Pit": opp_pit if opp_pit is not None else b.get("B10_Pit"),
            "B11_Str": opp_str if opp_str is not None else b.get("B11_Str"),
            "B14_wRCplus": None,
            "B15_HardHitPct": None,
            "B16_BarrelPct": None,
            "B17_OPSplus": None,
        }
        upsert_row(conn, batters_table, batting_row)

        # 2b Starter
        starter_id = find_starting_pitcher_id(tblock)
        sp_stats = pitcher_line(tblock, starter_id) if starter_id else {}

        pitcher_name = None
        if starter_id:
            pkey = f"ID{starter_id}"
            pdata = (tblock.get("players") or {}).get(pkey) or {}
            person = pdata.get("person") or {}
            pitcher_name = person.get("fullName")

        sp_ip = innings_to_float(sp_stats.get("inningsPitched"))
        sp_h = sp_stats.get("hits")
        sp_bb = sp_stats.get("baseOnBalls")
        sp_er = sp_stats.get("earnedRuns")

        # Derive ERA if not provided
        sp_era = _to_float(sp_stats.get("era"))
        if sp_era is None and sp_ip is not None and sp_ip > 0 and isinstance(sp_er, int):
            sp_era = 9.0 * sp_er / sp_ip

        starter_row = {
            "gamePk": gamePk,
            "season": cfg.season,
            "gameDate": ctx["gameDate"],
            "isHome": is_home,
            "teamId": team_id,
            "teamName": team_name,
            "opponentId": opponent_id,
            "opponentName": opponent_name,
            "teamScore": team_score,
            "oppScore": opp_score,
            "won": won,
            "pitcherId": starter_id,
            "pitcherName": pitcher_name,
            "SP1_IP": sp_ip,
            "SP2_H": sp_stats.get("hits"),
            "SP3_BB": sp_stats.get("baseOnBalls"),
            "SP4_SO": sp_stats.get("strikeOuts"),
            "SP5_HR": sp_stats.get("homeRuns"),
            "SP6_ERA": sp_era,
            "SP7_BF": sp_stats.get("battersFaced"),
            "SP8_Pit": sp_stats.get("pitchesThrown"),
            "SP9_Str": sp_stats.get("strikes"),
            "WHIP": compute_whip(sp_h, sp_bb, sp_ip),
        }
        upsert_row(conn, starters_table, starter_row)

        # 2c Bullpen totals (relief only)
        rel = agg_reliever_totals(tblock, starter_id)
        pitchers_row = {
            "gamePk": gamePk,
            "season": cfg.season,
            "gameDate": ctx["gameDate"],
            "isHome": is_home,
            "teamId": team_id,
            "teamName": team_name,
            "opponentId": opponent_id,
            "opponentName": opponent_name,
            "teamScore": team_score,
            "oppScore": opp_score,
            "won": won,
            **rel,
        }
        upsert_row(conn, pitchers_table, pitchers_row)

    conn.commit()


def run_step2(conn: sqlite3.Connection, session: requests.Session, cfg: ScrapeConfig, limit: Optional[int]) -> None:
    conn.execute("INSERT INTO scrape_runs(note) VALUES (?)", (f"step2-boxscores season={cfg.season}",))
    conn.commit()

    rows = conn.execute(
        """
        SELECT gamePk
        FROM schedule_games
        WHERE season=? AND homeWin IS NOT NULL
        ORDER BY gameDate ASC
        """,
        (cfg.season,),
    ).fetchall()

    gamepks = [int(r[0]) for r in rows]
    if limit is not None:
        gamepks = gamepks[:limit]

    print(f"\n[step2] season={cfg.season} final games to process: {len(gamepks)}")
    for i, gamePk in enumerate(gamepks, 1):
        if i % 50 == 0:
            print(f"[step2] processed {i}/{len(gamepks)} ...")
        try:
            process_game(conn, session, cfg, gamePk)
        except Exception as e:
            print(f"[step2][WARN] gamePk={gamePk} failed: {e}")

    print("[step2] done.")

from collections import deque
from math import sqrt

# ----------------------------
# Step 3: Modeling table build
# ----------------------------

BAT_COLS = [
    "B1_AB","B2_H","B3_BB","B4_SO","B5_PA","B6_BA","B7_OBP","B8_SLG","B9_OPS",
    "B10_Pit","B11_Str","B12_PO","B13_A",
    "B14_wRCplus","B15_HardHitPct","B16_BarrelPct","B17_OPSplus",
]

SP_COLS = ["SP1_IP","SP2_H","SP3_BB","SP4_SO","SP5_HR","SP6_ERA","SP7_BF","SP8_Pit","SP9_Str","WHIP"]

RP_COLS = ["P1_IP","P2_H","P3_BB","P4_SO","P5_HR","P6_ERA","P7_BF","P8_Pit","P9_Str","P17_IR","P18_IS"]


class RollingStats:
    """
    Maintains mean/std for:
      - season-to-date (unbounded)
      - last-N rolling window (bounded)
    Stores sum and sumsq per metric for O(1) update.
    """
    def __init__(self, cols, window: int | None):
        self.cols = cols
        self.window = window
        self.n = 0
        self.sum = {c: 0.0 for c in cols}
        self.sumsq = {c: 0.0 for c in cols}
        self.q = deque() if window is not None else None  # stores dict of {col: float}

    def _as_float(self, v):
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        try:
            return float(str(v))
        except Exception:
            return None

    def add(self, row: dict):
        """
        row: dict of {col: value}; ignores missing/None values
        """
        vals = {}
        for c in self.cols:
            v = self._as_float(row.get(c))
            if v is None:
                continue
            vals[c] = v

        # If bounded, evict oldest first
        if self.window is not None:
            self.q.append(vals)
            # Add new
            for c, v in vals.items():
                self.sum[c] += v
                self.sumsq[c] += v * v
            self.n += 1

            # Evict if needed
            while self.n > self.window:
                old = self.q.popleft()
                for c, v in old.items():
                    self.sum[c] -= v
                    self.sumsq[c] -= v * v
                self.n -= 1
        else:
            # season-to-date (unbounded)
            for c, v in vals.items():
                self.sum[c] += v
                self.sumsq[c] += v * v
            self.n += 1

    def mean_std(self, fill=-1.0):
        """
        Returns dict: {col_mean: ..., col_std: ...}
        Uses population std: sqrt(E[x^2] - E[x]^2)
        If n==0 -> fill
        """
        out = {}
        if self.n <= 0:
            for c in self.cols:
                out[f"{c}_mean"] = fill
                out[f"{c}_std"] = fill
            return out

        # NEW: for bounded windows, require a full window before emitting stats
        if self.window is not None and self.n < self.window:
            for c in self.cols:
                out[f"{c}_mean"] = fill
                out[f"{c}_std"] = fill
            return out

        # Normal case
        for c in self.cols:
            mu = self.sum[c] / self.n
            ex2 = self.sumsq[c] / self.n
            var = max(0.0, ex2 - mu * mu)
            sd = sqrt(var)
            out[f"{c}_mean"] = mu
            out[f"{c}_std"] = sd

        return out
    
def _to_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return int(x)
    if isinstance(x, str):
        s = x.strip()
        if s == "":
            return None
        try:
            return int(s)
        except ValueError:
            # sometimes "1,234"
            try:
                return int(s.replace(",", ""))
            except ValueError:
                return None
    return None

def _get_starter_id_for_game(conn: sqlite3.Connection, starters_table: str | None, gamePk: int) -> Optional[int]:
    if not starters_table:
        return None
    r = conn.execute(f'SELECT pitcherId FROM "{starters_table}" WHERE gamePk=?', (int(gamePk),)).fetchone()
    if not r or r[0] is None:
        return None
    try:
        return int(r[0])
    except Exception:
        return None

PITCHER_CAREER_CACHE: dict[int, dict] = {}

def fetch_pitcher_career_stats(session, pitcher_id: int, cfg) -> dict:
    """
    Career pitching stats via dedicated endpoint:
      /api/v1/people/{id}/stats?stats=career&group=pitching&gameType=R

    Returns dict with keys matching SP_COLS, missing -> -1.0
    """
    if pitcher_id in PITCHER_CAREER_CACHE:
        return PITCHER_CAREER_CACHE[pitcher_id]

    out = {c: -1.0 for c in SP_COLS}

    url = f"{STATSAPI_BASE}/api/v1/people/{pitcher_id}/stats"
    params = {
        "stats": "career",
        "group": "pitching",
        "gameType": "R",   # regular season career
    }

    data = fetch_json(session, url, params=params, cfg=cfg)

    # Expected shape: {"stats":[{"splits":[{"stat":{...}}]}]}
    stats_blocks = data.get("stats") or []
    if not stats_blocks:
        PITCHER_CAREER_CACHE[pitcher_id] = out
        return out

    splits = stats_blocks[0].get("splits") or []
    if not splits:
        PITCHER_CAREER_CACHE[pitcher_id] = out
        return out

    stat = splits[0].get("stat") or {}

    ip = innings_to_float(stat.get("inningsPitched"))
    h  = _to_int(stat.get("hits"))
    bb = _to_int(stat.get("baseOnBalls"))
    so = _to_int(stat.get("strikeOuts"))
    hr = _to_int(stat.get("homeRuns"))
    bf = _to_int(stat.get("battersFaced"))

    era = _to_float(stat.get("era"))
    whip = _to_float(stat.get("whip"))

    if ip is not None: out["SP1_IP"] = float(ip)
    if h  is not None: out["SP2_H"]  = float(h)
    if bb is not None: out["SP3_BB"] = float(bb)
    if so is not None: out["SP4_SO"] = float(so)
    if hr is not None: out["SP5_HR"] = float(hr)
    if era is not None: out["SP6_ERA"] = float(era)
    if bf is not None: out["SP7_BF"] = float(bf)

    # These typically aren't provided in career endpoint
    out["SP8_Pit"] = -1.0
    out["SP9_Str"] = -1.0

    if whip is not None:
        out["WHIP"] = float(whip)
    elif ip is not None and ip > 0 and h is not None and bb is not None:
        out["WHIP"] = float(h + bb) / float(ip)

    PITCHER_CAREER_CACHE[pitcher_id] = out
    return out




def _list_team_tables(conn: sqlite3.Connection, season: int, kind_suffix: str) -> list[str]:
    # kind_suffix like "_batters_2025"
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE ?",
        (f"%{kind_suffix}",),
    ).fetchall()
    return [r[0] for r in rows]


def _load_team_rows(conn: sqlite3.Connection, table: str, cols: list[str]) -> dict[int, dict]:
    """
    Load team table into mapping:
      gamePk -> row dict with requested cols + gameDate + teamId
    """
    select_cols = ["gamePk", "gameDate", "teamId"] + cols
    q = f'SELECT {",".join(select_cols)} FROM "{table}" ORDER BY gameDate ASC'
    out = {}
    for row in conn.execute(q).fetchall():
        d = dict(zip(select_cols, row))
        out[int(d["gamePk"])] = d
    return out


def _get_team_table_by_teamId(conn: sqlite3.Connection, season: int, kind: str, teamId: int) -> str | None:
    """
    kind in {'batters','starters','pitchers'}
    Finds the table for this teamId by scanning candidate tables and checking a sample row.
    (30 teams -> cheap.)
    """
    suffix = f"_{kind}_{season}"
    candidates = _list_team_tables(conn, season, suffix)
    for t in candidates:
        r = conn.execute(f'SELECT teamId FROM "{t}" LIMIT 1').fetchone()
        if r and int(r[0]) == int(teamId):
            return t
    return None


def _ensure_games_table(conn: sqlite3.Connection, season: int) -> str:
    table = f"games_table_{season}"

    # Build columns dynamically
    cols = [
        "gamePk INTEGER PRIMARY KEY",
        "season INTEGER",
        "gameDate TEXT",
        "homeTeamId INTEGER",
        "awayTeamId INTEGER",
        "homeTeamName TEXT",
        "awayTeamName TEXT",
        "homeWin REAL",
    ]

    def add_block(prefix: str, stat_prefix: str, base_cols: list[str]):
        for c in base_cols:
            cols.append(f"{prefix}{stat_prefix}{c}_mean REAL")
            cols.append(f"{prefix}{stat_prefix}{c}_std REAL")

    # Batting: season, last10, last20
    add_block("home_", "bat_season_", BAT_COLS)
    add_block("home_", "bat_last10_", BAT_COLS)
    add_block("home_", "bat_last20_", BAT_COLS)
    add_block("away_", "bat_season_", BAT_COLS)
    add_block("away_", "bat_last10_", BAT_COLS)
    add_block("away_", "bat_last20_", BAT_COLS)

    # Starters: season, last3
    add_block("home_", "sp_season_", SP_COLS)
    add_block("home_", "sp_last3_", SP_COLS)
    add_block("away_", "sp_season_", SP_COLS)
    add_block("away_", "sp_last3_", SP_COLS)

    # Bullpen: season only
    add_block("home_", "rp_season_", RP_COLS)
    add_block("away_", "rp_season_", RP_COLS)

    # Pitcher career placeholders (we’ll fill later from an external source)
    for side in ("home_", "away_"):
        for c in SP_COLS:
            cols.append(f"{side}sp_career_{c} REAL")

    ddl = f'CREATE TABLE IF NOT EXISTS "{table}" (\n  ' + ",\n  ".join(cols) + "\n)"
    conn.execute(ddl)
    conn.commit()
    return table


def _insert_games_row(conn: sqlite3.Connection, table: str, row: dict):
    cols = list(row.keys())
    placeholders = ",".join(["?"] * len(cols))
    updates = ",".join([f"{c}=excluded.{c}" for c in cols if c != "gamePk"])
    sql = f"""
    INSERT INTO "{table}" ({",".join(cols)})
    VALUES ({placeholders})
    ON CONFLICT(gamePk) DO UPDATE SET {updates}
    """
    conn.execute(sql, [row[c] for c in cols])


def run_step3(conn: sqlite3.Connection, session: requests.Session, cfg: ScrapeConfig, limit: int | None = None):
    """
    Builds modeling table by iterating games in chronological order and
    computing aggregates from *prior* history only (no leakage).
    """
    conn.execute("INSERT INTO scrape_runs(note) VALUES (?)", (f"step3-games_table season={cfg.season}",))
    conn.commit()

    out_table = _ensure_games_table(conn, cfg.season)

    games = conn.execute(
        """
        SELECT gamePk, gameDate,
               homeTeamId, homeTeamName,
               awayTeamId, awayTeamName,
               homeWin
        FROM schedule_games
        WHERE season=? AND homeWin IS NOT NULL
        ORDER BY gameDate ASC
        """,
        (cfg.season,),
    ).fetchall()

    if limit is not None:
        games = games[:limit]

    # Discover per-team table names (once)
    team_ids = sorted({int(g[2]) for g in games} | {int(g[4]) for g in games})
    tables = {}
    for tid in team_ids:
        tables[(tid, "batters")] = _get_team_table_by_teamId(conn, cfg.season, "batters", tid)
        tables[(tid, "starters")] = _get_team_table_by_teamId(conn, cfg.season, "starters", tid)
        tables[(tid, "pitchers")] = _get_team_table_by_teamId(conn, cfg.season, "pitchers", tid)

    # Load all team rows into memory (fast lookups by gamePk)
    team_bat = {}
    team_sp = {}
    team_rp = {}
    for tid in team_ids:
        tb = tables[(tid, "batters")]
        ts = tables[(tid, "starters")]
        tr = tables[(tid, "pitchers")]
        team_bat[tid] = _load_team_rows(conn, tb, BAT_COLS) if tb else {}
        team_sp[tid] = _load_team_rows(conn, ts, SP_COLS) if ts else {}
        team_rp[tid] = _load_team_rows(conn, tr, RP_COLS) if tr else {}

    # Rolling stats objects per team
    bat_season = {tid: RollingStats(BAT_COLS, window=None) for tid in team_ids}
    bat_last10 = {tid: RollingStats(BAT_COLS, window=10) for tid in team_ids}
    bat_last20 = {tid: RollingStats(BAT_COLS, window=20) for tid in team_ids}

    sp_season = {tid: RollingStats(SP_COLS, window=None) for tid in team_ids}
    sp_last3 = {tid: RollingStats(SP_COLS, window=3) for tid in team_ids}

    rp_season = {tid: RollingStats(RP_COLS, window=None) for tid in team_ids}

    def build_side_features(prefix: str, tid: int, gamePk: int) -> dict:
        # batting
        feats = {}
        feats.update({f"{prefix}bat_season_{k}": v for k, v in bat_season[tid].mean_std().items()})
        feats.update({f"{prefix}bat_last10_{k}": v for k, v in bat_last10[tid].mean_std().items()})
        feats.update({f"{prefix}bat_last20_{k}": v for k, v in bat_last20[tid].mean_std().items()})

        # starters
        feats.update({f"{prefix}sp_season_{k}": v for k, v in sp_season[tid].mean_std().items()})
        feats.update({f"{prefix}sp_last3_{k}": v for k, v in sp_last3[tid].mean_std().items()})

        # bullpen
        feats.update({f"{prefix}rp_season_{k}": v for k, v in rp_season[tid].mean_std().items()})

        # career stats: use starterId from the starters table for THIS game
        starters_table = tables.get((tid, "starters"))
        pid = _get_starter_id_for_game(conn, starters_table, gamePk)

        career = fetch_pitcher_career_stats(session, int(pid), cfg) if pid else {c: -1.0 for c in SP_COLS}
        for c in SP_COLS:
            feats[f"{prefix}sp_career_{c}"] = career.get(c, -1.0)

        return feats

    print(f"\n[step3] building {out_table} from {len(games)} games (season={cfg.season})")

    for i, (gamePk, gameDate, homeId, homeName, awayId, awayName, homeWin) in enumerate(games, 1):
        homeId = int(homeId); awayId = int(awayId); gamePk = int(gamePk)

        row = {
            "gamePk": gamePk,
            "season": cfg.season,
            "gameDate": gameDate,
            "homeTeamId": homeId,
            "awayTeamId": awayId,
            "homeTeamName": homeName,
            "awayTeamName": awayName,
            "homeWin": float(homeWin),
        }

        # Use ONLY prior data (rolling objects currently represent games before this one)
        row.update(build_side_features("home_", homeId, gamePk))
        row.update(build_side_features("away_", awayId, gamePk))


        _insert_games_row(conn, out_table, row)

        # Now "advance time": add this game’s actual stats into the rolling objects
        hb = team_bat[homeId].get(gamePk)
        ab = team_bat[awayId].get(gamePk)
        if hb:
            bat_season[homeId].add(hb); bat_last10[homeId].add(hb); bat_last20[homeId].add(hb)
        if ab:
            bat_season[awayId].add(ab); bat_last10[awayId].add(ab); bat_last20[awayId].add(ab)

        hs = team_sp[homeId].get(gamePk)
        a_s = team_sp[awayId].get(gamePk)
        if hs:
            sp_season[homeId].add(hs); sp_last3[homeId].add(hs)
        if a_s:
            sp_season[awayId].add(a_s); sp_last3[awayId].add(a_s)

        hr = team_rp[homeId].get(gamePk)
        ar = team_rp[awayId].get(gamePk)
        if hr:
            rp_season[homeId].add(hr)
        if ar:
            rp_season[awayId].add(ar)

        if i % 200 == 0:
            conn.commit()
            print(f"[step3] processed {i}/{len(games)} ...")

    conn.commit()
    print(f"[step3] done. wrote/updated {len(games)} rows in {out_table}.")


# ----------------------------
# CLI
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["step1", "step2", "step3", "step1+2", "all"], help="Which step(s) to run")
    ap.add_argument("--season", type=int, required=True, help="Season year, e.g. 2025")
    ap.add_argument("--db", type=str, default="mlb_scrape.sqlite", help="SQLite path")
    ap.add_argument("--limit", type=int, default=None, help="Limit games for step2 (debug)")

    args = ap.parse_args()

    cfg = ScrapeConfig(season=args.season, sqlite_path=args.db)
    conn = init_sqlite(args.db)
    session = make_session()

    if args.command == "step1":
        run_step1(conn, session, cfg)
    elif args.command == "step2":
        run_step2(conn, session, cfg, limit=args.limit)
    elif args.command == "step3":
        run_step3(conn, session, cfg, limit=args.limit)
    elif args.command == "all":
        run_step1(conn, session, cfg)
        run_step2(conn, session, cfg, limit=args.limit)
        run_step3(conn, session, cfg, limit=args.limit)
    else:
        run_step1(conn, session, cfg)
        run_step2(conn, session, cfg, limit=args.limit)

    conn.close()


if __name__ == "__main__":
    main()
