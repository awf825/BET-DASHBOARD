#!/usr/bin/env python3
"""
NHL scraper (NHL Web API: https://api-web.nhle.com/) — modeled after the structure of your MLB scraper.

High-level workflow (mirrors your MLB multi-step approach):
  Step 1) Pull game schedule + final scores and store in SQLite (games table)
  Step 2) For each game_id, pull gamecenter boxscore and store team-level stats (team_boxscore table)

Design notes:
- We store *post-game* stats. Your modeling notebook should later create *pre-game* features
  via rolling windows / lagging (to avoid leakage).
- We do NOT require "starting goalie" knowledge. If the boxscore returns goalie lines, we
  store them optionally in goalie_boxscore; you can later decide whether to use team-level
  goalie aggregates as lagged features.

Primary endpoints used (per NHL API reference):
  - /v1/score/{date}            (daily games + status + scores + teams)  (alt: /v1/schedule/{date})
  - /v1/gamecenter/{gameId}/boxscore (team boxscore + skaters + goalies)
Reference: https://github.com/Zmalski/NHL-API-Reference

Examples:
  # scrape 2015-16 through 2024-25 (inclusive), regular season + playoffs within Sept 1..Jun 30 windows
  python nhl_scraper.py --db nhl_scrape.sqlite --start 2015 --end 2025 step1
  python nhl_scraper.py --db nhl_scrape.sqlite --start 2015 --end 2025 step2

  # only a small date range (debug)
  python nhl_scraper.py --db nhl_scrape.sqlite --date-min 2024-10-01 --date-max 2024-10-07 step1 step2
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests


API_BASE = "https://api-web.nhle.com"


# ----------------------------
# HTTP helpers (polite + retry)
# ----------------------------

def polite_sleep(min_s: float = 0.35, max_s: float = 0.85) -> None:
    time.sleep(random.uniform(min_s, max_s))


def make_session(user_agent: str = "Mozilla/5.0 (compatible; nhl-scraper/1.0)") -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": user_agent,
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    })
    return s


def fetch_json(
    session: requests.Session,
    url: str,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = 30,
    max_retries: int = 5,
    backoff_s: float = 1.0
) -> Any:
    """
    Simple GET with exponential-ish backoff.
    """
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            r = session.get(url, params=params, timeout=timeout)
            if r.status_code == 429:
                # rate-limited
                sleep_s = backoff_s * attempt + random.uniform(0, 0.5)
                time.sleep(sleep_s)
                continue
            r.raise_for_status()
            # DEBUG — print & save ONE sample boxscore payload
            import json
            if not getattr(run_step2, "_printed_one", False):
                print("DEBUG boxscore payload type:", type(r.json()))
                if isinstance(r.json(), dict):
                    print("DEBUG boxscore top-level keys:", sorted(r.json().keys()))
                    with open("debug_boxscore_sample.json", "w") as f:
                        json.dump(r.json(), f, indent=2)
                    print("DEBUG wrote debug_boxscore_sample.json")
                run_step2._printed_one = True
            return r.json()
        except Exception as e:
            last_err = e
            sleep_s = backoff_s * attempt + random.uniform(0, 0.5)
            time.sleep(sleep_s)
    raise RuntimeError(f"Failed GET {url} after {max_retries} tries: {last_err}")


# ----------------------------
# SQLite schema + upserts
# ----------------------------

def init_sqlite(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def ensure_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS games (
            game_id TEXT PRIMARY KEY,
            season INTEGER,
            game_date TEXT,
            game_type TEXT,
            away_abbrev TEXT,
            away_name TEXT,
            home_abbrev TEXT,
            home_name TEXT,
            away_score INTEGER,
            home_score INTEGER,
            venue TEXT,
            game_state TEXT,
            start_time_utc TEXT,
            last_updated_utc TEXT
        );
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS team_boxscore (
            game_id TEXT,
            team_side TEXT,              -- 'home' or 'away'
            team_abbrev TEXT,
            goals INTEGER,
            shots INTEGER,
            hits INTEGER,
            blocks INTEGER,
            pim INTEGER,
            giveaways INTEGER,
            takeaways INTEGER,
            faceoff_pct REAL,

            pp_goals INTEGER,
            pp_opps INTEGER,
            pp_pct REAL,

            -- derived from opponent PP (i.e., PK on the night) if available
            pk_pct REAL,

            -- special team shorthand: shorthanded goals for/against if present
            sh_goals_for INTEGER,
            sh_goals_against INTEGER,

            -- optional xG fields if present (some boxscores provide)
            x_goals REAL,
            x_goals_against REAL,

            -- additional "team totals" fields that often appear
            team_toi TEXT,

            PRIMARY KEY (game_id, team_side),
            FOREIGN KEY (game_id) REFERENCES games(game_id)
        );
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS goalie_boxscore (
            game_id TEXT,
            team_side TEXT,              -- 'home' or 'away'
            player_id INTEGER,
            player_name TEXT,
            toi TEXT,
            shots_against INTEGER,
            goals_against INTEGER,
            saves INTEGER,
            save_pct REAL,
            decision TEXT,
            PRIMARY KEY (game_id, team_side, player_id),
            FOREIGN KEY (game_id) REFERENCES games(game_id)
        );
        """
    )

    # Optional: keep raw JSON for debugging & future feature work
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS raw_game_json (
            game_id TEXT PRIMARY KEY,
            boxscore_json TEXT,
            landing_json TEXT,
            last_updated_utc TEXT,
            FOREIGN KEY (game_id) REFERENCES games(game_id)
        );
        """
    )

    conn.commit()

import json

def _as_text(v):
    """SQLite-friendly TEXT: dict/list -> compact JSON, leave str/int/None alone."""
    if v is None:
        return None
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False, separators=(",", ":"))
    return v  # keep ints as ints if they appear; sqlite can handle it

def upsert_game(conn, row: dict) -> None:
    params = (
        _as_text(row.get("game_id")),
        row.get("season"),
        _as_text(row.get("game_date")),
        _as_text(row.get("game_type")),
        _as_text(row.get("away_abbrev")),
        _as_text(row.get("away_name")),   # <-- was dict
        _as_text(row.get("home_abbrev")),
        _as_text(row.get("home_name")),   # <-- might also be dict
        row.get("away_score"),
        row.get("home_score"),
        _as_text(row.get("venue")),
        _as_text(row.get("game_state")),
        _as_text(row.get("start_time_utc")),
        _as_text(row.get("last_updated_utc")),
    )

    conn.execute(
        """
        INSERT INTO games (
            game_id, season, game_date, game_type,
            away_abbrev, away_name,
            home_abbrev, home_name,
            away_score, home_score,
            venue, game_state,
            start_time_utc, last_updated_utc
        )
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(game_id) DO UPDATE SET
            season=excluded.season,
            game_date=excluded.game_date,
            game_type=excluded.game_type,
            away_abbrev=excluded.away_abbrev,
            away_name=excluded.away_name,
            home_abbrev=excluded.home_abbrev,
            home_name=excluded.home_name,
            away_score=excluded.away_score,
            home_score=excluded.home_score,
            venue=excluded.venue,
            game_state=excluded.game_state,
            start_time_utc=excluded.start_time_utc,
            last_updated_utc=excluded.last_updated_utc
        """,
        params,
    )



def upsert_team_boxscore(conn: sqlite3.Connection, row: Dict[str, Any]) -> None:
    cols = [
        "game_id", "team_side", "team_abbrev",
        "goals", "shots", "hits", "blocks", "pim",
        "giveaways", "takeaways", "faceoff_pct",
        "pp_goals", "pp_opps", "pp_pct",
        "pk_pct", "sh_goals_for", "sh_goals_against",
        "x_goals", "x_goals_against", "team_toi",
    ]
    values = [row.get(c) for c in cols]
    placeholders = ",".join(["?"] * len(cols))
    updates = ",".join([f"{c}=excluded.{c}" for c in cols if c not in ("game_id", "team_side")])
    conn.execute(
        f"""
        INSERT INTO team_boxscore ({",".join(cols)}) VALUES ({placeholders})
        ON CONFLICT(game_id, team_side) DO UPDATE SET {updates};
        """,
        values,
    )


def upsert_goalie_boxscore(conn: sqlite3.Connection, row: Dict[str, Any]) -> None:
    cols = [
        "game_id", "team_side", "player_id", "player_name",
        "toi", "shots_against", "goals_against", "saves",
        "save_pct", "decision",
    ]
    values = [row.get(c) for c in cols]
    placeholders = ",".join(["?"] * len(cols))
    updates = ",".join([f"{c}=excluded.{c}" for c in cols if c not in ("game_id", "team_side", "player_id")])
    conn.execute(
        f"""
        INSERT INTO goalie_boxscore ({",".join(cols)}) VALUES ({placeholders})
        ON CONFLICT(game_id, team_side, player_id) DO UPDATE SET {updates};
        """,
        values,
    )


def upsert_raw_json(conn: sqlite3.Connection, game_id: str, boxscore: Any = None, landing: Any = None) -> None:
    now = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    conn.execute(
        """
        INSERT INTO raw_game_json (game_id, boxscore_json, landing_json, last_updated_utc)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(game_id) DO UPDATE SET
          boxscore_json=COALESCE(excluded.boxscore_json, raw_game_json.boxscore_json),
          landing_json=COALESCE(excluded.landing_json, raw_game_json.landing_json),
          last_updated_utc=excluded.last_updated_utc;
        """,
        (
            game_id,
            json.dumps(boxscore) if boxscore is not None else None,
            json.dumps(landing) if landing is not None else None,
            now,
        ),
    )


# ----------------------------
# Date helpers
# ----------------------------

def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def iter_dates(d0: date, d1: date) -> Iterable[date]:
    """Inclusive range."""
    cur = d0
    while cur <= d1:
        yield cur
        cur += timedelta(days=1)


def season_window(start_year: int) -> Tuple[date, date]:
    """
    NHL season "window" that comfortably covers regular season + playoffs.
    Example: start_year=2015 => window [2015-11-01, 2016-04-15]
    """
    return date(start_year, 11, 1), date(start_year + 1, 4, 15)


def season_id_from_year(start_year: int) -> int:
    # NHL API uses YYYYYYYY (e.g., 20232024) in many places.
    return int(f"{start_year}{start_year+1}")


# ----------------------------
# Step 1 — schedule + scores
# ----------------------------

def extract_games_from_score_payload(payload: Dict[str, Any], fallback_date: date) -> List[Dict[str, Any]]:
    """
    /v1/score/{date} format can evolve; we defensively traverse fields.
    We only need: game_id, teams, scores, game state, start time, venue (if present), season (if present).
    """
    games_out: List[Dict[str, Any]] = []

    # Most commonly: payload has "games": [...]
    games = payload.get("games")

    if not isinstance(games, list):
        gw = payload.get("gameWeek")

        if isinstance(gw, dict):
            games = gw.get("games") or []
        elif isinstance(gw, list):
            # sometimes it's a list of buckets, each containing "games"
            games = []
            for item in gw:
                if isinstance(item, dict) and isinstance(item.get("games"), list):
                    games.extend(item["games"])
        else:
            games = []

    for g in games:
        game_id = str(g.get("id") or g.get("gameId") or g.get("gamePk") or "")
        if not game_id:
            continue

        away = g.get("awayTeam") or g.get("away") or {}
        home = g.get("homeTeam") or g.get("home") or {}

        # abbreviations are usually like "BOS", "MTL", etc.
        away_abbrev = away.get("abbrev") or away.get("triCode") or away.get("abbreviation")
        home_abbrev = home.get("abbrev") or home.get("triCode") or home.get("abbreviation")

        away_name = away.get("name") or away.get("placeName", {}).get("default") or away.get("commonName", {}).get("default")
        home_name = home.get("name") or home.get("placeName", {}).get("default") or home.get("commonName", {}).get("default")

        away_score = away.get("score")
        home_score = home.get("score")

        # Some payloads include start time as "startTimeUTC"
        start_utc = g.get("startTimeUTC") or g.get("startTime") or g.get("gameDate")
        game_state = g.get("gameState") or g.get("status", {}).get("abstractGameState") or g.get("gameStatus")

        venue = None
        if "venue" in g and isinstance(g["venue"], dict):
            venue = g["venue"].get("default") or g["venue"].get("name")
        elif isinstance(g.get("venue"), str):
            venue = g.get("venue")

        game_date = (g.get("gameDate") or g.get("date") or fallback_date.isoformat())[:10]
        game_type = g.get("gameType") or g.get("gameTypeId") or g.get("type")  # often "R"/"P" etc.

        season = g.get("season") or g.get("seasonId")
        try:
            season = int(season) if season is not None else None
        except Exception:
            season = None

        games_out.append({
            "game_id": game_id,
            "season": season,
            "game_date": game_date,
            "game_type": str(game_type) if game_type is not None else None,
            "away_abbrev": away_abbrev,
            "away_name": away_name,
            "home_abbrev": home_abbrev,
            "home_name": home_name,
            "away_score": away_score,
            "home_score": home_score,
            "venue": venue,
            "game_state": game_state,
            "start_time_utc": start_utc,
            "last_updated_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        })

    return games_out


def run_step1(
    conn: sqlite3.Connection,
    session: requests.Session,
    start_year: int,
    end_year: int,
    date_min: Optional[date] = None,
    date_max: Optional[date] = None,
    endpoint: str = "score",
) -> None:
    """
    Pull daily games for each date in each season window.
    endpoint: "score" (default) or "schedule"
    """
    assert endpoint in ("score", "schedule"), "endpoint must be 'score' or 'schedule'"

    ensure_tables(conn)

    for y in range(start_year, end_year + 1):
        d0, d1 = season_window(y)
        if date_min:
            d0 = max(d0, date_min)
        if date_max:
            d1 = min(d1, date_max)
        if d0 > d1:
            continue

        print(f"[step1] season {y}-{y+1}: {d0}..{d1}")

        for d in iter_dates(d0, d1):
            url = f"{API_BASE}/v1/{endpoint}/{d.isoformat()}"
            payload = fetch_json(session, url)
            games = extract_games_from_score_payload(payload, fallback_date=d)

            if not games:
                continue

            cur = conn.cursor()
            for row in games:
                # if API did not provide season, infer from window
                if row["season"] is None:
                    row["season"] = season_id_from_year(y)
                upsert_game(conn, row)
            conn.commit()

            # be nice
            polite_sleep()


# ----------------------------
# Step 2 — boxscore -> team_boxscore (+ optional goalie_boxscore)
# ----------------------------

def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, str) and x.strip() == "":
            return None
        return float(x)
    except Exception:
        return None


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        if isinstance(x, bool):
            return int(x)
        if isinstance(x, str) and x.strip() == "":
            return None
        return int(x)
    except Exception:
        return None


def _extract_team_stats_from_boxscore(
    box: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns (away_team_row, home_team_row, away_goalies, home_goalies)

    This NHL "gamecenter boxscore" payload provides:
      - Team score + SOG at top-level: box["awayTeam"], box["homeTeam"]
      - Most other skater totals only at player-level: box["playerByGameStats"][side]["forwards"/"defense"]
      - Goalies at: box["playerByGameStats"][side]["goalies"]

    We aggregate skater totals by summing player stats. Some "team-level" stats
    like FO% and PP opportunities/PP% are NOT present here, so we set them to None.
    """

    away = box.get("awayTeam") or {}
    home = box.get("homeTeam") or {}

    away_abbrev = away.get("abbrev") or away.get("triCode") or away.get("abbreviation")
    home_abbrev = home.get("abbrev") or home.get("triCode") or home.get("abbreviation")

    pbg = box.get("playerByGameStats") or {}
    away_pbg = pbg.get("awayTeam") or {}
    home_pbg = pbg.get("homeTeam") or {}

    def _collect_skaters(team_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
        skaters: List[Dict[str, Any]] = []
        for grp in ("forwards", "defense"):
            arr = team_obj.get(grp) or []
            if isinstance(arr, list):
                skaters.extend([p for p in arr if isinstance(p, dict)])
        return skaters

    def _sum_stat(players: List[Dict[str, Any]], key: str) -> Optional[int]:
        total = 0
        seen = False
        for p in players:
            v = p.get(key)
            if isinstance(v, (int, float)):
                total += int(v)
                seen = True
        return total if seen else None

    away_skaters = _collect_skaters(away_pbg)
    home_skaters = _collect_skaters(home_pbg)

    # Team-level goals/SOG are available at top-level objects
    away_goals = _safe_int(away.get("score"))
    home_goals = _safe_int(home.get("score"))
    away_sog = _safe_int(away.get("sog"))
    home_sog = _safe_int(home.get("sog"))

    # Aggregate common boxscore totals from skaters
    away_hits = _sum_stat(away_skaters, "hits")
    home_hits = _sum_stat(home_skaters, "hits")

    away_blocks = _sum_stat(away_skaters, "blockedShots")
    home_blocks = _sum_stat(home_skaters, "blockedShots")

    away_pim = _sum_stat(away_skaters, "pim")
    home_pim = _sum_stat(home_skaters, "pim")

    away_give = _sum_stat(away_skaters, "giveaways")
    home_give = _sum_stat(home_skaters, "giveaways")

    away_take = _sum_stat(away_skaters, "takeaways")
    home_take = _sum_stat(home_skaters, "takeaways")

    away_ppg = _sum_stat(away_skaters, "powerPlayGoals")
    home_ppg = _sum_stat(home_skaters, "powerPlayGoals")

    # Shorthanded goals are not consistently present; try to sum if key exists
    away_shg = _sum_stat(away_skaters, "shorthandedGoals")  # usually not present
    home_shg = _sum_stat(home_skaters, "shorthandedGoals")

    def team_row(side: str, abbrev: Any, goals: Any, sog: Any,
                 hits: Any, blocks: Any, pim: Any, giveaways: Any, takeaways: Any,
                 pp_goals: Any, sh_goals: Any) -> Dict[str, Any]:
        return {
            "team_side": side,
            "team_abbrev": abbrev,
            "goals": goals,
            "shots": sog,                 # map "shots" column to SOG
            "hits": hits,
            "blocks": blocks,
            "pim": pim,
            "giveaways": giveaways,
            "takeaways": takeaways,
            "faceoff_pct": None,          # not available reliably at team-level in this payload
            "pp_goals": pp_goals,
            "pp_opps": None,              # not available here
            "pp_pct": None,               # not available here
            "pk_pct": None,               # can't derive without opp PP opps
            "sh_goals_for": sh_goals,
            "sh_goals_against": None,
            "x_goals": None,              # not available here
            "x_goals_against": None,
            "team_toi": None,             # not available here at team-level
        }

    away_row = team_row(
        "away", away_abbrev, away_goals, away_sog,
        away_hits, away_blocks, away_pim, away_give, away_take,
        away_ppg, away_shg
    )
    home_row = team_row(
        "home", home_abbrev, home_goals, home_sog,
        home_hits, home_blocks, home_pim, home_give, home_take,
        home_ppg, home_shg
    )

    # Goalies: best source is playerByGameStats side goalies list
    def parse_goalies(goalie_list: Any, side: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not goalie_list:
            return out
        if isinstance(goalie_list, dict):
            goalie_list = list(goalie_list.values())
        if not isinstance(goalie_list, list):
            return out

        for g in goalie_list:
            if not isinstance(g, dict):
                continue
            pid = g.get("playerId") or g.get("id")
            if pid is None:
                continue

            name = None
            if isinstance(g.get("name"), dict):
                name = g["name"].get("default")
            name = name or g.get("name") or g.get("fullName")

            out.append({
                "team_side": side,
                "player_id": int(pid),
                "player_name": name,
                "toi": g.get("toi") or g.get("timeOnIce"),
                "shots_against": _safe_int(g.get("shotsAgainst") or g.get("shots") or g.get("sa")),
                "goals_against": _safe_int(g.get("goalsAgainst") or g.get("goals") or g.get("ga")),
                "saves": _safe_int(g.get("saves")),
                "save_pct": _safe_float(g.get("savePctg") or g.get("savePercentage")),
                "decision": g.get("decision"),
            })
        return out

    away_goalies = parse_goalies(away_pbg.get("goalies"), "away")
    home_goalies = parse_goalies(home_pbg.get("goalies"), "home")

    return away_row, home_row, away_goalies, home_goalies



def list_game_ids(conn: sqlite3.Connection, only_missing_boxscore: bool = False) -> List[str]:
    if only_missing_boxscore:
        rows = conn.execute(
            """
            SELECT g.game_id
            FROM games g
            LEFT JOIN team_boxscore t ON t.game_id = g.game_id
            WHERE t.game_id IS NULL
            ORDER BY g.game_date;
            """
        ).fetchall()
    else:
        rows = conn.execute("SELECT game_id FROM games ORDER BY game_date;").fetchall()
    return [r[0] for r in rows]

def filter_game_ids_by_date(
    conn: sqlite3.Connection,
    game_ids: list[str],
    date_min: str | None,
    date_max: str | None,
) -> list[str]:
    if not date_min and not date_max:
        return game_ids

    q = """
        SELECT game_id
        FROM games
        WHERE game_id IN ({})
    """.format(",".join("?" * len(game_ids)))

    params: list[str] = list(game_ids)

    if date_min:
        q += " AND game_date >= ?"
        params.append(date_min)

    if date_max:
        q += " AND game_date <= ?"
        params.append(date_max)

    cur = conn.cursor()
    cur.execute(q, params)
    return [r[0] for r in cur.fetchall()]



def run_step2(
    conn: sqlite3.Connection,
    session: requests.Session,
    only_missing: bool = True,
    save_raw_json: bool = True,
    also_fetch_landing: bool = False,
    date_min: str | None = None,   # YYYY-MM-DD
    date_max: str | None = None,   # YYYY-MM-DD
) -> None:

    ensure_tables(conn)

    game_ids = list_game_ids(conn, only_missing_boxscore=only_missing)

    # Apply optional date window
    if date_min is not None and date_max is not None:
        game_ids = filter_game_ids_by_date(
            conn,
            game_ids,
            date_min=date_min,
            date_max=date_max,
        )

    print(
        f"[step2] games to process: {len(game_ids)} "
        f"(date_min={date_min}, date_max={date_max})"
    )

    for i, game_id in enumerate(game_ids, 1):
        url_box = f"{API_BASE}/v1/gamecenter/{game_id}/boxscore"
        box = fetch_json(session, url_box)

        away_row, home_row, away_goalies, home_goalies = _extract_team_stats_from_boxscore(box)

        away_row["game_id"] = str(game_id)
        home_row["game_id"] = str(game_id)

        upsert_team_boxscore(conn, away_row)
        upsert_team_boxscore(conn, home_row)

        for g in away_goalies:
            g["game_id"] = str(game_id)
            upsert_goalie_boxscore(conn, g)
        for g in home_goalies:
            g["game_id"] = str(game_id)
            upsert_goalie_boxscore(conn, g)

        if save_raw_json:
            landing = None
            if also_fetch_landing:
                url_land = f"{API_BASE}/v1/gamecenter/{game_id}/landing"
                landing = fetch_json(session, url_land)
            upsert_raw_json(conn, str(game_id), boxscore=box, landing=landing)

        conn.commit()

        if i % 50 == 0:
            print(f"[step2] processed {i}/{len(game_ids)} ...")

        polite_sleep()

from collections import defaultdict
import re

_CONV_RE = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\s*$")

def _parse_conv(x):
    """Parse '2/5' => (2,5) else (None,None)."""
    if x is None:
        return (None, None)
    m = _CONV_RE.match(str(x))
    if not m:
        return (None, None)
    return (int(m.group(1)), int(m.group(2)))

def _to_int(x):
    try:
        return int(x)
    except Exception:
        return None

def _pct_100(n, d):
    """Percent in 0..100."""
    if d in (None, 0) or n is None:
        return None
    return 100.0 * (float(n) / float(d))

from collections import defaultdict

def _extract_team_specialteams_from_pbp(pbp: dict) -> dict:
    away_id = (pbp.get("awayTeam") or {}).get("id")
    home_id = (pbp.get("homeTeam") or {}).get("id")

    plays = pbp.get("plays") or []
    if not away_id or not home_id or not isinstance(plays, list):
        return {"away": {}, "home": {}}

    # Faceoffs
    fo_total = 0
    fo_wins = {away_id: 0, home_id: 0}

    # Penalties grouped by timestamp
    pen_groups = defaultdict(list)

    # SH goals (optional)
    sh_goals = {away_id: 0, home_id: 0}

    # PP goals only used to compute pp_pct, but we will NOT store it
    pp_goals = {away_id: 0, home_id: 0}

    for p in plays:
        t = p.get("typeDescKey")
        details = p.get("details") or {}

        if t == "faceoff":
            winner = details.get("eventOwnerTeamId")
            if winner in (away_id, home_id):
                fo_total += 1
                fo_wins[winner] += 1

        elif t == "penalty":
            team = details.get("eventOwnerTeamId")
            type_code = details.get("typeCode")
            duration = details.get("duration")
            if team in (away_id, home_id) and type_code in ("MIN", "MAJ") and isinstance(duration, (int, float)) and duration > 0:
                key = ((p.get("periodDescriptor") or {}).get("number"), p.get("timeInPeriod"))
                pen_groups[key].append(team)

        elif t == "goal":
            scorer_team = details.get("eventOwnerTeamId")
            sc = p.get("situationCode")
            if scorer_team in (away_id, home_id) and isinstance(sc, str) and len(sc) >= 2:
                try:
                    away_skaters = int(sc[0])
                    home_skaters = int(sc[1])
                except Exception:
                    continue

                # Power play / short-handed logic
                if scorer_team == away_id:
                    if away_skaters > home_skaters:
                        pp_goals[away_id] += 1
                    elif away_skaters < home_skaters:
                        sh_goals[away_id] += 1
                else:
                    if home_skaters > away_skaters:
                        pp_goals[home_id] += 1
                    elif home_skaters < away_skaters:
                        sh_goals[home_id] += 1

    # Compute PP opps by netting coincidentals
    pp_opps = {away_id: 0, home_id: 0}
    for _, teams in pen_groups.items():
        a = sum(1 for x in teams if x == away_id)
        h = sum(1 for x in teams if x == home_id)
        m = min(a, h)
        a_rem = a - m
        h_rem = h - m
        if a_rem > 0:
            pp_opps[home_id] += a_rem
        if h_rem > 0:
            pp_opps[away_id] += h_rem

    def pct(n, d):
        if not d:
            return None
        return 100.0 * (n / d)

    away_fo_pct = pct(fo_wins[away_id], fo_total)
    home_fo_pct = pct(fo_wins[home_id], fo_total)

    away_pp_pct = pct(pp_goals[away_id], pp_opps[away_id])
    home_pp_pct = pct(pp_goals[home_id], pp_opps[home_id])

    # PK% = 100 * (1 - opp_pp_goals / opp_pp_opps)
    away_pk_pct = None if not pp_opps[home_id] else 100.0 * (1.0 - (pp_goals[home_id] / pp_opps[home_id]))
    home_pk_pct = None if not pp_opps[away_id] else 100.0 * (1.0 - (pp_goals[away_id] / pp_opps[away_id]))

    # Clamp to [0,100] to avoid weirdness
    def clamp01(x):
        if x is None:
            return None
        return float(max(0.0, min(100.0, x)))

    return {
        "away": {
            "faceoff_pct": away_fo_pct,
            "pp_opps": int(pp_opps[away_id]),
            "pp_pct": clamp01(away_pp_pct),
            "pk_pct": clamp01(away_pk_pct),
            "sh_goals_for": int(sh_goals[away_id]),
            "sh_goals_against": int(sh_goals[home_id]),
            # IMPORTANT: don't return pp_goals
        },
        "home": {
            "faceoff_pct": home_fo_pct,
            "pp_opps": int(pp_opps[home_id]),
            "pp_pct": clamp01(home_pp_pct),
            "pk_pct": clamp01(home_pk_pct),
            "sh_goals_for": int(sh_goals[home_id]),
            "sh_goals_against": int(sh_goals[away_id]),
        },
    }



def _update_team_boxscore_extras(conn, game_id: str, side: str, extras: dict) -> None:
    cur = conn.cursor()

    cur.execute(
        """
        UPDATE team_boxscore
        SET
            faceoff_pct = ?,
            pp_opps = ?,
            pp_pct = ?,
            pk_pct = ?,
            sh_goals_for = ?,
            sh_goals_against = ?,
            pp_goals = NULL
        WHERE game_id = ?
          AND team_side = ?
        """,
        (
            extras.get("faceoff_pct"),
            extras.get("pp_opps"),
            extras.get("pp_pct"),
            extras.get("pk_pct"),
            extras.get("sh_goals_for"),
            extras.get("sh_goals_against"),
            str(game_id),
            side,
        ),
    )


def season_code_from_year(y: int) -> int:
    # args.start=2015 => 20152016
    return int(f"{y}{y+1}")

def run_step2b(conn, args):
    """
    Step 2b:
    Backfill team-level special teams + faceoff stats from play-by-play.
    Respects args.start/args.end and args.date_min/date_max.
    """
    print("[step2b] backfilling team special teams + faceoffs from PBP")

    # season_min = season_code_from_year(args.start)
    # season_max = season_code_from_year(args.end)

    # If you pass date-min/date-max, prefer those filters too
    date_min = args.date_min
    date_max = args.date_max

    where = []
    params = []

    # season window
    # where.append("g.season BETWEEN ? AND ?")
    # params.extend([season_min, season_max])

    # optional date window
    if date_min:
        where.append("g.game_date >= ?")
        params.append(date_min)
    if date_max:
        where.append("g.game_date <= ?")
        params.append(date_max)

    where_sql = " AND ".join(where)

    cur = conn.cursor()
    cur.execute(f"""
        SELECT DISTINCT tb.game_id
        FROM team_boxscore tb
        JOIN games g
          ON g.game_id = tb.game_id
        WHERE ({where_sql})
          AND (
               tb.faceoff_pct IS NULL
            OR tb.pp_opps IS NULL
            OR tb.pp_pct IS NULL
            OR tb.pk_pct IS NULL
            OR tb.sh_goals_for IS NULL
            OR tb.sh_goals_against IS NULL
          )
        ORDER BY g.game_date, tb.game_id
    """, params)

    game_ids = [r[0] for r in cur.fetchall()]

    print(f"[step2b] games needing backfill: {len(game_ids)} "
        #   f"(season {season_min}–{season_max}"
          f"{', date ' + str(date_min) if date_min else ''}"
          f"{'..' + str(date_max) if date_max else ''})")

    session = requests.Session()

    for i, game_id in enumerate(game_ids, 1):
        try:
            url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
            r = session.get(url, timeout=30)
            r.raise_for_status()
            pbp = r.json()

            extras = _extract_team_specialteams_from_pbp(pbp)

            if "away" in extras:
                _update_team_boxscore_extras(conn, game_id, "away", extras["away"])
            if "home" in extras:
                _update_team_boxscore_extras(conn, game_id, "home", extras["home"])

            if i % 50 == 0:
                conn.commit()
                print(f"[step2b] processed {i}/{len(game_ids)}")

        except Exception as e:
            print(f"[step2b] failed game {game_id}: {e}")

    conn.commit()
    print("[step2b] done")


# ----------------------------
# CLI
# ----------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NHL scraper -> SQLite (schedule + boxscore).")
    p.add_argument("--db", default="nhl_scrape.sqlite", help="SQLite output path")
    p.add_argument("--start", type=int, default=2015, help="Start season YEAR (e.g., 2015 means 2015-16)")
    p.add_argument("--end", type=int, default=2025, help="End season YEAR (e.g., 2025 means 2025-26 window)")
    p.add_argument("--date-min", type=str, default=None, help="Override minimum date (YYYY-MM-DD)")
    p.add_argument("--date-max", type=str, default=None, help="Override maximum date (YYYY-MM-DD)")
    p.add_argument("--endpoint", choices=["score", "schedule"], default="score", help="Daily endpoint to use for step1")
    p.add_argument("--only-missing", action="store_true", help="Step2: only fetch games missing team_boxscore")
    p.add_argument("--no-raw", action="store_true", help="Step2: do not store raw JSON payloads")
    p.add_argument("--landing", action="store_true", help="Step2: also store landing JSON (extra request per game)")
    p.add_argument("steps", nargs="+", choices=["step1", "step2", "step2b"], help="Which steps to run")

    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    conn = init_sqlite(args.db)
    ensure_tables(conn)
    session = make_session()

    date_min = parse_date(args.date_min) if args.date_min else None
    date_max = parse_date(args.date_max) if args.date_max else None

    if "step1" in args.steps:
        run_step1(
            conn=conn,
            session=session,
            start_year=args.start,
            end_year=args.end,
            date_min=date_min,
            date_max=date_max,
            endpoint=args.endpoint,
        )

    if "step2" in args.steps:
        run_step2(
            conn,
            session,
            only_missing=args.only_missing,
            save_raw_json=not args.no_raw,
            also_fetch_landing=args.landing,
            date_min=args.date_min,
            date_max=args.date_max,
        )


    if "step2b" in args.steps:
        run_step2b(conn, args)



    print("done.")


if __name__ == "__main__":
    main()

# python nhl_scraper.py --db nhl_scrape.sqlite --date-min 2025-12-18 --date-max 2025-12-18 step1
# python nhl_scraper.py --db nhl_scrape.sqlite --date-min 2025-12-18 --date-max 2025-12-18 step2
# python nhl_scraper.py --db nhl_scrape.sqlite --date-min 2025-12-18 --date-max 2025-12-18 step2b

# python nhl_feature_builder.py build --start-year 2025 --end-year 2025  