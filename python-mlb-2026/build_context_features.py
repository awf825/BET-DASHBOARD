#!/usr/bin/env python3
"""
build_context_features.py

Purpose
-------
Create and incrementally enrich a SQLite table `context_game` keyed by `gamePk`
with Option C contextual features:

1) Park factors (runs) from a local CSV (optional)
2) Weather at game time from Open-Meteo Archive API (no key) (optional but supported)
3) Starter handedness and starter IDs from MLB StatsAPI feed/live (supported)

This script is designed to work AFTER you've already scraped multiple seasons.
It discovers your existing `games_table_YYYY` tables and UPSERTs context rows
without requiring you to re-run the scraper.

Notes
-----
- Park factors: you provide a CSV via --park-csv. If omitted, park fields stay NULL.
- Weather: Open-Meteo archive is used; if unavailable or rate-limited, fields stay NULL.
- Starter handedness: fetched from MLB StatsAPI; cached locally in SQLite.

Run
---
python build_context_features.py --db mlb_scrape.sqlite
python build_context_features.py --db mlb_scrape.sqlite --seasons 2016-2025 --fill-starters --fill-weather --park-csv park_factors.csv

"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import requests
except ImportError as e:
    raise SystemExit("Missing dependency: requests. Install with: pip install requests") from e


STATSAPI_BASE = "https://statsapi.mlb.com"
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"


# ----------------------------
# Helpers: parsing / discovery
# ----------------------------

def parse_seasons_arg(s: str) -> List[int]:
    """
    Parse seasons like:
      "2016-2025"
      "2019,2021-2023"
      "2018"
    """
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
    """
    Return [(season, table_name), ...] for tables matching games_table_YYYY
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name LIKE 'games_table_%'
        ORDER BY name
    """)
    names = [r[0] for r in cur.fetchall()]
    out: List[Tuple[int, str]] = []
    for n in names:
        m = re.match(r"^games_table_(\d{4})$", n)
        if m:
            out.append((int(m.group(1)), n))
    return out


def table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    return [r[1] for r in cur.fetchall()]


def pick_col(cols: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def iso_to_utc_datetime(s: str) -> Optional[dt.datetime]:
    """
    Parse gameDate strings commonly found in MLB data:
    - '2024-04-01T23:10:00Z'
    - '2024-04-01T19:10:00-04:00'
    - '2024-04-01 23:10:00'
    Returns aware UTC datetime.
    """
    if not s:
        return None
    s = str(s).strip()
    try:
        # Normalize Z
        if s.endswith("Z"):
            return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(dt.timezone.utc)
        # If has offset
        if re.search(r"[+-]\d\d:\d\d$", s):
            return dt.datetime.fromisoformat(s).astimezone(dt.timezone.utc)
        # Naive datetime, assume UTC
        d = dt.datetime.fromisoformat(s)
        if d.tzinfo is None:
            d = d.replace(tzinfo=dt.timezone.utc)
        return d.astimezone(dt.timezone.utc)
    except Exception:
        return None


# ----------------------------
# SQLite schema + UPSERT
# ----------------------------

CONTEXT_GAME_DDL = """
CREATE TABLE IF NOT EXISTS context_game (
    gamePk INTEGER PRIMARY KEY,
    season INTEGER,
    gameDateUtc TEXT,

    homeTeamId INTEGER,
    awayTeamId INTEGER,

    venueId INTEGER,
    venueName TEXT,

    -- Starter IDs and handedness
    homeStarterId INTEGER,
    awayStarterId INTEGER,
    homeStarterThrows TEXT,   -- 'R' / 'L' / 'S' / NULL
    awayStarterThrows TEXT,

    -- Park factors (user-provided)
    parkFactorRuns REAL,

    -- Weather (Open-Meteo, UTC hour nearest game time)
    temperatureF REAL,
    windMph REAL,
    windDirDeg REAL,
    humidityPct REAL,
    precipMm REAL,
    pressureHpa REAL,

    created_at_utc TEXT DEFAULT (datetime('now')),
    updated_at_utc TEXT
);

CREATE INDEX IF NOT EXISTS idx_context_game_season ON context_game(season);
CREATE INDEX IF NOT EXISTS idx_context_game_venue ON context_game(venueId);
"""

PEOPLE_CACHE_DDL = """
CREATE TABLE IF NOT EXISTS people_cache (
    personId INTEGER PRIMARY KEY,
    payload_json TEXT,
    updated_at_utc TEXT DEFAULT (datetime('now'))
);
"""

VENUE_CACHE_DDL = """
CREATE TABLE IF NOT EXISTS venue_cache (
    venueId INTEGER PRIMARY KEY,
    payload_json TEXT,
    updated_at_utc TEXT DEFAULT (datetime('now'))
);
"""

WEATHER_CACHE_DDL = """
-- Cache Open-Meteo hourly payloads per (lat, lon, date) in UTC
CREATE TABLE IF NOT EXISTS weather_cache (
    cacheKey TEXT PRIMARY KEY,           -- f"{lat:.4f},{lon:.4f},{YYYY-MM-DD}"
    latitude REAL,
    longitude REAL,
    date_utc TEXT,
    payload_json TEXT,
    updated_at_utc TEXT DEFAULT (datetime('now'))
);
"""


UPSERT_CONTEXT_SQL = """
INSERT INTO context_game (
    gamePk, season, gameDate,
    homeTeamId, awayTeamId,
    venueId, venueName,
    home_sp_id, away_sp_id,
    home_sp_throws, away_sp_throws,
    park_pf_runs,
    temp_f, wind_mph, wind_dir_deg, humidity, precip_mm, pressure_hpa,
    updated_at_utc
) VALUES (
    :gamePk, :season, :gameDate,
    :homeTeamId, :awayTeamId,
    :venueId, :venueName,
    :home_sp_id, :away_sp_id,
    :home_sp_throws, :away_sp_throws,
    :park_pf_runs,
    :temp_f, :wind_mph, :wind_dir_deg, :humidity, :precip_mm, :pressure_hpa,
    datetime('now')
)
ON CONFLICT(gamePk) DO UPDATE SET
    season        = COALESCE(excluded.season, context_game.season),
    gameDate      = COALESCE(excluded.gameDate, context_game.gameDate),
    homeTeamId    = COALESCE(excluded.homeTeamId, context_game.homeTeamId),
    awayTeamId    = COALESCE(excluded.awayTeamId, context_game.awayTeamId),
    venueId       = COALESCE(excluded.venueId, context_game.venueId),
    venueName     = COALESCE(excluded.venueName, context_game.venueName),

    home_sp_id    = COALESCE(excluded.home_sp_id, context_game.home_sp_id),
    away_sp_id    = COALESCE(excluded.away_sp_id, context_game.away_sp_id),
    home_sp_throws= COALESCE(excluded.home_sp_throws, context_game.home_sp_throws),
    away_sp_throws= COALESCE(excluded.away_sp_throws, context_game.away_sp_throws),

    park_pf_runs  = COALESCE(excluded.park_pf_runs, context_game.park_pf_runs),

    temp_f        = COALESCE(excluded.temp_f, context_game.temp_f),
    wind_mph      = COALESCE(excluded.wind_mph, context_game.wind_mph),
    wind_dir_deg  = COALESCE(excluded.wind_dir_deg, context_game.wind_dir_deg),
    humidity      = COALESCE(excluded.humidity, context_game.humidity),
    precip_mm     = COALESCE(excluded.precip_mm, context_game.precip_mm),
    pressure_hpa  = COALESCE(excluded.pressure_hpa, context_game.pressure_hpa),

    updated_at_utc = datetime('now')
;
"""



def init_db(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    # cur.executescript(CONTEXT_GAME_DDL)
    cur.executescript(PEOPLE_CACHE_DDL)
    cur.executescript(VENUE_CACHE_DDL)
    cur.executescript(WEATHER_CACHE_DDL)
    conn.commit()


def upsert_context(conn: sqlite3.Connection, row: Dict[str, Any]) -> None:
    conn.execute(UPSERT_CONTEXT_SQL, row)


# ----------------------------
# Park factors (optional CSV)
# ----------------------------

def load_park_csv(path: str) -> Dict[Tuple[int, int], float]:
    """
    Load a park factors CSV and return {(season, venueId) or (season, teamId): pf_runs} mapping.

    Supported columns (case-insensitive):
      - season (required)
      - parkFactorRuns OR pf_runs OR runs_factor (required)
      - venueId OR venue_id (preferred)
      - homeTeamId OR teamId OR home_team_id (fallback)

    If venueId exists, we key by (season, venueId). Otherwise by (season, homeTeamId).
    """
    import csv
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return {}

    # normalize keys
    def g(row, key):
        for k in row.keys():
            if k.lower() == key.lower():
                return row[k]
        return None

    out: Dict[Tuple[int, int], float] = {}
    for r in rows:
        season_s = g(r, "season")
        pf_s = (
            g(r, "park_pf_runs")
            or g(r, "parkFactorRuns")
            or g(r, "pf_runs")
            or g(r, "runs_factor")
        )
        if not season_s or pf_s is None:
            continue
        try:
            season = int(season_s)
            pf = float(pf_s)
        except Exception:
            continue

        venue_s = g(r, "venueId") or g(r, "venue_id")
        team_s = g(r, "homeTeamId") or g(r, "teamId") or g(r, "home_team_id")
        if venue_s:
            try:
                vid = int(float(venue_s))
                out[(season, vid)] = pf
            except Exception:
                pass
        elif team_s:
            try:
                tid = int(float(team_s))
                out[(season, tid)] = pf
            except Exception:
                pass

    return out


# ----------------------------
# StatsAPI: starters + venues
# ----------------------------

class HttpClient:
    def __init__(self, timeout: int = 30, sleep_s: float = 0.05, max_retries: int = 3):
        self.timeout = timeout
        self.sleep_s = sleep_s
        self.max_retries = max_retries
        self.sess = requests.Session()
        self.sess.headers.update({
            "User-Agent": "python-mlb-context-features/1.0"
        })

    def get_json(self, url: str, params: Optional[dict] = None) -> dict:
        last_err = None
        for i in range(self.max_retries):
            try:
                r = self.sess.get(url, params=params, timeout=self.timeout)
                if r.status_code == 429:
                    time.sleep(1.0 + i)
                    continue
                r.raise_for_status()
                if self.sleep_s:
                    time.sleep(self.sleep_s)
                return r.json()
            except Exception as e:
                last_err = e
                time.sleep(0.5 + i * 0.5)
        raise RuntimeError(f"GET failed: {url} ({last_err})")


def get_person_cached(conn: sqlite3.Connection, http: HttpClient, person_id: int) -> dict:
    cur = conn.cursor()
    cur.execute("SELECT payload_json FROM people_cache WHERE personId = ?", (person_id,))
    row = cur.fetchone()
    if row and row[0]:
        return json.loads(row[0])

    payload = http.get_json(f"{STATSAPI_BASE}/api/v1/people/{person_id}")
    conn.execute(
        "INSERT OR REPLACE INTO people_cache(personId, payload_json, updated_at_utc) VALUES(?,?,datetime('now'))",
        (person_id, json.dumps(payload),)
    )
    conn.commit()
    return payload


def person_throws(person_payload: dict) -> Optional[str]:
    # StatsAPI response: {"people":[{"pitchHand":{"code":"R"}}]}
    try:
        p0 = person_payload.get("people", [])[0]
        ph = p0.get("pitchHand") or p0.get("throwingHand") or {}
        code = ph.get("code")
        if code:
            return str(code).upper()
    except Exception:
        pass
    return None


def get_venue_cached(conn: sqlite3.Connection, http: HttpClient, venue_id: int) -> dict:
    cur = conn.cursor()
    cur.execute("SELECT payload_json FROM venue_cache WHERE venueId = ?", (venue_id,))
    row = cur.fetchone()
    if row and row[0]:
        return json.loads(row[0])

    payload = http.get_json(
        f"{STATSAPI_BASE}/api/v1/venues/{venue_id}",
        params={"hydrate": "location,timezone"}
    )

    conn.execute(
        "INSERT OR REPLACE INTO venue_cache(venueId, payload_json, updated_at_utc) VALUES(?,?,datetime('now'))",
        (venue_id, json.dumps(payload),)
    )
    conn.commit()
    return payload


def venue_lat_lon_name(venue_payload: dict):
    try:
        v0 = venue_payload.get("venues", [])[0]
        name = v0.get("name")

        loc = v0.get("location") or {}

        # MLB StatsAPI commonly nests coordinates here:
        coords = loc.get("defaultCoordinates") or {}

        lat = coords.get("latitude", loc.get("latitude"))
        lon = coords.get("longitude", loc.get("longitude"))

        lat = float(lat) if lat is not None else None
        lon = float(lon) if lon is not None else None

        return lat, lon, name
    except Exception:
        return None, None, None



def get_game_live_feed(http: HttpClient, game_pk: int) -> dict:
    # v1.1 feed is commonly used in community code; v1 also works in many cases.
    url = f"{STATSAPI_BASE}/api/v1.1/game/{game_pk}/feed/live"
    return http.get_json(url)


def starters_and_venue_from_feed(feed: dict) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[str]]:
    """
    Return (homeStarterId, awayStarterId, venueId, venueName) if discoverable.
    Strategy:
      - Use boxscore teams.{home|away}.pitchers list (starter is typically first pitcher used)
      - Fall back to probablePitchers if present
      - Venue from gameData.venue
    """
    home_sp = away_sp = None

    # Venue
    venue_id = None
    venue_name = None
    try:
        vd = (feed.get("gameData") or {}).get("venue") or {}
        venue_id = vd.get("id")
        venue_name = vd.get("name")
        venue_id = int(venue_id) if venue_id is not None else None
    except Exception:
        pass

    # Starters from boxscore
    try:
        bs = feed.get("liveData", {}).get("boxscore", {})
        teams = bs.get("teams", {})
        home_pitchers = (teams.get("home", {}) or {}).get("pitchers") or []
        away_pitchers = (teams.get("away", {}) or {}).get("pitchers") or []
        if home_pitchers:
            home_sp = int(home_pitchers[0])
        if away_pitchers:
            away_sp = int(away_pitchers[0])
    except Exception:
        pass

    # Fallback: probablePitchers
    try:
        gd = feed.get("gameData", {}) or {}
        prob = gd.get("probablePitchers") or {}
        if home_sp is None and prob.get("home"):
            home_sp = int(prob["home"].get("id"))
        if away_sp is None and prob.get("away"):
            away_sp = int(prob["away"].get("id"))
    except Exception:
        pass

    return home_sp, away_sp, venue_id, venue_name


# ----------------------------
# Weather via Open-Meteo Archive
# ----------------------------

def cache_key(lat: float, lon: float, date_utc: str) -> str:
    return f"{lat:.4f},{lon:.4f},{date_utc}"


def get_weather_day_cached(
    conn: sqlite3.Connection,
    http: HttpClient,
    lat: float,
    lon: float,
    date_utc: str,
) -> Optional[dict]:
    """
    Fetch hourly weather for a UTC date and cache it.
    """
    key = cache_key(lat, lon, date_utc)
    cur = conn.cursor()
    cur.execute("SELECT payload_json FROM weather_cache WHERE cacheKey = ?", (key,))
    row = cur.fetchone()
    if row and row[0]:
        return json.loads(row[0])

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date_utc,
        "end_date": date_utc,
        "hourly": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "pressure_msl",
            "wind_speed_10m",
            "wind_direction_10m",
        ]),
        "timezone": "UTC",
    }
    try:
        payload = http.get_json(OPEN_METEO_ARCHIVE, params=params)
    except Exception:
        return None

    conn.execute(
        "INSERT OR REPLACE INTO weather_cache(cacheKey, latitude, longitude, date_utc, payload_json, updated_at_utc) "
        "VALUES(?,?,?,?,?,datetime('now'))",
        (key, lat, lon, date_utc, json.dumps(payload))
    )
    conn.commit()
    return payload


def nearest_hour_index(times: List[str], target: dt.datetime) -> Optional[int]:
    """
    times are ISO strings like '2024-04-01T23:00'
    target is aware UTC.
    """
    if not times:
        return None
    best_i = None
    best_dt = None
    for i, t in enumerate(times):
        try:
            # Open-Meteo returns no seconds, no Z; treat as UTC
            d = dt.datetime.fromisoformat(t).replace(tzinfo=dt.timezone.utc)
            delta = abs((d - target).total_seconds())
            if best_dt is None or delta < best_dt:
                best_dt = delta
                best_i = i
        except Exception:
            continue
    return best_i


def c_to_f(c: Optional[float]) -> Optional[float]:
    if c is None:
        return None
    return (c * 9.0 / 5.0) + 32.0


def ms_to_mph(ms: Optional[float]) -> Optional[float]:
    if ms is None:
        return None
    return ms * 2.2369362920544


def hpa_from_msl(maybe: Optional[float]) -> Optional[float]:
    if maybe is None:
        return None
    return float(maybe)


def extract_weather_at_time(payload: dict, game_dt_utc: dt.datetime) -> Dict[str, Any]:
    """
    From Open-Meteo hourly payload, pick nearest hour.
    """
    out = {
        "temp_f": None,
        "wind_mph": None,
        "wind_dir_deg": None,
        "humidity": None,
        "precip_mm": None,
        "pressure_hpa": None,
    }
    if not payload:
        return out
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    idx = nearest_hour_index(times, game_dt_utc)
    if idx is None:
        return out

    def getv(key):
        arr = hourly.get(key) or []
        if idx < len(arr):
            return arr[idx]
        return None

    temp_c = getv("temperature_2m")
    wind_ms = getv("wind_speed_10m")
    out["temp_f"] = c_to_f(float(temp_c)) if temp_c is not None else None
    out["wind_mph"] = ms_to_mph(float(wind_ms)) if wind_ms is not None else None
    out["wind_dir_deg"] = float(getv("wind_direction_10m")) if getv("wind_direction_10m") is not None else None
    out["humidity"] = float(getv("relative_humidity_2m")) if getv("relative_humidity_2m") is not None else None
    out["precip_mm"] = float(getv("precipitation")) if getv("precipitation") is not None else None
    out["pressure_hpa"] = hpa_from_msl(float(getv("pressure_msl"))) if getv("pressure_msl") is not None else None
    return out


# ----------------------------
# Main ETL
# ----------------------------

def iter_games(conn: sqlite3.Connection, seasons: List[int], limit: Optional[int]) -> Iterable[Dict[str, Any]]:
    """
    Yield game rows from games_table_YYYY with minimal fields needed.
    Tries to adapt to your column names automatically.
    """
    tables = dict(find_games_tables(conn))
    for season in seasons:
        t = tables.get(season)
        if not t:
            print(f"[warn] No games_table_{season} found; skipping.")
            continue

        cols = table_columns(conn, t)
        gamepk_col = pick_col(cols, ["gamePk", "game_pk", "game_id"])
        if not gamepk_col:
            print(f"[warn] {t}: no gamePk column found; skipping.")
            continue

        date_col = pick_col(cols, ["gameDate", "game_date", "date", "game_datetime", "startTime"])
        season_col = pick_col(cols, ["season", "year"])
        home_col = pick_col(cols, ["homeTeamId", "home_team_id", "home_id", "homeTeam"])
        away_col = pick_col(cols, ["awayTeamId", "away_team_id", "away_id", "awayTeam"])

        # Select only what's available
        select_cols = [gamepk_col]
        if season_col:
            select_cols.append(season_col)
        if date_col:
            select_cols.append(date_col)
        if home_col:
            select_cols.append(home_col)
        if away_col:
            select_cols.append(away_col)

        sql = f"SELECT {', '.join(select_cols)} FROM {t}"
        if limit:
            sql += f" LIMIT {int(limit)}"

        cur = conn.cursor()
        cur.execute(sql)
        for r in cur.fetchall():
            rec = dict(zip(select_cols, r))
            yield {
                "gamePk": int(rec.get(gamepk_col)),
                "season": int(rec.get(season_col)) if season_col and rec.get(season_col) is not None else season,
                "gameDateRaw": rec.get(date_col) if date_col else None,
                "homeTeamId": int(rec.get(home_col)) if home_col and rec.get(home_col) is not None else None,
                "awayTeamId": int(rec.get(away_col)) if away_col and rec.get(away_col) is not None else None,
            }


def process_games(
    conn: sqlite3.Connection,
    seasons: List[int],
    limit: Optional[int],
    fill_starters: bool,
    fill_weather: bool,
    park_map: Optional[Dict[Tuple[int, int], float]],
    batch_commit: int = 500,
) -> None:
    http = HttpClient()
    init_db(conn)

    n = 0
    for g in iter_games(conn, seasons, limit):
        game_pk = g["gamePk"]
        season = g["season"]
        game_dt = iso_to_utc_datetime(g.get("gameDateRaw") or "")
        gameDateUtc = game_dt.isoformat().replace("+00:00", "Z") if game_dt else None

        row = {
            "gamePk": game_pk,
            "season": season,
            "gameDate": g.get("gameDateRaw"),   # keep raw string or convert to ISO if you want
            "homeTeamId": g.get("homeTeamId"),
            "awayTeamId": g.get("awayTeamId"),
            "venueId": None,
            "venueName": None,
            "home_sp_id": None,
            "away_sp_id": None,
            "home_sp_throws": None,
            "away_sp_throws": None,
            "park_pf_runs": None,
            "temp_f": None,
            "wind_mph": None,
            "wind_dir_deg": None,
            "humidity": None,
            "precip_mm": None,
            "pressure_hpa": None,
        }


        venue_lat = venue_lon = None

        # Starters + venue from feed
        if fill_starters:
            try:
                feed = get_game_live_feed(http, game_pk)
                hsp, asp, vid, vname = starters_and_venue_from_feed(feed)
                row["home_sp_id"] = hsp
                row["away_sp_id"] = asp
                row["venueId"] = vid
                row["venueName"] = vname

                # Handedness
                if hsp:
                    p = get_person_cached(conn, http, int(hsp))
                    row["home_sp_throws"] = person_throws(p)
                if asp:
                    p = get_person_cached(conn, http, int(asp))
                    row["away_sp_throws"] = person_throws(p)

            except Exception as e:
                # leave NULLs; continue
                pass

        # Park factor: by (season, venueId) preferred, else by (season, homeTeamId)
        # print(f"[info] loaded park factors: {len(park_map):,} mappings")
        # print("[info] sample keys:", list(park_map.items())[:3])

        if park_map:
            # If we're not filling starters this run, venueId may be missing in `row`.
            # Pull existing venueId from context_game so park/weather passes can work.
            if row.get("venueId") is None:
                cur = conn.execute("SELECT venueId FROM context_game WHERE gamePk = ?", (game_pk,))
                existing = cur.fetchone()
                if existing and existing[0] is not None:
                    row["venueId"] = int(existing[0])
            pf = None
            if row.get("venueId") is not None:
                pf = park_map.get((season, int(row["venueId"])))
            if pf is None and row.get("homeTeamId") is not None:
                pf = park_map.get((season, int(row["homeTeamId"])))
            row["park_pf_runs"] = pf

        # Weather: needs venue lat/lon and game_dt
        if fill_weather and game_dt:
            # Need venueId first; if not, try to get from StatsAPI quickly
            if row.get("venueId") is None:
                try:
                    feed = get_game_live_feed(http, game_pk)
                    _, _, vid, vname = starters_and_venue_from_feed(feed)
                    row["venueId"] = vid
                    row["venueName"] = vname
                except Exception:
                    pass

            if row.get("venueId") is not None:
                try:
                    v_payload = get_venue_cached(conn, http, int(row["venueId"]))
                    venue_lat, venue_lon, vname2 = venue_lat_lon_name(v_payload)
                    if row.get("venueName") is None and vname2:
                        row["venueName"] = vname2
                except Exception:
                    venue_lat = venue_lon = None

            if venue_lat is not None and venue_lon is not None:
                date_utc = game_dt.date().isoformat()
                wp = get_weather_day_cached(conn, http, venue_lat, venue_lon, date_utc)
                wfeat = extract_weather_at_time(wp, game_dt)
                row.update(wfeat)

        upsert_context(conn, row)
        n += 1
        if n % 200 == 0:
            print(f"[progress] {n} games processed | weather_cache={conn.execute('SELECT COUNT(*) FROM weather_cache').fetchone()[0]}")

        if n % batch_commit == 0:
            conn.commit()
            print(f"[progress] upserted {n} games...")

    conn.commit()
    print(f"[done] upserted {n} games into context_game")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=str, default="mlb_scrape.sqlite", help="Path to SQLite DB (same as DB_PATH).")
    ap.add_argument("--seasons", type=str, default=None, help="Seasons like '2016-2025' or '2019,2021-2023'. Default: discover all.")
    ap.add_argument("--limit", type=int, default=None, help="Optional limit per-season for testing.")
    ap.add_argument("--fill-starters", action="store_true", help="Fetch starter IDs + handedness from StatsAPI feed/live.")
    ap.add_argument("--fill-weather", action="store_true", help="Fetch weather from Open-Meteo (requires venue lat/lon).")
    ap.add_argument("--park-csv", type=str, default=None, help="Optional park factor CSV to fill parkFactorRuns.")
    ap.add_argument("--batch-commit", type=int, default=500, help="Commit every N upserts.")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")

    discovered = find_games_tables(conn)
    if not discovered:
        raise SystemExit("No games_table_YYYY tables found in DB.")

    if args.seasons:
        seasons = parse_seasons_arg(args.seasons)
    else:
        seasons = [s for s, _ in discovered]

    park_map = None
    if args.park_csv:
        park_map = load_park_csv(args.park_csv)
        print(f"[info] loaded park factors: {len(park_map):,} mappings")

    process_games(
        conn=conn,
        seasons=seasons,
        limit=args.limit,
        fill_starters=args.fill_starters,
        fill_weather=args.fill_weather,
        park_map=park_map,
        batch_commit=args.batch_commit,
    )


if __name__ == "__main__":
    main()
