#!/usr/bin/env python3
"""
Async NBA odds scraper for SportsbookReview (SBR), modeled after get_nhl_odds.py.

What it does
------------
- Loads an NBA schedule from one or more CSV files (like the provided nba-2023-UTC.csv / nba-2024-UTC.csv).
- Scrapes SBR odds tables for each date in a date range:
    * point spread (default page)
    * money line
    * totals
- Merges multiple odds types into a single JSON per game keyed by away/home matchup.

Schedule CSV expectations
-------------------------
Your CSV should have at least:
  - "Date" (e.g. "24/10/2023 23:30" in UTC)
  - "Home Team"
  - "Away Team"

Notes
-----
- SBR is a Next.js site; we parse odds from the embedded __NEXT_DATA__ JSON.
- Matching schedule games to SBR rows is done via normalized team names + a small alias map
  (e.g., "Los Angeles Lakers" <-> "LA Lakers").
"""

import argparse
import aiohttp
import asyncio
import json
import re
import random
import time
import functools
import os
from datetime import datetime, timedelta
from typing import Dict, Tuple, List, Optional

import pandas as pd
from tqdm import tqdm


NEXT_DATA_PATTERN = re.compile(
    r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
    re.DOTALL
)

# --- Team normalization / aliases ------------------------------------------------

def _base_normalize(s: str) -> str:
    s = (s or "").strip().lower()
    # standardize punctuation/whitespace
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# Common SBR display variants vs "full" names found in schedules.
# Canonicalize to a single form so schedule + odds rows match.
TEAM_ALIASES = {
    # Los Angeles teams
    "los angeles lakers": "la lakers",
    "la lakers": "la lakers",
    "los angeles clippers": "la clippers",
    "la clippers": "la clippers",

    # Sometimes SBR uses shortened city forms
    "oklahoma city thunder": "oklahoma city thunder",
    "golden state warriors": "golden state warriors",

    # 76ers formatting
    "philadelphia 76ers": "philadelphia 76ers",
    "philadelphia seventy sixers": "philadelphia 76ers",
    "philadelphia sixers": "philadelphia 76ers",
}

def normalize_team_name(name: str) -> str:
    n = _base_normalize(name)

    # quick fixes for some numeric team names
    n = n.replace("76ers", "76ers")

    # canonicalize via alias map (two-pass: exact then loose)
    if n in TEAM_ALIASES:
        return TEAM_ALIASES[n]

    # loose patterns
    if n.startswith("los angeles "):
        if "lakers" in n:
            return "la lakers"
        if "clippers" in n:
            return "la clippers"

    if "seventy sixers" in n or "sixers" in n:
        return "philadelphia 76ers"

    return n

# --- SBR odds parsing ------------------------------------------------------------

def extract_odds_data(odds: dict, odds_type: str):
    """Extract opening/current line subsets depending on odds_type."""
    opening_line = odds.get("openingLine", {}) or {}
    current_line = odds.get("currentLine", {}) or {}

    if odds_type == "moneyline":
        opening_keys = ["homeOdds", "awayOdds"]
        current_keys = ["homeOdds", "awayOdds"]
    elif odds_type == "pointspread":
        opening_keys = ["homeOdds", "awayOdds", "homeSpread", "awaySpread"]
        current_keys = ["homeOdds", "awayOdds", "homeSpread", "awaySpread"]
    else:  # totals
        opening_keys = ["overOdds", "underOdds", "total"]
        current_keys = ["overOdds", "underOdds", "total"]

    opening_line_cleaned = {k: opening_line.get(k) for k in opening_keys}
    current_line_cleaned = {k: current_line.get(k) for k in current_keys}

    return opening_line_cleaned, current_line_cleaned


def get_odds_url(date: str, odds_type: str) -> str:
    """
    Build SBR URL for NBA odds tables.

    Verified pages:
      - Point Spread (full game): https://www.sportsbookreview.com/betting-odds/nba-basketball/
      - Money Line (full game):   https://www.sportsbookreview.com/betting-odds/nba-basketball/money-line/full-game/
      - Totals (full game):       https://www.sportsbookreview.com/betting-odds/nba-basketball/totals/full-game/
    """
    base = "https://www.sportsbookreview.com/betting-odds/nba-basketball"

    if odds_type == "pointspread":
        return f"{base}/?date={date}"
    if odds_type == "moneyline":
        return f"{base}/money-line/full-game/?date={date}"
    if odds_type == "totals":
        return f"{base}/totals/full-game/?date={date}"

    raise ValueError(f"Unknown odds type: {odds_type}")


@functools.lru_cache(maxsize=64)
def _sleep_jitter(delay: float) -> float:
    # deterministic-ish jitter per delay bucket (cache reduces overhead)
    return delay + random.random() * min(1.0, delay)


async def get_html_async(
    session: aiohttp.ClientSession,
    url: str,
    semaphore: asyncio.Semaphore,
    retries: int = 3,
    base_delay: float = 2.0,
) -> Optional[str]:
    """Fetch HTML with basic retry/backoff and concurrency control."""
    for attempt in range(retries):
        async with semaphore:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": (
                    "text/html,application/xhtml+xml,application/xml;"
                    "q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8"
                ),
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
            }

            try:
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 200:
                        return await resp.text()
                    # Rate limiting / transient
                    if resp.status in (429, 500, 502, 503, 504):
                        delay = _sleep_jitter(base_delay * (2 ** attempt))
                        await asyncio.sleep(delay)
                        continue

                    # Other statuses: return body (useful for debugging; may still contain __NEXT_DATA__)
                    body = None
                    try:
                        body = await resp.text()
                    except Exception:
                        body = None
                    if os.environ.get("SBR_DEBUG") == "1":
                        snippet = (body or "")[:500]
                        print(f"[SBR_DEBUG] Non-200 status {resp.status} for {url}; body head: {snippet!r}")
                    return body
            except (asyncio.TimeoutError, aiohttp.ClientError):
                delay = _sleep_jitter(base_delay * (2 ** attempt))
                await asyncio.sleep(delay)

    return None


# --- Schedule loading ------------------------------------------------------------

def load_nba_schedule_csvs(
    csv_paths: List[str],
    start_date: str,
    end_date: str,
    schedule_tz: str = "America/New_York",
) -> Dict[str, Dict[Tuple[str, str], str]]:
    """
    Load schedule rows from one or more CSV files and return:
      { 'YYYY-MM-DD': { (away_norm, home_norm): gameType } }

    gameType is set to "R" (regular) by default since the provided seasons are regular season schedules.
    """
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()

    schedule_map: Dict[str, Dict[Tuple[str, str], str]] = {}

    for p in csv_paths:
        if not p:
            continue
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"[schedule] Failed to read {p}: {e}")
            continue

        needed = {"Date", "Home Team", "Away Team"}
        missing = needed - set(df.columns)
        if missing:
            print(f"[schedule] {p} is missing columns: {sorted(missing)} (skipping)")
            continue

        # Parse: the provided files are UTC timestamps in dd/mm/yyyy HH:MM format
        # Convert to schedule_tz and take the *local date* for matching SBR's date filter.
        ts = pd.to_datetime(df["Date"], dayfirst=True, utc=True, errors="coerce")
        if ts.isna().all():
            print(f"[schedule] Could not parse any dates in {p} (skipping)")
            continue

        local_dates = ts.dt.tz_convert(schedule_tz).dt.date

        for ld, home, away in zip(local_dates, df["Home Team"], df["Away Team"]):
            if ld is None or pd.isna(ld):
                continue
            if ld < start or ld > end:
                continue

            date_key = ld.strftime("%Y-%m-%d")
            away_n = normalize_team_name(str(away))
            home_n = normalize_team_name(str(home))
            if not away_n or not home_n:
                continue

            schedule_map.setdefault(date_key, {})[(away_n, home_n)] = "R"

    return schedule_map


# --- Scraping -------------------------------------------------------------------

def _parse_next_data(html: str) -> Optional[dict]:
    m = NEXT_DATA_PATTERN.search(html or "")
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None


def _extract_games_from_next_data(next_data: dict) -> List[dict]:
    """
    Extract SBR "gameRows" from embedded __NEXT_DATA__.

    SBR has (for years) exposed odds rows at:
      props.pageProps.oddsTables[0].oddsTableModel.gameRows

    But the exact nesting can drift, so we try a few fallbacks.
    """
    if not isinstance(next_data, dict):
        return []

    props = (((next_data.get("props") or {}).get("pageProps")) or {})

    # 1) Primary: props.pageProps.oddsTables[0].oddsTableModel.gameRows
    odds_tables = props.get("oddsTables")
    if isinstance(odds_tables, list) and odds_tables:
        ot0 = odds_tables[0] or {}
        game_rows = ((ot0.get("oddsTableModel") or {}).get("gameRows"))
        if isinstance(game_rows, list) and game_rows:
            return game_rows

    # 2) Sometimes nested under initialState / initialReduxState
    initial = (props.get("initialState") or props.get("initialReduxState") or props.get("initial_state") or {})
    odds_tables = initial.get("oddsTables")
    if isinstance(odds_tables, list) and odds_tables:
        ot0 = odds_tables[0] or {}
        game_rows = ((ot0.get("oddsTableModel") or {}).get("gameRows"))
        if isinstance(game_rows, list) and game_rows:
            return game_rows

    # 3) Heuristic fallback: search anywhere for a non-empty "gameRows" list.
    #    (We also accept the older oddsTableModel -> gameRows shape.)
    found: List[dict] = []

    def walk(obj):
        if isinstance(obj, dict):
            # direct gameRows
            gr_direct = obj.get("gameRows")
            if isinstance(gr_direct, list) and gr_direct:
                # sanity check: at least one element looks like a game row
                if isinstance(gr_direct[0], dict) and ("gameView" in gr_direct[0] or "oddsViews" in gr_direct[0]):
                    found.extend(gr_direct)

            # oddsTableModel -> gameRows
            if "oddsTableModel" in obj:
                gr = ((obj.get("oddsTableModel") or {}).get("gameRows"))
                if isinstance(gr, list) and gr:
                    found.extend(gr)
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    walk(props)
    return found


async def scrape_nba_odds_async(
    session: aiohttp.ClientSession,
    date: str,
    odds_type: str,
    game_type_map: Dict[str, Dict[Tuple[str, str], str]],
    semaphore: asyncio.Semaphore,
    base_delay: float = 2.0,
):
    url = get_odds_url(date, odds_type)
    html = await get_html_async(session, url, semaphore, base_delay=base_delay)

    if not html:
        return date, odds_type, []

    next_data = _parse_next_data(html)
    if not next_data:
        return date, odds_type, []

    game_rows = _extract_games_from_next_data(next_data)
    if not game_rows:
        # Some days may have no games; that's OK
        return date, odds_type, []

    cleaned_games = []

    for game in game_rows:
        try:
            game_view = game.get("gameView", {}) or {}
            away_full = (game_view.get("awayTeam", {}) or {}).get("fullName") or game_view.get("awayTeamName") or "Unknown"
            home_full = (game_view.get("homeTeam", {}) or {}).get("fullName") or game_view.get("homeTeamName") or "Unknown"

            away = normalize_team_name(away_full)
            home = normalize_team_name(home_full)

            game_key = f"{away}_vs_{home}"

            cleaned_game = {
                "gameKey": game_key,
                "gameView": {},
                "oddsViews": [],
            }

            # Copy some useful game view fields if present
            for key in ["startDate", "awayTeam", "awayTeamScore", "homeTeam", "homeTeamScore", "gameStatusText", "venueName"]:
                if key in game_view:
                    cleaned_game["gameView"][key] = game_view.get(key)

            cleaned_game["gameView"]["gameType"] = game_type_map.get(date, {}).get((away, home), "Unknown")

            cleaned_odds_views = []
            for odds in game.get("oddsViews", []) or []:
                if odds is None:
                    continue
                sportsbook = odds.get("sportsbook", "Unknown")
                try:
                    opening_line, current_line = extract_odds_data(odds, odds_type)
                except Exception:
                    continue

                cleaned_odds_views.append({
                    "sportsbook": sportsbook,
                    "openingLine": opening_line,
                    "currentLine": current_line,
                })

            cleaned_game["oddsViews"] = cleaned_odds_views
            cleaned_games.append(cleaned_game)

        except Exception:
            continue

    return date, odds_type, cleaned_games


def merge_odds_data(all_results, odds_types: List[str]):
    """Merge odds data from different types into a single per-date structure."""
    merged_data = {}

    # Group results by date
    date_results = {}
    for date, odds_type, games in all_results:
        date_results.setdefault(date, {})[odds_type] = games

    for date, odds_by_type in date_results.items():
        merged_games = {}

        # Match NHL output format: each game object has {gameView, odds}
        # We still use an internal game_key (away_vs_home) only for merging.
        for odds_type in odds_types:
            for game in odds_by_type.get(odds_type, []):
                game_key = game.get("gameKey")
                if not game_key:
                    continue

                if game_key not in merged_games:
                    merged_games[game_key] = {
                        "gameView": (game.get("gameView") or {}).copy(),
                        "odds": {},
                    }

                merged_games[game_key]["odds"][odds_type] = game.get("oddsViews", [])

        merged_data[date] = list(merged_games.values())

    return merged_data


async def scrape_range_async(
    start_date: str,
    end_date: str,
    fast: bool,
    max_concurrent: int,
    odds_types: List[str],
    schedule_files: List[str],
    schedule_tz: str,
):
    print("Loading NBA schedule...")
    game_type_map = load_nba_schedule_csvs(schedule_files, start_date, end_date, schedule_tz=schedule_tz)

    if not game_type_map:
        print("No games found in date range (check your schedule files / timezone).")
        return {}

    dates = sorted(game_type_map.keys())
    print(f"Found {len(dates)} dates to scrape for odds types: {', '.join(odds_types)}")

    semaphore = asyncio.Semaphore(max_concurrent)
    base_delay = 0.5 if fast else 2.0

    connector = aiohttp.TCPConnector(limit=max_concurrent, ttl_dns_cache=300)
    timeout = aiohttp.ClientTimeout(total=60)

    all_results = []
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = []
        for date in dates:
            for odds_type in odds_types:
                tasks.append(scrape_nba_odds_async(session, date, odds_type, game_type_map, semaphore, base_delay=base_delay))

        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Scraping"):
            res = await f
            all_results.append(res)

    merged = merge_odds_data(all_results, odds_types)
    return merged


def main():
    parser = argparse.ArgumentParser(description="Async NBA odds scraper (SportsbookReview).")
    parser.add_argument("start_date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("end_date", help="End date (YYYY-MM-DD)")
    parser.add_argument("-f", "--fast", action="store_true", help="Fast mode (reduced delays)")
    parser.add_argument("-c", "--concurrent", type=int, default=5, help="Max concurrent requests (default: 5)")
    parser.add_argument("-o", "--output", default="nba_odds.json", help="Output filename")
    parser.add_argument(
        "-t", "--types", nargs="+", default=["moneyline"],
        choices=["moneyline", "pointspread", "totals"],
        help="Types of odds to retrieve (can specify multiple)"
    )
    parser.add_argument(
        "-s", "--schedule-files", nargs="+", required=True,
        help="One or more NBA schedule CSVs (expects columns: Date, Home Team, Away Team)"
    )
    parser.add_argument(
        "--schedule-tz", default="America/New_York",
        help="Timezone used to convert schedule timestamps to local date (default: America/New_York)"
    )

    args = parser.parse_args()

    # Validate dates
    try:
        datetime.strptime(args.start_date, "%Y-%m-%d")
        datetime.strptime(args.end_date, "%Y-%m-%d")
    except ValueError:
        print("Invalid date format. Use YYYY-MM-DD.")
        return

    if args.concurrent < 1 or args.concurrent > 20:
        print("Concurrent requests should be between 1 and 20")
        return

    # Remove duplicates while preserving order
    seen = set()
    odds_types = []
    for t in args.types:
        if t not in seen:
            seen.add(t)
            odds_types.append(t)

    start_time = time.time()
    data = asyncio.run(scrape_range_async(
        args.start_date,
        args.end_date,
        args.fast,
        args.concurrent,
        odds_types,
        args.schedule_files,
        args.schedule_tz,
    ))

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    end_time = time.time()
    total_games = sum(len(games) for games in data.values())

    print(f"Scraped {total_games} games from {len(data)} dates")
    print(f"Runtime: {end_time - start_time:.2f} seconds")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
