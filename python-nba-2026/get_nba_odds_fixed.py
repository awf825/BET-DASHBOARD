#!/usr/bin/env python3
"""
NBA moneyline odds scraper for SportsbookReview (SBR).

Outputs JSON shaped like your nhl_odds.json:
{
  "YYYY-MM-DD": [
    {"gameView": {...}, "odds": {"moneyline": [{"sportsbook": "...", "openingLine": {...}, "currentLine": {...}}, ...]}}
  ],
  ...
}

Example:
  python get_nba_odds.py 2025-01-01 2025-01-07 -t moneyline -o nba_ml.json
"""

import argparse
import asyncio
import json
import random
import re
import sys
import time
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from tqdm import tqdm

BASE = "https://www.sportsbookreview.com/betting-odds/nba-basketball"
ML_URL_TEMPLATE = BASE + "/money-line/full-game/?date={date}"

NEXT_DATA_PATTERN = re.compile(
    r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
    re.DOTALL,
)


def daterange(start: str, end: str) -> List[str]:
    s = date.fromisoformat(start)
    e = date.fromisoformat(end)
    if e < s:
        raise ValueError("end_date must be >= start_date")
    out = []
    cur = s
    while cur <= e:
        out.append(cur.isoformat())
        cur += timedelta(days=1)
    return out


def parse_next_data(html: str) -> Optional[dict]:
    m = NEXT_DATA_PATTERN.search(html or "")
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None


def _walk(obj: Any):
    """Yield every nested object (dict/list) in a JSON-like structure."""
    stack = [obj]
    while stack:
        cur = stack.pop()
        yield cur
        if isinstance(cur, dict):
            stack.extend(cur.values())
        elif isinstance(cur, list):
            stack.extend(cur)


def find_game_rows(next_data: dict) -> List[dict]:
    """
    Find the odds table game rows. For NHL SBR this typically lives at:
      props.pageProps.oddsTables[0].oddsTableModel.gameRows

    NBA sometimes uses the same shape; if not, we search defensively.
    """
    if not isinstance(next_data, dict):
        return []

    props = (next_data.get("props") or {}).get("pageProps") or {}

    # 1) Try the common oddsTables path
    odds_tables = props.get("oddsTables")
    if isinstance(odds_tables, list) and odds_tables:
        ot0 = odds_tables[0] or {}
        otm = ot0.get("oddsTableModel") or {}
        game_rows = otm.get("gameRows")
        if isinstance(game_rows, list):
            return [r for r in game_rows if isinstance(r, dict)]

    # 2) Some pages store a single oddsTableModel
    otm = props.get("oddsTableModel")
    if isinstance(otm, dict) and isinstance(otm.get("gameRows"), list):
        return [r for r in otm["gameRows"] if isinstance(r, dict)]

    # 3) Fallback: search for any dict with a gameRows list of dicts containing gameView + oddsViews
    for node in _walk(props):
        if isinstance(node, dict) and "gameRows" in node and isinstance(node["gameRows"], list):
            rows = [r for r in node["gameRows"] if isinstance(r, dict)]
            if rows and isinstance(rows[0].get("gameView"), dict) and isinstance(rows[0].get("oddsViews"), list):
                return rows

    # 4) As a last resort, look for a list of row dicts (gameView+oddsViews) anywhere
    for node in _walk(props):
        if isinstance(node, list) and node:
            first = node[0]
            if isinstance(first, dict) and isinstance(first.get("gameView"), dict) and isinstance(first.get("oddsViews"), list):
                return [r for r in node if isinstance(r, dict)]

    return []


def extract_moneyline(odds: dict) -> Tuple[dict, dict]:
    opening = odds.get("openingLine") or {}
    current = odds.get("currentLine") or {}
    opening_clean = {k: opening.get(k) for k in ("homeOdds", "awayOdds")}
    current_clean = {k: current.get(k) for k in ("homeOdds", "awayOdds")}
    return opening_clean, current_clean


def sportsbook_name(ov: dict) -> str:
    sb = ov.get("sportsbook")
    if isinstance(sb, dict):
        return sb.get("name") or sb.get("shortName") or sb.get("key") or "Unknown"
    if isinstance(sb, str):
        return sb
    return "Unknown"


def normalize_game_output(row: dict) -> dict:
    gv = row.get("gameView") or {}
    odds_views = row.get("oddsViews") or []

    ml_list: List[dict] = []
    for ov in odds_views:
        if not isinstance(ov, dict):
            continue
        odds = ov.get("odds") if isinstance(ov.get("odds"), dict) else ov
        if not isinstance(odds, dict):
            continue
        opening, current = extract_moneyline(odds)
        ml_list.append(
            {
                "sportsbook": sportsbook_name(ov),
                "openingLine": opening,
                "currentLine": current,
            }
        )

    return {"gameView": gv, "odds": {"moneyline": ml_list}}


async def fetch_html(
    session: aiohttp.ClientSession,
    url: str,
    sem: asyncio.Semaphore,
    retries: int,
    base_delay: float,
    verbose: bool,
) -> Optional[str]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

    for attempt in range(1, retries + 1):
        try:
            async with sem:
                async with session.get(url, headers=headers) as r:
                    text = await r.text()
                    if r.status != 200:
                        if verbose:
                            print(f"[HTTP {r.status}] {url}", file=sys.stderr)
                            print(text[:400], file=sys.stderr)
                        raise RuntimeError(f"HTTP {r.status}")
                    return text
        except Exception as e:
            if attempt == retries:
                if verbose:
                    print(f"[FAIL] {url} :: {e}", file=sys.stderr)
                return None
            await asyncio.sleep(base_delay * attempt + random.random())
    return None


async def scrape_date(
    session: aiohttp.ClientSession,
    d: str,
    sem: asyncio.Semaphore,
    retries: int,
    base_delay: float,
    verbose: bool,
) -> Tuple[str, List[dict]]:
    url = ML_URL_TEMPLATE.format(date=d)
    html = await fetch_html(session, url, sem, retries, base_delay, verbose)
    if not html:
        return d, []
    nd = parse_next_data(html)
    if not nd:
        if verbose:
            print(f"[NO __NEXT_DATA__] {d}", file=sys.stderr)
        return d, []
    rows = find_game_rows(nd)
    if not rows:
        if verbose:
            print(f"[0 gameRows] {d}", file=sys.stderr)
        return d, []
    games = [normalize_game_output(r) for r in rows if isinstance(r, dict)]
    return d, games


async def scrape_range(
    start_date: str,
    end_date: str,
    concurrent: int,
    retries: int,
    fast: bool,
    verbose: bool,
) -> Dict[str, List[dict]]:
    dates = daterange(start_date, end_date)
    sem = asyncio.Semaphore(concurrent)
    base_delay = 0.25 if fast else 1.5

    timeout = aiohttp.ClientTimeout(total=60)
    connector = aiohttp.TCPConnector(limit=concurrent, ttl_dns_cache=300)

    results: Dict[str, List[dict]] = {}
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = [scrape_date(session, d, sem, retries, base_delay, verbose) for d in dates]
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Scraping"):
            d, games = await coro
            results[d] = games
    return results


def main():
    ap = argparse.ArgumentParser(description="Scrape NBA odds from SBR (moneyline).")
    ap.add_argument("start_date", help="YYYY-MM-DD")
    ap.add_argument("end_date", help="YYYY-MM-DD")
    ap.add_argument("-t", "--types", nargs="+", default=["moneyline"], help="Only supports: moneyline")
    ap.add_argument("-o", "--output", default="nba_odds.json", help="Output JSON file")
    ap.add_argument("-c", "--concurrent", type=int, default=5)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("-f", "--fast", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")

    args = ap.parse_args()
    if any(t != "moneyline" for t in args.types):
        print("This script currently supports only moneyline.", file=sys.stderr)
        sys.exit(2)

    t0 = time.time()
    data = asyncio.run(
        scrape_range(
            args.start_date,
            args.end_date,
            concurrent=args.concurrent,
            retries=args.retries,
            fast=args.fast,
            verbose=args.verbose,
        )
    )
    runtime = time.time() - t0
    total_games = sum(len(v) for v in data.values())

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Scraped {total_games} games from {len(data)} dates")
    print(f"Runtime: {runtime:.2f} seconds")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
