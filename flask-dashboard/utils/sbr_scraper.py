"""
SportsBookReview Scraper
Scrapes opening and current lines from sportsbookreview.com
"""
import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

PREFERRED_BOOKS = ['fanduel', 'draftkings', 'betmgm', 'caesars', 'bet365', 'fanatics']


@dataclass
class GameLines:
    """Container for a game's line data."""
    game_id: int
    date: str
    home_team: str
    home_abbrev: str
    away_team: str
    away_abbrev: str
    status: str
    start_time: Optional[str] = None
    # Opening lines (from preferred book)
    open_home_ml: Optional[float] = None
    open_away_ml: Optional[float] = None
    # Current/closing lines (from preferred book)
    current_home_ml: Optional[float] = None
    current_away_ml: Optional[float] = None
    # Which book the lines are from
    book: Optional[str] = None
    # All books data for reference
    all_books: Optional[Dict] = None

    def to_dict(self) -> Dict:
        return {
            'game_id': self.game_id,
            'date': self.date,
            'home_team': self.home_team,
            'home_abbrev': self.home_abbrev,
            'away_team': self.away_team,
            'away_abbrev': self.away_abbrev,
            'status': self.status,
            'open_home_ml': self.open_home_ml,
            'open_away_ml': self.open_away_ml,
            'current_home_ml': self.current_home_ml,
            'current_away_ml': self.current_away_ml,
            'book': self.book,
        }


def scrape_nhl_odds(date: Optional[str] = None) -> Dict[str, Any]:
    """
    Scrape NHL odds from SportsBookReview.

    Args:
        date: Date string in YYYY-MM-DD format. Defaults to today.

    Returns:
        Dict with 'success', 'games' (list of GameLines), and 'error' if failed.
    """
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')

    url = f"https://www.sportsbookreview.com/betting-odds/nhl-hockey/?date={date}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        next_data = soup.find('script', id='__NEXT_DATA__')

        if not next_data:
            return {
                'success': False,
                'error': 'Could not find __NEXT_DATA__ on page',
                'games': []
            }

        data = json.loads(next_data.string)
        odds_tables = data.get('props', {}).get('pageProps', {}).get('oddsTables', [])

        if not odds_tables:
            return {
                'success': True,
                'games': [],
                'message': f'No games found for {date}'
            }

        games = []
        for table in odds_tables:
            game_rows = table.get('oddsTableModel', {}).get('gameRows', [])

            for game_row in game_rows:
                game_view = game_row.get('gameView', {})
                odds_views = game_row.get('oddsViews', [])

                # Extract game info
                home_team = game_view.get('homeTeam', {})
                away_team = game_view.get('awayTeam', {})

                game_lines = GameLines(
                    game_id=game_view.get('gameId'),
                    date=date,
                    home_team=home_team.get('fullName', ''),
                    home_abbrev=home_team.get('shortName', ''),
                    away_team=away_team.get('fullName', ''),
                    away_abbrev=away_team.get('shortName', ''),
                    status=game_view.get('gameStatusText', ''),
                    start_time=game_view.get('startDate', ''),
                    all_books={}
                )

                # Extract lines from each book
                for odds_view in odds_views:
                    if odds_view is None:
                        continue
                    book = odds_view.get('sportsbook', '')
                    opening = odds_view.get('openingLine') or {}
                    current = odds_view.get('currentLine') or {}

                    book_data = {
                        'open_home_ml': opening.get('homeOdds'),
                        'open_away_ml': opening.get('awayOdds'),
                        'current_home_ml': current.get('homeOdds'),
                        'current_away_ml': current.get('awayOdds'),
                    }
                    game_lines.all_books[book] = book_data

                # Pick preferred book with complete data
                for book in PREFERRED_BOOKS:
                    if book in game_lines.all_books:
                        book_data = game_lines.all_books[book]
                        if (book_data['current_home_ml'] is not None and
                            book_data['current_away_ml'] is not None):
                            game_lines.book = book
                            game_lines.open_home_ml = book_data['open_home_ml']
                            game_lines.open_away_ml = book_data['open_away_ml']
                            game_lines.current_home_ml = book_data['current_home_ml']
                            game_lines.current_away_ml = book_data['current_away_ml']
                            break

                games.append(game_lines)

        return {
            'success': True,
            'games': games,
            'date': date,
            'count': len(games)
        }

    except requests.RequestException as e:
        return {
            'success': False,
            'error': f'Request failed: {str(e)}',
            'games': []
        }
    except json.JSONDecodeError as e:
        return {
            'success': False,
            'error': f'Failed to parse JSON: {str(e)}',
            'games': []
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'Unexpected error: {str(e)}',
            'games': []
        }


def get_today_and_tomorrow_lines() -> Dict[str, Any]:
    """
    Get lines for both today (closing) and tomorrow (opening).

    Returns:
        Dict with 'today' and 'tomorrow' keys, each containing scrape results.
    """
    today = datetime.now().strftime('%Y-%m-%d')
    tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

    return {
        'today': scrape_nhl_odds(today),
        'tomorrow': scrape_nhl_odds(tomorrow)
    }


# Team abbreviation normalization (SBR uses different abbrevs than your DB)
ABBREV_MAP = {
    'WAS': 'WSH',
    'MON': 'MTL',
    'TB': 'TBL',
    'LA': 'LAK',
    'NJ': 'NJD',
    'SJ': 'SJS',
    'CAL': 'CGY',
    'CLB': 'CBJ',
    'NAS': 'NSH',
    'WIN': 'WPG',
    'UTAH': 'UTA',
}


def normalize_abbrev(abbrev: str) -> str:
    """Normalize team abbreviation to match database format."""
    abbrev = abbrev.upper().strip()
    return ABBREV_MAP.get(abbrev, abbrev)


if __name__ == '__main__':
    # Test the scraper
    result = scrape_nhl_odds('2026-01-01')
    print(f"Success: {result['success']}")
    print(f"Games found: {len(result['games'])}")

    for game in result['games']:
        print(f"\n{game.away_abbrev} @ {game.home_abbrev} ({game.book})")
        print(f"  Opening:  Home {game.open_home_ml}, Away {game.open_away_ml}")
        print(f"  Current:  Home {game.current_home_ml}, Away {game.current_away_ml}")
