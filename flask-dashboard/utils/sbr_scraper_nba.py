"""
SportsBookReview NBA Spread Scraper
Scrapes opening and current point spreads from sportsbookreview.com
"""
import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Only use FanDuel - don't show games unless FanDuel has posted lines
REQUIRED_BOOK = 'fanduel'


@dataclass
class NBAGameSpreads:
    """Container for an NBA game's spread data."""
    game_id: int
    date: str
    home_team: str
    home_abbrev: str
    away_team: str
    away_abbrev: str
    status: str
    start_time: Optional[str] = None
    # Opening spreads (from preferred book)
    open_home_spread: Optional[float] = None
    open_away_spread: Optional[float] = None
    open_home_odds: Optional[float] = None  # juice (-110, etc)
    open_away_odds: Optional[float] = None
    # Current spreads (from preferred book)
    current_home_spread: Optional[float] = None
    current_away_spread: Optional[float] = None
    current_home_odds: Optional[float] = None
    current_away_odds: Optional[float] = None
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
            'start_time': self.start_time,
            'open_home_spread': self.open_home_spread,
            'open_away_spread': self.open_away_spread,
            'open_home_odds': self.open_home_odds,
            'open_away_odds': self.open_away_odds,
            'current_home_spread': self.current_home_spread,
            'current_away_spread': self.current_away_spread,
            'current_home_odds': self.current_home_odds,
            'current_away_odds': self.current_away_odds,
            'book': self.book,
        }


def scrape_nba_spreads(date: Optional[str] = None) -> Dict[str, Any]:
    """
    Scrape NBA point spreads from SportsBookReview.

    Args:
        date: Date string in YYYY-MM-DD format. Defaults to today.

    Returns:
        Dict with 'success', 'games' (list of NBAGameSpreads), and 'error' if failed.
    """
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')

    url = f"https://www.sportsbookreview.com/betting-odds/nba-basketball/pointspread/?date={date}"
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

                game_spreads = NBAGameSpreads(
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

                # Extract spreads from each book
                for odds_view in odds_views:
                    if odds_view is None:
                        continue
                    book = odds_view.get('sportsbook', '')
                    opening = odds_view.get('openingLine') or {}
                    current = odds_view.get('currentLine') or {}

                    book_data = {
                        'open_home_spread': opening.get('homeSpread'),
                        'open_away_spread': opening.get('awaySpread'),
                        'open_home_odds': opening.get('homeOdds'),
                        'open_away_odds': opening.get('awayOdds'),
                        'current_home_spread': current.get('homeSpread'),
                        'current_away_spread': current.get('awaySpread'),
                        'current_home_odds': current.get('homeOdds'),
                        'current_away_odds': current.get('awayOdds'),
                    }
                    game_spreads.all_books[book] = book_data

                # Only use FanDuel - don't fall back to other books
                if REQUIRED_BOOK in game_spreads.all_books:
                    book_data = game_spreads.all_books[REQUIRED_BOOK]
                    if (book_data['current_home_spread'] is not None and
                        book_data['current_away_spread'] is not None):
                        game_spreads.book = REQUIRED_BOOK
                        game_spreads.open_home_spread = book_data['open_home_spread']
                        game_spreads.open_away_spread = book_data['open_away_spread']
                        game_spreads.open_home_odds = book_data['open_home_odds']
                        game_spreads.open_away_odds = book_data['open_away_odds']
                        game_spreads.current_home_spread = book_data['current_home_spread']
                        game_spreads.current_away_spread = book_data['current_away_spread']
                        game_spreads.current_home_odds = book_data['current_home_odds']
                        game_spreads.current_away_odds = book_data['current_away_odds']
                # If FanDuel doesn't have lines, spreads remain None (will show "No spreads available")

                games.append(game_spreads)

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


def get_today_and_tomorrow_spreads() -> Dict[str, Any]:
    """
    Get spreads for both today and tomorrow.

    Returns:
        Dict with 'today' and 'tomorrow' keys, each containing scrape results.
    """
    today = datetime.now().strftime('%Y-%m-%d')
    tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

    return {
        'today': scrape_nba_spreads(today),
        'tomorrow': scrape_nba_spreads(tomorrow)
    }


# Team name normalization (SBR fullName -> our model's team names)
NBA_TEAM_NAME_MAP = {
    'LA Clippers': 'Los Angeles Clippers',
    'L.A. Clippers': 'Los Angeles Clippers',
    # Add more mappings as needed
}

# Abbreviation normalization
NBA_ABBREV_MAP = {
    'BK': 'BKN',
    'BRK': 'BKN',
    'PHO': 'PHX',
    'SA': 'SAS',
    'NO': 'NOP',
    'NY': 'NYK',
    'GS': 'GSW',
    'UTAH': 'UTA',
}


def normalize_team_name(name: str) -> str:
    """Normalize team name to match model's format."""
    name = name.strip()
    return NBA_TEAM_NAME_MAP.get(name, name)


def normalize_abbrev(abbrev: str) -> str:
    """Normalize team abbreviation."""
    abbrev = abbrev.upper().strip()
    return NBA_ABBREV_MAP.get(abbrev, abbrev)


if __name__ == '__main__':
    # Test the scraper
    result = scrape_nba_spreads('2026-01-02')
    print(f"Success: {result['success']}")
    print(f"Games found: {len(result['games'])}")

    for game in result['games']:
        print(f"\n{game.away_abbrev} @ {game.home_abbrev} ({game.book})")
        print(f"  Opening:  Home {game.open_home_spread} ({game.open_home_odds}), Away {game.open_away_spread} ({game.open_away_odds})")
        print(f"  Current:  Home {game.current_home_spread} ({game.current_home_odds}), Away {game.current_away_spread} ({game.current_away_odds})")
