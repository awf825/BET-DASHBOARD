"""
NBA Predictions Runner
Wrapper to run NBA spread predictions from the Flask app.
Uses SBR scraper to get both opening and current spreads.
"""
import sys
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd

# Add the NBA project to path
NBA_PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../python-nba-2026'))
sys.path.insert(0, NBA_PROJECT_PATH)

# Import from the NBA project
try:
    import joblib
    from nba_api.stats.endpoints import leaguedashteamstats, commonteamroster, teamgamelog, leaguegamefinder, playergamelog, commonallplayers
    from src.Utils.Dictionaries import team_index_current, TEAM_TO_CODE
    NBA_AVAILABLE = True
except ImportError as e:
    NBA_AVAILABLE = False
    IMPORT_ERROR = str(e)

# Cache for player name -> ID lookup
_player_id_cache = {}

# Import our SBR scraper
from utils.sbr_scraper_nba import scrape_nba_spreads, normalize_team_name

# Import lineup scraper for injury alerts
try:
    from utils.lineup_scraper import get_all_lineups, get_injured_starters_with_fallback
    LINEUPS_AVAILABLE = True
except ImportError:
    LINEUPS_AVAILABLE = False
    def get_all_lineups():
        return {}
    def get_injured_starters_with_fallback(team_abbrev, starters=None):
        return {'source': 'none', 'injured_starters': []}

# Team ID mapping (team name -> NBA API team ID)
TEAM_NAME_TO_ID = {
    'Atlanta Hawks': 1610612737,
    'Boston Celtics': 1610612738,
    'Brooklyn Nets': 1610612751,
    'Charlotte Hornets': 1610612766,
    'Chicago Bulls': 1610612741,
    'Cleveland Cavaliers': 1610612739,
    'Dallas Mavericks': 1610612742,
    'Denver Nuggets': 1610612743,
    'Detroit Pistons': 1610612765,
    'Golden State Warriors': 1610612744,
    'Houston Rockets': 1610612745,
    'Indiana Pacers': 1610612754,
    'Los Angeles Clippers': 1610612746,
    'LA Clippers': 1610612746,
    'Los Angeles Lakers': 1610612747,
    'Memphis Grizzlies': 1610612763,
    'Miami Heat': 1610612748,
    'Milwaukee Bucks': 1610612749,
    'Minnesota Timberwolves': 1610612750,
    'New Orleans Pelicans': 1610612740,
    'New York Knicks': 1610612752,
    'Oklahoma City Thunder': 1610612760,
    'Orlando Magic': 1610612753,
    'Philadelphia 76ers': 1610612755,
    'Phoenix Suns': 1610612756,
    'Portland Trail Blazers': 1610612757,
    'Sacramento Kings': 1610612758,
    'San Antonio Spurs': 1610612759,
    'Toronto Raptors': 1610612761,
    'Utah Jazz': 1610612762,
    'Washington Wizards': 1610612764,
}

# Team name -> abbreviation mapping (for lineup lookup)
TEAM_NAME_TO_ABBREV = {
    'Atlanta Hawks': 'ATL',
    'Boston Celtics': 'BOS',
    'Brooklyn Nets': 'BKN',
    'Charlotte Hornets': 'CHA',
    'Chicago Bulls': 'CHI',
    'Cleveland Cavaliers': 'CLE',
    'Dallas Mavericks': 'DAL',
    'Denver Nuggets': 'DEN',
    'Detroit Pistons': 'DET',
    'Golden State Warriors': 'GSW',
    'Houston Rockets': 'HOU',
    'Indiana Pacers': 'IND',
    'Los Angeles Clippers': 'LAC',
    'LA Clippers': 'LAC',
    'Los Angeles Lakers': 'LAL',
    'Memphis Grizzlies': 'MEM',
    'Miami Heat': 'MIA',
    'Milwaukee Bucks': 'MIL',
    'Minnesota Timberwolves': 'MIN',
    'New Orleans Pelicans': 'NOP',
    'New York Knicks': 'NYK',
    'Oklahoma City Thunder': 'OKC',
    'Orlando Magic': 'ORL',
    'Philadelphia 76ers': 'PHI',
    'Phoenix Suns': 'PHX',
    'Portland Trail Blazers': 'POR',
    'Sacramento Kings': 'SAC',
    'San Antonio Spurs': 'SAS',
    'Toronto Raptors': 'TOR',
    'Utah Jazz': 'UTA',
    'Washington Wizards': 'WAS',
}

# Paths to NBA resources
NBA_MODEL_PATH = os.path.join(NBA_PROJECT_PATH, 'Models/homewin_logreg_final.joblib')
NBA_SCHEDULE_PATH = os.path.join(NBA_PROJECT_PATH, 'Data/nba-2025-UTC.csv')

# Default strategy parameters
DEFAULT_PARAMS = {
    'bankroll': 5000.0,
    'kelly_scale': 0.10,
    'min_edge': 0.03,  # 3% edge threshold for spreads
    'max_stake_frac': 0.02,
}


def check_nba_available() -> Dict[str, Any]:
    """Check if NBA predictions are available."""
    if not NBA_AVAILABLE:
        return {
            'available': False,
            'error': f'NBA module not available: {IMPORT_ERROR}'
        }

    if not os.path.exists(NBA_MODEL_PATH):
        return {
            'available': False,
            'error': f'NBA model not found: {NBA_MODEL_PATH}'
        }

    if not os.path.exists(NBA_SCHEDULE_PATH):
        return {
            'available': False,
            'error': f'NBA schedule not found: {NBA_SCHEDULE_PATH}'
        }

    return {'available': True}


def get_date_string(date_type: str) -> str:
    """Get date string for today or tomorrow."""
    today = datetime.now()
    if date_type == 'tomorrow':
        target = today + timedelta(days=1)
    else:
        target = today
    return target.strftime('%Y-%m-%d')


def calculate_days_rest(team_name: str, target_date: datetime, schedule_df: pd.DataFrame) -> int:
    """
    Calculate days rest for a team relative to a target date.

    Args:
        team_name: Full team name (e.g., 'Los Angeles Lakers')
        target_date: The date of the game we're predicting
        schedule_df: DataFrame with schedule data

    Returns:
        Number of days rest (1 = back-to-back, 2 = one day off, etc.)
    """
    # Get all games for this team
    team_games = schedule_df[
        (schedule_df['Home Team'] == team_name) |
        (schedule_df['Away Team'] == team_name)
    ]

    # Get most recent game BEFORE the target date
    previous_games = team_games[team_games['Date'] < target_date].sort_values('Date', ascending=False)

    if len(previous_games) > 0:
        last_game_date = previous_games.iloc[0]['Date']
        # Days rest = target_date - last_game_date (1 means back-to-back)
        days_rest = (target_date - last_game_date).days
        return max(1, days_rest)
    else:
        # No previous game found, assume well-rested
        return 7


def get_teams_playing_on_date(target_date: str) -> set:
    """
    Get set of team abbreviations playing on a given date.
    Used to detect back-to-back games.
    """
    scrape_result = scrape_nba_spreads(target_date)
    if not scrape_result['success']:
        return set()

    teams = set()
    for game in scrape_result['games']:
        teams.add(game.home_abbrev)
        teams.add(game.away_abbrev)

    return teams


def implied_prob_from_spread(spread: float) -> float:
    """
    Convert spread to implied win probability.
    Uses approximate relationship: each point of spread ~ 2.8% win probability.
    Spread of 0 = 50% win probability.
    Negative spread (favorite) = higher win probability.
    """
    # Home team spread: negative means favorite
    # -7 spread = roughly 70% win prob
    prob = 0.5 - (spread * 0.028)
    return max(0.01, min(0.99, prob))


def calculate_edge(p_model: float, spread: float) -> float:
    """
    Calculate edge: model probability - implied probability from spread.
    Positive edge means model thinks team is undervalued.
    """
    implied = implied_prob_from_spread(spread)
    return p_model - implied


def fetch_with_retry(fetch_func, max_retries=3, delay=5):
    """Wrapper to retry NBA API calls with delay between attempts."""
    last_error = None
    for attempt in range(max_retries):
        try:
            result = fetch_func()
            # Add delay after successful call to avoid rate limiting
            time.sleep(1)
            return result
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(delay)
    raise last_error


def get_player_id(player_name: str) -> Optional[int]:
    """Look up player ID from name. Uses cache to avoid repeated API calls."""
    global _player_id_cache

    # Check cache first
    if player_name in _player_id_cache:
        return _player_id_cache[player_name]

    try:
        # Load all players if cache is empty
        if not _player_id_cache:
            def fetch():
                return commonallplayers.CommonAllPlayers(is_only_current_season=1)
            players = fetch_with_retry(fetch, max_retries=2, delay=3)
            df = players.get_data_frames()[0]

            # Build cache from full names
            for _, row in df.iterrows():
                full_name = row['DISPLAY_FIRST_LAST']
                _player_id_cache[full_name] = row['PERSON_ID']

        # Try exact match first
        if player_name in _player_id_cache:
            return _player_id_cache[player_name]

        # Try partial match (for names like "K. Towns" -> "Karl-Anthony Towns")
        for cached_name, player_id in _player_id_cache.items():
            # Match last name
            if player_name.split()[-1].lower() in cached_name.lower():
                # Check first initial if present
                if '.' in player_name:
                    first_initial = player_name.split('.')[0].strip()
                    if cached_name.lower().startswith(first_initial.lower()):
                        _player_id_cache[player_name] = player_id
                        return player_id
                else:
                    _player_id_cache[player_name] = player_id
                    return player_id

        return None
    except Exception:
        return None


def get_record_with_without_player(team_id: int, player_name: str) -> Dict[str, Any]:
    """
    Calculate team's record with and without a specific player.
    Returns dict with 'with_record' and 'without_record' strings.
    """
    try:
        player_id = get_player_id(player_name)
        if not player_id:
            return {'with_record': None, 'without_record': None}

        # Get player's games this season
        def fetch_player():
            return playergamelog.PlayerGameLog(player_id=player_id, season='2025-26')
        player_log = fetch_with_retry(fetch_player, max_retries=2, delay=3)
        player_df = player_log.get_data_frames()[0]
        player_games = set(player_df['Game_ID'].tolist())

        # Get team's games this season
        def fetch_team():
            return leaguegamefinder.LeagueGameFinder(
                team_id_nullable=team_id,
                season_nullable='2025-26',
                season_type_nullable='Regular Season'
            )
        team_finder = fetch_with_retry(fetch_team, max_retries=2, delay=3)
        team_df = team_finder.get_data_frames()[0]

        # Calculate records
        with_player = team_df[team_df['GAME_ID'].isin(player_games)]
        without_player = team_df[~team_df['GAME_ID'].isin(player_games)]

        with_wins = len(with_player[with_player['WL'] == 'W'])
        with_losses = len(with_player[with_player['WL'] == 'L'])
        without_wins = len(without_player[without_player['WL'] == 'W'])
        without_losses = len(without_player[without_player['WL'] == 'L'])

        return {
            'with_record': f"{with_wins}-{with_losses}",
            'without_record': f"{without_wins}-{without_losses}" if (without_wins + without_losses) > 0 else "N/A"
        }
    except Exception:
        return {'with_record': None, 'without_record': None}


# Stats to compare with/without player
STAT_DIFF_COLUMNS = [
    'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT',
    'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB',
    'AST', 'TOV', 'STL', 'BLK', 'BLKA', 'PF', 'PFD', 'PTS', 'PLUS_MINUS'
]


def get_stats_with_without_player(team_id: int, player_name: str) -> Dict[str, Any]:
    """
    Calculate team's stat averages with and without a specific player.

    Returns dict with:
    - 'with_player': dict of stat averages when player plays
    - 'without_player': dict of stat averages when player is out
    - 'diff': dict of differences (with - without)
    - 'sample_size_with': number of games with player
    - 'sample_size_without': number of games without player
    - 'with_record': W-L record with player
    - 'without_record': W-L record without player
    """
    try:
        player_id = get_player_id(player_name)
        if not player_id:
            return {
                'error': f'Player not found: {player_name}',
                'with_player': None,
                'without_player': None,
                'diff': None,
                'sample_size_with': 0,
                'sample_size_without': 0
            }

        # Get player's games this season
        def fetch_player():
            return playergamelog.PlayerGameLog(player_id=player_id, season='2025-26')
        player_log = fetch_with_retry(fetch_player, max_retries=2, delay=3)
        player_df = player_log.get_data_frames()[0]
        player_games = set(player_df['Game_ID'].tolist())

        # Get team's games this season with full stats
        def fetch_team():
            return leaguegamefinder.LeagueGameFinder(
                team_id_nullable=team_id,
                season_nullable='2025-26',
                season_type_nullable='Regular Season'
            )
        team_finder = fetch_with_retry(fetch_team, max_retries=2, delay=3)
        team_df = team_finder.get_data_frames()[0]

        # Split games with/without player
        with_player_df = team_df[team_df['GAME_ID'].isin(player_games)]
        without_player_df = team_df[~team_df['GAME_ID'].isin(player_games)]

        sample_size_with = len(with_player_df)
        sample_size_without = len(without_player_df)

        # Calculate W-L records
        with_wins = len(with_player_df[with_player_df['WL'] == 'W'])
        with_losses = len(with_player_df[with_player_df['WL'] == 'L'])
        without_wins = len(without_player_df[without_player_df['WL'] == 'W'])
        without_losses = len(without_player_df[without_player_df['WL'] == 'L'])

        with_record = f"{with_wins}-{with_losses}"
        without_record = f"{without_wins}-{without_losses}" if sample_size_without > 0 else "N/A"

        # Calculate stat averages
        with_stats = {}
        without_stats = {}
        diff_stats = {}

        for stat in STAT_DIFF_COLUMNS:
            if stat in team_df.columns:
                # With player averages
                if sample_size_with > 0:
                    with_stats[stat] = round(with_player_df[stat].mean(), 2)
                else:
                    with_stats[stat] = None

                # Without player averages
                if sample_size_without > 0:
                    without_stats[stat] = round(without_player_df[stat].mean(), 2)
                else:
                    without_stats[stat] = None

                # Difference (with - without)
                if with_stats[stat] is not None and without_stats[stat] is not None:
                    diff_stats[stat] = round(with_stats[stat] - without_stats[stat], 2)
                else:
                    diff_stats[stat] = None

        return {
            'player_name': player_name,
            'player_id': player_id,
            'with_player': with_stats,
            'without_player': without_stats,
            'diff': diff_stats,
            'sample_size_with': sample_size_with,
            'sample_size_without': sample_size_without,
            'with_record': with_record,
            'without_record': without_record
        }

    except Exception as e:
        return {
            'error': str(e),
            'with_player': None,
            'without_player': None,
            'diff': None,
            'sample_size_with': 0,
            'sample_size_without': 0
        }


def get_team_average_age(team_id: int) -> Optional[float]:
    """Get average age of team roster."""
    try:
        def fetch():
            return commonteamroster.CommonTeamRoster(team_id=team_id)
        roster = fetch_with_retry(fetch, max_retries=2, delay=5)
        df = roster.get_data_frames()[0]
        if 'AGE' in df.columns and len(df) > 0:
            ages = df['AGE'].dropna()
            if len(ages) > 0:
                return round(ages.mean(), 1)
    except Exception:
        pass
    return None


def get_team_pace(team_name: str, advanced_stats_df: pd.DataFrame) -> Optional[float]:
    """Get team's pace from advanced stats."""
    try:
        team_row = advanced_stats_df[advanced_stats_df['TEAM_NAME'] == team_name]
        if len(team_row) > 0 and 'PACE' in team_row.columns:
            return round(team_row.iloc[0]['PACE'], 1)
    except Exception:
        pass
    return None


def get_team_ats_record(team_id: int, team_name: str, num_games: int = 8) -> Dict[str, Any]:
    """
    Get team's ATS (against the spread) record for last N games.

    Returns dict with 'wins', 'losses', 'pushes', 'record_str'
    """
    try:
        # Get game log
        gamelog = teamgamelog.TeamGameLog(
            team_id=team_id,
            season='2025-26'
        )
        games_df = gamelog.get_data_frames()[0]

        if len(games_df) == 0:
            return {'wins': 0, 'losses': 0, 'pushes': 0, 'record_str': '0-0'}

        # Get last N games (most recent first)
        recent_games = games_df.head(num_games)

        ats_wins = 0
        ats_losses = 0
        ats_pushes = 0

        for _, game in recent_games.iterrows():
            game_date = game['GAME_DATE']
            # Parse date - format is like "DEC 31, 2025"
            try:
                parsed_date = datetime.strptime(game_date, '%b %d, %Y')
                date_str = parsed_date.strftime('%Y-%m-%d')
            except:
                continue

            # Get the spread for this game from SBR
            spread_result = scrape_nba_spreads(date_str)
            if not spread_result['success']:
                continue

            # Find this game in the spread data
            matchup = game['MATCHUP']
            is_home = ' vs. ' in matchup

            # Extract opponent from matchup
            if is_home:
                opponent = matchup.split(' vs. ')[1].strip()
            else:
                opponent = matchup.split(' @ ')[1].strip()

            # Find matching game in spread data
            for spread_game in spread_result['games']:
                # Check if this is our game
                spread_home = spread_game.home_team
                spread_away = spread_game.away_team

                game_matches = False
                our_spread = None

                if is_home and team_name in spread_home:
                    game_matches = True
                    our_spread = spread_game.current_home_spread
                elif not is_home and team_name in spread_away:
                    game_matches = True
                    our_spread = spread_game.current_away_spread

                if game_matches and our_spread is not None:
                    # Calculate if team covered
                    pts_scored = game['PTS']
                    # We need opponent's score - get from matchup
                    # The game log doesn't have opponent score directly
                    # We can infer from W/L and point differential
                    wl = game['WL']

                    # For simplicity, use the spread result from game status
                    if spread_game.status == 'Final':
                        # Game is complete, check actual result
                        # This is tricky without opponent score
                        # For now, skip games we can't verify
                        pass
                    break

        # If we couldn't get ATS data, return empty
        # For now, return placeholder - ATS requires more complex data tracking
        return {
            'wins': ats_wins,
            'losses': ats_losses,
            'pushes': ats_pushes,
            'record_str': f'{ats_wins}-{ats_losses}' + (f'-{ats_pushes}' if ats_pushes > 0 else '')
        }

    except Exception as e:
        return {'wins': 0, 'losses': 0, 'pushes': 0, 'record_str': 'N/A', 'error': str(e)}


def get_team_last_games_info(team_id: int, num_games: int = 8) -> List[Dict]:
    """Get info about team's last N games with ATS results."""
    try:
        def fetch():
            return leaguegamefinder.LeagueGameFinder(
                team_id_nullable=team_id,
                season_nullable='2025-26',
                season_type_nullable='Regular Season'
            )
        gamefinder = fetch_with_retry(fetch, max_retries=2, delay=5)
        games_df = gamefinder.get_data_frames()[0]

        if len(games_df) == 0:
            return []

        # Sort by date descending and get most recent games
        games_df = games_df.sort_values('GAME_DATE', ascending=False)
        recent_games = games_df.head(num_games)

        games_info = []
        for _, game in recent_games.iterrows():
            # Get margin (PLUS_MINUS is the point differential)
            margin = int(game['PLUS_MINUS']) if 'PLUS_MINUS' in game and pd.notna(game['PLUS_MINUS']) else None

            # Parse date for spread lookup (format is YYYY-MM-DD)
            game_date_str = str(game['GAME_DATE'])
            date_for_scrape = game_date_str  # Already in correct format

            # Format date for display
            try:
                parsed_date = datetime.strptime(game_date_str, '%Y-%m-%d')
                display_date = parsed_date.strftime('%b %d, %Y').upper()
            except:
                display_date = game_date_str

            # Determine if home or away
            matchup = game['MATCHUP']
            is_home = ' vs. ' in matchup

            games_info.append({
                'date': display_date,
                'date_fmt': date_for_scrape,
                'matchup': matchup,
                'wl': game['WL'],
                'pts': int(game['PTS']),
                'margin': margin,
                'is_home': is_home,
            })

        return games_info
    except Exception as e:
        print(f"Error fetching games: {e}")
        return []


def get_ats_for_games(team_name: str, games_info: List[Dict]) -> List[Dict]:
    """
    Calculate ATS (against the spread) results for a list of games.
    Fetches closing spreads from SBR for each game date.
    """
    if not games_info:
        return games_info

    # Group games by date for efficient scraping
    dates_to_scrape = set()
    for game in games_info:
        if game.get('date_fmt'):
            dates_to_scrape.add(game['date_fmt'])

    # Fetch spreads for each date (cached internally by SBR scraper)
    spread_cache = {}
    for date_str in dates_to_scrape:
        try:
            result = scrape_nba_spreads(date_str)
            if result['success']:
                for g in result['games']:
                    # Key by date + home team
                    key = f"{date_str}_{g.home_team}"
                    spread_cache[key] = {
                        'home_spread': g.current_home_spread,
                        'away_spread': g.current_away_spread,
                        'home_team': g.home_team,
                        'away_team': g.away_team,
                        'home_abbrev': g.home_abbrev,
                        'away_abbrev': g.away_abbrev,
                    }
        except Exception:
            continue

    # Get team abbreviation for matching
    team_abbrev = TEAM_NAME_TO_ABBREV.get(team_name, '')

    # Calculate ATS for each game
    for game in games_info:
        game['ats'] = None  # Default to unknown
        game['spread'] = None

        if game.get('margin') is None or game.get('date_fmt') is None:
            continue

        # Find the matching spread
        matchup = game['matchup']
        is_home = game['is_home']

        # Extract opponent and find the game in spread cache
        for key, spread_data in spread_cache.items():
            if not key.startswith(game['date_fmt']):
                continue

            # Check if this is our game (match by abbreviation for reliability)
            our_team_matches = False
            our_spread = None

            if is_home and (team_abbrev == spread_data.get('home_abbrev') or team_name in spread_data['home_team']):
                our_team_matches = True
                our_spread = spread_data['home_spread']
            elif not is_home and (team_abbrev == spread_data.get('away_abbrev') or team_name in spread_data['away_team']):
                our_team_matches = True
                our_spread = spread_data['away_spread']

            if our_team_matches and our_spread is not None:
                game['spread'] = our_spread
                margin = game['margin']

                # ATS calculation: margin + spread > 0 means covered
                # Example: Team is -5.5 favorite, wins by 7 -> 7 + (-5.5) = 1.5 > 0 -> covered
                # Example: Team is +3.5 underdog, loses by 2 -> -2 + 3.5 = 1.5 > 0 -> covered
                ats_margin = margin + our_spread

                if ats_margin > 0:
                    game['ats'] = 'W'  # Covered
                elif ats_margin < 0:
                    game['ats'] = 'L'  # Didn't cover
                else:
                    game['ats'] = 'P'  # Push
                break

    return games_info


def get_team_details(team_id: int, team_name: str) -> Dict[str, Any]:
    """
    Fetch team details on demand (lazy loading).
    Called when user expands a game row.

    Includes injured starter stat differentials showing team performance
    with vs without each injured starting player.
    """
    try:
        # Get average age from roster
        avg_age = get_team_average_age(team_id)

        # Get recent games
        recent_games = get_team_last_games_info(team_id, 8)

        # Get ATS results for recent games
        recent_games = get_ats_for_games(team_name, recent_games)

        # Calculate last 8 ATS record (against the spread)
        ats_wins = sum(1 for g in recent_games if g.get('ats') == 'W')
        ats_losses = sum(1 for g in recent_games if g.get('ats') == 'L')
        ats_pushes = sum(1 for g in recent_games if g.get('ats') == 'P')
        ats_unknown = sum(1 for g in recent_games if g.get('ats') is None)

        # Format ATS record string
        if ats_unknown == len(recent_games):
            last8_ats_record = 'N/A'
        else:
            last8_ats_record = f"{ats_wins}-{ats_losses}"
            if ats_pushes > 0:
                last8_ats_record += f"-{ats_pushes}"

        # Get injured starters using Rotowire (primary) or ESPN (fallback)
        injured_starters = []
        injury_source = 'none'

        if LINEUPS_AVAILABLE:
            team_abbrev = TEAM_NAME_TO_ABBREV.get(team_name)
            if team_abbrev:
                try:
                    # Use fallback-enabled function (Rotowire -> ESPN)
                    injury_result = get_injured_starters_with_fallback(team_abbrev)
                    injury_source = injury_result.get('source', 'none')
                    injured_starters = injury_result.get('injured_starters', [])

                    # Enrich each injured starter with full stat differentials
                    for starter in injured_starters:
                        player_name = starter.get('name', '')

                        # Get full stat differentials (includes W-L record and all stats)
                        stat_diff = get_stats_with_without_player(team_id, player_name)

                        # Add records
                        starter['with_record'] = stat_diff.get('with_record')
                        starter['without_record'] = stat_diff.get('without_record')

                        # Add sample sizes
                        starter['sample_size_with'] = stat_diff.get('sample_size_with', 0)
                        starter['sample_size_without'] = stat_diff.get('sample_size_without', 0)

                        # Add stat averages and differences
                        starter['stats_with'] = stat_diff.get('with_player')
                        starter['stats_without'] = stat_diff.get('without_player')
                        starter['stats_diff'] = stat_diff.get('diff')

                except Exception as e:
                    print(f"Error enriching injury data: {e}")

        return {
            'success': True,
            'team_id': team_id,
            'team_name': team_name,
            'avg_age': avg_age,
            'last8_ats_record': last8_ats_record,
            'recent_games': recent_games[:8],  # Last 8 for display
            'injured_starters': injured_starters,  # Only starters with injury concerns
            'injury_source': injury_source,  # 'rotowire', 'espn', or 'none'
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'team_id': team_id,
            'team_name': team_name,
        }


def run_nba_predictions(
    edge_strategy: str = 'closing',
    date: Optional[str] = None,
    bankroll: float = DEFAULT_PARAMS['bankroll'],
) -> Dict[str, Any]:
    """
    Run NBA predictions for a given date and strategy.

    Args:
        edge_strategy: 'opening' or 'closing'
        date: YYYY-MM-DD format, defaults to today/tomorrow based on strategy
        bankroll: Bankroll for stake calculations

    Returns:
        Dict with 'success', 'games', 'summary', and optionally 'error'
    """
    # Check availability
    status = check_nba_available()
    if not status['available']:
        return {
            'success': False,
            'error': status['error'],
            'games': [],
            'summary': {}
        }

    # Set date
    if date is None:
        if edge_strategy == 'opening':
            date = get_date_string('tomorrow')
        else:
            date = get_date_string('today')

    try:
        # Load model
        model = joblib.load(NBA_MODEL_PATH)

        # Load schedule for days rest calculation
        schedule_df = pd.read_csv(
            NBA_SCHEDULE_PATH,
            parse_dates=['Date'],
            date_format='%d/%m/%Y %H:%M'
        )

        # Parse target date for days rest calculation
        target_date = datetime.strptime(date, '%Y-%m-%d')

        # Fetch team stats from NBA API (Base stats for model) with retry
        try:
            def fetch_base_stats():
                return leaguedashteamstats.LeagueDashTeamStats(
                    season='2025-26',
                    season_type_all_star='Regular Season',
                    measure_type_detailed_defense='Base',
                    per_mode_detailed='PerGame',
                    plus_minus='N',
                    pace_adjust='N',
                    rank='N',
                    date_from_nullable='2025-10-20',
                    last_n_games='0',
                )
            team_stats = fetch_with_retry(fetch_base_stats, max_retries=3, delay=5)
            stats_df = team_stats.get_data_frames()[0]
        except Exception as e:
            return {
                'success': False,
                'error': f'Failed to fetch NBA stats after retries: {str(e)}',
                'games': [],
                'summary': {}
            }

        # Fetch advanced stats for pace with retry
        try:
            def fetch_advanced_stats():
                return leaguedashteamstats.LeagueDashTeamStats(
                    season='2025-26',
                    season_type_all_star='Regular Season',
                    measure_type_detailed_defense='Advanced',
                    per_mode_detailed='PerGame',
                )
            advanced_stats = fetch_with_retry(fetch_advanced_stats, max_retries=2, delay=5)
            advanced_df = advanced_stats.get_data_frames()[0]
        except Exception:
            advanced_df = pd.DataFrame()  # Empty df if we can't get pace

        # Fetch games from SBR scraper
        scrape_result = scrape_nba_spreads(date)

        if not scrape_result['success']:
            return {
                'success': False,
                'error': scrape_result.get('error', 'Failed to scrape spreads'),
                'games': [],
                'summary': {}
            }

        games = scrape_result['games']

        # For tomorrow's games, detect back-to-back teams (teams playing today)
        teams_playing_today = set()
        if edge_strategy == 'opening':
            today_date = get_date_string('today')
            teams_playing_today = get_teams_playing_on_date(today_date)

        if not games:
            return {
                'success': True,
                'games': [],
                'summary': {
                    'date': date,
                    'edge_strategy': edge_strategy,
                    'total_games': 0,
                    'games_with_edge': 0,
                    'message': f'No games found for {date}'
                }
            }

        # Process each game
        results = []
        for game in games:
            # Normalize team names to match our model's format
            home_team = normalize_team_name(game.home_team)
            away_team = normalize_team_name(game.away_team)

            # Check if either team is on a back-to-back
            is_back_to_back = (game.home_abbrev in teams_playing_today or
                               game.away_abbrev in teams_playing_today)

            # Check if teams exist in current index
            if home_team not in team_index_current or away_team not in team_index_current:
                results.append({
                    'date': date,
                    'game_id': game.game_id,
                    'home': home_team,
                    'away': away_team,
                    'home_abbrev': game.home_abbrev,
                    'away_abbrev': game.away_abbrev,
                    'status': 'UNKNOWN_TEAM',
                    'book': game.book,
                    'back_to_back': is_back_to_back,
                })
                continue

            # Get spreads
            current_home_spread = game.current_home_spread
            current_away_spread = game.current_away_spread
            open_home_spread = game.open_home_spread
            open_away_spread = game.open_away_spread

            if current_home_spread is None or current_away_spread is None:
                results.append({
                    'date': date,
                    'game_id': game.game_id,
                    'home': home_team,
                    'away': away_team,
                    'home_abbrev': game.home_abbrev,
                    'away_abbrev': game.away_abbrev,
                    'status': 'NO_ODDS',
                    'book': game.book,
                    'back_to_back': is_back_to_back,
                })
                continue

            # Calculate days rest for TARGET DATE (not today)
            home_days_rest = calculate_days_rest(home_team, target_date, schedule_df)
            away_days_rest = calculate_days_rest(away_team, target_date, schedule_df)

            # Build feature vector - must match exactly how original main.py does it
            try:
                home_idx = team_index_current[home_team]
                away_idx = team_index_current[away_team]

                # Get team stats as Series (exactly like original main.py)
                home_team_series = stats_df.iloc[home_idx]
                away_team_series = stats_df.iloc[away_idx]

                # Concatenate the two series (this creates a combined series with all features)
                stats = pd.concat([home_team_series, away_team_series])
                stats['Days-Rest-Home'] = home_days_rest
                stats['Days-Rest-Away'] = away_days_rest

                # Create a single-row dataframe (like original code)
                match_data = [stats]
                games_data_frame = pd.concat(match_data, ignore_index=True, axis=1)
                games_data_frame = games_data_frame.T

                # Drop TEAM_ID and TEAM_NAME columns and convert to float
                frame_ml = games_data_frame.drop(columns=['TEAM_ID', 'TEAM_NAME'])
                features = frame_ml.values.astype(float)

            except Exception as e:
                results.append({
                    'date': date,
                    'game_id': game.game_id,
                    'home': home_team,
                    'away': away_team,
                    'home_abbrev': game.home_abbrev,
                    'away_abbrev': game.away_abbrev,
                    'open_home_spread': open_home_spread,
                    'open_away_spread': open_away_spread,
                    'current_home_spread': current_home_spread,
                    'current_away_spread': current_away_spread,
                    'book': game.book,
                    'status': 'MISSING_FEATURES',
                    'error': str(e),
                    'back_to_back': is_back_to_back,
                })
                continue

            # Run prediction
            if hasattr(model, 'predict_proba'):
                p_home = model.predict_proba(features)[0, 1]
            else:
                z = model.decision_function(features)[0]
                p_home = 1.0 / (1.0 + np.exp(-z))

            p_away = 1.0 - p_home

            # Determine side (pick the team with higher model probability)
            side = 'home' if p_home >= 0.5 else 'away'

            # Calculate edges vs spreads
            edge_vs_open_home = calculate_edge(p_home, open_home_spread) if open_home_spread else None
            edge_vs_open_away = calculate_edge(p_away, open_away_spread) if open_away_spread else None
            edge_vs_current_home = calculate_edge(p_home, current_home_spread)
            edge_vs_current_away = calculate_edge(p_away, current_away_spread)

            # Edge for our pick
            if side == 'home':
                edge_vs_open = edge_vs_open_home
                edge_vs_current = edge_vs_current_home
                pick_spread = current_home_spread
            else:
                edge_vs_open = edge_vs_open_away
                edge_vs_current = edge_vs_current_away
                pick_spread = current_away_spread

            # Calculate line movement (in points)
            if open_home_spread is not None and current_home_spread is not None:
                # Positive = line moved in our favor (spread got better for our pick)
                if side == 'home':
                    line_movement = open_home_spread - current_home_spread
                else:
                    line_movement = open_away_spread - current_away_spread
            else:
                line_movement = None

            # Determine if this is a bet (based on edge threshold)
            min_edge = DEFAULT_PARAMS['min_edge']
            if edge_strategy == 'opening':
                bet_edge = edge_vs_open if edge_vs_open is not None else edge_vs_current
            else:
                bet_edge = edge_vs_current

            is_bet = bool(bet_edge >= min_edge)  # Convert numpy bool to Python bool

            # Calculate stake using Kelly criterion (simplified)
            if is_bet:
                # Approximate Kelly for spread betting
                # f* = (bp - q) / b where b = 1 (even money at -110), p = edge + 0.5
                p_win = 0.5 + bet_edge
                q = 1 - p_win
                kelly_f = (p_win - q)  # Simplified for -110 odds
                kelly_f = max(0, kelly_f)
                stake_frac = min(kelly_f * DEFAULT_PARAMS['kelly_scale'], DEFAULT_PARAMS['max_stake_frac'])
                stake = round(bankroll * stake_frac, 2)
            else:
                kelly_f = 0
                stake_frac = 0
                stake = 0

            # Get team IDs for lazy-loading details later
            home_team_id = TEAM_NAME_TO_ID.get(home_team)
            away_team_id = TEAM_NAME_TO_ID.get(away_team)

            # Get pace from already-fetched advanced stats (no extra API call)
            home_pace = get_team_pace(home_team, advanced_df) if len(advanced_df) > 0 else None
            away_pace = get_team_pace(away_team, advanced_df) if len(advanced_df) > 0 else None

            results.append({
                'date': date,
                'game_id': game.game_id,
                'home': home_team,
                'away': away_team,
                'home_abbrev': game.home_abbrev,
                'away_abbrev': game.away_abbrev,
                'game_status': game.status,
                'start_time': game.start_time,
                'book': game.book,
                # Team IDs for lazy loading details
                'home_team_id': home_team_id,
                'away_team_id': away_team_id,
                # Days rest
                'home_days_rest': home_days_rest,
                'away_days_rest': away_days_rest,
                # Opening spreads
                'open_home_spread': open_home_spread,
                'open_away_spread': open_away_spread,
                # Current spreads
                'current_home_spread': current_home_spread,
                'current_away_spread': current_away_spread,
                # Model predictions
                'p_home': round(p_home, 4),
                'p_away': round(p_away, 4),
                # Line movement (in points)
                'line_movement': round(line_movement, 1) if line_movement is not None else None,
                # Pace (from already-fetched advanced stats)
                'home_pace': home_pace,
                'away_pace': away_pace,
                'status': 'OK',
                # Back-to-back indicator (team playing today and tomorrow)
                'back_to_back': is_back_to_back,
            })

        # Sort by game time (start_time)
        results_sorted = sorted(
            results,
            key=lambda x: (
                x.get('status') != 'OK',
                x.get('start_time') or ''
            )
        )

        # Summary stats
        ok_results = [r for r in results if r.get('status') == 'OK']

        # Count uncertain games (<55%) and bullish games (>=75%)
        uncertain_games = sum(1 for r in ok_results if max(r.get('p_home', 0), r.get('p_away', 0)) < 0.55)
        bullish_games = sum(1 for r in ok_results if max(r.get('p_home', 0), r.get('p_away', 0)) >= 0.75)

        return {
            'success': True,
            'games': results_sorted,
            'summary': {
                'date': date,
                'total_games': len(ok_results),
                'uncertain_games': uncertain_games,  # Model <55%
                'bullish_games': bullish_games,      # Model >=75%
            }
        }

    except Exception as e:
        import traceback
        return {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'games': [],
            'summary': {}
        }
