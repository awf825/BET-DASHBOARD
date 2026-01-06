"""
NBA Lineup Scraper
Scrapes starting lineups and injury status from Rotowire and ESPN depth charts.
Focus: Only show injury alerts for STARTING players.
"""
import requests
from bs4 import BeautifulSoup
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class PlayerStatus:
    """Container for a player's status."""
    name: str
    position: str
    is_starter: bool
    status: Optional[str]  # None, 'GTD', 'Ques', 'Prob', 'Out', 'Susp'
    pct_play: Optional[int]  # 0, 25, 50, 75, 100


@dataclass
class TeamLineup:
    """Container for a team's lineup."""
    team_abbrev: str
    team_name: str
    starters: List[PlayerStatus]
    injured_starters: List[PlayerStatus]  # Starters with injury status


# Team name normalization (abbreviation to full name)
ABBREV_TO_FULL = {
    'ATL': 'Atlanta Hawks',
    'BOS': 'Boston Celtics',
    'BKN': 'Brooklyn Nets',
    'BRK': 'Brooklyn Nets',
    'CHA': 'Charlotte Hornets',
    'CHI': 'Chicago Bulls',
    'CLE': 'Cleveland Cavaliers',
    'DAL': 'Dallas Mavericks',
    'DEN': 'Denver Nuggets',
    'DET': 'Detroit Pistons',
    'GSW': 'Golden State Warriors',
    'GS': 'Golden State Warriors',
    'HOU': 'Houston Rockets',
    'IND': 'Indiana Pacers',
    'LAC': 'LA Clippers',
    'LAL': 'Los Angeles Lakers',
    'MEM': 'Memphis Grizzlies',
    'MIA': 'Miami Heat',
    'MIL': 'Milwaukee Bucks',
    'MIN': 'Minnesota Timberwolves',
    'NOP': 'New Orleans Pelicans',
    'NO': 'New Orleans Pelicans',
    'NYK': 'New York Knicks',
    'NY': 'New York Knicks',
    'OKC': 'Oklahoma City Thunder',
    'ORL': 'Orlando Magic',
    'PHI': 'Philadelphia 76ers',
    'PHX': 'Phoenix Suns',
    'PHO': 'Phoenix Suns',
    'POR': 'Portland Trail Blazers',
    'SAC': 'Sacramento Kings',
    'SAS': 'San Antonio Spurs',
    'SA': 'San Antonio Spurs',
    'TOR': 'Toronto Raptors',
    'UTA': 'Utah Jazz',
    'UTAH': 'Utah Jazz',
    'WAS': 'Washington Wizards',
}


def scrape_espn_depth_charts() -> Dict[str, List[str]]:
    """
    Scrape ESPN NBA depth charts to get starting 5 for each team.
    Returns dict of {team_abbrev: [list of starter names]}

    ESPN shows starters as the first player at each position (PG, SG, SF, PF, C).
    Bolded players have moved to starting lineup in past 7 days.
    """
    url = "https://www.espn.com/nba/depth"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        starters_by_team = {}

        # ESPN depth chart structure: table with teams as rows
        # Each row has cells for PG, SG, SF, PF, C positions
        tables = soup.find_all('table')

        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all('td')
                if len(cells) < 6:  # Team name + 5 positions
                    continue

                # First cell is team name
                team_cell = cells[0]
                team_link = team_cell.find('a')
                if not team_link:
                    continue

                # Extract team abbreviation from link or text
                team_href = team_link.get('href', '')
                team_match = re.search(r'/nba/team/_/name/(\w+)', team_href)
                if team_match:
                    team_abbrev = team_match.group(1).upper()
                else:
                    continue

                # Get starter from each position column (cells 1-5)
                starters = []
                for i in range(1, 6):
                    if i < len(cells):
                        cell = cells[i]
                        # First player link in cell is the starter
                        player_links = cell.find_all('a')
                        if player_links:
                            starter_name = player_links[0].get_text(strip=True)
                            starters.append(starter_name)

                if starters:
                    starters_by_team[team_abbrev] = starters

        return starters_by_team

    except Exception as e:
        print(f"Error scraping ESPN depth charts: {e}")
        return {}


def scrape_rotowire_lineups() -> Dict[str, Any]:
    """
    Scrape Rotowire NBA lineups for today's games.
    Returns dict of games with lineup info and injury alerts for starters.

    Structure:
    {
        'success': True/False,
        'games': [
            {
                'away': {
                    'abbrev': 'MIN',
                    'starters': [PlayerStatus, ...],
                    'injured_starters': [PlayerStatus, ...]  # Only starters with issues
                },
                'home': {
                    'abbrev': 'MIA',
                    'starters': [PlayerStatus, ...],
                    'injured_starters': [PlayerStatus, ...]
                }
            }
        ]
    }
    """
    url = "https://www.rotowire.com/basketball/nba-lineups.php"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        games = []

        # Find all lineup cards
        lineup_divs = soup.find_all('div', class_='lineup')

        for lineup_div in lineup_divs:
            if 'is-nba' not in lineup_div.get('class', []):
                continue

            game_data = {'away': None, 'home': None}

            # Get team abbreviations from lineup__abbr divs
            abbr_divs = lineup_div.find_all('div', class_='lineup__abbr')
            team_abbrevs = [div.get_text(strip=True) for div in abbr_divs]

            # Process away and home lineups
            away_list = lineup_div.find('ul', class_='is-visit')
            home_list = lineup_div.find('ul', class_='is-home')

            if away_list and len(team_abbrevs) >= 1:
                away_abbrev = team_abbrevs[0] if team_abbrevs else 'UNK'
                game_data['away'] = parse_team_lineup(away_list, away_abbrev)

            if home_list and len(team_abbrevs) >= 2:
                home_abbrev = team_abbrevs[1] if len(team_abbrevs) > 1 else 'UNK'
                game_data['home'] = parse_team_lineup(home_list, home_abbrev)

            if game_data['away'] or game_data['home']:
                games.append(game_data)

        return {
            'success': True,
            'games': games
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'games': []
        }


def parse_team_lineup(lineup_ul, team_abbrev: str) -> Dict[str, Any]:
    """
    Parse a team's lineup from Rotowire.
    Returns dict with starters and injured starters.
    """
    starters = []
    injured_starters = []

    # Get all player list items
    player_items = lineup_ul.find_all('li', class_='lineup__player')

    starter_count = 0
    hit_may_not_play = False

    for item in lineup_ul.find_all('li'):
        # Check if we've hit the "MAY NOT PLAY" section
        if 'lineup__title' in item.get('class', []):
            title_text = item.get_text(strip=True)
            if 'MAY NOT PLAY' in title_text.upper():
                hit_may_not_play = True
            continue

        # Skip status items
        if 'lineup__status' in item.get('class', []):
            continue

        # Check if this is a player item
        if 'lineup__player' not in item.get('class', []):
            continue

        # Parse player info
        player_link = item.find('a')
        if not player_link:
            continue

        player_name = player_link.get('title', player_link.get_text(strip=True))

        # Get position
        pos_div = item.find('div', class_='lineup__pos')
        position = pos_div.get_text(strip=True) if pos_div else ''

        # Get injury status
        inj_span = item.find('span', class_='lineup__inj')
        status = inj_span.get_text(strip=True) if inj_span else None

        # Get play percentage from class
        pct_play = None
        item_classes = item.get('class', [])
        for cls in item_classes:
            if cls.startswith('is-pct-play-'):
                try:
                    pct_play = int(cls.replace('is-pct-play-', ''))
                except:
                    pass

        # Determine if this is a starter (first 5 before MAY NOT PLAY section)
        is_starter = not hit_may_not_play and starter_count < 5

        player = PlayerStatus(
            name=player_name,
            position=position,
            is_starter=is_starter,
            status=status,
            pct_play=pct_play
        )

        if is_starter:
            starters.append(player)
            starter_count += 1

            # Check if this starter has an injury concern
            if status or (pct_play is not None and pct_play < 100):
                injured_starters.append(player)

    return {
        'abbrev': team_abbrev,
        'team_name': ABBREV_TO_FULL.get(team_abbrev, team_abbrev),
        'starters': [
            {
                'name': p.name,
                'position': p.position,
                'status': p.status,
                'pct_play': p.pct_play
            } for p in starters
        ],
        'injured_starters': [
            {
                'name': p.name,
                'position': p.position,
                'status': p.status,
                'pct_play': p.pct_play
            } for p in injured_starters
        ]
    }


def get_lineup_for_team(team_abbrev: str) -> Optional[Dict[str, Any]]:
    """
    Get lineup info for a specific team.
    Returns None if team not playing today or lineup not available.
    """
    result = scrape_rotowire_lineups()

    if not result['success']:
        return None

    for game in result['games']:
        if game['away'] and game['away']['abbrev'] == team_abbrev:
            return game['away']
        if game['home'] and game['home']['abbrev'] == team_abbrev:
            return game['home']

    return None


def get_all_lineups() -> Dict[str, Any]:
    """
    Get all lineups for today's games.
    Returns dict keyed by team abbreviation.
    """
    result = scrape_rotowire_lineups()

    if not result['success']:
        return {}

    lineups_by_team = {}

    for game in result['games']:
        if game['away']:
            lineups_by_team[game['away']['abbrev']] = game['away']
        if game['home']:
            lineups_by_team[game['home']['abbrev']] = game['home']

    return lineups_by_team


# ESPN team URL slugs (differs from standard abbreviations)
ABBREV_TO_ESPN_SLUG = {
    'ATL': 'atl/atlanta-hawks',
    'BOS': 'bos/boston-celtics',
    'BKN': 'bkn/brooklyn-nets',
    'CHA': 'cha/charlotte-hornets',
    'CHI': 'chi/chicago-bulls',
    'CLE': 'cle/cleveland-cavaliers',
    'DAL': 'dal/dallas-mavericks',
    'DEN': 'den/denver-nuggets',
    'DET': 'det/detroit-pistons',
    'GSW': 'gs/golden-state-warriors',
    'HOU': 'hou/houston-rockets',
    'IND': 'ind/indiana-pacers',
    'LAC': 'lac/los-angeles-clippers',
    'LAL': 'lal/los-angeles-lakers',
    'MEM': 'mem/memphis-grizzlies',
    'MIA': 'mia/miami-heat',
    'MIL': 'mil/milwaukee-bucks',
    'MIN': 'min/minnesota-timberwolves',
    'NOP': 'no/new-orleans-pelicans',
    'NYK': 'ny/new-york-knicks',
    'OKC': 'okc/oklahoma-city-thunder',
    'ORL': 'orl/orlando-magic',
    'PHI': 'phi/philadelphia-76ers',
    'PHX': 'phx/phoenix-suns',
    'POR': 'por/portland-trail-blazers',
    'SAC': 'sac/sacramento-kings',
    'SAS': 'sa/san-antonio-spurs',
    'TOR': 'tor/toronto-raptors',
    'UTA': 'utah/utah-jazz',
    'WAS': 'wsh/washington-wizards',
}

# ESPN team IDs for depth chart API
ABBREV_TO_ESPN_TEAM_ID = {
    'ATL': '1', 'BOS': '2', 'BKN': '17', 'CHA': '30', 'CHI': '4',
    'CLE': '5', 'DAL': '6', 'DEN': '7', 'DET': '8', 'GSW': '9',
    'HOU': '10', 'IND': '11', 'LAC': '12', 'LAL': '13', 'MEM': '29',
    'MIA': '14', 'MIL': '15', 'MIN': '16', 'NOP': '3', 'NYK': '18',
    'OKC': '25', 'ORL': '19', 'PHI': '20', 'PHX': '21', 'POR': '22',
    'SAC': '23', 'SAS': '24', 'TOR': '28', 'UTA': '26', 'WAS': '27',
}


def get_espn_starters(team_abbrev: str) -> List[str]:
    """
    Fetch starting 5 from ESPN depth chart API.
    Returns list of starter names (first player at each position: PG, SG, SF, PF, C).
    """
    team_id = ABBREV_TO_ESPN_TEAM_ID.get(team_abbrev)
    if not team_id:
        return []

    api_url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team_id}/depthcharts"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }

    try:
        response = requests.get(api_url, headers=headers, timeout=15)
        response.raise_for_status()

        data = response.json()
        starters = []

        # Get depth chart positions
        depthchart = data.get('depthchart', [])
        if not depthchart:
            return []

        positions = depthchart[0].get('positions', {})

        # Get first player at each standard position
        for pos_key in ['pg', 'sg', 'sf', 'pf', 'c']:
            if pos_key in positions:
                athletes = positions[pos_key].get('athletes', [])
                if athletes:
                    starter_name = athletes[0].get('displayName', '')
                    if starter_name:
                        starters.append(starter_name)

        return starters

    except Exception as e:
        print(f"Error fetching ESPN depth chart: {e}")
        return []


def scrape_espn_injuries(team_abbrev: str) -> Dict[str, Any]:
    """
    Fetch injuries from ESPN API for a specific team.
    Uses ESPN's public injuries API for reliable data.

    Returns:
    {
        'success': True/False,
        'injuries': [
            {
                'name': 'Player Name',
                'position': '',
                'status': 'Out',  # Out, Day-To-Day, etc.
                'comment': 'Ankle - Expected to be out 2-3 weeks'
            }
        ]
    }
    """
    team_full_name = ABBREV_TO_FULL.get(team_abbrev)
    if not team_full_name:
        return {'success': False, 'error': f'Unknown team abbreviation: {team_abbrev}', 'injuries': []}

    # ESPN general injuries API returns all NBA injuries grouped by team
    api_url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }

    try:
        response = requests.get(api_url, headers=headers, timeout=15)
        response.raise_for_status()

        data = response.json()

        injuries = []

        # Find our team in the injuries list
        for team_data in data.get('injuries', []):
            if team_data.get('displayName', '').lower() == team_full_name.lower():
                # Found our team, extract injuries
                for injury in team_data.get('injuries', []):
                    athlete = injury.get('athlete', {})
                    player_name = athlete.get('displayName', '')
                    status = injury.get('status', '')
                    short_comment = injury.get('shortComment', '')
                    long_comment = injury.get('longComment', '')

                    if player_name:
                        injuries.append({
                            'name': player_name,
                            'position': '',
                            'status': status,
                            'comment': short_comment or long_comment
                        })
                break

        return {
            'success': True,
            'injuries': injuries,
            'api_url': api_url
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'injuries': []
        }


def get_injured_starters_with_fallback(team_abbrev: str, starters: List[str] = None) -> Dict[str, Any]:
    """
    Get injured starters for a team.
    Uses ESPN depth chart to identify regular starters, then checks ESPN injuries API
    to find which starters are injured.

    Args:
        team_abbrev: Team abbreviation (e.g., 'NYK')
        starters: Optional list of starter names to filter against

    Returns:
    {
        'source': 'espn',
        'injured_starters': [
            {
                'name': 'Player Name',
                'position': 'PG',
                'status': 'Out',
                'pct_play': 0
            }
        ],
        'starters': ['Player1', 'Player2', ...]  # List of starter names
    }
    """
    # Get regular starters from ESPN depth chart (not today's lineup)
    espn_starters = starters or get_espn_starters(team_abbrev)

    # Get injuries from ESPN API
    espn_result = scrape_espn_injuries(team_abbrev)

    if espn_result['success']:
        injured_starters = []

        # Filter injuries to only regular starters using depth chart
        if espn_starters:
            starter_names_lower = [s.lower() for s in espn_starters]
            for injury in espn_result['injuries']:
                # Check if injured player is in starters list
                player_name_lower = injury['name'].lower()
                is_starter = any(
                    player_name_lower == starter or  # Exact match
                    player_name_lower in starter or
                    starter in player_name_lower
                    for starter in starter_names_lower
                )
                if is_starter:
                    injured_starters.append({
                        'name': injury['name'],
                        'position': injury['position'],
                        'status': injury['status'],
                        'pct_play': 0 if injury['status'].lower() == 'out' else None,
                        'comment': injury.get('comment', '')
                    })
        else:
            # No starters list available, return empty (can't determine starters)
            injured_starters = []

        return {
            'source': 'espn',
            'injured_starters': injured_starters,
            'starters': espn_starters,
            'espn_url': espn_result.get('api_url')
        }

    # ESPN failed
    return {
        'source': 'none',
        'injured_starters': [],
        'starters': espn_starters or [],
        'error': 'Could not fetch injury data from ESPN'
    }


if __name__ == '__main__':
    # Test the scraper
    print("Testing Rotowire lineup scraper...")
    result = scrape_rotowire_lineups()
    print(f"Success: {result['success']}")
    print(f"Games found: {len(result['games'])}")

    for game in result['games'][:2]:  # Show first 2 games
        print(f"\n--- Game ---")
        if game['away']:
            away = game['away']
            print(f"Away: {away['abbrev']}")
            print(f"  Starters: {[s['name'] for s in away['starters']]}")
            if away['injured_starters']:
                print(f"  Injured starters: {[(s['name'], s['status']) for s in away['injured_starters']]}")
        if game['home']:
            home = game['home']
            print(f"Home: {home['abbrev']}")
            print(f"  Starters: {[s['name'] for s in home['starters']]}")
            if home['injured_starters']:
                print(f"  Injured starters: {[(s['name'], s['status']) for s in home['injured_starters']]}")
