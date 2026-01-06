"""
Azure Function: NHL Data Scraper & Feature Builder

Runs daily at 6 AM ET to:
1. Scrape latest NHL game results and stats
2. Build rolling features for each team
3. Save features CSV to Azure Blob Storage

Timer schedule: 0 0 11 * * * (6 AM ET = 11 AM UTC)
"""
import azure.functions as func
import logging
import os
import io
from datetime import datetime, timedelta
import pandas as pd
import requests
from bs4 import BeautifulSoup
import json

app = func.FunctionApp()

# Azure Blob Storage settings
STORAGE_CONNECTION_STRING = os.environ.get('AzureWebJobsStorage')
CONTAINER_NAME = os.environ.get('STORAGE_CONTAINER', 'betting-data')

# NHL API endpoints
NHL_API_BASE = 'https://api-web.nhle.com/v1'


def get_blob_service_client():
    """Get Azure Blob Service Client."""
    from azure.storage.blob import BlobServiceClient
    return BlobServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)


def upload_dataframe_to_blob(df: pd.DataFrame, blob_path: str):
    """Upload DataFrame as CSV to blob storage."""
    client = get_blob_service_client()
    blob_client = client.get_blob_client(container=CONTAINER_NAME, blob=blob_path)

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode('utf-8')

    blob_client.upload_blob(csv_bytes, overwrite=True)
    logging.info(f'Uploaded {blob_path} to blob storage ({len(df)} rows)')


def download_blob_to_dataframe(blob_path: str) -> pd.DataFrame:
    """Download CSV from blob storage to DataFrame."""
    client = get_blob_service_client()
    blob_client = client.get_blob_client(container=CONTAINER_NAME, blob=blob_path)

    try:
        data = blob_client.download_blob().readall()
        return pd.read_csv(io.BytesIO(data))
    except Exception as e:
        logging.warning(f'Could not download {blob_path}: {e}')
        return None


def fetch_nhl_standings() -> dict:
    """Fetch current NHL standings from NHL API."""
    url = f'{NHL_API_BASE}/standings/now'
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_team_schedule(team_abbrev: str, season: str = '20252026') -> list:
    """Fetch team's schedule for the season."""
    url = f'{NHL_API_BASE}/club-schedule-season/{team_abbrev}/{season}'
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json().get('games', [])


def calculate_rolling_features(games_df: pd.DataFrame, team: str, window: int = 10) -> dict:
    """
    Calculate rolling features for a team from recent games.

    Returns dict with home_* and away_* features.
    """
    # Filter to team's games
    home_games = games_df[games_df['home_team'] == team].tail(window)
    away_games = games_df[games_df['away_team'] == team].tail(window)
    all_games = games_df[(games_df['home_team'] == team) | (games_df['away_team'] == team)].tail(window)

    features = {'team': team}

    # Calculate home features
    if len(home_games) > 0:
        features['home_goals_for_avg'] = home_games['home_score'].mean()
        features['home_goals_against_avg'] = home_games['away_score'].mean()
        features['home_win_pct'] = (home_games['home_score'] > home_games['away_score']).mean()
        features['home_shots_avg'] = home_games.get('home_shots', pd.Series([30])).mean()
    else:
        features['home_goals_for_avg'] = 2.5
        features['home_goals_against_avg'] = 2.5
        features['home_win_pct'] = 0.5
        features['home_shots_avg'] = 30.0

    # Calculate away features
    if len(away_games) > 0:
        features['away_goals_for_avg'] = away_games['away_score'].mean()
        features['away_goals_against_avg'] = away_games['home_score'].mean()
        features['away_win_pct'] = (away_games['away_score'] > away_games['home_score']).mean()
        features['away_shots_avg'] = away_games.get('away_shots', pd.Series([28])).mean()
    else:
        features['away_goals_for_avg'] = 2.5
        features['away_goals_against_avg'] = 2.5
        features['away_win_pct'] = 0.5
        features['away_shots_avg'] = 28.0

    # Overall features
    if len(all_games) > 0:
        # Win streak
        wins = []
        for _, g in all_games.iterrows():
            if g['home_team'] == team:
                wins.append(1 if g['home_score'] > g['away_score'] else 0)
            else:
                wins.append(1 if g['away_score'] > g['home_score'] else 0)

        streak = 0
        for w in reversed(wins):
            if w == wins[-1]:
                streak += 1
            else:
                break
        features['win_streak'] = streak if wins[-1] == 1 else -streak
        features['last10_win_pct'] = sum(wins) / len(wins)
    else:
        features['win_streak'] = 0
        features['last10_win_pct'] = 0.5

    return features


def build_all_team_features(games_df: pd.DataFrame) -> pd.DataFrame:
    """Build features DataFrame for all teams."""
    teams = set(games_df['home_team'].unique()) | set(games_df['away_team'].unique())

    features_list = []
    for team in teams:
        try:
            features = calculate_rolling_features(games_df, team)
            features_list.append(features)
        except Exception as e:
            logging.error(f'Error building features for {team}: {e}')

    return pd.DataFrame(features_list)


def scrape_game_results() -> pd.DataFrame:
    """
    Scrape recent NHL game results.
    This is a simplified version - you may want to adapt your existing nhl_scraper.py logic.
    """
    # Fetch standings to get team list
    standings = fetch_nhl_standings()

    all_games = []
    processed_game_ids = set()

    for record in standings.get('standings', []):
        team_abbrev = record.get('teamAbbrev', {}).get('default', '')
        if not team_abbrev:
            continue

        try:
            # Get team's schedule
            schedule = fetch_team_schedule(team_abbrev)

            for game in schedule:
                game_id = game.get('id')
                if game_id in processed_game_ids:
                    continue

                # Only include completed games
                if game.get('gameState') != 'OFF':
                    continue

                processed_game_ids.add(game_id)

                home_team = game.get('homeTeam', {}).get('abbrev', '')
                away_team = game.get('awayTeam', {}).get('abbrev', '')
                home_score = game.get('homeTeam', {}).get('score', 0)
                away_score = game.get('awayTeam', {}).get('score', 0)
                game_date = game.get('gameDate', '')

                all_games.append({
                    'game_id': game_id,
                    'date': game_date,
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_score': home_score,
                    'away_score': away_score,
                })
        except Exception as e:
            logging.warning(f'Error fetching schedule for {team_abbrev}: {e}')

    df = pd.DataFrame(all_games)
    if len(df) > 0:
        df = df.sort_values('date').drop_duplicates('game_id')

    return df


@app.timer_trigger(schedule="0 0 11 * * *", arg_name="mytimer", run_on_startup=False)
def nhl_daily_scraper(mytimer: func.TimerRequest) -> None:
    """
    Timer-triggered function that runs daily at 6 AM ET (11 AM UTC).

    1. Scrapes latest NHL game results
    2. Builds rolling features for each team
    3. Uploads features CSV to blob storage
    """
    utc_timestamp = datetime.utcnow().isoformat()
    logging.info(f'NHL scraper triggered at {utc_timestamp}')

    try:
        # Step 1: Load existing games data (if any)
        existing_games = download_blob_to_dataframe('data/nhl_games_history.csv')

        # Step 2: Scrape recent game results
        logging.info('Scraping NHL game results...')
        new_games = scrape_game_results()
        logging.info(f'Scraped {len(new_games)} games')

        # Step 3: Merge with existing data
        if existing_games is not None and len(existing_games) > 0:
            all_games = pd.concat([existing_games, new_games]).drop_duplicates('game_id')
        else:
            all_games = new_games

        # Step 4: Save updated games history
        upload_dataframe_to_blob(all_games, 'data/nhl_games_history.csv')

        # Step 5: Build team features
        logging.info('Building team features...')
        features_df = build_all_team_features(all_games)
        logging.info(f'Built features for {len(features_df)} teams')

        # Step 6: Save features to blob
        upload_dataframe_to_blob(features_df, 'data/nhl_features_latest.csv')

        # Also save timestamped version for history
        timestamp = datetime.utcnow().strftime('%Y%m%d')
        upload_dataframe_to_blob(features_df, f'data/nhl_features_{timestamp}.csv')

        logging.info('NHL scraper completed successfully')

    except Exception as e:
        logging.error(f'NHL scraper failed: {e}')
        raise
