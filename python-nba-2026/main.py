import argparse
from datetime import datetime, timedelta

import pandas as pd
import tensorflow as tf
from colorama import Fore, Style

from src.DataProviders.SbrOddsProvider import SbrOddsProvider
from src.Predict import LogReg_Runner
from src.Utils.Dictionaries import team_index_current
from src.Utils.tools import create_todays_games_from_odds, get_json_data, to_data_frame, get_todays_games_json, create_todays_games
from nba_api.stats.endpoints import leaguedashteamstats

def createTodaysGames(games, df, odds, target_date=None):
    """
    Create feature matrix for games.

    Args:
        games: List of [home_team, away_team] pairs
        df: DataFrame with team statistics
        odds: Dictionary of odds or None for manual input
        target_date: The date of the games (datetime object). Defaults to today.
                     Use this to correctly calculate days rest for tomorrow's games.
    """
    match_data = []
    todays_games_uo = []
    home_team_odds = []
    away_team_odds = []

    home_team_days_rest = []
    away_team_days_rest = []

    # Use target_date if provided, otherwise use today
    if target_date is None:
        target_date = datetime.today()

    for game in games:
        home_team = game[0]
        away_team = game[1]
        if home_team not in team_index_current or away_team not in team_index_current:
            continue
        if odds is not None:
            # print(odds)
            game_odds = odds[home_team + ':' + away_team]
            todays_games_uo.append(game_odds['under_over_odds'])

            home_team_odds.append(game_odds[home_team]['money_line_odds'])
            away_team_odds.append(game_odds[away_team]['money_line_odds'])

        else:
            todays_games_uo.append(input(home_team + ' vs ' + away_team + ': '))

            home_team_odds.append(input(home_team + ' odds: '))
            away_team_odds.append(input(away_team + ' odds: '))

        # calculate days rest for both teams relative to target_date
        schedule_df = pd.read_csv('/Users/aidenflynn/BET-DASHBOARD/python-nba-2026/Data/nba-2025-UTC.csv', parse_dates=['Date'], date_format='%d/%m/%Y %H:%M')
        home_games = schedule_df[(schedule_df['Home Team'] == home_team) | (schedule_df['Away Team'] == home_team)]
        away_games = schedule_df[(schedule_df['Home Team'] == away_team) | (schedule_df['Away Team'] == away_team)]
        # Filter games BEFORE target_date (not today) to correctly calculate rest for future games
        previous_home_games = home_games.loc[home_games['Date'] < target_date].sort_values('Date', ascending=False).head(1)['Date']
        previous_away_games = away_games.loc[away_games['Date'] < target_date].sort_values('Date', ascending=False).head(1)['Date']
        if len(previous_home_games) > 0:
            last_home_date = previous_home_games.iloc[0]
            home_days_off = (target_date - last_home_date).days
            home_days_off = max(1, home_days_off)  # At least 1 day
        else:
            home_days_off = 7
        if len(previous_away_games) > 0:
            last_away_date = previous_away_games.iloc[0]
            away_days_off = (target_date - last_away_date).days
            away_days_off = max(1, away_days_off)  # At least 1 day
        else:
            away_days_off = 7
        # print(f"{away_team} days off: {away_days_off} @ {home_team} days off: {home_days_off}")

        home_team_days_rest.append(home_days_off)
        away_team_days_rest.append(away_days_off)
        home_team_series = df.iloc[team_index_current.get(home_team)]
        away_team_series = df.iloc[team_index_current.get(away_team)]
        stats = pd.concat([home_team_series, away_team_series])
        stats['Days-Rest-Home'] = home_days_off
        stats['Days-Rest-Away'] = away_days_off
        match_data.append(stats)

    games_data_frame = pd.concat(match_data, ignore_index=True, axis=1)
    games_data_frame = games_data_frame.T

    frame_ml = games_data_frame.drop(columns=['TEAM_ID', 'TEAM_NAME'])
    data = frame_ml.values
    data = data.astype(float)

    return data, todays_games_uo, frame_ml, home_team_odds, away_team_odds


def main():
    odds = SbrOddsProvider(sportsbook="fanduel").get_odds()
    games = create_todays_games_from_odds(odds)

    if len(games) == 0:
        print("No games found.")
        return

    # Request team stats for the 2025–26 regular season
    data = leaguedashteamstats.LeagueDashTeamStats(
        season='2025-26',
        season_type_all_star='Regular Season',
        measure_type_detailed_defense='Base',
        per_mode_detailed='PerGame',
        plus_minus='N',
        pace_adjust='N',
        rank='N',
        date_from_nullable='2025-10-20',
        # league_id='00',
        last_n_games='0',
    )
    # Convert to pandas DataFrame
    df = data.get_data_frames()[0]
    df.to_csv('games.csv', index=False)
    data, todays_games_uo, frame_ml, home_team_odds, away_team_odds = createTodaysGames(games, df, odds)


    print("---------------LogReg Model Predictions---------------")
    LogReg_Runner.logreg_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, True)
    print("-------------------------------------------------------")

if __name__ == "__main__":
    main()
