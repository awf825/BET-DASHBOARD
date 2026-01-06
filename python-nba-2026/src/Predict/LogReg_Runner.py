import numpy as np
import joblib
from colorama import Fore, Style, init, deinit
from src.Utils.Dictionaries import TEAM_TO_CODE

init(strip=False, convert=False, autoreset=True)

class LogRegRunner:
    def __init__(self, ml_model_path: str, uo_model_path: str | None = None):
        self.ml_model = joblib.load(ml_model_path)
        self.uo_model = joblib.load(uo_model_path) if uo_model_path else None

    def predict_ml(self, X: np.ndarray) -> np.ndarray:
        # returns prob of home win (class 1) if model trained that way
        if hasattr(self.ml_model, "predict_proba"):
            p = self.ml_model.predict_proba(X)[:, 1]
        else:
            # decision_function to prob via logistic curve if needed
            z = self.ml_model.decision_function(X)
            p = 1.0 / (1.0 + np.exp(-z))
        return p

    def predict_uo(self, X: np.ndarray) -> np.ndarray:
        if self.uo_model is None:
            raise RuntimeError("UO model not loaded.")
        if hasattr(self.uo_model, "predict_proba"):
            p_over = self.uo_model.predict_proba(X)[:, 1]
        else:
            z = self.uo_model.decision_function(X)
            p_over = 1.0 / (1.0 + np.exp(-z))
        return p_over

def logreg_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, kelly_criterion,
                  ml_model_path='/Users/aidenflynn/BET-DASHBOARD/python-nba-2026/Models/homewin_logreg_final.joblib', uo_model_path=None):
    # data is already feature-matrix for ML; frame_ml used for OU when needed
    runner = LogRegRunner(ml_model_path, uo_model_path)
    ml_p = runner.predict_ml(np.array(data))
    count = 0

    for game in games:
        home_team = game[0]
        away_team = game[1]
        winner_confidence = round(ml_p[count] * 100, 1)
        if winner_confidence >= 50:
            print(
                Fore.GREEN + home_team + Style.RESET_ALL + Fore.CYAN + f" ({winner_confidence}%)" + Style.RESET_ALL + ' vs ' + Fore.RED + away_team + Style.RESET_ALL)
        else:
            print(
                Fore.RED + home_team + Style.RESET_ALL + ' vs ' + Fore.GREEN + away_team + Style.RESET_ALL + Fore.CYAN + f" ({100-winner_confidence}%)" + Style.RESET_ALL)

        print("\n")

        count += 1
