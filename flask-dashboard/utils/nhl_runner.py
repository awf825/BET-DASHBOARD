"""
NHL Predictions Runner
Wrapper to run NHL edge predictions from the Flask app.
Uses SBR scraper to get both opening and current lines.

Supports both local development (SQLite) and production (Azure Blob Storage).
In production, set AZURE_STORAGE_CONNECTION_STRING environment variable.
"""
import sys
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd

# Check if we're in production mode
from utils.azure_storage import IS_PRODUCTION, NHLPaths

# Add the NHL project to path (for local development)
NHL_PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../python-nhl-2026'))
sys.path.insert(0, NHL_PROJECT_PATH)

# Import from the NHL script (local development)
NHL_AVAILABLE = False
IMPORT_ERROR = None

if not IS_PRODUCTION:
    try:
        import joblib
        from nhl_daily_edges import (
            build_matchup_features_from_latest_snapshots,
            compute_edges_and_stakes,
            _table_columns,
            norm_abbrev as nhl_norm_abbrev,
            american_to_decimal,
            implied_prob_from_american,
        )
        NHL_AVAILABLE = True
    except ImportError as e:
        IMPORT_ERROR = str(e)
else:
    # Production mode - import minimal dependencies
    import joblib
    NHL_AVAILABLE = True

    # Production implementations of required functions
    def nhl_norm_abbrev(abbrev: str) -> str:
        """Normalize NHL team abbreviation."""
        mapping = {
            'WAS': 'WSH', 'MON': 'MTL', 'TB': 'TBL', 'LA': 'LAK',
            'NJ': 'NJD', 'SJ': 'SJS', 'CAL': 'CGY', 'CLB': 'CBJ',
            'NAS': 'NSH', 'WIN': 'WPG', 'UTAH': 'UTA',
        }
        return mapping.get(abbrev.upper(), abbrev.upper())

    def implied_prob_from_american(odds: float) -> float:
        """Convert American odds to implied probability."""
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return abs(odds) / (abs(odds) + 100)

    def american_to_decimal(odds: float) -> float:
        """Convert American odds to decimal odds."""
        if odds > 0:
            return (odds / 100) + 1
        else:
            return (100 / abs(odds)) + 1

# Import our SBR scraper
from utils.sbr_scraper import scrape_nhl_odds, normalize_abbrev

# Paths to NHL resources (local development)
NHL_DB_PATH = os.path.join(NHL_PROJECT_PATH, 'nhl_scrape.sqlite')
NHL_MODEL_PATH = os.path.join(NHL_PROJECT_PATH, 'model_logit.joblib')
NHL_TABLE = 'games_table_20252026'

# Default strategy parameters
DEFAULT_PARAMS = {
    'bankroll': 5000.0,
    'kelly_scale': 0.10,
    'min_edge': 0.05,
    'max_edge': 0.12,
    'max_stake_frac': 0.02,
}

# =============================================================================
# Market Agreement Strategy Configuration
# =============================================================================

STRATEGY_CONFIG = {
    # Opening Strategy (Alpha-seeking): bet early when edge exists before market moves
    'opening': {
        'enabled_segments': ['road_underdog'],  # Toggleable segments
        'edge_range': (0.02, 0.05),  # 2% to 5% edge vs open
        'require_agreement': True,
        'compression_range': (0.3, 0.9),  # Market partially corrects but doesn't eliminate
        'allow_high_compression_if_price_improved': True,  # Allow compression > 1 if move_toward_pick > 0
    },
    # Closing Strategy (Confirmation-seeking): bet when edge remains after market correction
    'closing': {
        'enabled_segments': ['home_underdog', 'road_favorite'],  # Based on heatmaps
        'conditional_segments': {
            # Road underdogs only if market agrees (imp_drift > 0 = market got more bullish)
            'road_underdog': {'require_market_agreement': True},
        },
        'edge_range': (0.05, 0.08),  # 5% to 8% edge vs close (default)
        # Segment-specific edge ranges (in addition to default range)
        'segment_edge_ranges': {
            # Road underdogs also accept 2%-3% closing edge
            'road_underdog': [(0.02, 0.03)],
        },
        'require_agreement': True,
        'compression_range': (0.3, 1.1),  # Edge didn't explode
        'trap_threshold': {
            'edge': 0.08,  # Edge >= 8% is suspicious
            'compression': 1.3,  # Compression > 1.3 is suspicious
        },
    },
}


def determine_segment_type(side: str, p_model: float, is_market_underdog: bool) -> str:
    """
    Determine the segment type based on pick side and market status.

    Args:
        side: 'home' or 'away'
        p_model: Model probability for the pick
        is_market_underdog: Whether the pick is a market underdog (implied prob < 0.5)

    Returns:
        One of: 'road_underdog', 'road_favorite', 'home_underdog', 'home_favorite'
    """
    location = 'home' if side == 'home' else 'road'
    status = 'underdog' if is_market_underdog else 'favorite'
    return f'{location}_{status}'


def calculate_market_agreement_signals(
    side: str,
    edge_vs_open: Optional[float],
    edge_vs_close: Optional[float],
    open_home_odds: Optional[float],
    open_away_odds: Optional[float],
    current_home_odds: Optional[float],
    current_away_odds: Optional[float],
) -> Dict[str, Any]:
    """
    Calculate market agreement signals for betting decision.

    Returns:
        Dict with:
        - move_toward_pick: American odds difference (positive = better payout)
        - imp_drift: Implied probability drift (positive = market more bullish on pick)
        - agreement: Whether edge signs match and open edge > 0
        - compression: Ratio of close edge to open edge (capped 0-2)
    """
    result = {
        'move_toward_pick': None,
        'imp_drift': None,
        'agreement': None,
        'compression': None,
    }

    # Need all odds to calculate
    if None in (open_home_odds, open_away_odds, current_home_odds, current_away_odds):
        return result

    # Calculate move_toward_pick (American odds difference)
    if side == 'away':
        result['move_toward_pick'] = current_away_odds - open_away_odds
    else:
        result['move_toward_pick'] = current_home_odds - open_home_odds

    # Calculate implied probability drift
    if side == 'away':
        open_imp = implied_prob_from_american(open_away_odds)
        current_imp = implied_prob_from_american(current_away_odds)
    else:
        open_imp = implied_prob_from_american(open_home_odds)
        current_imp = implied_prob_from_american(current_home_odds)

    if np.isfinite(open_imp) and np.isfinite(current_imp):
        # Positive = market got more bullish (price worse for bettor)
        result['imp_drift'] = current_imp - open_imp

    # Calculate agreement: same sign edges and positive open edge
    if edge_vs_open is not None and edge_vs_close is not None:
        same_sign = (edge_vs_open >= 0 and edge_vs_close >= 0) or (edge_vs_open < 0 and edge_vs_close < 0)
        result['agreement'] = same_sign and abs(edge_vs_open) > 0

        # Calculate compression (ratio of close to open edge, capped 0-2)
        if abs(edge_vs_open) > 0.001:  # Avoid division by tiny numbers
            compression = abs(edge_vs_close) / abs(edge_vs_open)
            result['compression'] = min(max(compression, 0), 2.0)
        else:
            result['compression'] = None

    return result


def apply_strategy_bet_decision(
    edge_strategy: str,
    segment_type: str,
    edge_vs_open: Optional[float],
    edge_vs_close: Optional[float],
    agreement: Optional[bool],
    compression: Optional[float],
    move_toward_pick: Optional[float],
    imp_drift: Optional[float],
    base_bet: bool,
    base_stake: float,
    enabled_segments_override: Optional[Dict[str, bool]] = None,
) -> Dict[str, Any]:
    """
    Apply strategy-specific bet decision logic.

    Args:
        imp_drift: Implied probability drift (positive = market got more bullish on pick)
        enabled_segments_override: If provided, overrides the default segment config.
            Dict like {'road_underdog': True, 'home_underdog': False, ...}

    Returns:
        Dict with:
        - bet: Final bet decision
        - stake: Final stake (0 if no bet)
        - bet_reason: Reason for bet decision
        - is_trap: Whether this is flagged as a potential trap
        - trap_reason: Reason for trap flag if applicable
    """
    config = STRATEGY_CONFIG.get(edge_strategy, {})

    result = {
        'bet': False,
        'stake': 0,
        'bet_reason': None,
        'is_trap': False,
        'trap_reason': None,
    }

    # Determine enabled segments - use override if provided, otherwise use config defaults
    if enabled_segments_override is not None:
        segment_allowed = enabled_segments_override.get(segment_type, False)
    else:
        default_enabled = config.get('enabled_segments', [])
        segment_allowed = segment_type in default_enabled

    # Special handling for road_underdog in closing strategy - requires market agreement
    # Market agrees when imp_drift > 0 (market got more bullish on the pick)
    if segment_type == 'road_underdog' and edge_strategy == 'closing' and segment_allowed:
        conditional_segments = config.get('conditional_segments', {})
        if segment_type in conditional_segments:
            cond_config = conditional_segments[segment_type]
            if cond_config.get('require_market_agreement'):
                if imp_drift is None or imp_drift <= 0:
                    drift_pct = f'{imp_drift*100:.1f}%' if imp_drift is not None else 'None'
                    result['bet_reason'] = f'road_dog_market_disagrees_drift={drift_pct}'
                    return result

    if not segment_allowed:
        result['bet_reason'] = f'segment_{segment_type}_not_enabled'
        return result

    # Get edge to check based on strategy
    if edge_strategy == 'opening':
        edge_to_check = edge_vs_open
    else:
        edge_to_check = edge_vs_close

    if edge_to_check is None:
        result['bet_reason'] = 'missing_edge_data'
        return result

    # Check edge range - first check default range, then segment-specific ranges
    edge_min, edge_max = config.get('edge_range', (0.05, 0.12))
    in_default_range = edge_min <= edge_to_check <= edge_max

    # Check segment-specific edge ranges (additional ranges for specific segments)
    in_segment_range = False
    segment_edge_ranges = config.get('segment_edge_ranges', {}).get(segment_type, [])
    for seg_min, seg_max in segment_edge_ranges:
        if seg_min <= edge_to_check <= seg_max:
            in_segment_range = True
            break

    if not (in_default_range or in_segment_range):
        result['bet_reason'] = f'edge_{edge_to_check:.3f}_outside_range_{edge_min}-{edge_max}'
        return result

    # Check agreement requirement
    if config.get('require_agreement', False):
        if agreement is None or not agreement:
            result['bet_reason'] = 'no_agreement'
            return result

    # Check compression range
    comp_range = config.get('compression_range')
    if comp_range and compression is not None:
        comp_min, comp_max = comp_range

        if edge_strategy == 'opening':
            # Opening strategy: allow high compression if price improved
            if compression > comp_max:
                if config.get('allow_high_compression_if_price_improved', False):
                    if move_toward_pick is None or move_toward_pick <= 0:
                        result['bet_reason'] = f'high_compression_{compression:.2f}_without_price_improvement'
                        return result
                    # Price improved, allow the bet with caution noted
                else:
                    result['bet_reason'] = f'compression_{compression:.2f}_above_max_{comp_max}'
                    return result
            elif compression < comp_min:
                result['bet_reason'] = f'compression_{compression:.2f}_below_min_{comp_min}'
                return result
        else:
            # Closing strategy: strict compression check
            if not (comp_min <= compression <= comp_max):
                result['bet_reason'] = f'compression_{compression:.2f}_outside_range_{comp_min}-{comp_max}'
                return result

    # Trap filter for closing strategy
    if edge_strategy == 'closing':
        trap_config = config.get('trap_threshold', {})
        trap_edge = trap_config.get('edge', 0.08)
        trap_compression = trap_config.get('compression', 1.3)

        if edge_to_check >= trap_edge or (compression is not None and compression > trap_compression):
            result['is_trap'] = True
            if edge_to_check >= trap_edge:
                result['trap_reason'] = f'edge_{edge_to_check:.3f}_>=_{trap_edge}'
            else:
                result['trap_reason'] = f'compression_{compression:.2f}_>_{trap_compression}'
            # Still flag as trap but don't bet by default
            result['bet_reason'] = 'trap_flagged'
            return result

    # All checks passed - bet!
    result['bet'] = True
    result['stake'] = base_stake
    result['bet_reason'] = 'all_criteria_met'

    return result


def check_nhl_available() -> Dict[str, Any]:
    """Check if NHL predictions are available."""
    if not NHL_AVAILABLE:
        return {
            'available': False,
            'error': f'NHL module not available: {IMPORT_ERROR}'
        }

    if IS_PRODUCTION:
        # Production: check Azure blob storage
        from utils.azure_storage import blob_exists
        model_path = NHLPaths.model()
        features_path = NHLPaths.features_csv()

        if not blob_exists(model_path.blob):
            return {
                'available': False,
                'error': f'NHL model not found in blob storage: {model_path.blob}'
            }

        if not blob_exists(features_path.blob):
            return {
                'available': False,
                'error': f'NHL features not found in blob storage: {features_path.blob}'
            }
    else:
        # Local development: check local files
        if not os.path.exists(NHL_DB_PATH):
            return {
                'available': False,
                'error': f'NHL database not found: {NHL_DB_PATH}'
            }

        if not os.path.exists(NHL_MODEL_PATH):
            return {
                'available': False,
                'error': f'NHL model not found: {NHL_MODEL_PATH}'
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


def calculate_edge(p_model: float, odds: float) -> float:
    """Calculate edge: model probability - implied probability."""
    implied = implied_prob_from_american(odds)
    if not np.isfinite(implied):
        return 0.0
    return p_model - implied


# =============================================================================
# Production Mode Functions (Azure Blob Storage)
# =============================================================================

def load_model_production():
    """Load the NHL model from Azure Blob Storage."""
    from utils.azure_storage import load_joblib_model
    return load_joblib_model(NHLPaths.model().blob)


def load_features_production() -> pd.DataFrame:
    """Load pre-computed team features from Azure Blob Storage."""
    from utils.azure_storage import read_csv_from_blob
    return read_csv_from_blob(NHLPaths.features_csv().blob)


def build_matchup_features_production(
    features_df: pd.DataFrame,
    home_abbrev: str,
    away_abbrev: str
) -> Optional[pd.DataFrame]:
    """
    Build feature row for a matchup from pre-computed team features.

    The features CSV should have columns:
    - team: Team abbreviation
    - home_* and away_* columns with team stats

    For each matchup, we look up the home team's features and away team's features,
    then combine them into a single row.
    """
    # Look up home team (use their "home_*" features)
    home_row = features_df[features_df['team'] == home_abbrev]
    if len(home_row) == 0:
        return None

    # Look up away team (use their "away_*" features)
    away_row = features_df[features_df['team'] == away_abbrev]
    if len(away_row) == 0:
        return None

    # Get feature columns (all columns starting with home_ or away_)
    home_cols = [c for c in features_df.columns if c.startswith('home_')]
    away_cols = [c for c in features_df.columns if c.startswith('away_')]

    # Build combined feature row
    combined = {}
    for col in home_cols:
        combined[col] = home_row[col].values[0]
    for col in away_cols:
        combined[col] = away_row[col].values[0]

    return pd.DataFrame([combined])


def compute_edges_production(
    df_row: pd.DataFrame,
    bundle: dict,
    home_odds: float,
    away_odds: float,
    bankroll: float,
    min_edge: float,
    max_edge: float,
    kelly_scale: float,
    max_stake_frac: float,
    edge_strategy: str
) -> Dict[str, Any]:
    """
    Compute edges and stakes in production mode (simplified version).
    """
    # Extract model from bundle
    model = bundle['model'] if isinstance(bundle, dict) else bundle

    # Get predictions
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(df_row.values)
        p_home = probs[0, 1]
    else:
        # Fallback for models without predict_proba
        z = model.decision_function(df_row.values)[0]
        p_home = 1.0 / (1.0 + np.exp(-z))

    p_away = 1.0 - p_home

    # Determine side
    side = 'home' if p_home >= 0.5 else 'away'

    # Calculate implied probabilities
    home_imp = implied_prob_from_american(home_odds)
    away_imp = implied_prob_from_american(away_odds)

    # Calculate edges
    edge_home = p_home - home_imp
    edge_away = p_away - away_imp
    edge = edge_home if side == 'home' else edge_away

    # Determine segment
    if edge < 0:
        segment = 'negative'
    elif edge < min_edge:
        segment = 'below_threshold'
    elif edge < max_edge:
        segment = 'target'
    else:
        segment = 'high_edge'

    # Kelly criterion for stake
    if side == 'home':
        decimal_odds = american_to_decimal(home_odds)
    else:
        decimal_odds = american_to_decimal(away_odds)

    p = p_home if side == 'home' else p_away
    q = 1 - p
    b = decimal_odds - 1

    kelly_f = (b * p - q) / b if b > 0 else 0
    kelly_f = max(0, kelly_f)

    # Apply scaling
    stake_frac = min(kelly_f * kelly_scale, max_stake_frac)

    # Determine if we should bet (old threshold logic - kept for reference)
    should_bet = edge >= min_edge and edge <= max_edge

    # Always calculate stake - let caller decide whether to use it
    stake = bankroll * stake_frac

    return {
        'p_home': p_home,
        'p_away': p_away,
        'side': side,
        'home_imp': home_imp,
        'away_imp': away_imp,
        'edge': edge,
        'segment': segment,
        'kelly_f': kelly_f,
        'stake_frac': stake_frac,
        'stake': stake,
        'bet': should_bet,
    }


def run_nhl_predictions(
    edge_strategy: str = 'closing',
    date: Optional[str] = None,
    bankroll: float = DEFAULT_PARAMS['bankroll'],
    enabled_segments: Optional[Dict[str, Dict[str, bool]]] = None,
) -> Dict[str, Any]:
    """
    Run NHL predictions for a given date and strategy.
    Now uses SBR scraper to get both opening and current lines.

    Args:
        edge_strategy: 'opening' or 'closing'
        date: YYYY-MM-DD format, defaults to today/tomorrow based on strategy
        bankroll: Bankroll for stake calculations
        enabled_segments: Dict with segment toggles per strategy, e.g.:
            {
                'opening': {'road_underdog': True, 'home_underdog': False, ...},
                'closing': {'road_underdog': True, 'home_underdog': True, ...}
            }

    Returns:
        Dict with 'success', 'games', 'summary', and optionally 'error'
    """
    # Check availability
    status = check_nhl_available()
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
        # Load model and data source based on environment
        if IS_PRODUCTION:
            # Production: load from Azure Blob Storage
            bundle = load_model_production()
            features_df = load_features_production()
            conn = None  # No database connection in production
        else:
            # Local development: load from local files
            bundle = joblib.load(NHL_MODEL_PATH)
            conn = sqlite3.connect(NHL_DB_PATH)

            # Get feature columns from table
            cols = _table_columns(conn, NHL_TABLE)
            feature_cols = [c for c in cols if c.startswith('home_') or c.startswith('away_')]

            if not feature_cols:
                conn.close()
                return {
                    'success': False,
                    'error': f'No feature columns found in {NHL_TABLE}',
                    'games': [],
                    'summary': {}
                }
            features_df = None  # Not used in local mode

        # Fetch games from SBR scraper (gets both opening and current lines)
        scrape_result = scrape_nhl_odds(date)

        if not scrape_result['success']:
            if conn:
                conn.close()
            return {
                'success': False,
                'error': scrape_result.get('error', 'Failed to scrape odds'),
                'games': [],
                'summary': {}
            }

        games = scrape_result['games']

        # For tomorrow's games, detect back-to-back teams (teams playing today)
        teams_playing_today = set()
        today_date = get_date_string('today')
        if date != today_date:
            # We're looking at a future date, check if teams played today
            teams_playing_today = get_teams_playing_on_date(today_date)

        if not games:
            if conn:
                conn.close()
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
            # Normalize abbreviations to match database
            home_abbrev = normalize_abbrev(game.home_abbrev)
            away_abbrev = normalize_abbrev(game.away_abbrev)

            # Also apply NHL script's normalization
            home_abbrev = nhl_norm_abbrev(home_abbrev)
            away_abbrev = nhl_norm_abbrev(away_abbrev)

            # Get current (closing) odds - what we use for bet decisions
            current_home_odds = game.current_home_ml
            current_away_odds = game.current_away_ml

            # Get opening odds
            open_home_odds = game.open_home_ml
            open_away_odds = game.open_away_ml

            # Check if either team is on a back-to-back
            is_back_to_back = (home_abbrev in teams_playing_today or
                               away_abbrev in teams_playing_today)

            if current_home_odds is None or current_away_odds is None:
                results.append({
                    'date': date,
                    'game_id': game.game_id,
                    'home': home_abbrev,
                    'away': away_abbrev,
                    'status': 'NO_ODDS',
                    'book': game.book,
                    'start_time': game.start_time,
                    'bet': False,
                    'bet_reason': 'no_odds',
                    'is_trap': False,
                    'trap_reason': None,
                    'back_to_back': is_back_to_back,
                })
                continue

            # Build features - different approach for local vs production
            if IS_PRODUCTION:
                df_row = build_matchup_features_production(features_df, home_abbrev, away_abbrev)
            else:
                df_row, dbg = build_matchup_features_from_latest_snapshots(
                    conn, NHL_TABLE, home_abbrev, away_abbrev,
                    feature_cols, asof_date=date
                )

            if df_row is None:
                # Missing features - still include in results
                results.append({
                    'date': date,
                    'game_id': game.game_id,
                    'home': home_abbrev,
                    'away': away_abbrev,
                    'open_home_ml': open_home_odds,
                    'open_away_ml': open_away_odds,
                    'current_home_ml': current_home_odds,
                    'current_away_ml': current_away_odds,
                    'book': game.book,
                    'start_time': game.start_time,
                    'status': 'MISSING_FEATURES',
                    'p_home': None,
                    'p_away': None,
                    'side': None,
                    'segment': None,
                    'segment_type': None,
                    'edge_vs_open': None,
                    'edge_vs_current': None,
                    'move_toward_pick': None,
                    'imp_drift': None,
                    'agreement': None,
                    'compression': None,
                    'bet': False,
                    'bet_reason': 'missing_features',
                    'is_trap': False,
                    'trap_reason': None,
                    'stake': None,
                    'back_to_back': is_back_to_back,
                })
                continue

            # Compute predictions and edges against CURRENT lines (for bet decision)
            if IS_PRODUCTION:
                scored = compute_edges_production(
                    df_row=df_row,
                    bundle=bundle,
                    home_odds=current_home_odds,
                    away_odds=current_away_odds,
                    bankroll=bankroll,
                    min_edge=DEFAULT_PARAMS['min_edge'],
                    max_edge=DEFAULT_PARAMS['max_edge'],
                    kelly_scale=DEFAULT_PARAMS['kelly_scale'],
                    max_stake_frac=DEFAULT_PARAMS['max_stake_frac'],
                    edge_strategy=edge_strategy
                )
            else:
                scored = compute_edges_and_stakes(
                    df_row=df_row,
                    bundle=bundle,
                    home_odds=current_home_odds,
                    away_odds=current_away_odds,
                    bankroll=bankroll,
                    min_edge=DEFAULT_PARAMS['min_edge'],
                    max_edge=DEFAULT_PARAMS['max_edge'],
                    kelly_scale=DEFAULT_PARAMS['kelly_scale'],
                    max_stake_frac=DEFAULT_PARAMS['max_stake_frac'],
                    edge_strategy=edge_strategy
                )

            p_home = scored['p_home']
            p_away = scored['p_away']
            side = scored['side']

            # Calculate edge vs opening lines
            if open_home_odds is not None and open_away_odds is not None:
                edge_vs_open_home = calculate_edge(p_home, open_home_odds)
                edge_vs_open_away = calculate_edge(p_away, open_away_odds)
                edge_vs_open = edge_vs_open_home if side == 'home' else edge_vs_open_away
            else:
                edge_vs_open = None

            # Edge vs current is what the model already computed
            edge_vs_current = scored['edge']

            # Calculate implied probabilities for opening lines
            open_home_imp = implied_prob_from_american(open_home_odds) if open_home_odds else None
            open_away_imp = implied_prob_from_american(open_away_odds) if open_away_odds else None

            # Determine segment type (road_underdog, home_favorite, etc.)
            current_pick_imp = scored['home_imp'] if side == 'home' else scored['away_imp']
            is_market_underdog = current_pick_imp < 0.5
            segment_type = determine_segment_type(side, p_home if side == 'home' else p_away, is_market_underdog)

            # Calculate market agreement signals
            market_signals = calculate_market_agreement_signals(
                side=side,
                edge_vs_open=edge_vs_open,
                edge_vs_close=edge_vs_current,
                open_home_odds=open_home_odds,
                open_away_odds=open_away_odds,
                current_home_odds=current_home_odds,
                current_away_odds=current_away_odds,
            )

            # Get enabled segments for current strategy
            strategy_segments = None
            if enabled_segments is not None:
                strategy_segments = enabled_segments.get(edge_strategy)

            # Calculate stake from stake_frac (don't rely on scored['stake'] which may be 0)
            # Recalculate Kelly for the picked side specifically
            if side == 'away':
                pick_decimal = american_to_decimal(current_away_odds)
                pick_p = p_away
            else:
                pick_decimal = american_to_decimal(current_home_odds)
                pick_p = p_home

            pick_b = pick_decimal - 1
            pick_q = 1 - pick_p
            pick_kelly = (pick_b * pick_p - pick_q) / pick_b if pick_b > 0 else 0
            pick_kelly = max(0, pick_kelly)
            calculated_stake_frac = min(pick_kelly * DEFAULT_PARAMS['kelly_scale'], DEFAULT_PARAMS['max_stake_frac'])
            calculated_stake = bankroll * calculated_stake_frac

            # Apply strategy-specific bet decision
            strategy_decision = apply_strategy_bet_decision(
                edge_strategy=edge_strategy,
                segment_type=segment_type,
                edge_vs_open=edge_vs_open,
                edge_vs_close=edge_vs_current,
                agreement=market_signals['agreement'],
                compression=market_signals['compression'],
                move_toward_pick=market_signals['move_toward_pick'],
                imp_drift=market_signals['imp_drift'],
                base_bet=scored['bet'],
                base_stake=calculated_stake,
                enabled_segments_override=strategy_segments,
            )

            results.append({
                'date': date,
                'game_id': game.game_id,
                'home': home_abbrev,
                'away': away_abbrev,
                'game_status': game.status,
                'book': game.book,
                'start_time': game.start_time,
                # Opening lines
                'open_home_ml': open_home_odds,
                'open_away_ml': open_away_odds,
                'open_home_imp': round(open_home_imp, 4) if open_home_imp else None,
                'open_away_imp': round(open_away_imp, 4) if open_away_imp else None,
                # Current/closing lines
                'current_home_ml': current_home_odds,
                'current_away_ml': current_away_odds,
                'current_home_imp': round(scored['home_imp'], 4),
                'current_away_imp': round(scored['away_imp'], 4),
                # Model predictions
                'p_home': round(p_home, 4),
                'p_away': round(p_away, 4),
                'side': side,
                'segment': scored['segment'],
                'segment_type': segment_type,
                # Edges
                'edge_vs_open': round(edge_vs_open, 4) if edge_vs_open is not None else None,
                'edge_vs_current': round(edge_vs_current, 4),
                # Market agreement signals
                'move_toward_pick': round(market_signals['move_toward_pick'], 2) if market_signals['move_toward_pick'] is not None else None,
                'imp_drift': round(market_signals['imp_drift'], 4) if market_signals['imp_drift'] is not None else None,
                'agreement': market_signals['agreement'],
                'compression': round(market_signals['compression'], 3) if market_signals['compression'] is not None else None,
                # Line movement (positive = line moved in your favor if betting the pick)
                'line_movement': calculate_line_movement(
                    side, open_home_odds, open_away_odds,
                    current_home_odds, current_away_odds
                ),
                # Bet decision (now using market agreement strategy)
                'bet': strategy_decision['bet'],
                'bet_reason': strategy_decision['bet_reason'],
                'is_trap': strategy_decision['is_trap'],
                'trap_reason': strategy_decision['trap_reason'],
                'stake': round(strategy_decision['stake'], 2) if strategy_decision['bet'] else None,
                'stake_frac': round(scored['stake_frac'], 4) if strategy_decision['bet'] else None,
                'kelly_f': round(scored['kelly_f'], 4),
                'status': 'OK',
                # Back-to-back indicator (team playing today and tomorrow)
                'back_to_back': is_back_to_back,
            })

        if conn:
            conn.close()

        # Sort: bets first, then by edge vs current
        results_sorted = sorted(
            results,
            key=lambda x: (
                x.get('status') != 'OK',
                not x.get('bet', False),
                -(x.get('edge_vs_current') or -999)
            )
        )

        # Summary stats
        ok_results = [r for r in results if r.get('status') == 'OK']
        games_with_edge = sum(1 for r in ok_results if r.get('bet'))
        total_stake = sum(r.get('stake', 0) or 0 for r in ok_results if r.get('bet'))
        traps_flagged = sum(1 for r in ok_results if r.get('is_trap'))

        # Average edges
        edges_open = [r['edge_vs_open'] for r in ok_results if r.get('edge_vs_open') is not None]
        edges_current = [r['edge_vs_current'] for r in ok_results if r.get('edge_vs_current') is not None]

        # Count by segment type
        segment_counts = {}
        for r in ok_results:
            seg = r.get('segment_type')
            if seg:
                segment_counts[seg] = segment_counts.get(seg, 0) + 1

        # Bet reason breakdown
        reason_counts = {}
        for r in ok_results:
            reason = r.get('bet_reason')
            if reason:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1

        return {
            'success': True,
            'games': results_sorted,
            'summary': {
                'date': date,
                'edge_strategy': edge_strategy,
                'enabled_segments': enabled_segments,
                'total_games': len(ok_results),
                'games_with_edge': games_with_edge,
                'traps_flagged': traps_flagged,
                'total_stake': round(total_stake, 2),
                'bankroll': bankroll,
                'avg_edge_vs_open': round(np.mean(edges_open), 4) if edges_open else None,
                'avg_edge_vs_current': round(np.mean(edges_current), 4) if edges_current else None,
                'segment_counts': segment_counts,
                'bet_reason_counts': reason_counts,
                'strategy_config': STRATEGY_CONFIG.get(edge_strategy, {}),
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


def get_teams_playing_on_date(target_date: str) -> set:
    """
    Get set of team abbreviations playing on a given date.
    Used to detect back-to-back games.
    """
    scrape_result = scrape_nhl_odds(target_date)
    if not scrape_result['success']:
        return set()

    teams = set()
    for game in scrape_result['games']:
        home = normalize_abbrev(game.home_abbrev)
        away = normalize_abbrev(game.away_abbrev)
        home = nhl_norm_abbrev(home)
        away = nhl_norm_abbrev(away)
        teams.add(home)
        teams.add(away)

    return teams


def calculate_line_movement(
    side: str,
    open_home: Optional[float],
    open_away: Optional[float],
    current_home: Optional[float],
    current_away: Optional[float]
) -> Optional[float]:
    """
    Calculate line movement in implied probability terms.
    Positive = line moved in your favor (you're getting better odds now).
    """
    if None in (open_home, open_away, current_home, current_away):
        return None

    if side == 'home':
        open_imp = implied_prob_from_american(open_home)
        current_imp = implied_prob_from_american(current_home)
    else:
        open_imp = implied_prob_from_american(open_away)
        current_imp = implied_prob_from_american(current_away)

    if not (np.isfinite(open_imp) and np.isfinite(current_imp)):
        return None

    # If current implied is lower, you're getting better odds (positive movement)
    return round(open_imp - current_imp, 4)
