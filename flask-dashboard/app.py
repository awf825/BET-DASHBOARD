"""
Sports Betting Dashboard - Flask Application
"""
import os
from flask import Flask, render_template, jsonify, request, send_from_directory
from config import Config

app = Flask(__name__)
app.config.from_object(Config)


# ============================================================================
# ROUTES - Pages
# ============================================================================

@app.route('/')
def index():
    """Main dashboard landing page."""
    return render_template('index.html')


@app.route('/nba')
def nba_dashboard():
    """NBA predictions dashboard."""
    return render_template('nba.html', sport='NBA')


@app.route('/nhl')
def nhl_dashboard():
    """NHL predictions dashboard."""
    return render_template('nhl.html', sport='NHL')


@app.route('/mlb')
def mlb_dashboard():
    """MLB predictions dashboard."""
    return render_template('mlb.html', sport='MLB')


@app.route('/edge-analysis')
def edge_analysis():
    """Edge analysis - compare predictions vs lines."""
    return render_template('edge_analysis.html')


@app.route('/tools')
def tools():
    """Tools and utilities page."""
    return render_template('tools.html')


# ============================================================================
# API ROUTES - Data endpoints for AJAX calls
# ============================================================================

@app.route('/api/predictions/<sport>')
def get_predictions(sport):
    """
    Get predictions for a specific sport.
    TODO: Integrate with actual model predictions from python-{sport}-2026
    """
    # Placeholder data structure
    mock_data = {
        'sport': sport.upper(),
        'predictions': [
            {
                'game_id': '001',
                'home_team': 'Team A',
                'away_team': 'Team B',
                'predicted_winner': 'Team A',
                'predicted_spread': -3.5,
                'predicted_total': 215.5,
                'confidence': 0.68,
                'opening_line': -4.0,
                'closing_line': -3.5,
                'edge_spread': 0.5,
                'edge_total': 2.0
            }
        ]
    }
    return jsonify(mock_data)


@app.route('/api/edge/<sport>')
def get_edge_data(sport):
    """
    Get edge analysis data comparing model predictions vs lines.
    TODO: Implement actual edge calculation logic
    """
    mock_edge = {
        'sport': sport.upper(),
        'total_games_analyzed': 150,
        'positive_edge_games': 42,
        'avg_edge_spread': 1.2,
        'avg_edge_total': 0.8,
        'roi_if_bet_all_edges': 4.5,
        'edge_by_confidence': [
            {'confidence_range': '60-65%', 'avg_edge': 0.5, 'count': 25},
            {'confidence_range': '65-70%', 'avg_edge': 1.1, 'count': 12},
            {'confidence_range': '70%+', 'avg_edge': 2.3, 'count': 5}
        ]
    }
    return jsonify(mock_edge)


@app.route('/api/historical/<sport>')
def get_historical(sport):
    """
    Get historical performance data.
    TODO: Pull from Azure data storage once deployed
    """
    return jsonify({
        'sport': sport.upper(),
        'historical_accuracy': 0.54,
        'total_predictions': 500,
        'profit_loss': 12.5,
        'message': 'Historical data endpoint - connect to Azure storage'
    })


# ============================================================================
# NHL API ROUTES - Live predictions
# ============================================================================

@app.route('/api/nhl/status')
def nhl_status():
    """Check if NHL predictions are available."""
    from utils.nhl_runner import check_nhl_available
    return jsonify(check_nhl_available())


@app.route('/api/nhl/run', methods=['POST'])
def run_nhl_predictions():
    """
    Run NHL predictions.

    POST body (JSON):
        - edge_strategy: 'opening' or 'closing' (required)
        - date_type: 'today' or 'tomorrow' (optional, used if date not provided)
        - date: YYYY-MM-DD (optional, overrides date_type)
        - bankroll: float (optional, default 5000)
        - enabled_segments: dict with opening/closing segment toggles (optional)
    """
    from utils.nhl_runner import run_nhl_predictions as run_predictions, get_date_string

    data = request.get_json() or {}
    edge_strategy = data.get('edge_strategy', 'closing')
    date = data.get('date')
    date_type = data.get('date_type')
    bankroll = data.get('bankroll', 5000.0)
    enabled_segments = data.get('enabled_segments')

    if edge_strategy not in ('opening', 'closing'):
        return jsonify({
            'success': False,
            'error': 'edge_strategy must be "opening" or "closing"'
        }), 400

    # If date_type is provided but date is not, convert date_type to date
    if date is None and date_type in ('today', 'tomorrow'):
        date = get_date_string(date_type)

    result = run_predictions(
        edge_strategy=edge_strategy,
        date=date,
        bankroll=bankroll,
        enabled_segments=enabled_segments,
    )

    return jsonify(result)


@app.route('/api/nhl/today')
def nhl_today():
    """Get today's NHL predictions using closing line strategy."""
    from utils.nhl_runner import run_nhl_predictions as run_predictions
    return jsonify(run_predictions(edge_strategy='closing'))


@app.route('/api/nhl/tomorrow')
def nhl_tomorrow():
    """Get tomorrow's NHL predictions using opening line strategy."""
    from utils.nhl_runner import run_nhl_predictions as run_predictions
    return jsonify(run_predictions(edge_strategy='opening'))


@app.route('/api/nhl/visuals/<filename>')
def nhl_visuals(filename):
    """Serve NHL visualization images (heatmaps)."""
    visuals_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../python-nhl-2026/visuals'))
    return send_from_directory(visuals_dir, filename)


@app.route('/api/nhl/visuals')
def nhl_visuals_list():
    """List available NHL visualizations."""
    visuals_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../python-nhl-2026/visuals'))
    try:
        files = [f for f in os.listdir(visuals_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.svg'))]
        return jsonify({
            'success': True,
            'visuals': [{'name': f, 'url': f'/api/nhl/visuals/{f}'} for f in sorted(files)]
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ============================================================================
# NBA API ROUTES - Live predictions with spreads
# ============================================================================

@app.route('/api/nba/status')
def nba_status():
    """Check if NBA predictions are available."""
    from utils.nba_runner import check_nba_available
    return jsonify(check_nba_available())


@app.route('/api/nba/run', methods=['POST'])
def run_nba_predictions():
    """
    Run NBA spread predictions.

    POST body (JSON):
        - edge_strategy: 'opening' or 'closing' (required)
        - date: YYYY-MM-DD (optional, defaults based on strategy)
        - bankroll: float (optional, default 5000)
    """
    from utils.nba_runner import run_nba_predictions as run_predictions

    data = request.get_json() or {}
    edge_strategy = data.get('edge_strategy', 'closing')
    date = data.get('date')
    bankroll = data.get('bankroll', 5000.0)

    if edge_strategy not in ('opening', 'closing'):
        return jsonify({
            'success': False,
            'error': 'edge_strategy must be "opening" or "closing"'
        }), 400

    result = run_predictions(
        edge_strategy=edge_strategy,
        date=date,
        bankroll=bankroll
    )

    return jsonify(result)


@app.route('/api/nba/today')
def nba_today():
    """Get today's NBA predictions using closing spread strategy."""
    from utils.nba_runner import run_nba_predictions as run_predictions
    return jsonify(run_predictions(edge_strategy='closing'))


@app.route('/api/nba/tomorrow')
def nba_tomorrow():
    """Get tomorrow's NBA predictions using opening spread strategy."""
    from utils.nba_runner import run_nba_predictions as run_predictions
    return jsonify(run_predictions(edge_strategy='opening'))


@app.route('/api/nba/team-details/<int:team_id>')
def nba_team_details(team_id):
    """
    Lazy-load team details (age, recent games) when user expands a row.
    """
    from utils.nba_runner import get_team_details, TEAM_NAME_TO_ID

    # Find team name from ID
    team_name = None
    for name, tid in TEAM_NAME_TO_ID.items():
        if tid == team_id:
            team_name = name
            break

    if team_name is None:
        return jsonify({
            'success': False,
            'error': f'Unknown team ID: {team_id}'
        }), 404

    return jsonify(get_team_details(team_id, team_name))


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404


@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500


# ============================================================================
# RUN
# ============================================================================

if __name__ == '__main__':
    app.run(debug=True, port=5000)
