/**
 * Sports Betting Dashboard - Main JavaScript
 */

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Format a spread value with + or - prefix
 */
function formatSpread(spread) {
    if (spread === null || spread === undefined) return '--';
    return spread > 0 ? `+${spread}` : spread.toString();
}

/**
 * Format an edge value
 */
function formatEdge(edge) {
    if (edge === null || edge === undefined) return '--';
    return edge > 0 ? `+${edge.toFixed(1)}` : edge.toFixed(1);
}

/**
 * Get CSS class for edge value
 */
function getEdgeClass(edge) {
    if (!edge) return '';
    if (edge > 1) return 'edge-positive-strong';
    if (edge > 0) return 'edge-positive';
    if (edge < -1) return 'edge-negative-strong';
    if (edge < 0) return 'edge-negative';
    return '';
}

/**
 * Format percentage
 */
function formatPercent(value) {
    if (value === null || value === undefined) return '--';
    return `${(value * 100).toFixed(1)}%`;
}

/**
 * Format currency
 */
function formatCurrency(value) {
    if (value === null || value === undefined) return '--';
    const prefix = value >= 0 ? '+' : '';
    return `${prefix}$${Math.abs(value).toFixed(2)}`;
}

// ============================================================================
// API Functions
// ============================================================================

/**
 * Fetch predictions for a sport
 */
async function fetchPredictions(sport) {
    try {
        const response = await fetch(`/api/predictions/${sport}`);
        if (!response.ok) throw new Error('Network response was not ok');
        return await response.json();
    } catch (error) {
        console.error(`Error fetching ${sport} predictions:`, error);
        return null;
    }
}

/**
 * Fetch edge data for a sport
 */
async function fetchEdgeData(sport) {
    try {
        const response = await fetch(`/api/edge/${sport}`);
        if (!response.ok) throw new Error('Network response was not ok');
        return await response.json();
    } catch (error) {
        console.error(`Error fetching ${sport} edge data:`, error);
        return null;
    }
}

/**
 * Fetch historical data for a sport
 */
async function fetchHistorical(sport) {
    try {
        const response = await fetch(`/api/historical/${sport}`);
        if (!response.ok) throw new Error('Network response was not ok');
        return await response.json();
    } catch (error) {
        console.error(`Error fetching ${sport} historical data:`, error);
        return null;
    }
}

// ============================================================================
// Table Rendering
// ============================================================================

/**
 * Render predictions into a table body
 */
function renderPredictionsTable(tableBodyId, predictions) {
    const tbody = document.getElementById(tableBodyId);
    if (!tbody) return;

    if (!predictions || predictions.length === 0) {
        tbody.innerHTML = '<tr><td colspan="11">No predictions available</td></tr>';
        return;
    }

    tbody.innerHTML = predictions.map(game => `
        <tr data-confidence="${game.confidence || 0}" data-edge="${game.edge_spread || 0}">
            <td>${game.away_team} @ ${game.home_team}</td>
            <td><strong>${game.predicted_winner}</strong></td>
            <td>${formatSpread(game.predicted_spread)}</td>
            <td>${formatSpread(game.opening_line)}</td>
            <td>${formatSpread(game.closing_line)}</td>
            <td class="${getEdgeClass(game.edge_spread)}">${formatEdge(game.edge_spread)}</td>
            <td>${game.predicted_total || '--'}</td>
            <td>${game.opening_total || '--'}</td>
            <td>${game.closing_total || '--'}</td>
            <td class="${getEdgeClass(game.edge_total)}">${formatEdge(game.edge_total)}</td>
            <td>${formatPercent(game.confidence)}</td>
        </tr>
    `).join('');
}

// ============================================================================
// Notifications
// ============================================================================

/**
 * Show a notification message
 */
function showNotification(message, type = 'info') {
    // Create notification element if it doesn't exist
    let container = document.getElementById('notification-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'notification-container';
        container.style.cssText = `
            position: fixed;
            top: 80px;
            right: 20px;
            z-index: 1000;
            display: flex;
            flex-direction: column;
            gap: 10px;
        `;
        document.body.appendChild(container);
    }

    const notification = document.createElement('div');
    notification.style.cssText = `
        padding: 1rem 1.5rem;
        border-radius: 8px;
        color: white;
        font-weight: 500;
        animation: slideIn 0.3s ease;
        background-color: ${type === 'error' ? '#ef4444' : type === 'success' ? '#22c55e' : '#2563eb'};
    `;
    notification.textContent = message;
    container.appendChild(notification);

    // Auto-remove after 3 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// ============================================================================
// Local Storage
// ============================================================================

/**
 * Save user preferences to local storage
 */
function savePreference(key, value) {
    localStorage.setItem(`bet_dashboard_${key}`, JSON.stringify(value));
}

/**
 * Load user preference from local storage
 */
function loadPreference(key, defaultValue = null) {
    const stored = localStorage.getItem(`bet_dashboard_${key}`);
    return stored ? JSON.parse(stored) : defaultValue;
}

// ============================================================================
// Initialization
// ============================================================================

document.addEventListener('DOMContentLoaded', function() {
    // Add CSS animations
    const style = document.createElement('style');
    style.textContent = `
        @keyframes slideIn {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        @keyframes slideOut {
            from { transform: translateX(0); opacity: 1; }
            to { transform: translateX(100%); opacity: 0; }
        }
    `;
    document.head.appendChild(style);

    console.log('Sports Betting Dashboard initialized');
});
