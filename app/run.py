
"""
Refactored Stock Price Predictor Application
Main application file using modular architecture
"""

from flask import Flask, render_template
import os
import logging
from flask_login import LoginManager, login_required
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
import atexit
from dotenv import load_dotenv

# Import modular components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.data.yfinance_data import yfinance_data
from app.database.db_operations import db_operations
from app.auth.auth_module import auth_manager
from app.components.watchlist_alerts import watchlist_alerts_manager
from app.api.api_routes import api_routes

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__,
           static_url_path='/static',
           template_folder='templates',
           static_folder='static')

app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-here')  # Use env var or fallback
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup LoginManager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Initialize scheduler for price alerts
scheduler = BackgroundScheduler()
scheduler.start()
atexit.register(lambda: scheduler.shutdown())

@login_manager.user_loader
def load_user(user_id):
    """Load user for Flask-Login"""
    return db_operations.get_user(int(user_id))


def check_price_alerts():
    """Check price alerts and trigger notifications"""
    with app.app_context():
        watchlist_alerts_manager.check_price_alerts()


# Schedule alert checker
scheduler.add_job(
    func=check_price_alerts,
    trigger=IntervalTrigger(minutes=5),
    id='check_price_alerts',
    name='Check price alerts every 5 minutes'
)


# Route definitions using modular components
@app.route('/')
def index():
    """Main index page"""
    return api_routes.index_route()


@app.route('/financial-literacy')
def financial_literacy():
    """Financial literacy page with paper trading"""
    return api_routes.financial_literacy_route()


@app.route('/dashboard')
def dashboard():
    """Main dashboard page"""
    return api_routes.dashboard_route()


# Authentication routes
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """User signup"""
    return api_routes.signup_route()


@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    return api_routes.login_route()


@app.route('/logout')
def logout():
    """User logout"""
    return api_routes.logout_route()


# API endpoints
@app.route('/stock_scoring', methods=['POST'])
def stock_scoring():
    """Handle stock scoring requests"""
    return api_routes.stock_scoring_route()


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    return api_routes.predict_route()


@app.route('/run_backtest', methods=['POST'])
def run_backtest():
    """Handle backtest requests"""
    return api_routes.run_backtest_route()


# Portfolio management routes
@app.route('/add_to_portfolio', methods=['POST'])
def add_to_portfolio():
    """Add stock to portfolio"""
    return api_routes.add_to_portfolio_route()


@app.route('/add_to_watchlist', methods=['POST'])
def add_to_watchlist():
    """Add stock to watchlist"""
    return api_routes.add_to_watchlist_route()


@app.route('/add_alert', methods=['POST'])
def add_alert():
    """Add price alert"""
    return api_routes.add_alert_route()


@app.route('/remove_from_portfolio/<int:item_id>')
def remove_from_portfolio(item_id):
    """Remove item from portfolio"""
    return api_routes.remove_from_portfolio_route(item_id)


@app.route('/remove_from_watchlist/<int:item_id>')
def remove_from_watchlist(item_id):
    """Remove item from watchlist"""
    return api_routes.remove_from_watchlist_route(item_id)


@app.route('/remove_alert/<int:alert_id>')
def remove_alert(alert_id):
    """Remove price alert"""
    return api_routes.remove_alert_route(alert_id)


@app.route('/mark_notification_read/<int:notification_id>')
def mark_notification_read(notification_id):
    """Mark notification as read"""
    return api_routes.mark_notification_read_route(notification_id)


# Paper trading routes
@app.route('/paper_buy', methods=['POST'])
def paper_buy():
    """Execute paper buy order"""
    return api_routes.paper_buy_route()


@app.route('/paper_sell', methods=['POST'])
def paper_sell():
    """Execute paper sell order"""
    return api_routes.paper_sell_route()


@app.route('/paper_transactions')
def paper_transactions():
    """Get paper trading transactions"""
    return api_routes.paper_transactions_route()


@app.route('/reset_paper_portfolio')
def reset_paper_portfolio():
    """Reset paper portfolio"""
    return api_routes.reset_paper_portfolio_route()


@app.route('/get_stock_price/<symbol>')
def get_stock_price(symbol):
    """Get current stock price"""
    return api_routes.get_stock_price_route(symbol)


# Additional utility routes can be added here as needed
# For example: compare_stocks, value_at_risk, time_series_forecasting, options_pricing
# These would be implemented in api_routes.py following the same pattern

# Advanced Analytics Routes
@app.route('/compare_stocks', methods=['POST'])
@login_required
def compare_stocks():
    """Handle stock comparison requests"""
    return api_routes.compare_stocks_route()


@app.route('/value_at_risk', methods=['POST'])
@login_required
def value_at_risk():
    """Handle Value at Risk analysis requests"""
    return api_routes.value_at_risk_route()


@app.route('/time_series_forecasting', methods=['POST'])
@login_required
def time_series_forecasting():
    """Handle time series forecasting requests"""
    return api_routes.time_series_forecasting_route()


@app.route('/options_pricing', methods=['POST'])
@login_required
def options_pricing():
    """Handle options pricing requests"""
    return api_routes.options_pricing_route()


# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors"""
    return render_template('error.html', error='Page not found'), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return render_template('error.html', error='Internal server error'), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
