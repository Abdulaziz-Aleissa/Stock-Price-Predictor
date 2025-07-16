"""Dashboard route."""

from flask import Blueprint, render_template
from flask_login import login_required, current_user
import logging

from ...services.portfolio_service import PortfolioService
from ...services.alert_service import AlertService
from ...config.database import get_db


logger = logging.getLogger(__name__)

dashboard_bp = Blueprint('dashboard', __name__)


@dashboard_bp.route('/dashboard')
@login_required
def dashboard():
    """Main dashboard page."""
    try:
        db = get_db()
        portfolio_service = PortfolioService(db)
        alert_service = AlertService(db)
        
        # Get user data
        portfolio_data = portfolio_service.get_portfolio(current_user.id)
        watchlist_data = portfolio_service.get_watchlist(current_user.id)
        portfolio_summary = portfolio_service.get_portfolio_summary(current_user.id)
        alerts_data = alert_service.get_user_alerts(current_user.id)
        notifications = alert_service.get_user_notifications(current_user.id, unread_only=True, limit=10)
        
        logger.info(f"Dashboard loaded for user {current_user.username}")
        
        return render_template(
            'dashboard.html',
            portfolio=portfolio_data,
            watchlist=watchlist_data,
            alerts=alerts_data,
            notifications=notifications,
            summary=portfolio_summary
        )
        
    except Exception as e:
        logger.error(f"Error loading dashboard for user {current_user.id}: {str(e)}")
        return render_template('error.html', error="Unable to load dashboard")
    finally:
        if 'db' in locals():
            db.close()