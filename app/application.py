"""Factory function to create the Flask application."""

from flask import Flask
from flask_login import LoginManager
from apscheduler.schedulers.background import BackgroundScheduler
import atexit
import logging

from .config.settings import get_config
from .config.database import init_database, create_tables
from .models.user import User
from .services.alert_service import AlertService
from .services.stock_service import StockService
from .config.database import get_db


def create_app(config_name=None):
    """Create and configure the Flask application."""
    
    # Create Flask app
    app = Flask(__name__, 
                static_url_path='/static',
                template_folder='templates',
                static_folder='static')
    
    # Load configuration
    config = get_config(config_name)
    app.config.from_object(config)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize database
    with app.app_context():
        init_database(config)
        create_tables()
    
    # Setup LoginManager
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'
    login_manager.login_message = 'Please log in to access this page.'
    login_manager.login_message_category = 'info'
    
    @login_manager.user_loader
    def load_user(user_id):
        """Load user by ID."""
        db = get_db()
        try:
            return db.get(User, int(user_id))
        finally:
            db.close()
    
    # Register blueprints
    register_blueprints(app)
    
    # Setup background scheduler for alerts
    setup_scheduler(app)
    
    # Register error handlers
    register_error_handlers(app)
    
    # Add template filters
    register_template_filters(app)
    
    logger.info(f"Application created with config: {config.__class__.__name__}")
    
    return app


def register_blueprints(app):
    """Register all blueprints."""
    from .api.routes.main import main_bp
    from .api.routes.auth import auth_bp
    from .api.routes.stock import stock_bp
    from .api.routes.portfolio import portfolio_bp
    from .api.routes.dashboard import dashboard_bp
    from .api.routes.alerts import alert_bp
    
    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(stock_bp)
    app.register_blueprint(portfolio_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(alert_bp)


def setup_scheduler(app):
    """Setup background scheduler for periodic tasks."""
    from apscheduler.triggers.interval import IntervalTrigger
    
    scheduler = BackgroundScheduler()
    
    def check_price_alerts():
        """Check all price alerts and trigger notifications."""
        with app.app_context():
            try:
                db = get_db()
                alert_service = AlertService(db)
                triggered_alerts = alert_service.check_all_alerts()
                
                if triggered_alerts:
                    app.logger.info(f"Triggered {len(triggered_alerts)} alerts")
                
            except Exception as e:
                app.logger.error(f"Error checking alerts: {str(e)}")
            finally:
                if 'db' in locals():
                    db.close()
    
    # Schedule alert checker every 5 minutes
    scheduler.add_job(
        func=check_price_alerts,
        trigger=IntervalTrigger(minutes=app.config.get('ALERT_CHECK_INTERVAL_MINUTES', 5)),
        id='check_price_alerts',
        name='Check price alerts every 5 minutes',
        replace_existing=True
    )
    
    scheduler.start()
    
    # Shut down the scheduler when exiting the app
    atexit.register(lambda: scheduler.shutdown())


def register_error_handlers(app):
    """Register error handlers."""
    from flask import render_template
    
    @app.errorhandler(404)
    def not_found_error(error):
        return render_template('error.html', error='Page not found'), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return render_template('error.html', error='Internal server error'), 500
    
    @app.errorhandler(403)
    def forbidden_error(error):
        return render_template('error.html', error='Access forbidden'), 403


def register_template_filters(app):
    """Register custom template filters."""
    from .utils.formatters import format_currency, format_percentage
    from .utils.helpers import calculate_time_ago
    
    @app.template_filter('currency')
    def currency_filter(value):
        """Format value as currency."""
        return format_currency(value)
    
    @app.template_filter('percentage')
    def percentage_filter(value):
        """Format value as percentage."""
        return format_percentage(value)
    
    @app.template_filter('timeago')
    def timeago_filter(value):
        """Format datetime as time ago."""
        return calculate_time_ago(value)


# Global application instance
app = None


def get_app():
    """Get the application instance."""
    global app
    if app is None:
        app = create_app()
    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)