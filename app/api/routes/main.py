"""Main application routes."""

from flask import Blueprint, render_template

main_bp = Blueprint('main', __name__)


@main_bp.route('/')
def index():
    """Home page."""
    return render_template('main.html')


@main_bp.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors."""
    return render_template('error.html', error='Page not found'), 404


@main_bp.errorhandler(500)  
def internal_error(error):
    """Handle 500 errors."""
    return render_template('error.html', error='Internal server error'), 500