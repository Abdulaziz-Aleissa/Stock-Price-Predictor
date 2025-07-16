"""Authentication middleware."""

from functools import wraps
from flask import request, jsonify, current_app
from flask_login import current_user
import logging


logger = logging.getLogger(__name__)


def api_login_required(f):
    """Decorator for API routes that require authentication."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            return jsonify({
                'error': 'Authentication required',
                'message': 'Please log in to access this resource'
            }), 401
        return f(*args, **kwargs)
    return decorated_function


def admin_required(f):
    """Decorator for routes that require admin privileges."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            return jsonify({
                'error': 'Authentication required',
                'message': 'Please log in to access this resource'
            }), 401
        
        # Check if user has admin privileges (extend User model if needed)
        if not getattr(current_user, 'is_admin', False):
            return jsonify({
                'error': 'Admin access required',
                'message': 'You do not have permission to access this resource'
            }), 403
        
        return f(*args, **kwargs)
    return decorated_function


def validate_json(required_fields=None):
    """Decorator to validate JSON request data."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not request.is_json:
                return jsonify({
                    'error': 'Content-Type must be application/json'
                }), 400
            
            data = request.get_json()
            if not data:
                return jsonify({
                    'error': 'No JSON data provided'
                }), 400
            
            if required_fields:
                missing_fields = [field for field in required_fields if field not in data]
                if missing_fields:
                    return jsonify({
                        'error': 'Missing required fields',
                        'missing_fields': missing_fields
                    }), 400
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator


def rate_limit_by_user(max_requests=100, window_minutes=60):
    """Simple rate limiting by user (in-memory, not persistent)."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # This is a simple implementation - in production, use Redis or similar
            if current_user.is_authenticated:
                user_id = current_user.id
                # Rate limiting logic would go here
                # For now, just log the request
                logger.info(f"API request from user {user_id}")
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator