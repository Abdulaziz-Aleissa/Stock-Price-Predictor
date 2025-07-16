"""Validation middleware."""

from functools import wraps
from flask import request, jsonify
import logging

from ...core.constants import VALIDATION_RULES
from ...core.exceptions import ValidationError


logger = logging.getLogger(__name__)


def validate_ticker_symbol(f):
    """Decorator to validate ticker symbol in request."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check for ticker in form data, JSON, or URL parameter
        ticker = None
        
        if request.method == 'POST':
            if request.is_json:
                ticker = request.json.get('ticker') or request.json.get('symbol')
            else:
                ticker = request.form.get('ticker') or request.form.get('symbol')
        elif 'symbol' in kwargs:
            ticker = kwargs['symbol']
        elif 'ticker' in request.args:
            ticker = request.args.get('ticker')
        
        if ticker:
            ticker = ticker.upper().strip()
            
            # Validate ticker format
            if not ticker or len(ticker) < VALIDATION_RULES['TICKER_MIN_LENGTH']:
                return jsonify({
                    'error': 'Invalid ticker symbol',
                    'message': f'Ticker must be at least {VALIDATION_RULES["TICKER_MIN_LENGTH"]} character(s)'
                }), 400
            
            if len(ticker) > VALIDATION_RULES['TICKER_MAX_LENGTH']:
                return jsonify({
                    'error': 'Invalid ticker symbol',
                    'message': f'Ticker must be at most {VALIDATION_RULES["TICKER_MAX_LENGTH"]} characters'
                }), 400
            
            # Update the ticker in the request context
            if 'symbol' in kwargs:
                kwargs['symbol'] = ticker
        
        return f(*args, **kwargs)
    return decorated_function


def validate_price_data(f):
    """Decorator to validate price and financial data."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            # Get data from form or JSON
            if request.is_json:
                data = request.json
            else:
                data = request.form.to_dict()
            
            # Validate price fields
            price_fields = ['price', 'target_price', 'purchase_price']
            for field in price_fields:
                if field in data:
                    try:
                        value = float(data[field])
                        if value <= VALIDATION_RULES['PRICE_MIN_VALUE']:
                            return jsonify({
                                'error': f'Invalid {field}',
                                'message': f'{field} must be greater than ${VALIDATION_RULES["PRICE_MIN_VALUE"]}'
                            }), 400
                        
                        if value > VALIDATION_RULES['PRICE_MAX_VALUE']:
                            return jsonify({
                                'error': f'Invalid {field}',
                                'message': f'{field} must be less than ${VALIDATION_RULES["PRICE_MAX_VALUE"]}'
                            }), 400
                    
                    except ValueError:
                        return jsonify({
                            'error': f'Invalid {field}',
                            'message': f'{field} must be a valid number'
                        }), 400
            
            # Validate shares
            if 'shares' in data:
                try:
                    shares = float(data['shares'])
                    if shares <= VALIDATION_RULES['SHARES_MIN_VALUE']:
                        return jsonify({
                            'error': 'Invalid shares',
                            'message': f'Shares must be greater than {VALIDATION_RULES["SHARES_MIN_VALUE"]}'
                        }), 400
                    
                    if shares > VALIDATION_RULES['SHARES_MAX_VALUE']:
                        return jsonify({
                            'error': 'Invalid shares',
                            'message': f'Shares must be less than {VALIDATION_RULES["SHARES_MAX_VALUE"]}'
                        }), 400
                
                except ValueError:
                    return jsonify({
                        'error': 'Invalid shares',
                        'message': 'Shares must be a valid number'
                    }), 400
            
            return f(*args, **kwargs)
            
        except Exception as e:
            logger.error(f"Error in price validation: {str(e)}")
            return jsonify({
                'error': 'Validation error',
                'message': 'An error occurred during validation'
            }), 500
    
    return decorated_function


def validate_alert_condition(f):
    """Decorator to validate alert conditions."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Get condition from form or JSON
        condition = None
        if request.is_json:
            condition = request.json.get('condition')
        else:
            condition = request.form.get('condition')
        
        if condition:
            condition = condition.lower().strip()
            valid_conditions = ['above', 'below']
            
            if condition not in valid_conditions:
                return jsonify({
                    'error': 'Invalid condition',
                    'message': f'Condition must be one of: {", ".join(valid_conditions)}'
                }), 400
        
        return f(*args, **kwargs)
    return decorated_function


def sanitize_input(f):
    """Decorator to sanitize input data."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Basic input sanitization
        if request.is_json:
            data = request.json
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, str):
                        # Remove potentially dangerous characters
                        data[key] = value.strip()
        
        return f(*args, **kwargs)
    return decorated_function