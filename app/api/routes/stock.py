"""Stock-related routes."""

from flask import Blueprint, render_template, request, jsonify
from flask_login import login_required
import logging
from datetime import datetime

from ...services.stock_service import StockService
from ...services.prediction_service import PredictionService
from ...core.exceptions import InvalidTickerError, DataFetchError, PredictionError
from ...config.database import get_db


logger = logging.getLogger(__name__)

stock_bp = Blueprint('stock', __name__)

# Initialize services
stock_service = StockService()
prediction_service = PredictionService()


@stock_bp.route('/')
def index():
    """Main page for stock prediction."""
    return render_template('main.html')


@stock_bp.route('/predict', methods=['POST'])
def predict():
    """Stock price prediction endpoint."""
    try:
        stock_ticker = request.form.get('ticker', '').upper().strip()
        
        if not stock_ticker:
            return render_template('error.html', error="Please provide a ticker symbol")
        
        # Validate ticker
        if not stock_service.is_valid_ticker(stock_ticker):
            return render_template('error.html', error=f"Invalid ticker symbol: {stock_ticker}")
        
        logger.info(f"Processing prediction request for {stock_ticker}")
        
        # Get stock data
        try:
            df = stock_service.get_historical_data(stock_ticker)
            current_price = stock_service.get_current_price(stock_ticker)
            market_context = stock_service.get_market_context(stock_ticker)
            
            if not current_price:
                return render_template('error.html', error="Could not fetch current price")
            
        except DataFetchError as e:
            return render_template('error.html', error=str(e))
        
        # Generate prediction
        try:
            prediction_result = prediction_service.predict_price(
                stock_ticker, df, current_price
            )
            
            logger.info(f"Prediction completed for {stock_ticker}: "
                       f"${prediction_result['predicted_price']:.2f} "
                       f"({prediction_result['price_change_percent']:+.2f}%)")
            
            return render_template(
                'go.html',
                ticker=stock_ticker,
                prediction=round(prediction_result['predicted_price'], 2),
                current_price=round(current_price, 2),
                price_change_pct=round(prediction_result['price_change_percent'], 2),
                dates=prediction_result['chart_data']['dates'],
                actual_prices=prediction_result['chart_data']['actual_prices'],
                predicted_prices=prediction_result['chart_data']['predicted_prices'],
                market_context=market_context,
                current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                confidence_metrics={
                    'r2_score': f"{prediction_result['confidence_metrics']['r2_score']:.3f}",
                    'mae': f"${prediction_result['confidence_metrics']['mae']:.2f}",
                    'rmse': f"${prediction_result['confidence_metrics']['rmse']:.2f}"
                }
            )
            
        except PredictionError as e:
            logger.error(f"Prediction error for {stock_ticker}: {str(e)}")
            return render_template('error.html', error=f"Prediction failed: {str(e)}")
    
    except Exception as e:
        logger.error(f"Unexpected error in prediction: {str(e)}")
        return render_template('error.html', error="An unexpected error occurred")


@stock_bp.route('/api/stock/<symbol>/info')
def get_stock_info(symbol):
    """API endpoint for stock information."""
    try:
        symbol = symbol.upper()
        
        if not stock_service.is_valid_ticker(symbol):
            return jsonify({'error': 'Invalid ticker symbol'}), 400
        
        info = stock_service.get_market_context(symbol)
        if not info:
            return jsonify({'error': 'Could not fetch stock information'}), 404
        
        return jsonify(info)
        
    except Exception as e:
        logger.error(f"Error getting stock info for {symbol}: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


@stock_bp.route('/api/stock/<symbol>/price')
def get_current_price(symbol):
    """API endpoint for current stock price."""
    try:
        symbol = symbol.upper()
        
        if not stock_service.is_valid_ticker(symbol):
            return jsonify({'error': 'Invalid ticker symbol'}), 400
        
        price = stock_service.get_current_price(symbol)
        if price is None:
            return jsonify({'error': 'Could not fetch current price'}), 404
        
        return jsonify({
            'symbol': symbol,
            'price': price,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting price for {symbol}: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


@stock_bp.route('/compare_stocks', methods=['POST'])
@login_required
def compare_stocks():
    """Stock comparison endpoint."""
    try:
        symbol1 = request.form.get('symbol1', '').upper().strip()
        symbol2 = request.form.get('symbol2', '').upper().strip()
        timeframe = request.form.get('timeframe', '1y')
        
        if not symbol1 or not symbol2:
            return jsonify({'error': 'Please provide both stock symbols'}), 400
        
        if not stock_service.is_valid_ticker(symbol1):
            return jsonify({'error': f'Invalid ticker symbol: {symbol1}'}), 400
        
        if not stock_service.is_valid_ticker(symbol2):
            return jsonify({'error': f'Invalid ticker symbol: {symbol2}'}), 400
        
        comparison_data = stock_service.compare_stocks(symbol1, symbol2, timeframe)
        
        logger.info(f"Stock comparison completed: {symbol1} vs {symbol2}")
        return jsonify(comparison_data)
        
    except DataFetchError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Error comparing stocks: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


@stock_bp.route('/api/validate-ticker/<symbol>')
def validate_ticker(symbol):
    """API endpoint to validate ticker symbol."""
    try:
        symbol = symbol.upper()
        is_valid = stock_service.is_valid_ticker(symbol)
        
        return jsonify({
            'symbol': symbol,
            'valid': is_valid
        })
        
    except Exception as e:
        logger.error(f"Error validating ticker {symbol}: {str(e)}")
        return jsonify({
            'symbol': symbol,
            'valid': False,
            'error': 'Validation failed'
        })