"""
API Routes Module
Handle all API endpoints
"""

from flask import request, jsonify, render_template, redirect, url_for, flash
from flask_login import login_required, current_user
from app.auth.auth_module import auth_routes
from app.components.scoring_module import stock_scoring_manager
from app.components.portfolio_holdings import portfolio_holdings_manager
from app.components.portfolio_summary import portfolio_summary_calculator
from app.components.watchlist_alerts import watchlist_alerts_manager
from app.data.yfinance_data import yfinance_data
from app.database.db_operations import db_operations
from app.components.options_module import OptionsManager
from app.components.risk_module import RiskManager
from app.components.forecast_module import ForecastManager
from app.components.compare_module import compare_manager
from models.train_classifier import evaluate_model, load_data as load_db_data
from sklearn.ensemble import GradientBoostingRegressor
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class APIRoutes:
    """Handle all API routes"""
    
    def __init__(self):
        self.scoring = stock_scoring_manager
        self.portfolio_holdings = portfolio_holdings_manager
        self.portfolio_summary = portfolio_summary_calculator
        self.watchlist_alerts = watchlist_alerts_manager
        self.data_fetcher = yfinance_data
        self.db_ops = db_operations
        self.auth_routes = auth_routes
        self.options_manager = OptionsManager()
        self.risk_manager = RiskManager()
        self.forecast_manager = ForecastManager()
        self.compare_manager = compare_manager
    
    # Authentication routes
    def signup_route(self):
        """Handle signup route"""
        return self.auth_routes.signup_handler()
    
    def login_route(self):
        """Handle login route"""
        return self.auth_routes.login_handler()
    
    def logout_route(self):
        """Handle logout route"""
        return self.auth_routes.logout_handler()
    
    # Main application routes
    def index_route(self):
        """Handle index route"""
        return render_template('main.html')
    
    def financial_literacy_route(self):
        """Handle financial literacy route"""
        if not current_user.is_authenticated:
            return redirect(url_for('login'))
        
        paper_portfolio_data = self.portfolio_summary.calculate_paper_portfolio_summary(current_user.id)
        
        return render_template('financial_literacy.html',
                             paper_portfolio=paper_portfolio_data['paper_portfolio_data'],
                             paper_summary=paper_portfolio_data['paper_summary'])
    
    def dashboard_route(self):
        """Handle dashboard route"""
        if not current_user.is_authenticated:
            return redirect(url_for('login'))
        
        # Get portfolio data
        real_portfolio_data = self.portfolio_summary.calculate_real_portfolio_summary(current_user.id)
        paper_portfolio_data = self.portfolio_summary.calculate_paper_portfolio_summary(current_user.id)
        
        # Get watchlist and alerts data
        watchlist_data = self.watchlist_alerts.get_watchlist_data(current_user.id)
        alerts_data = self.watchlist_alerts.get_alerts_data(current_user.id)
        
        # Get notifications
        notifications = self.watchlist_alerts.get_user_notifications(current_user.id, unread_only=True)
        
        return render_template('dashboard.html',
                             portfolio=real_portfolio_data['portfolio_data'],
                             watchlist=watchlist_data,
                             alerts=alerts_data,
                             notifications=notifications,
                             summary=real_portfolio_data['summary'],
                             paper_portfolio=paper_portfolio_data['paper_portfolio_data'],
                             paper_summary=paper_portfolio_data['paper_summary'])
    
    # API endpoints
    def stock_scoring_route(self):
        """Handle stock scoring requests"""
        try:
            symbols_input = request.form.get('symbols', '').strip()
            
            if not symbols_input:
                return jsonify({'error': 'Please provide at least one stock symbol'})
            
            # Parse symbols (comma-separated)
            symbols = [s.strip().upper() for s in symbols_input.split(',') if s.strip()]
            
            if not symbols:
                return jsonify({'error': 'Please provide valid stock symbols'})
            
            # Limit to reasonable number of stocks
            if len(symbols) > 10:
                return jsonify({'error': 'Please limit analysis to 10 stocks or fewer'})
            
            # Analyze stocks
            results = self.scoring.analyze_multiple_stocks(symbols)
            
            return jsonify({
                'success': True,
                'results': results
            })
            
        except Exception as e:
            logger.error(f"Stock scoring error: {str(e)}")
            return jsonify({'error': f'An error occurred: {str(e)}'})
    
    def predict_route(self):
        """Handle prediction requests"""
        try:
            stock_ticker = request.form.get('ticker', '').strip().upper()  # Changed from 'stock_ticker' to 'ticker'
            
            if not stock_ticker:
                return render_template('error.html', error='Please provide a stock ticker')
            
            if not self.data_fetcher.is_valid_ticker(stock_ticker):
                return render_template('error.html', error=f'Invalid ticker symbol: {stock_ticker}')
            
            # Get market context
            market_context = self.data_fetcher.get_market_context(stock_ticker)
            if not market_context:
                return render_template('error.html', error='Unable to fetch market data')
            
            # Load and prepare data for prediction
            data = load_db_data()
            if data is None or data.empty:
                return render_template('error.html', error='Unable to load training data')
            
            # Get stock data for prediction
            stock_data = self.data_fetcher.get_historical_data(stock_ticker, period="2y")
            if stock_data is None or stock_data.empty:
                return render_template('error.html', error='Unable to fetch stock data')
            
            # Prepare features for prediction
            features = self._prepare_prediction_features(stock_data, market_context)
            if features is None:
                return render_template('error.html', error='Unable to prepare prediction features')
            
            # Load model and make prediction
            try:
                model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                X_train = data[['Open', 'High', 'Low', 'Volume']].fillna(0)
                y_train = data['Close'].fillna(0)
                model.fit(X_train, y_train)
                
                prediction = model.predict([features])[0]
                current_price = stock_data['Close'].iloc[-1]
                
                # Calculate metrics
                metrics = evaluate_model(model, X_train, y_train)
                
                # Store prediction
                price_change_pct = ((prediction - current_price) / current_price) * 100
                self.db_ops.store_prediction(stock_ticker, prediction, current_price, price_change_pct, metrics)
                
                # Get additional data for template
                dates = stock_data.index.strftime('%Y-%m-%d').tolist()
                actual_prices = stock_data['Close'].tolist()
                predicted_prices = [prediction] * len(actual_prices)  # Simplified for now
                
                return render_template(
                    'go.html',
                    ticker=stock_ticker,
                    prediction=round(prediction, 2),
                    current_price=round(current_price, 2),
                    price_change_pct=round(price_change_pct, 2),
                    dates=dates,
                    actual_prices=actual_prices,
                    predicted_prices=predicted_prices,
                    market_context=market_context,
                    current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    confidence_metrics={
                        'r2_score': f"{metrics.get('r2', 0):.3f}",
                        'mae': f"${metrics.get('mae', 0):.2f}",
                        'rmse': f"${metrics.get('rmse', 0):.2f}"
                    },
                    news_articles=[],  # Can be populated later
                    news_summary="",   # Can be populated later
                    news_error_message=""
                )
                
            except Exception as model_error:
                logger.error(f"Model prediction error: {str(model_error)}")
                return render_template('error.html', error='Prediction model failed')
            
        except Exception as e:
            logger.error(f"Prediction route error: {str(e)}")
            return render_template('error.html', error=str(e))
    
    def run_backtest_route(self):
        """Handle backtest requests"""
        try:
            stock_symbol = request.form.get('stock_symbol', '').strip().upper()
            days_back = int(request.form.get('days_back', 365))
            
            if not stock_symbol:
                return jsonify({'error': 'Please provide a stock symbol'})
            
            if not self.data_fetcher.is_valid_ticker(stock_symbol):
                return jsonify({'error': 'Invalid ticker symbol'})
            
            # Calculate backtest metrics
            backtest_results = self._calculate_backtest_metrics(stock_symbol, days_back)
            
            return jsonify({
                'success': True,
                'results': backtest_results
            })
            
        except Exception as e:
            logger.error(f"Backtest error: {str(e)}")
            return jsonify({'error': f'Backtest failed: {str(e)}'})
    
    # Portfolio management routes
    def add_to_portfolio_route(self):
        """Handle add to portfolio requests"""
        if not current_user.is_authenticated:
            return redirect(url_for('login'))
        
        try:
            symbol = request.form['symbol'].upper()
            shares = float(request.form['shares'])
            purchase_price = float(request.form['purchase_price'])
            
            if self.portfolio_holdings.add_real_holding(current_user.id, symbol, shares, purchase_price):
                flash('Stock added to portfolio successfully')
            else:
                flash('Failed to add stock to portfolio')
            
            return redirect(url_for('dashboard'))
            
        except Exception as e:
            logger.error(f"Add to portfolio error: {str(e)}")
            flash('Error adding stock to portfolio')
            return redirect(url_for('dashboard'))
    
    def add_to_watchlist_route(self):
        """Handle add to watchlist requests"""
        if not current_user.is_authenticated:
            return redirect(url_for('login'))
        
        try:
            symbol = request.form['symbol'].upper()
            target_price = float(request.form['target_price'])
            
            if self.watchlist_alerts.add_to_watchlist(current_user.id, symbol, target_price):
                flash('Stock added to watchlist successfully')
            else:
                flash('Failed to add stock to watchlist')
            
            return redirect(url_for('dashboard'))
            
        except Exception as e:
            logger.error(f"Add to watchlist error: {str(e)}")
            flash('Error adding stock to watchlist')
            return redirect(url_for('dashboard'))
    
    def add_alert_route(self):
        """Handle add alert requests"""
        if not current_user.is_authenticated:
            return redirect(url_for('login'))
        
        try:
            symbol = request.form['symbol'].upper()
            target_price = float(request.form['target_price'])
            condition = request.form['condition']
            
            if self.watchlist_alerts.add_price_alert(current_user.id, symbol, condition, target_price):
                flash('Alert added successfully')
            else:
                flash('Failed to add alert')
            
            return redirect(url_for('dashboard'))
            
        except Exception as e:
            logger.error(f"Add alert error: {str(e)}")
            flash('Error adding alert')
            return redirect(url_for('dashboard'))
    
    # Remove routes
    def remove_from_portfolio_route(self, item_id):
        """Handle remove from portfolio requests"""
        if not current_user.is_authenticated:
            return redirect(url_for('login'))
        
        if self.portfolio_holdings.remove_real_holding(item_id):
            flash('Item removed from portfolio')
        else:
            flash('Failed to remove item')
        
        return redirect(url_for('dashboard'))
    
    def remove_from_watchlist_route(self, item_id):
        """Handle remove from watchlist requests"""
        if not current_user.is_authenticated:
            return redirect(url_for('login'))
        
        if self.watchlist_alerts.remove_from_watchlist(item_id):
            flash('Item removed from watchlist')
        else:
            flash('Failed to remove item')
        
        return redirect(url_for('dashboard'))
    
    def remove_alert_route(self, alert_id):
        """Handle remove alert requests"""
        if not current_user.is_authenticated:
            return redirect(url_for('login'))
        
        if self.watchlist_alerts.remove_price_alert(alert_id):
            flash('Alert removed')
        else:
            flash('Failed to remove alert')
        
        return redirect(url_for('dashboard'))
    
    def mark_notification_read_route(self, notification_id):
        """Handle mark notification as read requests"""
        if not current_user.is_authenticated:
            return jsonify({'success': False})
        
        success = self.watchlist_alerts.mark_notification_read(notification_id)
        return jsonify({'success': success})
    
    # Paper trading routes
    def paper_buy_route(self):
        """Handle paper buy requests"""
        if not current_user.is_authenticated:
            return redirect(url_for('login'))
        
        try:
            symbol = request.form.get('symbol', '').strip().upper()
            shares = float(request.form.get('shares', 0))
            price = float(request.form.get('price', 0))
            
            if not symbol:
                flash('Please enter a stock symbol')
                return redirect(url_for('dashboard'))
            
            if shares <= 0:
                flash('Please enter a valid number of shares')
                return redirect(url_for('dashboard'))
            
            if price <= 0:
                flash('Please enter a valid price')
                return redirect(url_for('dashboard'))
            
            # Validate ticker
            if not self.data_fetcher.is_valid_ticker(symbol):
                flash(f'Stock symbol "{symbol}" not found. Please check the symbol and try again.')
                return redirect(url_for('dashboard'))
            
            result = self.portfolio_holdings.execute_paper_buy(current_user.id, symbol, shares)
            
            if result.get('success'):
                flash(f'Successfully bought {shares} shares of {symbol}')
            else:
                flash(f'Failed to buy shares: {result.get("message", "Unknown error")}')
            
            return redirect(url_for('dashboard'))
            
        except Exception as e:
            logger.error(f"Paper buy error: {str(e)}")
            flash('Error executing buy order')
            return redirect(url_for('dashboard'))
    
    def paper_sell_route(self):
        """Handle paper sell requests"""
        if not current_user.is_authenticated:
            return redirect(url_for('login'))
        
        try:
            symbol = request.form.get('symbol', '').strip().upper()
            shares = float(request.form.get('shares', 0))
            price = float(request.form.get('price', 0))
            
            if not symbol:
                flash('Please enter a stock symbol')
                return redirect(url_for('dashboard'))
            
            if shares <= 0:
                flash('Please enter a valid number of shares')
                return redirect(url_for('dashboard'))
            
            if price <= 0:
                flash('Please enter a valid price')
                return redirect(url_for('dashboard'))
            
            # Validate ticker
            if not self.data_fetcher.is_valid_ticker(symbol):
                flash(f'Stock symbol "{symbol}" not found. Please check the symbol and try again.')
                return redirect(url_for('dashboard'))
            
            result = self.portfolio_holdings.execute_paper_sell(current_user.id, symbol, shares)
            
            if result.get('success'):
                flash(f'Successfully sold {shares} shares of {symbol}')
            else:
                flash(f'Failed to sell shares: {result.get("message", "Unknown error")}')
            
            return redirect(url_for('dashboard'))
            
        except Exception as e:
            logger.error(f"Paper sell error: {str(e)}")
            flash('Error executing sell order')
            return redirect(url_for('dashboard'))
    
    def paper_transactions_route(self):
        """Handle paper transactions requests"""
        if not current_user.is_authenticated:
            return jsonify([])
        
        try:
            transactions = self.portfolio_holdings.get_paper_transactions(current_user.id)
            return jsonify(transactions)
        except Exception as e:
            logger.error(f"Paper transactions error: {str(e)}")
            return jsonify([])
    
    def reset_paper_portfolio_route(self):
        """Handle reset paper portfolio requests"""
        if not current_user.is_authenticated:
            return jsonify({'success': False})
        
        try:
            success = self.portfolio_holdings.reset_paper_holdings(current_user.id)
            return jsonify({'success': success})
        except Exception as e:
            logger.error(f"Reset paper portfolio error: {str(e)}")
            return jsonify({'success': False})
    
    def get_stock_price_route(self, symbol):
        """Handle get stock price requests"""
        try:
            price = self.data_fetcher.get_current_price(symbol.upper())
            return jsonify({'symbol': symbol.upper(), 'price': price})
        except Exception as e:
            logger.error(f"Get stock price error: {str(e)}")
            return jsonify({'symbol': symbol.upper(), 'price': None, 'error': str(e)})
    
    # Utility methods
    def _prepare_prediction_features(self, stock_data, market_context):
        """Prepare features for prediction model"""
        try:
            if stock_data.empty:
                return None
            
            latest = stock_data.iloc[-1]
            features = [
                latest['Open'],
                latest['High'],
                latest['Low'],
                latest['Volume']
            ]
            return features
        except Exception as e:
            logger.error(f"Error preparing prediction features: {str(e)}")
            return None
    
    def _calculate_backtest_metrics(self, stock_symbol, days_back):
        """Calculate backtest metrics for a stock"""
        try:
            # Get predictions history
            predictions = self.db_ops.get_predictions_history(stock_symbol, days_back)
            
            if not predictions:
                return {'error': 'No prediction history found'}
            
            # Calculate accuracy and other metrics
            accurate_predictions = 0
            total_predictions = len(predictions)
            
            for prediction in predictions:
                # Simple accuracy check - if direction was correct
                predicted_change = prediction.predicted_change_pct
                # You'd need to calculate actual change for the target date
                # For now, return basic metrics
                pass
            
            return {
                'total_predictions': total_predictions,
                'accuracy': 0,  # Placeholder
                'mae': 0,       # Placeholder
                'rmse': 0       # Placeholder
            }
            
        except Exception as e:
            logger.error(f"Error calculating backtest metrics: {str(e)}")
            return {'error': f'Backtest calculation failed: {str(e)}'}
    
    # Advanced Analytics Routes
    def compare_stocks_route(self):
        """Handle stock comparison requests"""
        try:
            symbol1 = request.form.get('symbol1', '').strip().upper()
            symbol2 = request.form.get('symbol2', '').strip().upper()
            timeframe = request.form.get('timeframe', '1y')
            
            if not symbol1 or not symbol2:
                return jsonify({'error': 'Both stock symbols are required'}), 400
            
            comparison_data = self.compare_manager.compare_stocks(symbol1, symbol2, timeframe)
            
            if comparison_data:
                return jsonify(comparison_data)
            else:
                return jsonify({'error': 'Unable to compare stocks. Please check symbols and try again.'}), 400
                
        except Exception as e:
            logger.error(f"Error in compare_stocks: {str(e)}")
            return jsonify({'error': f'Stock comparison failed: {str(e)}'}), 500
    
    def value_at_risk_route(self):
        """Handle Value at Risk analysis requests"""
        try:
            symbol = request.form.get('symbol', '').strip().upper()
            confidence_level = float(request.form.get('confidence_level', 0.95))
            time_horizon = int(request.form.get('time_horizon', 1))
            
            if not symbol:
                return jsonify({'error': 'Stock symbol is required'}), 400
            
            if not (0 < confidence_level < 1):
                return jsonify({'error': 'Confidence level must be between 0 and 1'}), 400
                
            var_result = self.risk_manager.calculate_individual_var(
                symbol, confidence_level, time_horizon
            )
            
            if var_result:
                return jsonify(var_result)
            else:
                return jsonify({'error': 'Unable to calculate VaR. Please check symbol and try again.'}), 400
                
        except ValueError as e:
            return jsonify({'error': 'Invalid input parameters'}), 400
        except Exception as e:
            logger.error(f"Error in value_at_risk: {str(e)}")
            return jsonify({'error': f'VaR analysis failed: {str(e)}'}), 500
    
    def time_series_forecasting_route(self):
        """Handle time series forecasting requests"""
        try:
            symbol = request.form.get('symbol', '').strip().upper()
            forecast_days = int(request.form.get('forecast_days', 30))
            model_type = request.form.get('model_type', 'arima')
            
            if not symbol:
                return jsonify({'error': 'Stock symbol is required'}), 400
            
            if forecast_days <= 0 or forecast_days > 365:
                return jsonify({'error': 'Forecast days must be between 1 and 365'}), 400
            
            forecast_result = self.forecast_manager.generate_price_forecast(
                symbol, forecast_days, model_type
            )
            
            if forecast_result:
                return jsonify(forecast_result)
            else:
                return jsonify({'error': 'Unable to generate forecast. Please check symbol and try again.'}), 400
                
        except ValueError as e:
            return jsonify({'error': 'Invalid input parameters'}), 400
        except Exception as e:
            logger.error(f"Error in time_series_forecasting: {str(e)}")
            return jsonify({'error': f'Time series forecasting failed: {str(e)}'}), 500
    
    def options_pricing_route(self):
        """Handle options pricing requests"""
        try:
            symbol = request.form.get('symbol', '').strip().upper()
            strike_price = float(request.form.get('strike_price', 0))
            time_to_expiry = float(request.form.get('time_to_expiry', 30)) / 365  # Convert days to years
            risk_free_rate = float(request.form.get('risk_free_rate', 0.02))
            volatility = float(request.form.get('volatility', 0.2))
            option_type = request.form.get('option_type', 'call').lower()
            
            if not symbol:
                return jsonify({'error': 'Stock symbol is required'}), 400
            
            # Get current stock price
            current_price = self.data_fetcher.get_current_price(symbol)
            if not current_price:
                return jsonify({'error': 'Unable to fetch current stock price'}), 400
            
            spot_price = float(current_price)
            
            # Calculate Black-Scholes price
            bs_result = self.options_manager.calculate_black_scholes(
                spot_price, strike_price, time_to_expiry, risk_free_rate, volatility, option_type
            )
            
            # Calculate Greeks
            greeks_result = self.options_manager.calculate_greeks(
                spot_price, strike_price, time_to_expiry, risk_free_rate, volatility, option_type
            )
            
            if bs_result and greeks_result:
                return jsonify({
                    'symbol': symbol,
                    'current_price': spot_price,
                    'strike_price': strike_price,
                    'time_to_expiry_days': int(time_to_expiry * 365),
                    'option_type': option_type,
                    'black_scholes': bs_result,
                    'greeks': greeks_result
                })
            else:
                return jsonify({'error': 'Unable to calculate options pricing'}), 400
                
        except ValueError as e:
            return jsonify({'error': 'Invalid input parameters'}), 400
        except Exception as e:
            logger.error(f"Error in options_pricing: {str(e)}")
            return jsonify({'error': f'Options pricing failed: {str(e)}'}), 500


# Global instance to be used across the application
api_routes = APIRoutes()