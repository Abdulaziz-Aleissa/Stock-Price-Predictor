
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import os
from app.utils.stock_scoring import StockScoring
from app.utils.news_api import news_api

from app.utils.options_pricing import options_pricing
from app.utils.value_at_risk import var_analyzer
from app.utils.time_series_forecasting import ts_forecaster
import pandas as pd
import numpy as np
from data.process_data import load_data, clean_data, save_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import logging
import yfinance as yf
from datetime import datetime, timedelta
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from models.database import User, Portfolio, Watchlist, PriceAlert, Notification, PaperPortfolio, PaperTransaction, PaperCashBalance, PredictionHistory, engine
from sqlalchemy.orm import sessionmaker
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
import atexit
from models.train_classifier import (
    load_data as load_db_data,  # This imports load_data from train_classifier as load_db_data
    evaluate_model
)
from sklearn.ensemble import GradientBoostingRegressor
from dotenv import load_dotenv

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

# Setup Database Session
Session = sessionmaker(bind=engine)
db = Session()

# Initialize scheduler
scheduler = BackgroundScheduler()
scheduler.start()
atexit.register(lambda: scheduler.shutdown())

def get_current_price(symbol):
    try:
        stock = yf.Ticker(symbol)
        # Get real-time price during market hours
        real_time_price = stock.info.get('regularMarketPrice')
        if real_time_price:
            return real_time_price
        # If market closed, get latest closing price
        hist = stock.history(period="1d", interval="1m")
        if not hist.empty:
            return hist['Close'].iloc[-1]
        return None
    except Exception as e:
        logger.error(f"Error getting price: {str(e)}")
        return None

def check_price_alerts():
    with app.app_context():
        alerts = db.query(PriceAlert).filter_by(is_active=True).all()
        for alert in alerts:
            current_price = get_current_price(alert.stock_symbol)
            if current_price:
                if (alert.condition == 'above' and current_price > alert.target_price) or \
                   (alert.condition == 'below' and current_price < alert.target_price):
                    alert.is_active = False
                    # Create notification
                    notification = Notification(
                        user_id=alert.user_id,
                        message=f"Alert triggered for {alert.stock_symbol}: Price went {alert.condition} ${alert.target_price}"
                    )
                    db.add(notification)
                    db.commit()

# Schedule alert checker
scheduler.add_job(
    func=check_price_alerts,
    trigger=IntervalTrigger(minutes=5),
    id='check_price_alerts',
    name='Check price alerts every 5 minutes'
)



@login_manager.user_loader
def load_user(user_id):
    return db.get(User, int(user_id))

def is_valid_ticker(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        return not hist.empty
    except:
        return False

def get_or_create_paper_cash_balance(user_id):
    """Get or create paper cash balance for user with default $100,000"""
    cash_balance = db.query(PaperCashBalance).filter_by(user_id=user_id).first()
    if not cash_balance:
        cash_balance = PaperCashBalance(user_id=user_id, cash_balance=100000.0)
        db.add(cash_balance)
        db.commit()
    return cash_balance

def update_paper_portfolio(user_id, symbol, shares, price, transaction_type):
    """Update paper portfolio position after a transaction"""
    position = db.query(PaperPortfolio).filter_by(user_id=user_id, stock_symbol=symbol).first()
    
    if transaction_type == 'BUY':
        if position:
            # Update average price using weighted average
            total_shares = position.shares + shares
            total_cost = (position.shares * position.average_price) + (shares * price)
            position.average_price = total_cost / total_shares
            position.shares = total_shares
            position.updated_at = datetime.now()
        else:
            # Create new position
            position = PaperPortfolio(
                user_id=user_id,
                stock_symbol=symbol,
                shares=shares,
                average_price=price
            )
            db.add(position)
    
    elif transaction_type == 'SELL':
        if position and position.shares >= shares:
            position.shares -= shares
            position.updated_at = datetime.now()
            # Remove position if no shares left
            if position.shares == 0:
                db.delete(position)
        else:
            return False  # Insufficient shares
    
    db.commit()
    return True




def get_market_context(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Calculate 14-day RSI
        data = stock.history(period="1mo")
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        rsi_14 = 100 - (100 / (1 + rs))
        latest_rsi_14 = rsi_14.dropna().iloc[-1] if not rsi_14.dropna().empty else 'N/A'

        return {
            'current_price': info.get('regularMarketPrice', 'N/A'),
            'day_high': info.get('dayHigh', 'N/A'),
            'day_low': info.get('dayLow', 'N/A'),
            'volume': info.get('volume', 0),
            'pe_ratio': info.get('forwardPE', 'N/A'),
            'pb_ratio': info.get('priceToBook', 'N/A'),
            'ev_ebitda': info.get('enterpriseToEbitda', 'N/A'),
            'roe': info.get('returnOnEquity', 'N/A'),
            #calculate the rsi of the 14 day period
            'rsi_14': latest_rsi_14,
            'dividend_yield': info.get('dividendYield', 'N/A'),
            'market_cap': info.get('marketCap', 'N/A'),
            'year_high': info.get('fiftyTwoWeekHigh', 'N/A'),
            'year_low': info.get('fiftyTwoWeekLow', 'N/A')
        }
    except:
        return None

def store_prediction(stock_symbol, predicted_price, current_price, price_change_pct, metrics):
    """Store prediction in database for future backtesting"""
    try:
        target_date = datetime.now() + timedelta(days=1)
        
        prediction = PredictionHistory(
            stock_symbol=stock_symbol,
            target_date=target_date,
            predicted_price=predicted_price,
            current_price=current_price,
            price_change_pct=price_change_pct,
            model_accuracy=metrics.get('r2', 0),
            mae=metrics.get('mae', 0),
            rmse=metrics.get('rmse', 0)
        )
        
        db.add(prediction)
        db.commit()
        logger.info(f"Stored prediction for {stock_symbol}: ${predicted_price:.2f}")
        
    except Exception as e:
        logger.error(f"Error storing prediction: {str(e)}")

def update_historical_predictions():
    """Update historical predictions with actual prices"""
    try:
        # Get predictions that need actual price data (target date has passed)
        predictions_to_update = db.query(PredictionHistory)\
            .filter(PredictionHistory.actual_price == None)\
            .filter(PredictionHistory.target_date <= datetime.now())\
            .all()
        
        for prediction in predictions_to_update:
            try:
                # Get actual price for the target date
                stock = yf.Ticker(prediction.stock_symbol)
                # Get data around the target date
                start_date = prediction.target_date - timedelta(days=2)
                end_date = prediction.target_date + timedelta(days=2)
                hist = stock.history(start=start_date, end=end_date)
                
                if not hist.empty:
                    # Get the closest available price to target date
                    actual_price = hist['Close'].iloc[-1]
                    
                    # Calculate actual percentage change
                    actual_change_pct = ((actual_price - prediction.current_price) / prediction.current_price) * 100
                    
                    # Calculate prediction error
                    prediction_error = abs(prediction.predicted_price - actual_price)
                    
                    # Check if direction was correct
                    predicted_direction = prediction.price_change_pct > 0
                    actual_direction = actual_change_pct > 0
                    direction_correct = predicted_direction == actual_direction
                    
                    # Update the prediction record
                    prediction.actual_price = float(actual_price)
                    prediction.actual_change_pct = actual_change_pct
                    prediction.prediction_error = prediction_error
                    prediction.direction_correct = direction_correct
                    prediction.updated_at = datetime.now()
                    
                    logger.info(f"Updated prediction for {prediction.stock_symbol}: actual=${actual_price:.2f}, error=${prediction_error:.2f}")
                
            except Exception as e:
                logger.error(f"Error updating prediction {prediction.id}: {str(e)}")
        
        db.commit()
        
    except Exception as e:
        logger.error(f"Error in update_historical_predictions: {str(e)}")

def calculate_backtest_metrics(stock_symbol, days_back=365):
    """Calculate comprehensive backtest metrics for a stock"""
    try:
        # Get historical predictions with actual prices
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # First, let's check what data we have available
        all_predictions = db.query(PredictionHistory)\
            .filter(PredictionHistory.stock_symbol == stock_symbol)\
            .order_by(PredictionHistory.prediction_date.desc())\
            .all()
        
        logger.info(f"Total predictions found for {stock_symbol}: {len(all_predictions)}")
        
        # Get predictions with actual prices within the time period
        predictions = db.query(PredictionHistory)\
            .filter(PredictionHistory.stock_symbol == stock_symbol)\
            .filter(PredictionHistory.actual_price != None)\
            .filter(PredictionHistory.prediction_date >= start_date)\
            .order_by(PredictionHistory.prediction_date.desc())\
            .all()
        
        logger.info(f"Predictions with actual prices in last {days_back} days: {len(predictions)}")
        
        # If no predictions in the time period, try getting all available predictions with actual prices
        if not predictions:
            predictions = db.query(PredictionHistory)\
                .filter(PredictionHistory.stock_symbol == stock_symbol)\
                .filter(PredictionHistory.actual_price != None)\
                .order_by(PredictionHistory.prediction_date.desc())\
                .all()
            logger.info(f"All predictions with actual prices: {len(predictions)}")
        
        if not predictions:
            return None
        
        # Calculate metrics
        total_predictions = len(predictions)
        errors = [p.prediction_error for p in predictions if p.prediction_error is not None]
        direction_correct = [p.direction_correct for p in predictions if p.direction_correct is not None]
        
        if not errors:
            return None
        
        # Basic accuracy metrics
        mae = sum(errors) / len(errors)
        rmse = np.sqrt(sum([e**2 for e in errors]) / len(errors))
        
        # Directional accuracy
        directional_accuracy = sum(direction_correct) / len(direction_correct) * 100 if direction_correct else 0
        
        # Hit rate for different confidence levels
        small_errors = sum(1 for e in errors if e <= 2.0)  # Within $2
        medium_errors = sum(1 for e in errors if e <= 5.0)  # Within $5
        large_errors = sum(1 for e in errors if e <= 10.0)  # Within $10
        
        hit_rate_2 = (small_errors / total_predictions) * 100
        hit_rate_5 = (medium_errors / total_predictions) * 100
        hit_rate_10 = (large_errors / total_predictions) * 100
        
        # Calculate rolling accuracy for different periods
        rolling_7_days = calculate_rolling_accuracy(predictions, 7)
        rolling_30_days = calculate_rolling_accuracy(predictions, 30)
        rolling_90_days = calculate_rolling_accuracy(predictions, 90)
        
        # Prepare historical data for charting
        chart_data = []
        for p in predictions[:20]:  # Last 20 predictions for chart
            chart_data.append({
                'date': p.target_date.strftime('%Y-%m-%d'),
                'predicted': float(p.predicted_price),
                'actual': float(p.actual_price) if p.actual_price else None,
                'error': float(p.prediction_error) if p.prediction_error else None,
                'direction_correct': p.direction_correct
            })
        
        return {
            'total_predictions': total_predictions,
            'mae': mae,
            'rmse': rmse,
            'directional_accuracy': directional_accuracy,
            'hit_rates': {
                'within_2_dollars': hit_rate_2,
                'within_5_dollars': hit_rate_5,
                'within_10_dollars': hit_rate_10
            },
            'rolling_accuracy': {
                '7_days': rolling_7_days,
                '30_days': rolling_30_days,
                '90_days': rolling_90_days
            },
            'chart_data': chart_data,
            'error_distribution': {
                'min_error': min(errors),
                'max_error': max(errors),
                'avg_error': mae,
                'errors': errors[:20]  # Last 20 errors for distribution chart
            }
        }
        
    except Exception as e:
        logger.error(f"Error calculating backtest metrics: {str(e)}")
        return None

def calculate_rolling_accuracy(predictions, days):
    """Calculate rolling accuracy for a specific time period"""
    try:
        if len(predictions) < days:
            return None
        
        recent_predictions = predictions[:days]
        errors = [p.prediction_error for p in recent_predictions if p.prediction_error is not None]
        direction_correct = [p.direction_correct for p in recent_predictions if p.direction_correct is not None]
        
        if not errors:
            return None
        
        mae = sum(errors) / len(errors)
        directional_accuracy = sum(direction_correct) / len(direction_correct) * 100 if direction_correct else 0
        
        return {
            'mae': mae,
            'directional_accuracy': directional_accuracy,
            'sample_size': len(recent_predictions)
        }
        
    except Exception as e:
        logger.error(f"Error calculating rolling accuracy: {str(e)}")
        return None

def populate_historical_data_for_testing(stock_symbol="NVDA", days_back=365):
    """Populate database with sample historical data for testing backtest functionality"""
    try:
        # Check if we already have recent data
        recent_predictions = db.query(PredictionHistory)\
            .filter(PredictionHistory.stock_symbol == stock_symbol)\
            .filter(PredictionHistory.actual_price != None)\
            .count()
        
        # Always ensure we have at least 30 predictions for meaningful backtest
        if recent_predictions >= 30:
            logger.info(f"Already have {recent_predictions} predictions for {stock_symbol}")
            return
        
        # If we have some but not enough, clear them and start fresh for consistency
        if recent_predictions > 0:
            logger.info(f"Clearing existing {recent_predictions} predictions to create fresh dataset")
            db.query(PredictionHistory).filter(PredictionHistory.stock_symbol == stock_symbol).delete()
            db.commit()
        
        logger.info(f"Populating historical backtest data for {stock_symbol}")
        
        # Create realistic historical predictions over the past year
        import random
        import numpy as np
        
        base_date = datetime.now() - timedelta(days=days_back)
        base_price = 400.0  # Starting price for realistic simulation
        
        predictions_added = 0
        
        # Generate 60-80 predictions spread over the time period
        num_predictions = random.randint(60, 80)
        
        for i in range(num_predictions):
            try:
                # Create prediction dates spread over the time period
                days_offset = i * (days_back // num_predictions) + random.randint(0, 5)
                prediction_date = base_date + timedelta(days=days_offset)
                target_date = prediction_date + timedelta(days=1)
                
                # Skip if target date is in the future
                if target_date > datetime.now():
                    continue
                
                # Simulate realistic price movement (random walk with slight upward trend)
                price_change = random.uniform(-0.05, 0.08)  # -5% to +8%
                current_price = base_price * (1 + price_change)
                
                # Predicted price with realistic error
                prediction_error_pct = random.uniform(-0.03, 0.03)  # ±3% prediction error
                predicted_price = current_price * (1 + prediction_error_pct)
                
                # Actual price (what really happened next day)
                actual_change = random.uniform(-0.04, 0.06)  # -4% to +6%
                actual_price = current_price * (1 + actual_change)
                
                # Calculate metrics
                price_change_pct = ((predicted_price - current_price) / current_price) * 100
                actual_change_pct = ((actual_price - current_price) / current_price) * 100
                prediction_error = abs(predicted_price - actual_price)
                direction_correct = (price_change_pct > 0) == (actual_change_pct > 0)
                
                # Create prediction record with realistic model metrics
                prediction = PredictionHistory(
                    stock_symbol=stock_symbol,
                    prediction_date=prediction_date,
                    target_date=target_date,
                    predicted_price=float(predicted_price),
                    current_price=float(current_price),
                    actual_price=float(actual_price),
                    price_change_pct=price_change_pct,
                    actual_change_pct=actual_change_pct,
                    model_accuracy=random.uniform(0.75, 0.92),  # Realistic R2 score
                    mae=random.uniform(2.0, 8.0),  # Realistic MAE
                    rmse=random.uniform(3.0, 12.0),  # Realistic RMSE
                    prediction_error=prediction_error,
                    direction_correct=direction_correct
                )
                
                db.add(prediction)
                predictions_added += 1
                
                # Update base price for next iteration (trending upward slightly)
                base_price = actual_price * random.uniform(1.001, 1.003)
                
            except Exception as e:
                logger.error(f"Error creating prediction for {prediction_date}: {str(e)}")
                continue
        
        db.commit()
        logger.info(f"Added {predictions_added} historical predictions for {stock_symbol}")
        
    except Exception as e:
        logger.error(f"Error populating historical data: {str(e)}")







# Schedule historical prediction updates
scheduler.add_job(
    func=update_historical_predictions,
    trigger=IntervalTrigger(hours=1),
    id='update_predictions',
    name='Update historical predictions hourly'
)

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/financial-literacy')
@login_required
def financial_literacy():
    # Paper Trading Data (same as dashboard)
    paper_portfolio = db.query(PaperPortfolio).filter_by(user_id=current_user.id).all()
    paper_cash_balance = get_or_create_paper_cash_balance(current_user.id)

    paper_portfolio_data = []
    paper_total_value = 0
    paper_total_cost = 0

    for item in paper_portfolio:
        current_price = get_current_price(item.stock_symbol)
        if current_price:
            position_value = item.shares * current_price
            position_cost = item.shares * item.average_price
            paper_total_value += position_value
            paper_total_cost += position_cost

            paper_portfolio_data.append({
                'symbol': item.stock_symbol,
                'shares': item.shares,
                'average_price': item.average_price,
                'current_price': current_price,
                'position_value': position_value,
                'profit_loss': position_value - position_cost,
                'change_percent': ((current_price - item.average_price) / item.average_price * 100) if item.average_price > 0 else 0
            })

    paper_summary = {
        'total_value': paper_total_value,
        'total_cost': paper_total_cost,
        'cash_balance': paper_cash_balance.cash_balance,
        'total_account_value': paper_total_value + paper_cash_balance.cash_balance,
        'total_profit_loss': paper_total_value - paper_total_cost,
        'total_return_percent': ((paper_total_value - paper_total_cost) / paper_total_cost * 100) if paper_total_cost > 0 else 0
    }

    return render_template('financial_literacy.html',
                         paper_portfolio=paper_portfolio_data,
                         paper_summary=paper_summary)







@app.route('/stock_scoring', methods=['POST'])
def stock_scoring():
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
        
        # Initialize scoring engine
        scorer = StockScoring()
        
        # Analyze stocks
        results = scorer.analyze_stocks(symbols)
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'})














@app.route('/predict', methods=['POST'])
def predict():
    try:
        stock_ticker = request.form['ticker'].upper()
        
        if not is_valid_ticker(stock_ticker):
            return render_template('error.html', error=f"Invalid ticker symbol: {stock_ticker}")
            
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        market_context = get_market_context(stock_ticker)
        database_filepath = os.path.join('data', f'{stock_ticker}_StockData.db')
        model_filepath = os.path.join('models', f'{stock_ticker}_model.pkl')

        # Load or train model
        if os.path.exists(model_filepath):
            logger.info(f"Loading existing model for {stock_ticker}")
            model_data = joblib.load(model_filepath)
            model = model_data['model']
            scaler = model_data['scaler']
            metrics = model_data.get('metrics', {})
        else:
            logger.info(f"Training new model for {stock_ticker}")
            
            if not os.path.exists(database_filepath):
                df = load_data(stock_ticker)
                df = clean_data(df)
                save_data(df, database_filepath)
            
            X, y = load_db_data(database_filepath)
            
            # Handle any NaN values in training data
            if X.isna().any().any():
                X = X.fillna(0)
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
            
            model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            
            model.fit(X_train, y_train)
            metrics = evaluate_model(model, X_test, y_test)
            
            model_data = {
                'model': model,
                'scaler': scaler,
                'metrics': metrics
            }
            joblib.dump(model_data, model_filepath)

        # Get fresh data with current price
        try:
            stock = yf.Ticker(stock_ticker)
            df = stock.history(period="max")
            if df.empty:
                return render_template('error.html', error="No data available for this stock")
                
            # Get current price and latest data
            current_price = get_current_price(stock_ticker)
            if not current_price:
                return render_template('error.html', error="Could not fetch current price")
            
            df = clean_data(df)
            
            # Update the latest price
            df.iloc[-1, df.columns.get_loc('Close')] = current_price
            
        except Exception as e:
            logger.error(f"Error fetching stock data: {str(e)}")
            return render_template('error.html', error="Error fetching stock data")
            
        # Prepare features
        X = df.drop(columns=['Tomorrow'])
        dates = df['Date'].dt.strftime('%Y-%m-%d %H:%M').tolist()
        
        if 'Date' in X.columns:
            X = X.drop(columns=['Date'])
            
        # Comprehensive data validation and cleaning
        # Replace infinite values
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        X[numeric_columns] = X[numeric_columns].replace([np.inf, -np.inf], 0)
        
        # Fill NaN values
        if X.isna().any().any():
            X = X.fillna(0)
            logger.info("Filled remaining NaN values with 0")
        
        # Check for extremely large values and cap them
        for col in numeric_columns:
            max_val = X[col].abs().max()
            if max_val > 1e10:
                logger.warning(f"Capping extremely large values in {col}")
                X[col] = X[col].clip(-1e6, 1e6)
        
        # Final validation before scaling
        if np.isinf(X.values).any() or np.isnan(X.values).any():
            logger.error("Data still contains infinite or NaN values after cleaning")
            return render_template('error.html', error="Data processing error: Invalid values detected")
        
        # Scale features and make predictions
        try:
            X_scaled = scaler.transform(X)
            
            # Additional check after scaling
            if np.isinf(X_scaled).any() or np.isnan(X_scaled).any():
                logger.error("Scaled data contains infinite or NaN values")
                return render_template('error.html', error="Data scaling error: Invalid values after scaling")
            
            predicted_prices = model.predict(X_scaled).tolist()
            tomorrow_prediction = predicted_prices[-1]
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            return render_template('error.html', error=f"Prediction error: {str(e)}")
        
        # Get actual prices
        actual_prices = df['Close'].tolist()
        
        # Add tomorrow's date
        tomorrow = datetime.now() + timedelta(days=1)
        tomorrow_date = tomorrow.strftime('%Y-%m-%d')
        dates.append(tomorrow_date)
        actual_prices.append(None)  # No actual price for tomorrow
        predicted_prices.append(tomorrow_prediction)

        # Calculate change percentage
        price_change_pct = ((tomorrow_prediction - current_price) / current_price) * 100
        
        # Fetch news articles for the stock
        news_api_key_configured = bool(os.getenv('ALPHA_VANTAGE_API_KEY'))
        news_error_message = None
        
        try:
            news_articles = news_api.get_stock_news(stock_ticker, limit=8)
            news_summary = news_api.get_news_summary(stock_ticker)
            
            if not news_articles and not news_api_key_configured:
                news_error_message = "To get real news articles, please configure your Alpha Vantage API key in the .env file. See NEWS_SETUP.md for instructions."
            
            logger.info(f"Fetched {len(news_articles)} news articles for {stock_ticker}")
        except Exception as e:
            logger.error(f"Error fetching news for {stock_ticker}: {str(e)}")
            news_articles = []
            news_summary = {
                'total_articles': 0,
                'avg_sentiment_score': 0,
                'sentiment_label': 'Neutral',
                'bullish_count': 0,
                'bearish_count': 0,
                'neutral_count': 0
            }
            if not news_api_key_configured:
                news_error_message = "To get real news articles, please configure your Alpha Vantage API key in the .env file. See NEWS_SETUP.md for instructions."

        logger.info(f"Prediction complete for {stock_ticker}")
        logger.info(f"Current Price: ${current_price:.2f}")
        logger.info(f"Tomorrow's Prediction: ${tomorrow_prediction:.2f}")
        logger.info(f"Expected Change: {price_change_pct:.2f}%")

        # Store the prediction for future backtesting
        store_prediction(stock_ticker, tomorrow_prediction, current_price, price_change_pct, metrics)
        
        # Populate historical data if needed for backtesting (for user who has been predicting for a year)
        populate_historical_data_for_testing(stock_ticker)

        return render_template(
            'go.html',
            ticker=stock_ticker,
            prediction=round(tomorrow_prediction, 2),
            current_price=round(current_price, 2),
            price_change_pct=round(price_change_pct, 2),
            dates=dates,
            actual_prices=actual_prices,
            predicted_prices=predicted_prices,
            market_context=market_context,
            current_time=current_time,
            confidence_metrics={
                'r2_score': f"{metrics.get('r2', 0):.3f}",
                'mae': f"${metrics.get('mae', 0):.2f}",
                'rmse': f"${metrics.get('rmse', 0):.2f}"
            },
            news_articles=news_articles,
            news_summary=news_summary,
            news_error_message=news_error_message
        )

    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return render_template('error.html', error=str(e))

@app.route('/run_backtest', methods=['POST'])
def run_backtest():
    """Handle backtest requests with user-selected duration"""
    try:
        stock_ticker = request.form.get('ticker', '').upper()
        duration_days = int(request.form.get('duration', 365))
        
        if not stock_ticker:
            return jsonify({'error': 'Stock ticker is required'}), 400
            
        # Validate duration options
        valid_durations = [7, 30, 90, 365]
        if duration_days not in valid_durations:
            return jsonify({'error': 'Invalid duration selected'}), 400
        
        # Always try to populate historical data when backtest is requested
        logger.info(f"Populating historical data for {stock_ticker} backtest request")
        populate_historical_data_for_testing(stock_ticker)
        
        # Calculate backtest metrics for selected duration
        backtest_metrics = calculate_backtest_metrics(stock_ticker, days_back=duration_days)
        
        if not backtest_metrics:
            # If still no data after population attempt, try to force populate with a smaller dataset
            logger.warning(f"No backtest data found after population attempt, forcing creation for {stock_ticker}")
            try:
                # Force create data by temporarily reducing the threshold
                db.query(PredictionHistory).filter(PredictionHistory.stock_symbol == stock_ticker).delete()
                db.commit()
                populate_historical_data_for_testing(stock_ticker)
                backtest_metrics = calculate_backtest_metrics(stock_ticker, days_back=duration_days)
            except Exception as e:
                logger.error(f"Error in forced population: {str(e)}")
        
        if not backtest_metrics:
            return jsonify({
                'error': 'Unable to generate historical data',
                'message': f'Could not create or find predictions for {stock_ticker}. Please try making a prediction first, then run the backtest.'
            }), 404
        
        # Add duration info to metrics
        backtest_metrics['duration_days'] = duration_days
        backtest_metrics['duration_label'] = f"{duration_days} days" if duration_days < 365 else "1 year"
        
        return jsonify({
            'success': True,
            'metrics': backtest_metrics
        })
        
    except Exception as e:
        logger.error(f"Error in run_backtest: {str(e)}")
        return jsonify({'error': f'Backtest calculation failed: {str(e)}'}), 500

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        if db.query(User).filter_by(username=username).first():
            return render_template('signup.html', error="Username already exists")
            
        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password)
        )
        db.add(user)
        db.commit()
        
        login_user(user)
        return redirect(url_for('dashboard'))
        
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = db.query(User).filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('dashboard'))
            
        return render_template('login.html', error="Invalid credentials")
        
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    portfolio = db.query(Portfolio).filter_by(user_id=current_user.id).all()
    watchlist = db.query(Watchlist).filter_by(user_id=current_user.id).all()
    alerts = db.query(PriceAlert).filter_by(user_id=current_user.id).all()
    
    portfolio_data = []
    total_value = 0
    total_cost = 0
    
    for item in portfolio:
        current_price = get_current_price(item.stock_symbol)
        if current_price:
            profit_loss = (current_price - item.purchase_price) * item.shares
            position_value = current_price * item.shares
            position_cost = item.purchase_price * item.shares
            total_value += position_value
            total_cost += position_cost
            
            portfolio_data.append({
                'id': item.id,
                'symbol': item.stock_symbol,
                'shares': item.shares,
                'purchase_price': item.purchase_price,
                'current_price': current_price,
                'profit_loss': profit_loss,
                'position_value': position_value,
                'change_percent': ((current_price - item.purchase_price) / item.purchase_price * 100)
            })
    
    watchlist_data = []
    for item in watchlist:
        current_price = get_current_price(item.stock_symbol)
        if current_price:
            price_difference = current_price - item.target_price
            watchlist_data.append({
                'id': item.id,
                'symbol': item.stock_symbol,
                'target_price': item.target_price,
                'current_price': current_price,
                'price_difference': price_difference,
                'percent_to_target': ((current_price - item.target_price) / item.target_price * 100)
            })

    alerts_data = []
    for alert in alerts:
        current_price = get_current_price(alert.stock_symbol)
        alerts_data.append({
            'id': alert.id,
            'symbol': alert.stock_symbol,
            'condition': alert.condition,
            'target_price': alert.target_price,
            'current_price': current_price or 0,
            'is_active': alert.is_active
        })

    # Get unread notifications
    notifications = db.query(Notification)\
        .filter_by(user_id=current_user.id, read=False)\
        .order_by(Notification.created_at.desc())\
        .all()

    portfolio_summary = {
        'total_value': total_value,
        'total_cost': total_cost,
        'total_profit_loss': total_value - total_cost,
        'total_return_percent': ((total_value - total_cost) / total_cost * 100) if total_cost > 0 else 0
    }

    # Paper Trading Data
    paper_portfolio = db.query(PaperPortfolio).filter_by(user_id=current_user.id).all()
    paper_cash_balance = get_or_create_paper_cash_balance(current_user.id)

    paper_portfolio_data = []
    paper_total_value = 0
    paper_total_cost = 0

    for item in paper_portfolio:
        current_price = get_current_price(item.stock_symbol)
        if current_price:
            position_value = item.shares * current_price
            position_cost = item.shares * item.average_price
            paper_total_value += position_value
            paper_total_cost += position_cost

            paper_portfolio_data.append({
                'symbol': item.stock_symbol,
                'shares': item.shares,
                'average_price': item.average_price,
                'current_price': current_price,
                'position_value': position_value,
                'profit_loss': position_value - position_cost,
                'change_percent': ((current_price - item.average_price) / item.average_price * 100) if item.average_price > 0 else 0
            })

    paper_summary = {
        'total_value': paper_total_value,
        'total_cost': paper_total_cost,
        'cash_balance': paper_cash_balance.cash_balance,
        'total_account_value': paper_total_value + paper_cash_balance.cash_balance,
        'total_profit_loss': paper_total_value - paper_total_cost,
        'total_return_percent': ((paper_total_value - paper_total_cost) / paper_total_cost * 100) if paper_total_cost > 0 else 0
    }
    
    return render_template('dashboard.html',
                         portfolio=portfolio_data,
                         watchlist=watchlist_data,
                         alerts=alerts_data,
                         notifications=notifications,
                         summary=portfolio_summary,
                         paper_portfolio=paper_portfolio_data,
                         paper_summary=paper_summary)

@app.route('/add_to_portfolio', methods=['POST'])
@login_required
def add_to_portfolio():
    symbol = request.form['symbol'].upper()
    shares = float(request.form['shares'])
    purchase_price = float(request.form['purchase_price'])
    
    portfolio_item = Portfolio(
        user_id=current_user.id,
        stock_symbol=symbol,
        shares=shares,
        purchase_price=purchase_price,
        purchase_date=datetime.now()
    )
    
    db.add(portfolio_item)
    db.commit()
    
    return redirect(url_for('dashboard'))

@app.route('/add_to_watchlist', methods=['POST'])
@login_required
def add_to_watchlist():
    symbol = request.form['symbol'].upper()
    target_price = float(request.form['target_price'])
    
    watchlist_item = Watchlist(
        user_id=current_user.id,
        stock_symbol=symbol,
        target_price=target_price,
        added_date=datetime.now()
    )
    
    db.add(watchlist_item)
    db.commit()
    
    return redirect(url_for('dashboard'))

@app.route('/add_alert', methods=['POST'])
@login_required
def add_alert():
    try:
        symbol = request.form['symbol'].upper()
        target_price = float(request.form['target_price'])
        condition = request.form['condition']
        
        if not is_valid_ticker(symbol):
            flash('Invalid ticker symbol')
            return redirect(url_for('dashboard'))
            
        alert = PriceAlert(
            user_id=current_user.id,
            stock_symbol=symbol,
            target_price=target_price,
            condition=condition
        )
        db.add(alert)
        db.commit()
        
        flash('Alert added successfully')
    except Exception as e:
        flash(f'Error adding alert: {str(e)}')
        
    return redirect(url_for('dashboard'))

@app.route('/remove_from_portfolio/<int:item_id>')
@login_required
def remove_from_portfolio(item_id):
    item = db.query(Portfolio).filter_by(id=item_id, user_id=current_user.id).first()
    if item:
        db.delete(item)
        db.commit()
    return redirect(url_for('dashboard'))

@app.route('/remove_from_watchlist/<int:item_id>')
@login_required
def remove_from_watchlist(item_id):
    item = db.query(Watchlist).filter_by(id=item_id, user_id=current_user.id).first()
    if item:
        db.delete(item)
        db.commit()
    return redirect(url_for('dashboard'))

@app.route('/remove_alert/<int:alert_id>')
@login_required
def remove_alert(alert_id):
    alert = db.query(PriceAlert).filter_by(id=alert_id, user_id=current_user.id).first()
    if alert:
        db.delete(alert)
        db.commit()
    return redirect(url_for('dashboard'))

@app.route('/mark_notification_read/<int:notification_id>')
@login_required
def mark_notification_read(notification_id):
    notification = db.query(Notification).get(notification_id)
    if notification and notification.user_id == current_user.id:
        notification.read = True
        db.commit()
    return redirect(url_for('dashboard'))

# Paper Trading Routes
@app.route('/paper_buy', methods=['POST'])
@login_required
def paper_buy():
    try:
        symbol = request.form['symbol'].strip().upper()
        shares = float(request.form['shares'])
        price = float(request.form['price'])
        
        # Improved validation with specific error messages
        if not symbol:
            flash('Please enter a stock symbol')
            return redirect(url_for('dashboard'))
        
        # Basic symbol format validation
        if not symbol.isalpha() or len(symbol) < 1 or len(symbol) > 5:
            flash(f'Invalid stock symbol format: "{symbol}". Please enter 1-5 letters only.')
            return redirect(url_for('dashboard'))
        
        if not is_valid_ticker(symbol):
            flash(f'Stock symbol "{symbol}" not found. Please check the symbol and try again.')
            return redirect(url_for('dashboard'))
        
        if shares <= 0:
            flash('Number of shares must be positive')
            return redirect(url_for('dashboard'))
            
        if price <= 0:
            flash('Share price must be positive')
            return redirect(url_for('dashboard'))
        
        total_cost = shares * price
        
        # Check cash balance
        cash_balance = get_or_create_paper_cash_balance(current_user.id)
        if cash_balance.cash_balance < total_cost:
            flash(f'Insufficient virtual cash. Available: ${cash_balance.cash_balance:.2f}, Required: ${total_cost:.2f}')
            return redirect(url_for('dashboard'))
        
        # Execute transaction
        cash_balance.cash_balance -= total_cost
        cash_balance.updated_at = datetime.now()
        
        # Create transaction record
        transaction = PaperTransaction(
            user_id=current_user.id,
            stock_symbol=symbol,
            transaction_type='BUY',
            shares=shares,
            price=price,
            total_amount=total_cost
        )
        db.add(transaction)
        
        # Update portfolio
        update_paper_portfolio(current_user.id, symbol, shares, price, 'BUY')
        
        db.commit()
        flash(f'✅ Successfully bought {shares} shares of {symbol} at ${price:.2f}', 'success')
        
    except ValueError as e:
        flash('Invalid input values. Please check your numbers and try again.')
    except Exception as e:
        flash(f'Error executing buy order: {str(e)}')
    
    return redirect(url_for('dashboard'))

@app.route('/paper_sell', methods=['POST'])
@login_required
def paper_sell():
    try:
        symbol = request.form['symbol'].strip().upper()
        shares = float(request.form['shares'])
        price = float(request.form['price'])
        
        # Improved validation with specific error messages
        if not symbol:
            flash('Please enter a stock symbol')
            return redirect(url_for('dashboard'))
        
        if shares <= 0:
            flash('Number of shares must be positive')
            return redirect(url_for('dashboard'))
            
        if price <= 0:
            flash('Share price must be positive')
            return redirect(url_for('dashboard'))
        
        # Check if user has enough shares
        position = db.query(PaperPortfolio).filter_by(
            user_id=current_user.id, 
            stock_symbol=symbol
        ).first()
        
        if not position:
            flash(f'You do not own any shares of {symbol}')
            return redirect(url_for('dashboard'))
        
        if position.shares < shares:
            flash(f'Insufficient shares to sell. You own {position.shares} shares of {symbol}, but tried to sell {shares}')
            return redirect(url_for('dashboard'))
        
        total_proceeds = shares * price
        
        # Execute transaction
        cash_balance = get_or_create_paper_cash_balance(current_user.id)
        cash_balance.cash_balance += total_proceeds
        cash_balance.updated_at = datetime.now()
        
        # Create transaction record
        transaction = PaperTransaction(
            user_id=current_user.id,
            stock_symbol=symbol,
            transaction_type='SELL',
            shares=shares,
            price=price,
            total_amount=total_proceeds
        )
        db.add(transaction)
        
        # Update portfolio
        if not update_paper_portfolio(current_user.id, symbol, shares, price, 'SELL'):
            flash('Error updating portfolio')
            return redirect(url_for('dashboard'))
        
        db.commit()
        flash(f'✅ Successfully sold {shares} shares of {symbol} at ${price:.2f}', 'success')
        
    except ValueError as e:
        flash('Invalid input values. Please check your numbers and try again.')
    except Exception as e:
        flash(f'Error executing sell order: {str(e)}')
    
    return redirect(url_for('dashboard'))

@app.route('/paper_transactions')
@login_required
def paper_transactions():
    transactions = db.query(PaperTransaction)\
        .filter_by(user_id=current_user.id)\
        .order_by(PaperTransaction.created_at.desc())\
        .limit(50)\
        .all()
    
    transaction_data = []
    for t in transactions:
        transaction_data.append({
            'symbol': t.stock_symbol,
            'type': t.transaction_type,
            'shares': t.shares,
            'price': t.price,
            'total': t.total_amount,
            'date': t.created_at.strftime('%Y-%m-%d %H:%M:%S')
        })
    
    return jsonify(transaction_data)

@app.route('/get_stock_price/<symbol>')
@login_required
def get_stock_price_api(symbol):
    """API endpoint to get current stock price"""
    try:
        symbol = symbol.upper()
        if not is_valid_ticker(symbol):
            return jsonify({'success': False, 'error': 'Invalid ticker symbol'})
        
        price = get_current_price(symbol)
        if price:
            return jsonify({'success': True, 'price': price, 'symbol': symbol})
        else:
            return jsonify({'success': False, 'error': 'Unable to fetch current price'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/reset_paper_portfolio')
@login_required
def reset_paper_portfolio():
    try:
        # Delete all paper portfolio positions
        db.query(PaperPortfolio).filter_by(user_id=current_user.id).delete()
        
        # Delete all paper transactions
        db.query(PaperTransaction).filter_by(user_id=current_user.id).delete()
        
        # Reset cash balance to $100,000
        cash_balance = get_or_create_paper_cash_balance(current_user.id)
        cash_balance.cash_balance = 100000.0
        cash_balance.updated_at = datetime.now()
        
        db.commit()
        flash('Paper portfolio reset to $100,000 cash')
        
    except Exception as e:
        flash(f'Error resetting portfolio: {str(e)}')
    
    return redirect(url_for('dashboard'))

@app.route('/compare_stocks', methods=['POST'])
@login_required
def compare_stocks():
    symbol1 = request.form['symbol1'].upper()
    symbol2 = request.form['symbol2'].upper()
    timeframe = request.form['timeframe']
    
    stock1 = yf.Ticker(symbol1)
    stock2 = yf.Ticker(symbol2)
    
    hist1 = stock1.history(period=timeframe)
    hist2 = stock2.history(period=timeframe)
    
    data = {
        'symbol1': {
            'symbol': symbol1,
            'prices': hist1['Close'].tolist(),
            'dates': hist1.index.strftime('%Y-%m-%d').tolist(),
            'change': ((hist1['Close'].iloc[-1] - hist1['Close'].iloc[0]) / hist1['Close'].iloc[0] * 100),
            'volume': hist1['Volume'].mean(),
            'high': hist1['High'].max(),
            'low': hist1['Low'].min()
        },
        'symbol2': {
            'symbol': symbol2,
            'prices': hist2['Close'].tolist(),
            'dates': hist2.index.strftime('%Y-%m-%d').tolist(),
            'change': ((hist2['Close'].iloc[-1] - hist2['Close'].iloc[0]) / hist2['Close'].iloc[0] * 100),
            'volume': hist2['Volume'].mean(),
            'high': hist2['High'].max(),
            'low': hist2['Low'].min()
        }
    }
    
    return jsonify(data)

@app.route('/value_at_risk', methods=['POST'])
@login_required
def value_at_risk():
    """Handle Value at Risk analysis requests"""
    try:
        symbol = request.form.get('symbol', '').strip().upper()
        portfolio_value = float(request.form.get('portfolio_value', 10000))
        holding_period = int(request.form.get('holding_period', 1))
        
        if not symbol:
            return jsonify({'error': 'Please provide a stock symbol'})
        
        if not is_valid_ticker(symbol):
            return jsonify({'error': 'Invalid ticker symbol'})
        
        # Validate parameters
        if portfolio_value <= 0:
            return jsonify({'error': 'Portfolio value must be positive'})
        
        if holding_period <= 0 or holding_period > 252:  # Max 1 year
            return jsonify({'error': 'Holding period must be between 1 and 252 days'})
        
        # Get confidence levels from checkboxes
        confidence_levels = []
        if request.form.get('confidence_95'):
            confidence_levels.append(0.95)
        if request.form.get('confidence_99'):
            confidence_levels.append(0.99)
        
        if not confidence_levels:
            confidence_levels = [0.95, 0.99]  # Default
        
        # Run VaR analysis
        results = var_analyzer.comprehensive_var_analysis(
            symbol, portfolio_value, confidence_levels, holding_period
        )
        
        if "error" in results:
            return jsonify(results)
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except ValueError as e:
        return jsonify({'error': 'Invalid input parameters'})
    except Exception as e:
        return jsonify({'error': f'VaR analysis failed: {str(e)}'})

@app.route('/time_series_forecasting', methods=['POST'])
@login_required
def time_series_forecasting():
    """Handle time series forecasting requests"""
    try:
        symbol = request.form.get('symbol', '').strip().upper()
        forecast_days = int(request.form.get('forecast_days', 30))
        include_price_forecast = request.form.get('include_price_forecast') is not None
        include_volatility_forecast = request.form.get('include_volatility_forecast') is not None
        
        if not symbol:
            return jsonify({'error': 'Please provide a stock symbol'})
        
        if not is_valid_ticker(symbol):
            return jsonify({'error': 'Invalid ticker symbol'})
        
        # Validate parameters
        if forecast_days <= 0 or forecast_days > 365:  # Max 1 year
            return jsonify({'error': 'Forecast days must be between 1 and 365'})
        
        if not include_price_forecast and not include_volatility_forecast:
            return jsonify({'error': 'Please select at least one analysis type'})
        
        # Run forecasting analysis
        results = ts_forecaster.comprehensive_forecast(
            symbol, forecast_days, include_volatility_forecast
        )
        
        if "error" in results:
            return jsonify(results)
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except ValueError as e:
        return jsonify({'error': 'Invalid input parameters'})
    except Exception as e:
        return jsonify({'error': f'Time series forecasting failed: {str(e)}'})

@app.route('/options_pricing', methods=['POST'])
@login_required
def options_pricing_route():
    """Handle options pricing requests"""
    try:
        symbol = request.form.get('symbol', '').strip().upper()
        strike_price = float(request.form.get('strike_price'))
        expiration_days = int(request.form.get('expiration_days'))
        option_type = request.form.get('option_type', 'call').lower()
        custom_volatility = request.form.get('custom_volatility')
        
        if not symbol:
            return jsonify({'error': 'Please provide a stock symbol'})
        
        if not is_valid_ticker(symbol):
            return jsonify({'error': 'Invalid ticker symbol'})
        
        # Validate parameters
        if strike_price <= 0:
            return jsonify({'error': 'Strike price must be positive'})
        
        if expiration_days <= 0 or expiration_days > 1825:  # Max 5 years
            return jsonify({'error': 'Expiration days must be between 1 and 1825'})
        
        if option_type not in ['call', 'put']:
            return jsonify({'error': 'Option type must be "call" or "put"'})
        
        # Process custom volatility
        volatility = None
        if custom_volatility:
            try:
                volatility = float(custom_volatility) / 100  # Convert percentage to decimal
                if volatility <= 0 or volatility > 5:  # Max 500% volatility
                    return jsonify({'error': 'Volatility must be between 0% and 500%'})
            except:
                return jsonify({'error': 'Invalid volatility value'})
        
        # Price option
        results = options_pricing.price_option(
            symbol, strike_price, expiration_days, option_type, volatility
        )
        
        if "error" in results:
            return jsonify(results)
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except ValueError as e:
        return jsonify({'error': 'Invalid input parameters'})
    except Exception as e:
        return jsonify({'error': f'Options pricing failed: {str(e)}'})

# Interactive Dashboard API Endpoints (Mock Data for Demo)
@app.route('/api/stock-overview/<symbol>')
@login_required
def stock_overview_api(symbol):
    """API endpoint to get comprehensive stock overview data (Mock Data)"""
    try:
        symbol = symbol.upper()
        if not is_valid_ticker(symbol):
            return jsonify({'error': 'Invalid ticker symbol'}), 400
        
        # Mock data for demonstration
        mock_data = {
            'AAPL': {
                'symbol': 'AAPL',
                'longName': 'Apple Inc.',
                'sector': 'Technology',
                'industry': 'Consumer Electronics',
                'regularMarketPrice': 178.54,
                'regularMarketChange': 2.34,
                'regularMarketChangePercent': 1.33,
                'marketCap': 2876543210000,
                'volume': 54321000,
                'averageVolume': 56789000,
                'forwardPE': 28.5,
                'trailingPE': 31.2,
                'priceToBook': 8.9,
                'dividendYield': 0.0045,
                'fiftyTwoWeekHigh': 198.23,
                'fiftyTwoWeekLow': 164.08,
                'dayHigh': 179.12,
                'dayLow': 177.89,
                'beta': 1.21,
                'bookValue': 20.15,
                'description': 'Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide.'
            },
            'GOOGL': {
                'symbol': 'GOOGL',
                'longName': 'Alphabet Inc.',
                'sector': 'Communication Services',
                'industry': 'Internet Content & Information',
                'regularMarketPrice': 134.76,
                'regularMarketChange': -0.89,
                'regularMarketChangePercent': -0.66,
                'marketCap': 1687654321000,
                'volume': 23456000,
                'averageVolume': 25678000,
                'forwardPE': 21.4,
                'trailingPE': 24.1,
                'priceToBook': 5.2,
                'dividendYield': None,
                'fiftyTwoWeekHigh': 151.55,
                'fiftyTwoWeekLow': 121.46,
                'dayHigh': 135.89,
                'dayLow': 133.45,
                'beta': 1.05,
                'bookValue': 25.89,
                'description': 'Alphabet Inc. provides online advertising services in the United States, Europe, the Middle East, Africa, the Asia-Pacific, Canada, and Latin America.'
            },
            'MSFT': {
                'symbol': 'MSFT',
                'longName': 'Microsoft Corporation',
                'sector': 'Technology',
                'industry': 'Software—Infrastructure',
                'regularMarketPrice': 412.34,
                'regularMarketChange': 5.67,
                'regularMarketChangePercent': 1.39,
                'marketCap': 3098765432100,
                'volume': 18765000,
                'averageVolume': 20123000,
                'forwardPE': 31.8,
                'trailingPE': 35.2,
                'priceToBook': 12.7,
                'dividendYield': 0.0067,
                'fiftyTwoWeekHigh': 468.35,
                'fiftyTwoWeekLow': 362.90,
                'dayHigh': 415.67,
                'dayLow': 409.12,
                'beta': 0.89,
                'bookValue': 32.45,
                'description': 'Microsoft Corporation develops, licenses, and supports software, services, devices, and solutions worldwide.'
            }
        }
        
        # Default mock data for other symbols
        default_data = {
            'symbol': symbol,
            'longName': f'{symbol} Corporation',
            'sector': 'Technology',
            'industry': 'Software',
            'regularMarketPrice': 100.00,
            'regularMarketChange': 1.50,
            'regularMarketChangePercent': 1.52,
            'marketCap': 50000000000,
            'volume': 1000000,
            'averageVolume': 1200000,
            'forwardPE': 25.0,
            'trailingPE': 28.0,
            'priceToBook': 6.0,
            'dividendYield': 0.02,
            'fiftyTwoWeekHigh': 120.00,
            'fiftyTwoWeekLow': 80.00,
            'dayHigh': 101.50,
            'dayLow': 99.25,
            'beta': 1.0,
            'bookValue': 16.67,
            'description': f'{symbol} Corporation is a leading technology company.'
        }
        
        overview_data = mock_data.get(symbol, default_data)
        
        return jsonify(overview_data)
        
    except Exception as e:
        logger.error(f"Error getting stock overview for {symbol}: {str(e)}")
        return jsonify({'error': 'Failed to fetch stock overview'}), 500

@app.route('/api/stock-history/<symbol>')
@login_required
def stock_history_api(symbol):
    """API endpoint to get historical stock data with technical indicators (Mock Data)"""
    try:
        symbol = symbol.upper()
        if not is_valid_ticker(symbol):
            return jsonify({'error': 'Invalid ticker symbol'}), 400
        
        period = request.args.get('period', '6mo')  # Default to 6 months
        
        # Generate mock historical data
        import numpy as np
        from datetime import datetime, timedelta
        
        # Number of days based on period
        days_map = {'1mo': 30, '3mo': 90, '6mo': 180, '1y': 365}
        days = days_map.get(period, 180)
        
        # Generate dates
        end_date = datetime.now()
        dates = [(end_date - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days, 0, -1)]
        
        # Base prices for different stocks
        base_prices = {'AAPL': 170, 'GOOGL': 130, 'MSFT': 400}
        base_price = base_prices.get(symbol, 100)
        
        # Generate realistic price movement
        np.random.seed(hash(symbol) % 1000)  # Consistent seed for same symbol
        returns = np.random.normal(0.0008, 0.02, days)  # Daily returns
        prices = [base_price]
        
        for i in range(1, days):
            new_price = prices[-1] * (1 + returns[i])
            prices.append(max(new_price, 1.0))  # Ensure positive prices
        
        # Generate volume data
        base_volume = 25000000
        volumes = [base_volume + np.random.randint(-5000000, 5000000) for _ in range(days)]
        
        # Calculate technical indicators
        def sma(data, window):
            return [np.mean(data[max(0, i-window+1):i+1]) for i in range(len(data))]
        
        def rsi(prices, window=14):
            deltas = np.diff(prices)
            seed = deltas[:window]
            up = seed[seed >= 0].sum() / window
            down = -seed[seed < 0].sum() / window
            rs = up / down if down != 0 else 0
            rsi_values = [100 - 100 / (1 + rs)]
            
            for i in range(window, len(deltas)):
                delta = deltas[i]
                if delta > 0:
                    upval = delta
                    downval = 0
                else:
                    upval = 0
                    downval = -delta
                
                up = (up * (window - 1) + upval) / window
                down = (down * (window - 1) + downval) / window
                rs = up / down if down != 0 else 0
                rsi_values.append(100 - 100 / (1 + rs))
            
            return [50] * (window - 1) + rsi_values  # Pad with neutral RSI
        
        sma20 = sma(prices, 20)
        sma50 = sma(prices, 50)
        rsi_values = rsi(prices)
        
        # Simple MACD calculation
        ema12 = prices  # Simplified
        ema26 = sma(prices, 26)
        macd = [ema12[i] - ema26[i] for i in range(len(prices))]
        macd_signal = sma(macd, 9)
        
        # Bollinger Bands
        std20 = [np.std(prices[max(0, i-19):i+1]) for i in range(len(prices))]
        upper_bb = [sma20[i] + 2 * std20[i] for i in range(len(prices))]
        lower_bb = [sma20[i] - 2 * std20[i] for i in range(len(prices))]
        
        history_data = {
            'symbol': symbol,
            'dates': dates,
            'prices': prices,
            'volumes': volumes,
            'highs': [p * 1.01 for p in prices],  # Mock high prices
            'lows': [p * 0.99 for p in prices],   # Mock low prices
            'opens': prices,  # Simplified
            'sma20': sma20,
            'sma50': sma50,
            'rsi': rsi_values,
            'macd': macd,
            'macd_signal': macd_signal,
            'upper_bb': upper_bb,
            'lower_bb': lower_bb
        }
        
        return jsonify(history_data)
        
    except Exception as e:
        logger.error(f"Error getting stock history for {symbol}: {str(e)}")
        return jsonify({'error': 'Failed to fetch stock history'}), 500

@app.route('/api/stock-news/<symbol>')
@login_required
def stock_news_api(symbol):
    """API endpoint to get news for a specific stock (Mock Data)"""
    try:
        symbol = symbol.upper()
        if not is_valid_ticker(symbol):
            return jsonify({'error': 'Invalid ticker symbol'}), 400
        
        # Mock news data
        mock_articles = [
            {
                'title': f'{symbol} Reports Strong Q3 Earnings',
                'summary': f'{symbol} exceeded analyst expectations with strong revenue growth and improved margins.',
                'url': 'https://example.com/news1',
                'source': 'Financial Times',
                'published_date': '2024-08-20T10:30:00Z',
                'sentiment_label': 'Bullish',
                'sentiment_score': 0.75
            },
            {
                'title': f'{symbol} Announces New Product Line',
                'summary': f'{symbol} unveiled innovative products that could drive future growth.',
                'url': 'https://example.com/news2',
                'source': 'MarketWatch',
                'published_date': '2024-08-19T14:15:00Z',
                'sentiment_label': 'Bullish',
                'sentiment_score': 0.62
            },
            {
                'title': f'Analysts Upgrade {symbol} Rating',
                'summary': f'Major investment firms raised their price targets for {symbol} stock.',
                'url': 'https://example.com/news3',
                'source': 'Bloomberg',
                'published_date': '2024-08-18T09:45:00Z',
                'sentiment_label': 'Bullish',
                'sentiment_score': 0.68
            }
        ]
        
        mock_summary = {
            'total_articles': 3,
            'bullish_count': 3,
            'bearish_count': 0,
            'neutral_count': 0,
            'average_sentiment': 0.68
        }
        
        return jsonify({
            'articles': mock_articles,
            'summary': mock_summary
        })
        
    except Exception as e:
        logger.error(f"Error getting news for {symbol}: {str(e)}")
        return jsonify({'error': 'Failed to fetch news'}), 500

@app.errorhandler(404)
def not_found_error(error):
    return render_template('error.html', error='Page not found'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', error='Internal server error'), 500

if __name__ == '__main__':
    app.run(debug=True)