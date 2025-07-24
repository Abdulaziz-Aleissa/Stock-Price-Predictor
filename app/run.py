
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import os
from app.utils.stock_scoring import StockScoring
import pandas as pd
import numpy as np
from data.process_data import load_data, clean_data, save_data,calculate_technical_indicators
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import logging
import yfinance as yf
from datetime import datetime, timedelta
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from models.database import User, Portfolio, Watchlist, PriceAlert, Notification, engine
from sqlalchemy.orm import sessionmaker
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
import atexit
import os
from models.train_classifier import (
    load_data as load_db_data,  # This imports load_data from train_classifier as load_db_data
    evaluate_model,
    build_model
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer








app = Flask(__name__, 
           static_url_path='/static',
           template_folder='templates',
           static_folder='static')

app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this in production
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

# def get_market_context(ticker):
#     try:
#         stock = yf.Ticker(ticker)
#         info = stock.info
#         return {]
#             'current_price': info.get('regularMarketPrice', 'N/A'),
#             'day_high': info.get('dayHigh', 'N/A'),
#             'day_low': info.get('dayLow', 'N/A'),
#             'volume': info.get('volume', 0),
#             'pe_ratio': info.get('forwardPE', 'N/A'),
#             'pb_ratio': info.get('priceToBook', 'N/A'),
#             'ev_ebitda': info.get('enterpriseToEbitda', 'N/A'),
#             'roe': info.get('returnOnEquity', 'N/A'),
#             #calculate the rsi of the 14 day period
           


#             'dividend_yield': info.get('dividendYield', 'N/A'),
#             'market_cap': info.get('marketCap', 'N/A'),
#             'year_high': info.get('fiftyTwoWeekHigh', 'N/A'),
#             'year_low': info.get('fiftyTwoWeekLow', 'N/A')
#         }
#     except:
#         return None
    


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







@app.route('/')
def index():
    return render_template('main.html')

@app.route('/financial-literacy')
def financial_literacy():
    return render_template('financial_literacy.html')







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
            
        # Final check for NaN values
        if X.isna().any().any():
            X = X.fillna(0)
            logger.info("Filled remaining NaN values with 0")
        
        # Scale features and make predictions
        X_scaled = scaler.transform(X)
        predicted_prices = model.predict(X_scaled).tolist()
        tomorrow_prediction = predicted_prices[-1]
        
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

        logger.info(f"Prediction complete for {stock_ticker}")
        logger.info(f"Current Price: ${current_price:.2f}")
        logger.info(f"Tomorrow's Prediction: ${tomorrow_prediction:.2f}")
        logger.info(f"Expected Change: {price_change_pct:.2f}%")

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
            }
        )

    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return render_template('error.html', error=str(e))


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
    
    return render_template('dashboard.html',
                         portfolio=portfolio_data,
                         watchlist=watchlist_data,
                         alerts=alerts_data,
                         notifications=notifications,
                         summary=portfolio_summary)

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

@app.errorhandler(404)
def not_found_error(error):
    return render_template('error.html', error='Page not found'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', error='Internal server error'), 500

if __name__ == '__main__':
    app.run(debug=True)