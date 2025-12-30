
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from flask_login import UserMixin

Base = declarative_base()

class User(Base, UserMixin):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True)
    email = Column(String(120), unique=True)
    password_hash = Column(String(128))
    is_active = Column(Boolean, default=True)
    portfolios = relationship('Portfolio', backref='user')
    watchlists = relationship('Watchlist', backref='user')
    alerts = relationship('PriceAlert', backref='user')

class Portfolio(Base):
    __tablename__ = 'portfolios'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    stock_symbol = Column(String(10))
    shares = Column(Float)
    purchase_price = Column(Float)
    purchase_date = Column(DateTime)

class Watchlist(Base):
    __tablename__ = 'watchlists'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    stock_symbol = Column(String(10))
    target_price = Column(Float)
    added_date = Column(DateTime)

class PriceAlert(Base):
    __tablename__ = 'price_alerts'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    stock_symbol = Column(String(10))
    target_price = Column(Float)
    condition = Column(String(10))  # 'above' or 'below'
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.now)

class Notification(Base):
    __tablename__ = 'notifications'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    message = Column(String(500))
    read = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.now)

class PaperPortfolio(Base):
    __tablename__ = 'paper_portfolios'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    stock_symbol = Column(String(10))
    shares = Column(Float)
    average_price = Column(Float)  # Average cost basis
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

class PaperTransaction(Base):
    __tablename__ = 'paper_transactions'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    stock_symbol = Column(String(10))
    transaction_type = Column(String(4))  # 'BUY' or 'SELL'
    shares = Column(Float)
    price = Column(Float)
    total_amount = Column(Float)  # shares * price
    created_at = Column(DateTime, default=datetime.now)

class PaperCashBalance(Base):
    __tablename__ = 'paper_cash_balances'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), unique=True)
    cash_balance = Column(Float, default=100000.0)  # $100,000 starting balance
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

class PredictionHistory(Base):
    __tablename__ = 'prediction_history'
    id = Column(Integer, primary_key=True)
    stock_symbol = Column(String(10), nullable=False)
    prediction_date = Column(DateTime, default=datetime.now)
    target_date = Column(DateTime, nullable=False)  # Date the prediction was for
    predicted_price = Column(Float, nullable=False)
    current_price = Column(Float, nullable=False)  # Price when prediction was made
    actual_price = Column(Float)  # Actual price on target date (filled later)
    price_change_pct = Column(Float)  # Predicted percentage change
    actual_change_pct = Column(Float)  # Actual percentage change (filled later)
    model_accuracy = Column(Float)  # R2 score at time of prediction
    mae = Column(Float)  # Mean absolute error at time of prediction
    rmse = Column(Float)  # Root mean square error at time of prediction
    prediction_error = Column(Float)  # |predicted - actual| (filled later)
    direction_correct = Column(Boolean)  # Whether direction was correct (filled later)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

class SoldOrder(Base):
    __tablename__ = 'sold_orders'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    stock_symbol = Column(String(10))
    shares = Column(Float)
    purchase_price = Column(Float)  # Original purchase price
    sell_price = Column(Float)  # Price at which stock was sold
    profit_loss = Column(Float)  # (sell_price - purchase_price) * shares
    sell_date = Column(DateTime, default=datetime.now)
    purchase_date = Column(DateTime)  # Original purchase date

class InvestmentBalance(Base):
    __tablename__ = 'investment_balances'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), unique=True)
    initial_amount = Column(Float, default=10000.0)  # Default $10,000
    available_amount = Column(Float, default=10000.0)  # Current available amount
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

engine = create_engine('sqlite:///stock_predictor.db')
Base.metadata.create_all(engine)
