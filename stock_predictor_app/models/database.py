
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

engine = create_engine('sqlite:///stock_predictor.db')
Base.metadata.create_all(engine)
