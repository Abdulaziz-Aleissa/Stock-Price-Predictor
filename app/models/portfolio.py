"""Portfolio model."""

from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime

from .base import BaseModel


class Portfolio(BaseModel):
    """Portfolio model for tracking user stock holdings."""
    
    __tablename__ = 'portfolios'
    
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    stock_symbol = Column(String(10), nullable=False)
    shares = Column(Float, nullable=False)
    purchase_price = Column(Float, nullable=False)
    purchase_date = Column(DateTime, default=datetime.now)
    
    # Relationships
    user = relationship('User', back_populates='portfolios')
    
    def __repr__(self):
        """String representation of the portfolio item."""
        return f"<Portfolio(id={self.id}, symbol='{self.stock_symbol}', shares={self.shares})>"
    
    @property
    def total_cost(self):
        """Calculate total cost of the position."""
        return self.shares * self.purchase_price
    
    def calculate_profit_loss(self, current_price: float):
        """Calculate profit/loss for this position."""
        current_value = self.shares * current_price
        return current_value - self.total_cost
    
    def calculate_percentage_change(self, current_price: float):
        """Calculate percentage change from purchase price."""
        return ((current_price - self.purchase_price) / self.purchase_price) * 100


class Watchlist(BaseModel):
    """Watchlist model for tracking stocks of interest."""
    
    __tablename__ = 'watchlists'
    
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    stock_symbol = Column(String(10), nullable=False)
    target_price = Column(Float, nullable=False)
    added_date = Column(DateTime, default=datetime.now)
    
    # Relationships
    user = relationship('User', back_populates='watchlists')
    
    def __repr__(self):
        """String representation of the watchlist item."""
        return f"<Watchlist(id={self.id}, symbol='{self.stock_symbol}', target=${self.target_price})>"
    
    def calculate_distance_to_target(self, current_price: float):
        """Calculate distance from current price to target."""
        return current_price - self.target_price
    
    def calculate_percentage_to_target(self, current_price: float):
        """Calculate percentage distance to target."""
        return ((current_price - self.target_price) / self.target_price) * 100