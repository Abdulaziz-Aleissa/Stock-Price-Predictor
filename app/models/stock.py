"""Stock model for stock-related data."""

from sqlalchemy import Column, String, Float, Text, DateTime
from datetime import datetime

from .base import BaseModel


class Stock(BaseModel):
    """Stock model for caching stock information."""
    
    __tablename__ = 'stocks'
    
    symbol = Column(String(10), unique=True, nullable=False)
    name = Column(String(200))
    sector = Column(String(100))
    industry = Column(String(100))
    market_cap = Column(Float)
    current_price = Column(Float)
    day_high = Column(Float)
    day_low = Column(Float)
    year_high = Column(Float)
    year_low = Column(Float)
    volume = Column(Float)
    pe_ratio = Column(Float)
    description = Column(Text)
    last_updated = Column(DateTime, default=datetime.now)
    
    def __repr__(self):
        """String representation of the stock."""
        return f"<Stock(symbol='{self.symbol}', price=${self.current_price})>"
    
    def update_price_data(self, price_data: dict):
        """Update stock with fresh price data."""
        self.current_price = price_data.get('current_price')
        self.day_high = price_data.get('day_high')
        self.day_low = price_data.get('day_low')
        self.year_high = price_data.get('year_high')
        self.year_low = price_data.get('year_low')
        self.volume = price_data.get('volume')
        self.pe_ratio = price_data.get('pe_ratio')
        self.market_cap = price_data.get('market_cap')
        self.last_updated = datetime.now()
    
    def is_data_stale(self, max_age_minutes: int = 15) -> bool:
        """Check if cached data is stale."""
        if not self.last_updated:
            return True
        
        age = datetime.now() - self.last_updated
        return age.total_seconds() > (max_age_minutes * 60)


class StockPrice(BaseModel):
    """Stock price history model."""
    
    __tablename__ = 'stock_prices'
    
    symbol = Column(String(10), nullable=False)
    date = Column(DateTime, nullable=False)
    open_price = Column(Float, nullable=False)
    high_price = Column(Float, nullable=False)
    low_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    
    def __repr__(self):
        """String representation of the stock price."""
        return f"<StockPrice(symbol='{self.symbol}', date={self.date}, close=${self.close_price})>"