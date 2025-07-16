"""Stock service for handling stock data operations."""

import yfinance as yf
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

from ..core.exceptions import InvalidTickerError, DataFetchError
from ..core.constants import StockPeriod, DEFAULTS


logger = logging.getLogger(__name__)


class StockService:
    """Service class for stock-related operations."""
    
    def __init__(self):
        """Initialize the stock service."""
        self.cache = {}
        self.cache_expiry = DEFAULTS['CACHE_EXPIRY']
    
    def is_valid_ticker(self, ticker: str) -> bool:
        """Validate if ticker symbol is valid."""
        try:
            if not ticker or len(ticker) < 1 or len(ticker) > 10:
                return False
            
            stock = yf.Ticker(ticker.upper())
            hist = stock.history(period="1d")
            return not hist.empty
        except Exception as e:
            logger.warning(f"Ticker validation failed for {ticker}: {str(e)}")
            return False
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a stock symbol."""
        try:
            stock = yf.Ticker(symbol.upper())
            
            # Try to get real-time price during market hours
            info = stock.info
            real_time_price = info.get('regularMarketPrice')
            if real_time_price:
                return float(real_time_price)
            
            # If market closed, get latest closing price
            hist = stock.history(period="1d", interval="1m")
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {str(e)}")
            return None
    
    def get_market_context(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive market context for a stock."""
        try:
            stock = yf.Ticker(ticker.upper())
            info = stock.info
            
            context = {
                'symbol': ticker.upper(),
                'name': info.get('longName', 'N/A'),
                'current_price': info.get('regularMarketPrice', 'N/A'),
                'day_high': info.get('dayHigh', 'N/A'),
                'day_low': info.get('dayLow', 'N/A'),
                'volume': info.get('volume', 0),
                'pe_ratio': info.get('forwardPE', 'N/A'),
                'market_cap': info.get('marketCap', 'N/A'),
                'year_high': info.get('fiftyTwoWeekHigh', 'N/A'),
                'year_low': info.get('fiftyTwoWeekLow', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'dividend_yield': info.get('dividendYield', 'N/A'),
                'beta': info.get('beta', 'N/A')
            }
            
            return context
            
        except Exception as e:
            logger.error(f"Error getting market context for {ticker}: {str(e)}")
            return None
    
    def get_historical_data(self, symbol: str, period: str = "max", 
                          start_date: Optional[str] = None, 
                          end_date: Optional[str] = None) -> Optional[Any]:
        """Get historical stock data."""
        try:
            stock = yf.Ticker(symbol.upper())
            
            if start_date and end_date:
                df = stock.history(start=start_date, end=end_date)
            else:
                df = stock.history(period=period)
            
            if df.empty:
                raise DataFetchError(f"No data found for ticker {symbol}")
            
            logger.info(f"Successfully fetched {len(df)} records for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            raise DataFetchError(f"Failed to fetch data for {symbol}")
    
    def compare_stocks(self, symbol1: str, symbol2: str, timeframe: str = "1y") -> Dict[str, Any]:
        """Compare two stocks over a timeframe."""
        try:
            stock1 = yf.Ticker(symbol1.upper())
            stock2 = yf.Ticker(symbol2.upper())
            
            hist1 = stock1.history(period=timeframe)
            hist2 = stock2.history(period=timeframe)
            
            if hist1.empty or hist2.empty:
                raise DataFetchError("Unable to fetch comparison data")
            
            comparison_data = {
                'symbol1': {
                    'symbol': symbol1.upper(),
                    'prices': hist1['Close'].tolist(),
                    'dates': hist1.index.strftime('%Y-%m-%d').tolist(),
                    'change_percent': ((hist1['Close'].iloc[-1] - hist1['Close'].iloc[0]) / hist1['Close'].iloc[0] * 100),
                    'volume_avg': hist1['Volume'].mean(),
                    'high': hist1['High'].max(),
                    'low': hist1['Low'].min(),
                    'volatility': hist1['Close'].pct_change().std() * 100
                },
                'symbol2': {
                    'symbol': symbol2.upper(),
                    'prices': hist2['Close'].tolist(),
                    'dates': hist2.index.strftime('%Y-%m-%d').tolist(),
                    'change_percent': ((hist2['Close'].iloc[-1] - hist2['Close'].iloc[0]) / hist2['Close'].iloc[0] * 100),
                    'volume_avg': hist2['Volume'].mean(),
                    'high': hist2['High'].max(),
                    'low': hist2['Low'].min(),
                    'volatility': hist2['Close'].pct_change().std() * 100
                },
                'timeframe': timeframe,
                'comparison_date': datetime.now().isoformat()
            }
            
            return comparison_data
            
        except Exception as e:
            logger.error(f"Error comparing stocks {symbol1} vs {symbol2}: {str(e)}")
            raise DataFetchError(f"Failed to compare stocks")
    
    def get_stock_info_cached(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get stock info with caching."""
        cache_key = f"{symbol.upper()}_info"
        
        # Check cache
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if (datetime.now() - timestamp).total_seconds() < self.cache_expiry:
                return cached_data
        
        # Fetch fresh data
        info = self.get_market_context(symbol)
        if info:
            self.cache[cache_key] = (info, datetime.now())
        
        return info
    
    def clear_cache(self):
        """Clear the service cache."""
        self.cache.clear()
        logger.info("Stock service cache cleared")