"""
Central Data Module for yfinance operations
Single module to fetch all yfinance data and export it to other modules
"""

import yfinance as yf
import pandas as pd
import logging
from typing import Dict, Optional, List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class YFinanceDataFetcher:
    """Central class for all yfinance data operations"""
    
    def __init__(self):
        self.cache = {}
        self.cache_timeout = 300  # 5 minutes
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current stock price"""
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
            logger.error(f"Error getting price for {symbol}: {str(e)}")
            return None
    
    def is_valid_ticker(self, ticker: str) -> bool:
        """Validate if ticker exists and has data"""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1d")
            return not hist.empty
        except:
            return False
    
    def get_market_context(self, ticker: str) -> Optional[Dict]:
        """Get comprehensive market data for a stock"""
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
                'rsi_14': latest_rsi_14,
                'dividend_yield': info.get('dividendYield', 'N/A'),
                'market_cap': info.get('marketCap', 'N/A'),
                'year_high': info.get('fiftyTwoWeekHigh', 'N/A'),
                'year_low': info.get('fiftyTwoWeekLow', 'N/A')
            }
        except Exception as e:
            logger.error(f"Error getting market context for {ticker}: {str(e)}")
            return None
    
    def get_historical_data(self, symbol: str, period: str = "1y", interval: str = "1d") -> Optional[pd.DataFrame]:
        """Get historical stock data"""
        try:
            stock = yf.Ticker(symbol)
            return stock.history(period=period, interval=interval)
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {str(e)}")
            return None
    
    def get_company_info(self, symbol: str) -> Optional[Dict]:
        """Get company information"""
        try:
            stock = yf.Ticker(symbol)
            return stock.info
        except Exception as e:
            logger.error(f"Error getting company info for {symbol}: {str(e)}")
            return None
    
    def get_multiple_stock_data(self, symbols: List[str]) -> Dict[str, Optional[Dict]]:
        """Get market data for multiple stocks"""
        result = {}
        for symbol in symbols:
            result[symbol] = self.get_market_context(symbol)
        return result
    
    def get_price_correlation(self, symbol1: str, symbol2: str, period: str = "1y") -> Optional[float]:
        """Calculate price correlation between two stocks"""
        try:
            stock1 = yf.Ticker(symbol1)
            stock2 = yf.Ticker(symbol2)
            
            hist1 = stock1.history(period=period)
            hist2 = stock2.history(period=period)
            
            if hist1.empty or hist2.empty:
                return None
                
            # Align dates and calculate correlation
            common_dates = hist1.index.intersection(hist2.index)
            if len(common_dates) < 20:  # Need sufficient data points
                return None
                
            prices1 = hist1.loc[common_dates, 'Close']
            prices2 = hist2.loc[common_dates, 'Close']
            
            return prices1.corr(prices2)
        except Exception as e:
            logger.error(f"Error calculating correlation between {symbol1} and {symbol2}: {str(e)}")
            return None


# Global instance to be used across the application
yfinance_data = YFinanceDataFetcher()