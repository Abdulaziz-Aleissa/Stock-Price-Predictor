"""
Enhanced Data Collection Module
Supports multiple data sources and real-time data integration
"""
import pandas as pd
import numpy as np
import requests
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import time
import json
from abc import ABC, abstractmethod

# Import configuration and helpers
from ..utils.config import config
from ..utils.helpers import DataValidator, DateHelper, performance_tracker

logger = logging.getLogger(__name__)

class DataSource(ABC):
    """Abstract base class for data sources"""
    
    @abstractmethod
    def get_stock_data(self, symbol: str, period: str = "max") -> pd.DataFrame:
        """Get stock data from the source"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if data source is available"""
        pass

class YahooFinanceSource(DataSource):
    """Yahoo Finance data source"""
    
    def __init__(self):
        self.name = "Yahoo Finance"
        self.base_url = "https://query1.finance.yahoo.com/v8/finance/chart/"
    
    def get_stock_data(self, symbol: str, period: str = "max") -> pd.DataFrame:
        """Get stock data from Yahoo Finance"""
        try:
            performance_tracker.start_timer(f"yahoo_fetch_{symbol}")
            
            # Try using yfinance if available
            try:
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period)
                
                if not df.empty:
                    df.index = pd.to_datetime(df.index)
                    performance_tracker.end_timer(f"yahoo_fetch_{symbol}")
                    logger.info(f"Successfully fetched {len(df)} records from Yahoo Finance for {symbol}")
                    return df
                    
            except ImportError:
                logger.warning("yfinance not available, using manual API approach")
                pass
            
            # Manual API approach if yfinance not available
            url = f"{self.base_url}{symbol}"
            
            # Define period mapping
            period_map = {
                "1d": "1d",
                "5d": "5d", 
                "1mo": "1mo",
                "3mo": "3mo",
                "6mo": "6mo",
                "1y": "1y",
                "2y": "2y",
                "5y": "5y",
                "10y": "10y",
                "ytd": "ytd",
                "max": "max"
            }
            
            params = {
                "range": period_map.get(period, "max"),
                "interval": "1d",
                "includePrePost": "false",
                "events": "div,splits"
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if "chart" not in data or not data["chart"]["result"]:
                raise ValueError(f"No data found for symbol {symbol}")
            
            result = data["chart"]["result"][0]
            timestamps = result["timestamp"]
            ohlcv = result["indicators"]["quote"][0]
            
            df = pd.DataFrame({
                "Open": ohlcv["open"],
                "High": ohlcv["high"], 
                "Low": ohlcv["low"],
                "Close": ohlcv["close"],
                "Volume": ohlcv["volume"]
            })
            
            df.index = pd.to_datetime(timestamps, unit='s')
            df = df.dropna()
            
            performance_tracker.end_timer(f"yahoo_fetch_{symbol}")
            logger.info(f"Successfully fetched {len(df)} records from Yahoo Finance API for {symbol}")
            return df
            
        except Exception as e:
            performance_tracker.end_timer(f"yahoo_fetch_{symbol}")
            logger.error(f"Error fetching data from Yahoo Finance for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def is_available(self) -> bool:
        """Check if Yahoo Finance is available"""
        try:
            response = requests.get("https://finance.yahoo.com", timeout=10)
            return response.status_code == 200
        except:
            return False

class AlphaVantageSource(DataSource):
    """Alpha Vantage data source"""
    
    def __init__(self):
        self.name = "Alpha Vantage"
        self.api_key = config.ALPHA_VANTAGE_API_KEY
        self.base_url = "https://www.alphavantage.co/query"
        self.rate_limit_delay = 12  # seconds (5 calls per minute for free tier)
    
    def get_stock_data(self, symbol: str, period: str = "max") -> pd.DataFrame:
        """Get stock data from Alpha Vantage"""
        try:
            if not self.api_key:
                logger.warning("Alpha Vantage API key not configured")
                return pd.DataFrame()
            
            performance_tracker.start_timer(f"alpha_fetch_{symbol}")
            
            # Use TIME_SERIES_DAILY_ADJUSTED for comprehensive data
            params = {
                "function": "TIME_SERIES_DAILY_ADJUSTED",
                "symbol": symbol,
                "outputsize": "full",
                "apikey": self.api_key
            }
            
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if "Error Message" in data:
                raise ValueError(f"Alpha Vantage error: {data['Error Message']}")
            
            if "Note" in data:
                logger.warning(f"Alpha Vantage rate limit warning: {data['Note']}")
                time.sleep(self.rate_limit_delay)
            
            if "Time Series (Daily)" not in data:
                raise ValueError("No daily time series data found")
            
            time_series = data["Time Series (Daily)"]
            
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.columns = ['Open', 'High', 'Low', 'Close', 'Adjusted_Close', 'Volume', 'Dividend', 'Split']
            
            # Convert to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Use adjusted close as close price
            df['Close'] = df['Adjusted_Close']
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            
            performance_tracker.end_timer(f"alpha_fetch_{symbol}")
            logger.info(f"Successfully fetched {len(df)} records from Alpha Vantage for {symbol}")
            
            # Respect rate limits
            time.sleep(self.rate_limit_delay)
            
            return df
            
        except Exception as e:
            performance_tracker.end_timer(f"alpha_fetch_{symbol}")
            logger.error(f"Error fetching data from Alpha Vantage for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def is_available(self) -> bool:
        """Check if Alpha Vantage is available"""
        try:
            if not self.api_key:
                return False
            
            # Test with a simple API call
            params = {
                "function": "GLOBAL_QUOTE",
                "symbol": "AAPL",
                "apikey": self.api_key
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            data = response.json()
            
            return "Global Quote" in data and "Error Message" not in data
        except:
            return False

class DataCollector:
    """Main data collection class that manages multiple sources"""
    
    def __init__(self):
        self.sources = {
            'yahoo': YahooFinanceSource(),
            'alpha_vantage': AlphaVantageSource()
        }
        self.primary_source = 'yahoo'
        self.fallback_sources = ['alpha_vantage']
        self.cache = {}
        self.cache_duration = config.DATA_SETTINGS['cache_duration']
    
    def get_stock_data(self, symbol: str, period: str = "max", 
                      force_refresh: bool = False) -> pd.DataFrame:
        """
        Get stock data with fallback mechanism
        
        Args:
            symbol: Stock ticker symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            force_refresh: Force refresh data (ignore cache)
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Validate symbol
            if not DataValidator.validate_ticker(symbol):
                raise ValueError(f"Invalid ticker symbol: {symbol}")
            
            symbol = symbol.upper()
            cache_key = f"{symbol}_{period}"
            
            # Check cache first
            if not force_refresh and cache_key in self.cache:
                cached_data, timestamp = self.cache[cache_key]
                if (datetime.now() - timestamp).seconds < self.cache_duration:
                    logger.info(f"Using cached data for {symbol}")
                    return cached_data
            
            # Try primary source first
            df = self._fetch_from_source(self.primary_source, symbol, period)
            
            # Try fallback sources if primary fails
            if df.empty:
                for source_name in self.fallback_sources:
                    logger.info(f"Trying fallback source: {source_name}")
                    df = self._fetch_from_source(source_name, symbol, period)
                    if not df.empty:
                        break
            
            if df.empty:
                raise ValueError(f"Unable to fetch data for {symbol} from any source")
            
            # Validate and clean data
            df = self._validate_and_clean_data(df, symbol)
            
            # Cache the data
            self.cache[cache_key] = (df.copy(), datetime.now())
            
            # Log data quality metrics
            quality_metrics = DataValidator.check_data_quality(df)
            logger.info(f"Data quality for {symbol}: {quality_metrics.get('quality_score', 0)}/100")
            
            return df
            
        except Exception as e:
            logger.error(f"Error collecting data for {symbol}: {str(e)}")
            raise
    
    def get_multiple_stocks(self, symbols: List[str], period: str = "max") -> Dict[str, pd.DataFrame]:
        """Get data for multiple stocks"""
        try:
            results = {}
            
            for symbol in symbols:
                try:
                    df = self.get_stock_data(symbol, period)
                    if not df.empty:
                        results[symbol] = df
                        logger.info(f"Successfully collected data for {symbol}")
                    else:
                        logger.warning(f"No data collected for {symbol}")
                except Exception as e:
                    logger.error(f"Failed to collect data for {symbol}: {str(e)}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Error collecting multiple stocks data: {str(e)}")
            return {}
    
    def get_real_time_price(self, symbol: str) -> Optional[float]:
        """Get real-time price for a symbol"""
        try:
            # For real-time prices, we'll use the latest available data
            df = self.get_stock_data(symbol, period="1d")
            if not df.empty:
                return float(df['Close'].iloc[-1])
            return None
            
        except Exception as e:
            logger.error(f"Error getting real-time price for {symbol}: {str(e)}")
            return None
    
    def _fetch_from_source(self, source_name: str, symbol: str, period: str) -> pd.DataFrame:
        """Fetch data from a specific source"""
        try:
            if source_name not in self.sources:
                logger.error(f"Unknown data source: {source_name}")
                return pd.DataFrame()
            
            source = self.sources[source_name]
            
            if not source.is_available():
                logger.warning(f"Data source {source_name} is not available")
                return pd.DataFrame()
            
            return source.get_stock_data(symbol, period)
            
        except Exception as e:
            logger.error(f"Error fetching from source {source_name}: {str(e)}")
            return pd.DataFrame()
    
    def _validate_and_clean_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Validate and clean the fetched data"""
        try:
            if df.empty:
                return df
            
            # Ensure required columns exist
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Validate data integrity
            # Check for negative prices (should not happen in real data)
            price_columns = ['Open', 'High', 'Low', 'Close']
            for col in price_columns:
                if (df[col] < 0).any():
                    logger.warning(f"Negative prices found in {col} for {symbol}, removing invalid rows")
                    df = df[df[col] >= 0]
            
            # Check for logical inconsistencies (High < Low, etc.)
            invalid_rows = df['High'] < df['Low']
            if invalid_rows.any():
                logger.warning(f"Found {invalid_rows.sum()} rows with High < Low for {symbol}, removing")
                df = df[~invalid_rows]
            
            # Clean numeric data
            df = DataValidator.clean_numeric_data(df)
            
            # Sort by date
            df = df.sort_index()
            
            # Remove duplicates
            df = df[~df.index.duplicated(keep='last')]
            
            logger.info(f"Data validation completed for {symbol}: {len(df)} valid records")
            return df
            
        except Exception as e:
            logger.error(f"Error validating data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_source_status(self) -> Dict[str, bool]:
        """Get status of all data sources"""
        status = {}
        for name, source in self.sources.items():
            try:
                status[name] = source.is_available()
            except:
                status[name] = False
        return status
    
    def clear_cache(self):
        """Clear the data cache"""
        self.cache.clear()
        logger.info("Data cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'cached_items': len(self.cache),
            'cache_keys': list(self.cache.keys()),
            'cache_size_mb': sum(len(str(data)) for data, _ in self.cache.values()) / 1024 / 1024
        }

# Create global data collector instance
data_collector = DataCollector()