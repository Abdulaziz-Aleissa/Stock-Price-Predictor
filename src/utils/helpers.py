"""
Helper utilities for the Stock Price Predictor
"""
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataValidator:
    """Validates and cleans data for ML models"""
    
    @staticmethod
    def validate_ticker(ticker: str) -> bool:
        """Validate stock ticker format"""
        try:
            # Basic validation - should be uppercase letters, 1-5 characters
            if not ticker or not isinstance(ticker, str):
                return False
            ticker = ticker.strip().upper()
            return len(ticker) >= 1 and len(ticker) <= 5 and ticker.isalpha()
        except:
            return False
    
    @staticmethod
    def clean_numeric_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean numeric data by handling infinities and NaNs"""
        try:
            # Replace infinite values with NaN
            df = df.replace([np.inf, -np.inf], np.nan)
            
            # Fill NaN values with appropriate methods
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                if df[col].isna().any():
                    # Forward fill first, then backward fill, then fill with 0
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            return df
            
        except Exception as e:
            logger.error(f"Error cleaning numeric data: {str(e)}")
            raise
    
    @staticmethod
    def validate_date_range(start_date: datetime, end_date: datetime) -> bool:
        """Validate date range"""
        try:
            if not isinstance(start_date, datetime) or not isinstance(end_date, datetime):
                return False
            
            return start_date < end_date and end_date <= datetime.now()
        except:
            return False
    
    @staticmethod
    def check_data_quality(df: pd.DataFrame, min_rows: int = 100) -> Dict[str, Any]:
        """Check data quality and return metrics"""
        try:
            quality_metrics = {
                'row_count': len(df),
                'column_count': len(df.columns),
                'missing_data_percent': (df.isna().sum().sum() / (len(df) * len(df.columns))) * 100,
                'duplicate_rows': df.duplicated().sum(),
                'data_types': df.dtypes.to_dict(),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
                'quality_score': 0
            }
            
            # Calculate quality score (0-100)
            score = 100
            if quality_metrics['row_count'] < min_rows:
                score -= 30
            if quality_metrics['missing_data_percent'] > 10:
                score -= 20
            if quality_metrics['duplicate_rows'] > len(df) * 0.05:
                score -= 10
            
            quality_metrics['quality_score'] = max(0, score)
            quality_metrics['is_good_quality'] = quality_metrics['quality_score'] >= 70
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Error checking data quality: {str(e)}")
            return {'error': str(e)}

class DateHelper:
    """Helper functions for date operations"""
    
    @staticmethod
    def get_trading_days(start_date: datetime, end_date: datetime) -> int:
        """Calculate number of trading days between dates"""
        try:
            # Simple approximation: 5/7 of total days (accounting for weekends)
            total_days = (end_date - start_date).days
            return int(total_days * 5 / 7)
        except:
            return 0
    
    @staticmethod
    def get_next_trading_day(date: datetime) -> datetime:
        """Get next trading day (skip weekends)"""
        try:
            next_day = date + timedelta(days=1)
            while next_day.weekday() > 4:  # Monday=0, Sunday=6
                next_day += timedelta(days=1)
            return next_day
        except:
            return date + timedelta(days=1)
    
    @staticmethod
    def get_market_calendar(year: int) -> List[datetime]:
        """Get trading days for a given year (simplified - excludes holidays)"""
        try:
            start_date = datetime(year, 1, 1)
            end_date = datetime(year, 12, 31)
            
            trading_days = []
            current_date = start_date
            
            while current_date <= end_date:
                if current_date.weekday() < 5:  # Monday=0, Friday=4
                    trading_days.append(current_date)
                current_date += timedelta(days=1)
            
            return trading_days
        except:
            return []

class MathUtils:
    """Mathematical utility functions"""
    
    @staticmethod
    def calculate_returns(prices: pd.Series, method: str = 'simple') -> pd.Series:
        """Calculate returns from price series"""
        try:
            if method == 'simple':
                return prices.pct_change().fillna(0)
            elif method == 'log':
                return np.log(prices / prices.shift(1)).fillna(0)
            else:
                raise ValueError("Method must be 'simple' or 'log'")
        except Exception as e:
            logger.error(f"Error calculating returns: {str(e)}")
            return pd.Series(index=prices.index, data=0)
    
    @staticmethod
    def calculate_volatility(returns: pd.Series, window: int = 20, annualize: bool = True) -> pd.Series:
        """Calculate rolling volatility"""
        try:
            vol = returns.rolling(window=window).std()
            if annualize:
                vol = vol * np.sqrt(252)  # Annualize assuming 252 trading days
            return vol.fillna(0)
        except Exception as e:
            logger.error(f"Error calculating volatility: {str(e)}")
            return pd.Series(index=returns.index, data=0)
    
    @staticmethod
    def normalize_data(data: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """Normalize data using different methods"""
        try:
            if method == 'minmax':
                return (data - np.min(data)) / (np.max(data) - np.min(data))
            elif method == 'zscore':
                return (data - np.mean(data)) / np.std(data)
            elif method == 'robust':
                median = np.median(data)
                mad = np.median(np.abs(data - median))
                return (data - median) / mad
            else:
                raise ValueError("Method must be 'minmax', 'zscore', or 'robust'")
        except Exception as e:
            logger.error(f"Error normalizing data: {str(e)}")
            return data
    
    @staticmethod
    def safe_divide(numerator: Union[float, np.ndarray], denominator: Union[float, np.ndarray], 
                   default: float = 0.0) -> Union[float, np.ndarray]:
        """Safe division that handles division by zero"""
        try:
            with np.errstate(divide='ignore', invalid='ignore'):
                result = np.divide(numerator, denominator)
                if isinstance(result, np.ndarray):
                    result[~np.isfinite(result)] = default
                elif not np.isfinite(result):
                    result = default
                return result
        except:
            return default

class PerformanceTracker:
    """Track and measure performance of various operations"""
    
    def __init__(self):
        self.timings = {}
        self.counters = {}
    
    def start_timer(self, name: str):
        """Start timing an operation"""
        self.timings[name] = {'start': datetime.now()}
    
    def end_timer(self, name: str) -> float:
        """End timing an operation and return duration in seconds"""
        if name in self.timings and 'start' in self.timings[name]:
            duration = (datetime.now() - self.timings[name]['start']).total_seconds()
            self.timings[name]['duration'] = duration
            return duration
        return 0.0
    
    def increment_counter(self, name: str, value: int = 1):
        """Increment a counter"""
        self.counters[name] = self.counters.get(name, 0) + value
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        return {
            'timings': {k: v.get('duration', 0) for k, v in self.timings.items()},
            'counters': self.counters.copy()
        }

# Create global performance tracker
performance_tracker = PerformanceTracker()

def setup_logging(level: str = 'INFO', log_file: Optional[str] = None):
    """Setup logging configuration"""
    try:
        log_level = getattr(logging, level.upper())
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Setup handlers
        handlers = []
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)
        
        # File handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            handlers.append(file_handler)
        
        # Configure root logger
        logging.basicConfig(
            level=log_level,
            handlers=handlers,
            force=True
        )
        
        return True
        
    except Exception as e:
        print(f"Error setting up logging: {str(e)}")
        return False

def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage statistics"""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': process.memory_percent()
        }
    except ImportError:
        return {'error': 'psutil not available'}
    except Exception as e:
        return {'error': str(e)}

def format_currency(amount: float, currency: str = 'USD') -> str:
    """Format currency amount"""
    try:
        if currency == 'USD':
            return f"${amount:,.2f}"
        else:
            return f"{amount:,.2f} {currency}"
    except:
        return str(amount)

def format_percentage(value: float, decimals: int = 2) -> str:
    """Format percentage value"""
    try:
        return f"{value:.{decimals}f}%"
    except:
        return str(value)

def chunks(lst: List[Any], n: int) -> List[List[Any]]:
    """Yield successive n-sized chunks from list"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]