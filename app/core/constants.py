"""Application constants."""

from enum import Enum
from typing import Dict, List


class AlertCondition(Enum):
    """Alert condition types."""
    ABOVE = "above"
    BELOW = "below"


class ModelType(Enum):
    """Machine learning model types."""
    GRADIENT_BOOSTING = "gradient_boosting"
    RANDOM_FOREST = "random_forest"
    LINEAR_REGRESSION = "linear_regression"


class StockPeriod(Enum):
    """Stock data time periods."""
    ONE_DAY = "1d"
    FIVE_DAYS = "5d"
    ONE_MONTH = "1mo"
    THREE_MONTHS = "3mo"
    SIX_MONTHS = "6mo"
    ONE_YEAR = "1y"
    TWO_YEARS = "2y"
    FIVE_YEARS = "5y"
    TEN_YEARS = "10y"
    MAX = "max"


class TimeFrame(Enum):
    """Time frames for stock comparison."""
    ONE_WEEK = "1wk"
    ONE_MONTH = "1mo"
    THREE_MONTHS = "3mo"
    SIX_MONTHS = "6mo"
    ONE_YEAR = "1y"
    TWO_YEARS = "2y"


# Technical Indicators Configuration
TECHNICAL_INDICATORS = {
    'RSI_PERIOD': 14,
    'SMA_SHORT_PERIOD': 20,
    'SMA_LONG_PERIOD': 50,
    'MACD_FAST_PERIOD': 12,
    'MACD_SLOW_PERIOD': 26,
    'MACD_SIGNAL_PERIOD': 9,
    'BOLLINGER_PERIOD': 20,
    'BOLLINGER_STD': 2,
    'VOLATILITY_PERIOD': 20
}

# Model Configuration
MODEL_CONFIG = {
    'GRADIENT_BOOSTING': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 5,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'subsample': 0.8,
        'random_state': 42
    },
    'RANDOM_FOREST': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42
    }
}

# Data Processing Constants
DEFAULT_FEATURES = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'Price_Change', 'Volume_Change', 'High_Low_Range',
    'Daily_Return', 'Volatility', 'SMA_20', 'SMA_50',
    'RSI', 'MACD', 'MACD_Signal'
]

# API Response Messages
MESSAGES = {
    'INVALID_TICKER': 'Invalid ticker symbol provided',
    'DATA_FETCH_ERROR': 'Failed to fetch stock data',
    'MODEL_NOT_FOUND': 'Prediction model not found',
    'PREDICTION_ERROR': 'Failed to generate prediction',
    'AUTHENTICATION_REQUIRED': 'Authentication required',
    'INSUFFICIENT_DATA': 'Insufficient data for analysis',
    'VALIDATION_ERROR': 'Data validation failed'
}

# Default Values
DEFAULTS = {
    'STOCK_PERIOD': 'max',
    'PREDICTION_DAYS': 1,
    'MODEL_TYPE': ModelType.GRADIENT_BOOSTING.value,
    'ALERT_CHECK_INTERVAL': 5,  # minutes
    'CACHE_EXPIRY': 3600,  # seconds
    'MIN_DATA_POINTS': 100
}

# File Extensions and Paths
FILE_EXTENSIONS = {
    'MODEL': '.pkl',
    'DATA': '.db',
    'CONFIG': '.json',
    'LOG': '.log'
}

# Validation Rules
VALIDATION_RULES = {
    'TICKER_MAX_LENGTH': 10,
    'TICKER_MIN_LENGTH': 1,
    'PRICE_MIN_VALUE': 0.01,
    'PRICE_MAX_VALUE': 100000,
    'SHARES_MIN_VALUE': 0.001,
    'SHARES_MAX_VALUE': 1000000
}

# HTTP Status Codes (for API responses)
HTTP_STATUS = {
    'OK': 200,
    'CREATED': 201,
    'BAD_REQUEST': 400,
    'UNAUTHORIZED': 401,
    'FORBIDDEN': 403,
    'NOT_FOUND': 404,
    'INTERNAL_ERROR': 500
}