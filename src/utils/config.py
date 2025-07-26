"""
Configuration management for the Stock Price Predictor
"""
import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the stock predictor application"""
    
    # API Keys
    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', '')
    REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID', '')
    REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET', '')
    TWITTER_BEARER_TOKEN = os.getenv('TWITTER_BEARER_TOKEN', '')
    
    # Database
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///stock_predictor.db')
    
    # Application Settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-here')
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    
    # Model Parameters
    MODEL_PARAMS = {
        'lstm': {
            'units': [50, 50],
            'dropout': 0.2,
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 0.001
        },
        'gru': {
            'units': [50, 50],
            'dropout': 0.2,
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 0.001
        },
        'arima': {
            'order': (1, 1, 1),
            'seasonal_order': (1, 1, 1, 12)
        },
        'prophet': {
            'yearly_seasonality': True,
            'weekly_seasonality': True,
            'daily_seasonality': False,
            'changepoint_prior_scale': 0.05
        },
        'random_forest': {
            'n_estimators': 200,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        },
        'xgboost': {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        },
        'gradient_boost': {
            'n_estimators': 200,
            'learning_rate': 0.1,
            'max_depth': 5,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'subsample': 0.8,
            'random_state': 42
        }
    }
    
    # Data Collection Settings
    DATA_SETTINGS = {
        'default_period': 'max',
        'min_data_points': 252,  # 1 year of trading days
        'refresh_interval': 300,  # 5 minutes in seconds
        'cache_duration': 3600,   # 1 hour in seconds
    }
    
    # Risk Settings
    RISK_SETTINGS = {
        'confidence_levels': [0.95, 0.99],
        'monte_carlo_simulations': 10000,
        'var_holding_periods': [1, 5, 10, 22],  # days
        'max_portfolio_size': 50,
        'risk_free_rate': 0.02  # 2% annual risk-free rate
    }
    
    # Prediction Horizons
    PREDICTION_HORIZONS = {
        '1_day': 1,
        '1_week': 7,
        '1_month': 30,
        '3_months': 90,
        '1_year': 365
    }
    
    # Technical Indicators Settings
    TECHNICAL_INDICATORS = {
        'rsi_period': 14,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'bb_period': 20,
        'bb_std': 2,
        'sma_periods': [20, 50, 200],
        'ema_periods': [12, 26]
    }
    
    # Sentiment Analysis Settings
    SENTIMENT_SETTINGS = {
        'news_sources': ['yahoo', 'alpha_vantage'],
        'social_sources': ['reddit', 'twitter'],
        'sentiment_window': 7,  # days
        'max_articles': 100,
        'relevance_threshold': 0.7
    }
    
    # Portfolio Optimization Settings
    PORTFOLIO_SETTINGS = {
        'rebalance_frequency': 'monthly',
        'risk_tolerance': 'moderate',  # conservative, moderate, aggressive
        'max_position_size': 0.1,  # 10% max per position
        'min_correlation_threshold': 0.7,
        'optimization_method': 'efficient_frontier'  # efficient_frontier, equal_weight, risk_parity
    }

    @classmethod
    def get_model_params(cls, model_name: str) -> Dict[str, Any]:
        """Get parameters for a specific model"""
        return cls.MODEL_PARAMS.get(model_name, {})
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate configuration settings"""
        required_keys = ['SECRET_KEY']
        for key in required_keys:
            if not getattr(cls, key):
                return False
        return True
    
    @classmethod
    def get_api_keys(cls) -> Dict[str, str]:
        """Get all API keys"""
        return {
            'alpha_vantage': cls.ALPHA_VANTAGE_API_KEY,
            'reddit_client_id': cls.REDDIT_CLIENT_ID,
            'reddit_client_secret': cls.REDDIT_CLIENT_SECRET,
            'twitter_bearer': cls.TWITTER_BEARER_TOKEN
        }

# Create global config instance
config = Config()