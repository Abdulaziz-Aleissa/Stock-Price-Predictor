"""Application configuration settings."""

import os
from typing import Optional


class Config:
    """Base configuration class."""
    
    # Flask Settings
    SECRET_KEY: str = os.environ.get('SECRET_KEY', 'your-secret-key-here')
    DEBUG: bool = False
    TESTING: bool = False
    
    # Database Settings
    DATABASE_URL: str = os.environ.get('DATABASE_URL', 'sqlite:///stock_predictor.db')
    
    # API Settings
    API_RATE_LIMIT: str = os.environ.get('API_RATE_LIMIT', '100/hour')
    
    # Stock Data Settings
    DEFAULT_STOCK_PERIOD: str = 'max'
    DEFAULT_PREDICTION_DAYS: int = 1
    
    # Model Settings
    MODEL_CACHE_DIR: str = os.environ.get('MODEL_CACHE_DIR', 'models')
    DATA_CACHE_DIR: str = os.environ.get('DATA_CACHE_DIR', 'data')
    
    # Alert Settings
    ALERT_CHECK_INTERVAL_MINUTES: int = int(os.environ.get('ALERT_CHECK_INTERVAL_MINUTES', '5'))
    
    # Email Settings (for future notifications)
    MAIL_SERVER: Optional[str] = os.environ.get('MAIL_SERVER')
    MAIL_PORT: int = int(os.environ.get('MAIL_PORT', '587'))
    MAIL_USE_TLS: bool = os.environ.get('MAIL_USE_TLS', 'true').lower() in ['true', '1', 'yes']
    MAIL_USERNAME: Optional[str] = os.environ.get('MAIL_USERNAME')
    MAIL_PASSWORD: Optional[str] = os.environ.get('MAIL_PASSWORD')


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key')


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    SECRET_KEY = os.environ.get('SECRET_KEY')
    
    @classmethod
    def validate(cls) -> None:
        """Validate production configuration."""
        if not cls.SECRET_KEY:
            raise ValueError("SECRET_KEY must be set in production")


class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    DATABASE_URL = 'sqlite:///:memory:'
    SECRET_KEY = 'testing-secret-key'


# Configuration mapping
config_map = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


def get_config(config_name: Optional[str] = None) -> Config:
    """Get configuration based on environment."""
    config_name = config_name or os.environ.get('FLASK_ENV', 'default')
    config_class = config_map.get(config_name, DevelopmentConfig)
    
    if config_name == 'production':
        config_class.validate()
    
    return config_class()