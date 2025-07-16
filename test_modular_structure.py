"""Basic tests for the modular application structure."""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import the application components
from app.config.settings import get_config, DevelopmentConfig
from app.core.exceptions import ValidationError, InvalidTickerError
from app.core.constants import AlertCondition, VALIDATION_RULES
from app.data.processors.technical_indicators import TechnicalIndicators
from app.data.validators.stock_validator import StockValidator
from app.utils.helpers import format_currency, format_percentage, safe_float


class TestConfiguration(unittest.TestCase):
    """Test configuration management."""
    
    def test_get_config_default(self):
        """Test getting default configuration."""
        config = get_config()
        self.assertIsNotNone(config)
        self.assertEqual(config.__class__, DevelopmentConfig)
    
    def test_config_attributes(self):
        """Test configuration attributes."""
        config = get_config('development')
        self.assertTrue(config.DEBUG)
        self.assertIn('sqlite:', config.DATABASE_URL)
        self.assertIsNotNone(config.SECRET_KEY)


class TestExceptions(unittest.TestCase):
    """Test custom exceptions."""
    
    def test_validation_error(self):
        """Test ValidationError exception."""
        with self.assertRaises(ValidationError):
            raise ValidationError("Test validation error")
    
    def test_invalid_ticker_error(self):
        """Test InvalidTickerError exception."""
        with self.assertRaises(InvalidTickerError):
            raise InvalidTickerError("Invalid ticker")


class TestConstants(unittest.TestCase):
    """Test application constants."""
    
    def test_alert_conditions(self):
        """Test alert condition enum."""
        self.assertEqual(AlertCondition.ABOVE.value, "above")
        self.assertEqual(AlertCondition.BELOW.value, "below")
    
    def test_validation_rules(self):
        """Test validation rules."""
        self.assertIn('TICKER_MAX_LENGTH', VALIDATION_RULES)
        self.assertIn('PRICE_MIN_VALUE', VALIDATION_RULES)
        self.assertGreater(VALIDATION_RULES['TICKER_MAX_LENGTH'], 0)


class TestTechnicalIndicators(unittest.TestCase):
    """Test technical indicators calculation."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample stock data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        
        self.df = pd.DataFrame({
            'Open': 100 + np.random.randn(len(dates)).cumsum(),
            'High': 105 + np.random.randn(len(dates)).cumsum(),
            'Low': 95 + np.random.randn(len(dates)).cumsum(),
            'Close': 100 + np.random.randn(len(dates)).cumsum(),
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
        
        # Ensure High >= Low and Close is within range
        self.df['High'] = np.maximum(self.df['High'], self.df[['Open', 'Close']].max(axis=1))
        self.df['Low'] = np.minimum(self.df['Low'], self.df[['Open', 'Close']].min(axis=1))
        
        self.indicators = TechnicalIndicators()
    
    def test_sma_calculation(self):
        """Test Simple Moving Average calculation."""
        sma_20 = self.indicators.calculate_sma(self.df, 20)
        self.assertEqual(len(sma_20), len(self.df))
        self.assertFalse(sma_20.iloc[20:].isnull().any())
    
    def test_rsi_calculation(self):
        """Test RSI calculation."""
        rsi = self.indicators.calculate_rsi(self.df)
        self.assertEqual(len(rsi), len(self.df))
        # RSI should be between 0 and 100
        valid_rsi = rsi.dropna()
        self.assertTrue((valid_rsi >= 0).all())
        self.assertTrue((valid_rsi <= 100).all())
    
    def test_macd_calculation(self):
        """Test MACD calculation."""
        macd_data = self.indicators.calculate_macd(self.df)
        self.assertIn('MACD', macd_data)
        self.assertIn('MACD_Signal', macd_data)
        self.assertEqual(len(macd_data['MACD']), len(self.df))


class TestStockValidator(unittest.TestCase):
    """Test stock data validation."""
    
    def setUp(self):
        """Set up test data."""
        self.validator = StockValidator()
    
    def test_ticker_validation(self):
        """Test ticker symbol validation."""
        # Valid tickers
        self.assertTrue(self.validator.validate_ticker_symbol('AAPL'))
        self.assertTrue(self.validator.validate_ticker_symbol('MSFT'))
        self.assertTrue(self.validator.validate_ticker_symbol('BRK.A'))
        
        # Invalid tickers
        self.assertFalse(self.validator.validate_ticker_symbol(''))
        self.assertFalse(self.validator.validate_ticker_symbol('A'))  # Too short
        self.assertFalse(self.validator.validate_ticker_symbol('VERYLONGTICKER'))  # Too long
        self.assertFalse(self.validator.validate_ticker_symbol('AAP!'))  # Invalid character
    
    def test_price_validation(self):
        """Test price validation."""
        # Valid prices
        self.assertTrue(self.validator.validate_price(100.50))
        self.assertTrue(self.validator.validate_price(0.01))
        
        # Invalid prices
        self.assertFalse(self.validator.validate_price(0))
        self.assertFalse(self.validator.validate_price(-10))
        self.assertFalse(self.validator.validate_price(1000000))  # Too high
    
    def test_shares_validation(self):
        """Test shares validation."""
        # Valid shares
        self.assertTrue(self.validator.validate_shares(10))
        self.assertTrue(self.validator.validate_shares(0.5))
        
        # Invalid shares
        self.assertFalse(self.validator.validate_shares(0))
        self.assertFalse(self.validator.validate_shares(-5))
    
    def test_dataframe_validation(self):
        """Test DataFrame validation."""
        # Valid DataFrame
        valid_df = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97],
            'Close': [102, 103, 104],
            'Volume': [1000000, 1100000, 1200000]
        })
        
        is_valid, errors = self.validator.validate_dataframe(valid_df)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        
        # Invalid DataFrame (missing columns)
        invalid_df = pd.DataFrame({
            'Close': [100, 101, 102]
        })
        
        is_valid, errors = self.validator.validate_dataframe(invalid_df)
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)


class TestUtilityHelpers(unittest.TestCase):
    """Test utility helper functions."""
    
    def test_format_currency(self):
        """Test currency formatting."""
        self.assertEqual(format_currency(1234.56), "$1,234.56")
        self.assertEqual(format_currency(0), "$0.00")
        self.assertEqual(format_currency(1000000), "$1,000,000.00")
    
    def test_format_percentage(self):
        """Test percentage formatting."""
        self.assertEqual(format_percentage(15.5), "+15.50%")
        self.assertEqual(format_percentage(-5.2), "-5.20%")
        self.assertEqual(format_percentage(0), "+0.00%")
    
    def test_safe_float(self):
        """Test safe float conversion."""
        self.assertEqual(safe_float("123.45"), 123.45)
        self.assertEqual(safe_float("invalid"), 0.0)
        self.assertEqual(safe_float(None), 0.0)
        self.assertEqual(safe_float(""), 0.0)
        self.assertEqual(safe_float("123.45", 999), 123.45)
        self.assertEqual(safe_float("invalid", 999), 999)


class TestApplicationStructure(unittest.TestCase):
    """Test overall application structure."""
    
    def test_import_structure(self):
        """Test that all modules can be imported without errors."""
        try:
            from app import create_app
            from app.services.stock_service import StockService
            from app.services.prediction_service import PredictionService
            from app.services.portfolio_service import PortfolioService
            from app.services.alert_service import AlertService
            from app.models.user import User
            from app.models.portfolio import Portfolio, Watchlist
            from app.models.alert import PriceAlert, Notification
            from app.utils.helpers import format_currency
            from app.utils.decorators import timer
            from app.utils.formatters import format_portfolio_item
        except ImportError as e:
            self.fail(f"Import failed: {e}")
    
    def test_application_creation(self):
        """Test that application can be created."""
        try:
            from app import create_app
            app = create_app('testing')
            self.assertIsNotNone(app)
            self.assertTrue(app.config['TESTING'])
        except Exception as e:
            self.fail(f"Application creation failed: {e}")


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)