"""Custom exceptions for the stock predictor application."""


class StockPredictorError(Exception):
    """Base exception for stock predictor application."""
    pass


class ConfigurationError(StockPredictorError):
    """Raised when there's a configuration error."""
    pass


class DatabaseError(StockPredictorError):
    """Raised when there's a database-related error."""
    pass


class StockDataError(StockPredictorError):
    """Raised when there's an error with stock data."""
    pass


class InvalidTickerError(StockDataError):
    """Raised when an invalid ticker symbol is provided."""
    pass


class DataFetchError(StockDataError):
    """Raised when stock data cannot be fetched."""
    pass


class ModelError(StockPredictorError):
    """Raised when there's an error with ML models."""
    pass


class ModelNotFoundError(ModelError):
    """Raised when a required model cannot be found."""
    pass


class ModelTrainingError(ModelError):
    """Raised when model training fails."""
    pass


class PredictionError(ModelError):
    """Raised when prediction fails."""
    pass


class ValidationError(StockPredictorError):
    """Raised when data validation fails."""
    pass


class AuthenticationError(StockPredictorError):
    """Raised when authentication fails."""
    pass


class PortfolioError(StockPredictorError):
    """Raised when there's an error with portfolio operations."""
    pass


class AlertError(StockPredictorError):
    """Raised when there's an error with alert operations."""
    pass