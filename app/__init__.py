"""Stock Price Predictor Application

A modular Flask application for stock price prediction and portfolio management.
"""

from .application import create_app

__version__ = "2.0.0"
__all__ = ["create_app"]
