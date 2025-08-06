"""
Modules package for Stock Price Predictor

This package contains modular components for:
- Stock comparison functionality
- Paper trading system
- Advanced analytics tools
"""

from .compare import stock_comparison
from .paper_trading import create_paper_trading
from .advanced_analytics import advanced_analytics

__all__ = ['stock_comparison', 'create_paper_trading', 'advanced_analytics']