"""Stock data validators."""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Tuple

from ...core.exceptions import ValidationError
from ...core.constants import VALIDATION_RULES


logger = logging.getLogger(__name__)


class StockValidator:
    """Validator for stock data and parameters."""
    
    def __init__(self):
        """Initialize the validator."""
        self.required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    def validate_ticker_symbol(self, ticker: str) -> bool:
        """Validate ticker symbol format."""
        if not ticker:
            return False
        
        ticker = ticker.strip().upper()
        
        # Check length
        if len(ticker) < 1:  # Allow single character tickers
            return False
        
        if len(ticker) > VALIDATION_RULES['TICKER_MAX_LENGTH']:
            return False
        
        # Check for valid characters (letters, numbers, dots)
        import re
        if not re.match(r'^[A-Z0-9.]+$', ticker):
            return False
        
        return True
    
    def validate_price(self, price: float) -> bool:
        """Validate price value."""
        try:
            price = float(price)
            return (VALIDATION_RULES['PRICE_MIN_VALUE'] <= price <= VALIDATION_RULES['PRICE_MAX_VALUE'])
        except (ValueError, TypeError):
            return False
    
    def validate_shares(self, shares: float) -> bool:
        """Validate shares value."""
        try:
            shares = float(shares)
            return (VALIDATION_RULES['SHARES_MIN_VALUE'] <= shares <= VALIDATION_RULES['SHARES_MAX_VALUE'])
        except (ValueError, TypeError):
            return False
    
    def validate_dataframe(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate stock data DataFrame."""
        errors = []
        
        # Check if DataFrame is empty
        if df is None or df.empty:
            errors.append("DataFrame is empty")
            return False, errors
        
        # Check for required columns
        missing_columns = set(self.required_columns) - set(df.columns)
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        
        # Check for negative prices
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            if col in df.columns:
                if (df[col] <= 0).any():
                    errors.append(f"Found non-positive values in {col}")
        
        # Check for negative volume
        if 'Volume' in df.columns:
            if (df['Volume'] < 0).any():
                errors.append("Found negative volume values")
        
        # Check for logical consistency
        if 'High' in df.columns and 'Low' in df.columns:
            if (df['High'] < df['Low']).any():
                errors.append("Found rows where High < Low")
        
        if all(col in df.columns for col in ['High', 'Low', 'Close']):
            if (df['High'] < df['Close']).any():
                errors.append("Found rows where High < Close")
            if (df['Close'] < df['Low']).any():
                errors.append("Found rows where Close < Low")
        
        # Check data size
        if len(df) < 30:
            errors.append("Dataset is very small (< 30 data points)")
        
        # Check for excessive missing values
        missing_percentage = df.isnull().sum() / len(df) * 100
        problematic_columns = missing_percentage[missing_percentage > 50]
        if not problematic_columns.empty:
            errors.append(f"Columns with >50% missing values: {problematic_columns.index.tolist()}")
        
        return len(errors) == 0, errors
    
    def validate_date_range(self, start_date: str, end_date: str) -> bool:
        """Validate date range."""
        try:
            from datetime import datetime
            
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
            
            # Check if start is before end
            if start >= end:
                return False
            
            # Check if dates are not too far in the future
            now = datetime.now()
            if start > now or end > now:
                return False
            
            return True
            
        except (ValueError, TypeError):
            return False
    
    def validate_portfolio_data(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate portfolio input data."""
        errors = []
        required_fields = ['symbol', 'shares', 'purchase_price']
        
        # Check required fields
        for field in required_fields:
            if field not in data or data[field] is None:
                errors.append(f"Missing required field: {field}")
        
        # Validate ticker symbol
        if 'symbol' in data:
            if not self.validate_ticker_symbol(data['symbol']):
                errors.append("Invalid ticker symbol")
        
        # Validate shares
        if 'shares' in data:
            if not self.validate_shares(data['shares']):
                errors.append("Invalid shares amount")
        
        # Validate price
        if 'purchase_price' in data:
            if not self.validate_price(data['purchase_price']):
                errors.append("Invalid purchase price")
        
        return len(errors) == 0, errors
    
    def validate_alert_data(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate alert input data."""
        errors = []
        required_fields = ['symbol', 'target_price', 'condition']
        
        # Check required fields
        for field in required_fields:
            if field not in data or data[field] is None:
                errors.append(f"Missing required field: {field}")
        
        # Validate ticker symbol
        if 'symbol' in data:
            if not self.validate_ticker_symbol(data['symbol']):
                errors.append("Invalid ticker symbol")
        
        # Validate price
        if 'target_price' in data:
            if not self.validate_price(data['target_price']):
                errors.append("Invalid target price")
        
        # Validate condition
        if 'condition' in data:
            valid_conditions = ['above', 'below']
            if data['condition'].lower() not in valid_conditions:
                errors.append(f"Invalid condition. Must be one of: {valid_conditions}")
        
        return len(errors) == 0, errors
    
    def sanitize_ticker(self, ticker: str) -> str:
        """Sanitize ticker symbol."""
        if not ticker:
            return ""
        
        import re
        # Remove whitespace and convert to uppercase
        ticker = ticker.strip().upper()
        
        # Remove invalid characters
        ticker = re.sub(r'[^A-Z0-9.]', '', ticker)
        
        return ticker
    
    def get_validation_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive validation summary."""
        is_valid, errors = self.validate_dataframe(df)
        
        summary = {
            'is_valid': is_valid,
            'errors': errors,
            'row_count': len(df),
            'column_count': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'date_range': {
                'start': df.index.min() if hasattr(df.index, 'min') else None,
                'end': df.index.max() if hasattr(df.index, 'max') else None
            }
        }
        
        # Add price statistics if available
        if 'Close' in df.columns:
            summary['price_stats'] = {
                'min': df['Close'].min(),
                'max': df['Close'].max(),
                'mean': df['Close'].mean(),
                'std': df['Close'].std()
            }
        
        return summary