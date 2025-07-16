"""Base data processor class."""

import pandas as pd
import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from ...core.exceptions import ValidationError


logger = logging.getLogger(__name__)


class BaseProcessor(ABC):
    """Base class for data processors."""
    
    def __init__(self):
        """Initialize the base processor."""
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def process(self, data: Any, **kwargs) -> Any:
        """Process the input data."""
        pass
    
    def validate_dataframe(self, df: pd.DataFrame, required_columns: list = None) -> None:
        """Validate a pandas DataFrame."""
        if df is None or df.empty:
            raise ValidationError("DataFrame is empty or None")
        
        if required_columns:
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                raise ValidationError(f"Missing required columns: {missing_columns}")
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'forward_fill') -> pd.DataFrame:
        """Handle missing values in DataFrame."""
        if strategy == 'forward_fill':
            return df.fillna(method='ffill').fillna(method='bfill')
        elif strategy == 'backward_fill':
            return df.fillna(method='bfill').fillna(method='ffill')
        elif strategy == 'interpolate':
            return df.interpolate().fillna(method='ffill').fillna(method='bfill')
        elif strategy == 'drop':
            return df.dropna()
        elif strategy == 'zero':
            return df.fillna(0)
        else:
            raise ValidationError(f"Unknown missing value strategy: {strategy}")
    
    def ensure_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure DataFrame has datetime index."""
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'Date' in df.columns:
                df = df.set_index('Date')
            elif 'date' in df.columns:
                df = df.set_index('date')
            elif 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            
            # Convert index to datetime if it's not already
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                raise ValidationError(f"Cannot convert index to datetime: {str(e)}")
        
        return df
    
    def calculate_returns(self, df: pd.DataFrame, price_column: str = 'Close') -> pd.DataFrame:
        """Calculate returns for a price series."""
        df = df.copy()
        df['Daily_Return'] = df[price_column].pct_change()
        df['Log_Return'] = np.log(df[price_column] / df[price_column].shift(1))
        return df
    
    def calculate_volatility(self, df: pd.DataFrame, window: int = 20, 
                           price_column: str = 'Close') -> pd.DataFrame:
        """Calculate rolling volatility."""
        df = df.copy()
        if 'Daily_Return' not in df.columns:
            df['Daily_Return'] = df[price_column].pct_change()
        
        df['Volatility'] = df['Daily_Return'].rolling(window=window).std()
        return df
    
    def remove_outliers(self, df: pd.DataFrame, columns: list = None, 
                       method: str = 'iqr', factor: float = 1.5) -> pd.DataFrame:
        """Remove outliers from DataFrame."""
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for column in columns:
            if method == 'iqr':
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
            
            elif method == 'zscore':
                z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
                df = df[z_scores < factor]
        
        return df
    
    def log_processing_stats(self, original_df: pd.DataFrame, processed_df: pd.DataFrame, 
                           operation: str) -> None:
        """Log processing statistics."""
        original_rows = len(original_df)
        processed_rows = len(processed_df)
        rows_changed = original_rows - processed_rows
        
        self.logger.info(f"{operation}: {original_rows} -> {processed_rows} rows "
                        f"({rows_changed} removed)")
        
        if processed_rows == 0:
            self.logger.warning(f"{operation}: All data was removed!")
        elif rows_changed > original_rows * 0.5:
            self.logger.warning(f"{operation}: Removed more than 50% of data")