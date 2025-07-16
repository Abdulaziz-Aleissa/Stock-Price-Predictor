"""Stock data processor for cleaning and preparing stock data."""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Optional

from .base_processor import BaseProcessor
from .technical_indicators import TechnicalIndicators
from ...core.exceptions import ValidationError


logger = logging.getLogger(__name__)


class StockProcessor(BaseProcessor):
    """Processor for stock data cleaning and feature engineering."""
    
    def __init__(self):
        """Initialize the stock processor."""
        super().__init__()
        self.technical_indicators = TechnicalIndicators()
    
    def process(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Main processing method."""
        return self.clean_data(df)
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare stock data."""
        try:
            # Validate required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            self.validate_dataframe(df, required_columns)
            
            # Ensure datetime index
            df = self.ensure_datetime_index(df)
            
            # Create a copy to avoid modifying original
            cleaned_df = df.copy()
            
            # Basic price and volume calculations
            cleaned_df = self._calculate_basic_features(cleaned_df)
            
            # Calculate technical indicators
            cleaned_df = self.technical_indicators.calculate_all_indicators(cleaned_df)
            
            # Create target variable (tomorrow's price)
            cleaned_df["Tomorrow"] = cleaned_df['Close'].shift(-1)
            
            # Add date column for compatibility
            cleaned_df['Date'] = cleaned_df.index
            
            # Remove the last row as it will have NaN in Tomorrow
            cleaned_df = cleaned_df.iloc[:-1].copy()
            
            # Handle any remaining NaN values
            cleaned_df = self.handle_missing_values(cleaned_df, strategy='forward_fill')
            
            self.log_processing_stats(df, cleaned_df, "Stock data cleaning")
            
            return cleaned_df
            
        except Exception as e:
            logger.error(f"Error in stock data cleaning: {str(e)}")
            raise ValidationError(f"Failed to clean stock data: {str(e)}")
    
    def process_for_training(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process data specifically for model training."""
        try:
            # Clean the data
            processed_df = self.clean_data(df)
            
            # Remove outliers for training
            processed_df = self.remove_outliers(
                processed_df, 
                columns=['Close', 'Volume'], 
                method='iqr', 
                factor=2.0
            )
            
            # Ensure we have enough data points
            min_points = 100
            if len(processed_df) < min_points:
                raise ValidationError(f"Insufficient data for training. Need at least {min_points} points, got {len(processed_df)}")
            
            logger.info(f"Processed {len(processed_df)} data points for training")
            return processed_df
            
        except Exception as e:
            logger.error(f"Error processing data for training: {str(e)}")
            raise
    
    def process_for_prediction(self, df: pd.DataFrame, current_price: Optional[float] = None) -> pd.DataFrame:
        """Process data for making predictions."""
        try:
            # Clean the data
            processed_df = self.clean_data(df)
            
            # Update the latest price if provided
            if current_price is not None:
                latest_idx = processed_df.index[-1]
                processed_df.loc[latest_idx, 'Close'] = current_price
                
                # Recalculate features that depend on the close price
                processed_df = self._recalculate_price_dependent_features(processed_df)
            
            logger.info(f"Processed {len(processed_df)} data points for prediction")
            return processed_df
            
        except Exception as e:
            logger.error(f"Error processing data for prediction: {str(e)}")
            raise
    
    def _calculate_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic price and volume features."""
        df = df.copy()
        
        # Price change features
        df['Price_Change'] = df['Close'].pct_change().fillna(0)
        df['Volume_Change'] = df['Volume'].pct_change().fillna(0)
        df['High_Low_Range'] = df['High'] - df['Low']
        
        # Daily return and volatility
        df = self.calculate_returns(df)
        df = self.calculate_volatility(df, window=20)
        
        # Price position within daily range
        df['Close_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        df['Close_Position'] = df['Close_Position'].fillna(0.5)
        
        # Volume relative to average
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        df['Volume_Ratio'] = df['Volume_Ratio'].fillna(1.0)
        
        return df
    
    def _recalculate_price_dependent_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Recalculate features that depend on the close price."""
        df = df.copy()
        
        # Recalculate price change for the last row
        if len(df) > 1:
            df.iloc[-1, df.columns.get_loc('Price_Change')] = (
                df['Close'].iloc[-1] - df['Close'].iloc[-2]
            ) / df['Close'].iloc[-2]
        
        # Recalculate close position
        last_idx = len(df) - 1
        high_val = df['High'].iloc[last_idx]
        low_val = df['Low'].iloc[last_idx]
        close_val = df['Close'].iloc[last_idx]
        
        if high_val != low_val:
            df.iloc[last_idx, df.columns.get_loc('Close_Position')] = (
                (close_val - low_val) / (high_val - low_val)
            )
        
        # Recalculate moving averages (only last few values affected)
        window = 20
        if len(df) >= window:
            start_idx = max(0, len(df) - window - 1)
            
            # SMA
            df.loc[df.index[start_idx:], 'SMA_20'] = (
                df['Close'].rolling(window=20).mean().loc[df.index[start_idx:]]
            )
            df.loc[df.index[start_idx:], 'SMA_50'] = (
                df['Close'].rolling(window=50).mean().loc[df.index[start_idx:]]
            )
            
            # RSI (recalculate last values)
            df.loc[df.index[start_idx:], 'RSI'] = (
                self.technical_indicators.calculate_rsi(df).loc[df.index[start_idx:]]
            )
        
        return df
    
    def validate_stock_data(self, df: pd.DataFrame) -> bool:
        """Validate stock data quality."""
        try:
            # Check for required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return False
            
            # Check for negative prices
            price_columns = ['Open', 'High', 'Low', 'Close']
            for col in price_columns:
                if (df[col] <= 0).any():
                    logger.warning(f"Found non-positive values in {col}")
            
            # Check for negative volume
            if (df['Volume'] < 0).any():
                logger.warning("Found negative volume values")
            
            # Check for logical consistency (High >= Low, etc.)
            if (df['High'] < df['Low']).any():
                logger.error("Found rows where High < Low")
                return False
            
            if (df['High'] < df['Close']).any() or (df['Close'] < df['Low']).any():
                logger.warning("Found rows where Close is outside High-Low range")
            
            # Check data density
            if len(df) < 30:
                logger.warning("Dataset is very small (< 30 data points)")
            
            logger.info("Stock data validation completed")
            return True
            
        except Exception as e:
            logger.error(f"Error validating stock data: {str(e)}")
            return False