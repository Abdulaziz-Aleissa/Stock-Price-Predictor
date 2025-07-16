"""Technical indicators calculator."""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any

from ...core.constants import TECHNICAL_INDICATORS


logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Class for calculating technical indicators."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize with configuration."""
        self.config = config or TECHNICAL_INDICATORS
    
    def calculate_sma(self, df: pd.DataFrame, window: int, price_column: str = 'Close') -> pd.Series:
        """Calculate Simple Moving Average."""
        return df[price_column].rolling(window=window).mean()
    
    def calculate_ema(self, df: pd.DataFrame, window: int, price_column: str = 'Close') -> pd.Series:
        """Calculate Exponential Moving Average."""
        return df[price_column].ewm(span=window, adjust=False).mean()
    
    def calculate_rsi(self, df: pd.DataFrame, window: int = None, price_column: str = 'Close') -> pd.Series:
        """Calculate Relative Strength Index."""
        window = window or self.config['RSI_PERIOD']
        
        delta = df[price_column].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50)  # Neutral RSI for NaN values
    
    def calculate_macd(self, df: pd.DataFrame, fast_period: int = None, 
                      slow_period: int = None, signal_period: int = None,
                      price_column: str = 'Close') -> Dict[str, pd.Series]:
        """Calculate MACD and Signal line."""
        fast_period = fast_period or self.config['MACD_FAST_PERIOD']
        slow_period = slow_period or self.config['MACD_SLOW_PERIOD']
        signal_period = signal_period or self.config['MACD_SIGNAL_PERIOD']
        
        exp1 = df[price_column].ewm(span=fast_period, adjust=False).mean()
        exp2 = df[price_column].ewm(span=slow_period, adjust=False).mean()
        
        macd = exp1 - exp2
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        histogram = macd - signal
        
        return {
            'MACD': macd.fillna(0),
            'MACD_Signal': signal.fillna(0),
            'MACD_Histogram': histogram.fillna(0)
        }
    
    def calculate_bollinger_bands(self, df: pd.DataFrame, window: int = None, 
                                 std_factor: float = None, price_column: str = 'Close') -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands."""
        window = window or self.config['BOLLINGER_PERIOD']
        std_factor = std_factor or self.config['BOLLINGER_STD']
        
        sma = df[price_column].rolling(window=window).mean()
        std = df[price_column].rolling(window=window).std()
        
        upper_band = sma + (std * std_factor)
        lower_band = sma - (std * std_factor)
        
        return {
            'BB_Upper': upper_band.fillna(method='bfill').fillna(df[price_column]),
            'BB_Middle': sma.fillna(method='bfill').fillna(df[price_column]),
            'BB_Lower': lower_band.fillna(method='bfill').fillna(df[price_column])
        }
    
    def calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, 
                           d_period: int = 3) -> Dict[str, pd.Series]:
        """Calculate Stochastic Oscillator."""
        lowest_low = df['Low'].rolling(window=k_period).min()
        highest_high = df['High'].rolling(window=k_period).max()
        
        k_percent = 100 * ((df['Close'] - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return {
            'Stoch_K': k_percent.fillna(50),
            'Stoch_D': d_percent.fillna(50)
        }
    
    def calculate_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R."""
        highest_high = df['High'].rolling(window=period).max()
        lowest_low = df['Low'].rolling(window=period).min()
        
        williams_r = -100 * ((highest_high - df['Close']) / (highest_high - lowest_low))
        
        return williams_r.fillna(-50)
    
    def calculate_cci(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index."""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mean_deviation = typical_price.rolling(window=period).apply(
            lambda x: np.mean(np.abs(x - np.mean(x)))
        )
        
        cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
        
        return cci.fillna(0)
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = df['High'] - df['Low']
        high_close_prev = np.abs(df['High'] - df['Close'].shift())
        low_close_prev = np.abs(df['Low'] - df['Close'].shift())
        
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr.fillna(0)
    
    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> Dict[str, pd.Series]:
        """Calculate Average Directional Index and directional indicators."""
        high_diff = df['High'].diff()
        low_diff = df['Low'].diff()
        
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        
        atr = self.calculate_atr(df, period)
        
        plus_di = 100 * (pd.Series(plus_dm).rolling(window=period).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm).rolling(window=period).mean() / atr)
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return {
            'ADX': adx.fillna(0),
            'Plus_DI': plus_di.fillna(0),
            'Minus_DI': minus_di.fillna(0)
        }
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators."""
        result_df = df.copy()
        
        try:
            # Moving Averages
            result_df['SMA_20'] = self.calculate_sma(df, self.config['SMA_SHORT_PERIOD'])
            result_df['SMA_50'] = self.calculate_sma(df, self.config['SMA_LONG_PERIOD'])
            result_df['EMA_12'] = self.calculate_ema(df, 12)
            result_df['EMA_26'] = self.calculate_ema(df, 26)
            
            # RSI
            result_df['RSI'] = self.calculate_rsi(df)
            
            # MACD
            macd_data = self.calculate_macd(df)
            for key, series in macd_data.items():
                result_df[key] = series
            
            # Bollinger Bands
            bb_data = self.calculate_bollinger_bands(df)
            for key, series in bb_data.items():
                result_df[key] = series
            
            # Stochastic
            stoch_data = self.calculate_stochastic(df)
            for key, series in stoch_data.items():
                result_df[key] = series
            
            # Other indicators
            result_df['Williams_R'] = self.calculate_williams_r(df)
            result_df['CCI'] = self.calculate_cci(df)
            result_df['ATR'] = self.calculate_atr(df)
            
            # ADX
            adx_data = self.calculate_adx(df)
            for key, series in adx_data.items():
                result_df[key] = series
            
            logger.info(f"Calculated {len(result_df.columns) - len(df.columns)} technical indicators")
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
            raise
        
        return result_df