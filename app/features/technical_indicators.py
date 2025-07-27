"""
Advanced Technical Indicators Module

This module implements sophisticated technical analysis indicators beyond basic RSI and MACD,
including advanced momentum, volatility, and trend indicators.
"""

import numpy as np
import pandas as pd
import logging
from typing import Union, Tuple
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AdvancedTechnicalIndicators:
    """
    Advanced technical indicators for comprehensive market analysis
    """
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Williams %R - Momentum oscillator measuring overbought/oversold levels
        """
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100
        return williams_r.fillna(0)
    
    @staticmethod
    def stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, 
                            k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        Stochastic Oscillator - %K and %D lines for momentum analysis
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = ((close - lowest_low) / (highest_high - lowest_low)) * 100
        k_percent = k_percent.fillna(50)  # Neutral value
        
        d_percent = k_percent.rolling(window=d_period).mean().fillna(50)
        
        return k_percent, d_percent
    
    @staticmethod
    def bollinger_bands(close: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Enhanced Bollinger Bands with additional metrics
        """
        sma = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        # Fill NaN values
        sma = sma.fillna(close)
        upper_band = upper_band.fillna(close)
        lower_band = lower_band.fillna(close)
        
        return upper_band, sma, lower_band
    
    @staticmethod
    def bollinger_bandwidth(close: pd.Series, period: int = 20, std_dev: float = 2) -> pd.Series:
        """
        Bollinger Band Width - Measure of volatility
        """
        upper, middle, lower = AdvancedTechnicalIndicators.bollinger_bands(close, period, std_dev)
        bandwidth = ((upper - lower) / middle) * 100
        return bandwidth.fillna(0)
    
    @staticmethod
    def bollinger_percent_b(close: pd.Series, period: int = 20, std_dev: float = 2) -> pd.Series:
        """
        Bollinger %B - Position within Bollinger Bands
        """
        upper, middle, lower = AdvancedTechnicalIndicators.bollinger_bands(close, period, std_dev)
        
        # Avoid division by zero
        band_width = upper - lower
        percent_b = np.where(band_width != 0, (close - lower) / band_width, 0.5)
        
        return pd.Series(percent_b, index=close.index).fillna(0.5)
    
    @staticmethod
    def commodity_channel_index(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """
        Commodity Channel Index (CCI) - Momentum oscillator
        """
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mean_deviation = typical_price.rolling(window=period).apply(
            lambda x: np.mean(np.abs(x - x.mean())), raw=True
        )
        
        # Avoid division by zero
        cci = np.where(mean_deviation != 0, (typical_price - sma_tp) / (0.015 * mean_deviation), 0)
        return pd.Series(cci, index=close.index).fillna(0)
    
    @staticmethod
    def average_true_range(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Average True Range (ATR) - Volatility indicator
        """
        high_low = high - low
        high_close_prev = np.abs(high - close.shift(1))
        low_close_prev = np.abs(low - close.shift(1))
        
        true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
        atr = pd.Series(true_range, index=close.index).rolling(window=period).mean()
        
        return atr.fillna(0)
    
    @staticmethod
    def parabolic_sar(high: pd.Series, low: pd.Series, close: pd.Series, 
                     af_start: float = 0.02, af_max: float = 0.2) -> pd.Series:
        """
        Parabolic SAR - Trend-following indicator
        """
        length = len(close)
        sar = np.zeros(length)
        trend = np.zeros(length)  # 1 for uptrend, -1 for downtrend
        af = np.zeros(length)
        ep = np.zeros(length)  # Extreme point
        
        # Initialize first values
        sar[0] = low.iloc[0]
        trend[0] = 1
        af[0] = af_start
        ep[0] = high.iloc[0]
        
        for i in range(1, length):
            # Calculate SAR
            sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
            
            # Determine trend
            if trend[i-1] == 1:  # Uptrend
                if low.iloc[i] <= sar[i]:  # Trend reversal
                    trend[i] = -1
                    sar[i] = ep[i-1]
                    af[i] = af_start
                    ep[i] = low.iloc[i]
                else:  # Continue uptrend
                    trend[i] = 1
                    if high.iloc[i] > ep[i-1]:
                        ep[i] = high.iloc[i]
                        af[i] = min(af[i-1] + af_start, af_max)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
                        
                    # Ensure SAR doesn't exceed recent lows
                    sar[i] = min(sar[i], min(low.iloc[i-1], low.iloc[i-2] if i > 1 else low.iloc[i-1]))
                    
            else:  # Downtrend
                if high.iloc[i] >= sar[i]:  # Trend reversal
                    trend[i] = 1
                    sar[i] = ep[i-1]
                    af[i] = af_start
                    ep[i] = high.iloc[i]
                else:  # Continue downtrend
                    trend[i] = -1
                    if low.iloc[i] < ep[i-1]:
                        ep[i] = low.iloc[i]
                        af[i] = min(af[i-1] + af_start, af_max)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
                        
                    # Ensure SAR doesn't exceed recent highs
                    sar[i] = max(sar[i], max(high.iloc[i-1], high.iloc[i-2] if i > 1 else high.iloc[i-1]))
        
        return pd.Series(sar, index=close.index)
    
    @staticmethod
    def money_flow_index(high: pd.Series, low: pd.Series, close: pd.Series, 
                        volume: pd.Series, period: int = 14) -> pd.Series:
        """
        Money Flow Index (MFI) - Volume-weighted RSI
        """
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        # Positive and negative money flow
        positive_flow = pd.Series(0.0, index=close.index)
        negative_flow = pd.Series(0.0, index=close.index)
        
        for i in range(1, len(typical_price)):
            if typical_price.iloc[i] > typical_price.iloc[i-1]:
                positive_flow.iloc[i] = money_flow.iloc[i]
            elif typical_price.iloc[i] < typical_price.iloc[i-1]:
                negative_flow.iloc[i] = money_flow.iloc[i]
        
        # Rolling sums
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        # Money Flow Index calculation
        money_ratio = np.where(negative_mf != 0, positive_mf / negative_mf, 1)
        mfi = 100 - (100 / (1 + money_ratio))
        
        return pd.Series(mfi, index=close.index).fillna(50)
    
    @staticmethod
    def on_balance_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        On Balance Volume (OBV) - Volume-based momentum indicator
        """
        obv = np.zeros(len(close))
        obv[0] = volume.iloc[0]
        
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv[i] = obv[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv[i] = obv[i-1] - volume.iloc[i]
            else:
                obv[i] = obv[i-1]
        
        return pd.Series(obv, index=close.index)
    
    @staticmethod
    def accumulation_distribution_line(high: pd.Series, low: pd.Series, 
                                     close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Accumulation/Distribution Line - Volume flow indicator
        """
        clv = np.where(high != low, ((close - low) - (high - close)) / (high - low), 0)
        ad_line = (clv * volume).cumsum()
        
        return pd.Series(ad_line, index=close.index)
    
    @staticmethod
    def chaikin_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, 
                          volume: pd.Series, fast_period: int = 3, slow_period: int = 10) -> pd.Series:
        """
        Chaikin Oscillator - A/D Line momentum
        """
        ad_line = AdvancedTechnicalIndicators.accumulation_distribution_line(high, low, close, volume)
        
        fast_ema = ad_line.ewm(span=fast_period).mean()
        slow_ema = ad_line.ewm(span=slow_period).mean()
        
        chaikin_osc = fast_ema - slow_ema
        return chaikin_osc.fillna(0)
    
    @staticmethod
    def vortex_indicator(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Tuple[pd.Series, pd.Series]:
        """
        Vortex Indicator - Trend strength and direction
        """
        tr = AdvancedTechnicalIndicators.average_true_range(high, low, close, 1)
        
        vm_plus = np.abs(high - low.shift(1))
        vm_minus = np.abs(low - high.shift(1))
        
        vm_plus_sum = vm_plus.rolling(window=period).sum()
        vm_minus_sum = vm_minus.rolling(window=period).sum()
        tr_sum = tr.rolling(window=period).sum()
        
        vi_plus = np.where(tr_sum != 0, vm_plus_sum / tr_sum, 1)
        vi_minus = np.where(tr_sum != 0, vm_minus_sum / tr_sum, 1)
        
        return pd.Series(vi_plus, index=close.index).fillna(1), pd.Series(vi_minus, index=close.index).fillna(1)
    
    @staticmethod
    def keltner_channels(high: pd.Series, low: pd.Series, close: pd.Series, 
                        period: int = 20, multiplier: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Keltner Channels - Volatility-based envelope
        """
        ema = close.ewm(span=period).mean()
        atr = AdvancedTechnicalIndicators.average_true_range(high, low, close, period)
        
        upper_channel = ema + (multiplier * atr)
        lower_channel = ema - (multiplier * atr)
        
        return upper_channel.fillna(close), ema.fillna(close), lower_channel.fillna(close)
    
    @staticmethod
    def donchian_channels(high: pd.Series, low: pd.Series, period: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Donchian Channels - Breakout system
        """
        upper_channel = high.rolling(window=period).max()
        lower_channel = low.rolling(window=period).min()
        middle_channel = (upper_channel + lower_channel) / 2
        
        return (upper_channel.fillna(high), 
                middle_channel.fillna((high + low) / 2), 
                lower_channel.fillna(low))
    
    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all advanced technical indicators for a given dataframe
        """
        logger.info("Calculating advanced technical indicators...")
        
        # Make a copy to avoid modifying original data
        result_df = df.copy()
        
        try:
            high, low, close, volume = df['High'], df['Low'], df['Close'], df['Volume']
            
            # Momentum Indicators
            result_df['Williams_R'] = AdvancedTechnicalIndicators.williams_r(high, low, close)
            
            stoch_k, stoch_d = AdvancedTechnicalIndicators.stochastic_oscillator(high, low, close)
            result_df['Stoch_K'] = stoch_k
            result_df['Stoch_D'] = stoch_d
            
            result_df['CCI'] = AdvancedTechnicalIndicators.commodity_channel_index(high, low, close)
            result_df['MFI'] = AdvancedTechnicalIndicators.money_flow_index(high, low, close, volume)
            
            # Volatility Indicators
            result_df['ATR'] = AdvancedTechnicalIndicators.average_true_range(high, low, close)
            
            bb_upper, bb_middle, bb_lower = AdvancedTechnicalIndicators.bollinger_bands(close)
            result_df['BB_Upper'] = bb_upper
            result_df['BB_Middle'] = bb_middle
            result_df['BB_Lower'] = bb_lower
            result_df['BB_Width'] = AdvancedTechnicalIndicators.bollinger_bandwidth(close)
            result_df['BB_PercentB'] = AdvancedTechnicalIndicators.bollinger_percent_b(close)
            
            kelt_upper, kelt_middle, kelt_lower = AdvancedTechnicalIndicators.keltner_channels(high, low, close)
            result_df['Keltner_Upper'] = kelt_upper
            result_df['Keltner_Middle'] = kelt_middle
            result_df['Keltner_Lower'] = kelt_lower
            
            don_upper, don_middle, don_lower = AdvancedTechnicalIndicators.donchian_channels(high, low)
            result_df['Donchian_Upper'] = don_upper
            result_df['Donchian_Middle'] = don_middle
            result_df['Donchian_Lower'] = don_lower
            
            # Trend Indicators
            result_df['Parabolic_SAR'] = AdvancedTechnicalIndicators.parabolic_sar(high, low, close)
            
            vi_plus, vi_minus = AdvancedTechnicalIndicators.vortex_indicator(high, low, close)
            result_df['Vortex_Plus'] = vi_plus
            result_df['Vortex_Minus'] = vi_minus
            
            # Volume Indicators
            result_df['OBV'] = AdvancedTechnicalIndicators.on_balance_volume(close, volume)
            result_df['AD_Line'] = AdvancedTechnicalIndicators.accumulation_distribution_line(high, low, close, volume)
            result_df['Chaikin_Osc'] = AdvancedTechnicalIndicators.chaikin_oscillator(high, low, close, volume)
            
            # Additional derived features
            result_df['Price_vs_BB_Upper'] = (close - bb_upper) / bb_upper
            result_df['Price_vs_BB_Lower'] = (close - bb_lower) / bb_lower
            result_df['SAR_Signal'] = np.where(close > result_df['Parabolic_SAR'], 1, -1)
            result_df['Stoch_Signal'] = np.where((stoch_k > 80) | (stoch_d > 80), -1, 
                                               np.where((stoch_k < 20) | (stoch_d < 20), 1, 0))
            
            # Replace any infinite values with 0
            numeric_columns = result_df.select_dtypes(include=[np.number]).columns
            result_df[numeric_columns] = result_df[numeric_columns].replace([np.inf, -np.inf], 0)
            
            # Fill any remaining NaN values
            result_df = result_df.fillna(0)
            
            logger.info(f"Successfully calculated {len(numeric_columns) - len(df.columns)} additional indicators")
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
            raise
            
        return result_df