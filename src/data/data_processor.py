"""
Enhanced Data Processing Module
Handles data preprocessing, feature engineering, and technical indicators
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Import configuration and helpers
from ..utils.config import config
from ..utils.helpers import DataValidator, MathUtils, performance_tracker

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """Calculate various technical indicators"""
    
    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = MathUtils.safe_divide(gain, loss, default=1.0)
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.fillna(50)  # Neutral RSI for NaN values
            
        except Exception as e:
            logger.error(f"Error calculating RSI: {str(e)}")
            return pd.Series(index=prices.index, data=50)  # Return neutral RSI
    
    @staticmethod
    def macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean()
            histogram = macd_line - signal_line
            
            return {
                'MACD': macd_line.fillna(0),
                'MACD_Signal': signal_line.fillna(0),
                'MACD_Histogram': histogram.fillna(0)
            }
            
        except Exception as e:
            logger.error(f"Error calculating MACD: {str(e)}")
            return {
                'MACD': pd.Series(index=prices.index, data=0),
                'MACD_Signal': pd.Series(index=prices.index, data=0),
                'MACD_Histogram': pd.Series(index=prices.index, data=0)
            }
    
    @staticmethod
    def bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        try:
            sma = prices.rolling(window=period).mean()
            rolling_std = prices.rolling(window=period).std()
            
            upper_band = sma + (rolling_std * std_dev)
            lower_band = sma - (rolling_std * std_dev)
            
            return {
                'BB_Upper': upper_band.fillna(method='bfill').fillna(method='ffill'),
                'BB_Middle': sma.fillna(method='bfill').fillna(method='ffill'),
                'BB_Lower': lower_band.fillna(method='bfill').fillna(method='ffill'),
                'BB_Width': (upper_band - lower_band).fillna(0),
                'BB_Position': MathUtils.safe_divide(prices - lower_band, upper_band - lower_band, default=0.5)
            }
            
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {str(e)}")
            return {
                'BB_Upper': prices.copy(),
                'BB_Middle': prices.copy(),
                'BB_Lower': prices.copy(),
                'BB_Width': pd.Series(index=prices.index, data=0),
                'BB_Position': pd.Series(index=prices.index, data=0.5)
            }
    
    @staticmethod
    def moving_averages(prices: pd.Series, periods: List[int] = [20, 50, 200]) -> Dict[str, pd.Series]:
        """Calculate Simple and Exponential Moving Averages"""
        try:
            result = {}
            
            for period in periods:
                # Simple Moving Average
                sma = prices.rolling(window=period).mean()
                result[f'SMA_{period}'] = sma.fillna(method='bfill').fillna(prices)
                
                # Exponential Moving Average
                ema = prices.ewm(span=period).mean()
                result[f'EMA_{period}'] = ema.fillna(method='bfill').fillna(prices)
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating moving averages: {str(e)}")
            return {}
    
    @staticmethod
    def volume_indicators(df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate volume-based indicators"""
        try:
            volume = df['Volume']
            close = df['Close']
            high = df['High']
            low = df['Low']
            
            # Volume Moving Average
            volume_sma_20 = volume.rolling(window=20).mean()
            
            # Volume Rate of Change
            volume_roc = volume.pct_change(periods=1).fillna(0)
            
            # On-Balance Volume (OBV)
            price_change = close.diff()
            obv = np.where(price_change > 0, volume, 
                          np.where(price_change < 0, -volume, 0)).cumsum()
            
            # Volume Weighted Average Price (VWAP)
            typical_price = (high + low + close) / 3
            vwap = (typical_price * volume).cumsum() / volume.cumsum()
            
            # Money Flow Index (MFI)
            typical_price_change = typical_price.diff()
            money_flow = typical_price * volume
            positive_flow = np.where(typical_price_change > 0, money_flow, 0)
            negative_flow = np.where(typical_price_change < 0, money_flow, 0)
            
            mfr = MathUtils.safe_divide(
                pd.Series(positive_flow).rolling(14).sum(),
                pd.Series(negative_flow).rolling(14).sum(),
                default=1.0
            )
            mfi = 100 - (100 / (1 + mfr))
            
            return {
                'Volume_SMA_20': volume_sma_20.fillna(volume.mean()),
                'Volume_ROC': pd.Series(volume_roc),
                'OBV': pd.Series(obv, index=df.index),
                'VWAP': pd.Series(vwap).fillna(method='bfill').fillna(close),
                'MFI': pd.Series(mfi).fillna(50),
                'Volume_Ratio': MathUtils.safe_divide(volume, volume_sma_20, default=1.0)
            }
            
        except Exception as e:
            logger.error(f"Error calculating volume indicators: {str(e)}")
            return {}
    
    @staticmethod
    def momentum_indicators(df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate momentum indicators"""
        try:
            close = df['Close']
            high = df['High']
            low = df['Low']
            
            # Rate of Change
            roc_1 = close.pct_change(periods=1).fillna(0)
            roc_5 = close.pct_change(periods=5).fillna(0)
            roc_10 = close.pct_change(periods=10).fillna(0)
            
            # Williams %R
            highest_high = high.rolling(14).max()
            lowest_low = low.rolling(14).min()
            williams_r = MathUtils.safe_divide(
                (highest_high - close) * -100,
                highest_high - lowest_low,
                default=-50
            )
            
            # Stochastic Oscillator
            stoch_k = MathUtils.safe_divide(
                (close - lowest_low) * 100,
                highest_high - lowest_low,
                default=50
            )
            stoch_d = stoch_k.rolling(3).mean()
            
            # Commodity Channel Index (CCI)
            typical_price = (high + low + close) / 3
            sma_tp = typical_price.rolling(20).mean()
            mad = typical_price.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
            cci = MathUtils.safe_divide((typical_price - sma_tp), (0.015 * mad), default=0)
            
            return {
                'ROC_1': roc_1,
                'ROC_5': roc_5,
                'ROC_10': roc_10,
                'Williams_R': pd.Series(williams_r).fillna(-50),
                'Stoch_K': pd.Series(stoch_k).fillna(50),
                'Stoch_D': pd.Series(stoch_d).fillna(50),
                'CCI': pd.Series(cci).fillna(0)
            }
            
        except Exception as e:
            logger.error(f"Error calculating momentum indicators: {str(e)}")
            return {}

class DataProcessor:
    """Main data processing class"""
    
    def __init__(self):
        self.technical_indicators = TechnicalIndicators()
        self.scalers = {}
    
    def process_stock_data(self, df: pd.DataFrame, symbol: str, 
                          add_technical_indicators: bool = True,
                          add_volume_analysis: bool = True,
                          add_momentum_indicators: bool = True) -> pd.DataFrame:
        """
        Process stock data with feature engineering
        
        Args:
            df: Raw OHLCV data
            symbol: Stock symbol
            add_technical_indicators: Add technical indicators
            add_volume_analysis: Add volume indicators
            add_momentum_indicators: Add momentum indicators
        
        Returns:
            Processed DataFrame with features
        """
        try:
            performance_tracker.start_timer(f"process_data_{symbol}")
            
            if df.empty:
                return df
            
            # Make a copy to avoid modifying original data
            processed_df = df.copy()
            
            # Basic price features
            processed_df = self._add_basic_features(processed_df)
            
            # Technical indicators
            if add_technical_indicators:
                processed_df = self._add_technical_indicators(processed_df)
            
            # Volume analysis
            if add_volume_analysis:
                processed_df = self._add_volume_indicators(processed_df)
            
            # Momentum indicators
            if add_momentum_indicators:
                processed_df = self._add_momentum_indicators(processed_df)
            
            # Market microstructure features
            processed_df = self._add_microstructure_features(processed_df)
            
            # Time-based features
            processed_df = self._add_time_features(processed_df)
            
            # Clean and validate final data
            processed_df = self._final_cleanup(processed_df)
            
            performance_tracker.end_timer(f"process_data_{symbol}")
            
            logger.info(f"Data processing completed for {symbol}: {len(processed_df)} rows, {len(processed_df.columns)} features")
            
            return processed_df
            
        except Exception as e:
            logger.error(f"Error processing data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic price and return features"""
        try:
            # Price changes and returns
            df['Price_Change'] = df['Close'].diff().fillna(0)
            df['Price_Change_Pct'] = df['Close'].pct_change().fillna(0)
            df['Log_Return'] = MathUtils.calculate_returns(df['Close'], method='log')
            
            # High-Low spread
            df['HL_Spread'] = df['High'] - df['Low']
            df['HL_Spread_Pct'] = MathUtils.safe_divide(df['HL_Spread'], df['Close'], default=0)
            
            # Open-Close relationship
            df['OC_Spread'] = df['Close'] - df['Open']
            df['OC_Spread_Pct'] = MathUtils.safe_divide(df['OC_Spread'], df['Open'], default=0)
            
            # Volatility measures
            df['Volatility_20'] = MathUtils.calculate_volatility(df['Price_Change_Pct'], window=20, annualize=False)
            df['Volatility_5'] = MathUtils.calculate_volatility(df['Price_Change_Pct'], window=5, annualize=False)
            
            # Price position within day's range
            df['Price_Position'] = MathUtils.safe_divide(
                df['Close'] - df['Low'],
                df['High'] - df['Low'],
                default=0.5
            )
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding basic features: {str(e)}")
            return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to DataFrame"""
        try:
            close_prices = df['Close']
            
            # RSI
            df['RSI'] = self.technical_indicators.rsi(close_prices, config.TECHNICAL_INDICATORS['rsi_period'])
            
            # MACD
            macd_data = self.technical_indicators.macd(
                close_prices,
                config.TECHNICAL_INDICATORS['macd_fast'],
                config.TECHNICAL_INDICATORS['macd_slow'],
                config.TECHNICAL_INDICATORS['macd_signal']
            )
            for key, value in macd_data.items():
                df[key] = value
            
            # Bollinger Bands
            bb_data = self.technical_indicators.bollinger_bands(
                close_prices,
                config.TECHNICAL_INDICATORS['bb_period'],
                config.TECHNICAL_INDICATORS['bb_std']
            )
            for key, value in bb_data.items():
                df[key] = value
            
            # Moving Averages
            ma_data = self.technical_indicators.moving_averages(
                close_prices,
                config.TECHNICAL_INDICATORS['sma_periods']
            )
            for key, value in ma_data.items():
                df[key] = value
            
            # Add crossover signals
            if 'SMA_20' in df.columns and 'SMA_50' in df.columns:
                df['SMA_20_50_Cross'] = np.where(df['SMA_20'] > df['SMA_50'], 1, -1)
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {str(e)}")
            return df
    
    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume indicators to DataFrame"""
        try:
            volume_data = self.technical_indicators.volume_indicators(df)
            for key, value in volume_data.items():
                df[key] = value
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding volume indicators: {str(e)}")
            return df
    
    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators to DataFrame"""
        try:
            momentum_data = self.technical_indicators.momentum_indicators(df)
            for key, value in momentum_data.items():
                df[key] = value
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding momentum indicators: {str(e)}")
            return df
    
    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features"""
        try:
            # Gap analysis
            df['Gap'] = df['Open'] - df['Close'].shift(1)
            df['Gap_Pct'] = MathUtils.safe_divide(df['Gap'], df['Close'].shift(1), default=0)
            
            # Intraday patterns
            df['Intraday_Return'] = MathUtils.safe_divide(df['Close'] - df['Open'], df['Open'], default=0)
            df['Upper_Shadow'] = df['High'] - np.maximum(df['Open'], df['Close'])
            df['Lower_Shadow'] = np.minimum(df['Open'], df['Close']) - df['Low']
            
            # Normalize shadows by price
            df['Upper_Shadow_Pct'] = MathUtils.safe_divide(df['Upper_Shadow'], df['Close'], default=0)
            df['Lower_Shadow_Pct'] = MathUtils.safe_divide(df['Lower_Shadow'], df['Close'], default=0)
            
            # Body size (candle body)
            df['Body_Size'] = np.abs(df['Close'] - df['Open'])
            df['Body_Size_Pct'] = MathUtils.safe_divide(df['Body_Size'], df['Close'], default=0)
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding microstructure features: {str(e)}")
            return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        try:
            # Day of week (0=Monday, 6=Sunday)
            df['DayOfWeek'] = df.index.dayofweek
            
            # Month
            df['Month'] = df.index.month
            
            # Quarter
            df['Quarter'] = df.index.quarter
            
            # Day of year
            df['DayOfYear'] = df.index.dayofyear
            
            # Is beginning/end of month
            df['IsMonthStart'] = df.index.is_month_start.astype(int)
            df['IsMonthEnd'] = df.index.is_month_end.astype(int)
            
            # Is beginning/end of quarter
            df['IsQuarterStart'] = df.index.is_quarter_start.astype(int)
            df['IsQuarterEnd'] = df.index.is_quarter_end.astype(int)
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding time features: {str(e)}")
            return df
    
    def _final_cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final data cleanup and validation"""
        try:
            # Replace infinite values
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], np.nan)
            
            # Fill NaN values
            df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Remove any columns with all NaN or constant values
            df = df.loc[:, df.var() != 0]
            
            # Cap extreme values
            for col in numeric_columns:
                if col in df.columns:
                    q99 = df[col].quantile(0.99)
                    q01 = df[col].quantile(0.01)
                    df[col] = df[col].clip(lower=q01, upper=q99)
            
            return df
            
        except Exception as e:
            logger.error(f"Error in final cleanup: {str(e)}")
            return df
    
    def prepare_features_for_ml(self, df: pd.DataFrame, target_col: str = 'Close',
                               prediction_horizon: int = 1, 
                               feature_scaling: str = 'standard') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for machine learning
        
        Args:
            df: Processed DataFrame
            target_col: Target column name
            prediction_horizon: Days ahead to predict
            feature_scaling: Type of scaling ('standard', 'minmax', 'robust', 'none')
        
        Returns:
            Tuple of (features_df, target_series)
        """
        try:
            if df.empty:
                return pd.DataFrame(), pd.Series()
            
            # Create target variable (future price)
            target = df[target_col].shift(-prediction_horizon)
            
            # Remove rows with NaN targets
            valid_indices = ~target.isna()
            features_df = df[valid_indices].copy()
            target_series = target[valid_indices].copy()
            
            # Remove target column from features if it exists
            if target_col in features_df.columns:
                features_df = features_df.drop(columns=[target_col])
            
            # Remove OHLCV columns as they are raw data
            ohlcv_cols = ['Open', 'High', 'Low', 'Volume']
            features_df = features_df.drop(columns=[col for col in ohlcv_cols if col in features_df.columns])
            
            # Apply feature scaling
            if feature_scaling != 'none' and not features_df.empty:
                features_df = self._scale_features(features_df, method=feature_scaling)
            
            logger.info(f"Prepared ML features: {len(features_df)} samples, {len(features_df.columns)} features")
            
            return features_df, target_series
            
        except Exception as e:
            logger.error(f"Error preparing features for ML: {str(e)}")
            return pd.DataFrame(), pd.Series()
    
    def _scale_features(self, df: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """Scale features using specified method"""
        try:
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            elif method == 'robust':
                scaler = RobustScaler()
            else:
                return df
            
            # Store scaler for later use
            scaler_key = f"{method}_{hash(tuple(df.columns))}"
            
            if scaler_key not in self.scalers:
                scaled_data = scaler.fit_transform(df)
                self.scalers[scaler_key] = scaler
            else:
                scaled_data = self.scalers[scaler_key].transform(df)
            
            scaled_df = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)
            
            return scaled_df
            
        except Exception as e:
            logger.error(f"Error scaling features: {str(e)}")
            return df
    
    def get_feature_importance_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get data for feature importance analysis"""
        try:
            # Calculate basic statistics for each feature
            feature_stats = {}
            
            for col in df.select_dtypes(include=[np.number]).columns:
                feature_stats[col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'skewness': df[col].skew(),
                    'kurtosis': df[col].kurtosis(),
                    'missing_pct': (df[col].isna().sum() / len(df)) * 100
                }
            
            return {
                'feature_stats': feature_stats,
                'correlation_matrix': df.corr().to_dict(),
                'feature_count': len(df.columns),
                'sample_count': len(df)
            }
            
        except Exception as e:
            logger.error(f"Error getting feature importance data: {str(e)}")
            return {}

# Create global data processor instance
data_processor = DataProcessor()