"""
Simplified Time Series Model (Alternative to Transformer)

Since deep learning libraries are not available, this module implements
a sophisticated time series model using traditional ML techniques that
capture temporal patterns effectively.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import logging

logger = logging.getLogger(__name__)

class TimeSeriesModel(BaseEstimator, RegressorMixin):
    """
    Advanced time series model using feature engineering to capture temporal patterns
    """
    
    def __init__(self, 
                 lookback_window: int = 30,
                 seasonal_periods: list = [5, 10, 20, 60],
                 trend_periods: list = [5, 10, 20],
                 volatility_periods: list = [5, 10, 20],
                 random_state: int = 42):
        """
        Initialize time series model
        
        Args:
            lookback_window: Number of historical periods to consider
            seasonal_periods: Periods for seasonal decomposition
            trend_periods: Periods for trend analysis
            volatility_periods: Periods for volatility features
        """
        self.lookback_window = lookback_window
        self.seasonal_periods = seasonal_periods
        self.trend_periods = trend_periods
        self.volatility_periods = volatility_periods
        self.random_state = random_state
        
        # Internal models
        self.trend_model = RandomForestRegressor(
            n_estimators=50, max_depth=8, random_state=random_state
        )
        self.seasonal_model = Ridge(alpha=1.0, random_state=random_state)
        self.residual_model = RandomForestRegressor(
            n_estimators=30, max_depth=6, random_state=random_state
        )
        
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def _create_temporal_features(self, X) -> pd.DataFrame:
        """Create temporal features from input data"""
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        if 'Close' not in X.columns:
            # Assume first column is the target price series
            price_col = X.columns[0]
        else:
            price_col = 'Close'
            
        prices = X[price_col]
        features = pd.DataFrame(index=X.index)
        
        # Lagged features
        for lag in range(1, min(self.lookback_window + 1, len(prices))):
            features[f'lag_{lag}'] = prices.shift(lag).fillna(method='bfill')
        
        # Rolling statistics for different periods
        for period in self.trend_periods:
            if period < len(prices):
                features[f'mean_{period}'] = prices.rolling(period).mean().fillna(method='bfill')
                features[f'std_{period}'] = prices.rolling(period).std().fillna(0)
                features[f'min_{period}'] = prices.rolling(period).min().fillna(method='bfill')
                features[f'max_{period}'] = prices.rolling(period).max().fillna(method='bfill')
        
        # Momentum features
        for period in [3, 7, 14]:
            if period < len(prices):
                features[f'momentum_{period}'] = (prices / prices.shift(period) - 1).fillna(0)
                features[f'roc_{period}'] = prices.pct_change(period).fillna(0)
        
        # Volatility features
        returns = prices.pct_change().fillna(0)
        for period in self.volatility_periods:
            if period < len(returns):
                features[f'volatility_{period}'] = returns.rolling(period).std().fillna(0)
                features[f'realized_vol_{period}'] = (returns ** 2).rolling(period).sum().fillna(0)
        
        # Trend strength
        for period in [10, 20]:
            if period < len(prices):
                trend_strength = (prices - prices.shift(period)) / prices.shift(period)
                features[f'trend_strength_{period}'] = trend_strength.fillna(0)
        
        # Seasonal features (if datetime index available)
        if hasattr(X.index, 'dayofweek'):
            features['day_of_week'] = X.index.dayofweek
            features['month'] = X.index.month
            features['quarter'] = X.index.quarter
        
        # Fourier features for seasonality
        for period in self.seasonal_periods:
            if period < len(prices):
                features[f'sin_{period}'] = np.sin(2 * np.pi * np.arange(len(prices)) / period)
                features[f'cos_{period}'] = np.cos(2 * np.pi * np.arange(len(prices)) / period)
        
        # Fill any remaining NaN values
        features = features.fillna(0)
        
        # Replace infinite values
        features = features.replace([np.inf, -np.inf], 0)
        
        return features
    
    def fit(self, X, y):
        """Fit the time series model"""
        logger.info("Training Time Series Model...")
        
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        
        # Create temporal features
        temporal_features = self._create_temporal_features(X)
        
        # Combine with original features
        all_features = pd.concat([X, temporal_features], axis=1)
        all_features = all_features.fillna(0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(all_features)
        
        # Decompose the target into trend, seasonal, and residual components
        y_series = pd.Series(y.values, index=X.index) if hasattr(X, 'index') else pd.Series(y.values)
        
        # Trend component (using rolling mean)
        trend_window = min(20, len(y_series) // 4)
        trend_component = y_series.rolling(trend_window, center=True).mean().fillna(method='bfill').fillna(method='ffill')
        
        # Seasonal component (simplified)
        detrended = y_series - trend_component
        seasonal_period = min(self.seasonal_periods[0], len(detrended) // 4) if self.seasonal_periods else 5
        seasonal_component = detrended.rolling(seasonal_period).mean().fillna(0)
        
        # Residual component
        residual_component = y_series - trend_component - seasonal_component
        
        # Train individual models
        try:
            self.trend_model.fit(X_scaled, trend_component)
            self.seasonal_model.fit(X_scaled, seasonal_component)
            self.residual_model.fit(X_scaled, residual_component)
        except Exception as e:
            logger.warning(f"Error training component models: {str(e)}")
            # Fallback to single model
            self.trend_model.fit(X_scaled, y_series)
        
        self.feature_names_ = all_features.columns.tolist()
        self.is_fitted = True
        
        return self
    
    def predict(self, X) -> np.ndarray:
        """Make predictions using the time series model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Create temporal features
        temporal_features = self._create_temporal_features(X)
        
        # Combine with original features
        all_features = pd.concat([X, temporal_features], axis=1)
        
        # Ensure all training features are present
        for feature in self.feature_names_:
            if feature not in all_features.columns:
                all_features[feature] = 0
        
        # Reorder columns to match training
        all_features = all_features[self.feature_names_]
        all_features = all_features.fillna(0)
        
        # Scale features
        X_scaled = self.scaler.transform(all_features)
        
        # Get predictions from component models
        try:
            trend_pred = self.trend_model.predict(X_scaled)
            seasonal_pred = self.seasonal_model.predict(X_scaled)
            residual_pred = self.residual_model.predict(X_scaled)
            
            # Combine predictions
            final_pred = trend_pred + seasonal_pred + residual_pred
        except Exception as e:
            logger.warning(f"Error in component prediction: {str(e)}")
            # Fallback to trend model only
            final_pred = self.trend_model.predict(X_scaled)
        
        return final_pred
    
    def get_feature_importance(self) -> dict:
        """Get feature importance from the model"""
        if not self.is_fitted:
            return {}
        
        try:
            # Get importance from tree-based models
            trend_importance = getattr(self.trend_model, 'feature_importances_', None)
            residual_importance = getattr(self.residual_model, 'feature_importances_', None)
            
            if trend_importance is not None and residual_importance is not None:
                # Weighted average of importances
                combined_importance = (trend_importance * 0.6 + residual_importance * 0.4)
                
                importance_dict = {
                    feature: importance 
                    for feature, importance in zip(self.feature_names_, combined_importance)
                }
                
                return importance_dict
            
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {str(e)}")
        
        return {}


class LSTMAlternativeModel(BaseEstimator, RegressorMixin):
    """
    Alternative to LSTM using sequential feature engineering and ensemble methods
    """
    
    def __init__(self, 
                 sequence_length: int = 20,
                 hidden_units: int = 50,
                 attention_heads: int = 4,
                 random_state: int = 42):
        """
        Initialize LSTM alternative model
        
        Args:
            sequence_length: Length of input sequences
            hidden_units: Number of hidden units (features to create)
            attention_heads: Number of attention-like mechanisms
        """
        self.sequence_length = sequence_length
        self.hidden_units = hidden_units
        self.attention_heads = attention_heads
        self.random_state = random_state
        
        # Component models to simulate LSTM layers
        self.sequence_models = []
        for i in range(3):  # 3 "layers"
            model = RandomForestRegressor(
                n_estimators=30,
                max_depth=6,
                random_state=random_state + i
            )
            self.sequence_models.append(model)
        
        # Attention-like mechanism using Ridge regression
        self.attention_models = []
        for i in range(attention_heads):
            model = Ridge(alpha=0.1, random_state=random_state + i)
            self.attention_models.append(model)
        
        # Final output layer
        self.output_model = RandomForestRegressor(
            n_estimators=50,
            max_depth=8,
            random_state=random_state
        )
        
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def _create_sequences(self, X) -> np.ndarray:
        """Create sequence-like features from time series data"""
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        if 'Close' not in X.columns:
            price_col = X.columns[0]
        else:
            price_col = 'Close'
        
        prices = X[price_col].values
        sequences = []
        
        # Create overlapping sequences
        for i in range(len(prices)):
            start_idx = max(0, i - self.sequence_length + 1)
            sequence = prices[start_idx:i+1]
            
            # Pad if necessary
            if len(sequence) < self.sequence_length:
                padding = [sequence[0]] * (self.sequence_length - len(sequence))
                sequence = padding + list(sequence)
            
            # Create sequence features
            seq_features = []
            
            # Basic sequence statistics
            seq_features.extend([
                np.mean(sequence),
                np.std(sequence),
                np.min(sequence),
                np.max(sequence),
                sequence[-1],  # Current value
                sequence[-1] - sequence[0] if len(sequence) > 1 else 0,  # Change
            ])
            
            # Momentum at different points in sequence
            for j in [1, 5, 10]:
                if j < len(sequence):
                    momentum = sequence[-1] - sequence[-j-1]
                    seq_features.append(momentum)
                else:
                    seq_features.append(0)
            
            # Trend strength
            if len(sequence) > 1:
                x = np.arange(len(sequence))
                trend = np.polyfit(x, sequence, 1)[0]
                seq_features.append(trend)
            else:
                seq_features.append(0)
            
            # Volatility measures
            if len(sequence) > 1:
                returns = np.diff(sequence) / sequence[:-1]
                seq_features.extend([
                    np.std(returns),
                    np.mean(np.abs(returns))
                ])
            else:
                seq_features.extend([0, 0])
            
            sequences.append(seq_features)
        
        return np.array(sequences)
    
    def _apply_attention_mechanism(self, X: np.ndarray) -> np.ndarray:
        """Apply attention-like mechanism using multiple models"""
        attention_outputs = []
        
        for model in self.attention_models:
            if hasattr(model, 'predict'):
                try:
                    # Use model to create attention weights
                    attention_scores = model.predict(X)
                    
                    # Normalize scores to create attention weights
                    attention_weights = np.exp(attention_scores) / np.sum(np.exp(attention_scores))
                    
                    # Apply attention to input features
                    attended_features = X * attention_weights.reshape(-1, 1)
                    attention_outputs.append(attended_features.mean(axis=1))
                    
                except Exception as e:
                    logger.warning(f"Attention mechanism failed: {str(e)}")
                    attention_outputs.append(X.mean(axis=1))
        
        return np.column_stack(attention_outputs) if attention_outputs else X
    
    def fit(self, X, y):
        """Fit the LSTM alternative model"""
        logger.info("Training LSTM Alternative Model...")
        
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        
        # Create sequence features
        sequence_features = self._create_sequences(X)
        
        # Add original features
        original_features = X.values
        combined_features = np.column_stack([sequence_features, original_features])
        
        # Scale features
        X_scaled = self.scaler.fit_transform(combined_features)
        
        # Train sequence models (simulating LSTM layers)
        current_input = X_scaled
        
        for i, model in enumerate(self.sequence_models):
            try:
                model.fit(current_input, y)
                
                # Create "hidden state" for next layer
                predictions = model.predict(current_input)
                current_input = np.column_stack([current_input, predictions])
                
            except Exception as e:
                logger.warning(f"Error training sequence model {i}: {str(e)}")
        
        # Train attention models
        for model in self.attention_models:
            try:
                model.fit(X_scaled, y)
            except Exception as e:
                logger.warning(f"Error training attention model: {str(e)}")
        
        # Train output model
        try:
            # Apply attention mechanism
            attended_features = self._apply_attention_mechanism(X_scaled)
            
            # Combine with sequence model outputs
            sequence_outputs = []
            for model in self.sequence_models:
                if hasattr(model, 'predict'):
                    sequence_outputs.append(model.predict(X_scaled))
            
            if sequence_outputs:
                final_features = np.column_stack([attended_features] + sequence_outputs)
            else:
                final_features = attended_features
            
            self.output_model.fit(final_features, y)
            
        except Exception as e:
            logger.warning(f"Error training output model: {str(e)}")
            # Fallback to simple model
            self.output_model.fit(X_scaled, y)
        
        # Store feature count for prediction
        self.n_features_ = combined_features.shape[1]
        self.is_fitted = True
        return self
    
    def predict(self, X) -> np.ndarray:
        """Make predictions using the LSTM alternative model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Create sequence features
        sequence_features = self._create_sequences(X)
        
        # Add original features
        original_features = X.values
        combined_features = np.column_stack([sequence_features, original_features])
        
        # Ensure consistent feature count
        if hasattr(self, 'n_features_') and combined_features.shape[1] != self.n_features_:
            # Pad or truncate features to match training
            if combined_features.shape[1] < self.n_features_:
                padding = np.zeros((combined_features.shape[0], self.n_features_ - combined_features.shape[1]))
                combined_features = np.column_stack([combined_features, padding])
            else:
                combined_features = combined_features[:, :self.n_features_]
        
        # Scale features
        X_scaled = self.scaler.transform(combined_features)
        
        try:
            # Apply attention mechanism
            attended_features = self._apply_attention_mechanism(X_scaled)
            
            # Get sequence model outputs
            sequence_outputs = []
            for model in self.sequence_models:
                if hasattr(model, 'predict'):
                    sequence_outputs.append(model.predict(X_scaled))
            
            if sequence_outputs:
                final_features = np.column_stack([attended_features] + sequence_outputs)
            else:
                final_features = attended_features
            
            predictions = self.output_model.predict(final_features)
            
        except Exception as e:
            logger.warning(f"Error in prediction: {str(e)}")
            # Fallback prediction
            predictions = self.output_model.predict(X_scaled)
        
        return predictions