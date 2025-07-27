"""
Meta-Learner for Intelligent Model Combination

This module implements a sophisticated meta-learning approach that learns
how to optimally combine predictions from multiple base models based on
market conditions and historical performance.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class MetaLearner(BaseEstimator, RegressorMixin):
    """
    Intelligent meta-learner that learns optimal model combination strategies
    """
    
    def __init__(self,
                 meta_model_type: str = 'random_forest',
                 use_market_regime: bool = True,
                 use_prediction_confidence: bool = True,
                 use_historical_performance: bool = True,
                 regime_lookback_days: int = 20,
                 performance_window: int = 50,
                 random_state: int = 42):
        """
        Initialize meta-learner
        
        Args:
            meta_model_type: Type of meta-model ('random_forest', 'ridge', 'elastic_net')
            use_market_regime: Include market regime features
            use_prediction_confidence: Include prediction confidence features
            use_historical_performance: Include historical performance features
            regime_lookback_days: Days to look back for regime classification
            performance_window: Window for calculating historical performance
        """
        self.meta_model_type = meta_model_type
        self.use_market_regime = use_market_regime
        self.use_prediction_confidence = use_prediction_confidence
        self.use_historical_performance = use_historical_performance
        self.regime_lookback_days = regime_lookback_days
        self.performance_window = performance_window
        self.random_state = random_state
        
        # Initialize meta-model
        if meta_model_type == 'random_forest':
            self.meta_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                min_samples_split=5,
                random_state=random_state
            )
        elif meta_model_type == 'ridge':
            self.meta_model = Ridge(alpha=1.0, random_state=random_state)
        elif meta_model_type == 'elastic_net':
            self.meta_model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=random_state)
        else:
            raise ValueError(f"Unknown meta_model_type: {meta_model_type}")
        
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.base_model_names = []
        self.historical_performance = {}
        
    def _extract_market_regime_features(self, X: pd.DataFrame, prices: pd.Series) -> pd.DataFrame:
        """Extract market regime features"""
        regime_features = pd.DataFrame(index=X.index)
        
        try:
            # Volatility regime
            returns = prices.pct_change().fillna(0)
            short_vol = returns.rolling(5).std()
            long_vol = returns.rolling(20).std()
            regime_features['volatility_regime'] = (short_vol / long_vol).fillna(1.0)
            
            # Trend regime
            sma_5 = prices.rolling(5).mean()
            sma_20 = prices.rolling(20).mean()
            regime_features['trend_strength'] = ((sma_5 - sma_20) / sma_20).fillna(0)
            
            # Momentum regime
            momentum_5 = (prices / prices.shift(5) - 1).fillna(0)
            momentum_20 = (prices / prices.shift(20) - 1).fillna(0)
            regime_features['momentum_divergence'] = momentum_5 - momentum_20
            
            # Market stress indicator
            rolling_returns = returns.rolling(self.regime_lookback_days)
            regime_features['market_stress'] = (
                rolling_returns.quantile(0.05) / rolling_returns.quantile(0.95)
            ).fillna(0)
            
            # Price level relative to recent range
            recent_high = prices.rolling(self.regime_lookback_days).max()
            recent_low = prices.rolling(self.regime_lookback_days).min()
            regime_features['price_position'] = (
                (prices - recent_low) / (recent_high - recent_low)
            ).fillna(0.5)
            
        except Exception as e:
            logger.warning(f"Error extracting market regime features: {str(e)}")
            # Fill with default values
            for col in ['volatility_regime', 'trend_strength', 'momentum_divergence', 'market_stress', 'price_position']:
                regime_features[col] = 0.5
        
        return regime_features
    
    def _extract_prediction_confidence_features(self, 
                                              base_predictions: np.ndarray,
                                              prediction_std: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Extract features related to prediction confidence and agreement"""
        n_samples, n_models = base_predictions.shape
        confidence_features = pd.DataFrame()
        
        # Prediction agreement/disagreement
        pred_mean = np.mean(base_predictions, axis=1)
        pred_std = np.std(base_predictions, axis=1)
        confidence_features['prediction_agreement'] = 1.0 / (1.0 + pred_std / np.abs(pred_mean))
        
        # Range of predictions
        pred_range = np.max(base_predictions, axis=1) - np.min(base_predictions, axis=1)
        confidence_features['prediction_range'] = pred_range / np.abs(pred_mean)
        
        # Model consensus strength
        confidence_features['consensus_strength'] = 1.0 - (pred_std / np.abs(pred_mean))
        
        # Prediction uncertainty (if available)
        if prediction_std is not None:
            confidence_features['prediction_uncertainty'] = prediction_std
        else:
            confidence_features['prediction_uncertainty'] = pred_std
        
        # Outlier detection in predictions
        z_scores = np.abs((base_predictions - pred_mean.reshape(-1, 1)) / pred_std.reshape(-1, 1))
        confidence_features['max_outlier_score'] = np.max(z_scores, axis=1)
        confidence_features['outlier_count'] = np.sum(z_scores > 2, axis=1)
        
        # Fill NaN values
        confidence_features = confidence_features.fillna(0)
        confidence_features = confidence_features.replace([np.inf, -np.inf], 0)
        
        return confidence_features
    
    def _calculate_historical_performance(self, 
                                        base_predictions: np.ndarray,
                                        actual_values: np.ndarray,
                                        window_size: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Calculate rolling historical performance for each base model"""
        if window_size is None:
            window_size = self.performance_window
        
        n_samples, n_models = base_predictions.shape
        performance_metrics = {
            'rolling_mae': np.zeros((n_samples, n_models)),
            'rolling_directional_accuracy': np.zeros((n_samples, n_models)),
            'rolling_bias': np.zeros((n_samples, n_models))
        }
        
        for i in range(n_models):
            model_predictions = base_predictions[:, i]
            
            for j in range(n_samples):
                start_idx = max(0, j - window_size + 1)
                end_idx = j + 1
                
                if end_idx - start_idx < 3:  # Need at least 3 samples
                    continue
                
                window_pred = model_predictions[start_idx:end_idx]
                window_actual = actual_values[start_idx:end_idx]
                
                # Rolling MAE
                mae = np.mean(np.abs(window_actual - window_pred))
                performance_metrics['rolling_mae'][j, i] = mae
                
                # Rolling directional accuracy
                if len(window_pred) > 1:
                    pred_direction = np.sign(np.diff(window_pred))
                    actual_direction = np.sign(np.diff(window_actual))
                    directional_acc = np.mean(pred_direction == actual_direction)
                    performance_metrics['rolling_directional_accuracy'][j, i] = directional_acc
                
                # Rolling bias
                bias = np.mean(window_pred - window_actual)
                performance_metrics['rolling_bias'][j, i] = bias
        
        return performance_metrics
    
    def _create_meta_features(self, 
                            X: pd.DataFrame,
                            base_predictions: np.ndarray,
                            prices: pd.Series,
                            actual_values: Optional[np.ndarray] = None,
                            prediction_std: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Create comprehensive meta-features for the meta-learner"""
        
        meta_features = pd.DataFrame(index=X.index)
        n_samples, n_models = base_predictions.shape
        
        # Base predictions as features
        for i in range(n_models):
            meta_features[f'base_pred_{i}'] = base_predictions[:, i]
        
        # Prediction statistics
        meta_features['pred_mean'] = np.mean(base_predictions, axis=1)
        meta_features['pred_std'] = np.std(base_predictions, axis=1)
        meta_features['pred_min'] = np.min(base_predictions, axis=1)
        meta_features['pred_max'] = np.max(base_predictions, axis=1)
        meta_features['pred_median'] = np.median(base_predictions, axis=1)
        
        # Market regime features
        if self.use_market_regime:
            regime_features = self._extract_market_regime_features(X, prices)
            meta_features = pd.concat([meta_features, regime_features], axis=1)
        
        # Prediction confidence features
        if self.use_prediction_confidence:
            confidence_features = self._extract_prediction_confidence_features(base_predictions, prediction_std)
            meta_features = pd.concat([meta_features, confidence_features], axis=1)
        
        # Historical performance features
        if self.use_historical_performance and actual_values is not None:
            try:
                performance_metrics = self._calculate_historical_performance(base_predictions, actual_values)
                
                # Add performance features
                for metric_name, metric_values in performance_metrics.items():
                    for i in range(n_models):
                        meta_features[f'{metric_name}_model_{i}'] = metric_values[:, i]
                    
                    # Aggregate performance features
                    meta_features[f'{metric_name}_mean'] = np.mean(metric_values, axis=1)
                    meta_features[f'{metric_name}_std'] = np.std(metric_values, axis=1)
                    meta_features[f'{metric_name}_best'] = np.min(metric_values, axis=1) if 'mae' in metric_name else np.max(metric_values, axis=1)
                
            except Exception as e:
                logger.warning(f"Error calculating historical performance features: {str(e)}")
        
        # Market context features (from original X)
        important_features = ['RSI', 'MACD', 'Volume_Change', 'Volatility', 'SMA_20', 'SMA_50']
        for feature in important_features:
            if feature in X.columns:
                meta_features[f'context_{feature}'] = X[feature]
        
        # Time-based features
        if hasattr(X.index, 'dayofweek'):
            meta_features['day_of_week'] = X.index.dayofweek
            meta_features['month'] = X.index.month
        
        # Clean up features
        meta_features = meta_features.fillna(0)
        meta_features = meta_features.replace([np.inf, -np.inf], 0)
        
        return meta_features
    
    def fit(self, 
            X: pd.DataFrame, 
            y: pd.Series, 
            base_predictions: np.ndarray,
            base_model_names: List[str],
            prices: pd.Series,
            prediction_std: Optional[np.ndarray] = None):
        """
        Fit the meta-learner
        
        Args:
            X: Original features
            y: Target values
            base_predictions: Predictions from base models (n_samples, n_models)
            base_model_names: Names of base models
            prices: Price series for regime detection
            prediction_std: Prediction uncertainties (optional)
        """
        logger.info("Training Meta-Learner...")
        
        self.base_model_names = base_model_names
        
        # Create meta-features
        meta_features = self._create_meta_features(
            X, base_predictions, prices, y.values, prediction_std
        )
        
        # Scale meta-features
        X_meta_scaled = self.scaler.fit_transform(meta_features)
        
        # Train meta-model
        try:
            self.meta_model.fit(X_meta_scaled, y)
            
            # Evaluate meta-model performance
            tscv = TimeSeriesSplit(n_splits=3)
            cv_scores = cross_val_score(
                self.meta_model, X_meta_scaled, y, 
                cv=tscv, scoring='neg_mean_absolute_error'
            )
            
            self.meta_performance = {
                'cv_mae': -np.mean(cv_scores),
                'cv_std': np.std(cv_scores),
                'feature_count': meta_features.shape[1]
            }
            
            logger.info(f"Meta-learner CV MAE: {self.meta_performance['cv_mae']:.4f}")
            
        except Exception as e:
            logger.error(f"Error training meta-model: {str(e)}")
            raise
        
        self.feature_names_ = meta_features.columns.tolist()
        self.is_fitted = True
        
        return self
    
    def predict(self, 
                X: pd.DataFrame,
                base_predictions: np.ndarray,
                prices: pd.Series,
                prediction_std: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Make meta-predictions
        
        Args:
            X: Original features
            base_predictions: Predictions from base models
            prices: Price series for regime detection
            prediction_std: Prediction uncertainties (optional)
        """
        if not self.is_fitted:
            raise ValueError("Meta-learner must be fitted before making predictions")
        
        # Create meta-features (without actual values for historical performance)
        meta_features = self._create_meta_features(
            X, base_predictions, prices, None, prediction_std
        )
        
        # Ensure all training features are present
        for feature in self.feature_names_:
            if feature not in meta_features.columns:
                meta_features[feature] = 0
        
        # Reorder columns to match training
        meta_features = meta_features[self.feature_names_]
        meta_features = meta_features.fillna(0)
        
        # Scale meta-features
        X_meta_scaled = self.scaler.transform(meta_features)
        
        # Make meta-prediction
        meta_predictions = self.meta_model.predict(X_meta_scaled)
        
        return meta_predictions
    
    def get_model_weights(self, 
                         X: pd.DataFrame,
                         base_predictions: np.ndarray,
                         prices: pd.Series,
                         prediction_std: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get dynamic model weights based on current conditions
        
        Returns:
            weights: Array of shape (n_samples, n_models) with weights for each model
        """
        if not self.is_fitted:
            raise ValueError("Meta-learner must be fitted before getting weights")
        
        # Create meta-features
        meta_features = self._create_meta_features(
            X, base_predictions, prices, None, prediction_std
        )
        
        # Get base model predictions from meta-features
        n_models = len(self.base_model_names)
        base_pred_features = meta_features[[f'base_pred_{i}' for i in range(n_models)]].values
        
        # Calculate weights based on prediction agreement and historical performance
        weights = np.ones((len(X), n_models))
        
        try:
            # Use prediction agreement to adjust weights
            pred_mean = np.mean(base_pred_features, axis=1, keepdims=True)
            pred_errors = np.abs(base_pred_features - pred_mean)
            agreement_weights = 1.0 / (1.0 + pred_errors)
            
            # Normalize weights
            weights = agreement_weights / np.sum(agreement_weights, axis=1, keepdims=True)
            
        except Exception as e:
            logger.warning(f"Error calculating dynamic weights: {str(e)}")
            # Fallback to equal weights
            weights = weights / n_models
        
        return weights
    
    def get_feature_importance(self) -> pd.Series:
        """Get feature importance from meta-model"""
        if not self.is_fitted:
            return pd.Series()
        
        if hasattr(self.meta_model, 'feature_importances_'):
            importance = pd.Series(
                self.meta_model.feature_importances_,
                index=self.feature_names_
            ).sort_values(ascending=False)
            return importance
        elif hasattr(self.meta_model, 'coef_'):
            # For linear models, use absolute coefficients
            importance = pd.Series(
                np.abs(self.meta_model.coef_),
                index=self.feature_names_
            ).sort_values(ascending=False)
            return importance
        else:
            return pd.Series()
    
    def get_performance_metrics(self) -> Dict:
        """Get meta-learner performance metrics"""
        if not hasattr(self, 'meta_performance'):
            return {}
        
        metrics = self.meta_performance.copy()
        metrics.update({
            'model_type': self.meta_model_type,
            'base_models': len(self.base_model_names),
            'feature_types': {
                'market_regime': self.use_market_regime,
                'prediction_confidence': self.use_prediction_confidence,
                'historical_performance': self.use_historical_performance
            }
        })
        
        return metrics