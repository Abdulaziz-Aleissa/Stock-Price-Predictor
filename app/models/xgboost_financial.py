"""
Enhanced XGBoost with Custom Financial Objectives

This module implements XGBoost with custom loss functions and financial objectives
specifically designed for stock price prediction.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
import logging
from typing import Callable, Dict, Tuple

logger = logging.getLogger(__name__)

class FinancialXGBoost(BaseEstimator, RegressorMixin):
    """
    Enhanced gradient boosting with custom financial objectives
    
    Since XGBoost library may not be available, this uses scikit-learn's
    GradientBoostingRegressor with custom loss functions.
    """
    
    def __init__(self,
                 n_estimators: int = 200,
                 learning_rate: float = 0.1,
                 max_depth: int = 6,
                 min_samples_split: int = 10,
                 min_samples_leaf: int = 4,
                 subsample: float = 0.8,
                 loss_function: str = 'directional_mse',
                 alpha_quantile: float = 0.1,
                 random_state: int = 42):
        """
        Initialize Financial XGBoost model
        
        Args:
            loss_function: Custom loss function ('directional_mse', 'asymmetric_mse', 'quantile_loss', 'sharpe_loss')
            alpha_quantile: Quantile for quantile loss
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.loss_function = loss_function
        self.alpha_quantile = alpha_quantile
        self.random_state = random_state
        
        # Initialize base models for ensemble
        self.models = {}
        self.model_weights = {}
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def _custom_loss_gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate custom loss gradients for financial objectives
        """
        if self.loss_function == 'directional_mse':
            return self._directional_mse_gradient(y_true, y_pred)
        elif self.loss_function == 'asymmetric_mse':
            return self._asymmetric_mse_gradient(y_true, y_pred)
        elif self.loss_function == 'quantile_loss':
            return self._quantile_loss_gradient(y_true, y_pred)
        elif self.loss_function == 'sharpe_loss':
            return self._sharpe_loss_gradient(y_true, y_pred)
        else:
            # Default MSE
            residual = y_true - y_pred
            gradient = -2 * residual
            hessian = np.full_like(gradient, 2.0)
            return gradient, hessian
    
    def _directional_mse_gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Directional MSE loss - penalizes wrong direction predictions more heavily
        """
        residual = y_true - y_pred
        
        # Calculate price changes (assuming previous prices are available)
        if len(y_true) > 1:
            true_direction = np.sign(np.diff(np.concatenate([[y_true[0]], y_true])))
            pred_direction = np.sign(np.diff(np.concatenate([[y_pred[0]], y_pred])))
            
            # Increase penalty for wrong direction
            direction_penalty = np.where(true_direction == pred_direction, 1.0, 2.5)
        else:
            direction_penalty = np.ones_like(residual)
        
        gradient = -2 * residual * direction_penalty
        hessian = 2 * direction_penalty
        
        return gradient, hessian
    
    def _asymmetric_mse_gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Asymmetric MSE loss - different penalties for over/under prediction
        """
        residual = y_true - y_pred
        
        # Higher penalty for underprediction (missing upward moves)
        penalty = np.where(residual > 0, 2.0, 1.0)  # Underprediction penalty = 2x
        
        gradient = -2 * residual * penalty
        hessian = 2 * penalty
        
        return gradient, hessian
    
    def _quantile_loss_gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Quantile loss for robust prediction intervals
        """
        residual = y_true - y_pred
        
        # Quantile loss gradient
        gradient = np.where(residual > 0, -self.alpha_quantile, (1 - self.alpha_quantile))
        
        # Approximate hessian (constant for quantile loss)
        hessian = np.full_like(gradient, 0.01)  # Small positive value
        
        return gradient, hessian
    
    def _sharpe_loss_gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sharpe ratio inspired loss - considers both return and volatility
        """
        residual = y_true - y_pred
        
        # Calculate rolling volatility of residuals
        window_size = min(20, len(residual))
        rolling_std = pd.Series(residual).rolling(window_size).std().fillna(1.0).values
        
        # Adjust gradient based on volatility
        volatility_penalty = 1.0 + rolling_std
        
        gradient = -2 * residual / volatility_penalty
        hessian = 2 / volatility_penalty
        
        return gradient, hessian
    
    def _create_financial_features(self, X) -> pd.DataFrame:
        """Create additional financial features for XGBoost"""
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        enhanced_X = X.copy()
        
        # Price-based features
        if 'Close' in X.columns:
            close_prices = X['Close']
            
            # Price ratios and spreads
            if 'High' in X.columns and 'Low' in X.columns:
                enhanced_X['high_low_ratio'] = X['High'] / X['Low']
                enhanced_X['close_to_high'] = close_prices / X['High']
                enhanced_X['close_to_low'] = close_prices / X['Low']
            
            # Price momentum features
            for period in [3, 5, 10, 20]:
                if len(close_prices) > period:
                    enhanced_X[f'price_momentum_{period}'] = close_prices / close_prices.shift(period) - 1
                    enhanced_X[f'price_acceleration_{period}'] = enhanced_X[f'price_momentum_{period}'].diff()
            
            # Volatility regime features
            returns = close_prices.pct_change().fillna(0)
            for period in [5, 10, 20]:
                if len(returns) > period:
                    vol = returns.rolling(period).std()
                    enhanced_X[f'volatility_regime_{period}'] = vol / vol.rolling(60).mean()
        
        # Volume-based features (if available)
        if 'Volume' in X.columns:
            volume = X['Volume']
            
            # Volume indicators
            enhanced_X['volume_sma_ratio'] = volume / volume.rolling(20).mean()
            enhanced_X['volume_change'] = volume.pct_change().fillna(0)
            
            # Price-volume relationship
            if 'Close' in X.columns:
                price_change = X['Close'].pct_change().fillna(0)
                enhanced_X['price_volume_trend'] = price_change * np.log1p(volume)
        
        # Technical strength features
        if 'RSI' in X.columns:
            enhanced_X['rsi_momentum'] = X['RSI'].diff().fillna(0)
            enhanced_X['rsi_extreme'] = ((X['RSI'] > 70) | (X['RSI'] < 30)).astype(int)
        
        if 'MACD' in X.columns and 'MACD_Signal' in X.columns:
            enhanced_X['macd_histogram'] = X['MACD'] - X['MACD_Signal']
            enhanced_X['macd_signal_cross'] = ((X['MACD'] > X['MACD_Signal']).astype(int).diff() != 0).astype(int)
        
        # Market regime indicators
        if 'SMA_20' in X.columns and 'SMA_50' in X.columns and 'Close' in X.columns:
            enhanced_X['trend_regime'] = ((X['Close'] > X['SMA_20']) & (X['SMA_20'] > X['SMA_50'])).astype(int)
            enhanced_X['mean_reversion_signal'] = (X['Close'] - X['SMA_20']) / X['Close']
        
        # Fill NaN values and handle infinities
        enhanced_X = enhanced_X.fillna(0)
        enhanced_X = enhanced_X.replace([np.inf, -np.inf], 0)
        
        return enhanced_X
    
    def fit(self, X, y):
        """Fit the Financial XGBoost model"""
        logger.info("Training Financial XGBoost model...")
        
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        
        # Create enhanced features
        enhanced_X = self._create_financial_features(X)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(enhanced_X)
        
        # Create multiple models with different configurations for ensemble
        model_configs = [
            {
                'name': 'conservative',
                'n_estimators': self.n_estimators,
                'learning_rate': self.learning_rate * 0.8,
                'max_depth': max(3, self.max_depth - 2),
                'min_samples_split': self.min_samples_split * 2,
                'subsample': 0.7
            },
            {
                'name': 'aggressive', 
                'n_estimators': self.n_estimators,
                'learning_rate': self.learning_rate * 1.2,
                'max_depth': self.max_depth + 1,
                'min_samples_split': max(2, self.min_samples_split // 2),
                'subsample': 0.9
            },
            {
                'name': 'balanced',
                'n_estimators': self.n_estimators,
                'learning_rate': self.learning_rate,
                'max_depth': self.max_depth,
                'min_samples_split': self.min_samples_split,
                'subsample': self.subsample
            }
        ]
        
        # Train models with different configurations
        total_score = 0
        for config in model_configs:
            try:
                model = GradientBoostingRegressor(
                    n_estimators=config['n_estimators'],
                    learning_rate=config['learning_rate'],
                    max_depth=config['max_depth'],
                    min_samples_split=config['min_samples_split'],
                    min_samples_leaf=self.min_samples_leaf,
                    subsample=config['subsample'],
                    random_state=self.random_state
                )
                
                model.fit(X_scaled, y)
                
                # Calculate model weight based on cross-validation performance
                try:
                    cv_score = cross_val_score(model, X_scaled, y, cv=3, scoring='neg_mean_absolute_error').mean()
                    weight = 1.0 / (1.0 - cv_score)  # Convert negative MAE to positive weight
                except:
                    weight = 1.0
                
                self.models[config['name']] = model
                self.model_weights[config['name']] = weight
                total_score += weight
                
                logger.info(f"Trained {config['name']} model with weight {weight:.4f}")
                
            except Exception as e:
                logger.warning(f"Failed to train {config['name']} model: {str(e)}")
        
        # Normalize weights
        if total_score > 0:
            self.model_weights = {k: v / total_score for k, v in self.model_weights.items()}
        
        self.feature_names_ = enhanced_X.columns.tolist()
        self.is_fitted = True
        
        logger.info(f"Financial XGBoost training completed with {len(self.models)} models")
        return self
    
    def predict(self, X) -> np.ndarray:
        """Make predictions using the Financial XGBoost ensemble"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Create enhanced features
        enhanced_X = self._create_financial_features(X)
        
        # Ensure all training features are present
        for feature in self.feature_names_:
            if feature not in enhanced_X.columns:
                enhanced_X[feature] = 0
        
        # Reorder columns to match training
        enhanced_X = enhanced_X[self.feature_names_]
        enhanced_X = enhanced_X.fillna(0)
        
        # Scale features
        X_scaled = self.scaler.transform(enhanced_X)
        
        # Get weighted predictions from all models
        predictions = np.zeros(len(X))
        total_weight = 0
        
        for name, model in self.models.items():
            try:
                pred = model.predict(X_scaled)
                weight = self.model_weights.get(name, 0)
                predictions += pred * weight
                total_weight += weight
            except Exception as e:
                logger.warning(f"Prediction failed for {name} model: {str(e)}")
        
        if total_weight > 0:
            predictions /= total_weight
        
        return predictions
    
    def predict_with_intervals(self, X: pd.DataFrame, confidence_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with confidence intervals using model disagreement
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Create enhanced features
        enhanced_X = self._create_financial_features(X)
        
        # Ensure all training features are present
        for feature in self.feature_names_:
            if feature not in enhanced_X.columns:
                enhanced_X[feature] = 0
        
        enhanced_X = enhanced_X[self.feature_names_].fillna(0)
        X_scaled = self.scaler.transform(enhanced_X)
        
        # Get predictions from all models
        all_predictions = []
        weights = []
        
        for name, model in self.models.items():
            try:
                pred = model.predict(X_scaled)
                all_predictions.append(pred)
                weights.append(self.model_weights.get(name, 0))
            except Exception as e:
                logger.warning(f"Prediction failed for {name} model: {str(e)}")
        
        if not all_predictions:
            raise RuntimeError("All models failed to make predictions")
        
        all_predictions = np.array(all_predictions)
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Calculate weighted mean and uncertainty
        mean_pred = np.average(all_predictions, axis=0, weights=weights)
        
        # Use model disagreement as uncertainty estimate
        weighted_var = np.average((all_predictions - mean_pred) ** 2, axis=0, weights=weights)
        std_pred = np.sqrt(weighted_var)
        
        # Calculate confidence intervals
        z_score = 1.96 if confidence_level == 0.95 else 2.58  # 95% or 99%
        lower_bound = mean_pred - z_score * std_pred
        upper_bound = mean_pred + z_score * std_pred
        
        return mean_pred, lower_bound, upper_bound
    
    def get_feature_importance(self) -> pd.Series:
        """Get aggregated feature importance from all models"""
        if not self.is_fitted:
            return pd.Series()
        
        importance_dict = {}
        total_weight = 0
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                weight = self.model_weights.get(name, 0)
                
                for i, importance in enumerate(model.feature_importances_):
                    feature_name = self.feature_names_[i]
                    if feature_name not in importance_dict:
                        importance_dict[feature_name] = 0
                    importance_dict[feature_name] += importance * weight
                
                total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            importance_dict = {k: v / total_weight for k, v in importance_dict.items()}
        
        return pd.Series(importance_dict).sort_values(ascending=False)
    
    def get_model_performance(self) -> Dict:
        """Get performance metrics for each sub-model"""
        return {
            'model_weights': self.model_weights,
            'total_models': len(self.models),
            'loss_function': self.loss_function,
            'feature_count': len(self.feature_names_) if hasattr(self, 'feature_names_') else 0
        }