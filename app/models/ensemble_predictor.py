"""
Advanced Multi-Model Ensemble Prediction System

This module implements a sophisticated ensemble combining multiple machine learning models
for improved stock price prediction accuracy and robustness.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, 
    ExtraTreesRegressor, AdaBoostRegressor, VotingRegressor
)
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import custom models
from .transformer_model import TimeSeriesModel, LSTMAlternativeModel
from .xgboost_financial import FinancialXGBoost
from .meta_learner import MetaLearner

logger = logging.getLogger(__name__)

class AdvancedEnsemblePredictor:
    """
    Advanced ensemble predictor combining multiple models with intelligent weighting
    """
    
    def __init__(self, random_state=42, use_meta_learner=True):
        self.random_state = random_state
        self.use_meta_learner = use_meta_learner
        self.models = {}
        self.model_weights = {}
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.is_fitted = False
        self.feature_importance_ = None
        self.model_performances = {}
        self.meta_learner = None
        
        # Initialize base models with optimized hyperparameters
        self._init_base_models()
        
    def _init_base_models(self):
        """Initialize base models with optimized hyperparameters"""
        
        # Traditional ML Models
        # Gradient Boosting - Primary model for trend capture
        self.models['gradient_boost'] = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.08,
            max_depth=6,
            min_samples_split=10,
            min_samples_leaf=4,
            subsample=0.8,
            random_state=self.random_state
        )
        
        # Random Forest - Strong baseline with feature importance
        self.models['random_forest'] = RandomForestRegressor(
            n_estimators=150,
            max_depth=12,
            min_samples_split=8,
            min_samples_leaf=4,
            max_features='sqrt',
            bootstrap=True,
            random_state=self.random_state
        )
        
        # Extra Trees - High variance, good for ensemble diversity
        self.models['extra_trees'] = ExtraTreesRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=False,
            random_state=self.random_state
        )
        
        # AdaBoost - Sequential boosting for hard cases
        self.models['ada_boost'] = AdaBoostRegressor(
            n_estimators=100,
            learning_rate=0.1,
            loss='linear',
            random_state=self.random_state
        )
        
        # Support Vector Regression - Non-linear patterns
        self.models['svr'] = SVR(
            kernel='rbf',
            C=100,
            gamma='scale',
            epsilon=0.1
        )
        
        # Ridge Regression - Linear baseline with regularization
        self.models['ridge'] = Ridge(
            alpha=1.0,
            random_state=self.random_state
        )
        
        # ElasticNet - Feature selection capabilities
        self.models['elastic_net'] = ElasticNet(
            alpha=0.1,
            l1_ratio=0.5,
            random_state=self.random_state
        )
        
        # Multi-layer Perceptron - Neural network approximation
        self.models['mlp'] = MLPRegressor(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=500,
            random_state=self.random_state
        )
        
        # Advanced Custom Models
        # Time Series Model (Transformer alternative)
        self.models['time_series'] = TimeSeriesModel(
            lookback_window=30,
            seasonal_periods=[5, 10, 20, 60],
            random_state=self.random_state
        )
        
        # LSTM Alternative Model
        self.models['lstm_alternative'] = LSTMAlternativeModel(
            sequence_length=20,
            hidden_units=50,
            attention_heads=4,
            random_state=self.random_state
        )
        
        # Financial XGBoost with custom objectives
        self.models['financial_xgboost'] = FinancialXGBoost(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            loss_function='directional_mse',
            random_state=self.random_state
        )
        
        # Initialize meta-learner if requested
        if self.use_meta_learner:
            self.meta_learner = MetaLearner(
                meta_model_type='random_forest',
                use_market_regime=True,
                use_prediction_confidence=True,
                use_historical_performance=True,
                random_state=self.random_state
            )
        
    def _calculate_model_weights(self, X, y):
        """Calculate intelligent model weights based on cross-validation performance"""
        logger.info("Calculating dynamic model weights based on performance...")
        
        # Use TimeSeriesSplit for financial data
        tscv = TimeSeriesSplit(n_splits=5)
        weights = {}
        
        for name, model in self.models.items():
            try:
                # Cross-validation scores (negative MAE)
                cv_scores = cross_val_score(
                    model, X, y, cv=tscv, 
                    scoring='neg_mean_absolute_error', 
                    n_jobs=-1
                )
                
                # Convert to positive MAE and calculate weight
                mae_scores = -cv_scores
                avg_mae = np.mean(mae_scores)
                
                # Weight inversely proportional to error (better models get higher weight)
                weight = 1.0 / (1.0 + avg_mae)
                weights[name] = weight
                
                self.model_performances[name] = {
                    'cv_mae': avg_mae,
                    'cv_std': np.std(mae_scores),
                    'weight': weight
                }
                
                logger.info(f"{name}: MAE={avg_mae:.4f} ± {np.std(mae_scores):.4f}, Weight={weight:.4f}")
                
            except Exception as e:
                logger.warning(f"Failed to calculate weight for {name}: {str(e)}")
                weights[name] = 0.1  # Minimal weight for failed models
                
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        if total_weight > 0:
            self.model_weights = {k: v / total_weight for k, v in weights.items()}
        else:
            # Equal weights if all models failed
            self.model_weights = {k: 1.0 / len(self.models) for k in self.models.keys()}
            
        logger.info(f"Final model weights: {self.model_weights}")
        
    def fit(self, X, y):
        """
        Fit the ensemble model with advanced preprocessing and weight calculation
        """
        logger.info("Training Advanced Ensemble Predictor...")
        
        # Validate input
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")
        
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
            
        # Handle any remaining NaN values
        X_clean = X.fillna(0)
        y_clean = y.fillna(y.mean())
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_clean)
        
        # Calculate dynamic model weights (only use traditional models for cross-validation)
        traditional_models = {}
        for name, model in self.models.items():
            if name in ['gradient_boost', 'random_forest', 'extra_trees', 'ada_boost', 'svr', 'ridge', 'elastic_net', 'mlp']:
                traditional_models[name] = model
        
        # Calculate weights for traditional models using scaled features
        temp_models = self.models
        self.models = traditional_models
        self._calculate_model_weights(X_scaled, y_clean)
        
        # Restore all models and add default weights for custom models
        self.models = temp_models
        custom_model_weight = 0.1  # Default weight for custom models
        for name in self.models.keys():
            if name not in self.model_weights:
                self.model_weights[name] = custom_model_weight
        
        # Train all models
        trained_models = {}
        base_predictions = []
        
        for name, model in self.models.items():
            try:
                logger.info(f"Training {name}...")
                model.fit(X_clean, y_clean)
                trained_models[name] = model
                
                # Get predictions for meta-learner training
                if self.use_meta_learner:
                    pred = model.predict(X_clean)
                    base_predictions.append(pred)
                
            except Exception as e:
                logger.warning(f"Failed to train {name}: {str(e)}")
                # Remove failed model from weights
                if name in self.model_weights:
                    del self.model_weights[name]
        
        # Update models to only include successfully trained ones
        self.models = trained_models
        
        # Renormalize weights after removing failed models
        total_weight = sum(self.model_weights.values())
        if total_weight > 0:
            self.model_weights = {k: v / total_weight for k, v in self.model_weights.items()}
        
        # Train meta-learner if enabled and we have base predictions
        if self.use_meta_learner and len(base_predictions) > 0:
            try:
                logger.info("Training meta-learner...")
                base_predictions_array = np.column_stack(base_predictions)
                
                # Extract price series for regime detection
                if 'Close' in X_clean.columns:
                    prices = X_clean['Close']
                else:
                    prices = y_clean  # Use target as price proxy
                
                self.meta_learner.fit(
                    X_clean, y_clean, base_predictions_array,
                    list(self.models.keys()), prices
                )
                
                logger.info("Meta-learner training completed")
                
            except Exception as e:
                logger.warning(f"Meta-learner training failed: {str(e)}")
                self.use_meta_learner = False
        
        # Calculate ensemble feature importance
        self._calculate_feature_importance(X.columns.tolist() if hasattr(X, 'columns') else None)
        
        self.is_fitted = True
        logger.info(f"Ensemble training completed with {len(self.models)} models")
        
        return self
        
    def _calculate_feature_importance(self, feature_names=None):
        """Calculate ensemble feature importance from models that support it"""
        if not feature_names:
            return
            
        importance_models = ['gradient_boost', 'random_forest', 'extra_trees']
        importances = []
        weights = []
        
        for name in importance_models:
            if name in self.models and hasattr(self.models[name], 'feature_importances_'):
                importances.append(self.models[name].feature_importances_)
                weights.append(self.model_weights.get(name, 0))
                
        if importances:
            # Weighted average of feature importances
            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalize weights
            
            weighted_importance = np.zeros_like(importances[0])
            for imp, weight in zip(importances, weights):
                weighted_importance += imp * weight
                
            self.feature_importance_ = pd.Series(
                weighted_importance, 
                index=feature_names
            ).sort_values(ascending=False)
            
    def predict(self, X):
        """
        Make ensemble predictions with meta-learner optimization
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        # Handle NaN values
        X_clean = X.fillna(0)
        
        # Get predictions from all base models
        base_predictions = []
        model_names = []
        
        for name, model in self.models.items():
            try:
                if name in ['gradient_boost', 'random_forest', 'extra_trees', 'ada_boost', 'svr', 'ridge', 'elastic_net', 'mlp']:
                    # Traditional models need scaled features
                    X_scaled = self.scaler.transform(X_clean)
                    pred = model.predict(X_scaled)
                else:
                    # Custom models handle scaling internally
                    pred = model.predict(X_clean)
                
                base_predictions.append(pred)
                model_names.append(name)
                
            except Exception as e:
                logger.warning(f"Prediction failed for {name}: {str(e)}")
                
        if not base_predictions:
            raise RuntimeError("All models failed to make predictions")
        
        base_predictions_array = np.column_stack(base_predictions)
        
        # Use meta-learner if available and trained
        if self.use_meta_learner and self.meta_learner and self.meta_learner.is_fitted:
            try:
                # Extract price series for regime detection
                if 'Close' in X_clean.columns:
                    prices = X_clean['Close']
                else:
                    # Use mean of predictions as price proxy
                    prices = pd.Series(np.mean(base_predictions_array, axis=1), index=X_clean.index)
                
                meta_predictions = self.meta_learner.predict(
                    X_clean, base_predictions_array, prices
                )
                
                return meta_predictions
                
            except Exception as e:
                logger.warning(f"Meta-learner prediction failed: {str(e)}")
                # Fall back to weighted ensemble
        
        # Weighted ensemble prediction (fallback)
        ensemble_pred = np.zeros(len(X))
        total_weight = 0
        
        for i, name in enumerate(model_names):
            weight = self.model_weights.get(name, 0)
            ensemble_pred += base_predictions_array[:, i] * weight
            total_weight += weight
            
        if total_weight > 0:
            ensemble_pred /= total_weight
            
        return ensemble_pred
        
    def predict_with_uncertainty(self, X, return_std=True):
        """
        Make predictions with uncertainty estimates using all models
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        # Handle NaN values
        X_clean = X.fillna(0)
        
        # Get predictions from all base models
        all_predictions = []
        weights = []
        model_names = []
        
        for name, model in self.models.items():
            try:
                if name in ['gradient_boost', 'random_forest', 'extra_trees', 'ada_boost', 'svr', 'ridge', 'elastic_net', 'mlp']:
                    # Traditional models need scaled features
                    X_scaled = self.scaler.transform(X_clean)
                    pred = model.predict(X_scaled)
                else:
                    # Custom models handle scaling internally
                    pred = model.predict(X_clean)
                
                all_predictions.append(pred)
                weights.append(self.model_weights.get(name, 0))
                model_names.append(name)
                
            except Exception as e:
                logger.warning(f"Prediction failed for {name}: {str(e)}")
                
        if not all_predictions:
            raise RuntimeError("All models failed to make predictions")
            
        all_predictions_array = np.array(all_predictions)
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
        
        # Try meta-learner first
        ensemble_pred = None
        if self.use_meta_learner and self.meta_learner and self.meta_learner.is_fitted:
            try:
                # Extract price series for regime detection
                if 'Close' in X_clean.columns:
                    prices = X_clean['Close']
                else:
                    prices = pd.Series(np.mean(all_predictions_array, axis=1), index=X_clean.index)
                
                ensemble_pred = self.meta_learner.predict(
                    X_clean, all_predictions_array.T, prices
                )
                
            except Exception as e:
                logger.warning(f"Meta-learner prediction failed: {str(e)}")
        
        # Fallback to weighted ensemble
        if ensemble_pred is None:
            ensemble_pred = np.average(all_predictions_array, axis=0, weights=weights)
        
        if return_std:
            # Calculate prediction uncertainty as weighted standard deviation
            weighted_var = np.average(
                (all_predictions_array - ensemble_pred) ** 2, 
                axis=0, weights=weights
            )
            prediction_std = np.sqrt(weighted_var)
            
            return ensemble_pred, prediction_std
        else:
            return ensemble_pred
            
    def get_model_performance(self):
        """Get detailed performance metrics for each model including meta-learner"""
        performance = self.model_performances.copy()
        
        # Add meta-learner performance if available
        if self.use_meta_learner and self.meta_learner:
            meta_performance = self.meta_learner.get_performance_metrics()
            performance['meta_learner'] = meta_performance
        
        # Add ensemble-level metrics
        performance['ensemble_info'] = {
            'total_models': len(self.models),
            'use_meta_learner': self.use_meta_learner,
            'model_types': list(self.models.keys())
        }
        
        return performance
        
    def get_feature_importance(self, top_n=20):
        """Get top N most important features from ensemble and meta-learner"""
        # Get base ensemble feature importance
        base_importance = None
        if self.feature_importance_ is not None:
            base_importance = self.feature_importance_.head(top_n)
        
        # Get meta-learner feature importance if available
        meta_importance = None
        if self.use_meta_learner and self.meta_learner and self.meta_learner.is_fitted:
            try:
                meta_importance = self.meta_learner.get_feature_importance()
                if len(meta_importance) > 0:
                    meta_importance = meta_importance.head(top_n)
            except Exception as e:
                logger.warning(f"Could not get meta-learner feature importance: {str(e)}")
        
        # Combine importances
        result = {}
        if base_importance is not None:
            result['base_ensemble'] = base_importance
        if meta_importance is not None:
            result['meta_learner'] = meta_importance
        
        return result if result else base_importance
        
    def save_model(self, filepath):
        """Save the trained ensemble model including meta-learner"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
            
        model_data = {
            'models': self.models,
            'model_weights': self.model_weights,
            'scaler': self.scaler,
            'feature_importance': self.feature_importance_,
            'model_performances': self.model_performances,
            'use_meta_learner': self.use_meta_learner,
            'meta_learner': self.meta_learner,
            'is_fitted': self.is_fitted,
            'metadata': {
                'created_at': datetime.now(),
                'model_type': 'AdvancedEnsemblePredictor',
                'version': '2.0'
            }
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Enhanced ensemble model saved to {filepath}")
        
    @classmethod
    def load_model(cls, filepath):
        """Load a trained ensemble model including meta-learner"""
        model_data = joblib.load(filepath)
        
        # Determine if this is an old or new model format
        use_meta_learner = model_data.get('use_meta_learner', False)
        
        ensemble = cls(use_meta_learner=use_meta_learner)
        ensemble.models = model_data['models']
        ensemble.model_weights = model_data['model_weights']
        ensemble.scaler = model_data['scaler']
        ensemble.feature_importance_ = model_data.get('feature_importance')
        ensemble.model_performances = model_data.get('model_performances', {})
        ensemble.use_meta_learner = use_meta_learner
        ensemble.meta_learner = model_data.get('meta_learner')
        ensemble.is_fitted = model_data['is_fitted']
        
        logger.info(f"Enhanced ensemble model loaded from {filepath}")
        return ensemble