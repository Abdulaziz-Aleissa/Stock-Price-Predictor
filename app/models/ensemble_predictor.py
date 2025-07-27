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

logger = logging.getLogger(__name__)

class AdvancedEnsemblePredictor:
    """
    Advanced ensemble predictor combining multiple models with intelligent weighting
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.model_weights = {}
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.is_fitted = False
        self.feature_importance_ = None
        self.model_performances = {}
        
        # Initialize base models with optimized hyperparameters
        self._init_base_models()
        
    def _init_base_models(self):
        """Initialize base models with optimized hyperparameters"""
        
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
        
        # Calculate dynamic model weights
        self._calculate_model_weights(X_scaled, y_clean)
        
        # Train all models
        trained_models = {}
        for name, model in self.models.items():
            try:
                logger.info(f"Training {name}...")
                model.fit(X_scaled, y_clean)
                trained_models[name] = model
                
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
        Make ensemble predictions with uncertainty quantification
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        # Handle NaN values and scale
        X_clean = X.fillna(0)
        X_scaled = self.scaler.transform(X_clean)
        
        # Get predictions from all models
        predictions = {}
        for name, model in self.models.items():
            try:
                pred = model.predict(X_scaled)
                predictions[name] = pred
            except Exception as e:
                logger.warning(f"Prediction failed for {name}: {str(e)}")
                
        if not predictions:
            raise RuntimeError("All models failed to make predictions")
            
        # Weighted ensemble prediction
        ensemble_pred = np.zeros(len(X))
        total_weight = 0
        
        for name, pred in predictions.items():
            weight = self.model_weights.get(name, 0)
            ensemble_pred += pred * weight
            total_weight += weight
            
        if total_weight > 0:
            ensemble_pred /= total_weight
            
        return ensemble_pred
        
    def predict_with_uncertainty(self, X, return_std=True):
        """
        Make predictions with uncertainty estimates
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        # Handle NaN values and scale
        X_clean = X.fillna(0)
        X_scaled = self.scaler.transform(X_clean)
        
        # Get predictions from all models
        all_predictions = []
        weights = []
        
        for name, model in self.models.items():
            try:
                pred = model.predict(X_scaled)
                all_predictions.append(pred)
                weights.append(self.model_weights.get(name, 0))
            except Exception as e:
                logger.warning(f"Prediction failed for {name}: {str(e)}")
                
        if not all_predictions:
            raise RuntimeError("All models failed to make predictions")
            
        all_predictions = np.array(all_predictions)
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
        
        # Weighted ensemble prediction
        ensemble_pred = np.average(all_predictions, axis=0, weights=weights)
        
        if return_std:
            # Calculate prediction uncertainty as weighted standard deviation
            weighted_var = np.average(
                (all_predictions - ensemble_pred) ** 2, 
                axis=0, weights=weights
            )
            prediction_std = np.sqrt(weighted_var)
            
            return ensemble_pred, prediction_std
        else:
            return ensemble_pred
            
    def get_model_performance(self):
        """Get detailed performance metrics for each model"""
        return self.model_performances
        
    def get_feature_importance(self, top_n=20):
        """Get top N most important features"""
        if self.feature_importance_ is None:
            return None
        return self.feature_importance_.head(top_n)
        
    def save_model(self, filepath):
        """Save the trained ensemble model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
            
        model_data = {
            'models': self.models,
            'model_weights': self.model_weights,
            'scaler': self.scaler,
            'feature_importance': self.feature_importance_,
            'model_performances': self.model_performances,
            'is_fitted': self.is_fitted,
            'metadata': {
                'created_at': datetime.now(),
                'model_type': 'AdvancedEnsemblePredictor'
            }
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Ensemble model saved to {filepath}")
        
    @classmethod
    def load_model(cls, filepath):
        """Load a trained ensemble model"""
        model_data = joblib.load(filepath)
        
        ensemble = cls()
        ensemble.models = model_data['models']
        ensemble.model_weights = model_data['model_weights']
        ensemble.scaler = model_data['scaler']
        ensemble.feature_importance_ = model_data.get('feature_importance')
        ensemble.model_performances = model_data.get('model_performances', {})
        ensemble.is_fitted = model_data['is_fitted']
        
        logger.info(f"Ensemble model loaded from {filepath}")
        return ensemble