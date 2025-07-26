"""
Random Forest Model for Stock Price Prediction
Implements Random Forest ensemble method for stock price forecasting
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import configuration and helpers
from ..utils.config import config
from ..utils.helpers import performance_tracker

logger = logging.getLogger(__name__)

class RandomForestModel:
    """Random Forest model for stock price prediction"""
    
    def __init__(self, symbol: str = ""):
        self.symbol = symbol
        self.model = None
        self.scaler = None
        self.trained = False
        self.feature_columns = []
        self.feature_importance = {}
        
        # Model parameters from config
        self.params = config.get_model_params('random_forest')
        
        # Check if scikit-learn is available
        self.sklearn_available = self._check_sklearn()
    
    def _check_sklearn(self) -> bool:
        """Check if scikit-learn is available"""
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            from sklearn.model_selection import train_test_split
            
            return True
            
        except ImportError as e:
            logger.warning(f"Scikit-learn not available: {str(e)}")
            return False
    
    def train(self, X: pd.DataFrame, y: pd.Series, validation_split: float = 0.2, 
             scale_features: bool = True) -> Dict[str, Any]:
        """
        Train the Random Forest model
        
        Args:
            X: Feature DataFrame
            y: Target Series
            validation_split: Fraction of data to use for validation
            scale_features: Whether to scale features
        
        Returns:
            Dictionary with training results
        """
        try:
            if not self.sklearn_available:
                return {'error': 'Scikit-learn not available for Random Forest training'}
            
            performance_tracker.start_timer(f"rf_train_{self.symbol}")
            
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            from sklearn.model_selection import train_test_split
            
            if X.empty or y.empty:
                return {'error': 'Empty training data provided'}
            
            # Store feature columns
            self.feature_columns = X.columns.tolist()
            
            # Handle missing values
            X_clean = X.fillna(X.mean())
            y_clean = y.fillna(y.mean())
            
            # Scale features if requested
            if scale_features:
                self.scaler = StandardScaler()
                X_scaled = self.scaler.fit_transform(X_clean)
                X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
            else:
                X_scaled = X_clean
                self.scaler = None
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y_clean, test_size=validation_split, random_state=42, shuffle=False
            )
            
            # Initialize Random Forest model
            self.model = RandomForestRegressor(
                n_estimators=self.params['n_estimators'],
                max_depth=self.params['max_depth'],
                min_samples_split=self.params['min_samples_split'],
                min_samples_leaf=self.params['min_samples_leaf'],
                random_state=self.params['random_state'],
                n_jobs=-1,  # Use all available cores
                bootstrap=True,
                oob_score=True,  # Out-of-bag score for additional validation
                max_features='auto'
            )
            
            # Train model
            logger.info(f"Training Random Forest model for {self.symbol}")
            self.model.fit(X_train, y_train)
            
            self.trained = True
            
            # Make predictions for evaluation
            train_pred = self.model.predict(X_train)
            val_pred = self.model.predict(X_val)
            
            # Calculate metrics
            train_mae = mean_absolute_error(y_train, train_pred)
            val_mae = mean_absolute_error(y_val, val_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            train_r2 = r2_score(y_train, train_pred)
            val_r2 = r2_score(y_val, val_pred)
            
            # Calculate MAPE (Mean Absolute Percentage Error)
            train_mape = np.mean(np.abs((y_train - train_pred) / y_train)) * 100
            val_mape = np.mean(np.abs((y_val - val_pred) / y_val)) * 100
            
            # Feature importance
            self.feature_importance = dict(zip(
                self.feature_columns,
                self.model.feature_importances_
            ))
            
            # Sort feature importance
            sorted_importance = sorted(
                self.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Out-of-bag score
            oob_score = self.model.oob_score_ if hasattr(self.model, 'oob_score_') else None
            
            training_duration = performance_tracker.end_timer(f"rf_train_{self.symbol}")
            
            results = {
                'model_type': 'Random Forest',
                'training_samples': len(X_train),
                'validation_samples': len(X_val),
                'features': len(self.feature_columns),
                'training_duration': training_duration,
                'metrics': {
                    'train_mae': train_mae,
                    'val_mae': val_mae,
                    'train_rmse': train_rmse,
                    'val_rmse': val_rmse,
                    'train_r2': train_r2,
                    'val_r2': val_r2,
                    'train_mape': train_mape,
                    'val_mape': val_mape,
                    'oob_score': oob_score
                },
                'model_params': self.params,
                'feature_columns': self.feature_columns,
                'feature_importance': {
                    'top_10': sorted_importance[:10],
                    'all_features': self.feature_importance
                },
                'n_estimators': self.model.n_estimators,
                'scaled_features': scale_features
            }
            
            logger.info(f"Random Forest training completed for {self.symbol}")
            logger.info(f"Validation MAE: ${val_mae:.2f}, RMSE: ${val_rmse:.2f}, R²: {val_r2:.4f}")
            logger.info(f"OOB Score: {oob_score:.4f}" if oob_score else "OOB Score: N/A")
            
            return results
            
        except Exception as e:
            performance_tracker.end_timer(f"rf_train_{self.symbol}")
            logger.error(f"Error training Random Forest model: {str(e)}")
            return {'error': str(e)}
    
    def predict(self, X: pd.DataFrame, return_confidence: bool = True) -> Dict[str, Any]:
        """
        Make predictions using the trained Random Forest model
        
        Args:
            X: Feature DataFrame
            return_confidence: Whether to return prediction intervals
        
        Returns:
            Dictionary with predictions and metadata
        """
        try:
            if not self.trained or self.model is None:
                return {'error': 'Model not trained'}
            
            if X.empty:
                return {'error': 'Empty input data'}
            
            performance_tracker.start_timer(f"rf_predict_{self.symbol}")
            
            # Ensure feature columns match training data
            missing_cols = set(self.feature_columns) - set(X.columns)
            if missing_cols:
                return {'error': f'Missing feature columns: {missing_cols}'}
            
            # Reorder columns to match training data
            X_ordered = X[self.feature_columns]
            
            # Handle missing values
            X_clean = X_ordered.fillna(X_ordered.mean())
            
            # Scale features if scaler was used during training
            if self.scaler is not None:
                X_scaled = self.scaler.transform(X_clean)
                X_scaled = pd.DataFrame(X_scaled, columns=X_clean.columns, index=X_clean.index)
            else:
                X_scaled = X_clean
            
            # Make prediction
            prediction = self.model.predict(X_scaled)
            
            # For single prediction, extract the value
            if len(prediction) == 1:
                prediction_value = float(prediction[0])
            else:
                prediction_value = float(prediction[-1])  # Use the last prediction
            
            # Calculate confidence intervals using individual tree predictions
            confidence_intervals = None
            if return_confidence:
                confidence_intervals = self._calculate_confidence_intervals(X_scaled)
            
            prediction_duration = performance_tracker.end_timer(f"rf_predict_{self.symbol}")
            
            results = {
                'model_type': 'Random Forest',
                'prediction': prediction_value,
                'confidence_intervals': confidence_intervals,
                'input_samples': len(X),
                'prediction_duration': prediction_duration,
                'model_params': self.params,
                'n_estimators_used': self.model.n_estimators
            }
            
            logger.info(f"Random Forest prediction completed for {self.symbol}: ${prediction_value:.2f}")
            
            return results
            
        except Exception as e:
            performance_tracker.end_timer(f"rf_predict_{self.symbol}")
            logger.error(f"Error making Random Forest prediction: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_confidence_intervals(self, X: pd.DataFrame) -> Optional[Dict[str, float]]:
        """Calculate confidence intervals using individual tree predictions"""
        try:
            # Get predictions from all individual trees
            tree_predictions = np.array([
                tree.predict(X) for tree in self.model.estimators_
            ])
            
            # For single sample prediction
            if X.shape[0] == 1:
                individual_preds = tree_predictions[:, 0]
            else:
                # Use last sample for multiple samples
                individual_preds = tree_predictions[:, -1]
            
            # Calculate statistics
            mean_pred = np.mean(individual_preds)
            std_pred = np.std(individual_preds)
            
            # Calculate percentiles for confidence intervals
            lower_95 = np.percentile(individual_preds, 2.5)
            upper_95 = np.percentile(individual_preds, 97.5)
            lower_68 = np.percentile(individual_preds, 16)
            upper_68 = np.percentile(individual_preds, 84)
            
            return {
                'mean': float(mean_pred),
                'std': float(std_pred),
                'lower_95': float(lower_95),
                'upper_95': float(upper_95),
                'lower_68': float(lower_68),
                'upper_68': float(upper_68),
                'tree_std': float(std_pred)  # Standard deviation across trees
            }
            
        except Exception as e:
            logger.error(f"Error calculating confidence intervals: {str(e)}")
            return None
    
    def predict_multiple_horizons(self, X: pd.DataFrame, horizons: List[int]) -> Dict[int, Dict[str, Any]]:
        """
        Predict multiple time horizons
        
        Args:
            X: Feature DataFrame
            horizons: List of prediction horizons (days ahead)
        
        Returns:
            Dictionary mapping horizons to predictions
        """
        try:
            if not self.trained:
                return {h: {'error': 'Model not trained'} for h in horizons}
            
            results = {}
            
            for horizon in horizons:
                try:
                    # For Random Forest, we'll use the same prediction for all horizons
                    # In practice, you'd want separate models for different horizons
                    single_pred = self.predict(X, return_confidence=True)
                    
                    if 'error' not in single_pred:
                        results[horizon] = single_pred.copy()
                        results[horizon]['horizon'] = horizon
                        
                        # Adjust confidence intervals for longer horizons
                        if single_pred.get('confidence_intervals'):
                            ci = single_pred['confidence_intervals']
                            # Increase uncertainty for longer horizons
                            uncertainty_multiplier = np.sqrt(horizon)
                            
                            results[horizon]['confidence_intervals'] = {
                                'mean': ci['mean'],
                                'std': ci['std'] * uncertainty_multiplier,
                                'lower_95': ci['mean'] - (ci['mean'] - ci['lower_95']) * uncertainty_multiplier,
                                'upper_95': ci['mean'] + (ci['upper_95'] - ci['mean']) * uncertainty_multiplier,
                                'lower_68': ci['mean'] - (ci['mean'] - ci['lower_68']) * uncertainty_multiplier,
                                'upper_68': ci['mean'] + (ci['upper_68'] - ci['mean']) * uncertainty_multiplier,
                                'tree_std': ci['tree_std'] * uncertainty_multiplier
                            }
                    else:
                        results[horizon] = single_pred
                        
                except Exception as e:
                    results[horizon] = {'error': f'Error predicting horizon {horizon}: {str(e)}'}
            
            return results
            
        except Exception as e:
            logger.error(f"Error predicting multiple horizons: {str(e)}")
            return {h: {'error': str(e)} for h in horizons}
    
    def get_feature_importance(self, top_n: int = 10) -> Dict[str, Any]:
        """Get feature importance analysis"""
        try:
            if not self.trained or not self.feature_importance:
                return {'error': 'Model not trained or feature importance not available'}
            
            # Sort features by importance
            sorted_importance = sorted(
                self.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Calculate cumulative importance
            total_importance = sum(self.feature_importance.values())
            cumulative_importance = 0
            cumulative_data = []
            
            for feature, importance in sorted_importance:
                cumulative_importance += importance
                cumulative_data.append({
                    'feature': feature,
                    'importance': importance,
                    'cumulative_importance': cumulative_importance,
                    'percentage': (importance / total_importance) * 100,
                    'cumulative_percentage': (cumulative_importance / total_importance) * 100
                })
            
            return {
                'top_features': sorted_importance[:top_n],
                'all_features': sorted_importance,
                'cumulative_analysis': cumulative_data,
                'total_features': len(self.feature_importance),
                'top_10_contribution': sum([imp for _, imp in sorted_importance[:10]]) / total_importance * 100
            }
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return {'error': str(e)}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model"""
        try:
            info = {
                'model_type': 'Random Forest',
                'symbol': self.symbol,
                'trained': self.trained,
                'sklearn_available': self.sklearn_available,
                'feature_count': len(self.feature_columns),
                'feature_columns': self.feature_columns,
                'model_params': self.params,
                'scaled_features': self.scaler is not None
            }
            
            if self.model and self.trained:
                try:
                    info['model_details'] = {
                        'n_estimators': self.model.n_estimators,
                        'max_depth': self.model.max_depth,
                        'min_samples_split': self.model.min_samples_split,
                        'min_samples_leaf': self.model.min_samples_leaf,
                        'oob_score': getattr(self.model, 'oob_score_', None)
                    }
                except:
                    pass
            
            if self.feature_importance:
                # Get top 5 most important features
                sorted_importance = sorted(
                    self.feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                info['top_features'] = sorted_importance[:5]
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return {'error': str(e)}
    
    def save_model(self, filepath: str) -> bool:
        """Save the trained model"""
        try:
            if not self.trained:
                return False
            
            import pickle
            
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'feature_importance': self.feature_importance,
                'params': self.params,
                'symbol': self.symbol
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Random Forest model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving Random Forest model: {str(e)}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load a trained model"""
        try:
            import pickle
            
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data.get('model')
            self.scaler = model_data.get('scaler')
            self.feature_columns = model_data.get('feature_columns', [])
            self.feature_importance = model_data.get('feature_importance', {})
            self.params = model_data.get('params', self.params)
            self.symbol = model_data.get('symbol', self.symbol)
            
            self.trained = True
            
            logger.info(f"Random Forest model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading Random Forest model: {str(e)}")
            return False
    
    def analyze_predictions(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Analyze model predictions in detail"""
        try:
            if not self.trained:
                return {'error': 'Model not trained'}
            
            predictions = self.predict(X, return_confidence=False)
            
            if 'error' in predictions:
                return predictions
            
            # Calculate residuals
            predicted_values = np.full(len(y), predictions['prediction'])
            residuals = y - predicted_values
            
            # Residual analysis
            residual_stats = {
                'mean': float(np.mean(residuals)),
                'std': float(np.std(residuals)),
                'min': float(np.min(residuals)),
                'max': float(np.max(residuals)),
                'q25': float(np.percentile(residuals, 25)),
                'q75': float(np.percentile(residuals, 75))
            }
            
            # Prediction accuracy by ranges
            actual_ranges = pd.qcut(y, q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
            range_accuracy = {}
            
            for range_name in actual_ranges.cat.categories:
                mask = actual_ranges == range_name
                if mask.sum() > 0:
                    range_residuals = residuals[mask]
                    range_accuracy[range_name] = {
                        'count': int(mask.sum()),
                        'mean_residual': float(np.mean(range_residuals)),
                        'std_residual': float(np.std(range_residuals)),
                        'mae': float(np.mean(np.abs(range_residuals)))
                    }
            
            return {
                'residual_statistics': residual_stats,
                'accuracy_by_price_range': range_accuracy,
                'overall_mae': float(np.mean(np.abs(residuals))),
                'overall_rmse': float(np.sqrt(np.mean(residuals**2))),
                'samples_analyzed': len(y)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing predictions: {str(e)}")
            return {'error': str(e)}