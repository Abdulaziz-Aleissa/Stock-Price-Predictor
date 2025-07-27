"""
Time Series Cross-Validation Module

This module implements proper time series cross-validation techniques that respect
the temporal nature of financial data and prevent look-ahead bias.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Generator
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class TimeSeriesCV:
    """
    Advanced time series cross-validation for financial models
    """
    
    def __init__(self, 
                 n_splits: int = 5,
                 test_size: Optional[int] = None,
                 gap: int = 0,
                 max_train_size: Optional[int] = None):
        """
        Initialize time series cross-validator
        
        Args:
            n_splits: Number of splits
            test_size: Size of test set (if None, uses 1/n_splits of data)
            gap: Gap between train and test to avoid look-ahead bias
            max_train_size: Maximum size of training set
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.max_train_size = max_train_size
        
    def split(self, X: pd.DataFrame, y: pd.Series = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test splits for time series data
        """
        n_samples = len(X)
        
        if self.test_size is None:
            test_size = n_samples // (self.n_splits + 1)
        else:
            test_size = self.test_size
            
        # Calculate split points
        for i in range(self.n_splits):
            # Test set always at the end of available data for this split
            test_end = n_samples - (self.n_splits - 1 - i) * test_size
            test_start = test_end - test_size
            
            # Training set ends before test set with gap
            train_end = test_start - self.gap
            
            # Training set start
            if self.max_train_size is not None:
                train_start = max(0, train_end - self.max_train_size)
            else:
                train_start = 0
            
            # Ensure we have enough data
            if train_start >= train_end or test_start >= test_end or train_end <= 0:
                continue
                
            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)
            
            yield train_indices, test_indices
    
    def cross_validate_model(self,
                           model,
                           X: pd.DataFrame,
                           y: pd.Series,
                           scoring: str = 'mae',
                           return_predictions: bool = False) -> Dict:
        """
        Perform cross-validation with proper time series splits
        """
        try:
            scores = []
            predictions = []
            actuals = []
            feature_importances = []
            
            for fold, (train_idx, test_idx) in enumerate(self.split(X, y)):
                logger.info(f"Processing fold {fold + 1}/{self.n_splits}")
                
                # Split data
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Train model
                model_copy = self._clone_model(model)
                model_copy.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model_copy.predict(X_test)
                
                # Calculate score
                if scoring == 'mae':
                    score = mean_absolute_error(y_test, y_pred)
                elif scoring == 'mse':
                    score = mean_squared_error(y_test, y_pred)
                elif scoring == 'rmse':
                    score = np.sqrt(mean_squared_error(y_test, y_pred))
                elif scoring == 'r2':
                    score = r2_score(y_test, y_pred)
                else:
                    raise ValueError(f"Unknown scoring method: {scoring}")
                
                scores.append(score)
                
                if return_predictions:
                    predictions.extend(y_pred)
                    actuals.extend(y_test.values)
                
                # Store feature importance if available
                if hasattr(model_copy, 'feature_importances_'):
                    feature_importances.append(model_copy.feature_importances_)
            
            # Calculate statistics
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            
            result = {
                'scores': scores,
                'mean_score': mean_score,
                'std_score': std_score,
                'scoring_method': scoring,
                'n_splits': len(scores),
                'confidence_interval': {
                    'lower': mean_score - 1.96 * std_score / np.sqrt(len(scores)),
                    'upper': mean_score + 1.96 * std_score / np.sqrt(len(scores))
                }
            }
            
            if return_predictions:
                result['predictions'] = predictions
                result['actuals'] = actuals
                
                # Additional metrics if we have predictions
                if predictions and actuals:
                    result['overall_mae'] = mean_absolute_error(actuals, predictions)
                    result['overall_mse'] = mean_squared_error(actuals, predictions)
                    result['overall_r2'] = r2_score(actuals, predictions)
                    
                    # Directional accuracy
                    if len(actuals) > 1:
                        actual_directions = np.sign(np.diff(actuals))
                        pred_directions = np.sign(np.diff(predictions))
                        directional_accuracy = np.mean(actual_directions == pred_directions)
                        result['directional_accuracy'] = directional_accuracy
            
            if feature_importances:
                result['mean_feature_importance'] = np.mean(feature_importances, axis=0)
                result['std_feature_importance'] = np.std(feature_importances, axis=0)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {str(e)}")
            return {'error': str(e)}
    
    def _clone_model(self, model):
        """Clone model for cross-validation"""
        try:
            # Try sklearn clone first
            from sklearn.base import clone
            return clone(model)
        except:
            # Fallback to simple copy for custom models
            import copy
            return copy.deepcopy(model)
    
    def walk_forward_validation(self,
                              model,
                              X: pd.DataFrame,
                              y: pd.Series,
                              initial_train_size: int = 252,  # 1 year
                              step_size: int = 21,  # 1 month
                              max_iterations: int = 50) -> Dict:
        """
        Perform walk-forward validation simulating real trading conditions
        """
        try:
            if len(X) < initial_train_size + step_size:
                return {'error': 'Insufficient data for walk-forward validation'}
            
            results = {
                'predictions': [],
                'actuals': [],
                'dates': [],
                'train_sizes': [],
                'model_scores': []
            }
            
            current_start = 0
            iterations = 0
            
            while (current_start + initial_train_size + step_size < len(X) and 
                   iterations < max_iterations):
                
                # Define training and test periods
                train_end = current_start + initial_train_size
                test_start = train_end
                test_end = min(test_start + step_size, len(X))
                
                # Extract data
                X_train = X.iloc[current_start:train_end]
                y_train = y.iloc[current_start:train_end]
                X_test = X.iloc[test_start:test_end]
                y_test = y.iloc[test_start:test_end]
                
                # Train model
                model_copy = self._clone_model(model)
                model_copy.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model_copy.predict(X_test)
                
                # Store results
                results['predictions'].extend(y_pred)
                results['actuals'].extend(y_test.values)
                results['train_sizes'].append(len(X_train))
                
                # Store dates if available
                if hasattr(X_test, 'index'):
                    results['dates'].extend(X_test.index.tolist())
                else:
                    results['dates'].extend(range(test_start, test_end))
                
                # Calculate model score on test set
                test_mae = mean_absolute_error(y_test, y_pred)
                results['model_scores'].append(test_mae)
                
                # Move forward
                current_start += step_size
                iterations += 1
            
            # Calculate summary statistics
            if results['predictions']:
                overall_mae = mean_absolute_error(results['actuals'], results['predictions'])
                overall_mse = mean_squared_error(results['actuals'], results['predictions'])
                overall_r2 = r2_score(results['actuals'], results['predictions'])
                
                # Directional accuracy
                directional_accuracy = 0
                if len(results['actuals']) > 1:
                    actual_directions = np.sign(np.diff(results['actuals']))
                    pred_directions = np.sign(np.diff(results['predictions']))
                    directional_accuracy = np.mean(actual_directions == pred_directions)
                
                # Performance over time
                rolling_mae = []
                window_size = min(20, len(results['predictions']) // 4)
                
                for i in range(window_size, len(results['predictions'])):
                    window_actuals = results['actuals'][i-window_size:i]
                    window_preds = results['predictions'][i-window_size:i]
                    window_mae = mean_absolute_error(window_actuals, window_preds)
                    rolling_mae.append(window_mae)
                
                results['summary'] = {
                    'overall_mae': overall_mae,
                    'overall_mse': overall_mse,
                    'overall_r2': overall_r2,
                    'directional_accuracy': directional_accuracy,
                    'mean_model_score': np.mean(results['model_scores']),
                    'std_model_score': np.std(results['model_scores']),
                    'total_predictions': len(results['predictions']),
                    'iterations': iterations,
                    'rolling_mae_trend': np.mean(np.diff(rolling_mae)) if len(rolling_mae) > 1 else 0
                }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in walk-forward validation: {str(e)}")
            return {'error': str(e)}
    
    def seasonal_validation(self,
                          model,
                          X: pd.DataFrame,
                          y: pd.Series,
                          date_column: str = None,
                          seasons: Dict[str, List[int]] = None) -> Dict:
        """
        Validate model performance across different seasons/periods
        """
        try:
            if seasons is None:
                # Default seasons (months)
                seasons = {
                    'Q1': [1, 2, 3],     # Winter/Spring
                    'Q2': [4, 5, 6],     # Spring/Summer
                    'Q3': [7, 8, 9],     # Summer/Fall
                    'Q4': [10, 11, 12]   # Fall/Winter
                }
            
            # Extract dates
            if date_column and date_column in X.columns:
                dates = pd.to_datetime(X[date_column])
            elif hasattr(X, 'index') and isinstance(X.index, pd.DatetimeIndex):
                dates = X.index
            else:
                logger.warning("No date information found, using synthetic dates")
                dates = pd.date_range(start='2020-01-01', periods=len(X), freq='D')
            
            seasonal_results = {}
            
            for season_name, months in seasons.items():
                # Filter data for this season
                season_mask = dates.month.isin(months)
                
                if season_mask.sum() < 10:  # Need at least 10 samples
                    continue
                
                X_season = X[season_mask]
                y_season = y[season_mask]
                
                # Perform cross-validation for this season
                seasonal_cv = TimeSeriesCV(n_splits=min(3, len(X_season) // 10))
                season_results = seasonal_cv.cross_validate_model(
                    model, X_season, y_season, return_predictions=True
                )
                
                seasonal_results[season_name] = {
                    'cv_results': season_results,
                    'sample_size': len(X_season),
                    'date_range': f"{dates[season_mask].min()} to {dates[season_mask].max()}"
                }
            
            # Compare seasonal performance
            if len(seasonal_results) > 1:
                season_scores = {name: results['cv_results']['mean_score'] 
                               for name, results in seasonal_results.items() 
                               if 'error' not in results['cv_results']}
                
                if season_scores:
                    best_season = min(season_scores.keys(), key=lambda k: season_scores[k])
                    worst_season = max(season_scores.keys(), key=lambda k: season_scores[k])
                    
                    seasonal_results['comparison'] = {
                        'best_season': best_season,
                        'worst_season': worst_season,
                        'performance_gap': season_scores[worst_season] - season_scores[best_season],
                        'seasonal_stability': 1.0 / (1.0 + np.std(list(season_scores.values())))
                    }
            
            return seasonal_results
            
        except Exception as e:
            logger.error(f"Error in seasonal validation: {str(e)}")
            return {'error': str(e)}
    
    def regime_aware_validation(self,
                              model,
                              X: pd.DataFrame,
                              y: pd.Series,
                              regime_column: str = None,
                              volatility_threshold: float = 0.02) -> Dict:
        """
        Validate model performance across different market regimes
        """
        try:
            # Determine market regimes
            if regime_column and regime_column in X.columns:
                regimes = X[regime_column]
            else:
                # Create regimes based on volatility
                returns = y.pct_change().fillna(0)
                rolling_vol = returns.rolling(20).std()
                
                regimes = pd.Series('Normal', index=X.index)
                regimes[rolling_vol > volatility_threshold * 1.5] = 'High Volatility'
                regimes[rolling_vol < volatility_threshold * 0.5] = 'Low Volatility'
            
            regime_results = {}
            unique_regimes = regimes.unique()
            
            for regime in unique_regimes:
                regime_mask = regimes == regime
                
                if regime_mask.sum() < 20:  # Need sufficient samples
                    continue
                
                X_regime = X[regime_mask]
                y_regime = y[regime_mask]
                
                # Cross-validation for this regime
                regime_cv = TimeSeriesCV(n_splits=min(3, len(X_regime) // 15))
                regime_cv_results = regime_cv.cross_validate_model(
                    model, X_regime, y_regime, return_predictions=True
                )
                
                regime_results[regime] = {
                    'cv_results': regime_cv_results,
                    'sample_size': len(X_regime),
                    'regime_frequency': regime_mask.sum() / len(X)
                }
            
            # Analyze regime performance differences
            if len(regime_results) > 1:
                regime_scores = {name: results['cv_results']['mean_score'] 
                               for name, results in regime_results.items() 
                               if 'error' not in results['cv_results']}
                
                if regime_scores:
                    best_regime = min(regime_scores.keys(), key=lambda k: regime_scores[k])
                    worst_regime = max(regime_scores.keys(), key=lambda k: regime_scores[k])
                    
                    regime_results['analysis'] = {
                        'best_regime': best_regime,
                        'worst_regime': worst_regime,
                        'regime_sensitivity': regime_scores[worst_regime] - regime_scores[best_regime],
                        'regime_consistency': 1.0 / (1.0 + np.std(list(regime_scores.values())))
                    }
            
            return regime_results
            
        except Exception as e:
            logger.error(f"Error in regime-aware validation: {str(e)}")
            return {'error': str(e)}