"""
Model Evaluator for Stock Price Prediction
Handles model comparison, evaluation, and automatic selection
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import models
from .lstm_model import LSTMModel
from .gru_model import GRUModel
from .arima_model import ARIMAModel
from .prophet_model import ProphetModel
from .random_forest_model import RandomForestModel

# Import configuration and helpers
from ..utils.config import config
from ..utils.helpers import performance_tracker

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Model evaluation and comparison system"""
    
    def __init__(self, symbol: str = ""):
        self.symbol = symbol
        self.models = {}
        self.evaluation_results = {}
        self.best_model = None
        self.ensemble_weights = {}
        
        # Available model types
        self.available_models = {
            'lstm': LSTMModel,
            'gru': GRUModel,
            'arima': ARIMAModel,
            'prophet': ProphetModel,
            'random_forest': RandomForestModel
        }
        
        # Evaluation metrics weights for model selection
        self.metric_weights = {
            'mae': 0.3,
            'rmse': 0.25,
            'r2': 0.25,
            'mape': 0.15,
            'stability': 0.05  # Model stability across different data splits
        }
    
    def train_all_models(self, X: pd.DataFrame, y: pd.Series, 
                        models_to_train: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Train all available models
        
        Args:
            X: Feature DataFrame
            y: Target Series
            models_to_train: List of model names to train (if None, train all)
        
        Returns:
            Dictionary with training results for all models
        """
        try:
            if models_to_train is None:
                models_to_train = list(self.available_models.keys())
            
            performance_tracker.start_timer(f"train_all_models_{self.symbol}")
            
            training_results = {}
            
            for model_name in models_to_train:
                if model_name not in self.available_models:
                    logger.warning(f"Unknown model type: {model_name}")
                    continue
                
                try:
                    logger.info(f"Training {model_name.upper()} model for {self.symbol}")
                    
                    # Initialize model
                    model_class = self.available_models[model_name]
                    model = model_class(symbol=self.symbol)
                    
                    # Train model based on type
                    if model_name in ['lstm', 'gru', 'random_forest']:
                        # These models use feature DataFrame
                        result = model.train(X, y)
                    elif model_name == 'arima':
                        # ARIMA uses only the target series
                        result = model.train(y)
                    elif model_name == 'prophet':
                        # Prophet uses target series, can optionally use features
                        result = model.train(y, additional_features=X)
                    else:
                        result = {'error': f'Unknown training method for {model_name}'}
                    
                    if 'error' not in result:
                        self.models[model_name] = model
                        training_results[model_name] = result
                        logger.info(f"Successfully trained {model_name.upper()} model")
                    else:
                        logger.error(f"Failed to train {model_name}: {result['error']}")
                        training_results[model_name] = result
                        
                except Exception as e:
                    error_msg = f"Error training {model_name}: {str(e)}"
                    logger.error(error_msg)
                    training_results[model_name] = {'error': error_msg}
            
            total_duration = performance_tracker.end_timer(f"train_all_models_{self.symbol}")
            
            # Store training results
            self.evaluation_results = training_results
            
            summary = {
                'symbol': self.symbol,
                'total_training_duration': total_duration,
                'models_attempted': len(models_to_train),
                'models_successful': len([r for r in training_results.values() if 'error' not in r]),
                'models_failed': len([r for r in training_results.values() if 'error' in r]),
                'training_results': training_results
            }
            
            logger.info(f"Model training completed for {self.symbol}: "
                       f"{summary['models_successful']}/{summary['models_attempted']} successful")
            
            return summary
            
        except Exception as e:
            performance_tracker.end_timer(f"train_all_models_{self.symbol}")
            logger.error(f"Error in train_all_models: {str(e)}")
            return {'error': str(e)}
    
    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate all trained models on test data
        
        Args:
            X_test: Test feature DataFrame
            y_test: Test target Series
        
        Returns:
            Dictionary with evaluation results
        """
        try:
            if not self.models:
                return {'error': 'No trained models available for evaluation'}
            
            performance_tracker.start_timer(f"evaluate_models_{self.symbol}")
            
            evaluation_results = {}
            
            for model_name, model in self.models.items():
                try:
                    logger.info(f"Evaluating {model_name.upper()} model")
                    
                    # Make predictions based on model type
                    if model_name in ['lstm', 'gru', 'random_forest']:
                        predictions_result = model.predict(X_test, return_confidence=True)
                    elif model_name == 'arima':
                        predictions_result = model.predict(steps=len(y_test), return_confidence=True)
                    elif model_name == 'prophet':
                        predictions_result = model.predict(steps=len(y_test), return_confidence=True)
                    else:
                        predictions_result = {'error': f'Unknown prediction method for {model_name}'}
                    
                    if 'error' in predictions_result:
                        evaluation_results[model_name] = predictions_result
                        continue
                    
                    # Extract predictions
                    if 'predictions' in predictions_result and len(predictions_result['predictions']) > 1:
                        predictions = np.array(predictions_result['predictions'][:len(y_test)])
                    else:
                        # Single prediction, replicate for evaluation
                        pred_value = predictions_result.get('prediction', 0)
                        predictions = np.full(len(y_test), pred_value)
                    
                    # Calculate evaluation metrics
                    metrics = self._calculate_metrics(y_test.values, predictions)
                    
                    # Add model-specific information
                    evaluation_results[model_name] = {
                        'metrics': metrics,
                        'predictions': predictions.tolist(),
                        'model_type': predictions_result.get('model_type', model_name),
                        'confidence_intervals': predictions_result.get('confidence_intervals'),
                        'prediction_duration': predictions_result.get('prediction_duration', 0)
                    }
                    
                    logger.info(f"{model_name.upper()} evaluation - MAE: ${metrics['mae']:.2f}, "
                               f"RMSE: ${metrics['rmse']:.2f}, R²: {metrics['r2']:.4f}")
                    
                except Exception as e:
                    error_msg = f"Error evaluating {model_name}: {str(e)}"
                    logger.error(error_msg)
                    evaluation_results[model_name] = {'error': error_msg}
            
            # Calculate model rankings
            rankings = self._rank_models(evaluation_results)
            
            # Select best model
            self.best_model = self._select_best_model(evaluation_results, rankings)
            
            # Calculate ensemble weights
            self.ensemble_weights = self._calculate_ensemble_weights(evaluation_results)
            
            total_duration = performance_tracker.end_timer(f"evaluate_models_{self.symbol}")
            
            summary = {
                'symbol': self.symbol,
                'evaluation_duration': total_duration,
                'models_evaluated': len([r for r in evaluation_results.values() if 'error' not in r]),
                'evaluation_results': evaluation_results,
                'model_rankings': rankings,
                'best_model': self.best_model,
                'ensemble_weights': self.ensemble_weights,
                'test_samples': len(y_test)
            }
            
            if self.best_model:
                logger.info(f"Best model for {self.symbol}: {self.best_model['name']} "
                           f"(Score: {self.best_model['score']:.4f})")
            
            return summary
            
        except Exception as e:
            performance_tracker.end_timer(f"evaluate_models_{self.symbol}")
            logger.error(f"Error in evaluate_models: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        try:
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            
            # Remove any NaN values
            mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            y_true_clean = y_true[mask]
            y_pred_clean = y_pred[mask]
            
            if len(y_true_clean) == 0:
                return {'mae': np.inf, 'rmse': np.inf, 'r2': -np.inf, 'mape': np.inf}
            
            # Calculate metrics
            mae = mean_absolute_error(y_true_clean, y_pred_clean)
            mse = mean_squared_error(y_true_clean, y_pred_clean)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true_clean, y_pred_clean)
            
            # MAPE (Mean Absolute Percentage Error)
            mape = np.mean(np.abs((y_true_clean - y_pred_clean) / np.maximum(y_true_clean, 1e-8))) * 100
            
            # Directional accuracy (what percentage of price movements are predicted correctly)
            if len(y_true_clean) > 1:
                true_directions = np.diff(y_true_clean) > 0
                pred_directions = np.diff(y_pred_clean) > 0
                directional_accuracy = np.mean(true_directions == pred_directions) * 100
            else:
                directional_accuracy = 50.0  # Random guess baseline
            
            # Maximum error
            max_error = np.max(np.abs(y_true_clean - y_pred_clean))
            
            return {
                'mae': float(mae),
                'mse': float(mse),
                'rmse': float(rmse),
                'r2': float(r2),
                'mape': float(mape),
                'directional_accuracy': float(directional_accuracy),
                'max_error': float(max_error),
                'samples': len(y_true_clean)
            }
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {'mae': np.inf, 'rmse': np.inf, 'r2': -np.inf, 'mape': np.inf}
    
    def _rank_models(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Rank models based on performance metrics"""
        try:
            valid_results = {
                name: result for name, result in evaluation_results.items()
                if 'error' not in result and 'metrics' in result
            }
            
            if not valid_results:
                return {}
            
            # Extract metrics for ranking
            model_scores = {}
            
            for model_name, result in valid_results.items():
                metrics = result['metrics']
                
                # Calculate composite score (lower is better for MAE, RMSE, MAPE; higher for R²)
                # Normalize metrics to 0-1 scale
                mae_score = 1 / (1 + metrics.get('mae', np.inf))
                rmse_score = 1 / (1 + metrics.get('rmse', np.inf))
                r2_score = max(0, min(1, (metrics.get('r2', -np.inf) + 1) / 2))
                mape_score = 1 / (1 + metrics.get('mape', np.inf) / 100)
                
                # Weighted composite score
                composite_score = (
                    self.metric_weights['mae'] * mae_score +
                    self.metric_weights['rmse'] * rmse_score +
                    self.metric_weights['r2'] * r2_score +
                    self.metric_weights['mape'] * mape_score
                )
                
                model_scores[model_name] = {
                    'composite_score': composite_score,
                    'individual_scores': {
                        'mae_score': mae_score,
                        'rmse_score': rmse_score,
                        'r2_score': r2_score,
                        'mape_score': mape_score
                    },
                    'raw_metrics': metrics
                }
            
            # Sort by composite score (descending)
            ranked_models = sorted(
                model_scores.items(),
                key=lambda x: x[1]['composite_score'],
                reverse=True
            )
            
            # Create ranking dictionary
            rankings = {}
            for i, (model_name, scores) in enumerate(ranked_models):
                rankings[model_name] = {
                    'rank': i + 1,
                    'score': scores['composite_score'],
                    'individual_scores': scores['individual_scores'],
                    'metrics': scores['raw_metrics']
                }
            
            return rankings
            
        except Exception as e:
            logger.error(f"Error ranking models: {str(e)}")
            return {}
    
    def _select_best_model(self, evaluation_results: Dict[str, Any], 
                          rankings: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Select the best performing model"""
        try:
            if not rankings:
                return None
            
            # Get the top-ranked model
            best_model_name = min(rankings.keys(), key=lambda x: rankings[x]['rank'])
            best_ranking = rankings[best_model_name]
            
            return {
                'name': best_model_name,
                'model': self.models.get(best_model_name),
                'rank': best_ranking['rank'],
                'score': best_ranking['score'],
                'metrics': best_ranking['metrics']
            }
            
        except Exception as e:
            logger.error(f"Error selecting best model: {str(e)}")
            return None
    
    def _calculate_ensemble_weights(self, evaluation_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate weights for ensemble predictions based on model performance"""
        try:
            valid_results = {
                name: result for name, result in evaluation_results.items()
                if 'error' not in result and 'metrics' in result
            }
            
            if not valid_results:
                return {}
            
            # Calculate inverse error weights (better models get higher weights)
            weights = {}
            total_weight = 0
            
            for model_name, result in valid_results.items():
                metrics = result['metrics']
                
                # Use inverse of MAE as weight (add small constant to avoid division by zero)
                mae = metrics.get('mae', np.inf)
                weight = 1 / (mae + 1e-6) if mae != np.inf else 0
                
                weights[model_name] = weight
                total_weight += weight
            
            # Normalize weights to sum to 1
            if total_weight > 0:
                for model_name in weights:
                    weights[model_name] /= total_weight
            
            return weights
            
        except Exception as e:
            logger.error(f"Error calculating ensemble weights: {str(e)}")
            return {}
    
    def predict_ensemble(self, X: pd.DataFrame, horizons: List[int] = [1]) -> Dict[str, Any]:
        """
        Make ensemble predictions using all trained models
        
        Args:
            X: Feature DataFrame
            horizons: List of prediction horizons
        
        Returns:
            Dictionary with ensemble predictions
        """
        try:
            if not self.models or not self.ensemble_weights:
                return {'error': 'Models not trained or ensemble weights not calculated'}
            
            performance_tracker.start_timer(f"ensemble_predict_{self.symbol}")
            
            ensemble_predictions = {}
            
            for horizon in horizons:
                horizon_predictions = []
                horizon_weights = []
                model_predictions = {}
                
                for model_name, model in self.models.items():
                    try:
                        # Get prediction based on model type
                        if model_name in ['lstm', 'gru', 'random_forest']:
                            pred_result = model.predict_multiple_horizons(X, [horizon])
                            pred = pred_result.get(horizon, {}).get('prediction')
                        elif model_name == 'arima':
                            pred_result = model.predict_multiple_horizons([horizon])
                            pred = pred_result.get(horizon, {}).get('prediction')
                        elif model_name == 'prophet':
                            pred_result = model.predict_multiple_horizons([horizon])
                            pred = pred_result.get(horizon, {}).get('prediction')
                        else:
                            pred = None
                        
                        if pred is not None and not np.isnan(pred):
                            weight = self.ensemble_weights.get(model_name, 0)
                            horizon_predictions.append(pred)
                            horizon_weights.append(weight)
                            model_predictions[model_name] = {
                                'prediction': pred,
                                'weight': weight
                            }
                    
                    except Exception as e:
                        logger.warning(f"Error getting prediction from {model_name}: {str(e)}")
                        continue
                
                if horizon_predictions:
                    # Calculate weighted ensemble prediction
                    total_weight = sum(horizon_weights)
                    if total_weight > 0:
                        weighted_pred = sum(p * w for p, w in zip(horizon_predictions, horizon_weights)) / total_weight
                    else:
                        weighted_pred = np.mean(horizon_predictions)
                    
                    # Calculate prediction variance (uncertainty)
                    pred_variance = np.var(horizon_predictions) if len(horizon_predictions) > 1 else 0
                    pred_std = np.sqrt(pred_variance)
                    
                    ensemble_predictions[horizon] = {
                        'ensemble_prediction': float(weighted_pred),
                        'individual_predictions': model_predictions,
                        'prediction_std': float(pred_std),
                        'confidence_intervals': {
                            'lower_95': float(weighted_pred - 1.96 * pred_std),
                            'upper_95': float(weighted_pred + 1.96 * pred_std),
                            'lower_68': float(weighted_pred - pred_std),
                            'upper_68': float(weighted_pred + pred_std)
                        },
                        'models_used': len(horizon_predictions),
                        'horizon': horizon
                    }
                else:
                    ensemble_predictions[horizon] = {'error': f'No valid predictions for horizon {horizon}'}
            
            prediction_duration = performance_tracker.end_timer(f"ensemble_predict_{self.symbol}")
            
            result = {
                'symbol': self.symbol,
                'ensemble_predictions': ensemble_predictions,
                'ensemble_weights': self.ensemble_weights,
                'prediction_duration': prediction_duration,
                'models_available': list(self.models.keys())
            }
            
            # Log best prediction
            if 1 in ensemble_predictions and 'ensemble_prediction' in ensemble_predictions[1]:
                next_day_pred = ensemble_predictions[1]['ensemble_prediction']
                logger.info(f"Ensemble prediction for {self.symbol} (1 day): ${next_day_pred:.2f}")
            
            return result
            
        except Exception as e:
            performance_tracker.end_timer(f"ensemble_predict_{self.symbol}")
            logger.error(f"Error in ensemble prediction: {str(e)}")
            return {'error': str(e)}
    
    def get_model_comparison(self) -> Dict[str, Any]:
        """Get detailed comparison of all models"""
        try:
            if not self.evaluation_results:
                return {'error': 'No evaluation results available'}
            
            comparison = {
                'symbol': self.symbol,
                'models_compared': len(self.models),
                'evaluation_summary': {},
                'performance_ranking': [],
                'best_model': self.best_model,
                'ensemble_available': len(self.ensemble_weights) > 0
            }
            
            # Create performance summary
            for model_name, model in self.models.items():
                model_info = model.get_model_info()
                
                comparison['evaluation_summary'][model_name] = {
                    'model_type': model_info.get('model_type', model_name),
                    'trained': model_info.get('trained', False),
                    'available': True
                }
                
                # Add training results if available
                if model_name in self.evaluation_results:
                    training_result = self.evaluation_results[model_name]
                    if 'error' not in training_result:
                        comparison['evaluation_summary'][model_name]['training_metrics'] = training_result.get('metrics', {})
                        comparison['evaluation_summary'][model_name]['training_duration'] = training_result.get('training_duration', 0)
            
            # Add ranking information
            if hasattr(self, 'model_rankings') and self.model_rankings:
                for model_name, ranking in self.model_rankings.items():
                    comparison['performance_ranking'].append({
                        'model': model_name,
                        'rank': ranking['rank'],
                        'score': ranking['score'],
                        'metrics': ranking['metrics']
                    })
                
                # Sort by rank
                comparison['performance_ranking'].sort(key=lambda x: x['rank'])
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error getting model comparison: {str(e)}")
            return {'error': str(e)}
    
    def save_all_models(self, base_path: str) -> Dict[str, bool]:
        """Save all trained models"""
        try:
            save_results = {}
            
            for model_name, model in self.models.items():
                try:
                    filepath = f"{base_path}/{self.symbol}_{model_name}_model.pkl"
                    success = model.save_model(filepath)
                    save_results[model_name] = success
                    
                    if success:
                        logger.info(f"Saved {model_name} model to {filepath}")
                    else:
                        logger.error(f"Failed to save {model_name} model")
                        
                except Exception as e:
                    logger.error(f"Error saving {model_name} model: {str(e)}")
                    save_results[model_name] = False
            
            return save_results
            
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            return {}
    
    def load_all_models(self, base_path: str) -> Dict[str, bool]:
        """Load all saved models"""
        try:
            load_results = {}
            
            for model_name, model_class in self.available_models.items():
                try:
                    filepath = f"{base_path}/{self.symbol}_{model_name}_model.pkl"
                    model = model_class(symbol=self.symbol)
                    success = model.load_model(filepath)
                    
                    if success:
                        self.models[model_name] = model
                        logger.info(f"Loaded {model_name} model from {filepath}")
                    
                    load_results[model_name] = success
                    
                except Exception as e:
                    logger.error(f"Error loading {model_name} model: {str(e)}")
                    load_results[model_name] = False
            
            return load_results
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return {}