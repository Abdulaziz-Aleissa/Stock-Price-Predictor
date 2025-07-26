"""
Prophet Model for Stock Price Prediction
Implements Facebook Prophet for time series forecasting with trend and seasonality
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import configuration and helpers
from ..utils.config import config
from ..utils.helpers import performance_tracker

logger = logging.getLogger(__name__)

class ProphetModel:
    """Prophet model for stock price prediction"""
    
    def __init__(self, symbol: str = ""):
        self.symbol = symbol
        self.model = None
        self.trained = False
        self.training_data = None
        self.future_dates = None
        
        # Model parameters from config
        self.params = config.get_model_params('prophet')
        
        # Check if Prophet is available
        self.prophet_available = self._check_prophet()
    
    def _check_prophet(self) -> bool:
        """Check if Prophet is available"""
        try:
            import prophet
            from prophet import Prophet
            from prophet.diagnostics import cross_validation, performance_metrics
            
            return True
            
        except ImportError as e:
            logger.warning(f"Prophet not available: {str(e)}")
            return False
    
    def _prepare_data(self, data: pd.Series) -> pd.DataFrame:
        """Prepare data for Prophet (requires 'ds' and 'y' columns)"""
        try:
            df = pd.DataFrame()
            df['ds'] = data.index
            df['y'] = data.values
            
            # Remove any NaN values
            df = df.dropna()
            
            # Ensure datetime index
            df['ds'] = pd.to_datetime(df['ds'])
            
            return df
            
        except Exception as e:
            logger.error(f"Error preparing data for Prophet: {str(e)}")
            return pd.DataFrame()
    
    def _add_regressors(self, df: pd.DataFrame, additional_features: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Add external regressors to the Prophet DataFrame"""
        try:
            if additional_features is None:
                return df
            
            # Align dates between main data and additional features
            aligned_features = additional_features.reindex(df['ds'])
            
            # Add numeric columns as regressors
            numeric_cols = aligned_features.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols[:5]:  # Limit to 5 additional regressors to avoid overfitting
                if not aligned_features[col].isna().all():
                    df[col] = aligned_features[col].fillna(aligned_features[col].mean())
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding regressors: {str(e)}")
            return df
    
    def train(self, data: pd.Series, additional_features: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Train the Prophet model
        
        Args:
            data: Time series data (prices)
            additional_features: Additional features to use as regressors
        
        Returns:
            Dictionary with training results
        """
        try:
            if not self.prophet_available:
                return {'error': 'Prophet not available for training'}
            
            performance_tracker.start_timer(f"prophet_train_{self.symbol}")
            
            from prophet import Prophet
            
            if data.empty:
                return {'error': 'Empty training data provided'}
            
            # Prepare data
            df = self._prepare_data(data)
            if df.empty:
                return {'error': 'Failed to prepare data for Prophet'}
            
            # Add external regressors if provided
            df = self._add_regressors(df, additional_features)
            
            # Store training data
            self.training_data = df.copy()
            
            # Initialize Prophet model
            self.model = Prophet(
                yearly_seasonality=self.params.get('yearly_seasonality', True),
                weekly_seasonality=self.params.get('weekly_seasonality', True),
                daily_seasonality=self.params.get('daily_seasonality', False),
                changepoint_prior_scale=self.params.get('changepoint_prior_scale', 0.05),
                seasonality_prior_scale=10.0,
                holidays_prior_scale=10.0,
                seasonality_mode='multiplicative',
                interval_width=0.80
            )
            
            # Add additional regressors to model
            regressor_columns = [col for col in df.columns if col not in ['ds', 'y']]
            for col in regressor_columns:
                self.model.add_regressor(col)
            
            # Fit the model
            logger.info(f"Training Prophet model for {self.symbol}")
            self.model.fit(df)
            
            self.trained = True
            
            # Make in-sample predictions for evaluation
            forecast = self.model.predict(df)
            
            # Calculate metrics
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            
            actual = df['y'].values
            predicted = forecast['yhat'].values
            
            mae = mean_absolute_error(actual, predicted)
            rmse = np.sqrt(mean_squared_error(actual, predicted))
            r2 = r2_score(actual, predicted)
            mape = np.mean(np.abs((actual - predicted) / actual)) * 100
            
            # Model diagnostics
            try:
                # Extract trend and seasonality components
                trend = forecast['trend'].values
                weekly_seasonality = forecast.get('weekly', np.zeros(len(forecast)))
                yearly_seasonality = forecast.get('yearly', np.zeros(len(forecast)))
                
                # Calculate component importance
                trend_importance = np.std(trend) / np.std(actual) if np.std(actual) > 0 else 0
                weekly_importance = np.std(weekly_seasonality) / np.std(actual) if np.std(actual) > 0 else 0
                yearly_importance = np.std(yearly_seasonality) / np.std(actual) if np.std(actual) > 0 else 0
                
                component_importance = {
                    'trend': float(trend_importance),
                    'weekly': float(weekly_importance),
                    'yearly': float(yearly_importance)
                }
                
            except Exception as e:
                logger.warning(f"Could not calculate component importance: {str(e)}")
                component_importance = {}
            
            training_duration = performance_tracker.end_timer(f"prophet_train_{self.symbol}")
            
            results = {
                'model_type': 'Prophet',
                'training_samples': len(df),
                'training_duration': training_duration,
                'features_used': len(regressor_columns),
                'regressor_columns': regressor_columns,
                'metrics': {
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'mape': mape
                },
                'model_params': self.params,
                'component_importance': component_importance,
                'changepoints_detected': len(self.model.changepoints) if hasattr(self.model, 'changepoints') else 0
            }
            
            logger.info(f"Prophet training completed for {self.symbol}")
            logger.info(f"MAE: ${mae:.2f}, RMSE: ${rmse:.2f}, R²: {r2:.4f}")
            
            return results
            
        except Exception as e:
            performance_tracker.end_timer(f"prophet_train_{self.symbol}")
            logger.error(f"Error training Prophet model: {str(e)}")
            return {'error': str(e)}
    
    def predict(self, steps: int = 1, return_confidence: bool = True, 
               additional_features: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Make predictions using the trained Prophet model
        
        Args:
            steps: Number of steps ahead to forecast
            return_confidence: Whether to return confidence intervals
            additional_features: Additional features for prediction
        
        Returns:
            Dictionary with predictions and metadata
        """
        try:
            if not self.trained or self.model is None:
                return {'error': 'Model not trained'}
            
            performance_tracker.start_timer(f"prophet_predict_{self.symbol}")
            
            # Create future dates
            last_date = self.training_data['ds'].max()
            future_dates = []
            
            for i in range(1, steps + 1):
                next_date = last_date + timedelta(days=i)
                # Skip weekends for stock predictions
                while next_date.weekday() > 4:
                    next_date += timedelta(days=1)
                future_dates.append(next_date)
            
            # Create future DataFrame
            future_df = pd.DataFrame({'ds': future_dates})
            
            # Add regressors if provided
            if additional_features is not None:
                regressor_columns = [col for col in self.training_data.columns if col not in ['ds', 'y']]
                
                for col in regressor_columns:
                    if col in additional_features.columns:
                        # Use the latest available value for future predictions
                        latest_value = additional_features[col].dropna().iloc[-1] if not additional_features[col].dropna().empty else 0
                        future_df[col] = latest_value
                    else:
                        # Use mean from training data
                        future_df[col] = self.training_data[col].mean()
            else:
                # Add regressor columns with mean values from training
                regressor_columns = [col for col in self.training_data.columns if col not in ['ds', 'y']]
                for col in regressor_columns:
                    future_df[col] = self.training_data[col].mean()
            
            # Make prediction
            forecast = self.model.predict(future_df)
            
            # Extract predictions
            predictions = forecast['yhat'].tolist()
            prediction = predictions[0] if predictions else 0.0
            
            # Extract confidence intervals
            confidence_intervals = None
            if return_confidence:
                try:
                    lower_ci = forecast['yhat_lower'].tolist()
                    upper_ci = forecast['yhat_upper'].tolist()
                    
                    if steps == 1:
                        confidence_intervals = {
                            'mean': prediction,
                            'lower_95': lower_ci[0] if lower_ci else prediction * 0.95,
                            'upper_95': upper_ci[0] if upper_ci else prediction * 1.05,
                            'lower_68': lower_ci[0] + (upper_ci[0] - lower_ci[0]) * 0.16 if lower_ci and upper_ci else prediction * 0.98,
                            'upper_68': upper_ci[0] - (upper_ci[0] - lower_ci[0]) * 0.16 if lower_ci and upper_ci else prediction * 1.02,
                            'std': (upper_ci[0] - lower_ci[0]) / 4 if lower_ci and upper_ci else abs(prediction) * 0.05
                        }
                    else:
                        confidence_intervals = {
                            'lower_95': lower_ci,
                            'upper_95': upper_ci,
                            'predictions': predictions
                        }
                        
                except Exception as e:
                    logger.warning(f"Could not extract confidence intervals: {str(e)}")
                    confidence_intervals = None
            
            # Extract trend and seasonality components for the prediction
            components = {}
            try:
                if 'trend' in forecast.columns:
                    components['trend'] = forecast['trend'].iloc[0] if steps == 1 else forecast['trend'].tolist()
                if 'weekly' in forecast.columns:
                    components['weekly'] = forecast['weekly'].iloc[0] if steps == 1 else forecast['weekly'].tolist()
                if 'yearly' in forecast.columns:
                    components['yearly'] = forecast['yearly'].iloc[0] if steps == 1 else forecast['yearly'].tolist()
            except:
                pass
            
            prediction_duration = performance_tracker.end_timer(f"prophet_predict_{self.symbol}")
            
            results = {
                'model_type': 'Prophet',
                'prediction': float(prediction),
                'predictions': predictions if steps > 1 else [prediction],
                'steps': steps,
                'confidence_intervals': confidence_intervals,
                'components': components,
                'prediction_dates': [date.strftime('%Y-%m-%d') for date in future_dates],
                'prediction_duration': prediction_duration,
                'model_params': self.params
            }
            
            logger.info(f"Prophet prediction completed for {self.symbol}: ${prediction:.2f}")
            
            return results
            
        except Exception as e:
            performance_tracker.end_timer(f"prophet_predict_{self.symbol}")
            logger.error(f"Error making Prophet prediction: {str(e)}")
            return {'error': str(e)}
    
    def predict_multiple_horizons(self, horizons: List[int], 
                                 additional_features: Optional[pd.DataFrame] = None) -> Dict[int, Dict[str, Any]]:
        """
        Predict multiple time horizons
        
        Args:
            horizons: List of prediction horizons (days ahead)
            additional_features: Additional features for prediction
        
        Returns:
            Dictionary mapping horizons to predictions
        """
        try:
            if not self.trained:
                return {h: {'error': 'Model not trained'} for h in horizons}
            
            max_horizon = max(horizons)
            
            # Get predictions for the maximum horizon
            pred_result = self.predict(
                steps=max_horizon,
                return_confidence=True,
                additional_features=additional_features
            )
            
            if 'error' in pred_result:
                return {h: pred_result for h in horizons}
            
            results = {}
            predictions = pred_result['predictions']
            prediction_dates = pred_result['prediction_dates']
            
            for horizon in horizons:
                if horizon <= len(predictions):
                    results[horizon] = {
                        'model_type': 'Prophet',
                        'prediction': predictions[horizon - 1],
                        'horizon': horizon,
                        'prediction_date': prediction_dates[horizon - 1],
                        'model_params': self.params
                    }
                    
                    # Extract components for this horizon
                    if pred_result.get('components'):
                        components = pred_result['components']
                        results[horizon]['components'] = {}
                        
                        for comp_name, comp_values in components.items():
                            if isinstance(comp_values, list) and horizon <= len(comp_values):
                                results[horizon]['components'][comp_name] = comp_values[horizon - 1]
                            elif not isinstance(comp_values, list):
                                results[horizon]['components'][comp_name] = comp_values
                    
                    # Extract confidence intervals for this horizon
                    if pred_result.get('confidence_intervals'):
                        ci = pred_result['confidence_intervals']
                        if isinstance(ci.get('lower_95'), list):
                            results[horizon]['confidence_intervals'] = {
                                'lower_95': ci['lower_95'][horizon - 1],
                                'upper_95': ci['upper_95'][horizon - 1],
                                'mean': predictions[horizon - 1]
                            }
                            
                            # Estimate std and 68% CI
                            std_est = (ci['upper_95'][horizon - 1] - ci['lower_95'][horizon - 1]) / 4
                            results[horizon]['confidence_intervals'].update({
                                'std': std_est,
                                'lower_68': predictions[horizon - 1] - std_est,
                                'upper_68': predictions[horizon - 1] + std_est
                            })
                        else:
                            # Single horizon prediction case handled elsewhere
                            pass
                else:
                    results[horizon] = {'error': f'Horizon {horizon} exceeds maximum predicted steps'}
            
            return results
            
        except Exception as e:
            logger.error(f"Error predicting multiple horizons: {str(e)}")
            return {h: {'error': str(e)} for h in horizons}
    
    def cross_validate(self, initial_days: int = 365, period_days: int = 30, 
                      horizon_days: int = 30) -> Optional[Dict[str, Any]]:
        """
        Perform time series cross-validation
        
        Args:
            initial_days: Initial training period in days
            period_days: Period between cutoff dates in days
            horizon_days: Forecast horizon in days
        
        Returns:
            Cross-validation results
        """
        try:
            if not self.trained or not self.prophet_available:
                return None
            
            from prophet.diagnostics import cross_validation, performance_metrics
            
            # Perform cross-validation
            cv_results = cross_validation(
                self.model,
                initial=f'{initial_days} days',
                period=f'{period_days} days',
                horizon=f'{horizon_days} days'
            )
            
            # Calculate performance metrics
            metrics = performance_metrics(cv_results)
            
            return {
                'cv_results': cv_results.to_dict(),
                'metrics': metrics.to_dict(),
                'mean_mae': metrics['mae'].mean(),
                'mean_mape': metrics['mape'].mean(),
                'mean_rmse': metrics['rmse'].mean()
            }
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {str(e)}")
            return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model"""
        try:
            info = {
                'model_type': 'Prophet',
                'symbol': self.symbol,
                'trained': self.trained,
                'prophet_available': self.prophet_available,
                'model_params': self.params
            }
            
            if self.training_data is not None:
                info['training_data_length'] = len(self.training_data)
                info['date_range'] = {
                    'start': self.training_data['ds'].min().strftime('%Y-%m-%d'),
                    'end': self.training_data['ds'].max().strftime('%Y-%m-%d')
                }
            
            if self.model and self.trained:
                try:
                    info['seasonalities'] = list(self.model.seasonalities.keys()) if hasattr(self.model, 'seasonalities') else []
                    info['regressors'] = list(self.model.extra_regressors.keys()) if hasattr(self.model, 'extra_regressors') else []
                    info['changepoints_count'] = len(self.model.changepoints) if hasattr(self.model, 'changepoints') else 0
                except:
                    pass
            
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
                'params': self.params,
                'training_data': self.training_data,
                'symbol': self.symbol
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Prophet model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving Prophet model: {str(e)}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load a trained model"""
        try:
            import pickle
            
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data.get('model')
            self.params = model_data.get('params', self.params)
            self.training_data = model_data.get('training_data')
            self.symbol = model_data.get('symbol', self.symbol)
            
            self.trained = True
            
            logger.info(f"Prophet model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading Prophet model: {str(e)}")
            return False