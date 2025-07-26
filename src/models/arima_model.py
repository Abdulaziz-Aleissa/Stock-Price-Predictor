"""
ARIMA Model for Stock Price Prediction
Implements AutoRegressive Integrated Moving Average models for time series forecasting
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

class ARIMAModel:
    """ARIMA model for stock price prediction"""
    
    def __init__(self, symbol: str = ""):
        self.symbol = symbol
        self.model = None
        self.fitted_model = None
        self.trained = False
        self.training_data = None
        self.order = config.get_model_params('arima').get('order', (1, 1, 1))
        self.seasonal_order = config.get_model_params('arima').get('seasonal_order', (1, 1, 1, 12))
        self.training_history = {}
        
        # Check if required libraries are available
        self.statsmodels_available = self._check_statsmodels()
    
    def _check_statsmodels(self) -> bool:
        """Check if statsmodels is available"""
        try:
            import statsmodels.api as sm
            from statsmodels.tsa.arima.model import ARIMA
            from statsmodels.tsa.seasonal import seasonal_decompose
            from statsmodels.tsa.stattools import adfuller, kpss
            from statsmodels.stats.diagnostic import acorr_ljungbox
            
            return True
            
        except ImportError as e:
            logger.warning(f"Statsmodels not available: {str(e)}")
            return False
    
    def _check_stationarity(self, ts: pd.Series) -> Dict[str, Any]:
        """Check if time series is stationary using ADF and KPSS tests"""
        try:
            if not self.statsmodels_available:
                return {'error': 'Statsmodels not available'}
            
            from statsmodels.tsa.stattools import adfuller, kpss
            
            # Augmented Dickey-Fuller test
            adf_result = adfuller(ts.dropna(), autolag='AIC')
            adf_stationary = adf_result[1] < 0.05  # p-value < 0.05 means stationary
            
            # KPSS test
            try:
                kpss_result = kpss(ts.dropna(), regression='c')
                kpss_stationary = kpss_result[1] > 0.05  # p-value > 0.05 means stationary
            except:
                kpss_stationary = None
            
            return {
                'adf_statistic': adf_result[0],
                'adf_pvalue': adf_result[1],
                'adf_stationary': adf_stationary,
                'kpss_statistic': kpss_result[0] if kpss_stationary is not None else None,
                'kpss_pvalue': kpss_result[1] if kpss_stationary is not None else None,
                'kpss_stationary': kpss_stationary,
                'is_stationary': adf_stationary and (kpss_stationary if kpss_stationary is not None else True)
            }
            
        except Exception as e:
            logger.error(f"Error checking stationarity: {str(e)}")
            return {'error': str(e)}
    
    def _auto_arima_order(self, ts: pd.Series, max_p: int = 5, max_d: int = 2, max_q: int = 5) -> Tuple[int, int, int]:
        """Automatically determine ARIMA order using AIC/BIC"""
        try:
            if not self.statsmodels_available:
                return self.order
            
            from statsmodels.tsa.arima.model import ARIMA
            
            best_aic = np.inf
            best_order = self.order
            
            # Grid search for best parameters
            for p in range(max_p + 1):
                for d in range(max_d + 1):
                    for q in range(max_q + 1):
                        try:
                            model = ARIMA(ts, order=(p, d, q))
                            fitted_model = model.fit(method_kwargs={'warn_convergence': False})
                            
                            if fitted_model.aic < best_aic:
                                best_aic = fitted_model.aic
                                best_order = (p, d, q)
                                
                        except Exception:
                            continue
            
            logger.info(f"Auto ARIMA selected order: {best_order} (AIC: {best_aic:.2f})")
            return best_order
            
        except Exception as e:
            logger.error(f"Error in auto ARIMA: {str(e)}")
            return self.order
    
    def _decompose_series(self, ts: pd.Series) -> Optional[Dict[str, pd.Series]]:
        """Decompose time series into trend, seasonal, and residual components"""
        try:
            if not self.statsmodels_available or len(ts) < 24:
                return None
            
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            # Try additive decomposition first
            try:
                decomp = seasonal_decompose(ts, model='additive', period=12)
                return {
                    'trend': decomp.trend,
                    'seasonal': decomp.seasonal,
                    'resid': decomp.resid,
                    'observed': decomp.observed
                }
            except:
                # Fallback to multiplicative
                try:
                    decomp = seasonal_decompose(ts, model='multiplicative', period=12)
                    return {
                        'trend': decomp.trend,
                        'seasonal': decomp.seasonal,
                        'resid': decomp.resid,
                        'observed': decomp.observed
                    }
                except:
                    return None
            
        except Exception as e:
            logger.error(f"Error decomposing series: {str(e)}")
            return None
    
    def train(self, data: pd.Series, auto_order: bool = True, seasonal: bool = False) -> Dict[str, Any]:
        """
        Train the ARIMA model
        
        Args:
            data: Time series data (prices)
            auto_order: Whether to automatically determine ARIMA order
            seasonal: Whether to use seasonal ARIMA (SARIMA)
        
        Returns:
            Dictionary with training results
        """
        try:
            if not self.statsmodels_available:
                return {'error': 'Statsmodels not available for ARIMA training'}
            
            performance_tracker.start_timer(f"arima_train_{self.symbol}")
            
            from statsmodels.tsa.arima.model import ARIMA
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            
            if data.empty:
                return {'error': 'Empty training data provided'}
            
            # Store training data
            self.training_data = data.copy()
            
            # Check stationarity
            stationarity_test = self._check_stationarity(data)
            
            # Decompose series
            decomposition = self._decompose_series(data)
            
            # Auto-determine order if requested
            if auto_order:
                self.order = self._auto_arima_order(data)
            
            # Choose model type
            try:
                if seasonal and len(data) >= 24:
                    # Use SARIMA for seasonal data
                    self.model = SARIMAX(
                        data,
                        order=self.order,
                        seasonal_order=self.seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                    model_type = 'SARIMA'
                else:
                    # Use regular ARIMA
                    self.model = ARIMA(data, order=self.order)
                    model_type = 'ARIMA'
                
                # Fit the model
                logger.info(f"Training {model_type} model for {self.symbol}")
                self.fitted_model = self.model.fit(method_kwargs={'warn_convergence': False})
                
                self.trained = True
                
                # Get model diagnostics
                aic = self.fitted_model.aic
                bic = self.fitted_model.bic
                hqic = self.fitted_model.hqic
                llf = self.fitted_model.llf
                
                # Residual diagnostics
                residuals = self.fitted_model.resid
                residual_mean = np.mean(residuals)
                residual_std = np.std(residuals)
                
                # Ljung-Box test for residual autocorrelation
                try:
                    from statsmodels.stats.diagnostic import acorr_ljungbox
                    ljung_box = acorr_ljungbox(residuals, lags=10, return_df=True)
                    ljung_box_pvalue = ljung_box['lb_pvalue'].iloc[-1]
                except:
                    ljung_box_pvalue = None
                
                # In-sample predictions for evaluation
                predictions = self.fitted_model.fittedvalues
                
                # Calculate metrics
                from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                
                # Remove NaN values for metric calculation
                valid_idx = ~(np.isnan(data) | np.isnan(predictions))
                if valid_idx.sum() > 0:
                    data_clean = data[valid_idx]
                    pred_clean = predictions[valid_idx]
                    
                    mae = mean_absolute_error(data_clean, pred_clean)
                    rmse = np.sqrt(mean_squared_error(data_clean, pred_clean))
                    r2 = r2_score(data_clean, pred_clean)
                    mape = np.mean(np.abs((data_clean - pred_clean) / data_clean)) * 100
                else:
                    mae = rmse = r2 = mape = np.nan
                
                training_duration = performance_tracker.end_timer(f"arima_train_{self.symbol}")
                
                results = {
                    'model_type': model_type,
                    'order': self.order,
                    'seasonal_order': self.seasonal_order if seasonal else None,
                    'training_samples': len(data),
                    'training_duration': training_duration,
                    'model_fit': {
                        'aic': aic,
                        'bic': bic,
                        'hqic': hqic,
                        'log_likelihood': llf,
                        'converged': self.fitted_model.mle_retvals['converged'] if hasattr(self.fitted_model, 'mle_retvals') else True
                    },
                    'metrics': {
                        'mae': mae,
                        'rmse': rmse,
                        'r2': r2,
                        'mape': mape
                    },
                    'residual_diagnostics': {
                        'mean': residual_mean,
                        'std': residual_std,
                        'ljung_box_pvalue': ljung_box_pvalue
                    },
                    'stationarity_test': stationarity_test,
                    'decomposition_available': decomposition is not None
                }
                
                logger.info(f"{model_type} training completed for {self.symbol}")
                logger.info(f"AIC: {aic:.2f}, MAE: ${mae:.2f}, RMSE: ${rmse:.2f}")
                
                return results
                
            except Exception as e:
                performance_tracker.end_timer(f"arima_train_{self.symbol}")
                logger.error(f"Error fitting ARIMA model: {str(e)}")
                return {'error': f'Model fitting failed: {str(e)}'}
            
        except Exception as e:
            performance_tracker.end_timer(f"arima_train_{self.symbol}")
            logger.error(f"Error training ARIMA model: {str(e)}")
            return {'error': str(e)}
    
    def predict(self, steps: int = 1, return_confidence: bool = True, alpha: float = 0.05) -> Dict[str, Any]:
        """
        Make predictions using the trained ARIMA model
        
        Args:
            steps: Number of steps ahead to forecast
            return_confidence: Whether to return confidence intervals
            alpha: Significance level for confidence intervals (0.05 = 95% CI)
        
        Returns:
            Dictionary with predictions and metadata
        """
        try:
            if not self.trained or self.fitted_model is None:
                return {'error': 'Model not trained'}
            
            performance_tracker.start_timer(f"arima_predict_{self.symbol}")
            
            # Make forecast
            forecast_result = self.fitted_model.forecast(
                steps=steps,
                alpha=alpha if return_confidence else None
            )
            
            if steps == 1:
                prediction = float(forecast_result)
                predictions = [prediction]
            else:
                predictions = forecast_result.tolist()
                prediction = predictions[0]  # Next day prediction
            
            # Get prediction intervals if requested
            confidence_intervals = None
            if return_confidence:
                try:
                    pred_ci = self.fitted_model.get_prediction(
                        start=len(self.training_data),
                        end=len(self.training_data) + steps - 1,
                        dynamic=False
                    )
                    
                    ci_df = pred_ci.conf_int(alpha=alpha)
                    
                    if steps == 1:
                        confidence_intervals = {
                            'lower_95': float(ci_df.iloc[0, 0]),
                            'upper_95': float(ci_df.iloc[0, 1]),
                            'alpha': alpha
                        }
                        
                        # Estimate 68% CI (assuming normal distribution)
                        std_err = (confidence_intervals['upper_95'] - confidence_intervals['lower_95']) / (2 * 1.96)
                        confidence_intervals['lower_68'] = prediction - std_err
                        confidence_intervals['upper_68'] = prediction + std_err
                        confidence_intervals['std'] = std_err
                        confidence_intervals['mean'] = prediction
                    else:
                        confidence_intervals = {
                            'lower_95': ci_df.iloc[:, 0].tolist(),
                            'upper_95': ci_df.iloc[:, 1].tolist(),
                            'alpha': alpha
                        }
                        
                except Exception as e:
                    logger.warning(f"Could not calculate confidence intervals: {str(e)}")
                    confidence_intervals = None
            
            prediction_duration = performance_tracker.end_timer(f"arima_predict_{self.symbol}")
            
            results = {
                'model_type': 'ARIMA',
                'prediction': prediction,
                'predictions': predictions if steps > 1 else [prediction],
                'steps': steps,
                'confidence_intervals': confidence_intervals,
                'prediction_duration': prediction_duration,
                'order': self.order
            }
            
            logger.info(f"ARIMA prediction completed for {self.symbol}: ${prediction:.2f}")
            
            return results
            
        except Exception as e:
            performance_tracker.end_timer(f"arima_predict_{self.symbol}")
            logger.error(f"Error making ARIMA prediction: {str(e)}")
            return {'error': str(e)}
    
    def predict_multiple_horizons(self, horizons: List[int]) -> Dict[int, Dict[str, Any]]:
        """
        Predict multiple time horizons
        
        Args:
            horizons: List of prediction horizons (days ahead)
        
        Returns:
            Dictionary mapping horizons to predictions
        """
        try:
            if not self.trained:
                return {h: {'error': 'Model not trained'} for h in horizons}
            
            max_horizon = max(horizons)
            
            # Get predictions for the maximum horizon
            pred_result = self.predict(steps=max_horizon, return_confidence=True)
            
            if 'error' in pred_result:
                return {h: pred_result for h in horizons}
            
            results = {}
            predictions = pred_result['predictions']
            
            for horizon in horizons:
                if horizon <= len(predictions):
                    results[horizon] = {
                        'model_type': 'ARIMA',
                        'prediction': predictions[horizon - 1],
                        'horizon': horizon,
                        'order': self.order
                    }
                    
                    # Adjust confidence intervals for this horizon
                    if pred_result.get('confidence_intervals'):
                        ci = pred_result['confidence_intervals']
                        if isinstance(ci.get('lower_95'), list):
                            results[horizon]['confidence_intervals'] = {
                                'lower_95': ci['lower_95'][horizon - 1],
                                'upper_95': ci['upper_95'][horizon - 1],
                                'alpha': ci['alpha']
                            }
                        else:
                            # Single step prediction, scale uncertainty
                            uncertainty_multiplier = np.sqrt(horizon)
                            std_err = ci.get('std', 0)
                            mean_pred = predictions[horizon - 1]
                            
                            results[horizon]['confidence_intervals'] = {
                                'mean': mean_pred,
                                'std': std_err * uncertainty_multiplier,
                                'lower_95': mean_pred - 1.96 * std_err * uncertainty_multiplier,
                                'upper_95': mean_pred + 1.96 * std_err * uncertainty_multiplier,
                                'lower_68': mean_pred - std_err * uncertainty_multiplier,
                                'upper_68': mean_pred + std_err * uncertainty_multiplier
                            }
                else:
                    results[horizon] = {'error': f'Horizon {horizon} exceeds maximum predicted steps'}
            
            return results
            
        except Exception as e:
            logger.error(f"Error predicting multiple horizons: {str(e)}")
            return {h: {'error': str(e)} for h in horizons}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model"""
        try:
            info = {
                'model_type': 'ARIMA',
                'symbol': self.symbol,
                'trained': self.trained,
                'statsmodels_available': self.statsmodels_available,
                'order': self.order,
                'seasonal_order': self.seasonal_order
            }
            
            if self.fitted_model:
                try:
                    info['model_summary'] = {
                        'aic': self.fitted_model.aic,
                        'bic': self.fitted_model.bic,
                        'log_likelihood': self.fitted_model.llf,
                        'params_count': len(self.fitted_model.params)
                    }
                except:
                    pass
            
            if self.training_data is not None:
                info['training_data_length'] = len(self.training_data)
            
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
                'fitted_model': self.fitted_model,
                'order': self.order,
                'seasonal_order': self.seasonal_order,
                'training_data': self.training_data,
                'symbol': self.symbol
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"ARIMA model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving ARIMA model: {str(e)}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load a trained model"""
        try:
            import pickle
            
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.fitted_model = model_data.get('fitted_model')
            self.order = model_data.get('order', self.order)
            self.seasonal_order = model_data.get('seasonal_order', self.seasonal_order)
            self.training_data = model_data.get('training_data')
            self.symbol = model_data.get('symbol', self.symbol)
            
            self.trained = True
            
            logger.info(f"ARIMA model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading ARIMA model: {str(e)}")
            return False