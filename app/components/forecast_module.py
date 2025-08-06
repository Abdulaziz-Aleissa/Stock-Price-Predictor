"""
Forecast Module
Handle time series forecasting
"""

from app.utils.time_series_forecasting import ts_forecaster
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ForecastManager:
    """Handle time series forecasting operations"""
    
    def __init__(self):
        self.ts_forecaster = ts_forecaster
    
    def generate_price_forecast(self, symbol: str, 
                               forecast_days: int = 30, 
                               model_type: str = 'arima') -> Optional[Dict]:
        """Generate price forecast for a stock"""
        try:
            return self.ts_forecaster.forecast_price(
                symbol, forecast_days, model_type
            )
        except Exception as e:
            logger.error(f"Error generating price forecast for {symbol}: {str(e)}")
            return None
    
    def generate_volatility_forecast(self, symbol: str, 
                                   forecast_days: int = 30) -> Optional[Dict]:
        """Generate volatility forecast"""
        try:
            return self.ts_forecaster.forecast_volatility(symbol, forecast_days)
        except Exception as e:
            logger.error(f"Error generating volatility forecast for {symbol}: {str(e)}")
            return None
    
    def generate_trend_analysis(self, symbol: str, 
                               analysis_period: int = 252) -> Optional[Dict]:
        """Generate trend analysis"""
        try:
            return self.ts_forecaster.trend_analysis(symbol, analysis_period)
        except Exception as e:
            logger.error(f"Error generating trend analysis for {symbol}: {str(e)}")
            return None
    
    def seasonal_decomposition(self, symbol: str, 
                              period: int = 252) -> Optional[Dict]:
        """Perform seasonal decomposition of time series"""
        try:
            return self.ts_forecaster.seasonal_decompose(symbol, period)
        except Exception as e:
            logger.error(f"Error performing seasonal decomposition for {symbol}: {str(e)}")
            return None
    
    def forecast_multiple_stocks(self, symbols: List[str], 
                                forecast_days: int = 30) -> Dict:
        """Generate forecasts for multiple stocks"""
        try:
            forecasts = {}
            
            for symbol in symbols:
                forecast = self.generate_price_forecast(symbol, forecast_days)
                if forecast:
                    forecasts[symbol] = forecast
            
            return forecasts
            
        except Exception as e:
            logger.error(f"Error generating multiple forecasts: {str(e)}")
            return {}
    
    def compare_forecast_models(self, symbol: str, 
                               models: List[str] = ['arima', 'lstm', 'prophet']) -> Dict:
        """Compare different forecasting models"""
        try:
            model_results = {}
            
            for model in models:
                try:
                    forecast = self.generate_price_forecast(symbol, 30, model)
                    if forecast:
                        model_results[model] = forecast
                except Exception as model_error:
                    logger.warning(f"Model {model} failed for {symbol}: {str(model_error)}")
                    model_results[model] = None
            
            # Determine best model based on accuracy metrics
            best_model = self._select_best_model(model_results)
            
            return {
                'symbol': symbol,
                'model_results': model_results,
                'best_model': best_model,
                'comparison_metrics': self._calculate_comparison_metrics(model_results)
            }
            
        except Exception as e:
            logger.error(f"Error comparing forecast models for {symbol}: {str(e)}")
            return {}
    
    def generate_portfolio_forecast(self, portfolio_data: List[Dict], 
                                   forecast_days: int = 30) -> Dict:
        """Generate forecast for entire portfolio"""
        try:
            portfolio_forecasts = {}
            total_portfolio_value = 0
            forecasted_portfolio_value = 0
            
            for holding in portfolio_data:
                symbol = holding.get('symbol')
                shares = holding.get('shares', 0)
                current_price = holding.get('current_price', 0)
                
                if symbol and shares > 0 and current_price > 0:
                    forecast = self.generate_price_forecast(symbol, forecast_days)
                    
                    if forecast:
                        forecasted_price = forecast.get('forecasted_price', current_price)
                        current_value = shares * current_price
                        forecasted_value = shares * forecasted_price
                        
                        portfolio_forecasts[symbol] = {
                            'current_price': current_price,
                            'forecasted_price': forecasted_price,
                            'current_value': current_value,
                            'forecasted_value': forecasted_value,
                            'expected_return': ((forecasted_price - current_price) / current_price) * 100,
                            'forecast_details': forecast
                        }
                        
                        total_portfolio_value += current_value
                        forecasted_portfolio_value += forecasted_value
            
            portfolio_expected_return = 0
            if total_portfolio_value > 0:
                portfolio_expected_return = ((forecasted_portfolio_value - total_portfolio_value) / total_portfolio_value) * 100
            
            return {
                'individual_forecasts': portfolio_forecasts,
                'portfolio_summary': {
                    'current_value': total_portfolio_value,
                    'forecasted_value': forecasted_portfolio_value,
                    'expected_return_percent': portfolio_expected_return,
                    'forecast_horizon_days': forecast_days
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating portfolio forecast: {str(e)}")
            return {}
    
    def _select_best_model(self, model_results: Dict) -> Optional[str]:
        """Select best performing model based on accuracy metrics"""
        try:
            best_model = None
            best_accuracy = 0
            
            for model, result in model_results.items():
                if result and isinstance(result, dict):
                    accuracy = result.get('accuracy', 0)
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_model = model
            
            return best_model
            
        except Exception:
            return None
    
    def _calculate_comparison_metrics(self, model_results: Dict) -> Dict:
        """Calculate comparison metrics between models"""
        try:
            metrics = {
                'models_tested': len([m for m in model_results.values() if m is not None]),
                'models_failed': len([m for m in model_results.values() if m is None]),
                'accuracy_range': {},
                'prediction_variance': 0
            }
            
            accuracies = [r.get('accuracy', 0) for r in model_results.values() if r is not None]
            predictions = [r.get('forecasted_price', 0) for r in model_results.values() if r is not None]
            
            if accuracies:
                metrics['accuracy_range'] = {
                    'min': min(accuracies),
                    'max': max(accuracies),
                    'avg': sum(accuracies) / len(accuracies)
                }
            
            if len(predictions) > 1:
                mean_prediction = sum(predictions) / len(predictions)
                variance = sum([(p - mean_prediction) ** 2 for p in predictions]) / len(predictions)
                metrics['prediction_variance'] = variance
            
            return metrics
            
        except Exception:
            return {}


# Global instance to be used across the application
forecast_manager = ForecastManager()