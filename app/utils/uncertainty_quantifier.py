"""
Uncertainty Quantification Module

This module implements Monte Carlo simulation and various uncertainty quantification
techniques for providing confidence intervals and prediction reliability metrics.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class UncertaintyQuantifier:
    """
    Advanced uncertainty quantification for stock price predictions
    """
    
    def __init__(self, n_simulations: int = 1000, random_state: int = 42):
        """
        Initialize uncertainty quantifier
        
        Args:
            n_simulations: Number of Monte Carlo simulations
            random_state: Random seed for reproducibility
        """
        self.n_simulations = n_simulations
        self.random_state = random_state
        np.random.seed(random_state)
        
    def monte_carlo_price_simulation(self,
                                   current_price: float,
                                   predicted_return: float,
                                   volatility: float,
                                   time_horizon: int = 1,
                                   distribution: str = 'normal') -> Dict:
        """
        Perform Monte Carlo simulation for price prediction uncertainty
        
        Args:
            current_price: Current stock price
            predicted_return: Expected return (as decimal)
            volatility: Historical volatility (annualized)
            time_horizon: Prediction horizon in days
            distribution: Distribution type ('normal', 'log_normal', 't_distribution')
        """
        try:
            # Scale volatility for time horizon
            scaled_volatility = volatility * np.sqrt(time_horizon / 252)
            
            # Generate random returns based on distribution
            if distribution == 'normal':
                random_returns = np.random.normal(predicted_return, scaled_volatility, self.n_simulations)
            elif distribution == 'log_normal':
                # For log-normal, use geometric returns
                mu = predicted_return - 0.5 * scaled_volatility**2
                random_returns = np.random.lognormal(mu, scaled_volatility, self.n_simulations) - 1
            elif distribution == 't_distribution':
                # t-distribution with 3 degrees of freedom (fat tails)
                t_samples = stats.t.rvs(df=3, size=self.n_simulations)
                random_returns = predicted_return + scaled_volatility * t_samples
            else:
                raise ValueError(f"Unknown distribution: {distribution}")
            
            # Calculate simulated prices
            simulated_prices = current_price * (1 + random_returns)
            
            # Calculate statistics
            mean_price = np.mean(simulated_prices)
            median_price = np.median(simulated_prices)
            std_price = np.std(simulated_prices)
            
            # Confidence intervals
            confidence_levels = [0.05, 0.10, 0.25, 0.75, 0.90, 0.95]
            percentiles = np.percentile(simulated_prices, [100 * p for p in confidence_levels])
            
            # Probability metrics
            prob_positive = np.mean(simulated_prices > current_price)
            prob_above_prediction = np.mean(simulated_prices > current_price * (1 + predicted_return))
            
            # Risk metrics
            var_95 = current_price - percentiles[0]  # 5th percentile loss
            var_99 = current_price - np.percentile(simulated_prices, 1)  # 1st percentile loss
            expected_shortfall = current_price - np.mean(simulated_prices[simulated_prices < percentiles[0]])
            
            return {
                'simulated_prices': simulated_prices.tolist(),
                'statistics': {
                    'mean': mean_price,
                    'median': median_price,
                    'std': std_price,
                    'min': np.min(simulated_prices),
                    'max': np.max(simulated_prices)
                },
                'confidence_intervals': {
                    '90%': {'lower': percentiles[0], 'upper': percentiles[5]},
                    '80%': {'lower': percentiles[1], 'upper': percentiles[4]},
                    '50%': {'lower': percentiles[2], 'upper': percentiles[3]}
                },
                'probabilities': {
                    'positive_return': prob_positive,
                    'above_prediction': prob_above_prediction,
                    'loss_probability': 1 - prob_positive
                },
                'risk_metrics': {
                    'var_95': max(0, var_95),
                    'var_99': max(0, var_99),
                    'expected_shortfall': max(0, expected_shortfall),
                    'maximum_loss': max(0, current_price - np.min(simulated_prices))
                },
                'parameters': {
                    'n_simulations': self.n_simulations,
                    'distribution': distribution,
                    'volatility': volatility,
                    'time_horizon': time_horizon
                }
            }
            
        except Exception as e:
            logger.error(f"Error in Monte Carlo simulation: {str(e)}")
            return {
                'simulated_prices': [],
                'statistics': {'mean': current_price, 'median': current_price, 'std': 0},
                'confidence_intervals': {'90%': {'lower': current_price, 'upper': current_price}},
                'probabilities': {'positive_return': 0.5, 'above_prediction': 0.5},
                'risk_metrics': {'var_95': 0, 'var_99': 0, 'expected_shortfall': 0},
                'parameters': {'error': str(e)}
            }
    
    def bootstrap_prediction_intervals(self,
                                     predictions: List[float],
                                     actuals: List[float],
                                     new_prediction: float,
                                     confidence_levels: List[float] = [0.8, 0.9, 0.95]) -> Dict:
        """
        Use bootstrap resampling to estimate prediction intervals
        """
        try:
            if len(predictions) != len(actuals) or len(predictions) < 10:
                return {'error': 'Insufficient historical data for bootstrap'}
            
            # Calculate residuals
            residuals = np.array(actuals) - np.array(predictions)
            
            # Bootstrap resampling
            bootstrap_predictions = []
            
            for _ in range(self.n_simulations):
                # Sample residuals with replacement
                bootstrap_residuals = np.random.choice(residuals, size=len(residuals), replace=True)
                
                # Add to new prediction
                bootstrap_pred = new_prediction + np.random.choice(bootstrap_residuals)
                bootstrap_predictions.append(bootstrap_pred)
            
            bootstrap_predictions = np.array(bootstrap_predictions)
            
            # Calculate confidence intervals
            intervals = {}
            for conf_level in confidence_levels:
                alpha = 1 - conf_level
                lower_percentile = (alpha / 2) * 100
                upper_percentile = (1 - alpha / 2) * 100
                
                intervals[f'{int(conf_level * 100)}%'] = {
                    'lower': np.percentile(bootstrap_predictions, lower_percentile),
                    'upper': np.percentile(bootstrap_predictions, upper_percentile)
                }
            
            return {
                'bootstrap_predictions': bootstrap_predictions.tolist(),
                'mean_prediction': np.mean(bootstrap_predictions),
                'prediction_std': np.std(bootstrap_predictions),
                'confidence_intervals': intervals,
                'bias_estimate': np.mean(residuals),
                'residual_std': np.std(residuals)
            }
            
        except Exception as e:
            logger.error(f"Error in bootstrap prediction intervals: {str(e)}")
            return {'error': str(e)}
    
    def calculate_prediction_reliability(self,
                                       historical_predictions: List[float],
                                       historical_actuals: List[float],
                                       prediction_dates: List[str] = None) -> Dict:
        """
        Calculate various reliability metrics for predictions
        """
        try:
            if len(historical_predictions) != len(historical_actuals):
                return {'error': 'Predictions and actuals length mismatch'}
            
            if len(historical_predictions) < 5:
                return {'error': 'Insufficient historical data'}
            
            predictions = np.array(historical_predictions)
            actuals = np.array(historical_actuals)
            
            # Basic accuracy metrics
            mae = mean_absolute_error(actuals, predictions)
            mse = mean_squared_error(actuals, predictions)
            rmse = np.sqrt(mse)
            
            # Percentage errors
            mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
            
            # Directional accuracy
            pred_direction = np.sign(np.diff(predictions))
            actual_direction = np.sign(np.diff(actuals))
            directional_accuracy = np.mean(pred_direction == actual_direction) if len(pred_direction) > 0 else 0
            
            # Correlation
            correlation = np.corrcoef(predictions, actuals)[0, 1] if len(predictions) > 1 else 0
            
            # Bias analysis
            bias = np.mean(predictions - actuals)
            bias_pct = (bias / np.mean(actuals)) * 100
            
            # Consistency metrics (rolling accuracy)
            window_size = min(10, len(predictions) // 2)
            rolling_errors = []
            
            if window_size >= 2:
                for i in range(window_size, len(predictions)):
                    window_preds = predictions[i-window_size:i]
                    window_actuals = actuals[i-window_size:i]
                    window_mae = mean_absolute_error(window_actuals, window_preds)
                    rolling_errors.append(window_mae)
            
            consistency_score = 1.0 / (1.0 + np.std(rolling_errors)) if rolling_errors else 0.5
            
            # Calibration analysis (how well confidence matches accuracy)
            residuals = np.abs(predictions - actuals)
            calibration_score = 1.0 - (np.std(residuals) / np.mean(residuals)) if np.mean(residuals) > 0 else 0.5
            calibration_score = max(0, min(1, calibration_score))
            
            # Overall reliability score
            reliability_components = {
                'accuracy': 1.0 / (1.0 + mae / np.mean(np.abs(actuals))),
                'directional': directional_accuracy,
                'correlation': max(0, correlation),
                'consistency': consistency_score,
                'calibration': calibration_score
            }
            
            overall_reliability = np.mean(list(reliability_components.values()))
            
            return {
                'accuracy_metrics': {
                    'mae': mae,
                    'mse': mse,
                    'rmse': rmse,
                    'mape': mape,
                    'bias': bias,
                    'bias_pct': bias_pct
                },
                'directional_accuracy': directional_accuracy,
                'correlation': correlation,
                'reliability_components': reliability_components,
                'overall_reliability': overall_reliability,
                'consistency_score': consistency_score,
                'calibration_score': calibration_score,
                'sample_size': len(predictions),
                'reliability_level': self._get_reliability_level(overall_reliability)
            }
            
        except Exception as e:
            logger.error(f"Error calculating prediction reliability: {str(e)}")
            return {'error': str(e)}
    
    def _get_reliability_level(self, reliability_score: float) -> str:
        """Convert reliability score to text description"""
        if reliability_score >= 0.85:
            return 'Excellent'
        elif reliability_score >= 0.75:
            return 'Very Good'
        elif reliability_score >= 0.65:
            return 'Good'
        elif reliability_score >= 0.55:
            return 'Fair'
        elif reliability_score >= 0.45:
            return 'Poor'
        else:
            return 'Very Poor'
    
    def scenario_analysis(self,
                         current_price: float,
                         base_prediction: float,
                         volatility: float,
                         scenarios: Dict[str, float] = None) -> Dict:
        """
        Perform scenario analysis with different market conditions
        """
        try:
            if scenarios is None:
                scenarios = {
                    'bull_market': 1.5,     # 50% higher volatility
                    'bear_market': 2.0,     # 100% higher volatility
                    'market_crash': 3.0,    # 200% higher volatility
                    'low_volatility': 0.5   # 50% of normal volatility
                }
            
            base_return = (base_prediction - current_price) / current_price
            results = {}
            
            for scenario_name, vol_multiplier in scenarios.items():
                scenario_vol = volatility * vol_multiplier
                
                # Monte Carlo for this scenario
                scenario_results = self.monte_carlo_price_simulation(
                    current_price=current_price,
                    predicted_return=base_return,
                    volatility=scenario_vol,
                    time_horizon=1
                )
                
                results[scenario_name] = {
                    'volatility_multiplier': vol_multiplier,
                    'expected_price': scenario_results['statistics']['mean'],
                    'confidence_90': scenario_results['confidence_intervals']['90%'],
                    'probability_positive': scenario_results['probabilities']['positive_return'],
                    'var_95': scenario_results['risk_metrics']['var_95'],
                    'max_loss': scenario_results['risk_metrics']['maximum_loss']
                }
            
            # Compare scenarios
            scenario_comparison = {
                'most_optimistic': max(results.keys(), key=lambda k: results[k]['expected_price']),
                'most_pessimistic': min(results.keys(), key=lambda k: results[k]['expected_price']),
                'highest_risk': max(results.keys(), key=lambda k: results[k]['var_95']),
                'lowest_risk': min(results.keys(), key=lambda k: results[k]['var_95'])
            }
            
            return {
                'scenarios': results,
                'comparison': scenario_comparison,
                'base_parameters': {
                    'current_price': current_price,
                    'base_prediction': base_prediction,
                    'base_volatility': volatility
                }
            }
            
        except Exception as e:
            logger.error(f"Error in scenario analysis: {str(e)}")
            return {'error': str(e)}
    
    def adaptive_confidence_intervals(self,
                                    prediction: float,
                                    model_uncertainty: float,
                                    data_uncertainty: float,
                                    market_regime_uncertainty: float = 0.1) -> Dict:
        """
        Calculate adaptive confidence intervals considering multiple uncertainty sources
        """
        try:
            # Combine different uncertainty sources
            total_uncertainty = np.sqrt(
                model_uncertainty**2 + 
                data_uncertainty**2 + 
                market_regime_uncertainty**2
            )
            
            # Adaptive confidence levels based on total uncertainty
            if total_uncertainty < 0.02:  # Low uncertainty
                confidence_levels = [0.80, 0.90, 0.95]
                multipliers = [1.28, 1.64, 1.96]  # Standard normal quantiles
            elif total_uncertainty < 0.05:  # Medium uncertainty
                confidence_levels = [0.75, 0.85, 0.90]
                multipliers = [1.15, 1.44, 1.64]
            else:  # High uncertainty
                confidence_levels = [0.70, 0.80, 0.85]
                multipliers = [1.04, 1.28, 1.44]
            
            intervals = {}
            for conf_level, multiplier in zip(confidence_levels, multipliers):
                margin = multiplier * total_uncertainty * prediction
                intervals[f'{int(conf_level * 100)}%'] = {
                    'lower': prediction - margin,
                    'upper': prediction + margin,
                    'margin': margin
                }
            
            # Uncertainty breakdown
            uncertainty_breakdown = {
                'model_uncertainty': model_uncertainty,
                'data_uncertainty': data_uncertainty,
                'market_regime_uncertainty': market_regime_uncertainty,
                'total_uncertainty': total_uncertainty,
                'dominant_source': max(
                    [('model', model_uncertainty), ('data', data_uncertainty), 
                     ('market_regime', market_regime_uncertainty)],
                    key=lambda x: x[1]
                )[0]
            }
            
            return {
                'confidence_intervals': intervals,
                'uncertainty_breakdown': uncertainty_breakdown,
                'adaptive_levels': confidence_levels,
                'recommendation': self._get_uncertainty_recommendation(total_uncertainty)
            }
            
        except Exception as e:
            logger.error(f"Error calculating adaptive confidence intervals: {str(e)}")
            return {'error': str(e)}
    
    def _get_uncertainty_recommendation(self, total_uncertainty: float) -> str:
        """Get recommendation based on total uncertainty level"""
        if total_uncertainty < 0.02:
            return "Low uncertainty - prediction is highly reliable"
        elif total_uncertainty < 0.05:
            return "Moderate uncertainty - proceed with normal caution"
        elif total_uncertainty < 0.10:
            return "High uncertainty - consider reducing position size"
        else:
            return "Very high uncertainty - avoid trading or wait for more data"