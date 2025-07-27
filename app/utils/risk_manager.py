"""
Risk Management and Position Sizing Module

This module implements advanced risk management techniques including position sizing,
Value at Risk calculations, and uncertainty quantification for stock predictions.
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

class RiskManager:
    """
    Advanced risk management system for stock predictions
    """
    
    def __init__(self, max_position_size: float = 0.05, max_portfolio_risk: float = 0.02):
        """
        Initialize risk manager
        
        Args:
            max_position_size: Maximum position size as fraction of portfolio (default 5%)
            max_portfolio_risk: Maximum portfolio risk per trade (default 2%)
        """
        self.max_position_size = max_position_size
        self.max_portfolio_risk = max_portfolio_risk
        
    def calculate_position_size(self, 
                              portfolio_value: float,
                              entry_price: float,
                              stop_loss_price: float,
                              prediction_confidence: float) -> Dict:
        """
        Calculate optimal position size based on risk parameters and prediction confidence
        """
        try:
            # Basic risk per share
            risk_per_share = abs(entry_price - stop_loss_price)
            
            if risk_per_share == 0:
                return {
                    'position_size': 0,
                    'shares': 0,
                    'max_loss': 0,
                    'confidence_factor': prediction_confidence,
                    'risk_reward_ratio': 0
                }
            
            # Maximum dollar risk
            max_dollar_risk = portfolio_value * self.max_portfolio_risk
            
            # Base shares calculation
            base_shares = int(max_dollar_risk / risk_per_share)
            
            # Maximum shares based on position size limit
            max_shares_by_position = int((portfolio_value * self.max_position_size) / entry_price)
            
            # Adjust shares based on prediction confidence (0.0 to 1.0)
            confidence_multiplier = max(0.1, min(1.0, prediction_confidence))
            adjusted_shares = int(base_shares * confidence_multiplier)
            
            # Final shares is minimum of all constraints
            final_shares = min(adjusted_shares, max_shares_by_position, base_shares)
            
            # Calculate final metrics
            position_value = final_shares * entry_price
            position_size_pct = position_value / portfolio_value
            max_loss = final_shares * risk_per_share
            
            return {
                'shares': final_shares,
                'position_value': position_value,
                'position_size_pct': position_size_pct,
                'max_loss': max_loss,
                'risk_per_share': risk_per_share,
                'confidence_factor': confidence_multiplier,
                'risk_reward_ratio': risk_per_share / entry_price if entry_price > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return {
                'shares': 0,
                'position_value': 0,
                'position_size_pct': 0,
                'max_loss': 0,
                'risk_per_share': 0,
                'confidence_factor': 0,
                'risk_reward_ratio': 0
            }
    
    def calculate_portfolio_var(self, 
                               positions: List[Dict],
                               confidence_level: float = 0.95,
                               time_horizon: int = 1) -> Dict:
        """
        Calculate Value at Risk for the entire portfolio
        
        Args:
            positions: List of portfolio positions with returns data
            confidence_level: VaR confidence level (default 95%)
            time_horizon: Time horizon in days (default 1 day)
        """
        try:
            if not positions:
                return {'var': 0, 'expected_shortfall': 0, 'confidence_level': confidence_level}
            
            # Extract returns data
            all_returns = []
            weights = []
            
            for position in positions:
                if 'returns' in position and 'weight' in position:
                    returns = np.array(position['returns'])
                    if len(returns) > 0:
                        all_returns.append(returns)
                        weights.append(position['weight'])
            
            if not all_returns:
                return {'var': 0, 'expected_shortfall': 0, 'confidence_level': confidence_level}
            
            # Create returns matrix
            min_length = min(len(returns) for returns in all_returns)
            returns_matrix = np.column_stack([returns[-min_length:] for returns in all_returns])
            weights = np.array(weights[:len(all_returns)])
            weights = weights / weights.sum()  # Normalize weights
            
            # Calculate portfolio returns
            portfolio_returns = np.dot(returns_matrix, weights)
            
            # Scale for time horizon
            if time_horizon > 1:
                portfolio_returns = portfolio_returns * np.sqrt(time_horizon)
            
            # Calculate VaR using historical simulation
            var_percentile = (1 - confidence_level) * 100
            var = np.percentile(portfolio_returns, var_percentile)
            
            # Calculate Expected Shortfall (Conditional VaR)
            tail_returns = portfolio_returns[portfolio_returns <= var]
            expected_shortfall = np.mean(tail_returns) if len(tail_returns) > 0 else var
            
            return {
                'var': abs(var),
                'expected_shortfall': abs(expected_shortfall),
                'confidence_level': confidence_level,
                'time_horizon': time_horizon,
                'portfolio_volatility': np.std(portfolio_returns),
                'worst_case_loss': abs(np.min(portfolio_returns))
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio VaR: {str(e)}")
            return {'var': 0, 'expected_shortfall': 0, 'confidence_level': confidence_level}
    
    def calculate_prediction_confidence(self,
                                      prediction: float,
                                      prediction_std: float,
                                      historical_accuracy: float) -> Dict:
        """
        Calculate confidence metrics for a prediction
        """
        try:
            # Normalize prediction standard deviation to confidence score
            # Lower std = higher confidence
            std_confidence = 1.0 / (1.0 + prediction_std / abs(prediction) if prediction != 0 else 1.0)
            std_confidence = max(0.1, min(1.0, std_confidence))
            
            # Historical accuracy confidence (0.5 to 1.0 range)
            accuracy_confidence = max(0.5, min(1.0, historical_accuracy))
            
            # Combined confidence (weighted average)
            combined_confidence = (std_confidence * 0.6) + (accuracy_confidence * 0.4)
            
            # Confidence intervals (assuming normal distribution)
            confidence_95_lower = prediction - (1.96 * prediction_std)
            confidence_95_upper = prediction + (1.96 * prediction_std)
            confidence_99_lower = prediction - (2.58 * prediction_std)
            confidence_99_upper = prediction + (2.58 * prediction_std)
            
            return {
                'overall_confidence': combined_confidence,
                'prediction_uncertainty': prediction_std,
                'historical_accuracy': historical_accuracy,
                'confidence_intervals': {
                    '95%': {'lower': confidence_95_lower, 'upper': confidence_95_upper},
                    '99%': {'lower': confidence_99_lower, 'upper': confidence_99_upper}
                },
                'confidence_level_text': self._get_confidence_text(combined_confidence)
            }
            
        except Exception as e:
            logger.error(f"Error calculating prediction confidence: {str(e)}")
            return {
                'overall_confidence': 0.5,
                'prediction_uncertainty': 0,
                'historical_accuracy': 0.5,
                'confidence_intervals': {'95%': {'lower': 0, 'upper': 0}, '99%': {'lower': 0, 'upper': 0}},
                'confidence_level_text': 'Low'
            }
    
    def _get_confidence_text(self, confidence: float) -> str:
        """Convert confidence score to text description"""
        if confidence >= 0.8:
            return 'Very High'
        elif confidence >= 0.7:
            return 'High'
        elif confidence >= 0.6:
            return 'Medium'
        elif confidence >= 0.5:
            return 'Low'
        else:
            return 'Very Low'
    
    def assess_market_conditions(self, 
                                price_data: pd.DataFrame,
                                volume_data: Optional[pd.Series] = None) -> Dict:
        """
        Assess current market conditions for risk adjustment
        """
        try:
            if 'Close' not in price_data.columns:
                return {'market_regime': 'Unknown', 'volatility_regime': 'Normal', 'risk_multiplier': 1.0}
            
            close_prices = price_data['Close'].dropna()
            returns = close_prices.pct_change().dropna()
            
            # Calculate volatility metrics
            current_volatility = returns.rolling(20).std().iloc[-1] * np.sqrt(252)  # Annualized
            long_term_volatility = returns.rolling(60).std().iloc[-1] * np.sqrt(252)
            
            # Trend analysis
            sma_20 = close_prices.rolling(20).mean().iloc[-1]
            sma_50 = close_prices.rolling(50).mean().iloc[-1]
            current_price = close_prices.iloc[-1]
            
            # Market regime classification
            if current_price > sma_20 > sma_50:
                market_regime = 'Bullish'
                risk_multiplier = 0.9  # Slightly lower risk in uptrend
            elif current_price < sma_20 < sma_50:
                market_regime = 'Bearish'
                risk_multiplier = 1.2  # Higher risk in downtrend
            else:
                market_regime = 'Sideways'
                risk_multiplier = 1.0
            
            # Volatility regime
            vol_ratio = current_volatility / long_term_volatility if long_term_volatility > 0 else 1.0
            
            if vol_ratio > 1.5:
                volatility_regime = 'High Volatility'
                risk_multiplier *= 1.3
            elif vol_ratio < 0.7:
                volatility_regime = 'Low Volatility'
                risk_multiplier *= 0.8
            else:
                volatility_regime = 'Normal Volatility'
            
            # Volume analysis if available
            volume_trend = 'Normal'
            if volume_data is not None:
                recent_volume = volume_data.rolling(10).mean().iloc[-1]
                long_term_volume = volume_data.rolling(50).mean().iloc[-1]
                
                if recent_volume > long_term_volume * 1.5:
                    volume_trend = 'High Volume'
                elif recent_volume < long_term_volume * 0.5:
                    volume_trend = 'Low Volume'
                    risk_multiplier *= 1.1  # Slightly higher risk in low volume
            
            return {
                'market_regime': market_regime,
                'volatility_regime': volatility_regime,
                'volume_trend': volume_trend,
                'current_volatility': current_volatility,
                'volatility_ratio': vol_ratio,
                'risk_multiplier': min(2.0, max(0.5, risk_multiplier)),  # Cap risk multiplier
                'trend_strength': abs(current_price - sma_50) / sma_50 if sma_50 > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error assessing market conditions: {str(e)}")
            return {
                'market_regime': 'Unknown',
                'volatility_regime': 'Normal',
                'volume_trend': 'Normal',
                'risk_multiplier': 1.0,
                'current_volatility': 0,
                'volatility_ratio': 1.0,
                'trend_strength': 0
            }
    
    def generate_risk_report(self,
                           prediction: float,
                           current_price: float,
                           prediction_confidence: Dict,
                           position_sizing: Dict,
                           market_conditions: Dict) -> Dict:
        """
        Generate comprehensive risk assessment report
        """
        try:
            # Calculate potential returns and risks
            expected_return_pct = ((prediction - current_price) / current_price) * 100
            
            # Risk-adjusted metrics
            adjusted_confidence = prediction_confidence['overall_confidence'] * (1 / market_conditions['risk_multiplier'])
            
            # Risk level assessment
            if abs(expected_return_pct) < 2 and prediction_confidence['overall_confidence'] > 0.7:
                risk_level = 'Low'
            elif abs(expected_return_pct) < 5 and prediction_confidence['overall_confidence'] > 0.6:
                risk_level = 'Medium'
            elif abs(expected_return_pct) < 10 and prediction_confidence['overall_confidence'] > 0.5:
                risk_level = 'High'
            else:
                risk_level = 'Very High'
            
            # Generate recommendations
            recommendations = []
            
            if adjusted_confidence < 0.5:
                recommendations.append("Consider reducing position size due to low confidence")
            
            if market_conditions['volatility_regime'] == 'High Volatility':
                recommendations.append("Exercise caution due to high market volatility")
            
            if market_conditions['market_regime'] == 'Bearish':
                recommendations.append("Consider defensive positioning in bearish market")
            
            if abs(expected_return_pct) > 10:
                recommendations.append("Large price movement predicted - verify with additional analysis")
            
            if not recommendations:
                recommendations.append("Risk parameters within acceptable range")
            
            return {
                'risk_level': risk_level,
                'expected_return_pct': expected_return_pct,
                'adjusted_confidence': adjusted_confidence,
                'position_recommendation': position_sizing,
                'market_assessment': market_conditions,
                'confidence_metrics': prediction_confidence,
                'recommendations': recommendations,
                'overall_score': {
                    'confidence': prediction_confidence['overall_confidence'],
                    'risk_adjusted_confidence': adjusted_confidence,
                    'market_favorability': 1.0 / market_conditions['risk_multiplier']
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating risk report: {str(e)}")
            return {
                'risk_level': 'Unknown',
                'expected_return_pct': 0,
                'adjusted_confidence': 0.5,
                'recommendations': ['Error in risk assessment - proceed with caution'],
                'overall_score': {'confidence': 0.5, 'risk_adjusted_confidence': 0.5, 'market_favorability': 0.5}
            }