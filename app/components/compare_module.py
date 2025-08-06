"""
Compare Module
Handle stock comparison functionality
"""

from app.data.yfinance_data import yfinance_data
from typing import Dict, List, Optional, Tuple
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class CompareManager:
    """Handle stock comparison operations"""
    
    def __init__(self):
        self.data_fetcher = yfinance_data
    
    def compare_stocks(self, symbol1: str, symbol2: str, 
                      timeframe: str = '1y') -> Optional[Dict]:
        """Compare two stocks across various metrics"""
        try:
            # Get historical data for both stocks
            data1 = self.data_fetcher.get_historical_data(symbol1, timeframe)
            data2 = self.data_fetcher.get_historical_data(symbol2, timeframe)
            
            if data1 is None or data2 is None:
                raise ValueError("Unable to fetch data for one or both stocks")
            
            # Calculate comparison metrics
            comparison_data = {
                'symbol1': symbol1,
                'symbol2': symbol2,
                'timeframe': timeframe,
                'performance': self._calculate_performance_comparison(data1, data2),
                'volatility': self._calculate_volatility_comparison(data1, data2),
                'correlation': self._calculate_correlation(data1, data2),
                'risk_metrics': self._calculate_risk_metrics(data1, data2),
                'price_data': self._prepare_price_data(data1, data2, symbol1, symbol2),
                'current_prices': {
                    symbol1: self.data_fetcher.get_current_price(symbol1),
                    symbol2: self.data_fetcher.get_current_price(symbol2)
                }
            }
            
            return comparison_data
            
        except Exception as e:
            logger.error(f"Error comparing stocks {symbol1} vs {symbol2}: {str(e)}")
            return None
    
    def _calculate_performance_comparison(self, data1: pd.DataFrame, 
                                        data2: pd.DataFrame) -> Dict:
        """Calculate performance metrics for comparison"""
        try:
            # Calculate returns
            returns1 = data1['Close'].pct_change().dropna()
            returns2 = data2['Close'].pct_change().dropna()
            
            # Calculate cumulative returns
            cumulative_returns1 = (1 + returns1).cumprod() - 1
            cumulative_returns2 = (1 + returns2).cumprod() - 1
            
            # Calculate total returns
            total_return1 = cumulative_returns1.iloc[-1] if len(cumulative_returns1) > 0 else 0
            total_return2 = cumulative_returns2.iloc[-1] if len(cumulative_returns2) > 0 else 0
            
            # Calculate annualized returns
            trading_days = len(returns1)
            annualized_return1 = ((1 + total_return1) ** (252 / trading_days)) - 1 if trading_days > 0 else 0
            annualized_return2 = ((1 + total_return2) ** (252 / trading_days)) - 1 if trading_days > 0 else 0
            
            return {
                'total_return': {
                    'stock1': round(total_return1 * 100, 2),
                    'stock2': round(total_return2 * 100, 2)
                },
                'annualized_return': {
                    'stock1': round(annualized_return1 * 100, 2),
                    'stock2': round(annualized_return2 * 100, 2)
                },
                'cumulative_returns': {
                    'stock1': cumulative_returns1.tolist() if len(cumulative_returns1) > 0 else [],
                    'stock2': cumulative_returns2.tolist() if len(cumulative_returns2) > 0 else []
                }
            }
        except Exception as e:
            logger.error(f"Error calculating performance comparison: {str(e)}")
            return {}
    
    def _calculate_volatility_comparison(self, data1: pd.DataFrame, 
                                       data2: pd.DataFrame) -> Dict:
        """Calculate volatility metrics for comparison"""
        try:
            returns1 = data1['Close'].pct_change().dropna()
            returns2 = data2['Close'].pct_change().dropna()
            
            # Calculate volatilities
            volatility1 = returns1.std() * np.sqrt(252)  # Annualized
            volatility2 = returns2.std() * np.sqrt(252)
            
            return {
                'annualized_volatility': {
                    'stock1': round(volatility1 * 100, 2),
                    'stock2': round(volatility2 * 100, 2)
                },
                'rolling_volatility': {
                    'stock1': (returns1.rolling(30).std() * np.sqrt(252) * 100).tolist(),
                    'stock2': (returns2.rolling(30).std() * np.sqrt(252) * 100).tolist()
                }
            }
        except Exception as e:
            logger.error(f"Error calculating volatility comparison: {str(e)}")
            return {}
    
    def _calculate_correlation(self, data1: pd.DataFrame, 
                             data2: pd.DataFrame) -> Dict:
        """Calculate correlation between stocks"""
        try:
            returns1 = data1['Close'].pct_change().dropna()
            returns2 = data2['Close'].pct_change().dropna()
            
            # Align the data
            aligned_data = pd.concat([returns1, returns2], axis=1, join='inner')
            
            if len(aligned_data) < 2:
                return {'correlation': 0, 'rolling_correlation': []}
            
            correlation = aligned_data.corr().iloc[0, 1]
            
            # Rolling correlation
            rolling_corr = aligned_data.iloc[:, 0].rolling(30).corr(aligned_data.iloc[:, 1])
            
            return {
                'correlation': round(correlation, 3),
                'rolling_correlation': rolling_corr.dropna().tolist()
            }
        except Exception as e:
            logger.error(f"Error calculating correlation: {str(e)}")
            return {'correlation': 0, 'rolling_correlation': []}
    
    def _calculate_risk_metrics(self, data1: pd.DataFrame, 
                               data2: pd.DataFrame) -> Dict:
        """Calculate risk metrics for comparison"""
        try:
            returns1 = data1['Close'].pct_change().dropna()
            returns2 = data2['Close'].pct_change().dropna()
            
            # Calculate Sharpe ratios (assuming risk-free rate of 2%)
            risk_free_rate = 0.02
            excess_returns1 = returns1.mean() * 252 - risk_free_rate
            excess_returns2 = returns2.mean() * 252 - risk_free_rate
            
            volatility1 = returns1.std() * np.sqrt(252)
            volatility2 = returns2.std() * np.sqrt(252)
            
            sharpe1 = excess_returns1 / volatility1 if volatility1 > 0 else 0
            sharpe2 = excess_returns2 / volatility2 if volatility2 > 0 else 0
            
            # Calculate maximum drawdown
            cumulative1 = (1 + returns1).cumprod()
            cumulative2 = (1 + returns2).cumprod()
            
            peak1 = cumulative1.expanding().max()
            peak2 = cumulative2.expanding().max()
            
            drawdown1 = (cumulative1 - peak1) / peak1
            drawdown2 = (cumulative2 - peak2) / peak2
            
            max_drawdown1 = drawdown1.min()
            max_drawdown2 = drawdown2.min()
            
            return {
                'sharpe_ratio': {
                    'stock1': round(sharpe1, 3),
                    'stock2': round(sharpe2, 3)
                },
                'max_drawdown': {
                    'stock1': round(max_drawdown1 * 100, 2),
                    'stock2': round(max_drawdown2 * 100, 2)
                }
            }
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {str(e)}")
            return {}
    
    def _prepare_price_data(self, data1: pd.DataFrame, data2: pd.DataFrame, 
                           symbol1: str, symbol2: str) -> Dict:
        """Prepare price data for charting"""
        try:
            # Normalize prices to start at 100 for comparison
            normalized1 = (data1['Close'] / data1['Close'].iloc[0] * 100).tolist()
            normalized2 = (data2['Close'] / data2['Close'].iloc[0] * 100).tolist()
            
            dates = data1.index.strftime('%Y-%m-%d').tolist()
            
            return {
                'dates': dates,
                'normalized_prices': {
                    symbol1: normalized1,
                    symbol2: normalized2
                },
                'actual_prices': {
                    symbol1: data1['Close'].tolist(),
                    symbol2: data2['Close'].tolist()
                },
                'volumes': {
                    symbol1: data1['Volume'].tolist() if 'Volume' in data1.columns else [],
                    symbol2: data2['Volume'].tolist() if 'Volume' in data2.columns else []
                }
            }
        except Exception as e:
            logger.error(f"Error preparing price data: {str(e)}")
            return {}
    
    def get_comparison_summary(self, symbol1: str, symbol2: str, 
                             timeframe: str = '1y') -> Optional[Dict]:
        """Get a summary comparison between two stocks"""
        try:
            comparison_data = self.compare_stocks(symbol1, symbol2, timeframe)
            
            if not comparison_data:
                return None
            
            # Extract key metrics for summary
            performance = comparison_data.get('performance', {})
            volatility = comparison_data.get('volatility', {})
            risk_metrics = comparison_data.get('risk_metrics', {})
            
            return {
                'winner': self._determine_winner(comparison_data),
                'key_metrics': {
                    'return_comparison': performance.get('total_return', {}),
                    'risk_comparison': volatility.get('annualized_volatility', {}),
                    'risk_adjusted_return': risk_metrics.get('sharpe_ratio', {})
                }
            }
        except Exception as e:
            logger.error(f"Error getting comparison summary: {str(e)}")
            return None
    
    def _determine_winner(self, comparison_data: Dict) -> Dict:
        """Determine which stock performs better based on risk-adjusted returns"""
        try:
            sharpe_ratios = comparison_data.get('risk_metrics', {}).get('sharpe_ratio', {})
            
            if not sharpe_ratios:
                return {'winner': 'tie', 'reason': 'Insufficient data for comparison'}
            
            stock1_sharpe = sharpe_ratios.get('stock1', 0)
            stock2_sharpe = sharpe_ratios.get('stock2', 0)
            
            if stock1_sharpe > stock2_sharpe:
                return {
                    'winner': comparison_data['symbol1'],
                    'reason': f'Higher risk-adjusted return (Sharpe: {stock1_sharpe})'
                }
            elif stock2_sharpe > stock1_sharpe:
                return {
                    'winner': comparison_data['symbol2'],
                    'reason': f'Higher risk-adjusted return (Sharpe: {stock2_sharpe})'
                }
            else:
                return {'winner': 'tie', 'reason': 'Similar risk-adjusted returns'}
        except Exception as e:
            logger.error(f"Error determining winner: {str(e)}")
            return {'winner': 'tie', 'reason': 'Error in comparison'}


# Create global instance
compare_manager = CompareManager()