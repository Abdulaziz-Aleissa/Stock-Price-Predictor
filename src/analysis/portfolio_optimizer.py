"""
Portfolio Optimization Module
Implements Modern Portfolio Theory and various optimization strategies
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
from ..utils.helpers import performance_tracker, MathUtils

logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    """Portfolio optimization using Modern Portfolio Theory and other methods"""
    
    def __init__(self):
        self.portfolio_settings = config.PORTFOLIO_SETTINGS
        self.risk_free_rate = config.RISK_SETTINGS['risk_free_rate']
        
        # Check if optimization libraries are available
        self.scipy_available = self._check_scipy()
        self.cvxpy_available = self._check_cvxpy()
        
    def _check_scipy(self) -> bool:
        """Check if SciPy is available for optimization"""
        try:
            from scipy.optimize import minimize
            from scipy import linalg
            return True
        except ImportError:
            logger.warning("SciPy not available for portfolio optimization")
            return False
    
    def _check_cvxpy(self) -> bool:
        """Check if CVXPY is available for convex optimization"""
        try:
            import cvxpy as cp
            return True
        except ImportError:
            logger.warning("CVXPY not available for advanced portfolio optimization")
            return False
    
    def calculate_portfolio_metrics(self, weights: np.ndarray, returns: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate portfolio performance metrics
        
        Args:
            weights: Portfolio weights
            returns: Asset returns DataFrame
        
        Returns:
            Dictionary with portfolio metrics
        """
        try:
            # Ensure weights sum to 1
            weights = weights / np.sum(weights)
            
            # Portfolio returns
            portfolio_returns = returns @ weights
            
            # Basic metrics
            annual_return = portfolio_returns.mean() * 252
            annual_volatility = portfolio_returns.std() * np.sqrt(252)
            
            # Sharpe ratio
            sharpe_ratio = (annual_return - self.risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
            
            # Maximum drawdown
            cumulative_returns = (1 + portfolio_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # Value at Risk (95% confidence)
            var_95 = np.percentile(portfolio_returns, 5)
            
            # Expected Shortfall (Conditional VaR)
            es_95 = portfolio_returns[portfolio_returns <= var_95].mean()
            
            # Sortino ratio (using downside deviation)
            downside_returns = portfolio_returns[portfolio_returns < 0]
            downside_deviation = downside_returns.std() * np.sqrt(252)
            sortino_ratio = (annual_return - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
            
            # Information ratio (if benchmark is available, use market return as proxy)
            # For simplicity, we'll use the equal-weighted portfolio as benchmark
            benchmark_returns = returns.mean(axis=1)
            excess_returns = portfolio_returns - benchmark_returns
            tracking_error = excess_returns.std() * np.sqrt(252)
            information_ratio = excess_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0
            
            return {
                'annual_return': float(annual_return),
                'annual_volatility': float(annual_volatility),
                'sharpe_ratio': float(sharpe_ratio),
                'sortino_ratio': float(sortino_ratio),
                'information_ratio': float(information_ratio),
                'max_drawdown': float(max_drawdown),
                'var_95': float(var_95),
                'expected_shortfall_95': float(es_95),
                'calmar_ratio': float(annual_return / abs(max_drawdown)) if max_drawdown != 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {str(e)}")
            return {}
    
    def optimize_portfolio(self, returns: pd.DataFrame, method: str = 'mean_variance',
                          target_return: Optional[float] = None,
                          risk_tolerance: str = 'moderate',
                          constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Optimize portfolio using specified method
        
        Args:
            returns: Asset returns DataFrame
            method: Optimization method ('mean_variance', 'risk_parity', 'max_sharpe', 'min_variance')
            target_return: Target annual return (for mean-variance optimization)
            risk_tolerance: Risk tolerance level ('conservative', 'moderate', 'aggressive')
            constraints: Additional constraints
        
        Returns:
            Dictionary with optimization results
        """
        try:
            if not self.scipy_available:
                return {'error': 'SciPy not available for portfolio optimization'}
            
            performance_tracker.start_timer("portfolio_optimization")
            
            # Validate inputs
            if returns.empty:
                return {'error': 'Empty returns data provided'}
            
            n_assets = len(returns.columns)
            
            if n_assets < 2:
                return {'error': 'Need at least 2 assets for portfolio optimization'}
            
            # Calculate expected returns and covariance matrix
            expected_returns = returns.mean() * 252  # Annualized
            cov_matrix = returns.cov() * 252  # Annualized
            
            # Set default constraints
            default_constraints = {
                'max_weight': self.portfolio_settings['max_position_size'],
                'min_weight': 0.0,
                'long_only': True
            }
            
            if constraints:
                default_constraints.update(constraints)
            
            # Optimize based on method
            if method == 'mean_variance':
                result = self._optimize_mean_variance(
                    expected_returns, cov_matrix, target_return, default_constraints
                )
            elif method == 'max_sharpe':
                result = self._optimize_max_sharpe(
                    expected_returns, cov_matrix, default_constraints
                )
            elif method == 'min_variance':
                result = self._optimize_min_variance(
                    cov_matrix, default_constraints
                )
            elif method == 'risk_parity':
                result = self._optimize_risk_parity(
                    cov_matrix, default_constraints
                )
            elif method == 'equal_weight':
                result = self._equal_weight_portfolio(n_assets, returns.columns)
            else:
                return {'error': f'Unknown optimization method: {method}'}
            
            if 'error' in result:
                return result
            
            # Calculate portfolio metrics
            weights = result['weights']
            metrics = self.calculate_portfolio_metrics(weights, returns)
            
            # Add optimization details
            result.update({
                'portfolio_metrics': metrics,
                'optimization_method': method,
                'n_assets': n_assets,
                'asset_names': returns.columns.tolist(),
                'optimization_duration': performance_tracker.end_timer("portfolio_optimization"),
                'constraints_used': default_constraints,
                'risk_tolerance': risk_tolerance
            })
            
            logger.info(f"Portfolio optimization completed using {method}")
            logger.info(f"Expected return: {metrics.get('annual_return', 0):.2%}, "
                       f"Volatility: {metrics.get('annual_volatility', 0):.2%}, "
                       f"Sharpe: {metrics.get('sharpe_ratio', 0):.3f}")
            
            return result
            
        except Exception as e:
            performance_tracker.end_timer("portfolio_optimization")
            logger.error(f"Error in portfolio optimization: {str(e)}")
            return {'error': str(e)}
    
    def _optimize_mean_variance(self, expected_returns: pd.Series, cov_matrix: pd.DataFrame,
                               target_return: Optional[float], constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize portfolio using mean-variance optimization"""
        try:
            from scipy.optimize import minimize
            
            n_assets = len(expected_returns)
            
            # Objective function: minimize portfolio variance
            def portfolio_variance(weights):
                return np.dot(weights.T, np.dot(cov_matrix, weights))
            
            # Constraints
            constraint_list = []
            
            # Weights sum to 1
            constraint_list.append({
                'type': 'eq',
                'fun': lambda x: np.sum(x) - 1
            })
            
            # Target return constraint (if specified)
            if target_return is not None:
                constraint_list.append({
                    'type': 'eq',
                    'fun': lambda x: np.dot(x, expected_returns) - target_return
                })
            
            # Bounds for weights
            if constraints['long_only']:
                bounds = [(constraints['min_weight'], constraints['max_weight']) for _ in range(n_assets)]
            else:
                bounds = [(-1, 1) for _ in range(n_assets)]  # Allow short selling
            
            # Initial guess (equal weights)
            x0 = np.array([1/n_assets] * n_assets)
            
            # Optimize
            result = minimize(
                portfolio_variance,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraint_list,
                options={'maxiter': 1000}
            )
            
            if not result.success:
                return {'error': f'Optimization failed: {result.message}'}
            
            weights = result.x
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_risk = np.sqrt(portfolio_variance(weights))
            
            return {
                'weights': weights,
                'expected_return': float(portfolio_return),
                'expected_risk': float(portfolio_risk),
                'optimization_success': True,
                'target_return_achieved': target_return is None or abs(portfolio_return - target_return) < 1e-6
            }
            
        except Exception as e:
            logger.error(f"Error in mean-variance optimization: {str(e)}")
            return {'error': str(e)}
    
    def _optimize_max_sharpe(self, expected_returns: pd.Series, cov_matrix: pd.DataFrame,
                            constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize portfolio for maximum Sharpe ratio"""
        try:
            from scipy.optimize import minimize
            
            n_assets = len(expected_returns)
            
            # Objective function: minimize negative Sharpe ratio
            def negative_sharpe(weights):
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                
                if portfolio_risk == 0:
                    return -np.inf
                
                sharpe = (portfolio_return - self.risk_free_rate) / portfolio_risk
                return -sharpe  # Negative because we minimize
            
            # Constraints
            constraint_list = []
            
            # Weights sum to 1
            constraint_list.append({
                'type': 'eq',
                'fun': lambda x: np.sum(x) - 1
            })
            
            # Bounds for weights
            if constraints['long_only']:
                bounds = [(constraints['min_weight'], constraints['max_weight']) for _ in range(n_assets)]
            else:
                bounds = [(-1, 1) for _ in range(n_assets)]
            
            # Initial guess (equal weights)
            x0 = np.array([1/n_assets] * n_assets)
            
            # Optimize
            result = minimize(
                negative_sharpe,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraint_list,
                options={'maxiter': 1000}
            )
            
            if not result.success:
                return {'error': f'Optimization failed: {result.message}'}
            
            weights = result.x
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk
            
            return {
                'weights': weights,
                'expected_return': float(portfolio_return),
                'expected_risk': float(portfolio_risk),
                'sharpe_ratio': float(sharpe_ratio),
                'optimization_success': True
            }
            
        except Exception as e:
            logger.error(f"Error in max Sharpe optimization: {str(e)}")
            return {'error': str(e)}
    
    def _optimize_min_variance(self, cov_matrix: pd.DataFrame, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize portfolio for minimum variance"""
        try:
            from scipy.optimize import minimize
            
            n_assets = len(cov_matrix)
            
            # Objective function: minimize portfolio variance
            def portfolio_variance(weights):
                return np.dot(weights.T, np.dot(cov_matrix, weights))
            
            # Constraints
            constraint_list = []
            
            # Weights sum to 1
            constraint_list.append({
                'type': 'eq',
                'fun': lambda x: np.sum(x) - 1
            })
            
            # Bounds for weights
            if constraints['long_only']:
                bounds = [(constraints['min_weight'], constraints['max_weight']) for _ in range(n_assets)]
            else:
                bounds = [(-1, 1) for _ in range(n_assets)]
            
            # Initial guess (equal weights)
            x0 = np.array([1/n_assets] * n_assets)
            
            # Optimize
            result = minimize(
                portfolio_variance,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraint_list,
                options={'maxiter': 1000}
            )
            
            if not result.success:
                return {'error': f'Optimization failed: {result.message}'}
            
            weights = result.x
            portfolio_risk = np.sqrt(portfolio_variance(weights))
            
            return {
                'weights': weights,
                'expected_risk': float(portfolio_risk),
                'optimization_success': True
            }
            
        except Exception as e:
            logger.error(f"Error in min variance optimization: {str(e)}")
            return {'error': str(e)}
    
    def _optimize_risk_parity(self, cov_matrix: pd.DataFrame, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize portfolio using risk parity approach"""
        try:
            from scipy.optimize import minimize
            
            n_assets = len(cov_matrix)
            
            # Objective function: minimize sum of squared differences in risk contributions
            def risk_parity_objective(weights):
                portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
                
                if portfolio_variance == 0:
                    return np.inf
                
                # Risk contributions
                marginal_contrib = np.dot(cov_matrix, weights)
                risk_contrib = weights * marginal_contrib / portfolio_variance
                
                # Target risk contribution (equal for all assets)
                target_contrib = 1 / n_assets
                
                # Sum of squared differences
                return np.sum((risk_contrib - target_contrib) ** 2)
            
            # Constraints
            constraint_list = []
            
            # Weights sum to 1
            constraint_list.append({
                'type': 'eq',
                'fun': lambda x: np.sum(x) - 1
            })
            
            # Bounds for weights (risk parity typically requires positive weights)
            bounds = [(0.01, constraints['max_weight']) for _ in range(n_assets)]
            
            # Initial guess (equal weights)
            x0 = np.array([1/n_assets] * n_assets)
            
            # Optimize
            result = minimize(
                risk_parity_objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraint_list,
                options={'maxiter': 1000}
            )
            
            if not result.success:
                return {'error': f'Optimization failed: {result.message}'}
            
            weights = result.x
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            portfolio_risk = np.sqrt(portfolio_variance)
            
            # Calculate risk contributions
            marginal_contrib = np.dot(cov_matrix, weights)
            risk_contrib = weights * marginal_contrib / portfolio_variance
            
            return {
                'weights': weights,
                'expected_risk': float(portfolio_risk),
                'risk_contributions': risk_contrib.tolist(),
                'optimization_success': True
            }
            
        except Exception as e:
            logger.error(f"Error in risk parity optimization: {str(e)}")
            return {'error': str(e)}
    
    def _equal_weight_portfolio(self, n_assets: int, asset_names: pd.Index) -> Dict[str, Any]:
        """Create equal-weighted portfolio"""
        try:
            weights = np.array([1/n_assets] * n_assets)
            
            return {
                'weights': weights,
                'optimization_success': True,
                'method_note': 'Equal weighting - no optimization performed'
            }
            
        except Exception as e:
            logger.error(f"Error creating equal weight portfolio: {str(e)}")
            return {'error': str(e)}
    
    def calculate_efficient_frontier(self, returns: pd.DataFrame, n_points: int = 100) -> Dict[str, Any]:
        """
        Calculate the efficient frontier
        
        Args:
            returns: Asset returns DataFrame
            n_points: Number of points on the frontier
        
        Returns:
            Dictionary with efficient frontier data
        """
        try:
            if not self.scipy_available:
                return {'error': 'SciPy not available for efficient frontier calculation'}
            
            performance_tracker.start_timer("efficient_frontier")
            
            # Calculate expected returns and covariance matrix
            expected_returns = returns.mean() * 252
            cov_matrix = returns.cov() * 252
            
            # Range of target returns
            min_return = expected_returns.min()
            max_return = expected_returns.max()
            target_returns = np.linspace(min_return, max_return, n_points)
            
            # Calculate efficient portfolios
            efficient_portfolios = []
            
            for target_return in target_returns:
                try:
                    result = self._optimize_mean_variance(
                        expected_returns, cov_matrix, target_return, 
                        {'max_weight': 1.0, 'min_weight': 0.0, 'long_only': True}
                    )
                    
                    if 'error' not in result:
                        efficient_portfolios.append({
                            'target_return': target_return,
                            'expected_return': result['expected_return'],
                            'expected_risk': result['expected_risk'],
                            'weights': result['weights']
                        })
                        
                except Exception:
                    continue  # Skip problematic points
            
            if not efficient_portfolios:
                return {'error': 'Could not calculate efficient frontier'}
            
            # Calculate maximum Sharpe ratio portfolio
            max_sharpe_result = self._optimize_max_sharpe(
                expected_returns, cov_matrix,
                {'max_weight': 1.0, 'min_weight': 0.0, 'long_only': True}
            )
            
            # Calculate minimum variance portfolio
            min_var_result = self._optimize_min_variance(
                cov_matrix,
                {'max_weight': 1.0, 'min_weight': 0.0, 'long_only': True}
            )
            
            duration = performance_tracker.end_timer("efficient_frontier")
            
            return {
                'efficient_portfolios': efficient_portfolios,
                'max_sharpe_portfolio': max_sharpe_result if 'error' not in max_sharpe_result else None,
                'min_variance_portfolio': min_var_result if 'error' not in min_var_result else None,
                'n_points': len(efficient_portfolios),
                'asset_names': returns.columns.tolist(),
                'calculation_duration': duration,
                'risk_free_rate': self.risk_free_rate
            }
            
        except Exception as e:
            performance_tracker.end_timer("efficient_frontier")
            logger.error(f"Error calculating efficient frontier: {str(e)}")
            return {'error': str(e)}
    
    def portfolio_rebalancing_analysis(self, current_weights: np.ndarray, 
                                     target_weights: np.ndarray,
                                     prices: pd.Series,
                                     transaction_costs: float = 0.001) -> Dict[str, Any]:
        """
        Analyze portfolio rebalancing requirements
        
        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            prices: Current asset prices
            transaction_costs: Transaction cost rate
        
        Returns:
            Dictionary with rebalancing analysis
        """
        try:
            # Calculate weight differences
            weight_diff = target_weights - current_weights
            
            # Assets to buy (positive differences)
            assets_to_buy = weight_diff > 0.001  # Small threshold to avoid tiny trades
            
            # Assets to sell (negative differences)
            assets_to_sell = weight_diff < -0.001
            
            # Calculate turnover
            turnover = np.sum(np.abs(weight_diff)) / 2  # Divide by 2 to avoid double counting
            
            # Estimate transaction costs
            estimated_costs = turnover * transaction_costs
            
            # Calculate rebalancing urgency (higher when weights are far from target)
            max_deviation = np.max(np.abs(weight_diff))
            avg_deviation = np.mean(np.abs(weight_diff))
            
            # Rebalancing recommendation
            if max_deviation > 0.05:  # 5% deviation threshold
                recommendation = 'Immediate rebalancing recommended'
                urgency = 'High'
            elif max_deviation > 0.03:  # 3% deviation threshold
                recommendation = 'Rebalancing suggested within a week'
                urgency = 'Medium'
            elif max_deviation > 0.01:  # 1% deviation threshold
                recommendation = 'Monitor and consider rebalancing'
                urgency = 'Low'
            else:
                recommendation = 'No rebalancing needed'
                urgency = 'None'
            
            return {
                'current_weights': current_weights.tolist(),
                'target_weights': target_weights.tolist(),
                'weight_differences': weight_diff.tolist(),
                'assets_to_buy': assets_to_buy.tolist(),
                'assets_to_sell': assets_to_sell.tolist(),
                'turnover': float(turnover),
                'estimated_transaction_costs': float(estimated_costs),
                'max_deviation': float(max_deviation),
                'avg_deviation': float(avg_deviation),
                'recommendation': recommendation,
                'urgency': urgency,
                'rebalancing_threshold': 0.05
            }
            
        except Exception as e:
            logger.error(f"Error in rebalancing analysis: {str(e)}")
            return {'error': str(e)}
    
    def analyze_portfolio_diversification(self, weights: np.ndarray, 
                                        returns: pd.DataFrame,
                                        sector_mapping: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Analyze portfolio diversification
        
        Args:
            weights: Portfolio weights
            returns: Asset returns DataFrame
            sector_mapping: Mapping of assets to sectors
        
        Returns:
            Dictionary with diversification analysis
        """
        try:
            n_assets = len(weights)
            
            # Concentration metrics
            herfindahl_index = np.sum(weights ** 2)  # Lower is more diversified
            effective_assets = 1 / herfindahl_index  # Effective number of assets
            
            # Weight distribution analysis
            max_weight = np.max(weights)
            min_weight = np.min(weights)
            weight_std = np.std(weights)
            
            # Correlation analysis
            corr_matrix = returns.corr()
            
            # Average correlation
            avg_correlation = (corr_matrix.values.sum() - n_assets) / (n_assets * (n_assets - 1))
            
            # Weighted average correlation
            weighted_correlations = []
            for i in range(n_assets):
                for j in range(i+1, n_assets):
                    weighted_corr = corr_matrix.iloc[i, j] * weights[i] * weights[j]
                    weighted_correlations.append(weighted_corr)
            
            weighted_avg_correlation = np.sum(weighted_correlations) / np.sum([weights[i] * weights[j] 
                                                                             for i in range(n_assets) 
                                                                             for j in range(i+1, n_assets)])
            
            # Diversification ratio (Choueifaty & Coignard, 2008)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
            weighted_avg_vol = np.sum(weights * returns.std() * np.sqrt(252))
            diversification_ratio = weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 1
            
            # Sector diversification (if sector mapping provided)
            sector_analysis = None
            if sector_mapping:
                sector_weights = {}
                for i, asset in enumerate(returns.columns):
                    sector = sector_mapping.get(asset, 'Unknown')
                    sector_weights[sector] = sector_weights.get(sector, 0) + weights[i]
                
                sector_herfindahl = sum(w ** 2 for w in sector_weights.values())
                effective_sectors = 1 / sector_herfindahl if sector_herfindahl > 0 else 1
                
                sector_analysis = {
                    'sector_weights': sector_weights,
                    'sector_herfindahl_index': sector_herfindahl,
                    'effective_sectors': effective_sectors,
                    'most_concentrated_sector': max(sector_weights, key=sector_weights.get) if sector_weights else None
                }
            
            # Overall diversification score (0-100, higher is better)
            diversification_score = (
                30 * min(1, effective_assets / 10) +  # Asset diversification (30%)
                25 * max(0, 1 - abs(avg_correlation)) +  # Low correlation (25%)
                25 * min(1, diversification_ratio / 1.5) +  # Diversification ratio (25%)
                20 * (1 - herfindahl_index)  # Weight concentration (20%)
            )
            
            return {
                'herfindahl_index': float(herfindahl_index),
                'effective_assets': float(effective_assets),
                'max_weight': float(max_weight),
                'min_weight': float(min_weight),
                'weight_std': float(weight_std),
                'avg_correlation': float(avg_correlation),
                'weighted_avg_correlation': float(weighted_avg_correlation),
                'diversification_ratio': float(diversification_ratio),
                'diversification_score': float(diversification_score),
                'sector_analysis': sector_analysis,
                'diversification_grade': self._get_diversification_grade(diversification_score),
                'recommendations': self._get_diversification_recommendations(
                    herfindahl_index, avg_correlation, max_weight, effective_assets
                )
            }
            
        except Exception as e:
            logger.error(f"Error analyzing diversification: {str(e)}")
            return {'error': str(e)}
    
    def _get_diversification_grade(self, score: float) -> str:
        """Get diversification grade based on score"""
        if score >= 80:
            return 'A (Excellent)'
        elif score >= 70:
            return 'B (Good)'
        elif score >= 60:
            return 'C (Fair)'
        elif score >= 50:
            return 'D (Poor)'
        else:
            return 'F (Very Poor)'
    
    def _get_diversification_recommendations(self, herfindahl: float, avg_corr: float, 
                                           max_weight: float, effective_assets: float) -> List[str]:
        """Get diversification improvement recommendations"""
        recommendations = []
        
        if herfindahl > 0.3:
            recommendations.append("Consider reducing concentration by adding more assets or rebalancing weights")
        
        if avg_corr > 0.7:
            recommendations.append("Portfolio has high correlation - consider adding assets from different sectors/regions")
        
        if max_weight > 0.4:
            recommendations.append("Largest position is very concentrated - consider reducing maximum position size")
        
        if effective_assets < 5:
            recommendations.append("Portfolio is equivalent to fewer than 5 assets - consider broader diversification")
        
        if not recommendations:
            recommendations.append("Portfolio shows good diversification characteristics")
        
        return recommendations
    
    def monte_carlo_portfolio_simulation(self, returns: pd.DataFrame, weights: np.ndarray,
                                       n_simulations: int = 1000, time_horizon: int = 252) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation for portfolio performance
        
        Args:
            returns: Historical returns DataFrame
            weights: Portfolio weights
            n_simulations: Number of simulation paths
            time_horizon: Time horizon in days
        
        Returns:
            Dictionary with simulation results
        """
        try:
            performance_tracker.start_timer("monte_carlo_portfolio")
            
            # Calculate portfolio returns
            portfolio_returns = returns @ weights
            
            # Estimate return distribution parameters
            mean_return = portfolio_returns.mean()
            std_return = portfolio_returns.std()
            
            # Run simulations
            simulation_results = []
            
            for _ in range(n_simulations):
                # Generate random returns
                random_returns = np.random.normal(mean_return, std_return, time_horizon)
                
                # Calculate cumulative performance
                cumulative_return = np.prod(1 + random_returns) - 1
                final_value = 1 + cumulative_return
                
                # Calculate maximum drawdown for this path
                cumulative_values = np.cumprod(1 + random_returns)
                rolling_max = np.maximum.accumulate(cumulative_values)
                drawdowns = (cumulative_values - rolling_max) / rolling_max
                max_drawdown = np.min(drawdowns)
                
                simulation_results.append({
                    'final_value': final_value,
                    'total_return': cumulative_return,
                    'max_drawdown': max_drawdown,
                    'volatility': np.std(random_returns) * np.sqrt(252)
                })
            
            # Analyze results
            final_values = [r['final_value'] for r in simulation_results]
            total_returns = [r['total_return'] for r in simulation_results]
            max_drawdowns = [r['max_drawdown'] for r in simulation_results]
            
            # Calculate percentiles
            percentiles = [5, 10, 25, 50, 75, 90, 95]
            return_percentiles = {f'p{p}': np.percentile(total_returns, p) for p in percentiles}
            value_percentiles = {f'p{p}': np.percentile(final_values, p) for p in percentiles}
            drawdown_percentiles = {f'p{p}': np.percentile(max_drawdowns, p) for p in percentiles}
            
            # Probability of loss
            prob_loss = len([r for r in total_returns if r < 0]) / n_simulations
            
            # Expected shortfall (5% worst cases)
            worst_5_percent = sorted(total_returns)[:int(0.05 * n_simulations)]
            expected_shortfall = np.mean(worst_5_percent) if worst_5_percent else 0
            
            duration = performance_tracker.end_timer("monte_carlo_portfolio")
            
            return {
                'n_simulations': n_simulations,
                'time_horizon_days': time_horizon,
                'time_horizon_years': time_horizon / 252,
                'expected_return': float(np.mean(total_returns)),
                'return_std': float(np.std(total_returns)),
                'probability_of_loss': float(prob_loss),
                'expected_shortfall_5pct': float(expected_shortfall),
                'return_percentiles': {k: float(v) for k, v in return_percentiles.items()},
                'value_percentiles': {k: float(v) for k, v in value_percentiles.items()},
                'drawdown_percentiles': {k: float(v) for k, v in drawdown_percentiles.items()},
                'best_case_return': float(max(total_returns)),
                'worst_case_return': float(min(total_returns)),
                'simulation_duration': duration
            }
            
        except Exception as e:
            performance_tracker.end_timer("monte_carlo_portfolio")
            logger.error(f"Error in Monte Carlo simulation: {str(e)}")
            return {'error': str(e)}

# Create global portfolio optimizer instance
portfolio_optimizer = PortfolioOptimizer()