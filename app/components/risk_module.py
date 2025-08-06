"""
Risk Module
Handle VaR and risk calculations
"""

from app.utils.value_at_risk import var_analyzer
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class RiskManager:
    """Handle VaR and risk calculations"""
    
    def __init__(self):
        self.var_analyzer = var_analyzer
    
    def calculate_portfolio_var(self, portfolio_data: List[Dict], 
                               confidence_level: float = 0.95, 
                               time_horizon: int = 1) -> Optional[Dict]:
        """Calculate Value at Risk for portfolio"""
        try:
            return self.var_analyzer.calculate_portfolio_var(
                portfolio_data, confidence_level, time_horizon
            )
        except Exception as e:
            logger.error(f"Error calculating portfolio VaR: {str(e)}")
            return None
    
    def calculate_individual_var(self, symbol: str, 
                                confidence_level: float = 0.95, 
                                time_horizon: int = 1) -> Optional[Dict]:
        """Calculate VaR for individual stock"""
        try:
            return self.var_analyzer.calculate_var(
                symbol, confidence_level, time_horizon
            )
        except Exception as e:
            logger.error(f"Error calculating VaR for {symbol}: {str(e)}")
            return None
    
    def calculate_expected_shortfall(self, returns: List[float], 
                                   confidence_level: float = 0.95) -> Optional[float]:
        """Calculate Expected Shortfall (Conditional VaR)"""
        try:
            return self.var_analyzer.expected_shortfall(returns, confidence_level)
        except Exception as e:
            logger.error(f"Error calculating Expected Shortfall: {str(e)}")
            return None
    
    def monte_carlo_simulation(self, symbol: str, 
                             simulation_days: int = 252, 
                             num_simulations: int = 1000) -> Optional[Dict]:
        """Run Monte Carlo simulation for risk analysis"""
        try:
            return self.var_analyzer.monte_carlo_var(
                symbol, simulation_days, num_simulations
            )
        except Exception as e:
            logger.error(f"Error running Monte Carlo simulation: {str(e)}")
            return None
    
    def stress_test_portfolio(self, portfolio_data: List[Dict], 
                             stress_scenarios: List[Dict]) -> Dict:
        """Perform stress testing on portfolio"""
        try:
            results = {}
            for scenario in stress_scenarios:
                scenario_name = scenario.get('name', 'Unknown')
                stress_factor = scenario.get('stress_factor', 1.0)
                
                # Apply stress factor to portfolio
                stressed_portfolio = []
                for holding in portfolio_data:
                    stressed_holding = holding.copy()
                    stressed_holding['current_price'] *= stress_factor
                    stressed_portfolio.append(stressed_holding)
                
                # Calculate stressed VaR
                stressed_var = self.calculate_portfolio_var(stressed_portfolio)
                results[scenario_name] = stressed_var
            
            return results
            
        except Exception as e:
            logger.error(f"Error performing stress test: {str(e)}")
            return {}
    
    def calculate_portfolio_beta(self, portfolio_data: List[Dict], 
                                market_symbol: str = 'SPY') -> Optional[float]:
        """Calculate portfolio beta against market"""
        try:
            return self.var_analyzer.calculate_beta(portfolio_data, market_symbol)
        except Exception as e:
            logger.error(f"Error calculating portfolio beta: {str(e)}")
            return None
    
    def risk_metrics_summary(self, portfolio_data: List[Dict]) -> Dict:
        """Get comprehensive risk metrics summary"""
        try:
            var_95 = self.calculate_portfolio_var(portfolio_data, 0.95)
            var_99 = self.calculate_portfolio_var(portfolio_data, 0.99)
            beta = self.calculate_portfolio_beta(portfolio_data)
            
            return {
                'var_95': var_95,
                'var_99': var_99,
                'portfolio_beta': beta,
                'risk_level': self._assess_risk_level(var_95, beta),
                'recommendations': self._generate_risk_recommendations(var_95, beta)
            }
            
        except Exception as e:
            logger.error(f"Error generating risk metrics summary: {str(e)}")
            return {}
    
    def _assess_risk_level(self, var_data: Optional[Dict], beta: Optional[float]) -> str:
        """Assess overall portfolio risk level"""
        try:
            if not var_data or beta is None:
                return 'Unknown'
            
            var_percentage = var_data.get('var_percentage', 0)
            
            if var_percentage > 5 or abs(beta) > 1.5:
                return 'High Risk'
            elif var_percentage > 2 or abs(beta) > 1.0:
                return 'Medium Risk'
            else:
                return 'Low Risk'
                
        except Exception:
            return 'Unknown'
    
    def _generate_risk_recommendations(self, var_data: Optional[Dict], beta: Optional[float]) -> List[str]:
        """Generate risk management recommendations"""
        try:
            recommendations = []
            
            if var_data:
                var_percentage = var_data.get('var_percentage', 0)
                if var_percentage > 5:
                    recommendations.append("Consider reducing position sizes or diversifying portfolio")
                if var_percentage > 10:
                    recommendations.append("Portfolio shows high volatility - review risk tolerance")
            
            if beta is not None:
                if beta > 1.5:
                    recommendations.append("Portfolio is highly sensitive to market movements")
                elif beta < 0.5:
                    recommendations.append("Portfolio may be overly conservative")
            
            if not recommendations:
                recommendations.append("Portfolio risk appears well-managed")
            
            return recommendations
            
        except Exception:
            return ["Unable to generate recommendations due to insufficient data"]


# Global instance to be used across the application
risk_manager = RiskManager()