"""
Value at Risk (VaR) Analysis Module
Provides comprehensive VaR calculations and risk assessment
"""
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ValueAtRiskAnalyzer:
    """Comprehensive Value at Risk analysis toolkit"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def get_stock_data(self, symbol, period="2y"):
        """Get historical stock data for VaR analysis"""
        try:
            stock = yf.Ticker(symbol)
            hist_data = stock.history(period=period)
            
            if hist_data.empty:
                return None, None
            
            # Get current price
            current_price = hist_data['Close'].iloc[-1]
            
            # Calculate daily returns
            returns = hist_data['Close'].pct_change().dropna()
            
            return float(current_price), returns
            
        except Exception as e:
            self.logger.error(f"Error getting stock data for {symbol}: {str(e)}")
            return None, None
    
    def historical_var(self, returns, confidence_levels=[0.95, 0.99], holding_period=1):
        """
        Calculate Historical Value at Risk
        
        Args:
            returns: Series of daily returns
            confidence_levels: List of confidence levels (e.g., [0.95, 0.99])
            holding_period: Number of days to hold the position
            
        Returns:
            Dictionary with VaR values for each confidence level
        """
        try:
            # Adjust returns for holding period
            if holding_period > 1:
                adjusted_returns = returns * np.sqrt(holding_period)
            else:
                adjusted_returns = returns
            
            var_results = {}
            
            for confidence in confidence_levels:
                # Calculate percentile (lower tail)
                percentile = 1 - confidence
                var_value = np.percentile(adjusted_returns, percentile * 100)
                
                # Convert to positive loss value
                var_results[f"VaR_{int(confidence*100)}"] = abs(var_value) * 100  # As percentage
            
            return var_results
            
        except Exception as e:
            self.logger.error(f"Historical VaR calculation error: {str(e)}")
            return {}
    
    def parametric_var(self, returns, confidence_levels=[0.95, 0.99], holding_period=1):
        """
        Calculate Parametric VaR assuming normal distribution
        
        Args:
            returns: Series of daily returns
            confidence_levels: List of confidence levels
            holding_period: Number of days to hold the position
            
        Returns:
            Dictionary with parametric VaR values
        """
        try:
            # Calculate mean and standard deviation
            mean_return = returns.mean()
            std_return = returns.std()
            
            # Adjust for holding period
            if holding_period > 1:
                mean_return = mean_return * holding_period
                std_return = std_return * np.sqrt(holding_period)
            
            var_results = {}
            
            for confidence in confidence_levels:
                # Calculate Z-score for confidence level
                z_score = stats.norm.ppf(1 - confidence)
                
                # Calculate VaR (mean + z_score * std)
                var_value = mean_return + z_score * std_return
                
                # Convert to positive loss value
                var_results[f"Parametric_VaR_{int(confidence*100)}"] = abs(var_value) * 100  # As percentage
            
            return var_results
            
        except Exception as e:
            self.logger.error(f"Parametric VaR calculation error: {str(e)}")
            return {}
    
    def monte_carlo_var(self, returns, confidence_levels=[0.95, 0.99], 
                       holding_period=1, simulations=10000):
        """
        Calculate Monte Carlo VaR using simulated returns
        
        Args:
            returns: Series of daily returns
            confidence_levels: List of confidence levels
            holding_period: Number of days to hold the position
            simulations: Number of Monte Carlo simulations
            
        Returns:
            Dictionary with Monte Carlo VaR values and simulation results
        """
        try:
            # Calculate parameters from historical data
            mean_return = returns.mean()
            std_return = returns.std()
            
            # Generate random returns
            np.random.seed(42)  # For reproducibility
            simulated_returns = np.random.normal(
                mean_return * holding_period, 
                std_return * np.sqrt(holding_period), 
                simulations
            )
            
            var_results = {}
            
            for confidence in confidence_levels:
                # Calculate percentile from simulated returns
                percentile = 1 - confidence
                var_value = np.percentile(simulated_returns, percentile * 100)
                
                # Convert to positive loss value
                var_results[f"MC_VaR_{int(confidence*100)}"] = abs(var_value) * 100  # As percentage
            
            # Add simulation statistics
            var_results["simulated_returns"] = simulated_returns.tolist()[:1000]  # First 1000 for visualization
            var_results["mean_simulated"] = float(np.mean(simulated_returns))
            var_results["std_simulated"] = float(np.std(simulated_returns))
            
            return var_results
            
        except Exception as e:
            self.logger.error(f"Monte Carlo VaR calculation error: {str(e)}")
            return {}
    
    def expected_shortfall(self, returns, confidence_level=0.95, holding_period=1):
        """
        Calculate Expected Shortfall (Conditional VaR)
        
        Args:
            returns: Series of daily returns
            confidence_level: Confidence level for calculation
            holding_period: Number of days to hold the position
            
        Returns:
            Expected Shortfall value
        """
        try:
            # Adjust returns for holding period
            if holding_period > 1:
                adjusted_returns = returns * np.sqrt(holding_period)
            else:
                adjusted_returns = returns
            
            # Calculate VaR threshold
            percentile = 1 - confidence_level
            var_threshold = np.percentile(adjusted_returns, percentile * 100)
            
            # Calculate Expected Shortfall (mean of returns below VaR)
            tail_returns = adjusted_returns[adjusted_returns <= var_threshold]
            
            if len(tail_returns) > 0:
                expected_shortfall = abs(tail_returns.mean()) * 100  # As percentage
            else:
                expected_shortfall = 0
            
            return float(expected_shortfall)
            
        except Exception as e:
            self.logger.error(f"Expected Shortfall calculation error: {str(e)}")
            return 0
    
    def risk_metrics(self, returns, current_price, portfolio_value):
        """
        Calculate comprehensive risk metrics
        
        Args:
            returns: Series of daily returns
            current_price: Current stock price
            portfolio_value: Current portfolio value
            
        Returns:
            Dictionary with various risk metrics
        """
        try:
            metrics = {}
            
            # Basic statistics
            metrics["mean_return"] = float(returns.mean() * 100)  # Daily mean return %
            metrics["volatility"] = float(returns.std() * 100 * np.sqrt(252))  # Annualized volatility %
            metrics["skewness"] = float(returns.skew())
            metrics["kurtosis"] = float(returns.kurtosis())
            
            # Sharpe ratio (assuming 2% risk-free rate)
            risk_free_rate = 0.02 / 252  # Daily risk-free rate
            excess_returns = returns - risk_free_rate
            if returns.std() != 0:
                metrics["sharpe_ratio"] = float(excess_returns.mean() / returns.std() * np.sqrt(252))
            else:
                metrics["sharpe_ratio"] = 0
            
            # Maximum drawdown
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            metrics["max_drawdown"] = float(abs(drawdown.min()) * 100)  # As percentage
            
            # Beta calculation (vs SPY as benchmark)
            try:
                spy = yf.Ticker("SPY")
                spy_data = spy.history(period="1y")
                if not spy_data.empty:
                    spy_returns = spy_data['Close'].pct_change().dropna()
                    
                    # Align dates
                    common_dates = returns.index.intersection(spy_returns.index)
                    if len(common_dates) > 50:  # Need sufficient data
                        aligned_returns = returns.loc[common_dates]
                        aligned_spy = spy_returns.loc[common_dates]
                        
                        covariance = np.cov(aligned_returns, aligned_spy)[0, 1]
                        spy_variance = np.var(aligned_spy)
                        
                        if spy_variance != 0:
                            metrics["beta"] = float(covariance / spy_variance)
                        else:
                            metrics["beta"] = 1.0
                    else:
                        metrics["beta"] = 1.0
                else:
                    metrics["beta"] = 1.0
            except:
                metrics["beta"] = 1.0
            
            # Portfolio impact metrics
            metrics["portfolio_std"] = float(portfolio_value * returns.std())  # Daily portfolio std
            metrics["annual_portfolio_volatility"] = float(portfolio_value * returns.std() * np.sqrt(252))
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Risk metrics calculation error: {str(e)}")
            return {}
    
    def comprehensive_var_analysis(self, symbol, portfolio_value=10000, 
                                 confidence_levels=[0.95, 0.99], holding_period=1):
        """
        Perform comprehensive VaR analysis
        
        Args:
            symbol: Stock ticker symbol
            portfolio_value: Portfolio value in dollars
            confidence_levels: List of confidence levels
            holding_period: Holding period in days
            
        Returns:
            Complete VaR analysis results
        """
        try:
            # Get stock data
            current_price, returns = self.get_stock_data(symbol)
            if current_price is None or returns is None:
                return {"error": "Unable to fetch stock data"}
            
            if len(returns) < 50:
                return {"error": "Insufficient historical data for VaR analysis"}
            
            # Calculate different VaR methods
            historical_var = self.historical_var(returns, confidence_levels, holding_period)
            parametric_var = self.parametric_var(returns, confidence_levels, holding_period)
            monte_carlo_var = self.monte_carlo_var(returns, confidence_levels, holding_period)
            
            # Calculate Expected Shortfall
            expected_shortfall_95 = self.expected_shortfall(returns, 0.95, holding_period)
            expected_shortfall_99 = self.expected_shortfall(returns, 0.99, holding_period)
            
            # Calculate comprehensive risk metrics
            risk_metrics = self.risk_metrics(returns, current_price, portfolio_value)
            
            # Convert VaR percentages to dollar amounts
            var_dollar_amounts = {}
            for method in ["Historical", "Parametric", "MC"]:
                for conf in confidence_levels:
                    key = f"VaR_{int(conf*100)}" if method == "Historical" else f"{method.replace('Historical', '')}_VaR_{int(conf*100)}"
                    if method == "Historical":
                        var_key = key
                    elif method == "Parametric":
                        var_key = f"Parametric_{key}"
                    else:
                        var_key = f"MC_{key}"
                    
                    if method == "Historical" and var_key in historical_var:
                        var_dollar_amounts[f"{method}_VaR_{int(conf*100)}_dollars"] = historical_var[var_key] * portfolio_value / 100
                    elif method == "Parametric" and var_key in parametric_var:
                        var_dollar_amounts[f"{method}_VaR_{int(conf*100)}_dollars"] = parametric_var[var_key] * portfolio_value / 100
                    elif method == "MC" and var_key in monte_carlo_var:
                        var_dollar_amounts[f"{method}_VaR_{int(conf*100)}_dollars"] = monte_carlo_var[var_key] * portfolio_value / 100
            
            # Risk assessment and recommendations
            avg_var_95 = np.mean([
                historical_var.get("VaR_95", 0),
                parametric_var.get("Parametric_VaR_95", 0),
                monte_carlo_var.get("MC_VaR_95", 0)
            ])
            
            risk_level = "Low"
            if avg_var_95 > 5:
                risk_level = "High"
            elif avg_var_95 > 3:
                risk_level = "Medium"
            
            recommendations = []
            if avg_var_95 > 5:
                recommendations.append("Consider diversifying your portfolio to reduce concentration risk")
                recommendations.append("High VaR indicates significant potential losses - consider position sizing")
            if risk_metrics.get("max_drawdown", 0) > 20:
                recommendations.append("Historical maximum drawdown is high - implement stop-loss strategies")
            if risk_metrics.get("sharpe_ratio", 0) < 1:
                recommendations.append("Risk-adjusted returns are below optimal - consider alternative investments")
            if not recommendations:
                recommendations.append("Risk levels appear manageable for current portfolio allocation")
            
            return {
                "success": True,
                "symbol": symbol,
                "current_price": current_price,
                "portfolio_value": portfolio_value,
                "holding_period": holding_period,
                "data_points": len(returns),
                "historical_var": historical_var,
                "parametric_var": parametric_var,
                "monte_carlo_var": monte_carlo_var,
                "var_dollar_amounts": var_dollar_amounts,
                "expected_shortfall": {
                    "ES_95": expected_shortfall_95,
                    "ES_99": expected_shortfall_99,
                    "ES_95_dollars": expected_shortfall_95 * portfolio_value / 100,
                    "ES_99_dollars": expected_shortfall_99 * portfolio_value / 100
                },
                "risk_metrics": risk_metrics,
                "risk_assessment": {
                    "risk_level": risk_level,
                    "average_var_95": avg_var_95,
                    "recommendations": recommendations
                },
                "returns_data": {
                    "returns": returns.tail(252).tolist(),  # Last year for visualization
                    "dates": [d.strftime('%Y-%m-%d') for d in returns.tail(252).index]
                }
            }
            
        except Exception as e:
            self.logger.error(f"VaR analysis error: {str(e)}")
            return {"error": f"VaR analysis failed: {str(e)}"}


# Global instance
var_analyzer = ValueAtRiskAnalyzer()