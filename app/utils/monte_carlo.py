"""
Monte Carlo Simulation for Stock Risk Analysis
"""
import numpy as np
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class MonteCarloSimulator:
    """Monte Carlo simulation for stock price analysis and risk assessment"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def get_stock_data(self, symbol, period="1y"):
        """Fetch historical stock data"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            return data
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def calculate_returns(self, prices):
        """Calculate daily returns from price data"""
        return prices.pct_change().dropna()
    
    def simulate_price_paths(self, symbol, days=30, simulations=1000):
        """
        Run Monte Carlo simulation for stock price paths
        
        Args:
            symbol: Stock ticker symbol
            days: Number of days to simulate
            simulations: Number of simulation paths
            
        Returns:
            Dictionary with simulation results
        """
        try:
            # Get historical data
            data = self.get_stock_data(symbol, period="1y")
            if data is None or data.empty:
                return {"error": "Unable to fetch historical data"}
            
            # Calculate parameters
            prices = data['Close']
            returns = self.calculate_returns(prices)
            
            current_price = prices.iloc[-1]
            mu = returns.mean()  # Average daily return
            sigma = returns.std()  # Volatility
            
            # Generate simulation paths
            dt = 1  # Daily time step
            price_paths = np.zeros((simulations, days + 1))
            price_paths[:, 0] = current_price
            
            # Generate random walks
            for i in range(simulations):
                for t in range(1, days + 1):
                    # Geometric Brownian Motion
                    drift = (mu - 0.5 * sigma**2) * dt
                    shock = sigma * np.sqrt(dt) * np.random.normal()
                    price_paths[i, t] = price_paths[i, t-1] * np.exp(drift + shock)
            
            # Calculate statistics
            final_prices = price_paths[:, -1]
            
            # Risk metrics
            percentiles = {
                '5th': np.percentile(final_prices, 5),
                '25th': np.percentile(final_prices, 25),
                '50th': np.percentile(final_prices, 50),
                '75th': np.percentile(final_prices, 75),
                '95th': np.percentile(final_prices, 95)
            }
            
            # Value at Risk (VaR)
            var_5 = (percentiles['5th'] - current_price) / current_price * 100
            var_1 = (np.percentile(final_prices, 1) - current_price) / current_price * 100
            
            # Expected shortfall (Conditional VaR)
            es_5 = np.mean(final_prices[final_prices <= percentiles['5th']])
            expected_shortfall = (es_5 - current_price) / current_price * 100
            
            # Probability of profit/loss
            prob_profit = np.sum(final_prices > current_price) / simulations * 100
            prob_loss = 100 - prob_profit
            
            # Generate dates for plotting
            start_date = datetime.now()
            dates = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days + 1)]
            
            # Select sample paths for visualization (max 50 for performance)
            sample_paths = price_paths[:min(50, simulations), :]
            
            return {
                "success": True,
                "symbol": symbol,
                "current_price": float(current_price),
                "simulation_days": days,
                "simulations": simulations,
                "annual_return": float(mu * 252 * 100),  # Annualized return
                "annual_volatility": float(sigma * np.sqrt(252) * 100),  # Annualized volatility
                "percentiles": {k: float(v) for k, v in percentiles.items()},
                "var_5": float(var_5),
                "var_1": float(var_1),
                "expected_shortfall": float(expected_shortfall),
                "probability_profit": float(prob_profit),
                "probability_loss": float(prob_loss),
                "dates": dates,
                "sample_paths": sample_paths.tolist(),
                "price_distribution": final_prices.tolist()
            }
            
        except Exception as e:
            self.logger.error(f"Monte Carlo simulation error: {str(e)}")
            return {"error": f"Simulation failed: {str(e)}"}
    
    def risk_analysis(self, symbol, investment_amount=10000):
        """
        Perform comprehensive risk analysis using Monte Carlo
        
        Args:
            symbol: Stock ticker symbol
            investment_amount: Investment amount in dollars
            
        Returns:
            Risk analysis results
        """
        try:
            simulation_results = self.simulate_price_paths(symbol, days=30, simulations=5000)
            
            if "error" in simulation_results:
                return simulation_results
            
            current_price = simulation_results["current_price"]
            percentiles = simulation_results["percentiles"]
            
            # Calculate investment scenarios
            shares = investment_amount / current_price
            
            scenarios = {}
            for percentile, price in percentiles.items():
                portfolio_value = shares * price
                profit_loss = portfolio_value - investment_amount
                return_pct = (profit_loss / investment_amount) * 100
                
                scenarios[percentile] = {
                    "price": price,
                    "portfolio_value": portfolio_value,
                    "profit_loss": profit_loss,
                    "return_percentage": return_pct
                }
            
            # Risk recommendations
            risk_level = "Low"
            if simulation_results["annual_volatility"] > 30:
                risk_level = "High"
            elif simulation_results["annual_volatility"] > 20:
                risk_level = "Medium"
            
            recommendation = self._generate_risk_recommendation(
                simulation_results["var_5"],
                simulation_results["probability_profit"],
                simulation_results["annual_volatility"]
            )
            
            return {
                **simulation_results,
                "investment_amount": investment_amount,
                "shares": shares,
                "scenarios": scenarios,
                "risk_level": risk_level,
                "recommendation": recommendation
            }
            
        except Exception as e:
            self.logger.error(f"Risk analysis error: {str(e)}")
            return {"error": f"Risk analysis failed: {str(e)}"}
    
    def _generate_risk_recommendation(self, var_5, prob_profit, volatility):
        """Generate investment recommendation based on risk metrics"""
        recommendations = []
        
        if var_5 < -20:
            recommendations.append("⚠️ High downside risk - consider position sizing")
        elif var_5 < -10:
            recommendations.append("⚡ Moderate risk - monitor closely")
        else:
            recommendations.append("✅ Acceptable risk level")
        
        if prob_profit > 70:
            recommendations.append("📈 High probability of profit")
        elif prob_profit > 50:
            recommendations.append("⚖️ Balanced risk-reward profile")
        else:
            recommendations.append("📉 Higher probability of loss - exercise caution")
        
        if volatility > 40:
            recommendations.append("🌪️ Very high volatility - expect large price swings")
        elif volatility > 25:
            recommendations.append("📊 High volatility - suitable for risk-tolerant investors")
        else:
            recommendations.append("🏛️ Moderate volatility - relatively stable")
        
        return recommendations


# Global instance
monte_carlo_simulator = MonteCarloSimulator()