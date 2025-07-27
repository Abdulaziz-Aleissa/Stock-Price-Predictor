"""
Monte Carlo Simulation for Stock Risk Analysis
Self-contained implementation using only Python standard library
"""
import random
import math
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class MonteCarloSimulator:
    """Monte Carlo simulation for stock price analysis and risk assessment"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def _generate_mock_data(self, symbol, num_days=252):
        """Generate realistic mock stock data using only standard library"""
        # Mock prices based on symbol (realistic starting prices)
        base_prices = {
            'AAPL': 150.0, 'GOOGL': 2800.0, 'MSFT': 300.0, 'TSLA': 200.0,
            'AMZN': 3000.0, 'META': 250.0, 'NVDA': 400.0, 'SPY': 400.0,
            'QQQ': 350.0, 'NFLX': 400.0, 'AMD': 90.0, 'INTC': 50.0
        }
        
        start_price = base_prices.get(symbol.upper(), 100.0)
        
        # Generate realistic price movements using geometric brownian motion
        random.seed(42)  # For consistent demo data
        prices = [start_price]
        
        # Simulate daily returns with realistic parameters
        mu = 0.0005  # Average daily return (about 0.05%)
        sigma = 0.02  # Daily volatility (about 2%)
        
        for i in range(num_days - 1):
            # Generate random return using Box-Muller transform for normal distribution
            u1 = random.random()
            u2 = random.random()
            z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
            daily_return = mu + sigma * z
            
            new_price = prices[-1] * (1 + daily_return)
            prices.append(max(new_price, 0.01))  # Ensure price doesn't go negative
        
        self.logger.info(f"Generated mock data for {symbol}: {len(prices)} days, starting at ${start_price:.2f}")
        return prices
    
    def _calculate_statistics(self, prices):
        """Calculate basic statistics from price list"""
        if len(prices) < 2:
            return None, None
        
        # Calculate daily returns
        returns = []
        for i in range(1, len(prices)):
            ret = (prices[i] - prices[i-1]) / prices[i-1]
            returns.append(ret)
        
        if not returns:
            return None, None
        
        # Calculate mean return
        mu = sum(returns) / len(returns)
        
        # Calculate standard deviation (volatility)
        variance = sum((r - mu) ** 2 for r in returns) / len(returns)
        sigma = math.sqrt(variance)
        
        return mu, sigma
    
    def _calculate_percentile(self, data, percentile):
        """Calculate percentile from sorted data"""
        if not data:
            return 0
        
        sorted_data = sorted(data)
        n = len(sorted_data)
        index = (percentile / 100.0) * (n - 1)
        
        if index == int(index):
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
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
            self.logger.info(f"Running Monte Carlo simulation for {symbol}: {simulations} simulations, {days} days")
            
            # Generate historical mock data
            historical_prices = self._generate_mock_data(symbol, 252)
            current_price = historical_prices[-1]
            
            # Calculate statistical parameters
            mu, sigma = self._calculate_statistics(historical_prices)
            
            if mu is None or sigma is None or sigma == 0:
                error_msg = f"Unable to calculate valid statistical parameters for {symbol}"
                self.logger.error(error_msg)
                return {"error": error_msg}
            
            self.logger.info(f"Simulation parameters for {symbol}: mu={mu:.6f}, sigma={sigma:.6f}, current_price=${current_price:.2f}")
            
            # Generate simulation paths
            final_prices = []
            sample_paths = []
            
            for sim in range(simulations):
                path = [current_price]
                price = current_price
                
                for day in range(days):
                    # Generate random return using Box-Muller transform
                    u1 = random.random()
                    u2 = random.random()
                    z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
                    
                    # Geometric Brownian Motion
                    drift = (mu - 0.5 * sigma**2)
                    shock = sigma * z
                    price = price * math.exp(drift + shock)
                    path.append(price)
                
                final_prices.append(price)
                # Store first 50 paths for visualization
                if sim < 50:
                    sample_paths.append(path)
            
            # Calculate risk metrics
            percentiles = {
                '5th': self._calculate_percentile(final_prices, 5),
                '25th': self._calculate_percentile(final_prices, 25),
                '50th': self._calculate_percentile(final_prices, 50),
                '75th': self._calculate_percentile(final_prices, 75),
                '95th': self._calculate_percentile(final_prices, 95)
            }
            
            # Value at Risk (VaR)
            var_5 = (percentiles['5th'] - current_price) / current_price * 100
            var_1 = (self._calculate_percentile(final_prices, 1) - current_price) / current_price * 100
            
            # Expected shortfall (Conditional VaR)
            var_5_threshold = percentiles['5th']
            losses_beyond_var = [p for p in final_prices if p <= var_5_threshold]
            if losses_beyond_var:
                es_5 = sum(losses_beyond_var) / len(losses_beyond_var)
                expected_shortfall = (es_5 - current_price) / current_price * 100
            else:
                expected_shortfall = var_5
            
            # Probability of profit/loss
            profitable_outcomes = sum(1 for p in final_prices if p > current_price)
            prob_profit = (profitable_outcomes / simulations) * 100
            prob_loss = 100 - prob_profit
            
            # Generate dates for plotting
            start_date = datetime.now()
            dates = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days + 1)]
            
            self.logger.info(f"Monte Carlo simulation completed successfully for {symbol}")
            
            return {
                "success": True,
                "symbol": symbol,
                "current_price": current_price,
                "simulation_days": days,
                "simulations": simulations,
                "annual_return": mu * 252 * 100,  # Annualized return
                "annual_volatility": sigma * math.sqrt(252) * 100,  # Annualized volatility
                "percentiles": percentiles,
                "var_5": var_5,
                "var_1": var_1,
                "expected_shortfall": expected_shortfall,
                "probability_profit": prob_profit,
                "probability_loss": prob_loss,
                "dates": dates,
                "sample_paths": sample_paths,
                "price_distribution": final_prices[:100]  # Limit for performance
            }
            
        except Exception as e:
            error_msg = f"Monte Carlo simulation error for {symbol}: {str(e)}"
            self.logger.error(error_msg)
            return {"error": error_msg}
    
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