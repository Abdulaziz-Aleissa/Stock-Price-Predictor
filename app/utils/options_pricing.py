"""
Options Pricing Models - Black-Scholes and Greeks Calculator
"""
import numpy as np
import yfinance as yf
import pandas as pd
from scipy.stats import norm
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class OptionsPricing:
    """Options pricing using Black-Scholes model and Greeks calculation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.risk_free_rate = 0.02  # 2% default risk-free rate
    
    def get_stock_data(self, symbol):
        """Get current stock price and calculate volatility"""
        try:
            stock = yf.Ticker(symbol)
            
            # Get current price
            info = stock.info
            current_price = info.get('regularMarketPrice')
            if not current_price:
                hist = stock.history(period="1d")
                current_price = hist['Close'].iloc[-1] if not hist.empty else None
            
            # Calculate historical volatility
            hist_data = stock.history(period="1y")
            if hist_data.empty:
                return None, None
            
            returns = hist_data['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            
            return float(current_price), float(volatility)
            
        except Exception as e:
            self.logger.error(f"Error getting stock data for {symbol}: {str(e)}")
            return None, None
    
    def black_scholes_call(self, S, K, T, r, sigma):
        """
        Black-Scholes formula for European call option
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility
            
        Returns:
            Call option price
        """
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return call_price
    
    def black_scholes_put(self, S, K, T, r, sigma):
        """
        Black-Scholes formula for European put option
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility
            
        Returns:
            Put option price
        """
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return put_price
    
    def calculate_greeks(self, S, K, T, r, sigma, option_type='call'):
        """
        Calculate option Greeks (Delta, Gamma, Theta, Vega, Rho)
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            
        Returns:
            Dictionary with Greeks values
        """
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Delta
        if option_type == 'call':
            delta = norm.cdf(d1)
        else:  # put
            delta = norm.cdf(d1) - 1
        
        # Gamma (same for call and put)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        # Theta
        theta_common = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T)
        if option_type == 'call':
            theta = (theta_common * norm.cdf(d2)) / 365  # Per day
        else:  # put
            theta = (theta_common * norm.cdf(-d2)) / 365  # Per day
        
        # Vega (same for call and put)
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # Per 1% change in volatility
        
        # Rho
        if option_type == 'call':
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100  # Per 1% change in interest rate
        else:  # put
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100  # Per 1% change in interest rate
        
        return {
            'delta': float(delta),
            'gamma': float(gamma),
            'theta': float(theta),
            'vega': float(vega),
            'rho': float(rho)
        }
    
    def price_option(self, symbol, strike_price, expiration_days, option_type='call', custom_volatility=None):
        """
        Price an option using Black-Scholes model
        
        Args:
            symbol: Stock ticker symbol
            strike_price: Option strike price
            expiration_days: Days until expiration
            option_type: 'call' or 'put'
            custom_volatility: Optional custom volatility (as decimal)
            
        Returns:
            Option pricing results
        """
        try:
            # Get stock data
            current_price, historical_volatility = self.get_stock_data(symbol)
            if current_price is None:
                return {"error": "Unable to fetch stock data"}
            
            # Use custom volatility or historical
            volatility = custom_volatility if custom_volatility else historical_volatility
            if volatility is None:
                return {"error": "Unable to calculate volatility"}
            
            # Convert expiration to years
            T = expiration_days / 365.0
            
            # Calculate option price
            if option_type.lower() == 'call':
                option_price = self.black_scholes_call(current_price, strike_price, T, self.risk_free_rate, volatility)
            else:
                option_price = self.black_scholes_put(current_price, strike_price, T, self.risk_free_rate, volatility)
            
            # Calculate Greeks
            greeks = self.calculate_greeks(current_price, strike_price, T, self.risk_free_rate, volatility, option_type)
            
            # Calculate moneyness
            moneyness = current_price / strike_price
            if option_type.lower() == 'call':
                itm_probability = norm.cdf((np.log(moneyness) + (self.risk_free_rate + 0.5 * volatility**2) * T) / (volatility * np.sqrt(T)))
            else:
                itm_probability = 1 - norm.cdf((np.log(moneyness) + (self.risk_free_rate + 0.5 * volatility**2) * T) / (volatility * np.sqrt(T)))
            
            # Calculate breakeven
            if option_type.lower() == 'call':
                breakeven = strike_price + option_price
            else:
                breakeven = strike_price - option_price
            
            return {
                "success": True,
                "symbol": symbol,
                "option_type": option_type.upper(),
                "current_price": float(current_price),
                "strike_price": float(strike_price),
                "expiration_days": expiration_days,
                "time_to_expiration": float(T),
                "volatility": float(volatility * 100),  # As percentage
                "risk_free_rate": float(self.risk_free_rate * 100),  # As percentage
                "option_price": float(option_price),
                "greeks": greeks,
                "moneyness": float(moneyness),
                "itm_probability": float(itm_probability * 100),  # As percentage
                "breakeven": float(breakeven),
                "intrinsic_value": float(max(0, current_price - strike_price if option_type.lower() == 'call' else strike_price - current_price)),
                "time_value": float(option_price - max(0, current_price - strike_price if option_type.lower() == 'call' else strike_price - current_price))
            }
            
        except Exception as e:
            self.logger.error(f"Option pricing error: {str(e)}")
            return {"error": f"Option pricing failed: {str(e)}"}
    
    def volatility_smile(self, symbol, expiration_days, strikes_range=0.2):
        """
        Generate volatility smile data for different strike prices
        
        Args:
            symbol: Stock ticker symbol
            expiration_days: Days until expiration
            strikes_range: Range around current price (e.g., 0.2 = ±20%)
            
        Returns:
            Volatility smile data
        """
        try:
            current_price, base_volatility = self.get_stock_data(symbol)
            if current_price is None:
                return {"error": "Unable to fetch stock data"}
            
            # Generate strike prices
            strikes = []
            implied_vols = []
            call_prices = []
            put_prices = []
            
            price_range = current_price * strikes_range
            strike_prices = np.linspace(current_price - price_range, current_price + price_range, 11)
            
            T = expiration_days / 365.0
            
            for strike in strike_prices:
                # For demonstration, we'll use a simple volatility smile model
                # In practice, this would come from market data
                moneyness = current_price / strike
                vol_adjustment = 0.02 * (np.abs(moneyness - 1) ** 1.5)  # Smile effect
                implied_vol = base_volatility + vol_adjustment
                
                call_price = self.black_scholes_call(current_price, strike, T, self.risk_free_rate, implied_vol)
                put_price = self.black_scholes_put(current_price, strike, T, self.risk_free_rate, implied_vol)
                
                strikes.append(float(strike))
                implied_vols.append(float(implied_vol * 100))
                call_prices.append(float(call_price))
                put_prices.append(float(put_price))
            
            return {
                "success": True,
                "symbol": symbol,
                "current_price": float(current_price),
                "expiration_days": expiration_days,
                "strikes": strikes,
                "implied_volatilities": implied_vols,
                "call_prices": call_prices,
                "put_prices": put_prices
            }
            
        except Exception as e:
            self.logger.error(f"Volatility smile error: {str(e)}")
            return {"error": f"Volatility smile calculation failed: {str(e)}"}
    
    def option_strategy_analyzer(self, symbol, strategy_type, expiration_days, strikes=None):
        """
        Analyze popular option strategies
        
        Args:
            symbol: Stock ticker symbol
            strategy_type: Type of strategy (e.g., 'straddle', 'strangle', 'iron_condor')
            expiration_days: Days until expiration
            strikes: List of strike prices for complex strategies
            
        Returns:
            Strategy analysis results
        """
        try:
            current_price, volatility = self.get_stock_data(symbol)
            if current_price is None:
                return {"error": "Unable to fetch stock data"}
            
            if strategy_type.lower() == 'straddle':
                # Long straddle: Buy call + put at same strike
                strike = strikes[0] if strikes else current_price
                call_result = self.price_option(symbol, strike, expiration_days, 'call')
                put_result = self.price_option(symbol, strike, expiration_days, 'put')
                
                if "error" in call_result or "error" in put_result:
                    return {"error": "Unable to price options"}
                
                total_cost = call_result["option_price"] + put_result["option_price"]
                breakeven_up = strike + total_cost
                breakeven_down = strike - total_cost
                
                return {
                    "success": True,
                    "strategy": "Long Straddle",
                    "symbol": symbol,
                    "current_price": current_price,
                    "strike": strike,
                    "expiration_days": expiration_days,
                    "call_price": call_result["option_price"],
                    "put_price": put_result["option_price"],
                    "total_cost": total_cost,
                    "max_profit": "Unlimited",
                    "max_loss": total_cost,
                    "breakeven_upper": breakeven_up,
                    "breakeven_lower": breakeven_down,
                    "best_scenario": "High volatility, large price movement",
                    "worst_scenario": f"Stock price stays near ${strike:.2f} at expiration"
                }
            
            elif strategy_type.lower() == 'strangle':
                # Long strangle: Buy OTM call + OTM put
                call_strike = strikes[0] if strikes and len(strikes) >= 1 else current_price * 1.05
                put_strike = strikes[1] if strikes and len(strikes) >= 2 else current_price * 0.95
                
                call_result = self.price_option(symbol, call_strike, expiration_days, 'call')
                put_result = self.price_option(symbol, put_strike, expiration_days, 'put')
                
                if "error" in call_result or "error" in put_result:
                    return {"error": "Unable to price options"}
                
                total_cost = call_result["option_price"] + put_result["option_price"]
                breakeven_up = call_strike + total_cost
                breakeven_down = put_strike - total_cost
                
                return {
                    "success": True,
                    "strategy": "Long Strangle",
                    "symbol": symbol,
                    "current_price": current_price,
                    "call_strike": call_strike,
                    "put_strike": put_strike,
                    "expiration_days": expiration_days,
                    "call_price": call_result["option_price"],
                    "put_price": put_result["option_price"],
                    "total_cost": total_cost,
                    "max_profit": "Unlimited",
                    "max_loss": total_cost,
                    "breakeven_upper": breakeven_up,
                    "breakeven_lower": breakeven_down,
                    "best_scenario": "Very high volatility, large price movement",
                    "worst_scenario": f"Stock price between ${put_strike:.2f} and ${call_strike:.2f} at expiration"
                }
            
            else:
                return {"error": f"Strategy '{strategy_type}' not implemented yet"}
                
        except Exception as e:
            self.logger.error(f"Option strategy analysis error: {str(e)}")
            return {"error": f"Strategy analysis failed: {str(e)}"}


# Global instance
options_pricing = OptionsPricing()