"""
Time Series Forecasting Module
Implements ARIMA and GARCH models for stock price and volatility forecasting
"""
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
import warnings
import logging
from datetime import datetime, timedelta

# Statistical modeling libraries
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller, acf, pacf
    from statsmodels.tsa.seasonal import seasonal_decompose
    from arch import arch_model
    ADVANCED_LIBS_AVAILABLE = True
except ImportError:
    ADVANCED_LIBS_AVAILABLE = False

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class TimeSeriesForecaster:
    """Advanced time series forecasting using ARIMA and GARCH models"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.advanced_available = ADVANCED_LIBS_AVAILABLE
    
    def get_stock_data(self, symbol, period="2y"):
        """Get historical stock data for time series analysis"""
        try:
            stock = yf.Ticker(symbol)
            hist_data = stock.history(period=period)
            
            if hist_data.empty:
                return None, None, None
            
            # Prepare data
            prices = hist_data['Close']
            returns = prices.pct_change().dropna()
            log_returns = np.log(prices / prices.shift(1)).dropna()
            
            return prices, returns, log_returns
            
        except Exception as e:
            self.logger.error(f"Error getting stock data for {symbol}: {str(e)}")
            return None, None, None
    
    def check_stationarity(self, series, significance_level=0.05):
        """
        Check if time series is stationary using Augmented Dickey-Fuller test
        
        Args:
            series: Time series data
            significance_level: Significance level for the test
            
        Returns:
            Dictionary with stationarity test results
        """
        try:
            if not self.advanced_available:
                return {"stationary": False, "error": "Advanced libraries not available"}
            
            # Perform ADF test
            adf_result = adfuller(series.dropna())
            
            is_stationary = adf_result[1] <= significance_level
            
            return {
                "stationary": is_stationary,
                "adf_statistic": float(adf_result[0]),
                "p_value": float(adf_result[1]),
                "critical_values": {
                    "1%": float(adf_result[4]['1%']),
                    "5%": float(adf_result[4]['5%']),
                    "10%": float(adf_result[4]['10%'])
                },
                "interpretation": "Stationary" if is_stationary else "Non-stationary"
            }
            
        except Exception as e:
            self.logger.error(f"Stationarity test error: {str(e)}")
            return {"stationary": False, "error": str(e)}
    
    def simple_arima_forecast(self, prices, forecast_days=30):
        """
        Simple ARIMA forecasting without advanced libraries
        Uses basic autoregressive approach
        """
        try:
            # Use simple moving average and trend for basic forecasting
            returns = prices.pct_change().dropna()
            
            # Calculate basic statistics
            mean_return = returns.mean()
            std_return = returns.std()
            
            # Simple trend calculation
            recent_prices = prices.tail(30)
            if len(recent_prices) > 1:
                trend = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / len(recent_prices)
            else:
                trend = 0
            
            # Generate forecast
            last_price = prices.iloc[-1]
            forecast_prices = []
            forecast_returns = []
            
            current_price = last_price
            for i in range(forecast_days):
                # Simple random walk with drift
                expected_return = mean_return + (trend / last_price) * 0.1  # Dampened trend
                random_component = np.random.normal(0, std_return * 0.5)  # Reduced volatility
                
                forecast_return = expected_return + random_component
                current_price = current_price * (1 + forecast_return)
                
                forecast_prices.append(float(current_price))
                forecast_returns.append(float(forecast_return))
            
            # Generate forecast dates
            last_date = prices.index[-1]
            forecast_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=forecast_days,
                freq='D'
            )
            
            # Calculate confidence intervals (simple approach)
            forecast_std = std_return * np.sqrt(np.arange(1, forecast_days + 1))
            upper_bound = [last_price * (1 + mean_return * i + 1.96 * forecast_std[i-1]) 
                          for i in range(1, forecast_days + 1)]
            lower_bound = [last_price * (1 + mean_return * i - 1.96 * forecast_std[i-1]) 
                          for i in range(1, forecast_days + 1)]
            
            return {
                "success": True,
                "method": "Simple Autoregressive",
                "forecast_prices": forecast_prices,
                "forecast_returns": forecast_returns,
                "forecast_dates": [d.strftime('%Y-%m-%d') for d in forecast_dates],
                "confidence_intervals": {
                    "upper_95": upper_bound,
                    "lower_95": lower_bound
                },
                "model_info": {
                    "method": "Simple random walk with drift",
                    "mean_return": float(mean_return),
                    "volatility": float(std_return),
                    "trend": float(trend)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Simple ARIMA forecast error: {str(e)}")
            return {"error": f"Simple forecasting failed: {str(e)}"}
    
    def arima_forecast(self, prices, forecast_days=30, max_p=3, max_d=2, max_q=3):
        """
        ARIMA model forecasting with automatic parameter selection
        
        Args:
            prices: Historical price series
            forecast_days: Number of days to forecast
            max_p, max_d, max_q: Maximum values for ARIMA parameters
            
        Returns:
            ARIMA forecast results
        """
        try:
            if not self.advanced_available:
                return self.simple_arima_forecast(prices, forecast_days)
            
            # Use log prices for better modeling
            log_prices = np.log(prices)
            
            # Check stationarity and difference if needed
            stationarity = self.check_stationarity(log_prices)
            
            # Auto-select ARIMA parameters (simplified approach)
            best_aic = float('inf')
            best_order = (1, 1, 1)  # Default
            
            for p in range(max_p + 1):
                for d in range(max_d + 1):
                    for q in range(max_q + 1):
                        try:
                            if p == 0 and d == 0 and q == 0:
                                continue
                            
                            model = ARIMA(log_prices, order=(p, d, q))
                            fitted_model = model.fit()
                            
                            if fitted_model.aic < best_aic:
                                best_aic = fitted_model.aic
                                best_order = (p, d, q)
                        except:
                            continue
            
            # Fit the best model
            final_model = ARIMA(log_prices, order=best_order)
            fitted_model = final_model.fit()
            
            # Generate forecasts
            forecast_result = fitted_model.forecast(steps=forecast_days, alpha=0.05)
            forecast_log_prices = forecast_result
            
            # Convert back to actual prices
            forecast_prices = np.exp(forecast_log_prices).tolist()
            
            # Generate forecast dates
            last_date = prices.index[-1]
            forecast_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=forecast_days,
                freq='D'
            )
            
            # Get confidence intervals
            forecast_ci = fitted_model.get_forecast(steps=forecast_days).conf_int()
            upper_bound = np.exp(forecast_ci.iloc[:, 1]).tolist()
            lower_bound = np.exp(forecast_ci.iloc[:, 0]).tolist()
            
            # Calculate returns from forecasted prices
            last_price = prices.iloc[-1]
            forecast_returns = [(forecast_prices[0] / last_price) - 1]
            for i in range(1, len(forecast_prices)):
                ret = (forecast_prices[i] / forecast_prices[i-1]) - 1
                forecast_returns.append(ret)
            
            return {
                "success": True,
                "method": "ARIMA",
                "arima_order": best_order,
                "forecast_prices": [float(p) for p in forecast_prices],
                "forecast_returns": [float(r) for r in forecast_returns],
                "forecast_dates": [d.strftime('%Y-%m-%d') for d in forecast_dates],
                "confidence_intervals": {
                    "upper_95": [float(u) for u in upper_bound],
                    "lower_95": [float(l) for l in lower_bound]
                },
                "model_info": {
                    "aic": float(fitted_model.aic),
                    "bic": float(fitted_model.bic),
                    "p_value_params": fitted_model.pvalues.tolist(),
                    "residual_std": float(np.std(fitted_model.resid))
                },
                "stationarity_test": stationarity
            }
            
        except Exception as e:
            self.logger.error(f"ARIMA forecast error: {str(e)}")
            # Fallback to simple method
            return self.simple_arima_forecast(prices, forecast_days)
    
    def simple_volatility_forecast(self, returns, forecast_days=30):
        """
        Simple volatility forecasting without GARCH
        """
        try:
            # Calculate rolling volatility
            rolling_vol = returns.rolling(window=30).std()
            
            # Current volatility estimate
            current_vol = rolling_vol.iloc[-1]
            
            # Simple volatility forecast (mean reversion to long-term average)
            long_term_vol = returns.std()
            mean_reversion_speed = 0.1  # Adjust towards long-term mean
            
            forecast_vols = []
            vol = current_vol
            
            for i in range(forecast_days):
                # Mean reversion formula
                vol = vol + mean_reversion_speed * (long_term_vol - vol) + np.random.normal(0, long_term_vol * 0.1)
                vol = max(0.001, vol)  # Ensure positive volatility
                forecast_vols.append(float(vol))
            
            # Generate forecast dates
            last_date = returns.index[-1]
            forecast_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=forecast_days,
                freq='D'
            )
            
            return {
                "success": True,
                "method": "Simple Mean Reversion",
                "forecast_volatility": forecast_vols,
                "forecast_dates": [d.strftime('%Y-%m-%d') for d in forecast_dates],
                "current_volatility": float(current_vol),
                "long_term_volatility": float(long_term_vol),
                "annualized_current_vol": float(current_vol * np.sqrt(252) * 100),
                "annualized_longterm_vol": float(long_term_vol * np.sqrt(252) * 100)
            }
            
        except Exception as e:
            self.logger.error(f"Simple volatility forecast error: {str(e)}")
            return {"error": f"Simple volatility forecasting failed: {str(e)}"}
    
    def garch_volatility_forecast(self, returns, forecast_days=30):
        """
        GARCH model for volatility forecasting
        
        Args:
            returns: Historical return series
            forecast_days: Number of days to forecast volatility
            
        Returns:
            GARCH volatility forecast results
        """
        try:
            if not self.advanced_available:
                return self.simple_volatility_forecast(returns, forecast_days)
            
            # Remove any infinite or extremely large values
            clean_returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
            clean_returns = clean_returns[np.abs(clean_returns) < 0.5]  # Remove extreme outliers
            
            if len(clean_returns) < 100:
                return {"error": "Insufficient clean data for GARCH modeling"}
            
            # Convert to percentage returns for better numerical stability
            percent_returns = clean_returns * 100
            
            # Fit GARCH(1,1) model
            garch_model = arch_model(
                percent_returns, 
                vol='Garch', 
                p=1, 
                q=1,
                mean='Constant',
                dist='normal'
            ).fit(disp='off')
            
            # Generate volatility forecasts
            volatility_forecast = garch_model.forecast(horizon=forecast_days)
            forecast_variance = volatility_forecast.variance.iloc[-1].values
            forecast_vols = np.sqrt(forecast_variance) / 100  # Convert back to decimal
            
            # Generate forecast dates
            last_date = returns.index[-1]
            forecast_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=forecast_days,
                freq='D'
            )
            
            # Current volatility
            current_vol = float(garch_model.conditional_volatility.iloc[-1] / 100)
            
            return {
                "success": True,
                "method": "GARCH(1,1)",
                "forecast_volatility": [float(v) for v in forecast_vols],
                "forecast_dates": [d.strftime('%Y-%m-%d') for d in forecast_dates],
                "current_volatility": current_vol,
                "annualized_forecast_vol": [float(v * np.sqrt(252) * 100) for v in forecast_vols],
                "annualized_current_vol": float(current_vol * np.sqrt(252) * 100),
                "model_info": {
                    "log_likelihood": float(garch_model.loglikelihood),
                    "aic": float(garch_model.aic),
                    "bic": float(garch_model.bic),
                    "alpha": float(garch_model.params['alpha[1]']),
                    "beta": float(garch_model.params['beta[1]']),
                    "omega": float(garch_model.params['omega'])
                }
            }
            
        except Exception as e:
            self.logger.error(f"GARCH forecast error: {str(e)}")
            # Fallback to simple method
            return self.simple_volatility_forecast(returns, forecast_days)
    
    def comprehensive_forecast(self, symbol, forecast_days=30, include_volatility=True):
        """
        Comprehensive time series forecasting combining price and volatility predictions
        
        Args:
            symbol: Stock ticker symbol
            forecast_days: Number of days to forecast
            include_volatility: Whether to include volatility forecasting
            
        Returns:
            Complete forecasting analysis
        """
        try:
            # Get stock data
            prices, returns, log_returns = self.get_stock_data(symbol)
            if prices is None:
                return {"error": "Unable to fetch stock data"}
            
            if len(prices) < 100:
                return {"error": "Insufficient historical data for forecasting"}
            
            # Price forecasting using ARIMA
            price_forecast = self.arima_forecast(prices, forecast_days)
            
            # Volatility forecasting using GARCH
            volatility_forecast = None
            if include_volatility:
                volatility_forecast = self.garch_volatility_forecast(returns, forecast_days)
            
            # Calculate forecast accuracy metrics on historical data
            # Use the last 30 days for validation
            if len(prices) > 60:
                validation_size = min(30, len(prices) // 4)
                train_prices = prices[:-validation_size]
                actual_prices = prices[-validation_size:]
                
                # Generate forecast for validation period
                validation_forecast = self.arima_forecast(train_prices, validation_size)
                
                if validation_forecast.get("success"):
                    forecast_vals = validation_forecast["forecast_prices"][:len(actual_prices)]
                    actual_vals = actual_prices.values
                    
                    # Calculate metrics
                    mae = np.mean(np.abs(np.array(forecast_vals) - actual_vals))
                    rmse = np.sqrt(np.mean((np.array(forecast_vals) - actual_vals) ** 2))
                    mape = np.mean(np.abs((actual_vals - np.array(forecast_vals)) / actual_vals)) * 100
                    
                    accuracy_metrics = {
                        "mae": float(mae),
                        "rmse": float(rmse),
                        "mape": float(mape),
                        "validation_days": validation_size
                    }
                else:
                    accuracy_metrics = {"error": "Could not calculate validation metrics"}
            else:
                accuracy_metrics = {"error": "Insufficient data for validation"}
            
            # Generate insights and recommendations
            insights = []
            recommendations = []
            
            if price_forecast.get("success"):
                forecast_prices = price_forecast["forecast_prices"]
                current_price = prices.iloc[-1]
                
                # Price trend analysis
                price_change = (forecast_prices[-1] - current_price) / current_price * 100
                if price_change > 5:
                    insights.append(f"Model forecasts significant upward trend (+{price_change:.1f}%)")
                    recommendations.append("Consider long positions if fundamentals support the forecast")
                elif price_change < -5:
                    insights.append(f"Model forecasts significant downward trend ({price_change:.1f}%)")
                    recommendations.append("Consider risk management strategies or short positions")
                else:
                    insights.append("Model forecasts relatively stable price movement")
                    recommendations.append("Suitable for range-bound trading strategies")
            
            if volatility_forecast and volatility_forecast.get("success"):
                forecast_vols = volatility_forecast["forecast_volatility"]
                current_vol = volatility_forecast["current_volatility"]
                
                # Volatility trend analysis
                avg_forecast_vol = np.mean(forecast_vols)
                vol_change = (avg_forecast_vol - current_vol) / current_vol * 100
                
                if vol_change > 20:
                    insights.append(f"Volatility expected to increase significantly (+{vol_change:.1f}%)")
                    recommendations.append("Consider volatility strategies like straddles or protective puts")
                elif vol_change < -20:
                    insights.append(f"Volatility expected to decrease significantly ({vol_change:.1f}%)")
                    recommendations.append("Good environment for covered calls or cash-secured puts")
            
            if not insights:
                insights.append("Forecasts suggest normal market conditions")
                recommendations.append("Standard investment strategies should perform adequately")
            
            return {
                "success": True,
                "symbol": symbol,
                "current_price": float(prices.iloc[-1]),
                "forecast_days": forecast_days,
                "data_points": len(prices),
                "price_forecast": price_forecast,
                "volatility_forecast": volatility_forecast,
                "accuracy_metrics": accuracy_metrics,
                "analysis": {
                    "insights": insights,
                    "recommendations": recommendations
                },
                "historical_data": {
                    "prices": prices.tail(60).tolist(),  # Last 60 days for visualization
                    "returns": returns.tail(60).tolist(),
                    "dates": [d.strftime('%Y-%m-%d') for d in prices.tail(60).index]
                }
            }
            
        except Exception as e:
            self.logger.error(f"Comprehensive forecast error: {str(e)}")
            return {"error": f"Time series forecasting failed: {str(e)}"}


# Global instance
ts_forecaster = TimeSeriesForecaster()