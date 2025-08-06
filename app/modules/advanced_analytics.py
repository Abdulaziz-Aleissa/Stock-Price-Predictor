"""
Advanced analytics module - modular functionality for advanced financial analysis
"""
import logging

logger = logging.getLogger(__name__)

class AdvancedAnalytics:
    """Advanced analytics functionality"""
    
    def __init__(self):
        self.logger = logger
    
    def import_analytics_modules(self):
        """Import analytics modules with error handling"""
        try:
            from app.utils.value_at_risk import var_analyzer
            from app.utils.time_series_forecasting import ts_forecaster
            from app.utils.options_pricing import options_pricing
            
            return {
                'var_analyzer': var_analyzer,
                'ts_forecaster': ts_forecaster,
                'options_pricing': options_pricing
            }
        except ImportError as e:
            self.logger.error(f"Error importing analytics modules: {str(e)}")
            return None
    
    def validate_ticker(self, symbol):
        """Validate ticker symbol"""
        try:
            import yfinance as yf
            stock = yf.Ticker(symbol)
            hist = stock.history(period="1d")
            return not hist.empty
        except:
            return False
    
    def calculate_value_at_risk(self, symbol, portfolio_value, confidence_levels, holding_period):
        """
        Calculate Value at Risk analysis
        
        Args:
            symbol (str): Stock symbol
            portfolio_value (float): Portfolio value
            confidence_levels (list): List of confidence levels (e.g., [0.95, 0.99])
            holding_period (int): Holding period in days
            
        Returns:
            dict: VaR analysis results or error
        """
        try:
            # Input validation
            if not symbol or not symbol.strip():
                return {'error': 'Please provide a stock symbol'}
            
            symbol = symbol.strip().upper()
            
            if not self.validate_ticker(symbol):
                return {'error': 'Invalid ticker symbol'}
            
            if portfolio_value <= 0:
                return {'error': 'Portfolio value must be positive'}
            
            if holding_period <= 0 or holding_period > 252:  # Max 1 year
                return {'error': 'Holding period must be between 1 and 252 days'}
            
            if not confidence_levels:
                confidence_levels = [0.95, 0.99]  # Default
            
            # Import VaR analyzer
            modules = self.import_analytics_modules()
            if not modules:
                return {'error': 'Analytics modules not available'}
            
            var_analyzer = modules['var_analyzer']
            
            # Run VaR analysis
            results = var_analyzer.comprehensive_var_analysis(
                symbol, portfolio_value, confidence_levels, holding_period
            )
            
            if "error" in results:
                return results
            
            return {
                'success': True,
                'results': results
            }
            
        except Exception as e:
            self.logger.error(f"Error in VaR analysis: {str(e)}")
            return {'error': f'VaR analysis failed: {str(e)}'}
    
    def time_series_forecasting(self, symbol, forecast_days, include_volatility_forecast=True):
        """
        Perform time series forecasting
        
        Args:
            symbol (str): Stock symbol
            forecast_days (int): Number of days to forecast
            include_volatility_forecast (bool): Include volatility forecast
            
        Returns:
            dict: Forecasting results or error
        """
        try:
            # Input validation
            if not symbol or not symbol.strip():
                return {'error': 'Please provide a stock symbol'}
            
            symbol = symbol.strip().upper()
            
            if not self.validate_ticker(symbol):
                return {'error': 'Invalid ticker symbol'}
            
            if forecast_days <= 0 or forecast_days > 365:  # Max 1 year
                return {'error': 'Forecast days must be between 1 and 365'}
            
            # Import time series forecaster
            modules = self.import_analytics_modules()
            if not modules:
                return {'error': 'Analytics modules not available'}
            
            ts_forecaster = modules['ts_forecaster']
            
            # Run forecasting analysis
            results = ts_forecaster.comprehensive_forecast(
                symbol, forecast_days, include_volatility_forecast
            )
            
            if "error" in results:
                return results
            
            return {
                'success': True,
                'results': results
            }
            
        except Exception as e:
            self.logger.error(f"Error in time series forecasting: {str(e)}")
            return {'error': f'Time series forecasting failed: {str(e)}'}
    
    def options_pricing(self, symbol, strike_price, expiration_days, option_type='call', volatility=None):
        """
        Calculate options pricing
        
        Args:
            symbol (str): Stock symbol
            strike_price (float): Strike price
            expiration_days (int): Days to expiration
            option_type (str): 'call' or 'put'
            volatility (float): Custom volatility (optional)
            
        Returns:
            dict: Options pricing results or error
        """
        try:
            # Input validation
            if not symbol or not symbol.strip():
                return {'error': 'Please provide a stock symbol'}
            
            symbol = symbol.strip().upper()
            
            if not self.validate_ticker(symbol):
                return {'error': 'Invalid ticker symbol'}
            
            if strike_price <= 0:
                return {'error': 'Strike price must be positive'}
            
            if expiration_days <= 0 or expiration_days > 1825:  # Max 5 years
                return {'error': 'Expiration days must be between 1 and 1825'}
            
            if option_type.lower() not in ['call', 'put']:
                return {'error': 'Option type must be "call" or "put"'}
            
            # Process custom volatility
            if volatility is not None:
                if volatility <= 0 or volatility > 5:  # Max 500% volatility
                    return {'error': 'Volatility must be between 0% and 500%'}
            
            # Import options pricing
            modules = self.import_analytics_modules()
            if not modules:
                return {'error': 'Analytics modules not available'}
            
            options_pricing = modules['options_pricing']
            
            # Price option
            results = options_pricing.price_option(
                symbol, strike_price, expiration_days, option_type.lower(), volatility
            )
            
            if "error" in results:
                return results
            
            return {
                'success': True,
                'results': results
            }
            
        except Exception as e:
            self.logger.error(f"Error in options pricing: {str(e)}")
            return {'error': f'Options pricing failed: {str(e)}'}
    
    def comprehensive_analysis(self, symbol, analysis_types):
        """
        Perform comprehensive analysis based on selected types
        
        Args:
            symbol (str): Stock symbol
            analysis_types (dict): Dict of analysis types to perform
            
        Returns:
            dict: Comprehensive analysis results
        """
        try:
            if not symbol or not symbol.strip():
                return {'error': 'Please provide a stock symbol'}
            
            symbol = symbol.strip().upper()
            
            if not self.validate_ticker(symbol):
                return {'error': 'Invalid ticker symbol'}
            
            results = {'symbol': symbol, 'analyses': {}}
            
            # Value at Risk
            if analysis_types.get('var'):
                var_params = analysis_types['var']
                var_result = self.calculate_value_at_risk(
                    symbol,
                    var_params.get('portfolio_value', 10000),
                    var_params.get('confidence_levels', [0.95, 0.99]),
                    var_params.get('holding_period', 1)
                )
                results['analyses']['var'] = var_result
            
            # Time Series Forecasting
            if analysis_types.get('forecasting'):
                forecast_params = analysis_types['forecasting']
                forecast_result = self.time_series_forecasting(
                    symbol,
                    forecast_params.get('forecast_days', 30),
                    forecast_params.get('include_volatility', True)
                )
                results['analyses']['forecasting'] = forecast_result
            
            # Options Pricing
            if analysis_types.get('options'):
                options_params = analysis_types['options']
                options_result = self.options_pricing(
                    symbol,
                    options_params.get('strike_price'),
                    options_params.get('expiration_days'),
                    options_params.get('option_type', 'call'),
                    options_params.get('volatility')
                )
                results['analyses']['options'] = options_result
            
            return {
                'success': True,
                'results': results
            }
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive analysis: {str(e)}")
            return {'error': f'Comprehensive analysis failed: {str(e)}'}
    
    def get_analysis_summary(self, symbol):
        """Get a quick analysis summary for a stock"""
        try:
            if not self.validate_ticker(symbol):
                return {'error': 'Invalid ticker symbol'}
            
            import yfinance as yf
            
            # Get basic stock info
            stock = yf.Ticker(symbol)
            info = stock.info
            hist = stock.history(period="1mo")
            
            if hist.empty:
                return {'error': 'No data available for analysis'}
            
            # Calculate basic metrics
            current_price = hist['Close'].iloc[-1]
            month_start_price = hist['Close'].iloc[0]
            monthly_return = ((current_price - month_start_price) / month_start_price * 100)
            
            # Calculate volatility
            daily_returns = hist['Close'].pct_change().dropna()
            volatility = daily_returns.std() * 100
            
            summary = {
                'symbol': symbol,
                'current_price': float(current_price),
                'monthly_return': float(monthly_return),
                'volatility': float(volatility),
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('forwardPE'),
                'dividend_yield': info.get('dividendYield'),
                'beta': info.get('beta'),
                'recommendation': self._get_quick_recommendation(monthly_return, volatility)
            }
            
            return {
                'success': True,
                'summary': summary
            }
            
        except Exception as e:
            self.logger.error(f"Error getting analysis summary: {str(e)}")
            return {'error': f'Failed to get analysis summary: {str(e)}'}
    
    def _get_quick_recommendation(self, monthly_return, volatility):
        """Get a quick recommendation based on return and volatility"""
        if monthly_return > 5 and volatility < 20:
            return "Strong Buy - Good returns with low volatility"
        elif monthly_return > 0 and volatility < 25:
            return "Buy - Positive returns with manageable risk"
        elif monthly_return > -5 and volatility < 30:
            return "Hold - Moderate performance and risk"
        elif volatility > 40:
            return "Caution - High volatility, high risk"
        else:
            return "Sell - Poor performance indicators"

# Initialize the analytics module
advanced_analytics = AdvancedAnalytics()