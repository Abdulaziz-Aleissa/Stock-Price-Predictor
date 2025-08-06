"""
Stock comparison module - modular functionality for comparing multiple stocks
"""
import yfinance as yf
import logging

logger = logging.getLogger(__name__)

class StockComparison:
    """Stock comparison functionality"""
    
    def __init__(self):
        self.logger = logger
    
    def compare_stocks(self, symbol1, symbol2, timeframe):
        """
        Compare two stocks across various metrics
        
        Args:
            symbol1 (str): First stock symbol
            symbol2 (str): Second stock symbol  
            timeframe (str): Time period for comparison
            
        Returns:
            dict: Comparison data or error
        """
        try:
            symbol1 = symbol1.upper().strip()
            symbol2 = symbol2.upper().strip()
            
            # Validate inputs
            if not symbol1 or not symbol2:
                return {'error': 'Please provide both stock symbols'}
            
            if symbol1 == symbol2:
                return {'error': 'Please select different stocks to compare'}
            
            # Valid timeframes
            valid_timeframes = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
            if timeframe not in valid_timeframes:
                return {'error': f'Invalid timeframe. Valid options: {", ".join(valid_timeframes)}'}
            
            # Get data for both stocks
            stock1 = yf.Ticker(symbol1)
            stock2 = yf.Ticker(symbol2)
            
            hist1 = stock1.history(period=timeframe)
            hist2 = stock2.history(period=timeframe)
            
            # Check if data is available
            if hist1.empty:
                return {'error': f'No data available for {symbol1}'}
            if hist2.empty:
                return {'error': f'No data available for {symbol2}'}
            
            # Calculate metrics for stock 1
            stock1_data = self._calculate_stock_metrics(symbol1, hist1, stock1)
            
            # Calculate metrics for stock 2
            stock2_data = self._calculate_stock_metrics(symbol2, hist2, stock2)
            
            return {
                'success': True,
                'timeframe': timeframe,
                'symbol1': stock1_data,
                'symbol2': stock2_data,
                'comparison': self._compare_metrics(stock1_data, stock2_data)
            }
            
        except Exception as e:
            self.logger.error(f"Error comparing stocks {symbol1} vs {symbol2}: {str(e)}")
            return {'error': f'Comparison failed: {str(e)}'}
    
    def _calculate_stock_metrics(self, symbol, hist_data, ticker_obj):
        """Calculate various metrics for a stock"""
        try:
            # Price data
            prices = hist_data['Close'].tolist()
            dates = hist_data.index.strftime('%Y-%m-%d').tolist()
            
            # Performance metrics
            start_price = hist_data['Close'].iloc[0]
            end_price = hist_data['Close'].iloc[-1]
            total_return = ((end_price - start_price) / start_price * 100)
            
            # Volatility (standard deviation of daily returns)
            daily_returns = hist_data['Close'].pct_change().dropna()
            volatility = daily_returns.std() * 100
            
            # Volume metrics
            avg_volume = hist_data['Volume'].mean()
            
            # High/Low metrics
            period_high = hist_data['High'].max()
            period_low = hist_data['Low'].min()
            
            # Try to get additional info
            try:
                info = ticker_obj.info
                market_cap = info.get('marketCap', None)
                pe_ratio = info.get('forwardPE', None)
                dividend_yield = info.get('dividendYield', None)
            except:
                market_cap = None
                pe_ratio = None
                dividend_yield = None
            
            return {
                'symbol': symbol,
                'prices': prices,
                'dates': dates,
                'start_price': float(start_price),
                'end_price': float(end_price),
                'total_return': float(total_return),
                'volatility': float(volatility),
                'avg_volume': float(avg_volume),
                'period_high': float(period_high),
                'period_low': float(period_low),
                'market_cap': market_cap,
                'pe_ratio': pe_ratio,
                'dividend_yield': dividend_yield
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics for {symbol}: {str(e)}")
            return {
                'symbol': symbol,
                'error': f'Failed to calculate metrics: {str(e)}'
            }
    
    def _compare_metrics(self, stock1_data, stock2_data):
        """Compare metrics between two stocks"""
        try:
            comparison = {}
            
            # Performance comparison
            if 'total_return' in stock1_data and 'total_return' in stock2_data:
                if stock1_data['total_return'] > stock2_data['total_return']:
                    comparison['better_performer'] = stock1_data['symbol']
                    comparison['performance_difference'] = stock1_data['total_return'] - stock2_data['total_return']
                else:
                    comparison['better_performer'] = stock2_data['symbol']
                    comparison['performance_difference'] = stock2_data['total_return'] - stock1_data['total_return']
            
            # Volatility comparison
            if 'volatility' in stock1_data and 'volatility' in stock2_data:
                if stock1_data['volatility'] < stock2_data['volatility']:
                    comparison['less_volatile'] = stock1_data['symbol']
                    comparison['volatility_difference'] = stock2_data['volatility'] - stock1_data['volatility']
                else:
                    comparison['less_volatile'] = stock2_data['symbol']
                    comparison['volatility_difference'] = stock1_data['volatility'] - stock2_data['volatility']
            
            # Volume comparison
            if 'avg_volume' in stock1_data and 'avg_volume' in stock2_data:
                if stock1_data['avg_volume'] > stock2_data['avg_volume']:
                    comparison['higher_volume'] = stock1_data['symbol']
                else:
                    comparison['higher_volume'] = stock2_data['symbol']
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Error comparing metrics: {str(e)}")
            return {'error': 'Failed to compare metrics'}

# Initialize the comparison module
stock_comparison = StockComparison()