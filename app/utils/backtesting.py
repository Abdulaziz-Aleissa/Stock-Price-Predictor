"""
Backtesting Framework for Stock Trading Strategies
Self-contained implementation using only Python standard library
"""
import random
import math
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class BacktestingFramework:
    """Backtesting framework for evaluating trading strategies"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def _generate_mock_data(self, symbol, num_days=504):
        """Generate realistic mock stock data for backtesting"""
        # Mock prices based on symbol
        base_prices = {
            'AAPL': 150.0, 'GOOGL': 2800.0, 'MSFT': 300.0, 'TSLA': 200.0,
            'AMZN': 3000.0, 'META': 250.0, 'NVDA': 400.0, 'SPY': 400.0,
            'QQQ': 350.0, 'NFLX': 400.0, 'AMD': 90.0, 'INTC': 50.0
        }
        
        start_price = base_prices.get(symbol.upper(), 100.0)
        
        # Generate realistic price movements
        random.seed(42)  # For consistent demo data
        prices = [start_price]
        
        # Simulate daily returns with realistic parameters
        mu = 0.0005  # Average daily return
        sigma = 0.02  # Daily volatility
        
        for i in range(num_days - 1):
            # Generate random return using Box-Muller transform
            u1 = random.random()
            u2 = random.random()
            z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
            daily_return = mu + sigma * z
            
            new_price = prices[-1] * (1 + daily_return)
            prices.append(max(new_price, 0.01))  # Ensure price doesn't go negative
        
        # Generate dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=num_days)
        dates = []
        current_date = start_date
        
        for i in range(num_days):
            # Skip weekends for trading days
            while current_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
                current_date += timedelta(days=1)
            dates.append(current_date)
            current_date += timedelta(days=1)
        
        # Create mock OHLCV data
        data = []
        for i, price in enumerate(prices):
            open_price = price * random.uniform(0.995, 1.005)
            high_price = price * random.uniform(1.005, 1.03)
            low_price = price * random.uniform(0.97, 0.995)
            volume = random.randint(1000000, 10000000)
            
            data.append({
                'Date': dates[i] if i < len(dates) else dates[-1] + timedelta(days=i - len(dates) + 1),
                'Open': open_price,
                'High': high_price,
                'Low': low_price,
                'Close': price,
                'Volume': volume
            })
        
        self.logger.info(f"Generated mock data for {symbol}: {len(data)} days, starting at ${start_price:.2f}")
        return data
    
    def _calculate_moving_average(self, prices, window):
        """Calculate simple moving average"""
        if len(prices) < window:
            return [None] * len(prices)
        
        averages = [None] * (window - 1)
        
        for i in range(window - 1, len(prices)):
            avg = sum(prices[i - window + 1:i + 1]) / window
            averages.append(avg)
        
        return averages
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return [None] * len(prices)
        
        rsi_values = [None] * period
        
        # Calculate price changes
        changes = []
        for i in range(1, len(prices)):
            changes.append(prices[i] - prices[i-1])
        
        # Calculate initial average gains and losses
        gains = [max(0, change) for change in changes[:period]]
        losses = [max(0, -change) for change in changes[:period]]
        
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        
        # Calculate first RSI
        if avg_loss == 0:
            rsi_values.append(100)
        else:
            rs = avg_gain / avg_loss
            rsi_values.append(100 - (100 / (1 + rs)))
        
        # Calculate subsequent RSI values using smoothed averages
        for i in range(period, len(changes)):
            change = changes[i]
            gain = max(0, change)
            loss = max(0, -change)
            
            avg_gain = (avg_gain * (period - 1) + gain) / period
            avg_loss = (avg_loss * (period - 1) + loss) / period
            
            if avg_loss == 0:
                rsi_values.append(100)
            else:
                rs = avg_gain / avg_loss
                rsi_values.append(100 - (100 / (1 + rs)))
        
        return rsi_values
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD (Moving Average Convergence Divergence)"""
        if len(prices) < slow:
            return [None] * len(prices), [None] * len(prices)
        
        # Calculate exponential moving averages
        def calc_ema(data, period):
            multiplier = 2 / (period + 1)
            ema = [data[0]]  # Start with first price
            
            for i in range(1, len(data)):
                ema.append((data[i] * multiplier) + (ema[i-1] * (1 - multiplier)))
            
            return ema
        
        ema_fast = calc_ema(prices, fast)
        ema_slow = calc_ema(prices, slow)
        
        # Calculate MACD line
        macd_line = []
        for i in range(len(prices)):
            if i < slow - 1:
                macd_line.append(None)
            else:
                macd_line.append(ema_fast[i] - ema_slow[i])
        
        # Calculate signal line (EMA of MACD)
        macd_values = [x for x in macd_line if x is not None]
        if len(macd_values) >= signal:
            signal_ema = calc_ema(macd_values, signal)
            signal_line = [None] * (len(macd_line) - len(signal_ema)) + signal_ema
        else:
            signal_line = [None] * len(macd_line)
        
        return macd_line, signal_line
    
    def _calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            return [None] * len(prices), [None] * len(prices), [None] * len(prices)
        
        middle_band = self._calculate_moving_average(prices, period)
        upper_band = []
        lower_band = []
        
        for i in range(len(prices)):
            if middle_band[i] is None:
                upper_band.append(None)
                lower_band.append(None)
            else:
                # Calculate standard deviation for the period
                start_idx = i - period + 1
                period_prices = prices[start_idx:i + 1]
                mean = sum(period_prices) / len(period_prices)
                variance = sum((p - mean) ** 2 for p in period_prices) / len(period_prices)
                std = math.sqrt(variance)
                
                upper_band.append(middle_band[i] + std_dev * std)
                lower_band.append(middle_band[i] - std_dev * std)
        
        return upper_band, middle_band, lower_band
    
    def moving_average_crossover_strategy(self, data, short_window=20, long_window=50):
        """Simple moving average crossover strategy"""
        prices = [d['Close'] for d in data]
        
        short_ma = self._calculate_moving_average(prices, short_window)
        long_ma = self._calculate_moving_average(prices, long_window)
        
        signals = []
        for i in range(len(data)):
            if (short_ma[i] is not None and long_ma[i] is not None and
                i > 0 and short_ma[i-1] is not None and long_ma[i-1] is not None):
                
                # Buy signal: short MA crosses above long MA
                if short_ma[i] > long_ma[i] and short_ma[i-1] <= long_ma[i-1]:
                    signals.append(1)
                # Sell signal: short MA crosses below long MA
                elif short_ma[i] < long_ma[i] and short_ma[i-1] >= long_ma[i-1]:
                    signals.append(-1)
                else:
                    signals.append(0)
            else:
                signals.append(0)
        
        return signals
    
    def rsi_strategy(self, data, rsi_oversold=30, rsi_overbought=70):
        """RSI-based mean reversion strategy"""
        prices = [d['Close'] for d in data]
        rsi_values = self._calculate_rsi(prices)
        
        signals = []
        for i, rsi in enumerate(rsi_values):
            if rsi is None:
                signals.append(0)
            elif rsi < rsi_oversold:
                signals.append(1)  # Buy signal (oversold)
            elif rsi > rsi_overbought:
                signals.append(-1)  # Sell signal (overbought)
            else:
                signals.append(0)
        
        return signals
    
    def macd_strategy(self, data):
        """MACD strategy"""
        prices = [d['Close'] for d in data]
        macd_line, signal_line = self._calculate_macd(prices)
        
        signals = []
        for i in range(len(data)):
            if (macd_line[i] is not None and signal_line[i] is not None and
                i > 0 and macd_line[i-1] is not None and signal_line[i-1] is not None):
                
                # Buy signal: MACD crosses above signal line
                if macd_line[i] > signal_line[i] and macd_line[i-1] <= signal_line[i-1]:
                    signals.append(1)
                # Sell signal: MACD crosses below signal line
                elif macd_line[i] < signal_line[i] and macd_line[i-1] >= signal_line[i-1]:
                    signals.append(-1)
                else:
                    signals.append(0)
            else:
                signals.append(0)
        
        return signals
    
    def bollinger_bands_strategy(self, data):
        """Bollinger Bands mean reversion strategy"""
        prices = [d['Close'] for d in data]
        upper_band, middle_band, lower_band = self._calculate_bollinger_bands(prices)
        
        signals = []
        for i in range(len(data)):
            if upper_band[i] is not None and lower_band[i] is not None:
                # Buy signal: price touches lower band (oversold)
                if prices[i] <= lower_band[i]:
                    signals.append(1)
                # Sell signal: price touches upper band (overbought)
                elif prices[i] >= upper_band[i]:
                    signals.append(-1)
                else:
                    signals.append(0)
            else:
                signals.append(0)
        
        return signals
    
    def buy_and_hold_strategy(self, data):
        """Simple buy and hold strategy for comparison"""
        return [1] + [0] * (len(data) - 1)  # Buy on first day, hold thereafter
    
    def _calculate_performance_metrics(self, returns):
        """Calculate comprehensive performance metrics"""
        try:
            if not returns:
                return {}
            
            # Basic metrics
            total_return = 1.0
            for ret in returns:
                total_return *= (1 + ret)
            total_return -= 1
            
            annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
            
            # Volatility
            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
            volatility = math.sqrt(variance) * math.sqrt(252)
            
            # Sharpe ratio (assuming 2% risk-free rate)
            risk_free_rate = 0.02
            sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
            
            # Downside metrics
            negative_returns = [r for r in returns if r < 0]
            if negative_returns:
                downside_variance = sum(r ** 2 for r in negative_returns) / len(negative_returns)
                downside_deviation = math.sqrt(downside_variance) * math.sqrt(252)
                sortino_ratio = (annualized_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
            else:
                sortino_ratio = 0
            
            # Drawdown analysis
            cumulative = [1.0]
            for ret in returns:
                cumulative.append(cumulative[-1] * (1 + ret))
            
            running_max = [cumulative[0]]
            for i in range(1, len(cumulative)):
                running_max.append(max(running_max[-1], cumulative[i]))
            
            drawdowns = [(cumulative[i] - running_max[i]) / running_max[i] for i in range(len(cumulative))]
            max_drawdown = min(drawdowns) if drawdowns else 0
            
            # Win rate
            positive_returns = [r for r in returns if r > 0]
            win_rate = len(positive_returns) / len(returns) * 100 if returns else 0
            
            return {
                'total_return': total_return * 100,
                'annualized_return': annualized_return * 100,
                'volatility': volatility * 100,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown * 100,
                'win_rate': win_rate,
                'total_trades': len([r for r in returns if r != 0])
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {str(e)}")
            return {}
    
    def backtest_strategy(self, symbol, strategy_name, initial_capital=10000, commission=0.001):
        """
        Comprehensive backtesting of a trading strategy
        
        Args:
            symbol: Stock ticker symbol
            strategy_name: Name of strategy to test
            initial_capital: Starting capital
            commission: Commission rate per trade
            
        Returns:
            Backtesting results dictionary
        """
        try:
            self.logger.info(f"Running backtest for {symbol} using {strategy_name} strategy with ${initial_capital} initial capital")
            
            # Generate historical mock data (2 years)
            data = self._generate_mock_data(symbol, num_days=504)
            
            if len(data) < 50:  # Need at least 50 days for meaningful backtest
                error_msg = f"Insufficient historical data for {symbol} (only {len(data)} days available)"
                self.logger.error(error_msg)
                return {"error": error_msg}
            
            # Apply strategy
            if strategy_name == "moving_average_crossover":
                signals = self.moving_average_crossover_strategy(data)
            elif strategy_name == "rsi_strategy":
                signals = self.rsi_strategy(data)
            elif strategy_name == "macd_strategy":
                signals = self.macd_strategy(data)
            elif strategy_name == "bollinger_bands":
                signals = self.bollinger_bands_strategy(data)
            elif strategy_name == "buy_and_hold":
                signals = self.buy_and_hold_strategy(data)
            else:
                error_msg = f"Unknown strategy: {strategy_name}"
                self.logger.error(error_msg)
                return {"error": error_msg}
            
            # Simulate trading
            portfolio_value = initial_capital
            position = 0
            cash = initial_capital
            trades = []
            portfolio_values = [initial_capital]
            
            for i, (day_data, signal) in enumerate(zip(data, signals)):
                current_price = day_data['Close']
                
                # Buy signal
                if signal == 1 and position == 0:
                    shares_to_buy = int(cash / (current_price * (1 + commission)))
                    if shares_to_buy > 0:
                        cost = shares_to_buy * current_price * (1 + commission)
                        cash -= cost
                        position = shares_to_buy
                        trades.append({
                            'date': day_data['Date'].strftime('%Y-%m-%d'),
                            'action': 'BUY',
                            'shares': shares_to_buy,
                            'price': current_price,
                            'cost': cost
                        })
                
                # Sell signal
                elif signal == -1 and position > 0:
                    proceeds = position * current_price * (1 - commission)
                    cash += proceeds
                    trades.append({
                        'date': day_data['Date'].strftime('%Y-%m-%d'),
                        'action': 'SELL',
                        'shares': position,
                        'price': current_price,
                        'proceeds': proceeds
                    })
                    position = 0
                
                # Calculate portfolio value
                portfolio_value = cash + (position * current_price)
                portfolio_values.append(portfolio_value)
            
            # Calculate returns
            strategy_returns = []
            for i in range(1, len(portfolio_values)):
                ret = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
                strategy_returns.append(ret)
            
            # Calculate benchmark returns (buy and hold)
            benchmark_returns = []
            for i in range(1, len(data)):
                ret = (data[i]['Close'] - data[i-1]['Close']) / data[i-1]['Close']
                benchmark_returns.append(ret)
            
            # Performance metrics
            performance = self._calculate_performance_metrics(strategy_returns)
            benchmark_performance = self._calculate_performance_metrics(benchmark_returns)
            
            # Generate visualization data
            dates = [d['Date'].strftime('%Y-%m-%d') for d in data]
            
            # Calculate cumulative returns for plotting
            strategy_cumulative = [initial_capital]
            for ret in strategy_returns:
                strategy_cumulative.append(strategy_cumulative[-1] * (1 + ret))
            
            benchmark_cumulative = [initial_capital]
            for ret in benchmark_returns:
                benchmark_cumulative.append(benchmark_cumulative[-1] * (1 + ret))
            
            self.logger.info(f"Backtesting completed successfully for {symbol}: {len(trades)} trades, final value ${portfolio_values[-1]:.2f}")
            
            return {
                "success": True,
                "symbol": symbol,
                "strategy": strategy_name,
                "initial_capital": initial_capital,
                "final_portfolio_value": portfolio_values[-1],
                "total_trades": len(trades),
                "performance": performance,
                "benchmark_performance": benchmark_performance,
                "dates": dates,
                "strategy_equity_curve": strategy_cumulative,
                "benchmark_equity_curve": benchmark_cumulative,
                "trades": trades[-10:],  # Last 10 trades for display
                "outperformance": performance.get('total_return', 0) - benchmark_performance.get('total_return', 0)
            }
            
        except Exception as e:
            error_msg = f"Backtesting error for {symbol}: {str(e)}"
            self.logger.error(error_msg)
            return {"error": error_msg}


# Global instance
backtesting_framework = BacktestingFramework()