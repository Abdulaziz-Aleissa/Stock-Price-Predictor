"""
Paper trading module - modular functionality for virtual trading
"""
import logging
from datetime import datetime
import yfinance as yf

logger = logging.getLogger(__name__)

class PaperTrading:
    """Paper trading functionality"""
    
    def __init__(self, db_session):
        self.db = db_session
        self.logger = logger
    
    def validate_ticker(self, symbol):
        """Validate if ticker symbol exists"""
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period="1d")
            return not hist.empty
        except:
            return False
    
    def get_current_price(self, symbol):
        """Get current stock price"""
        try:
            stock = yf.Ticker(symbol)
            # Get real-time price during market hours
            real_time_price = stock.info.get('regularMarketPrice')
            if real_time_price:
                return real_time_price
            # If market closed, get latest closing price
            hist = stock.history(period="1d", interval="1m")
            if not hist.empty:
                return hist['Close'].iloc[-1]
            return None
        except Exception as e:
            self.logger.error(f"Error getting price for {symbol}: {str(e)}")
            return None
    
    def validate_buy_order(self, symbol, shares, price, cash_balance):
        """
        Validate a buy order
        
        Args:
            symbol (str): Stock symbol
            shares (float): Number of shares
            price (float): Price per share
            cash_balance (float): Available cash
            
        Returns:
            dict: Validation result
        """
        try:
            # Input validation
            if not symbol or not symbol.strip():
                return {'valid': False, 'error': 'Please enter a stock symbol'}
            
            symbol = symbol.strip().upper()
            
            # Basic symbol format validation
            if not symbol.isalpha() or len(symbol) < 1 or len(symbol) > 5:
                return {'valid': False, 'error': f'Invalid stock symbol format: "{symbol}". Please enter 1-5 letters only.'}
            
            if not self.validate_ticker(symbol):
                return {'valid': False, 'error': f'Stock symbol "{symbol}" not found. Please check the symbol and try again.'}
            
            if shares <= 0:
                return {'valid': False, 'error': 'Number of shares must be positive'}
                
            if price <= 0:
                return {'valid': False, 'error': 'Share price must be positive'}
            
            total_cost = shares * price
            
            if cash_balance < total_cost:
                return {'valid': False, 'error': f'Insufficient virtual cash. Available: ${cash_balance:.2f}, Required: ${total_cost:.2f}'}
            
            return {'valid': True, 'total_cost': total_cost}
            
        except ValueError:
            return {'valid': False, 'error': 'Invalid input values. Please check your numbers and try again.'}
        except Exception as e:
            return {'valid': False, 'error': f'Validation error: {str(e)}'}
    
    def validate_sell_order(self, symbol, shares, price, user_position):
        """
        Validate a sell order
        
        Args:
            symbol (str): Stock symbol
            shares (float): Number of shares
            price (float): Price per share
            user_position: User's current position in this stock
            
        Returns:
            dict: Validation result
        """
        try:
            # Input validation
            if not symbol or not symbol.strip():
                return {'valid': False, 'error': 'Please enter a stock symbol'}
            
            symbol = symbol.strip().upper()
            
            if shares <= 0:
                return {'valid': False, 'error': 'Number of shares must be positive'}
                
            if price <= 0:
                return {'valid': False, 'error': 'Share price must be positive'}
            
            # Check if user has enough shares
            if not user_position:
                return {'valid': False, 'error': f'You do not own any shares of {symbol}'}
            
            if user_position.shares < shares:
                return {'valid': False, 'error': f'Insufficient shares to sell. You own {user_position.shares} shares of {symbol}, but tried to sell {shares}'}
            
            total_proceeds = shares * price
            
            return {'valid': True, 'total_proceeds': total_proceeds}
            
        except ValueError:
            return {'valid': False, 'error': 'Invalid input values. Please check your numbers and try again.'}
        except Exception as e:
            return {'valid': False, 'error': f'Validation error: {str(e)}'}
    
    def execute_buy_order(self, user_id, symbol, shares, price, PaperCashBalance, PaperTransaction, PaperPortfolio):
        """
        Execute a buy order
        
        Args:
            user_id: User ID
            symbol: Stock symbol
            shares: Number of shares
            price: Price per share
            PaperCashBalance: Database model for cash balance
            PaperTransaction: Database model for transactions
            PaperPortfolio: Database model for portfolio
            
        Returns:
            dict: Execution result
        """
        try:
            # Get or create cash balance
            cash_balance = self.db.query(PaperCashBalance).filter_by(user_id=user_id).first()
            if not cash_balance:
                cash_balance = PaperCashBalance(user_id=user_id, cash_balance=100000.0)
                self.db.add(cash_balance)
                self.db.commit()
            
            # Validate the order
            validation = self.validate_buy_order(symbol, shares, price, cash_balance.cash_balance)
            if not validation['valid']:
                return validation
            
            total_cost = validation['total_cost']
            
            # Execute transaction
            cash_balance.cash_balance -= total_cost
            cash_balance.updated_at = datetime.now()
            
            # Create transaction record
            transaction = PaperTransaction(
                user_id=user_id,
                stock_symbol=symbol,
                transaction_type='BUY',
                shares=shares,
                price=price,
                total_amount=total_cost
            )
            self.db.add(transaction)
            
            # Update portfolio
            self._update_portfolio_buy(user_id, symbol, shares, price, PaperPortfolio)
            
            self.db.commit()
            
            return {
                'success': True,
                'message': f'Successfully bought {shares} shares of {symbol} at ${price:.2f}',
                'total_cost': total_cost
            }
            
        except Exception as e:
            self.db.rollback()
            self.logger.error(f"Error executing buy order: {str(e)}")
            return {'success': False, 'error': f'Error executing buy order: {str(e)}'}
    
    def execute_sell_order(self, user_id, symbol, shares, price, PaperCashBalance, PaperTransaction, PaperPortfolio):
        """
        Execute a sell order
        
        Args:
            user_id: User ID
            symbol: Stock symbol
            shares: Number of shares
            price: Price per share
            PaperCashBalance: Database model for cash balance
            PaperTransaction: Database model for transactions
            PaperPortfolio: Database model for portfolio
            
        Returns:
            dict: Execution result
        """
        try:
            # Get user position
            position = self.db.query(PaperPortfolio).filter_by(
                user_id=user_id, 
                stock_symbol=symbol
            ).first()
            
            # Validate the order
            validation = self.validate_sell_order(symbol, shares, price, position)
            if not validation['valid']:
                return validation
            
            total_proceeds = validation['total_proceeds']
            
            # Get cash balance
            cash_balance = self.db.query(PaperCashBalance).filter_by(user_id=user_id).first()
            if not cash_balance:
                cash_balance = PaperCashBalance(user_id=user_id, cash_balance=100000.0)
                self.db.add(cash_balance)
            
            # Execute transaction
            cash_balance.cash_balance += total_proceeds
            cash_balance.updated_at = datetime.now()
            
            # Create transaction record
            transaction = PaperTransaction(
                user_id=user_id,
                stock_symbol=symbol,
                transaction_type='SELL',
                shares=shares,
                price=price,
                total_amount=total_proceeds
            )
            self.db.add(transaction)
            
            # Update portfolio
            self._update_portfolio_sell(user_id, symbol, shares, price, PaperPortfolio)
            
            self.db.commit()
            
            return {
                'success': True,
                'message': f'Successfully sold {shares} shares of {symbol} at ${price:.2f}',
                'total_proceeds': total_proceeds
            }
            
        except Exception as e:
            self.db.rollback()
            self.logger.error(f"Error executing sell order: {str(e)}")
            return {'success': False, 'error': f'Error executing sell order: {str(e)}'}
    
    def _update_portfolio_buy(self, user_id, symbol, shares, price, PaperPortfolio):
        """Update portfolio after a buy transaction"""
        position = self.db.query(PaperPortfolio).filter_by(user_id=user_id, stock_symbol=symbol).first()
        
        if position:
            # Update average price using weighted average
            total_shares = position.shares + shares
            total_cost = (position.shares * position.average_price) + (shares * price)
            position.average_price = total_cost / total_shares
            position.shares = total_shares
            position.updated_at = datetime.now()
        else:
            # Create new position
            position = PaperPortfolio(
                user_id=user_id,
                stock_symbol=symbol,
                shares=shares,
                average_price=price
            )
            self.db.add(position)
    
    def _update_portfolio_sell(self, user_id, symbol, shares, price, PaperPortfolio):
        """Update portfolio after a sell transaction"""
        position = self.db.query(PaperPortfolio).filter_by(user_id=user_id, stock_symbol=symbol).first()
        
        if position and position.shares >= shares:
            position.shares -= shares
            position.updated_at = datetime.now()
            # Remove position if no shares left
            if position.shares == 0:
                self.db.delete(position)
            return True
        else:
            return False
    
    def get_portfolio_summary(self, user_id, PaperPortfolio, PaperCashBalance):
        """Get portfolio summary for a user"""
        try:
            # Get portfolio positions
            portfolio = self.db.query(PaperPortfolio).filter_by(user_id=user_id).all()
            
            # Get cash balance
            cash_balance = self.db.query(PaperCashBalance).filter_by(user_id=user_id).first()
            if not cash_balance:
                cash_balance = PaperCashBalance(user_id=user_id, cash_balance=100000.0)
                self.db.add(cash_balance)
                self.db.commit()
            
            portfolio_data = []
            total_value = 0
            total_cost = 0
            
            for item in portfolio:
                current_price = self.get_current_price(item.stock_symbol)
                if current_price:
                    position_value = item.shares * current_price
                    position_cost = item.shares * item.average_price
                    total_value += position_value
                    total_cost += position_cost
                    
                    portfolio_data.append({
                        'symbol': item.stock_symbol,
                        'shares': item.shares,
                        'average_price': item.average_price,
                        'current_price': current_price,
                        'position_value': position_value,
                        'profit_loss': position_value - position_cost,
                        'change_percent': ((current_price - item.average_price) / item.average_price * 100) if item.average_price > 0 else 0
                    })
            
            summary = {
                'total_value': total_value,
                'total_cost': total_cost,
                'cash_balance': cash_balance.cash_balance,
                'total_account_value': total_value + cash_balance.cash_balance,
                'total_profit_loss': total_value - total_cost,
                'total_return_percent': ((total_value - total_cost) / total_cost * 100) if total_cost > 0 else 0
            }
            
            return {
                'success': True,
                'portfolio': portfolio_data,
                'summary': summary
            }
            
        except Exception as e:
            self.logger.error(f"Error getting portfolio summary: {str(e)}")
            return {'success': False, 'error': f'Error getting portfolio data: {str(e)}'}
    
    def reset_portfolio(self, user_id, PaperPortfolio, PaperTransaction, PaperCashBalance):
        """Reset user's paper portfolio"""
        try:
            # Delete all paper portfolio positions
            self.db.query(PaperPortfolio).filter_by(user_id=user_id).delete()
            
            # Delete all paper transactions
            self.db.query(PaperTransaction).filter_by(user_id=user_id).delete()
            
            # Reset cash balance to $100,000
            cash_balance = self.db.query(PaperCashBalance).filter_by(user_id=user_id).first()
            if not cash_balance:
                cash_balance = PaperCashBalance(user_id=user_id, cash_balance=100000.0)
                self.db.add(cash_balance)
            else:
                cash_balance.cash_balance = 100000.0
                cash_balance.updated_at = datetime.now()
            
            self.db.commit()
            
            return {
                'success': True,
                'message': 'Paper portfolio reset to $100,000 cash'
            }
            
        except Exception as e:
            self.db.rollback()
            self.logger.error(f"Error resetting portfolio: {str(e)}")
            return {'success': False, 'error': f'Error resetting portfolio: {str(e)}'}

# Function to create paper trading instance
def create_paper_trading(db_session):
    """Create a paper trading instance with database session"""
    return PaperTrading(db_session)