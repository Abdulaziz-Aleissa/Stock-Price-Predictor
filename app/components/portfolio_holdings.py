"""
Portfolio Holdings Module
Handle Portfolio Holdings display and calculations
"""

from app.database.db_operations import db_operations
from app.data.yfinance_data import yfinance_data
from app.auth.auth_module import auth_manager
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class PortfolioHoldingsManager:
    """Handle portfolio holdings operations"""
    
    def __init__(self):
        self.db_ops = db_operations
        self.data_fetcher = yfinance_data
        self.auth = auth_manager
    
    def get_real_holdings(self, user_id: int) -> List[Dict]:
        """Get real portfolio holdings with current market data"""
        try:
            portfolio = self.db_ops.get_portfolio(user_id)
            holdings = []
            
            for item in portfolio:
                current_price = self.data_fetcher.get_current_price(item.stock_symbol)
                if current_price:
                    profit_loss = (current_price - item.purchase_price) * item.shares
                    position_value = current_price * item.shares
                    position_cost = item.purchase_price * item.shares
                    
                    holdings.append({
                        'id': item.id,
                        'symbol': item.stock_symbol,
                        'shares': item.shares,
                        'purchase_price': item.purchase_price,
                        'purchase_date': item.purchase_date,
                        'current_price': current_price,
                        'profit_loss': profit_loss,
                        'position_value': position_value,
                        'position_cost': position_cost,
                        'change_percent': ((current_price - item.purchase_price) / item.purchase_price * 100),
                        'market_context': self.data_fetcher.get_market_context(item.stock_symbol)
                    })
            
            return holdings
            
        except Exception as e:
            logger.error(f"Error getting real holdings: {str(e)}")
            return []
    
    def get_paper_holdings(self, user_id: int) -> List[Dict]:
        """Get paper portfolio holdings with current market data"""
        try:
            paper_portfolio = self.db_ops.get_paper_portfolio(user_id)
            holdings = []
            
            for item in paper_portfolio:
                current_price = self.data_fetcher.get_current_price(item.stock_symbol)
                if current_price:
                    position_value = item.shares * current_price
                    position_cost = item.shares * item.average_price
                    
                    holdings.append({
                        'symbol': item.stock_symbol,
                        'shares': item.shares,
                        'average_price': item.average_price,
                        'current_price': current_price,
                        'position_value': position_value,
                        'position_cost': position_cost,
                        'profit_loss': position_value - position_cost,
                        'change_percent': ((current_price - item.average_price) / item.average_price * 100) if item.average_price > 0 else 0,
                        'market_context': self.data_fetcher.get_market_context(item.stock_symbol),
                        'created_at': item.created_at,
                        'updated_at': item.updated_at
                    })
            
            return holdings
            
        except Exception as e:
            logger.error(f"Error getting paper holdings: {str(e)}")
            return []
    
    def add_real_holding(self, user_id: int, symbol: str, shares: float, purchase_price: float) -> bool:
        """Add stock to real portfolio"""
        try:
            # Validate ticker
            if not self.data_fetcher.is_valid_ticker(symbol):
                logger.warning(f"Invalid ticker symbol: {symbol}")
                return False
            
            # Add to portfolio
            self.db_ops.add_to_portfolio(user_id, symbol.upper(), shares, purchase_price)
            return True
            
        except Exception as e:
            logger.error(f"Error adding real holding: {str(e)}")
            return False
    
    def remove_real_holding(self, item_id: int) -> bool:
        """Remove stock from real portfolio"""
        try:
            return self.db_ops.remove_from_portfolio(item_id)
        except Exception as e:
            logger.error(f"Error removing real holding: {str(e)}")
            return False
    
    def execute_paper_buy(self, user_id: int, symbol: str, shares: float) -> Dict:
        """Execute paper buy order"""
        try:
            # Validate ticker
            if not self.data_fetcher.is_valid_ticker(symbol):
                return {'success': False, 'message': 'Invalid ticker symbol'}
            
            # Get current price
            current_price = self.data_fetcher.get_current_price(symbol)
            if not current_price:
                return {'success': False, 'message': 'Unable to get current price'}
            
            # Calculate total cost
            total_cost = shares * current_price
            
            # Check cash balance
            cash_balance = self.db_ops.get_or_create_paper_cash_balance(user_id)
            if cash_balance.cash_balance < total_cost:
                return {'success': False, 'message': 'Insufficient cash balance'}
            
            # Update portfolio
            if self.db_ops.update_paper_portfolio(user_id, symbol.upper(), shares, current_price, 'BUY'):
                # Update cash balance
                self.db_ops.update_paper_cash_balance(user_id, -total_cost)
                
                # Record transaction
                self.db_ops.add_paper_transaction(user_id, symbol.upper(), 'BUY', shares, current_price)
                
                return {
                    'success': True, 
                    'message': f'Successfully bought {shares} shares of {symbol} at ${current_price:.2f}',
                    'transaction_details': {
                        'symbol': symbol.upper(),
                        'shares': shares,
                        'price': current_price,
                        'total_cost': total_cost
                    }
                }
            else:
                return {'success': False, 'message': 'Failed to update portfolio'}
                
        except Exception as e:
            logger.error(f"Error executing paper buy: {str(e)}")
            return {'success': False, 'message': 'Transaction failed due to server error'}
    
    def execute_paper_sell(self, user_id: int, symbol: str, shares: float) -> Dict:
        """Execute paper sell order"""
        try:
            # Validate ticker
            if not self.data_fetcher.is_valid_ticker(symbol):
                return {'success': False, 'message': 'Invalid ticker symbol'}
            
            # Get current price
            current_price = self.data_fetcher.get_current_price(symbol)
            if not current_price:
                return {'success': False, 'message': 'Unable to get current price'}
            
            # Update portfolio
            if self.db_ops.update_paper_portfolio(user_id, symbol.upper(), shares, current_price, 'SELL'):
                # Calculate proceeds
                total_proceeds = shares * current_price
                
                # Update cash balance
                self.db_ops.update_paper_cash_balance(user_id, total_proceeds)
                
                # Record transaction
                self.db_ops.add_paper_transaction(user_id, symbol.upper(), 'SELL', shares, current_price)
                
                return {
                    'success': True, 
                    'message': f'Successfully sold {shares} shares of {symbol} at ${current_price:.2f}',
                    'transaction_details': {
                        'symbol': symbol.upper(),
                        'shares': shares,
                        'price': current_price,
                        'total_proceeds': total_proceeds
                    }
                }
            else:
                return {'success': False, 'message': 'Insufficient shares to sell'}
                
        except Exception as e:
            logger.error(f"Error executing paper sell: {str(e)}")
            return {'success': False, 'message': 'Transaction failed due to server error'}
    
    def get_paper_transactions(self, user_id: int) -> List[Dict]:
        """Get paper trading transaction history"""
        try:
            transactions = self.db_ops.get_paper_transactions(user_id)
            transaction_data = []
            
            for transaction in transactions:
                transaction_data.append({
                    'id': transaction.id,
                    'symbol': transaction.stock_symbol,
                    'type': transaction.transaction_type,
                    'shares': transaction.shares,
                    'price': transaction.price,
                    'total': transaction.shares * transaction.price,
                    'date': transaction.transaction_date
                })
            
            return transaction_data
            
        except Exception as e:
            logger.error(f"Error getting paper transactions: {str(e)}")
            return []
    
    def reset_paper_holdings(self, user_id: int) -> bool:
        """Reset paper portfolio"""
        try:
            return self.db_ops.reset_paper_portfolio(user_id)
        except Exception as e:
            logger.error(f"Error resetting paper holdings: {str(e)}")
            return False
    
    def get_holding_performance_metrics(self, user_id: int) -> Dict:
        """Get performance metrics for holdings"""
        try:
            real_holdings = self.get_real_holdings(user_id)
            paper_holdings = self.get_paper_holdings(user_id)
            
            # Calculate metrics
            metrics = {
                'real_portfolio': {
                    'total_positions': len(real_holdings),
                    'profitable_positions': len([h for h in real_holdings if h['profit_loss'] > 0]),
                    'losing_positions': len([h for h in real_holdings if h['profit_loss'] < 0]),
                    'best_performer': max(real_holdings, key=lambda x: x['change_percent']) if real_holdings else None,
                    'worst_performer': min(real_holdings, key=lambda x: x['change_percent']) if real_holdings else None
                },
                'paper_portfolio': {
                    'total_positions': len(paper_holdings),
                    'profitable_positions': len([h for h in paper_holdings if h['profit_loss'] > 0]),
                    'losing_positions': len([h for h in paper_holdings if h['profit_loss'] < 0]),
                    'best_performer': max(paper_holdings, key=lambda x: x['change_percent']) if paper_holdings else None,
                    'worst_performer': min(paper_holdings, key=lambda x: x['change_percent']) if paper_holdings else None
                }
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting holding performance metrics: {str(e)}")
            return {'real_portfolio': {}, 'paper_portfolio': {}}


# Global instance to be used across the application
portfolio_holdings_manager = PortfolioHoldingsManager()