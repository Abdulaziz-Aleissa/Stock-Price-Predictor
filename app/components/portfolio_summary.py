"""
Portfolio Summary Module
Calculate Total Portfolio Value, Total Investment, Profit/Loss, Active Positions
"""

from app.database.db_operations import db_operations
from app.data.yfinance_data import yfinance_data
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class PortfolioSummaryCalculator:
    """Handle portfolio summary calculations"""
    
    def __init__(self):
        self.db_ops = db_operations
        self.data_fetcher = yfinance_data
    
    def calculate_real_portfolio_summary(self, user_id: int) -> Dict:
        """Calculate real portfolio summary data"""
        try:
            portfolio = self.db_ops.get_portfolio(user_id)
            
            portfolio_data = []
            total_value = 0
            total_cost = 0
            
            for item in portfolio:
                current_price = self.data_fetcher.get_current_price(item.stock_symbol)
                if current_price:
                    profit_loss = (current_price - item.purchase_price) * item.shares
                    position_value = current_price * item.shares
                    position_cost = item.purchase_price * item.shares
                    total_value += position_value
                    total_cost += position_cost
                    
                    portfolio_data.append({
                        'id': item.id,
                        'symbol': item.stock_symbol,
                        'shares': item.shares,
                        'purchase_price': item.purchase_price,
                        'current_price': current_price,
                        'profit_loss': profit_loss,
                        'position_value': position_value,
                        'change_percent': ((current_price - item.purchase_price) / item.purchase_price * 100)
                    })
            
            return {
                'portfolio_data': portfolio_data,
                'summary': {
                    'total_value': total_value,
                    'total_cost': total_cost,
                    'total_profit_loss': total_value - total_cost,
                    'total_return_percent': ((total_value - total_cost) / total_cost * 100) if total_cost > 0 else 0,
                    'active_positions': len(portfolio_data)
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating real portfolio summary: {str(e)}")
            return {
                'portfolio_data': [],
                'summary': {
                    'total_value': 0,
                    'total_cost': 0,
                    'total_profit_loss': 0,
                    'total_return_percent': 0,
                    'active_positions': 0
                }
            }
    
    def calculate_paper_portfolio_summary(self, user_id: int) -> Dict:
        """Calculate paper portfolio summary data"""
        try:
            paper_portfolio = self.db_ops.get_paper_portfolio(user_id)
            paper_cash_balance = self.db_ops.get_or_create_paper_cash_balance(user_id)
            
            paper_portfolio_data = []
            paper_total_value = 0
            paper_total_cost = 0
            
            for item in paper_portfolio:
                current_price = self.data_fetcher.get_current_price(item.stock_symbol)
                if current_price:
                    position_value = item.shares * current_price
                    position_cost = item.shares * item.average_price
                    paper_total_value += position_value
                    paper_total_cost += position_cost

                    paper_portfolio_data.append({
                        'symbol': item.stock_symbol,
                        'shares': item.shares,
                        'average_price': item.average_price,
                        'current_price': current_price,
                        'position_value': position_value,
                        'profit_loss': position_value - position_cost,
                        'change_percent': ((current_price - item.average_price) / item.average_price * 100) if item.average_price > 0 else 0
                    })
            
            return {
                'paper_portfolio_data': paper_portfolio_data,
                'paper_summary': {
                    'total_value': paper_total_value,
                    'total_cost': paper_total_cost,
                    'cash_balance': paper_cash_balance.cash_balance,
                    'total_account_value': paper_total_value + paper_cash_balance.cash_balance,
                    'total_profit_loss': paper_total_value - paper_total_cost,
                    'total_return_percent': ((paper_total_value - paper_total_cost) / paper_total_cost * 100) if paper_total_cost > 0 else 0,
                    'active_positions': len(paper_portfolio_data)
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating paper portfolio summary: {str(e)}")
            return {
                'paper_portfolio_data': [],
                'paper_summary': {
                    'total_value': 0,
                    'total_cost': 0,
                    'cash_balance': 100000.0,  # Default cash balance
                    'total_account_value': 100000.0,
                    'total_profit_loss': 0,
                    'total_return_percent': 0,
                    'active_positions': 0
                }
            }
    
    def get_portfolio_performance(self, user_id: int) -> Dict:
        """Get comprehensive portfolio performance metrics"""
        try:
            real_summary = self.calculate_real_portfolio_summary(user_id)
            paper_summary = self.calculate_paper_portfolio_summary(user_id)
            
            return {
                'real_portfolio': real_summary,
                'paper_portfolio': paper_summary,
                'combined_metrics': {
                    'total_positions': real_summary['summary']['active_positions'] + paper_summary['paper_summary']['active_positions'],
                    'real_vs_paper_performance': {
                        'real_return': real_summary['summary']['total_return_percent'],
                        'paper_return': paper_summary['paper_summary']['total_return_percent']
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting portfolio performance: {str(e)}")
            return {
                'real_portfolio': {'portfolio_data': [], 'summary': {}},
                'paper_portfolio': {'paper_portfolio_data': [], 'paper_summary': {}},
                'combined_metrics': {'total_positions': 0, 'real_vs_paper_performance': {}}
            }


# Global instance to be used across the application
portfolio_summary_calculator = PortfolioSummaryCalculator()