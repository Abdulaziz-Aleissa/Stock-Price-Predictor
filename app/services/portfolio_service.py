"""Portfolio service for managing user portfolios."""

import logging
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session

from ..models.portfolio import Portfolio, Watchlist
from ..models.user import User
from ..services.stock_service import StockService
from ..core.exceptions import PortfolioError, ValidationError
from ..core.constants import VALIDATION_RULES


logger = logging.getLogger(__name__)


class PortfolioService:
    """Service class for portfolio management operations."""
    
    def __init__(self, db_session: Session, stock_service: StockService = None):
        """Initialize the portfolio service."""
        self.db = db_session
        self.stock_service = stock_service or StockService()
    
    def add_to_portfolio(self, user_id: int, symbol: str, shares: float, 
                        purchase_price: float) -> Portfolio:
        """Add a stock to user's portfolio."""
        try:
            # Validate inputs
            self._validate_portfolio_input(symbol, shares, purchase_price)
            
            # Validate ticker
            if not self.stock_service.is_valid_ticker(symbol):
                raise ValidationError(f"Invalid ticker symbol: {symbol}")
            
            # Create portfolio item
            portfolio_item = Portfolio(
                user_id=user_id,
                stock_symbol=symbol.upper(),
                shares=shares,
                purchase_price=purchase_price
            )
            
            self.db.add(portfolio_item)
            self.db.commit()
            
            logger.info(f"Added {shares} shares of {symbol} to portfolio for user {user_id}")
            return portfolio_item
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error adding to portfolio: {str(e)}")
            raise PortfolioError(f"Failed to add {symbol} to portfolio")
    
    def remove_from_portfolio(self, user_id: int, portfolio_id: int) -> bool:
        """Remove a stock from user's portfolio."""
        try:
            portfolio_item = self.db.query(Portfolio).filter_by(
                id=portfolio_id, user_id=user_id
            ).first()
            
            if not portfolio_item:
                raise PortfolioError("Portfolio item not found")
            
            symbol = portfolio_item.stock_symbol
            self.db.delete(portfolio_item)
            self.db.commit()
            
            logger.info(f"Removed {symbol} from portfolio for user {user_id}")
            return True
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error removing from portfolio: {str(e)}")
            raise PortfolioError("Failed to remove item from portfolio")
    
    def get_portfolio(self, user_id: int) -> List[Dict[str, Any]]:
        """Get user's portfolio with current values."""
        try:
            portfolio_items = self.db.query(Portfolio).filter_by(user_id=user_id).all()
            portfolio_data = []
            
            for item in portfolio_items:
                current_price = self.stock_service.get_current_price(item.stock_symbol)
                
                if current_price is not None:
                    profit_loss = item.calculate_profit_loss(current_price)
                    change_percent = item.calculate_percentage_change(current_price)
                    position_value = item.shares * current_price
                    
                    portfolio_data.append({
                        'id': item.id,
                        'symbol': item.stock_symbol,
                        'shares': item.shares,
                        'purchase_price': item.purchase_price,
                        'purchase_date': item.purchase_date.isoformat() if item.purchase_date else None,
                        'current_price': current_price,
                        'position_value': position_value,
                        'total_cost': item.total_cost,
                        'profit_loss': profit_loss,
                        'change_percent': change_percent
                    })
            
            return portfolio_data
            
        except Exception as e:
            logger.error(f"Error getting portfolio for user {user_id}: {str(e)}")
            raise PortfolioError("Failed to retrieve portfolio")
    
    def get_portfolio_summary(self, user_id: int) -> Dict[str, float]:
        """Get portfolio summary with totals."""
        try:
            portfolio_data = self.get_portfolio(user_id)
            
            total_value = sum(item['position_value'] for item in portfolio_data)
            total_cost = sum(item['total_cost'] for item in portfolio_data)
            total_profit_loss = total_value - total_cost
            total_return_percent = ((total_value - total_cost) / total_cost * 100) if total_cost > 0 else 0
            
            return {
                'total_value': total_value,
                'total_cost': total_cost,
                'total_profit_loss': total_profit_loss,
                'total_return_percent': total_return_percent,
                'position_count': len(portfolio_data)
            }
            
        except Exception as e:
            logger.error(f"Error getting portfolio summary for user {user_id}: {str(e)}")
            raise PortfolioError("Failed to calculate portfolio summary")
    
    def update_portfolio_item(self, user_id: int, portfolio_id: int, 
                             shares: Optional[float] = None, 
                             purchase_price: Optional[float] = None) -> Portfolio:
        """Update a portfolio item."""
        try:
            portfolio_item = self.db.query(Portfolio).filter_by(
                id=portfolio_id, user_id=user_id
            ).first()
            
            if not portfolio_item:
                raise PortfolioError("Portfolio item not found")
            
            if shares is not None:
                if shares <= 0 or shares > VALIDATION_RULES['SHARES_MAX_VALUE']:
                    raise ValidationError("Invalid shares amount")
                portfolio_item.shares = shares
            
            if purchase_price is not None:
                if purchase_price <= 0 or purchase_price > VALIDATION_RULES['PRICE_MAX_VALUE']:
                    raise ValidationError("Invalid purchase price")
                portfolio_item.purchase_price = purchase_price
            
            self.db.commit()
            
            logger.info(f"Updated portfolio item {portfolio_id} for user {user_id}")
            return portfolio_item
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error updating portfolio item: {str(e)}")
            raise PortfolioError("Failed to update portfolio item")
    
    def add_to_watchlist(self, user_id: int, symbol: str, target_price: float) -> Watchlist:
        """Add a stock to user's watchlist."""
        try:
            # Validate inputs
            self._validate_watchlist_input(symbol, target_price)
            
            # Validate ticker
            if not self.stock_service.is_valid_ticker(symbol):
                raise ValidationError(f"Invalid ticker symbol: {symbol}")
            
            # Check if already in watchlist
            existing = self.db.query(Watchlist).filter_by(
                user_id=user_id, stock_symbol=symbol.upper()
            ).first()
            
            if existing:
                raise PortfolioError(f"{symbol} is already in your watchlist")
            
            # Create watchlist item
            watchlist_item = Watchlist(
                user_id=user_id,
                stock_symbol=symbol.upper(),
                target_price=target_price
            )
            
            self.db.add(watchlist_item)
            self.db.commit()
            
            logger.info(f"Added {symbol} to watchlist for user {user_id}")
            return watchlist_item
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error adding to watchlist: {str(e)}")
            raise PortfolioError(f"Failed to add {symbol} to watchlist")
    
    def remove_from_watchlist(self, user_id: int, watchlist_id: int) -> bool:
        """Remove a stock from user's watchlist."""
        try:
            watchlist_item = self.db.query(Watchlist).filter_by(
                id=watchlist_id, user_id=user_id
            ).first()
            
            if not watchlist_item:
                raise PortfolioError("Watchlist item not found")
            
            symbol = watchlist_item.stock_symbol
            self.db.delete(watchlist_item)
            self.db.commit()
            
            logger.info(f"Removed {symbol} from watchlist for user {user_id}")
            return True
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error removing from watchlist: {str(e)}")
            raise PortfolioError("Failed to remove item from watchlist")
    
    def get_watchlist(self, user_id: int) -> List[Dict[str, Any]]:
        """Get user's watchlist with current prices."""
        try:
            watchlist_items = self.db.query(Watchlist).filter_by(user_id=user_id).all()
            watchlist_data = []
            
            for item in watchlist_items:
                current_price = self.stock_service.get_current_price(item.stock_symbol)
                
                if current_price is not None:
                    distance_to_target = item.calculate_distance_to_target(current_price)
                    percent_to_target = item.calculate_percentage_to_target(current_price)
                    
                    watchlist_data.append({
                        'id': item.id,
                        'symbol': item.stock_symbol,
                        'target_price': item.target_price,
                        'current_price': current_price,
                        'distance_to_target': distance_to_target,
                        'percent_to_target': percent_to_target,
                        'added_date': item.added_date.isoformat() if item.added_date else None
                    })
            
            return watchlist_data
            
        except Exception as e:
            logger.error(f"Error getting watchlist for user {user_id}: {str(e)}")
            raise PortfolioError("Failed to retrieve watchlist")
    
    def _validate_portfolio_input(self, symbol: str, shares: float, price: float) -> None:
        """Validate portfolio input parameters."""
        if not symbol or len(symbol) > VALIDATION_RULES['TICKER_MAX_LENGTH']:
            raise ValidationError("Invalid ticker symbol")
        
        if shares <= 0 or shares > VALIDATION_RULES['SHARES_MAX_VALUE']:
            raise ValidationError("Invalid shares amount")
        
        if price <= 0 or price > VALIDATION_RULES['PRICE_MAX_VALUE']:
            raise ValidationError("Invalid price")
    
    def _validate_watchlist_input(self, symbol: str, target_price: float) -> None:
        """Validate watchlist input parameters."""
        if not symbol or len(symbol) > VALIDATION_RULES['TICKER_MAX_LENGTH']:
            raise ValidationError("Invalid ticker symbol")
        
        if target_price <= 0 or target_price > VALIDATION_RULES['PRICE_MAX_VALUE']:
            raise ValidationError("Invalid target price")