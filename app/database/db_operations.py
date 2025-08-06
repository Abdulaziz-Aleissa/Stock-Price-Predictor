"""
Database Operations Module
Handle all database operations for the Stock Predictor application
"""

from datetime import datetime, timedelta
from sqlalchemy.orm import sessionmaker
from models.database import (
    User, Portfolio, Watchlist, PriceAlert, Notification, 
    PaperPortfolio, PaperTransaction, PaperCashBalance, 
    PredictionHistory, engine
)
from typing import Optional, List, Dict
import logging

logger = logging.getLogger(__name__)

# Setup Database Session
Session = sessionmaker(bind=engine)


class DatabaseOperations:
    """Handle all database operations"""
    
    def __init__(self):
        self.db = Session()
    
    def get_session(self):
        """Get database session"""
        return self.db
    
    def close_session(self):
        """Close database session"""
        self.db.close()
    
    # User operations
    def get_user(self, user_id: int) -> Optional[User]:
        """Get user by ID"""
        return self.db.get(User, user_id)
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        return self.db.query(User).filter_by(username=username).first()
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        return self.db.query(User).filter_by(email=email).first()
    
    def create_user(self, username: str, email: str, password_hash: str) -> User:
        """Create new user"""
        user = User(username=username, email=email, password_hash=password_hash)
        self.db.add(user)
        self.db.commit()
        return user
    
    # Paper Portfolio operations
    def get_or_create_paper_cash_balance(self, user_id: int) -> PaperCashBalance:
        """Get or create paper cash balance for user with default $100,000"""
        cash_balance = self.db.query(PaperCashBalance).filter_by(user_id=user_id).first()
        if not cash_balance:
            cash_balance = PaperCashBalance(user_id=user_id, cash_balance=100000.0)
            self.db.add(cash_balance)
            self.db.commit()
        return cash_balance
    
    def update_paper_portfolio(self, user_id: int, symbol: str, shares: float, price: float, transaction_type: str) -> bool:
        """Update paper portfolio position after a transaction"""
        position = self.db.query(PaperPortfolio).filter_by(user_id=user_id, stock_symbol=symbol).first()
        
        if transaction_type == 'BUY':
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
        
        elif transaction_type == 'SELL':
            if position and position.shares >= shares:
                position.shares -= shares
                position.updated_at = datetime.now()
                # Remove position if no shares left
                if position.shares == 0:
                    self.db.delete(position)
            else:
                return False  # Insufficient shares
        
        self.db.commit()
        return True
    
    def get_paper_portfolio(self, user_id: int) -> List[PaperPortfolio]:
        """Get user's paper portfolio"""
        return self.db.query(PaperPortfolio).filter_by(user_id=user_id).all()
    
    def add_paper_transaction(self, user_id: int, symbol: str, transaction_type: str, shares: float, price: float) -> PaperTransaction:
        """Add paper transaction record"""
        transaction = PaperTransaction(
            user_id=user_id,
            stock_symbol=symbol,
            transaction_type=transaction_type,
            shares=shares,
            price=price
        )
        self.db.add(transaction)
        self.db.commit()
        return transaction
    
    def get_paper_transactions(self, user_id: int) -> List[PaperTransaction]:
        """Get user's paper transactions"""
        return self.db.query(PaperTransaction).filter_by(user_id=user_id).order_by(PaperTransaction.transaction_date.desc()).all()
    
    def update_paper_cash_balance(self, user_id: int, amount: float) -> bool:
        """Update user's paper cash balance"""
        try:
            cash_balance = self.get_or_create_paper_cash_balance(user_id)
            cash_balance.cash_balance += amount
            cash_balance.updated_at = datetime.now()
            self.db.commit()
            return True
        except Exception as e:
            logger.error(f"Error updating cash balance: {str(e)}")
            return False
    
    def reset_paper_portfolio(self, user_id: int) -> bool:
        """Reset user's paper portfolio"""
        try:
            # Delete all positions
            self.db.query(PaperPortfolio).filter_by(user_id=user_id).delete()
            # Delete all transactions
            self.db.query(PaperTransaction).filter_by(user_id=user_id).delete()
            # Reset cash balance
            cash_balance = self.get_or_create_paper_cash_balance(user_id)
            cash_balance.cash_balance = 100000.0
            cash_balance.updated_at = datetime.now()
            self.db.commit()
            return True
        except Exception as e:
            logger.error(f"Error resetting portfolio: {str(e)}")
            return False
    
    # Portfolio operations
    def get_portfolio(self, user_id: int) -> List[Portfolio]:
        """Get user's portfolio"""
        return self.db.query(Portfolio).filter_by(user_id=user_id).all()
    
    def add_to_portfolio(self, user_id: int, symbol: str, shares: float, purchase_price: float) -> Portfolio:
        """Add stock to portfolio"""
        portfolio_item = Portfolio(
            user_id=user_id,
            stock_symbol=symbol,
            shares=shares,
            purchase_price=purchase_price,
            purchase_date=datetime.now()
        )
        self.db.add(portfolio_item)
        self.db.commit()
        return portfolio_item
    
    def remove_from_portfolio(self, item_id: int) -> bool:
        """Remove item from portfolio"""
        try:
            item = self.db.get(Portfolio, item_id)
            if item:
                self.db.delete(item)
                self.db.commit()
                return True
            return False
        except Exception as e:
            logger.error(f"Error removing from portfolio: {str(e)}")
            return False
    
    # Watchlist operations
    def get_watchlist(self, user_id: int) -> List[Watchlist]:
        """Get user's watchlist"""
        return self.db.query(Watchlist).filter_by(user_id=user_id).all()
    
    def add_to_watchlist(self, user_id: int, symbol: str) -> Watchlist:
        """Add stock to watchlist"""
        watchlist_item = Watchlist(user_id=user_id, stock_symbol=symbol)
        self.db.add(watchlist_item)
        self.db.commit()
        return watchlist_item
    
    def remove_from_watchlist(self, item_id: int) -> bool:
        """Remove item from watchlist"""
        try:
            item = self.db.get(Watchlist, item_id)
            if item:
                self.db.delete(item)
                self.db.commit()
                return True
            return False
        except Exception as e:
            logger.error(f"Error removing from watchlist: {str(e)}")
            return False
    
    # Price alerts operations
    def get_active_price_alerts(self) -> List[PriceAlert]:
        """Get all active price alerts"""
        return self.db.query(PriceAlert).filter_by(is_active=True).all()
    
    def get_user_alerts(self, user_id: int) -> List[PriceAlert]:
        """Get user's price alerts"""
        return self.db.query(PriceAlert).filter_by(user_id=user_id).all()
    
    def add_price_alert(self, user_id: int, symbol: str, condition: str, target_price: float) -> PriceAlert:
        """Add price alert"""
        alert = PriceAlert(
            user_id=user_id,
            stock_symbol=symbol,
            condition=condition,
            target_price=target_price
        )
        self.db.add(alert)
        self.db.commit()
        return alert
    
    def remove_price_alert(self, alert_id: int) -> bool:
        """Remove price alert"""
        try:
            alert = self.db.get(PriceAlert, alert_id)
            if alert:
                self.db.delete(alert)
                self.db.commit()
                return True
            return False
        except Exception as e:
            logger.error(f"Error removing price alert: {str(e)}")
            return False
    
    def deactivate_price_alert(self, alert_id: int) -> bool:
        """Deactivate price alert"""
        try:
            alert = self.db.get(PriceAlert, alert_id)
            if alert:
                alert.is_active = False
                self.db.commit()
                return True
            return False
        except Exception as e:
            logger.error(f"Error deactivating price alert: {str(e)}")
            return False
    
    # Notification operations
    def add_notification(self, user_id: int, message: str) -> Notification:
        """Add notification"""
        notification = Notification(user_id=user_id, message=message)
        self.db.add(notification)
        self.db.commit()
        return notification
    
    def get_user_notifications(self, user_id: int) -> List[Notification]:
        """Get user's notifications"""
        return self.db.query(Notification).filter_by(user_id=user_id).order_by(Notification.created_at.desc()).all()
    
    def mark_notification_read(self, notification_id: int) -> bool:
        """Mark notification as read"""
        try:
            notification = self.db.get(Notification, notification_id)
            if notification:
                notification.is_read = True
                self.db.commit()
                return True
            return False
        except Exception as e:
            logger.error(f"Error marking notification as read: {str(e)}")
            return False
    
    # Prediction operations
    def store_prediction(self, stock_symbol: str, predicted_price: float, current_price: float, price_change_pct: float, metrics: Dict) -> PredictionHistory:
        """Store prediction in database for future backtesting"""
        try:
            target_date = datetime.now() + timedelta(days=1)
            prediction = PredictionHistory(
                stock_symbol=stock_symbol,
                prediction_date=datetime.now(),
                target_date=target_date,
                predicted_price=predicted_price,
                actual_current_price=current_price,
                predicted_change_pct=price_change_pct,
                model_accuracy=metrics.get('accuracy', 0),
                mae=metrics.get('mae', 0),
                rmse=metrics.get('rmse', 0)
            )
            self.db.add(prediction)
            self.db.commit()
            return prediction
        except Exception as e:
            logger.error(f"Error storing prediction: {str(e)}")
            return None
    
    def get_predictions_history(self, stock_symbol: str = None, days_back: int = 365) -> List[PredictionHistory]:
        """Get prediction history"""
        cutoff_date = datetime.now() - timedelta(days=days_back)
        query = self.db.query(PredictionHistory).filter(PredictionHistory.prediction_date >= cutoff_date)
        
        if stock_symbol:
            query = query.filter(PredictionHistory.stock_symbol == stock_symbol)
        
        return query.order_by(PredictionHistory.prediction_date.desc()).all()


# Global instance to be used across the application
db_operations = DatabaseOperations()