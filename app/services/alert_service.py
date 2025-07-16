"""Alert service for managing price alerts and notifications."""

import logging
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from datetime import datetime

from ..models.alert import PriceAlert, Notification
from ..models.user import User
from ..services.stock_service import StockService
from ..core.exceptions import AlertError, ValidationError
from ..core.constants import AlertCondition, VALIDATION_RULES


logger = logging.getLogger(__name__)


class AlertService:
    """Service class for alert and notification management."""
    
    def __init__(self, db_session: Session, stock_service: StockService = None):
        """Initialize the alert service."""
        self.db = db_session
        self.stock_service = stock_service or StockService()
    
    def create_alert(self, user_id: int, symbol: str, target_price: float, 
                    condition: str) -> PriceAlert:
        """Create a new price alert."""
        try:
            # Validate inputs
            self._validate_alert_input(symbol, target_price, condition)
            
            # Validate ticker
            if not self.stock_service.is_valid_ticker(symbol):
                raise ValidationError(f"Invalid ticker symbol: {symbol}")
            
            # Check for duplicate alerts
            existing_alert = self.db.query(PriceAlert).filter_by(
                user_id=user_id,
                stock_symbol=symbol.upper(),
                target_price=target_price,
                condition=condition,
                is_active=True
            ).first()
            
            if existing_alert:
                raise AlertError(f"Alert already exists for {symbol} {condition} ${target_price}")
            
            # Create alert
            alert = PriceAlert(
                user_id=user_id,
                stock_symbol=symbol.upper(),
                target_price=target_price,
                condition=condition.lower()
            )
            
            self.db.add(alert)
            self.db.commit()
            
            logger.info(f"Created alert for user {user_id}: {symbol} {condition} ${target_price}")
            return alert
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating alert: {str(e)}")
            raise AlertError(f"Failed to create alert for {symbol}")
    
    def remove_alert(self, user_id: int, alert_id: int) -> bool:
        """Remove a price alert."""
        try:
            alert = self.db.query(PriceAlert).filter_by(
                id=alert_id, user_id=user_id
            ).first()
            
            if not alert:
                raise AlertError("Alert not found")
            
            symbol = alert.stock_symbol
            self.db.delete(alert)
            self.db.commit()
            
            logger.info(f"Removed alert {alert_id} for user {user_id}")
            return True
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error removing alert: {str(e)}")
            raise AlertError("Failed to remove alert")
    
    def get_user_alerts(self, user_id: int, active_only: bool = True) -> List[Dict[str, Any]]:
        """Get all alerts for a user."""
        try:
            query = self.db.query(PriceAlert).filter_by(user_id=user_id)
            
            if active_only:
                query = query.filter_by(is_active=True)
            
            alerts = query.all()
            alerts_data = []
            
            for alert in alerts:
                current_price = self.stock_service.get_current_price(alert.stock_symbol)
                
                alert_data = {
                    'id': alert.id,
                    'symbol': alert.stock_symbol,
                    'target_price': alert.target_price,
                    'condition': alert.condition,
                    'is_active': alert.is_active,
                    'created_at': alert.created_at.isoformat() if alert.created_at else None,
                    'triggered_at': alert.triggered_at.isoformat() if alert.triggered_at else None,
                    'current_price': current_price or 0
                }
                
                # Check if condition is met
                if current_price and alert.is_active:
                    alert_data['condition_met'] = alert.check_condition(current_price)
                else:
                    alert_data['condition_met'] = False
                
                alerts_data.append(alert_data)
            
            return alerts_data
            
        except Exception as e:
            logger.error(f"Error getting alerts for user {user_id}: {str(e)}")
            raise AlertError("Failed to retrieve alerts")
    
    def check_all_alerts(self) -> List[Dict[str, Any]]:
        """Check all active alerts and trigger notifications."""
        triggered_alerts = []
        
        try:
            active_alerts = self.db.query(PriceAlert).filter_by(is_active=True).all()
            
            for alert in active_alerts:
                current_price = self.stock_service.get_current_price(alert.stock_symbol)
                
                if current_price and alert.check_condition(current_price):
                    # Trigger the alert
                    alert.trigger()
                    
                    # Create notification
                    notification = Notification.create_alert_notification(
                        user_id=alert.user_id,
                        stock_symbol=alert.stock_symbol,
                        target_price=alert.target_price,
                        condition=alert.condition
                    )
                    
                    self.db.add(notification)
                    
                    triggered_alerts.append({
                        'alert_id': alert.id,
                        'user_id': alert.user_id,
                        'symbol': alert.stock_symbol,
                        'target_price': alert.target_price,
                        'condition': alert.condition,
                        'current_price': current_price
                    })
                    
                    logger.info(f"Triggered alert {alert.id}: {alert.stock_symbol} {alert.condition} ${alert.target_price}")
            
            if triggered_alerts:
                self.db.commit()
                logger.info(f"Processed {len(triggered_alerts)} triggered alerts")
            
            return triggered_alerts
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error checking alerts: {str(e)}")
            return []
    
    def toggle_alert(self, user_id: int, alert_id: int) -> PriceAlert:
        """Toggle alert active status."""
        try:
            alert = self.db.query(PriceAlert).filter_by(
                id=alert_id, user_id=user_id
            ).first()
            
            if not alert:
                raise AlertError("Alert not found")
            
            if alert.is_active:
                alert.is_active = False
            else:
                alert.reactivate()
            
            self.db.commit()
            
            logger.info(f"Toggled alert {alert_id} to {'active' if alert.is_active else 'inactive'}")
            return alert
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error toggling alert: {str(e)}")
            raise AlertError("Failed to toggle alert")
    
    def create_notification(self, user_id: int, title: str, message: str, 
                          notification_type: str = 'info') -> Notification:
        """Create a notification for a user."""
        try:
            notification = Notification(
                user_id=user_id,
                title=title,
                message=message,
                notification_type=notification_type
            )
            
            self.db.add(notification)
            self.db.commit()
            
            logger.info(f"Created notification for user {user_id}: {title}")
            return notification
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating notification: {str(e)}")
            raise AlertError("Failed to create notification")
    
    def get_user_notifications(self, user_id: int, unread_only: bool = False, 
                             limit: int = 50) -> List[Dict[str, Any]]:
        """Get notifications for a user."""
        try:
            query = self.db.query(Notification).filter_by(user_id=user_id)
            
            if unread_only:
                query = query.filter_by(read=False)
            
            notifications = query.order_by(Notification.created_at.desc()).limit(limit).all()
            
            return [
                {
                    'id': notif.id,
                    'title': notif.title,
                    'message': notif.message,
                    'type': notif.notification_type,
                    'read': notif.read,
                    'created_at': notif.created_at.isoformat() if notif.created_at else None,
                    'read_at': notif.read_at.isoformat() if notif.read_at else None
                }
                for notif in notifications
            ]
            
        except Exception as e:
            logger.error(f"Error getting notifications for user {user_id}: {str(e)}")
            raise AlertError("Failed to retrieve notifications")
    
    def mark_notification_read(self, user_id: int, notification_id: int) -> bool:
        """Mark a notification as read."""
        try:
            notification = self.db.query(Notification).filter_by(
                id=notification_id, user_id=user_id
            ).first()
            
            if not notification:
                raise AlertError("Notification not found")
            
            notification.mark_as_read()
            self.db.commit()
            
            logger.info(f"Marked notification {notification_id} as read for user {user_id}")
            return True
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error marking notification as read: {str(e)}")
            raise AlertError("Failed to mark notification as read")
    
    def mark_all_notifications_read(self, user_id: int) -> int:
        """Mark all notifications as read for a user."""
        try:
            unread_notifications = self.db.query(Notification).filter_by(
                user_id=user_id, read=False
            ).all()
            
            count = 0
            for notification in unread_notifications:
                notification.mark_as_read()
                count += 1
            
            self.db.commit()
            
            logger.info(f"Marked {count} notifications as read for user {user_id}")
            return count
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error marking all notifications as read: {str(e)}")
            raise AlertError("Failed to mark notifications as read")
    
    def _validate_alert_input(self, symbol: str, target_price: float, condition: str) -> None:
        """Validate alert input parameters."""
        if not symbol or len(symbol) > VALIDATION_RULES['TICKER_MAX_LENGTH']:
            raise ValidationError("Invalid ticker symbol")
        
        if target_price <= 0 or target_price > VALIDATION_RULES['PRICE_MAX_VALUE']:
            raise ValidationError("Invalid target price")
        
        if condition.lower() not in [AlertCondition.ABOVE.value, AlertCondition.BELOW.value]:
            raise ValidationError("Invalid alert condition. Must be 'above' or 'below'")