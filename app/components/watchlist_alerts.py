"""
Watchlist & Alerts Module
Handle Watchlist & Alerts functionality
"""

from app.database.db_operations import db_operations
from app.data.yfinance_data import yfinance_data
from app.auth.auth_module import auth_manager
from flask import current_app
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class WatchlistAlertsManager:
    """Handle watchlist and alerts operations"""
    
    def __init__(self):
        self.db_ops = db_operations
        self.data_fetcher = yfinance_data
        self.auth = auth_manager
    
    # Watchlist operations
    def get_watchlist_data(self, user_id: int) -> List[Dict]:
        """Get watchlist with current market data"""
        try:
            watchlist = self.db_ops.get_watchlist(user_id)
            watchlist_data = []
            
            for item in watchlist:
                current_price = self.data_fetcher.get_current_price(item.stock_symbol)
                if current_price:
                    price_difference = current_price - item.target_price
                    watchlist_data.append({
                        'id': item.id,
                        'symbol': item.stock_symbol,
                        'target_price': item.target_price,
                        'current_price': current_price,
                        'price_difference': price_difference,
                        'percent_to_target': ((current_price - item.target_price) / item.target_price * 100),
                        'added_date': item.added_date,
                        'market_context': self.data_fetcher.get_market_context(item.stock_symbol)
                    })
            
            return watchlist_data
            
        except Exception as e:
            logger.error(f"Error getting watchlist data: {str(e)}")
            return []
    
    def add_to_watchlist(self, user_id: int, symbol: str, target_price: float) -> bool:
        """Add stock to watchlist"""
        try:
            # Validate ticker
            if not self.data_fetcher.is_valid_ticker(symbol):
                logger.warning(f"Invalid ticker symbol: {symbol}")
                return False
            
            # Check if already in watchlist
            existing_watchlist = self.db_ops.get_watchlist(user_id)
            if any(item.stock_symbol == symbol.upper() for item in existing_watchlist):
                logger.warning(f"Symbol {symbol} already in watchlist")
                return False
            
            # Add to watchlist
            self.db_ops.add_to_watchlist(user_id, symbol.upper(), target_price)
            return True
            
        except Exception as e:
            logger.error(f"Error adding to watchlist: {str(e)}")
            return False
    
    def remove_from_watchlist(self, item_id: int) -> bool:
        """Remove stock from watchlist"""
        try:
            return self.db_ops.remove_from_watchlist(item_id)
        except Exception as e:
            logger.error(f"Error removing from watchlist: {str(e)}")
            return False
    
    def update_watchlist_target(self, item_id: int, new_target_price: float) -> bool:
        """Update target price for watchlist item"""
        try:
            # This would require adding an update method to db_operations
            # For now, we'll return True as a placeholder
            # TODO: Implement update_watchlist_target in db_operations
            return True
        except Exception as e:
            logger.error(f"Error updating watchlist target: {str(e)}")
            return False
    
    # Price alerts operations
    def get_alerts_data(self, user_id: int) -> List[Dict]:
        """Get price alerts with current market data"""
        try:
            alerts = self.db_ops.get_user_alerts(user_id)
            alerts_data = []
            
            for alert in alerts:
                current_price = self.data_fetcher.get_current_price(alert.stock_symbol)
                alerts_data.append({
                    'id': alert.id,
                    'symbol': alert.stock_symbol,
                    'condition': alert.condition,
                    'target_price': alert.target_price,
                    'current_price': current_price or 0,
                    'is_active': alert.is_active,
                    'created_at': alert.created_at,
                    'price_distance': self._calculate_price_distance(current_price, alert.target_price, alert.condition)
                })
            
            return alerts_data
            
        except Exception as e:
            logger.error(f"Error getting alerts data: {str(e)}")
            return []
    
    def add_price_alert(self, user_id: int, symbol: str, condition: str, target_price: float) -> bool:
        """Add price alert"""
        try:
            # Validate ticker
            if not self.data_fetcher.is_valid_ticker(symbol):
                logger.warning(f"Invalid ticker symbol: {symbol}")
                return False
            
            # Validate condition
            if condition not in ['above', 'below']:
                logger.warning(f"Invalid condition: {condition}")
                return False
            
            # Add alert
            self.db_ops.add_price_alert(user_id, symbol.upper(), condition, target_price)
            return True
            
        except Exception as e:
            logger.error(f"Error adding price alert: {str(e)}")
            return False
    
    def remove_price_alert(self, alert_id: int) -> bool:
        """Remove price alert"""
        try:
            return self.db_ops.remove_price_alert(alert_id)
        except Exception as e:
            logger.error(f"Error removing price alert: {str(e)}")
            return False
    
    def check_price_alerts(self) -> int:
        """Check all active price alerts and trigger notifications"""
        try:
            alerts_triggered = 0
            active_alerts = self.db_ops.get_active_price_alerts()
            
            for alert in active_alerts:
                current_price = self.data_fetcher.get_current_price(alert.stock_symbol)
                if current_price:
                    alert_triggered = False
                    
                    if alert.condition == 'above' and current_price > alert.target_price:
                        alert_triggered = True
                    elif alert.condition == 'below' and current_price < alert.target_price:
                        alert_triggered = True
                    
                    if alert_triggered:
                        # Deactivate alert
                        self.db_ops.deactivate_price_alert(alert.id)
                        
                        # Create notification
                        message = f"Alert triggered for {alert.stock_symbol}: Price went {alert.condition} ${alert.target_price:.2f} (Current: ${current_price:.2f})"
                        self.db_ops.add_notification(alert.user_id, message)
                        
                        alerts_triggered += 1
                        logger.info(f"Alert triggered for {alert.stock_symbol} - User {alert.user_id}")
            
            return alerts_triggered
            
        except Exception as e:
            logger.error(f"Error checking price alerts: {str(e)}")
            return 0
    
    # Notification operations
    def get_user_notifications(self, user_id: int, unread_only: bool = True) -> List[Dict]:
        """Get user notifications"""
        try:
            notifications = self.db_ops.get_user_notifications(user_id)
            
            if unread_only:
                notifications = [n for n in notifications if not n.is_read]
            
            notification_data = []
            for notification in notifications:
                notification_data.append({
                    'id': notification.id,
                    'message': notification.message,
                    'created_at': notification.created_at,
                    'is_read': notification.is_read
                })
            
            return notification_data
            
        except Exception as e:
            logger.error(f"Error getting user notifications: {str(e)}")
            return []
    
    def mark_notification_read(self, notification_id: int) -> bool:
        """Mark notification as read"""
        try:
            return self.db_ops.mark_notification_read(notification_id)
        except Exception as e:
            logger.error(f"Error marking notification as read: {str(e)}")
            return False
    
    def mark_all_notifications_read(self, user_id: int) -> bool:
        """Mark all user notifications as read"""
        try:
            notifications = self.db_ops.get_user_notifications(user_id)
            for notification in notifications:
                if not notification.is_read:
                    self.db_ops.mark_notification_read(notification.id)
            return True
        except Exception as e:
            logger.error(f"Error marking all notifications as read: {str(e)}")
            return False
    
    # Analysis and insights
    def get_watchlist_insights(self, user_id: int) -> Dict:
        """Get insights about watchlist performance"""
        try:
            watchlist_data = self.get_watchlist_data(user_id)
            
            if not watchlist_data:
                return {'total_stocks': 0, 'insights': []}
            
            insights = {
                'total_stocks': len(watchlist_data),
                'above_target': len([item for item in watchlist_data if item['price_difference'] > 0]),
                'below_target': len([item for item in watchlist_data if item['price_difference'] < 0]),
                'at_target': len([item for item in watchlist_data if abs(item['price_difference']) < 0.01]),
                'best_performer': max(watchlist_data, key=lambda x: x['percent_to_target']) if watchlist_data else None,
                'worst_performer': min(watchlist_data, key=lambda x: x['percent_to_target']) if watchlist_data else None,
                'average_distance_to_target': sum([abs(item['percent_to_target']) for item in watchlist_data]) / len(watchlist_data)
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting watchlist insights: {str(e)}")
            return {'total_stocks': 0, 'insights': []}
    
    def get_alert_statistics(self, user_id: int) -> Dict:
        """Get statistics about user's alerts"""
        try:
            alerts_data = self.get_alerts_data(user_id)
            
            if not alerts_data:
                return {'total_alerts': 0, 'statistics': {}}
            
            statistics = {
                'total_alerts': len(alerts_data),
                'active_alerts': len([alert for alert in alerts_data if alert['is_active']]),
                'triggered_alerts': len([alert for alert in alerts_data if not alert['is_active']]),
                'above_alerts': len([alert for alert in alerts_data if alert['condition'] == 'above']),
                'below_alerts': len([alert for alert in alerts_data if alert['condition'] == 'below']),
                'closest_to_trigger': min(alerts_data, key=lambda x: abs(x['price_distance'])) if alerts_data else None
            }
            
            return statistics
            
        except Exception as e:
            logger.error(f"Error getting alert statistics: {str(e)}")
            return {'total_alerts': 0, 'statistics': {}}
    
    def _calculate_price_distance(self, current_price: Optional[float], target_price: float, condition: str) -> float:
        """Calculate distance between current price and target price"""
        if current_price is None:
            return float('inf')
        
        if condition == 'above':
            return target_price - current_price  # Positive means still below target
        else:  # below
            return current_price - target_price  # Positive means still above target


# Global instance to be used across the application
watchlist_alerts_manager = WatchlistAlertsManager()