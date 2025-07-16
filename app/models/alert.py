"""Alert and notification models."""

from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey, Text, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime

from .base import BaseModel


class PriceAlert(BaseModel):
    """Price alert model for user notifications."""
    
    __tablename__ = 'price_alerts'
    
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    stock_symbol = Column(String(10), nullable=False)
    target_price = Column(Float, nullable=False)
    condition = Column(String(10), nullable=False)  # 'above' or 'below'
    is_active = Column(Boolean, default=True)
    triggered_at = Column(DateTime)
    
    # Relationships
    user = relationship('User', back_populates='alerts')
    
    def __repr__(self):
        """String representation of the price alert."""
        return f"<PriceAlert(id={self.id}, symbol='{self.stock_symbol}', {self.condition} ${self.target_price})>"
    
    def check_condition(self, current_price: float) -> bool:
        """Check if alert condition is met."""
        if not self.is_active:
            return False
            
        if self.condition == 'above':
            return current_price > self.target_price
        elif self.condition == 'below':
            return current_price < self.target_price
        
        return False
    
    def trigger(self):
        """Mark alert as triggered."""
        self.is_active = False
        self.triggered_at = datetime.now()
    
    def reactivate(self):
        """Reactivate the alert."""
        self.is_active = True
        self.triggered_at = None


class Notification(BaseModel):
    """Notification model for user messages."""
    
    __tablename__ = 'notifications'
    
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    title = Column(String(200))
    message = Column(Text, nullable=False)
    notification_type = Column(String(50), default='info')  # 'info', 'alert', 'warning', 'error'
    read = Column(Boolean, default=False)
    read_at = Column(DateTime)
    
    # Relationships
    user = relationship('User', back_populates='notifications')
    
    def __repr__(self):
        """String representation of the notification."""
        return f"<Notification(id={self.id}, type='{self.notification_type}', read={self.read})>"
    
    def mark_as_read(self):
        """Mark notification as read."""
        self.read = True
        self.read_at = datetime.now()
    
    def mark_as_unread(self):
        """Mark notification as unread."""
        self.read = False
        self.read_at = None
    
    @classmethod
    def create_alert_notification(cls, user_id: int, stock_symbol: str, target_price: float, condition: str):
        """Create a notification for a triggered alert."""
        message = f"Alert triggered for {stock_symbol}: Price went {condition} ${target_price:.2f}"
        return cls(
            user_id=user_id,
            title=f"{stock_symbol} Alert Triggered",
            message=message,
            notification_type='alert'
        )