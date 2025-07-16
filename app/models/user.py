"""User model."""

from sqlalchemy import Column, String, Boolean
from sqlalchemy.orm import relationship
from flask_login import UserMixin

from .base import BaseModel


class User(BaseModel, UserMixin):
    """User model for authentication and data management."""
    
    __tablename__ = 'users'
    
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    password_hash = Column(String(128), nullable=False)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    portfolios = relationship('Portfolio', back_populates='user', cascade='all, delete-orphan')
    watchlists = relationship('Watchlist', back_populates='user', cascade='all, delete-orphan')
    alerts = relationship('PriceAlert', back_populates='user', cascade='all, delete-orphan')
    notifications = relationship('Notification', back_populates='user', cascade='all, delete-orphan')
    
    def __repr__(self):
        """String representation of the user."""
        return f"<User(id={self.id}, username='{self.username}')>"
    
    def to_dict(self):
        """Convert user to dictionary (excluding sensitive data)."""
        data = super().to_dict()
        # Remove password hash for security
        data.pop('password_hash', None)
        return data