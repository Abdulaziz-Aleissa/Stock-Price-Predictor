"""
Authentication Module
Handle all authentication logic for the Stock Predictor application
"""

from flask import render_template, request, redirect, url_for, flash
from flask_login import login_user, logout_user, current_user, login_required
from werkzeug.security import generate_password_hash, check_password_hash
from app.database.db_operations import db_operations
import logging

logger = logging.getLogger(__name__)


class AuthenticationManager:
    """Handle all authentication operations"""
    
    def __init__(self):
        self.db_ops = db_operations
    
    def register_user(self, username: str, email: str, password: str) -> tuple:
        """
        Register a new user
        Returns: (success: bool, message: str, user: User or None)
        """
        try:
            # Check if username already exists
            if self.db_ops.get_user_by_username(username):
                return False, "Username already exists", None
            
            # Check if email already exists
            if self.db_ops.get_user_by_email(email):
                return False, "Email already exists", None
            
            # Create new user
            password_hash = generate_password_hash(password)
            user = self.db_ops.create_user(username, email, password_hash)
            
            return True, "User created successfully", user
            
        except Exception as e:
            logger.error(f"Error registering user: {str(e)}")
            return False, "Registration failed due to server error", None
    
    def authenticate_user(self, username: str, password: str) -> tuple:
        """
        Authenticate user credentials
        Returns: (success: bool, message: str, user: User or None)
        """
        try:
            user = self.db_ops.get_user_by_username(username)
            
            if user and check_password_hash(user.password_hash, password):
                return True, "Authentication successful", user
            else:
                return False, "Invalid credentials", None
                
        except Exception as e:
            logger.error(f"Error authenticating user: {str(e)}")
            return False, "Authentication failed due to server error", None
    
    def login_user_session(self, user) -> bool:
        """Login user and create session"""
        try:
            login_user(user)
            return True
        except Exception as e:
            logger.error(f"Error logging in user: {str(e)}")
            return False
    
    def logout_user_session(self) -> bool:
        """Logout user and clear session"""
        try:
            logout_user()
            return True
        except Exception as e:
            logger.error(f"Error logging out user: {str(e)}")
            return False
    
    def is_user_authenticated(self) -> bool:
        """Check if current user is authenticated"""
        return current_user.is_authenticated
    
    def get_current_user(self):
        """Get current authenticated user"""
        return current_user if current_user.is_authenticated else None
    
    def get_current_user_id(self) -> int:
        """Get current user ID"""
        return current_user.id if current_user.is_authenticated else None


# Authentication route handlers
class AuthRoutes:
    """Handle authentication routes"""
    
    def __init__(self):
        self.auth_manager = AuthenticationManager()
    
    def signup_handler(self):
        """Handle signup route"""
        if request.method == 'POST':
            username = request.form['username']
            email = request.form['email']
            password = request.form['password']
            
            success, message, user = self.auth_manager.register_user(username, email, password)
            
            if success:
                self.auth_manager.login_user_session(user)
                return redirect(url_for('dashboard'))
            else:
                return render_template('signup.html', error=message)
        
        return render_template('signup.html')
    
    def login_handler(self):
        """Handle login route"""
        if request.method == 'POST':
            username = request.form['username']
            password = request.form['password']
            
            success, message, user = self.auth_manager.authenticate_user(username, password)
            
            if success:
                self.auth_manager.login_user_session(user)
                return redirect(url_for('dashboard'))
            else:
                return render_template('login.html', error=message)
        
        return render_template('login.html')
    
    def logout_handler(self):
        """Handle logout route"""
        self.auth_manager.logout_user_session()
        return redirect(url_for('index'))


# Global instances to be used across the application
auth_manager = AuthenticationManager()
auth_routes = AuthRoutes()