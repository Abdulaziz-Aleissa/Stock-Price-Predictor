"""Authentication routes."""

from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import logging

from ...models.user import User
from ...config.database import get_db
from ...core.exceptions import AuthenticationError, ValidationError


logger = logging.getLogger(__name__)

auth_bp = Blueprint('auth', __name__)


@auth_bp.route('/signup', methods=['GET', 'POST'])
def signup():
    """User registration."""
    if request.method == 'POST':
        try:
            db = get_db()
            
            username = request.form.get('username', '').strip()
            email = request.form.get('email', '').strip()
            password = request.form.get('password', '')
            
            # Validate input
            if not username or len(username) < 3:
                raise ValidationError("Username must be at least 3 characters long")
            
            if not email or '@' not in email:
                raise ValidationError("Please provide a valid email address")
            
            if not password or len(password) < 6:
                raise ValidationError("Password must be at least 6 characters long")
            
            # Check if user exists
            existing_user = db.query(User).filter_by(username=username).first()
            if existing_user:
                raise ValidationError("Username already exists")
            
            existing_email = db.query(User).filter_by(email=email).first()
            if existing_email:
                raise ValidationError("Email already registered")
            
            # Create new user
            user = User(
                username=username,
                email=email,
                password_hash=generate_password_hash(password)
            )
            
            db.add(user)
            db.commit()
            
            # Log in the user
            login_user(user)
            
            logger.info(f"New user registered: {username}")
            flash('Account created successfully!', 'success')
            return redirect(url_for('dashboard.dashboard'))
            
        except ValidationError as e:
            flash(str(e), 'error')
        except Exception as e:
            logger.error(f"Error during signup: {str(e)}")
            flash('An error occurred during registration. Please try again.', 'error')
        finally:
            if 'db' in locals():
                db.close()
    
    return render_template('signup.html')


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """User login."""
    if request.method == 'POST':
        try:
            db = get_db()
            
            username = request.form.get('username', '').strip()
            password = request.form.get('password', '')
            
            if not username or not password:
                raise ValidationError("Please provide both username and password")
            
            # Find user
            user = db.query(User).filter_by(username=username).first()
            
            if not user or not check_password_hash(user.password_hash, password):
                raise AuthenticationError("Invalid username or password")
            
            if not user.is_active:
                raise AuthenticationError("Account is deactivated")
            
            # Log in the user
            login_user(user)
            
            logger.info(f"User logged in: {username}")
            flash('Logged in successfully!', 'success')
            
            # Redirect to next page or dashboard
            next_page = request.args.get('next')
            if next_page:
                return redirect(next_page)
            return redirect(url_for('dashboard.dashboard'))
            
        except (AuthenticationError, ValidationError) as e:
            flash(str(e), 'error')
        except Exception as e:
            logger.error(f"Error during login: {str(e)}")
            flash('An error occurred during login. Please try again.', 'error')
        finally:
            if 'db' in locals():
                db.close()
    
    return render_template('login.html')


@auth_bp.route('/logout')
@login_required
def logout():
    """User logout."""
    try:
        username = current_user.username
        logout_user()
        
        logger.info(f"User logged out: {username}")
        flash('Logged out successfully!', 'info')
        
    except Exception as e:
        logger.error(f"Error during logout: {str(e)}")
        flash('An error occurred during logout.', 'error')
    
    return redirect(url_for('main.index'))


@auth_bp.route('/profile')
@login_required
def profile():
    """User profile page."""
    return render_template('profile.html', user=current_user)


@auth_bp.route('/change-password', methods=['POST'])
@login_required
def change_password():
    """Change user password."""
    try:
        db = get_db()
        
        current_password = request.form.get('current_password', '')
        new_password = request.form.get('new_password', '')
        confirm_password = request.form.get('confirm_password', '')
        
        # Validate input
        if not current_password:
            raise ValidationError("Please provide your current password")
        
        if not new_password or len(new_password) < 6:
            raise ValidationError("New password must be at least 6 characters long")
        
        if new_password != confirm_password:
            raise ValidationError("Password confirmation does not match")
        
        # Verify current password
        if not check_password_hash(current_user.password_hash, current_password):
            raise AuthenticationError("Current password is incorrect")
        
        # Update password
        user = db.query(User).get(current_user.id)
        user.password_hash = generate_password_hash(new_password)
        db.commit()
        
        logger.info(f"Password changed for user: {current_user.username}")
        flash('Password changed successfully!', 'success')
        
    except (AuthenticationError, ValidationError) as e:
        flash(str(e), 'error')
    except Exception as e:
        logger.error(f"Error changing password: {str(e)}")
        flash('An error occurred while changing password.', 'error')
    finally:
        if 'db' in locals():
            db.close()
    
    return redirect(url_for('auth.profile'))