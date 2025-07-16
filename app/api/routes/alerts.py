"""Alert-related routes."""

from flask import Blueprint, request, redirect, url_for, flash, jsonify
from flask_login import login_required, current_user
import logging

from ...services.alert_service import AlertService
from ...core.exceptions import AlertError, ValidationError
from ...config.database import get_db


logger = logging.getLogger(__name__)

alert_bp = Blueprint('alert', __name__)


@alert_bp.route('/add_alert', methods=['POST'])
@login_required
def add_alert():
    """Add a new price alert."""
    try:
        db = get_db()
        alert_service = AlertService(db)
        
        symbol = request.form.get('symbol', '').upper().strip()
        target_price = float(request.form.get('target_price', 0))
        condition = request.form.get('condition', '').lower()
        
        alert_service.create_alert(
            user_id=current_user.id,
            symbol=symbol,
            target_price=target_price,
            condition=condition
        )
        
        flash(f'Alert created for {symbol} {condition} ${target_price:.2f}!', 'success')
        
    except (AlertError, ValidationError) as e:
        flash(str(e), 'error')
    except ValueError:
        flash('Please provide a valid target price.', 'error')
    except Exception as e:
        logger.error(f"Error adding alert: {str(e)}")
        flash('An error occurred while creating the alert.', 'error')
    finally:
        if 'db' in locals():
            db.close()
    
    return redirect(url_for('dashboard.dashboard'))


@alert_bp.route('/remove_alert/<int:alert_id>')
@login_required
def remove_alert(alert_id):
    """Remove a price alert."""
    try:
        db = get_db()
        alert_service = AlertService(db)
        
        alert_service.remove_alert(current_user.id, alert_id)
        flash('Alert removed successfully!', 'success')
        
    except AlertError as e:
        flash(str(e), 'error')
    except Exception as e:
        logger.error(f"Error removing alert: {str(e)}")
        flash('An error occurred while removing the alert.', 'error')
    finally:
        if 'db' in locals():
            db.close()
    
    return redirect(url_for('dashboard.dashboard'))


@alert_bp.route('/toggle_alert/<int:alert_id>')
@login_required
def toggle_alert(alert_id):
    """Toggle alert active status."""
    try:
        db = get_db()
        alert_service = AlertService(db)
        
        alert = alert_service.toggle_alert(current_user.id, alert_id)
        status = 'activated' if alert.is_active else 'deactivated'
        flash(f'Alert {status} successfully!', 'success')
        
    except AlertError as e:
        flash(str(e), 'error')
    except Exception as e:
        logger.error(f"Error toggling alert: {str(e)}")
        flash('An error occurred while updating the alert.', 'error')
    finally:
        if 'db' in locals():
            db.close()
    
    return redirect(url_for('dashboard.dashboard'))


@alert_bp.route('/mark_notification_read/<int:notification_id>')
@login_required
def mark_notification_read(notification_id):
    """Mark a notification as read."""
    try:
        db = get_db()
        alert_service = AlertService(db)
        
        alert_service.mark_notification_read(current_user.id, notification_id)
        
    except AlertError as e:
        logger.warning(f"Error marking notification as read: {str(e)}")
    except Exception as e:
        logger.error(f"Error marking notification as read: {str(e)}")
    finally:
        if 'db' in locals():
            db.close()
    
    return redirect(url_for('dashboard.dashboard'))


@alert_bp.route('/mark_all_notifications_read', methods=['POST'])
@login_required
def mark_all_notifications_read():
    """Mark all notifications as read."""
    try:
        db = get_db()
        alert_service = AlertService(db)
        
        count = alert_service.mark_all_notifications_read(current_user.id)
        flash(f'Marked {count} notifications as read.', 'info')
        
    except AlertError as e:
        flash(str(e), 'error')
    except Exception as e:
        logger.error(f"Error marking all notifications as read: {str(e)}")
        flash('An error occurred while updating notifications.', 'error')
    finally:
        if 'db' in locals():
            db.close()
    
    return redirect(url_for('dashboard.dashboard'))


@alert_bp.route('/api/alerts')
@login_required
def get_alerts_api():
    """API endpoint for user alerts."""
    try:
        db = get_db()
        alert_service = AlertService(db)
        
        active_only = request.args.get('active_only', 'true').lower() == 'true'
        alerts_data = alert_service.get_user_alerts(current_user.id, active_only)
        
        return jsonify({
            'alerts': alerts_data,
            'count': len(alerts_data)
        })
        
    except Exception as e:
        logger.error(f"Error getting alerts API data: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500
    finally:
        if 'db' in locals():
            db.close()


@alert_bp.route('/api/notifications')
@login_required
def get_notifications_api():
    """API endpoint for user notifications."""
    try:
        db = get_db()
        alert_service = AlertService(db)
        
        unread_only = request.args.get('unread_only', 'false').lower() == 'true'
        limit = int(request.args.get('limit', 50))
        
        notifications = alert_service.get_user_notifications(
            current_user.id, unread_only, limit
        )
        
        return jsonify({
            'notifications': notifications,
            'count': len(notifications)
        })
        
    except Exception as e:
        logger.error(f"Error getting notifications API data: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500
    finally:
        if 'db' in locals():
            db.close()


@alert_bp.route('/api/notifications/<int:notification_id>/read', methods=['POST'])
@login_required
def mark_notification_read_api(notification_id):
    """API endpoint to mark notification as read."""
    try:
        db = get_db()
        alert_service = AlertService(db)
        
        alert_service.mark_notification_read(current_user.id, notification_id)
        
        return jsonify({'success': True, 'message': 'Notification marked as read'})
        
    except AlertError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Error marking notification as read (API): {str(e)}")
        return jsonify({'success': False, 'error': 'Internal server error'}), 500
    finally:
        if 'db' in locals():
            db.close()