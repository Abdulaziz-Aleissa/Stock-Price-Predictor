"""Portfolio-related routes."""

from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from flask_login import login_required, current_user
import logging

from ...services.portfolio_service import PortfolioService
from ...services.stock_service import StockService
from ...core.exceptions import PortfolioError, ValidationError
from ...config.database import get_db


logger = logging.getLogger(__name__)

portfolio_bp = Blueprint('portfolio', __name__)


@portfolio_bp.route('/add_to_portfolio', methods=['POST'])
@login_required
def add_to_portfolio():
    """Add stock to user's portfolio."""
    try:
        db = get_db()
        portfolio_service = PortfolioService(db)
        
        symbol = request.form.get('symbol', '').upper().strip()
        shares = float(request.form.get('shares', 0))
        purchase_price = float(request.form.get('purchase_price', 0))
        
        portfolio_service.add_to_portfolio(
            user_id=current_user.id,
            symbol=symbol,
            shares=shares,
            purchase_price=purchase_price
        )
        
        flash(f'Added {shares} shares of {symbol} to your portfolio!', 'success')
        
    except (PortfolioError, ValidationError) as e:
        flash(str(e), 'error')
    except ValueError:
        flash('Please provide valid numeric values for shares and price.', 'error')
    except Exception as e:
        logger.error(f"Error adding to portfolio: {str(e)}")
        flash('An error occurred while adding to portfolio.', 'error')
    finally:
        if 'db' in locals():
            db.close()
    
    return redirect(url_for('dashboard.dashboard'))


@portfolio_bp.route('/remove_from_portfolio/<int:item_id>')
@login_required
def remove_from_portfolio(item_id):
    """Remove stock from user's portfolio."""
    try:
        db = get_db()
        portfolio_service = PortfolioService(db)
        
        portfolio_service.remove_from_portfolio(current_user.id, item_id)
        flash('Stock removed from portfolio!', 'success')
        
    except PortfolioError as e:
        flash(str(e), 'error')
    except Exception as e:
        logger.error(f"Error removing from portfolio: {str(e)}")
        flash('An error occurred while removing from portfolio.', 'error')
    finally:
        if 'db' in locals():
            db.close()
    
    return redirect(url_for('dashboard.dashboard'))


@portfolio_bp.route('/update_portfolio/<int:item_id>', methods=['POST'])
@login_required
def update_portfolio(item_id):
    """Update portfolio item."""
    try:
        db = get_db()
        portfolio_service = PortfolioService(db)
        
        shares = request.form.get('shares')
        purchase_price = request.form.get('purchase_price')
        
        update_data = {}
        if shares:
            update_data['shares'] = float(shares)
        if purchase_price:
            update_data['purchase_price'] = float(purchase_price)
        
        portfolio_service.update_portfolio_item(current_user.id, item_id, **update_data)
        flash('Portfolio item updated successfully!', 'success')
        
    except (PortfolioError, ValidationError) as e:
        flash(str(e), 'error')
    except ValueError:
        flash('Please provide valid numeric values.', 'error')
    except Exception as e:
        logger.error(f"Error updating portfolio: {str(e)}")
        flash('An error occurred while updating portfolio.', 'error')
    finally:
        if 'db' in locals():
            db.close()
    
    return redirect(url_for('dashboard.dashboard'))


@portfolio_bp.route('/add_to_watchlist', methods=['POST'])
@login_required
def add_to_watchlist():
    """Add stock to user's watchlist."""
    try:
        db = get_db()
        portfolio_service = PortfolioService(db)
        
        symbol = request.form.get('symbol', '').upper().strip()
        target_price = float(request.form.get('target_price', 0))
        
        portfolio_service.add_to_watchlist(
            user_id=current_user.id,
            symbol=symbol,
            target_price=target_price
        )
        
        flash(f'Added {symbol} to your watchlist!', 'success')
        
    except (PortfolioError, ValidationError) as e:
        flash(str(e), 'error')
    except ValueError:
        flash('Please provide a valid target price.', 'error')
    except Exception as e:
        logger.error(f"Error adding to watchlist: {str(e)}")
        flash('An error occurred while adding to watchlist.', 'error')
    finally:
        if 'db' in locals():
            db.close()
    
    return redirect(url_for('dashboard.dashboard'))


@portfolio_bp.route('/remove_from_watchlist/<int:item_id>')
@login_required
def remove_from_watchlist(item_id):
    """Remove stock from user's watchlist."""
    try:
        db = get_db()
        portfolio_service = PortfolioService(db)
        
        portfolio_service.remove_from_watchlist(current_user.id, item_id)
        flash('Stock removed from watchlist!', 'success')
        
    except PortfolioError as e:
        flash(str(e), 'error')
    except Exception as e:
        logger.error(f"Error removing from watchlist: {str(e)}")
        flash('An error occurred while removing from watchlist.', 'error')
    finally:
        if 'db' in locals():
            db.close()
    
    return redirect(url_for('dashboard.dashboard'))


@portfolio_bp.route('/api/portfolio')
@login_required
def get_portfolio_api():
    """API endpoint for portfolio data."""
    try:
        db = get_db()
        portfolio_service = PortfolioService(db)
        
        portfolio_data = portfolio_service.get_portfolio(current_user.id)
        portfolio_summary = portfolio_service.get_portfolio_summary(current_user.id)
        
        return jsonify({
            'portfolio': portfolio_data,
            'summary': portfolio_summary
        })
        
    except Exception as e:
        logger.error(f"Error getting portfolio API data: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500
    finally:
        if 'db' in locals():
            db.close()


@portfolio_bp.route('/api/watchlist')
@login_required
def get_watchlist_api():
    """API endpoint for watchlist data."""
    try:
        db = get_db()
        portfolio_service = PortfolioService(db)
        
        watchlist_data = portfolio_service.get_watchlist(current_user.id)
        
        return jsonify({
            'watchlist': watchlist_data
        })
        
    except Exception as e:
        logger.error(f"Error getting watchlist API data: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500
    finally:
        if 'db' in locals():
            db.close()


@portfolio_bp.route('/api/portfolio/performance')
@login_required
def get_portfolio_performance():
    """API endpoint for portfolio performance metrics."""
    try:
        db = get_db()
        portfolio_service = PortfolioService(db)
        stock_service = StockService()
        
        portfolio_data = portfolio_service.get_portfolio(current_user.id)
        
        # Calculate additional performance metrics
        performance_metrics = {
            'total_positions': len(portfolio_data),
            'profitable_positions': len([p for p in portfolio_data if p['profit_loss'] > 0]),
            'losing_positions': len([p for p in portfolio_data if p['profit_loss'] < 0]),
            'best_performer': None,
            'worst_performer': None
        }
        
        if portfolio_data:
            # Find best and worst performers
            best = max(portfolio_data, key=lambda x: x['change_percent'])
            worst = min(portfolio_data, key=lambda x: x['change_percent'])
            
            performance_metrics['best_performer'] = {
                'symbol': best['symbol'],
                'change_percent': best['change_percent']
            }
            performance_metrics['worst_performer'] = {
                'symbol': worst['symbol'],
                'change_percent': worst['change_percent']
            }
        
        return jsonify(performance_metrics)
        
    except Exception as e:
        logger.error(f"Error getting portfolio performance: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500
    finally:
        if 'db' in locals():
            db.close()