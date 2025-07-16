"""Formatting utilities for display and output."""

import json
from typing import Any, Dict, List, Optional
from datetime import datetime

from .helpers import format_currency, format_percentage, format_large_number, calculate_change_color


class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for handling datetime objects and other types."""
    
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        return super().default(obj)


def format_portfolio_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Format portfolio item for display."""
    formatted = item.copy()
    
    # Format currency values
    formatted['purchase_price_formatted'] = format_currency(item.get('purchase_price', 0))
    formatted['current_price_formatted'] = format_currency(item.get('current_price', 0))
    formatted['position_value_formatted'] = format_currency(item.get('position_value', 0))
    formatted['total_cost_formatted'] = format_currency(item.get('total_cost', 0))
    formatted['profit_loss_formatted'] = format_currency(item.get('profit_loss', 0))
    
    # Format percentage
    formatted['change_percent_formatted'] = format_percentage(item.get('change_percent', 0))
    
    # Add color class for profit/loss
    formatted['profit_loss_color'] = calculate_change_color(item.get('profit_loss', 0))
    formatted['change_percent_color'] = calculate_change_color(item.get('change_percent', 0))
    
    return formatted


def format_portfolio_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Format portfolio summary for display."""
    formatted = summary.copy()
    
    # Format currency values
    formatted['total_value_formatted'] = format_currency(summary.get('total_value', 0))
    formatted['total_cost_formatted'] = format_currency(summary.get('total_cost', 0))
    formatted['total_profit_loss_formatted'] = format_currency(summary.get('total_profit_loss', 0))
    
    # Format percentage
    formatted['total_return_percent_formatted'] = format_percentage(summary.get('total_return_percent', 0))
    
    # Add color classes
    formatted['profit_loss_color'] = calculate_change_color(summary.get('total_profit_loss', 0))
    formatted['return_percent_color'] = calculate_change_color(summary.get('total_return_percent', 0))
    
    return formatted


def format_watchlist_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Format watchlist item for display."""
    formatted = item.copy()
    
    # Format currency values
    formatted['target_price_formatted'] = format_currency(item.get('target_price', 0))
    formatted['current_price_formatted'] = format_currency(item.get('current_price', 0))
    formatted['distance_to_target_formatted'] = format_currency(item.get('distance_to_target', 0))
    
    # Format percentage
    formatted['percent_to_target_formatted'] = format_percentage(item.get('percent_to_target', 0))
    
    # Add color class
    formatted['distance_color'] = calculate_change_color(item.get('distance_to_target', 0))
    
    return formatted


def format_alert_item(alert: Dict[str, Any]) -> Dict[str, Any]:
    """Format alert item for display."""
    formatted = alert.copy()
    
    # Format prices
    formatted['target_price_formatted'] = format_currency(alert.get('target_price', 0))
    formatted['current_price_formatted'] = format_currency(alert.get('current_price', 0))
    
    # Format condition for display
    condition = alert.get('condition', '').lower()
    if condition == 'above':
        formatted['condition_symbol'] = '↑'
        formatted['condition_text'] = 'above'
    elif condition == 'below':
        formatted['condition_symbol'] = '↓'
        formatted['condition_text'] = 'below'
    else:
        formatted['condition_symbol'] = '?'
        formatted['condition_text'] = condition
    
    # Status formatting
    if alert.get('is_active'):
        formatted['status_text'] = 'Active'
        formatted['status_color'] = 'text-success'
    else:
        formatted['status_text'] = 'Inactive'
        formatted['status_color'] = 'text-muted'
    
    return formatted


def format_stock_info(info: Dict[str, Any]) -> Dict[str, Any]:
    """Format stock information for display."""
    formatted = info.copy()
    
    # Format currency values
    price_fields = ['current_price', 'day_high', 'day_low', 'year_high', 'year_low']
    for field in price_fields:
        if field in info:
            formatted[f'{field}_formatted'] = format_currency(info[field])
    
    # Format large numbers
    if 'market_cap' in info and info['market_cap'] != 'N/A':
        formatted['market_cap_formatted'] = format_large_number(info['market_cap'])
    
    if 'volume' in info and info['volume'] != 'N/A':
        formatted['volume_formatted'] = format_large_number(info['volume'])
    
    # Format PE ratio
    if 'pe_ratio' in info and info['pe_ratio'] != 'N/A':
        try:
            pe_ratio = float(info['pe_ratio'])
            formatted['pe_ratio_formatted'] = f"{pe_ratio:.2f}"
        except (ValueError, TypeError):
            formatted['pe_ratio_formatted'] = 'N/A'
    
    return formatted


def format_prediction_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Format prediction result for display."""
    formatted = result.copy()
    
    # Format prices
    formatted['current_price_formatted'] = format_currency(result.get('current_price', 0))
    formatted['predicted_price_formatted'] = format_currency(result.get('predicted_price', 0))
    formatted['price_change_formatted'] = format_currency(result.get('price_change', 0))
    
    # Format percentage
    formatted['price_change_percent_formatted'] = format_percentage(result.get('price_change_percent', 0))
    
    # Add color class
    formatted['change_color'] = calculate_change_color(result.get('price_change_percent', 0))
    
    # Format confidence metrics
    if 'confidence_metrics' in result:
        metrics = result['confidence_metrics']
        formatted['confidence_metrics_formatted'] = {
            'r2_score': f"{metrics.get('r2_score', 0):.3f}",
            'mae': format_currency(metrics.get('mae', 0)),
            'rmse': format_currency(metrics.get('rmse', 0))
        }
    
    return formatted


def format_comparison_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Format stock comparison data for display."""
    formatted = data.copy()
    
    for symbol_key in ['symbol1', 'symbol2']:
        if symbol_key in data:
            symbol_data = data[symbol_key]
            formatted_symbol = symbol_data.copy()
            
            # Format percentage change
            formatted_symbol['change_percent_formatted'] = format_percentage(
                symbol_data.get('change_percent', 0)
            )
            
            # Format volume
            formatted_symbol['volume_avg_formatted'] = format_large_number(
                symbol_data.get('volume_avg', 0)
            )
            
            # Format prices
            for price_field in ['high', 'low']:
                if price_field in symbol_data:
                    formatted_symbol[f'{price_field}_formatted'] = format_currency(
                        symbol_data[price_field]
                    )
            
            # Add color class
            formatted_symbol['change_color'] = calculate_change_color(
                symbol_data.get('change_percent', 0)
            )
            
            formatted[symbol_key] = formatted_symbol
    
    return formatted


def format_api_response(data: Any, status: str = 'success', message: str = None) -> Dict[str, Any]:
    """Format API response with consistent structure."""
    response = {
        'status': status,
        'timestamp': datetime.now().isoformat(),
        'data': data
    }
    
    if message:
        response['message'] = message
    
    return response


def format_error_response(error: str, details: str = None, error_code: str = None) -> Dict[str, Any]:
    """Format error response with consistent structure."""
    response = {
        'status': 'error',
        'timestamp': datetime.now().isoformat(),
        'error': error
    }
    
    if details:
        response['details'] = details
    
    if error_code:
        response['error_code'] = error_code
    
    return response


def to_json(data: Any, indent: int = None) -> str:
    """Convert data to JSON string using custom encoder."""
    return json.dumps(data, cls=JSONEncoder, indent=indent)


def format_table_data(data: List[Dict[str, Any]], headers: List[str]) -> Dict[str, Any]:
    """Format data for table display."""
    return {
        'headers': headers,
        'rows': data,
        'total_rows': len(data)
    }