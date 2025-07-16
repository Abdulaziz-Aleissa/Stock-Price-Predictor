"""Helper utility functions."""

import re
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP


logger = logging.getLogger(__name__)


def format_currency(amount: float, currency: str = 'USD', decimal_places: int = 2) -> str:
    """Format amount as currency string."""
    try:
        if currency == 'USD':
            return f"${amount:,.{decimal_places}f}"
        else:
            return f"{amount:,.{decimal_places}f} {currency}"
    except (ValueError, TypeError):
        return "N/A"


def format_percentage(value: float, decimal_places: int = 2) -> str:
    """Format value as percentage string."""
    try:
        return f"{value:+.{decimal_places}f}%"
    except (ValueError, TypeError):
        return "N/A"


def format_large_number(number: float) -> str:
    """Format large numbers with appropriate suffixes (K, M, B, T)."""
    try:
        if abs(number) >= 1e12:
            return f"{number/1e12:.1f}T"
        elif abs(number) >= 1e9:
            return f"{number/1e9:.1f}B"
        elif abs(number) >= 1e6:
            return f"{number/1e6:.1f}M"
        elif abs(number) >= 1e3:
            return f"{number/1e3:.1f}K"
        else:
            return f"{number:.0f}"
    except (ValueError, TypeError):
        return "N/A"


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float."""
    try:
        if value is None or value == '':
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert value to int."""
    try:
        if value is None or value == '':
            return default
        return int(float(value))  # Handle strings like "10.0"
    except (ValueError, TypeError):
        return default


def clean_ticker_symbol(ticker: str) -> str:
    """Clean and validate ticker symbol."""
    if not ticker:
        return ""
    
    # Remove whitespace and convert to uppercase
    ticker = ticker.strip().upper()
    
    # Remove any non-alphanumeric characters (except dots for some tickers)
    ticker = re.sub(r'[^A-Z0-9.]', '', ticker)
    
    return ticker


def validate_email(email: str) -> bool:
    """Validate email address format."""
    if not email:
        return False
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def calculate_time_ago(timestamp: datetime) -> str:
    """Calculate human-readable time ago string."""
    try:
        now = datetime.now()
        
        # Handle timezone-naive vs timezone-aware datetimes
        if timestamp.tzinfo is not None and now.tzinfo is None:
            now = now.replace(tzinfo=timestamp.tzinfo)
        elif timestamp.tzinfo is None and now.tzinfo is not None:
            timestamp = timestamp.replace(tzinfo=now.tzinfo)
        
        diff = now - timestamp
        
        if diff.days > 365:
            years = diff.days // 365
            return f"{years} year{'s' if years != 1 else ''} ago"
        elif diff.days > 30:
            months = diff.days // 30
            return f"{months} month{'s' if months != 1 else ''} ago"
        elif diff.days > 0:
            return f"{diff.days} day{'s' if diff.days != 1 else ''} ago"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        elif diff.seconds > 60:
            minutes = diff.seconds // 60
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        else:
            return "Just now"
    
    except Exception as e:
        logger.error(f"Error calculating time ago: {str(e)}")
        return "Unknown"


def round_to_decimal_places(value: float, places: int = 2) -> float:
    """Round value to specified decimal places."""
    try:
        decimal_value = Decimal(str(value))
        return float(decimal_value.quantize(Decimal('0.' + '0' * places), rounding=ROUND_HALF_UP))
    except (ValueError, TypeError):
        return 0.0


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split list into chunks of specified size."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def filter_dict(data: Dict[str, Any], allowed_keys: List[str]) -> Dict[str, Any]:
    """Filter dictionary to only include allowed keys."""
    return {key: value for key, value in data.items() if key in allowed_keys}


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple dictionaries."""
    result = {}
    for d in dicts:
        if d:
            result.update(d)
    return result


def get_nested_value(data: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """Get nested value from dictionary using dot notation."""
    try:
        keys = key_path.split('.')
        value = data
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError, AttributeError):
        return default


def calculate_change_color(change_percent: float) -> str:
    """Get color class based on change percentage."""
    if change_percent > 0:
        return "text-success"  # Green for positive
    elif change_percent < 0:
        return "text-danger"   # Red for negative
    else:
        return "text-muted"    # Gray for neutral


def format_date(date: datetime, format_string: str = "%Y-%m-%d") -> str:
    """Format datetime object as string."""
    try:
        return date.strftime(format_string)
    except (ValueError, TypeError, AttributeError):
        return "N/A"


def parse_date(date_string: str, format_string: str = "%Y-%m-%d") -> Optional[datetime]:
    """Parse date string to datetime object."""
    try:
        return datetime.strptime(date_string, format_string)
    except (ValueError, TypeError):
        return None


def is_market_hours() -> bool:
    """Check if current time is within market hours (9:30 AM - 4:00 PM ET)."""
    try:
        now = datetime.now()
        # Simple check - in production, you'd want to account for timezone and holidays
        weekday = now.weekday()  # 0 = Monday, 6 = Sunday
        hour = now.hour
        minute = now.minute
        
        # Check if it's a weekday
        if weekday >= 5:  # Saturday or Sunday
            return False
        
        # Check if it's within market hours (9:30 AM - 4:00 PM)
        start_time = 9 * 60 + 30  # 9:30 AM in minutes
        end_time = 16 * 60        # 4:00 PM in minutes
        current_time = hour * 60 + minute
        
        return start_time <= current_time <= end_time
    
    except Exception as e:
        logger.error(f"Error checking market hours: {str(e)}")
        return False


def generate_random_string(length: int = 10) -> str:
    """Generate random alphanumeric string."""
    import random
    import string
    
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))