"""Decorator utilities."""

from functools import wraps
import time
import logging
from typing import Callable, Any

from ..core.exceptions import ValidationError


logger = logging.getLogger(__name__)


def timer(func: Callable) -> Callable:
    """Decorator to time function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        
        logger.info(f"{func.__name__} executed in {execution_time:.4f} seconds")
        return result
    return wrapper


def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator to retry function execution on failure."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            current_delay = delay
            
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts >= max_attempts:
                        logger.error(f"{func.__name__} failed after {max_attempts} attempts: {str(e)}")
                        raise
                    
                    logger.warning(f"{func.__name__} attempt {attempts} failed: {str(e)}. "
                                 f"Retrying in {current_delay} seconds...")
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            return None
        return wrapper
    return decorator


def cache_result(ttl: int = 300):
    """Decorator to cache function results for a specified time."""
    def decorator(func: Callable) -> Callable:
        cache = {}
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Check if result is in cache and not expired
            if cache_key in cache:
                result, timestamp = cache[cache_key]
                if time.time() - timestamp < ttl:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return result
                else:
                    # Remove expired entry
                    del cache[cache_key]
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache[cache_key] = (result, time.time())
            logger.debug(f"Cache miss for {func.__name__}, result cached")
            
            return result
        
        # Add cache clearing method
        wrapper.clear_cache = lambda: cache.clear()
        
        return wrapper
    return decorator


def validate_params(**param_validators):
    """Decorator to validate function parameters."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate parameters
            for param_name, validator in param_validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    
                    if callable(validator):
                        if not validator(value):
                            raise ValidationError(f"Invalid value for parameter '{param_name}': {value}")
                    elif isinstance(validator, type):
                        if not isinstance(value, validator):
                            raise ValidationError(f"Parameter '{param_name}' must be of type {validator.__name__}")
                    elif isinstance(validator, (list, tuple)):
                        if value not in validator:
                            raise ValidationError(f"Parameter '{param_name}' must be one of {validator}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def log_calls(level: int = logging.INFO):
    """Decorator to log function calls."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.log(level, f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
            try:
                result = func(*args, **kwargs)
                logger.log(level, f"{func.__name__} completed successfully")
                return result
            except Exception as e:
                logger.log(logging.ERROR, f"{func.__name__} failed with error: {str(e)}")
                raise
        return wrapper
    return decorator


def handle_exceptions(default_return=None, exceptions=(Exception,)):
    """Decorator to handle exceptions and return default value."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                logger.error(f"Exception in {func.__name__}: {str(e)}")
                return default_return
        return wrapper
    return decorator


def require_non_empty(*param_names):
    """Decorator to ensure specified parameters are not empty."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            for param_name in param_names:
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if not value or (isinstance(value, str) and not value.strip()):
                        raise ValidationError(f"Parameter '{param_name}' cannot be empty")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def normalize_ticker(func: Callable) -> Callable:
    """Decorator to normalize ticker symbols in function arguments."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        import inspect
        
        # Get function signature
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        
        # Normalize ticker parameters
        ticker_params = ['ticker', 'symbol', 'stock_symbol']
        
        for param_name in ticker_params:
            if param_name in bound_args.arguments:
                value = bound_args.arguments[param_name]
                if isinstance(value, str):
                    bound_args.arguments[param_name] = value.upper().strip()
        
        return func(*bound_args.args, **bound_args.kwargs)
    return wrapper


def deprecated(replacement: str = None):
    """Decorator to mark functions as deprecated."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            message = f"{func.__name__} is deprecated"
            if replacement:
                message += f". Use {replacement} instead"
            
            logger.warning(message)
            return func(*args, **kwargs)
        return wrapper
    return decorator