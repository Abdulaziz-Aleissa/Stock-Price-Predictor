"""Backward compatibility wrapper for original data processing functions."""

import logging
from typing import Any
import pandas as pd

from .processors.stock_processor import StockProcessor

logger = logging.getLogger(__name__)

# Initialize the stock processor
stock_processor = StockProcessor()


def load_data(stock_symbol: str) -> pd.DataFrame:
    """Load stock data - wrapper for backward compatibility."""
    from ..services.stock_service import StockService
    
    stock_service = StockService()
    return stock_service.get_historical_data(stock_symbol)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean stock data - wrapper for backward compatibility."""
    return stock_processor.clean_data(df)


def save_data(df: pd.DataFrame, database_filepath: str) -> None:
    """Save data to database - wrapper for backward compatibility."""
    import os
    from sqlalchemy import create_engine
    
    logger.info(f"Saving data to {database_filepath}")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(database_filepath), exist_ok=True)
    
    engine = create_engine(f'sqlite:///{database_filepath}')
    table_name = f"{os.path.basename(database_filepath).replace('.db', '')}_table"
    
    df.to_sql(table_name, engine, index=False, if_exists='replace')
    logger.info(f"Successfully saved {len(df)} records to database")


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators - wrapper for backward compatibility."""
    from .processors.technical_indicators import TechnicalIndicators
    
    technical_indicators = TechnicalIndicators()
    return technical_indicators.calculate_all_indicators(df)