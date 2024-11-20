import yfinance as yf
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import sys
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_technical_indicators(df):
    """Calculate technical indicators using pandas"""
    try:
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Moving averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        sma = df['Close'].rolling(window=20).mean()
        std = df['Close'].rolling(window=20).std()
        df['Upper_BB'] = sma + (std * 2)
        df['Lower_BB'] = sma - (std * 2)
        
        return df
    except Exception as e:
        logger.error(f"Error calculating technical indicators: {str(e)}")
        raise

def load_data(stock_name):
    """
    Fetches historical stock data with error handling
    """
    try:
        logger.info(f"Fetching data for {stock_name}")
        stock = yf.Ticker(stock_name)
        df = stock.history(period="max")
        
        if df.empty:
            raise ValueError(f"No data found for ticker {stock_name}")
            
        logger.info(f"Successfully fetched {len(df)} records")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching data: {str(e)}")
        raise

def clean_data(df):
    """
    Prepares stock data by calculating various technical indicators
    """
    try:
        logger.info("Cleaning and processing data")
        
        # Ensure datetime index
        df.index = pd.to_datetime(df.index)
        
        # Basic price metrics
        df['Price_Change'] = df['Close'].pct_change()
        df['Volume_Change'] = df['Volume'].pct_change()
        df['High_Low_Range'] = df['High'] - df['Low']
        df['Daily_Return'] = df['Close'].pct_change()
        df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
        
        # Technical indicators
        df = calculate_technical_indicators(df)
        
        # Target variable
        df["Tomorrow"] = df['Close'].shift(-1)
        
        # Save date before resetting index
        df['Date'] = df.index
        
        # Clean up
        df = df.dropna()
        df.reset_index(drop=True, inplace=True)
        
        logger.info("Data processing completed")
        return df
        
    except Exception as e:
        logger.error(f"Error in data cleaning: {str(e)}")
        raise

def save_data(df, database_filepath):
    """
    Saves the DataFrame to an SQLite database with error handling
    """
    try:
        logger.info(f"Saving data to {database_filepath}")
        
        engine = create_engine(f'sqlite:///{database_filepath}')
        table_name = f"{os.path.basename(database_filepath).replace('.db', '')}_table"
        
        df.to_sql(table_name, engine, index=False, if_exists='replace')
        logger.info(f"Successfully saved {len(df)} records to database")
        
    except Exception as e:
        logger.error(f"Error saving to database: {str(e)}")
        raise

def main():
    """
    Main function to run the data processing pipeline
    """
    try:
        if len(sys.argv) == 3:
            stock_symbol = sys.argv[1]
            database_filepath = sys.argv[2]

            logger.info(f'Loading data for stock: {stock_symbol}')
            df = load_data(stock_symbol)

            logger.info('Cleaning and processing data...')
            df = clean_data(df)
            
            logger.info(f'Saving data to database: {database_filepath}')
            save_data(df, database_filepath)
            
            logger.info('Data processing completed successfully!')
            
        else:
            logger.error('Please provide the stock ticker symbol and database filepath')
            logger.info('Example: python process_data.py NVDA ../data/nvda_stock.db')
            
    except Exception as e:
        logger.error(f'Error in main process: {str(e)}')
        sys.exit(1)

if __name__ == '__main__':
    main()
