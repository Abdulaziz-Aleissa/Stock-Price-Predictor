import yfinance as yf
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import sys
import os
import logging
from datetime import datetime, timedelta
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
        # Use end date as tomorrow to ensure we have latest data
        end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        df = stock.history(period="max", end=end_date)
        
        if df.empty:
            raise ValueError(f"No data found for ticker {stock_name}")
            
        logger.info(f"Successfully fetched {len(df)} records, latest date: {df.index[-1]}")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching data: {str(e)}")
        raise

def clean_data(df):
    """
    Prepares stock data with NaN handling
    """
    try:
        # Ensure datetime index
        df.index = pd.to_datetime(df.index)
        
        # Basic calculations
        df['Price_Change'] = df['Close'].pct_change().fillna(0)
        df['Volume_Change'] = df['Volume'].pct_change().fillna(0)
        df['High_Low_Range'] = df['High'] - df['Low']
        df['Daily_Return'] = df['Close'].pct_change().fillna(0)
        
        # Handle rolling calculations
        df['Volatility'] = df['Daily_Return'].rolling(window=20).std().fillna(0)
        df['SMA_20'] = df['Close'].rolling(window=20).mean().fillna(method='bfill').fillna(df['Close'])
        df['SMA_50'] = df['Close'].rolling(window=50).mean().fillna(method='bfill').fillna(df['Close'])
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean().fillna(0)
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean().fillna(0)
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI'] = df['RSI'].fillna(50)  # Neutral RSI for NaN values
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = (exp1 - exp2).fillna(0)
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean().fillna(0)
        
        # Target variable - shifted close price
        df["Tomorrow"] = df['Close'].shift(-1)
        
        # Save date
        df['Date'] = df.index
        
        # Drop the last row as it will have NaN in Tomorrow
        df = df.iloc[:-1].copy()
        
        # Final check for any remaining NaN
        df = df.fillna(0)
        
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
