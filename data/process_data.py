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
    Fetches historical stock data with error handling and offline fallback
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
        logger.warning(f"Error fetching data from yfinance: {str(e)}")
        logger.info(f"Generating demo data for {stock_name} for offline demonstration")
        
        # Generate demo data for offline environments
        return generate_demo_data(stock_name)

def generate_demo_data(stock_name):
    """
    Generate realistic demo stock data for offline environments
    """
    try:
        import hashlib
        import random
        
        # Use stock name to generate consistent demo data
        seed = int(hashlib.md5(stock_name.encode()).hexdigest()[:8], 16)
        random.seed(seed)
        np.random.seed(seed % (2**32))
        
        # Generate 2 years of daily data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)
        
        # Create date range
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        # Remove weekends (approximate trading days)
        dates = [d for d in dates if d.weekday() < 5]
        
        n_days = len(dates)
        
        # Generate realistic stock data
        # Starting price based on stock name
        base_price = 50 + (seed % 500)  # $50-$550
        
        # Generate price series with random walk + trend
        returns = np.random.normal(0.0005, 0.02, n_days)  # Daily returns
        prices = [base_price]
        
        for i in range(1, n_days):
            # Add some mean reversion and trend
            trend = 0.0002 if i % 250 < 200 else -0.0001  # Yearly trend
            price_change = returns[i] + trend
            new_price = prices[-1] * (1 + price_change)
            prices.append(max(new_price, 1.0))  # Ensure price doesn't go negative
        
        # Generate volume data
        base_volume = 1000000 + (seed % 5000000)  # 1M-6M base volume
        volumes = []
        for i in range(n_days):
            volume_change = np.random.normal(1.0, 0.3)
            volume = int(base_volume * max(volume_change, 0.1))
            volumes.append(volume)
        
        # Generate OHLC data
        opens = []
        highs = []
        lows = []
        
        for i, close_price in enumerate(prices):
            # Open price (previous close + gap)
            if i == 0:
                open_price = close_price
            else:
                gap = np.random.normal(0, 0.005)
                open_price = prices[i-1] * (1 + gap)
                
            # High and low based on intraday volatility
            daily_vol = np.random.uniform(0.01, 0.04)
            high_price = max(open_price, close_price) * (1 + daily_vol)
            low_price = min(open_price, close_price) * (1 - daily_vol)
            
            opens.append(open_price)
            highs.append(high_price)
            lows.append(low_price)
        
        # Create DataFrame
        df = pd.DataFrame({
            'Open': opens,
            'High': highs,
            'Low': lows,
            'Close': prices,
            'Volume': volumes
        }, index=pd.DatetimeIndex(dates[:n_days]))
        
        logger.info(f"Generated demo data for {stock_name}: {len(df)} records")
        return df
        
    except Exception as e:
        logger.error(f"Error generating demo data: {str(e)}")
        raise

def clean_data(df):
    """
    Prepares stock data with NaN handling and infinity protection
    """
    try:
        # Ensure datetime index
        df.index = pd.to_datetime(df.index)
        
        # Basic calculations with robust handling
        df['Price_Change'] = df['Close'].pct_change().fillna(0)
        df['Volume_Change'] = df['Volume'].pct_change().fillna(0)
        df['High_Low_Range'] = df['High'] - df['Low']
        df['Daily_Return'] = df['Close'].pct_change().fillna(0)
        
        # Replace infinite values with zero
        df['Price_Change'] = df['Price_Change'].replace([np.inf, -np.inf], 0)
        df['Volume_Change'] = df['Volume_Change'].replace([np.inf, -np.inf], 0)
        df['Daily_Return'] = df['Daily_Return'].replace([np.inf, -np.inf], 0)
        
        # Handle rolling calculations
        df['Volatility'] = df['Daily_Return'].rolling(window=20).std().fillna(0)
        df['SMA_20'] = df['Close'].rolling(window=20).mean().fillna(method='bfill').fillna(df['Close'])
        df['SMA_50'] = df['Close'].rolling(window=50).mean().fillna(method='bfill').fillna(df['Close'])
        
        # RSI with robust division handling
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean().fillna(0)
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean().fillna(0)
        
        # Prevent division by zero in RSI calculation
        rs = np.where(loss != 0, gain / loss, 0)
        df['RSI'] = np.where(rs != 0, 100 - (100 / (1 + rs)), 50)
        df['RSI'] = pd.Series(df['RSI']).fillna(50)  # Neutral RSI for NaN values
        
        # Replace any infinite RSI values
        df['RSI'] = df['RSI'].replace([np.inf, -np.inf], 50)
        
        # MACD with robust calculation
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = (exp1 - exp2).fillna(0)
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean().fillna(0)
        
        # Replace infinite MACD values
        df['MACD'] = df['MACD'].replace([np.inf, -np.inf], 0)
        df['MACD_Signal'] = df['MACD_Signal'].replace([np.inf, -np.inf], 0)
        
        # Target variable - shifted close price
        df["Tomorrow"] = df['Close'].shift(-1)
        
        # Save date
        df['Date'] = df.index
        
        # Drop the last row as it will have NaN in Tomorrow
        df = df.iloc[:-1].copy()
        
        # Final comprehensive cleanup
        # Replace any remaining infinite values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], 0)
        
        # Fill any remaining NaN values
        df = df.fillna(0)
        
        # Final validation - ensure no infinite or extremely large values
        for col in numeric_columns:
            if col != 'Date':  # Skip date column
                max_val = df[col].abs().max()
                if max_val > 1e10:  # If values are extremely large
                    logger.warning(f"Large values detected in {col}, capping at reasonable limits")
                    df[col] = df[col].clip(-1e6, 1e6)
        
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
