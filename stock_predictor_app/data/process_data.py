import yfinance as yf
from sqlalchemy import create_engine
import pandas as pd
import sys
import os

def load_data(stock_name):
    """
    Fetches historical stock data for a given stock symbol using the yfinance library.

    Parameters:
    stock_name (str): The ticker symbol of the stock to load data for.

    Returns:
    pd.DataFrame: A DataFrame containing historical data for the specified stock.
    """
    df = yf.Ticker(stock_name).history(period="max")
    return df 

def clean_data(df):
    """
    Prepares stock data by ensuring the index is a DatetimeIndex and adding 
    necessary columns for analysis.
    """
    # Ensure the index is a DatetimeIndex
    df.index = pd.to_datetime(df.index)
    df["Tomorrow"] = df['Close'].shift(-1)
    df["Target"] = (df['Tomorrow'] > df['Close']).astype(int)
    df = df.dropna()

    # Reset the index to include the date as a column
    df.reset_index(inplace=True)
    return df




def save_data(df, database_filepath):
    """
    Saves the DataFrame to an SQLite database. The table name is generated based on
    the base name of the database filename (excluding its directory path).
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    
    # Extract only the base name of the database file (e.g., "NIO_StockData" from "data/NIO_StockData.db")
    table_name = f"{os.path.basename(database_filepath).replace('.db', '')}_table"
    
    print(f"Saving data to table: {table_name}")
    df.to_sql(table_name, engine, index=False, if_exists='replace')
    print(f"Data saved to table '{table_name}' in database '{database_filepath}'")


def main():
    if len(sys.argv) == 3:
        stock_symbol = sys.argv[1]
        database_filepath = sys.argv[2]

        print(f'Loading data for stock: {stock_symbol}')
        df = load_data(stock_symbol)

        print('Cleaning data...')
        df = clean_data(df)
        
        print(f'Saving data to database: {database_filepath}')
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the stock ticker symbol as the first argument, '\
              'and the filepath of the database to save the cleaned data to as the second argument. \n\n'\
              'Example: python process_data.py NVDA StockData.db')

if __name__ == '__main__':
    main()
