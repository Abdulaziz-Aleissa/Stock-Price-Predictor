import yfinance as yf
from sqlalchemy import create_engine
import pandas as pd
import sys

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
    Prepares stock data by adding a column that indicates if the next day's close price
    is higher than the current day's close price and removes any rows with missing values.

    Parameters:
    df (pd.DataFrame): The DataFrame containing historical stock data.

    Returns:
    pd.DataFrame: A DataFrame with an added 'Tomorrow' column for the next day's close price,
                  a 'Target' column for indicating price increase (1) or not (0), and no missing values.
    """
    df["Tomorrow"] = df['Close'].shift(-1)
    df["Target"] = (df['Tomorrow'] > df['Close']).astype(int)
    df = df.dropna()
    return df

def save_data(df, database_filepath):
    """
    Saves the DataFrame to an SQLite database. The table name is generated based on
    the database filename without its file extension.

    Parameters:
    df (pd.DataFrame): The DataFrame to be saved.
    database_filepath (str): The file path for the SQLite database where data will be saved.

    Returns:
    None
    """
    # Create the SQLite engine to connect to the database
    engine = create_engine(f'sqlite:///{database_filepath}')
    
    # Automatically generate a table name based on the database filename
    table_name = f"{database_filepath.split('/')[-1].replace('.db', '')}_table"
    
    # Write the DataFrame to the specified SQLite table
    df.to_sql(table_name, engine, index=False, if_exists='replace') 


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
