from flask import Flask, render_template, request
import os
import pandas as pd
from data.process_data import load_data, clean_data, save_data
from models.train_classifier import load_data as load_db_data, build_model, evaluate_model, save_model
from sklearn.model_selection import train_test_split
import pickle

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/predict', methods=['POST'])
def predict():
    stock_ticker = request.form['ticker']
    
    database_filepath = os.path.join('data', f'{stock_ticker}_StockData.db')
    model_filepath = os.path.join('models', f'{stock_ticker}_model.pkl')

    # Check if model exists
    if os.path.exists(model_filepath):
        print(f"Model already exists for {stock_ticker}. Loading model...")
        with open(model_filepath, 'rb') as file:
            model = pickle.load(file)
    else:
        print(f"Model does not exist for {stock_ticker}. Training a new model...")
        
        # Check if data exists
        if not os.path.exists(database_filepath):
            print(f"Fetching data for {stock_ticker}...")
            df = load_data(stock_ticker)
            df = clean_data(df)
            save_data(df, database_filepath)
        else:
            print(f"Database already exists for {stock_ticker}. Loading data...")

        print("Loading data for training...")
        X, y = load_db_data(database_filepath)

        # Drop non-numeric columns before training
        non_numeric_columns = ['Date'] if 'Date' in X.columns else []
        X = X.drop(columns=non_numeric_columns)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print("Building model...")
        model = build_model()

        print("Training model...")
        model.fit(X_train, y_train)

        print("Evaluating model...")
        evaluate_model(model, X_test, y_test)

        print("Saving model...")
        save_model(model, model_filepath)

    # Load data from the database for predictions
    table_name = f"{os.path.basename(database_filepath).replace('.db', '')}_table"
    df = pd.read_sql_table(table_name, f'sqlite:///{database_filepath}')

    # Ensure the index is a DatetimeIndex
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])  # Convert Date column to datetime
        df.set_index('Date', inplace=True)      # Set Date column as the index
        print("Converted 'Date' column to DatetimeIndex and set it as index.")
    else:
        raise ValueError("No 'Date' column found, and index is not datetime.")

    # Prepare the feature matrix for prediction
    non_numeric_columns = ['Date'] if 'Date' in df.columns else []
    X = df.drop(columns=['Tomorrow'] + non_numeric_columns)

    # Debugging: Print X to ensure it's numeric
    print("Feature matrix (X) for prediction:")
    print(X.head())

    # Predict prices for all dates
    predicted_prices = model.predict(X).tolist()

    # Predict the price for tomorrow
    tomorrow_prediction = model.predict([X.iloc[-1]])[0]

    # Prepare data for Plotly
    dates = df.index.strftime('%Y-%m-%d').tolist()
    actual_prices = df['Close'].tolist()

    return render_template(
        'go.html',
        ticker=stock_ticker,
        prediction=round(tomorrow_prediction, 2),
        dates=dates,
        actual_prices=actual_prices,
        predicted_prices=predicted_prices
    )

if __name__ == '__main__':
    app.run(debug=True)
