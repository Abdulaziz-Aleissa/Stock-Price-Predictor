from flask import Flask, render_template, request
import os
import pandas as pd
import json
from data.process_data import load_data, clean_data, save_data
from models.train_classifier import load_data as load_db_data, build_model, evaluate_model, save_model
from sklearn.model_selection import train_test_split

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/predict', methods=['POST'])
def predict():
    stock_ticker = request.form['ticker']
    
    database_filepath = os.path.join('data', f'{stock_ticker}_StockData.db')
    model_filepath = os.path.join('models', f'{stock_ticker}_model.pkl')

    print(f"Loading data for {stock_ticker}")
    df = load_data(stock_ticker)
    df = clean_data(df)
    save_data(df, database_filepath)

    print("Training model...")
    X, y = load_db_data(database_filepath)
    model = build_model()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model, model_filepath)

    # Predict the price for tomorrow
    tomorrow_prediction = model.predict([X.iloc[-1]])[0]
    
    # Prepare data for Plotly chart
    dates = df.index.strftime('%Y-%m-%d').tolist()
    actual_prices = df['Close'].tolist()
    predicted_prices = model.predict(X).tolist()  # Predicted prices for all dates

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
