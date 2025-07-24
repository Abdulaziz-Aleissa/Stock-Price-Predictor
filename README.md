# 📈 Stock Price Predictor & Portfolio Dashboard

A full-featured Flask web app that predicts the **next-day stock price** and helps users **track portfolios, set alerts, manage a watchlist**, and **compare stocks interactively**.

Built with real-time data (via `yfinance`), a trained ML model (GradientBoostingRegressor), and interactive charts (Plotly), this app blends predictive analytics with practical portfolio tools — all inside a sleek, dark-light-themed UI.

---

## 🌟 Features

- 🔮 **Stock Price Prediction** — Enter any ticker (e.g. TSLA) to get tomorrow’s price prediction and market overview.
- 📊 **Interactive Dashboard** — Track your holdings with profit/loss insights, current prices, and watchlist items.
- 📈 **Stock Comparison Tool** — Visualize two stocks side-by-side over time.
- ⏰ **Price Alerts** — Get notified when a stock crosses your target price.
- ✅ **User Authentication** — Sign up and log in securely to access portfolio features.
- 🎨 **Dark-Light UI Mode** — Stylish Bootstrap-based dark-light theme switch for smooth user experience.

---

## 🖼️ Screenshots

### 🔹 Home Page – Ticker Search + Auth
<img src="screenshots/mainPage.png" width="700"/>

---

### 🔹 Stock Analysis – Prediction, Chart & Market Data
<img src="screenshots/resaultPage.png" width="700"/>

---

### 🔹 Sign Up / Create Account
<img src="screenshots/signUpPage.png" width="700"/>

---

### 🔹 Login Page
<img src="screenshots/logInPage.png" width="700"/>

---

### 🔹 Dashboard – Portfolio, Watchlist, Alerts, Comparison
<img src="screenshots/dashboardPage.png" width="700"/>

---

## 🛠️ Technologies Used

| Tool          | Purpose                              |
|---------------|--------------------------------------|
| Flask         | Backend web framework                |
| yfinance      | Fetching real-time stock data        |
| SQLAlchemy    | SQLite database interaction          |
| scikit-learn  | ML model (GradientBoostingRegressor) |
| pandas        | Data cleaning and manipulation       |
| Plotly        | Interactive graphs and comparisons   |
| Bootstrap     | Responsive dark-themed UI            |

---

## 🚀 Getting Started

### 1. Clone the repository
type the code below in your terminal to clone this repository:
```bash
git clone https://github.com/Abdulaziz-Aleissa/stock_predictor_app.git
```
### 2. Navigate to the repository
type the code below in your terminal to navigate to the repository:
```bash
cd stock_predictor_app
```

### 3. Install dependencies
type the code below in your terminal to install dependencies:
```bash
pip install -r requirements.txt
pip install --upgrade yfinance
pip install Flask yfinance sqlalchemy pandas scikit-learn plotly
```


### 4. start the application
type the code below in your terminal to start the application:
```bash
python -m app.run
```
---

# 🧠 Usage Guide

This guide walks you through how to use the Stock Price Predictor & Portfolio Dashboard app.

---

## 🔹 Step 1: Predict a Stock Price

1. Go to the home page.
2. Enter a valid stock ticker symbol (e.g., `AAPL`, `TSLA`, `MSFT`) in the input field.
3. Click the **"Predict"** button.

> The app will:
- Fetch real-time historical data from Yahoo Finance
- Process the data
- Use a trained Random Forest model to predict the **next day's price**
- Show a Plotly chart with actual vs. predicted trends
- Display key stats like current price, high/low, volume, and PE ratio

---

## 🔹 Step 2: Sign Up / Log In

To access personalized features, click **"Sign Up"** or **"Login"** from the top menu.

- Signing up allows you to:
  - Save your portfolio and watchlist
  - Set and manage price alerts
  - Access the comparison tool

---

## 🔹 Step 3: Explore the Dashboard

Once logged in, you’ll be redirected to your personal dashboard with the following tabs:

### 📁 Portfolio
- Add stocks with purchase price and quantity.
- View real-time updates on:
  - Total value
  - Profit/loss (SAR and %)
  - Price movement

### 👁 Watchlist
- Add stocks you want to monitor.
- View current price and compare with your custom target.

### 🔔 Price Alerts
- Create alerts like:
  - “Notify me when MSFT goes above $350”
  - “Alert if AAPL drops below $150”
- Alerts are triggered when market conditions are met.

### 📊 Compare Stocks
- Select two or more stocks.
- See their performance over time in one interactive Plotly chart.

### ➕ New Prediction
- Navigate back to the home page anytime to make a new prediction.

---

## 💡 Notes

- The model may take **1–3 minutes** to train depending on stock history.
- Results are stored per session; logging in saves your data permanently.
- Ensure stable internet as data is fetched from Yahoo Finance live.

---

Happy predicting! 🚀📉📈
---

# License

This project is open-source and is a gift of knowledge, created to empower and inspire others. You are free to use, modify, and share this project.
