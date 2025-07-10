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
<img src="Screenshots/mainPage.png" width="700"/>

---

### 🔹 Stock Analysis – Prediction, Chart & Market Data
<img src="Screenshots/resaultPage.png" width="700"/>

---

### 🔹 Loading / Processing Page
<img src="Screenshots/processing-pag.png" width="700"/>

---

### 🔹 Sign Up / Create Account
<img src="Screenshots/signUpPage.png" width="700"/>

---

### 🔹 Login Page
<img src="Screenshots/logInPage.png" width="700"/>

---

### 🔹 Dashboard – Portfolio, Watchlist, Alerts, Comparison
<img src="Screenshots/dashboardPage.png" width="700"/>

---

## 🛠️ Technologies Used

| Tool          | Purpose                              |
|---------------|--------------------------------------|
| Flask         | Backend web framework                |
| yfinance      | Fetching real-time stock data        |
| SQLAlchemy    | SQLite database interaction          |
| scikit-learn  | ML model (Random Forest Regressor)   |
| pandas        | Data cleaning and manipulation       |
| Plotly        | Interactive graphs and comparisons   |
| Bootstrap     | Responsive dark-themed UI            |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Abdulaziz-Aleissa/stock_predictor_app.git
cd stock_predictor_app
