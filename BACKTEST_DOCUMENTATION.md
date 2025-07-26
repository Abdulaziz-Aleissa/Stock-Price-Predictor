# Backtest Feature Documentation

## Overview

The backtest feature provides comprehensive historical performance analysis for stock price predictions, enabling users to evaluate model accuracy and reliability over time.

## Features

### 1. Historical Prediction Tracking
- Automatically stores each prediction made by users
- Tracks prediction date, target date, predicted price, and current price
- Updates predictions with actual prices when available

### 2. Comprehensive Metrics
- **Mean Absolute Error (MAE)**: Average prediction error in dollars
- **Root Mean Square Error (RMSE)**: Weighted accuracy measure
- **Directional Accuracy**: Percentage of correct price direction predictions
- **Hit Rates**: Accuracy within $2, $5, and $10 price ranges
- **Rolling Performance**: 7-day, 30-day, and 90-day rolling metrics

### 3. Interactive Visualizations
- Historical predictions vs actual prices chart
- Prediction error distribution histogram
- Responsive design with theme support

## Database Schema

### PredictionHistory Table
```sql
CREATE TABLE prediction_history (
    id INTEGER PRIMARY KEY,
    stock_symbol VARCHAR(10) NOT NULL,
    prediction_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    target_date DATETIME NOT NULL,
    predicted_price FLOAT NOT NULL,
    current_price FLOAT NOT NULL,
    actual_price FLOAT,
    price_change_pct FLOAT,
    actual_change_pct FLOAT,
    model_accuracy FLOAT,
    mae FLOAT,
    rmse FLOAT,
    prediction_error FLOAT,
    direction_correct BOOLEAN,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

## API Endpoints

### Store Prediction
```python
def store_prediction(stock_symbol, predicted_price, current_price, price_change_pct, metrics):
    """Store prediction in database for future backtesting"""
```

### Update Historical Predictions
```python
def update_historical_predictions():
    """Update historical predictions with actual prices"""
```

### Calculate Backtest Metrics
```python
def calculate_backtest_metrics(stock_symbol, days_back=30):
    """Calculate comprehensive backtest metrics for a stock"""
```

## Frontend Components

### Backtest Results Section
```html
<div class="metrics-card">
    <h4>📊 Backtest Results</h4>
    <!-- Performance metrics cards -->
    <!-- Hit rate visualization -->
    <!-- Rolling performance metrics -->
    <!-- Interactive charts -->
</div>
```

### CSS Classes
- `.backtest-metric-card`: Main metric display cards
- `.hit-rate-card`: Accuracy threshold cards
- `.rolling-metric-card`: Time-period performance cards
- `.backtest-chart`: Chart container styling

### JavaScript Functions
- `createBacktestCharts()`: Generates interactive Plotly charts
- Theme-aware chart updates
- Responsive chart resizing

## Usage

### For Users
1. Make stock predictions through the normal interface
2. Return to view prediction results after the target date
3. Review backtest metrics in the dedicated section
4. Analyze charts to understand model performance patterns

### For Developers
1. Historical predictions are automatically stored
2. Scheduled updates run hourly to fetch actual prices
3. Metrics are calculated on-demand when viewing results
4. Charts are generated client-side using Plotly.js

## Performance Considerations

- Database queries are optimized with proper indexing
- Metrics calculation is cached for frequently accessed data
- Charts use responsive design for mobile compatibility
- Scheduled updates minimize API calls to stock price services

## Error Handling

- Graceful handling of missing historical data
- Fallback displays for stocks with no prediction history
- Robust error handling for API failures
- User-friendly messages for various error states

## Future Enhancements

- Export backtest results to CSV/PDF
- Comparison of multiple stocks' prediction accuracy
- Advanced statistical significance tests
- Confidence intervals for predictions
- Machine learning model performance comparison