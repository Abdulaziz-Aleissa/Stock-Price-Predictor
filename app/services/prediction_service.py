"""Prediction service for ML model operations."""

import os
import joblib
import logging
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ..core.exceptions import ModelError, ModelNotFoundError, PredictionError
from ..core.constants import MODEL_CONFIG, DEFAULTS
from ..data.processors.stock_processor import StockProcessor


logger = logging.getLogger(__name__)


class PredictionService:
    """Service class for stock price prediction operations."""
    
    def __init__(self, model_dir: str = None, data_dir: str = None):
        """Initialize the prediction service."""
        self.model_dir = model_dir or DEFAULTS.get('MODEL_CACHE_DIR', 'models')
        self.data_dir = data_dir or DEFAULTS.get('DATA_CACHE_DIR', 'data')
        self.stock_processor = StockProcessor()
        
        # Ensure directories exist
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
    
    def get_model_path(self, stock_symbol: str) -> str:
        """Get the file path for a stock model."""
        return os.path.join(self.model_dir, f'{stock_symbol.upper()}_model.pkl')
    
    def get_data_path(self, stock_symbol: str) -> str:
        """Get the file path for stock data."""
        return os.path.join(self.data_dir, f'{stock_symbol.upper()}_StockData.db')
    
    def model_exists(self, stock_symbol: str) -> bool:
        """Check if a trained model exists for the stock."""
        return os.path.exists(self.get_model_path(stock_symbol))
    
    def load_model(self, stock_symbol: str) -> Dict[str, Any]:
        """Load a trained model and its metadata."""
        model_path = self.get_model_path(stock_symbol)
        
        if not os.path.exists(model_path):
            raise ModelNotFoundError(f"Model not found for {stock_symbol}")
        
        try:
            model_data = joblib.load(model_path)
            logger.info(f"Loaded existing model for {stock_symbol}")
            return model_data
        except Exception as e:
            logger.error(f"Error loading model for {stock_symbol}: {str(e)}")
            raise ModelError(f"Failed to load model for {stock_symbol}")
    
    def save_model(self, stock_symbol: str, model_data: Dict[str, Any]) -> None:
        """Save a trained model and its metadata."""
        model_path = self.get_model_path(stock_symbol)
        
        try:
            joblib.dump(model_data, model_path)
            logger.info(f"Saved model for {stock_symbol} to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model for {stock_symbol}: {str(e)}")
            raise ModelError(f"Failed to save model for {stock_symbol}")
    
    def build_model(self, model_type: str = 'gradient_boosting') -> Any:
        """Build and configure a machine learning model."""
        config = MODEL_CONFIG.get(model_type.upper(), MODEL_CONFIG['GRADIENT_BOOSTING'])
        
        if model_type.lower() == 'gradient_boosting':
            return GradientBoostingRegressor(**config)
        else:
            # Default to gradient boosting
            return GradientBoostingRegressor(**MODEL_CONFIG['GRADIENT_BOOSTING'])
    
    def evaluate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        try:
            y_pred = model.predict(X_test)
            
            metrics = {
                'mae': mean_absolute_error(y_test, y_pred),
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred)
            }
            
            logger.info("Model Performance Metrics:")
            logger.info(f"MAE: ${metrics['mae']:.2f}")
            logger.info(f"RMSE: ${metrics['rmse']:.2f}")
            logger.info(f"R²: {metrics['r2']:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise ModelError("Failed to evaluate model")
    
    def train_model(self, stock_symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Train a new model for the stock."""
        try:
            logger.info(f"Training new model for {stock_symbol}")
            
            # Prepare data
            processed_df = self.stock_processor.process_for_training(df)
            
            # Extract features and target
            X = processed_df.drop(columns=['Tomorrow'])
            if 'Date' in X.columns:
                X = X.drop(columns=['Date'])
            
            y = processed_df['Tomorrow']
            
            # Handle NaN values
            if X.isna().any().any():
                X = X.fillna(0)
                logger.info("Filled NaN values in features")
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            # Build and train model
            model = self.build_model()
            model.fit(X_train, y_train)
            
            # Evaluate model
            metrics = self.evaluate_model(model, X_test, y_test)
            
            # Prepare model data
            model_data = {
                'model': model,
                'scaler': scaler,
                'metrics': metrics,
                'feature_names': X.columns.tolist(),
                'trained_at': datetime.now().isoformat(),
                'data_shape': X.shape
            }
            
            # Save model
            self.save_model(stock_symbol, model_data)
            
            logger.info(f"Model training completed for {stock_symbol}")
            return model_data
            
        except Exception as e:
            logger.error(f"Error training model for {stock_symbol}: {str(e)}")
            raise ModelError(f"Failed to train model for {stock_symbol}")
    
    def predict_price(self, stock_symbol: str, df: pd.DataFrame, 
                     current_price: float) -> Dict[str, Any]:
        """Generate price prediction for a stock."""
        try:
            # Load or train model
            if self.model_exists(stock_symbol):
                model_data = self.load_model(stock_symbol)
            else:
                model_data = self.train_model(stock_symbol, df)
            
            model = model_data['model']
            scaler = model_data['scaler']
            metrics = model_data.get('metrics', {})
            
            # Prepare features for prediction
            processed_df = self.stock_processor.process_for_prediction(df, current_price)
            
            X = processed_df.drop(columns=['Tomorrow'])
            dates = processed_df['Date'].dt.strftime('%Y-%m-%d %H:%M').tolist()
            
            if 'Date' in X.columns:
                X = X.drop(columns=['Date'])
            
            # Handle NaN values
            if X.isna().any().any():
                X = X.fillna(0)
                logger.info("Filled NaN values in prediction features")
            
            # Scale features and predict
            X_scaled = scaler.transform(X)
            predicted_prices = model.predict(X_scaled).tolist()
            tomorrow_prediction = predicted_prices[-1]
            
            # Calculate prediction metrics
            price_change = tomorrow_prediction - current_price
            price_change_pct = (price_change / current_price) * 100
            
            # Add tomorrow's date
            tomorrow = datetime.now() + timedelta(days=1)
            tomorrow_date = tomorrow.strftime('%Y-%m-%d')
            dates.append(tomorrow_date)
            
            # Prepare actual prices
            actual_prices = processed_df['Close'].tolist()
            actual_prices.append(None)  # No actual price for tomorrow
            predicted_prices.append(tomorrow_prediction)
            
            prediction_result = {
                'symbol': stock_symbol.upper(),
                'current_price': current_price,
                'predicted_price': tomorrow_prediction,
                'price_change': price_change,
                'price_change_percent': price_change_pct,
                'prediction_date': tomorrow_date,
                'confidence_metrics': {
                    'r2_score': metrics.get('r2', 0),
                    'mae': metrics.get('mae', 0),
                    'rmse': metrics.get('rmse', 0)
                },
                'chart_data': {
                    'dates': dates,
                    'actual_prices': actual_prices,
                    'predicted_prices': predicted_prices
                },
                'model_info': {
                    'trained_at': model_data.get('trained_at'),
                    'data_points': model_data.get('data_shape', [0, 0])[0]
                }
            }
            
            logger.info(f"Prediction completed for {stock_symbol}: ${tomorrow_prediction:.2f} ({price_change_pct:+.2f}%)")
            return prediction_result
            
        except Exception as e:
            logger.error(f"Error predicting price for {stock_symbol}: {str(e)}")
            raise PredictionError(f"Failed to generate prediction for {stock_symbol}")
    
    def retrain_model(self, stock_symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Force retrain a model with new data."""
        logger.info(f"Retraining model for {stock_symbol}")
        return self.train_model(stock_symbol, df)
    
    def get_model_info(self, stock_symbol: str) -> Optional[Dict[str, Any]]:
        """Get information about a trained model."""
        if not self.model_exists(stock_symbol):
            return None
        
        try:
            model_data = self.load_model(stock_symbol)
            return {
                'symbol': stock_symbol.upper(),
                'metrics': model_data.get('metrics', {}),
                'trained_at': model_data.get('trained_at'),
                'feature_count': len(model_data.get('feature_names', [])),
                'data_points': model_data.get('data_shape', [0, 0])[0]
            }
        except Exception as e:
            logger.error(f"Error getting model info for {stock_symbol}: {str(e)}")
            return None