
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(database_filepath):
    """
    Loads and prepares data from SQLite database
    """
    try:
        logger.info(f"Loading data from {database_filepath}")
        
        engine = create_engine(f'sqlite:///{database_filepath}')
        table_name = f"{os.path.basename(database_filepath).replace('.db', '')}_table"
        
        df = pd.read_sql_table(table_name, engine)
        logger.info(f"Loaded {len(df)} records from database")
        
        # Prepare features and target
        y = df['Tomorrow']
        X = df.drop(columns=['Tomorrow'])
        
        # Drop date column if exists
        if 'Date' in X.columns:
            X = X.drop(columns=['Date'])
            
        logger.info(f"Prepared features: {X.shape[1]} columns")
        return X, y
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def evaluate_model(model, X_test, y_test):
    """
    Evaluates model performance using multiple metrics
    """
    try:
        y_pred = model.predict(X_test)
        
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }
        
        logger.info("Model Performance Metrics:")
        logger.info(f"Mean Absolute Error: ${metrics['mae']:.2f}")
        logger.info(f"Root Mean Squared Error: ${metrics['rmse']:.2f}")
        logger.info(f"R-squared Score: {metrics['r2']:.4f}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        raise

def save_model(model_data, model_filepath):
    """
    Saves model and associated data
    """
    try:
        logger.info(f"Saving model to {model_filepath}")
        joblib.dump(model_data, model_filepath)
        logger.info("Model saved successfully")
        
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise

def build_model():
    """
    Creates and configures the model
    """
    return GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=42
    )

def main():
    try:
        if len(sys.argv) == 3:
            database_filepath, model_filepath = sys.argv[1:]
            
            # Load data
            logger.info(f'Loading data from database: {database_filepath}')
            X, y = load_data(database_filepath)
            
            # Scale features
            logger.info('Scaling features...')
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            # Train model
            logger.info('Training model...')
            model = build_model()
            model.fit(X_train, y_train)
            
            # Evaluate
            logger.info('Evaluating model...')
            metrics = evaluate_model(model, X_test, y_test)
            
            # Save model and metadata
            model_data = {
                'model': model,
                'scaler': scaler,
                'metrics': metrics,
                'feature_names': X.columns.tolist()
            }
            save_model(model_data, model_filepath)
            
            logger.info('Training completed successfully!')
            
        else:
            logger.error('Please provide the database filepath and model filepath as arguments')
            logger.info('Example: python train_classifier.py ../data/stock_data.db ../models/model.pkl')
            
    except Exception as e:
        logger.error(f'Error in training process: {str(e)}')
        sys.exit(1)

if __name__ == '__main__':
    main()
