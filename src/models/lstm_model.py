"""
LSTM Model for Stock Price Prediction
Implements Long Short-Term Memory neural networks for time series forecasting
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import configuration and helpers
from ..utils.config import config
from ..utils.helpers import performance_tracker

logger = logging.getLogger(__name__)

class LSTMModel:
    """LSTM model for stock price prediction"""
    
    def __init__(self, symbol: str = ""):
        self.symbol = symbol
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.sequence_length = 60  # Use 60 days of historical data
        self.trained = False
        self.feature_columns = []
        self.training_history = {}
        
        # Model parameters from config
        self.params = config.get_model_params('lstm')
        
        # Try to import required libraries
        self.tf_available = self._check_tensorflow()
    
    def _check_tensorflow(self) -> bool:
        """Check if TensorFlow is available"""
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.optimizers import Adam
            from sklearn.preprocessing import MinMaxScaler
            
            # Test if GPU is available
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                logger.info(f"TensorFlow GPU support detected: {len(gpus)} GPU(s)")
            else:
                logger.info("TensorFlow running on CPU")
            
            return True
            
        except ImportError as e:
            logger.warning(f"TensorFlow not available: {str(e)}")
            return False
    
    def _create_sequences(self, data: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        try:
            X, y = [], []
            
            for i in range(self.sequence_length, len(data)):
                X.append(data[i-self.sequence_length:i])
                y.append(target[i])
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            logger.error(f"Error creating sequences: {str(e)}")
            return np.array([]), np.array([])
    
    def _build_model(self, input_shape: Tuple[int, int]) -> Any:
        """Build LSTM model architecture"""
        try:
            if not self.tf_available:
                raise ImportError("TensorFlow not available")
            
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
            from tensorflow.keras.optimizers import Adam
            from tensorflow.keras.regularizers import l2
            
            model = Sequential()
            
            # First LSTM layer
            model.add(LSTM(
                units=self.params['units'][0],
                return_sequences=True,
                input_shape=input_shape,
                kernel_regularizer=l2(0.001)
            ))
            model.add(Dropout(self.params['dropout']))
            model.add(BatchNormalization())
            
            # Second LSTM layer
            if len(self.params['units']) > 1:
                model.add(LSTM(
                    units=self.params['units'][1],
                    return_sequences=False,
                    kernel_regularizer=l2(0.001)
                ))
                model.add(Dropout(self.params['dropout']))
                model.add(BatchNormalization())
            
            # Dense layers
            model.add(Dense(50, activation='relu', kernel_regularizer=l2(0.001)))
            model.add(Dropout(0.2))
            model.add(Dense(25, activation='relu'))
            model.add(Dense(1))
            
            # Compile model
            optimizer = Adam(learning_rate=self.params['learning_rate'])
            model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
            
            logger.info(f"LSTM model built with input shape: {input_shape}")
            return model
            
        except Exception as e:
            logger.error(f"Error building LSTM model: {str(e)}")
            return None
    
    def train(self, X: pd.DataFrame, y: pd.Series, validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Train the LSTM model
        
        Args:
            X: Feature DataFrame
            y: Target Series
            validation_split: Fraction of data to use for validation
        
        Returns:
            Dictionary with training results
        """
        try:
            if not self.tf_available:
                return {'error': 'TensorFlow not available for LSTM training'}
            
            performance_tracker.start_timer(f"lstm_train_{self.symbol}")
            
            # Import required libraries
            from sklearn.preprocessing import MinMaxScaler
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
            
            if X.empty or y.empty:
                return {'error': 'Empty training data provided'}
            
            # Store feature columns
            self.feature_columns = X.columns.tolist()
            
            # Scale features
            self.scaler_X = MinMaxScaler()
            X_scaled = self.scaler_X.fit_transform(X)
            
            # Scale target
            self.scaler_y = MinMaxScaler()
            y_scaled = self.scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()
            
            # Create sequences
            X_seq, y_seq = self._create_sequences(X_scaled, y_scaled)
            
            if len(X_seq) == 0:
                return {'error': 'Insufficient data for sequence creation'}
            
            # Split data
            split_idx = int(len(X_seq) * (1 - validation_split))
            X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
            y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
            
            # Build model
            input_shape = (X_seq.shape[1], X_seq.shape[2])
            self.model = self._build_model(input_shape)
            
            if self.model is None:
                return {'error': 'Failed to build LSTM model'}
            
            # Callbacks
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True
            )
            
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6
            )
            
            # Train model
            logger.info(f"Training LSTM model for {self.symbol}")
            history = self.model.fit(
                X_train, y_train,
                epochs=self.params['epochs'],
                batch_size=self.params['batch_size'],
                validation_data=(X_val, y_val),
                callbacks=[early_stopping, reduce_lr],
                verbose=0
            )
            
            self.trained = True
            self.training_history = history.history
            
            # Evaluate model
            train_loss = self.model.evaluate(X_train, y_train, verbose=0)
            val_loss = self.model.evaluate(X_val, y_val, verbose=0)
            
            # Make predictions for evaluation
            train_pred = self.model.predict(X_train, verbose=0)
            val_pred = self.model.predict(X_val, verbose=0)
            
            # Inverse transform predictions
            train_pred_actual = self.scaler_y.inverse_transform(train_pred).flatten()
            val_pred_actual = self.scaler_y.inverse_transform(val_pred).flatten()
            y_train_actual = self.scaler_y.inverse_transform(y_train.reshape(-1, 1)).flatten()
            y_val_actual = self.scaler_y.inverse_transform(y_val.reshape(-1, 1)).flatten()
            
            # Calculate metrics
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            
            train_mae = mean_absolute_error(y_train_actual, train_pred_actual)
            val_mae = mean_absolute_error(y_val_actual, val_pred_actual)
            train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_pred_actual))
            val_rmse = np.sqrt(mean_squared_error(y_val_actual, val_pred_actual))
            train_r2 = r2_score(y_train_actual, train_pred_actual)
            val_r2 = r2_score(y_val_actual, val_pred_actual)
            
            training_duration = performance_tracker.end_timer(f"lstm_train_{self.symbol}")
            
            results = {
                'model_type': 'LSTM',
                'training_samples': len(X_train),
                'validation_samples': len(X_val),
                'features': len(self.feature_columns),
                'sequence_length': self.sequence_length,
                'epochs_trained': len(history.history['loss']),
                'training_duration': training_duration,
                'metrics': {
                    'train_mae': train_mae,
                    'val_mae': val_mae,
                    'train_rmse': train_rmse,
                    'val_rmse': val_rmse,
                    'train_r2': train_r2,
                    'val_r2': val_r2,
                    'final_train_loss': train_loss[0] if isinstance(train_loss, list) else train_loss,
                    'final_val_loss': val_loss[0] if isinstance(val_loss, list) else val_loss
                },
                'model_params': self.params,
                'feature_columns': self.feature_columns
            }
            
            logger.info(f"LSTM training completed for {self.symbol}")
            logger.info(f"Validation MAE: ${val_mae:.2f}, RMSE: ${val_rmse:.2f}, R²: {val_r2:.4f}")
            
            return results
            
        except Exception as e:
            performance_tracker.end_timer(f"lstm_train_{self.symbol}")
            logger.error(f"Error training LSTM model: {str(e)}")
            return {'error': str(e)}
    
    def predict(self, X: pd.DataFrame, return_confidence: bool = True) -> Dict[str, Any]:
        """
        Make predictions using the trained LSTM model
        
        Args:
            X: Feature DataFrame
            return_confidence: Whether to return confidence intervals
        
        Returns:
            Dictionary with predictions and metadata
        """
        try:
            if not self.trained or self.model is None:
                return {'error': 'Model not trained'}
            
            if X.empty:
                return {'error': 'Empty input data'}
            
            performance_tracker.start_timer(f"lstm_predict_{self.symbol}")
            
            # Ensure feature columns match training data
            missing_cols = set(self.feature_columns) - set(X.columns)
            if missing_cols:
                return {'error': f'Missing feature columns: {missing_cols}'}
            
            # Reorder columns to match training data
            X_ordered = X[self.feature_columns]
            
            # Scale features using the fitted scaler
            X_scaled = self.scaler_X.transform(X_ordered)
            
            # Create sequences for prediction
            if len(X_scaled) < self.sequence_length:
                return {'error': f'Insufficient data for prediction. Need at least {self.sequence_length} samples'}
            
            # Use the last sequence_length samples for prediction
            X_seq = X_scaled[-self.sequence_length:].reshape(1, self.sequence_length, -1)
            
            # Make prediction
            pred_scaled = self.model.predict(X_seq, verbose=0)
            pred_actual = self.scaler_y.inverse_transform(pred_scaled).flatten()[0]
            
            # Monte Carlo Dropout for confidence intervals
            confidence_intervals = None
            if return_confidence:
                confidence_intervals = self._calculate_confidence_intervals(X_seq)
            
            prediction_duration = performance_tracker.end_timer(f"lstm_predict_{self.symbol}")
            
            results = {
                'model_type': 'LSTM',
                'prediction': float(pred_actual),
                'confidence_intervals': confidence_intervals,
                'input_samples': len(X),
                'sequence_length_used': self.sequence_length,
                'prediction_duration': prediction_duration,
                'model_params': self.params
            }
            
            logger.info(f"LSTM prediction completed for {self.symbol}: ${pred_actual:.2f}")
            
            return results
            
        except Exception as e:
            performance_tracker.end_timer(f"lstm_predict_{self.symbol}")
            logger.error(f"Error making LSTM prediction: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_confidence_intervals(self, X_seq: np.ndarray, n_samples: int = 100) -> Optional[Dict[str, float]]:
        """Calculate confidence intervals using Monte Carlo Dropout"""
        try:
            if not self.tf_available:
                return None
            
            # Enable dropout during inference
            import tensorflow as tf
            
            # Function to enable dropout during inference
            def enable_dropout(model):
                for layer in model.layers:
                    if hasattr(layer, 'training'):
                        layer.training = True
            
            predictions = []
            
            for _ in range(n_samples):
                enable_dropout(self.model)
                pred_scaled = self.model(X_seq, training=True)
                pred_actual = self.scaler_y.inverse_transform(pred_scaled.numpy()).flatten()[0]
                predictions.append(pred_actual)
            
            predictions = np.array(predictions)
            
            return {
                'mean': float(np.mean(predictions)),
                'std': float(np.std(predictions)),
                'lower_95': float(np.percentile(predictions, 2.5)),
                'upper_95': float(np.percentile(predictions, 97.5)),
                'lower_68': float(np.percentile(predictions, 16)),
                'upper_68': float(np.percentile(predictions, 84))
            }
            
        except Exception as e:
            logger.error(f"Error calculating confidence intervals: {str(e)}")
            return None
    
    def predict_multiple_horizons(self, X: pd.DataFrame, horizons: List[int]) -> Dict[int, Dict[str, Any]]:
        """
        Predict multiple time horizons
        
        Args:
            X: Feature DataFrame
            horizons: List of prediction horizons (days ahead)
        
        Returns:
            Dictionary mapping horizons to predictions
        """
        try:
            if not self.trained:
                return {h: {'error': 'Model not trained'} for h in horizons}
            
            results = {}
            
            for horizon in horizons:
                try:
                    # For multi-step prediction, we'll use recursive approach
                    current_X = X.copy()
                    predictions = []
                    
                    for step in range(horizon):
                        pred_result = self.predict(current_X, return_confidence=False)
                        
                        if 'error' in pred_result:
                            results[horizon] = pred_result
                            break
                        
                        prediction = pred_result['prediction']
                        predictions.append(prediction)
                        
                        # Update the input for next step (simplified approach)
                        # In a real implementation, we would properly update all features
                        if len(current_X) > 0:
                            # Add the prediction as the next day's data point
                            # This is a simplified approach - in practice, we'd need to 
                            # properly forecast all features
                            break  # For now, just predict one step
                    
                    if horizon not in results:
                        # For now, just return single-step prediction for all horizons
                        single_pred = self.predict(X, return_confidence=True)
                        results[horizon] = single_pred
                        
                        # Adjust confidence intervals for longer horizons
                        if 'confidence_intervals' in single_pred and single_pred['confidence_intervals']:
                            ci = single_pred['confidence_intervals']
                            # Increase uncertainty for longer horizons
                            uncertainty_multiplier = np.sqrt(horizon)
                            
                            results[horizon]['confidence_intervals'] = {
                                'mean': ci['mean'],
                                'std': ci['std'] * uncertainty_multiplier,
                                'lower_95': ci['mean'] - (ci['mean'] - ci['lower_95']) * uncertainty_multiplier,
                                'upper_95': ci['mean'] + (ci['upper_95'] - ci['mean']) * uncertainty_multiplier,
                                'lower_68': ci['mean'] - (ci['mean'] - ci['lower_68']) * uncertainty_multiplier,
                                'upper_68': ci['mean'] + (ci['upper_68'] - ci['mean']) * uncertainty_multiplier
                            }
                        
                        results[horizon]['horizon'] = horizon
                        
                except Exception as e:
                    results[horizon] = {'error': f'Error predicting horizon {horizon}: {str(e)}'}
            
            return results
            
        except Exception as e:
            logger.error(f"Error predicting multiple horizons: {str(e)}")
            return {h: {'error': str(e)} for h in horizons}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model"""
        try:
            info = {
                'model_type': 'LSTM',
                'symbol': self.symbol,
                'trained': self.trained,
                'tensorflow_available': self.tf_available,
                'sequence_length': self.sequence_length,
                'feature_count': len(self.feature_columns),
                'feature_columns': self.feature_columns,
                'model_params': self.params
            }
            
            if self.model and self.tf_available:
                try:
                    info['model_summary'] = {
                        'total_params': self.model.count_params(),
                        'trainable_params': sum([np.prod(p.shape) for p in self.model.trainable_weights]),
                        'layers': len(self.model.layers)
                    }
                except:
                    pass
            
            if self.training_history:
                info['training_history'] = {
                    'epochs': len(self.training_history.get('loss', [])),
                    'final_train_loss': self.training_history.get('loss', [])[-1] if self.training_history.get('loss') else None,
                    'final_val_loss': self.training_history.get('val_loss', [])[-1] if self.training_history.get('val_loss') else None
                }
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return {'error': str(e)}
    
    def save_model(self, filepath: str) -> bool:
        """Save the trained model"""
        try:
            if not self.trained or not self.tf_available:
                return False
            
            import pickle
            
            # Save model weights and scalers
            model_data = {
                'model_weights': self.model.get_weights() if self.model else None,
                'model_config': self.model.get_config() if self.model else None,
                'scaler_X': self.scaler_X,
                'scaler_y': self.scaler_y,
                'sequence_length': self.sequence_length,
                'feature_columns': self.feature_columns,
                'params': self.params,
                'training_history': self.training_history,
                'symbol': self.symbol
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"LSTM model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving LSTM model: {str(e)}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load a trained model"""
        try:
            if not self.tf_available:
                return False
            
            import pickle
            from tensorflow.keras.models import model_from_config
            
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            # Restore model
            if model_data.get('model_config') and model_data.get('model_weights'):
                self.model = model_from_config(model_data['model_config'])
                self.model.set_weights(model_data['model_weights'])
                
                # Recompile model
                from tensorflow.keras.optimizers import Adam
                optimizer = Adam(learning_rate=self.params['learning_rate'])
                self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
            
            # Restore other attributes
            self.scaler_X = model_data.get('scaler_X')
            self.scaler_y = model_data.get('scaler_y')
            self.sequence_length = model_data.get('sequence_length', 60)
            self.feature_columns = model_data.get('feature_columns', [])
            self.params = model_data.get('params', self.params)
            self.training_history = model_data.get('training_history', {})
            self.symbol = model_data.get('symbol', self.symbol)
            
            self.trained = True
            
            logger.info(f"LSTM model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading LSTM model: {str(e)}")
            return False