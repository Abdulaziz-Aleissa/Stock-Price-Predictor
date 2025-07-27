#!/usr/bin/env python3
"""
Test script for the enhanced ensemble prediction system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from app.models.ensemble_predictor import AdvancedEnsemblePredictor
from app.features.technical_indicators import AdvancedTechnicalIndicators
from app.utils.risk_manager import RiskManager
from app.utils.uncertainty_quantifier import UncertaintyQuantifier

def test_ensemble_system():
    """Test the ensemble prediction system with synthetic data"""
    print("Testing Enhanced Ensemble Prediction System...")
    
    # Create synthetic stock data
    np.random.seed(42)
    n_samples = 1000
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    
    # Generate realistic stock price data
    price = 100
    prices = []
    volumes = []
    
    for i in range(n_samples):
        # Random walk with slight upward bias
        change = np.random.normal(0.001, 0.02)  # 0.1% daily return, 2% volatility
        price = price * (1 + change)
        prices.append(price)
        volumes.append(np.random.randint(100000, 1000000))
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Open': prices,
        'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'Close': prices,
        'Volume': volumes
    })
    
    print(f"Created synthetic data with {len(df)} samples")
    
    # Test technical indicators
    try:
        print("Testing technical indicators...")
        df_with_indicators = AdvancedTechnicalIndicators.calculate_all_indicators(df)
        print(f"Added {len(df_with_indicators.columns) - len(df.columns)} technical indicators")
    except Exception as e:
        print(f"Error in technical indicators: {str(e)}")
        return False
    
    # Prepare data for modeling
    df_with_indicators['Tomorrow'] = df_with_indicators['Close'].shift(-1)
    df_model = df_with_indicators.dropna()
    
    X = df_model.drop(columns=['Tomorrow', 'Date'])
    y = df_model['Tomorrow']
    
    # Test ensemble predictor
    try:
        print("Testing enhanced ensemble predictor...")
        ensemble = AdvancedEnsemblePredictor(random_state=42, use_meta_learner=True)
        
        # Train on first 80% of data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"Training on {len(X_train)} samples...")
        ensemble.fit(X_train, y_train)
        
        print("Making predictions with uncertainty...")
        predictions, uncertainties = ensemble.predict_with_uncertainty(X_test)
        
        print(f"Enhanced model trained with {len(ensemble.models)} base models")
        print(f"Meta-learner enabled: {ensemble.use_meta_learner}")
        print(f"Model weights: {ensemble.model_weights}")
        
        # Show some example predictions
        for i in range(min(5, len(predictions))):
            actual = y_test.iloc[i]
            pred = predictions[i]
            std = uncertainties[i]
            error = abs(actual - pred)
            print(f"Actual: ${actual:.2f}, Predicted: ${pred:.2f} ± ${std:.2f}, Error: ${error:.2f}")
        
    except Exception as e:
        print(f"Error in enhanced ensemble predictor: {str(e)}")
        return False
    
    # Test risk manager
    try:
        print("\nTesting risk manager...")
        risk_manager = RiskManager()
        
        current_price = prices[-1]
        prediction = predictions[0] if len(predictions) > 0 else current_price * 1.01
        prediction_std = uncertainties[0] if len(uncertainties) > 0 else current_price * 0.02
        
        # Calculate position sizing
        position_info = risk_manager.calculate_position_size(
            portfolio_value=100000,
            entry_price=current_price,
            stop_loss_price=current_price * 0.95,  # 5% stop loss
            prediction_confidence=0.8
        )
        
        print(f"Position sizing recommendation: {position_info['shares']} shares")
        print(f"Position value: ${position_info['position_value']:.2f}")
        print(f"Max loss: ${position_info['max_loss']:.2f}")
        
    except Exception as e:
        print(f"Error in risk manager: {str(e)}")
        return False
    
    # Test uncertainty quantifier
    try:
        print("\nTesting uncertainty quantifier...")
        uncertainty_quantifier = UncertaintyQuantifier(n_simulations=100)  # Reduced for speed
        
        # Calculate returns and volatility
        returns = pd.Series(prices).pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized
        expected_return = (prediction - current_price) / current_price
        
        # Monte Carlo simulation
        mc_results = uncertainty_quantifier.monte_carlo_price_simulation(
            current_price=current_price,
            predicted_return=expected_return,
            volatility=volatility,
            time_horizon=1
        )
        
        print(f"Monte Carlo results:")
        print(f"  Mean price: ${mc_results['statistics']['mean']:.2f}")
        print(f"  90% CI: ${mc_results['confidence_intervals']['90%']['lower']:.2f} - ${mc_results['confidence_intervals']['90%']['upper']:.2f}")
        print(f"  Probability of positive return: {mc_results['probabilities']['positive_return']:.1%}")
        
    except Exception as e:
        print(f"Error in uncertainty quantifier: {str(e)}")
        return False
    
    print("\n✅ All tests passed! Enhanced ensemble system is working correctly.")
    return True

if __name__ == "__main__":
    success = test_ensemble_system()
    sys.exit(0 if success else 1)