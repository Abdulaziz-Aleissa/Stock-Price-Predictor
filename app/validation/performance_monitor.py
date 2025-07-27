"""
Performance Monitoring Module

This module implements comprehensive performance tracking and model drift detection
for the ensemble prediction system.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import sqlite3
import json
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """
    Comprehensive performance monitoring for prediction models
    """
    
    def __init__(self, db_path: str = "performance_monitoring.db"):
        """
        Initialize performance monitor
        
        Args:
            db_path: Path to SQLite database for storing performance metrics
        """
        self.db_path = db_path
        self._init_database()
        
    def _init_database(self):
        """Initialize the performance monitoring database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create performance metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    stock_symbol TEXT NOT NULL,
                    prediction_date TIMESTAMP NOT NULL,
                    target_date TIMESTAMP NOT NULL,
                    predicted_price REAL NOT NULL,
                    actual_price REAL,
                    prediction_error REAL,
                    confidence_score REAL,
                    model_version TEXT,
                    feature_count INTEGER,
                    training_samples INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create model drift table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_drift (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    stock_symbol TEXT NOT NULL,
                    drift_type TEXT NOT NULL,
                    drift_magnitude REAL NOT NULL,
                    detection_date TIMESTAMP NOT NULL,
                    drift_details TEXT,
                    action_taken TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create feature importance tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS feature_importance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    stock_symbol TEXT NOT NULL,
                    feature_name TEXT NOT NULL,
                    importance_score REAL NOT NULL,
                    importance_rank INTEGER,
                    model_version TEXT,
                    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error initializing performance database: {str(e)}")
            
    def log_prediction(self,
                      model_name: str,
                      stock_symbol: str,
                      predicted_price: float,
                      confidence_score: float,
                      model_version: str = "1.0",
                      feature_count: int = 0,
                      training_samples: int = 0) -> int:
        """
        Log a new prediction for performance tracking
        
        Returns:
            Prediction ID for later updating with actual results
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO model_performance 
                (model_name, stock_symbol, prediction_date, target_date, predicted_price, 
                 confidence_score, model_version, feature_count, training_samples)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                model_name, stock_symbol, datetime.now(), 
                datetime.now() + timedelta(days=1), predicted_price,
                confidence_score, model_version, feature_count, training_samples
            ))
            
            prediction_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            return prediction_id
            
        except Exception as e:
            logger.error(f"Error logging prediction: {str(e)}")
            return -1
    
    def update_prediction_actual(self, prediction_id: int, actual_price: float):
        """Update prediction with actual price when available"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get the predicted price first
            cursor.execute('''
                SELECT predicted_price FROM model_performance WHERE id = ?
            ''', (prediction_id,))
            
            result = cursor.fetchone()
            if result:
                predicted_price = result[0]
                prediction_error = abs(actual_price - predicted_price)
                
                cursor.execute('''
                    UPDATE model_performance 
                    SET actual_price = ?, prediction_error = ?
                    WHERE id = ?
                ''', (actual_price, prediction_error, prediction_id))
                
                conn.commit()
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error updating prediction actual: {str(e)}")
    
    def calculate_model_metrics(self,
                              model_name: str,
                              stock_symbol: str,
                              days_back: int = 30) -> Dict:
        """
        Calculate comprehensive performance metrics for a model
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Query recent predictions with actual values
            query = '''
                SELECT predicted_price, actual_price, prediction_error, confidence_score, 
                       prediction_date, target_date
                FROM model_performance 
                WHERE model_name = ? AND stock_symbol = ? 
                AND actual_price IS NOT NULL 
                AND prediction_date >= datetime('now', '-{} days')
                ORDER BY prediction_date DESC
            '''.format(days_back)
            
            df = pd.read_sql_query(query, conn, params=(model_name, stock_symbol))
            conn.close()
            
            if len(df) == 0:
                return {'error': 'No predictions with actual values found'}
            
            # Basic accuracy metrics
            mae = df['prediction_error'].mean()
            rmse = np.sqrt((df['prediction_error'] ** 2).mean())
            mape = (df['prediction_error'] / df['actual_price'] * 100).mean()
            
            # Directional accuracy
            predicted_direction = np.sign(df['predicted_price'].diff())
            actual_direction = np.sign(df['actual_price'].diff())
            directional_accuracy = (predicted_direction == actual_direction).mean()
            
            # Confidence calibration
            confidence_bins = pd.cut(df['confidence_score'], bins=5, labels=['Low', 'Med-Low', 'Medium', 'Med-High', 'High'])
            calibration_data = df.groupby(confidence_bins).agg({
                'prediction_error': ['mean', 'count']
            }).round(3)
            
            # Trend analysis
            recent_errors = df['prediction_error'].head(10).mean()
            older_errors = df['prediction_error'].tail(10).mean()
            error_trend = 'Improving' if recent_errors < older_errors else 'Degrading'
            
            # Consistency metrics
            error_consistency = 1.0 / (1.0 + df['prediction_error'].std())
            
            return {
                'sample_size': len(df),
                'time_period_days': days_back,
                'accuracy_metrics': {
                    'mae': mae,
                    'rmse': rmse,
                    'mape': mape,
                    'directional_accuracy': directional_accuracy
                },
                'confidence_calibration': calibration_data.to_dict() if not calibration_data.empty else {},
                'trend_analysis': {
                    'error_trend': error_trend,
                    'recent_mae': recent_errors,
                    'older_mae': older_errors,
                    'consistency_score': error_consistency
                },
                'performance_grade': self._calculate_performance_grade(mae, directional_accuracy, error_consistency)
            }
            
        except Exception as e:
            logger.error(f"Error calculating model metrics: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_performance_grade(self, mae: float, directional_accuracy: float, consistency: float) -> str:
        """Calculate overall performance grade"""
        # Normalize metrics (assuming reasonable ranges)
        mae_score = max(0, 1 - mae / 10)  # Assuming MAE of 10 is poor
        direction_score = directional_accuracy
        consistency_score = consistency
        
        overall_score = (mae_score + direction_score + consistency_score) / 3
        
        if overall_score >= 0.85:
            return 'A+'
        elif overall_score >= 0.80:
            return 'A'
        elif overall_score >= 0.75:
            return 'B+'
        elif overall_score >= 0.70:
            return 'B'
        elif overall_score >= 0.65:
            return 'C+'
        elif overall_score >= 0.60:
            return 'C'
        else:
            return 'D'
    
    def detect_model_drift(self,
                          model_name: str,
                          stock_symbol: str,
                          baseline_days: int = 60,
                          recent_days: int = 14,
                          drift_threshold: float = 0.15) -> Dict:
        """
        Detect if model performance has degraded (concept drift)
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get baseline performance
            baseline_query = '''
                SELECT prediction_error, confidence_score
                FROM model_performance 
                WHERE model_name = ? AND stock_symbol = ? 
                AND actual_price IS NOT NULL 
                AND prediction_date BETWEEN datetime('now', '-{} days') AND datetime('now', '-{} days')
            '''.format(baseline_days, recent_days)
            
            baseline_df = pd.read_sql_query(baseline_query, conn, params=(model_name, stock_symbol))
            
            # Get recent performance
            recent_query = '''
                SELECT prediction_error, confidence_score
                FROM model_performance 
                WHERE model_name = ? AND stock_symbol = ? 
                AND actual_price IS NOT NULL 
                AND prediction_date >= datetime('now', '-{} days')
            '''.format(recent_days)
            
            recent_df = pd.read_sql_query(recent_query, conn, params=(model_name, stock_symbol))
            conn.close()
            
            if len(baseline_df) < 5 or len(recent_df) < 3:
                return {'drift_detected': False, 'reason': 'Insufficient data for drift detection'}
            
            # Calculate performance metrics
            baseline_mae = baseline_df['prediction_error'].mean()
            recent_mae = recent_df['prediction_error'].mean()
            
            # Performance drift detection
            performance_drift = (recent_mae - baseline_mae) / baseline_mae
            performance_drift_detected = performance_drift > drift_threshold
            
            # Confidence drift detection
            baseline_confidence = baseline_df['confidence_score'].mean()
            recent_confidence = recent_df['confidence_score'].mean()
            confidence_drift = abs(recent_confidence - baseline_confidence) / baseline_confidence
            confidence_drift_detected = confidence_drift > drift_threshold
            
            # Overall drift assessment
            drift_detected = performance_drift_detected or confidence_drift_detected
            
            if drift_detected:
                # Log drift detection
                self._log_drift_detection(
                    model_name, stock_symbol, 
                    'performance' if performance_drift_detected else 'confidence',
                    max(performance_drift, confidence_drift),
                    {
                        'baseline_mae': baseline_mae,
                        'recent_mae': recent_mae,
                        'baseline_confidence': baseline_confidence,
                        'recent_confidence': recent_confidence
                    }
                )
            
            return {
                'drift_detected': drift_detected,
                'drift_details': {
                    'performance_drift': performance_drift,
                    'performance_drift_detected': performance_drift_detected,
                    'confidence_drift': confidence_drift,
                    'confidence_drift_detected': confidence_drift_detected,
                    'baseline_samples': len(baseline_df),
                    'recent_samples': len(recent_df)
                },
                'recommendation': self._get_drift_recommendation(drift_detected, performance_drift, confidence_drift)
            }
            
        except Exception as e:
            logger.error(f"Error detecting model drift: {str(e)}")
            return {'drift_detected': False, 'error': str(e)}
    
    def _log_drift_detection(self, model_name: str, stock_symbol: str, drift_type: str, 
                           drift_magnitude: float, drift_details: Dict):
        """Log drift detection event"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO model_drift 
                (model_name, stock_symbol, drift_type, drift_magnitude, detection_date, drift_details)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                model_name, stock_symbol, drift_type, drift_magnitude,
                datetime.now(), json.dumps(drift_details)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error logging drift detection: {str(e)}")
    
    def _get_drift_recommendation(self, drift_detected: bool, performance_drift: float, confidence_drift: float) -> str:
        """Get recommendation based on drift detection"""
        if not drift_detected:
            return "Model performance is stable - continue monitoring"
        
        if performance_drift > 0.3:
            return "Significant performance degradation - retrain model immediately"
        elif performance_drift > 0.15:
            return "Moderate performance drift - consider retraining within 1 week"
        elif confidence_drift > 0.2:
            return "Confidence calibration drift - review model confidence calculation"
        else:
            return "Minor drift detected - increase monitoring frequency"
    
    def log_feature_importance(self,
                             model_name: str,
                             stock_symbol: str,
                             feature_importance: Dict,
                             model_version: str = "1.0"):
        """Log feature importance for tracking feature drift"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Sort features by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            for rank, (feature_name, importance_score) in enumerate(sorted_features, 1):
                cursor.execute('''
                    INSERT INTO feature_importance 
                    (model_name, stock_symbol, feature_name, importance_score, importance_rank, model_version)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (model_name, stock_symbol, feature_name, importance_score, rank, model_version))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error logging feature importance: {str(e)}")
    
    def generate_performance_report(self,
                                  model_name: str,
                                  stock_symbol: str,
                                  days_back: int = 30) -> Dict:
        """Generate comprehensive performance report"""
        try:
            # Get model metrics
            model_metrics = self.calculate_model_metrics(model_name, stock_symbol, days_back)
            
            # Detect drift
            drift_analysis = self.detect_model_drift(model_name, stock_symbol)
            
            # Get recent feature importance trends
            feature_trends = self._analyze_feature_importance_trends(model_name, stock_symbol)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(model_metrics, drift_analysis)
            
            return {
                'model_name': model_name,
                'stock_symbol': stock_symbol,
                'report_date': datetime.now().isoformat(),
                'evaluation_period_days': days_back,
                'model_metrics': model_metrics,
                'drift_analysis': drift_analysis,
                'feature_trends': feature_trends,
                'recommendations': recommendations,
                'overall_health': self._assess_model_health(model_metrics, drift_analysis)
            }
            
        except Exception as e:
            logger.error(f"Error generating performance report: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_feature_importance_trends(self, model_name: str, stock_symbol: str) -> Dict:
        """Analyze trends in feature importance over time"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT feature_name, importance_score, importance_rank, recorded_at
                FROM feature_importance 
                WHERE model_name = ? AND stock_symbol = ?
                ORDER BY recorded_at DESC
                LIMIT 100
            '''
            
            df = pd.read_sql_query(query, conn, params=(model_name, stock_symbol))
            conn.close()
            
            if len(df) == 0:
                return {'status': 'No feature importance data available'}
            
            # Get most recent top features
            latest_features = df.head(20)  # Top 20 most recent
            top_features = latest_features.nsmallest(10, 'importance_rank')['feature_name'].tolist()
            
            # Check for feature stability
            feature_stability = {}
            for feature in top_features:
                feature_data = df[df['feature_name'] == feature].head(5)  # Last 5 recordings
                if len(feature_data) > 1:
                    rank_std = feature_data['importance_rank'].std()
                    stability_score = 1.0 / (1.0 + rank_std / 5.0)  # Normalize by rank range
                    feature_stability[feature] = stability_score
            
            return {
                'top_features': top_features,
                'feature_stability': feature_stability,
                'stable_features': [f for f, score in feature_stability.items() if score > 0.8],
                'unstable_features': [f for f, score in feature_stability.items() if score < 0.5]
            }
            
        except Exception as e:
            logger.error(f"Error analyzing feature importance trends: {str(e)}")
            return {'error': str(e)}
    
    def _generate_recommendations(self, model_metrics: Dict, drift_analysis: Dict) -> List[str]:
        """Generate actionable recommendations based on performance analysis"""
        recommendations = []
        
        if 'error' in model_metrics or 'error' in drift_analysis:
            recommendations.append("Insufficient data for comprehensive analysis - continue collecting predictions")
            return recommendations
        
        # Performance-based recommendations
        grade = model_metrics.get('performance_grade', 'D')
        if grade in ['A+', 'A']:
            recommendations.append("Excellent performance - maintain current model configuration")
        elif grade in ['B+', 'B']:
            recommendations.append("Good performance - consider minor hyperparameter tuning")
        elif grade in ['C+', 'C']:
            recommendations.append("Moderate performance - review feature engineering and model selection")
        else:
            recommendations.append("Poor performance - model requires significant improvement or replacement")
        
        # Drift-based recommendations
        if drift_analysis.get('drift_detected', False):
            recommendations.append(drift_analysis.get('recommendation', 'Address detected model drift'))
        
        # Directional accuracy recommendations
        directional_acc = model_metrics.get('accuracy_metrics', {}).get('directional_accuracy', 0)
        if directional_acc < 0.55:
            recommendations.append("Low directional accuracy - consider trend-following indicators or regime detection")
        
        return recommendations
    
    def _assess_model_health(self, model_metrics: Dict, drift_analysis: Dict) -> str:
        """Assess overall model health"""
        if 'error' in model_metrics or 'error' in drift_analysis:
            return 'Unknown'
        
        grade = model_metrics.get('performance_grade', 'D')
        drift_detected = drift_analysis.get('drift_detected', False)
        
        if grade in ['A+', 'A'] and not drift_detected:
            return 'Excellent'
        elif grade in ['B+', 'B'] and not drift_detected:
            return 'Good'
        elif grade in ['C+', 'C'] or drift_detected:
            return 'Needs Attention'
        else:
            return 'Critical'