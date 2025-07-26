"""
Enhanced Stock Price Predictor - Main Application
Integrates all components for comprehensive stock analysis and prediction
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import all modules
from src.data.data_collector import data_collector
from src.data.data_processor import data_processor
from src.data.sentiment_analyzer import sentiment_analyzer
from src.models.model_evaluator import ModelEvaluator
from src.analysis.portfolio_optimizer import portfolio_optimizer
from src.utils.config import config
from src.utils.helpers import setup_logging, performance_tracker

# Setup logging
logger = logging.getLogger(__name__)

class StockPricePredictor:
    """Main application class for enhanced stock price prediction"""
    
    def __init__(self):
        self.version = "2.0.0"
        self.initialized = False
        
        # Initialize logging
        setup_logging(level='INFO')
        
        # Component instances
        self.data_collector = data_collector
        self.data_processor = data_processor
        self.sentiment_analyzer = sentiment_analyzer
        self.portfolio_optimizer = portfolio_optimizer
        
        # Model evaluators for different symbols
        self.model_evaluators = {}
        
        logger.info(f"Stock Price Predictor v{self.version} initialized")
        
    def analyze_stock(self, symbol: str, prediction_horizons: List[int] = [1, 7, 30],
                     include_sentiment: bool = True, include_technicals: bool = True,
                     training_period: str = "2y") -> Dict[str, Any]:
        """
        Comprehensive stock analysis with predictions
        
        Args:
            symbol: Stock ticker symbol
            prediction_horizons: List of prediction horizons in days
            include_sentiment: Whether to include sentiment analysis
            include_technicals: Whether to include technical indicators
            training_period: Training data period
        
        Returns:
            Dictionary with comprehensive analysis results
        """
        try:
            performance_tracker.start_timer(f"analyze_stock_{symbol}")
            
            logger.info(f"Starting comprehensive analysis for {symbol}")
            
            # Step 1: Data Collection
            logger.info("Step 1: Collecting stock data...")
            stock_data = self.data_collector.get_stock_data(symbol, period=training_period)
            
            if stock_data.empty:
                return {'error': f'No data available for {symbol}'}
            
            # Step 2: Data Processing and Feature Engineering
            logger.info("Step 2: Processing data and engineering features...")
            processed_data = self.data_processor.process_stock_data(
                stock_data, 
                symbol,
                add_technical_indicators=include_technicals,
                add_volume_analysis=True,
                add_momentum_indicators=True
            )
            
            if processed_data.empty:
                return {'error': f'Data processing failed for {symbol}'}
            
            # Step 3: Prepare data for ML models
            logger.info("Step 3: Preparing features for machine learning...")
            features_df, target_series = self.data_processor.prepare_features_for_ml(
                processed_data, 
                target_col='Close',
                prediction_horizon=1,
                feature_scaling='standard'
            )
            
            if features_df.empty:
                return {'error': f'Feature preparation failed for {symbol}'}
            
            # Step 4: Train and evaluate models
            logger.info("Step 4: Training and evaluating ML models...")
            model_evaluator = ModelEvaluator(symbol=symbol)
            self.model_evaluators[symbol] = model_evaluator
            
            # Split data for training and testing
            split_idx = int(len(features_df) * 0.8)
            X_train, X_test = features_df.iloc[:split_idx], features_df.iloc[split_idx:]
            y_train, y_test = target_series.iloc[:split_idx], target_series.iloc[split_idx:]
            
            # Train all models
            training_results = model_evaluator.train_all_models(X_train, y_train)
            
            # Evaluate models
            evaluation_results = model_evaluator.evaluate_models(X_test, y_test)
            
            # Step 5: Generate predictions for multiple horizons
            logger.info("Step 5: Generating multi-horizon predictions...")
            predictions = model_evaluator.predict_ensemble(features_df, prediction_horizons)
            
            # Step 6: Sentiment Analysis (if requested)
            sentiment_data = {}
            if include_sentiment:
                logger.info("Step 6: Analyzing sentiment...")
                sentiment_data = self.sentiment_analyzer.get_comprehensive_sentiment(symbol, days=7)
            
            # Step 7: Technical Analysis Summary
            technical_summary = self._create_technical_summary(processed_data)
            
            # Step 8: Risk Assessment
            risk_assessment = self._calculate_risk_metrics(stock_data, processed_data)
            
            analysis_duration = performance_tracker.end_timer(f"analyze_stock_{symbol}")
            
            # Compile comprehensive results
            results = {
                'symbol': symbol,
                'analysis_timestamp': datetime.now().isoformat(),
                'analysis_duration': analysis_duration,
                'data_summary': {
                    'total_samples': len(stock_data),
                    'training_samples': len(X_train),
                    'test_samples': len(X_test),
                    'features_count': len(features_df.columns),
                    'date_range': {
                        'start': stock_data.index[0].strftime('%Y-%m-%d'),
                        'end': stock_data.index[-1].strftime('%Y-%m-%d')
                    }
                },
                'current_price': float(stock_data['Close'].iloc[-1]),
                'predictions': predictions,
                'model_evaluation': evaluation_results,
                'technical_analysis': technical_summary,
                'risk_assessment': risk_assessment,
                'sentiment_analysis': sentiment_data if include_sentiment else None,
                'recommendation': self._generate_recommendation(predictions, sentiment_data, risk_assessment)
            }
            
            logger.info(f"Comprehensive analysis completed for {symbol}")
            
            return results
            
        except Exception as e:
            performance_tracker.end_timer(f"analyze_stock_{symbol}")
            logger.error(f"Error analyzing stock {symbol}: {str(e)}")
            return {'error': str(e)}
    
    def analyze_portfolio(self, symbols: List[str], weights: Optional[List[float]] = None,
                         optimization_method: str = 'max_sharpe', 
                         rebalancing_frequency: str = 'monthly') -> Dict[str, Any]:
        """
        Comprehensive portfolio analysis and optimization
        
        Args:
            symbols: List of stock symbols
            weights: Portfolio weights (if None, will optimize)
            optimization_method: Portfolio optimization method
            rebalancing_frequency: Rebalancing frequency
        
        Returns:
            Dictionary with portfolio analysis results
        """
        try:
            performance_tracker.start_timer("analyze_portfolio")
            
            logger.info(f"Starting portfolio analysis for {len(symbols)} assets")
            
            # Step 1: Collect data for all symbols
            logger.info("Step 1: Collecting portfolio data...")
            portfolio_data = self.data_collector.get_multiple_stocks(symbols, period="2y")
            
            if not portfolio_data:
                return {'error': 'No data available for portfolio symbols'}
            
            # Filter out symbols with no data
            valid_symbols = list(portfolio_data.keys())
            logger.info(f"Valid symbols: {valid_symbols}")
            
            # Step 2: Align data and calculate returns
            logger.info("Step 2: Processing portfolio data...")
            aligned_data = self._align_portfolio_data(portfolio_data)
            returns_data = aligned_data.pct_change().dropna()
            
            # Step 3: Portfolio Optimization
            logger.info("Step 3: Running portfolio optimization...")
            
            if weights is None:
                # Optimize portfolio
                optimization_result = self.portfolio_optimizer.optimize_portfolio(
                    returns_data, 
                    method=optimization_method,
                    risk_tolerance='moderate'
                )
                
                if 'error' in optimization_result:
                    return optimization_result
                
                optimal_weights = optimization_result['weights']
            else:
                # Use provided weights
                if len(weights) != len(valid_symbols):
                    return {'error': 'Number of weights does not match number of valid symbols'}
                
                optimal_weights = np.array(weights)
                
                # Still calculate metrics for provided weights
                optimization_result = {
                    'weights': optimal_weights,
                    'portfolio_metrics': self.portfolio_optimizer.calculate_portfolio_metrics(
                        optimal_weights, returns_data
                    ),
                    'optimization_method': 'user_provided'
                }
            
            # Step 4: Calculate efficient frontier
            logger.info("Step 4: Calculating efficient frontier...")
            efficient_frontier = self.portfolio_optimizer.calculate_efficient_frontier(returns_data)
            
            # Step 5: Diversification analysis
            logger.info("Step 5: Analyzing diversification...")
            diversification_analysis = self.portfolio_optimizer.analyze_portfolio_diversification(
                optimal_weights, returns_data
            )
            
            # Step 6: Monte Carlo simulation
            logger.info("Step 6: Running Monte Carlo simulation...")
            monte_carlo_results = self.portfolio_optimizer.monte_carlo_portfolio_simulation(
                returns_data, optimal_weights, n_simulations=1000, time_horizon=252
            )
            
            # Step 7: Individual stock predictions
            logger.info("Step 7: Generating individual stock predictions...")
            individual_predictions = {}
            
            for symbol in valid_symbols:
                try:
                    prediction = self.analyze_stock(
                        symbol, 
                        prediction_horizons=[1, 30], 
                        include_sentiment=False,
                        include_technicals=True,
                        training_period="1y"
                    )
                    
                    if 'error' not in prediction:
                        individual_predictions[symbol] = {
                            'current_price': prediction['current_price'],
                            'next_day_prediction': prediction['predictions']['ensemble_predictions'].get(1, {}).get('ensemble_prediction'),
                            'monthly_prediction': prediction['predictions']['ensemble_predictions'].get(30, {}).get('ensemble_prediction'),
                            'risk_assessment': prediction['risk_assessment']
                        }
                        
                except Exception as e:
                    logger.warning(f"Could not analyze {symbol}: {str(e)}")
                    continue
            
            # Step 8: Rebalancing analysis
            logger.info("Step 8: Analyzing rebalancing needs...")
            current_prices = pd.Series({symbol: portfolio_data[symbol]['Close'].iloc[-1] 
                                       for symbol in valid_symbols})
            
            # Assume equal weights as current weights for demonstration
            current_weights = np.array([1/len(valid_symbols)] * len(valid_symbols))
            
            rebalancing_analysis = self.portfolio_optimizer.portfolio_rebalancing_analysis(
                current_weights, optimal_weights, current_prices
            )
            
            analysis_duration = performance_tracker.end_timer("analyze_portfolio")
            
            # Compile comprehensive portfolio results
            results = {
                'portfolio_summary': {
                    'symbols': valid_symbols,
                    'total_assets': len(valid_symbols),
                    'analysis_duration': analysis_duration,
                    'optimization_method': optimization_method,
                    'rebalancing_frequency': rebalancing_frequency
                },
                'optimal_allocation': {
                    'weights': optimal_weights.tolist(),
                    'weight_mapping': dict(zip(valid_symbols, optimal_weights.tolist())),
                    'optimization_result': optimization_result
                },
                'portfolio_metrics': optimization_result.get('portfolio_metrics', {}),
                'efficient_frontier': efficient_frontier,
                'diversification_analysis': diversification_analysis,
                'monte_carlo_simulation': monte_carlo_results,
                'individual_predictions': individual_predictions,
                'rebalancing_analysis': rebalancing_analysis,
                'portfolio_recommendation': self._generate_portfolio_recommendation(
                    optimization_result, diversification_analysis, monte_carlo_results
                )
            }
            
            logger.info("Portfolio analysis completed successfully")
            
            return results
            
        except Exception as e:
            performance_tracker.end_timer("analyze_portfolio")
            logger.error(f"Error analyzing portfolio: {str(e)}")
            return {'error': str(e)}
    
    def _align_portfolio_data(self, portfolio_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Align portfolio data across different stocks"""
        try:
            # Create aligned DataFrame with close prices
            aligned_data = pd.DataFrame()
            
            for symbol, data in portfolio_data.items():
                aligned_data[symbol] = data['Close']
            
            # Forward fill missing values and drop rows with any NaN
            aligned_data = aligned_data.fillna(method='ffill').dropna()
            
            return aligned_data
            
        except Exception as e:
            logger.error(f"Error aligning portfolio data: {str(e)}")
            return pd.DataFrame()
    
    def _create_technical_summary(self, processed_data: pd.DataFrame) -> Dict[str, Any]:
        """Create technical analysis summary"""
        try:
            latest_data = processed_data.iloc[-1]
            
            # RSI analysis
            rsi = latest_data.get('RSI', 50)
            if rsi > 70:
                rsi_signal = 'Overbought'
            elif rsi < 30:
                rsi_signal = 'Oversold'
            else:
                rsi_signal = 'Neutral'
            
            # MACD analysis
            macd = latest_data.get('MACD', 0)
            macd_signal = latest_data.get('MACD_Signal', 0)
            if macd > macd_signal:
                macd_trend = 'Bullish'
            else:
                macd_trend = 'Bearish'
            
            # Moving average analysis
            price = latest_data.get('Close', 0)
            sma_20 = latest_data.get('SMA_20', price)
            sma_50 = latest_data.get('SMA_50', price)
            
            if price > sma_20 > sma_50:
                ma_trend = 'Strong Uptrend'
            elif price > sma_20:
                ma_trend = 'Uptrend'
            elif price < sma_20 < sma_50:
                ma_trend = 'Strong Downtrend'
            else:
                ma_trend = 'Downtrend'
            
            # Bollinger Bands analysis
            bb_upper = latest_data.get('BB_Upper', price * 1.02)
            bb_lower = latest_data.get('BB_Lower', price * 0.98)
            
            if price > bb_upper:
                bb_signal = 'Above Upper Band - Potential Overbought'
            elif price < bb_lower:
                bb_signal = 'Below Lower Band - Potential Oversold'
            else:
                bb_signal = 'Within Bands - Normal Range'
            
            return {
                'rsi': {
                    'value': float(rsi),
                    'signal': rsi_signal
                },
                'macd': {
                    'value': float(macd),
                    'signal_line': float(macd_signal),
                    'trend': macd_trend
                },
                'moving_averages': {
                    'sma_20': float(sma_20),
                    'sma_50': float(sma_50),
                    'trend': ma_trend
                },
                'bollinger_bands': {
                    'upper': float(bb_upper),
                    'lower': float(bb_lower),
                    'signal': bb_signal
                },
                'overall_technical_signal': self._determine_overall_technical_signal(
                    rsi_signal, macd_trend, ma_trend
                )
            }
            
        except Exception as e:
            logger.error(f"Error creating technical summary: {str(e)}")
            return {}
    
    def _calculate_risk_metrics(self, stock_data: pd.DataFrame, 
                               processed_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics"""
        try:
            returns = stock_data['Close'].pct_change().dropna()
            
            # Volatility (annualized)
            volatility = returns.std() * np.sqrt(252)
            
            # Value at Risk (95% confidence)
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)
            
            # Expected Shortfall
            es_95 = returns[returns <= var_95].mean()
            es_99 = returns[returns <= var_99].mean()
            
            # Maximum Drawdown
            cumulative_returns = (1 + returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # Beta (using a market proxy - simplified calculation)
            # In practice, you'd use a proper market index
            market_returns = returns  # Simplified
            beta = 1.0  # Placeholder
            
            # Downside deviation
            downside_returns = returns[returns < 0]
            downside_deviation = downside_returns.std() * np.sqrt(252)
            
            return {
                'volatility': float(volatility),
                'var_95': float(var_95),
                'var_99': float(var_99),
                'expected_shortfall_95': float(es_95),
                'expected_shortfall_99': float(es_99),
                'max_drawdown': float(max_drawdown),
                'beta': float(beta),
                'downside_deviation': float(downside_deviation),
                'risk_level': self._determine_risk_level(volatility, max_drawdown)
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {str(e)}")
            return {}
    
    def _determine_overall_technical_signal(self, rsi_signal: str, macd_trend: str, ma_trend: str) -> str:
        """Determine overall technical signal"""
        bullish_signals = 0
        bearish_signals = 0
        
        # RSI signals
        if rsi_signal == 'Oversold':
            bullish_signals += 1
        elif rsi_signal == 'Overbought':
            bearish_signals += 1
        
        # MACD signals
        if macd_trend == 'Bullish':
            bullish_signals += 1
        else:
            bearish_signals += 1
        
        # Moving average signals
        if 'Uptrend' in ma_trend:
            bullish_signals += 1
        else:
            bearish_signals += 1
        
        if bullish_signals > bearish_signals:
            return 'Bullish'
        elif bearish_signals > bullish_signals:
            return 'Bearish'
        else:
            return 'Neutral'
    
    def _determine_risk_level(self, volatility: float, max_drawdown: float) -> str:
        """Determine risk level based on volatility and drawdown"""
        risk_score = 0
        
        # Volatility component
        if volatility > 0.4:  # 40% annual volatility
            risk_score += 2
        elif volatility > 0.25:  # 25% annual volatility
            risk_score += 1
        
        # Drawdown component
        if max_drawdown < -0.3:  # 30% maximum drawdown
            risk_score += 2
        elif max_drawdown < -0.15:  # 15% maximum drawdown
            risk_score += 1
        
        if risk_score >= 3:
            return 'High'
        elif risk_score >= 2:
            return 'Medium-High'
        elif risk_score >= 1:
            return 'Medium'
        else:
            return 'Low'
    
    def _generate_recommendation(self, predictions: Dict[str, Any], 
                                sentiment_data: Dict[str, Any], 
                                risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Generate investment recommendation"""
        try:
            # Extract key metrics
            next_day_pred = predictions.get('ensemble_predictions', {}).get(1, {})
            prediction_value = next_day_pred.get('ensemble_prediction', 0)
            confidence = next_day_pred.get('confidence_intervals', {})
            
            sentiment_score = sentiment_data.get('sentiment_score', 0) if sentiment_data else 0
            risk_level = risk_assessment.get('risk_level', 'Medium')
            
            # Generate recommendation logic
            recommendation_score = 0
            
            # Prediction component (40% weight)
            if prediction_value > 0:
                recommendation_score += 0.4
            elif prediction_value < 0:
                recommendation_score -= 0.4
            
            # Sentiment component (30% weight)
            if sentiment_score > 0.1:
                recommendation_score += 0.3
            elif sentiment_score < -0.1:
                recommendation_score -= 0.3
            
            # Risk component (30% weight)
            if risk_level == 'Low':
                recommendation_score += 0.3
            elif risk_level == 'High':
                recommendation_score -= 0.3
            
            # Determine recommendation
            if recommendation_score > 0.5:
                recommendation = 'Strong Buy'
            elif recommendation_score > 0.2:
                recommendation = 'Buy'
            elif recommendation_score > -0.2:
                recommendation = 'Hold'
            elif recommendation_score > -0.5:
                recommendation = 'Sell'
            else:
                recommendation = 'Strong Sell'
            
            return {
                'recommendation': recommendation,
                'confidence': abs(recommendation_score),
                'score': float(recommendation_score),
                'factors': {
                    'prediction_signal': 'Positive' if prediction_value > 0 else 'Negative',
                    'sentiment_signal': 'Positive' if sentiment_score > 0 else 'Negative',
                    'risk_level': risk_level
                },
                'rationale': self._generate_rationale(recommendation, prediction_value, sentiment_score, risk_level)
            }
            
        except Exception as e:
            logger.error(f"Error generating recommendation: {str(e)}")
            return {'recommendation': 'Hold', 'confidence': 0.0, 'error': str(e)}
    
    def _generate_rationale(self, recommendation: str, prediction: float, 
                           sentiment: float, risk_level: str) -> str:
        """Generate rationale for recommendation"""
        rationale_parts = []
        
        if prediction > 0:
            rationale_parts.append("ML models predict price appreciation")
        else:
            rationale_parts.append("ML models predict price decline")
        
        if sentiment > 0.1:
            rationale_parts.append("positive market sentiment")
        elif sentiment < -0.1:
            rationale_parts.append("negative market sentiment")
        
        if risk_level == 'Low':
            rationale_parts.append("low risk profile")
        elif risk_level == 'High':
            rationale_parts.append("high risk profile")
        
        return f"{recommendation} recommendation based on: " + ", ".join(rationale_parts)
    
    def _generate_portfolio_recommendation(self, optimization_result: Dict[str, Any],
                                         diversification_analysis: Dict[str, Any],
                                         monte_carlo_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate portfolio-level recommendation"""
        try:
            metrics = optimization_result.get('portfolio_metrics', {})
            sharpe_ratio = metrics.get('sharpe_ratio', 0)
            max_drawdown = metrics.get('max_drawdown', 0)
            
            diversification_score = diversification_analysis.get('diversification_score', 50)
            prob_loss = monte_carlo_results.get('probability_of_loss', 0.5)
            
            # Portfolio quality score
            portfolio_score = 0
            
            # Sharpe ratio component
            if sharpe_ratio > 1.5:
                portfolio_score += 3
            elif sharpe_ratio > 1.0:
                portfolio_score += 2
            elif sharpe_ratio > 0.5:
                portfolio_score += 1
            
            # Diversification component
            if diversification_score > 80:
                portfolio_score += 2
            elif diversification_score > 60:
                portfolio_score += 1
            
            # Risk component
            if max_drawdown > -0.1:  # Less than 10% max drawdown
                portfolio_score += 1
            
            # Determine rating
            if portfolio_score >= 5:
                rating = 'Excellent'
            elif portfolio_score >= 3:
                rating = 'Good'
            elif portfolio_score >= 2:
                rating = 'Fair'
            else:
                rating = 'Poor'
            
            return {
                'portfolio_rating': rating,
                'portfolio_score': portfolio_score,
                'key_metrics': {
                    'sharpe_ratio': float(sharpe_ratio),
                    'max_drawdown': float(max_drawdown),
                    'diversification_score': float(diversification_score),
                    'probability_of_loss': float(prob_loss)
                },
                'recommendations': self._get_portfolio_recommendations(
                    sharpe_ratio, diversification_score, max_drawdown
                )
            }
            
        except Exception as e:
            logger.error(f"Error generating portfolio recommendation: {str(e)}")
            return {'portfolio_rating': 'Unknown', 'error': str(e)}
    
    def _get_portfolio_recommendations(self, sharpe_ratio: float, 
                                     diversification_score: float, 
                                     max_drawdown: float) -> List[str]:
        """Get specific portfolio recommendations"""
        recommendations = []
        
        if sharpe_ratio < 0.5:
            recommendations.append("Consider reviewing asset selection to improve risk-adjusted returns")
        
        if diversification_score < 60:
            recommendations.append("Improve diversification by adding assets from different sectors/regions")
        
        if max_drawdown < -0.2:  # More than 20% drawdown
            recommendations.append("Consider adding defensive assets to reduce maximum drawdown risk")
        
        if sharpe_ratio > 1.0 and diversification_score > 70:
            recommendations.append("Portfolio shows strong risk-adjusted performance and diversification")
        
        return recommendations if recommendations else ["Portfolio appears well-balanced"]
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and component health"""
        try:
            # Check data sources
            data_sources_status = self.data_collector.get_source_status()
            
            # Check cache statistics
            cache_stats = self.data_collector.get_cache_stats()
            
            # Performance metrics
            performance_summary = performance_tracker.get_summary()
            
            # Model evaluators status
            evaluators_status = {
                symbol: len(evaluator.models) 
                for symbol, evaluator in self.model_evaluators.items()
            }
            
            return {
                'version': self.version,
                'status': 'Operational',
                'timestamp': datetime.now().isoformat(),
                'data_sources': data_sources_status,
                'cache_statistics': cache_stats,
                'performance_metrics': performance_summary,
                'active_model_evaluators': evaluators_status,
                'components': {
                    'data_collector': 'Available',
                    'data_processor': 'Available',
                    'sentiment_analyzer': 'Available',
                    'portfolio_optimizer': 'Available'
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {str(e)}")
            return {'error': str(e)}

def main():
    """Main application entry point"""
    try:
        # Initialize the predictor
        predictor = StockPricePredictor()
        
        # Example usage
        print(f"Stock Price Predictor v{predictor.version}")
        print("=" * 50)
        
        # Single stock analysis example
        print("Analyzing AAPL...")
        aapl_analysis = predictor.analyze_stock(
            'AAPL', 
            prediction_horizons=[1, 7, 30],
            include_sentiment=True,
            include_technicals=True
        )
        
        if 'error' not in aapl_analysis:
            print(f"Current Price: ${aapl_analysis['current_price']:.2f}")
            
            predictions = aapl_analysis['predictions']['ensemble_predictions']
            if 1 in predictions:
                next_day = predictions[1]['ensemble_prediction']
                print(f"Next Day Prediction: ${next_day:.2f}")
            
            recommendation = aapl_analysis['recommendation']
            print(f"Recommendation: {recommendation['recommendation']} (Confidence: {recommendation['confidence']:.2f})")
        
        print("\n" + "=" * 50)
        
        # Portfolio analysis example
        print("Analyzing Portfolio [AAPL, MSFT, GOOGL]...")
        portfolio_analysis = predictor.analyze_portfolio(
            ['AAPL', 'MSFT', 'GOOGL'],
            optimization_method='max_sharpe'
        )
        
        if 'error' not in portfolio_analysis:
            metrics = portfolio_analysis['portfolio_metrics']
            print(f"Expected Annual Return: {metrics.get('annual_return', 0):.2%}")
            print(f"Annual Volatility: {metrics.get('annual_volatility', 0):.2%}")
            print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
            
            recommendation = portfolio_analysis['portfolio_recommendation']
            print(f"Portfolio Rating: {recommendation['portfolio_rating']}")
        
        # System status
        print("\n" + "=" * 50)
        status = predictor.get_system_status()
        print(f"System Status: {status['status']}")
        print(f"Active Model Evaluators: {len(status['active_model_evaluators'])}")
        
    except Exception as e:
        logger.error(f"Error in main application: {str(e)}")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()