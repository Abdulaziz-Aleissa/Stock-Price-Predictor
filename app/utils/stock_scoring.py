import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

class StockScoring:
    def __init__(self):
        self.scoring_criteria = {
            'pe_ratio': {
                'excellent': (0, 10, 5),      # P/E ≤ 10 = 5 points
                'very_good': (10, 15, 4),     # P/E ≤ 15 = 4 points
                'good': (15, 20, 3),          # P/E ≤ 20 = 3 points
                'fair': (20, 25, 2),          # P/E ≤ 25 = 2 points
                'poor': (25, float('inf'), 1) # P/E > 25 = 1 point
            },
            'pb_ratio': {
                'excellent': (0, 1, 5),       # P/B ≤ 1 = 5 points
                'very_good': (1, 2, 4),       # P/B ≤ 2 = 4 points
                'good': (2, 3, 3),            # P/B ≤ 3 = 3 points
                'fair': (3, 4, 2),            # P/B ≤ 4 = 2 points
                'poor': (4, float('inf'), 1)  # P/B > 4 = 1 point
            },
            'roe': {
                'excellent': (30, float('inf'), 5), # ROE ≥ 30% = 5 points
                'very_good': (20, 30, 4),           # ROE ≥ 20% = 4 points
                'good': (15, 20, 3),                # ROE ≥ 15% = 3 points
                'fair': (10, 15, 2),                # ROE ≥ 10% = 2 points
                'poor': (0, 10, 1)                  # ROE < 10% = 1 point
            },
            'ev_ebitda': {
                'excellent': (0, 8, 5),       # EV/EBITDA ≤ 8 = 5 points
                'very_good': (8, 10, 4),      # EV/EBITDA ≤ 10 = 4 points
                'good': (10, 12, 3),          # EV/EBITDA ≤ 12 = 3 points
                'fair': (12, 15, 2),          # EV/EBITDA ≤ 15 = 2 points
                'poor': (15, float('inf'), 1) # EV/EBITDA > 15 = 1 point
            },
            'rsi': {
                'oversold': (0, 30, 5),       # RSI < 30 = 5 points (oversold)
                'neutral': (30, 70, 3),       # RSI 30-70 = 3 points (neutral)
                'overbought': (70, 100, 1)    # RSI > 70 = 1 point (overbought)
            },
            'volatility': {
                'excellent': (0, 0.15, 5),      # ≤15%
                'very_good': (0.15, 0.25, 4),   # 15%–25%
                'good': (0.25, 0.35, 3),        # 25%–35%
                'fair': (0.35, 0.50, 2),        # 35%–50%
                'poor': (0.50, float('inf'), 1) # >50%
            }
        }

    def get_stock_metrics(self, symbol: str) -> Dict:
        """Fetch comprehensive stock metrics for analysis"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="1y")

            metrics = {
                'pe_ratio': info.get('trailingPE'),
                'pb_ratio': info.get('priceToBook'),
                'roe': info.get('returnOnEquity'),
                'ev_ebitda': info.get('enterpriseToEbitda'),
                'rsi': self._calculate_rsi(hist['Close'], period=14),
                'volatility': self._calculate_volatility(hist['Close'], window=30),
                'market_cap': info.get('marketCap'),
                'current_price': info.get('currentPrice'),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow'),
                'dividend_yield': info.get('dividendYield'),
                'beta': info.get('beta'),
                'debt_to_equity': info.get('debtToEquity'),
                'current_ratio': info.get('currentRatio'),
                'profit_margins': info.get('profitMargins'),
                'revenue_growth': info.get('revenueGrowth')
            }

            # Convert decimals to percentages where appropriate
            if metrics['roe'] is not None:
                metrics['roe'] = metrics['roe'] * 100
            if metrics['dividend_yield'] is not None:
                metrics['dividend_yield'] = metrics['dividend_yield'] * 100
            if metrics['profit_margins'] is not None:
                metrics['profit_margins'] = metrics['profit_margins'] * 100
            if metrics['revenue_growth'] is not None:
                metrics['revenue_growth'] = metrics['revenue_growth'] * 100

            return metrics

        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return {}

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> Optional[float]:
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1] if not rsi.empty else None
        except Exception:
            return None

    def _calculate_volatility(self, prices: pd.Series, window: int = 30) -> Optional[float]:
        try:
            returns = prices.pct_change().dropna()
            volatility_30day = returns.rolling(window=window).std() * np.sqrt(252)
            return volatility_30day.iloc[-1] if not volatility_30day.empty else None
        except Exception:
            return None

    def calculate_score(self, metrics: Dict) -> Dict:
        """Calculate individual and total scores based on metrics"""
        scores = {}
        total_score = 0

        for metric, value in metrics.items():
            if metric in self.scoring_criteria and value is not None:
                score = self._get_metric_score(metric, value)
                scores[f"{metric}_score"] = score
                total_score += score
            else:
                scores[f"{metric}_score"] = 0

        scores['total_score'] = total_score
        return scores

    def _get_metric_score(self, metric: str, value: float) -> int:
        """Get score for a specific metric based on value"""
        criteria = self.scoring_criteria.get(metric, {})
        for _, (min_val, max_val, score) in criteria.items():
            if min_val <= value < max_val:
                return score
        return 0

    def get_recommendation(self, total_score: int, metrics: Dict) -> tuple:
        """Get investment recommendation based on total score"""
        if total_score >= 25:
            recommendation = "BUY"
            reasoning = self._generate_buy_reasoning(metrics, total_score)
        elif total_score <= 10:
            recommendation = "SELL"
            reasoning = self._generate_sell_reasoning(metrics, total_score)
        else:
            recommendation = "HOLD"
            reasoning = self._generate_hold_reasoning(metrics, total_score)
        return recommendation, reasoning

    def _generate_buy_reasoning(self, metrics: Dict, score: int) -> str:
        """Generate reasoning for BUY recommendation"""
        reasons = []
        if metrics.get('pe_ratio') is not None and metrics['pe_ratio'] <= 15:
            reasons.append("attractive valuation (low P/E)")
        if metrics.get('roe') is not None and metrics['roe'] >= 15:
            reasons.append("strong profitability (high ROE)")
        if metrics.get('rsi') is not None and metrics['rsi'] < 70:
            reasons.append("not overbought")
        if metrics.get('volatility') is not None and metrics['volatility'] < 0.3:
            reasons.append("reasonable volatility")
        base_reason = f"Strong fundamentals with score of {score}/30"
        if reasons:
            return f"{base_reason}. Key strengths: {', '.join(reasons[:3])}."
        return base_reason + "."

    def _generate_sell_reasoning(self, metrics: Dict, score: int) -> str:
        """Generate reasoning for SELL recommendation"""
        concerns = []
        if metrics.get('pe_ratio') is not None and metrics['pe_ratio'] > 30:
            concerns.append("overvalued (high P/E)")
        if metrics.get('roe') is not None and metrics['roe'] < 5:
            concerns.append("poor profitability (low ROE)")
        if metrics.get('rsi') is not None and metrics['rsi'] > 80:
            concerns.append("overbought conditions")
        if metrics.get('volatility') is not None and metrics['volatility'] > 0.5:
            concerns.append("high volatility")
        base_reason = f"Weak fundamentals with score of {score}/30"
        if concerns:
            return f"{base_reason}. Key concerns: {', '.join(concerns[:3])}."
        return base_reason + "."

    def _generate_hold_reasoning(self, metrics: Dict, score: int) -> str:
        """Generate reasoning for HOLD recommendation"""
        return f"Mixed fundamentals with moderate score of {score}/30. Consider waiting for better entry/exit points or further analysis."

    def analyze_stocks(self, symbols: List[str]) -> List[Dict]:
        """Analyze multiple stocks and return comprehensive results"""
        results = []
        for symbol in symbols:
            symbol = symbol.strip().upper()
            try:
                metrics = self.get_stock_metrics(symbol)
                if not metrics:
                    results.append({
                        'symbol': symbol,
                        'error': 'Unable to fetch data'
                    })
                    continue
                scores = self.calculate_score(metrics)
                recommendation, reasoning = self.get_recommendation(scores['total_score'], metrics)
                results.append({
                    'symbol': symbol,
                    'metrics': metrics,
                    'scores': scores,
                    'total_score': scores['total_score'],
                    'recommendation': recommendation,
                    'reasoning': reasoning
                })
            except Exception as e:
                results.append({
                    'symbol': symbol,
                    'error': f'Analysis failed: {str(e)}'
                })
        return results