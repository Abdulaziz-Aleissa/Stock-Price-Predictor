"""
News Module
Handle news fetching and processing
"""

from app.utils.news_api import news_api
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class NewsManager:
    """Handle news fetching and processing operations"""
    
    def __init__(self):
        self.news_api = news_api
    
    def get_stock_news(self, symbol: str, limit: int = 10) -> List[Dict]:
        """Get news for a specific stock symbol"""
        try:
            # Use the existing news_api utility
            return self.news_api.get_stock_news(symbol, limit)
        except Exception as e:
            logger.error(f"Error getting stock news for {symbol}: {str(e)}")
            return []
    
    def get_market_news(self, limit: int = 20) -> List[Dict]:
        """Get general market news"""
        try:
            return self.news_api.get_market_news(limit)
        except Exception as e:
            logger.error(f"Error getting market news: {str(e)}")
            return []
    
    def get_financial_news(self, category: str = 'business', limit: int = 15) -> List[Dict]:
        """Get financial news by category"""
        try:
            return self.news_api.get_financial_news(category, limit)
        except Exception as e:
            logger.error(f"Error getting financial news: {str(e)}")
            return []
    
    def search_news(self, query: str, limit: int = 10) -> List[Dict]:
        """Search for news articles"""
        try:
            return self.news_api.search_news(query, limit)
        except Exception as e:
            logger.error(f"Error searching news for '{query}': {str(e)}")
            return []
    
    def get_news_for_portfolio(self, symbols: List[str], limit_per_stock: int = 3) -> Dict:
        """Get news for all stocks in a portfolio"""
        try:
            portfolio_news = {}
            
            for symbol in symbols:
                news = self.get_stock_news(symbol, limit_per_stock)
                if news:
                    portfolio_news[symbol] = news
            
            return portfolio_news
            
        except Exception as e:
            logger.error(f"Error getting portfolio news: {str(e)}")
            return {}
    
    def get_trending_topics(self) -> List[str]:
        """Get trending financial topics"""
        try:
            # This would typically analyze news content to extract trending topics
            # For now, return common financial topics
            return [
                'Stock Market',
                'Cryptocurrency',
                'Federal Reserve',
                'Interest Rates',
                'Inflation',
                'Earnings Reports',
                'IPO',
                'Mergers and Acquisitions'
            ]
        except Exception as e:
            logger.error(f"Error getting trending topics: {str(e)}")
            return []
    
    def analyze_news_sentiment(self, news_articles: List[Dict]) -> Dict:
        """Analyze sentiment of news articles"""
        try:
            if not news_articles:
                return {'sentiment': 'neutral', 'confidence': 0}
            
            # Basic sentiment analysis based on keywords
            # In a real implementation, you'd use NLP libraries like VADER or TextBlob
            positive_keywords = ['growth', 'profit', 'gain', 'up', 'rise', 'bullish', 'success', 'strong']
            negative_keywords = ['loss', 'decline', 'down', 'fall', 'bearish', 'weak', 'crisis', 'problem']
            
            positive_count = 0
            negative_count = 0
            total_articles = len(news_articles)
            
            for article in news_articles:
                title = article.get('title', '').lower()
                description = article.get('description', '').lower()
                content = f"{title} {description}"
                
                pos_score = sum(1 for keyword in positive_keywords if keyword in content)
                neg_score = sum(1 for keyword in negative_keywords if keyword in content)
                
                if pos_score > neg_score:
                    positive_count += 1
                elif neg_score > pos_score:
                    negative_count += 1
            
            if positive_count > negative_count:
                sentiment = 'positive'
                confidence = (positive_count / total_articles) * 100
            elif negative_count > positive_count:
                sentiment = 'negative'
                confidence = (negative_count / total_articles) * 100
            else:
                sentiment = 'neutral'
                confidence = 50
            
            return {
                'sentiment': sentiment,
                'confidence': round(confidence, 2),
                'positive_articles': positive_count,
                'negative_articles': negative_count,
                'neutral_articles': total_articles - positive_count - negative_count,
                'total_articles': total_articles
            }
            
        except Exception as e:
            logger.error(f"Error analyzing news sentiment: {str(e)}")
            return {'sentiment': 'neutral', 'confidence': 0}
    
    def get_news_summary(self, symbol: str) -> Dict:
        """Get comprehensive news summary for a stock"""
        try:
            news = self.get_stock_news(symbol, 20)
            
            if not news:
                return {
                    'symbol': symbol,
                    'news_count': 0,
                    'sentiment': self.analyze_news_sentiment([]),
                    'recent_news': []
                }
            
            sentiment_analysis = self.analyze_news_sentiment(news)
            
            return {
                'symbol': symbol,
                'news_count': len(news),
                'sentiment': sentiment_analysis,
                'recent_news': news[:5],  # Most recent 5 articles
                'all_news': news
            }
            
        except Exception as e:
            logger.error(f"Error getting news summary for {symbol}: {str(e)}")
            return {
                'symbol': symbol,
                'news_count': 0,
                'sentiment': {'sentiment': 'neutral', 'confidence': 0},
                'recent_news': []
            }


# Global instance to be used across the application
news_manager = NewsManager()