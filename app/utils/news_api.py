"""
News API utility module for fetching stock-related news from Alpha Vantage News & Sentiment API.
This module provides functionality to fetch, cache, and process news articles for stock tickers.
"""
import os
import requests
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

@dataclass
class NewsArticle:
    """Data class representing a news article"""
    title: str
    summary: str
    url: str
    source: str
    published_date: str
    sentiment: Optional[Dict[str, float]] = None

class NewsCache:
    """Simple in-memory cache for news articles"""
    def __init__(self, cache_duration_minutes: int = 30):
        self.cache = {}
        self.cache_duration = timedelta(minutes=cache_duration_minutes)
    
    def get(self, key: str) -> Optional[List[NewsArticle]]:
        """Get cached news articles if not expired"""
        if key in self.cache:
            cached_time, articles = self.cache[key]
            if datetime.now() - cached_time < self.cache_duration:
                return articles
            else:
                # Remove expired entry
                del self.cache[key]
        return None
    
    def set(self, key: str, articles: List[NewsArticle]):
        """Cache news articles with timestamp"""
        self.cache[key] = (datetime.now(), articles)
    
    def clear(self):
        """Clear all cached entries"""
        self.cache.clear()

class NewsAPI:
    """News API client for Alpha Vantage News & Sentiment API"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('ALPHA_VANTAGE_API_KEY')
        self.base_url = "https://www.alphavantage.co/query"
        self.cache = NewsCache()
        self.rate_limit_delay = 1  # 1 second between requests (free tier limit)
        self.last_request_time = 0
        
    def _rate_limit(self):
        """Implement rate limiting to respect API limits"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last_request)
        self.last_request_time = time.time()
    
    def _make_request(self, params: Dict[str, str]) -> Optional[Dict]:
        """Make API request with error handling and rate limiting"""
        if not self.api_key:
            logger.error("Alpha Vantage API key not found. Real news requires an API key.")
            return None
        
        self._rate_limit()
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API error messages
            if 'Error Message' in data:
                logger.error(f"Alpha Vantage API error: {data['Error Message']}")
                return None
            
            if 'Information' in data:
                logger.warning(f"Alpha Vantage API info: {data['Information']}")
                return None
                
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error making API request: {str(e)}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing API response: {str(e)}")
            return None

    
    def _parse_news_article(self, article_data: Dict, ticker: str) -> Optional[NewsArticle]:
        """Parse API response data into NewsArticle objects"""
        try:
            # Format the published date
            time_published = article_data.get('time_published', '')
            try:
                # Parse the time format: YYYYMMDDTHHMMSS
                if len(time_published) >= 8:
                    year = time_published[:4]
                    month = time_published[4:6]
                    day = time_published[6:8]
                    formatted_date = f"{year}-{month}-{day}"
                else:
                    formatted_date = "Unknown"
            except:
                formatted_date = "Unknown"
                
            # Extract sentiment data for the specific ticker
            sentiment = None
            ticker_sentiments = article_data.get('ticker_sentiment', [])
            for ts in ticker_sentiments:
                if ts.get('ticker', '').upper() == ticker.upper():
                    sentiment = {
                        'score': float(ts.get('ticker_sentiment_score', 0)),
                        'label': ts.get('ticker_sentiment_label', 'Neutral'),
                        'relevance': float(ts.get('relevance_score', 0))
                    }
                    break
            
            # If no ticker-specific sentiment, use overall sentiment
            if sentiment is None:
                sentiment = {
                    'score': float(article_data.get('overall_sentiment_score', 0)),
                    'label': article_data.get('overall_sentiment_label', 'Neutral'),
                    'relevance': 0.5  # Default relevance
                }
            
            return NewsArticle(
                title=article_data.get('title', ''),
                summary=article_data.get('summary', ''),
                url=article_data.get('url', ''),
                source=article_data.get('source', 'Unknown'),
                published_date=formatted_date,
                sentiment=sentiment
            )
            
        except Exception as e:
            logger.error(f"Error parsing news article: {str(e)}")
            return None
    
    def get_stock_news(self, ticker: str, limit: int = 8) -> List[NewsArticle]:
        """
        Fetch news articles for a specific stock ticker
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            limit: Maximum number of articles to return (default: 8)
            
        Returns:
            List of NewsArticle objects
        """
        ticker = ticker.upper()
        cache_key = f"{ticker}_{limit}"
        
        # Check cache first
        cached_articles = self.cache.get(cache_key)
        if cached_articles:
            logger.info(f"Returning cached news for {ticker}")
            return cached_articles
        
        # Check if API key is configured
        if not self.api_key:
            logger.error(f"Cannot fetch real news for {ticker}: Alpha Vantage API key not configured")
            logger.info("To get real news articles, please:")
            logger.info("1. Get a free API key from https://www.alphavantage.co/support/#api-key")
            logger.info("2. Copy .env.example to .env")
            logger.info("3. Add your API key to the .env file")
            logger.info("4. See NEWS_SETUP.md for detailed instructions")
            return []
        
        # Prepare API parameters
        params = {
            'function': 'NEWS_SENTIMENT',
            'tickers': ticker,
            'limit': str(limit),
            'apikey': self.api_key
        }
        
        logger.info(f"Making real API call to Alpha Vantage for {ticker} news...")
        
        # Make API request
        data = self._make_request(params)
        if not data:
            logger.warning(f"Failed to fetch news for {ticker}")
            return []
        
        # Parse response
        articles = []
        feed = data.get('feed', [])
        
        for article_data in feed[:limit]:
            article = self._parse_news_article(article_data, ticker)
            if article:
                articles.append(article)
        
        # Cache the results
        self.cache.set(cache_key, articles)
        
        logger.info(f"Fetched {len(articles)} news articles for {ticker}")
        return articles
    
    def get_news_summary(self, ticker: str) -> Dict[str, any]:
        """
        Get a summary of news sentiment and key metrics
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary containing sentiment summary
        """
        articles = self.get_stock_news(ticker)
        
        if not articles:
            return {
                'total_articles': 0,
                'avg_sentiment_score': 0,
                'sentiment_label': 'Neutral',
                'bullish_count': 0,
                'bearish_count': 0,
                'neutral_count': 0
            }
        
        # Calculate sentiment metrics
        total_score = 0
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        
        for article in articles:
            if article.sentiment:
                score = article.sentiment['score']
                total_score += score
                
                if score >= 0.15:
                    bullish_count += 1
                elif score <= -0.15:
                    bearish_count += 1
                else:
                    neutral_count += 1
        
        avg_score = total_score / len(articles) if articles else 0
        
        # Determine overall sentiment label
        if avg_score >= 0.15:
            sentiment_label = 'Bullish'
        elif avg_score <= -0.15:
            sentiment_label = 'Bearish'
        else:
            sentiment_label = 'Neutral'
        
        return {
            'total_articles': len(articles),
            'avg_sentiment_score': avg_score,
            'sentiment_label': sentiment_label,
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'neutral_count': neutral_count
        }

# Global instance
news_api = NewsAPI()