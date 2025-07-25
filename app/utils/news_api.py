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
            logger.warning("Alpha Vantage API key not found. Using mock data.")
            return self._get_mock_news_data(params.get('tickers', ''))
        
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
    
    def _get_mock_news_data(self, ticker: str) -> Dict:
        """Return mock news data when API key is not available"""
        return {
            "items": str(5),
            "sentiment_score_definition": "x <= -0.35: Bearish; -0.35 < x <= -0.15: Somewhat-Bearish; -0.15 < x < 0.15: Neutral; 0.15 <= x < 0.35: Somewhat_Bullish; x >= 0.35: Bullish",
            "relevance_score_definition": "0 < x <= 1, with a higher score indicating higher relevance.",
            "feed": [
                {
                    "title": f"{ticker} Reports Strong Q4 Earnings with Record Revenue Growth",
                    "url": "https://example.com/news1",
                    "time_published": "20241201T120000",
                    "authors": ["Financial News"],
                    "summary": f"Company {ticker} announced impressive quarterly results, beating analyst expectations on both revenue and earnings per share. The strong performance was driven by increased demand and operational efficiency improvements.",
                    "banner_image": "https://example.com/image1.jpg",
                    "source": "MarketWatch",
                    "category_within_source": "Finance",
                    "source_domain": "marketwatch.com",
                    "topics": [
                        {
                            "topic": "Earnings",
                            "relevance_score": "0.9"
                        }
                    ],
                    "overall_sentiment_score": 0.4,
                    "overall_sentiment_label": "Bullish",
                    "ticker_sentiment": [
                        {
                            "ticker": ticker,
                            "relevance_score": "0.95",
                            "ticker_sentiment_score": "0.42",
                            "ticker_sentiment_label": "Bullish"
                        }
                    ]
                },
                {
                    "title": f"Analysts Upgrade {ticker} Stock Rating Following Strategic Partnership",
                    "url": "https://example.com/news2",
                    "time_published": "20241130T143000",
                    "authors": ["Investment Analysis"],
                    "summary": f"Multiple analysts have raised their price targets for {ticker} following announcement of a strategic partnership that could expand market reach and boost future revenue streams.",
                    "banner_image": "https://example.com/image2.jpg",
                    "source": "Yahoo Finance",
                    "category_within_source": "Markets",
                    "source_domain": "finance.yahoo.com",
                    "topics": [
                        {
                            "topic": "Financial Markets",
                            "relevance_score": "0.8"
                        }
                    ],
                    "overall_sentiment_score": 0.3,
                    "overall_sentiment_label": "Somewhat-Bullish",
                    "ticker_sentiment": [
                        {
                            "ticker": ticker,
                            "relevance_score": "0.88",
                            "ticker_sentiment_score": "0.31",
                            "ticker_sentiment_label": "Somewhat-Bullish"
                        }
                    ]
                },
                {
                    "title": f"Market Volatility Affects {ticker} Despite Strong Fundamentals",
                    "url": "https://example.com/news3",
                    "time_published": "20241129T094500",
                    "authors": ["Market Reporter"],
                    "summary": f"While {ticker} maintains solid business fundamentals, broader market uncertainty and sector-wide concerns have created some volatility in the stock price recently.",
                    "banner_image": "https://example.com/image3.jpg",
                    "source": "Reuters",
                    "category_within_source": "Business",
                    "source_domain": "reuters.com",
                    "topics": [
                        {
                            "topic": "Technology",
                            "relevance_score": "0.7"
                        }
                    ],
                    "overall_sentiment_score": -0.1,
                    "overall_sentiment_label": "Neutral",
                    "ticker_sentiment": [
                        {
                            "ticker": ticker,
                            "relevance_score": "0.82",
                            "ticker_sentiment_score": "-0.08",
                            "ticker_sentiment_label": "Neutral"
                        }
                    ]
                },
                {
                    "title": f"{ticker} Announces New Product Line Expected to Drive Growth",
                    "url": "https://example.com/news4",
                    "time_published": "20241128T160000",
                    "authors": ["Business News"],
                    "summary": f"The company unveiled its latest product innovation, which management expects to capture significant market share and contribute meaningfully to revenue growth in the coming quarters.",
                    "banner_image": "https://example.com/image4.jpg",
                    "source": "Bloomberg",
                    "category_within_source": "Technology",
                    "source_domain": "bloomberg.com",
                    "topics": [
                        {
                            "topic": "Technology",
                            "relevance_score": "0.9"
                        }
                    ],
                    "overall_sentiment_score": 0.35,
                    "overall_sentiment_label": "Bullish",
                    "ticker_sentiment": [
                        {
                            "ticker": ticker,
                            "relevance_score": "0.92",
                            "ticker_sentiment_score": "0.38",
                            "ticker_sentiment_label": "Bullish"
                        }
                    ]
                },
                {
                    "title": f"Industry Trends Support Long-term Outlook for {ticker}",
                    "url": "https://example.com/news5",
                    "time_published": "20241127T113000",
                    "authors": ["Industry Analysis"],
                    "summary": f"Sector analysis indicates favorable long-term trends that position {ticker} well for sustained growth, despite short-term market challenges facing the broader industry.",
                    "banner_image": "https://example.com/image5.jpg",
                    "source": "CNBC",
                    "category_within_source": "Investing",
                    "source_domain": "cnbc.com",
                    "topics": [
                        {
                            "topic": "Financial Markets",
                            "relevance_score": "0.75"
                        }
                    ],
                    "overall_sentiment_score": 0.2,
                    "overall_sentiment_label": "Somewhat-Bullish",
                    "ticker_sentiment": [
                        {
                            "ticker": ticker,
                            "relevance_score": "0.78",
                            "ticker_sentiment_score": "0.22",
                            "ticker_sentiment_label": "Somewhat-Bullish"
                        }
                    ]
                }
            ]
        }
    
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
        
        # Prepare API parameters
        params = {
            'function': 'NEWS_SENTIMENT',
            'tickers': ticker,
            'limit': str(limit),
            'apikey': self.api_key or 'demo'
        }
        
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