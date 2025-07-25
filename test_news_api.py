#!/usr/bin/env python3
"""
Test script for the news API functionality without external dependencies
"""

import os
import sys
import json
from datetime import datetime

# Add app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Mock the requests module to avoid external dependency
class MockResponse:
    def __init__(self, data):
        self.data = data
    
    def json(self):
        return self.data
    
    def raise_for_status(self):
        pass

class MockRequests:
    @staticmethod
    def get(url, params=None, timeout=None):
        # Return mock data that simulates Alpha Vantage response
        mock_data = {
            "items": "5",
            "sentiment_score_definition": "x <= -0.35: Bearish; -0.35 < x <= -0.15: Somewhat-Bearish; -0.15 < x < 0.15: Neutral; 0.15 <= x < 0.35: Somewhat_Bullish; x >= 0.35: Bullish",
            "feed": [
                {
                    "title": "AAPL Reports Strong Quarterly Earnings",
                    "url": "https://example.com/news1",
                    "time_published": "20241201T120000",
                    "summary": "Apple Inc. reported better than expected quarterly earnings with strong iPhone sales driving revenue growth.",
                    "source": "MarketWatch",
                    "overall_sentiment_score": 0.4,
                    "overall_sentiment_label": "Bullish",
                    "ticker_sentiment": [
                        {
                            "ticker": "AAPL",
                            "relevance_score": "0.95",
                            "ticker_sentiment_score": "0.42",
                            "ticker_sentiment_label": "Bullish"
                        }
                    ]
                }
            ]
        }
        return MockResponse(mock_data)

# Monkey patch requests
sys.modules['requests'] = MockRequests()

def test_news_api():
    print("Testing News API functionality...")
    
    try:
        from app.utils.news_api import NewsAPI, NewsArticle
        
        # Create news API instance
        news_api = NewsAPI()
        
        # Test fetching news
        print("\n1. Testing news fetching...")
        articles = news_api.get_stock_news('AAPL', limit=5)
        print(f"   ✓ Fetched {len(articles)} articles")
        
        if articles:
            article = articles[0]
            print(f"   ✓ First article: {article.title[:50]}...")
            print(f"   ✓ Source: {article.source}")
            print(f"   ✓ Date: {article.published_date}")
            print(f"   ✓ Sentiment: {article.sentiment}")
        
        # Test news summary
        print("\n2. Testing news summary...")
        summary = news_api.get_news_summary('AAPL')
        print(f"   ✓ Total articles: {summary['total_articles']}")
        print(f"   ✓ Overall sentiment: {summary['sentiment_label']}")
        print(f"   ✓ Bullish count: {summary['bullish_count']}")
        
        # Test caching
        print("\n3. Testing caching...")
        articles2 = news_api.get_stock_news('AAPL', limit=5)
        print(f"   ✓ Second fetch returned {len(articles2)} articles (should be cached)")
        
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_news_api()
    sys.exit(0 if success else 1)