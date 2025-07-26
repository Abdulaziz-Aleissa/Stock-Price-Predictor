"""
Enhanced Sentiment Analysis Module
Integrates news sentiment analysis with social media sentiment tracking
"""
import pandas as pd
import numpy as np
import requests
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import time
import re
import warnings
warnings.filterwarnings('ignore')

# Import configuration and helpers
from ..utils.config import config
from ..utils.helpers import performance_tracker

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Enhanced sentiment analysis with news and social media integration"""
    
    def __init__(self):
        self.news_sources = config.SENTIMENT_SETTINGS['news_sources']
        self.social_sources = config.SENTIMENT_SETTINGS['social_sources']
        self.api_keys = config.get_api_keys()
        
        # Check available libraries
        self.nltk_available = self._check_nltk()
        self.textblob_available = self._check_textblob()
        
        # Initialize sentiment models
        self.sentiment_models = {}
        self._initialize_sentiment_models()
    
    def _check_nltk(self) -> bool:
        """Check if NLTK is available"""
        try:
            import nltk
            from nltk.sentiment import SentimentIntensityAnalyzer
            
            # Try to download required NLTK data
            try:
                nltk.data.find('vader_lexicon')
            except LookupError:
                try:
                    nltk.download('vader_lexicon', quiet=True)
                except:
                    logger.warning("Could not download NLTK VADER lexicon")
                    return False
            
            return True
            
        except ImportError:
            logger.warning("NLTK not available for sentiment analysis")
            return False
    
    def _check_textblob(self) -> bool:
        """Check if TextBlob is available"""
        try:
            from textblob import TextBlob
            return True
        except ImportError:
            logger.warning("TextBlob not available for sentiment analysis")
            return False
    
    def _initialize_sentiment_models(self):
        """Initialize available sentiment analysis models"""
        try:
            # NLTK VADER Sentiment
            if self.nltk_available:
                try:
                    from nltk.sentiment import SentimentIntensityAnalyzer
                    self.sentiment_models['vader'] = SentimentIntensityAnalyzer()
                    logger.info("VADER sentiment analyzer initialized")
                except Exception as e:
                    logger.warning(f"Could not initialize VADER: {str(e)}")
            
            # TextBlob Sentiment  
            if self.textblob_available:
                self.sentiment_models['textblob'] = True
                logger.info("TextBlob sentiment analyzer initialized")
            
            # Simple pattern-based sentiment (fallback)
            self.sentiment_models['pattern'] = self._initialize_pattern_sentiment()
            
        except Exception as e:
            logger.error(f"Error initializing sentiment models: {str(e)}")
    
    def _initialize_pattern_sentiment(self) -> Dict[str, List[str]]:
        """Initialize simple pattern-based sentiment analyzer"""
        return {
            'positive': [
                'bullish', 'buy', 'strong', 'growth', 'profit', 'gain', 'up', 'rise',
                'surge', 'rally', 'boom', 'excellent', 'outstanding', 'great',
                'positive', 'optimistic', 'confident', 'upgrade', 'beat', 'exceed',
                'record', 'high', 'soar', 'jump', 'leap', 'breakthrough', 'success'
            ],
            'negative': [
                'bearish', 'sell', 'weak', 'decline', 'loss', 'fall', 'drop',
                'crash', 'plunge', 'sink', 'poor', 'disappointing', 'terrible',
                'negative', 'pessimistic', 'concern', 'downgrade', 'miss', 'below',
                'low', 'worst', 'fail', 'struggle', 'crisis', 'trouble', 'warning'
            ],
            'neutral': [
                'hold', 'stable', 'unchanged', 'flat', 'sideways', 'maintain',
                'neutral', 'mixed', 'average', 'moderate', 'steady'
            ]
        }
    
    def analyze_text_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of a single text using multiple methods
        
        Args:
            text: Text to analyze
        
        Returns:
            Dictionary with sentiment scores
        """
        try:
            if not text or not isinstance(text, str):
                return {'error': 'Invalid text input'}
            
            # Clean text
            cleaned_text = self._clean_text(text)
            
            sentiment_scores = {}
            
            # VADER Sentiment
            if 'vader' in self.sentiment_models:
                try:
                    vader_scores = self.sentiment_models['vader'].polarity_scores(cleaned_text)
                    sentiment_scores['vader'] = {
                        'compound': vader_scores['compound'],
                        'positive': vader_scores['pos'],
                        'negative': vader_scores['neg'],
                        'neutral': vader_scores['neu']
                    }
                except Exception as e:
                    logger.warning(f"VADER analysis failed: {str(e)}")
            
            # TextBlob Sentiment
            if 'textblob' in self.sentiment_models:
                try:
                    from textblob import TextBlob
                    blob = TextBlob(cleaned_text)
                    sentiment_scores['textblob'] = {
                        'polarity': blob.sentiment.polarity,  # -1 to 1
                        'subjectivity': blob.sentiment.subjectivity  # 0 to 1
                    }
                except Exception as e:
                    logger.warning(f"TextBlob analysis failed: {str(e)}")
            
            # Pattern-based sentiment
            pattern_score = self._analyze_pattern_sentiment(cleaned_text)
            sentiment_scores['pattern'] = pattern_score
            
            # Calculate composite sentiment
            composite_sentiment = self._calculate_composite_sentiment(sentiment_scores)
            
            return {
                'text': text[:200] + '...' if len(text) > 200 else text,
                'cleaned_text': cleaned_text,
                'individual_scores': sentiment_scores,
                'composite_sentiment': composite_sentiment,
                'confidence': self._calculate_confidence(sentiment_scores)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing text sentiment: {str(e)}")
            return {'error': str(e)}
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text for sentiment analysis"""
        try:
            # Convert to lowercase
            text = text.lower()
            
            # Remove URLs
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
            
            # Remove mentions and hashtags (but keep the text)
            text = re.sub(r'[@#]([a-zA-Z0-9_]+)', r'\\1', text)
            
            # Remove extra whitespace
            text = re.sub(r'\\s+', ' ', text).strip()
            
            # Remove non-alphanumeric characters (keep spaces and basic punctuation)
            text = re.sub(r'[^a-zA-Z0-9\\s.,!?-]', '', text)
            
            return text
            
        except Exception as e:
            logger.error(f"Error cleaning text: {str(e)}")
            return text
    
    def _analyze_pattern_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using pattern matching"""
        try:
            patterns = self.sentiment_models.get('pattern', {})
            
            positive_count = sum(1 for word in patterns.get('positive', []) if word in text)
            negative_count = sum(1 for word in patterns.get('negative', []) if word in text)
            neutral_count = sum(1 for word in patterns.get('neutral', []) if word in text)
            
            total_sentiment_words = positive_count + negative_count + neutral_count
            
            if total_sentiment_words == 0:
                return {'score': 0.0, 'label': 'neutral', 'confidence': 0.0}
            
            # Calculate sentiment score (-1 to 1)
            sentiment_score = (positive_count - negative_count) / total_sentiment_words
            
            # Determine label
            if sentiment_score > 0.1:
                label = 'positive'
            elif sentiment_score < -0.1:
                label = 'negative'
            else:
                label = 'neutral'
            
            # Calculate confidence based on number of sentiment words
            confidence = min(1.0, total_sentiment_words / 10.0)
            
            return {
                'score': sentiment_score,
                'label': label,
                'confidence': confidence,
                'positive_words': positive_count,
                'negative_words': negative_count,
                'neutral_words': neutral_count
            }
            
        except Exception as e:
            logger.error(f"Error in pattern sentiment analysis: {str(e)}")
            return {'score': 0.0, 'label': 'neutral', 'confidence': 0.0}
    
    def _calculate_composite_sentiment(self, sentiment_scores: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate composite sentiment from multiple analyzers"""
        try:
            scores = []
            weights = []
            
            # VADER (weight: 0.4)
            if 'vader' in sentiment_scores:
                scores.append(sentiment_scores['vader']['compound'])
                weights.append(0.4)
            
            # TextBlob (weight: 0.4)
            if 'textblob' in sentiment_scores:
                scores.append(sentiment_scores['textblob']['polarity'])
                weights.append(0.4)
            
            # Pattern (weight: 0.2)
            if 'pattern' in sentiment_scores:
                scores.append(sentiment_scores['pattern']['score'])
                weights.append(0.2)
            
            if not scores:
                return {'score': 0.0, 'label': 'neutral', 'confidence': 0.0}
            
            # Calculate weighted average
            weighted_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
            
            # Determine label
            if weighted_score > 0.1:
                label = 'positive'
            elif weighted_score < -0.1:
                label = 'negative'
            else:
                label = 'neutral'
            
            # Calculate confidence (agreement between methods)
            if len(scores) > 1:
                score_std = np.std(scores)
                confidence = max(0.0, 1.0 - score_std)
            else:
                confidence = 0.5
            
            return {
                'score': float(weighted_score),
                'label': label,
                'confidence': float(confidence),
                'methods_used': len(scores)
            }
            
        except Exception as e:
            logger.error(f"Error calculating composite sentiment: {str(e)}")
            return {'score': 0.0, 'label': 'neutral', 'confidence': 0.0}
    
    def _calculate_confidence(self, sentiment_scores: Dict[str, Any]) -> float:
        """Calculate overall confidence in sentiment analysis"""
        try:
            confidences = []
            
            # VADER confidence (based on compound score magnitude)
            if 'vader' in sentiment_scores:
                compound = abs(sentiment_scores['vader']['compound'])
                confidences.append(min(1.0, compound * 2))
            
            # TextBlob confidence (based on polarity magnitude)
            if 'textblob' in sentiment_scores:
                polarity = abs(sentiment_scores['textblob']['polarity'])
                confidences.append(polarity)
            
            # Pattern confidence
            if 'pattern' in sentiment_scores:
                confidences.append(sentiment_scores['pattern']['confidence'])
            
            return float(np.mean(confidences)) if confidences else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return 0.0
    
    def get_news_sentiment(self, symbol: str, days: int = 7) -> Dict[str, Any]:
        """
        Get sentiment analysis from financial news
        
        Args:
            symbol: Stock symbol
            days: Number of days to look back
        
        Returns:
            Dictionary with news sentiment analysis
        """
        try:
            performance_tracker.start_timer(f"news_sentiment_{symbol}")
            
            news_articles = []
            
            # Alpha Vantage News (if API key available)
            if self.api_keys.get('alpha_vantage'):
                av_articles = self._get_alpha_vantage_news(symbol, days)
                news_articles.extend(av_articles)
            
            # Yahoo Finance News (free, no API key required)
            yahoo_articles = self._get_yahoo_news(symbol, days)
            news_articles.extend(yahoo_articles)
            
            if not news_articles:
                return {'error': 'No news articles found'}
            
            # Analyze sentiment for each article
            analyzed_articles = []
            for article in news_articles:
                title_sentiment = self.analyze_text_sentiment(article.get('title', ''))
                summary_sentiment = self.analyze_text_sentiment(article.get('summary', ''))
                
                # Combine title and summary sentiment (title weighted more heavily)
                if 'error' not in title_sentiment and 'error' not in summary_sentiment:
                    combined_score = (
                        0.7 * title_sentiment['composite_sentiment']['score'] +
                        0.3 * summary_sentiment['composite_sentiment']['score']
                    )
                    combined_confidence = (
                        title_sentiment['confidence'] + summary_sentiment['confidence']
                    ) / 2
                elif 'error' not in title_sentiment:
                    combined_score = title_sentiment['composite_sentiment']['score']
                    combined_confidence = title_sentiment['confidence']
                elif 'error' not in summary_sentiment:
                    combined_score = summary_sentiment['composite_sentiment']['score']
                    combined_confidence = summary_sentiment['confidence']
                else:
                    combined_score = 0.0
                    combined_confidence = 0.0
                
                analyzed_articles.append({
                    'title': article.get('title', ''),
                    'url': article.get('url', ''),
                    'published_date': article.get('published_date', ''),
                    'source': article.get('source', ''),
                    'sentiment_score': combined_score,
                    'confidence': combined_confidence,
                    'title_sentiment': title_sentiment,
                    'summary_sentiment': summary_sentiment
                })
            
            # Calculate overall sentiment metrics
            scores = [a['sentiment_score'] for a in analyzed_articles if a['confidence'] > 0.1]
            confidences = [a['confidence'] for a in analyzed_articles]
            
            if scores:
                avg_sentiment = np.mean(scores)
                sentiment_std = np.std(scores)
                positive_articles = len([s for s in scores if s > 0.1])
                negative_articles = len([s for s in scores if s < -0.1])
                neutral_articles = len(scores) - positive_articles - negative_articles
                
                # Determine overall sentiment label
                if avg_sentiment > 0.1:
                    overall_label = 'positive'
                elif avg_sentiment < -0.1:
                    overall_label = 'negative'
                else:
                    overall_label = 'neutral'
            else:
                avg_sentiment = 0.0
                sentiment_std = 0.0
                positive_articles = negative_articles = neutral_articles = 0
                overall_label = 'neutral'
            
            duration = performance_tracker.end_timer(f"news_sentiment_{symbol}")
            
            return {
                'symbol': symbol,
                'analysis_period_days': days,
                'total_articles': len(news_articles),
                'analyzed_articles': len(analyzed_articles),
                'overall_sentiment': {
                    'score': float(avg_sentiment),
                    'label': overall_label,
                    'std': float(sentiment_std),
                    'confidence': float(np.mean(confidences)) if confidences else 0.0
                },
                'sentiment_distribution': {
                    'positive_articles': positive_articles,
                    'negative_articles': negative_articles,
                    'neutral_articles': neutral_articles
                },
                'articles': analyzed_articles,
                'analysis_duration': duration
            }
            
        except Exception as e:
            performance_tracker.end_timer(f"news_sentiment_{symbol}")
            logger.error(f"Error getting news sentiment: {str(e)}")
            return {'error': str(e)}
    
    def _get_alpha_vantage_news(self, symbol: str, days: int) -> List[Dict[str, Any]]:
        """Get news from Alpha Vantage API"""
        try:
            if not self.api_keys.get('alpha_vantage'):
                return []
            
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': symbol,
                'apikey': self.api_keys['alpha_vantage'],
                'limit': 50,
                'time_from': (datetime.now() - timedelta(days=days)).strftime('%Y%m%dT%H%M')
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if 'feed' not in data:
                logger.warning("No news feed found in Alpha Vantage response")
                return []
            
            articles = []
            for item in data['feed']:
                articles.append({
                    'title': item.get('title', ''),
                    'url': item.get('url', ''),
                    'summary': item.get('summary', ''),
                    'published_date': item.get('time_published', ''),
                    'source': 'Alpha Vantage',
                    'authors': item.get('authors', [])
                })
            
            # Respect API rate limits
            time.sleep(12)  # Alpha Vantage free tier: 5 calls per minute
            
            return articles
            
        except Exception as e:
            logger.error(f"Error getting Alpha Vantage news: {str(e)}")
            return []
    
    def _get_yahoo_news(self, symbol: str, days: int) -> List[Dict[str, Any]]:
        """Get news from Yahoo Finance (scraping approach)"""
        try:
            # This is a simplified approach - in practice, you'd use a proper news API
            # or web scraping library like BeautifulSoup
            
            # For demonstration, we'll return mock news data
            mock_articles = [
                {
                    'title': f'{symbol} Reports Strong Quarterly Earnings',
                    'url': f'https://finance.yahoo.com/news/{symbol.lower()}-earnings',
                    'summary': f'{symbol} exceeded analyst expectations with strong revenue growth.',
                    'published_date': (datetime.now() - timedelta(days=1)).isoformat(),
                    'source': 'Yahoo Finance'
                },
                {
                    'title': f'Analyst Upgrades {symbol} Price Target',
                    'url': f'https://finance.yahoo.com/news/{symbol.lower()}-upgrade',
                    'summary': f'Investment firm raises price target for {symbol} citing strong fundamentals.',
                    'published_date': (datetime.now() - timedelta(days=2)).isoformat(),
                    'source': 'Yahoo Finance'
                }
            ]
            
            return mock_articles
            
        except Exception as e:
            logger.error(f"Error getting Yahoo news: {str(e)}")
            return []
    
    def get_social_sentiment(self, symbol: str, days: int = 7) -> Dict[str, Any]:
        """
        Get sentiment analysis from social media
        
        Args:
            symbol: Stock symbol
            days: Number of days to look back
        
        Returns:
            Dictionary with social media sentiment analysis
        """
        try:
            performance_tracker.start_timer(f"social_sentiment_{symbol}")
            
            social_data = []
            
            # Reddit sentiment (if credentials available)
            if self.api_keys.get('reddit_client_id') and self.api_keys.get('reddit_client_secret'):
                reddit_data = self._get_reddit_sentiment(symbol, days)
                social_data.extend(reddit_data)
            
            # Twitter sentiment (if API key available)
            if self.api_keys.get('twitter_bearer'):
                twitter_data = self._get_twitter_sentiment(symbol, days)
                social_data.extend(twitter_data)
            
            if not social_data:
                return {'error': 'No social media data found or APIs not configured'}
            
            # Analyze sentiment for each post
            analyzed_posts = []
            for post in social_data:
                sentiment = self.analyze_text_sentiment(post.get('text', ''))
                
                if 'error' not in sentiment:
                    analyzed_posts.append({
                        'text': post.get('text', ''),
                        'platform': post.get('platform', ''),
                        'created_date': post.get('created_date', ''),
                        'author': post.get('author', ''),
                        'engagement': post.get('engagement', {}),
                        'sentiment_score': sentiment['composite_sentiment']['score'],
                        'confidence': sentiment['confidence'],
                        'sentiment_details': sentiment
                    })
            
            # Calculate overall social sentiment
            scores = [p['sentiment_score'] for p in analyzed_posts if p['confidence'] > 0.1]
            confidences = [p['confidence'] for p in analyzed_posts]
            
            if scores:
                avg_sentiment = np.mean(scores)
                sentiment_std = np.std(scores)
                positive_posts = len([s for s in scores if s > 0.1])
                negative_posts = len([s for s in scores if s < -0.1])
                neutral_posts = len(scores) - positive_posts - negative_posts
                
                # Determine overall sentiment label
                if avg_sentiment > 0.1:
                    overall_label = 'positive'
                elif avg_sentiment < -0.1:
                    overall_label = 'negative'
                else:
                    overall_label = 'neutral'
            else:
                avg_sentiment = 0.0
                sentiment_std = 0.0
                positive_posts = negative_posts = neutral_posts = 0
                overall_label = 'neutral'
            
            duration = performance_tracker.end_timer(f"social_sentiment_{symbol}")
            
            return {
                'symbol': symbol,
                'analysis_period_days': days,
                'total_posts': len(social_data),
                'analyzed_posts': len(analyzed_posts),
                'overall_sentiment': {
                    'score': float(avg_sentiment),
                    'label': overall_label,
                    'std': float(sentiment_std),
                    'confidence': float(np.mean(confidences)) if confidences else 0.0
                },
                'sentiment_distribution': {
                    'positive_posts': positive_posts,
                    'negative_posts': negative_posts,
                    'neutral_posts': neutral_posts
                },
                'platform_breakdown': self._calculate_platform_breakdown(analyzed_posts),
                'posts': analyzed_posts[:20],  # Return top 20 posts
                'analysis_duration': duration
            }
            
        except Exception as e:
            performance_tracker.end_timer(f"social_sentiment_{symbol}")
            logger.error(f"Error getting social sentiment: {str(e)}")
            return {'error': str(e)}
    
    def _get_reddit_sentiment(self, symbol: str, days: int) -> List[Dict[str, Any]]:
        """Get sentiment data from Reddit"""
        try:
            # This would require Reddit API integration
            # For demonstration, return mock data
            mock_posts = [
                {
                    'text': f'Bullish on {symbol}! Great fundamentals and strong growth potential.',
                    'platform': 'reddit',
                    'created_date': (datetime.now() - timedelta(days=1)).isoformat(),
                    'author': 'reddit_user_1',
                    'engagement': {'upvotes': 45, 'comments': 12}
                },
                {
                    'text': f'Not sure about {symbol} - seems overvalued at current levels.',
                    'platform': 'reddit',
                    'created_date': (datetime.now() - timedelta(days=2)).isoformat(),
                    'author': 'reddit_user_2',
                    'engagement': {'upvotes': 23, 'comments': 8}
                }
            ]
            
            return mock_posts
            
        except Exception as e:
            logger.error(f"Error getting Reddit sentiment: {str(e)}")
            return []
    
    def _get_twitter_sentiment(self, symbol: str, days: int) -> List[Dict[str, Any]]:
        """Get sentiment data from Twitter"""
        try:
            # This would require Twitter API integration
            # For demonstration, return mock data
            mock_tweets = [
                {
                    'text': f'${symbol} breaking out! Strong volume and momentum. #bullish',
                    'platform': 'twitter',
                    'created_date': (datetime.now() - timedelta(hours=6)).isoformat(),
                    'author': 'trader123',
                    'engagement': {'likes': 15, 'retweets': 8, 'replies': 3}
                },
                {
                    'text': f'Concerned about ${symbol} guidance. May see some selling pressure.',
                    'platform': 'twitter',
                    'created_date': (datetime.now() - timedelta(hours=12)).isoformat(),
                    'author': 'analyst_views',
                    'engagement': {'likes': 7, 'retweets': 2, 'replies': 5}
                }
            ]
            
            return mock_tweets
            
        except Exception as e:
            logger.error(f"Error getting Twitter sentiment: {str(e)}")
            return []
    
    def _calculate_platform_breakdown(self, analyzed_posts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate sentiment breakdown by platform"""
        try:
            platform_data = {}
            
            for post in analyzed_posts:
                platform = post.get('platform', 'unknown')
                
                if platform not in platform_data:
                    platform_data[platform] = {
                        'count': 0,
                        'scores': [],
                        'confidences': []
                    }
                
                platform_data[platform]['count'] += 1
                platform_data[platform]['scores'].append(post['sentiment_score'])
                platform_data[platform]['confidences'].append(post['confidence'])
            
            # Calculate platform summaries
            platform_summary = {}
            for platform, data in platform_data.items():
                scores = data['scores']
                if scores:
                    platform_summary[platform] = {
                        'posts': data['count'],
                        'avg_sentiment': float(np.mean(scores)),
                        'sentiment_std': float(np.std(scores)),
                        'avg_confidence': float(np.mean(data['confidences'])),
                        'positive_posts': len([s for s in scores if s > 0.1]),
                        'negative_posts': len([s for s in scores if s < -0.1]),
                        'neutral_posts': len([s for s in scores if -0.1 <= s <= 0.1])
                    }
            
            return platform_summary
            
        except Exception as e:
            logger.error(f"Error calculating platform breakdown: {str(e)}")
            return {}
    
    def get_comprehensive_sentiment(self, symbol: str, days: int = 7) -> Dict[str, Any]:
        """
        Get comprehensive sentiment analysis combining news and social media
        
        Args:
            symbol: Stock symbol
            days: Number of days to look back
        
        Returns:
            Dictionary with comprehensive sentiment analysis
        """
        try:
            performance_tracker.start_timer(f"comprehensive_sentiment_{symbol}")
            
            # Get news sentiment
            news_sentiment = self.get_news_sentiment(symbol, days)
            
            # Get social media sentiment
            social_sentiment = self.get_social_sentiment(symbol, days)
            
            # Combine sentiments
            combined_sentiment = self._combine_sentiments(news_sentiment, social_sentiment)
            
            duration = performance_tracker.end_timer(f"comprehensive_sentiment_{symbol}")
            
            return {
                'symbol': symbol,
                'analysis_period_days': days,
                'news_sentiment': news_sentiment,
                'social_sentiment': social_sentiment,
                'combined_sentiment': combined_sentiment,
                'analysis_duration': duration,
                'sentiment_score': combined_sentiment.get('overall_score', 0.0),
                'sentiment_label': combined_sentiment.get('overall_label', 'neutral'),
                'confidence': combined_sentiment.get('confidence', 0.0)
            }
            
        except Exception as e:
            performance_tracker.end_timer(f"comprehensive_sentiment_{symbol}")
            logger.error(f"Error getting comprehensive sentiment: {str(e)}")
            return {'error': str(e)}
    
    def _combine_sentiments(self, news_sentiment: Dict[str, Any], 
                           social_sentiment: Dict[str, Any]) -> Dict[str, Any]:
        """Combine news and social media sentiments"""
        try:
            scores = []
            weights = []
            confidences = []
            
            # News sentiment (weight: 0.6 - generally more reliable)
            if 'error' not in news_sentiment and news_sentiment.get('overall_sentiment'):
                news_score = news_sentiment['overall_sentiment']['score']
                news_confidence = news_sentiment['overall_sentiment']['confidence']
                scores.append(news_score)
                weights.append(0.6)
                confidences.append(news_confidence)
            
            # Social sentiment (weight: 0.4 - more volatile but timely)
            if 'error' not in social_sentiment and social_sentiment.get('overall_sentiment'):
                social_score = social_sentiment['overall_sentiment']['score']
                social_confidence = social_sentiment['overall_sentiment']['confidence']
                scores.append(social_score)
                weights.append(0.4)
                confidences.append(social_confidence)
            
            if not scores:
                return {
                    'overall_score': 0.0,
                    'overall_label': 'neutral',
                    'confidence': 0.0,
                    'data_sources': 0
                }
            
            # Calculate weighted average
            weighted_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
            
            # Determine label
            if weighted_score > 0.1:
                label = 'positive'
            elif weighted_score < -0.1:
                label = 'negative'
            else:
                label = 'neutral'
            
            # Calculate combined confidence
            combined_confidence = np.mean(confidences) if confidences else 0.0
            
            return {
                'overall_score': float(weighted_score),
                'overall_label': label,
                'confidence': float(combined_confidence),
                'data_sources': len(scores),
                'news_weight': 0.6 if len(scores) == 2 else (1.0 if 'error' not in news_sentiment else 0.0),
                'social_weight': 0.4 if len(scores) == 2 else (1.0 if 'error' not in social_sentiment else 0.0)
            }
            
        except Exception as e:
            logger.error(f"Error combining sentiments: {str(e)}")
            return {'overall_score': 0.0, 'overall_label': 'neutral', 'confidence': 0.0}

# Create global sentiment analyzer instance
sentiment_analyzer = SentimentAnalyzer()