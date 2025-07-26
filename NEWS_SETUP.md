# News Feature Setup Guide

This guide explains how to set up the news feature for the Stock Price Predictor application.

## Alpha Vantage API Setup

The news feature uses the Alpha Vantage News & Sentiment API to fetch relevant stock news articles.

### 1. Get Your API Key

1. Visit [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
2. Click "Get your free API key today"
3. Fill out the form with your details
4. You'll receive a free API key that allows:
   - 25 API requests per day (free tier)
   - Access to real-time news and sentiment data
   - No credit card required

### 2. Environment Configuration

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit the `.env` file and add your API key:
   ```env
   # Alpha Vantage API Configuration
   ALPHA_VANTAGE_API_KEY=your_actual_api_key_here
   
   # Flask Configuration
   SECRET_KEY=your-secret-key-here
   ```

3. **Important**: Never commit your `.env` file to version control. It's already included in `.gitignore`.

### 3. Testing the Setup

You can test the news API functionality:

```bash
# Run the test script
python3 test_news_api.py
```

If no API key is configured, the system will show a clear message indicating that an API key is required for real news articles.

**Important**: To get real, live news articles instead of demo data, you must configure a valid Alpha Vantage API key.

## Features

### News Integration
- **Real-time News**: Fetches the latest news articles for any stock ticker
- **Sentiment Analysis**: Shows sentiment scores (Bullish, Bearish, Neutral) for each article
- **Source Attribution**: Displays article source, publication date, and direct links
- **Caching**: Implements 30-minute caching to reduce API calls and improve performance

### UI Components
- **Collapsible Section**: Users can hide/show news to focus on predictions
- **Sentiment Overview**: Quick summary showing article count and overall sentiment
- **Responsive Design**: Optimized for both desktop and mobile devices
- **Theme Support**: Matches the existing dark/light theme system

### Error Handling
- **API Failures**: Graceful fallback when news service is unavailable
- **Rate Limiting**: Respects API rate limits (1 request per second)
- **Mock Data**: Automatic fallback to demo data when API key is missing

## API Rate Limits

The free tier of Alpha Vantage includes:
- **25 requests per day**
- **1 request per second**

The application automatically handles rate limiting and caching to optimize API usage.

## Upgrading API Plan

For production use or higher volume, consider upgrading to a paid Alpha Vantage plan:
- Standard: 75 API requests/day
- Premium: 1,200 API requests/day
- Professional: Unlimited requests

Visit [Alpha Vantage Pricing](https://www.alphavantage.co/premium/) for more details.

## Troubleshooting

### Common Issues

1. **"No news articles found"**
   - Check if your API key is valid
   - Verify the stock ticker exists
   - Check if you've exceeded daily API limits

2. **"API key not found. Using mock data."**
   - Ensure `.env` file exists with correct API key
   - Restart the Flask application after adding the key

3. **Slow news loading**
   - First request takes longer as it fetches fresh data
   - Subsequent requests use cached data for better performance

### Log Messages

The application logs news-related activity:
- News fetching attempts
- API errors or rate limiting
- Cache hits/misses
- Fallback to mock data

Check the application logs for detailed debugging information.

## Development Notes

### Mock Data
When no API key is provided, the system uses realistic mock data that includes:
- Sample news articles with proper formatting
- Realistic sentiment scores and labels
- Various news sources (MarketWatch, Yahoo Finance, Reuters, etc.)
- Different publication dates

This allows developers to work on the UI and functionality without requiring an API key.

### Customization
The news feature can be customized by modifying:
- `app/utils/news_api.py`: API client and caching logic
- `app/templates/go.html`: News section HTML structure
- `app/static/css/styles.css`: News styling and responsive design