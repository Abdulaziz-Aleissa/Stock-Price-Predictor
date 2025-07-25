# How to Get Real News Articles

Currently, the application is set up to show **real news articles** from Alpha Vantage, but it requires an API key to function.

## Quick Setup (5 minutes)

### Step 1: Get Your Free API Key
1. Visit: https://www.alphavantage.co/support/#api-key
2. Click "Get your free API key today"
3. Fill out the simple form (no credit card required)
4. Copy your API key

### Step 2: Configure Your API Key
1. In the project folder, copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Open the `.env` file and replace `your_alpha_vantage_api_key_here` with your actual API key:
   ```env
   ALPHA_VANTAGE_API_KEY=YOUR_ACTUAL_API_KEY_HERE
   ```

3. Save the file

### Step 3: Restart the Application
Restart your Flask application to load the new API key.

## What You Get With Real News
- ✅ **Live news articles** from major financial sources (Reuters, Bloomberg, MarketWatch, etc.)
- ✅ **Real sentiment analysis** for each stock
- ✅ **Current publication dates** (not demo data)
- ✅ **Actual article links** to full stories
- ✅ **25 free API calls per day** (more than enough for personal use)

## Troubleshooting
- **No articles showing?** Check that your `.env` file is in the root directory and contains the correct API key
- **"API key required" message?** Make sure you've restarted the Flask application after adding the API key
- **Need more requests?** Alpha Vantage offers paid plans with higher limits

## Free Tier Limits
- 25 API requests per day
- Perfect for personal stock analysis
- No credit card required

---
*Once configured, you'll see real, up-to-date news articles instead of demo content!*