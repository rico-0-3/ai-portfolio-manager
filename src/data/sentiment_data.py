"""
Sentiment data fetcher.
Collects and analyzes news and social media sentiment for stocks.
"""

import requests
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import logging
from pathlib import Path
import re

logger = logging.getLogger(__name__)


class SentimentDataFetcher:
    """Fetch news and sentiment data for stocks."""

    def __init__(
        self,
        news_api_key: Optional[str] = None,
        cache_dir: str = "data/raw/sentiment"
    ):
        """
        Initialize sentiment data fetcher.

        Args:
            news_api_key: News API key (from newsapi.org)
            cache_dir: Directory to cache sentiment data
        """
        self.news_api_key = news_api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info("SentimentDataFetcher initialized")

    def fetch_news(
        self,
        ticker: str,
        company_name: Optional[str] = None,
        days_back: int = 7
    ) -> List[Dict]:
        """
        Fetch news articles for a stock.

        Args:
            ticker: Stock ticker symbol
            company_name: Company name for better search results
            days_back: Number of days to look back

        Returns:
            List of news articles with metadata
        """
        if not self.news_api_key or self.news_api_key == 'YOUR_NEWS_API_KEY':
            logger.warning("News API key not configured")
            return []

        logger.info(f"Fetching news for {ticker}")

        # Prepare search query
        query = ticker if not company_name else f"{ticker} OR {company_name}"

        # Calculate date range
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days_back)

        url = "https://newsapi.org/v2/everything"
        params = {
            'q': query,
            'from': from_date.strftime('%Y-%m-%d'),
            'to': to_date.strftime('%Y-%m-%d'),
            'language': 'en',
            'sortBy': 'relevancy',
            'apiKey': self.news_api_key
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if data['status'] == 'ok':
                articles = data.get('articles', [])
                logger.info(f"Found {len(articles)} articles for {ticker}")
                return articles
            else:
                logger.error(f"News API error: {data.get('message', 'Unknown')}")
                return []

        except Exception as e:
            logger.error(f"Error fetching news for {ticker}: {e}")
            return []

    def fetch_financial_news(
        self,
        tickers: List[str],
        days_back: int = 7
    ) -> pd.DataFrame:
        """
        Fetch news for multiple tickers and return as DataFrame.

        Args:
            tickers: List of stock tickers
            days_back: Number of days to look back

        Returns:
            DataFrame with news articles
        """
        all_articles = []

        for ticker in tickers:
            articles = self.fetch_news(ticker, days_back=days_back)

            for article in articles:
                all_articles.append({
                    'ticker': ticker,
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'content': article.get('content', ''),
                    'source': article.get('source', {}).get('name', 'Unknown'),
                    'author': article.get('author', 'Unknown'),
                    'published_at': article.get('publishedAt', ''),
                    'url': article.get('url', '')
                })

        df = pd.DataFrame(all_articles)

        if not df.empty:
            df['published_at'] = pd.to_datetime(df['published_at'])
            df = df.sort_values('published_at', ascending=False)

        logger.info(f"Fetched {len(df)} total articles for {len(tickers)} tickers")
        return df

    def get_market_sentiment_keywords(self) -> Dict[str, List[str]]:
        """
        Get predefined keywords for sentiment classification.

        Returns:
            Dictionary with positive, negative, and neutral keywords
        """
        return {
            'positive': [
                'surge', 'soar', 'rally', 'gain', 'rise', 'jump', 'climb',
                'outperform', 'beat', 'exceed', 'strong', 'growth', 'profit',
                'bullish', 'upgrade', 'buy', 'optimistic', 'opportunity',
                'innovation', 'expansion', 'record', 'breakthrough'
            ],
            'negative': [
                'plunge', 'crash', 'fall', 'drop', 'decline', 'loss', 'sink',
                'underperform', 'miss', 'weak', 'bearish', 'downgrade', 'sell',
                'concern', 'risk', 'warning', 'challenge', 'threat', 'crisis',
                'debt', 'lawsuit', 'investigation', 'bankruptcy', 'layoff'
            ],
            'neutral': [
                'stable', 'unchanged', 'flat', 'maintain', 'hold', 'continue',
                'steady', 'consistent', 'moderate', 'expected'
            ]
        }

    def simple_sentiment_score(self, text: str) -> float:
        """
        Calculate a simple sentiment score based on keywords.

        This is a basic implementation. For production, use FinBERT or VADER.

        Args:
            text: Text to analyze

        Returns:
            Sentiment score between -1 (negative) and 1 (positive)
        """
        if not text:
            return 0.0

        text = text.lower()
        keywords = self.get_market_sentiment_keywords()

        positive_count = sum(1 for word in keywords['positive'] if word in text)
        negative_count = sum(1 for word in keywords['negative'] if word in text)

        total_count = positive_count + negative_count

        if total_count == 0:
            return 0.0

        score = (positive_count - negative_count) / total_count
        return score

    def analyze_news_sentiment(self, articles_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze sentiment for news articles.

        Args:
            articles_df: DataFrame with news articles

        Returns:
            DataFrame with sentiment scores added
        """
        if articles_df.empty:
            return articles_df

        logger.info("Analyzing sentiment for articles")

        # Combine title and description for analysis
        articles_df['full_text'] = (
            articles_df['title'].fillna('') + ' ' +
            articles_df['description'].fillna('')
        )

        # Calculate sentiment scores
        articles_df['sentiment_score'] = articles_df['full_text'].apply(
            self.simple_sentiment_score
        )

        # Categorize sentiment
        articles_df['sentiment_category'] = articles_df['sentiment_score'].apply(
            lambda x: 'positive' if x > 0.2 else ('negative' if x < -0.2 else 'neutral')
        )

        return articles_df

    def aggregate_ticker_sentiment(
        self,
        articles_df: pd.DataFrame,
        ticker: str
    ) -> Dict:
        """
        Aggregate sentiment for a specific ticker.

        Args:
            articles_df: DataFrame with analyzed articles
            ticker: Stock ticker

        Returns:
            Dictionary with aggregated sentiment metrics
        """
        ticker_articles = articles_df[articles_df['ticker'] == ticker]

        if ticker_articles.empty:
            return {
                'ticker': ticker,
                'article_count': 0,
                'avg_sentiment': 0.0,
                'positive_pct': 0.0,
                'negative_pct': 0.0,
                'neutral_pct': 0.0
            }

        total = len(ticker_articles)
        positive = len(ticker_articles[ticker_articles['sentiment_category'] == 'positive'])
        negative = len(ticker_articles[ticker_articles['sentiment_category'] == 'negative'])
        neutral = len(ticker_articles[ticker_articles['sentiment_category'] == 'neutral'])

        return {
            'ticker': ticker,
            'article_count': total,
            'avg_sentiment': ticker_articles['sentiment_score'].mean(),
            'positive_pct': (positive / total) * 100,
            'negative_pct': (negative / total) * 100,
            'neutral_pct': (neutral / total) * 100
        }

    def get_sentiment_summary(self, articles_df: pd.DataFrame) -> pd.DataFrame:
        """
        Get sentiment summary for all tickers.

        Args:
            articles_df: DataFrame with analyzed articles

        Returns:
            DataFrame with sentiment summary per ticker
        """
        tickers = articles_df['ticker'].unique()
        summaries = []

        for ticker in tickers:
            summary = self.aggregate_ticker_sentiment(articles_df, ticker)
            summaries.append(summary)

        summary_df = pd.DataFrame(summaries)
        summary_df = summary_df.sort_values('avg_sentiment', ascending=False)

        return summary_df
