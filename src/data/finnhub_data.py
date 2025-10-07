"""
Finnhub data fetcher.
Provides real-time market data and additional metrics.
"""

import requests
import pandas as pd
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta
import time

logger = logging.getLogger(__name__)


class FinnhubDataFetcher:
    """Fetch real-time data from Finnhub API."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Finnhub data fetcher.

        Args:
            api_key: Finnhub API key (get free at https://finnhub.io)
        """
        self.api_key = api_key
        self.base_url = "https://finnhub.io/api/v1"
        self.rate_limit_delay = 1.0  # 1 second delay for free tier

        if not api_key:
            logger.warning("Finnhub API key not provided. Additional data will not be available.")

    def get_quote(self, ticker: str) -> Dict:
        """
        Get real-time quote.

        Args:
            ticker: Stock ticker

        Returns:
            Dictionary with quote data
        """
        if not self.api_key:
            return {}

        try:
            url = f"{self.base_url}/quote"
            params = {
                'symbol': ticker,
                'token': self.api_key
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            time.sleep(self.rate_limit_delay)

            data = response.json()
            if data:
                logger.info(f"Fetched quote for {ticker}")
                return data
            else:
                return {}

        except requests.exceptions.HTTPError as e:
            if '403' in str(e):
                logger.debug(f"Finnhub API access forbidden for {ticker} (check API key)")
            else:
                logger.error(f"Error fetching quote for {ticker}: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error fetching quote for {ticker}: {e}")
            return {}

    def get_recommendation_trends(self, ticker: str) -> List[Dict]:
        """
        Get analyst recommendation trends.

        Args:
            ticker: Stock ticker

        Returns:
            List of recommendation trends
        """
        if not self.api_key:
            return []

        try:
            url = f"{self.base_url}/stock/recommendation"
            params = {
                'symbol': ticker,
                'token': self.api_key
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            time.sleep(self.rate_limit_delay)

            data = response.json()
            if data:
                logger.info(f"Fetched {len(data)} recommendation trends for {ticker}")
                return data
            else:
                return []

        except requests.exceptions.HTTPError as e:
            if '403' in str(e):
                logger.debug(f"Finnhub API access forbidden for {ticker} (check API key)")
            else:
                logger.error(f"Error fetching recommendations for {ticker}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error fetching recommendations for {ticker}: {e}")
            return []

    def get_price_target(self, ticker: str) -> Dict:
        """
        Get analyst price targets (PREMIUM - requires paid account).

        NOTE: This function is disabled for free accounts.
        Use recommendation_trends instead which is free.

        Args:
            ticker: Stock ticker

        Returns:
            Empty dict (Premium API not available)
        """
        logger.debug(f"Price target API is Premium - skipping for {ticker}")
        return {}

    def get_basic_financials(self, ticker: str) -> Dict:
        """
        Get basic financial metrics (FREE API).

        Args:
            ticker: Stock ticker

        Returns:
            Dictionary with basic financials
        """
        if not self.api_key:
            return {}

        try:
            url = f"{self.base_url}/stock/metric"
            params = {
                'symbol': ticker,
                'metric': 'all',
                'token': self.api_key
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            time.sleep(self.rate_limit_delay)

            data = response.json()
            if data:
                logger.info(f"Fetched basic financials for {ticker}")
                return data
            else:
                return {}

        except requests.exceptions.HTTPError as e:
            if '403' in str(e):
                logger.debug(f"Finnhub API access forbidden for {ticker} (check API key)")
            else:
                logger.error(f"Error fetching basic financials for {ticker}: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error fetching basic financials for {ticker}: {e}")
            return {}

    def get_earnings_surprises(self, ticker: str, limit: int = 4) -> List[Dict]:
        """
        Get EPS surprises (FREE API).

        Args:
            ticker: Stock ticker
            limit: Number of quarters to fetch

        Returns:
            List of earnings surprises
        """
        if not self.api_key:
            return []

        try:
            url = f"{self.base_url}/stock/earnings"
            params = {
                'symbol': ticker,
                'token': self.api_key
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            time.sleep(self.rate_limit_delay)

            data = response.json()
            if data:
                logger.info(f"Fetched earnings surprises for {ticker}")
                return data[:limit]
            else:
                return []

        except requests.exceptions.HTTPError as e:
            if '403' in str(e):
                logger.debug(f"Finnhub API access forbidden for {ticker} (check API key)")
            else:
                logger.error(f"Error fetching earnings surprises for {ticker}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error fetching earnings surprises for {ticker}: {e}")
            return []

    def get_insider_sentiment(self, ticker: str) -> Dict:
        """
        Get insider sentiment (FREE API).

        Args:
            ticker: Stock ticker

        Returns:
            Dictionary with insider sentiment data
        """
        if not self.api_key:
            return {}

        try:
            url = f"{self.base_url}/stock/insider-sentiment"
            params = {
                'symbol': ticker,
                'token': self.api_key
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            time.sleep(self.rate_limit_delay)

            data = response.json()
            if data and 'data' in data and len(data['data']) > 0:
                logger.info(f"Fetched insider sentiment for {ticker}")
                return data['data'][0]  # Most recent
            else:
                return {}

        except requests.exceptions.HTTPError as e:
            if '403' in str(e):
                logger.debug(f"Finnhub API access forbidden for {ticker} (check API key)")
            else:
                logger.error(f"Error fetching insider sentiment for {ticker}: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error fetching insider sentiment for {ticker}: {e}")
            return {}

    def get_analyst_features(self, ticker: str) -> Dict[str, float]:
        """
        Get analyst-based features for ML (all FREE APIs).

        Args:
            ticker: Stock ticker

        Returns:
            Dictionary of analyst features
        """
        features = {}

        # 1. Recommendation Trends (FREE)
        trends = self.get_recommendation_trends(ticker)
        if trends and len(trends) > 0:
            latest = trends[0]
            features['buy_signals'] = latest.get('buy', 0)
            features['hold_signals'] = latest.get('hold', 0)
            features['sell_signals'] = latest.get('sell', 0)
            features['strong_buy_signals'] = latest.get('strongBuy', 0)
            features['strong_sell_signals'] = latest.get('strongSell', 0)

            # Calculate consensus score (-1 to 1)
            total_signals = (
                features['strong_buy_signals'] +
                features['buy_signals'] +
                features['hold_signals'] +
                features['sell_signals'] +
                features['strong_sell_signals']
            )

            if total_signals > 0:
                consensus = (
                    features['strong_buy_signals'] * 1.0 +
                    features['buy_signals'] * 0.5 +
                    features['hold_signals'] * 0.0 +
                    features['sell_signals'] * -0.5 +
                    features['strong_sell_signals'] * -1.0
                ) / total_signals
                features['analyst_consensus'] = consensus

        # 2. Basic Financials (FREE - HIGH USAGE)
        financials = self.get_basic_financials(ticker)
        if financials and 'metric' in financials:
            metrics = financials['metric']
            # Key valuation metrics
            features['pe_ratio'] = metrics.get('peBasicExclExtraTTM', 0)
            features['pb_ratio'] = metrics.get('pbQuarterly', 0)
            features['ps_ratio'] = metrics.get('psQuarterly', 0)
            features['price_to_cash_flow'] = metrics.get('pfcfShareTTM', 0)

            # Profitability metrics
            features['roe'] = metrics.get('roeRfy', 0) or metrics.get('roeTTM', 0)
            features['roa'] = metrics.get('roaRfy', 0) or metrics.get('roaTTM', 0)
            features['profit_margin'] = metrics.get('netProfitMarginTTM', 0)

            # Growth metrics
            features['revenue_growth'] = metrics.get('revenueGrowthTTMYoy', 0)
            features['eps_growth'] = metrics.get('epsGrowthTTMYoy', 0)

            # Dividend metrics
            features['dividend_yield'] = metrics.get('dividendYieldIndicatedAnnual', 0)
            features['payout_ratio'] = metrics.get('payoutRatioTTM', 0)

            # Financial health
            features['current_ratio'] = metrics.get('currentRatioQuarterly', 0)
            features['debt_to_equity'] = metrics.get('totalDebt/totalEquityQuarterly', 0)
            features['quick_ratio'] = metrics.get('quickRatioQuarterly', 0)

        # 3. EPS Surprises (FREE - HIGH USAGE)
        surprises = self.get_earnings_surprises(ticker, limit=4)
        if surprises:
            # Calculate average surprise percentage
            surprise_pcts = []
            for surprise in surprises:
                actual = surprise.get('actual', 0)
                estimate = surprise.get('estimate', 0)
                if estimate and estimate != 0:
                    surprise_pct = (actual - estimate) / abs(estimate)
                    surprise_pcts.append(surprise_pct)

            if surprise_pcts:
                features['avg_eps_surprise'] = sum(surprise_pcts) / len(surprise_pcts)
                features['recent_eps_surprise'] = surprise_pcts[0]  # Most recent
                # Positive surprise rate
                positive_count = sum(1 for s in surprise_pcts if s > 0)
                features['positive_surprise_rate'] = positive_count / len(surprise_pcts)

        # 4. Insider Sentiment (FREE)
        insider = self.get_insider_sentiment(ticker)
        if insider:
            # Insider sentiment score
            mspr = insider.get('mspr', 0)  # Monthly share purchase ratio
            change = insider.get('change', 0)  # Net change in shares
            features['insider_mspr'] = mspr
            features['insider_change'] = change

            # Positive if insiders are buying (mspr > 0, change > 0)
            if mspr > 0 and change > 0:
                features['insider_bullish'] = 1
            elif mspr < 0 and change < 0:
                features['insider_bullish'] = 0
            else:
                features['insider_bullish'] = 0.5

        return features

    def enrich_with_analyst_data(
        self,
        tickers: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Enrich tickers with analyst data.

        Args:
            tickers: List of stock tickers

        Returns:
            Dictionary mapping ticker to analyst features
        """
        analyst_data = {}

        for ticker in tickers:
            logger.info(f"Fetching analyst data for {ticker}...")
            analyst_data[ticker] = self.get_analyst_features(ticker)
            time.sleep(self.rate_limit_delay)

        return analyst_data
