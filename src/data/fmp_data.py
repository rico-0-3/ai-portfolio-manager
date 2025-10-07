"""
Financial Modeling Prep (FMP) data fetcher.
Provides fundamental data for stocks.
"""

import requests
import pandas as pd
from typing import Dict, List, Optional
import logging
from datetime import datetime
import time

logger = logging.getLogger(__name__)


class FMPDataFetcher:
    """Fetch fundamental data from Financial Modeling Prep API."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize FMP data fetcher.

        Args:
            api_key: FMP API key (get free at https://financialmodelingprep.com)
        """
        self.api_key = api_key
        self.base_url = "https://financialmodelingprep.com/api/v3"
        self.rate_limit_delay = 1.0  # 1 second between calls for free tier

        if not api_key:
            logger.warning("FMP API key not provided. Fundamental data will not be available.")

    def get_company_profile(self, ticker: str) -> Dict:
        """
        Get company profile data.

        Args:
            ticker: Stock ticker

        Returns:
            Dictionary with company profile data
        """
        if not self.api_key:
            return {}

        try:
            url = f"{self.base_url}/profile/{ticker}"
            params = {'apikey': self.api_key}

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            time.sleep(self.rate_limit_delay)

            data = response.json()
            if data and len(data) > 0:
                profile = data[0]
                logger.info(f"Fetched profile for {ticker}")
                return profile
            else:
                logger.warning(f"No profile data for {ticker}")
                return {}

        except requests.exceptions.HTTPError as e:
            if '403' in str(e):
                logger.debug(f"FMP API access forbidden for {ticker} (check API key)")
            else:
                logger.error(f"Error fetching profile for {ticker}: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error fetching profile for {ticker}: {e}")
            return {}

    def get_key_metrics(self, ticker: str, period: str = 'annual', limit: int = 5) -> pd.DataFrame:
        """
        Get key financial metrics.

        Args:
            ticker: Stock ticker
            period: 'annual' or 'quarter'
            limit: Number of periods to fetch

        Returns:
            DataFrame with key metrics
        """
        if not self.api_key:
            return pd.DataFrame()

        try:
            url = f"{self.base_url}/key-metrics/{ticker}"
            params = {
                'apikey': self.api_key,
                'period': period,
                'limit': limit
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            time.sleep(self.rate_limit_delay)

            data = response.json()
            if data:
                df = pd.DataFrame(data)
                logger.info(f"Fetched {len(df)} key metrics for {ticker}")
                return df
            else:
                logger.warning(f"No key metrics for {ticker}")
                return pd.DataFrame()

        except requests.exceptions.HTTPError as e:
            if '403' in str(e):
                logger.debug(f"FMP API access forbidden for {ticker} (check API key)")
            else:
                logger.error(f"Error fetching key metrics for {ticker}: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching key metrics for {ticker}: {e}")
            return pd.DataFrame()

    def get_financial_ratios(self, ticker: str, period: str = 'annual', limit: int = 5) -> pd.DataFrame:
        """
        Get financial ratios.

        Args:
            ticker: Stock ticker
            period: 'annual' or 'quarter'
            limit: Number of periods

        Returns:
            DataFrame with financial ratios
        """
        if not self.api_key:
            return pd.DataFrame()

        try:
            url = f"{self.base_url}/ratios/{ticker}"
            params = {
                'apikey': self.api_key,
                'period': period,
                'limit': limit
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            time.sleep(self.rate_limit_delay)

            data = response.json()
            if data:
                df = pd.DataFrame(data)
                logger.info(f"Fetched {len(df)} financial ratios for {ticker}")
                return df
            else:
                return pd.DataFrame()

        except requests.exceptions.HTTPError as e:
            if '403' in str(e):
                logger.debug(f"FMP API access forbidden for {ticker} (check API key)")
            else:
                logger.error(f"Error fetching ratios for {ticker}: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching ratios for {ticker}: {e}")
            return pd.DataFrame()

    def get_fundamental_features(self, ticker: str) -> Dict[str, float]:
        """
        Get fundamental features for ML model.

        Args:
            ticker: Stock ticker

        Returns:
            Dictionary of fundamental features
        """
        features = {}

        # Get company profile
        profile = self.get_company_profile(ticker)
        if profile:
            features['market_cap'] = profile.get('mktCap', 0)
            features['pe_ratio'] = profile.get('price', 0) / (profile.get('eps', 1) + 1e-8)
            features['beta'] = profile.get('beta', 1.0)
            features['dividend_yield'] = profile.get('lastDiv', 0) / (profile.get('price', 1) + 1e-8)

        # Get key metrics
        metrics = self.get_key_metrics(ticker, limit=1)
        if not metrics.empty:
            latest = metrics.iloc[0]
            features['roe'] = latest.get('roe', 0)
            features['roa'] = latest.get('roic', 0)
            features['debt_to_equity'] = latest.get('debtToEquity', 0)
            features['current_ratio'] = latest.get('currentRatio', 0)
            features['quick_ratio'] = latest.get('quickRatio', 0)

        # Get ratios
        ratios = self.get_financial_ratios(ticker, limit=1)
        if not ratios.empty:
            latest = ratios.iloc[0]
            features['profit_margin'] = latest.get('netProfitMargin', 0)
            features['operating_margin'] = latest.get('operatingProfitMargin', 0)
            features['return_on_assets'] = latest.get('returnOnAssets', 0)
            features['return_on_equity'] = latest.get('returnOnEquity', 0)

        return features

    def enrich_with_fundamentals(
        self,
        tickers: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Enrich tickers with fundamental data.

        Args:
            tickers: List of stock tickers

        Returns:
            Dictionary mapping ticker to fundamental features
        """
        fundamentals = {}

        for ticker in tickers:
            logger.info(f"Fetching fundamentals for {ticker}...")
            fundamentals[ticker] = self.get_fundamental_features(ticker)
            time.sleep(self.rate_limit_delay)

        return fundamentals
