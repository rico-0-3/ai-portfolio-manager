"""
Alpha Vantage data fetcher.
Provides access to real-time and historical data with technical indicators.
"""

import requests
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging
import time
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class AlphaVantageDataFetcher:
    """Fetch data from Alpha Vantage API."""

    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str, cache_dir: str = "data/raw/alpha_vantage"):
        """
        Initialize Alpha Vantage data fetcher.

        Args:
            api_key: Alpha Vantage API key
            cache_dir: Directory to cache data
        """
        self.api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.last_request_time = 0
        self.rate_limit_delay = 12  # 5 requests per minute = 12 seconds between requests

        logger.info("AlphaVantageDataFetcher initialized")

    def _rate_limit(self):
        """Enforce rate limiting."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time

        if time_since_last_request < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last_request
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def _make_request(self, params: Dict) -> Dict:
        """
        Make API request with rate limiting.

        Args:
            params: Request parameters

        Returns:
            JSON response
        """
        self._rate_limit()

        params['apikey'] = self.api_key

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Check for API errors
            if 'Error Message' in data:
                raise ValueError(f"API Error: {data['Error Message']}")
            if 'Note' in data:
                logger.warning(f"API Note: {data['Note']}")

            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            raise

    def fetch_intraday(
        self,
        ticker: str,
        interval: str = '5min',
        outputsize: str = 'compact'
    ) -> pd.DataFrame:
        """
        Fetch intraday data.

        Args:
            ticker: Stock ticker
            interval: Time interval (1min, 5min, 15min, 30min, 60min)
            outputsize: 'compact' (last 100 data points) or 'full'

        Returns:
            DataFrame with intraday data
        """
        logger.info(f"Fetching intraday data for {ticker} ({interval})")

        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': ticker,
            'interval': interval,
            'outputsize': outputsize,
            'datatype': 'json'
        }

        data = self._make_request(params)
        time_series_key = f'Time Series ({interval})'

        if time_series_key not in data:
            logger.error(f"No intraday data found for {ticker}")
            return pd.DataFrame()

        df = pd.DataFrame.from_dict(data[time_series_key], orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.rename(columns={
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. volume': 'Volume'
        })

        df = df.astype(float)
        df = df.sort_index()

        return df

    def fetch_daily(
        self,
        ticker: str,
        outputsize: str = 'full'
    ) -> pd.DataFrame:
        """
        Fetch daily historical data.

        Args:
            ticker: Stock ticker
            outputsize: 'compact' (last 100 days) or 'full' (20+ years)

        Returns:
            DataFrame with daily data
        """
        logger.info(f"Fetching daily data for {ticker}")

        params = {
            'function': 'TIME_SERIES_DAILY_ADJUSTED',
            'symbol': ticker,
            'outputsize': outputsize,
            'datatype': 'json'
        }

        data = self._make_request(params)

        if 'Time Series (Daily)' not in data:
            logger.error(f"No daily data found for {ticker}")
            return pd.DataFrame()

        df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.rename(columns={
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. adjusted close': 'Adj Close',
            '6. volume': 'Volume',
            '7. dividend amount': 'Dividend',
            '8. split coefficient': 'Split'
        })

        df = df.astype(float)
        df = df.sort_index()

        return df

    def fetch_sma(
        self,
        ticker: str,
        interval: str = 'daily',
        time_period: int = 20,
        series_type: str = 'close'
    ) -> pd.DataFrame:
        """
        Fetch Simple Moving Average (SMA) indicator.

        Args:
            ticker: Stock ticker
            interval: Time interval
            time_period: Number of periods
            series_type: Price type (close, open, high, low)

        Returns:
            DataFrame with SMA values
        """
        logger.info(f"Fetching SMA-{time_period} for {ticker}")

        params = {
            'function': 'SMA',
            'symbol': ticker,
            'interval': interval,
            'time_period': time_period,
            'series_type': series_type
        }

        data = self._make_request(params)

        if 'Technical Analysis: SMA' not in data:
            return pd.DataFrame()

        df = pd.DataFrame.from_dict(data['Technical Analysis: SMA'], orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.rename(columns={'SMA': f'SMA_{time_period}'})
        df = df.astype(float)
        df = df.sort_index()

        return df

    def fetch_rsi(
        self,
        ticker: str,
        interval: str = 'daily',
        time_period: int = 14,
        series_type: str = 'close'
    ) -> pd.DataFrame:
        """
        Fetch Relative Strength Index (RSI) indicator.

        Args:
            ticker: Stock ticker
            interval: Time interval
            time_period: Number of periods
            series_type: Price type

        Returns:
            DataFrame with RSI values
        """
        logger.info(f"Fetching RSI-{time_period} for {ticker}")

        params = {
            'function': 'RSI',
            'symbol': ticker,
            'interval': interval,
            'time_period': time_period,
            'series_type': series_type
        }

        data = self._make_request(params)

        if 'Technical Analysis: RSI' not in data:
            return pd.DataFrame()

        df = pd.DataFrame.from_dict(data['Technical Analysis: RSI'], orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.rename(columns={'RSI': f'RSI_{time_period}'})
        df = df.astype(float)
        df = df.sort_index()

        return df

    def fetch_macd(
        self,
        ticker: str,
        interval: str = 'daily',
        series_type: str = 'close'
    ) -> pd.DataFrame:
        """
        Fetch MACD indicator.

        Args:
            ticker: Stock ticker
            interval: Time interval
            series_type: Price type

        Returns:
            DataFrame with MACD values
        """
        logger.info(f"Fetching MACD for {ticker}")

        params = {
            'function': 'MACD',
            'symbol': ticker,
            'interval': interval,
            'series_type': series_type
        }

        data = self._make_request(params)

        if 'Technical Analysis: MACD' not in data:
            return pd.DataFrame()

        df = pd.DataFrame.from_dict(data['Technical Analysis: MACD'], orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.rename(columns={
            'MACD': 'MACD',
            'MACD_Signal': 'MACD_Signal',
            'MACD_Hist': 'MACD_Hist'
        })
        df = df.astype(float)
        df = df.sort_index()

        return df

    def fetch_bbands(
        self,
        ticker: str,
        interval: str = 'daily',
        time_period: int = 20,
        series_type: str = 'close'
    ) -> pd.DataFrame:
        """
        Fetch Bollinger Bands indicator.

        Args:
            ticker: Stock ticker
            interval: Time interval
            time_period: Number of periods
            series_type: Price type

        Returns:
            DataFrame with Bollinger Bands
        """
        logger.info(f"Fetching Bollinger Bands for {ticker}")

        params = {
            'function': 'BBANDS',
            'symbol': ticker,
            'interval': interval,
            'time_period': time_period,
            'series_type': series_type
        }

        data = self._make_request(params)

        if 'Technical Analysis: BBANDS' not in data:
            return pd.DataFrame()

        df = pd.DataFrame.from_dict(data['Technical Analysis: BBANDS'], orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.rename(columns={
            'Real Upper Band': 'BB_Upper',
            'Real Middle Band': 'BB_Middle',
            'Real Lower Band': 'BB_Lower'
        })
        df = df.astype(float)
        df = df.sort_index()

        return df

    def fetch_company_overview(self, ticker: str) -> Dict:
        """
        Fetch company fundamental data.

        Args:
            ticker: Stock ticker

        Returns:
            Dictionary with company information
        """
        logger.info(f"Fetching company overview for {ticker}")

        params = {
            'function': 'OVERVIEW',
            'symbol': ticker
        }

        data = self._make_request(params)
        return data
