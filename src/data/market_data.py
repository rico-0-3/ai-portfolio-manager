"""
Market data acquisition module.
Handles downloading historical and real-time stock data from multiple sources.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union
import logging
from pathlib import Path
import time

logger = logging.getLogger(__name__)


class MarketDataFetcher:
    """Fetch market data from Yahoo Finance and other sources."""

    def __init__(self, cache_dir: str = "data/raw"):
        """
        Initialize the market data fetcher.

        Args:
            cache_dir: Directory to cache downloaded data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"MarketDataFetcher initialized with cache: {self.cache_dir}")

    def fetch_stock_data(
        self,
        tickers: Union[str, List[str]],
        period: str = "5y",
        interval: str = "1d",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_cache: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical stock data for given tickers.

        Args:
            tickers: Single ticker or list of tickers
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            start_date: Start date (YYYY-MM-DD) - overrides period
            end_date: End date (YYYY-MM-DD)
            use_cache: Whether to use cached data if available

        Returns:
            Dictionary mapping ticker to DataFrame with OHLCV data
        """
        if isinstance(tickers, str):
            tickers = [tickers]

        data_dict = {}

        for ticker in tickers:
            logger.info(f"Fetching data for {ticker}")

            # Check cache
            cache_file = self.cache_dir / f"{ticker}_{period}_{interval}.parquet"
            if use_cache and cache_file.exists():
                cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
                if cache_age < timedelta(hours=24):
                    logger.info(f"Loading {ticker} from cache (age: {cache_age})")
                    data_dict[ticker] = pd.read_parquet(cache_file)
                    continue

            try:
                # Download data
                stock = yf.Ticker(ticker)

                if start_date and end_date:
                    df = stock.history(start=start_date, end=end_date, interval=interval)
                else:
                    df = stock.history(period=period, interval=interval)

                if df.empty:
                    logger.warning(f"No data received for {ticker}")
                    continue

                # Clean data
                df = self._clean_data(df)

                # Cache data
                df.to_parquet(cache_file)
                logger.info(f"Cached {ticker} data to {cache_file}")

                data_dict[ticker] = df

                # Rate limiting
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Error fetching {ticker}: {e}")
                continue

        logger.info(f"Successfully fetched data for {len(data_dict)}/{len(tickers)} tickers")
        return data_dict

    def fetch_stock_info(self, ticker: str) -> Dict:
        """
        Fetch fundamental information about a stock.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with stock information
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Extract key metrics
            fundamental_data = {
                'ticker': ticker,
                'name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', None),
                'pe_ratio': info.get('trailingPE', None),
                'forward_pe': info.get('forwardPE', None),
                'peg_ratio': info.get('pegRatio', None),
                'price_to_book': info.get('priceToBook', None),
                'dividend_yield': info.get('dividendYield', None),
                'beta': info.get('beta', None),
                'profit_margin': info.get('profitMargins', None),
                'operating_margin': info.get('operatingMargins', None),
                'roe': info.get('returnOnEquity', None),
                'debt_to_equity': info.get('debtToEquity', None),
                'current_ratio': info.get('currentRatio', None),
                'revenue_growth': info.get('revenueGrowth', None),
                'earnings_growth': info.get('earningsGrowth', None),
            }

            logger.info(f"Fetched fundamental data for {ticker}")
            return fundamental_data

        except Exception as e:
            logger.error(f"Error fetching info for {ticker}: {e}")
            return {'ticker': ticker, 'error': str(e)}

    def fetch_multiple_info(self, tickers: List[str]) -> pd.DataFrame:
        """
        Fetch fundamental information for multiple stocks.

        Args:
            tickers: List of stock ticker symbols

        Returns:
            DataFrame with fundamental data
        """
        info_list = []

        for ticker in tickers:
            info = self.fetch_stock_info(ticker)
            info_list.append(info)
            time.sleep(0.2)  # Rate limiting

        df = pd.DataFrame(info_list)
        logger.info(f"Fetched fundamental data for {len(tickers)} stocks")

        return df

    def fetch_market_indices(
        self,
        period: str = "5y",
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch major market indices data.

        Args:
            period: Data period
            interval: Data interval

        Returns:
            Dictionary with index data
        """
        indices = {
            'SPY': 'S&P 500',
            'QQQ': 'NASDAQ-100',
            'DIA': 'Dow Jones',
            'IWM': 'Russell 2000',
            'VIX': 'Volatility Index'
        }

        logger.info("Fetching market indices data")
        return self.fetch_stock_data(list(indices.keys()), period=period, interval=interval)

    def get_sp500_tickers(self) -> List[str]:
        """
        Get list of S&P 500 tickers.

        Returns:
            List of ticker symbols
        """
        try:
            # Read S&P 500 from Wikipedia
            table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
            df = table[0]
            tickers = df['Symbol'].str.replace('.', '-').tolist()
            logger.info(f"Retrieved {len(tickers)} S&P 500 tickers")
            return tickers
        except Exception as e:
            logger.error(f"Error fetching S&P 500 tickers: {e}")
            return []

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare market data.

        Args:
            df: Raw DataFrame

        Returns:
            Cleaned DataFrame
        """
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]

        # Sort by date
        df = df.sort_index()

        # Forward fill missing values (holidays, etc.)
        df = df.fillna(method='ffill')

        # Drop remaining NaN (beginning of series)
        df = df.dropna()

        # Ensure numeric columns
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    def calculate_returns(
        self,
        data: pd.DataFrame,
        column: str = 'Close',
        periods: int = 1
    ) -> pd.Series:
        """
        Calculate returns for given data.

        Args:
            data: DataFrame with price data
            column: Column to calculate returns from
            periods: Number of periods for return calculation

        Returns:
            Series with returns
        """
        returns = data[column].pct_change(periods=periods)
        return returns

    def get_latest_price(self, ticker: str) -> Optional[float]:
        """
        Get the latest price for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Latest closing price or None
        """
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period='1d')
            if not data.empty:
                return float(data['Close'].iloc[-1])
            return None
        except Exception as e:
            logger.error(f"Error getting latest price for {ticker}: {e}")
            return None
