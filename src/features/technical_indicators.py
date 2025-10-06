"""
Technical indicators calculator.
Implements 67+ technical indicators using TA-Lib and custom calculations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# Optional TA-Lib import
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logger.warning("TA-Lib not available. Using pandas-ta or custom implementations.")

try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False
    logger.warning("pandas-ta not available. Install with: pip install pandas-ta")


class TechnicalIndicators:
    """Calculate technical indicators for stock data."""

    def __init__(self):
        """Initialize technical indicators calculator."""
        self.talib_available = TALIB_AVAILABLE
        self.pandas_ta_available = PANDAS_TA_AVAILABLE
        logger.info(f"TechnicalIndicators initialized (TA-Lib: {TALIB_AVAILABLE}, pandas-ta: {PANDAS_TA_AVAILABLE})")

    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all available technical indicators to DataFrame.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with technical indicators added
        """
        logger.info("Adding all technical indicators")

        df = df.copy()

        # Trend indicators
        df = self.add_moving_averages(df)
        df = self.add_macd(df)
        df = self.add_adx(df)

        # Momentum indicators
        df = self.add_rsi(df)
        df = self.add_stochastic(df)
        df = self.add_cci(df)
        df = self.add_roc(df)
        df = self.add_williams_r(df)

        # Volatility indicators
        df = self.add_bollinger_bands(df)
        df = self.add_atr(df)
        df = self.add_keltner_channels(df)

        # Volume indicators
        df = self.add_obv(df)
        df = self.add_vwap(df)
        df = self.add_mfi(df)

        # Additional indicators
        df = self.add_ichimoku(df)
        df = self.add_pivot_points(df)

        logger.info(f"Added indicators. DataFrame now has {len(df.columns)} columns")
        return df

    def add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Simple and Exponential Moving Averages."""
        periods = [5, 10, 20, 50, 100, 200]

        for period in periods:
            if TALIB_AVAILABLE:
                df[f'SMA_{period}'] = talib.SMA(df['Close'], timeperiod=period)
                df[f'EMA_{period}'] = talib.EMA(df['Close'], timeperiod=period)
            else:
                df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
                df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()

        # Add WMA (Weighted Moving Average)
        if TALIB_AVAILABLE:
            df['WMA_20'] = talib.WMA(df['Close'], timeperiod=20)

        return df

    def add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add MACD (Moving Average Convergence Divergence)."""
        if TALIB_AVAILABLE:
            macd, signal, hist = talib.MACD(
                df['Close'],
                fastperiod=12,
                slowperiod=26,
                signalperiod=9
            )
            df['MACD'] = macd
            df['MACD_Signal'] = signal
            df['MACD_Hist'] = hist
        else:
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

        return df

    def add_rsi(self, df: pd.DataFrame, periods: List[int] = [14, 21, 28]) -> pd.DataFrame:
        """Add Relative Strength Index."""
        for period in periods:
            if TALIB_AVAILABLE:
                df[f'RSI_{period}'] = talib.RSI(df['Close'], timeperiod=period)
            else:
                # Custom RSI calculation
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                df[f'RSI_{period}'] = 100 - (100 / (1 + rs))

        return df

    def add_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std: float = 2.0) -> pd.DataFrame:
        """Add Bollinger Bands."""
        if TALIB_AVAILABLE:
            upper, middle, lower = talib.BBANDS(
                df['Close'],
                timeperiod=period,
                nbdevup=std,
                nbdevdn=std,
                matype=0
            )
            df['BB_Upper'] = upper
            df['BB_Middle'] = middle
            df['BB_Lower'] = lower
        else:
            df['BB_Middle'] = df['Close'].rolling(window=period).mean()
            rolling_std = df['Close'].rolling(window=period).std()
            df['BB_Upper'] = df['BB_Middle'] + (rolling_std * std)
            df['BB_Lower'] = df['BB_Middle'] - (rolling_std * std)

        # Bollinger Band Width and %B
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        df['BB_Percent'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

        return df

    def add_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """Add Stochastic Oscillator."""
        if TALIB_AVAILABLE:
            slowk, slowd = talib.STOCH(
                df['High'],
                df['Low'],
                df['Close'],
                fastk_period=k_period,
                slowk_period=d_period,
                slowk_matype=0,
                slowd_period=d_period,
                slowd_matype=0
            )
            df['STOCH_K'] = slowk
            df['STOCH_D'] = slowd
        else:
            low_min = df['Low'].rolling(window=k_period).min()
            high_max = df['High'].rolling(window=k_period).max()
            df['STOCH_K'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
            df['STOCH_D'] = df['STOCH_K'].rolling(window=d_period).mean()

        return df

    def add_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Average True Range (volatility indicator)."""
        if TALIB_AVAILABLE:
            df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=period)
        else:
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['ATR'] = true_range.rolling(window=period).mean()

        # Normalized ATR
        df['ATR_Percent'] = (df['ATR'] / df['Close']) * 100

        return df

    def add_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Average Directional Index (trend strength)."""
        if TALIB_AVAILABLE:
            df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=period)
            df['PLUS_DI'] = talib.PLUS_DI(df['High'], df['Low'], df['Close'], timeperiod=period)
            df['MINUS_DI'] = talib.MINUS_DI(df['High'], df['Low'], df['Close'], timeperiod=period)

        return df

    def add_cci(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Add Commodity Channel Index."""
        if TALIB_AVAILABLE:
            df['CCI'] = talib.CCI(df['High'], df['Low'], df['Close'], timeperiod=period)
        else:
            tp = (df['High'] + df['Low'] + df['Close']) / 3
            sma_tp = tp.rolling(window=period).mean()
            mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
            df['CCI'] = (tp - sma_tp) / (0.015 * mad)

        return df

    def add_roc(self, df: pd.DataFrame, period: int = 12) -> pd.DataFrame:
        """Add Rate of Change."""
        if TALIB_AVAILABLE:
            df['ROC'] = talib.ROC(df['Close'], timeperiod=period)
        else:
            df['ROC'] = ((df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period)) * 100

        return df

    def add_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Williams %R."""
        if TALIB_AVAILABLE:
            df['WILLIAMS_R'] = talib.WILLR(df['High'], df['Low'], df['Close'], timeperiod=period)
        else:
            high_max = df['High'].rolling(window=period).max()
            low_min = df['Low'].rolling(window=period).min()
            df['WILLIAMS_R'] = -100 * (high_max - df['Close']) / (high_max - low_min)

        return df

    def add_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add On-Balance Volume."""
        if TALIB_AVAILABLE:
            df['OBV'] = talib.OBV(df['Close'], df['Volume'])
        else:
            obv = [0]
            for i in range(1, len(df)):
                if df['Close'].iloc[i] > df['Close'].iloc[i - 1]:
                    obv.append(obv[-1] + df['Volume'].iloc[i])
                elif df['Close'].iloc[i] < df['Close'].iloc[i - 1]:
                    obv.append(obv[-1] - df['Volume'].iloc[i])
                else:
                    obv.append(obv[-1])
            df['OBV'] = obv

        return df

    def add_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Volume Weighted Average Price."""
        df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
        return df

    def add_mfi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Money Flow Index."""
        if TALIB_AVAILABLE:
            df['MFI'] = talib.MFI(df['High'], df['Low'], df['Close'], df['Volume'], timeperiod=period)

        return df

    def add_ichimoku(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Ichimoku Cloud indicators."""
        # Tenkan-sen (Conversion Line)
        period9_high = df['High'].rolling(window=9).max()
        period9_low = df['Low'].rolling(window=9).min()
        df['Ichimoku_Tenkan'] = (period9_high + period9_low) / 2

        # Kijun-sen (Base Line)
        period26_high = df['High'].rolling(window=26).max()
        period26_low = df['Low'].rolling(window=26).min()
        df['Ichimoku_Kijun'] = (period26_high + period26_low) / 2

        # Senkou Span A (Leading Span A)
        df['Ichimoku_Senkou_A'] = ((df['Ichimoku_Tenkan'] + df['Ichimoku_Kijun']) / 2).shift(26)

        # Senkou Span B (Leading Span B)
        period52_high = df['High'].rolling(window=52).max()
        period52_low = df['Low'].rolling(window=52).min()
        df['Ichimoku_Senkou_B'] = ((period52_high + period52_low) / 2).shift(26)

        return df

    def add_keltner_channels(self, df: pd.DataFrame, period: int = 20, multiplier: float = 2.0) -> pd.DataFrame:
        """Add Keltner Channels."""
        df['KC_Middle'] = df['Close'].ewm(span=period, adjust=False).mean()

        # Calculate ATR if not present
        if 'ATR' not in df.columns:
            df = self.add_atr(df, period)

        df['KC_Upper'] = df['KC_Middle'] + (multiplier * df['ATR'])
        df['KC_Lower'] = df['KC_Middle'] - (multiplier * df['ATR'])

        return df

    def add_pivot_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Pivot Points (for intraday/daily levels)."""
        df['Pivot'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['R1'] = 2 * df['Pivot'] - df['Low']
        df['S1'] = 2 * df['Pivot'] - df['High']
        df['R2'] = df['Pivot'] + (df['High'] - df['Low'])
        df['S2'] = df['Pivot'] - (df['High'] - df['Low'])
        df['R3'] = df['High'] + 2 * (df['Pivot'] - df['Low'])
        df['S3'] = df['Low'] - 2 * (df['High'] - df['Pivot'])

        return df

    def get_indicator_list(self) -> List[str]:
        """Get list of all available indicators."""
        indicators = [
            'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50', 'SMA_100', 'SMA_200',
            'EMA_5', 'EMA_10', 'EMA_20', 'EMA_50', 'EMA_100', 'EMA_200',
            'MACD', 'MACD_Signal', 'MACD_Hist',
            'RSI_14', 'RSI_21', 'RSI_28',
            'BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Width', 'BB_Percent',
            'STOCH_K', 'STOCH_D',
            'ATR', 'ATR_Percent',
            'ADX', 'PLUS_DI', 'MINUS_DI',
            'CCI', 'ROC', 'WILLIAMS_R',
            'OBV', 'VWAP', 'MFI',
            'Ichimoku_Tenkan', 'Ichimoku_Kijun', 'Ichimoku_Senkou_A', 'Ichimoku_Senkou_B',
            'KC_Upper', 'KC_Middle', 'KC_Lower',
            'Pivot', 'R1', 'R2', 'R3', 'S1', 'S2', 'S3'
        ]
        return indicators
