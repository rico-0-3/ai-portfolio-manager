"""
Feature engineering module.
Creates derived features, handles scaling, and prepares data for ML models.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Engineer features for machine learning models."""

    def __init__(self, scaler_type: str = 'standard'):
        """
        Initialize feature engineer.

        Args:
            scaler_type: Type of scaler ('standard', 'minmax', 'robust')
        """
        self.scaler_type = scaler_type
        self.scaler = None
        self.feature_names = None
        self.pca = None

        logger.info(f"FeatureEngineer initialized with {scaler_type} scaler")

    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create price-based features.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with price features added
        """
        df = df.copy()

        # Returns
        df['Returns_1d'] = df['Close'].pct_change(1)
        df['Returns_5d'] = df['Close'].pct_change(5)
        df['Returns_10d'] = df['Close'].pct_change(10)
        df['Returns_20d'] = df['Close'].pct_change(20)

        # Log returns
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))

        # Price momentum
        df['Momentum_5d'] = df['Close'] - df['Close'].shift(5)
        df['Momentum_10d'] = df['Close'] - df['Close'].shift(10)
        df['Momentum_20d'] = df['Close'] - df['Close'].shift(20)

        # Price velocity (rate of change of returns)
        df['Velocity'] = df['Returns_1d'] - df['Returns_1d'].shift(1)

        # High-Low spread
        df['HL_Spread'] = df['High'] - df['Low']
        df['HL_Spread_Pct'] = (df['HL_Spread'] / df['Close']) * 100

        # Gap
        df['Gap'] = df['Open'] - df['Close'].shift(1)
        df['Gap_Pct'] = (df['Gap'] / df['Close'].shift(1)) * 100

        # Intraday range
        df['Intraday_Range'] = (df['High'] - df['Low']) / df['Open']

        return df

    def create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create volume-based features.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with volume features added
        """
        df = df.copy()

        # Volume changes
        df['Volume_Change'] = df['Volume'].pct_change(1)
        df['Volume_MA_5'] = df['Volume'].rolling(window=5).mean()
        df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()

        # Volume ratio
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20']

        # Price-Volume trend
        df['PV_Trend'] = df['Returns_1d'] * df['Volume']

        # Volume momentum
        df['Volume_Momentum'] = df['Volume'] - df['Volume'].shift(5)

        return df

    def create_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create volatility-based features.

        Args:
            df: DataFrame with price data

        Returns:
            DataFrame with volatility features added
        """
        df = df.copy()

        # Historical volatility (rolling std of returns)
        df['Volatility_5d'] = df['Returns_1d'].rolling(window=5).std()
        df['Volatility_10d'] = df['Returns_1d'].rolling(window=10).std()
        df['Volatility_20d'] = df['Returns_1d'].rolling(window=20).std()
        df['Volatility_60d'] = df['Returns_1d'].rolling(window=60).std()

        # Parkinson's volatility (using high-low)
        df['Parkinson_Vol'] = np.sqrt(
            (1 / (4 * np.log(2))) *
            ((np.log(df['High'] / df['Low'])) ** 2).rolling(window=20).mean()
        )

        # Garman-Klass volatility
        df['GK_Vol'] = np.sqrt(
            0.5 * ((np.log(df['High'] / df['Low'])) ** 2).rolling(window=20).mean() -
            (2 * np.log(2) - 1) * ((np.log(df['Close'] / df['Open'])) ** 2).rolling(window=20).mean()
        )

        return df

    def create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create statistical features.

        Args:
            df: DataFrame with price data

        Returns:
            DataFrame with statistical features added
        """
        df = df.copy()

        # Rolling statistics
        for window in [5, 10, 20, 60]:
            df[f'Mean_{window}d'] = df['Close'].rolling(window=window).mean()
            df[f'Std_{window}d'] = df['Close'].rolling(window=window).std()
            df[f'Skew_{window}d'] = df['Returns_1d'].rolling(window=window).skew()
            df[f'Kurt_{window}d'] = df['Returns_1d'].rolling(window=window).kurt()

        # Z-score (standardized price)
        df['Z_Score_20d'] = (df['Close'] - df['Mean_20d']) / df['Std_20d']

        # Percentile rank
        df['Percentile_20d'] = df['Close'].rolling(window=20).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1]
        )

        return df

    def create_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create trend identification features.

        Args:
            df: DataFrame with price data

        Returns:
            DataFrame with trend features added
        """
        df = df.copy()

        # Trend direction (based on moving averages)
        if 'SMA_20' in df.columns and 'SMA_50' in df.columns:
            df['Trend_Short'] = (df['SMA_20'] > df['SMA_50']).astype(int)

        if 'SMA_50' in df.columns and 'SMA_200' in df.columns:
            df['Trend_Long'] = (df['SMA_50'] > df['SMA_200']).astype(int)

        # Price position relative to MA
        if 'SMA_20' in df.columns:
            df['Price_to_SMA20'] = (df['Close'] / df['SMA_20']) - 1

        if 'SMA_50' in df.columns:
            df['Price_to_SMA50'] = (df['Close'] / df['SMA_50']) - 1

        # Linear regression slope (trend strength)
        def calculate_slope(series):
            if len(series) < 2:
                return 0
            x = np.arange(len(series))
            slope = np.polyfit(x, series, 1)[0]
            return slope

        df['Trend_Slope_10d'] = df['Close'].rolling(window=10).apply(calculate_slope)
        df['Trend_Slope_20d'] = df['Close'].rolling(window=20).apply(calculate_slope)

        return df

    def create_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create candlestick pattern features.

        Args:
            df: DataFrame with OHLC data

        Returns:
            DataFrame with pattern features added
        """
        df = df.copy()

        # Body and shadow sizes
        df['Body'] = abs(df['Close'] - df['Open'])
        df['Upper_Shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
        df['Lower_Shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']

        # Body ratio
        df['Body_Ratio'] = df['Body'] / (df['High'] - df['Low'])

        # Doji pattern (small body)
        df['Is_Doji'] = (df['Body_Ratio'] < 0.1).astype(int)

        # Hammer/Hanging Man
        df['Is_Hammer'] = (
            (df['Lower_Shadow'] > 2 * df['Body']) &
            (df['Upper_Shadow'] < df['Body'])
        ).astype(int)

        # Bullish/Bearish
        df['Is_Bullish'] = (df['Close'] > df['Open']).astype(int)

        return df

    def create_lagged_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        lags: List[int] = [1, 2, 3, 5, 10]
    ) -> pd.DataFrame:
        """
        Create lagged features for time series.

        Args:
            df: DataFrame with features
            columns: Columns to create lags for
            lags: List of lag periods

        Returns:
            DataFrame with lagged features added
        """
        df = df.copy()

        for col in columns:
            if col in df.columns:
                for lag in lags:
                    df[f'{col}_Lag{lag}'] = df[col].shift(lag)

        return df

    def create_rolling_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        windows: List[int] = [5, 10, 20]
    ) -> pd.DataFrame:
        """
        Create rolling window features.

        Args:
            df: DataFrame with features
            columns: Columns to create rolling features for
            windows: List of window sizes

        Returns:
            DataFrame with rolling features added
        """
        df = df.copy()

        for col in columns:
            if col in df.columns:
                for window in windows:
                    df[f'{col}_Mean_{window}'] = df[col].rolling(window=window).mean()
                    df[f'{col}_Std_{window}'] = df[col].rolling(window=window).std()
                    df[f'{col}_Min_{window}'] = df[col].rolling(window=window).min()
                    df[f'{col}_Max_{window}'] = df[col].rolling(window=window).max()

        return df

    def add_cross_sectional_features(
        self,
        df_dict: Dict[str, pd.DataFrame],
        feature_col: str = 'Returns_1d'
    ) -> Dict[str, pd.DataFrame]:
        """
        Add cross-sectional features (relative to universe of stocks).

        Args:
            df_dict: Dictionary mapping ticker to DataFrame
            feature_col: Column to compute cross-sectional stats for

        Returns:
            Dictionary with cross-sectional features added
        """
        # Combine all data
        combined = pd.concat(
            [df[[feature_col]].rename(columns={feature_col: ticker})
             for ticker, df in df_dict.items()],
            axis=1
        )

        # Calculate cross-sectional statistics
        cross_mean = combined.mean(axis=1)
        cross_std = combined.std(axis=1)
        cross_rank = combined.rank(axis=1, pct=True)

        # Add back to individual DataFrames
        for ticker, df in df_dict.items():
            df[f'{feature_col}_CrossMean'] = cross_mean
            df[f'{feature_col}_CrossStd'] = cross_std
            df[f'{feature_col}_Rank'] = cross_rank[ticker]
            df[f'{feature_col}_ZScore'] = (df[feature_col] - cross_mean) / cross_std

        return df_dict

    def scale_features(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Scale features using specified scaler.

        Args:
            df: DataFrame with features
            feature_columns: Columns to scale
            fit: Whether to fit the scaler (True) or use existing (False)

        Returns:
            DataFrame with scaled features
        """
        df = df.copy()

        # Initialize scaler if needed
        if self.scaler is None or fit:
            if self.scaler_type == 'standard':
                self.scaler = StandardScaler()
            elif self.scaler_type == 'minmax':
                self.scaler = MinMaxScaler()
            elif self.scaler_type == 'robust':
                self.scaler = RobustScaler()

        # Select features that exist
        available_features = [col for col in feature_columns if col in df.columns]

        if not available_features:
            logger.warning("No features to scale")
            return df

        # Fit and transform or just transform
        if fit:
            scaled_values = self.scaler.fit_transform(df[available_features])
            logger.info(f"Fitted scaler on {len(available_features)} features")
        else:
            scaled_values = self.scaler.transform(df[available_features])

        # Update DataFrame
        df[available_features] = scaled_values

        return df

    def apply_pca(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        n_components: int = 10,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Apply PCA for dimensionality reduction.

        Args:
            df: DataFrame with features
            feature_columns: Columns to apply PCA to
            n_components: Number of principal components
            fit: Whether to fit PCA (True) or use existing (False)

        Returns:
            DataFrame with PCA components
        """
        df = df.copy()

        # Select available features
        available_features = [col for col in feature_columns if col in df.columns]

        if not available_features:
            return df

        # Initialize PCA if needed
        if self.pca is None or fit:
            self.pca = PCA(n_components=n_components)

        # Fit and transform or just transform
        if fit:
            pca_values = self.pca.fit_transform(df[available_features].fillna(0))
            logger.info(f"PCA fitted. Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.4f}")
        else:
            pca_values = self.pca.transform(df[available_features].fillna(0))

        # Add PCA components to DataFrame
        for i in range(n_components):
            df[f'PCA_{i+1}'] = pca_values[:, i]

        return df

    def prepare_features_for_ml(
        self,
        df: pd.DataFrame,
        target_column: str = 'Returns_1d',
        sequence_length: int = 60,
        prediction_horizon: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and targets for ML models.

        Args:
            df: DataFrame with features
            target_column: Target column for prediction
            sequence_length: Length of input sequences
            prediction_horizon: Steps ahead to predict

        Returns:
            Tuple of (X, y) arrays
        """
        df = df.dropna()

        # Get feature columns (exclude target and non-numeric)
        feature_cols = [col for col in df.columns
                       if col != target_column and
                       df[col].dtype in ['float64', 'float32', 'int64', 'int32']]

        X_list = []
        y_list = []

        for i in range(sequence_length, len(df) - prediction_horizon + 1):
            X_list.append(df[feature_cols].iloc[i-sequence_length:i].values)
            y_list.append(df[target_column].iloc[i + prediction_horizon - 1])

        X = np.array(X_list)
        y = np.array(y_list)

        logger.info(f"Prepared features: X shape={X.shape}, y shape={y.shape}")

        return X, y
