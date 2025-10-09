"""
Advanced Feature Engineering - Match training pipeline exactly.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class AdvancedFeatureEngineer:
    """Advanced feature engineering beyond basic technical indicators."""

    def __init__(self):
        self.poly_features = None
        self.selected_features_for_interaction = None

    def create_lag_features(self, df: pd.DataFrame, lags: List[int] = [1, 5, 21]) -> pd.DataFrame:
        """Create lagged features."""
        logger.debug(f"  Creating lag features: {lags}")

        for lag in lags:
            df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
            df[f'Volume_lag_{lag}'] = df['Volume'].shift(lag)
            df[f'Return_lag_{lag}'] = df['Close'].pct_change(lag)

        return df

    def create_rolling_statistics(self, df: pd.DataFrame, windows: List[int] = [5, 10, 21, 60]) -> pd.DataFrame:
        """Create rolling statistical features."""
        logger.debug(f"  Creating rolling statistics: {windows}")

        for window in windows:
            # Mean and std
            df[f'Close_rolling_mean_{window}'] = df['Close'].rolling(window).mean()
            df[f'Close_rolling_std_{window}'] = df['Close'].rolling(window).std()

            # Skewness and kurtosis
            df[f'Return_rolling_skew_{window}'] = df['Close'].pct_change().rolling(window).skew()
            df[f'Return_rolling_kurt_{window}'] = df['Close'].pct_change().rolling(window).kurt()

            # Volume statistics
            if window <= 21:
                df[f'Volume_rolling_mean_{window}'] = df['Volume'].rolling(window).mean()
                df[f'Volume_rolling_std_{window}'] = df['Volume'].rolling(window).std()

        return df

    def create_fourier_features(self, df: pd.DataFrame, periods: List[int] = [5, 10, 21, 252]) -> pd.DataFrame:
        """Create Fourier transform features for seasonality."""
        logger.debug(f"  Creating Fourier features: {periods}")

        close_prices = df['Close'].fillna(method='ffill').fillna(method='bfill').values

        for period in periods:
            # Sine and cosine components
            df[f'Fourier_sin_{period}'] = np.sin(2 * np.pi * np.arange(len(df)) / period)
            df[f'Fourier_cos_{period}'] = np.cos(2 * np.pi * np.arange(len(df)) / period)

        return df

    def apply_rolling_zscore(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        window: int = 126,
        min_periods: int = 30,
        min_valid_ratio: float = 0.4,
    ) -> pd.DataFrame:
        """Normalize features using rolling z-score without look-ahead."""
        if not feature_columns:
            return df

        logger.debug(
            "  Applying rolling z-score normalization (window=%d, min_periods=%d) to %d features",
            window,
            min_periods,
            len(feature_columns),
        )

        subset = df[feature_columns]
        rolling_mean = subset.rolling(window=window, min_periods=min_periods).mean().shift(1)
        rolling_std = subset.rolling(window=window, min_periods=min_periods).std().shift(1)

        # Avoid division by zero; replace zeros with NaN so they can be filtered out
        rolling_std = rolling_std.replace(0, np.nan)

        normalized = (subset - rolling_mean) / rolling_std

        min_valid_count = max(int(len(df) * min_valid_ratio), min_periods)
        valid_cols = [col for col in feature_columns if normalized[col].notna().sum() >= min_valid_count]

        if valid_cols:
            df.loc[:, valid_cols] = normalized[valid_cols]
        dropped_cols = sorted(set(feature_columns) - set(valid_cols))
        if dropped_cols:
            logger.debug(
                "  Dropping %d features due to insufficient rolling coverage: %s",
                len(dropped_cols),
                ", ".join(dropped_cols[:10]) + ("..." if len(dropped_cols) > 10 else ""),
            )
            df = df.drop(columns=dropped_cols)

        return df

    def create_interaction_features(self, X: np.ndarray, feature_names: List[str], top_k: int = 10) -> Tuple[np.ndarray, List[str]]:
        """Create interaction features between top features."""
        logger.debug(f"  Creating interaction features (top {top_k})...")

        # Select top k features
        top_features = X[:, :min(top_k, X.shape[1])]
        top_feature_names = feature_names[:min(top_k, len(feature_names))]

        interaction_features = []
        interaction_names = []

        for i in range(len(top_feature_names)):
            for j in range(i+1, len(top_feature_names)):
                interaction = top_features[:, i] * top_features[:, j]
                interaction_features.append(interaction)
                interaction_names.append(f"{top_feature_names[i]}_x_{top_feature_names[j]}")

        if interaction_features:
            X_interactions = np.column_stack(interaction_features)
            X_combined = np.hstack([X, X_interactions])
            combined_names = feature_names + interaction_names

            logger.debug(f"  Interaction features: {X.shape[1]} → {X_combined.shape[1]} (added {len(interaction_names)})")
            return X_combined, combined_names
        else:
            return X, feature_names
