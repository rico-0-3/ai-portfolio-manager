"""Utilities to remove features prone to leakage or excessive drift."""

from typing import Iterable, Tuple, Optional
import pandas as pd

LEAKAGE_FEATURES = {
    'SMA_200',
    'EMA_200',
    'VWAP',
    'Ichimoku_Senkou_A',
    'Ichimoku_Senkou_B',
    'KC_Lower',
    'KC_Upper',
    'KC_Middle',
    'BB_Lower',
    'Volume_rolling_std_60',
    'Volume_rolling_mean_60',
    'Volume_rolling_std_42',
    'Volume_rolling_mean_42',
    'Return_rolling_skew_60',
    'Return_rolling_kurt_42',
}


def drop_leakage_prone_features(
    df: pd.DataFrame,
    logger: Optional[object] = None,
    extra_features: Optional[Iterable[str]] = None,
    log_level: str = "debug",
) -> Tuple[pd.DataFrame, Tuple[str, ...]]:
    """Drop features that are known to introduce temporal leakage.

    Args:
        df: DataFrame to sanitize.
        logger: Optional logger for reporting.
        extra_features: Additional feature names to drop.
        log_level: Logger method to call ("info", "debug", etc.).

    Returns:
        Tuple of (cleaned DataFrame, tuple of removed columns).
    """
    drop_set = set(LEAKAGE_FEATURES)
    if extra_features:
        drop_set.update(extra_features)

    existing = tuple(col for col in drop_set if col in df.columns)
    if existing:
        df = df.drop(columns=list(existing))
        if logger is not None:
            log_fn = getattr(logger, log_level, None)
            if callable(log_fn):
                log_fn("Removing leakage-prone features: %s", existing)

    return df, existing
