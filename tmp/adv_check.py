import pandas as pd
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.feature_selection import mutual_info_regression
from src.features.technical_indicators import TechnicalIndicators
from src.features.advanced_feature_engineering import AdvancedFeatureEngineer
from train_perfect_colab import purged_time_series_split, adversarial_validation

path = Path('data/raw/AAPL_5y_1d.parquet')
df = pd.read_parquet(path)
print(f"Loaded {len(df)} rows")

tech = TechnicalIndicators()
df = tech.add_all_indicators(df)
adv = AdvancedFeatureEngineer()
leaky_features = [
    'SMA_200',
    'VWAP',
    'EMA_200',
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
]
df = df.drop(columns=[feat for feat in leaky_features if feat in df.columns])

df['trend_40d'] = df['Close'].rolling(40, min_periods=15).mean().shift(1)
df['distance_from_trend_40d'] = (df['Close'] - df['trend_40d']) / (df['trend_40d'] + 1e-8)
df['trend_20d'] = df['Close'].rolling(20, min_periods=10).mean().shift(1)
df['distance_from_trend_20d'] = (df['Close'] - df['trend_20d']) / (df['trend_20d'] + 1e-8)

df = adv.create_lag_features(df, lags=[1, 5, 21])
df = adv.create_rolling_statistics(df, windows=[5, 10, 21, 42])
df = adv.create_fourier_features(df, periods=[5, 10, 21])

feature_cols_all = [
    c for c in df.columns
    if c not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
]
df = adv.apply_rolling_zscore(df, feature_cols_all, window=126, min_periods=30)

df['target'] = (df['Close'].shift(-5) - df['Close']) / df['Close']
mu, sigma = df['target'].mean(), df['target'].std()
df['target'] = df['target'].clip(mu - 3 * sigma, mu + 3 * sigma)
df['target_direction'] = (df['target'] > 0).astype(int)

df = df.replace([np.inf, -np.inf], np.nan).dropna()
print(f"After feature engineering: {df.shape}")

HOLDOUT_DAYS = 252
if len(df) <= HOLDOUT_DAYS:
    raise SystemExit('Not enough rows after feature generation')

df_train = df.iloc[:-HOLDOUT_DAYS].copy()
feature_cols = [c for c in df_train.columns if c not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'target', 'target_direction']]

X = df_train[feature_cols].values
y = df_train['target'].values

if X.shape[1] > 40:
    mi = mutual_info_regression(X, y, random_state=42, n_neighbors=5)
    top_idx = np.argsort(mi)[-40:]
    feature_cols = [feature_cols[i] for i in top_idx]
    X = X[:, top_idx]

X, feature_cols = adv.create_interaction_features(X, feature_cols, top_k=3)

splits = list(purged_time_series_split(len(X), n_splits=10, embargo_pct=0.01))
train_idx, val_idx = splits[-2]
X_train, X_val = X[train_idx], X[val_idx]

auc = adversarial_validation(X_train, X_val, feature_cols)
print(f"Adversarial validation AUC: {auc:.3f}")
PYTHON_FEATURE_NAMES = np.array(feature_cols)
print(f"Selected features: {len(PYTHON_FEATURE_NAMES)}")
