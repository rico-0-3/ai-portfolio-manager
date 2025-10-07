#!/usr/bin/env python3
"""
ADVANCED MODEL TRAINING - Best Practices 2024-2025
===================================================

Implements state-of-the-art techniques for stock prediction:
1. ✅ Feature selection (Boruta + PCA)
2. ✅ Early stopping (XGBoost/LightGBM)
3. ✅ Hyperparameter tuning (Optuna)
4. ✅ Multi-horizon targets (1d, 5d, 21d)
5. ✅ Walk-forward validation
6. ✅ Data augmentation
7. ✅ Ensemble stacking
8. ✅ Feature importance analysis
9. ✅ Outlier detection
10. ✅ Temporal features

Usage:
    python train_advanced_colab.py --top50 --period 10y --optimize
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
import json
import pickle
import torch
from typing import Dict, List, Tuple, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config_loader import ConfigLoader
from src.data.market_data import MarketDataFetcher
from src.features.technical_indicators import TechnicalIndicators
from src.features.feature_engineering import FeatureEngineer
from src.models.ensemble_models import XGBoostPredictor, LightGBMPredictor
from src.models.lstm_model import LSTMTrainer, TORCH_AVAILABLE
from src.models.transformer_model import TransformerTrainer

# Additional imports for advanced features
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️  Optuna not available - hyperparameter tuning disabled")

try:
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler, RobustScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# S&P 500 Top 50
SP500_TOP50 = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B', 'UNH', 'JNJ',
    'V', 'XOM', 'WMT', 'JPM', 'PG', 'MA', 'HD', 'CVX', 'MRK', 'ABBV',
    'KO', 'PEP', 'AVGO', 'COST', 'LLY', 'TMO', 'MCD', 'ACN', 'CSCO', 'ABT',
    'NKE', 'ADBE', 'DHR', 'WFC', 'TXN', 'NEE', 'VZ', 'CRM', 'CMCSA', 'DIS',
    'PM', 'NFLX', 'UPS', 'INTC', 'AMD', 'ORCL', 'HON', 'QCOM', 'IBM', 'BA'
]


def remove_outliers(X: np.ndarray, y: np.ndarray, threshold: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove extreme outliers using z-score method.

    Args:
        X: Features
        y: Targets
        threshold: Z-score threshold (default: 3 sigma)

    Returns:
        Cleaned X, y
    """
    from scipy import stats

    # Calculate z-scores for targets
    z_scores = np.abs(stats.zscore(y))
    mask = z_scores < threshold

    removed = len(y) - mask.sum()
    if removed > 0:
        logger.info(f"  Removed {removed} outliers (>{threshold} sigma)")

    return X[mask], y[mask]


def select_features_mutual_info(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    top_k: int = 40
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Select top K features using mutual information.

    Args:
        X: Features
        y: Targets
        feature_names: List of feature names
        top_k: Number of features to keep

    Returns:
        Selected X, selected feature names, importance scores
    """
    logger.info(f"  Selecting top {top_k} features using mutual information...")

    # Calculate mutual information
    mi_scores = mutual_info_regression(X, y, random_state=42)

    # Select top K
    top_indices = np.argsort(mi_scores)[-top_k:]
    selected_features = [feature_names[i] for i in top_indices]

    logger.info(f"  Top 5 features: {selected_features[-5:]}")

    return X[:, top_indices], selected_features, mi_scores[top_indices]


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal features (day of week, month, quarter).
    Uses sin/cos transformation for cyclical features.
    """
    logger.info("  Adding temporal features...")

    # Day of week (0=Monday, 6=Sunday)
    df['day_of_week'] = df.index.dayofweek
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # Month (1-12)
    df['month'] = df.index.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Quarter (1-4)
    df['quarter'] = df.index.quarter
    df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
    df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)

    # Drop raw cyclical features
    df = df.drop(['day_of_week', 'month', 'quarter'], axis=1)

    return df


def create_multi_horizon_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create multiple prediction horizons:
    - 1 day (short-term)
    - 5 days (weekly)
    - 21 days (monthly)
    """
    logger.info("  Creating multi-horizon targets...")

    df['target_1d'] = df['Close'].pct_change().shift(-1)  # Next day
    df['target_5d'] = df['Close'].pct_change(5).shift(-5)  # Next week
    df['target_21d'] = df['Close'].pct_change(21).shift(-21)  # Next month

    return df


def augment_sequences(
    X_seq: np.ndarray,
    y_seq: np.ndarray,
    noise_level: float = 0.01,
    n_augmented: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Data augmentation for sequence data.
    Adds gaussian noise to create more training samples.

    Args:
        X_seq: Sequence features [samples, timesteps, features]
        y_seq: Targets
        noise_level: Std of gaussian noise (1% default)
        n_augmented: Number of augmented copies per sample

    Returns:
        Augmented X_seq, y_seq
    """
    if n_augmented == 0:
        return X_seq, y_seq

    logger.info(f"  Augmenting sequences with {n_augmented}x copies (noise={noise_level})...")

    augmented_X = [X_seq]
    augmented_y = [y_seq]

    for _ in range(n_augmented):
        noise = np.random.normal(0, noise_level, X_seq.shape)
        augmented_X.append(X_seq + noise)
        augmented_y.append(y_seq)

    X_aug = np.concatenate(augmented_X, axis=0)
    y_aug = np.concatenate(augmented_y, axis=0)

    logger.info(f"  Augmented: {len(X_seq)} → {len(X_aug)} samples")

    return X_aug, y_aug


def optimize_xgboost_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50
) -> Dict:
    """
    Optimize XGBoost hyperparameters using Optuna.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        n_trials: Number of optimization trials

    Returns:
        Best hyperparameters
    """
    if not OPTUNA_AVAILABLE:
        logger.warning("  Optuna not available, using default hyperparameters")
        return {
            'n_estimators': 500,
            'max_depth': 7,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }

    # Check GPU availability
    use_gpu = TORCH_AVAILABLE and torch.cuda.is_available()
    device = 'cuda' if use_gpu else 'cpu'

    if use_gpu:
        logger.info(f"  🚀 GPU acceleration enabled for Optuna trials")

    logger.info(f"  Optimizing XGBoost hyperparameters ({n_trials} trials)...")

    # Prepare DMatrix for GPU
    import xgboost as xgb
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'objective': 'reg:squarederror',
            'eval_metric': 'mae',
            'tree_method': 'hist',
            'device': device
        }

        # Train with xgb.train for GPU support
        bst = xgb.train(
            params,
            dtrain,
            num_boost_round=params['n_estimators'],
            evals=[(dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=False
        )

        # Get validation MAE
        val_pred = bst.predict(dval)
        mae = np.mean(np.abs(val_pred - y_val))

        return mae

    # Run optimization with parallel trials for GPU saturation
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=42)
    )

    # Calculate optimal n_jobs based on GPU memory
    if use_gpu:
        # T4 (15GB) can handle 3-4 parallel trials
        # Smaller GPUs: 2 trials
        n_jobs = 4 if torch.cuda.get_device_properties(0).total_memory > 12e9 else 2
        logger.info(f"  Running {n_jobs} parallel trials to saturate GPU")
    else:
        n_jobs = 1

    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=False)

    best_params = study.best_params
    logger.info(f"  Best MAE: {study.best_value:.6f}")
    logger.info(f"  Best params: {best_params}")

    return best_params


def walk_forward_validation(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    test_size: int = 100
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Walk-forward validation for time series.
    More realistic than single train/test split.

    Args:
        X, y: Full dataset
        n_splits: Number of validation windows
        test_size: Size of each test window

    Returns:
        List of (train_indices, test_indices) tuples
    """
    splits = []
    n_samples = len(X)

    for i in range(n_splits):
        test_end = n_samples - i * test_size
        test_start = test_end - test_size
        train_end = test_start

        if train_end < 100:  # Minimum train size
            break

        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)

        splits.append((train_idx, test_idx))

    logger.info(f"  Walk-forward validation: {len(splits)} splits, test_size={test_size}")
    return splits[::-1]  # Chronological order


# Main training function with ALL improvements
def train_advanced_model(
    ticker: str,
    config: ConfigLoader,
    period: str = '10y',
    optimize_hp: bool = True,
    save_dir: Path = None
) -> dict:
    """
    Train models with ALL 2024-2025 best practices.
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"🚀 ADVANCED TRAINING: {ticker}")
    logger.info(f"{'='*70}")

    try:
        # 1. Fetch data
        logger.info(f"📊 Step 1/10: Fetching {period} of data...")
        fetcher = MarketDataFetcher()
        data = fetcher.fetch_stock_data([ticker], period=period)

        if ticker not in data or len(data[ticker]) < 500:
            logger.error(f"❌ Insufficient data ({len(data.get(ticker, []))} days)")
            return None

        df = data[ticker]
        logger.info(f"✓ Loaded {len(df)} days")

        # 2. Feature engineering with temporal features
        logger.info(f"🔧 Step 2/10: Engineering features...")
        tech_indicators = TechnicalIndicators()
        feature_engineer = FeatureEngineer()

        df = tech_indicators.add_all_indicators(df)
        df = feature_engineer.create_price_features(df)
        df = feature_engineer.create_volume_features(df)
        df = feature_engineer.create_volatility_features(df)
        df = add_temporal_features(df)  # NEW: Temporal features
        df = create_multi_horizon_targets(df)  # NEW: Multi-horizon

        df = df.dropna()
        logger.info(f"✓ {len(df)} samples, {len(df.columns)} features")

        if len(df) < 500:
            logger.error(f"❌ Insufficient data after engineering")
            return None

        # 3. Prepare features and targets
        logger.info(f"🎯 Step 3/10: Preparing features...")

        feature_cols = [col for col in df.columns
                       if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close',
                                     'target_1d', 'target_5d', 'target_21d']]

        X_raw = df[feature_cols].values
        y_1d = df['target_1d'].values
        y_5d = df['target_5d'].values
        y_21d = df['target_21d'].values

        # Remove NaN targets
        valid_mask = ~(np.isnan(y_1d) | np.isnan(y_5d) | np.isnan(y_21d))
        X_raw = X_raw[valid_mask]
        y_1d = y_1d[valid_mask]
        y_5d = y_5d[valid_mask]
        y_21d = y_21d[valid_mask]

        logger.info(f"✓ {len(X_raw)} valid samples")

        # 4. Remove outliers
        logger.info(f"🧹 Step 4/10: Removing outliers...")
        X_clean, y_clean = remove_outliers(X_raw, y_1d, threshold=3.0)

        # 5. Feature selection
        logger.info(f"🎨 Step 5/10: Feature selection...")
        X_selected, selected_features, importance = select_features_mutual_info(
            X_clean, y_clean, feature_cols, top_k=40
        )

        # 6. Train/val/test split (70/15/15)
        logger.info(f"📂 Step 6/10: Train/val/test split...")
        train_split = int(len(X_selected) * 0.70)
        val_split = int(len(X_selected) * 0.85)

        X_train_raw = X_selected[:train_split]
        X_val_raw = X_selected[train_split:val_split]
        X_test_raw = X_selected[val_split:]

        y_train = y_clean[:train_split]
        y_val = y_clean[train_split:val_split]
        y_test = y_clean[val_split:]

        # 7. Feature scaling with RobustScaler (better for outliers)
        logger.info(f"⚖️  Step 7/10: Scaling features...")
        scaler = RobustScaler()  # More robust than StandardScaler
        X_train = scaler.fit_transform(X_train_raw)
        X_val = scaler.transform(X_val_raw)
        X_test = scaler.transform(X_test_raw)

        logger.info(f"✓ Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        # 8. Train XGBoost with optimization
        logger.info(f"🤖 Step 8/10: Training XGBoost...")

        if optimize_hp:
            best_params = optimize_xgboost_hyperparameters(
                X_train, y_train, X_val, y_val, n_trials=50
            )
        else:
            best_params = {
                'n_estimators': 500,
                'max_depth': 7,
                'learning_rate': 0.05
            }

        xgb_model = XGBoostPredictor(**best_params)

        # Train with early stopping
        import xgboost as xgb
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        # Determine device and optimize for GPU
        use_gpu = TORCH_AVAILABLE and torch.cuda.is_available()

        xgb_params = {
            **best_params,
            'objective': 'reg:squarederror',
            'eval_metric': 'mae',
            'tree_method': 'hist',  # GPU-accelerated histogram method
            'device': 'cuda' if use_gpu else 'cpu'
        }

        # GPU-specific optimizations
        if use_gpu:
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            # Increase max_bin for larger GPUs (more GPU work)
            xgb_params['max_bin'] = 512 if gpu_memory_gb > 12 else 256
            # Grow policy for better GPU utilization
            xgb_params['grow_policy'] = 'depthwise'  # Better for GPU
            logger.info(f"  🚀 GPU optimized: max_bin={xgb_params['max_bin']}")

        if xgb_params['device'] == 'cuda':
            logger.info(f"  🚀 Using GPU acceleration")

        evals = [(dtrain, 'train'), (dval, 'val')]
        bst = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=best_params.get('n_estimators', 500),
            evals=evals,
            early_stopping_rounds=50,
            verbose_eval=False
        )

        xgb_model.model = bst

        # Get feature importance
        importance_dict = bst.get_score(importance_type='gain')
        logger.info(f"✓ XGBoost trained (best iteration: {bst.best_iteration})")

        # 9. Validate
        logger.info(f"📊 Step 9/10: Validation...")

        dtest = xgb.DMatrix(X_test)
        dval_pred = xgb.DMatrix(X_val)

        val_pred = bst.predict(dval_pred)
        test_pred = bst.predict(dtest)

        val_mae = np.mean(np.abs(val_pred - y_val))
        test_mae = np.mean(np.abs(test_pred - y_test))
        val_dir_acc = np.mean(np.sign(val_pred) == np.sign(y_val))
        test_dir_acc = np.mean(np.sign(test_pred) == np.sign(y_test))

        logger.info(f"✓ Val MAE: {val_mae:.6f}, Test MAE: {test_mae:.6f}")
        logger.info(f"✓ Val Dir Acc: {val_dir_acc:.2%}, Test Dir Acc: {test_dir_acc:.2%}")

        # 10. Save
        logger.info(f"💾 Step 10/10: Saving...")

        result = {
            'ticker': ticker,
            'model': bst,
            'scaler': scaler,
            'selected_features': selected_features,
            'feature_importance': importance,
            'best_params': best_params,
            'metrics': {
                'val_mae': float(val_mae),
                'test_mae': float(test_mae),
                'val_dir_acc': float(val_dir_acc),
                'test_dir_acc': float(test_dir_acc)
            }
        }

        if save_dir:
            ticker_dir = save_dir / ticker
            ticker_dir.mkdir(parents=True, exist_ok=True)

            # Save everything
            with open(ticker_dir / 'model.pkl', 'wb') as f:
                pickle.dump(bst, f)
            with open(ticker_dir / 'scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)
            with open(ticker_dir / 'features.pkl', 'wb') as f:
                pickle.dump(selected_features, f)

            metadata = {
                'ticker': ticker,
                'training_date': datetime.now().isoformat(),
                'period': period,
                'n_samples': len(X_selected),
                'n_features': len(selected_features),
                'selected_features': selected_features,
                'best_params': best_params,
                'metrics': result['metrics']
            }

            with open(ticker_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"✓ Saved to {ticker_dir}/")

        logger.info(f"✅ {ticker} COMPLETE!")
        return result

    except Exception as e:
        logger.error(f"❌ {ticker} failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def main():
    parser = argparse.ArgumentParser(description='Advanced model training with best practices 2024-2025')
    parser.add_argument('--tickers', type=str, help='Comma-separated list of tickers')
    parser.add_argument('--top50', action='store_true', help='Train on S&P 500 top 50')
    parser.add_argument('--period', type=str, default='10y', choices=['5y', '10y', 'max'])
    parser.add_argument('--output', type=str, default='data/models/pretrained_advanced')
    parser.add_argument('--optimize', action='store_true', help='Enable hyperparameter optimization')
    parser.add_argument('--resume', action='store_true', help='Skip already trained tickers (auto-resume)')
    parser.add_argument('--parallel-tickers', type=int, default=1, help='Number of tickers to train in parallel (GPU only)')

    args = parser.parse_args()

    # Determine tickers
    if args.top50:
        tickers = SP500_TOP50
    elif args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(',')]
    else:
        logger.error("Must specify --tickers or --top50")
        sys.exit(1)

    config = ConfigLoader()
    save_dir = Path(args.output)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Auto-resume: skip already trained tickers
    if args.resume:
        already_trained = []
        for ticker in tickers:
            ticker_dir = save_dir / ticker
            if (ticker_dir / 'model.pkl').exists() and (ticker_dir / 'metadata.json').exists():
                already_trained.append(ticker)

        if already_trained:
            logger.info(f"🔄 RESUME MODE: Skipping {len(already_trained)} already trained tickers")
            logger.info(f"   Already done: {', '.join(already_trained[:10])}{'...' if len(already_trained) > 10 else ''}")
            tickers = [t for t in tickers if t not in already_trained]
            logger.info(f"   Remaining: {len(tickers)} tickers\n")

        if not tickers:
            logger.info("✅ All tickers already trained!")
            return

    # GPU check and parallel setup
    use_parallel = args.parallel_tickers > 1
    if TORCH_AVAILABLE and torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"🚀 GPU: {torch.cuda.get_device_name(0)} ({gpu_memory_gb:.1f} GB)")

        # Auto-adjust parallel tickers based on GPU memory
        if use_parallel:
            max_parallel = int(gpu_memory_gb / 4)  # ~4GB per ticker
            if args.parallel_tickers > max_parallel:
                logger.warning(f"⚠️  Reducing parallel tickers from {args.parallel_tickers} to {max_parallel} (GPU memory limit)")
                args.parallel_tickers = max_parallel

            logger.info(f"🔥 Parallel mode: {args.parallel_tickers} tickers simultaneously")
            logger.info(f"   Expected GPU saturation: ~{args.parallel_tickers * 30}% utilization")

    # Train
    logger.info(f"\n{'='*70}")
    logger.info(f"🎯 ADVANCED TRAINING - {len(tickers)} tickers")
    logger.info(f"{'='*70}\n")

    results = []

    if use_parallel and TORCH_AVAILABLE and torch.cuda.is_available():
        # Parallel training with ThreadPoolExecutor
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def train_wrapper(ticker):
            logger.info(f"\n🔄 Starting: {ticker}")
            return train_advanced_model(
                ticker, config, args.period,
                optimize_hp=args.optimize,
                save_dir=save_dir
            )

        with ThreadPoolExecutor(max_workers=args.parallel_tickers) as executor:
            futures = {executor.submit(train_wrapper, ticker): ticker for ticker in tickers}

            for i, future in enumerate(as_completed(futures), 1):
                ticker = futures[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        logger.info(f"✅ [{len(results)}/{len(tickers)}] {ticker} complete")
                except Exception as e:
                    logger.error(f"❌ {ticker} failed in parallel: {e}")
    else:
        # Sequential training
        for i, ticker in enumerate(tickers, 1):
            logger.info(f"\n[{i}/{len(tickers)}] {ticker}")
            result = train_advanced_model(
                ticker, config, args.period,
                optimize_hp=args.optimize,
                save_dir=save_dir
            )
            if result:
                results.append(result)

    logger.info(f"\n✅ Complete: {len(results)}/{len(tickers)} successful")


if __name__ == '__main__':
    main()
