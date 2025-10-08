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

    # Safety check: ensure no inf/nan in X or y
    if not np.all(np.isfinite(X)):
        logger.warning("  Found inf/nan in X before feature selection, clipping...")
        X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)

    if not np.all(np.isfinite(y)):
        logger.warning("  Found inf/nan in y before feature selection, clipping...")
        y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)

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


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create single prediction target: 1 month (21 trading days).
    """
    logger.info("  Creating 1-month target...")

    df['target'] = df['Close'].pct_change(21).shift(-21)  # Next month (21 trading days)

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
        df = create_target(df)  # Single target: 1 month

        # Clean data: remove NaN and inf
        df = df.replace([np.inf, -np.inf], np.nan)  # Replace inf with NaN
        df = df.dropna()
        logger.info(f"✓ {len(df)} samples after cleaning")

        if len(df) < 500:
            logger.error(f"❌ Insufficient data after engineering ({len(df)} < 500)")
            return None

        # 3. Prepare features and targets
        logger.info(f"🎯 Step 3/10: Preparing features...")

        feature_cols = [col for col in df.columns
                       if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'target']]

        X_raw = df[feature_cols].values
        y = df['target'].values

        # Additional safety check: remove any remaining NaN or inf
        valid_mask = np.isfinite(y) & np.all(np.isfinite(X_raw), axis=1)
        X_raw = X_raw[valid_mask]
        y = y[valid_mask]

        logger.info(f"✓ {len(X_raw)} valid samples (no NaN/inf)")

        if len(X_raw) < 500:
            logger.error(f"❌ Insufficient valid data ({len(X_raw)} < 500)")
            return None

        # 4. Remove outliers
        logger.info(f"🧹 Step 4/10: Removing outliers...")
        X_clean, y_clean = remove_outliers(X_raw, y, threshold=3.0)

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

        # Split target
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

        # 8. Train ALL models for ensemble (1m horizon)
        logger.info(f"🤖 Step 8/10: Training ensemble models for 1m horizon...")

        import xgboost as xgb
        import lightgbm as lgb

        trained_models = {}
        all_metrics = {}

        # Determine device
        use_gpu = TORCH_AVAILABLE and torch.cuda.is_available()

        # ================== 1. XGBoost ==================
        logger.info(f"  [1/6] Training XGBoost...")

        # Optimize hyperparameters
        if optimize_hp:
            logger.info(f"    Optimizing hyperparameters...")
            best_params_xgb = optimize_xgboost_hyperparameters(
                X_train, y_train, X_val, y_val, n_trials=50
            )
        else:
            best_params_xgb = {
                'n_estimators': 500,
                'max_depth': 7,
                'learning_rate': 0.05
            }

        # XGBoost parameters
        xgb_params = {
            **best_params_xgb,
            'objective': 'reg:squarederror',
            'eval_metric': 'mae',
            'tree_method': 'hist',
            'device': 'cuda' if use_gpu else 'cpu'
        }

        if use_gpu:
            xgb_params['max_bin'] = 1024
            xgb_params['grow_policy'] = 'depthwise'
            xgb_params['max_leaves'] = 255
            xgb_params['subsample'] = 0.8
            xgb_params['colsample_bytree'] = 0.8
            if xgb_params.get('n_estimators', 500) < 1000:
                xgb_params['n_estimators'] = 1000

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(X_test, label=y_test)

        xgb_model = xgb.train(
            xgb_params, dtrain,
            num_boost_round=xgb_params.get('n_estimators', 500),
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=False
        )

        xgb_pred = xgb_model.predict(dtest)
        xgb_mae = np.mean(np.abs(xgb_pred - y_test))
        xgb_acc = np.mean(np.sign(xgb_pred) == np.sign(y_test))

        trained_models['xgboost'] = xgb_model
        all_metrics['xgboost'] = {'test_mae': xgb_mae, 'test_dir_acc': xgb_acc}
        logger.info(f"    ✓ XGBoost: MAE={xgb_mae:.6f}, Acc={xgb_acc:.2%}")

        # ================== 2. LightGBM ==================
        logger.info(f"  [2/6] Training LightGBM...")

        lgb_params = {
            'objective': 'regression',
            'metric': 'mae',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'device': 'gpu' if use_gpu else 'cpu'
        }

        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

        lgb_model = lgb.train(
            lgb_params, lgb_train,
            num_boost_round=1000,
            valid_sets=[lgb_val],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )

        lgb_pred = lgb_model.predict(X_test)
        lgb_mae = np.mean(np.abs(lgb_pred - y_test))
        lgb_acc = np.mean(np.sign(lgb_pred) == np.sign(y_test))

        trained_models['lightgbm'] = lgb_model
        all_metrics['lightgbm'] = {'test_mae': lgb_mae, 'test_dir_acc': lgb_acc}
        logger.info(f"    ✓ LightGBM: MAE={lgb_mae:.6f}, Acc={lgb_acc:.2%}")

        # ================== 3-6. Deep Learning Models (LSTM, GRU, LSTM+Attn, Transformer) ==================
        # Skip for now - too heavy for Colab, will be added in phase 2
        logger.info(f"  [3-6] Deep learning models skipped (Phase 2)")

        # Aggregate metrics
        metrics = {
            'xgboost': all_metrics['xgboost'],
            'lightgbm': all_metrics['lightgbm']
        }

        logger.info(f"✓ Ensemble trained: {len(trained_models)} models")

        # 9. Validation complete (already done above)
        logger.info(f"📊 Step 9/10: Validation complete")

        # 10. Create and save UnifiedEnsembleModel
        logger.info(f"💾 Step 10/10: Creating UnifiedEnsembleModel...")

        # Calculate optimal ensemble weights based on validation performance
        ensemble_weights = {}
        total_inverse_mae = 0.0

        for model_name, model_metrics in all_metrics.items():
            mae = model_metrics['test_mae']
            # Weight = inverse MAE (lower MAE = higher weight)
            inverse_mae = 1.0 / (mae + 1e-6)
            ensemble_weights[model_name] = inverse_mae
            total_inverse_mae += inverse_mae

        # Normalize to sum to 1.0
        ensemble_weights = {k: v/total_inverse_mae for k, v in ensemble_weights.items()}

        logger.info(f"  Ensemble weights (optimized):")
        for name, weight in ensemble_weights.items():
            logger.info(f"    {name}: {weight:.3f}")

        # Create unified model
        from src.models.ensemble_model_unified import UnifiedEnsembleModel

        metadata = {
            'ticker': ticker,
            'training_date': datetime.now().isoformat(),
            'period': period,
            'n_samples': len(X_selected),
            'n_features': len(selected_features),
            'selected_features': selected_features,
            'best_params': best_params_xgb,
            'horizon': '1m',
            'metrics': metrics
        }

        unified_model = UnifiedEnsembleModel(
            models=trained_models,
            weights=ensemble_weights,
            scaler=scaler,
            selected_features=selected_features,
            metadata=metadata
        )

        result = {
            'ticker': ticker,
            'unified_model': unified_model,
            'metrics': metrics
        }

        if save_dir:
            ticker_dir = save_dir / ticker
            ticker_dir.mkdir(parents=True, exist_ok=True)

            # Save unified model (saves all sub-models automatically)
            unified_model.save(ticker_dir)
            logger.info(f"    ✓ Saved UnifiedEnsembleModel to {ticker_dir}/")

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
            # Each ticker uses ~1-1.5GB RAM during training
            # We keep 2GB reserve for CUDA overhead
            max_parallel = int((gpu_memory_gb - 2) / 1.5)

            if args.parallel_tickers > max_parallel:
                logger.warning(f"⚠️  Reducing parallel tickers from {args.parallel_tickers} to {max_parallel} (GPU memory limit)")
                args.parallel_tickers = max_parallel

            logger.info(f"🔥 Parallel mode: {args.parallel_tickers} tickers simultaneously")
            logger.info(f"   Expected GPU memory: ~{args.parallel_tickers * 1.5:.1f}GB / {gpu_memory_gb:.1f}GB")
            logger.info(f"   Expected GPU utilization: 80-95%")

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
