import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import cross_val_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.advanced_feature_engineering import AdvancedFeatureEngineer
from src.features.leakage_filters import drop_leakage_prone_features
from src.features.technical_indicators import TechnicalIndicators
from train_perfect_colab import purged_time_series_split

MIN_TRAIN_DAYS = 150
MIN_HOLDOUT_DAYS = 21
DEFAULT_PATTERN = "*_2y_1d.parquet"
FIXED_FEATURES = [
    'RSI_21',
    'S1',
    'Volume_rolling_std_10',
    'R1',
    'EMA_5',
    'R2',
    'S3',
    'Volume_rolling_std_21',
    'ATR_Percent',
    'Ichimoku_Tenkan',
    'Pivot',
    'PLUS_DI',
    'Volume_rolling_mean_10',
    'distance_from_trend_40d',
    'OBV',
]


def compute_dynamic_holdout(total_rows: int, requested_days: int) -> int:
    if total_rows < MIN_TRAIN_DAYS + MIN_HOLDOUT_DAYS:
        raise ValueError(
            f"Not enough rows ({total_rows}) for minimum training ({MIN_TRAIN_DAYS}) + holdout ({MIN_HOLDOUT_DAYS})"
        )

    max_holdout = max(0, total_rows - MIN_TRAIN_DAYS)
    optimal_holdout = min(requested_days, int(total_rows * 0.25))
    return max(MIN_HOLDOUT_DAYS, min(optimal_holdout, max_holdout))


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df, _ = drop_leakage_prone_features(df)

    tech = TechnicalIndicators()
    df = tech.add_all_indicators(df)
    df, _ = drop_leakage_prone_features(df)

    df['trend_40d'] = df['Close'].rolling(40, min_periods=15).mean().shift(1)
    df['distance_from_trend_40d'] = (df['Close'] - df['trend_40d']) / (df['trend_40d'] + 1e-8)
    df['trend_20d'] = df['Close'].rolling(20, min_periods=10).mean().shift(1)
    df['distance_from_trend_20d'] = (df['Close'] - df['trend_20d']) / (df['trend_20d'] + 1e-8)

    adv = AdvancedFeatureEngineer()
    df = adv.create_lag_features(df, lags=[1, 5, 21])
    df = adv.create_rolling_statistics(df, windows=[5, 10, 21, 42])
    df = adv.create_fourier_features(df, periods=[5, 10, 21])

    feature_cols_all = [
        c for c in df.columns
        if c not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
    ]

    df = adv.apply_rolling_zscore(df, feature_cols_all, window=126, min_periods=30)
    df, _ = drop_leakage_prone_features(df)

    df['target'] = (df['Close'].shift(-5) - df['Close']) / df['Close']
    mu, sigma = df['target'].mean(), df['target'].std()
    df['target'] = df['target'].clip(mu - 3 * sigma, mu + 3 * sigma)
    df['target_direction'] = (df['target'] > 0).astype(int)

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


def select_features(df: pd.DataFrame, holdout_months: int) -> Tuple[np.ndarray, np.ndarray, List[str], int]:
    requested_holdout = holdout_months * 21
    holdout_days = compute_dynamic_holdout(len(df), requested_holdout)

    df_train = df.iloc[:-holdout_days].copy()
    feature_cols = [
        c for c in df_train.columns
        if c not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'target', 'target_direction']
    ]

    X = df_train[feature_cols].values
    y = df_train['target'].values

    if X.shape[1] > 40:
        mi = mutual_info_regression(X, y, random_state=42, n_neighbors=5)
        top_idx = np.argsort(mi)[-40:]
        feature_cols = [feature_cols[i] for i in top_idx]
        X = X[:, top_idx]

    adv = AdvancedFeatureEngineer()
    X, feature_cols = adv.create_interaction_features(X, feature_cols, top_k=3)

    splits = list(purged_time_series_split(len(X), n_splits=10, embargo_pct=0.01))
    train_idx, val_idx = splits[-2]
    X_train, X_val = X[train_idx], X[val_idx]

    return X_train, X_val, feature_cols, holdout_days


def compute_adv_auc(
    X_train: np.ndarray,
    X_val: np.ndarray,
    feature_names: List[str],
    return_importances: bool = False,
) -> Tuple[float, Dict[str, float]] | float:
    X_combined = np.vstack([X_train, X_val])
    y_combined = np.concatenate([
        np.zeros(len(X_train), dtype=int),
        np.ones(len(X_val), dtype=int),
    ])

    clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
    auc = cross_val_score(clf, X_combined, y_combined, cv=3, scoring='roc_auc').mean()

    if not return_importances:
        return auc

    clf_full = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42, n_jobs=-1)
    clf_full.fit(X_combined, y_combined)
    importances = clf_full.feature_importances_

    capped = min(len(feature_names), len(importances))
    importance_map = {
        feature_names[idx]: float(importances[idx])
        for idx in range(capped)
    }
    return auc, importance_map


def evaluate_path(
    path: Path,
    holdout_months: int,
    subset_size: int,
    top_leak: int,
) -> Dict:
    ticker = path.stem.split('_')[0]
    df = pd.read_parquet(path)
    df = engineer_features(df)

    X_train, X_val, feature_cols, holdout_days = select_features(df, holdout_months)
    if not feature_cols:
        raise ValueError(f"No features remain after preprocessing for {ticker}")

    baseline_auc, importances = compute_adv_auc(
        X_train, X_val, feature_cols, return_importances=True
    )

    fixed_features_available = [feat for feat in FIXED_FEATURES if feat in feature_cols]
    missing_features = [feat for feat in FIXED_FEATURES if feat not in feature_cols]

    if not fixed_features_available:
        raise ValueError(
            f"None of the fixed features are present for {ticker}. Missing: {', '.join(FIXED_FEATURES)}"
        )

    idx_map = {name: idx for idx, name in enumerate(feature_cols)}
    subset_indices = [idx_map[name] for name in fixed_features_available]

    X_train_subset = X_train[:, subset_indices]
    X_val_subset = X_val[:, subset_indices]
    subset_auc = compute_adv_auc(X_train_subset, X_val_subset, fixed_features_available)

    sorted_by_importance = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)
    high_leak_features = [feat for feat, _ in sorted_by_importance[:top_leak]]

    low_leak_details = [
        {
            'feature': feat,
            'importance': float(importances.get(feat, np.nan)),
            'rank': idx + 1,
        }
        for idx, feat in enumerate(fixed_features_available)
    ]

    return {
        'ticker': ticker,
        'path': str(path),
        'rows': len(df),
        'holdout_days': holdout_days,
        'baseline_auc': baseline_auc,
        'subset_auc': subset_auc,
        'low_leak_features': fixed_features_available,
        'low_leak_details': low_leak_details,
        'high_leak_features': high_leak_features,
        'importances': {feature: float(score) for feature, score in importances.items()},
        'missing_fixed_features': missing_features,
    }


def batch_evaluate(
    files: List[Path],
    holdout_months: int,
    subset_size: int,
    top_leak: int,
    limit: int | None,
) -> List[Dict]:
    results = []
    processed = 0

    for path in files:
        if limit is not None and processed >= limit:
            break
        try:
            result = evaluate_path(path, holdout_months, subset_size, top_leak)
            results.append(result)
            processed += 1
            print(
                f"[{result['ticker']}] baseline AUC={result['baseline_auc']:.3f} "
                f"→ subset AUC={result['subset_auc']:.3f} | holdout={result['holdout_days']}d"
            )
        except Exception as exc:
            print(f"[WARN] Skipping {path.name}: {exc}")

    return results


def summarize_results(results: List[Dict], subset_size: int, global_top: int) -> Dict:
    feature_counter = Counter()
    importance_totals = Counter()
    rank_totals = Counter()
    missing_counter = Counter()
    auc_deltas = []
    for res in results:
        feature_counter.update(res['low_leak_features'])
        missing_counter.update(res.get('missing_fixed_features', []))
        for detail in res.get('low_leak_details', []):
            feature = detail.get('feature')
            importance = detail.get('importance')
            rank = detail.get('rank')
            if feature is None:
                continue
            if importance is not None and not np.isnan(importance):
                importance_totals[feature] += importance
            if rank is not None and not np.isnan(rank):
                rank_totals[feature] += rank
        auc_deltas.append(res['subset_auc'] - res['baseline_auc'])

    top_common = feature_counter.most_common(subset_size * 2)

    aggregated = []
    for feature, count in feature_counter.items():
        total_importance = importance_totals.get(feature)
        total_rank = rank_totals.get(feature)
        avg_importance = (total_importance / count) if total_importance is not None else None
        avg_rank = (total_rank / count) if total_rank is not None else None
        aggregated.append({
            'feature': feature,
            'count': count,
            'avg_importance': float(avg_importance) if avg_importance is not None else None,
            'avg_rank': float(avg_rank) if avg_rank is not None else None,
        })

    aggregated_sorted = sorted(
        aggregated,
        key=lambda item: (
            -item['count'],
            item['avg_rank'] if item['avg_rank'] is not None else float('inf'),
            -(item['avg_importance'] if item['avg_importance'] is not None else float('-inf')),
            item['feature'],
        ),
    )

    global_low_leak_features = aggregated_sorted[:global_top]

    summary = {
        'tickers_evaluated': len(results),
        'average_auc_delta': float(np.mean(auc_deltas)) if auc_deltas else None,
        'median_auc_delta': float(np.median(auc_deltas)) if auc_deltas else None,
        'top_common_features': top_common,
        'global_low_leak_features': global_low_leak_features,
        'missing_fixed_features': missing_counter.most_common(),
    }
    return summary


def ensure_output_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch adversarial validation feature audit")
    parser.add_argument('--path', type=Path, help='Single parquet file to analyze')
    parser.add_argument('--raw-dir', type=Path, default=Path('data/raw'), help='Directory containing raw parquet files')
    parser.add_argument('--pattern', type=str, default=DEFAULT_PATTERN, help='Glob pattern for selecting files')
    parser.add_argument('--limit', type=int, help='Maximum number of files to process')
    parser.add_argument('--holdout-months', type=int, default=3, help='Holdout period in months')
    parser.add_argument('--subset-size', type=int, default=len(FIXED_FEATURES), help='(Deprecated) retained for compatibility; fixed-feature workflow uses the global list')
    parser.add_argument('--top-leak', type=int, default=10, help='Number of high-leak features to report per ticker')
    parser.add_argument('--global-top', type=int, default=15, help='Number of global low-leak features to surface from aggregate results')
    parser.add_argument('--output', type=Path, default=Path('results/adv_feature_audit.json'), help='Path to dump summary JSON')

    args = parser.parse_args()

    if args.path and not args.path.exists():
        raise SystemExit(f"Data file not found: {args.path}")

    if args.path:
        paths = [args.path]
    else:
        paths = sorted(args.raw_dir.glob(args.pattern))
        if not paths:
            raise SystemExit(f"No files match pattern {args.pattern} in {args.raw_dir}")

    results = batch_evaluate(paths, args.holdout_months, args.subset_size, args.top_leak, args.limit)

    if not results:
        raise SystemExit("No successful evaluations completed.")

    print("\nDetailed per-ticker summaries:\n" + "-" * 60)
    for res in results:
        print(f"Ticker: {res['ticker']}")
        print(f"  Baseline AUC   : {res['baseline_auc']:.3f}")
        print(f"  Subset AUC     : {res['subset_auc']:.3f} (delta {res['subset_auc'] - res['baseline_auc']:+.3f})")
        print(f"  Holdout days   : {res['holdout_days']}")
        print(f"  Low-leak ({len(res['low_leak_features'])}) features:")
        for feat in res['low_leak_features']:
            print(f"    - {feat}")
        print(f"  Highest-leak ({len(res['high_leak_features'])}) features:")
        for feat in res['high_leak_features']:
            print(f"    - {feat}")
        missing = res.get('missing_fixed_features', [])
        if missing:
            print(f"  Missing fixed features ({len(missing)}):")
            for feat in missing:
                print(f"    - {feat}")
        print("-" * 60)

    summary = summarize_results(results, args.subset_size, args.global_top)
    print("Aggregate summary:")
    print(f"  Tickers evaluated : {summary['tickers_evaluated']}")
    if summary['average_auc_delta'] is not None:
        print(f"  Avg subset AUC Δ  : {summary['average_auc_delta']:+.3f}")
        print(f"  Median subset AUC Δ: {summary['median_auc_delta']:+.3f}")
    print(f"  Top {min(args.subset_size * 2, len(summary['top_common_features']))} most common low-leak features:")
    for feature, count in summary['top_common_features']:
        print(f"    - {feature}: {count}")

    global_features = summary.get('global_low_leak_features', [])
    if global_features:
        print(f"\nTop {len(global_features)} global low-leak features (averaged across tickers):")
        for item in global_features:
            avg_rank = item.get('avg_rank')
            avg_importance = item.get('avg_importance')
            count = item.get('count')
            metrics = []
            if avg_rank is not None:
                metrics.append(f"avg rank {avg_rank:.2f}")
            if avg_importance is not None:
                metrics.append(f"avg importance {avg_importance:.4f}")
            if count is not None:
                metrics.append(f"count {count}")
            metric_str = ', '.join(metrics)
            print(f"    - {item['feature']} ({metric_str})")

    missing_summary = summary.get('missing_fixed_features', [])
    if missing_summary:
        print("\nMissing fixed features across tickers (descending frequency):")
        for feature, count in missing_summary:
            print(f"    - {feature}: missing in {count} ticker(s)")

    ensure_output_dir(args.output)
    payload = {
        'results': results,
        'summary': summary,
        'config': {
            'holdout_months': args.holdout_months,
            'subset_size': args.subset_size,
            'top_leak': args.top_leak,
            'global_top': args.global_top,
            'pattern': args.pattern,
            'limit': args.limit,
            'fixed_features': FIXED_FEATURES,
        }
    }
    with args.output.open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved summary to {args.output}")


if __name__ == '__main__':
    main()
