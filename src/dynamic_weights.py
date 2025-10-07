"""
Dynamic Weight Calibrator
Adjusts ML ensemble and portfolio optimization weights based on historical performance per ticker.
"""

import logging
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

logger = logging.getLogger(__name__)


class DynamicWeightCalibrator:
    """
    Calibrates weights dynamically for each ticker based on historical performance.

    Two-level calibration:
    1. ML Ensemble Weights - per ticker, based on prediction accuracy
    2. Portfolio Optimization Weights - per ticker, based on method effectiveness
    """

    def __init__(self, lookback_period: int = 60):
        """
        Args:
            lookback_period: Number of days to use for historical calibration
        """
        self.lookback_period = lookback_period
        self.ml_weights_cache = {}  # ticker -> model weights
        self.portfolio_weights_cache = {}  # ticker -> method weights

    def calibrate_ml_weights(
        self,
        ticker: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        models: Dict[str, object],
        default_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calibrate ML ensemble weights for a specific ticker based on validation performance.

        Args:
            ticker: Stock ticker
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            models: Dictionary of trained models {name: model_object}
            default_weights: Default weights from config

        Returns:
            Optimized weights dictionary {model_name: weight}
        """
        logger.info(f"{ticker}: Calibrating ML ensemble weights...")

        # Calculate performance for each model on validation set
        model_errors = {}

        for model_name, model in models.items():
            try:
                # Get predictions
                if hasattr(model, 'predict'):
                    if model_name in ['lstm', 'gru', 'lstm_attention', 'transformer']:
                        # Sequential models need 3D input
                        if X_val.ndim == 2:
                            seq_length = 60
                            if len(X_val) >= seq_length:
                                X_val_seq = np.array([X_val[i-seq_length:i] for i in range(seq_length, len(X_val))])
                                y_val_seq = y_val[seq_length:]
                                predictions = model.predict(X_val_seq)
                            else:
                                continue
                        else:
                            predictions = model.predict(X_val)
                            y_val_seq = y_val
                    else:
                        predictions = model.predict(X_val)
                        y_val_seq = y_val

                    # Calculate error metrics
                    if len(predictions) > 0 and len(y_val_seq) > 0:
                        # Ensure same length
                        min_len = min(len(predictions), len(y_val_seq))
                        predictions = predictions[:min_len]
                        y_val_seq = y_val_seq[:min_len]

                        mae = mean_absolute_error(y_val_seq, predictions)
                        rmse = np.sqrt(mean_squared_error(y_val_seq, predictions))

                        # Combined error (lower is better)
                        combined_error = (mae + rmse) / 2
                        model_errors[model_name] = combined_error

                        logger.debug(f"{ticker}: {model_name} - MAE={mae:.4f}, RMSE={rmse:.4f}")

            except Exception as e:
                logger.warning(f"{ticker}: Could not calibrate {model_name}: {e}")
                # Use default weight for failed models
                model_errors[model_name] = float('inf')

        if not model_errors or all(e == float('inf') for e in model_errors.values()):
            logger.warning(f"{ticker}: No valid model calibration, using default weights")
            return default_weights

        # Convert errors to weights using inverse softmax
        # Lower error = higher weight
        errors_array = np.array([model_errors.get(name, float('inf')) for name in default_weights.keys()])

        # Replace inf with max finite error * 2
        finite_errors = errors_array[np.isfinite(errors_array)]
        if len(finite_errors) > 0:
            max_error = finite_errors.max() * 2
            errors_array = np.where(np.isfinite(errors_array), errors_array, max_error)
        else:
            return default_weights

        # Inverse (lower error = higher score)
        scores = 1.0 / (errors_array + 1e-6)

        # Softmax to get weights
        exp_scores = np.exp(scores - scores.max())  # Numerical stability
        weights_array = exp_scores / exp_scores.sum()

        # Create weight dictionary
        calibrated_weights = {
            name: float(weight)
            for name, weight in zip(default_weights.keys(), weights_array)
        }

        # Log changes
        logger.info(f"{ticker}: ML weights calibrated:")
        for name in default_weights.keys():
            old = default_weights[name]
            new = calibrated_weights[name]
            change = ((new - old) / old * 100) if old > 0 else 0
            logger.info(f"  {name}: {old:.3f} -> {new:.3f} ({change:+.1f}%)")

        # Cache
        self.ml_weights_cache[ticker] = calibrated_weights

        return calibrated_weights

    def calibrate_portfolio_weights(
        self,
        ticker: str,
        historical_returns: pd.Series,
        optimization_results: Dict[str, Dict],
        default_weights: Dict[str, float],
        ml_prediction: float = None
    ) -> Dict[str, float]:
        """
        Calibrate portfolio optimization method weights for a specific ticker.
        Now uses ML predictions and method effectiveness instead of just historical Sharpe.

        Args:
            ticker: Stock ticker
            historical_returns: Historical return series
            optimization_results: Results from each optimization method
                                 {method_name: {'allocation': float, 'sharpe': float, 'uses_ml': bool}}
            default_weights: Default weights from config
            ml_prediction: ML predicted return for this ticker (optional)

        Returns:
            Optimized weights dictionary {method_name: weight}
        """
        logger.info(f"{ticker}: Calibrating portfolio optimization weights (ML-aware)...")

        # Score each method based on multiple factors
        method_scores = {}

        for method_name, result in optimization_results.items():
            if result is None or 'allocation' not in result:
                method_scores[method_name] = 0.0
                continue

            # Factor 1: Allocation success (did the method allocate to this ticker?)
            allocation = result.get('allocation', 0.0)
            allocation_score = 1.0 if allocation > 0 else 0.1  # Small score even if no allocation

            # Factor 2: Method type bonus
            # ML-driven methods get bonus when we have ML predictions
            ml_driven_methods = ['mean_variance', 'black_litterman']
            risk_driven_methods = ['risk_parity', 'cvar', 'rl_agent']

            if ml_prediction is not None:
                if method_name in ml_driven_methods:
                    # ML methods get bonus
                    method_type_bonus = 1.5
                else:
                    # Risk methods get penalty (they ignore ML)
                    method_type_bonus = 0.7
            else:
                # No ML prediction, all methods equal
                method_type_bonus = 1.0

            # Factor 3: ML prediction alignment
            # If we have ML prediction, score based on alignment with prediction
            if ml_prediction is not None:
                # ML-driven methods should be weighted higher when we trust ML
                # Risk-driven methods provide diversification regardless of ML
                if method_name in ml_driven_methods:
                    # ML methods: score based on having a prediction
                    ml_alignment_score = 1.5  # Strong preference for ML methods
                else:
                    # Risk methods: constant score (diversification value)
                    ml_alignment_score = 1.0
            else:
                # No ML prediction: fall back to historical Sharpe (normalized)
                sharpe = result.get('sharpe', 0.0)
                ml_alignment_score = max(0.5, sharpe + 1.0)  # Normalize: Sharpe 0 = 1, Sharpe 1 = 2

            # Combined score: allocation × method_type × ml_alignment
            score = allocation_score * method_type_bonus * ml_alignment_score
            method_scores[method_name] = score

            logger.debug(f"{ticker}: {method_name} - Alloc={allocation:.2%}, TypeBonus={method_type_bonus:.2f}, MLAlign={ml_alignment_score:.3f}, Score={score:.3f}")

        # Check if we have valid scores
        total_score = sum(method_scores.values())

        if total_score <= 0:
            logger.warning(f"{ticker}: No valid optimization scores, using default weights")
            return default_weights

        # Normalize scores to weights
        calibrated_weights = {
            name: score / total_score
            for name, score in method_scores.items()
        }

        # Apply smoothing with default weights (80% calibrated, 20% default)
        # This prevents over-fitting to recent performance
        alpha = 0.8
        smoothed_weights = {
            name: alpha * calibrated_weights.get(name, 0.0) + (1 - alpha) * default_weights.get(name, 0.0)
            for name in default_weights.keys()
        }

        # Renormalize
        total = sum(smoothed_weights.values())
        if total > 0:
            smoothed_weights = {name: w / total for name, w in smoothed_weights.items()}
        else:
            smoothed_weights = default_weights

        # Log changes
        logger.info(f"{ticker}: Portfolio optimization weights calibrated:")
        for name in default_weights.keys():
            old = default_weights[name]
            new = smoothed_weights[name]
            change = ((new - old) / old * 100) if old > 0 else 0
            logger.info(f"  {name}: {old:.3f} -> {new:.3f} ({change:+.1f}%)")

        # Cache
        self.portfolio_weights_cache[ticker] = smoothed_weights

        return smoothed_weights

    def get_cached_ml_weights(self, ticker: str, default_weights: Dict[str, float]) -> Dict[str, float]:
        """Get cached ML weights for ticker or return defaults."""
        return self.ml_weights_cache.get(ticker, default_weights)

    def get_cached_portfolio_weights(self, ticker: str, default_weights: Dict[str, float]) -> Dict[str, float]:
        """Get cached portfolio weights for ticker or return defaults."""
        return self.portfolio_weights_cache.get(ticker, default_weights)

    def reset_cache(self):
        """Clear all cached weights."""
        self.ml_weights_cache.clear()
        self.portfolio_weights_cache.clear()
        logger.info("Dynamic weight caches cleared")
