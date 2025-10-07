"""
Ensemble models using XGBoost, LightGBM, and CatBoost.
Provides gradient boosting implementations for stock prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Any
import logging
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available. Install: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not available. Install: pip install lightgbm")

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    logger.warning("CatBoost not available. Install: pip install catboost")


class XGBoostPredictor:
    """XGBoost model for stock prediction."""

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 7,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize XGBoost predictor.

        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            subsample: Subsample ratio
            colsample_bytree: Feature sampling ratio
            random_state: Random seed
            **kwargs: Additional XGBoost parameters
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost required. Install: pip install xgboost")

        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'random_state': random_state,
            'objective': 'reg:squarederror',
            'tree_method': 'hist',
            **kwargs
        }

        self.model = xgb.XGBRegressor(**self.params)
        self.feature_importance = None

        logger.info("XGBoost predictor initialized")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        early_stopping_rounds: int = 50,
        verbose: bool = True
    ):
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            early_stopping_rounds: Early stopping rounds
            verbose: Whether to print progress
        """
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        # Note: early_stopping_rounds is deprecated in newer XGBoost
        # Just train without early stopping for simplicity
        self.model.fit(
            X_train,
            y_train,
            verbose=verbose
        )

        # Store feature importance
        self.feature_importance = self.model.feature_importances_

        logger.info("XGBoost training completed")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)

    def get_feature_importance(self, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Get feature importance."""
        if self.feature_importance is None:
            logger.warning("Model not trained yet")
            return pd.DataFrame()

        importance_df = pd.DataFrame({
            'feature': feature_names if feature_names else [f'f{i}' for i in range(len(self.feature_importance))],
            'importance': self.feature_importance
        })

        importance_df = importance_df.sort_values('importance', ascending=False)
        return importance_df

    def save(self, path: str):
        """Save model."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f"XGBoost model saved to {path}")

    def load(self, path: str):
        """Load model."""
        self.model = joblib.load(path)
        self.feature_importance = self.model.feature_importances_
        logger.info(f"XGBoost model loaded from {path}")


class LightGBMPredictor:
    """LightGBM model for stock prediction."""

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 7,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize LightGBM predictor.

        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            num_leaves: Maximum number of leaves
            subsample: Subsample ratio
            colsample_bytree: Feature sampling ratio
            random_state: Random seed
            **kwargs: Additional LightGBM parameters
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM required. Install: pip install lightgbm")

        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'num_leaves': num_leaves,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'random_state': random_state,
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            **kwargs
        }

        self.model = lgb.LGBMRegressor(**self.params)
        self.feature_importance = None

        logger.info("LightGBM predictor initialized")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        early_stopping_rounds: int = 50,
        verbose: bool = False
    ):
        """Train the model."""
        callbacks = []
        if early_stopping_rounds:
            callbacks.append(lgb.early_stopping(early_stopping_rounds))

        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            callbacks=callbacks if callbacks else None
        )

        self.feature_importance = self.model.feature_importances_

        logger.info("LightGBM training completed")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)

    def get_feature_importance(self, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Get feature importance."""
        if self.feature_importance is None:
            logger.warning("Model not trained yet")
            return pd.DataFrame()

        importance_df = pd.DataFrame({
            'feature': feature_names if feature_names else [f'f{i}' for i in range(len(self.feature_importance))],
            'importance': self.feature_importance
        })

        importance_df = importance_df.sort_values('importance', ascending=False)
        return importance_df

    def save(self, path: str):
        """Save model."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f"LightGBM model saved to {path}")

    def load(self, path: str):
        """Load model."""
        self.model = joblib.load(path)
        self.feature_importance = self.model.feature_importances_
        logger.info(f"LightGBM model loaded from {path}")


class CatBoostPredictor:
    """CatBoost model for stock prediction."""

    def __init__(
        self,
        iterations: int = 500,
        depth: int = 7,
        learning_rate: float = 0.05,
        random_state: int = 42,
        **kwargs
    ):
        """Initialize CatBoost predictor."""
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost required. Install: pip install catboost")

        self.params = {
            'iterations': iterations,
            'depth': depth,
            'learning_rate': learning_rate,
            'random_state': random_state,
            'verbose': False,
            **kwargs
        }

        self.model = CatBoostRegressor(**self.params)
        self.feature_importance = None

        logger.info("CatBoost predictor initialized")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        early_stopping_rounds: int = 50
    ):
        """Train the model."""
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = (X_val, y_val)

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            early_stopping_rounds=early_stopping_rounds,
            verbose=False
        )

        self.feature_importance = self.model.get_feature_importance()

        logger.info("CatBoost training completed")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)

    def get_feature_importance(self, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Get feature importance."""
        if self.feature_importance is None:
            logger.warning("Model not trained yet")
            return pd.DataFrame()

        importance_df = pd.DataFrame({
            'feature': feature_names if feature_names else [f'f{i}' for i in range(len(self.feature_importance))],
            'importance': self.feature_importance
        })

        importance_df = importance_df.sort_values('importance', ascending=False)
        return importance_df

    def save(self, path: str):
        """Save model."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(path)
        logger.info(f"CatBoost model saved to {path}")

    def load(self, path: str):
        """Load model."""
        self.model.load_model(path)
        self.feature_importance = self.model.get_feature_importance()
        logger.info(f"CatBoost model loaded from {path}")


class EnsemblePredictor:
    """Ensemble of multiple models with weighted predictions."""

    def __init__(
        self,
        models: Dict[str, Any],
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize ensemble predictor.

        Args:
            models: Dictionary of model name to model instance
            weights: Dictionary of model name to weight (if None, equal weights)
        """
        self.models = models
        self.weights = weights

        if self.weights is None:
            # Equal weights
            n_models = len(models)
            self.weights = {name: 1.0/n_models for name in models.keys()}

        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {k: v/total_weight for k, v in self.weights.items()}

        logger.info(f"Ensemble predictor initialized with {len(models)} models")
        logger.info(f"Weights: {self.weights}")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ):
        """Train all models in the ensemble."""
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            model.train(X_train, y_train, X_val, y_val)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions."""
        predictions = np.zeros(len(X))

        for name, model in self.models.items():
            model_pred = model.predict(X)
            predictions += model_pred * self.weights[name]

        return predictions

    def get_individual_predictions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get predictions from each individual model."""
        return {name: model.predict(X) for name, model in self.models.items()}

    def update_weights(self, new_weights: Dict[str, float]):
        """Update ensemble weights."""
        # Normalize
        total_weight = sum(new_weights.values())
        self.weights = {k: v/total_weight for k, v in new_weights.items()}
        logger.info(f"Updated weights: {self.weights}")

    def save(self, directory: str):
        """Save all models in ensemble."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        for name, model in self.models.items():
            model_path = directory / f"{name}_model.pkl"
            model.save(str(model_path))

        # Save weights
        import json
        weights_path = directory / "ensemble_weights.json"
        with open(weights_path, 'w') as f:
            json.dump(self.weights, f)

        logger.info(f"Ensemble saved to {directory}")

    def load(self, directory: str):
        """Load all models in ensemble."""
        directory = Path(directory)

        for name, model in self.models.items():
            model_path = directory / f"{name}_model.pkl"
            if model_path.exists():
                model.load(str(model_path))

        # Load weights
        import json
        weights_path = directory / "ensemble_weights.json"
        if weights_path.exists():
            with open(weights_path, 'r') as f:
                self.weights = json.load(f)

        logger.info(f"Ensemble loaded from {directory}")
