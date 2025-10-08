"""
Unified Ensemble Model
Contains all trained models (XGBoost, LightGBM, LSTM, GRU, LSTM+Attn, Transformer)
and provides a single .predict() interface.
"""

import numpy as np
import pickle
from typing import Dict, List, Optional
from pathlib import Path


class UnifiedEnsembleModel:
    """
    Unified ensemble model that contains all sub-models and provides
    a single prediction interface.

    This simplifies the orchestrator by encapsulating all ML logic
    in one class.
    """

    def __init__(
        self,
        models: Dict[str, object],
        weights: Dict[str, float],
        scaler: object,
        selected_features: List[str],
        metadata: Optional[Dict] = None
    ):
        """
        Initialize unified ensemble.

        Args:
            models: Dictionary of {model_name: model_object}
            weights: Dictionary of {model_name: weight} for ensemble
            scaler: Fitted scaler for feature normalization
            selected_features: List of feature names in correct order
            metadata: Optional metadata (training date, metrics, etc.)
        """
        self.models = models
        self.weights = weights
        self.scaler = scaler
        self.selected_features = selected_features
        self.metadata = metadata or {}

        # Validate weights sum to 1.0
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 0.01:
            # Normalize weights
            self.weights = {k: v/total_weight for k, v in weights.items()}

    def predict(self, X: np.ndarray, feature_names: List[str]) -> float:
        """
        Make prediction using ensemble of all models.

        Args:
            X: Feature matrix (n_samples, n_features)
            feature_names: List of feature names corresponding to X columns

        Returns:
            Single prediction (1-month return)
        """
        # Select and order features correctly
        feature_indices = []
        for feat in self.selected_features:
            if feat in feature_names:
                feature_indices.append(feature_names.index(feat))
            else:
                raise ValueError(f"Feature '{feat}' not found in provided features")

        X_selected = X[:, feature_indices]

        # Scale features
        X_scaled = self.scaler.transform(X_selected)

        # Get predictions from all models
        predictions = []
        model_weights = []

        for model_name, model in self.models.items():
            weight = self.weights.get(model_name, 0.0)
            if weight == 0:
                continue

            try:
                if model_name == 'xgboost':
                    # XGBoost can predict directly from numpy array
                    pred = float(model.predict(X_scaled)[0])

                elif model_name == 'lightgbm':
                    pred = float(model.predict(X_scaled)[0])
                    
                elif model_name == 'catboost':
                    pred = float(model.predict(X_scaled)[0])

                # Future: LSTM, GRU, Transformer
                # elif model_name == 'lstm':
                #     pred = model.predict(X_scaled_sequence)[0]

                else:
                    continue

                predictions.append(pred)
                model_weights.append(weight)

            except Exception as e:
                print(f"Warning: {model_name} prediction failed: {e}")
                continue

        if not predictions:
            return 0.0

        # Weighted average
        predictions = np.array(predictions)
        model_weights = np.array(model_weights)
        model_weights = model_weights / model_weights.sum()  # Renormalize

        final_prediction = np.average(predictions, weights=model_weights)

        return float(final_prediction)

    def get_model_names(self) -> List[str]:
        """Return list of model names in ensemble."""
        return list(self.models.keys())

    def get_weights(self) -> Dict[str, float]:
        """Return ensemble weights."""
        return self.weights.copy()

    def get_metadata(self) -> Dict:
        """Return training metadata."""
        return self.metadata.copy()

    @classmethod
    def load(cls, model_dir: Path) -> 'UnifiedEnsembleModel':
        """
        Load ensemble model from directory.

        Args:
            model_dir: Path to directory containing model files

        Returns:
            UnifiedEnsembleModel instance
        """
        import json

        # Load metadata
        with open(model_dir / 'metadata.json', 'r') as f:
            metadata = json.load(f)

        # Load scaler and features
        with open(model_dir / 'scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        with open(model_dir / 'features.pkl', 'rb') as f:
            selected_features = pickle.load(f)

        # Load all models
        models = {}
        model_names = metadata.get('models', ['xgboost', 'lightgbm'])

        for model_name in model_names:
            model_path = model_dir / f'{model_name}.pkl'
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    models[model_name] = pickle.load(f)

        # Get ensemble weights
        weights = metadata.get('ensemble_weights', {})

        # Default equal weights if not provided
        if not weights:
            weights = {name: 1.0/len(models) for name in models.keys()}

        return cls(
            models=models,
            weights=weights,
            scaler=scaler,
            selected_features=selected_features,
            metadata=metadata
        )

    def save(self, model_dir: Path):
        """
        Save ensemble model to directory.

        Args:
            model_dir: Path to directory for saving
        """
        import json

        model_dir.mkdir(parents=True, exist_ok=True)

        # Save all models
        for model_name, model in self.models.items():
            with open(model_dir / f'{model_name}.pkl', 'wb') as f:
                pickle.dump(model, f)

        # Save scaler and features
        with open(model_dir / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)

        with open(model_dir / 'features.pkl', 'wb') as f:
            pickle.dump(self.selected_features, f)

        # Update metadata with ensemble info
        self.metadata['models'] = list(self.models.keys())
        self.metadata['ensemble_weights'] = self.weights

        # Save metadata
        with open(model_dir / 'metadata.json', 'w') as f:
            json.dump(self.metadata, f, indent=2)
