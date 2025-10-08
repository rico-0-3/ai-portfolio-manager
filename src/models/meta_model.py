"""
MetaModel: Unified model containing prediction ensemble + portfolio optimization strategy.

Architecture:
1. Prediction Layer: Ensemble of ML models (XGBoost, LightGBM, LSTM, etc.)
2. Portfolio Optimization Layer: ML-optimized blend of methods (Markowitz, BL, CVaR, RP, RL)

Interface:
- MetaModel.predict(market_data) -> portfolio_weights
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import json
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class MetaModel:
    """
    MetaModel: End-to-end portfolio allocation model.

    Phase 1: Predict returns for each ticker (ensemble)
    Phase 2: Optimize portfolio weights using ML-learned strategy
    """

    def __init__(
        self,
        prediction_models: Dict[str, Any],  # {ticker: UnifiedEnsembleModel}
        portfolio_optimizer_config: Dict[str, float],  # {'markowitz': 0.25, 'black_litterman': 0.25, ...}
        optimizer_weights_predictor: Optional[Any] = None,  # ML model that predicts optimal method weights
        metadata: Optional[Dict] = None
    ):
        """
        Args:
            prediction_models: Dictionary mapping ticker -> UnifiedEnsembleModel
            portfolio_optimizer_config: Static weights for each optimization method
            optimizer_weights_predictor: (Optional) ML model that dynamically adjusts method weights
            metadata: Training metadata (tickers, training date, accuracy metrics, etc.)
        """
        self.prediction_models = prediction_models
        self.portfolio_optimizer_config = portfolio_optimizer_config
        self.optimizer_weights_predictor = optimizer_weights_predictor
        self.metadata = metadata or {}

        logger.info(f"MetaModel initialized with {len(prediction_models)} tickers")
        logger.info(f"Portfolio optimizer config: {portfolio_optimizer_config}")

    def predict_returns(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Phase 1: Get return predictions for each ticker.

        Args:
            market_data: {ticker: DataFrame with features}

        Returns:
            {ticker: predicted_return_1m}
        """
        predictions = {}

        for ticker, df in market_data.items():
            if ticker not in self.prediction_models:
                logger.warning(f"No prediction model for {ticker}, skipping")
                continue

            try:
                ensemble_model = self.prediction_models[ticker]

                # Get latest features
                feature_cols = ensemble_model.selected_features
                X = df[feature_cols].values
                X_test = X[-1].reshape(1, -1)

                # Predict
                pred = ensemble_model.predict(X_test, feature_cols)
                predictions[ticker] = pred

                logger.debug(f"{ticker}: prediction = {pred*100:+.2f}%")

            except Exception as e:
                logger.error(f"{ticker}: prediction failed - {e}")
                continue

        return predictions

    def optimize_portfolio(
        self,
        predictions: Dict[str, float],
        historical_returns: pd.DataFrame,
        market_conditions: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Phase 2: Calculate optimal portfolio weights.

        Args:
            predictions: {ticker: predicted_return}
            historical_returns: DataFrame of historical returns for covariance calculation
            market_conditions: (Optional) Current market features for dynamic optimizer selection

        Returns:
            {ticker: weight}
        """

        # Dynamic optimizer weights (if ML model trained)
        if self.optimizer_weights_predictor is not None and market_conditions is not None:
            try:
                # ML model predicts optimal method weights based on market conditions
                optimizer_weights = self._predict_optimizer_weights(market_conditions)
                logger.info(f"Dynamic optimizer weights: {optimizer_weights}")
            except Exception as e:
                logger.warning(f"Failed to predict dynamic optimizer weights: {e}, using static config")
                optimizer_weights = self.portfolio_optimizer_config
        else:
            optimizer_weights = self.portfolio_optimizer_config

        # TEMPORARY FIX: Use PortfolioOptimizer class instead of standalone functions
        from src.portfolio.optimizer import PortfolioOptimizer
        
        try:
            optimizer = PortfolioOptimizer()
            
            # Convert predictions to expected return format
            expected_returns = pd.Series(predictions)
            
            # Calculate optimal weights
            weights = optimizer.optimize_portfolio(
                expected_returns=expected_returns,
                returns_df=historical_returns,
                method='markowitz'  # Use default method for now
            )
            
            return weights
            
        except Exception as e:
            logger.warning(f"Portfolio optimization failed: {e}, using equal weights")
            # Fallback: equal weights
            tickers = list(predictions.keys())
            return {ticker: 1.0 / len(tickers) for ticker in tickers}

        # OLD CODE (standalone functions don't exist):
        # from src.portfolio.optimizer import (
        #     optimize_portfolio_markowitz,
        #     optimize_portfolio_black_litterman,
        #     optimize_portfolio_risk_parity,
        #     optimize_portfolio_cvar
        # )

        # Calculate weights from each method
        method_weights = {}

        # Markowitz
        if optimizer_weights.get('markowitz', 0) > 0:
            try:
                mw = optimize_portfolio_markowitz(
                    returns=historical_returns,
                    ml_predictions=predictions,
                    risk_free_rate=0.02
                )
                method_weights['markowitz'] = mw
                logger.debug(f"Markowitz weights: {mw}")
            except Exception as e:
                logger.warning(f"Markowitz failed: {e}")

        # Black-Litterman
        if optimizer_weights.get('black_litterman', 0) > 0:
            try:
                blw = optimize_portfolio_black_litterman(
                    returns=historical_returns,
                    ml_predictions=predictions,
                    risk_free_rate=0.02
                )
                method_weights['black_litterman'] = blw
                logger.debug(f"Black-Litterman weights: {blw}")
            except Exception as e:
                logger.warning(f"Black-Litterman failed: {e}")

        # Risk Parity
        if optimizer_weights.get('risk_parity', 0) > 0:
            try:
                rpw = optimize_portfolio_risk_parity(
                    returns=historical_returns,
                    ml_predictions=predictions
                )
                method_weights['risk_parity'] = rpw
                logger.debug(f"Risk Parity weights: {rpw}")
            except Exception as e:
                logger.warning(f"Risk Parity failed: {e}")

        # CVaR
        if optimizer_weights.get('cvar', 0) > 0:
            try:
                cw = optimize_portfolio_cvar(
                    returns=historical_returns,
                    ml_predictions=predictions,
                    alpha=0.05
                )
                method_weights['cvar'] = cw
                logger.debug(f"CVaR weights: {cw}")
            except Exception as e:
                logger.warning(f"CVaR failed: {e}")

        # RL Agent (placeholder - will be trained in Phase 2)
        if optimizer_weights.get('rl_agent', 0) > 0:
            logger.warning("RL Agent not yet implemented, skipping")

        # Ensemble: weighted average of all methods
        final_weights = self._ensemble_portfolio_weights(method_weights, optimizer_weights)

        return final_weights

    def _ensemble_portfolio_weights(
        self,
        method_weights: Dict[str, Dict[str, float]],
        optimizer_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Combine weights from multiple optimization methods.

        Args:
            method_weights: {'markowitz': {ticker: weight}, 'black_litterman': {...}, ...}
            optimizer_weights: {'markowitz': 0.25, 'black_litterman': 0.25, ...}

        Returns:
            {ticker: final_weight}
        """
        if not method_weights:
            logger.error("No valid method weights, cannot ensemble")
            return {}

        # Get all tickers
        all_tickers = set()
        for weights in method_weights.values():
            all_tickers.update(weights.keys())

        # Weighted average
        final_weights = {ticker: 0.0 for ticker in all_tickers}

        total_method_weight = 0.0
        for method_name, weights in method_weights.items():
            method_weight = optimizer_weights.get(method_name, 0)
            total_method_weight += method_weight

            for ticker in all_tickers:
                ticker_weight = weights.get(ticker, 0.0)
                final_weights[ticker] += method_weight * ticker_weight

        # Normalize
        if total_method_weight > 0:
            final_weights = {k: v/total_method_weight for k, v in final_weights.items()}

        # Ensure weights sum to 1.0
        total_weight = sum(final_weights.values())
        if total_weight > 0:
            final_weights = {k: v/total_weight for k, v in final_weights.items()}

        return final_weights

    def _predict_optimizer_weights(self, market_conditions: Dict[str, float]) -> Dict[str, float]:
        """
        Use ML model to predict optimal optimizer method weights based on market conditions.

        Args:
            market_conditions: {'vix': 18.5, 'market_return_1m': 0.02, 'volatility': 0.15, ...}

        Returns:
            {'markowitz': 0.3, 'black_litterman': 0.2, ...}
        """
        # TODO: Implement ML predictor (Phase 2)
        # For now, return static config
        return self.portfolio_optimizer_config

    def predict(
        self,
        market_data: Dict[str, pd.DataFrame],
        historical_returns: pd.DataFrame,
        market_conditions: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        End-to-end prediction: market data -> portfolio weights.

        Args:
            market_data: {ticker: DataFrame with features}
            historical_returns: Historical returns for covariance
            market_conditions: (Optional) Current market features

        Returns:
            {ticker: weight}
        """
        # Phase 1: Predict returns
        predictions = self.predict_returns(market_data)

        if not predictions:
            logger.error("No predictions available, cannot optimize portfolio")
            return {}

        # Phase 2: Optimize portfolio
        weights = self.optimize_portfolio(predictions, historical_returns, market_conditions)

        return weights

    @classmethod
    def load(cls, model_dir: Path) -> "MetaModel":
        """Load MetaModel from directory."""
        model_dir = Path(model_dir)

        # Load metadata
        metadata_path = model_dir / "meta_model_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Load portfolio optimizer config
        optimizer_config_path = model_dir / "portfolio_optimizer_config.json"
        with open(optimizer_config_path, 'r') as f:
            portfolio_optimizer_config = json.load(f)

        # Load prediction models for each ticker
        from src.models.ensemble_model_unified import UnifiedEnsembleModel

        prediction_models = {}
        tickers = metadata.get('tickers', [])

        for ticker in tickers:
            ticker_dir = model_dir / ticker
            if not ticker_dir.exists():
                logger.warning(f"No model directory for {ticker}")
                continue

            try:
                ensemble_model = UnifiedEnsembleModel.load(ticker_dir)
                prediction_models[ticker] = ensemble_model
            except Exception as e:
                logger.error(f"Failed to load model for {ticker}: {e}")

        # Load optimizer weights predictor (if exists)
        optimizer_predictor_path = model_dir / "optimizer_weights_predictor.pkl"
        optimizer_weights_predictor = None
        if optimizer_predictor_path.exists():
            with open(optimizer_predictor_path, 'rb') as f:
                optimizer_weights_predictor = pickle.load(f)

        logger.info(f"MetaModel loaded: {len(prediction_models)} tickers")

        return cls(
            prediction_models=prediction_models,
            portfolio_optimizer_config=portfolio_optimizer_config,
            optimizer_weights_predictor=optimizer_weights_predictor,
            metadata=metadata
        )

    def save(self, model_dir: Path):
        """Save MetaModel to directory."""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata_path = model_dir / "meta_model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

        # Save portfolio optimizer config
        optimizer_config_path = model_dir / "portfolio_optimizer_config.json"
        with open(optimizer_config_path, 'w') as f:
            json.dump(self.portfolio_optimizer_config, f, indent=2)

        # Save each ticker's prediction model
        for ticker, ensemble_model in self.prediction_models.items():
            ticker_dir = model_dir / ticker
            ensemble_model.save(ticker_dir)

        # Save optimizer weights predictor (if exists)
        if self.optimizer_weights_predictor is not None:
            optimizer_predictor_path = model_dir / "optimizer_weights_predictor.pkl"
            with open(optimizer_predictor_path, 'wb') as f:
                pickle.dump(self.optimizer_weights_predictor, f)

        logger.info(f"MetaModel saved to {model_dir}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the MetaModel."""
        return {
            'num_tickers': len(self.prediction_models),
            'tickers': list(self.prediction_models.keys()),
            'portfolio_optimizer_config': self.portfolio_optimizer_config,
            'has_dynamic_optimizer': self.optimizer_weights_predictor is not None,
            'metadata': self.metadata
        }
