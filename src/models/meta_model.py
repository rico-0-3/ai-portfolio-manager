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
                
                # CRITICAL: Validate prediction is not NaN
                if pd.isna(pred):
                    logger.error(f"{ticker}: prediction is NaN! Skipping ticker")
                    logger.error(f"  Model: {ensemble_model}")
                    logger.error(f"  Features shape: {X_test.shape}")
                    logger.error(f"  Feature names: {feature_cols[:5]}...")
                    continue
                
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

        # Use PortfolioOptimizer class with ALL 5 methods
        from src.portfolio.optimizer import PortfolioOptimizer

        optimizer = PortfolioOptimizer(risk_free_rate=0.02)

        # Convert predictions to Dict format (required by optimizer methods)
        # Some methods expect Dict[str, float], not pd.Series
        if isinstance(predictions, pd.Series):
            expected_returns_dict = predictions.to_dict()
        else:
            expected_returns_dict = predictions

        # CRITICAL: Filter out NaN predictions BEFORE passing to optimizer
        # NaN predictions cause "Input contains NaN" errors in Markowitz
        clean_predictions = {}
        nan_tickers = []
        
        logger.debug(f"Filtering predictions: {expected_returns_dict}")
        
        for ticker, pred in expected_returns_dict.items():
            if pd.isna(pred):
                nan_tickers.append(ticker)
                logger.warning(f"Prediction for {ticker} is NaN (value={pred}), excluding from optimization")
            else:
                clean_predictions[ticker] = pred

        if nan_tickers:
            logger.warning(f"Excluded {len(nan_tickers)} tickers with NaN predictions: {nan_tickers}")
            logger.warning(f"Original predictions dict: {expected_returns_dict}")
            logger.warning(f"Clean predictions dict: {clean_predictions}")

        # If all predictions are NaN, fallback to equal weights
        if not clean_predictions:
            logger.error("All predictions are NaN! Using equal weights on all tickers")
            tickers = list(expected_returns_dict.keys())
            return {ticker: 1.0 / len(tickers) for ticker in tickers}

        # Filter historical_returns to match clean predictions
        valid_tickers = list(clean_predictions.keys())
        historical_returns = historical_returns[valid_tickers]
        expected_returns_dict = clean_predictions

        # FINAL VALIDATION: Double-check no NaN in clean predictions
        assert all(not pd.isna(v) for v in expected_returns_dict.values()), \
            f"CRITICAL: NaN found in clean_predictions! {expected_returns_dict}"

        logger.info(f"Optimizing portfolio with {len(expected_returns_dict)} valid predictions")
        logger.debug(f"Final predictions passed to optimizer: {expected_returns_dict}")

        # Calculate weights from each optimization method
        method_weights = {}

        # 1. Markowitz (Mean-Variance Optimization)
        if optimizer_weights.get('markowitz', 0) > 0:
            try:
                mw = optimizer.markowitz_optimization(
                    returns=historical_returns,
                    ml_predictions=expected_returns_dict
                )
                method_weights['markowitz'] = mw
                logger.debug(f"Markowitz weights: {mw}")
            except Exception as e:
                logger.warning(f"Markowitz failed: {e}")

        # 2. Black-Litterman (uses ml_predictions as views)
        if optimizer_weights.get('black_litterman', 0) > 0:
            try:
                # Black-Litterman uses 'views' parameter, not 'ml_predictions'
                blw = optimizer.black_litterman_optimization(
                    returns=historical_returns,
                    views=expected_returns_dict  # ml_predictions as absolute views
                )
                method_weights['black_litterman'] = blw
                logger.debug(f"Black-Litterman weights: {blw}")
            except Exception as e:
                logger.warning(f"Black-Litterman failed: {e}")

        # 3. Risk Parity
        if optimizer_weights.get('risk_parity', 0) > 0:
            try:
                rpw = optimizer.risk_parity_optimization(
                    returns=historical_returns,
                    ml_predictions=expected_returns_dict
                )
                method_weights['risk_parity'] = rpw
                logger.debug(f"Risk Parity weights: {rpw}")
            except Exception as e:
                logger.warning(f"Risk Parity failed: {e}")

        # 4. CVaR (Conditional Value at Risk)
        if optimizer_weights.get('cvar', 0) > 0:
            try:
                cw = optimizer.cvar_optimization(
                    returns=historical_returns,
                    ml_predictions=expected_returns_dict,
                    alpha=0.05
                )
                method_weights['cvar'] = cw
                logger.debug(f"CVaR weights: {cw}")
            except Exception as e:
                logger.warning(f"CVaR failed: {e}")

        # 5. RL Agent (Reinforcement Learning - PPO)
        if optimizer_weights.get('rl_agent', 0) > 0:
            try:
                rlw = optimizer.rl_agent_optimization(
                    returns=historical_returns,
                    ml_predictions=expected_returns_dict,
                    training_steps=10000  # 10k steps is reasonable for portfolio optimization
                )
                method_weights['rl_agent'] = rlw
                logger.debug(f"RL Agent weights: {rlw}")
            except Exception as e:
                logger.warning(f"RL Agent failed: {e}")

        # If no methods succeeded, fallback to equal weights
        if not method_weights:
            logger.warning("All optimization methods failed, using equal weights")
            tickers = list(predictions.keys())
            return {ticker: 1.0 / len(tickers) for ticker in tickers}

        # Ensemble: weighted average of all successful methods
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

        Strategy:
        1. If ML model exists (optimizer_weights_predictor), use it
        2. Otherwise, use rule-based adaptive weights based on market regime

        Research-backed rules:
        - High volatility + bear market: Risk Parity (40%) + CVaR (35%)
        - Low volatility + bull market: Markowitz (35%) + Black-Litterman (30%)
        - High correlation: Risk Parity (more diversification needed)
        - Trending market: Black-Litterman (incorporate views) + RL Agent
        - Uncertain/Mixed: Balanced ensemble

        Args:
            market_conditions: {'vix': 18.5, 'market_return_1m': 0.02, 'volatility': 0.15, ...}

        Returns:
            {'markowitz': 0.3, 'black_litterman': 0.2, ...}
        """
        # If ML model exists, use it
        if self.optimizer_weights_predictor is not None:
            try:
                from src.models.portfolio_optimizer_ml import PortfolioOptimizerML
                
                if isinstance(self.optimizer_weights_predictor, PortfolioOptimizerML):
                    weights = self.optimizer_weights_predictor.predict(
                        market_conditions=market_conditions
                    )
                    logger.info(f"ML-predicted weights: {weights}")
                    return weights
            except Exception as e:
                logger.warning(f"ML weight prediction failed: {e}, falling back to rule-based")

        # Rule-based adaptive weights based on market conditions
        volatility = market_conditions.get('volatility', 0.2)
        trend = market_conditions.get('trend', 0.0)
        correlation = market_conditions.get('correlation', 0.5)
        vix = market_conditions.get('vix', 0.2)
        regime = market_conditions.get('regime', 1)  # 1 = bull, -1 = bear
        max_drawdown = market_conditions.get('max_drawdown', 0.1)

        # Initialize weights
        weights = {
            'markowitz': 0.0,
            'black_litterman': 0.0,
            'risk_parity': 0.0,
            'cvar': 0.0,
            'rl_agent': 0.0
        }

        # Detect market regime
        high_volatility = volatility > 0.25 or vix > 0.3
        bear_market = regime < 0 or trend < -0.02
        bull_market = regime > 0 and trend > 0.02
        high_correlation = correlation > 0.7
        high_drawdown = max_drawdown > 0.15

        # SCENARIO 1: Crisis Mode (high vol + bear + high drawdown)
        if high_volatility and (bear_market or high_drawdown):
            logger.info("Market regime: CRISIS - defensive portfolio")
            weights['cvar'] = 0.40          # Focus on tail risk
            weights['risk_parity'] = 0.35   # Diversification
            weights['black_litterman'] = 0.15  # Some tactical views
            weights['markowitz'] = 0.05     # Minimal MVO
            weights['rl_agent'] = 0.05      # Adaptive learning

        # SCENARIO 2: Bull Market (low vol + positive trend)
        elif not high_volatility and bull_market:
            logger.info("Market regime: BULL - growth-oriented portfolio")
            weights['markowitz'] = 0.35      # Maximize Sharpe
            weights['black_litterman'] = 0.30  # Incorporate positive views
            weights['rl_agent'] = 0.20       # Adaptive to momentum
            weights['risk_parity'] = 0.10    # Some diversification
            weights['cvar'] = 0.05           # Minimal tail risk focus

        # SCENARIO 3: High Correlation (systemic risk)
        elif high_correlation:
            logger.info("Market regime: HIGH CORRELATION - diversification critical")
            weights['risk_parity'] = 0.45    # Maximum diversification
            weights['cvar'] = 0.25           # Tail risk protection
            weights['black_litterman'] = 0.15  # Tactical positioning
            weights['markowitz'] = 0.10
            weights['rl_agent'] = 0.05

        # SCENARIO 4: Moderate Volatility + Sideways Market
        elif volatility > 0.15 and abs(trend) < 0.01:
            logger.info("Market regime: SIDEWAYS - balanced approach")
            weights['risk_parity'] = 0.30
            weights['black_litterman'] = 0.25
            weights['markowitz'] = 0.20
            weights['cvar'] = 0.15
            weights['rl_agent'] = 0.10

        # SCENARIO 5: Default Balanced (uncertain conditions)
        else:
            logger.info("Market regime: BALANCED - ensemble approach")
            weights['markowitz'] = 0.25
            weights['black_litterman'] = 0.25
            weights['risk_parity'] = 0.20
            weights['cvar'] = 0.20
            weights['rl_agent'] = 0.10

        # Ensure weights sum to 1.0
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}

        logger.info(f"Rule-based adaptive weights: {weights}")
        return weights

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

        # Store last optimization details (for get_optimization_details())
        self._last_optimization_details = self._calculate_optimization_details()

        return weights

    def _calculate_optimization_details(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate per-ticker contribution from each optimization method.

        Returns:
            {ticker: {method: contribution_percentage}}
        """
        # This would require storing intermediate method_weights during optimize_portfolio()
        # For now, return empty dict - can be enhanced in future
        return {}

    def get_optimization_details(self) -> Dict[str, Dict[str, float]]:
        """
        Get details about which optimization methods contributed to each ticker's weight.

        Returns:
            {ticker: {method_name: contribution_percentage}}
        """
        # For now, return the static optimizer config as a proxy
        # In the future, this could track actual per-ticker method contributions
        return getattr(self, '_last_optimization_details', {})

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
