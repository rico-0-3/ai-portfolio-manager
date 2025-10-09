"""
ML-Based Portfolio Optimizer Weight Selection

Instead of fixed weights (25% Markowitz, 30% BL, etc.),
train an ML model to predict optimal method weights based on:
- Market conditions (volatility, trend, correlation)
- Historical performance (Sharpe ratio per method)
- Ticker characteristics (sector, beta, market cap)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class PortfolioOptimizerML:
    """
    ML model that predicts optimal portfolio optimization method weights
    based on market conditions and historical performance.
    """

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.method_names = ['markowitz', 'black_litterman', 'risk_parity', 'cvar', 'rl_agent']

    def extract_market_features(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Extract market condition features from returns.

        Args:
            returns: DataFrame of historical returns (tickers x time)

        Returns:
            Feature vector (1D array)
        """
        features = []

        # 1. Volatility (annualized)
        volatility = returns.std().mean() * np.sqrt(252)
        features.append(volatility)

        # 2. Trend (recent 21-day return)
        recent_return = returns.iloc[-21:].mean().mean()
        features.append(recent_return)

        # 3. Correlation (average pairwise correlation)
        corr_matrix = returns.corr()
        avg_correlation = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
        features.append(avg_correlation)

        # 4. Skewness (average)
        skewness = returns.skew().mean()
        features.append(skewness)

        # 5. Kurtosis (average)
        kurtosis = returns.kurt().mean()
        features.append(kurtosis)

        # 6. VIX proxy (rolling 21-day volatility)
        rolling_vol = returns.rolling(21).std().mean().mean() * np.sqrt(252)
        features.append(rolling_vol)

        # 7. Max drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min().mean()
        features.append(abs(max_drawdown))

        # 8. Sharpe ratio (recent 60 days)
        recent_sharpe = (returns.iloc[-60:].mean() / returns.iloc[-60:].std()).mean() * np.sqrt(252)
        features.append(recent_sharpe)

        # 9. Market regime: Bull (1) or Bear (-1)
        regime = 1 if recent_return > 0 else -1
        features.append(regime)

        # 10. Diversification ratio (portfolio vol / weighted avg vol)
        portfolio_vol = returns.mean(axis=1).std()
        weighted_avg_vol = returns.std().mean()
        diversification_ratio = weighted_avg_vol / (portfolio_vol + 1e-6)
        features.append(diversification_ratio)

        return np.array(features)

    def generate_training_data(
        self,
        historical_returns: pd.DataFrame,
        window_size: int = 252,
        step_size: int = 21
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate training data by simulating different market conditions
        and calculating optimal method weights.

        Args:
            historical_returns: Historical returns (long history, e.g., 5+ years)
            window_size: Rolling window size for each training sample (252 = 1 year)
            step_size: Step between windows (21 = 1 month)

        Returns:
            X: Market features (n_samples, n_features)
            y: Optimal method weights (n_samples, n_methods)
        """
        logger.info("Generating training data for Portfolio Optimizer ML...")

        from src.portfolio.optimizer import PortfolioOptimizer

        X_list = []
        y_list = []

        # Slide window through historical data
        for start in range(0, len(historical_returns) - window_size - 21, step_size):
            end = start + window_size
            future_end = end + 21  # Next month for testing

            # Current window
            window_returns = historical_returns.iloc[start:end]

            # Future returns (for scoring methods)
            future_returns = historical_returns.iloc[end:future_end]

            # Extract features
            features = self.extract_market_features(window_returns)

            # Initialize optimizer
            optimizer = PortfolioOptimizer(risk_free_rate=0.02)

            # Test each optimization method
            method_scores = {}

            try:
                # Markowitz
                weights_mv = optimizer.markowitz_optimization(window_returns)
                future_ret_mv = (future_returns * pd.Series(weights_mv)).sum(axis=1).mean()
                future_vol_mv = (future_returns * pd.Series(weights_mv)).sum(axis=1).std()
                sharpe_mv = future_ret_mv / (future_vol_mv + 1e-6) * np.sqrt(252)
                method_scores['markowitz'] = max(sharpe_mv, 0)  # Clip negative Sharpe

                # Black-Litterman
                weights_bl = optimizer.black_litterman_optimization(window_returns)
                future_ret_bl = (future_returns * pd.Series(weights_bl)).sum(axis=1).mean()
                future_vol_bl = (future_returns * pd.Series(weights_bl)).sum(axis=1).std()
                sharpe_bl = future_ret_bl / (future_vol_bl + 1e-6) * np.sqrt(252)
                method_scores['black_litterman'] = max(sharpe_bl, 0)

                # Risk Parity
                weights_rp = optimizer.risk_parity_optimization(window_returns)
                future_ret_rp = (future_returns * pd.Series(weights_rp)).sum(axis=1).mean()
                future_vol_rp = (future_returns * pd.Series(weights_rp)).sum(axis=1).std()
                sharpe_rp = future_ret_rp / (future_vol_rp + 1e-6) * np.sqrt(252)
                method_scores['risk_parity'] = max(sharpe_rp, 0)

                # CVaR
                weights_cvar = optimizer.cvar_optimization(window_returns, alpha=0.05)
                future_ret_cvar = (future_returns * pd.Series(weights_cvar)).sum(axis=1).mean()
                future_vol_cvar = (future_returns * pd.Series(weights_cvar)).sum(axis=1).std()
                sharpe_cvar = future_ret_cvar / (future_vol_cvar + 1e-6) * np.sqrt(252)
                method_scores['cvar'] = max(sharpe_cvar, 0)

                # RL Agent (placeholder - use equal weight for now)
                method_scores['rl_agent'] = 0.5

            except Exception as e:
                logger.warning(f"Skipping window {start}-{end}: {e}")
                continue

            # Convert scores to weights (softmax)
            scores_array = np.array([method_scores[m] for m in self.method_names])
            # Softmax with temperature
            exp_scores = np.exp(scores_array / 0.5)  # Temperature = 0.5
            weights = exp_scores / exp_scores.sum()

            X_list.append(features)
            y_list.append(weights)

        if not X_list:
            logger.error("No training data generated!")
            return None, None

        X = np.array(X_list)
        y = np.array(y_list)

        logger.info(f"Generated {len(X)} training samples")
        logger.info(f"Feature shape: {X.shape}, Target shape: {y.shape}")

        return X, y

    def train(self, historical_returns: pd.DataFrame):
        """
        Train ML model to predict optimal method weights.

        Args:
            historical_returns: Long historical returns (5+ years recommended)
        """
        logger.info("Training Portfolio Optimizer ML...")

        # Generate training data
        X, y = self.generate_training_data(historical_returns, window_size=252, step_size=21)

        if X is None or len(X) == 0:
            logger.error("No training data available")
            return

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train separate regressor for each method
        self.model = {}
        for i, method_name in enumerate(self.method_names):
            logger.info(f"  Training regressor for {method_name}...")
            regressor = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                random_state=42,
                n_jobs=-1
            )
            regressor.fit(X_scaled, y[:, i])
            self.model[method_name] = regressor

        logger.info("✓ Portfolio Optimizer ML trained")

    def predict(self, market_conditions: Dict[str, float] = None, returns: pd.DataFrame = None) -> Dict[str, float]:
        """
        Predict optimal method weights for current market conditions.

        Args:
            market_conditions: Optional dict with pre-computed features
            returns: Optional returns DataFrame (will extract features)

        Returns:
            Dict of {method_name: weight}
        """
        if self.model is None:
            logger.warning("Model not trained, returning equal weights")
            return {name: 1.0/len(self.method_names) for name in self.method_names}

        # Extract features
        if market_conditions:
            # Use provided features
            features = np.array([
                market_conditions.get('volatility', 0.2),
                market_conditions.get('trend', 0.0),
                market_conditions.get('correlation', 0.5),
                market_conditions.get('skewness', 0.0),
                market_conditions.get('kurtosis', 3.0),
                market_conditions.get('vix', 0.2),
                market_conditions.get('max_drawdown', 0.1),
                market_conditions.get('sharpe', 1.0),
                market_conditions.get('regime', 1),
                market_conditions.get('diversification_ratio', 1.5)
            ])
        elif returns is not None:
            features = self.extract_market_features(returns)
        else:
            logger.error("Must provide either market_conditions or returns")
            return {name: 1.0/len(self.method_names) for name in self.method_names}

        # Scale and predict
        features_scaled = self.scaler.transform(features.reshape(1, -1))

        weights = {}
        for method_name in self.method_names:
            weight = self.model[method_name].predict(features_scaled)[0]
            weights[method_name] = max(weight, 0.0)  # Clip negative weights

        # Normalize to sum = 1.0
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
        else:
            weights = {name: 1.0/len(self.method_names) for name in self.method_names}

        return weights

    def save(self, filepath: str):
        """Save trained model."""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({'model': self.model, 'scaler': self.scaler, 'method_names': self.method_names}, f)
        logger.info(f"Portfolio Optimizer ML saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'PortfolioOptimizerML':
        """Load trained model."""
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        instance = cls()
        instance.model = data['model']
        instance.scaler = data['scaler']
        instance.method_names = data['method_names']

        logger.info(f"Portfolio Optimizer ML loaded from {filepath}")
        return instance
