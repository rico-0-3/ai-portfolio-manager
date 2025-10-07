"""
Risk management module.
Implements VaR, CVaR, stop-loss, and position sizing.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class RiskManager:
    """Risk management system for portfolio."""

    def __init__(
        self,
        max_position_size: float = 0.20,
        max_portfolio_risk: float = 0.02,
        var_confidence: float = 0.95,
        max_drawdown: float = 0.25
    ):
        """
        Initialize risk manager.

        Args:
            max_position_size: Maximum position as fraction of portfolio
            max_portfolio_risk: Maximum portfolio risk per day
            var_confidence: VaR confidence level
            max_drawdown: Maximum allowed drawdown
        """
        self.max_position_size = max_position_size
        self.max_portfolio_risk = max_portfolio_risk
        self.var_confidence = var_confidence
        self.max_drawdown = max_drawdown

        logger.info("RiskManager initialized")

    def calculate_var(
        self,
        returns: pd.Series,
        confidence: float = 0.95,
        method: str = 'historical',
        ml_prediction: Optional[float] = None
    ) -> float:
        """
        Calculate Value at Risk with optional ML adjustment.

        Args:
            returns: Historical returns
            confidence: Confidence level
            method: 'historical', 'parametric', or 'monte_carlo'
            ml_prediction: Optional ML predicted return for adjustment

        Returns:
            VaR value (adjusted if ML prediction provided)
        """
        if method == 'historical':
            var = returns.quantile(1 - confidence)
        elif method == 'parametric':
            mu = returns.mean()
            sigma = returns.std()
            var = stats.norm.ppf(1 - confidence, mu, sigma)
        elif method == 'monte_carlo':
            simulations = np.random.normal(
                returns.mean(),
                returns.std(),
                10000
            )
            var = np.percentile(simulations, (1 - confidence) * 100)
        else:
            raise ValueError(f"Unknown method: {method}")

        # ML-adjustment: if ML predicts bearish market, increase VaR (more risk)
        if ml_prediction is not None:
            if ml_prediction < -0.01:  # Predicts >1% decline
                # Bearish: increase risk estimate by 30%
                adjustment_factor = 1.3
                logger.info(f"ML predicts bearish ({ml_prediction*100:.2f}%), increasing VaR by 30%")
            elif ml_prediction < 0:  # Slight negative
                # Cautious: increase risk by 15%
                adjustment_factor = 1.15
                logger.info(f"ML predicts slight negative ({ml_prediction*100:.2f}%), increasing VaR by 15%")
            elif ml_prediction > 0.02:  # Predicts >2% gain
                # Bullish: decrease risk estimate by 10%
                adjustment_factor = 0.9
                logger.info(f"ML predicts bullish ({ml_prediction*100:.2f}%), decreasing VaR by 10%")
            else:
                # Neutral
                adjustment_factor = 1.0

            var = var * adjustment_factor

        return var

    def calculate_cvar(
        self,
        returns: pd.Series,
        confidence: float = 0.95,
        ml_prediction: Optional[float] = None
    ) -> float:
        """
        Calculate Conditional VaR (Expected Shortfall) with optional ML adjustment.

        Args:
            returns: Historical returns
            confidence: Confidence level
            ml_prediction: Optional ML predicted return for adjustment

        Returns:
            CVaR value (adjusted if ML prediction provided)
        """
        var = self.calculate_var(returns, confidence, method='historical', ml_prediction=ml_prediction)
        cvar = returns[returns <= var].mean()

        # Apply same ML adjustment as VaR
        if ml_prediction is not None:
            if ml_prediction < -0.01:
                cvar = cvar * 1.3
            elif ml_prediction < 0:
                cvar = cvar * 1.15
            elif ml_prediction > 0.02:
                cvar = cvar * 0.9

        return cvar

    def calculate_position_size(
        self,
        portfolio_value: float,
        asset_volatility: float,
        max_risk: Optional[float] = None
    ) -> float:
        """
        Calculate optimal position size using volatility-based sizing.

        Args:
            portfolio_value: Total portfolio value
            asset_volatility: Asset's volatility
            max_risk: Maximum risk (if None, uses portfolio max_risk)

        Returns:
            Position size in dollars
        """
        if max_risk is None:
            max_risk = self.max_portfolio_risk

        # Risk per dollar position
        risk_per_dollar = asset_volatility

        # Position size
        position_size = (portfolio_value * max_risk) / risk_per_dollar

        # Apply max position constraint
        max_position = portfolio_value * self.max_position_size
        position_size = min(position_size, max_position)

        return position_size

    def check_position_limits(
        self,
        weights: Dict[str, float],
        max_weight: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Ensure position weights respect limits.

        Args:
            weights: Dictionary of ticker to weight
            max_weight: Maximum weight per position

        Returns:
            Adjusted weights
        """
        if max_weight is None:
            max_weight = self.max_position_size

        adjusted_weights = {}

        for ticker, weight in weights.items():
            adjusted_weights[ticker] = min(weight, max_weight)

        # Renormalize
        total = sum(adjusted_weights.values())
        if total > 0:
            adjusted_weights = {k: v/total for k, v in adjusted_weights.items()}

        return adjusted_weights

    def calculate_portfolio_var(
        self,
        returns: pd.DataFrame,
        weights: Dict[str, float],
        confidence: float = 0.95,
        ml_predictions: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate portfolio-level VaR with optional ML adjustment.

        Args:
            returns: DataFrame of asset returns
            weights: Portfolio weights
            confidence: Confidence level
            ml_predictions: Optional dict of ML predictions per ticker

        Returns:
            Portfolio VaR (ML-adjusted if predictions provided)
        """
        # Weight array
        w = np.array([weights.get(col, 0) for col in returns.columns])

        # Portfolio returns
        portfolio_returns = (returns * w).sum(axis=1)

        # Calculate weighted average ML prediction for portfolio
        ml_pred_portfolio = None
        if ml_predictions:
            ml_pred_portfolio = sum(weights.get(ticker, 0) * ml_predictions.get(ticker, 0)
                                   for ticker in returns.columns)

        # Calculate VaR with ML adjustment
        var = self.calculate_var(portfolio_returns, confidence, ml_prediction=ml_pred_portfolio)

        return var

    def calculate_portfolio_cvar(
        self,
        returns: pd.DataFrame,
        weights: Dict[str, float],
        confidence: float = 0.95,
        ml_predictions: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate portfolio CVaR with optional ML adjustment.

        Args:
            returns: DataFrame of asset returns
            weights: Portfolio weights
            confidence: Confidence level
            ml_predictions: Optional dict of ML predictions per ticker

        Returns:
            Portfolio CVaR (ML-adjusted if predictions provided)
        """
        w = np.array([weights.get(col, 0) for col in returns.columns])
        portfolio_returns = (returns * w).sum(axis=1)

        # Calculate weighted average ML prediction for portfolio
        ml_pred_portfolio = None
        if ml_predictions:
            ml_pred_portfolio = sum(weights.get(ticker, 0) * ml_predictions.get(ticker, 0)
                                   for ticker in returns.columns)

        cvar = self.calculate_cvar(portfolio_returns, confidence, ml_prediction=ml_pred_portfolio)
        return cvar

    def calculate_stress_scenarios(
        self,
        returns: pd.DataFrame,
        weights: Dict[str, float],
        scenarios: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Calculate portfolio performance under stress scenarios.

        Args:
            returns: Historical returns
            weights: Portfolio weights
            scenarios: Dict of scenario name to market shock (as fraction)

        Returns:
            Dict of scenario to portfolio impact
        """
        if scenarios is None:
            scenarios = {
                'market_crash_10': -0.10,
                'market_crash_20': -0.20,
                'market_crash_30': -0.30
            }

        w = np.array([weights.get(col, 0) for col in returns.columns])

        results = {}
        for scenario_name, shock in scenarios.items():
            # Assume all assets move with market
            scenario_return = shock
            portfolio_loss = scenario_return * sum(w)
            results[scenario_name] = portfolio_loss

        return results

    def calculate_stop_loss(
        self,
        entry_price: float,
        atr: float,
        multiplier: float = 2.0
    ) -> float:
        """
        Calculate ATR-based stop loss.

        Args:
            entry_price: Entry price
            atr: Average True Range
            multiplier: ATR multiplier

        Returns:
            Stop loss price
        """
        stop_loss = entry_price - (atr * multiplier)
        return stop_loss

    def calculate_trailing_stop(
        self,
        current_price: float,
        highest_price: float,
        trailing_percent: float = 0.10
    ) -> float:
        """
        Calculate trailing stop loss.

        Args:
            current_price: Current price
            highest_price: Highest price since entry
            trailing_percent: Trailing percentage

        Returns:
            Trailing stop price
        """
        trailing_stop = highest_price * (1 - trailing_percent)
        return trailing_stop

    def check_drawdown(
        self,
        portfolio_values: pd.Series,
        threshold: Optional[float] = None
    ) -> Tuple[float, bool]:
        """
        Check if drawdown exceeds threshold.

        Args:
            portfolio_values: Time series of portfolio values
            threshold: Drawdown threshold (if None, uses max_drawdown)

        Returns:
            Tuple of (current_drawdown, exceeds_threshold)
        """
        if threshold is None:
            threshold = self.max_drawdown

        running_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - running_max) / running_max
        current_drawdown = drawdown.iloc[-1]

        exceeds = current_drawdown < -threshold

        return current_drawdown, exceeds

    def calculate_kelly_criterion(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        Calculate Kelly Criterion for position sizing.

        Args:
            win_rate: Probability of winning trade
            avg_win: Average win amount
            avg_loss: Average loss amount

        Returns:
            Kelly percentage (fraction of capital to risk)
        """
        if avg_loss == 0:
            return 0

        win_loss_ratio = avg_win / avg_loss
        kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio

        # Apply fractional Kelly (half Kelly for safety)
        kelly = max(0, kelly) * 0.5

        return kelly

    def diversification_ratio(
        self,
        returns: pd.DataFrame,
        weights: Dict[str, float]
    ) -> float:
        """
        Calculate portfolio diversification ratio.

        Args:
            returns: Asset returns
            weights: Portfolio weights

        Returns:
            Diversification ratio (higher is better)
        """
        w = np.array([weights.get(col, 0) for col in returns.columns])

        # Weighted average volatility
        individual_vols = returns.std().values
        weighted_avg_vol = np.sum(w * individual_vols)

        # Portfolio volatility
        cov_matrix = returns.cov().values
        portfolio_vol = np.sqrt(w @ cov_matrix @ w)

        # Diversification ratio
        div_ratio = weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 0

        return div_ratio

    def risk_contribution(
        self,
        returns: pd.DataFrame,
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate risk contribution of each asset.

        Args:
            returns: Asset returns
            weights: Portfolio weights

        Returns:
            Dict of ticker to risk contribution
        """
        w = np.array([weights.get(col, 0) for col in returns.columns])
        cov_matrix = returns.cov().values

        # Portfolio variance
        portfolio_var = w @ cov_matrix @ w

        # Marginal contribution to risk
        marginal_contrib = cov_matrix @ w

        # Risk contribution
        risk_contrib = w * marginal_contrib / np.sqrt(portfolio_var)

        result = dict(zip(returns.columns, risk_contrib))
        return result
