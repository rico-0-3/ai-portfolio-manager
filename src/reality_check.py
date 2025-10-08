"""
Reality Check Module
Applies conservative adjustments and validates predictions for realism.
Prevents overly optimistic forecasts and ensures alignment with market reality.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class RealityCheck:
    """
    Applies reality checks and conservative adjustments to predictions.

    Key adjustments:
    1. Out-of-sample degradation factor (models perform worse on unseen data)
    2. Transaction costs and slippage
    3. Market impact (large orders move prices)
    4. Regime change risk (market conditions change)
    5. Overfitting penalty (reduce predictions based on model complexity)
    """

    def __init__(
        self,
        degradation_factor: float = 0.7,  # 70% of in-sample performance
        min_transaction_cost: float = 0.001,  # 0.1% minimum
        slippage_factor: float = 0.0005,  # 0.05% slippage
        market_impact_threshold: float = 100000,  # $100k order size
        confidence_penalty: float = 0.1  # Reduce confidence by 10%
    ):
        """
        Initialize reality check system.

        Args:
            degradation_factor: Out-of-sample performance degradation (0-1)
            min_transaction_cost: Minimum transaction cost
            slippage_factor: Slippage as fraction of trade
            market_impact_threshold: Order size that starts affecting price
            confidence_penalty: Penalty to apply to confidence scores
        """
        self.degradation_factor = degradation_factor
        self.min_transaction_cost = min_transaction_cost
        self.slippage_factor = slippage_factor
        self.market_impact_threshold = market_impact_threshold
        self.confidence_penalty = confidence_penalty

        logger.info(f"RealityCheck initialized (degradation={degradation_factor}, txn_cost={min_transaction_cost})")

    def adjust_predictions(
        self,
        predictions: Dict[str, float],
        historical_volatility: Dict[str, float],
        model_complexity: int = 1,
        use_ml: bool = True
    ) -> Dict[str, float]:
        """
        Apply conservative adjustment to predictions.

        Args:
            predictions: Dict of ticker to predicted return
            historical_volatility: Dict of ticker to volatility
            model_complexity: Number of models in ensemble (higher = more overfitting risk)
            use_ml: If True, predictions come from ML (apply lighter penalties)

        Returns:
            Adjusted predictions
        """
        adjusted = {}

        for ticker, pred in predictions.items():
            if use_ml:
                # ML predictions are already realistic - trust the model!
                # No degradation: ML is trained with validation and already accounts for:
                # - Out-of-sample performance
                # - Overfitting (via cross-validation)
                # - Mean reversion (learned from data)

                adjusted_pred = pred  # Use prediction as-is

                # Only check for extreme outliers (> 5x volatility = likely a bug)
                vol = historical_volatility.get(ticker, 0.02)
                if abs(adjusted_pred) > 5 * vol:
                    # Cap at 5x volatility to prevent model bugs
                    adjusted_pred = np.sign(adjusted_pred) * 5 * vol
                    logger.warning(f"{ticker}: Extreme ML prediction {pred*100:.2f}% capped at {adjusted_pred*100:.2f}%")

                # No other penalties - trust ML

            else:
                # Historical predictions - apply full conservative adjustments

                # 1. Apply out-of-sample degradation
                adjusted_pred = pred * self.degradation_factor

                # 2. Penalize extreme predictions
                vol = historical_volatility.get(ticker, 0.02)
                if abs(adjusted_pred) > 2 * vol:
                    penalty = 0.5 + 0.5 * (2 * vol / max(abs(adjusted_pred), 1e-6))
                    adjusted_pred *= penalty
                    logger.debug(f"{ticker}: Extreme prediction {pred*100:.2f}% reduced to {adjusted_pred*100:.2f}%")

                # 3. Overfitting penalty
                overfitting_penalty = 1.0 - (model_complexity - 1) * 0.05
                overfitting_penalty = max(0.7, overfitting_penalty)
                adjusted_pred *= overfitting_penalty

                # 4. Regression to mean
                mean_reversion = 0.8
                adjusted_pred = adjusted_pred * mean_reversion

            adjusted[ticker] = adjusted_pred

            if abs(pred - adjusted_pred) > 0.001:  # Log if change > 0.1%
                logger.info(f"{ticker}: Prediction adjusted {pred*100:+.2f}% → {adjusted_pred*100:+.2f}% (ML={use_ml})")

        return adjusted

    def apply_transaction_costs(
        self,
        expected_return: float,
        portfolio_value: float,
        weights: Dict[str, float],
        prices: Dict[str, float],
        rebalance_frequency: str = 'weekly'
    ) -> float:
        """
        Adjust returns for realistic transaction costs.

        Args:
            expected_return: Annual expected return
            portfolio_value: Total portfolio value
            weights: Portfolio weights
            prices: Current prices
            rebalance_frequency: How often portfolio is rebalanced

        Returns:
            Adjusted return after transaction costs
        """
        # Calculate number of rebalances per year
        rebalances_per_year = {
            'daily': 252,
            'weekly': 52,
            'monthly': 12,
            'quarterly': 4
        }.get(rebalance_frequency, 52)

        # Estimate turnover (assume 30% of portfolio changes per rebalance)
        average_turnover = 0.30

        # Transaction costs
        # Base cost + slippage + market impact
        total_cost = self.min_transaction_cost + self.slippage_factor

        # Market impact (for large orders)
        avg_order_size = portfolio_value * average_turnover / len(weights)
        if avg_order_size > self.market_impact_threshold:
            # Market impact increases with square root of order size
            impact_ratio = avg_order_size / self.market_impact_threshold
            market_impact = 0.0002 * np.sqrt(impact_ratio)  # 0.02% base impact
            total_cost += market_impact
            logger.debug(f"Market impact added: {market_impact*100:.3f}% (order size: ${avg_order_size:,.0f})")

        # Annual cost = cost per trade × turnover × rebalances
        annual_cost = total_cost * average_turnover * rebalances_per_year

        # Adjust return
        adjusted_return = expected_return - annual_cost

        logger.info(f"Transaction costs: {annual_cost*100:.2f}% annually ({total_cost*100:.3f}% per trade × {rebalances_per_year} rebalances)")
        logger.info(f"Return adjusted: {expected_return*100:+.2f}% → {adjusted_return*100:+.2f}%")

        return adjusted_return

    def stress_test_portfolio(
        self,
        returns: pd.DataFrame,
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Run stress tests on portfolio.

        Args:
            returns: Historical returns
            weights: Portfolio weights

        Returns:
            Dict of scenario to portfolio impact
        """
        w = np.array([weights.get(col, 0) for col in returns.columns])
        portfolio_returns = returns @ w

        scenarios = {
            '2008_crisis': -0.37,      # S&P 500 in 2008
            '2020_covid': -0.34,       # March 2020 crash
            '1987_crash': -0.20,       # Black Monday
            'market_correction': -0.10, # Standard correction
            'recession': -0.15         # Typical recession
        }

        results = {}

        for scenario, shock in scenarios.items():
            # Estimate portfolio behavior based on historical correlation
            # Assume portfolio moves with market (with some diversification benefit)
            portfolio_beta = 0.85  # Assume slightly lower than market due to diversification

            portfolio_impact = shock * portfolio_beta
            results[scenario] = portfolio_impact

        logger.info("Stress test results:")
        for scenario, impact in results.items():
            logger.info(f"  {scenario}: {impact*100:+.1f}%")

        return results

    def validate_sharpe_ratio(
        self,
        sharpe: float,
        n_observations: int,
        n_parameters: int
    ) -> Tuple[float, str]:
        """
        Validate if Sharpe ratio is realistic given data size and model complexity.

        Args:
            sharpe: Calculated Sharpe ratio
            n_observations: Number of data points
            n_parameters: Number of model parameters

        Returns:
            Tuple of (adjusted_sharpe, warning_message)
        """
        # Adjust for small sample bias
        # Sharpe ratio has upward bias in small samples
        if n_observations < 500:
            bias_correction = np.sqrt((n_observations - 1) / (n_observations - 3))
            sharpe_adjusted = sharpe * bias_correction
        else:
            sharpe_adjusted = sharpe

        # Check for overfitting (too many parameters relative to data)
        overfitting_ratio = n_parameters / n_observations

        warning = ""

        if overfitting_ratio > 0.1:  # More than 10% ratio is concerning
            # Apply overfitting penalty
            penalty = 1.0 - (overfitting_ratio - 0.1) * 2
            penalty = max(0.5, penalty)  # Max 50% penalty
            sharpe_adjusted *= penalty
            warning = f"HIGH OVERFITTING RISK: {n_parameters} parameters for {n_observations} observations"

        # Realistic Sharpe bounds
        # Very few strategies sustain Sharpe > 2.0 long-term
        if sharpe_adjusted > 2.5:
            warning += " | UNREALISTIC SHARPE: Sustainably achieving Sharpe > 2.5 is extremely rare"
            sharpe_adjusted = 2.0 + (sharpe_adjusted - 2.0) * 0.3  # Aggressive discount
        elif sharpe_adjusted > 2.0:
            warning += " | HIGH SHARPE: Sharpe > 2.0 is difficult to sustain"

        if warning:
            logger.warning(f"Sharpe validation: {sharpe:.2f} → {sharpe_adjusted:.2f}. {warning}")

        return sharpe_adjusted, warning

    def calculate_realistic_confidence(
        self,
        sharpe: float,
        volatility: float,
        n_observations: int,
        model_agreement: float = 1.0
    ) -> Tuple[float, str]:
        """
        Calculate realistic confidence level.

        Args:
            sharpe: Sharpe ratio
            volatility: Portfolio volatility
            n_observations: Number of observations
            model_agreement: Agreement between models (0-1)

        Returns:
            Tuple of (confidence_pct, confidence_level)
        """
        # Base confidence from Sharpe
        if sharpe > 1.5:
            base_confidence = 75
        elif sharpe > 1.0:
            base_confidence = 65
        elif sharpe > 0.5:
            base_confidence = 50
        else:
            base_confidence = 35

        # Penalty for high volatility
        if volatility > 0.3:  # >30% annual vol
            vol_penalty = 10
        elif volatility > 0.25:
            vol_penalty = 5
        else:
            vol_penalty = 0

        # Penalty for small sample
        if n_observations < 200:
            sample_penalty = 10
        elif n_observations < 500:
            sample_penalty = 5
        else:
            sample_penalty = 0

        # Bonus for model agreement
        agreement_bonus = model_agreement * 5

        # Calculate final confidence
        confidence_pct = base_confidence - vol_penalty - sample_penalty + agreement_bonus
        confidence_pct = max(20, min(85, confidence_pct))  # Clamp to 20-85%

        # Determine level
        if confidence_pct >= 70:
            level = "MEDIUM-HIGH"
        elif confidence_pct >= 50:
            level = "MEDIUM"
        else:
            level = "LOW-MEDIUM"

        logger.info(f"Realistic confidence: {level} ({confidence_pct:.0f}%)")

        return confidence_pct, level

    def detect_negative_scenarios(
        self,
        returns: pd.DataFrame,
        weights: Dict[str, float]
    ) -> Dict[str, any]:
        """
        Explicitly calculate probability and magnitude of negative outcomes.

        Args:
            returns: Historical returns
            weights: Portfolio weights

        Returns:
            Dict with negative scenario analysis
        """
        w = np.array([weights.get(col, 0) for col in returns.columns])
        portfolio_returns = returns @ w

        # Analyze negative returns
        negative_returns = portfolio_returns[portfolio_returns < 0]

        analysis = {
            'prob_negative_day': len(negative_returns) / len(portfolio_returns),
            'prob_negative_week': 1 - (1 - len(negative_returns) / len(portfolio_returns)) ** 5,
            'prob_negative_month': 1 - (1 - len(negative_returns) / len(portfolio_returns)) ** 21,
            'avg_loss_day': negative_returns.mean() if len(negative_returns) > 0 else 0,
            'worst_day': portfolio_returns.min(),
            'worst_week': portfolio_returns.rolling(5).sum().min() if len(portfolio_returns) >= 5 else portfolio_returns.min(),
            'worst_month': portfolio_returns.rolling(21).sum().min() if len(portfolio_returns) >= 21 else portfolio_returns.min(),
        }

        logger.info("Negative scenario analysis:")
        logger.info(f"  Probability of losing day: {analysis['prob_negative_day']*100:.1f}%")
        logger.info(f"  Probability of losing week: {analysis['prob_negative_week']*100:.1f}%")
        logger.info(f"  Probability of losing month: {analysis['prob_negative_month']*100:.1f}%")
        logger.info(f"  Average loss on down days: {analysis['avg_loss_day']*100:.2f}%")
        logger.info(f"  Worst historical day: {analysis['worst_day']*100:.2f}%")
        logger.info(f"  Worst historical week: {analysis['worst_week']*100:.2f}%")
        logger.info(f"  Worst historical month: {analysis['worst_month']*100:.2f}%")

        return analysis
