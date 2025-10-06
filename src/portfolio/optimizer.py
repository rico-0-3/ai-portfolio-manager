"""
Portfolio optimization using modern portfolio theory.
Implements Markowitz, Black-Litterman, Risk Parity, and CVaR optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

try:
    from pypfopt import (
        EfficientFrontier,
        BlackLittermanModel,
        risk_models,
        expected_returns,
        objective_functions,
        HRPOpt,
        CLA
    )
    from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
    PYPFOPT_AVAILABLE = True
except ImportError:
    PYPFOPT_AVAILABLE = False
    logger.warning("pypfopt not available. Install: pip install PyPortfolioOpt")

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    logger.warning("cvxpy not available. Install: pip install cvxpy")


class PortfolioOptimizer:
    """Portfolio optimization using various methods."""

    def __init__(
        self,
        risk_free_rate: float = 0.02,
        target_return: Optional[float] = None
    ):
        """
        Initialize portfolio optimizer.

        Args:
            risk_free_rate: Risk-free rate for Sharpe ratio
            target_return: Target return for optimization (if None, maximize Sharpe)
        """
        self.risk_free_rate = risk_free_rate
        self.target_return = target_return

        logger.info("PortfolioOptimizer initialized")

    def markowitz_optimization(
        self,
        returns: pd.DataFrame,
        method: str = 'max_sharpe',
        constraints: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Markowitz Mean-Variance Optimization.

        Args:
            returns: DataFrame of asset returns
            method: Optimization method ('max_sharpe', 'min_volatility', 'efficient_return')
            constraints: Additional constraints dict

        Returns:
            Dictionary of ticker to weight
        """
        if not PYPFOPT_AVAILABLE:
            logger.error("pypfopt required for Markowitz optimization")
            return {}

        logger.info(f"Running Markowitz optimization ({method})")

        # Calculate expected returns and covariance matrix
        mu = expected_returns.mean_historical_return(returns)
        S = risk_models.sample_cov(returns)

        # Create efficient frontier
        ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))

        # Apply optimization
        if method == 'max_sharpe':
            weights = ef.max_sharpe(risk_free_rate=self.risk_free_rate)
        elif method == 'min_volatility':
            weights = ef.min_volatility()
        elif method == 'efficient_return':
            if self.target_return is None:
                logger.warning("target_return not set, using max Sharpe instead")
                weights = ef.max_sharpe(risk_free_rate=self.risk_free_rate)
            else:
                weights = ef.efficient_return(self.target_return)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Clean weights
        weights = ef.clean_weights()

        # Get performance metrics
        perf = ef.portfolio_performance(risk_free_rate=self.risk_free_rate)
        logger.info(f"Expected return: {perf[0]:.4f}, Volatility: {perf[1]:.4f}, Sharpe: {perf[2]:.4f}")

        return weights

    def black_litterman_optimization(
        self,
        returns: pd.DataFrame,
        market_caps: Optional[Dict[str, float]] = None,
        views: Optional[Dict] = None,
        confidence: Optional[List[float]] = None
    ) -> Dict[str, float]:
        """
        Black-Litterman optimization with investor views.

        Args:
            returns: DataFrame of asset returns
            market_caps: Dictionary of market capitalizations
            views: Dictionary of absolute views (e.g., {'AAPL': 0.15})
            confidence: List of confidence levels for views (0-1)

        Returns:
            Dictionary of ticker to weight
        """
        if not PYPFOPT_AVAILABLE:
            logger.error("pypfopt required for Black-Litterman")
            return {}

        logger.info("Running Black-Litterman optimization")

        # Calculate covariance
        S = risk_models.sample_cov(returns)

        # Market-cap weights (if provided)
        if market_caps:
            total_cap = sum(market_caps.values())
            market_prior = {k: v/total_cap for k, v in market_caps.items()}
        else:
            # Equal weights as prior
            tickers = returns.columns
            market_prior = {ticker: 1.0/len(tickers) for ticker in tickers}

        # Create Black-Litterman model
        bl = BlackLittermanModel(S, pi=market_prior, risk_aversion=2.5)

        # Add views if provided
        if views:
            view_dict = views
            bl.bl_returns(view_dict, confidence)

        # Get posterior returns
        ret_bl = bl.bl_returns()

        # Optimize
        ef = EfficientFrontier(ret_bl, S)
        weights = ef.max_sharpe(risk_free_rate=self.risk_free_rate)
        weights = ef.clean_weights()

        perf = ef.portfolio_performance(risk_free_rate=self.risk_free_rate)
        logger.info(f"BL - Expected return: {perf[0]:.4f}, Volatility: {perf[1]:.4f}, Sharpe: {perf[2]:.4f}")

        return weights

    def risk_parity_optimization(self, returns: pd.DataFrame) -> Dict[str, float]:
        """
        Risk Parity optimization (equal risk contribution).

        Args:
            returns: DataFrame of asset returns

        Returns:
            Dictionary of ticker to weight
        """
        logger.info("Running Risk Parity optimization")

        cov_matrix = returns.cov().values
        n_assets = len(returns.columns)

        def risk_contribution(weights, cov_matrix):
            """Calculate risk contribution of each asset."""
            portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
            marginal_contrib = cov_matrix @ weights
            risk_contrib = weights * marginal_contrib / portfolio_vol
            return risk_contrib

        def risk_parity_objective(weights, cov_matrix):
            """Objective: minimize variance of risk contributions."""
            risk_contrib = risk_contribution(weights, cov_matrix)
            target = np.ones(len(weights)) / len(weights)
            return np.sum((risk_contrib - target * np.sum(risk_contrib)) ** 2)

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # weights sum to 1
        ]
        bounds = tuple((0, 1) for _ in range(n_assets))

        # Initial guess (equal weights)
        init_weights = np.ones(n_assets) / n_assets

        # Optimize
        result = minimize(
            risk_parity_objective,
            init_weights,
            args=(cov_matrix,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if result.success:
            weights = dict(zip(returns.columns, result.x))
            logger.info("Risk Parity optimization successful")
            return weights
        else:
            logger.error("Risk Parity optimization failed")
            return {}

    def cvar_optimization(
        self,
        returns: pd.DataFrame,
        alpha: float = 0.95,
        target_return: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Conditional Value-at-Risk (CVaR) optimization.

        Args:
            returns: DataFrame of asset returns
            alpha: Confidence level (e.g., 0.95 for 95% CVaR)
            target_return: Minimum target return

        Returns:
            Dictionary of ticker to weight
        """
        if not CVXPY_AVAILABLE:
            logger.error("cvxpy required for CVaR optimization")
            return self.markowitz_optimization(returns, method='min_volatility')

        logger.info(f"Running CVaR optimization (alpha={alpha})")

        n_assets = len(returns.columns)
        n_samples = len(returns)

        # Decision variables
        weights = cp.Variable(n_assets)
        var = cp.Variable()
        u = cp.Variable(n_samples)

        # Returns matrix
        returns_matrix = returns.values

        # Portfolio returns
        portfolio_returns = returns_matrix @ weights

        # CVaR objective
        cvar = var + (1 / ((1 - alpha) * n_samples)) * cp.sum(u)

        # Constraints
        constraints = [
            weights >= 0,                    # Long-only
            cp.sum(weights) == 1,            # Fully invested
            u >= 0,                          # Auxiliary variable
            u >= -portfolio_returns - var    # CVaR constraint
        ]

        # Target return constraint
        if target_return:
            expected_return = cp.sum(cp.multiply(returns.mean().values, weights))
            constraints.append(expected_return >= target_return)

        # Solve
        problem = cp.Problem(cp.Minimize(cvar), constraints)

        try:
            problem.solve(solver=cp.ECOS)

            if weights.value is not None:
                result_weights = dict(zip(returns.columns, weights.value))
                logger.info(f"CVaR optimization successful. CVaR: {cvar.value:.4f}")
                return result_weights
            else:
                logger.error("CVaR optimization failed")
                return {}
        except Exception as e:
            logger.error(f"CVaR optimization error: {e}")
            return {}

    def hierarchical_risk_parity(self, returns: pd.DataFrame) -> Dict[str, float]:
        """
        Hierarchical Risk Parity (HRP) optimization.

        Args:
            returns: DataFrame of asset returns

        Returns:
            Dictionary of ticker to weight
        """
        if not PYPFOPT_AVAILABLE:
            logger.error("pypfopt required for HRP")
            return {}

        logger.info("Running Hierarchical Risk Parity optimization")

        hrp = HRPOpt(returns)
        weights = hrp.optimize()

        logger.info("HRP optimization completed")
        return weights

    def allocate_to_positions(
        self,
        weights: Dict[str, float],
        total_portfolio_value: float,
        latest_prices: Dict[str, float],
        min_allocation: float = 0.02
    ) -> Dict[str, int]:
        """
        Convert portfolio weights to discrete share allocations.

        Args:
            weights: Dictionary of ticker to weight
            total_portfolio_value: Total portfolio value in dollars
            latest_prices: Dictionary of ticker to latest price
            min_allocation: Minimum allocation threshold

        Returns:
            Dictionary of ticker to number of shares
        """
        # Filter out small allocations
        weights = {k: v for k, v in weights.items() if v >= min_allocation}

        # Discrete allocation
        allocation = {}
        leftover = total_portfolio_value

        for ticker, weight in weights.items():
            if ticker in latest_prices:
                dollar_amount = weight * total_portfolio_value
                shares = int(dollar_amount / latest_prices[ticker])
                allocation[ticker] = shares
                leftover -= shares * latest_prices[ticker]

        logger.info(f"Allocated ${total_portfolio_value - leftover:.2f}, Leftover: ${leftover:.2f}")

        return allocation

    def get_portfolio_metrics(
        self,
        weights: Dict[str, float],
        returns: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate portfolio performance metrics.

        Args:
            weights: Portfolio weights
            returns: Historical returns

        Returns:
            Dictionary of performance metrics
        """
        # Convert weights to array
        weight_array = np.array([weights.get(col, 0) for col in returns.columns])

        # Portfolio returns
        portfolio_returns = returns @ weight_array

        # Calculate metrics
        annual_return = portfolio_returns.mean() * 252
        annual_vol = portfolio_returns.std() * np.sqrt(252)
        sharpe = (annual_return - self.risk_free_rate) / annual_vol

        # Drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Sortino ratio (downside deviation)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252)
        sortino = (annual_return - self.risk_free_rate) / downside_vol if downside_vol != 0 else 0

        return {
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown
        }
