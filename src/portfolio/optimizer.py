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
        constraints: Optional[Dict] = None,
        ml_predictions: Optional[Dict[str, float]] = None
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

        # Handle single asset case
        if len(returns.columns) == 1:
            logger.info("Single asset detected, returning 100% allocation")
            return {returns.columns[0]: 1.0}

        logger.info(f"Running Markowitz optimization ({method})")
        if ml_predictions is not None and len(ml_predictions) > 0:
            logger.info(f"  Using ML predictions (1m horizon) for {len(ml_predictions)} tickers")
        else:
            logger.info("  Using historical returns (no ML predictions)")

        try:
            # Additional validation: ensure no NaN in returns
            if returns.isnull().any().any():
                logger.warning("Returns contain NaN values before Markowitz, cleaning...")
                nan_cols = returns.columns[returns.isnull().any()].tolist()
                logger.warning(f"  Columns with NaN: {nan_cols}")
                for col in nan_cols:
                    logger.warning(f"  {col}: {returns[col].isnull().sum()}/{len(returns)} NaN values")
                returns = returns.dropna()

            if returns.empty or len(returns) < 20:
                raise ValueError(f"Insufficient clean data: {len(returns)} rows")

            # Calculate expected returns - ALWAYS prefer ML predictions over historical
            if ml_predictions is not None and len(ml_predictions) > 0:
                # Use ML predictions (simple float format for 1m)
                mu_dict = {}
                for ticker in returns.columns:
                    pred = ml_predictions.get(ticker, 0)
                    # CRITICAL: Check if prediction is NaN (should never happen if meta_model filters correctly)
                    if pd.isna(pred):
                        logger.error(f"ML prediction for {ticker} is NaN! This should be filtered in meta_model!")
                        raise ValueError(f"NaN prediction for {ticker} - check meta_model filtering")
                    # ML prediction is already 1-month return, use directly
                    mu_dict[ticker] = pred

                mu = pd.Series(mu_dict)
                logger.info("Using ML predictions (1m horizon) for expected returns")
                logger.debug(f"ML predictions: {ml_predictions}")
                logger.debug(f"Calculated mu: {mu.to_dict()}")

                # Check if all predictions are negative
                if (mu <= 0).all():
                    logger.warning("All ML predictions are negative or zero")
                    logger.warning("In bearish market conditions, using equal weights as conservative strategy")
                    tickers = returns.columns
                    return {ticker: 1.0/len(tickers) for ticker in tickers}
            else:
                # This should NEVER happen if meta_model.py filters correctly
                logger.error("CRITICAL: No ML predictions provided to Markowitz!")
                logger.error("Markowitz should ALWAYS receive ML predictions from meta_model")
                raise ValueError("ML predictions required for Markowitz optimization")

            # Check for NaN in mu and remove problematic tickers
            if mu.isnull().any():
                nan_tickers = mu[mu.isnull()].index.tolist()
                logger.warning(f"NaN in expected returns for: {nan_tickers}")
                logger.warning(f"Full mu series:\n{mu}")

                # Remove problematic tickers and retry
                logger.warning(f"Removing {nan_tickers} and retrying...")
                returns = returns.drop(columns=nan_tickers)

                # Handle single asset case after removal
                if len(returns.columns) == 1:
                    logger.info(f"Only one asset remains after removing NaN tickers: {returns.columns[0]}")
                    return {returns.columns[0]: 1.0}

                if len(returns.columns) == 0:
                    raise ValueError("All tickers have NaN expected returns")

                # Recalculate mu with remaining tickers
                if ml_predictions is not None and len(ml_predictions) > 0:
                    # Keep using ML predictions for remaining tickers
                    # CRITICAL: Filter out NaN predictions during retry
                    mu_dict_clean = {}
                    for ticker in returns.columns:
                        pred = ml_predictions.get(ticker, 0)
                        # Skip if prediction is NaN
                        if pd.isna(pred):
                            logger.warning(f"ML prediction for {ticker} is NaN in retry, using 0")
                            mu_dict_clean[ticker] = 0.0
                        else:
                            mu_dict_clean[ticker] = pred
                    mu = pd.Series(mu_dict_clean)
                    logger.info("Using ML predictions for remaining tickers (NaN filtered)")
                else:
                    mu = expected_returns.mean_historical_return(returns)

                # If still NaN, give up
                if mu.isnull().any():
                    raise ValueError("Expected returns still contain NaN after removing problematic tickers")

            # Use Ledoit-Wolf shrinkage for better conditioned covariance matrix
            # This fixes non-convex errors by ensuring positive semi-definite matrix
            S = risk_models.CovarianceShrinkage(returns).ledoit_wolf()

            # Check for NaN in covariance
            if np.isnan(S).any():
                logger.warning("NaN in covariance matrix")
                raise ValueError("Covariance matrix contains NaN")

            # Create efficient frontier with regularization
            ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))

            # Add L2 regularization to improve stability
            ef.add_objective(objective_functions.L2_reg, gamma=0.1)

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

        except Exception as e:
            logger.warning(f"Markowitz optimization failed: {e}")
            # Fallback to equal weights
            tickers = returns.columns
            equal_weights = {ticker: 1.0/len(tickers) for ticker in tickers}
            logger.warning("Using equal weights as fallback")
            return equal_weights

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

        # Handle single asset case
        if len(returns.columns) == 1:
            logger.info("Single asset detected, returning 100% allocation")
            return {returns.columns[0]: 1.0}

        logger.info("Running Black-Litterman optimization")

        try:
            # Additional validation: ensure no NaN in returns
            if returns.isnull().any().any():
                logger.warning("Returns contain NaN values, cleaning...")
                returns = returns.dropna()

            if returns.empty or len(returns) < 20:
                raise ValueError(f"Insufficient clean data: {len(returns)} rows")

            # Calculate covariance with Ledoit-Wolf shrinkage
            S = risk_models.CovarianceShrinkage(returns).ledoit_wolf()

            # Check for NaN in covariance
            if np.isnan(S).any():
                logger.warning("NaN in covariance matrix")
                raise ValueError("Covariance matrix contains NaN")

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
                # Convert views dict to proper format
                # views: {'AAPL': 0.15, 'MSFT': 0.10, ...}
                # Need to create P (picking matrix) and Q (view returns)
                view_tickers = list(views.keys())
                n_views = len(view_tickers)
                n_assets = len(returns.columns)
                ticker_to_idx = {ticker: i for i, ticker in enumerate(returns.columns)}

                # Create P matrix (picking matrix) - which assets each view refers to
                P = np.zeros((n_views, n_assets))
                Q = []  # View returns

                for i, (ticker, view_return) in enumerate(views.items()):
                    if ticker in ticker_to_idx:
                        P[i, ticker_to_idx[ticker]] = 1.0
                        Q.append(view_return)

                Q = np.array(Q)

                # Confidence in views (if not provided, use moderate confidence)
                if confidence is None:
                    confidence = [0.5] * n_views  # Moderate confidence

                # Omega (uncertainty in views) - diagonal matrix
                # Lower confidence = higher uncertainty
                omega = np.diag([1.0 / c if c > 0 else 1.0 for c in confidence])

                # Add views using proper format
                bl.bl_returns(P=P, Q=Q, omega=omega)

            # Get posterior returns
            ret_bl = bl.bl_returns()

            # Optimize with regularization
            ef = EfficientFrontier(ret_bl, S)
            ef.add_objective(objective_functions.L2_reg, gamma=0.1)
            weights = ef.max_sharpe(risk_free_rate=self.risk_free_rate)
            weights = ef.clean_weights()

            perf = ef.portfolio_performance(risk_free_rate=self.risk_free_rate)
            logger.info(f"BL - Expected return: {perf[0]:.4f}, Volatility: {perf[1]:.4f}, Sharpe: {perf[2]:.4f}")

            return weights

        except Exception as e:
            logger.debug(f"Black-Litterman optimization failed: {e}")
            # Fallback to equal weights (don't call Markowitz without ml_predictions!)
            logger.debug("Black-Litterman failed, using equal weights")
            tickers = returns.columns
            return {ticker: 1.0/len(tickers) for ticker in tickers}

    def risk_parity_optimization(
        self,
        returns: pd.DataFrame,
        ml_predictions: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Risk Parity optimization with optional ML-aware targets.

        Args:
            returns: DataFrame of asset returns
            ml_predictions: Optional ML predictions for ML-aware risk targets

        Returns:
            Dictionary of ticker to weight
        """
        logger.info("Running Risk Parity optimization" +
                   (" (ML-aware)" if ml_predictions else ""))

        cov_matrix = returns.cov().values
        n_assets = len(returns.columns)

        # ML-aware risk targets: adjust based on predictions
        if ml_predictions:
            risk_targets = []
            for ticker in returns.columns:
                ml_pred = ml_predictions.get(ticker, 0)
                # If ML predicts negative, reduce target risk contribution
                if ml_pred < -0.01:  # Strong negative
                    target = 0.7  # 30% less risk
                elif ml_pred < 0:  # Slight negative
                    target = 0.85  # 15% less risk
                elif ml_pred > 0.02:  # Strong positive
                    target = 1.2  # 20% more risk (opportunity)
                else:  # Neutral
                    target = 1.0
                risk_targets.append(target)

            # Normalize targets to sum to n_assets
            risk_targets = np.array(risk_targets)
            risk_targets = risk_targets * n_assets / risk_targets.sum()
            logger.info(f"ML-aware risk targets: {dict(zip(returns.columns, risk_targets))}")
        else:
            # Equal risk contribution (standard RP)
            risk_targets = np.ones(n_assets)

        def risk_contribution(weights, cov_matrix):
            """Calculate risk contribution of each asset."""
            portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
            marginal_contrib = cov_matrix @ weights
            risk_contrib = weights * marginal_contrib / portfolio_vol
            return risk_contrib

        def risk_parity_objective(weights, cov_matrix, targets):
            """Objective: minimize variance from target risk contributions."""
            risk_contrib = risk_contribution(weights, cov_matrix)
            # Target risk based on ML-aware targets
            target_contrib = targets / targets.sum() * np.sum(risk_contrib)
            return np.sum((risk_contrib - target_contrib) ** 2)

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
            args=(cov_matrix, risk_targets),
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
        target_return: Optional[float] = None,
        ml_predictions: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Conditional Value-at-Risk (CVaR) optimization.

        Args:
            returns: DataFrame of asset returns
            alpha: Confidence level (e.g., 0.95 for 95% CVaR)
            target_return: Minimum target return
            ml_predictions: Optional ML predictions for expected returns

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
            # Use ML predictions if available, otherwise historical mean
            if ml_predictions is not None and len(ml_predictions) > 0:
                mu_list = []
                for ticker in returns.columns:
                    pred = ml_predictions.get(ticker, 0)
                    # ML prediction is already 1-month return
                    mu_list.append(pred)
                mu = np.array(mu_list)
                logger.info("CVaR: Using ML predictions (1m horizon) for expected returns")
            else:
                mu = returns.mean().values
                logger.info("CVaR: Using historical mean for expected returns")

            expected_return = cp.sum(cp.multiply(mu, weights))
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
        returns: pd.DataFrame,
        ml_predictions: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Calculate portfolio performance metrics.

        Args:
            weights: Portfolio weights
            returns: Historical returns (for volatility/drawdown calculation)
            ml_predictions: Optional ML predictions for expected return

        Returns:
            Dictionary of performance metrics
        """
        # Convert weights to array
        weight_array = np.array([weights.get(col, 0) for col in returns.columns])

        # Portfolio returns (for risk metrics only)
        portfolio_returns = returns @ weight_array

        # Calculate expected return - use ML predictions if available
        if ml_predictions:
            # Use ML predictions for expected return (1m horizon only)
            monthly_return = 0.0

            for ticker in returns.columns:
                weight = weights.get(ticker, 0)
                pred = ml_predictions.get(ticker, 0)
                monthly_return += weight * pred

            # Store as single value
            annual_return = monthly_return  # We report monthly return directly

            logger.info(f"Using ML predictions (1m horizon): {monthly_return*100:+.2f}%")
        else:
            # Fallback to historical mean
            annual_return = portfolio_returns.mean() * 252
            logger.info(f"Using historical mean for expected return: {annual_return*100:.2f}%")
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

        metrics = {
            'monthly_return': annual_return,  # This is actually monthly return from 1m model
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown
        }

        return metrics

    def rl_agent_optimization(
        self,
        returns: pd.DataFrame,
        ml_predictions: Optional[Dict[str, float]] = None,
        training_steps: int = 10000
    ) -> Dict[str, float]:
        """
        Reinforcement Learning Agent (PPO) for portfolio optimization.

        Uses Stable-Baselines3 PPO to learn optimal portfolio allocation.
        State: returns, volatility, correlation, ML predictions
        Action: portfolio weights (continuous)
        Reward: Sharpe ratio

        Args:
            returns: DataFrame of asset returns
            ml_predictions: Optional ML predictions for state augmentation
            training_steps: Number of training steps (default 10000)

        Returns:
            Dictionary of ticker to weight
        """
        logger.info(f"Running RL Agent optimization (PPO, {training_steps} steps)")

        try:
            # Check if stable-baselines3 is available
            try:
                from stable_baselines3 import PPO
                from stable_baselines3.common.vec_env import DummyVecEnv
                import gymnasium as gym
                from gymnasium import spaces
            except ImportError:
                logger.warning("stable-baselines3 not available. Install: pip install stable-baselines3 gymnasium")
                logger.warning("Falling back to Risk Parity")
                return self.risk_parity_optimization(returns, ml_predictions)

            # Handle single asset case
            if len(returns.columns) == 1:
                logger.info("Single asset detected, returning 100% allocation")
                return {returns.columns[0]: 1.0}

            # ========== CUSTOM PORTFOLIO ENVIRONMENT ==========
            class PortfolioEnv(gym.Env):
                """
                Custom Gymnasium environment for portfolio optimization.

                State: [returns_mean, volatility, correlation_matrix_flattened, ML_predictions]
                Action: portfolio weights (continuous, sum to 1)
                Reward: Sharpe ratio of the portfolio
                """

                def __init__(self, returns_df, ml_preds=None):
                    super().__init__()

                    self.returns_df = returns_df
                    self.ml_preds = ml_preds or {}
                    self.tickers = returns_df.columns.tolist()
                    self.n_assets = len(self.tickers)

                    # Calculate statistics
                    self.returns_mean = returns_df.mean().values
                    self.returns_std = returns_df.std().values
                    self.corr_matrix = returns_df.corr().values

                    # State space:
                    # - n_assets: mean returns
                    # - n_assets: volatilities
                    # - n_assets*(n_assets-1)/2: unique correlation pairs
                    # - n_assets: ML predictions (if available)
                    n_corr_features = int(self.n_assets * (self.n_assets - 1) / 2)
                    state_dim = self.n_assets * 2 + n_corr_features
                    if ml_preds is not None and len(ml_preds) > 0:
                        state_dim += self.n_assets

                    self.observation_space = spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(state_dim,),
                        dtype=np.float32
                    )

                    # Action space: portfolio weights (continuous, 0 to 1)
                    self.action_space = spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=(self.n_assets,),
                        dtype=np.float32
                    )

                    self.current_step = 0
                    self.max_steps = 1000

                def reset(self, seed=None, options=None):
                    super().reset(seed=seed)
                    self.current_step = 0
                    return self._get_state(), {}

                def _get_state(self):
                    """Get current state representation."""
                    state = []

                    # Mean returns
                    state.extend(self.returns_mean)

                    # Volatilities
                    state.extend(self.returns_std)

                    # Correlation matrix (upper triangle, no diagonal)
                    for i in range(self.n_assets):
                        for j in range(i+1, self.n_assets):
                            state.append(self.corr_matrix[i, j])

                    # ML predictions (if available)
                    if self.ml_preds is not None and len(self.ml_preds) > 0:
                        for ticker in self.tickers:
                            state.append(self.ml_preds.get(ticker, 0.0))

                    return np.array(state, dtype=np.float32)

                def step(self, action):
                    """
                    Execute action (portfolio weights) and return reward.

                    Reward = Sharpe ratio of the portfolio
                    """
                    self.current_step += 1

                    # Normalize weights to sum to 1
                    weights = action / (action.sum() + 1e-8)

                    # Calculate portfolio metrics
                    portfolio_return = np.dot(weights, self.returns_mean) * 252  # Annualized
                    portfolio_variance = weights @ self.returns_df.cov().values @ weights
                    portfolio_vol = np.sqrt(portfolio_variance * 252)  # Annualized

                    # Reward: Sharpe ratio
                    risk_free_rate = 0.02
                    sharpe = (portfolio_return - risk_free_rate) / (portfolio_vol + 1e-8)

                    # Add penalty for extreme concentration (encourage diversification)
                    concentration_penalty = -np.sum(weights ** 2)  # Negative Herfindahl index

                    # Combined reward
                    reward = sharpe + 0.1 * concentration_penalty

                    # Episode termination
                    done = self.current_step >= self.max_steps
                    truncated = False

                    return self._get_state(), reward, done, truncated, {}

                def render(self):
                    pass

            # ========== TRAIN RL AGENT ==========

            # Create environment
            env = DummyVecEnv([lambda: PortfolioEnv(returns, ml_predictions)])

            # Create PPO agent
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,  # Encourage exploration
                verbose=0,
                seed=42
            )

            # Train
            logger.info(f"  Training PPO agent for {training_steps} steps...")
            model.learn(total_timesteps=training_steps, progress_bar=False)

            # Get optimal weights
            obs = env.reset()
            action, _ = model.predict(obs, deterministic=True)

            # Normalize weights
            weights_array = action[0] / (action[0].sum() + 1e-8)

            # Convert to dictionary
            weights = dict(zip(returns.columns, weights_array))

            # Calculate resulting Sharpe ratio
            portfolio_return = np.dot(weights_array, returns.mean().values) * 252
            portfolio_vol = np.sqrt(weights_array @ returns.cov().values @ weights_array * 252)
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol

            logger.info(f"  RL Agent Sharpe: {sharpe:.4f}")
            logger.info(f"  Weights: {weights}")

            return weights

        except Exception as e:
            logger.warning(f"RL Agent optimization failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            # Fallback to Risk Parity
            logger.warning("Falling back to Risk Parity")
            return self.risk_parity_optimization(returns, ml_predictions)
