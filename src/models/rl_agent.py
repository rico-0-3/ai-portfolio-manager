"""
Reinforcement Learning agent for portfolio management.
Implements PPO and SAC algorithms for dynamic allocation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
import gymnasium as gym
from gymnasium import spaces

logger = logging.getLogger(__name__)

try:
    from stable_baselines3 import PPO, SAC, DDPG
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    logger.warning("stable-baselines3 not available. Install: pip install stable-baselines3")


class PortfolioEnv(gym.Env):
    """Custom gym environment for portfolio management."""

    def __init__(
        self,
        price_data: Dict[str, pd.DataFrame],
        features: Dict[str, pd.DataFrame],
        initial_balance: Optional[float] = None,
        transaction_cost: Optional[float] = None,
        max_steps: Optional[int] = None,
        config: Optional[Dict] = None,
        ml_predictions: Optional[Dict[str, float]] = None
    ):
        """
        Initialize portfolio environment.

        Args:
            price_data: Dict of ticker to price DataFrame
            features: Dict of ticker to features DataFrame
            initial_balance: Starting balance (if None, uses config or default)
            transaction_cost: Transaction cost as fraction (if None, uses config or default)
            max_steps: Maximum steps per episode
            config: Config dict (optional)
            ml_predictions: Optional ML predictions for ML-aware reward
        """
        super(PortfolioEnv, self).__init__()

        self.price_data = price_data
        self.features = features
        self.tickers = list(price_data.keys())
        self.n_assets = len(self.tickers)
        self.ml_predictions = ml_predictions or {}

        # Load from config if provided
        if config:
            self.initial_balance = initial_balance or config.get('portfolio', {}).get('initial_budget', 10000)
            self.transaction_cost = transaction_cost or config.get('portfolio', {}).get('transaction_cost', 0.001)
        else:
            self.initial_balance = initial_balance or 10000
            self.transaction_cost = transaction_cost or 0.001

        # Get common dates
        all_dates = set.intersection(*[set(df.index) for df in price_data.values()])
        self.dates = sorted(all_dates)
        self.max_steps = max_steps or len(self.dates)

        # Action space: portfolio weights for each asset (continuous)
        self.action_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.n_assets,),
            dtype=np.float32
        )

        # Observation space: features for all assets
        n_features = len(features[self.tickers[0]].columns)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_assets * n_features,),
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        """Reset environment (Gymnasium API)."""
        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0
        self.balance = self.initial_balance
        self.portfolio_value = self.initial_balance
        self.positions = np.zeros(self.n_assets)

        obs = self._get_observation()
        info = {}

        return obs, info

    def _get_observation(self):
        """Get current observation."""
        if self.current_step >= len(self.dates):
            self.current_step = len(self.dates) - 1

        date = self.dates[self.current_step]
        obs = []

        for ticker in self.tickers:
            if date in self.features[ticker].index:
                features = self.features[ticker].loc[date].values
                obs.extend(features)
            else:
                obs.extend(np.zeros(len(self.features[ticker].columns)))

        return np.array(obs, dtype=np.float32)

    def step(self, action):
        """Execute one step."""
        # Normalize action to sum to 1
        action = np.clip(action, 0, 1)
        action = action / (action.sum() + 1e-8)

        date = self.dates[self.current_step]

        # Get current prices
        prices = np.array([
            self.price_data[ticker].loc[date, 'Close']
            if date in self.price_data[ticker].index else 0
            for ticker in self.tickers
        ])

        # Calculate current portfolio value
        self.portfolio_value = self.balance + np.sum(self.positions * prices)

        # Calculate target positions
        target_positions = (action * self.portfolio_value) / prices

        # Execute trades
        trades = target_positions - self.positions
        transaction_costs = np.sum(np.abs(trades) * prices) * self.transaction_cost

        self.positions = target_positions
        self.balance -= transaction_costs

        # Move to next step
        self.current_step += 1

        # Calculate reward (portfolio return)
        if self.current_step < len(self.dates):
            next_date = self.dates[self.current_step]
            next_prices = np.array([
                self.price_data[ticker].loc[next_date, 'Close']
                if next_date in self.price_data[ticker].index else prices[i]
                for i, ticker in enumerate(self.tickers)
            ])

            next_value = self.balance + np.sum(self.positions * next_prices)
            reward = (next_value - self.portfolio_value) / self.portfolio_value

            # ML-aware reward: bonus for actions aligned with ML predictions
            if self.ml_predictions:
                ml_alignment = 0.0
                for i, ticker in enumerate(self.tickers):
                    ml_pred = self.ml_predictions.get(ticker, 0)
                    weight = action[i]

                    # Reward allocating to assets with positive ML predictions
                    if ml_pred > 0.01:  # Strong positive prediction
                        ml_alignment += weight * ml_pred * 0.5  # 50% bonus
                    elif ml_pred < -0.01:  # Strong negative prediction
                        ml_alignment -= weight * abs(ml_pred) * 0.3  # 30% penalty

                # Add ML alignment bonus to reward
                reward += ml_alignment
        else:
            reward = 0

        # Check if done (Gymnasium uses terminated and truncated)
        terminated = self.current_step >= len(self.dates) - 1
        truncated = self.current_step >= self.max_steps

        # Get next observation
        obs = self._get_observation()

        info = {
            'portfolio_value': self.portfolio_value,
            'positions': self.positions.copy(),
            'date': date
        }

        return obs, reward, terminated, truncated, info


class RLPortfolioAgent:
    """Reinforcement Learning agent for portfolio management."""

    def __init__(
        self,
        algorithm: Optional[str] = None,
        learning_rate: Optional[float] = None,
        gamma: Optional[float] = None,
        config: Optional[Dict] = None,
        **kwargs
    ):
        """
        Initialize RL agent.

        Args:
            algorithm: Algorithm to use ('PPO', 'SAC', 'DDPG') - from config if None
            learning_rate: Learning rate - from config if None
            gamma: Discount factor - from config if None
            config: Config dict (optional)
            **kwargs: Additional algorithm parameters
        """
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 required")

        # Load from config if provided
        if config:
            rl_config = config.get('optimization', {}).get('reinforcement_learning', {})
            self.algorithm = algorithm or rl_config.get('algorithm', 'PPO')
            self.learning_rate = learning_rate or rl_config.get('learning_rate', 0.0003)
            self.gamma = gamma or rl_config.get('gamma', 0.99)
        else:
            self.algorithm = algorithm or 'PPO'
            self.learning_rate = learning_rate or 0.0003
            self.gamma = gamma or 0.99

        self.kwargs = kwargs
        self.model = None
        self.env = None
        self.config = config

        logger.info(f"RLPortfolioAgent initialized with {self.algorithm}")

    def create_environment(
        self,
        price_data: Dict[str, pd.DataFrame],
        features: Dict[str, pd.DataFrame],
        initial_balance: Optional[float] = None,
        transaction_cost: Optional[float] = None,
        ml_predictions: Optional[Dict[str, float]] = None
    ):
        """Create training environment with optional ML predictions."""
        self.env = PortfolioEnv(
            price_data=price_data,
            features=features,
            initial_balance=initial_balance,
            transaction_cost=transaction_cost,
            config=self.config,
            ml_predictions=ml_predictions
        )

        # Wrap in vectorized environment
        self.env = DummyVecEnv([lambda: self.env])

        logger.info("Environment created")

    def train(
        self,
        total_timesteps: int = 100000,
        callback: Optional[BaseCallback] = None
    ):
        """
        Train the agent.

        Args:
            total_timesteps: Total training timesteps
            callback: Optional callback
        """
        if self.env is None:
            raise ValueError("Environment not created. Call create_environment first.")

        # Initialize model
        if self.algorithm == 'PPO':
            self.model = PPO(
                'MlpPolicy',
                self.env,
                learning_rate=self.learning_rate,
                gamma=self.gamma,
                verbose=1,
                **self.kwargs
            )
        elif self.algorithm == 'SAC':
            self.model = SAC(
                'MlpPolicy',
                self.env,
                learning_rate=self.learning_rate,
                gamma=self.gamma,
                verbose=1,
                **self.kwargs
            )
        elif self.algorithm == 'DDPG':
            self.model = DDPG(
                'MlpPolicy',
                self.env,
                learning_rate=self.learning_rate,
                gamma=self.gamma,
                verbose=1,
                **self.kwargs
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        logger.info(f"Training {self.algorithm} for {total_timesteps} timesteps...")

        # Train
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback
        )

        logger.info("Training completed")

    def predict(self, observation):
        """
        Predict action for observation.

        Args:
            observation: Current observation

        Returns:
            Predicted action (portfolio weights)
        """
        if self.model is None:
            raise ValueError("Model not trained")

        action, _states = self.model.predict(observation, deterministic=True)
        return action

    def get_portfolio_allocation(
        self,
        current_features: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Get portfolio allocation from current features.

        Args:
            current_features: Dict of ticker to feature array

        Returns:
            Dict of ticker to weight
        """
        # Create observation
        obs = []
        tickers = sorted(current_features.keys())

        # First pass: determine max feature size
        feature_sizes = []
        for ticker in tickers:
            features = current_features[ticker]
            if isinstance(features, (int, float, np.number)):
                feature_sizes.append(1)
            else:
                features_array = np.asarray(features).flatten()
                feature_sizes.append(len(features_array))

        max_features = max(feature_sizes) if feature_sizes else 0

        # Second pass: pad features to same size
        for ticker in tickers:
            features = current_features[ticker]
            # Handle single value or array
            if isinstance(features, (int, float, np.number)):
                features_array = np.array([float(features)])
            else:
                # Convert to array and flatten
                features_array = np.asarray(features).flatten()

            # Pad with zeros if needed to reach max_features
            if len(features_array) < max_features:
                padding = np.zeros(max_features - len(features_array))
                features_array = np.concatenate([features_array, padding])

            obs.extend(features_array)

        obs = np.array(obs, dtype=np.float32)

        # Predict
        action = self.predict(obs)

        # Ensure action is array (PPO returns tuple, extract action)
        if isinstance(action, (list, tuple)):
            action = action[0]
        action = np.asarray(action).flatten()

        # Ensure action has correct length
        if len(action) != len(tickers):
            logger.warning(f"Action size mismatch: got {len(action)}, expected {len(tickers)}. Using equal weights.")
            action = np.ones(len(tickers))

        # Normalize
        action = np.clip(action, 0, 1)
        action = action / (action.sum() + 1e-8)

        # Create allocation dict
        allocation = dict(zip(tickers, action))

        return allocation

    def save(self, path: str):
        """Save model."""
        if self.model is None:
            raise ValueError("Model not trained")

        self.model.save(path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load model."""
        if self.algorithm == 'PPO':
            self.model = PPO.load(path)
        elif self.algorithm == 'SAC':
            self.model = SAC.load(path)
        elif self.algorithm == 'DDPG':
            self.model = DDPG.load(path)

        logger.info(f"Model loaded from {path}")
