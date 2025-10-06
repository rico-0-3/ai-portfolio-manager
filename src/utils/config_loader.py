"""
Configuration loader utility.
Handles loading and validation of configuration files.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Load and manage configuration settings."""

    def __init__(self, config_path: str = None):
        """
        Initialize the configuration loader.

        Args:
            config_path: Path to the config file. If None, uses default location.
        """
        if config_path is None:
            # Get project root (3 levels up from this file)
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "config.yaml"

        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Config file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing config file: {e}")
            raise

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.

        Args:
            key: Configuration key (e.g., 'data.yfinance.enabled')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_data_config(self) -> Dict[str, Any]:
        """Get data sources configuration."""
        return self.config.get('data', {})

    def get_portfolio_config(self) -> Dict[str, Any]:
        """Get portfolio settings."""
        return self.config.get('portfolio', {})

    def get_risk_config(self) -> Dict[str, Any]:
        """Get risk management settings."""
        return self.config.get('risk', {})

    def get_models_config(self) -> Dict[str, Any]:
        """Get ML models configuration."""
        return self.config.get('models', {})

    def get_optimization_config(self) -> Dict[str, Any]:
        """Get portfolio optimization settings."""
        return self.config.get('optimization', {})

    def get_backtesting_config(self) -> Dict[str, Any]:
        """Get backtesting settings."""
        return self.config.get('backtesting', {})

    def validate_api_keys(self) -> Dict[str, bool]:
        """
        Validate that required API keys are set.

        Returns:
            Dictionary of API key status
        """
        keys_status = {}

        # Check Alpha Vantage
        av_key = self.get('data.alpha_vantage.api_key')
        keys_status['alpha_vantage'] = (
            av_key is not None and av_key != 'YOUR_API_KEY_HERE'
        )

        # Check News API
        news_key = self.get('data.news_api.api_key')
        keys_status['news_api'] = (
            news_key is not None and news_key != 'YOUR_NEWS_API_KEY'
        )

        return keys_status

    def __repr__(self) -> str:
        """String representation."""
        return f"ConfigLoader(config_path='{self.config_path}')"
