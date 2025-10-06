"""
Main orchestrator for AI Portfolio Manager.
Coordinates all components: data, features, models, optimization.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import logging
from datetime import datetime

from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger
from src.data.market_data import MarketDataFetcher
from src.data.sentiment_data import SentimentDataFetcher
from src.data.sentiment_analyzer import HybridSentimentAnalyzer
from src.features.technical_indicators import TechnicalIndicators
from src.features.feature_engineering import FeatureEngineer
from src.models.lstm_model import LSTMTrainer
from src.models.ensemble_models import EnsemblePredictor, XGBoostPredictor, LightGBMPredictor
from src.models.rl_agent import RLPortfolioAgent, PortfolioEnv
from src.portfolio.optimizer import PortfolioOptimizer
from src.portfolio.risk_manager import RiskManager


class PortfolioOrchestrator:
    """Main orchestrator for the portfolio management system."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize orchestrator.

        Args:
            config_path: Path to config file
        """
        # Load config
        self.config = ConfigLoader(config_path)

        # Setup logging
        log_config = self.config.get('logging', {})
        self.logger = setup_logger(
            name="orchestrator",
            level=log_config.get('level', 'INFO'),
            log_file=log_config.get('log_file', 'logs/orchestrator.log')
        )

        # Initialize components
        self.market_fetcher = MarketDataFetcher(
            cache_dir=self.config.get('paths.data_raw', 'data/raw')
        )
        self.sentiment_fetcher = SentimentDataFetcher(
            news_api_key=self.config.get('data.news_api.api_key')
        )
        self.tech_indicators = TechnicalIndicators()
        self.feature_engineer = FeatureEngineer()
        self.optimizer = PortfolioOptimizer(
            risk_free_rate=self.config.get('optimization.mean_variance.risk_free_rate', 0.02)
        )
        self.risk_manager = RiskManager(
            max_position_size=self.config.get('portfolio.max_single_position', 0.20),
            max_drawdown=self.config.get('risk.max_drawdown', 0.25)
        )

        self.logger.info("PortfolioOrchestrator initialized")

    def run_full_pipeline(
        self,
        tickers: List[str],
        period: str = '2y',
        use_ml_predictions: bool = True
    ) -> Dict:
        """
        Run complete pipeline from data acquisition to portfolio allocation.
        Uses ALL optimization methods combined based on risk profile from config.

        Args:
            tickers: List of stock tickers
            period: Data period
            use_ml_predictions: Whether to use ML predictions

        Returns:
            Dict with results including weights, metrics, and allocations
        """
        self.logger.info("="*80)
        self.logger.info("STARTING FULL PIPELINE")
        self.logger.info(f"Tickers: {tickers}")
        self.logger.info(f"Period: {period}")
        self.logger.info("="*80)

        # Step 1: Fetch market data
        self.logger.info("\n[1/6] Fetching market data...")
        market_data = self.market_fetcher.fetch_stock_data(
            tickers,
            period=period,
            interval='1d'
        )
        self.logger.info(f"Fetched data for {len(market_data)} tickers")

        # Step 2: Fetch sentiment data
        self.logger.info("\n[2/6] Analyzing sentiment...")
        sentiment_scores = self._analyze_sentiment(tickers)

        # Step 3: Feature engineering
        self.logger.info("\n[3/6] Engineering features...")
        processed_data = self._engineer_features(market_data)

        # Step 4: ML predictions (optional)
        predictions = {}
        if use_ml_predictions:
            self.logger.info("\n[4/6] Generating ML predictions...")
            predictions = self._generate_predictions(processed_data)
        else:
            self.logger.info("\n[4/6] Skipping ML predictions")

        # Step 5: Portfolio optimization (uses ALL methods)
        self.logger.info("\n[5/6] Optimizing portfolio...")
        weights = self._optimize_portfolio(
            processed_data,
            sentiment_scores,
            predictions
        )

        # Step 6: Risk management and allocation
        self.logger.info("\n[6/6] Applying risk management...")
        final_weights = self.risk_manager.check_position_limits(weights)

        # Get latest prices for allocation
        latest_prices = {
            ticker: self.market_fetcher.get_latest_price(ticker)
            for ticker in tickers
        }
        latest_prices = {k: v for k, v in latest_prices.items() if v is not None}

        # Calculate discrete allocation
        budget = self.config.get('portfolio.initial_budget', 10000)
        allocation = self.optimizer.allocate_to_positions(
            final_weights,
            budget,
            latest_prices
        )

        # Calculate returns for metrics
        returns_df = pd.DataFrame({
            ticker: data['Close'].pct_change()
            for ticker, data in processed_data.items()
        }).dropna()

        # Calculate portfolio metrics
        metrics = self.optimizer.get_portfolio_metrics(final_weights, returns_df)

        # Calculate risk metrics
        portfolio_var = self.risk_manager.calculate_portfolio_var(
            returns_df,
            final_weights,
            confidence=self.config.get('risk.var_confidence', 0.95)
        )
        portfolio_cvar = self.risk_manager.calculate_portfolio_cvar(
            returns_df,
            final_weights,
            confidence=self.config.get('risk.var_confidence', 0.95)
        )

        results = {
            'weights': final_weights,
            'allocation': allocation,
            'metrics': metrics,
            'risk_metrics': {
                'var_95': portfolio_var,
                'cvar_95': portfolio_cvar
            },
            'sentiment_scores': sentiment_scores,
            'predictions': predictions,
            'latest_prices': latest_prices,
            'budget': budget
        }

        self.logger.info("\n" + "="*80)
        self.logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        self.logger.info("="*80)

        return results

    def _analyze_sentiment(self, tickers: List[str]) -> Dict[str, float]:
        """Analyze sentiment for tickers."""
        sentiment_scores = {}

        try:
            # Fetch news
            news_df = self.sentiment_fetcher.fetch_financial_news(
                tickers,
                days_back=7
            )

            if not news_df.empty:
                # Analyze sentiment
                news_with_sentiment = self.sentiment_fetcher.analyze_news_sentiment(news_df)
                summary = self.sentiment_fetcher.get_sentiment_summary(news_with_sentiment)

                for _, row in summary.iterrows():
                    sentiment_scores[row['ticker']] = row['avg_sentiment']

                self.logger.info(f"Sentiment analyzed for {len(sentiment_scores)} tickers")
            else:
                self.logger.warning("No news data available")

        except Exception as e:
            self.logger.error(f"Sentiment analysis error: {e}")

        return sentiment_scores

    def _engineer_features(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Engineer features for all tickers."""
        processed_data = {}

        for ticker, df in market_data.items():
            try:
                # Add technical indicators
                df = self.tech_indicators.add_all_indicators(df)

                # Add custom features
                df = self.feature_engineer.create_price_features(df)
                df = self.feature_engineer.create_volume_features(df)
                df = self.feature_engineer.create_volatility_features(df)

                # Drop NaN
                df = df.dropna()

                processed_data[ticker] = df
                self.logger.info(f"{ticker}: {len(df)} rows, {len(df.columns)} features")

            except Exception as e:
                self.logger.error(f"Feature engineering error for {ticker}: {e}")

        return processed_data

    def _generate_predictions(self, processed_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Generate ML predictions using LSTM and Ensemble models."""
        predictions = {}

        try:
            # Initialize ensemble predictor (XGBoost + LightGBM)
            ensemble = EnsemblePredictor()

            for ticker, df in processed_data.items():
                try:
                    # Prepare features (use last 60 days for LSTM lookback)
                    if len(df) < 100:
                        self.logger.warning(f"{ticker}: Insufficient data ({len(df)} rows), skipping")
                        predictions[ticker] = 0.0
                        continue

                    # Use technical indicators as features
                    feature_cols = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]

                    if not feature_cols:
                        self.logger.warning(f"{ticker}: No features available")
                        predictions[ticker] = 0.0
                        continue

                    X = df[feature_cols].values
                    y = df['Close'].pct_change().shift(-1).fillna(0).values  # Next day return

                    # Split: train on first 80%, predict on last point
                    split_idx = int(len(X) * 0.8)
                    X_train, y_train = X[:split_idx], y[:split_idx]
                    X_test = X[-1].reshape(1, -1)  # Last observation

                    # Quick train ensemble (XGBoost + LightGBM)
                    ensemble.train(X_train, y_train)
                    pred = ensemble.predict(X_test)[0]

                    predictions[ticker] = float(pred)
                    self.logger.info(f"{ticker}: Predicted return = {pred*100:+.2f}%")

                except Exception as e:
                    self.logger.error(f"Prediction error for {ticker}: {e}")
                    predictions[ticker] = 0.0

            self.logger.info(f"Generated predictions for {len(predictions)} tickers using ML ensemble")

        except Exception as e:
            self.logger.error(f"ML prediction failed: {e}")
            # Fallback to zero predictions
            predictions = {ticker: 0.0 for ticker in processed_data.keys()}

        return predictions

    def _optimize_portfolio(
        self,
        processed_data: Dict[str, pd.DataFrame],
        sentiment_scores: Dict[str, float],
        predictions: Dict[str, float],
        method: str = None
    ) -> Dict[str, float]:
        """
        Optimize portfolio allocation using ALL methods combined.
        Weights are determined by risk profile from config.
        """
        # Calculate returns
        returns_df = pd.DataFrame({
            ticker: data['Close'].pct_change()
            for ticker, data in processed_data.items()
        }).dropna()

        # Get risk profile and weights from config
        risk_profile = self.config.get('optimization.risk_profile', 'medium')
        profile_weights = self.config.get(f'optimization.risk_profiles.{risk_profile}', {})

        self.logger.info(f"\nUsing risk profile: {risk_profile.upper()}")
        self.logger.info(f"Combining all optimization methods with weights:")
        for method_name, weight in profile_weights.items():
            self.logger.info(f"  {method_name}: {weight*100:.1f}%")

        # Run all optimization methods
        all_weights = {}

        # 1. Markowitz Mean-Variance
        try:
            self.logger.info("\n  Running Markowitz optimization...")
            markowitz_weights = self.optimizer.markowitz_optimization(
                returns_df,
                method='max_sharpe'
            )
            all_weights['mean_variance'] = markowitz_weights
        except Exception as e:
            self.logger.error(f"  Markowitz failed: {e}")
            all_weights['mean_variance'] = {}

        # 2. Black-Litterman with sentiment + ML predictions
        try:
            self.logger.info("  Running Black-Litterman optimization...")
            # Combine sentiment and ML predictions for views
            views = {}
            for ticker in returns_df.columns:
                view_value = 0.0
                # Add sentiment component (weight: 0.5)
                if ticker in sentiment_scores:
                    view_value += sentiment_scores[ticker] * 0.05  # Scale sentiment
                # Add ML prediction component (weight: 0.5)
                if ticker in predictions:
                    view_value += predictions[ticker] * 0.5  # ML predicted return

                if abs(view_value) > 0.001:  # Only include meaningful views
                    views[ticker] = view_value

            bl_weights = self.optimizer.black_litterman_optimization(
                returns_df,
                views=views if views else None
            )
            all_weights['black_litterman'] = bl_weights
        except Exception as e:
            self.logger.error(f"  Black-Litterman failed: {e}")
            all_weights['black_litterman'] = {}

        # 3. Risk Parity
        try:
            self.logger.info("  Running Risk Parity optimization...")
            rp_weights = self.optimizer.risk_parity_optimization(returns_df)
            all_weights['risk_parity'] = rp_weights
        except Exception as e:
            self.logger.error(f"  Risk Parity failed: {e}")
            all_weights['risk_parity'] = {}

        # 4. CVaR
        try:
            self.logger.info("  Running CVaR optimization...")
            cvar_weights = self.optimizer.cvar_optimization(
                returns_df,
                alpha=self.config.get('risk.var_confidence', 0.95)
            )
            all_weights['cvar'] = cvar_weights
        except Exception as e:
            self.logger.error(f"  CVaR failed: {e}")
            all_weights['cvar'] = {}

        # 5. RL Agent
        try:
            self.logger.info("  Running RL agent optimization...")
            rl_weights = self._get_rl_allocation(processed_data, returns_df)
            all_weights['rl_agent'] = rl_weights
        except Exception as e:
            self.logger.error(f"  RL agent failed: {e}")
            all_weights['rl_agent'] = {}

        # Combine all methods using profile weights
        combined_weights = {}
        tickers = returns_df.columns

        for ticker in tickers:
            combined_weights[ticker] = 0.0

            for method_name, method_weights in all_weights.items():
                if method_weights and ticker in method_weights:
                    method_weight = profile_weights.get(method_name, 0.0)
                    combined_weights[ticker] += method_weights[ticker] * method_weight

        # Normalize to sum to 1
        total = sum(combined_weights.values())
        if total > 0:
            combined_weights = {k: v/total for k, v in combined_weights.items()}

        self.logger.info(f"\n  Combined {len([w for w in all_weights.values() if w])} methods successfully")

        return combined_weights

    def _get_rl_allocation(
        self,
        processed_data: Dict[str, pd.DataFrame],
        returns_df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Get portfolio allocation using RL agent.
        Trains a quick RL agent on historical data.
        """
        try:
            # Prepare price data and features for RL environment
            price_data = {ticker: df[['Open', 'High', 'Low', 'Close', 'Volume']]
                         for ticker, df in processed_data.items()}

            # Use all features for RL
            features = {}
            for ticker, df in processed_data.items():
                feature_cols = [col for col in df.columns
                               if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
                if feature_cols:
                    features[ticker] = df[feature_cols]
                else:
                    # Fallback: use price-based features
                    features[ticker] = df[['Close']].pct_change().fillna(0)

            # Initialize RL agent
            rl_config = self.config.get('optimization.reinforcement_learning', {})
            agent = RLPortfolioAgent(
                algorithm=rl_config.get('algorithm', 'PPO'),
                learning_rate=rl_config.get('learning_rate', 0.0003),
                gamma=rl_config.get('gamma', 0.99),
                config=self.config.config
            )

            # Create environment
            agent.create_environment(
                price_data=price_data,
                features=features,
                initial_balance=self.config.get('portfolio.initial_budget', 10000),
                transaction_cost=self.config.get('portfolio.transaction_cost', 0.001)
            )

            # Quick training (reduced episodes for speed)
            training_episodes = rl_config.get('training_episodes', 1000)
            quick_training = min(training_episodes, 500)  # Cap at 500 for speed

            self.logger.info(f"    Training RL agent for {quick_training} timesteps...")
            agent.train(total_timesteps=quick_training)

            # Get current features for prediction
            current_features = {}
            for ticker in processed_data.keys():
                if ticker in features and len(features[ticker]) > 0:
                    current_features[ticker] = features[ticker].iloc[-1].values

            # Get allocation
            allocation = agent.get_portfolio_allocation(current_features)

            self.logger.info(f"    RL allocation generated for {len(allocation)} assets")
            return allocation

        except Exception as e:
            self.logger.error(f"RL allocation failed: {e}")
            # Fallback to equal weights
            tickers = list(returns_df.columns)
            return {ticker: 1.0/len(tickers) for ticker in tickers}

    def print_results(self, results: Dict):
        """Print results in formatted way."""
        print("\n" + "="*80)
        print("PORTFOLIO ALLOCATION RESULTS")
        print("="*80)

        # Risk profile
        risk_profile = self.config.get('optimization.risk_profile', 'medium')
        print(f"\n🎯 RISK PROFILE: {risk_profile.upper()}")
        print(f"   Strategy: Combined all optimization methods")

        # Portfolio weights
        print("\n📊 PORTFOLIO WEIGHTS:")
        print("-"*80)
        weights = results['weights']
        for ticker in sorted(weights.keys(), key=lambda x: weights[x], reverse=True):
            if weights[ticker] > 0.01:
                print(f"  {ticker:6s}: {weights[ticker]*100:6.2f}%")

        # Discrete allocation
        print("\n💰 SHARE ALLOCATION (Budget: ${:,.2f}):".format(results['budget']))
        print("-"*80)
        allocation = results['allocation']
        prices = results['latest_prices']
        total_invested = 0

        for ticker in sorted(allocation.keys(), key=lambda x: allocation[x] * prices.get(x, 0), reverse=True):
            shares = allocation[ticker]
            price = prices.get(ticker, 0)
            value = shares * price
            total_invested += value
            print(f"  {ticker:6s}: {shares:4d} shares @ ${price:7.2f} = ${value:10,.2f}")

        print(f"\n  Total Invested: ${total_invested:,.2f}")
        print(f"  Cash Remaining: ${results['budget'] - total_invested:,.2f}")

        # Performance metrics
        print("\n📈 EXPECTED PERFORMANCE:")
        print("-"*80)
        metrics = results['metrics']
        print(f"  Annual Return:      {metrics['annual_return']*100:6.2f}%")
        print(f"  Annual Volatility:  {metrics['annual_volatility']*100:6.2f}%")
        print(f"  Sharpe Ratio:       {metrics['sharpe_ratio']:6.2f}")
        print(f"  Sortino Ratio:      {metrics['sortino_ratio']:6.2f}")
        print(f"  Max Drawdown:       {metrics['max_drawdown']*100:6.2f}%")

        # Risk metrics
        print("\n⚠️  RISK METRICS:")
        print("-"*80)
        risk = results['risk_metrics']
        print(f"  VaR (95%):          {risk['var_95']*100:6.2f}%")
        print(f"  CVaR (95%):         {risk['cvar_95']*100:6.2f}%")

        # Sentiment
        if results['sentiment_scores']:
            print("\n💭 SENTIMENT SCORES:")
            print("-"*80)
            for ticker, score in sorted(results['sentiment_scores'].items(), key=lambda x: x[1], reverse=True):
                sentiment = "POSITIVE" if score > 0.1 else ("NEGATIVE" if score < -0.1 else "NEUTRAL")
                print(f"  {ticker:6s}: {score:+.3f}  ({sentiment})")

        # ML Predictions
        if results['predictions']:
            print("\n🤖 ML PREDICTIONS (Next Day Return):")
            print("-"*80)
            for ticker, pred in sorted(results['predictions'].items(), key=lambda x: x[1], reverse=True):
                if abs(pred) > 0.001:
                    direction = "↑ BULLISH" if pred > 0 else "↓ BEARISH"
                    print(f"  {ticker:6s}: {pred*100:+6.2f}%  {direction}")

        print("\n" + "="*80)
        print(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80 + "\n")
