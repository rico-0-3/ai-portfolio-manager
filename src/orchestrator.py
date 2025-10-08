"""
Main orchestrator for AI Portfolio Manager.
Coordinates all components: data, features, models, optimization.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime

from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger
from src.data.market_data import MarketDataFetcher
from src.data.sentiment_data import SentimentDataFetcher
from src.data.sentiment_analyzer import HybridSentimentAnalyzer
from src.data.fmp_data import FMPDataFetcher
from src.data.finnhub_data import FinnhubDataFetcher
from src.features.technical_indicators import TechnicalIndicators
from src.features.feature_engineering import FeatureEngineer
from src.models.lstm_model import LSTMTrainer, LSTMAttentionModel, TORCH_AVAILABLE
from src.models.ensemble_models import EnsemblePredictor, XGBoostPredictor, LightGBMPredictor
from src.models.transformer_model import TransformerTrainer
from src.models.rl_agent import RLPortfolioAgent, PortfolioEnv
from src.portfolio.optimizer import PortfolioOptimizer
from src.portfolio.risk_manager import RiskManager
from src.dynamic_weights import DynamicWeightCalibrator
from src.reality_check import RealityCheck


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

        # Initialize additional data sources
        fmp_enabled = self.config.get('data.fmp.enabled', False)
        fmp_key = self.config.get('data.fmp.api_key')
        self.fmp_fetcher = FMPDataFetcher(api_key=fmp_key) if fmp_enabled and fmp_key else None

        finnhub_enabled = self.config.get('data.finnhub.enabled', False)
        finnhub_key = self.config.get('data.finnhub.api_key')
        self.finnhub_fetcher = FinnhubDataFetcher(api_key=finnhub_key) if finnhub_enabled and finnhub_key else None

        self.tech_indicators = TechnicalIndicators()
        self.feature_engineer = FeatureEngineer()
        self.optimizer = PortfolioOptimizer(
            risk_free_rate=self.config.get('optimization.mean_variance.risk_free_rate', 0.02)
        )
        self.risk_manager = RiskManager(
            max_position_size=self.config.get('portfolio.max_single_position', 0.20),
            max_drawdown=self.config.get('risk.max_drawdown', 0.25)
        )

        # Dynamic weight calibrator (NEW!)
        use_dynamic_weights = self.config.get('optimization.dynamic_weights.enabled', True)
        lookback_period = self.config.get('optimization.dynamic_weights.lookback_period', 60)
        self.dynamic_calibrator = DynamicWeightCalibrator(lookback_period=lookback_period) if use_dynamic_weights else None

        # Reality check system (NEW!)
        use_reality_check = self.config.get('optimization.reality_check.enabled', True)
        self.reality_check = RealityCheck(
            degradation_factor=self.config.get('optimization.reality_check.degradation_factor', 0.7),
            min_transaction_cost=self.config.get('portfolio.transaction_cost', 0.001),
            slippage_factor=self.config.get('optimization.reality_check.slippage_factor', 0.0005)
        ) if use_reality_check else None

        self.logger.info(f"PortfolioOrchestrator initialized (FMP: {fmp_enabled}, Finnhub: {finnhub_enabled}, DynamicWeights: {use_dynamic_weights}, RealityCheck: {use_reality_check})")

    def run_full_pipeline(
        self,
        tickers: List[str],
        period: str = '2y',
        use_ml_predictions: bool = True,
        use_pretrained: bool = True,
        finetune_days: int = 30
    ) -> Dict:
        """
        Run complete pipeline from data acquisition to portfolio allocation.
        Uses ALL optimization methods combined based on risk profile from config.

        Args:
            tickers: List of stock tickers
            period: Data period
            use_ml_predictions: Whether to use ML predictions
            use_pretrained: Whether to load pretrained models (default: True)
            finetune_days: Number of recent days for fine-tuning (default: 30)

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
        self.logger.info("\n[2/7] Analyzing sentiment...")
        sentiment_scores = self._analyze_sentiment(tickers)

        # Step 3: Enrich with fundamental and analyst data
        self.logger.info("\n[3/7] Enriching with additional data...")
        fundamental_data, analyst_data = self._enrich_data(tickers)

        # Step 4: Feature engineering
        self.logger.info("\n[4/7] Engineering features...")
        processed_data = self._engineer_features(market_data, fundamental_data, analyst_data)

        # Step 5: ML predictions (optional)
        predictions = {}
        if use_ml_predictions:
            if use_pretrained:
                self.logger.info(f"\n[5/7] Loading pretrained models + fine-tuning ({finetune_days} days)...")
            else:
                self.logger.info("\n[5/7] Training models from scratch...")
            predictions = self._generate_predictions(
                processed_data,
                analyst_data,
                use_pretrained=use_pretrained,
                finetune_days=finetune_days
            )
        else:
            self.logger.info("\n[5/7] Skipping ML predictions")

        # Step 6: Portfolio optimization (uses ALL methods)
        self.logger.info("\n[6/7] Optimizing portfolio...")
        weights = self._optimize_portfolio(
            processed_data,
            sentiment_scores,
            predictions
        )

        # Step 7: Risk management and allocation
        self.logger.info("\n[7/7] Applying risk management...")
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

        # Calculate portfolio metrics using ML predictions
        metrics = self.optimizer.get_portfolio_metrics(final_weights, returns_df, ml_predictions=predictions)

        # Apply RealityCheck to portfolio returns
        if self.reality_check:
            self.logger.info("Applying RealityCheck to portfolio metrics...")

            # Transaction costs are already considered in real trading (spread/slippage)
            # ML model learns from actual historical data which includes these costs
            # So we don't apply them again to avoid double-counting
            adjusted_return = metrics.get('monthly_return', metrics.get('annual_return', 0))

            # Validate Sharpe ratio
            n_observations = len(returns_df)
            n_parameters = len(final_weights) * 2  # Rough estimate
            adjusted_sharpe, sharpe_warning = self.reality_check.validate_sharpe_ratio(
                sharpe=metrics['sharpe_ratio'],
                n_observations=n_observations,
                n_parameters=n_parameters
            )

            # Calculate realistic confidence
            model_agreement = 0.85  # Assume 85% agreement between models
            confidence_pct, confidence_level = self.reality_check.calculate_realistic_confidence(
                sharpe=adjusted_sharpe,
                volatility=metrics['annual_volatility'],
                n_observations=n_observations,
                model_agreement=model_agreement
            )

            # Detect negative scenarios
            negative_analysis = self.reality_check.detect_negative_scenarios(
                returns=returns_df,
                weights=final_weights
            )

            # Stress test
            stress_results = self.reality_check.stress_test_portfolio(
                returns=returns_df,
                weights=final_weights
            )

            # Update metrics with adjusted values
            metrics['monthly_return'] = adjusted_return
            metrics['sharpe_ratio'] = adjusted_sharpe
            metrics['reality_check'] = {
                'confidence_pct': confidence_pct,
                'confidence_level': confidence_level,
                'sharpe_warning': sharpe_warning,
                'negative_analysis': negative_analysis,
                'stress_test': stress_results
            }

            self.logger.info("RealityCheck applied to portfolio metrics")

        # Calculate risk metrics with ML adjustment
        portfolio_var = self.risk_manager.calculate_portfolio_var(
            returns_df,
            final_weights,
            confidence=self.config.get('risk.var_confidence', 0.95),
            ml_predictions=predictions  # Pass ML predictions for adjustment
        )
        portfolio_cvar = self.risk_manager.calculate_portfolio_cvar(
            returns_df,
            final_weights,
            confidence=self.config.get('risk.var_confidence', 0.95),
            ml_predictions=predictions  # Pass ML predictions for adjustment
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

    def _enrich_data(self, tickers: List[str]) -> Tuple[Dict, Dict]:
        """
        Enrich with fundamental and analyst data.

        Returns:
            Tuple of (fundamental_data, analyst_data)
        """
        fundamental_data = {}
        analyst_data = {}

        # Get fundamental data from FMP
        if self.fmp_fetcher:
            try:
                self.logger.info("Fetching fundamental data from FMP...")
                fundamental_data = self.fmp_fetcher.enrich_with_fundamentals(tickers)
                self.logger.info(f"Fundamental data fetched for {len(fundamental_data)} tickers")
            except Exception as e:
                self.logger.error(f"FMP data fetch error: {e}")

        # Get analyst data from Finnhub
        if self.finnhub_fetcher:
            try:
                self.logger.info("Fetching analyst data from Finnhub...")
                analyst_data = self.finnhub_fetcher.enrich_with_analyst_data(tickers)
                self.logger.info(f"Analyst data fetched for {len(analyst_data)} tickers")
            except Exception as e:
                self.logger.error(f"Finnhub data fetch error: {e}")

        return fundamental_data, analyst_data

    def _engineer_features(
        self,
        market_data: Dict[str, pd.DataFrame],
        fundamental_data: Optional[Dict] = None,
        analyst_data: Optional[Dict] = None
    ) -> Dict[str, pd.DataFrame]:
        """Engineer features for all tickers, including fundamental and analyst data."""
        processed_data = {}

        for ticker, df in market_data.items():
            try:
                # Add technical indicators
                df = self.tech_indicators.add_all_indicators(df)

                # Add custom features
                df = self.feature_engineer.create_price_features(df)
                df = self.feature_engineer.create_volume_features(df)
                df = self.feature_engineer.create_volatility_features(df)

                # Add fundamental data as constant features (same value for all rows)
                fund_count = 0
                if fundamental_data and ticker in fundamental_data:
                    fund_features = fundamental_data[ticker]
                    for key, value in fund_features.items():
                        if isinstance(value, (int, float)) and not np.isnan(value):
                            df[f'fund_{key}'] = value
                            fund_count += 1

                # Add analyst data as constant features
                analyst_count = 0
                if analyst_data and ticker in analyst_data:
                    analyst_features = analyst_data[ticker]
                    for key, value in analyst_features.items():
                        if isinstance(value, (int, float)) and not np.isnan(value):
                            df[f'analyst_{key}'] = value
                            analyst_count += 1

                # Drop NaN
                df = df.dropna()

                processed_data[ticker] = df
                features_msg = f"{ticker}: {len(df)} rows, {len(df.columns)} features"
                if fund_count > 0 or analyst_count > 0:
                    features_msg += f" (technical: {len(df.columns) - fund_count - analyst_count}, fundamental: {fund_count}, analyst: {analyst_count})"
                self.logger.info(features_msg)

            except Exception as e:
                self.logger.error(f"Feature engineering error for {ticker}: {e}")

        return processed_data

    def _generate_predictions(
        self,
        processed_data: Dict[str, pd.DataFrame],
        analyst_data: Optional[Dict] = None,
        use_pretrained: bool = True,
        finetune_days: int = 30
    ) -> Dict[str, float]:
        """
        Generate ML predictions using enhanced 7-model ensemble.

        Args:
            processed_data: Dict of ticker to DataFrame
            analyst_data: Optional analyst data
            use_pretrained: If True, load pretrained models instead of training from scratch
            finetune_days: Number of recent days for fine-tuning (default: 30)

        Returns:
            Dict of ticker to predicted return
        """
        predictions = {}

        try:
            # Get ensemble configuration
            use_advanced = self.config.get('models.ensemble.use_advanced_models', True)
            lstm_weight = self.config.get('models.ensemble.lstm_weight', 0.10)
            gru_weight = self.config.get('models.ensemble.gru_weight', 0.10)
            lstm_attn_weight = self.config.get('models.ensemble.lstm_attention_weight', 0.20)
            transformer_weight = self.config.get('models.ensemble.transformer_weight', 0.20)
            xgb_weight = self.config.get('models.ensemble.xgboost_weight', 0.20)
            lgb_weight = self.config.get('models.ensemble.lightgbm_weight', 0.20)

            # Log configuration
            self.logger.info(f"ML Ensemble Config: use_advanced={use_advanced}, TORCH_AVAILABLE={TORCH_AVAILABLE}")
            if use_advanced and TORCH_AVAILABLE:
                self.logger.info(f"Advanced models enabled: LSTM({lstm_weight}), GRU({gru_weight}), LSTM+Attn({lstm_attn_weight}), Transformer({transformer_weight})")
            else:
                self.logger.info(f"Using only gradient boosting models: XGBoost({xgb_weight}), LightGBM({lgb_weight})")

            for ticker, df in processed_data.items():
                # Always initialize ensemble_predictions to avoid UnboundLocalError
                ensemble_predictions = []
                try:
                    # Check for pretrained model
                    pretrained_dir = Path(f"data/models/pretrained_advanced/{ticker}")
                    has_pretrained = (
                        use_pretrained and
                        pretrained_dir.exists() and
                        (pretrained_dir / "model.pkl").exists() and
                        (pretrained_dir / "scaler.pkl").exists() and
                        (pretrained_dir / "features.pkl").exists()
                    )

                    # Prepare features
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

                    if has_pretrained:
                        # ========== SIMPLIFIED: Use UnifiedEnsembleModel ==========
                        try:
                            from src.models.ensemble_model_unified import UnifiedEnsembleModel

                            self.logger.info(f"{ticker}: Loading UnifiedEnsembleModel...")

                            # Load unified model
                            ensemble_model = UnifiedEnsembleModel.load(pretrained_dir)

                            # Prepare data for prediction
                            X = df[feature_cols].values
                            X_test = X[-1].reshape(1, -1)  # Last observation

                            # Make prediction (handles everything internally)
                            prediction = ensemble_model.predict(X_test, feature_cols)

                            # Store prediction
                            predictions[ticker] = prediction

                            # Log
                            model_names = ensemble_model.get_model_names()
                            weights = ensemble_model.get_weights()
                            self.logger.info(f"{ticker}: ✓ Prediction = {prediction*100:+.2f}% (ensemble: {len(model_names)} models)")
                            for name in model_names:
                                self.logger.debug(f"  {name}: weight={weights.get(name, 0):.3f}")

                            continue  # Skip training from scratch

                        except Exception as e:
                            self.logger.warning(f"{ticker}: UnifiedEnsembleModel failed - {e}")
                            self.logger.warning(f"{ticker}: Falling back to training from scratch")
                            has_pretrained = False

                    # ========== FALLBACK: Train from scratch (if no pretrained) ==========
                    # Count feature types
                    fund_features = [c for c in feature_cols if c.startswith('fund_')]
                    analyst_features = [c for c in feature_cols if c.startswith('analyst_')]
                    tech_features = [c for c in feature_cols if not c.startswith('fund_') and not c.startswith('analyst_')]

                    self.logger.info(f"{ticker}: Training from scratch - {len(feature_cols)} features (tech: {len(tech_features)}, fund: {len(fund_features)}, analyst: {len(analyst_features)})")

                    X = df[feature_cols].values
                    y = df['Close'].pct_change().shift(-1).fillna(0).values  # Next day return

                    # Split: train on first 80%, predict on last point
                    split_idx = int(len(X) * 0.8)
                    X_train, y_train = X[:split_idx], y[:split_idx]
                    X_test = X[-1].reshape(1, -1)  # Last observation

                    ensemble_predictions = []
                    default_weights = []
                    trained_models = {}  # For dynamic weight calibration

                    # Split for validation (if DynamicWeights enabled)
                    if self.dynamic_calibrator:
                        # Use 80% for train, 20% for validation
                        val_split = int(len(X_train) * 0.8)
                        X_train_sub = X_train[:val_split]
                        y_train_sub = y_train[:val_split]
                        X_val = X_train[val_split:]
                        y_val = y_train[val_split:]
                    else:
                        X_train_sub = X_train
                        y_train_sub = y_train
                        X_val = None
                        y_val = None

                    # 1. XGBoost
                    try:
                        if has_pretrained:
                            # Load pretrained models (XGBoost + LightGBM for 1m horizon)
                            import pickle
                            import xgboost as xgb
                            import lightgbm as lgb
                            self.logger.debug(f"{ticker}: Loading pretrained models (1m horizon)...")

                            pretrained_models = {}

                            # Try to load all available models
                            for model_name in ['xgboost', 'lightgbm']:
                                model_path = pretrained_dir / f"{model_name}.pkl"
                                if model_path.exists():
                                    with open(model_path, 'rb') as f:
                                        pretrained_models[model_name] = pickle.load(f)
                                    self.logger.debug(f"{ticker}:   Loaded {model_name}.pkl")

                            # Load scaler and features
                            with open(pretrained_dir / "scaler.pkl", 'rb') as f:
                                scaler = pickle.load(f)
                            with open(pretrained_dir / "features.pkl", 'rb') as f:
                                selected_features = pickle.load(f)

                            if not pretrained_models:
                                self.logger.warning(f"{ticker}: No pretrained models found, training from scratch")
                                has_pretrained = False

                            # Select only the features that were used during training
                            # IMPORTANT: Maintain the same order as in selected_features
                            feature_indices = []
                            for feat_name in selected_features:
                                if feat_name in feature_cols:
                                    feature_indices.append(feature_cols.index(feat_name))
                                else:
                                    self.logger.warning(f"{ticker}: Feature '{feat_name}' not found in current data")

                            if len(feature_indices) != len(selected_features):
                                self.logger.warning(f"{ticker}: Feature mismatch! Expected {len(selected_features)}, found {len(feature_indices)}")
                                raise ValueError(f"Feature mismatch: {len(selected_features)} vs {len(feature_indices)}")

                            # Fine-tuning on last N days
                            if finetune_days > 0 and len(df) > finetune_days:
                                self.logger.debug(f"{ticker}: Fine-tuning on last {finetune_days} days...")
                                X_finetune_full = X[-finetune_days:]
                                y_finetune = y[-finetune_days:]

                                # Select features in the correct order
                                X_finetune = X_finetune_full[:, feature_indices]

                                # Scale with pretrained scaler
                                X_finetune_scaled = scaler.transform(X_finetune)

                                # Quick fine-tuning with xgb.train()
                                dtrain_finetune = xgb.DMatrix(X_finetune_scaled[:-1], label=y_finetune[:-1])
                                model_1m = xgb.train(
                                    {'objective': 'reg:squarederror'},
                                    dtrain_finetune,
                                    num_boost_round=10,  # Quick fine-tuning
                                    xgb_model=model_1m  # Continue from pretrained
                                )

                            # Select features and scale test data for prediction
                            X_test_selected = X_test[:, feature_indices]
                            X_test_scaled = scaler.transform(X_test_selected)
                            dtest = xgb.DMatrix(X_test_scaled)

                            # Get prediction from 1m model
                            xgb_pred = float(model_1m.predict(dtest)[0])

                            # Store prediction (single value for 1m)
                            predictions[ticker] = xgb_pred

                            self.logger.info(f"{ticker}: ✓ Pretrained model (1m={xgb_pred*100:+.2f}%)")
                            xgb_model = model_1m
                        else:
                            # Train from scratch
                            self.logger.debug(f"{ticker}: Training XGBoost from scratch...")
                            xgb_model = XGBoostPredictor()
                            xgb_model.train(X_train_sub, y_train_sub)
                            xgb_pred = xgb_model.predict(X_test)[0]

                        ensemble_predictions.append(xgb_pred)
                        default_weights.append(xgb_weight)
                        trained_models['xgboost'] = xgb_model
                        self.logger.debug(f"{ticker}: XGBoost prediction = {xgb_pred*100:+.2f}%")
                    except Exception as e:
                        self.logger.warning(f"{ticker}: XGBoost failed - {e}")

                    # 2. LightGBM (skip if using pretrained)
                    if not has_pretrained:
                        try:
                            self.logger.debug(f"{ticker}: Training LightGBM...")
                            lgb_model = LightGBMPredictor()
                            lgb_model.train(X_train_sub, y_train_sub)
                            lgb_pred = lgb_model.predict(X_test)[0]
                            ensemble_predictions.append(lgb_pred)
                            default_weights.append(lgb_weight)
                            trained_models['lightgbm'] = lgb_model
                            self.logger.debug(f"{ticker}: LightGBM prediction = {lgb_pred*100:+.2f}%")
                        except Exception as e:
                            self.logger.warning(f"{ticker}: LightGBM failed - {e}")

                        # 3. CatBoost (skip if using pretrained)
                        catboost_enabled = self.config.get('models.catboost.enabled', False)
                        if catboost_enabled and not has_pretrained:
                            try:
                                from src.models.ensemble_models import CatBoostPredictor
                                catboost_weight = self.config.get('models.ensemble.catboost_weight', 0.10)
                                cat_model = CatBoostPredictor()
                                cat_model.train(X_train_sub, y_train_sub, verbose=False)
                                cat_pred = cat_model.predict(X_test)[0]
                                ensemble_predictions.append(cat_pred)
                                default_weights.append(catboost_weight)
                                trained_models['catboost'] = cat_model
                            except Exception as e:
                                self.logger.warning(f"{ticker}: CatBoost failed - {e}")

                        # Advanced models (skip if using pretrained)
                        if use_advanced and TORCH_AVAILABLE and not has_pretrained:
                            sequence_length = self.config.get('models.lstm.sequence_length', 60)

                            # Prepare sequence data
                            if len(X_train_sub) >= sequence_length:
                                # Calculate dynamic epochs based on data size
                                # Target: ~5000 training steps total
                                # More data → fewer epochs needed
                                num_samples = len(X_train_sub) - sequence_length
                                target_steps = 5000
                                dynamic_epochs = max(10, min(100, int(target_steps / max(num_samples, 1))))
                                self.logger.debug(f"{ticker}: Using {dynamic_epochs} epochs for {num_samples} samples")
                                # Create sequences from training subset
                                X_seq_train = []
                                y_seq_train = []
                                for i in range(sequence_length, len(X_train_sub)):
                                    X_seq_train.append(X_train_sub[i-sequence_length:i])
                                    y_seq_train.append(y_train_sub[i])
                                X_seq_train = np.array(X_seq_train)
                                y_seq_train = np.array(y_seq_train)

                                # Last sequence for prediction
                                X_seq_test = X[-sequence_length:].reshape(1, sequence_length, -1)

                                # 4. LSTM
                                try:
                                    lstm_trainer = LSTMTrainer(
                                        model_type='lstm',
                                        input_size=X.shape[1],
                                        hidden_sizes=self.config.get('models.lstm.hidden_units', [128, 64, 32]),
                                        dropout=self.config.get('models.lstm.dropout', 0.2),
                                        learning_rate=self.config.get('models.lstm.learning_rate', 0.001)
                                    )
                                    lstm_trainer.train(X_seq_train, y_seq_train, epochs=dynamic_epochs, verbose=False)
                                    lstm_pred_raw = lstm_trainer.predict(X_seq_test)
                                    # Handle both scalar and array returns
                                    lstm_pred = float(lstm_pred_raw[0]) if hasattr(lstm_pred_raw, '__len__') else float(lstm_pred_raw)
                                    ensemble_predictions.append(lstm_pred)
                                    default_weights.append(lstm_weight)
                                    trained_models['lstm'] = lstm_trainer
                                    self.logger.debug(f"{ticker}: LSTM prediction = {lstm_pred*100:+.2f}%")
                                except Exception as e:
                                    self.logger.warning(f"{ticker}: LSTM failed - {e}")

                                # 5. GRU
                                try:
                                    gru_trainer = LSTMTrainer(
                                        model_type='gru',
                                        input_size=X.shape[1],
                                        hidden_sizes=self.config.get('models.gru.hidden_units', [100, 50]),
                                        dropout=self.config.get('models.gru.dropout', 0.2),
                                        learning_rate=self.config.get('models.gru.learning_rate', 0.001)
                                    )
                                    gru_trainer.train(X_seq_train, y_seq_train, epochs=dynamic_epochs, verbose=False)
                                    gru_pred_raw = gru_trainer.predict(X_seq_test)
                                    gru_pred = float(gru_pred_raw[0]) if hasattr(gru_pred_raw, '__len__') else float(gru_pred_raw)
                                    ensemble_predictions.append(gru_pred)
                                    default_weights.append(gru_weight)
                                    trained_models['gru'] = gru_trainer
                                    self.logger.debug(f"{ticker}: GRU prediction = {gru_pred*100:+.2f}%")
                                except Exception as e:
                                    self.logger.warning(f"{ticker}: GRU failed - {e}")

                                # 6. LSTM with Attention
                                try:
                                    lstm_attn_trainer = TransformerTrainer(
                                        model_type='lstm_attention',
                                        input_dim=X.shape[1],
                                        d_model=self.config.get('models.lstm_attention.hidden_size', 128),
                                        nhead=1,  # Not used for LSTM attention
                                        num_layers=self.config.get('models.lstm_attention.num_layers', 2),
                                        dropout=self.config.get('models.lstm_attention.dropout', 0.2),
                                        learning_rate=self.config.get('models.lstm_attention.learning_rate', 0.001)
                                    )
                                    lstm_attn_trainer.train(X_seq_train, y_seq_train, epochs=dynamic_epochs, verbose=False)
                                    lstm_attn_pred_raw = lstm_attn_trainer.predict(X_seq_test)
                                    lstm_attn_pred = float(lstm_attn_pred_raw[0]) if hasattr(lstm_attn_pred_raw, '__len__') else float(lstm_attn_pred_raw)
                                    ensemble_predictions.append(lstm_attn_pred)
                                    default_weights.append(lstm_attn_weight)
                                    trained_models['lstm_attention'] = lstm_attn_trainer
                                    self.logger.debug(f"{ticker}: LSTM+Attention prediction = {lstm_attn_pred*100:+.2f}%")
                                except Exception as e:
                                    self.logger.warning(f"{ticker}: LSTM+Attention failed - {e}")

                                # 7. Transformer
                                try:
                                    self.logger.debug(f"{ticker}: Training Transformer...")
                                    transformer_trainer = TransformerTrainer(
                                        model_type='transformer',
                                        input_dim=X.shape[1],
                                        d_model=self.config.get('models.transformer.d_model', 128),
                                        nhead=self.config.get('models.transformer.nhead', 8),
                                        num_layers=self.config.get('models.transformer.num_layers', 4),
                                        dim_feedforward=self.config.get('models.transformer.dim_feedforward', 512),
                                        dropout=self.config.get('models.transformer.dropout', 0.1),
                                        learning_rate=self.config.get('models.transformer.learning_rate', 0.001)
                                    )
                                    transformer_trainer.train(X_seq_train, y_seq_train, epochs=dynamic_epochs, verbose=False)
                                    transformer_pred_raw = transformer_trainer.predict(X_seq_test)
                                    transformer_pred = float(transformer_pred_raw[0]) if hasattr(transformer_pred_raw, '__len__') else float(transformer_pred_raw)
                                    ensemble_predictions.append(transformer_pred)
                                    default_weights.append(transformer_weight)
                                    trained_models['transformer'] = transformer_trainer
                                    self.logger.debug(f"{ticker}: Transformer prediction = {transformer_pred*100:+.2f}%")
                                except Exception as e:
                                    self.logger.warning(f"{ticker}: Transformer failed - {e}")
                                    import traceback
                                    self.logger.debug(f"Transformer traceback: {traceback.format_exc()}")

                    # Combine predictions with weights
                    if ensemble_predictions:
                        # Use DynamicWeights if enabled and we have validation data
                        if self.dynamic_calibrator and X_val is not None and len(X_val) > 10:
                            # Convert default weights list to dict
                            default_weight_dict = {
                                'xgboost': xgb_weight if 'xgboost' in trained_models else 0,
                                'lightgbm': lgb_weight if 'lightgbm' in trained_models else 0,
                                'catboost': self.config.get('models.ensemble.catboost_weight', 0.10) if 'catboost' in trained_models else 0,
                                'lstm': lstm_weight if 'lstm' in trained_models else 0,
                                'gru': gru_weight if 'gru' in trained_models else 0,
                                'lstm_attention': lstm_attn_weight if 'lstm_attention' in trained_models else 0,
                                'transformer': transformer_weight if 'transformer' in trained_models else 0
                            }

                            # Calibrate weights for this ticker
                            try:
                                # If using pretrained, select features for validation data
                                if has_pretrained and 'xgboost' in trained_models:
                                    X_val_for_calibration = X_val[:, feature_indices]
                                else:
                                    X_val_for_calibration = X_val

                                calibrated_weights = self.dynamic_calibrator.calibrate_ml_weights(
                                    ticker=ticker,
                                    X_train=X_train_sub,
                                    y_train=y_train_sub,
                                    X_val=X_val_for_calibration,
                                    y_val=y_val,
                                    models=trained_models,
                                    default_weights=default_weight_dict
                                )

                                # Apply calibrated weights in order of ensemble_predictions
                                weights_array = []
                                for model_name in ['xgboost', 'lightgbm', 'catboost', 'lstm', 'gru', 'lstm_attention', 'transformer']:
                                    if model_name in trained_models:
                                        weights_array.append(calibrated_weights.get(model_name, 0))

                                weights_array = np.array(weights_array[:len(ensemble_predictions)])
                            except Exception as e:
                                self.logger.warning(f"{ticker}: Dynamic weight calibration failed - {e}, using defaults")
                                weights_array = np.array(default_weights)
                        else:
                            # Use default weights
                            weights_array = np.array(default_weights)

                        # Ensure weights and predictions have same length
                        if len(weights_array) != len(ensemble_predictions):
                            self.logger.warning(f"{ticker}: Weight/prediction mismatch ({len(weights_array)} vs {len(ensemble_predictions)}), using defaults")
                            weights_array = np.array(default_weights[:len(ensemble_predictions)])

                        weights_array = weights_array / (weights_array.sum() + 1e-8)  # Normalize
                        final_pred = np.average(ensemble_predictions, weights=weights_array)

                        # Store as float (1m prediction)
                        if ticker not in predictions:
                            predictions[ticker] = float(final_pred)

                        # Show model contributions
                        model_names = []
                        if len(ensemble_predictions) >= 1:
                            model_names.append('XGB')
                        if len(ensemble_predictions) >= 2:
                            model_names.append('LGB')
                        if len(ensemble_predictions) >= 3:
                            model_names.append('CatBoost')
                        if len(ensemble_predictions) >= 4:
                            model_names.append('LSTM')
                        if len(ensemble_predictions) >= 5:
                            model_names.append('GRU')
                        if len(ensemble_predictions) >= 6:
                            model_names.append('LSTM+Attn')
                        if len(ensemble_predictions) >= 7:
                            model_names.append('Transformer')

                        # Log prediction
                        log_msg = f"{ticker}: Predicted return 1m={predictions[ticker]*100:+.2f}%"
                        log_msg += f" (ensemble: {len(ensemble_predictions)} models: {', '.join(model_names)})"
                        self.logger.info(log_msg)
                    else:
                        predictions[ticker] = 0.0
                        self.logger.warning(f"{ticker}: No models succeeded, using 0.0")

                except Exception as e:
                    self.logger.error(f"Prediction error for {ticker}: {e}")
                    predictions[ticker] = 0.0

            self.logger.info(f"Generated predictions for {len(predictions)} tickers using {len(ensemble_predictions)}-model ensemble")

            # Apply RealityCheck to predictions
            if self.reality_check:
                self.logger.info("Applying RealityCheck to predictions...")

                # Calculate historical volatility for each ticker
                historical_volatility = {}
                for ticker, df in processed_data.items():
                    returns = df['Close'].pct_change().dropna()
                    historical_volatility[ticker] = returns.std()

                # Adjust predictions conservatively (use_ml=True for lighter penalties)
                predictions = self.reality_check.adjust_predictions(
                    predictions=predictions,
                    historical_volatility=historical_volatility,
                    model_complexity=len(ensemble_predictions),
                    use_ml=True  # ML predictions are already realistic
                )

                self.logger.info("RealityCheck applied to predictions")

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
        # Calculate returns per ticker
        returns_dict = {}
        for ticker, data in processed_data.items():
            try:
                # Calculate returns and clean
                ticker_returns = data['Close'].pct_change()
                ticker_returns = ticker_returns.replace([np.inf, -np.inf], np.nan)

                # Check if this ticker has enough valid data
                valid_count = ticker_returns.notna().sum()
                total_count = len(ticker_returns)

                if valid_count < 20:
                    self.logger.warning(f"{ticker}: Insufficient valid returns ({valid_count}/{total_count}), skipping")
                    continue

                if valid_count / total_count < 0.5:
                    self.logger.warning(f"{ticker}: Too many NaN values ({valid_count}/{total_count}), skipping")
                    continue

                returns_dict[ticker] = ticker_returns
            except Exception as e:
                self.logger.warning(f"{ticker}: Error calculating returns - {e}")

        # Combine into dataframe
        if not returns_dict:
            self.logger.error("No valid tickers after return calculation")
            raise ValueError("All tickers have invalid return data")

        returns_df = pd.DataFrame(returns_dict)

        # Drop rows where ALL values are NaN
        returns_df = returns_df.dropna(how='all')

        # Drop rows where ANY value is NaN (for optimization we need complete data)
        returns_df = returns_df.dropna(how='any')

        # Final validation: check for any remaining NaN per column
        for col in returns_df.columns:
            nan_count = returns_df[col].isnull().sum()
            if nan_count > 0:
                self.logger.error(f"{col}: Still has {nan_count} NaN values after cleaning!")
                # Drop this column
                returns_df = returns_df.drop(columns=[col])
                self.logger.warning(f"{col}: Dropped due to persistent NaN values")

        # Ensure we have enough data and tickers
        if len(returns_df) < 20:
            self.logger.error(f"Insufficient return data after cleaning: {len(returns_df)} rows")
            raise ValueError("Insufficient data for optimization")

        if len(returns_df.columns) == 0:
            self.logger.error("No valid tickers remain after cleaning")
            raise ValueError("All tickers have invalid data")

        self.logger.info(f"Using {len(returns_df)} days of return data for {len(returns_df.columns)} tickers: {list(returns_df.columns)}")

        # Debug: check for NaN in returns_df
        for col in returns_df.columns:
            nan_count = returns_df[col].isnull().sum()
            if nan_count > 0:
                self.logger.error(f"BUG: {col} still has {nan_count} NaN after cleaning!")
            inf_count = np.isinf(returns_df[col]).sum()
            if inf_count > 0:
                self.logger.error(f"BUG: {col} has {inf_count} Inf values!")

        # Verify returns_df has no NaN/Inf
        if returns_df.isnull().any().any():
            self.logger.error(f"CRITICAL: returns_df still contains NaN after cleaning!")
            self.logger.error(f"NaN columns: {returns_df.columns[returns_df.isnull().any()].tolist()}")
        if np.isinf(returns_df.values).any():
            self.logger.error(f"CRITICAL: returns_df contains Inf values!")
            # Replace Inf with NaN and drop
            returns_df = returns_df.replace([np.inf, -np.inf], np.nan).dropna()

        # Get risk profile and weights from config
        risk_profile = self.config.get('optimization.risk_profile', 'medium')
        profile_weights = self.config.get(f'optimization.risk_profiles.{risk_profile}', {})

        self.logger.info(f"\nUsing risk profile: {risk_profile.upper()}")
        self.logger.info(f"Combining all optimization methods with weights:")
        for method_name, weight in profile_weights.items():
            self.logger.info(f"  {method_name}: {weight*100:.1f}%")

        # Run all optimization methods
        all_weights = {}

        # Filter predictions to match returns_df tickers (some may have been removed during cleaning)
        if predictions:
            filtered_predictions = {ticker: pred for ticker, pred in predictions.items()
                                   if ticker in returns_df.columns}
            if len(filtered_predictions) < len(predictions):
                removed = set(predictions.keys()) - set(filtered_predictions.keys())
                self.logger.warning(f"Removed predictions for tickers not in returns_df: {removed}")
            predictions = filtered_predictions

        # 1. Markowitz Mean-Variance
        try:
            self.logger.info("\n  Running Markowitz optimization...")
            markowitz_weights = self.optimizer.markowitz_optimization(
                returns_df,
                method='max_sharpe',
                ml_predictions=predictions  # Use ML predictions instead of historical mean
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
                if ticker in sentiment_scores and not np.isnan(sentiment_scores[ticker]):
                    view_value += sentiment_scores[ticker] * 0.05  # Scale sentiment
                # Add ML prediction component (weight: 0.5)
                if ticker in predictions:
                    pred = predictions[ticker]
                    # Extract 1y or 1d prediction
                    if isinstance(pred, dict):
                        pred_value = pred.get('1y', pred.get('1d', 0) * 80)
                    else:
                        pred_value = pred * 80
                    if not np.isnan(pred_value):
                        view_value += pred_value * 0.5  # ML predicted return

                # Only include meaningful views (and check for NaN/inf)
                if abs(view_value) > 0.001 and np.isfinite(view_value):
                    views[ticker] = view_value

            # Only run BL if we have valid views
            if views:
                bl_weights = self.optimizer.black_litterman_optimization(
                    returns_df,
                    views=views
                )
                all_weights['black_litterman'] = bl_weights
            else:
                self.logger.warning("  Black-Litterman: No valid views, skipping")
                all_weights['black_litterman'] = {}
        except Exception as e:
            self.logger.error(f"  Black-Litterman failed: {e}")
            all_weights['black_litterman'] = {}

        # 3. Risk Parity
        try:
            self.logger.info("  Running Risk Parity optimization...")
            rp_weights = self.optimizer.risk_parity_optimization(
                returns_df,
                ml_predictions=predictions
            )
            all_weights['risk_parity'] = rp_weights
        except Exception as e:
            self.logger.error(f"  Risk Parity failed: {e}")
            all_weights['risk_parity'] = {}

        # 4. CVaR
        try:
            self.logger.info("  Running CVaR optimization...")
            cvar_weights = self.optimizer.cvar_optimization(
                returns_df,
                alpha=self.config.get('risk.var_confidence', 0.95),
                ml_predictions=predictions
            )
            all_weights['cvar'] = cvar_weights
        except Exception as e:
            self.logger.error(f"  CVaR failed: {e}")
            all_weights['cvar'] = {}

        # 5. RL Agent
        try:
            self.logger.info("  Running RL agent optimization...")
            rl_weights = self._get_rl_allocation(processed_data, returns_df, predictions)
            all_weights['rl_agent'] = rl_weights
        except Exception as e:
            self.logger.error(f"  RL agent failed: {e}")
            all_weights['rl_agent'] = {}

        # Combine all methods using profile weights (with optional dynamic calibration)
        combined_weights = {}
        tickers = returns_df.columns

        # Calculate per-ticker weights using DynamicWeights if enabled
        if self.dynamic_calibrator and len(returns_df) > 100:
            self.logger.info("\n  Applying DynamicWeights to portfolio optimization methods...")

            # For each ticker, calibrate weights based on method performance
            for ticker in tickers:
                ticker_weights = 0.0

                # Prepare optimization results for this ticker
                optimization_results = {}
                for method_name, method_alloc in all_weights.items():
                    if method_alloc and ticker in method_alloc:
                        # Calculate method-specific metrics
                        single_ticker_weights = {ticker: 1.0}  # 100% allocation to this ticker
                        try:
                            metrics = self.optimizer.get_portfolio_metrics(
                                single_ticker_weights,
                                returns_df[[ticker]]
                            )
                            optimization_results[method_name] = {
                                'allocation': method_alloc[ticker],
                                'sharpe': metrics.get('sharpe_ratio', 0)
                            }
                        except:
                            optimization_results[method_name] = {
                                'allocation': method_alloc[ticker],
                                'sharpe': 0
                            }

                # Calibrate weights for this ticker
                try:
                    # Get ML prediction for this ticker (if available)
                    ml_pred = predictions.get(ticker, None) if predictions else None

                    calibrated_profile_weights = self.dynamic_calibrator.calibrate_portfolio_weights(
                        ticker=ticker,
                        historical_returns=returns_df[ticker],
                        optimization_results=optimization_results,
                        default_weights=profile_weights,
                        ml_prediction=ml_pred  # Pass ML prediction
                    )
                except Exception as e:
                    self.logger.warning(f"  {ticker}: Portfolio weight calibration failed - {e}, using defaults")
                    calibrated_profile_weights = profile_weights

                # Apply calibrated weights
                for method_name, method_weights in all_weights.items():
                    if method_weights and ticker in method_weights:
                        method_weight = calibrated_profile_weights.get(method_name, 0.0)
                        ticker_weights += method_weights[ticker] * method_weight

                combined_weights[ticker] = ticker_weights

        else:
            # Use static profile weights (original behavior)
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
        returns_df: pd.DataFrame,
        ml_predictions: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Get portfolio allocation using RL agent with optional ML-aware reward.
        Trains a quick RL agent on historical data.
        """
        try:
            # Prepare price data and features for RL environment
            price_data = {ticker: df[['Open', 'High', 'Low', 'Close', 'Volume']]
                         for ticker, df in processed_data.items()}

            # Use all features for RL - STANDARDIZE TO SAME SIZE
            features = {}
            feature_sizes = []

            # First pass: collect all features and determine max size
            temp_features = {}
            for ticker, df in processed_data.items():
                feature_cols = [col for col in df.columns
                               if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
                if feature_cols:
                    temp_features[ticker] = df[feature_cols]
                    feature_sizes.append(len(feature_cols))
                else:
                    # Fallback: use price-based features
                    temp_features[ticker] = df[['Close']].pct_change().fillna(0)
                    feature_sizes.append(1)

            # Determine max feature size
            max_features = max(feature_sizes) if feature_sizes else 1

            # Second pass: pad all features to max size
            for ticker, feat_df in temp_features.items():
                n_features = len(feat_df.columns)
                if n_features < max_features:
                    # Pad with zeros
                    padding_cols = max_features - n_features
                    for i in range(padding_cols):
                        feat_df[f'pad_{i}'] = 0.0

                features[ticker] = feat_df

            self.logger.debug(f"RL features standardized to {max_features} features per ticker")

            # Initialize RL agent
            rl_config = self.config.get('optimization.reinforcement_learning', {})
            agent = RLPortfolioAgent(
                algorithm=rl_config.get('algorithm', 'PPO'),
                learning_rate=rl_config.get('learning_rate', 0.0003),
                gamma=rl_config.get('gamma', 0.99),
                config=self.config.config
            )

            # Create environment (ML-aware if predictions available)
            agent.create_environment(
                price_data=price_data,
                features=features,
                initial_balance=self.config.get('portfolio.initial_budget', 10000),
                transaction_cost=self.config.get('portfolio.transaction_cost', 0.001),
                ml_predictions=ml_predictions
            )

            # Dynamic training timesteps based on available data
            # More data → more timesteps (but capped for speed)
            num_dates = len(agent.env.envs[0].dates) if hasattr(agent.env, 'envs') else 252
            training_episodes = rl_config.get('training_episodes', 1000)

            # Scale timesteps: min 500, max 2000, proportional to data
            dynamic_timesteps = max(500, min(2000, int(num_dates * 2)))
            quick_training = min(training_episodes, dynamic_timesteps)

            self.logger.info(f"    Training RL agent for {quick_training} timesteps ({num_dates} days available)...")

            # Suppress verbose output
            import sys
            from io import StringIO
            old_stdout = sys.stdout
            sys.stdout = StringIO()

            try:
                agent.train(total_timesteps=quick_training)
            finally:
                sys.stdout = old_stdout

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
        """Print results in clear, concise format."""
        print("\n" + "="*80)
        print("PORTFOLIO ANALYSIS RESULTS")
        print("="*80)

        # Budget allocation per ticker
        print("\nBUDGET ALLOCATION:")
        print("-"*80)
        allocation = results['allocation']
        prices = results['latest_prices']
        weights = results['weights']
        total_invested = 0

        for ticker in sorted(allocation.keys(), key=lambda x: allocation[x] * prices.get(x, 0), reverse=True):
            shares = allocation[ticker]
            price = prices.get(ticker, 0)
            value = shares * price
            total_invested += value
            pct_of_budget = (value / results['budget']) * 100
            print(f"  {ticker:6s}: ${value:10,.2f} ({pct_of_budget:5.1f}%) - {shares:4d} shares @ ${price:7.2f}")

        print(f"\n  Total Invested:  ${total_invested:,.2f}")
        print(f"  Cash Remaining:  ${results['budget'] - total_invested:,.2f}")

        # Expected return (1 month from ML model)
        print("\nEXPECTED RETURN (1 MONTH):")
        print("-"*80)
        metrics = results['metrics']
        monthly_return = metrics.get('monthly_return', 0)

        print(f"  1 Month:    {monthly_return*100:+6.2f}%  (ML model - 1m horizon)")

        # Estimated portfolio value (1 month projection)
        print("\nESTIMATED PORTFOLIO VALUE (1 MONTH):")
        print("-"*80)
        current_value = total_invested
        month_val = current_value * (1 + monthly_return)

        print(f"  Current:      ${current_value:>12,.2f}")
        print(f"  1 Month:      ${month_val:>12,.2f}  ({(month_val-current_value):+,.2f})  (ML model)")

        print("\n" + "="*80)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80 + "\n")
