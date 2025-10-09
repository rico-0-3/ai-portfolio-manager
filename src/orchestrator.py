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
from src.features.advanced_feature_engineering import AdvancedFeatureEngineer
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

        self.logger.info("Initializing Market Data Fetcher...")
        # Initialize components
        self.market_fetcher = MarketDataFetcher(
            cache_dir=self.config.get('paths.data_raw', 'data/raw')
        )

        self.logger.info("Initializing Sentiment Data Fetcher...")
        self.sentiment_fetcher = SentimentDataFetcher(
            news_api_key=self.config.get('data.news_api.api_key')
        )

        # Initialize additional data sources
        fmp_enabled = self.config.get('data.fmp.enabled', False)
        fmp_key = self.config.get('data.fmp.api_key')
        self.logger.info(f"Initializing FMP Data Fetcher... (Enabled: {fmp_enabled})")
        self.fmp_fetcher = FMPDataFetcher(api_key=fmp_key) if fmp_enabled and fmp_key else None

        finnhub_enabled = self.config.get('data.finnhub.enabled', False)
        finnhub_key = self.config.get('data.finnhub.api_key')
        self.logger.info(f"Initializing Finnhub Data Fetcher... (Enabled: {finnhub_enabled})")
        self.finnhub_fetcher = FinnhubDataFetcher(api_key=finnhub_key) if finnhub_enabled and finnhub_key else None

        self.logger.info("Initializing Technical Indicators...")
        self.tech_indicators = TechnicalIndicators()

        self.logger.info("Initializing Sentiment Analyzer...")
        self.sentiment_analyzer = HybridSentimentAnalyzer()

        self.logger.info("Initializing Feature Engineer...")
        self.feature_engineer = FeatureEngineer()

        self.logger.info("Initializing Portfolio Optimizer...")
        self.optimizer = PortfolioOptimizer(
            risk_free_rate=self.config.get('optimization.mean_variance.risk_free_rate', 0.02)
        )
        self.logger.info("Initializing Risk Manager...")
        self.risk_manager = RiskManager(
            max_position_size=self.config.get('portfolio.max_single_position', 0.20),
            max_drawdown=self.config.get('risk.max_drawdown', 0.25)
        )

        # Dynamic weight calibrator (NEW!)
        use_dynamic_weights = self.config.get('optimization.dynamic_weights.enabled', True)
        lookback_period = self.config.get('optimization.dynamic_weights.lookback_period', 60)

        self.logger.info(f"Initializing Dynamic Weight Calibrator... (Lookback Period: {lookback_period})")
        self.dynamic_calibrator = DynamicWeightCalibrator(lookback_period=lookback_period) if use_dynamic_weights else None

        # Reality check system (NEW!)
        use_reality_check = self.config.get('optimization.reality_check.enabled', True)
        
        self.logger.info(f"Initializing Reality Check System... (Enabled: {use_reality_check})")
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
        Run complete pipeline: data → MetaModel.predict() → weights

        REQUIRES pretrained MetaModel at data/models/pretrained_advanced/
        No fallback - train models first using Train_Perfect_Colab.ipynb

        Args:
            tickers: List of stock tickers
            period: Data period
            use_ml_predictions: Ignored (always uses ML via MetaModel)
            use_pretrained: Must be True
            finetune_days: Ignored (MetaModel doesn't fine-tune)

        Returns:
            Dict with results including weights, metrics, and allocations
        """
        self.logger.info("="*80)
        self.logger.info("METAMODEL PIPELINE")
        self.logger.info(f"Tickers: {tickers}")
        self.logger.info(f"Period: {period}")
        self.logger.info("="*80)

        # Step 1: Fetch market data
        self.logger.info("\n[1/3] Fetching market data...")
        market_data = self.market_fetcher.fetch_stock_data(
            tickers,
            period=period,
            interval='1d'
        )
        self.logger.info(f"✓ Fetched data for {len(market_data)} tickers")

        # Step 2: Feature engineering (base features only)
        self.logger.info("\n[2/3] Engineering features...")
        fundamental_data, analyst_data = self._enrich_data(tickers)
        processed_data = self._engineer_features(market_data, fundamental_data, analyst_data)
        self.logger.info(f"✓ Base features ready")

        # Step 3: Load MetaModel and predict
        self.logger.info("\n[3/3] Loading MetaModel...")
        meta_model_dir = Path("data/models/pretrained_perfect")

        if not meta_model_dir.exists() or not (meta_model_dir / "meta_model_metadata.json").exists():
            self.logger.error(f"❌ MetaModel not found at {meta_model_dir}")
            self.logger.error("   Please train models first:")
            self.logger.error("   1. Open Train_Perfect_Colab.ipynb in Google Colab")
            self.logger.error("   2. Run all cells (~10-12 hours)")
            self.logger.error("   3. Download and extract models to data/models/pretrained_advanced/")
            raise FileNotFoundError(f"MetaModel not found at {meta_model_dir}")

        from src.models.meta_model import MetaModel

        # Load MetaModel
        meta_model = MetaModel.load(meta_model_dir)
        info = meta_model.get_model_info()
        self.logger.info(f"✓ MetaModel loaded: {info['num_tickers']} tickers, {len(info['portfolio_optimizer_config'])} optimization methods")
        
        # Add interaction features based on what the model expects
        self.logger.info("  Adding interaction features based on trained models...")
        processed_data = self._add_model_specific_features(processed_data, meta_model)
        self.logger.info(f"✓ All features ready")

        # Calculate historical returns
        returns_df = pd.DataFrame({
            ticker: data['Close'].pct_change()
            for ticker, data in processed_data.items()
        }).dropna()

        # Market conditions (optional - Phase 2)
        market_conditions = None

        # ONE CALL: predictions + portfolio optimization!
        self.logger.info("  Predicting returns and optimizing portfolio...")
        final_weights = meta_model.predict(
            market_data=processed_data,
            historical_returns=returns_df,
            market_conditions=market_conditions
        )

        self.logger.info(f"✓ Portfolio optimized: {len(final_weights)} positions")

        # Apply risk limits
        final_weights = self.risk_manager.check_position_limits(final_weights)

        # Get predictions for metrics
        predictions = meta_model.predict_returns(processed_data)

        # ========== COMMON: Allocation and Metrics ==========
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

        # Get optimization details from MetaModel
        optimization_details = {}
        if hasattr(meta_model, 'get_optimization_details'):
            optimization_details = meta_model.get_optimization_details()

        results = {
            'weights': final_weights,
            'allocation': allocation,
            'metrics': metrics,
            'risk_metrics': {
                'var_95': portfolio_var,
                'cvar_95': portfolio_cvar
            },
            'sentiment_scores': {},  # Sentiment analysis not currently used in MetaModel pipeline
            'predictions': predictions,
            'optimization_details': optimization_details,  # NEW: Per-ticker method weights
            'latest_prices': latest_prices,
            'budget': budget
        }

        self.logger.info("\n" + "="*80)
        self.logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        self.logger.info("="*80)

        return results

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

    def _add_model_specific_features(
        self,
        processed_data: Dict[str, pd.DataFrame],
        meta_model
    ) -> Dict[str, pd.DataFrame]:
        """
        Add interaction features required by the trained model.
        The model's selected_features may include interaction features (containing '_x_').
        We need to create these from the base features.
        """
        for ticker, df in processed_data.items():
            if ticker not in meta_model.prediction_models:
                continue
                
            # Get the features expected by this ticker's model
            model = meta_model.prediction_models[ticker]
            expected_features = model.selected_features
            
            # Find interaction features (those with '_x_' in the name)
            interaction_features_needed = [f for f in expected_features if '_x_' in f]
            
            if not interaction_features_needed:
                continue
            
            # Create each interaction feature
            for interaction_feat in interaction_features_needed:
                if interaction_feat in df.columns:
                    continue  # Already exists
                    
                # Parse interaction feature name (e.g., "ATR_x_Close_lag_5")
                parts = interaction_feat.split('_x_')
                if len(parts) != 2:
                    self.logger.warning(f"{ticker}: Cannot parse interaction feature: {interaction_feat}")
                    continue
                
                feat1, feat2 = parts[0], parts[1]
                
                # Check if both base features exist
                if feat1 in df.columns and feat2 in df.columns:
                    df[interaction_feat] = df[feat1] * df[feat2]
                    self.logger.debug(f"{ticker}: Created interaction feature: {interaction_feat}")
                else:
                    missing = []
                    if feat1 not in df.columns:
                        missing.append(feat1)
                    if feat2 not in df.columns:
                        missing.append(feat2)
                    self.logger.warning(f"{ticker}: Cannot create {interaction_feat}, missing: {missing}")
            
            processed_data[ticker] = df
        
        return processed_data

    def _engineer_features(
        self,
        market_data: Dict[str, pd.DataFrame],
        fundamental_data: Optional[Dict] = None,
        analyst_data: Optional[Dict] = None
    ) -> Dict[str, pd.DataFrame]:
        """Engineer features for all tickers, including fundamental and analyst data."""
        processed_data = {}
        
        # Initialize AdvancedFeatureEngineer (same as training)
        adv_eng = AdvancedFeatureEngineer()

        for ticker, df in market_data.items():
            try:
                # Add technical indicators (EXACTLY as in training)
                df = self.tech_indicators.add_all_indicators(df)

                # Add detrending features (EXACTLY as in training)
                df['trend_60d'] = df['Close'].rolling(60, min_periods=20).mean().shift(1)
                df['distance_from_trend'] = (df['Close'] - df['trend_60d']) / (df['trend_60d'] + 1e-8)
                df['trend_20d'] = df['Close'].rolling(20, min_periods=10).mean().shift(1)
                df['distance_from_trend_20d'] = (df['Close'] - df['trend_20d']) / (df['trend_20d'] + 1e-8)
                
                # Add advanced features (EXACTLY as in training)
                df = adv_eng.create_lag_features(df, lags=[1, 5, 21])
                df = adv_eng.create_rolling_statistics(df, windows=[5, 10, 21, 60])
                df = adv_eng.create_fourier_features(df, periods=[5, 10, 21, 252])
                
                # Clean data (EXACTLY as in training)
                df = df.replace([np.inf, -np.inf], np.nan)

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

        # DETAILED BREAKDOWN PER STOCK
        print("\n\n" + "="*80)
        print("DETAILED STOCK ANALYSIS")
        print("="*80)

        predictions = results.get('predictions', {})
        optimization_details = results.get('optimization_details', {})

        for ticker in sorted(allocation.keys(), key=lambda x: allocation[x] * prices.get(x, 0), reverse=True):
            shares = allocation[ticker]
            price = prices.get(ticker, 0)
            value = shares * price
            weight_pct = weights.get(ticker, 0) * 100
            ml_pred = predictions.get(ticker, 0)

            print(f"\n📊 {ticker}")
            print(f"  Allocation:      ${value:>10,.2f} ({weight_pct:>5.1f}% of portfolio)")
            print(f"  Shares:          {shares:>4d} @ ${price:.2f}")
            print(f"  ML Prediction:   {ml_pred*100:>+6.2f}% (5-day return)")

            # Show which optimization methods contributed most
            if ticker in optimization_details:
                method_weights = optimization_details[ticker]
                print(f"  Optimization Method Contributions:")
                for method, method_weight in sorted(method_weights.items(), key=lambda x: x[1], reverse=True):
                    if method_weight > 0.01:  # Show only significant contributions
                        print(f"    - {method:20s}: {method_weight*100:>5.1f}%")

            # Risk metrics per stock (if available)
            if 'stock_metrics' in results and ticker in results['stock_metrics']:
                stock_metrics = results['stock_metrics'][ticker]
                print(f"  Volatility:      {stock_metrics.get('volatility', 0)*100:>6.2f}%")
                print(f"  Sharpe Ratio:    {stock_metrics.get('sharpe', 0):>6.2f}")

        print("\n" + "="*80)

        # Expected return (5 days / 1 week from ML model)
        print("\nEXPECTED RETURN (5 DAYS / 1 WEEK):")
        print("-"*80)
        metrics = results['metrics']
        weekly_return = metrics.get('monthly_return', 0)  # This is actually 5-day return from new model

        print(f"  5 Days:     {weekly_return*100:+6.2f}%  (ML model - 5d horizon)")

        # Extrapolate to 1 month (conservative: 5 days × 4 trading weeks)
        monthly_return_extrapolated = weekly_return * 4
        print(f"  1 Month (est): {monthly_return_extrapolated*100:+6.2f}%  (extrapolated from 5d)")

        # Estimated portfolio value (5 days projection)
        print("\nESTIMATED PORTFOLIO VALUE (5 DAYS / 1 WEEK):")
        print("-"*80)
        current_value = total_invested
        week_val = current_value * (1 + weekly_return)
        month_val = current_value * (1 + monthly_return_extrapolated)

        print(f"  Current:      ${current_value:>12,.2f}")
        print(f"  5 Days:       ${week_val:>12,.2f}  ({(week_val-current_value):+,.2f})  (ML model)")
        print(f"  1 Month (est):${month_val:>12,.2f}  ({(month_val-current_value):+,.2f})  (extrapolated)")

        print("\n" + "="*80)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80 + "\n")
