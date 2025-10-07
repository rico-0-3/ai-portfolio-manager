# AI Portfolio Manager - Project Overview

## 📋 Project Summary

AI Portfolio Manager is a sophisticated automated portfolio management system that combines multiple machine learning models, sentiment analysis, and advanced portfolio optimization techniques to generate optimal stock allocations.

**Key Features:**
- 🤖 **5 Optimization Methods Combined** (Markowitz, Black-Litterman, Risk Parity, CVaR, RL)
- 🧠 **Advanced ML Ensemble** (XGBoost + LightGBM + LSTM+Attention + Transformer)
- 💭 **Multi-Source Sentiment** (FinBERT transformer + VADER + News API)
- 📊 **Fundamental Data Integration** (FMP API: PE ratio, ROE, debt ratios, financials)
- 📈 **Analyst Data Integration** (Finnhub API: price targets, buy/sell signals, consensus)
- 🎯 **Risk-Based Profiles** (Low/Medium/High with different optimization weights)
- 📉 **67+ Technical Indicators** (via TA-Lib when available)
- 🔄 **Backtesting Engine** (with commission/slippage simulation)
- 🎮 **Reinforcement Learning** (PPO/SAC/DDPG agents for dynamic allocation)
- 🔬 **State-of-the-Art Models** (Transformer with attention, LSTM with attention)
- 🎲 **Dynamic Weight Calibration** (NEW! Per-ticker weight optimization based on historical performance)
- ✅ **Reality Check System** (NEW! Conservative adjustments to prevent overly optimistic predictions)

---

## 🏗️ Project Structure

```
ai-portfolio-manager/
├── config/
│   └── config.yaml                 # Central configuration (all parameters)
├── src/
│   ├── data/                       # Data acquisition layer
│   │   ├── market_data.py         # Yahoo Finance data fetcher
│   │   ├── alpha_vantage_data.py  # Alpha Vantage API wrapper
│   │   ├── fmp_data.py            # Financial Modeling Prep (fundamentals)
│   │   ├── finnhub_data.py        # Finnhub (analyst data & price targets)
│   │   ├── sentiment_data.py      # News fetcher
│   │   └── sentiment_analyzer.py  # FinBERT + VADER sentiment analysis
│   ├── features/                   # Feature engineering
│   │   ├── technical_indicators.py # 67+ technical indicators
│   │   └── feature_engineering.py  # Advanced feature creation
│   ├── models/                     # ML models
│   │   ├── lstm_model.py          # LSTM/GRU + LSTM with Attention
│   │   ├── transformer_model.py   # Transformer with multi-head attention
│   │   ├── ensemble_models.py     # XGBoost + LightGBM + CatBoost
│   │   └── rl_agent.py            # Reinforcement learning (PPO/SAC/DDPG)
│   ├── portfolio/                  # Portfolio optimization
│   │   ├── optimizer.py           # 4 optimization methods
│   │   └── risk_manager.py        # VaR, CVaR, position limits
│   ├── backtesting/                # Backtesting framework
│   │   └── backtest_engine.py     # Historical performance simulation
│   ├── utils/                      # Utilities
│   │   ├── config_loader.py       # YAML config loader
│   │   └── logger.py              # Logging utilities
│   ├── dynamic_weights.py          # Dynamic weight calibration system (NEW!)
│   ├── reality_check.py            # Reality check & conservative adjustments (NEW!)
│   └── orchestrator.py             # Main pipeline coordinator
├── data/                           # Data storage
│   ├── raw/                       # Raw market data
│   ├── processed/                 # Processed features
│   ├── models/                    # Saved ML models
│   └── results/                   # Backtest results
├── logs/                          # Application logs
├── setup.sh                       # Installation script
├── predict.sh                     # Main prediction script
├── requirements.txt               # Python dependencies
├── README.md                      # Full documentation
├── QUICKSTART.md                  # Quick start guide
└── LICENSE                        # MIT license

```

---

## 🔧 Architecture & Components

### 1. Data Layer (`src/data/`)

**Market Data Fetcher** (`market_data.py`)
- Uses `yfinance` for OHLCV data
- Caches data locally to avoid repeated API calls
- Supports multiple tickers and timeframes

**Sentiment Analyzer** (`sentiment_analyzer.py`)
- **FinBERT**: Transformer-based model (when `transformers` installed)
- **VADER**: Rule-based sentiment (lightweight fallback)
- Fetches financial news from News API
- Generates sentiment scores per ticker

**Financial Modeling Prep (FMP) Fetcher** (`fmp_data.py`)
- Fetches fundamental data (PE ratio, ROE, debt-to-equity, etc.)
- Company profiles (market cap, beta, dividend yield)
- Key metrics (ROA, current ratio, profit margins)
- Financial ratios and growth metrics
- Free tier: 250 API calls per day

**Finnhub Data Fetcher** (`finnhub_data.py`)
- Fetches analyst recommendations (buy/sell/hold signals)
- Price targets (analyst consensus high/low/mean)
- Calculates analyst consensus score
- Computes upside potential from current price
- Free tier: 60 API calls per minute

### 2. Feature Engineering (`src/features/`)

**Technical Indicators** (`technical_indicators.py`)
- 67+ indicators including:
  - Moving Averages (SMA, EMA, WMA)
  - Momentum (RSI, MACD, Stochastic, ROC)
  - Volatility (Bollinger Bands, ATR, Keltner)
  - Volume (OBV, VWAP, MFI, CMF)
  - Trend (ADX, Aroon, Ichimoku)
- Uses TA-Lib when available, custom implementations otherwise

**Feature Engineer** (`feature_engineering.py`)
- Price features (returns, log returns, gaps)
- Volume features (volume ratios, spikes)
- Volatility features (rolling std, ranges)
- Cross-asset features (correlation, beta)

### 3. ML Models (`src/models/`)

**Ensemble Predictor** (`ensemble_models.py`)
- **XGBoost**: Gradient boosting (primary model)
- **LightGBM**: Fast gradient boosting
- **CatBoost**: Handles categorical features (optional)
- Combines predictions with weighted average
- Predicts next-day returns

**LSTM Models** (`lstm_model.py`)
- **Standard LSTM**: Multi-layer LSTM with dropout
- **GRU**: Gated Recurrent Unit variant
- **LSTM with Attention**: Enhanced LSTM with attention mechanism
  - Attention layer focuses on relevant time periods
  - Better performance during volatile markets
  - Weighted sum of LSTM outputs based on importance
- Sequence-based prediction (60-day lookback)
- PyTorch implementation
- Used for time-series forecasting

**Transformer Model** (`transformer_model.py`)
- **State-of-the-art architecture** for time series prediction
- Multi-head attention mechanism (8 heads)
- Positional encoding for sequence order
- 4 transformer encoder layers
- 512-dimensional feedforward network
- Superior at capturing long-term patterns
- Best performance on complex market dynamics

**RL Agent** (`rl_agent.py`)
- **Custom Gymnasium Environment**: Portfolio management as RL problem
- **Algorithms**: PPO (default), SAC, DDPG
- **State**: Technical features for all assets
- **Action**: Portfolio weights (continuous)
- **Reward**: Portfolio return minus transaction costs
- Trains on historical data and generates allocation
- Compatible with Gymnasium API (successor to OpenAI Gym)

### 4. Portfolio Optimization (`src/portfolio/`)

**Optimizer** (`optimizer.py`)
Implements 4 classical optimization methods:

1. **Markowitz Mean-Variance**
   - Maximizes Sharpe ratio
   - Classic efficient frontier approach

2. **Black-Litterman**
   - Combines market equilibrium with views
   - **Views = Sentiment (50%) + ML Predictions (50%)**
   - More stable than pure Markowitz

3. **Risk Parity**
   - Equal risk contribution from each asset
   - Better diversification

4. **CVaR (Conditional Value at Risk)**
   - Minimizes tail risk
   - Focus on worst-case scenarios

**Risk Manager** (`risk_manager.py`)
- Position limits (max 20% per stock by default)
- VaR and CVaR calculation
- Maximum drawdown monitoring
- Portfolio-level risk metrics

### 5. Reinforcement Learning (`src/models/rl_agent.py`)

**RL Integration**:
- Trains a quick RL agent (500 timesteps) on historical data
- Uses PPO (Proximal Policy Optimization) by default
- Learns optimal portfolio allocation through trial-and-error
- Combined with other methods in ensemble (15-20% weight)

### 6. Backtesting (`src/backtesting/`)

**Backtest Engine** (`backtest_engine.py`)
- Realistic trading simulation
- Commission costs (0.1% default)
- Slippage modeling (0.05% default)
- Rebalancing frequencies (daily/weekly/monthly)
- Performance metrics (Sharpe, Sortino, max drawdown)

### 7. Dynamic Weight Calibration (`src/dynamic_weights.py`) **NEW!**

**DynamicWeightCalibrator**
Adapts weights per ticker based on historical performance.

**Two-Level Calibration:**

1. **ML Ensemble Weights** (`calibrate_ml_weights()`)
   - Splits data into train (80%) + validation (20%)
   - Tests each model (XGBoost, LightGBM, LSTM, etc.) on validation set
   - Calculates MAE and RMSE per model
   - Applies inverse softmax → models with lower errors get higher weights
   - **Result**: AAPL might prefer XGBoost (35%) + LSTM (28%), while TSLA prefers LightGBM (42%)

2. **Portfolio Optimization Weights** (`calibrate_portfolio_weights()`)
   - Evaluates each method (Markowitz, Black-Litterman, Risk Parity, CVaR, RL) per ticker
   - Scores based on Sharpe ratio and allocation success
   - Applies 80/20 smoothing with default weights to prevent overfitting
   - **Result**: Tech stocks might favor Black-Litterman (40%), utilities favor Risk Parity (45%)

**Key Benefits:**
- Each ticker uses its optimal models and methods
- Prevents one-size-fits-all approach
- Automatic adaptation to market regime changes
- Cached results for performance

### 8. Reality Check System (`src/reality_check.py`) **NEW!**

**RealityCheck**
Applies conservative adjustments to prevent overly optimistic predictions.

**Adjustments Applied:**

1. **Out-of-Sample Degradation** (70% factor)
   - ML models perform 20-40% worse on unseen data (research-backed)
   - Reduces predictions by 30%

2. **Extreme Prediction Penalty**
   - If prediction > 2× historical volatility → 50% reduction
   - Prevents overfitting artifacts

3. **Overfitting Penalty** (5% per model)
   - More models = higher overfitting risk
   - 6-model ensemble → 25% penalty

4. **Mean Reversion** (80% factor)
   - Extreme predictions tend to revert to mean
   - Pulls 20% toward zero

5. **Transaction Costs**
   - Base cost (0.1%) + slippage (0.05%) + market impact
   - Weekly rebalancing → ~2-3% annual cost
   - Directly reduces expected returns

6. **Sharpe Ratio Validation**
   - Small sample bias correction
   - Overfitting ratio check (params/observations)
   - Cap at 2.5 (Renaissance Technologies ~2.0 is world-class)

7. **Negative Scenario Analysis**
   - Probability of losing day/week/month
   - Worst historical drawdowns
   - Average loss on down days

8. **Stress Testing**
   - 2008 Crisis (-37%), 2020 COVID (-34%), Recession (-15%)
   - Applies portfolio beta (0.85) for realistic estimates

**Key Benefits:**
- Prevents unrealistic expectations
- Forces consideration of downside
- Research-based adjustments
- Transparent risk disclosure

### 9. Main Orchestrator (`src/orchestrator.py`)

**Pipeline Flow** (7 Steps):
```
1. Fetch Market Data (yfinance)
          ↓
2. Analyze Sentiment (FinBERT + VADER + News API)
          ↓
3. Enrich with Additional Data (NEW!)
   - FMP: Fundamental data (PE, ROE, debt ratios)
   - Finnhub: Analyst data (price targets, consensus)
          ↓
4. Engineer Features (67+ indicators + fundamentals)
          ↓
5. Generate ML Predictions (6-model ensemble)
   - XGBoost, LightGBM (gradient boosting)
   - LSTM, GRU (recurrent networks)
   - LSTM with Attention, Transformer (advanced architectures)
   - **DynamicWeights applied**: Each ticker uses optimal model weights
          ↓
5a. Apply RealityCheck to Predictions (NEW!)
   - Out-of-sample degradation (30% reduction)
   - Overfitting penalties (25% for 6 models)
   - Extreme prediction caps, mean reversion
          ↓
6. Optimize Portfolio (5 methods combined)
   - Markowitz, Black-Litterman, Risk Parity, CVaR, RL Agent
   - **DynamicWeights applied**: Each ticker uses optimal method weights
          ↓
7. Apply Risk Management + RealityCheck (NEW!)
   - Position limits, VaR, CVaR
   - Transaction costs (-2-3% annually)
   - Sharpe validation, negative scenarios, stress testing
          ↓
8. Calculate Discrete Allocation (exact shares to buy)
```

**Output Format** (Concise & Actionable):
- Budget allocation per ticker ($ amount, % of budget, shares)
- Time-based predictions (1 week, 1 month, 3 months, 1 year)
- Realistic confidence level (MEDIUM-HIGH/MEDIUM/LOW-MEDIUM with %)
- Risk metrics (max drawdown, VaR, CVaR)
- **Negative scenario probabilities** (NEW! - losing day/week/month)
- **Stress test results** (NEW! - 2008 crisis, COVID, recession impacts)
- Estimated portfolio value at each time period

---

## ⚙️ Configuration System

All parameters are centralized in `config/config.yaml`:

### Risk Profiles

**Low Risk (Conservative)** - Basato su ricerca 2024-2025:
```yaml
mean_variance: 0.15    # Conservative returns
black_litterman: 0.30  # HIGH - BL+CVaR best for tail risk (research 2025)
risk_parity: 0.30      # HIGH - Good in uncertain markets (research 2024)
cvar: 0.20             # HIGH - Critical for downside protection
rl_agent: 0.05         # LOW - PPO basic config underperforms (research 2024)
```
*Reasoning*: Ricerca 2024 mostra che BL+CVaR ha migliori risk-adjusted returns. Risk Parity ha Sharpe 14.92% in mercati incerti. RL ridotto per evitare underperformance.

**Medium Risk (Balanced)** [DEFAULT] - Basato su ricerca 2024-2025:
```yaml
mean_variance: 0.25    # Highest Sharpe in studies (15.06%)
black_litterman: 0.30  # HIGH - BL+LSTM outperforms traditional (research 2025)
risk_parity: 0.20      # Moderate diversification
cvar: 0.15             # Moderate - BL+CVaR combination effective
rl_agent: 0.10         # LOW - Needs CNN features for good performance
```
*Reasoning*: Markowitz ha ottenuto Sharpe 15.06% in studi 2024. BL con LSTM supera max-Sharpe tradizionale. RL ridotto perché richiede configurazione avanzata.

**High Risk (Aggressive)** - Basato su ricerca 2024-2025:
```yaml
mean_variance: 0.40    # HIGH - Max Sharpe priority
black_litterman: 0.25  # Leverage ML views
risk_parity: 0.10      # LOW - Accept concentration
cvar: 0.10             # LOW - Some downside protection
rl_agent: 0.15         # MODERATE - Can be 1.85x better with good config
```
*Reasoning*: Focus su Markowitz per max Sharpe. RL aumentato perché research mostra DRL può essere 1.85x meglio di MVO se ben configurato.

### API Configuration

```yaml
data:
  # Financial Modeling Prep (Fundamental Data)
  fmp:
    enabled: true
    api_key: "YOUR_FMP_API_KEY"  # Get at https://financialmodelingprep.com/
    # Free tier: 250 calls/day

  # Finnhub (Analyst Data)
  finnhub:
    enabled: true
    api_key: "YOUR_FINNHUB_API_KEY"  # Get at https://finnhub.io/
    # Free tier: 60 calls/minute
```

### Dynamic Weight Configuration (NEW!)

```yaml
optimization:
  # Dynamic Weight Calibration
  dynamic_weights:
    enabled: true              # Enable per-ticker weight optimization
    lookback_period: 60        # Days for historical calibration
    # When enabled:
    # - ML models tested on validation set per ticker
    # - Portfolio methods scored by Sharpe per ticker
    # - Weights automatically adapted to each stock's characteristics

  # Reality Check System
  reality_check:
    enabled: true              # Enable conservative adjustments
    degradation_factor: 0.7    # Out-of-sample performance (0.5=very conservative, 0.9=optimistic)
    slippage_factor: 0.0005    # 0.05% slippage per trade
    # When enabled:
    # - Predictions reduced by 30-60%
    # - Transaction costs applied (-2-3% annually)
    # - Sharpe capped at 2.5
    # - Negative scenarios calculated
    # - Stress tests performed
```

### ML Model Configuration

```yaml
models:
  # LSTM with Attention (NEW!)
  lstm_attention:
    enabled: true
    sequence_length: 60
    hidden_size: 128
    num_layers: 2
    dropout: 0.2
    epochs: 100
    batch_size: 32
    learning_rate: 0.001

  # Transformer (NEW!)
  transformer:
    enabled: true
    d_model: 128
    nhead: 8
    num_layers: 4
    dim_feedforward: 512
    dropout: 0.1
    epochs: 100
    batch_size: 32
    learning_rate: 0.001

  # XGBoost
  xgboost:
    enabled: true
    n_estimators: 500
    max_depth: 7
    learning_rate: 0.05

  # LightGBM
  lightgbm:
    enabled: true
    n_estimators: 500
    max_depth: 7
    learning_rate: 0.05

  # Ensemble weights (NEW!)
  ensemble:
    use_advanced_models: true
    lstm_weight: 0.10
    gru_weight: 0.10
    lstm_attention_weight: 0.20  # LSTM with attention
    transformer_weight: 0.20     # Transformer
    xgboost_weight: 0.20
    lightgbm_weight: 0.20

optimization:
  reinforcement_learning:
    algorithm: "PPO"  # or SAC, DDPG
    training_episodes: 1000
    learning_rate: 0.0003
    gamma: 0.99
```

---

## 🚀 Usage

### Installation
```bash
./setup.sh
```

### Basic Usage

**Quick Start** (auto-selects top 50 US stocks):
```bash
./predict.sh
```

**Custom Tickers**:
```bash
./predict.sh AAPL MSFT GOOGL AMZN
```

**With Risk Profile**:
```bash
./predict.sh AAPL MSFT GOOGL --risk low --budget 50000
```

**Aggressive on Auto-Selected Stocks**:
```bash
./predict.sh --risk high --budget 100000
```

**Save Results**:
```bash
./predict.sh --output results/portfolio_$(date +%Y%m%d).json
```

### Script Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `TICKERS` | Stock symbols | Top 50 US stocks |
| `--budget` | Investment budget | From config.yaml |
| `--period` | Data period (1y, 2y, 5y) | 2y |
| `--risk` | Risk profile (low/medium/high) | medium |
| `--no-ml` | Disable ML predictions | ML enabled |
| `--output` | Save results to JSON | No file |

---

## 📊 Output (NEW FORMAT!)

The system provides **concise, actionable results**:

### 💰 Budget Allocation
- $ amount per ticker
- % of total budget
- Exact number of shares to buy
- Cash remaining

### 📈 Time-Based Predictions
- **1 Week**: Expected return (+/- %)
- **1 Month**: Expected return (+/- %)
- **3 Months**: Expected return (+/- %)
- **1 Year**: Expected return (+/- %)

### 🎯 Prediction Confidence
- **Overall Confidence**: HIGH/MEDIUM/LOW (percentage)
- **Sharpe Ratio**: Risk-adjusted return metric
- **Sortino Ratio**: Downside risk metric
- **Annual Volatility**: Expected price fluctuation

### ⚠️ Risk Analysis
- **Maximum Drawdown**: Worst historical decline
- **VaR (95%)**: Value at Risk
- **CVaR (95%)**: Conditional VaR (tail risk)

### 💎 Estimated Portfolio Value
- Current value
- After 1 week (+gain/loss)
- After 1 month (+gain/loss)
- After 3 months (+gain/loss)
- After 1 year (+gain/loss)

---

## 🔬 Technical Details

### Optimization Method Combination

The system runs ALL 5 optimization methods and combines them:

```python
combined_weight[ticker] = (
    markowitz_weight * profile['mean_variance'] +
    bl_weight * profile['black_litterman'] +
    rp_weight * profile['risk_parity'] +
    cvar_weight * profile['cvar'] +
    rl_weight * profile['rl_agent']
)
```

### Black-Litterman Views

Views are computed from sentiment + ML predictions:

```python
view[ticker] = (
    sentiment_score * 0.05 +      # Sentiment component
    ml_prediction * 0.5            # ML predicted return
)
```

### ML Prediction Pipeline (7-Model Ensemble)

1. Extract 67+ technical features + fundamental data per ticker
2. Train ensemble on first 80% of historical data:
   - **XGBoost** (gradient boosting)
   - **LightGBM** (fast gradient boosting)
   - **CatBoost** (optional, handles categorical features)
   - **LSTM** (sequence-based time series)
   - **GRU** (gated recurrent unit)
   - **LSTM with Attention** (focuses on important time periods)
   - **Transformer** (state-of-the-art with multi-head attention)
3. Each model predicts next-day return independently
4. Combine predictions using weighted average from config
5. Use ensemble prediction in Black-Litterman optimization

**Prediction Weights** (from config):
```python
xgboost: 0.20
lightgbm: 0.20
lstm: 0.10
gru: 0.10
lstm_attention: 0.20  # Best for volatility
transformer: 0.20     # Best for long patterns
```

### RL Training

1. Create custom Gymnasium environment with portfolio state
2. Train PPO agent for 2048 timesteps (quick training)
3. Agent learns to maximize returns - transaction costs
4. Generate allocation from trained agent
5. Combine with other methods (15-20% weight depending on risk profile)

### Time-Based Return Calculation

Returns for different time periods are calculated using compound interest:

```python
weekly_return = (1 + annual_return) ** (7/252) - 1    # 7 trading days
monthly_return = (1 + annual_return) ** (21/252) - 1  # ~21 trading days
quarterly_return = (1 + annual_return) ** (63/252) - 1 # ~63 trading days
annual_return = expected_return  # From portfolio optimization
```

### Confidence Scoring

Confidence is derived from Sharpe ratio:

```python
if sharpe_ratio > 1.0:
    confidence = "HIGH"
    confidence_pct = min(85 + (sharpe - 1.0) * 10, 95)
elif sharpe_ratio > 0.5:
    confidence = "MEDIUM"
    confidence_pct = 65 + (sharpe - 0.5) * 40
else:
    confidence = "LOW"
    confidence_pct = max(30, 65 * sharpe)
```

---

## 🛠️ Dependencies

### Core (Required)
- `numpy`, `pandas`, `scipy` - Data processing
- `yfinance` - Market data
- `scikit-learn` - ML utilities
- `xgboost`, `lightgbm` - Gradient boosting
- `PyPortfolioOpt`, `cvxpy` - Optimization
- `vaderSentiment` - Sentiment analysis
- `matplotlib`, `seaborn`, `plotly` - Visualization

### Optional (Heavy but included)
- `torch`, `torchvision` - Deep learning (LSTM/GRU/Transformer/Attention)
- `transformers`, `sentencepiece` - FinBERT sentiment
- `stable-baselines3`, `gymnasium` - RL agents (PPO/SAC/DDPG)
- `catboost` - Additional gradient boosting
- `TA-Lib` - Technical indicators (requires system install)
- `pyarrow` - Required for yfinance data handling

### APIs Used (Free Tiers)
- **Yahoo Finance** (via yfinance) - Market data (OHLCV)
- **Alpha Vantage** (optional) - Additional market data
- **News API** (optional) - Financial news for sentiment
- **Financial Modeling Prep (FMP)** - Fundamental data (PE, ROE, debt ratios)
  - Free tier: 250 API calls per day
- **Finnhub** - Analyst recommendations and consensus
  - Free tier: 60 API calls per minute
  - Note: Only free endpoints used (Premium APIs disabled)

---

## 🎯 Key Design Decisions

1. **All Methods Combined**: Instead of choosing one optimization method, we combine ALL 5 methods weighted by risk profile. This provides more robust allocations.

2. **Sentiment + ML in Black-Litterman**: Black-Litterman naturally incorporates "views". We use sentiment analysis + ML predictions + fundamentals as views, making it more intelligent.

3. **7-Model Ensemble**: Combines XGBoost, LightGBM, CatBoost, LSTM, GRU, LSTM+Attention, and Transformer for maximum prediction accuracy. Each model captures different aspects of market dynamics.

4. **Advanced Deep Learning**: Uses state-of-the-art architectures (Transformer with multi-head attention, LSTM with attention) for superior time series forecasting.

5. **Fundamental + Technical Analysis**: Integrates both fundamental data (PE ratios, ROE, debt) from FMP and technical indicators (67+) for comprehensive analysis.

6. **Analyst Consensus Integration**: Uses Finnhub's free analyst recommendation data to incorporate professional insights into predictions.

7. **RL Integration**: RL agent (PPO) is trained on 2048 timesteps during each prediction run, learning optimal portfolio allocation dynamically.

8. **Config-Driven**: ALL parameters are in `config.yaml`. No hardcoded values. Easy to tune without changing code.

9. **Automatic Stock Selection**: If no tickers provided, automatically uses top 50 US stocks (diversified across sectors). Zero-friction quick start.

10. **Parameter Override**: CLI arguments override config IN MEMORY only. Config file never modified.

11. **Concise Output**: Results focus on actionable metrics (time-based predictions, confidence levels, portfolio value projections) rather than overwhelming technical details.

---

## 📈 Future Enhancements (Planned)

- **Real-time Trading Integration**: Connect to broker APIs (Alpaca, Interactive Brokers)
- **Live Rebalancing**: Periodic automatic rebalancing
- **Pre-trained RL Models**: Save/load trained RL agents
- **Multi-timeframe Analysis**: Combine daily/weekly/monthly signals
- **Options Strategies**: Add options for hedging
- **Walk-forward Optimization**: Rolling window backtesting
- **Web Dashboard**: Interactive UI with Streamlit/Flask
- **Alert System**: Email/SMS notifications for opportunities

---

## ⚠️ Important Notes

1. **Educational Purpose**: This is for learning and research, NOT professional financial advice.

2. **Risk Disclaimer**: All investments carry risk. Past performance doesn't guarantee future results.

3. **Data Quality**: Results depend on quality of market data and news sentiment.

4. **Transaction Costs**: Real trading involves more costs (taxes, fees, spreads).

5. **Market Conditions**: Models trained on historical data may not work in different market regimes.

6. **API Keys**: System works better with FMP + Finnhub + News API keys. All use free tiers.

---

## 🔑 Key Files to Understand

### For Modifications:

1. **config/config.yaml** - Change all parameters here
2. **src/orchestrator.py** - Main 7-step pipeline logic
3. **src/portfolio/optimizer.py** - 5 optimization methods
4. **src/models/ensemble_models.py** - XGBoost, LightGBM, CatBoost
5. **src/models/lstm_model.py** - LSTM, GRU, LSTM+Attention
6. **src/models/transformer_model.py** - Transformer with attention
7. **src/models/rl_agent.py** - RL agent (PPO/SAC/DDPG)
8. **src/data/fmp_data.py** - Fundamental data fetcher
9. **src/data/finnhub_data.py** - Analyst data fetcher
10. **predict.sh** - Entry point script

### For Adding Features:

- **New indicator**: Add to `src/features/technical_indicators.py`
- **New ML model**: Add to `src/models/` and integrate in `orchestrator.py` `_generate_predictions()`
- **New optimization method**: Add to `src/portfolio/optimizer.py` and combine in `orchestrator.py`
- **New data source**: Add to `src/data/` and integrate in `orchestrator.py` `_enrich_data()`

---

## 📝 Example Workflow

```bash
# 1. Install
./setup.sh

# 2. Quick test with auto-selected stocks
./predict.sh

# 3. Test specific stocks with conservative profile
./predict.sh AAPL MSFT GOOGL AMZN --risk low --budget 10000

# 4. Save results for analysis
./predict.sh --risk medium --output results/test_$(date +%Y%m%d).json

# 5. Review logs
tail -f logs/portfolio_manager.log

# 6. Adjust config if needed
nano config/config.yaml

# 7. Run again
./predict.sh
```

---

## 🤝 Contributing

To extend this project:

1. All parameters should be in `config.yaml`
2. Follow the existing modular structure
3. Add proper error handling and logging
4. Update this documentation (claude.md)
5. Test with different risk profiles and stock sets

---

## 📚 References

- **Modern Portfolio Theory**: Markowitz (1952)
- **Black-Litterman Model**: Black & Litterman (1992)
- **Risk Parity**: Qian (2005)
- **CVaR Optimization**: Rockafellar & Uryasev (2000)
- **Reinforcement Learning**: Sutton & Barto (2018)
- **FinBERT**: Araci (2019)

---

**Last Updated**: 2025-01-07
**Version**: 2.0.0
**Author**: AI Portfolio Manager Team
**License**: MIT

---

## 🆕 What's New in v3.0.0

### Major New Features:

1. **Dynamic Weight Calibration System** 🎲
   - Per-ticker ML model weight optimization based on validation performance
   - Per-ticker portfolio method weight optimization based on historical Sharpe
   - Automatic adaptation: AAPL might prefer XGBoost+LSTM, TSLA prefers LightGBM
   - Two-level calibration: ML ensemble + portfolio optimization
   - 80/20 smoothing to prevent overfitting

2. **Reality Check System** ✅
   - Out-of-sample degradation (30% prediction reduction based on research)
   - Overfitting penalties (5% per model, up to 30% total)
   - Extreme prediction caps (2× volatility threshold)
   - Transaction cost modeling (-2-3% annually for weekly rebalancing)
   - Sharpe ratio validation (cap at 2.5, overfitting detection)
   - **Negative scenario analysis** - probability of losing days/weeks/months
   - **Stress testing** - 2008 crisis, COVID, recession impact estimates
   - Realistic confidence scoring (considers sample size, volatility, agreement)

3. **Comprehensive Bug Fixes**
   - RL agent broadcast error FIXED (feature standardization with padding)
   - Markowitz non-convex error FIXED (Ledoit-Wolf shrinkage + L2 regularization)
   - Black-Litterman Q array error FIXED (proper P matrix + Q array conversion)
   - NaN errors in returns FIXED (double cleaning + validation)

### Previous Features (v2.0.0):
1. **6-Model ML Ensemble** - XGBoost, LightGBM, CatBoost, LSTM, GRU, LSTM+Attention, Transformer
2. **Fundamental Data Integration** - FMP API for PE ratios, ROE, debt-to-equity
3. **Analyst Data Integration** - Finnhub API for recommendations and consensus
4. **Concise Output Format** - Time-based predictions, confidence levels
5. **Gymnasium API** - Updated from deprecated OpenAI Gym

### Key Improvements:
- **Predictions now realistic** - No more +35% annual returns on negative predictions
- **All optimization methods work** - Markowitz, BL, Risk Parity, CVaR, RL all functional
- **Transparent risk disclosure** - Shows probability of losses and stress scenarios
- **Adaptive per ticker** - Each stock uses its optimal models and methods
- **Research-backed adjustments** - All penalties based on academic literature
