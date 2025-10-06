# AI Portfolio Manager - Project Overview

## 📋 Project Summary

AI Portfolio Manager is a sophisticated automated portfolio management system that combines multiple machine learning models, sentiment analysis, and advanced portfolio optimization techniques to generate optimal stock allocations.

**Key Features:**
- 🤖 **5 Optimization Methods Combined** (Markowitz, Black-Litterman, Risk Parity, CVaR, RL)
- 🧠 **Machine Learning Predictions** (XGBoost + LightGBM ensemble)
- 💭 **Sentiment Analysis** (FinBERT transformer + VADER)
- 🎯 **Risk-Based Profiles** (Low/Medium/High with different optimization weights)
- 📊 **67+ Technical Indicators** (via TA-Lib when available)
- 🔄 **Backtesting Engine** (with commission/slippage simulation)
- 🎮 **Reinforcement Learning** (PPO/SAC/DDPG agents for dynamic allocation)

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
│   │   ├── sentiment_data.py      # News fetcher
│   │   └── sentiment_analyzer.py  # FinBERT + VADER sentiment analysis
│   ├── features/                   # Feature engineering
│   │   ├── technical_indicators.py # 67+ technical indicators
│   │   └── feature_engineering.py  # Advanced feature creation
│   ├── models/                     # ML models
│   │   ├── lstm_model.py          # LSTM/GRU deep learning
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
- LSTM and GRU architectures
- Sequence-based prediction (60-day lookback)
- PyTorch implementation
- Used for time-series forecasting

**RL Agent** (`rl_agent.py`)
- **Custom Gym Environment**: Portfolio management as RL problem
- **Algorithms**: PPO (default), SAC, DDPG
- **State**: Technical features for all assets
- **Action**: Portfolio weights (continuous)
- **Reward**: Portfolio return minus transaction costs
- Trains on historical data and generates allocation

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

### 7. Main Orchestrator (`src/orchestrator.py`)

**Pipeline Flow**:
```
1. Fetch Market Data (yfinance)
          ↓
2. Analyze Sentiment (FinBERT + VADER)
          ↓
3. Engineer Features (67+ indicators)
          ↓
4. Generate ML Predictions (XGBoost + LightGBM)
          ↓
5. Optimize Portfolio (5 methods combined by risk profile)
   - Markowitz
   - Black-Litterman (uses sentiment + ML)
   - Risk Parity
   - CVaR
   - RL Agent (trains on historical data)
          ↓
6. Apply Risk Management (position limits, VaR)
          ↓
7. Calculate Discrete Allocation (exact shares to buy)
```

---

## ⚙️ Configuration System

All parameters are centralized in `config/config.yaml`:

### Risk Profiles

**Low Risk (Conservative)**:
```yaml
mean_variance: 0.10    # Less focus on max returns
black_litterman: 0.20  # Market views
risk_parity: 0.35      # HIGH - Equal risk distribution
cvar: 0.20             # HIGH - Minimize tail risk
rl_agent: 0.15         # Moderate RL
```

**Medium Risk (Balanced)** [DEFAULT]:
```yaml
mean_variance: 0.30    # Balanced returns
black_litterman: 0.25  # Market views + sentiment
risk_parity: 0.15      # Some diversification
cvar: 0.10             # Moderate tail risk
rl_agent: 0.20         # Moderate RL
```

**High Risk (Aggressive)**:
```yaml
mean_variance: 0.45    # HIGH - Max Sharpe ratio
black_litterman: 0.25  # Leverage views
risk_parity: 0.05      # LOW - Less diversification
cvar: 0.05             # LOW - Accept more tail risk
rl_agent: 0.20         # Moderate RL
```

### ML Model Configuration

```yaml
models:
  xgboost:
    n_estimators: 500
    max_depth: 7
    learning_rate: 0.05

  lightgbm:
    n_estimators: 500
    max_depth: 7
    learning_rate: 0.05

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

## 📊 Output

The system provides:

1. **Portfolio Weights** - Percentage allocation per stock
2. **Share Allocation** - Exact number of shares to buy
3. **Performance Metrics**:
   - Expected annual return
   - Annual volatility
   - Sharpe ratio
   - Sortino ratio
   - Max drawdown
4. **Risk Metrics**:
   - VaR (95%)
   - CVaR (95%)
5. **Sentiment Scores** - Per-ticker sentiment (POSITIVE/NEUTRAL/NEGATIVE)
6. **ML Predictions** - Next-day return predictions (BULLISH/BEARISH)

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

### ML Prediction Pipeline

1. Extract 67+ features per ticker
2. Train XGBoost + LightGBM on first 80% of data
3. Predict next-day return on latest data
4. Use predictions in Black-Litterman optimization

### RL Training

1. Create custom Gym environment with portfolio state
2. Train PPO agent for 500 timesteps (quick training)
3. Agent learns to maximize returns - transaction costs
4. Generate allocation from trained agent
5. Combine with other methods (15-20% weight)

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
- `torch`, `torchvision` - Deep learning (LSTM/GRU)
- `transformers`, `sentencepiece` - FinBERT sentiment
- `stable-baselines3`, `gymnasium` - RL agents
- `catboost` - Additional gradient boosting
- `TA-Lib` - Technical indicators (requires system install)

---

## 🎯 Key Design Decisions

1. **All Methods Combined**: Instead of choosing one optimization method, we combine ALL 5 methods weighted by risk profile. This provides more robust allocations.

2. **Sentiment + ML in Black-Litterman**: Black-Litterman naturally incorporates "views". We use sentiment analysis + ML predictions as views, making it more intelligent.

3. **RL Integration**: RL is trained quickly (500 steps) on historical data during each prediction. While not fully trained, it adds a dynamic learning component.

4. **Config-Driven**: ALL parameters are in `config.yaml`. No hardcoded values. Easy to tune without changing code.

5. **Automatic Stock Selection**: If no tickers provided, automatically uses top 50 US stocks (diversified across sectors). Zero-friction quick start.

6. **Parameter Override**: CLI arguments override config IN MEMORY only. Config file never modified.

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

6. **API Keys**: Sentiment analysis works better with Alpha Vantage + News API keys.

---

## 🔑 Key Files to Understand

### For Modifications:

1. **config/config.yaml** - Change all parameters here
2. **src/orchestrator.py** - Main pipeline logic
3. **src/portfolio/optimizer.py** - Optimization methods
4. **src/models/ensemble_models.py** - ML predictions
5. **src/models/rl_agent.py** - RL agent implementation
6. **predict.sh** - Entry point script

### For Adding Features:

- **New indicator**: Add to `src/features/technical_indicators.py`
- **New ML model**: Add to `src/models/` and integrate in `orchestrator.py`
- **New optimization method**: Add to `src/portfolio/optimizer.py`
- **New data source**: Add to `src/data/`

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

**Last Updated**: 2025-01-06
**Version**: 1.0.0
**Author**: AI Portfolio Manager Team
**License**: MIT
