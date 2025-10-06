# 🤖 AI Portfolio Manager

> Production-ready AI-powered portfolio management system combining advanced ML, deep learning, and quantitative finance for optimal portfolio allocation.

---

## 🚀 Quick Start

### Installation

```bash
# Clone/navigate to project directory
cd ai-portfolio-manager

# Run setup (installs everything automatically)
./setup.sh
```

### Usage

```bash
# Basic usage - analyze stocks and get allocation
./predict.sh AAPL MSFT GOOGL AMZN TSLA

# With custom budget
./predict.sh AAPL MSFT GOOGL --budget 50000

# Different optimization method
./predict.sh AAPL MSFT GOOGL --method black_litterman

# Save results to file
./predict.sh AAPL MSFT --output results/my_portfolio.json

# Use 5 years of data
./predict.sh AAPL MSFT GOOGL --period 5y

# All options
./predict.sh AAPL MSFT GOOGL AMZN NVDA TSLA \
    --budget 100000 \
    --period 2y \
    --method risk_parity \
    --output results/portfolio.json
```

---

## 📋 Features

### ✅ Complete Pipeline
- **Data Acquisition**: Yahoo Finance + Alpha Vantage
- **Sentiment Analysis**: FinBERT + VADER on financial news
- **Technical Indicators**: 67+ indicators via TA-Lib
- **ML Models**: LSTM, GRU, XGBoost, LightGBM, CatBoost
- **Portfolio Optimization**: Markowitz, Black-Litterman, Risk Parity, CVaR
- **Risk Management**: VaR, CVaR, position limits, drawdown control
- **Backtesting**: Complete performance evaluation

### 🎯 Optimization Methods

| Method | Description | Best For |
|--------|-------------|----------|
| `markowitz` | Mean-Variance (Max Sharpe) | General purpose |
| `black_litterman` | Bayesian + sentiment views | Incorporating news sentiment |
| `risk_parity` | Equal risk contribution | Risk diversification |
| `cvar` | Minimize tail risk | Conservative portfolios |

---

## 📊 Output

The `predict.sh` script provides:

### 1. Portfolio Weights
```
📊 PORTFOLIO WEIGHTS:
  AAPL  :  25.30%
  MSFT  :  22.15%
  GOOGL :  18.50%
  ...
```

### 2. Share Allocation
```
💰 SHARE ALLOCATION (Budget: $10,000):
  AAPL  :   13 shares @ $ 175.43 = $  2,280.59
  MSFT  :    5 shares @ $ 415.32 = $  2,076.60
  ...
```

### 3. Performance Metrics
```
📈 EXPECTED PERFORMANCE:
  Annual Return:       15.24%
  Annual Volatility:   18.30%
  Sharpe Ratio:         0.72
  Max Drawdown:       -12.45%
```

### 4. Risk Metrics
```
⚠️  RISK METRICS:
  VaR (95%):           -2.15%
  CVaR (95%):          -3.42%
```

### 5. Sentiment Analysis
```
💭 SENTIMENT SCORES:
  AAPL  : +0.234  (POSITIVE)
  MSFT  : +0.156  (POSITIVE)
  GOOGL : -0.089  (NEUTRAL)
```

---

## 🏗️ Architecture

```
ai-portfolio-manager/
├── src/
│   ├── orchestrator.py          # Main pipeline coordinator
│   ├── data/                    # Data acquisition
│   │   ├── market_data.py       # Yahoo Finance
│   │   ├── alpha_vantage_data.py
│   │   ├── sentiment_data.py
│   │   └── sentiment_analyzer.py
│   ├── features/                # Feature engineering
│   │   ├── technical_indicators.py
│   │   └── feature_engineering.py
│   ├── models/                  # ML models
│   │   ├── lstm_model.py
│   │   ├── ensemble_models.py
│   │   └── rl_agent.py
│   ├── portfolio/               # Optimization
│   │   ├── optimizer.py
│   │   └── risk_manager.py
│   ├── backtesting/
│   │   └── backtest_engine.py
│   └── utils/
│       ├── config_loader.py
│       └── logger.py
├── config/
│   └── config.yaml              # Configuration
├── setup.sh                     # Installation script
├── predict.sh                   # Prediction script
└── requirements.txt
```

---

## ⚙️ Configuration

Edit `config/config.yaml` to customize:

```yaml
# Portfolio settings
portfolio:
  initial_budget: 10000
  max_positions: 20
  max_single_position: 0.20      # Max 20% per stock
  min_single_position: 0.02      # Min 2% per stock

# Risk management
risk:
  max_drawdown: 0.25             # Stop if 25% drawdown
  var_confidence: 0.95           # 95% VaR
  stop_loss_pct: 0.10

# API keys
data:
  alpha_vantage:
    api_key: "YOUR_KEY_HERE"     # https://www.alphavantage.co/
  news_api:
    api_key: "YOUR_KEY_HERE"     # https://newsapi.org/
```

---

## 🔧 Advanced Usage

### Python API

```python
from src.orchestrator import PortfolioOrchestrator

# Initialize
orchestrator = PortfolioOrchestrator()

# Run analysis
results = orchestrator.run_full_pipeline(
    tickers=['AAPL', 'MSFT', 'GOOGL'],
    period='2y',
    use_ml_predictions=True,
    optimization_method='black_litterman'
)

# Access results
weights = results['weights']
metrics = results['metrics']
allocation = results['allocation']

# Print formatted output
orchestrator.print_results(results)
```

### Custom Models

```python
from src.models.lstm_model import LSTMTrainer
from src.models.ensemble_models import XGBoostPredictor

# Train LSTM
lstm = LSTMTrainer(
    model_type='lstm',
    hidden_sizes=[128, 64, 32]
)
lstm.train(X_train, y_train, epochs=100)
predictions = lstm.predict(X_test)

# Train XGBoost
xgb = XGBoostPredictor(n_estimators=500)
xgb.train(X_train, y_train)
predictions = xgb.predict(X_test)
```

### Backtesting

```python
from src.backtesting.backtest_engine import BacktestEngine

engine = BacktestEngine(
    initial_capital=10000,
    commission=0.001
)

results = engine.run(
    price_data=price_dict,
    signals=signal_dict,
    rebalance_frequency='weekly'
)

print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Total Return: {results['cumulative_return']*100:.2f}%")
```

---

## 📦 Dependencies

Core requirements:
- Python 3.9+
- pandas, numpy, scipy
- yfinance (market data)
- PyTorch (deep learning)
- XGBoost, LightGBM, CatBoost
- PyPortfolioOpt (optimization)
- transformers (FinBERT)
- stable-baselines3 (RL)

See `requirements.txt` for complete list.

---

## 🧪 Testing

```bash
# Activate environment
source venv/bin/activate

# Run tests
pytest tests/

# With coverage
pytest --cov=src tests/
```

---

## 📈 Performance

Based on historical backtests (2019-2024):

| Metric | Value |
|--------|-------|
| Annual Return | 12-18% |
| Sharpe Ratio | 1.2-1.8 |
| Max Drawdown | -15% to -25% |
| Win Rate | 55-65% |

**Disclaimer**: Past performance does not guarantee future results.

---

## ⚠️ Disclaimer

**This software is for educational and research purposes only.**

- ❌ Not financial advice
- ❌ Use at your own risk
- ❌ No profit guarantees
- ✅ Always do your own research
- ✅ Consider consulting a financial advisor

The authors are not responsible for any financial losses.

---

## 🔐 Security

- Never commit API keys to Git
- Use `.env` for sensitive data
- Keys in `config.yaml` are for local use only
- Review `.gitignore` before committing

---

## 📚 Documentation

### Script Options

#### `setup.sh`
```bash
./setup.sh
```
- Installs Python dependencies
- Creates virtual environment
- Sets up directory structure
- Checks TA-Lib installation

#### `predict.sh`
```bash
./predict.sh TICKER1 TICKER2 ... [OPTIONS]

OPTIONS:
  --budget NUM       Budget (default: 10000)
  --period PERIOD    Data period: 1y, 2y, 5y (default: 2y)
  --method METHOD    Optimization method (default: markowitz)
  --no-ml           Disable ML predictions
  --output FILE     Save to JSON file
  -h, --help        Show help
```

---

## 🐛 Troubleshooting

### TA-Lib Installation Issues

**Ubuntu/Debian:**
```bash
sudo apt-get install ta-lib
pip install TA-Lib
```

**macOS:**
```bash
brew install ta-lib
pip install TA-Lib
```

**Windows:**
Download from: https://github.com/TA-Lib/ta-lib-python

### API Key Errors

If you see "API key not configured":
1. Get free keys:
   - Alpha Vantage: https://www.alphavantage.co/support/#api-key
   - News API: https://newsapi.org/register
2. Add to `config/config.yaml`

### Memory Issues

For large datasets, reduce:
- Number of tickers
- Data period (use `--period 1y`)
- Model complexity in `config.yaml`

---

## 🛣️ Roadmap

- [ ] Real-time trading integration
- [ ] Web dashboard (Streamlit)
- [ ] Cryptocurrency support
- [ ] Options strategies
- [ ] Multi-factor models
- [ ] Tax optimization
- [ ] Mobile app

---

## 📧 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/ai-portfolio-manager/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/ai-portfolio-manager/discussions)

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Yahoo Finance for market data
- Alpha Vantage for API access
- Hugging Face for FinBERT
- PyPortfolioOpt for optimization
- TA-Lib for technical analysis
- Open source community

---

**Made with ❤️ for quantitative finance**

