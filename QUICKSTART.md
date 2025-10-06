# AI Portfolio Manager - Quickstart Guide

## 🚀 Installation (1 command)

```bash
./setup.sh
```

This will:
- Install Python dependencies
- Create virtual environment
- Setup directory structure

## 💡 Usage

### Quick Start (No tickers needed!)
```bash
# Analyze top 50 US stocks automatically
./predict.sh
```

### Basic Examples
```bash
# Specific stocks
./predict.sh AAPL MSFT GOOGL

# Custom settings
./predict.sh AAPL MSFT GOOGL AMZN --budget 50000 --risk low

# Aggressive on top 50 stocks
./predict.sh --risk high --budget 100000

# Save results to JSON
./predict.sh --output results/my_portfolio.json
```

## 🎯 Risk Profiles

The system uses **ALL 4 optimization methods** combined:

### `--risk low` (Conservative)
- 40% Risk Parity (equal risk distribution)
- 25% Black-Litterman (market views)
- 20% CVaR (minimize tail risk)
- 15% Markowitz (moderate returns)

### `--risk medium` (Balanced) **[DEFAULT]**
- 35% Markowitz (balanced returns)
- 30% Black-Litterman (sentiment + views)
- 20% Risk Parity (diversification)
- 15% CVaR (moderate risk control)

### `--risk high` (Aggressive)
- 50% Markowitz (maximum Sharpe ratio)
- 30% Black-Litterman (leverage views)
- 10% Risk Parity (less diversification)
- 10% CVaR (minimal tail risk focus)

## 📊 Output

The script provides:
1. **Portfolio Weights** - Percentage allocation per stock
2. **Share Allocation** - Exact number of shares to buy
3. **Performance Metrics** - Expected return, volatility, Sharpe ratio
4. **Risk Metrics** - VaR, CVaR
5. **Sentiment Analysis** - News sentiment per stock

## ⚙️ Configuration

Edit `config/config.yaml` to customize:
- Default budget
- Risk profile
- Transaction costs
- API keys (Alpha Vantage, News API)

## 📝 Examples

```bash
# Quick analysis of top 50 US stocks (default: medium risk, config budget)
./predict.sh

# Conservative auto-selected portfolio with $100k
./predict.sh --risk low --budget 100000

# Aggressive tech portfolio
./predict.sh NVDA AMD TSLA PLTR --risk high

# Save analysis for later
./predict.sh --output results/portfolio_$(date +%Y%m%d).json

# Specific stocks with custom settings
./predict.sh AAPL MSFT GOOGL AMZN META --risk medium --period 5y
```

## 🔐 API Keys (Optional but Recommended)

Get free API keys:
1. **Alpha Vantage**: https://www.alphavantage.co/support/#api-key
2. **News API**: https://newsapi.org/register

Add to `config/config.yaml`:
```yaml
data:
  alpha_vantage:
    api_key: "YOUR_KEY_HERE"
  news_api:
    api_key: "YOUR_KEY_HERE"
```

## ⚠️ Disclaimer

**Educational purposes only. Not financial advice.**

---

For full documentation, see [README.md](README.md)
