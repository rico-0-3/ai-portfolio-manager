"""
Backtesting engine for portfolio strategies.
Evaluates historical performance with realistic trading simulation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Backtesting engine for portfolio strategies."""

    def __init__(
        self,
        initial_capital: Optional[float] = None,
        commission: Optional[float] = None,
        slippage: Optional[float] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize backtest engine.

        Args:
            initial_capital: Starting capital (if None, uses config)
            commission: Commission per trade (if None, uses config)
            slippage: Slippage per trade (if None, uses config)
            config: Config dict (optional)
        """
        # Load from config if not provided
        if config:
            self.initial_capital = initial_capital or config.get('portfolio', {}).get('initial_budget', 10000)
            self.commission = commission or config.get('portfolio', {}).get('transaction_cost', 0.001)
            self.slippage = slippage or 0.0005  # Not in config, reasonable default
        else:
            self.initial_capital = initial_capital or 10000
            self.commission = commission or 0.001
            self.slippage = slippage or 0.0005

        self.portfolio_value = []
        self.positions = {}
        self.trades = []
        self.cash = initial_capital

        logger.info(f"BacktestEngine initialized with ${initial_capital}")

    def run(
        self,
        price_data: Dict[str, pd.DataFrame],
        signals: Dict[str, pd.DataFrame],
        rebalance_frequency: str = 'weekly'
    ) -> Dict:
        """
        Run backtest.

        Args:
            price_data: Dict of ticker to price DataFrame
            signals: Dict of ticker to signal DataFrame (weights)
            rebalance_frequency: 'daily', 'weekly', 'monthly'

        Returns:
            Dictionary with backtest results
        """
        logger.info("Starting backtest...")

        all_dates = sorted(set().union(*[df.index for df in price_data.values()]))

        for date in all_dates:
            # Check if rebalance day
            if self._should_rebalance(date, rebalance_frequency):
                self._rebalance(date, price_data, signals)

            # Update portfolio value
            self._update_portfolio_value(date, price_data)

        results = self._calculate_results()
        logger.info("Backtest completed")

        return results

    def _should_rebalance(self, date: datetime, frequency: str) -> bool:
        """Check if should rebalance on this date."""
        if frequency == 'daily':
            return True
        elif frequency == 'weekly':
            return date.weekday() == 0  # Monday
        elif frequency == 'monthly':
            return date.day == 1
        return False

    def _rebalance(
        self,
        date: datetime,
        price_data: Dict[str, pd.DataFrame],
        signals: Dict[str, pd.DataFrame]
    ):
        """Rebalance portfolio based on signals."""
        target_weights = {}

        for ticker, signal_df in signals.items():
            if date in signal_df.index:
                target_weights[ticker] = signal_df.loc[date, 'weight']

        # Calculate target positions
        total_value = self._get_total_value(date, price_data)

        for ticker, weight in target_weights.items():
            if ticker not in price_data or date not in price_data[ticker].index:
                continue

            price = price_data[ticker].loc[date, 'Close']
            target_value = total_value * weight
            target_shares = int(target_value / price)

            current_shares = self.positions.get(ticker, 0)
            shares_to_trade = target_shares - current_shares

            if shares_to_trade != 0:
                self._execute_trade(ticker, shares_to_trade, price, date)

    def _execute_trade(self, ticker: str, shares: int, price: float, date: datetime):
        """Execute a trade."""
        # Apply slippage
        if shares > 0:
            execution_price = price * (1 + self.slippage)
        else:
            execution_price = price * (1 - self.slippage)

        trade_value = abs(shares) * execution_price
        commission_cost = trade_value * self.commission

        # Update cash
        if shares > 0:  # Buy
            total_cost = trade_value + commission_cost
            if total_cost <= self.cash:
                self.cash -= total_cost
                self.positions[ticker] = self.positions.get(ticker, 0) + shares
                self.trades.append({
                    'date': date,
                    'ticker': ticker,
                    'action': 'BUY',
                    'shares': shares,
                    'price': execution_price,
                    'commission': commission_cost
                })
        else:  # Sell
            self.cash += trade_value - commission_cost
            self.positions[ticker] = self.positions.get(ticker, 0) + shares
            if self.positions[ticker] == 0:
                del self.positions[ticker]
            self.trades.append({
                'date': date,
                'ticker': ticker,
                'action': 'SELL',
                'shares': abs(shares),
                'price': execution_price,
                'commission': commission_cost
            })

    def _get_total_value(self, date: datetime, price_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate total portfolio value."""
        total = self.cash

        for ticker, shares in self.positions.items():
            if ticker in price_data and date in price_data[ticker].index:
                price = price_data[ticker].loc[date, 'Close']
                total += shares * price

        return total

    def _update_portfolio_value(self, date: datetime, price_data: Dict[str, pd.DataFrame]):
        """Update portfolio value for date."""
        total_value = self._get_total_value(date, price_data)
        self.portfolio_value.append({
            'date': date,
            'value': total_value
        })

    def _calculate_results(self) -> Dict:
        """Calculate backtest performance metrics."""
        df = pd.DataFrame(self.portfolio_value)
        df.set_index('date', inplace=True)

        # Returns
        df['returns'] = df['value'].pct_change()

        # Cumulative returns
        cumulative_return = (df['value'].iloc[-1] / self.initial_capital) - 1

        # Annual metrics
        total_days = (df.index[-1] - df.index[0]).days
        annual_return = (1 + cumulative_return) ** (365 / total_days) - 1
        annual_volatility = df['returns'].std() * np.sqrt(252)

        # Sharpe ratio
        sharpe = annual_return / annual_volatility if annual_volatility > 0 else 0

        # Drawdown
        cumulative = df['value']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Win rate
        winning_trades = len([t for t in self.trades if t['action'] == 'SELL'])
        total_trades = len(self.trades) / 2  # Buy and sell pairs

        return {
            'initial_capital': self.initial_capital,
            'final_value': df['value'].iloc[-1],
            'cumulative_return': cumulative_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'total_trades': int(total_trades),
            'portfolio_history': df,
            'trades': self.trades
        }


class PerformanceAnalyzer:
    """Analyze backtest performance and generate reports."""

    @staticmethod
    def calculate_metrics(portfolio_history: pd.DataFrame) -> Dict:
        """Calculate comprehensive performance metrics."""
        returns = portfolio_history['returns'].dropna()

        # Basic metrics
        total_return = (portfolio_history['value'].iloc[-1] / portfolio_history['value'].iloc[0]) - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        annual_vol = returns.std() * np.sqrt(252)

        # Risk metrics
        sharpe = (annual_return - 0.02) / annual_vol if annual_vol > 0 else 0

        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252)
        sortino = (annual_return - 0.02) / downside_vol if downside_vol > 0 else 0

        # Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()

        # VaR and CVaR
        var_95 = returns.quantile(0.05)
        cvar_95 = returns[returns <= var_95].mean()

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_dd,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis()
        }

    @staticmethod
    def compare_to_benchmark(
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> Dict:
        """Compare portfolio to benchmark."""
        # Align dates
        aligned = pd.DataFrame({
            'portfolio': portfolio_returns,
            'benchmark': benchmark_returns
        }).dropna()

        # Excess returns
        excess_returns = aligned['portfolio'] - aligned['benchmark']

        # Information ratio
        tracking_error = excess_returns.std() * np.sqrt(252)
        info_ratio = (excess_returns.mean() * 252) / tracking_error if tracking_error > 0 else 0

        # Beta
        covariance = np.cov(aligned['portfolio'], aligned['benchmark'])[0][1]
        benchmark_variance = aligned['benchmark'].var()
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0

        # Alpha
        portfolio_return = aligned['portfolio'].mean() * 252
        benchmark_return = aligned['benchmark'].mean() * 252
        alpha = portfolio_return - (0.02 + beta * (benchmark_return - 0.02))

        return {
            'alpha': alpha,
            'beta': beta,
            'information_ratio': info_ratio,
            'tracking_error': tracking_error,
            'correlation': aligned.corr().iloc[0, 1]
        }
