"""
Logging utility for the portfolio manager.
Provides consistent logging across all modules.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(
    name: str = "portfolio_manager",
    level: str = "INFO",
    log_file: Optional[str] = None,
    console_output: bool = True
) -> logging.Logger:
    """
    Set up a logger with file and console handlers.

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file. If None, no file logging
        console_output: Whether to output to console

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers to avoid duplicates
    logger.handlers = []

    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class PerformanceLogger:
    """Logger for tracking model and portfolio performance."""

    def __init__(self, log_dir: str = "logs/performance"):
        """
        Initialize performance logger.

        Args:
            log_dir: Directory for performance logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger("performance", log_file=str(self.log_dir / "performance.log"))

    def log_prediction(self, model_name: str, ticker: str, prediction: float, actual: Optional[float] = None):
        """Log model prediction."""
        log_msg = f"Model: {model_name} | Ticker: {ticker} | Prediction: {prediction:.4f}"
        if actual is not None:
            error = abs(prediction - actual)
            log_msg += f" | Actual: {actual:.4f} | Error: {error:.4f}"
        self.logger.info(log_msg)

    def log_portfolio_allocation(self, allocations: dict, total_value: float):
        """Log portfolio allocation decisions."""
        self.logger.info(f"Portfolio Total Value: ${total_value:.2f}")
        for ticker, weight in allocations.items():
            self.logger.info(f"  {ticker}: {weight*100:.2f}%")

    def log_trade(self, action: str, ticker: str, quantity: float, price: float):
        """Log trade execution."""
        total = quantity * price
        self.logger.info(f"TRADE | {action} | {ticker} | Qty: {quantity} | Price: ${price:.2f} | Total: ${total:.2f}")

    def log_performance_metrics(self, metrics: dict):
        """Log performance metrics."""
        self.logger.info("=" * 60)
        self.logger.info("PERFORMANCE METRICS")
        for metric, value in metrics.items():
            if isinstance(value, float):
                self.logger.info(f"  {metric}: {value:.4f}")
            else:
                self.logger.info(f"  {metric}: {value}")
        self.logger.info("=" * 60)
