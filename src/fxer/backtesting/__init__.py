"""Backtesting module for the fxEr trading system."""

from fxer.backtesting.engine import BacktestEngine
from fxer.backtesting.metrics import compute_trade_metrics
from fxer.backtesting.tracker import TradeTracker
from fxer.backtesting.types import BacktestResult, BacktestTrade, RegimeMetrics, TradeMetrics

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "BacktestTrade",
    "RegimeMetrics",
    "TradeMetrics",
    "TradeTracker",
    "compute_trade_metrics",
]