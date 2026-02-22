"""Backtest data types for the fxEr trading system."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from fxer.regime.types import RegimeState
from fxer.signals.types import Direction


@dataclass(frozen=True, slots=True)
class BacktestTrade:
    """Record of a completed or open trade during backtesting.

    Attributes:
        symbol: Trading symbol.
        direction: Trade direction (LONG/SHORT).
        entry_timestamp: When the trade was opened.
        entry_price: Opening price.
        exit_timestamp: When the trade was closed (None if still open).
        exit_price: Closing price (None if still open).
        spread_pct: Spread cost as percentage of entry price.
        predicted_horizon: Expected holding period in bars.
        bars_held: Actual bars held.
        pnl_pct: P&L percentage after spread (None if still open).
        regime_state: Market regime when trade was opened.
    """

    symbol: str
    direction: Direction
    entry_timestamp: datetime
    entry_price: Decimal
    exit_timestamp: datetime | None
    exit_price: Decimal | None
    spread_pct: float
    predicted_horizon: int
    bars_held: int
    pnl_pct: float | None
    regime_state: RegimeState | None


@dataclass(frozen=True, slots=True)
class RegimeMetrics:
    """Performance metrics for a specific regime state.

    Attributes:
        trade_count: Number of trades in this regime.
        win_rate: Percentage of profitable trades.
        sharpe_ratio: Risk-adjusted return metric.
        total_return: Compound return for this regime.
    """

    trade_count: int
    win_rate: float
    sharpe_ratio: float
    total_return: float


@dataclass(frozen=True, slots=True)
class TradeMetrics:
    """Comprehensive performance metrics for a set of trades.

    Attributes:
        total_return: Compound return across all trades.
        sharpe_ratio: Annualized risk-adjusted return.
        sortino_ratio: Downside risk-adjusted return.
        profit_factor: Ratio of wins to losses.
        win_rate: Percentage of profitable trades.
        max_drawdown: Maximum peak-to-trough decline.
        trade_count: Total number of trades.
        avg_win: Average return of winning trades.
        avg_loss: Average return of losing trades.
        best_trade: Best single trade return.
        worst_trade: Worst single trade return.
    """

    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    profit_factor: float
    win_rate: float
    max_drawdown: float
    trade_count: int
    avg_win: float
    avg_loss: float
    best_trade: float
    worst_trade: float


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Complete backtest results and metadata.

    Attributes:
        symbol: Trading symbol tested.
        timeframe: Bar timeframe (e.g. "5m").
        start_date: First bar timestamp.
        end_date: Last bar timestamp.
        bars_processed: Total bars processed.
        signals_generated: Number of signals generated.
        trades_executed: Number of trades completed.
        spread_pct: Spread cost used.
        total_return: Compound return.
        sharpe_ratio: Risk-adjusted return.
        sortino_ratio: Downside risk-adjusted return.
        profit_factor: Ratio of wins to losses.
        win_rate: Percentage of profitable trades.
        max_drawdown: Maximum drawdown.
        meets_minimum: Whether results meet minimum criteria.
        regime_breakdown: Performance by regime state.
    """

    symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    bars_processed: int
    signals_generated: int
    trades_executed: int
    spread_pct: float
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    profit_factor: float
    win_rate: float
    max_drawdown: float
    meets_minimum: bool
    regime_breakdown: dict[str, RegimeMetrics]