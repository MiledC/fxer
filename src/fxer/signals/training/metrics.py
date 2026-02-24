"""Trading performance metrics for signal evaluation."""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SignalMetrics:
    """Comprehensive trading metrics for signal evaluation.

    Targets from project.md:
        Sharpe >= 1.0 (target 1.5+)
        Sortino >= 1.5 (target 2.0+)
        Profit Factor >= 1.3 (target 1.5+)
        Win Rate >= 40% (target 50%+)
        Max Drawdown <= 20% (target <15%)
        Trade Count >= 100 (target 200+)
    """

    sharpe_ratio: float
    sortino_ratio: float
    profit_factor: float
    win_rate: float
    max_drawdown: float
    trade_count: int
    total_return: float
    mean_return: float
    std_return: float
    accuracy: float  # Classification accuracy

    def meets_minimum(self) -> bool:
        """Check if metrics meet minimum thresholds from project.md."""
        return (
            self.sharpe_ratio >= 1.0
            and self.sortino_ratio >= 1.5
            and self.profit_factor >= 1.3
            and self.win_rate >= 0.40
            and self.max_drawdown <= 0.20
            and self.trade_count >= 100
        )

    def meets_target(self) -> bool:
        """Check if metrics meet target thresholds from project.md."""
        return (
            self.sharpe_ratio >= 1.5
            and self.sortino_ratio >= 2.0
            and self.profit_factor >= 1.5
            and self.win_rate >= 0.50
            and self.max_drawdown <= 0.15
            and self.trade_count >= 200
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "profit_factor": self.profit_factor,
            "win_rate": self.win_rate,
            "max_drawdown": self.max_drawdown,
            "trade_count": self.trade_count,
            "total_return": self.total_return,
            "mean_return": self.mean_return,
            "std_return": self.std_return,
            "accuracy": self.accuracy,
        }


def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    returns: np.ndarray,
    periods_per_year: float = 252 * 78,  # 5m bars per trading year
    horizon_bars: int = 1,
) -> SignalMetrics:
    """Compute trading metrics from predictions, labels, and returns.

    Args:
        predictions: Model predictions (1 = long, 0 = short).
        labels: True labels (1 = long, 0 = short).
        returns: Per-period returns corresponding to each prediction.
        periods_per_year: Number of bars per trading year for annualization.
            Default assumes 5-minute bars (252 days * 78 bars/day).
        horizon_bars: Number of bars per holding period. When > 1, each
            return spans multiple bars so ``periods_per_year`` is adjusted
            to ``periods_per_year / horizon_bars`` for correct annualization.

    Returns:
        SignalMetrics with all computed metrics.
    """
    if horizon_bars > 1:
        periods_per_year = periods_per_year / horizon_bars
    if len(predictions) == 0 or len(labels) == 0:
        return SignalMetrics(
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            profit_factor=0.0,
            win_rate=0.0,
            max_drawdown=0.0,
            trade_count=0,
            total_return=0.0,
            mean_return=0.0,
            std_return=0.0,
            accuracy=0.0,
        )

    # Strategy returns: go long when predicted long, short when predicted short
    positions = np.where(predictions == 1, 1.0, -1.0)
    strategy_returns = positions * returns

    trade_count = len(strategy_returns)
    winning_trades = strategy_returns[strategy_returns > 0]
    losing_trades = strategy_returns[strategy_returns < 0]

    # Win rate
    win_rate = len(winning_trades) / trade_count if trade_count > 0 else 0.0

    # Sharpe ratio (annualized)
    mean_ret = float(np.mean(strategy_returns))
    std_ret = float(np.std(strategy_returns, ddof=1)) if trade_count > 1 else 1.0
    sharpe = (mean_ret / std_ret * np.sqrt(periods_per_year)) if std_ret > 0 else 0.0

    # Sortino ratio (annualized, using downside deviation)
    downside = strategy_returns[strategy_returns < 0]
    downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 1.0
    sortino = (mean_ret / downside_std * np.sqrt(periods_per_year)) if downside_std > 0 else 0.0

    # Profit factor (gross gains / gross losses)
    gross_gains = float(np.sum(winning_trades)) if len(winning_trades) > 0 else 0.0
    gross_losses = float(np.abs(np.sum(losing_trades))) if len(losing_trades) > 0 else 1.0
    profit_factor = gross_gains / gross_losses if gross_losses > 0 else float("inf")

    # Max drawdown
    cumulative = np.cumsum(strategy_returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    # Classification accuracy
    accuracy = float(np.mean(predictions == labels))

    total_return = float(np.sum(strategy_returns))

    return SignalMetrics(
        sharpe_ratio=float(sharpe),
        sortino_ratio=float(sortino),
        profit_factor=float(profit_factor),
        win_rate=float(win_rate),
        max_drawdown=float(max_dd),
        trade_count=trade_count,
        total_return=total_return,
        mean_return=mean_ret,
        std_return=std_ret,
        accuracy=accuracy,
    )
