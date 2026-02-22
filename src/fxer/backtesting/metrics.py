"""Trade metrics computation for backtesting."""

from __future__ import annotations

import numpy as np

from fxer.backtesting.types import BacktestTrade, RegimeMetrics, TradeMetrics


def compute_trade_metrics(
    trades: list[BacktestTrade],
    periods_per_year: float = 252 * 6.5,  # ~1638 trading hours/year
) -> TradeMetrics:
    """Compute comprehensive performance metrics from completed trades.

    Args:
        trades: List of completed BacktestTrade objects.
        periods_per_year: Annualization factor for Sharpe/Sortino.

    Returns:
        TradeMetrics with all performance indicators.
    """
    if not trades:
        # Return zeros for empty trade list
        return TradeMetrics(
            total_return=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            profit_factor=0.0,
            win_rate=0.0,
            max_drawdown=0.0,
            trade_count=0,
            avg_win=0.0,
            avg_loss=0.0,
            best_trade=0.0,
            worst_trade=0.0,
        )

    # Extract P&L percentages from closed trades
    pnl_list = []
    bars_held_list = []
    for trade in trades:
        if trade.pnl_pct is not None:  # Only closed trades
            pnl_list.append(trade.pnl_pct)
            bars_held_list.append(trade.bars_held)

    if not pnl_list:
        # No closed trades
        return TradeMetrics(
            total_return=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            profit_factor=0.0,
            win_rate=0.0,
            max_drawdown=0.0,
            trade_count=len(trades),
            avg_win=0.0,
            avg_loss=0.0,
            best_trade=0.0,
            worst_trade=0.0,
        )

    pnl_array = np.array(pnl_list)

    # Total return: compound returns
    total_return = float(np.prod(1 + pnl_array / 100) - 1) * 100

    # Win rate
    wins = pnl_array > 0
    win_rate = float(np.mean(wins)) * 100

    # Average win/loss
    winning_trades = pnl_array[wins]
    losing_trades = pnl_array[~wins]

    avg_win = float(np.mean(winning_trades)) if len(winning_trades) > 0 else 0.0
    avg_loss = float(np.mean(losing_trades)) if len(losing_trades) > 0 else 0.0

    # Best/worst trade
    best_trade = float(np.max(pnl_array)) if len(pnl_array) > 0 else 0.0
    worst_trade = float(np.min(pnl_array)) if len(pnl_array) > 0 else 0.0

    # Profit factor
    if len(losing_trades) > 0:
        total_wins = np.sum(winning_trades)
        total_losses = np.abs(np.sum(losing_trades))
        if total_losses > 0:
            profit_factor = float(total_wins / total_losses)
            # Cap at 99.99 to avoid inf
            profit_factor = min(profit_factor, 99.99)
        else:
            profit_factor = 99.99
    else:
        profit_factor = 99.99 if len(winning_trades) > 0 else 0.0

    # Sharpe ratio (annualized)
    if len(pnl_array) > 1:
        mean_return = np.mean(pnl_array)
        std_return = np.std(pnl_array, ddof=1)
        avg_bars_per_trade = np.mean(bars_held_list)

        if std_return > 0 and avg_bars_per_trade > 0:
            # Annualization factor: trades_per_year = periods_per_year / avg_bars_per_trade
            annualization_factor = np.sqrt(periods_per_year / avg_bars_per_trade)
            sharpe_ratio = float(mean_return / std_return * annualization_factor)
        else:
            sharpe_ratio = 0.0
    else:
        sharpe_ratio = 0.0

    # Sortino ratio (using downside deviation)
    if len(pnl_array) > 1:
        mean_return = np.mean(pnl_array)
        downside_returns = pnl_array[pnl_array < 0]

        if len(downside_returns) > 0:
            downside_std = np.std(downside_returns, ddof=1)
            avg_bars_per_trade = np.mean(bars_held_list)

            if downside_std > 0 and avg_bars_per_trade > 0:
                annualization_factor = np.sqrt(periods_per_year / avg_bars_per_trade)
                sortino_ratio = float(mean_return / downside_std * annualization_factor)
            else:
                sortino_ratio = 0.0
        else:
            # No losing trades
            sortino_ratio = float(sharpe_ratio * 2.0) if sharpe_ratio > 0 else 0.0
    else:
        sortino_ratio = 0.0

    # Maximum drawdown
    max_drawdown = _calculate_max_drawdown(pnl_array)

    return TradeMetrics(
        total_return=round(total_return, 2),
        sharpe_ratio=round(sharpe_ratio, 2),
        sortino_ratio=round(sortino_ratio, 2),
        profit_factor=round(profit_factor, 2),
        win_rate=round(win_rate, 1),
        max_drawdown=round(max_drawdown, 1),
        trade_count=len(trades),
        avg_win=round(avg_win, 2),
        avg_loss=round(avg_loss, 2),
        best_trade=round(best_trade, 2),
        worst_trade=round(worst_trade, 2),
    )


def compute_regime_breakdown(
    trades: list[BacktestTrade],
    periods_per_year: float = 252 * 6.5,
) -> dict[str, RegimeMetrics]:
    """Compute performance metrics grouped by regime state.

    Args:
        trades: List of completed BacktestTrade objects.
        periods_per_year: Annualization factor for Sharpe.

    Returns:
        Dictionary mapping regime state names to their metrics.
    """
    breakdown: dict[str, RegimeMetrics] = {}

    # Group trades by regime
    regime_trades: dict[str, list[BacktestTrade]] = {}
    for trade in trades:
        if trade.regime_state is not None and trade.pnl_pct is not None:
            regime_key = trade.regime_state.value
            if regime_key not in regime_trades:
                regime_trades[regime_key] = []
            regime_trades[regime_key].append(trade)

    # Compute metrics per regime
    for regime_name, regime_trade_list in regime_trades.items():
        if not regime_trade_list:
            continue

        # Extract P&L and bars held
        pnl_list = []
        bars_held_list = []
        for trade in regime_trade_list:
            if trade.pnl_pct is not None:
                pnl_list.append(trade.pnl_pct)
                bars_held_list.append(trade.bars_held)

        if not pnl_list:
            continue

        pnl_array = np.array(pnl_list)

        # Total return
        total_return = float(np.prod(1 + pnl_array / 100) - 1) * 100

        # Win rate
        win_rate = float(np.mean(pnl_array > 0)) * 100

        # Sharpe ratio
        if len(pnl_array) > 1:
            mean_return = np.mean(pnl_array)
            std_return = np.std(pnl_array, ddof=1)
            avg_bars_per_trade = np.mean(bars_held_list)

            if std_return > 0 and avg_bars_per_trade > 0:
                annualization_factor = np.sqrt(periods_per_year / avg_bars_per_trade)
                sharpe_ratio = float(mean_return / std_return * annualization_factor)
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0

        breakdown[regime_name] = RegimeMetrics(
            trade_count=len(regime_trade_list),
            win_rate=round(win_rate, 1),
            sharpe_ratio=round(sharpe_ratio, 2),
            total_return=round(total_return, 2),
        )

    return breakdown


def _calculate_max_drawdown(pnl_array: np.ndarray) -> float:
    """Calculate maximum drawdown from P&L percentages.

    Args:
        pnl_array: Array of P&L percentages.

    Returns:
        Maximum drawdown as percentage.
    """
    if len(pnl_array) == 0:
        return 0.0

    # Build cumulative equity curve
    equity_curve = np.cumprod(1 + pnl_array / 100)

    # Calculate running maximum
    running_max = np.maximum.accumulate(equity_curve)

    # Calculate drawdown at each point
    drawdown = (equity_curve - running_max) / running_max * 100

    # Find maximum drawdown
    max_dd = float(np.min(drawdown))

    return abs(max_dd)