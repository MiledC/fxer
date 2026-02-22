"""Trade position tracking for backtesting."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from fxer.backtesting.types import BacktestTrade
from fxer.regime.types import RegimeState
from fxer.signals.types import Direction


@dataclass
class _OpenPosition:
    """Mutable internal state for an open position."""

    symbol: str
    direction: Direction
    entry_price: Decimal
    entry_timestamp: datetime
    regime_state: RegimeState | None
    bars_since_entry: int = 0


class TradeTracker:
    """Manages position state and P&L tracking during backtesting.

    Tracks one position at a time, computing P&L with spread costs.
    Spread is specified in price units and converted to percentage at trade time.
    """

    def __init__(self, spread: float = 0.30, predicted_horizon: int = 12) -> None:
        """Initialize the trade tracker.

        Args:
            spread: Spread cost in price units (e.g. 0.30 USD for XAUUSD).
            predicted_horizon: Expected holding period in bars.
        """
        self._spread = spread  # In price units
        self._predicted_horizon = predicted_horizon
        self._open_trade: _OpenPosition | None = None
        self._closed_trades: list[BacktestTrade] = []

    def has_open_position(self) -> bool:
        """Check if there's an open position."""
        return self._open_trade is not None

    def open_trade(
        self,
        symbol: str,
        direction: Direction,
        entry_price: Decimal,
        entry_timestamp: datetime,
        regime_state: RegimeState | None = None,
    ) -> None:
        """Open a new position.

        Args:
            symbol: Trading symbol.
            direction: LONG or SHORT.
            entry_price: Entry price (typically next bar's open).
            entry_timestamp: Entry timestamp.
            regime_state: Current regime state.

        Raises:
            ValueError: If a position is already open.
        """
        if self._open_trade is not None:
            raise ValueError("Cannot open trade: position already open")

        if entry_price <= 0:
            raise ValueError(f"Invalid entry price: {entry_price}")

        self._open_trade = _OpenPosition(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            entry_timestamp=entry_timestamp,
            regime_state=regime_state,
            bars_since_entry=0,
        )

    def update_bar(self) -> None:
        """Increment bars counter for open position."""
        if self._open_trade is not None:
            self._open_trade.bars_since_entry += 1

    def should_exit(self) -> bool:
        """Check if position has reached predicted horizon."""
        if self._open_trade is None:
            return False
        return self._open_trade.bars_since_entry >= self._predicted_horizon

    def close_trade(
        self,
        exit_price: Decimal,
        exit_timestamp: datetime,
    ) -> BacktestTrade:
        """Close the current position and compute P&L.

        Args:
            exit_price: Exit price (typically bar's close).
            exit_timestamp: Exit timestamp.

        Returns:
            Frozen BacktestTrade with computed P&L.

        Raises:
            ValueError: If no position is open.
        """
        if self._open_trade is None:
            raise ValueError("Cannot close trade: no position open")

        # Compute spread percentage based on entry price
        entry_price_float = float(self._open_trade.entry_price)
        spread_pct = self._spread / entry_price_float if entry_price_float > 0 else 0.0

        # Compute P&L based on direction
        exit_price_float = float(exit_price)

        if self._open_trade.direction == Direction.LONG:
            # LONG: (exit - entry) / entry - spread
            raw_pnl = (exit_price_float - entry_price_float) / entry_price_float
            pnl_pct = (raw_pnl - spread_pct) * 100
        elif self._open_trade.direction == Direction.SHORT:
            # SHORT: (entry - exit) / entry - spread
            raw_pnl = (entry_price_float - exit_price_float) / entry_price_float
            pnl_pct = (raw_pnl - spread_pct) * 100
        else:
            # NEUTRAL shouldn't happen
            pnl_pct = -spread_pct * 100

        # Create frozen trade record
        trade = BacktestTrade(
            symbol=self._open_trade.symbol,
            direction=self._open_trade.direction,
            entry_timestamp=self._open_trade.entry_timestamp,
            entry_price=self._open_trade.entry_price,
            exit_timestamp=exit_timestamp,
            exit_price=exit_price,
            spread_pct=spread_pct,
            predicted_horizon=self._predicted_horizon,
            bars_held=self._open_trade.bars_since_entry,
            pnl_pct=pnl_pct,
            regime_state=self._open_trade.regime_state,
        )

        # Clear open position and record closed trade
        self._open_trade = None
        self._closed_trades.append(trade)

        return trade

    def force_close_all(
        self,
        exit_price: Decimal,
        exit_timestamp: datetime,
    ) -> BacktestTrade | None:
        """Force close any open position.

        Used at end of backtest or before weekends.

        Args:
            exit_price: Exit price.
            exit_timestamp: Exit timestamp.

        Returns:
            The closed trade if one was open, None otherwise.
        """
        if self._open_trade is None:
            return None
        return self.close_trade(exit_price, exit_timestamp)

    def get_closed_trades(self) -> list[BacktestTrade]:
        """Get all closed trades."""
        return self._closed_trades.copy()