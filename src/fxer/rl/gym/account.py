"""Account state management with lot-based position tracking for RL environment."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from fxer.rl.types import PositionState


@dataclass
class _Position:
    """Mutable internal state for an open position."""

    state: PositionState  # LONG or SHORT
    entry_price: Decimal
    lots: float
    entry_step: int
    bars_held: int = 0


class AccountState:
    """Manages account balance, equity, margin, and position state.

    Tracks lot-based positions with commission model (half on open, half on close).
    Monitors margin requirements and liquidation conditions.
    Mirrors TradeTracker pattern but adapted for lot-based sizing with commission.
    """

    def __init__(
        self,
        initial_balance: float,
        leverage: float,
        units_per_lot: float,
        commission_per_lot: float,
        margin_call_pct: float,
        lot_size: float,
    ) -> None:
        """Initialize account state with trading parameters.

        Args:
            initial_balance: Starting account balance in USD.
            leverage: Maximum leverage ratio (e.g., 30 for 30:1).
            units_per_lot: Units per lot (e.g., 100 for gold).
            commission_per_lot: Round-trip commission per lot (e.g., 7.0 USD).
            margin_call_pct: Margin call threshold percentage (e.g., 50.0).
            lot_size: Fixed lot size to use for all trades.
        """
        self._initial_balance = initial_balance
        self._balance = initial_balance
        self._leverage = leverage
        self._units_per_lot = units_per_lot
        self._commission_per_lot = commission_per_lot
        self._margin_call_pct = margin_call_pct
        self._lot_size = lot_size

        # Position tracking
        self._position: _Position | None = None

        # Account metrics
        self._peak_equity = initial_balance
        self._total_trades = 0
        self._total_commission_paid = 0.0
        self._is_liquidated = False

    @property
    def balance(self) -> float:
        """Current account balance (realized P&L)."""
        return self._balance

    @property
    def equity(self) -> float:
        """Current equity (balance + unrealized P&L)."""
        return self._balance + self.unrealized_pnl

    @property
    def position_state(self) -> PositionState:
        """Current position state (FLAT, LONG, or SHORT)."""
        if self._position is None:
            return PositionState.FLAT
        return self._position.state

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized P&L in USD."""
        if self._position is None:
            return 0.0

        # Get current price from last update
        if not hasattr(self, '_current_price'):
            return 0.0

        return self._calculate_pnl(
            self._position.entry_price,
            self._current_price,
            self._position.state == PositionState.LONG,
            self._position.lots
        )

    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized P&L as percentage of entry notional."""
        if self._position is None:
            return 0.0

        entry_notional = float(self._position.entry_price) * self._position.lots * self._units_per_lot
        if entry_notional == 0:
            return 0.0

        return (self.unrealized_pnl / entry_notional) * 100.0

    @property
    def bars_held(self) -> int:
        """Number of bars position has been held."""
        if self._position is None:
            return 0
        return self._position.bars_held

    @property
    def drawdown_pct(self) -> float:
        """Current drawdown percentage from peak equity."""
        if self._peak_equity == 0:
            return 0.0
        return ((self._peak_equity - self.equity) / self._peak_equity) * 100.0

    @property
    def is_liquidated(self) -> bool:
        """Whether account has been liquidated."""
        return self._is_liquidated

    @property
    def total_trades(self) -> int:
        """Total number of completed trades."""
        return self._total_trades

    @property
    def margin_used(self) -> float:
        """Current margin requirement for open position."""
        if self._position is None:
            return 0.0

        if not hasattr(self, '_current_price'):
            # Use entry price if current price not set
            position_value = float(self._position.entry_price) * self._position.lots * self._units_per_lot
        else:
            position_value = float(self._current_price) * self._position.lots * self._units_per_lot

        return position_value / self._leverage

    @property
    def total_commission_paid(self) -> float:
        """Total commission paid across all trades."""
        return self._total_commission_paid

    def open_position(
        self,
        state: PositionState,
        price: Decimal,
        step: int
    ) -> None:
        """Open a new position with commission charge.

        Args:
            state: Position state (LONG or SHORT).
            price: Entry price.
            step: Current environment step.

        Raises:
            ValueError: If position already open or invalid state.
        """
        if self._position is not None:
            raise ValueError("Cannot open position: already have open position")

        if state not in (PositionState.LONG, PositionState.SHORT):
            raise ValueError(f"Invalid position state for opening: {state}")

        if price <= 0:
            raise ValueError(f"Invalid entry price: {price}")

        # Charge half commission on open
        half_commission = self._lot_size * self._commission_per_lot / 2
        self._balance -= half_commission
        self._total_commission_paid += half_commission

        # Create position
        self._position = _Position(
            state=state,
            entry_price=price,
            lots=self._lot_size,
            entry_step=step,
            bars_held=0
        )

        # Store current price for unrealized P&L calculation
        self._current_price = price

    def close_position(self, price: Decimal) -> float:
        """Close current position and realize P&L.

        Args:
            price: Exit price.

        Returns:
            Realized P&L in USD (after commission).

        Raises:
            ValueError: If no position to close.
        """
        if self._position is None:
            raise ValueError("Cannot close position: no open position")

        # Calculate gross P&L
        gross_pnl = self._calculate_pnl(
            self._position.entry_price,
            price,
            self._position.state == PositionState.LONG,
            self._position.lots
        )

        # Charge half commission on close
        half_commission = self._position.lots * self._commission_per_lot / 2
        self._total_commission_paid += half_commission

        # Net P&L after commission
        net_pnl = gross_pnl - half_commission

        # Update balance
        self._balance += gross_pnl - half_commission

        # Update trade counter
        self._total_trades += 1

        # Clear position
        self._position = None
        if hasattr(self, '_current_price'):
            delattr(self, '_current_price')

        return net_pnl

    def update_price(self, price: Decimal, step: int) -> None:
        """Update current price and check liquidation.

        Args:
            price: Current market price.
            step: Current environment step.
        """
        # Store current price
        self._current_price = price

        if self._position is not None:
            # Increment bars held
            self._position.bars_held += 1

            # Update peak equity
            current_equity = self.equity
            if current_equity > self._peak_equity:
                self._peak_equity = current_equity

            # Check liquidation condition
            margin_required = self.margin_used
            if margin_required > 0:
                liquidation_threshold = margin_required * (self._margin_call_pct / 100.0)
                if current_equity <= liquidation_threshold:
                    self._is_liquidated = True

    def reset(self) -> None:
        """Reset account to initial state."""
        self._balance = self._initial_balance
        self._position = None
        self._peak_equity = self._initial_balance
        self._total_trades = 0
        self._total_commission_paid = 0.0
        self._is_liquidated = False
        if hasattr(self, '_current_price'):
            delattr(self, '_current_price')

    def _calculate_pnl(
        self,
        entry: Decimal,
        exit_price: Decimal,
        is_long: bool,
        lots: float
    ) -> float:
        """Calculate P&L using Decimal arithmetic.

        Args:
            entry: Entry price.
            exit_price: Exit price.
            is_long: Whether position is long.
            lots: Number of lots.

        Returns:
            P&L in USD.
        """
        if is_long:
            pnl_per_unit = exit_price - entry
        else:
            pnl_per_unit = entry - exit_price

        return float(pnl_per_unit) * lots * self._units_per_lot