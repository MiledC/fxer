"""Unit tests for the AccountState module."""

from decimal import Decimal

import pytest

from fxer.rl.gym.account import AccountState
from fxer.rl.types import PositionState


class TestAccountState:
    """Test AccountState trading account management."""

    def test_initialization(self):
        """Verify account initializes with correct values."""
        account = AccountState(
            initial_balance=10_000.0,
            leverage=100.0,
            units_per_lot=100.0,
            commission_per_lot=7.0,
            margin_call_pct=50.0,
            lot_size=0.01,
        )

        assert account.balance == 10_000.0
        assert account.equity == 10_000.0
        assert account.position_state == PositionState.FLAT
        assert account.unrealized_pnl == 0.0
        assert account.unrealized_pnl_pct == 0.0
        assert account.bars_held == 0
        assert account.drawdown_pct == 0.0
        assert account.is_liquidated is False
        assert account.total_trades == 0
        assert account.margin_used == 0.0
        assert account.total_commission_paid == 0.0

    def test_open_long_position(self):
        """Verify opening a long position charges commission."""
        account = AccountState(
            initial_balance=10_000.0,
            leverage=100.0,
            units_per_lot=100.0,
            commission_per_lot=7.0,
            margin_call_pct=50.0,
            lot_size=0.01,
        )

        entry_price = Decimal("2050.00")
        account.open_position(PositionState.LONG, entry_price, step=0)

        # Half commission charged on open
        half_commission = 0.01 * 7.0 / 2
        assert account.balance == 10_000.0 - half_commission
        assert account.position_state == PositionState.LONG
        assert account.total_commission_paid == half_commission
        assert account.bars_held == 0  # Not incremented yet

    def test_open_short_position(self):
        """Verify opening a short position charges commission."""
        account = AccountState(
            initial_balance=10_000.0,
            leverage=100.0,
            units_per_lot=100.0,
            commission_per_lot=7.0,
            margin_call_pct=50.0,
            lot_size=0.1,
        )

        entry_price = Decimal("2050.00")
        account.open_position(PositionState.SHORT, entry_price, step=5)

        # Half commission charged on open
        half_commission = 0.1 * 7.0 / 2
        assert account.balance == 10_000.0 - half_commission
        assert account.position_state == PositionState.SHORT
        assert account.total_commission_paid == half_commission

    def test_close_long_position_profit(self):
        """Verify closing a long position with profit."""
        account = AccountState(
            initial_balance=10_000.0,
            leverage=100.0,
            units_per_lot=100.0,
            commission_per_lot=7.0,
            margin_call_pct=50.0,
            lot_size=0.01,
        )

        # Open long
        entry_price = Decimal("2050.00")
        account.open_position(PositionState.LONG, entry_price, step=0)

        # Close with profit
        exit_price = Decimal("2060.00")
        realized_pnl = account.close_position(exit_price)

        # Calculate expected P&L
        gross_pnl = float(exit_price - entry_price) * 0.01 * 100.0  # 10.0
        full_commission = 0.01 * 7.0  # 0.07
        net_pnl = gross_pnl - full_commission / 2  # Already paid half on open

        assert realized_pnl == pytest.approx(net_pnl, abs=1e-5)
        assert account.position_state == PositionState.FLAT
        assert account.total_trades == 1
        assert account.total_commission_paid == full_commission

        # Balance should be initial + gross_pnl - full_commission
        expected_balance = 10_000.0 + gross_pnl - full_commission
        assert account.balance == pytest.approx(expected_balance, abs=1e-5)

    def test_close_short_position_profit(self):
        """Verify closing a short position with profit."""
        account = AccountState(
            initial_balance=10_000.0,
            leverage=100.0,
            units_per_lot=100.0,
            commission_per_lot=7.0,
            margin_call_pct=50.0,
            lot_size=0.01,
        )

        # Open short
        entry_price = Decimal("2050.00")
        account.open_position(PositionState.SHORT, entry_price, step=0)

        # Close with profit (price went down)
        exit_price = Decimal("2040.00")
        realized_pnl = account.close_position(exit_price)

        # Calculate expected P&L
        gross_pnl = float(entry_price - exit_price) * 0.01 * 100.0  # 10.0
        full_commission = 0.01 * 7.0  # 0.07
        net_pnl = gross_pnl - full_commission / 2  # Already paid half on open

        assert realized_pnl == pytest.approx(net_pnl, abs=1e-5)
        assert account.position_state == PositionState.FLAT
        assert account.total_trades == 1

    def test_close_long_position_loss(self):
        """Verify closing a long position with loss."""
        account = AccountState(
            initial_balance=10_000.0,
            leverage=100.0,
            units_per_lot=100.0,
            commission_per_lot=7.0,
            margin_call_pct=50.0,
            lot_size=0.01,
        )

        # Open long
        entry_price = Decimal("2050.00")
        account.open_position(PositionState.LONG, entry_price, step=0)

        # Close with loss
        exit_price = Decimal("2040.00")
        realized_pnl = account.close_position(exit_price)

        # Calculate expected P&L
        gross_pnl = float(exit_price - entry_price) * 0.01 * 100.0  # -10.0
        full_commission = 0.01 * 7.0  # 0.07
        net_pnl = gross_pnl - full_commission / 2  # Already paid half on open

        assert realized_pnl == pytest.approx(net_pnl, abs=1e-5)
        assert realized_pnl < 0  # Loss

    def test_unrealized_pnl_long(self):
        """Verify unrealized P&L calculation for long position."""
        account = AccountState(
            initial_balance=10_000.0,
            leverage=100.0,
            units_per_lot=100.0,
            commission_per_lot=7.0,
            margin_call_pct=50.0,
            lot_size=0.01,
        )

        # Open long
        entry_price = Decimal("2050.00")
        account.open_position(PositionState.LONG, entry_price, step=0)

        # Update price (profit)
        current_price = Decimal("2055.00")
        account.update_price(current_price, step=1)

        expected_unrealized = float(current_price - entry_price) * 0.01 * 100.0
        assert account.unrealized_pnl == pytest.approx(expected_unrealized, abs=1e-5)

        # Update price (loss)
        current_price = Decimal("2045.00")
        account.update_price(current_price, step=2)

        expected_unrealized = float(current_price - entry_price) * 0.01 * 100.0
        assert account.unrealized_pnl == pytest.approx(expected_unrealized, abs=1e-5)

    def test_unrealized_pnl_short(self):
        """Verify unrealized P&L calculation for short position."""
        account = AccountState(
            initial_balance=10_000.0,
            leverage=100.0,
            units_per_lot=100.0,
            commission_per_lot=7.0,
            margin_call_pct=50.0,
            lot_size=0.01,
        )

        # Open short
        entry_price = Decimal("2050.00")
        account.open_position(PositionState.SHORT, entry_price, step=0)

        # Update price (profit - price went down)
        current_price = Decimal("2045.00")
        account.update_price(current_price, step=1)

        expected_unrealized = float(entry_price - current_price) * 0.01 * 100.0
        assert account.unrealized_pnl == pytest.approx(expected_unrealized, abs=1e-5)

        # Update price (loss - price went up)
        current_price = Decimal("2055.00")
        account.update_price(current_price, step=2)

        expected_unrealized = float(entry_price - current_price) * 0.01 * 100.0
        assert account.unrealized_pnl == pytest.approx(expected_unrealized, abs=1e-5)

    def test_unrealized_pnl_pct(self):
        """Verify unrealized P&L percentage calculation."""
        account = AccountState(
            initial_balance=10_000.0,
            leverage=100.0,
            units_per_lot=100.0,
            commission_per_lot=7.0,
            margin_call_pct=50.0,
            lot_size=0.01,
        )

        # Open long
        entry_price = Decimal("2000.00")
        account.open_position(PositionState.LONG, entry_price, step=0)

        # Update price (5% profit on position)
        current_price = Decimal("2100.00")
        account.update_price(current_price, step=1)

        # Unrealized P&L = 100 * 0.01 * 100 = 100
        # Entry notional = 2000 * 0.01 * 100 = 2000
        # Percentage = 100 / 2000 * 100 = 5%
        assert account.unrealized_pnl_pct == pytest.approx(5.0, abs=1e-5)

    def test_bars_held_tracking(self):
        """Verify bars held counter increments correctly."""
        account = AccountState(
            initial_balance=10_000.0,
            leverage=100.0,
            units_per_lot=100.0,
            commission_per_lot=7.0,
            margin_call_pct=50.0,
            lot_size=0.01,
        )

        assert account.bars_held == 0

        # Open position
        account.open_position(PositionState.LONG, Decimal("2050.00"), step=0)
        assert account.bars_held == 0  # Not incremented yet

        # Update price increments bars held
        account.update_price(Decimal("2051.00"), step=1)
        assert account.bars_held == 1

        account.update_price(Decimal("2052.00"), step=2)
        assert account.bars_held == 2

        account.update_price(Decimal("2053.00"), step=3)
        assert account.bars_held == 3

        # Close position resets bars held
        account.close_position(Decimal("2054.00"))
        assert account.bars_held == 0

    def test_drawdown_tracking(self):
        """Verify drawdown percentage tracking."""
        account = AccountState(
            initial_balance=10_000.0,
            leverage=100.0,
            units_per_lot=100.0,
            commission_per_lot=7.0,
            margin_call_pct=50.0,
            lot_size=0.1,
        )

        # Initially no drawdown
        assert account.drawdown_pct == 0.0

        # Open long position
        entry_price = Decimal("2000.00")
        account.open_position(PositionState.LONG, entry_price, step=0)

        # Price goes up - new peak
        account.update_price(Decimal("2100.00"), step=1)
        assert account.drawdown_pct == 0.0  # At peak

        # Price drops - drawdown from peak
        account.update_price(Decimal("2050.00"), step=2)

        # Unrealized P&L dropped from 1000 to 500
        # Peak equity was ~10999.65 (10000 - 0.35 commission + 1000 unrealized)
        # Current equity is ~10499.65 (10000 - 0.35 commission + 500 unrealized)
        # Drawdown = 500 / 10999.65 * 100 ≈ 4.55%
        assert account.drawdown_pct > 0
        assert account.drawdown_pct < 5.0

    def test_margin_used_calculation(self):
        """Verify margin requirement calculation."""
        account = AccountState(
            initial_balance=10_000.0,
            leverage=100.0,
            units_per_lot=100.0,
            commission_per_lot=7.0,
            margin_call_pct=50.0,
            lot_size=0.1,
        )

        # No position, no margin used
        assert account.margin_used == 0.0

        # Open position
        entry_price = Decimal("2000.00")
        account.open_position(PositionState.LONG, entry_price, step=0)

        # Margin = (price * lots * units_per_lot) / leverage
        # = (2000 * 0.1 * 100) / 100 = 200
        expected_margin = (2000.0 * 0.1 * 100.0) / 100.0
        assert account.margin_used == pytest.approx(expected_margin, abs=1e-5)

        # Update price changes margin requirement
        account.update_price(Decimal("2100.00"), step=1)
        expected_margin = (2100.0 * 0.1 * 100.0) / 100.0
        assert account.margin_used == pytest.approx(expected_margin, abs=1e-5)

    def test_liquidation_check(self):
        """Verify liquidation is triggered when equity falls below threshold."""
        account = AccountState(
            initial_balance=1000.0,  # Small balance for easier testing
            leverage=100.0,
            units_per_lot=100.0,
            commission_per_lot=7.0,
            margin_call_pct=50.0,
            lot_size=0.1,
        )

        # Open position
        entry_price = Decimal("2000.00")
        account.open_position(PositionState.LONG, entry_price, step=0)

        # Margin used = (2000 * 0.1 * 100) / 100 = 200
        # Liquidation threshold = 200 * 0.5 = 100

        # Price drops significantly
        # Need equity to drop below 100
        # Starting balance ≈ 999.65 (after commission)
        # Need unrealized loss > 899.65
        # Loss per dollar = 0.1 * 100 = 10
        # Need price drop > 89.965 ≈ 90

        account.update_price(Decimal("1900.00"), step=1)

        # Check if liquidated (equity should be around 999.65 - 1000 = -0.35)
        # Actually equity = 999.65 + (-100 * 0.1 * 100) = 999.65 - 1000 = -0.35
        # This is below threshold of 100
        assert account.is_liquidated is True

    def test_liquidation_not_triggered_above_threshold(self):
        """Verify liquidation is not triggered when equity is above threshold."""
        account = AccountState(
            initial_balance=10_000.0,
            leverage=100.0,
            units_per_lot=100.0,
            commission_per_lot=7.0,
            margin_call_pct=50.0,
            lot_size=0.01,
        )

        # Open position
        entry_price = Decimal("2000.00")
        account.open_position(PositionState.LONG, entry_price, step=0)

        # Small price drop
        account.update_price(Decimal("1990.00"), step=1)

        # Unrealized loss = -10 * 0.01 * 100 = -10
        # Equity ≈ 9999.965 - 10 = 9989.965
        # Margin = 1990 * 0.01 * 100 / 100 = 19.9
        # Threshold = 19.9 * 0.5 = 9.95
        # Equity is well above threshold
        assert account.is_liquidated is False

    def test_reset(self):
        """Verify reset restores initial state."""
        account = AccountState(
            initial_balance=10_000.0,
            leverage=100.0,
            units_per_lot=100.0,
            commission_per_lot=7.0,
            margin_call_pct=50.0,
            lot_size=0.01,
        )

        # Open position and make some trades
        account.open_position(PositionState.LONG, Decimal("2050.00"), step=0)
        account.update_price(Decimal("2060.00"), step=1)
        account.close_position(Decimal("2060.00"))
        account.open_position(PositionState.SHORT, Decimal("2060.00"), step=2)

        # Reset
        account.reset()

        # Check all values reset
        assert account.balance == 10_000.0
        assert account.equity == 10_000.0
        assert account.position_state == PositionState.FLAT
        assert account.unrealized_pnl == 0.0
        assert account.bars_held == 0
        assert account.drawdown_pct == 0.0
        assert account.is_liquidated is False
        assert account.total_trades == 0
        assert account.margin_used == 0.0
        assert account.total_commission_paid == 0.0

    def test_error_open_position_when_already_open(self):
        """Verify error when trying to open position with existing position."""
        account = AccountState(
            initial_balance=10_000.0,
            leverage=100.0,
            units_per_lot=100.0,
            commission_per_lot=7.0,
            margin_call_pct=50.0,
            lot_size=0.01,
        )

        # Open first position
        account.open_position(PositionState.LONG, Decimal("2050.00"), step=0)

        # Try to open another
        with pytest.raises(ValueError, match="Cannot open position: already have open position"):
            account.open_position(PositionState.SHORT, Decimal("2051.00"), step=1)

    def test_error_open_position_invalid_state(self):
        """Verify error when trying to open position with FLAT state."""
        account = AccountState(
            initial_balance=10_000.0,
            leverage=100.0,
            units_per_lot=100.0,
            commission_per_lot=7.0,
            margin_call_pct=50.0,
            lot_size=0.01,
        )

        with pytest.raises(ValueError, match="Invalid position state for opening"):
            account.open_position(PositionState.FLAT, Decimal("2050.00"), step=0)

    def test_error_open_position_invalid_price(self):
        """Verify error when trying to open position with invalid price."""
        account = AccountState(
            initial_balance=10_000.0,
            leverage=100.0,
            units_per_lot=100.0,
            commission_per_lot=7.0,
            margin_call_pct=50.0,
            lot_size=0.01,
        )

        # Zero price
        with pytest.raises(ValueError, match="Invalid entry price"):
            account.open_position(PositionState.LONG, Decimal("0.00"), step=0)

        # Negative price
        with pytest.raises(ValueError, match="Invalid entry price"):
            account.open_position(PositionState.LONG, Decimal("-100.00"), step=0)

    def test_error_close_position_when_flat(self):
        """Verify error when trying to close non-existent position."""
        account = AccountState(
            initial_balance=10_000.0,
            leverage=100.0,
            units_per_lot=100.0,
            commission_per_lot=7.0,
            margin_call_pct=50.0,
            lot_size=0.01,
        )

        with pytest.raises(ValueError, match="Cannot close position: no open position"):
            account.close_position(Decimal("2050.00"))

    def test_multiple_trades_tracking(self):
        """Verify multiple trades are tracked correctly."""
        account = AccountState(
            initial_balance=10_000.0,
            leverage=100.0,
            units_per_lot=100.0,
            commission_per_lot=7.0,
            margin_call_pct=50.0,
            lot_size=0.01,
        )

        # Trade 1: Long
        account.open_position(PositionState.LONG, Decimal("2050.00"), step=0)
        account.close_position(Decimal("2055.00"))
        assert account.total_trades == 1

        # Trade 2: Short
        account.open_position(PositionState.SHORT, Decimal("2055.00"), step=5)
        account.close_position(Decimal("2050.00"))
        assert account.total_trades == 2

        # Trade 3: Long
        account.open_position(PositionState.LONG, Decimal("2050.00"), step=10)
        account.close_position(Decimal("2060.00"))
        assert account.total_trades == 3

        # Commission should accumulate
        expected_commission = 3 * 0.01 * 7.0
        assert account.total_commission_paid == pytest.approx(expected_commission, abs=1e-5)

    def test_equity_calculation_with_position(self):
        """Verify equity = balance + unrealized P&L."""
        account = AccountState(
            initial_balance=10_000.0,
            leverage=100.0,
            units_per_lot=100.0,
            commission_per_lot=7.0,
            margin_call_pct=50.0,
            lot_size=0.1,
        )

        # Open position
        account.open_position(PositionState.LONG, Decimal("2000.00"), step=0)

        # Balance reduced by half commission
        half_commission = 0.1 * 7.0 / 2
        assert account.balance == pytest.approx(10_000.0 - half_commission, abs=1e-5)

        # Update price for unrealized profit
        account.update_price(Decimal("2050.00"), step=1)

        # Unrealized P&L = 50 * 0.1 * 100 = 500
        assert account.unrealized_pnl == pytest.approx(500.0, abs=1e-5)

        # Equity = balance + unrealized
        expected_equity = (10_000.0 - half_commission) + 500.0
        assert account.equity == pytest.approx(expected_equity, abs=1e-5)