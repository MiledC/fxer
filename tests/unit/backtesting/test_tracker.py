"""Unit tests for the TradeTracker module."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from fxer.backtesting.tracker import TradeTracker
from fxer.backtesting.types import BacktestTrade
from fxer.regime.types import RegimeState
from fxer.signals.types import Direction


class TestTradeTracker:
    """Test TradeTracker behavior."""

    def test_open_trade_creates_position(self):
        """Verify opening a LONG trade sets has_open_position to True."""
        # Arrange
        tracker = TradeTracker(spread=0.30, predicted_horizon=12)

        # Act
        tracker.open_trade(
            symbol="XAUUSD",
            direction=Direction.LONG,
            entry_price=Decimal("2000.00"),
            entry_timestamp=datetime(2024, 1, 2, 10, 0, 0, tzinfo=timezone.utc),
        )

        # Assert
        assert tracker.has_open_position() is True

    def test_open_trade_short(self):
        """Verify opening a SHORT trade works correctly."""
        # Arrange
        tracker = TradeTracker(spread=0.30, predicted_horizon=12)

        # Act
        tracker.open_trade(
            symbol="XAUUSD",
            direction=Direction.SHORT,
            entry_price=Decimal("2000.00"),
            entry_timestamp=datetime(2024, 1, 2, 10, 0, 0, tzinfo=timezone.utc),
        )

        # Assert
        assert tracker.has_open_position() is True

    def test_cannot_open_when_position_exists(self):
        """Verify raises ValueError if already open position exists."""
        # Arrange
        tracker = TradeTracker(spread=0.30, predicted_horizon=12)
        tracker.open_trade(
            symbol="XAUUSD",
            direction=Direction.LONG,
            entry_price=Decimal("2000.00"),
            entry_timestamp=datetime(2024, 1, 2, 10, 0, 0, tzinfo=timezone.utc),
        )

        # Act & Assert
        with pytest.raises(ValueError, match="Cannot open trade: position already open"):
            tracker.open_trade(
                symbol="XAUUSD",
                direction=Direction.SHORT,
                entry_price=Decimal("2005.00"),
                entry_timestamp=datetime(2024, 1, 2, 10, 5, 0, tzinfo=timezone.utc),
            )

    def test_update_bar_increments_counter(self):
        """Verify bars_since_entry increases with update_bar()."""
        # Arrange
        tracker = TradeTracker(spread=0.30, predicted_horizon=12)
        tracker.open_trade(
            symbol="XAUUSD",
            direction=Direction.LONG,
            entry_price=Decimal("2000.00"),
            entry_timestamp=datetime(2024, 1, 2, 10, 0, 0, tzinfo=timezone.utc),
        )

        # Act - simulate 5 bars passing
        for _ in range(5):
            tracker.update_bar()

        # Assert - check internal state via should_exit
        assert tracker.should_exit() is False  # 5 < 12

        # Act - continue to horizon
        for _ in range(7):
            tracker.update_bar()

        # Assert
        assert tracker.should_exit() is True  # 12 >= 12

    def test_should_exit_at_horizon(self):
        """Verify should_exit returns True after predicted_horizon bars."""
        # Arrange
        tracker = TradeTracker(spread=0.30, predicted_horizon=6)
        tracker.open_trade(
            symbol="XAUUSD",
            direction=Direction.LONG,
            entry_price=Decimal("2000.00"),
            entry_timestamp=datetime(2024, 1, 2, 10, 0, 0, tzinfo=timezone.utc),
        )

        # Act
        for _ in range(6):
            tracker.update_bar()

        # Assert
        assert tracker.should_exit() is True

    def test_should_exit_false_before_horizon(self):
        """Verify should_exit returns False before horizon reached."""
        # Arrange
        tracker = TradeTracker(spread=0.30, predicted_horizon=6)
        tracker.open_trade(
            symbol="XAUUSD",
            direction=Direction.LONG,
            entry_price=Decimal("2000.00"),
            entry_timestamp=datetime(2024, 1, 2, 10, 0, 0, tzinfo=timezone.utc),
        )

        # Act
        for _ in range(5):
            tracker.update_bar()

        # Assert
        assert tracker.should_exit() is False

    def test_close_trade_long_profitable(self):
        """Verify LONG trade with exit > entry produces positive P&L."""
        # Arrange
        tracker = TradeTracker(spread=0.30, predicted_horizon=12)
        tracker.open_trade(
            symbol="XAUUSD",
            direction=Direction.LONG,
            entry_price=Decimal("2000.00"),
            entry_timestamp=datetime(2024, 1, 2, 10, 0, 0, tzinfo=timezone.utc),
            regime_state=RegimeState.LOW_VOL_TREND,
        )

        # Simulate 3 bars held
        for _ in range(3):
            tracker.update_bar()

        # Act - close with profit
        trade = tracker.close_trade(
            exit_price=Decimal("2010.00"),  # +10 points
            exit_timestamp=datetime(2024, 1, 2, 10, 15, 0, tzinfo=timezone.utc),
        )

        # Assert
        # P&L = ((2010 - 2000) / 2000 - 0.30/2000) * 100
        #     = (0.005 - 0.00015) * 100 = 0.485%
        assert trade.pnl_pct == pytest.approx(0.485, abs=1e-3)
        assert trade.bars_held == 3
        assert trade.direction == Direction.LONG
        assert trade.regime_state == RegimeState.LOW_VOL_TREND
        assert tracker.has_open_position() is False

    def test_close_trade_long_losing(self):
        """Verify LONG trade with exit < entry produces negative P&L."""
        # Arrange
        tracker = TradeTracker(spread=0.30, predicted_horizon=12)
        tracker.open_trade(
            symbol="XAUUSD",
            direction=Direction.LONG,
            entry_price=Decimal("2000.00"),
            entry_timestamp=datetime(2024, 1, 2, 10, 0, 0, tzinfo=timezone.utc),
        )

        tracker.update_bar()
        tracker.update_bar()

        # Act - close with loss
        trade = tracker.close_trade(
            exit_price=Decimal("1990.00"),  # -10 points
            exit_timestamp=datetime(2024, 1, 2, 10, 10, 0, tzinfo=timezone.utc),
        )

        # Assert
        # P&L = ((1990 - 2000) / 2000 - 0.30/2000) * 100
        #     = (-0.005 - 0.00015) * 100 = -0.515%
        assert trade.pnl_pct == pytest.approx(-0.515, abs=1e-3)
        assert trade.bars_held == 2

    def test_close_trade_short_profitable(self):
        """Verify SHORT trade with exit < entry produces positive P&L."""
        # Arrange
        tracker = TradeTracker(spread=0.30, predicted_horizon=12)
        tracker.open_trade(
            symbol="XAUUSD",
            direction=Direction.SHORT,
            entry_price=Decimal("2000.00"),
            entry_timestamp=datetime(2024, 1, 2, 10, 0, 0, tzinfo=timezone.utc),
        )

        for _ in range(4):
            tracker.update_bar()

        # Act - close with profit (price went down, good for short)
        trade = tracker.close_trade(
            exit_price=Decimal("1990.00"),  # -10 points
            exit_timestamp=datetime(2024, 1, 2, 10, 20, 0, tzinfo=timezone.utc),
        )

        # Assert
        # SHORT P&L = ((2000 - 1990) / 2000 - 0.30/2000) * 100
        #          = (0.005 - 0.00015) * 100 = 0.485%
        assert trade.pnl_pct == pytest.approx(0.485, abs=1e-3)
        assert trade.bars_held == 4
        assert trade.direction == Direction.SHORT

    def test_close_trade_short_losing(self):
        """Verify SHORT trade with exit > entry produces negative P&L."""
        # Arrange
        tracker = TradeTracker(spread=0.30, predicted_horizon=12)
        tracker.open_trade(
            symbol="XAUUSD",
            direction=Direction.SHORT,
            entry_price=Decimal("2000.00"),
            entry_timestamp=datetime(2024, 1, 2, 10, 0, 0, tzinfo=timezone.utc),
        )

        tracker.update_bar()

        # Act - close with loss (price went up, bad for short)
        trade = tracker.close_trade(
            exit_price=Decimal("2010.00"),  # +10 points
            exit_timestamp=datetime(2024, 1, 2, 10, 5, 0, tzinfo=timezone.utc),
        )

        # Assert
        # SHORT P&L = ((2000 - 2010) / 2000 - 0.30/2000) * 100
        #          = (-0.005 - 0.00015) * 100 = -0.515%
        assert trade.pnl_pct == pytest.approx(-0.515, abs=1e-3)
        assert trade.bars_held == 1

    def test_close_trade_includes_spread(self):
        """Verify spread is correctly deducted from P&L."""
        # Arrange - zero spread
        tracker_no_spread = TradeTracker(spread=0.0, predicted_horizon=12)
        tracker_no_spread.open_trade(
            symbol="XAUUSD",
            direction=Direction.LONG,
            entry_price=Decimal("2000.00"),
            entry_timestamp=datetime(2024, 1, 2, 10, 0, 0, tzinfo=timezone.utc),
        )

        # With spread
        tracker_with_spread = TradeTracker(spread=0.30, predicted_horizon=12)
        tracker_with_spread.open_trade(
            symbol="XAUUSD",
            direction=Direction.LONG,
            entry_price=Decimal("2000.00"),
            entry_timestamp=datetime(2024, 1, 2, 10, 0, 0, tzinfo=timezone.utc),
        )

        # Act - same exit for both
        trade_no_spread = tracker_no_spread.close_trade(
            exit_price=Decimal("2000.00"),  # breakeven
            exit_timestamp=datetime(2024, 1, 2, 10, 5, 0, tzinfo=timezone.utc),
        )

        trade_with_spread = tracker_with_spread.close_trade(
            exit_price=Decimal("2000.00"),  # breakeven
            exit_timestamp=datetime(2024, 1, 2, 10, 5, 0, tzinfo=timezone.utc),
        )

        # Assert
        assert trade_no_spread.pnl_pct == pytest.approx(0.0, abs=1e-5)
        # With spread: -0.30/2000 * 100 = -0.015%
        assert trade_with_spread.pnl_pct == pytest.approx(-0.015, abs=1e-5)

    def test_close_trade_returns_frozen_dataclass(self):
        """Verify returned BacktestTrade is frozen (immutable)."""
        # Arrange
        tracker = TradeTracker(spread=0.30, predicted_horizon=12)
        tracker.open_trade(
            symbol="XAUUSD",
            direction=Direction.LONG,
            entry_price=Decimal("2000.00"),
            entry_timestamp=datetime(2024, 1, 2, 10, 0, 0, tzinfo=timezone.utc),
        )

        # Act
        trade = tracker.close_trade(
            exit_price=Decimal("2005.00"),
            exit_timestamp=datetime(2024, 1, 2, 10, 5, 0, tzinfo=timezone.utc),
        )

        # Assert - frozen dataclass should not allow modification
        from dataclasses import FrozenInstanceError
        with pytest.raises(FrozenInstanceError):
            trade.pnl_pct = 999.0

    def test_force_close_with_open_position(self):
        """Verify force_close_all returns the closed trade."""
        # Arrange
        tracker = TradeTracker(spread=0.30, predicted_horizon=12)
        tracker.open_trade(
            symbol="XAUUSD",
            direction=Direction.LONG,
            entry_price=Decimal("2000.00"),
            entry_timestamp=datetime(2024, 1, 2, 10, 0, 0, tzinfo=timezone.utc),
        )

        for _ in range(8):
            tracker.update_bar()

        # Act
        trade = tracker.force_close_all(
            exit_price=Decimal("2005.00"),
            exit_timestamp=datetime(2024, 1, 2, 10, 40, 0, tzinfo=timezone.utc),
        )

        # Assert
        assert trade is not None
        assert isinstance(trade, BacktestTrade)
        assert trade.bars_held == 8
        assert tracker.has_open_position() is False

    def test_force_close_without_position(self):
        """Verify force_close_all returns None when no position open."""
        # Arrange
        tracker = TradeTracker(spread=0.30, predicted_horizon=12)

        # Act
        trade = tracker.force_close_all(
            exit_price=Decimal("2005.00"),
            exit_timestamp=datetime(2024, 1, 2, 10, 40, 0, tzinfo=timezone.utc),
        )

        # Assert
        assert trade is None

    def test_get_closed_trades_returns_copy(self):
        """Verify get_closed_trades returns independent copy of list."""
        # Arrange
        tracker = TradeTracker(spread=0.30, predicted_horizon=12)

        # Create and close first trade
        tracker.open_trade(
            symbol="XAUUSD",
            direction=Direction.LONG,
            entry_price=Decimal("2000.00"),
            entry_timestamp=datetime(2024, 1, 2, 10, 0, 0, tzinfo=timezone.utc),
        )
        trade1 = tracker.close_trade(
            exit_price=Decimal("2005.00"),
            exit_timestamp=datetime(2024, 1, 2, 10, 5, 0, tzinfo=timezone.utc),
        )

        # Create and close second trade
        tracker.open_trade(
            symbol="XAUUSD",
            direction=Direction.SHORT,
            entry_price=Decimal("2010.00"),
            entry_timestamp=datetime(2024, 1, 2, 10, 10, 0, tzinfo=timezone.utc),
        )
        trade2 = tracker.close_trade(
            exit_price=Decimal("2008.00"),
            exit_timestamp=datetime(2024, 1, 2, 10, 15, 0, tzinfo=timezone.utc),
        )

        # Act
        closed_trades = tracker.get_closed_trades()

        # Assert
        assert len(closed_trades) == 2
        assert closed_trades[0] == trade1
        assert closed_trades[1] == trade2

        # Verify it's a copy (modifications don't affect internal state)
        closed_trades.clear()
        assert len(tracker.get_closed_trades()) == 2

    def test_no_lookahead_in_pnl_calculation(self):
        """Verify P&L calculation uses only entry and exit prices correctly."""
        # This test verifies that P&L is computed using information available
        # at trade time, not future information

        # Arrange
        tracker = TradeTracker(spread=0.30, predicted_horizon=12)
        entry_timestamp = datetime(2024, 1, 2, 10, 0, 0, tzinfo=timezone.utc)
        exit_timestamp = datetime(2024, 1, 2, 10, 30, 0, tzinfo=timezone.utc)

        tracker.open_trade(
            symbol="XAUUSD",
            direction=Direction.LONG,
            entry_price=Decimal("2000.00"),
            entry_timestamp=entry_timestamp,
        )

        # Simulate bars passing
        for _ in range(6):
            tracker.update_bar()

        # Act - close trade
        trade = tracker.close_trade(
            exit_price=Decimal("2010.00"),
            exit_timestamp=exit_timestamp,
        )

        # Assert - P&L should only depend on entry and exit prices
        # The calculation should be: ((exit - entry) / entry - spread_pct) * 100
        expected_spread_pct = 0.30 / 2000.0
        expected_raw_pnl = (2010.0 - 2000.0) / 2000.0
        expected_pnl_pct = (expected_raw_pnl - expected_spread_pct) * 100

        assert trade.pnl_pct == pytest.approx(expected_pnl_pct, abs=1e-5)
        assert trade.entry_price == Decimal("2000.00")
        assert trade.exit_price == Decimal("2010.00")
        assert trade.entry_timestamp == entry_timestamp
        assert trade.exit_timestamp == exit_timestamp

    def test_close_trade_no_position_raises(self):
        """Verify close_trade raises ValueError when no position is open."""
        # Arrange
        tracker = TradeTracker(spread=0.30, predicted_horizon=12)

        # Act & Assert
        with pytest.raises(ValueError, match="Cannot close trade: no position open"):
            tracker.close_trade(
                exit_price=Decimal("2000.00"),
                exit_timestamp=datetime(2024, 1, 2, 10, 0, 0, tzinfo=timezone.utc),
            )

    def test_should_exit_no_position_returns_false(self):
        """Verify should_exit returns False when no position is open."""
        # Arrange
        tracker = TradeTracker(spread=0.30, predicted_horizon=12)

        # Act & Assert
        assert tracker.should_exit() is False

    def test_update_bar_no_position_does_nothing(self):
        """Verify update_bar does nothing when no position is open."""
        # Arrange
        tracker = TradeTracker(spread=0.30, predicted_horizon=12)

        # Act - should not raise
        tracker.update_bar()
        tracker.update_bar()

        # Assert - still no position
        assert tracker.has_open_position() is False