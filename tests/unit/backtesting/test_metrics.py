"""Unit tests for the trade metrics computation module."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import numpy as np
import pytest

from fxer.backtesting.metrics import (
    _calculate_max_drawdown,
    compute_regime_breakdown,
    compute_trade_metrics,
)
from fxer.backtesting.types import BacktestTrade, RegimeMetrics, TradeMetrics
from fxer.regime.types import RegimeState
from fxer.signals.types import Direction


def _make_backtest_trade(
    pnl_pct: float,
    bars_held: int = 12,
    direction: Direction = Direction.LONG,
    regime_state: RegimeState | None = None,
) -> BacktestTrade:
    """Helper to create a BacktestTrade with minimal fields."""
    return BacktestTrade(
        symbol="XAUUSD",
        direction=direction,
        entry_timestamp=datetime(2024, 1, 2, 10, 0, 0, tzinfo=timezone.utc),
        entry_price=Decimal("2000.00"),
        exit_timestamp=datetime(2024, 1, 2, 11, 0, 0, tzinfo=timezone.utc),
        exit_price=Decimal("2010.00"),
        spread_pct=0.00015,  # 0.30 / 2000
        predicted_horizon=12,
        bars_held=bars_held,
        pnl_pct=pnl_pct,
        regime_state=regime_state,
    )


class TestComputeTradeMetrics:
    """Test compute_trade_metrics function."""

    def test_compute_trade_metrics_all_wins(self):
        """Verify metrics when all trades are profitable."""
        # Arrange - 5 winning trades
        trades = [
            _make_backtest_trade(pnl_pct=0.5, bars_held=12),
            _make_backtest_trade(pnl_pct=1.0, bars_held=10),
            _make_backtest_trade(pnl_pct=0.8, bars_held=14),
            _make_backtest_trade(pnl_pct=0.3, bars_held=8),
            _make_backtest_trade(pnl_pct=1.2, bars_held=16),
        ]

        # Act
        metrics = compute_trade_metrics(trades)

        # Assert
        assert metrics.trade_count == 5
        assert metrics.win_rate == 100.0
        assert metrics.avg_win == pytest.approx(0.76, abs=0.01)  # (0.5+1.0+0.8+0.3+1.2)/5
        assert metrics.avg_loss == 0.0
        assert metrics.best_trade == 1.2
        assert metrics.worst_trade == 0.3
        assert metrics.profit_factor == 99.99  # Capped when no losses
        assert metrics.max_drawdown == 0.0  # No drawdown with all wins
        # Total return: compound
        expected_return = ((1.005 * 1.01 * 1.008 * 1.003 * 1.012) - 1) * 100
        assert metrics.total_return == pytest.approx(expected_return, abs=0.01)

    def test_compute_trade_metrics_all_losses(self):
        """Verify metrics when all trades are losing."""
        # Arrange - 4 losing trades
        trades = [
            _make_backtest_trade(pnl_pct=-0.3, bars_held=12),
            _make_backtest_trade(pnl_pct=-0.5, bars_held=10),
            _make_backtest_trade(pnl_pct=-0.8, bars_held=14),
            _make_backtest_trade(pnl_pct=-0.2, bars_held=8),
        ]

        # Act
        metrics = compute_trade_metrics(trades)

        # Assert
        assert metrics.trade_count == 4
        assert metrics.win_rate == 0.0
        assert metrics.avg_win == 0.0
        assert metrics.avg_loss == pytest.approx(-0.45, abs=0.01)  # (-0.3-0.5-0.8-0.2)/4
        assert metrics.best_trade == -0.2
        assert metrics.worst_trade == -0.8
        assert metrics.profit_factor == 0.0  # No wins
        # Total return: compound losses
        expected_return = ((0.997 * 0.995 * 0.992 * 0.998) - 1) * 100
        assert metrics.total_return == pytest.approx(expected_return, abs=0.01)

    def test_compute_trade_metrics_mixed(self):
        """Verify metrics with mix of wins and losses."""
        # Arrange - 3 wins, 2 losses
        trades = [
            _make_backtest_trade(pnl_pct=1.0, bars_held=12),   # Win
            _make_backtest_trade(pnl_pct=-0.5, bars_held=10),  # Loss
            _make_backtest_trade(pnl_pct=0.8, bars_held=14),   # Win
            _make_backtest_trade(pnl_pct=-0.3, bars_held=8),   # Loss
            _make_backtest_trade(pnl_pct=0.6, bars_held=16),   # Win
        ]

        # Act
        metrics = compute_trade_metrics(trades)

        # Assert
        assert metrics.trade_count == 5
        assert metrics.win_rate == 60.0  # 3/5
        assert metrics.avg_win == pytest.approx(0.8, abs=0.01)  # (1.0+0.8+0.6)/3
        assert metrics.avg_loss == pytest.approx(-0.4, abs=0.01)  # (-0.5-0.3)/2
        assert metrics.best_trade == 1.0
        assert metrics.worst_trade == -0.5

        # Profit factor = sum(wins) / abs(sum(losses))
        # = (1.0+0.8+0.6) / abs(-0.5-0.3) = 2.4 / 0.8 = 3.0
        assert metrics.profit_factor == pytest.approx(3.0, abs=0.01)

    def test_compute_trade_metrics_empty(self):
        """Verify empty trade list returns zeros."""
        # Act
        metrics = compute_trade_metrics([])

        # Assert
        assert metrics.trade_count == 0
        assert metrics.total_return == 0.0
        assert metrics.sharpe_ratio == 0.0
        assert metrics.sortino_ratio == 0.0
        assert metrics.profit_factor == 0.0
        assert metrics.win_rate == 0.0
        assert metrics.max_drawdown == 0.0
        assert metrics.avg_win == 0.0
        assert metrics.avg_loss == 0.0
        assert metrics.best_trade == 0.0
        assert metrics.worst_trade == 0.0

    def test_sharpe_ratio_annualized(self):
        """Verify Sharpe ratio uses annualization factor correctly."""
        # Arrange - trades with known mean/std
        trades = [
            _make_backtest_trade(pnl_pct=1.0, bars_held=12),
            _make_backtest_trade(pnl_pct=0.5, bars_held=12),
            _make_backtest_trade(pnl_pct=-0.5, bars_held=12),
            _make_backtest_trade(pnl_pct=0.8, bars_held=12),
            _make_backtest_trade(pnl_pct=-0.3, bars_held=12),
        ]

        # Act
        metrics = compute_trade_metrics(trades, periods_per_year=252 * 6.5)

        # Assert
        pnl_array = np.array([1.0, 0.5, -0.5, 0.8, -0.3])
        mean_return = np.mean(pnl_array)
        std_return = np.std(pnl_array, ddof=1)

        # With 12 bars per trade and 252*6.5 periods per year
        # annualization_factor = sqrt(1638 / 12) = sqrt(136.5) ≈ 11.68
        annualization_factor = np.sqrt(252 * 6.5 / 12)
        expected_sharpe = mean_return / std_return * annualization_factor

        assert metrics.sharpe_ratio == pytest.approx(expected_sharpe, abs=0.1)

    def test_sortino_ratio_no_losses(self):
        """Verify Sortino ratio when there are no losing trades."""
        # Arrange - all winning trades
        trades = [
            _make_backtest_trade(pnl_pct=1.0, bars_held=12),
            _make_backtest_trade(pnl_pct=0.5, bars_held=12),
            _make_backtest_trade(pnl_pct=0.8, bars_held=12),
        ]

        # Act
        metrics = compute_trade_metrics(trades)

        # Assert
        # When no losses, sortino should be 2x sharpe (as coded)
        assert metrics.sortino_ratio == pytest.approx(metrics.sharpe_ratio * 2.0, abs=0.01)

    def test_profit_factor_no_losses(self):
        """Verify profit factor is capped at 99.99 when no losses."""
        # Arrange
        trades = [
            _make_backtest_trade(pnl_pct=1.0),
            _make_backtest_trade(pnl_pct=0.5),
        ]

        # Act
        metrics = compute_trade_metrics(trades)

        # Assert
        assert metrics.profit_factor == 99.99

    def test_max_drawdown_calculation(self):
        """Verify max drawdown calculation with known scenario."""
        # Arrange - sequence with known drawdown
        trades = [
            _make_backtest_trade(pnl_pct=2.0),   # Up 2%
            _make_backtest_trade(pnl_pct=1.0),   # Up 1%
            _make_backtest_trade(pnl_pct=-3.0),  # Down 3% (drawdown starts)
            _make_backtest_trade(pnl_pct=-2.0),  # Down 2% (drawdown deepens)
            _make_backtest_trade(pnl_pct=1.0),   # Up 1% (recovery)
        ]

        # Act
        metrics = compute_trade_metrics(trades)

        # Assert
        # Equity curve: 1.02 * 1.01 * 0.97 * 0.98 * 1.01
        # Peak after trade 2: 1.02 * 1.01 = 1.0302
        # Trough after trade 4: 1.0302 * 0.97 * 0.98 ≈ 0.9788
        # Drawdown = (0.9788 - 1.0302) / 1.0302 ≈ -5.0%
        assert metrics.max_drawdown > 4.0  # Should be around 5%

    def test_win_rate_calculation(self):
        """Verify win rate calculation: 3 wins out of 5 = 60%."""
        # Arrange
        trades = [
            _make_backtest_trade(pnl_pct=1.0),   # Win
            _make_backtest_trade(pnl_pct=-0.5),  # Loss
            _make_backtest_trade(pnl_pct=0.8),   # Win
            _make_backtest_trade(pnl_pct=-0.3),  # Loss
            _make_backtest_trade(pnl_pct=0.6),   # Win
        ]

        # Act
        metrics = compute_trade_metrics(trades)

        # Assert
        assert metrics.win_rate == 60.0


class TestComputeRegimeBreakdown:
    """Test compute_regime_breakdown function."""

    def test_regime_breakdown_groups_correctly(self):
        """Verify trades are grouped by regime state correctly."""
        # Arrange
        trades = [
            _make_backtest_trade(pnl_pct=1.0, bars_held=12, regime_state=RegimeState.LOW_VOL_TREND),
            _make_backtest_trade(pnl_pct=0.5, bars_held=12, regime_state=RegimeState.LOW_VOL_TREND),
            _make_backtest_trade(pnl_pct=-0.5, bars_held=10, regime_state=RegimeState.HIGH_VOL_TREND),
            _make_backtest_trade(pnl_pct=0.8, bars_held=14, regime_state=RegimeState.HIGH_VOL_TREND),
            _make_backtest_trade(pnl_pct=-0.3, bars_held=8, regime_state=RegimeState.RANGING),
            _make_backtest_trade(pnl_pct=0.2, bars_held=16, regime_state=RegimeState.RANGING),
        ]

        # Act
        breakdown = compute_regime_breakdown(trades)

        # Assert
        assert len(breakdown) == 3
        assert "low_vol_trend" in breakdown
        assert "high_vol_trend" in breakdown
        assert "ranging" in breakdown

        # Low vol trend: 2 trades, both wins
        low_vol = breakdown["low_vol_trend"]
        assert low_vol.trade_count == 2
        assert low_vol.win_rate == 100.0

        # High vol trend: 2 trades, 1 win, 1 loss
        high_vol = breakdown["high_vol_trend"]
        assert high_vol.trade_count == 2
        assert high_vol.win_rate == 50.0

        # Ranging: 2 trades, 1 win, 1 loss
        ranging = breakdown["ranging"]
        assert ranging.trade_count == 2
        assert ranging.win_rate == 50.0

    def test_regime_breakdown_empty_trades(self):
        """Verify empty trade list returns empty dict."""
        # Act
        breakdown = compute_regime_breakdown([])

        # Assert
        assert breakdown == {}

    def test_regime_breakdown_no_regime_trades(self):
        """Verify trades without regime state are ignored."""
        # Arrange
        trades = [
            _make_backtest_trade(pnl_pct=1.0, regime_state=None),
            _make_backtest_trade(pnl_pct=0.5, regime_state=None),
        ]

        # Act
        breakdown = compute_regime_breakdown(trades)

        # Assert
        assert breakdown == {}

    def test_regime_breakdown_mixed_with_none(self):
        """Verify mix of trades with and without regime state."""
        # Arrange
        trades = [
            _make_backtest_trade(pnl_pct=1.0, regime_state=RegimeState.LOW_VOL_TREND),
            _make_backtest_trade(pnl_pct=0.5, regime_state=None),  # Ignored
            _make_backtest_trade(pnl_pct=-0.5, regime_state=RegimeState.LOW_VOL_TREND),
        ]

        # Act
        breakdown = compute_regime_breakdown(trades)

        # Assert
        assert len(breakdown) == 1
        assert "low_vol_trend" in breakdown
        low_vol = breakdown["low_vol_trend"]
        assert low_vol.trade_count == 2  # Only 2 with regime state

    def test_regime_metrics_total_return(self):
        """Verify total return is computed per regime."""
        # Arrange
        trades = [
            _make_backtest_trade(pnl_pct=2.0, regime_state=RegimeState.LOW_VOL_TREND),
            _make_backtest_trade(pnl_pct=1.0, regime_state=RegimeState.LOW_VOL_TREND),
        ]

        # Act
        breakdown = compute_regime_breakdown(trades)

        # Assert
        low_vol = breakdown["low_vol_trend"]
        # Compound: (1.02 * 1.01 - 1) * 100 = 3.02%
        expected_return = ((1.02 * 1.01) - 1) * 100
        assert low_vol.total_return == pytest.approx(expected_return, abs=0.01)

    def test_regime_metrics_sharpe_ratio(self):
        """Verify Sharpe ratio is computed per regime."""
        # Arrange
        trades = [
            _make_backtest_trade(pnl_pct=1.0, bars_held=12, regime_state=RegimeState.HIGH_VOL_TREND),
            _make_backtest_trade(pnl_pct=-0.5, bars_held=12, regime_state=RegimeState.HIGH_VOL_TREND),
            _make_backtest_trade(pnl_pct=0.8, bars_held=12, regime_state=RegimeState.HIGH_VOL_TREND),
        ]

        # Act
        breakdown = compute_regime_breakdown(trades, periods_per_year=252 * 6.5)

        # Assert
        high_vol = breakdown["high_vol_trend"]

        # Manual calculation
        pnl_array = np.array([1.0, -0.5, 0.8])
        mean_return = np.mean(pnl_array)
        std_return = np.std(pnl_array, ddof=1)
        annualization_factor = np.sqrt(252 * 6.5 / 12)
        expected_sharpe = mean_return / std_return * annualization_factor

        assert high_vol.sharpe_ratio == pytest.approx(expected_sharpe, abs=0.1)


class TestMaxDrawdown:
    """Test the _calculate_max_drawdown helper function."""

    def test_max_drawdown_empty_array(self):
        """Verify empty array returns 0 drawdown."""
        assert _calculate_max_drawdown(np.array([])) == 0.0

    def test_max_drawdown_single_trade(self):
        """Verify single trade has no drawdown."""
        pnl = np.array([1.0])
        assert _calculate_max_drawdown(pnl) == 0.0

    def test_max_drawdown_all_positive(self):
        """Verify all positive returns have no drawdown."""
        pnl = np.array([1.0, 0.5, 2.0, 1.5])
        assert _calculate_max_drawdown(pnl) == 0.0

    def test_max_drawdown_simple_case(self):
        """Verify simple drawdown calculation."""
        # Arrange
        # Start at 100, go to 110, drop to 95, recover to 105
        # Max drawdown is from 110 to 95 = (95-110)/110 = -13.6%
        pnl = np.array([10.0, -13.18, 10.53])  # Results in equity: 1.1, 0.95, 1.05

        # Act
        dd = _calculate_max_drawdown(pnl)

        # Assert
        # Equity: 1.1 * 0.8682 * 1.1053 ≈ 1.05
        # Peak at 1.1, trough at 1.1 * 0.8682 ≈ 0.955
        # Drawdown = (0.955 - 1.1) / 1.1 ≈ -13.18%
        assert dd == pytest.approx(13.18, abs=0.1)

    def test_max_drawdown_multiple_drawdowns(self):
        """Verify maximum of multiple drawdowns is returned."""
        # Arrange - two drawdown periods
        pnl = np.array([
            5.0,   # Up to 1.05
            3.0,   # Up to 1.0815
            -2.0,  # Down to 1.0599 (small drawdown)
            1.0,   # Up to 1.0705
            -5.0,  # Down to 1.017 (larger drawdown from peak)
            2.0,   # Partial recovery
        ])

        # Act
        dd = _calculate_max_drawdown(pnl)

        # Assert
        # Peak is at 1.0815 (after trade 2)
        # Lowest point relative to peak happens later
        assert dd > 5.0  # Should capture the larger drawdown