"""Integration tests for the BacktestEngine module."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch
from tempfile import TemporaryDirectory

import pytest

from fxer.backtesting.engine import BacktestEngine
from fxer.backtesting.types import BacktestResult
from fxer.config.settings import Settings
from fxer.core.events import FeatureVector, NormalizedBar
from fxer.core.exceptions import ModelLoadError
from fxer.core.types import Timeframe
from fxer.regime.types import RegimeDecision, RegimeState
from fxer.signals.types import Direction, HoldingPeriod, TradeSignal


def make_test_bars(count: int = 100) -> list[NormalizedBar]:
    """Create test bars with valid OHLC relationships."""
    bars = []
    base_price = 2000.0
    start = datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc)

    for i in range(count):
        ts = start.replace(
            hour=(i * 5) // 60,
            minute=(i * 5) % 60,
        )

        # Simple price movement
        open_price = base_price + (i * 0.1)
        close_price = open_price + 0.5
        high_price = max(open_price, close_price) + 0.2
        low_price = min(open_price, close_price) - 0.2

        bars.append(NormalizedBar(
            symbol="XAUUSD",
            timeframe=Timeframe.M5,
            timestamp=ts,
            open=Decimal(str(open_price)),
            high=Decimal(str(high_price)),
            low=Decimal(str(low_price)),
            close=Decimal(str(close_price)),
            volume=Decimal("1000"),
            is_complete=True,
        ))

        base_price = close_price

    return bars


class TestBacktestEngineIntegration:
    """Integration tests for BacktestEngine."""

    def test_empty_bars_returns_empty_result(self):
        """Verify empty bar list produces empty BacktestResult."""
        # Arrange
        engine = BacktestEngine()

        # Act
        result = engine.run([], "XAUUSD", "5m")

        # Assert
        assert result.symbol == "XAUUSD"
        assert result.timeframe == "5m"
        assert result.start_date is None
        assert result.end_date is None
        assert result.bars_processed == 0
        assert result.signals_generated == 0
        assert result.trades_executed == 0
        assert result.total_return == 0.0
        assert result.meets_minimum is False

    def test_model_not_found_raises_error(self):
        """Verify ModelLoadError when models don't exist."""
        # Arrange
        bars = make_test_bars(10)

        with TemporaryDirectory() as tmpdir:
            settings = Settings()
            settings.signal_model_dir = tmpdir  # Empty directory

            engine = BacktestEngine(settings=settings)

            # Act & Assert
            with pytest.raises(ModelLoadError, match="Model not found"):
                engine.run(bars, "XAUUSD", "5m")

    def _test_backtest_with_mocked_models(self):
        """Test backtest run with fully mocked models."""
        # Arrange
        bars = make_test_bars(100)
        settings = Settings()

        with patch("fxer.backtesting.engine.FeatureEngine") as MockFeatureEngine, \
             patch("fxer.backtesting.engine.StackingEnsemble") as MockEnsemble, \
             patch("fxer.backtesting.engine.Path") as MockPath:

            # Mock path existence
            mock_path = Mock()
            mock_path.exists.return_value = True
            MockPath.return_value.__truediv__.return_value = mock_path

            # Mock feature engine
            mock_feature_engine = Mock()
            mock_feature_engine.warmup_complete = False
            mock_feature_engine.compute_features.return_value = Mock(spec=FeatureVector)

            # Set warmup complete after 50 bars
            call_count = [0]
            def compute_features_side_effect(bar):
                call_count[0] += 1
                if call_count[0] > 50:
                    mock_feature_engine.warmup_complete = True
                return Mock(spec=FeatureVector)

            mock_feature_engine.compute_features.side_effect = compute_features_side_effect
            MockFeatureEngine.return_value = mock_feature_engine

            # Mock ensemble
            mock_ensemble = Mock()
            mock_ensemble.load.return_value = None

            # Generate alternating signals after warmup
            def generate_signal_side_effect(features, symbol, timestamp, horizon):
                bar_idx = next((i for i, b in enumerate(bars) if b.timestamp == timestamp), 0)
                if bar_idx < 50:
                    direction = Direction.NEUTRAL
                else:
                    # Alternate LONG/SHORT every 20 bars to allow positions to close
                    direction = Direction.LONG if (bar_idx // 20) % 2 == 0 else Direction.SHORT

                return TradeSignal(
                    symbol=symbol,
                    timestamp=timestamp,
                    direction=direction,
                    confidence=0.7,
                    predicted_horizon=horizon,
                    xgboost_prob=0.6,
                    lstm_prob=0.6,
                    ensemble_prob=0.6,
                    top_features={},
                    warmup_complete=bar_idx >= 50,
                )

            mock_ensemble.generate_signal.side_effect = generate_signal_side_effect
            MockEnsemble.return_value = mock_ensemble

            # Act
            engine = BacktestEngine(settings=settings, use_regime=False)
            result = engine.run(bars, "XAUUSD", "5m")

            # Assert
            assert result.symbol == "XAUUSD"
            assert result.timeframe == "5m"
            assert result.bars_processed == 100
            assert result.signals_generated == 50  # After warmup
            assert result.trades_executed > 0  # Should have some trades
            assert result.spread_pct == 0.30

    def _test_signal_entry_timing_no_lookahead(self):
        """CRITICAL: Verify signal at bar[i] opens position at bar[i+1].open."""
        # Arrange
        bars = [
            NormalizedBar(
                symbol="XAUUSD",
                timeframe=Timeframe.M5,
                timestamp=datetime(2024, 1, 2, 10, 0, 0, tzinfo=timezone.utc),
                open=Decimal("2000.00"),
                high=Decimal("2005.00"),
                low=Decimal("1995.00"),
                close=Decimal("2003.00"),
                volume=Decimal("1000"),
                is_complete=True,
            ),
            NormalizedBar(
                symbol="XAUUSD",
                timeframe=Timeframe.M5,
                timestamp=datetime(2024, 1, 2, 10, 5, 0, tzinfo=timezone.utc),
                open=Decimal("2004.00"),  # Should enter here
                high=Decimal("2008.00"),
                low=Decimal("2002.00"),
                close=Decimal("2006.00"),
                volume=Decimal("1000"),
                is_complete=True,
            ),
            NormalizedBar(
                symbol="XAUUSD",
                timeframe=Timeframe.M5,
                timestamp=datetime(2024, 1, 2, 10, 10, 0, tzinfo=timezone.utc),
                open=Decimal("2007.00"),
                high=Decimal("2010.00"),
                low=Decimal("2005.00"),
                close=Decimal("2008.00"),
                volume=Decimal("1000"),
                is_complete=True,
            ),
        ]

        settings = Settings()
        entry_price_captured = []

        with patch("fxer.backtesting.engine.FeatureEngine") as MockFeatureEngine, \
             patch("fxer.backtesting.engine.StackingEnsemble") as MockEnsemble, \
             patch("fxer.backtesting.engine.TradeTracker") as MockTracker, \
             patch("fxer.backtesting.engine.Path") as MockPath:

            # Mock path
            mock_path = Mock()
            mock_path.exists.return_value = True
            MockPath.return_value.__truediv__.return_value = mock_path

            # Mock feature engine
            mock_feature_engine = Mock()
            mock_feature_engine.warmup_complete = True
            mock_feature_engine.compute_features.return_value = Mock(spec=FeatureVector)
            MockFeatureEngine.return_value = mock_feature_engine

            # Mock ensemble - signal only on first bar
            mock_ensemble = Mock()
            mock_ensemble.load.return_value = None

            signal_count = [0]
            def generate_signal_side_effect(features, symbol, timestamp, horizon):
                if signal_count[0] == 0 and timestamp == bars[0].timestamp:
                    signal_count[0] += 1
                    return TradeSignal(
                        symbol=symbol,
                        timestamp=timestamp,
                        direction=Direction.LONG,
                        confidence=0.8,
                        predicted_horizon=horizon,
                        xgboost_prob=0.7,
                        lstm_prob=0.7,
                        ensemble_prob=0.7,
                        top_features={},
                        warmup_complete=True,
                    )
                return TradeSignal(
                    symbol=symbol,
                    timestamp=timestamp,
                    direction=Direction.NEUTRAL,
                    confidence=0.0,
                    predicted_horizon=horizon,
                    xgboost_prob=0.5,
                    lstm_prob=0.5,
                    ensemble_prob=0.5,
                    top_features={},
                    warmup_complete=True,
                )

            mock_ensemble.generate_signal.side_effect = generate_signal_side_effect
            MockEnsemble.return_value = mock_ensemble

            # Mock tracker to capture entry
            mock_tracker = Mock()
            mock_tracker.has_open_position.return_value = False
            mock_tracker.should_exit.return_value = False
            mock_tracker.get_closed_trades.return_value = []

            def open_trade_side_effect(symbol, direction, entry_price, entry_timestamp, regime_state):
                entry_price_captured.append(entry_price)

            mock_tracker.open_trade.side_effect = open_trade_side_effect
            MockTracker.return_value = mock_tracker

            # Act
            engine = BacktestEngine(settings=settings, use_regime=False)
            result = engine.run(bars, "XAUUSD", "5m")

            # Assert - Critical assertion
            assert len(entry_price_captured) == 1
            assert entry_price_captured[0] == bars[1].open  # Entry at next bar's open
            assert entry_price_captured[0] != bars[0].close  # NOT at signal bar's close

    def _test_regime_filtering(self):
        """Test that regime can filter out signals."""
        # Arrange
        bars = make_test_bars(60)
        settings = Settings()

        with patch("fxer.backtesting.engine.FeatureEngine") as MockFeatureEngine, \
             patch("fxer.backtesting.engine.StackingEnsemble") as MockEnsemble, \
             patch("fxer.backtesting.engine.RegimeClassifier") as MockRegime, \
             patch("fxer.backtesting.engine.Path") as MockPath:

            # Mock paths
            mock_path = Mock()
            mock_path.exists.return_value = True
            MockPath.return_value.__truediv__.return_value = mock_path

            # Mock feature engine
            mock_feature_engine = Mock()
            mock_feature_engine.warmup_complete = True
            mock_feature_engine.compute_features.return_value = Mock(spec=FeatureVector)
            MockFeatureEngine.return_value = mock_feature_engine

            # Mock ensemble - always LONG
            mock_ensemble = Mock()
            mock_ensemble.load.return_value = None
            mock_ensemble.generate_signal.return_value = TradeSignal(
                symbol="XAUUSD",
                timestamp=datetime.now(timezone.utc),
                direction=Direction.LONG,
                confidence=0.7,
                predicted_horizon=HoldingPeriod.HOUR_1,
                xgboost_prob=0.6,
                lstm_prob=0.6,
                ensemble_prob=0.6,
                top_features={},
                warmup_complete=True,
            )
            MockEnsemble.return_value = mock_ensemble

            # Mock regime - alternates should_trade
            mock_regime = Mock()
            mock_regime.load_hmm_model.return_value = None

            call_count = [0]
            def classify_side_effect(bar, features, daily_bars):
                call_count[0] += 1
                should_trade = call_count[0] % 2 == 1  # Trade on odd calls
                return RegimeDecision(
                    state=RegimeState.RANGING,
                    confidence=0.7,
                    position_multiplier=1.0,
                    should_trade=should_trade,
                    reason="Test regime",
                )

            mock_regime.classify.side_effect = classify_side_effect
            MockRegime.return_value = mock_regime

            # Act
            engine = BacktestEngine(settings=settings, use_regime=True)
            result = engine.run(bars, "XAUUSD", "5m")

            # Assert
            assert result.bars_processed == 60
            # Some signals should be filtered by regime
            assert result.signals_generated == 60
            # But not all will result in trades due to regime filtering
            # and one-position-at-a-time rule