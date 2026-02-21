"""Integration tests for RegimeClassifier and SignalGenerator interaction.

Tests the full event flow:
1. SignalGenerator receives feature events
2. RegimeClassifier evaluates market conditions
3. Regime events are published to EventBus
4. Signals are gated based on regime decision
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch
from decimal import Decimal

import pytest

from fxer.config.settings import Settings
from fxer.core.events import FeatureVector, NormalizedBar
from fxer.core.types import Timeframe
from fxer.messaging.bus import EventBus
from fxer.regime.classifier import RegimeClassifier
from fxer.regime.types import RegimeDecision, RegimeEvent, RegimeState
from fxer.signals.generator import SignalGenerator
from fxer.signals.types import Direction, HoldingPeriod, TradeSignal


# ---------- Helper Factories ----------


def _make_feature_vector(**kwargs):
    """Create a FeatureVector with defaults."""
    defaults = {
        "symbol": "XAUUSD",
        "timestamp": datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        "timeframe": Timeframe.M5,
        "rsi_14": 55.0,
        "rsi_7": 60.0,
        "macd_line": 0.5,
        "macd_signal": 0.3,
        "macd_histogram": 0.2,
        "bb_upper": 2050.0,
        "bb_middle": 2040.0,
        "bb_lower": 2030.0,
        "bb_width": 0.01,
        "atr_14": 5.0,
        "is_london_session": True,
        "is_ny_session": True,
        "is_overlap_session": True,
        "is_asian_session": False,
        "hour_of_day": 14,
        "day_of_week": 2,
        "is_month_turn": False,
        "warmup_complete": True,
        "dxy_return_1h": -0.001,
        "dxy_rsi_14": 48.0,
        "vix_level": 15.5,
        "vix_change": 0.3,
    }
    defaults.update(kwargs)
    return FeatureVector(**defaults)


def _make_normalized_bar(**kwargs):
    """Create a NormalizedBar with defaults."""
    defaults = {
        "symbol": "XAUUSD",
        "timeframe": Timeframe.M5,
        "timestamp": datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        "open": Decimal("2040.00"),
        "high": Decimal("2042.00"),
        "low": Decimal("2038.00"),
        "close": Decimal("2041.00"),
        "volume": Decimal("1500"),
        "is_complete": True,
    }
    defaults.update(kwargs)
    return NormalizedBar(**defaults)


def _make_trade_signal(**kwargs):
    """Create a TradeSignal with defaults."""
    defaults = {
        "symbol": "XAUUSD",
        "timestamp": datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        "direction": Direction.LONG,
        "confidence": 0.75,
        "predicted_horizon": HoldingPeriod.HOUR_1,
        "xgboost_prob": 0.72,
        "lstm_prob": 0.78,
        "ensemble_prob": 0.75,
        "top_features": {"rsi_14": 0.15, "macd_line": 0.10},
        "warmup_complete": True,
    }
    defaults.update(kwargs)
    return TradeSignal(**defaults)


# ---------- Fixtures ----------


@pytest.fixture
def mock_event_bus():
    """Create a mock EventBus for testing."""
    bus = MagicMock(spec=EventBus)
    bus.subscribe = AsyncMock()
    bus.unsubscribe = AsyncMock()
    bus.publish = AsyncMock()
    bus.start = AsyncMock()
    bus.stop = AsyncMock()
    return bus


@pytest.fixture
def settings():
    """Create test settings."""
    return Settings(
        # Regime settings
        regime_persistence_bars=2,
        regime_confidence_minimum=0.5,
        regime_adx_range_threshold=25.0,
        regime_adx_trend_threshold=35.0,
        regime_atr_expansion_threshold=1.2,
        # Signal settings
        signal_neutral_threshold=0.65,
        signal_horizon_bars=12,  # 1 hour at 5m bars
    )


@pytest.fixture
def mock_regime_classifier():
    """Create a mock RegimeClassifier."""
    classifier = MagicMock(spec=RegimeClassifier)
    classifier.is_ready = True
    classifier.classify = MagicMock()
    return classifier


# ---------- Integration Tests ----------


class TestRegimeSignalIntegration:
    """Test integration between RegimeClassifier and SignalGenerator."""

    @pytest.mark.asyncio
    async def test_regime_blocks_signal_when_should_trade_false(
        self, mock_event_bus, settings, mock_regime_classifier
    ):
        """Verify signals are not published when regime says don't trade."""
        # Setup regime to block trading
        mock_regime_classifier.classify.return_value = RegimeDecision(
            state=RegimeState.HIGH_VOL_TREND,
            confidence=0.3,  # Below minimum
            position_multiplier=0.5,
            should_trade=False,
            reason="Low confidence (0.30 < 0.50)",
        )

        # Create SignalGenerator with regime classifier
        generator = SignalGenerator(
            event_bus=mock_event_bus,
            settings=settings,
            regime_classifier=mock_regime_classifier,
        )

        # Mock the ensemble to return a valid signal
        with patch.object(type(generator._ensemble), "is_fitted", new_callable=PropertyMock, return_value=True), \
             patch.object(generator._ensemble, "generate_signal") as mock_generate:

            mock_generate.return_value = _make_trade_signal(
                direction=Direction.LONG,
                confidence=0.80,
            )

            # Process a feature event
            feature_event = _make_feature_vector(warmup_complete=True)
            await generator._on_feature_event(
                "features.xauusd",
                feature_event.to_dict()
            )

            # Verify regime was checked
            mock_regime_classifier.classify.assert_called_once()

            # Verify regime event was published
            regime_publish_calls = [
                call for call in mock_event_bus.publish.call_args_list
                if call[0][0].startswith("regime.")
            ]
            assert len(regime_publish_calls) == 1

            # Verify signal was NOT published (blocked by regime)
            signal_publish_calls = [
                call for call in mock_event_bus.publish.call_args_list
                if call[0][0].startswith("signal.")
            ]
            assert len(signal_publish_calls) == 0

            # Verify signal generation was never called (early return)
            mock_generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_regime_allows_signal_when_should_trade_true(
        self, mock_event_bus, settings, mock_regime_classifier
    ):
        """Verify signals are published when regime allows trading."""
        # Setup regime to allow trading
        mock_regime_classifier.classify.return_value = RegimeDecision(
            state=RegimeState.LOW_VOL_TREND,
            confidence=0.85,
            position_multiplier=1.5,
            should_trade=True,
            reason="low_vol_trend regime — London/NY overlap (1.2x)",
        )

        # Create SignalGenerator with regime classifier
        generator = SignalGenerator(
            event_bus=mock_event_bus,
            settings=settings,
            regime_classifier=mock_regime_classifier,
        )

        # Mock the ensemble to return a valid signal
        with patch.object(type(generator._ensemble), "is_fitted", new_callable=PropertyMock, return_value=True), \
             patch.object(generator._ensemble, "generate_signal") as mock_generate:

            trade_signal = _make_trade_signal(
                direction=Direction.LONG,
                confidence=0.80,
            )
            mock_generate.return_value = trade_signal

            # Process a feature event
            feature_event = _make_feature_vector(warmup_complete=True)
            await generator._on_feature_event(
                "features.xauusd",
                feature_event.to_dict()
            )

            # Verify regime was checked
            mock_regime_classifier.classify.assert_called_once()

            # Verify both regime and signal were published
            assert mock_event_bus.publish.call_count == 2

            # Check regime event
            regime_call = mock_event_bus.publish.call_args_list[0]
            assert regime_call[0][0] == "regime.xauusd"
            regime_data = regime_call[0][1]
            assert regime_data["decision"]["state"] == "low_vol_trend"
            assert regime_data["decision"]["confidence"] == 0.85
            assert regime_data["decision"]["should_trade"] is True

            # Check signal event
            signal_call = mock_event_bus.publish.call_args_list[1]
            assert signal_call[0][0] == "signal.xauusd"
            signal_data = signal_call[0][1]
            assert signal_data["direction"] == "long"
            assert signal_data["confidence"] == 0.80

    @pytest.mark.asyncio
    async def test_regime_event_published_with_position_multiplier(
        self, mock_event_bus, settings, mock_regime_classifier
    ):
        """Verify regime event includes position_multiplier."""
        # Setup regime with specific position multiplier
        mock_regime_classifier.classify.return_value = RegimeDecision(
            state=RegimeState.RANGING,
            confidence=0.70,
            position_multiplier=1.0,
            should_trade=True,
            reason="ranging regime — London session (1.0x)",
            adx_value=22.5,
            atr_expanding=False,
            session_bias=1.0,
        )

        # Create SignalGenerator with regime classifier
        generator = SignalGenerator(
            event_bus=mock_event_bus,
            settings=settings,
            regime_classifier=mock_regime_classifier,
        )

        # Mock the ensemble
        with patch.object(type(generator._ensemble), "is_fitted", new_callable=PropertyMock, return_value=True), \
             patch.object(generator._ensemble, "generate_signal") as mock_generate:

            mock_generate.return_value = _make_trade_signal(direction=Direction.NEUTRAL)

            # Process a feature event
            feature_event = _make_feature_vector(warmup_complete=True)
            await generator._on_feature_event(
                "features.xauusd",
                feature_event.to_dict()
            )

            # Verify regime event was published
            assert mock_event_bus.publish.called
            regime_call = mock_event_bus.publish.call_args_list[0]

            # Verify event structure
            assert regime_call[0][0] == "regime.xauusd"
            regime_data = regime_call[0][1]

            # Check all fields are present
            assert "decision" in regime_data
            decision = regime_data["decision"]
            assert decision["position_multiplier"] == 1.0
            assert decision["state"] == "ranging"
            assert decision["confidence"] == 0.70
            assert decision["adx_value"] == 22.5
            assert decision["atr_expanding"] is False
            assert decision["session_bias"] == 1.0

    @pytest.mark.asyncio
    async def test_no_regime_check_when_classifier_not_ready(
        self, mock_event_bus, settings, mock_regime_classifier
    ):
        """Verify regime is not checked when classifier is not ready."""
        # Setup regime as not ready (insufficient data)
        mock_regime_classifier.is_ready = False

        # Create SignalGenerator with regime classifier
        generator = SignalGenerator(
            event_bus=mock_event_bus,
            settings=settings,
            regime_classifier=mock_regime_classifier,
        )

        # Mock the ensemble
        with patch.object(type(generator._ensemble), "is_fitted", new_callable=PropertyMock, return_value=True), \
             patch.object(generator._ensemble, "generate_signal") as mock_generate:

            mock_generate.return_value = _make_trade_signal()

            # Process a feature event
            feature_event = _make_feature_vector(warmup_complete=True)
            await generator._on_feature_event(
                "features.xauusd",
                feature_event.to_dict()
            )

            # Verify regime was NOT checked
            mock_regime_classifier.classify.assert_not_called()

            # Verify no regime event was published
            regime_publish_calls = [
                call for call in mock_event_bus.publish.call_args_list
                if call[0][0].startswith("regime.")
            ]
            assert len(regime_publish_calls) == 0

            # Verify signal was still generated and published
            signal_publish_calls = [
                call for call in mock_event_bus.publish.call_args_list
                if call[0][0].startswith("signal.")
            ]
            assert len(signal_publish_calls) == 1

    @pytest.mark.asyncio
    async def test_no_regime_check_when_classifier_is_none(
        self, mock_event_bus, settings
    ):
        """Verify system works without regime classifier."""
        # Create SignalGenerator without regime classifier
        generator = SignalGenerator(
            event_bus=mock_event_bus,
            settings=settings,
            regime_classifier=None,  # No classifier
        )

        # Mock the ensemble
        with patch.object(type(generator._ensemble), "is_fitted", new_callable=PropertyMock, return_value=True), \
             patch.object(generator._ensemble, "generate_signal") as mock_generate:

            mock_generate.return_value = _make_trade_signal()

            # Process a feature event
            feature_event = _make_feature_vector(warmup_complete=True)
            await generator._on_feature_event(
                "features.xauusd",
                feature_event.to_dict()
            )

            # Verify signal was generated normally
            mock_generate.assert_called_once()

            # Verify only signal event was published (no regime event)
            assert mock_event_bus.publish.call_count == 1
            signal_call = mock_event_bus.publish.call_args_list[0]
            assert signal_call[0][0] == "signal.xauusd"

    @pytest.mark.asyncio
    async def test_make_bar_from_features_creates_valid_bar(
        self, mock_event_bus, settings, mock_regime_classifier
    ):
        """Verify _make_bar_from_features creates a valid NormalizedBar."""
        generator = SignalGenerator(
            event_bus=mock_event_bus,
            settings=settings,
            regime_classifier=mock_regime_classifier,
        )

        # Create feature vector with specific values
        fv = _make_feature_vector(
            bb_middle=2045.50,
            atr_14=6.25,
            symbol="XAUUSD",
            timeframe=Timeframe.M5,
            timestamp=datetime(2025, 1, 15, 14, 30, 0, tzinfo=timezone.utc),
        )

        # Create bar from features
        bar = generator._make_bar_from_features(fv)

        # Verify bar properties
        assert bar.symbol == "XAUUSD"
        assert bar.timeframe == Timeframe.M5
        assert bar.timestamp == fv.timestamp
        assert float(bar.close) == 2045.50  # BB middle
        assert float(bar.open) == 2045.50
        assert float(bar.high) == 2045.50 + 6.25  # price + ATR
        assert float(bar.low) == 2045.50 - 6.25  # price - ATR
        assert float(bar.volume) == 0

        # Verify OHLC consistency (should not raise)
        assert bar.low <= bar.open <= bar.high
        assert bar.low <= bar.close <= bar.high

    @pytest.mark.asyncio
    async def test_neutral_signal_not_published(
        self, mock_event_bus, settings, mock_regime_classifier
    ):
        """Verify neutral signals are not published to EventBus."""
        # Setup regime to allow trading
        mock_regime_classifier.classify.return_value = RegimeDecision(
            state=RegimeState.RANGING,
            confidence=0.65,
            position_multiplier=1.0,
            should_trade=True,
            reason="ranging regime",
        )

        generator = SignalGenerator(
            event_bus=mock_event_bus,
            settings=settings,
            regime_classifier=mock_regime_classifier,
        )

        # Mock ensemble to return neutral signal
        with patch.object(type(generator._ensemble), "is_fitted", new_callable=PropertyMock, return_value=True), \
             patch.object(generator._ensemble, "generate_signal") as mock_generate:

            mock_generate.return_value = _make_trade_signal(
                direction=Direction.NEUTRAL,
                confidence=0.50,
            )

            # Process feature event
            feature_event = _make_feature_vector(warmup_complete=True)
            await generator._on_feature_event(
                "features.xauusd",
                feature_event.to_dict()
            )

            # Verify only regime event was published, not signal
            assert mock_event_bus.publish.call_count == 1
            regime_call = mock_event_bus.publish.call_args_list[0]
            assert regime_call[0][0] == "regime.xauusd"

    @pytest.mark.asyncio
    async def test_warmup_incomplete_skips_processing(
        self, mock_event_bus, settings, mock_regime_classifier
    ):
        """Verify features with incomplete warmup are skipped."""
        generator = SignalGenerator(
            event_bus=mock_event_bus,
            settings=settings,
            regime_classifier=mock_regime_classifier,
        )

        with patch.object(type(generator._ensemble), "is_fitted", new_callable=PropertyMock, return_value=True):
            # Process feature event with incomplete warmup
            feature_event = _make_feature_vector(warmup_complete=False)
            await generator._on_feature_event(
                "features.xauusd",
                feature_event.to_dict()
            )

            # Verify nothing was processed
            mock_regime_classifier.classify.assert_not_called()
            mock_event_bus.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_regime_decision_serialization_in_event(
        self, mock_event_bus, settings, mock_regime_classifier
    ):
        """Verify RegimeDecision is correctly serialized in RegimeEvent."""
        # Create a complete regime decision
        regime_decision = RegimeDecision(
            state=RegimeState.LOW_VOL_TREND,
            confidence=0.92,
            position_multiplier=1.5,
            should_trade=True,
            reason="Optimal trading conditions",
            hmm_prob=0.88,
            adx_value=42.3,
            atr_expanding=True,
            session_bias=1.2,
        )
        mock_regime_classifier.classify.return_value = regime_decision

        generator = SignalGenerator(
            event_bus=mock_event_bus,
            settings=settings,
            regime_classifier=mock_regime_classifier,
        )

        with patch.object(type(generator._ensemble), "is_fitted", new_callable=PropertyMock, return_value=True), \
             patch.object(generator._ensemble, "generate_signal") as mock_generate:

            mock_generate.return_value = _make_trade_signal(direction=Direction.NEUTRAL)

            # Process feature event
            feature_event = _make_feature_vector(warmup_complete=True)
            await generator._on_feature_event(
                "features.xauusd",
                feature_event.to_dict()
            )

            # Verify regime event was published with all fields
            regime_call = mock_event_bus.publish.call_args_list[0]
            regime_data = regime_call[0][1]

            # Check complete serialization
            decision_data = regime_data["decision"]
            assert decision_data["state"] == "low_vol_trend"
            assert decision_data["confidence"] == 0.92
            assert decision_data["position_multiplier"] == 1.5
            assert decision_data["should_trade"] is True
            assert decision_data["reason"] == "Optimal trading conditions"
            assert decision_data["hmm_prob"] == 0.88
            assert decision_data["adx_value"] == 42.3
            assert decision_data["atr_expanding"] is True
            assert decision_data["session_bias"] == 1.2

            # Check event metadata
            assert regime_data["symbol"] == "XAUUSD"
            assert "timestamp" in regime_data
            assert "event_time" in regime_data


class TestRealRegimeClassifierIntegration:
    """Test with a real RegimeClassifier instance (not mocked)."""

    @pytest.mark.asyncio
    async def test_real_regime_classifier_flow(self, mock_event_bus, settings):
        """Test the full flow with a real RegimeClassifier."""
        # Create real classifier
        regime_classifier = RegimeClassifier(settings=settings)

        # Create generator with real classifier
        generator = SignalGenerator(
            event_bus=mock_event_bus,
            settings=settings,
            regime_classifier=regime_classifier,
        )

        # Mock only the ensemble
        with patch.object(type(generator._ensemble), "is_fitted", new_callable=PropertyMock, return_value=True), \
             patch.object(generator._ensemble, "generate_signal") as mock_generate:

            mock_generate.return_value = _make_trade_signal()

            # Process multiple feature events to warm up the classifier
            for i in range(5):
                fv = _make_feature_vector(
                    warmup_complete=True,
                    atr_14=5.0 + i * 0.5,  # Varying ATR
                    timestamp=datetime(2025, 1, 15, 12, i * 5, 0, tzinfo=timezone.utc),
                )
                await generator._on_feature_event(
                    "features.xauusd",
                    fv.to_dict()
                )

            # After warmup, classifier should be ready (needs at least 2 ATR values)
            # Check if classifier is ready or has at least processed data

            # Verify regime events were published once ready
            regime_events = [
                call for call in mock_event_bus.publish.call_args_list
                if call[0][0].startswith("regime.")
            ]

            # If classifier became ready, we should have regime events
            if regime_classifier.is_ready and len(regime_events) > 0:
                # Should have regime events for bars where classifier was ready (at least last 3)
                assert len(regime_events) >= 3

                # Check last regime event has valid decision
                last_regime = regime_events[-1][0][1]
                decision = last_regime["decision"]
                assert decision["state"] in ["low_vol_trend", "high_vol_trend", "ranging"]
                assert 0.0 <= decision["confidence"] <= 1.0
                assert decision["position_multiplier"] > 0
                assert isinstance(decision["should_trade"], bool)
                assert decision["reason"] is not None
            else:
                # If not ready yet, at least verify signal events were published
                signal_events = [
                    call for call in mock_event_bus.publish.call_args_list
                    if call[0][0].startswith("signal.")
                ]
                assert len(signal_events) >= 5  # All 5 feature events should generate signals

    @pytest.mark.asyncio
    async def test_regime_persistence_behavior(self, mock_event_bus, settings):
        """Test that regime persistence prevents whipsaws."""
        # Create classifier with persistence_bars=2
        settings.regime_persistence_bars = 2
        regime_classifier = RegimeClassifier(settings=settings)

        # Mock the sub-components to control proposed states
        with patch.object(regime_classifier._intraday_filter, "classify") as mock_intraday:
            # Start with RANGING
            mock_intraday.return_value = (RegimeState.RANGING, 0.7)

            # Warm up the classifier
            for i in range(3):
                bar = _make_normalized_bar()
                fv = _make_feature_vector(atr_14=5.0)
                decision = regime_classifier.classify(bar, fv)

            # Current state should be RANGING
            assert regime_classifier._current_state == RegimeState.RANGING

            # Change to LOW_VOL_TREND but only for 1 bar
            mock_intraday.return_value = (RegimeState.LOW_VOL_TREND, 0.8)
            bar = _make_normalized_bar()
            fv = _make_feature_vector(atr_14=5.0)
            decision1 = regime_classifier.classify(bar, fv)

            # Should still be RANGING (persistence)
            assert decision1.state == RegimeState.RANGING

            # Second bar with LOW_VOL_TREND
            decision2 = regime_classifier.classify(bar, fv)

            # Now should switch (2 bars of persistence met)
            assert decision2.state == RegimeState.LOW_VOL_TREND
            assert regime_classifier._current_state == RegimeState.LOW_VOL_TREND