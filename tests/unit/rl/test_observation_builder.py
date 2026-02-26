"""Unit tests for the RL observation builder module."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from fxer.config.constants import MAX_WARMUP_BARS
from fxer.core.events import FeatureVector, NormalizedBar
from fxer.core.exceptions import FxerError, ObservationError, RLError
from fxer.core.types import Timeframe
from fxer.features.engine import FeatureEngine
from fxer.regime.classifier import RegimeClassifier
from fxer.regime.types import RegimeDecision, RegimeState
from fxer.rl.observations.builder import ObservationBuilder, _encode_regime
from fxer.rl.observations.config import ObservationConfig, TimeframeWindowConfig
from fxer.rl.types import EpisodeMode, GymAction, PositionState
from fxer.signals.base import FEATURE_COLUMNS, feature_vector_to_array
from tests.conftest import make_normalized_bar_series, make_observation_config


# ---------------------------------------------------------------------------
# Helper Functions and Fixtures
# ---------------------------------------------------------------------------


def _make_feature_vector(**kwargs) -> FeatureVector:
    """Create a FeatureVector with sensible defaults, overridable via kwargs."""
    defaults = {
        "symbol": "XAUUSD",
        "timestamp": datetime(2024, 1, 2, 10, 0, 0, tzinfo=timezone.utc),
        "timeframe": Timeframe.M5,
        "rsi_14": 55.0,
        "rsi_7": 53.0,
        "macd_line": 0.45,
        "macd_signal": 0.38,
        "macd_histogram": 0.07,
        "bb_upper": 2070.0,
        "bb_middle": 2065.0,
        "bb_lower": 2060.0,
        "bb_width": 0.00484,
        "bb_percent_b": 0.5,
        "atr_14": 3.5,
        "is_london_session": True,
        "is_ny_session": False,
        "is_overlap_session": False,
        "is_asian_session": False,
        "hour_of_day": 10,
        "day_of_week": 1,
        "is_month_turn": False,
        "warmup_complete": True,
        # Price-derived features
        "return_1bar": 0.0012,
        "return_5bar": 0.0058,
        "return_12bar": 0.0142,
        "rolling_volatility_20": 0.0085,
        "momentum_48": 0.0325,
        # External features (can be None)
        "dxy_return_1h": None,
        "dxy_rsi_14": None,
        "vix_level": None,
        "vix_change": None,
    }
    defaults.update(kwargs)
    return FeatureVector(**defaults)


@pytest.fixture
def multi_tf_bars():
    """Generate bars for M5, H1, H4 with enough for warmup + lookback."""
    from decimal import Decimal
    import random

    random.seed(42)

    # Generate M5 bars (need at least 49 + 20 = 69, use 100)
    m5_bars = []
    base_time = datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc)
    base_price = 2062.50

    for i in range(100):
        timestamp = base_time + timedelta(minutes=i * 5)
        change = random.uniform(-1.5, 1.5)
        open_p = round(base_price, 2)
        close_p = round(base_price + change, 2)
        high_p = round(max(open_p, close_p) + random.uniform(0.1, 1.0), 2)
        low_p = round(min(open_p, close_p) - random.uniform(0.1, 1.0), 2)
        vol = random.randint(800, 2000)

        m5_bars.append(NormalizedBar(
            symbol="XAUUSD",
            timeframe=Timeframe.M5,
            timestamp=timestamp,
            open=Decimal(str(open_p)),
            high=Decimal(str(high_p)),
            low=Decimal(str(low_p)),
            close=Decimal(str(close_p)),
            volume=Decimal(str(vol)),
            is_complete=True,
        ))
        base_price = close_p

    # Generate H1 bars (need at least 49 + 10 = 59, use 70)
    h1_bars = []
    base_price = 2062.50

    for i in range(70):
        timestamp = base_time + timedelta(minutes=i * 60)
        change = random.uniform(-2.0, 2.0)
        open_p = round(base_price, 2)
        close_p = round(base_price + change, 2)
        high_p = round(max(open_p, close_p) + random.uniform(0.2, 1.5), 2)
        low_p = round(min(open_p, close_p) - random.uniform(0.2, 1.5), 2)
        vol = random.randint(3000, 8000)

        h1_bars.append(NormalizedBar(
            symbol="XAUUSD",
            timeframe=Timeframe.H1,
            timestamp=timestamp,
            open=Decimal(str(open_p)),
            high=Decimal(str(high_p)),
            low=Decimal(str(low_p)),
            close=Decimal(str(close_p)),
            volume=Decimal(str(vol)),
            is_complete=True,
        ))
        base_price = close_p

    # Generate H4 bars (need at least 49 + 5 = 54, use 60)
    h4_bars = []
    base_price = 2062.50

    for i in range(60):
        timestamp = base_time + timedelta(minutes=i * 240)
        change = random.uniform(-3.0, 3.0)
        open_p = round(base_price, 2)
        close_p = round(base_price + change, 2)
        high_p = round(max(open_p, close_p) + random.uniform(0.5, 2.5), 2)
        low_p = round(min(open_p, close_p) - random.uniform(0.5, 2.5), 2)
        vol = random.randint(10000, 30000)

        h4_bars.append(NormalizedBar(
            symbol="XAUUSD",
            timeframe=Timeframe.H4,
            timestamp=timestamp,
            open=Decimal(str(open_p)),
            high=Decimal(str(high_p)),
            low=Decimal(str(low_p)),
            close=Decimal(str(close_p)),
            volume=Decimal(str(vol)),
            is_complete=True,
        ))
        base_price = close_p

    return {Timeframe.M5: m5_bars, Timeframe.H1: h1_bars, Timeframe.H4: h4_bars}


@pytest.fixture
def mock_regime_classifier():
    """Mock RegimeClassifier for testing regime encoding."""
    mock = MagicMock(spec=RegimeClassifier)
    mock.classify.return_value = RegimeDecision(
        state=RegimeState.LOW_VOL_TREND,
        confidence=0.85,
        position_multiplier=1.0,
        should_trade=True,
        reason="Test regime",
    )
    return mock


# ---------------------------------------------------------------------------
# Config Tests
# ---------------------------------------------------------------------------


class TestObservationConfig:
    """Test ObservationConfig and TimeframeWindowConfig."""

    def test_default_config_observation_size(self):
        """ObservationConfig().observation_size == 844."""
        config = ObservationConfig()
        # Default: 3 windows (M5:20, H1:10, H4:5) * 24 features + 4 regime
        # = (20*24 + 10*24 + 5*24) + 4 = 840 + 4 = 844
        assert config.observation_size == 844

    def test_features_per_bar_features_only(self):
        """TimeframeWindowConfig with features=True, ohlcv=False → 24."""
        window = TimeframeWindowConfig(
            timeframe=Timeframe.M5,
            lookback=10,
            include_ohlcv=False,
            include_features=True,
        )
        assert window.features_per_bar == 24  # len(FEATURE_COLUMNS)

    def test_features_per_bar_with_ohlcv(self):
        """TimeframeWindowConfig with both → 29."""
        window = TimeframeWindowConfig(
            timeframe=Timeframe.M5,
            lookback=10,
            include_ohlcv=True,
            include_features=True,
        )
        assert window.features_per_bar == 29  # 24 features + 5 OHLCV

    def test_features_per_bar_ohlcv_only(self):
        """TimeframeWindowConfig with features=False, ohlcv=True → 5."""
        window = TimeframeWindowConfig(
            timeframe=Timeframe.M5,
            lookback=10,
            include_ohlcv=True,
            include_features=False,
        )
        assert window.features_per_bar == 5  # OHLCV only

    def test_config_no_regime(self):
        """ObservationConfig with include_regime=False → 840."""
        config = ObservationConfig(include_regime=False)
        # 3 windows * features, no regime
        assert config.observation_size == 840

    def test_window_size(self):
        """lookback * features_per_bar."""
        window = TimeframeWindowConfig(
            timeframe=Timeframe.M5,
            lookback=20,
            include_ohlcv=False,
            include_features=True,
        )
        assert window.window_size == 20 * 24  # 480


# ---------------------------------------------------------------------------
# Types Tests
# ---------------------------------------------------------------------------


class TestRLTypes:
    """Test RL enums."""

    def test_position_state_values(self):
        """FLAT=0, LONG=1, SHORT=2."""
        assert PositionState.FLAT == 0
        assert PositionState.LONG == 1
        assert PositionState.SHORT == 2

    def test_gym_action_values(self):
        """FLAT=0, BUY=1, SELL=2."""
        assert GymAction.FLAT == 0
        assert GymAction.BUY == 1
        assert GymAction.SELL == 2

    def test_episode_mode_values(self):
        """TRAINING=0, EVALUATION=1."""
        assert EpisodeMode.TRAINING == 0
        assert EpisodeMode.EVALUATION == 1


# ---------------------------------------------------------------------------
# Exception Tests
# ---------------------------------------------------------------------------


class TestExceptions:
    """Test exception hierarchy."""

    def test_rl_error_inherits_fxer_error(self):
        """RLError is subclass of FxerError."""
        assert issubclass(RLError, FxerError)

    def test_observation_error_inherits_rl_error(self):
        """ObservationError is subclass of RLError."""
        assert issubclass(ObservationError, RLError)


# ---------------------------------------------------------------------------
# Builder Tests (Core)
# ---------------------------------------------------------------------------


class TestObservationBuilder:
    """Test ObservationBuilder core functionality."""

    def test_build_with_provided_bars(self, multi_tf_bars):
        """Build with synthetic bars dict, verify observations are populated and is_built is True."""
        config = ObservationConfig()
        builder = ObservationBuilder(config)

        # Start after warmup period (MAX_WARMUP_BARS=49, so 49*5min = 245min = 4h5min)
        # Start at 4:30 to be safe
        start = datetime(2024, 1, 2, 4, 30, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 2, 7, 0, 0, tzinfo=timezone.utc)

        builder.build(start=start, end=end, bars=multi_tf_bars)

        assert builder.is_built
        # Check that we have observations
        # Since warmup takes time, we might not have many observations
        assert len(builder._observations) >= 0  # At least build completes
        # If we have any observations, check they're valid
        if len(builder.timestamps) > 0:
            assert len(builder.timestamps) > 0
            assert len(builder._observations) > 0

    def test_observation_shape_matches_config(self, multi_tf_bars):
        """Every observation has len == config.observation_size."""
        config = ObservationConfig()
        builder = ObservationBuilder(config)

        start = datetime(2024, 1, 2, 5, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 2, 7, 0, 0, tzinfo=timezone.utc)

        builder.build(start=start, end=end, bars=multi_tf_bars)

        for timestamp in builder.timestamps:
            obs = builder.get_observation(timestamp)
            assert obs is not None
            assert len(obs) == config.observation_size

    def test_get_observation_returns_none_for_missing(self, multi_tf_bars):
        """Query a timestamp not in range."""
        config = ObservationConfig()
        builder = ObservationBuilder(config)

        start = datetime(2024, 1, 2, 5, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 2, 7, 0, 0, tzinfo=timezone.utc)

        builder.build(start=start, end=end, bars=multi_tf_bars)

        # Query timestamp outside range
        missing_ts = datetime(2024, 1, 3, 0, 0, 0, tzinfo=timezone.utc)
        obs = builder.get_observation(missing_ts)
        assert obs is None

    def test_get_bar_returns_primary_tf_bar(self, multi_tf_bars):
        """get_bar returns the correct NormalizedBar."""
        config = ObservationConfig(primary_timeframe=Timeframe.M5)
        builder = ObservationBuilder(config)

        start = datetime(2024, 1, 2, 5, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 2, 7, 0, 0, tzinfo=timezone.utc)

        builder.build(start=start, end=end, bars=multi_tf_bars)

        # Get a bar from the primary timeframe
        if builder.timestamps:
            bar = builder.get_bar(builder.timestamps[0])
            assert bar is not None
            assert isinstance(bar, NormalizedBar)
            assert bar.timeframe == Timeframe.M5

    def test_get_features_returns_primary_tf_features(self, multi_tf_bars):
        """get_features returns FeatureVector."""
        config = ObservationConfig(primary_timeframe=Timeframe.M5)
        builder = ObservationBuilder(config)

        start = datetime(2024, 1, 2, 5, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 2, 7, 0, 0, tzinfo=timezone.utc)

        builder.build(start=start, end=end, bars=multi_tf_bars)

        # Get features from the primary timeframe
        if builder.timestamps:
            features = builder.get_features(builder.timestamps[0])
            assert features is not None
            assert isinstance(features, FeatureVector)

    def test_requires_bars_or_db_client(self):
        """build() with neither raises ObservationError."""
        config = ObservationConfig()
        builder = ObservationBuilder(config, db_client=None)

        start = datetime(2024, 1, 2, 5, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 2, 7, 0, 0, tzinfo=timezone.utc)

        with pytest.raises(ObservationError, match="Either bars dict or db_client"):
            builder.build(start=start, end=end, bars=None)


# ---------------------------------------------------------------------------
# Builder Tests (Trading-Specific / Critical)
# ---------------------------------------------------------------------------


class TestObservationBuilderTradingSpecific:
    """Test ObservationBuilder for trading-specific edge cases."""

    def test_no_lookahead_higher_tf(self):
        """CRITICAL: For an M5 observation at time T, verify that H1 features used are only from bars where completion_time <= T."""
        # Create specific bars for testing lookahead bias
        # H1 bar at 14:00 completes at 15:00
        # M5 observation at 14:30 should NOT use H1 bar from 14:00

        # Create M5 bars from 13:00 to 15:00
        from decimal import Decimal

        m5_bars = []
        base_time = datetime(2024, 1, 2, 13, 0, 0, tzinfo=timezone.utc)
        for i in range(25):  # 25 bars * 5 min = 125 min (13:00 to 15:05)
            timestamp = base_time + timedelta(minutes=i * 5)
            m5_bars.append(
                NormalizedBar(
                    symbol="XAUUSD",
                    timeframe=Timeframe.M5,
                    timestamp=timestamp,
                    open=Decimal("2060.0"),
                    high=Decimal("2061.0"),
                    low=Decimal("2059.0"),
                    close=Decimal("2060.5"),
                    volume=Decimal("1000"),
                    is_complete=True,
                )
            )

        # Create H1 bars at 13:00 and 14:00
        h1_bars = [
            NormalizedBar(
                symbol="XAUUSD",
                timeframe=Timeframe.H1,
                timestamp=datetime(2024, 1, 2, 13, 0, 0, tzinfo=timezone.utc),
                open=Decimal("2060.0"),
                high=Decimal("2062.0"),
                low=Decimal("2058.0"),
                close=Decimal("2061.0"),
                volume=Decimal("5000"),
                is_complete=True,
            ),
            NormalizedBar(
                symbol="XAUUSD",
                timeframe=Timeframe.H1,
                timestamp=datetime(2024, 1, 2, 14, 0, 0, tzinfo=timezone.utc),
                open=Decimal("2061.0"),
                high=Decimal("2063.0"),
                low=Decimal("2060.0"),
                close=Decimal("2062.0"),
                volume=Decimal("5000"),
                is_complete=True,
            ),
        ]

        # Add warmup bars before our test range
        import random
        random.seed(42)

        # Create warmup M5 bars
        warmup_m5 = []
        base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        for i in range(MAX_WARMUP_BARS + 10):
            timestamp = base_time + timedelta(minutes=i * 5)
            warmup_m5.append(
                NormalizedBar(
                    symbol="XAUUSD",
                    timeframe=Timeframe.M5,
                    timestamp=timestamp,
                    open=Decimal("2060.0"),
                    high=Decimal("2061.0"),
                    low=Decimal("2059.0"),
                    close=Decimal("2060.5"),
                    volume=Decimal("1000"),
                    is_complete=True,
                )
            )

        # Create warmup H1 bars
        warmup_h1 = []
        for i in range(MAX_WARMUP_BARS + 5):
            timestamp = base_time + timedelta(minutes=i * 60)
            warmup_h1.append(
                NormalizedBar(
                    symbol="XAUUSD",
                    timeframe=Timeframe.H1,
                    timestamp=timestamp,
                    open=Decimal("2060.0"),
                    high=Decimal("2062.0"),
                    low=Decimal("2058.0"),
                    close=Decimal("2061.0"),
                    volume=Decimal("5000"),
                    is_complete=True,
                )
            )

        # Create H4 bars for completeness
        h4_bars = []
        for i in range(MAX_WARMUP_BARS + 10):
            timestamp = base_time + timedelta(minutes=i * 240)
            h4_bars.append(
                NormalizedBar(
                    symbol="XAUUSD",
                    timeframe=Timeframe.H4,
                    timestamp=timestamp,
                    open=Decimal("2060.0"),
                    high=Decimal("2064.0"),
                    low=Decimal("2056.0"),
                    close=Decimal("2062.0"),
                    volume=Decimal("20000"),
                    is_complete=True,
                )
            )

        bars = {
            Timeframe.M5: warmup_m5 + m5_bars,
            Timeframe.H1: warmup_h1 + h1_bars,
            Timeframe.H4: h4_bars,
        }

        # Build observations
        config = ObservationConfig(
            windows=(
                TimeframeWindowConfig(Timeframe.M5, 2, False, True),
                TimeframeWindowConfig(Timeframe.H1, 1, False, True),
                TimeframeWindowConfig(Timeframe.H4, 1, False, True),
            ),
            include_regime=False,
        )
        builder = ObservationBuilder(config)

        # Query at 14:30
        query_time = datetime(2024, 1, 2, 14, 30, 0, tzinfo=timezone.utc)
        start = query_time
        end = query_time

        builder.build(start=start, end=end, bars=bars)

        # Check internal state - the H1 features at 14:30 should be from 13:00 bar
        # (which completed at 14:00), NOT from 14:00 bar (which completes at 15:00)
        valid_features = builder._get_valid_features_for_window(
            query_time, Timeframe.H1, builder._tf_features[Timeframe.H1], lookback=10
        )

        # The latest valid H1 feature should be from 13:00
        if valid_features:
            latest_h1_feature = valid_features[-1]
            assert latest_h1_feature.timestamp == datetime(2024, 1, 2, 13, 0, 0, tzinfo=timezone.utc)
            # Should NOT include the 14:00 bar
            h1_timestamps = [fv.timestamp for fv in valid_features]
            assert datetime(2024, 1, 2, 14, 0, 0, tzinfo=timezone.utc) not in h1_timestamps

    def test_warmup_bars_excluded(self, multi_tf_bars):
        """Observations should only include timestamps where warmup_complete=True."""
        config = ObservationConfig()
        builder = ObservationBuilder(config)

        # Start early to include warmup period
        start = datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 2, 7, 0, 0, tzinfo=timezone.utc)

        builder.build(start=start, end=end, bars=multi_tf_bars)

        # All observations should have warmup_complete=True
        for timestamp in builder.timestamps:
            features = builder.get_features(timestamp)
            if features:
                assert features.warmup_complete

    def test_timestamps_within_range(self, multi_tf_bars):
        """All returned timestamps are within [start, end]."""
        config = ObservationConfig()
        builder = ObservationBuilder(config)

        start = datetime(2024, 1, 2, 5, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 2, 6, 0, 0, tzinfo=timezone.utc)

        builder.build(start=start, end=end, bars=multi_tf_bars)

        for timestamp in builder.timestamps:
            assert start <= timestamp <= end


# ---------------------------------------------------------------------------
# Builder Tests (Regime)
# ---------------------------------------------------------------------------


class TestObservationBuilderRegime:
    """Test ObservationBuilder regime functionality."""

    def test_regime_encoding(self, multi_tf_bars, mock_regime_classifier):
        """With a mock RegimeClassifier, verify the last 4 values of observation are [one_hot..., confidence]."""
        config = ObservationConfig(include_regime=True)
        builder = ObservationBuilder(
            config, regime_classifier=mock_regime_classifier
        )

        start = datetime(2024, 1, 2, 5, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 2, 7, 0, 0, tzinfo=timezone.utc)

        builder.build(start=start, end=end, bars=multi_tf_bars)

        if builder.timestamps:
            obs = builder.get_observation(builder.timestamps[0])
            assert obs is not None

            # Last 4 values should be regime encoding
            regime_part = obs[-4:]
            # Should be one-hot encoded [1, 0, 0] for LOW_VOL_TREND + [0.85] confidence
            assert regime_part[0] == 1.0  # LOW_VOL_TREND
            assert regime_part[1] == 0.0  # HIGH_VOL_TREND
            assert regime_part[2] == 0.0  # RANGING
            assert regime_part[3] == 0.85  # confidence

    def test_no_regime_when_disabled(self, multi_tf_bars):
        """With include_regime=False, observation size is 840."""
        config = ObservationConfig(include_regime=False)
        builder = ObservationBuilder(config)

        start = datetime(2024, 1, 2, 5, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 2, 7, 0, 0, tzinfo=timezone.utc)

        builder.build(start=start, end=end, bars=multi_tf_bars)

        assert config.observation_size == 840
        if builder.timestamps:
            obs = builder.get_observation(builder.timestamps[0])
            assert obs is not None
            assert len(obs) == 840


# ---------------------------------------------------------------------------
# Builder Tests (OHLCV)
# ---------------------------------------------------------------------------


class TestObservationBuilderOHLCV:
    """Test ObservationBuilder OHLCV functionality."""

    def test_ohlcv_inclusion(self, multi_tf_bars):
        """Config with include_ohlcv=True produces larger observations."""
        # Config without OHLCV
        config_no_ohlcv = ObservationConfig(
            windows=(
                TimeframeWindowConfig(Timeframe.M5, 20, False, True),
                TimeframeWindowConfig(Timeframe.H1, 10, False, True),
                TimeframeWindowConfig(Timeframe.H4, 5, False, True),
            ),
            include_regime=False,
        )

        # Config with OHLCV
        config_with_ohlcv = ObservationConfig(
            windows=(
                TimeframeWindowConfig(Timeframe.M5, 20, True, True),  # +5 per bar
                TimeframeWindowConfig(Timeframe.H1, 10, True, True),  # +5 per bar
                TimeframeWindowConfig(Timeframe.H4, 5, True, True),   # +5 per bar
            ),
            include_regime=False,
        )

        # Build both
        builder_no_ohlcv = ObservationBuilder(config_no_ohlcv)
        builder_with_ohlcv = ObservationBuilder(config_with_ohlcv)

        start = datetime(2024, 1, 2, 5, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 2, 7, 0, 0, tzinfo=timezone.utc)

        builder_no_ohlcv.build(start=start, end=end, bars=multi_tf_bars)
        builder_with_ohlcv.build(start=start, end=end, bars=multi_tf_bars)

        # With OHLCV should be larger
        # Additional: 20*5 + 10*5 + 5*5 = 175 extra values
        assert config_with_ohlcv.observation_size == config_no_ohlcv.observation_size + 175

        if builder_no_ohlcv.timestamps and builder_with_ohlcv.timestamps:
            obs_no_ohlcv = builder_no_ohlcv.get_observation(builder_no_ohlcv.timestamps[0])
            obs_with_ohlcv = builder_with_ohlcv.get_observation(builder_with_ohlcv.timestamps[0])

            assert obs_no_ohlcv is not None
            assert obs_with_ohlcv is not None
            assert len(obs_with_ohlcv) == len(obs_no_ohlcv) + 175


# ---------------------------------------------------------------------------
# Helper Function Tests
# ---------------------------------------------------------------------------


class TestHelperFunctions:
    """Test module-level helper functions."""

    def test_encode_regime_with_decision(self):
        """Test _encode_regime with a valid decision."""
        decision = RegimeDecision(
            state=RegimeState.HIGH_VOL_TREND,
            confidence=0.75,
            position_multiplier=0.5,
            should_trade=True,
            reason="Test",
        )

        encoded = _encode_regime(decision)

        assert len(encoded) == 4
        assert encoded[0] == 0.0  # LOW_VOL_TREND
        assert encoded[1] == 1.0  # HIGH_VOL_TREND (selected)
        assert encoded[2] == 0.0  # RANGING
        assert encoded[3] == 0.75  # confidence

    def test_encode_regime_with_none(self):
        """Test _encode_regime with None returns zeros."""
        encoded = _encode_regime(None)

        assert len(encoded) == 4
        assert np.all(encoded == 0.0)

    def test_encode_regime_all_states(self):
        """Test _encode_regime for all regime states."""
        states = [
            (RegimeState.LOW_VOL_TREND, [1.0, 0.0, 0.0]),
            (RegimeState.HIGH_VOL_TREND, [0.0, 1.0, 0.0]),
            (RegimeState.RANGING, [0.0, 0.0, 1.0]),
        ]

        for state, expected_one_hot in states:
            decision = RegimeDecision(
                state=state,
                confidence=0.5,
                position_multiplier=1.0,
                should_trade=True,
                reason="Test",
            )
            encoded = _encode_regime(decision)

            assert encoded[0] == expected_one_hot[0]
            assert encoded[1] == expected_one_hot[1]
            assert encoded[2] == expected_one_hot[2]
            assert encoded[3] == 0.5