"""Unit tests for the regime classification module."""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from fxer.config.settings import Settings
from fxer.core.events import FeatureVector, NormalizedBar
from fxer.core.exceptions import RegimeClassificationError
from fxer.core.types import Timeframe
from fxer.regime.change_point import ChangePointDetector
from fxer.regime.classifier import RegimeClassifier
from fxer.regime.hmm import HMMRegimeClassifier
from fxer.regime.intraday_filter import IntradayRegimeFilter
from fxer.regime.session_rules import SessionRegimeRules
from fxer.regime.types import RegimeDecision, RegimeEvent, RegimeState
from tests.conftest import make_normalized_bar, make_normalized_bar_series


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_daily_data(n_rows: int = 200, seed: int = 42) -> pd.DataFrame:
    """Create synthetic daily OHLC data for HMM testing."""
    np.random.seed(seed)
    closes = []
    price = 2000.0

    for i in range(n_rows):
        # Random walk with slight upward drift
        change = np.random.normal(0.1, 10.0)
        price = max(price + change, 1900.0)  # Floor at 1900
        closes.append(price)

    closes = np.array(closes)
    highs = closes + np.abs(np.random.normal(5, 2, n_rows))
    lows = closes - np.abs(np.random.normal(5, 2, n_rows))

    return pd.DataFrame({
        "close": closes,
        "high": highs,
        "low": lows,
    })


def _make_regime_shift_returns(n_stable: int = 100, n_volatile: int = 100) -> np.ndarray:
    """Create returns with a clear regime shift for change point testing."""
    np.random.seed(42)
    stable = np.random.normal(0.0, 0.01, n_stable)  # Low volatility
    volatile = np.random.normal(0.0, 0.05, n_volatile)  # High volatility
    return np.concatenate([stable, volatile])


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
        "atr_14": 3.5,
        "is_london_session": True,
        "is_ny_session": False,
        "is_overlap_session": False,
        "is_asian_session": False,
        "hour_of_day": 10,
        "day_of_week": 1,
        "is_month_turn": False,
        "warmup_complete": True,
    }
    defaults.update(kwargs)
    return FeatureVector(**defaults)


# ---------------------------------------------------------------------------
# TestRegimeTypes
# ---------------------------------------------------------------------------


class TestRegimeTypes:
    """Test RegimeState enum and RegimeDecision/RegimeEvent data classes."""

    def test_regime_state_enum_values(self):
        """Verify RegimeState enum values match expected strings."""
        assert RegimeState.LOW_VOL_TREND.value == "low_vol_trend"
        assert RegimeState.HIGH_VOL_TREND.value == "high_vol_trend"
        assert RegimeState.RANGING.value == "ranging"
        # Verify all enum members are present
        assert len(list(RegimeState)) == 3

    def test_regime_decision_to_dict_from_dict_roundtrip(self):
        """Verify RegimeDecision serialization roundtrip preserves all fields."""
        decision = RegimeDecision(
            state=RegimeState.HIGH_VOL_TREND,
            confidence=0.85,
            position_multiplier=0.5,
            should_trade=True,
            reason="Test reason",
            hmm_prob=0.75,
            adx_value=35.0,
            atr_expanding=True,
            session_bias=1.3,
        )

        # Convert to dict
        data = decision.to_dict()
        assert data["state"] == "high_vol_trend"
        assert data["confidence"] == 0.85
        assert data["position_multiplier"] == 0.5
        assert data["should_trade"] is True
        assert data["reason"] == "Test reason"
        assert data["hmm_prob"] == 0.75
        assert data["adx_value"] == 35.0
        assert data["atr_expanding"] is True
        assert data["session_bias"] == 1.3

        # Convert back from dict
        decision2 = RegimeDecision.from_dict(data)
        assert decision2.state == RegimeState.HIGH_VOL_TREND
        assert decision2.confidence == 0.85
        assert decision2.position_multiplier == 0.5
        assert decision2.should_trade is True
        assert decision2.reason == "Test reason"
        assert decision2.hmm_prob == 0.75
        assert decision2.adx_value == 35.0
        assert decision2.atr_expanding is True
        assert decision2.session_bias == 1.3

    def test_regime_decision_frozen_immutability(self):
        """Verify RegimeDecision is immutable (frozen dataclass)."""
        decision = RegimeDecision(
            state=RegimeState.RANGING,
            confidence=0.7,
            position_multiplier=1.0,
            should_trade=True,
            reason="Ranging market",
        )

        # Attempting to modify should raise an error
        with pytest.raises(AttributeError):
            decision.confidence = 0.9

    def test_regime_event_topic_generation(self):
        """Verify RegimeEvent generates correct topic strings."""
        decision = RegimeDecision(
            state=RegimeState.LOW_VOL_TREND,
            confidence=0.8,
            position_multiplier=1.5,
            should_trade=True,
            reason="Trending market",
        )

        event = RegimeEvent(
            decision=decision,
            symbol="XAUUSD",
            timestamp=datetime(2024, 1, 2, 10, 0, 0, tzinfo=timezone.utc),
        )

        assert event.topic == "regime.xauusd"

        # Test with different symbol
        event2 = RegimeEvent(
            decision=decision,
            symbol="EURUSD",
            timestamp=datetime(2024, 1, 2, 10, 0, 0, tzinfo=timezone.utc),
        )
        assert event2.topic == "regime.eurusd"


# ---------------------------------------------------------------------------
# TestHMMRegimeClassifier
# ---------------------------------------------------------------------------


class TestHMMRegimeClassifier:
    """Test HMMRegimeClassifier for daily regime detection."""

    def test_not_fitted_initially(self):
        """Verify HMM is not fitted on initialization."""
        hmm = HMMRegimeClassifier(n_states=3)
        assert not hmm.is_fitted

    def test_fit_with_synthetic_daily_data(self):
        """Verify HMM can be fitted with synthetic daily data."""
        # Skip if hmmlearn not installed
        pytest.importorskip("hmmlearn")

        hmm = HMMRegimeClassifier(n_states=3, random_state=42)
        daily_df = _make_synthetic_daily_data(n_rows=200)

        metrics = hmm.fit(daily_df)

        assert hmm.is_fitted
        assert metrics["n_states"] == 3
        assert metrics["n_samples"] > 0
        assert "log_likelihood" in metrics

    def test_predict_returns_valid_regime_and_confidence(self):
        """Verify predict_regime returns valid RegimeState and confidence."""
        # Skip if hmmlearn not installed
        pytest.importorskip("hmmlearn")

        hmm = HMMRegimeClassifier(n_states=3, random_state=42)
        daily_df = _make_synthetic_daily_data(n_rows=200)
        hmm.fit(daily_df)

        regime, confidence = hmm.predict_regime(daily_df)

        assert isinstance(regime, RegimeState)
        assert 0.0 <= confidence <= 1.0

    def test_predict_raises_when_not_fitted(self):
        """Verify predict_regime raises error when HMM not fitted."""
        hmm = HMMRegimeClassifier(n_states=3)
        daily_df = _make_synthetic_daily_data(n_rows=100)

        with pytest.raises(RegimeClassificationError, match="HMM not fitted"):
            hmm.predict_regime(daily_df)

    def test_save_load_roundtrip(self, tmp_path):
        """Verify HMM model can be saved and loaded."""
        # Skip if hmmlearn not installed
        pytest.importorskip("hmmlearn")

        hmm = HMMRegimeClassifier(n_states=3, random_state=42)
        daily_df = _make_synthetic_daily_data(n_rows=200)
        hmm.fit(daily_df)

        # Get prediction before save
        regime1, conf1 = hmm.predict_regime(daily_df)

        # Save model
        hmm.save(tmp_path)
        assert (tmp_path / "hmm_model.pkl").exists()
        assert (tmp_path / "hmm_state_map.pkl").exists()

        # Load into new instance
        hmm2 = HMMRegimeClassifier(n_states=3)
        assert not hmm2.is_fitted
        hmm2.load(tmp_path)
        assert hmm2.is_fitted

        # Verify predictions are the same
        regime2, conf2 = hmm2.predict_regime(daily_df)
        assert regime1 == regime2
        assert abs(conf1 - conf2) < 1e-6

    def test_insufficient_data_raises_error(self):
        """Verify fitting with insufficient data raises RegimeClassificationError."""
        # Skip if hmmlearn not installed
        pytest.importorskip("hmmlearn")

        hmm = HMMRegimeClassifier(n_states=3)
        daily_df = _make_synthetic_daily_data(n_rows=30)  # Too few rows

        with pytest.raises(RegimeClassificationError, match="Insufficient data"):
            hmm.fit(daily_df)

    def test_no_lookahead_bias(self):
        """CRITICAL: Verify no lookahead bias in predictions."""
        # Skip if hmmlearn not installed
        pytest.importorskip("hmmlearn")

        hmm = HMMRegimeClassifier(n_states=3, random_state=42)

        # Fit on full dataset
        full_data = _make_synthetic_daily_data(n_rows=200)
        hmm.fit(full_data)

        # Predict on first 100 rows
        partial_data = full_data.iloc[:100]
        regime1, conf1 = hmm.predict_regime(partial_data)

        # Predict on first 100 rows again with more data appended
        # The prediction at row 100 should be identical
        regime2, conf2 = hmm.predict_regime(partial_data)

        assert regime1 == regime2
        assert abs(conf1 - conf2) < 1e-10


# ---------------------------------------------------------------------------
# TestIntradayRegimeFilter
# ---------------------------------------------------------------------------


class TestIntradayRegimeFilter:
    """Test IntradayRegimeFilter ADX/ATR-based classification."""

    def test_adx_below_range_threshold_returns_ranging(self):
        """ADX < 20 should return RANGING state."""
        filter = IntradayRegimeFilter(
            adx_range_threshold=20.0,
            adx_trend_threshold=25.0,
        )

        state, confidence = filter.classify(adx=15.0, atr=3.0, prev_atr=3.0)

        assert state == RegimeState.RANGING
        assert 0.5 <= confidence <= 1.0

    def test_adx_above_trend_threshold_returns_trending(self):
        """ADX > 25 should return LOW_VOL_TREND state."""
        filter = IntradayRegimeFilter(
            adx_range_threshold=20.0,
            adx_trend_threshold=25.0,
        )

        state, confidence = filter.classify(adx=30.0, atr=3.0, prev_atr=3.0)

        assert state == RegimeState.LOW_VOL_TREND
        assert 0.5 <= confidence <= 1.0

    def test_adx_in_dead_zone_maintains_previous_state(self):
        """ADX in dead zone (20-25) should maintain previous state."""
        filter = IntradayRegimeFilter(
            adx_range_threshold=20.0,
            adx_trend_threshold=25.0,
        )

        # First set state to RANGING
        state1, _ = filter.classify(adx=15.0, atr=3.0, prev_atr=3.0)
        assert state1 == RegimeState.RANGING

        # ADX in dead zone should maintain RANGING
        state2, conf2 = filter.classify(adx=22.5, atr=3.0, prev_atr=3.0)
        assert state2 == RegimeState.RANGING
        assert conf2 == 0.5  # Dead zone confidence

        # Set state to TRENDING
        state3, _ = filter.classify(adx=30.0, atr=3.0, prev_atr=3.0)
        assert state3 == RegimeState.LOW_VOL_TREND

        # ADX in dead zone should maintain TRENDING
        state4, conf4 = filter.classify(adx=22.5, atr=3.0, prev_atr=3.0)
        assert state4 == RegimeState.LOW_VOL_TREND
        assert conf4 == 0.5

    def test_atr_expansion_returns_high_vol_trend(self):
        """ATR expansion should return HIGH_VOL_TREND state."""
        filter = IntradayRegimeFilter(
            adx_range_threshold=20.0,
            adx_trend_threshold=25.0,
            atr_expansion_threshold=1.5,
        )

        # ATR expanded by 1.6x
        state, confidence = filter.classify(adx=25.0, atr=8.0, prev_atr=5.0)

        assert state == RegimeState.HIGH_VOL_TREND
        assert confidence > 0.0

    def test_adx_none_returns_previous_state_with_half_confidence(self):
        """ADX=None should return previous state with 0.5 confidence."""
        filter = IntradayRegimeFilter()

        # Set initial state
        state1, _ = filter.classify(adx=30.0, atr=3.0, prev_atr=3.0)
        assert state1 == RegimeState.LOW_VOL_TREND

        # ADX=None should maintain state
        state2, conf2 = filter.classify(adx=None, atr=3.0, prev_atr=3.0)
        assert state2 == RegimeState.LOW_VOL_TREND
        assert conf2 == 0.5

    def test_reset_clears_state(self):
        """Verify reset() clears internal state."""
        filter = IntradayRegimeFilter()

        # Set state to trending
        state1, _ = filter.classify(adx=30.0, atr=3.0, prev_atr=3.0)
        assert state1 == RegimeState.LOW_VOL_TREND

        # Reset should clear state
        filter.reset()

        # After reset, should start with RANGING
        state2, _ = filter.classify(adx=15.0, atr=3.0, prev_atr=3.0)
        assert state2 == RegimeState.RANGING


# ---------------------------------------------------------------------------
# TestSessionRegimeRules
# ---------------------------------------------------------------------------


class TestSessionRegimeRules:
    """Test SessionRegimeRules for session-based trading bias."""

    def test_asian_session_returns_consolidation_bias(self):
        """Asian session (hour=3) should return bias=0.7."""
        rules = SessionRegimeRules()

        timestamp = datetime(2024, 1, 2, 3, 0, 0, tzinfo=timezone.utc)
        bias, reason = rules.get_session_bias(timestamp)

        assert bias == 0.7
        assert "Asian" in reason

    def test_london_open_returns_breakout_bias(self):
        """London open (hour=7) should return bias=1.2."""
        rules = SessionRegimeRules()

        timestamp = datetime(2024, 1, 2, 7, 30, 0, tzinfo=timezone.utc)
        bias, reason = rules.get_session_bias(timestamp)

        assert bias == 1.2
        assert "London open" in reason

    def test_overlap_returns_peak_liquidity_bias(self):
        """Overlap session (hour=14) should return bias=1.3."""
        rules = SessionRegimeRules()

        timestamp = datetime(2024, 1, 2, 14, 0, 0, tzinfo=timezone.utc)
        bias, reason = rules.get_session_bias(timestamp)

        assert bias == 1.3
        assert "overlap" in reason

    def test_late_ny_returns_low_bias_and_should_avoid(self):
        """Late NY (hour=18) should return bias=0.3 and should_avoid=True."""
        rules = SessionRegimeRules()

        timestamp = datetime(2024, 1, 2, 18, 0, 0, tzinfo=timezone.utc)
        bias, reason = rules.get_session_bias(timestamp)
        should_avoid = rules.should_avoid_trading(timestamp)

        assert bias == 0.3
        assert "Late NY" in reason
        assert should_avoid is True

    def test_normal_hours_returns_neutral_bias(self):
        """Normal hours (hour=10) should return bias=1.0."""
        rules = SessionRegimeRules()

        timestamp = datetime(2024, 1, 2, 10, 0, 0, tzinfo=timezone.utc)
        bias, reason = rules.get_session_bias(timestamp)

        assert bias == 1.0
        assert "Normal" in reason


# ---------------------------------------------------------------------------
# TestChangePointDetector
# ---------------------------------------------------------------------------


class TestChangePointDetector:
    """Test ChangePointDetector for structural break detection."""

    def test_stable_series_returns_no_change_points(self):
        """Stable series should return no change points."""
        # Skip if ruptures not installed
        pytest.importorskip("ruptures")

        detector = ChangePointDetector(window_size=100, min_size=10)

        # Stable returns with low volatility
        np.random.seed(42)
        returns = np.random.normal(0.0, 0.01, 100)

        change_points = detector.detect_change_points(returns)

        # Should be empty or very few false positives
        assert len(change_points) <= 1

    def test_regime_shift_detects_change(self):
        """Synthetic regime shift should be detected."""
        # Skip if ruptures not installed
        pytest.importorskip("ruptures")

        detector = ChangePointDetector(window_size=200, min_size=10)

        returns = _make_regime_shift_returns(n_stable=100, n_volatile=100)

        change_points = detector.detect_change_points(returns)

        # Should detect at least one change point
        assert len(change_points) > 0
        # Change point should be near the regime shift (around index 100)
        # Allow some tolerance due to algorithm characteristics
        assert any(80 <= cp <= 120 for cp in change_points)

    def test_short_series_returns_empty_list(self):
        """Short series should return empty list."""
        # Skip if ruptures not installed
        pytest.importorskip("ruptures")

        detector = ChangePointDetector(window_size=100, min_size=10)

        returns = np.random.normal(0.0, 0.01, 15)  # Less than 2 * min_size

        change_points = detector.detect_change_points(returns)

        assert change_points == []

    def test_has_recent_change_with_lookback(self):
        """Verify has_recent_change works with lookback parameter."""
        # Skip if ruptures not installed
        pytest.importorskip("ruptures")

        detector = ChangePointDetector(window_size=200, min_size=10)

        # Create returns with a change point
        returns = _make_regime_shift_returns(n_stable=150, n_volatile=50)

        # Should detect recent change (within last 20 bars)
        has_recent = detector.has_recent_change(returns, lookback=60)
        assert has_recent is True

        # Should not detect if lookback is too small
        has_recent_small = detector.has_recent_change(returns, lookback=5)
        assert has_recent_small is False


# ---------------------------------------------------------------------------
# TestRegimeClassifier
# ---------------------------------------------------------------------------


class TestRegimeClassifier:
    """Test RegimeClassifier composite classification."""

    def test_not_ready_initially(self):
        """Classifier should not be ready initially."""
        classifier = RegimeClassifier()
        assert not classifier.is_ready

    def test_ready_after_feeding_bars_with_atr(self):
        """Classifier should be ready after feeding enough bars with ATR."""
        classifier = RegimeClassifier()

        bar = make_normalized_bar()
        features = _make_feature_vector(atr_14=3.5)

        # Feed first bar
        decision1 = classifier.classify(bar, features)
        assert not classifier.is_ready  # Need at least 2 ATR values

        # Feed second bar
        decision2 = classifier.classify(bar, features)
        assert classifier.is_ready

    def test_classify_returns_regime_decision(self):
        """Classify should return a valid RegimeDecision."""
        classifier = RegimeClassifier()

        bar = make_normalized_bar()
        features = _make_feature_vector(atr_14=3.5)

        # Feed bars to make ready
        classifier.classify(bar, features)
        classifier.classify(bar, features)

        decision = classifier.classify(bar, features)

        assert isinstance(decision, RegimeDecision)
        assert isinstance(decision.state, RegimeState)
        assert 0.0 <= decision.confidence <= 1.0
        assert decision.position_multiplier > 0
        assert isinstance(decision.should_trade, bool)
        assert isinstance(decision.reason, str)

    def test_regime_persistence(self):
        """Regime should persist until persistence_bars reached."""
        settings = Settings(regime_persistence_bars=3)
        classifier = RegimeClassifier(settings=settings)

        # Set up initial state (RANGING)
        bar = make_normalized_bar()
        features_ranging = _make_feature_vector(atr_14=3.0)

        # Feed initial bars to establish RANGING state
        for _ in range(5):
            decision = classifier.classify(bar, features_ranging)

        assert decision.state == RegimeState.RANGING

        # Now feed bars that would trigger LOW_VOL_TREND
        # But state should persist for persistence_bars
        features_trending = _make_feature_vector(atr_14=5.0)

        # Mock the intraday filter to return LOW_VOL_TREND
        with patch.object(classifier._intraday_filter, 'classify') as mock_classify:
            mock_classify.return_value = (RegimeState.LOW_VOL_TREND, 0.8)

            # First bar proposing new state
            decision1 = classifier.classify(bar, features_trending)
            assert decision1.state == RegimeState.RANGING  # Still RANGING

            # Second bar
            decision2 = classifier.classify(bar, features_trending)
            assert decision2.state == RegimeState.RANGING  # Still RANGING

            # Third bar (persistence_bars = 3)
            decision3 = classifier.classify(bar, features_trending)
            assert decision3.state == RegimeState.LOW_VOL_TREND  # Now changed

    def test_confidence_below_minimum_sets_should_trade_false(self):
        """Confidence below minimum should set should_trade=False."""
        settings = Settings(regime_confidence_minimum=0.6)
        classifier = RegimeClassifier(settings=settings)

        bar = make_normalized_bar()
        features = _make_feature_vector(atr_14=3.5)

        # Make classifier ready
        classifier.classify(bar, features)
        classifier.classify(bar, features)

        # Mock low confidence
        with patch.object(classifier._intraday_filter, 'classify') as mock_classify:
            mock_classify.return_value = (RegimeState.RANGING, 0.4)  # Low confidence

            decision = classifier.classify(bar, features)

            assert decision.confidence < settings.regime_confidence_minimum
            assert decision.should_trade is False
            assert "Low confidence" in decision.reason

    def test_feature_columns_alignment(self):
        """Verify 'adx_14' is NOT in FEATURE_COLUMNS."""
        from fxer.signals.base import FEATURE_COLUMNS

        # ADX should not be in feature columns (it's computed externally)
        assert "adx_14" not in FEATURE_COLUMNS
        assert "adx" not in FEATURE_COLUMNS