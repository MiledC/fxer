"""Unit tests for the feature engine and temporal features."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from fxer.config.constants import MAX_WARMUP_BARS
from fxer.core.events import FeatureVector, NormalizedBar
from fxer.core.types import Timeframe
from fxer.features.engine import FeatureEngine
from fxer.features.temporal.time_features import (
    compute_time_features,
    day_of_week,
    hour_of_day,
    is_asian_session,
    is_london_session,
    is_month_turn,
    is_ny_session,
    is_overlap_session,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bar(
    close: float = 2000.0,
    high: float | None = None,
    low: float | None = None,
    open_: float | None = None,
    timestamp: datetime | None = None,
    symbol: str = "XAUUSD",
) -> NormalizedBar:
    """Build a NormalizedBar with sensible defaults."""
    c = Decimal(str(close))
    h = Decimal(str(high if high is not None else close + 2.0))
    l = Decimal(str(low if low is not None else close - 2.0))
    o = Decimal(str(open_ if open_ is not None else close - 0.5))
    return NormalizedBar(
        symbol=symbol,
        timeframe=Timeframe.H1,
        timestamp=timestamp or datetime(2025, 6, 15, 14, 0, 0, tzinfo=timezone.utc),
        open=o,
        high=h,
        low=l,
        close=c,
        volume=Decimal("100"),
    )


def _make_bars(n: int, base_close: float = 2000.0) -> list[NormalizedBar]:
    """Generate n bars with slightly varying prices."""
    bars = []
    for i in range(n):
        close = base_close + (i % 10) * 1.5 - 5.0
        ts = datetime(2025, 6, 15, i % 24, 0, 0, tzinfo=timezone.utc)
        bars.append(_make_bar(close=close, timestamp=ts))
    return bars


# ---------------------------------------------------------------------------
# Session Detection Tests
# ---------------------------------------------------------------------------

class TestSessionDetection:
    def test_london_session_active(self):
        ts = datetime(2025, 6, 15, 10, 0, 0, tzinfo=timezone.utc)
        assert is_london_session(ts) is True

    def test_london_session_start_boundary(self):
        ts = datetime(2025, 6, 15, 7, 0, 0, tzinfo=timezone.utc)
        assert is_london_session(ts) is True

    def test_london_session_end_boundary(self):
        # End is exclusive: 16:00 is NOT in London session
        ts = datetime(2025, 6, 15, 16, 0, 0, tzinfo=timezone.utc)
        assert is_london_session(ts) is False

    def test_london_session_inactive(self):
        ts = datetime(2025, 6, 15, 5, 0, 0, tzinfo=timezone.utc)
        assert is_london_session(ts) is False

    def test_ny_session_active(self):
        ts = datetime(2025, 6, 15, 15, 0, 0, tzinfo=timezone.utc)
        assert is_ny_session(ts) is True

    def test_ny_session_inactive(self):
        ts = datetime(2025, 6, 15, 22, 0, 0, tzinfo=timezone.utc)
        assert is_ny_session(ts) is False

    def test_overlap_session(self):
        ts = datetime(2025, 6, 15, 14, 0, 0, tzinfo=timezone.utc)
        assert is_overlap_session(ts) is True
        assert is_london_session(ts) is True
        assert is_ny_session(ts) is True

    def test_overlap_not_active_outside(self):
        ts = datetime(2025, 6, 15, 10, 0, 0, tzinfo=timezone.utc)
        assert is_overlap_session(ts) is False

    def test_asian_session_active_late(self):
        ts = datetime(2025, 6, 15, 23, 0, 0, tzinfo=timezone.utc)
        assert is_asian_session(ts) is True

    def test_asian_session_active_early(self):
        ts = datetime(2025, 6, 16, 3, 0, 0, tzinfo=timezone.utc)
        assert is_asian_session(ts) is True

    def test_asian_session_inactive(self):
        ts = datetime(2025, 6, 15, 10, 0, 0, tzinfo=timezone.utc)
        assert is_asian_session(ts) is False

    def test_asian_session_boundary_start(self):
        ts = datetime(2025, 6, 15, 22, 0, 0, tzinfo=timezone.utc)
        assert is_asian_session(ts) is True

    def test_asian_session_boundary_end(self):
        # End is exclusive: 07:00 is NOT in Asian session
        ts = datetime(2025, 6, 16, 7, 0, 0, tzinfo=timezone.utc)
        assert is_asian_session(ts) is False


# ---------------------------------------------------------------------------
# Temporal Feature Tests
# ---------------------------------------------------------------------------

class TestTemporalFeatures:
    def test_hour_of_day(self):
        ts = datetime(2025, 6, 15, 14, 30, 0, tzinfo=timezone.utc)
        assert hour_of_day(ts) == 14

    def test_day_of_week_monday(self):
        # 2025-06-16 is a Monday
        ts = datetime(2025, 6, 16, 0, 0, 0, tzinfo=timezone.utc)
        assert day_of_week(ts) == 0

    def test_day_of_week_friday(self):
        # 2025-06-20 is a Friday
        ts = datetime(2025, 6, 20, 0, 0, 0, tzinfo=timezone.utc)
        assert day_of_week(ts) == 4

    def test_is_month_turn_start(self):
        ts = datetime(2025, 7, 1, 10, 0, 0, tzinfo=timezone.utc)
        assert is_month_turn(ts) is True

    def test_is_month_turn_day_2(self):
        ts = datetime(2025, 7, 2, 10, 0, 0, tzinfo=timezone.utc)
        assert is_month_turn(ts) is True

    def test_is_month_turn_mid_month(self):
        ts = datetime(2025, 7, 15, 10, 0, 0, tzinfo=timezone.utc)
        assert is_month_turn(ts) is False

    def test_is_month_turn_last_day(self):
        ts = datetime(2025, 6, 30, 10, 0, 0, tzinfo=timezone.utc)
        assert is_month_turn(ts) is True

    def test_is_month_turn_second_to_last(self):
        ts = datetime(2025, 6, 29, 10, 0, 0, tzinfo=timezone.utc)
        assert is_month_turn(ts) is True

    def test_is_month_turn_february(self):
        ts = datetime(2025, 2, 28, 10, 0, 0, tzinfo=timezone.utc)
        assert is_month_turn(ts) is True

    def test_compute_time_features_returns_all_keys(self):
        ts = datetime(2025, 6, 15, 14, 0, 0, tzinfo=timezone.utc)
        feats = compute_time_features(ts)
        expected_keys = {
            "is_london_session",
            "is_ny_session",
            "is_overlap_session",
            "is_asian_session",
            "hour_of_day",
            "day_of_week",
            "is_month_turn",
        }
        assert set(feats.keys()) == expected_keys

    def test_compute_time_features_overlap_hour(self):
        ts = datetime(2025, 6, 15, 14, 0, 0, tzinfo=timezone.utc)
        feats = compute_time_features(ts)
        assert feats["is_london_session"] is True
        assert feats["is_ny_session"] is True
        assert feats["is_overlap_session"] is True
        assert feats["is_asian_session"] is False
        assert feats["hour_of_day"] == 14


# ---------------------------------------------------------------------------
# FeatureEngine Tests
# ---------------------------------------------------------------------------

class TestFeatureEngine:
    def test_get_warmup_bars(self):
        engine = FeatureEngine()
        assert engine.get_warmup_bars() == MAX_WARMUP_BARS

    def test_not_warmed_up_initially(self):
        engine = FeatureEngine()
        assert engine.warmup_complete is False
        assert engine.bar_count == 0

    def test_single_bar_returns_feature_vector(self):
        engine = FeatureEngine()
        bar = _make_bar()
        fv = engine.compute_features(bar)
        assert isinstance(fv, FeatureVector)
        assert fv.symbol == "XAUUSD"
        assert fv.timestamp == bar.timestamp
        assert fv.warmup_complete is False

    def test_warmup_completes_after_enough_bars(self):
        engine = FeatureEngine()
        bars = _make_bars(MAX_WARMUP_BARS + 5)
        for bar in bars:
            fv = engine.compute_features(bar)
        assert engine.warmup_complete is True
        assert fv.warmup_complete is True

    def test_indicators_none_during_warmup(self):
        engine = FeatureEngine()
        bar = _make_bar()
        fv = engine.compute_features(bar)
        assert fv.rsi_14 is None
        assert fv.rsi_7 is None
        assert fv.macd_line is None
        assert fv.macd_signal is None
        assert fv.macd_histogram is None
        assert fv.bb_upper is None
        assert fv.atr_14 is None

    def test_indicators_populated_after_warmup(self):
        engine = FeatureEngine()
        bars = _make_bars(MAX_WARMUP_BARS + 5)
        fv = None
        for bar in bars:
            fv = engine.compute_features(bar)
        assert fv is not None
        assert fv.rsi_14 is not None
        assert fv.rsi_7 is not None
        assert fv.macd_line is not None
        assert fv.macd_signal is not None
        assert fv.macd_histogram is not None
        assert fv.bb_upper is not None
        assert fv.bb_middle is not None
        assert fv.bb_lower is not None
        assert fv.bb_width is not None
        assert fv.atr_14 is not None

    def test_temporal_features_present_from_start(self):
        engine = FeatureEngine()
        ts = datetime(2025, 6, 15, 14, 0, 0, tzinfo=timezone.utc)
        bar = _make_bar(timestamp=ts)
        fv = engine.compute_features(bar)
        assert fv.hour_of_day == 14
        assert fv.is_london_session is True
        assert fv.is_ny_session is True
        assert fv.is_overlap_session is True

    def test_compute_batch(self):
        engine = FeatureEngine()
        bars = _make_bars(MAX_WARMUP_BARS + 5)
        results = engine.compute_batch(bars)
        assert len(results) == len(bars)
        assert all(isinstance(fv, FeatureVector) for fv in results)
        assert results[-1].warmup_complete is True
        assert results[0].warmup_complete is False

    def test_bar_count_increments(self):
        engine = FeatureEngine()
        bars = _make_bars(10)
        for bar in bars:
            engine.compute_features(bar)
        assert engine.bar_count == 10

    def test_reset(self):
        engine = FeatureEngine()
        bars = _make_bars(MAX_WARMUP_BARS + 5)
        engine.compute_batch(bars)
        assert engine.warmup_complete is True
        assert engine.bar_count > 0

        engine.reset()
        assert engine.warmup_complete is False
        assert engine.bar_count == 0

    def test_rsi_values_bounded(self):
        engine = FeatureEngine()
        bars = _make_bars(MAX_WARMUP_BARS + 10)
        for bar in bars:
            fv = engine.compute_features(bar)
            if fv.rsi_14 is not None:
                assert 0.0 <= fv.rsi_14 <= 100.0
            if fv.rsi_7 is not None:
                assert 0.0 <= fv.rsi_7 <= 100.0

    def test_bb_ordering(self):
        engine = FeatureEngine()
        bars = _make_bars(MAX_WARMUP_BARS + 10)
        for bar in bars:
            fv = engine.compute_features(bar)
            if fv.bb_upper is not None:
                assert fv.bb_upper >= fv.bb_middle >= fv.bb_lower

    def test_atr_positive(self):
        engine = FeatureEngine()
        bars = _make_bars(MAX_WARMUP_BARS + 10)
        for bar in bars:
            fv = engine.compute_features(bar)
            if fv.atr_14 is not None:
                assert fv.atr_14 > 0.0

    def test_feature_vector_serialization_roundtrip(self):
        engine = FeatureEngine()
        bars = _make_bars(MAX_WARMUP_BARS + 5)
        results = engine.compute_batch(bars)
        fv = results[-1]
        d = fv.to_dict()
        restored = FeatureVector.from_dict(d)
        assert restored.symbol == fv.symbol
        assert restored.rsi_14 == fv.rsi_14
        assert restored.warmup_complete == fv.warmup_complete
