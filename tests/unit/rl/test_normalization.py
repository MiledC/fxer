"""Unit tests for RL observation normalization module."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import numpy as np
import pytest

from fxer.rl.observations.config import (
    NormalizationConfig,
    NormalizationMethod,
    ObservationConfig,
    TimeframeWindowConfig,
)
from fxer.rl.observations.normalization import (
    FEATURE_CLASS_MAP,
    RESCALE_RANGES,
    FeatureClass,
    ObservationNormalizer,
    OnlineStats,
    TradingSession,
    classify_session,
)
from fxer.signals.base import FEATURE_COLUMNS
from fxer.core.types import Timeframe


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_timestamp(hour: int, minute: int = 0) -> datetime:
    return datetime(2024, 6, 15, hour, minute, 0, tzinfo=timezone.utc)


def _make_feature_vector_mock(**overrides):
    """Create a mock FeatureVector with needed attributes."""
    fv = MagicMock()
    fv.rolling_volatility_20 = overrides.get("rolling_volatility_20", 0.005)
    return fv


def _simple_obs_config(include_regime: bool = False) -> ObservationConfig:
    """Minimal single-window config for testing normalizer index mapping."""
    return ObservationConfig(
        windows=(TimeframeWindowConfig(Timeframe.M5, lookback=1),),
        include_regime=include_regime,
        normalization=NormalizationConfig(method=NormalizationMethod.FEATURE_SPECIFIC),
    )


# ---------------------------------------------------------------------------
# TradingSession & classify_session
# ---------------------------------------------------------------------------

class TestClassifySession:

    @pytest.mark.parametrize("hour,expected", [
        (22, TradingSession.ASIAN),
        (23, TradingSession.ASIAN),
        (0, TradingSession.ASIAN),
        (1, TradingSession.ASIAN),
        (6, TradingSession.ASIAN),
    ])
    def test_asian_session(self, hour, expected):
        assert classify_session(_make_timestamp(hour)) == expected

    @pytest.mark.parametrize("hour,expected", [
        (7, TradingSession.LONDON),
        (8, TradingSession.LONDON),
        (11, TradingSession.LONDON),
    ])
    def test_london_session(self, hour, expected):
        assert classify_session(_make_timestamp(hour)) == expected

    @pytest.mark.parametrize("hour,expected", [
        (12, TradingSession.OVERLAP),
        (13, TradingSession.OVERLAP),
        (15, TradingSession.OVERLAP),
    ])
    def test_overlap_session(self, hour, expected):
        assert classify_session(_make_timestamp(hour)) == expected

    @pytest.mark.parametrize("hour,expected", [
        (16, TradingSession.NY),
        (17, TradingSession.NY),
        (21, TradingSession.NY),
    ])
    def test_ny_session(self, hour, expected):
        assert classify_session(_make_timestamp(hour)) == expected

    def test_all_hours_classified(self):
        """Every hour of the day maps to a session."""
        for hour in range(24):
            result = classify_session(_make_timestamp(hour))
            assert isinstance(result, TradingSession)


# ---------------------------------------------------------------------------
# FEATURE_CLASS_MAP completeness
# ---------------------------------------------------------------------------

class TestFeatureClassMap:

    def test_all_feature_columns_mapped(self):
        for col in FEATURE_COLUMNS:
            assert col in FEATURE_CLASS_MAP, f"{col} missing from FEATURE_CLASS_MAP"

    def test_no_extra_keys(self):
        for key in FEATURE_CLASS_MAP:
            assert key in FEATURE_COLUMNS, f"{key} in map but not in FEATURE_COLUMNS"

    @pytest.mark.parametrize("name", [
        "is_london_session", "is_ny_session", "is_overlap_session",
        "is_asian_session", "is_month_turn", "bb_percent_b",
    ])
    def test_passthrough_features(self, name):
        assert FEATURE_CLASS_MAP[name] == FeatureClass.PASSTHROUGH

    @pytest.mark.parametrize("name", [
        "rsi_14", "rsi_7", "dxy_rsi_14", "hour_of_day", "day_of_week",
    ])
    def test_rescale_features(self, name):
        assert FEATURE_CLASS_MAP[name] == FeatureClass.RESCALE

    @pytest.mark.parametrize("name", [
        "macd_line", "macd_signal", "macd_histogram",
        "atr_14", "vix_change", "dxy_return_1h",
    ])
    def test_zscore_features(self, name):
        assert FEATURE_CLASS_MAP[name] == FeatureClass.ZSCORE

    @pytest.mark.parametrize("name", [
        "return_1bar", "return_5bar", "return_12bar", "momentum_48",
    ])
    def test_vol_adjust_features(self, name):
        assert FEATURE_CLASS_MAP[name] == FeatureClass.VOL_ADJUST

    @pytest.mark.parametrize("name", [
        "bb_width", "rolling_volatility_20", "vix_level",
    ])
    def test_log_zscore_features(self, name):
        assert FEATURE_CLASS_MAP[name] == FeatureClass.LOG_ZSCORE


# ---------------------------------------------------------------------------
# OnlineStats
# ---------------------------------------------------------------------------

class TestOnlineStatsExpandingWindow:

    def test_mean_matches_numpy(self):
        stats = OnlineStats(min_samples=50)
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        ts = _make_timestamp(10)
        for v in values:
            stats.update(v, ts)

        mean, std = stats.get_stats()
        assert mean == pytest.approx(np.mean(values), abs=1e-10)

    def test_variance_matches_numpy(self):
        stats = OnlineStats(min_samples=50)
        rng = np.random.RandomState(42)
        values = rng.normal(10.0, 2.0, size=30).tolist()
        ts = _make_timestamp(10)
        for v in values:
            stats.update(v, ts)

        mean, std = stats.get_stats()
        assert mean == pytest.approx(np.mean(values), abs=1e-6)
        # Welford produces population std; numpy std() also uses N by default
        assert std == pytest.approx(np.std(values), abs=0.1)

    def test_passthrough_during_warmup(self):
        stats = OnlineStats(min_samples=20)
        ts = _make_timestamp(10)
        for i in range(10):
            stats.update(float(i), ts)

        # Not enough samples yet — normalize should return value unchanged
        assert stats.normalize(42.0) == 42.0


class TestOnlineStatsEMA:

    def test_ema_activates_after_min_samples(self):
        stats = OnlineStats(alpha=0.97, min_samples=5)
        ts = _make_timestamp(10)
        for i in range(5):
            stats.update(10.0, ts)

        # Now in EMA mode — feeding a different value should shift the mean
        stats.update(20.0, ts)
        mean, _ = stats.get_stats()
        # Mean should have moved toward 20 but not reached it
        assert 10.0 < mean < 20.0

    def test_ema_weights_recent_values_more(self):
        stats = OnlineStats(alpha=0.9, min_samples=5)
        ts = _make_timestamp(10)
        # Prime with 10.0
        for _ in range(10):
            stats.update(10.0, ts)
        mean_before, _ = stats.get_stats()

        # Shift to 20.0
        for _ in range(50):
            stats.update(20.0, ts)
        mean_after, _ = stats.get_stats()

        assert mean_after == pytest.approx(20.0, abs=0.5)
        assert mean_before == pytest.approx(10.0, abs=0.5)

    def test_normalize_produces_zscore(self):
        stats = OnlineStats(alpha=0.97, min_samples=5)
        ts = _make_timestamp(10)
        rng = np.random.RandomState(99)
        values = rng.normal(50.0, 5.0, size=100).tolist()
        for v in values:
            stats.update(v, ts)

        mean, std = stats.get_stats()
        z = stats.normalize(mean)
        assert z == pytest.approx(0.0, abs=0.3)

        z_high = stats.normalize(mean + std)
        assert z_high == pytest.approx(1.0, abs=0.3)


class TestOnlineStatsWinsorization:

    def test_extreme_outlier_does_not_corrupt_stats(self):
        stats = OnlineStats(alpha=0.97, min_samples=5, winsorize_threshold=5.0)
        ts = _make_timestamp(10)

        # Build up normal stats around 0, std ~1
        rng = np.random.RandomState(42)
        for v in rng.normal(0.0, 1.0, size=30):
            stats.update(v, ts)

        mean_before, std_before = stats.get_stats()

        # Inject extreme outlier (100 std devs away)
        stats.update(mean_before + 100 * std_before, ts)

        mean_after, std_after = stats.get_stats()
        # The outlier should be winsorized to 5σ, so stats shouldn't blow up
        assert abs(mean_after - mean_before) < 2.0 * std_before
        assert std_after < 3.0 * std_before


class TestOnlineStatsGapDetection:

    def test_gap_dampens_alpha(self):
        """After a large time gap, the EMA uses a dampened alpha (alpha*0.5).

        With dampened alpha the new value receives MORE weight (1 - alpha*0.5),
        so the mean shifts MORE toward the new value compared to the normal
        case. We verify the gap is detected by checking the mean moved
        differently than the normal (non-gap) case.
        """
        base = _make_timestamp(10)

        # --- normal case (no gap) ---
        stats_normal = OnlineStats(alpha=0.97, min_samples=5, gap_threshold_minutes=180)
        for i in range(20):
            stats_normal.update(10.0, base + timedelta(minutes=i * 5))
        mean_before = stats_normal.get_stats()[0]

        ts_normal_next = base + timedelta(minutes=20 * 5)
        stats_normal.update(20.0, ts_normal_next)
        mean_normal = stats_normal.get_stats()[0]

        # --- gap case (200+ min gap) ---
        stats_gap = OnlineStats(alpha=0.97, min_samples=5, gap_threshold_minutes=180)
        for i in range(20):
            stats_gap.update(10.0, base + timedelta(minutes=i * 5))

        ts_gap = base + timedelta(minutes=19 * 5 + 200)
        stats_gap.update(20.0, ts_gap)
        mean_gap = stats_gap.get_stats()[0]

        # Both means should have shifted toward 20, but by different amounts
        normal_shift = abs(mean_normal - mean_before)
        gap_shift = abs(mean_gap - mean_before)
        assert normal_shift != pytest.approx(gap_shift, abs=1e-6), (
            "Gap detection should cause a different mean shift"
        )


class TestOnlineStatsSessionAware:

    def test_independent_session_stats(self):
        stats = OnlineStats(min_samples=3, session_aware=True)
        ts = _make_timestamp(10)

        # Feed low values to Asian session
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            stats.update(v, ts, TradingSession.ASIAN)

        # Feed high values to London session
        for v in [100.0, 200.0, 300.0, 400.0, 500.0]:
            stats.update(v, ts, TradingSession.LONDON)

        asian_mean, _ = stats.get_stats(TradingSession.ASIAN)
        london_mean, _ = stats.get_stats(TradingSession.LONDON)

        assert asian_mean < 10.0
        assert london_mean > 90.0

    def test_reset_clears_all_sessions(self):
        stats = OnlineStats(min_samples=3, session_aware=True)
        ts = _make_timestamp(10)

        stats.update(100.0, ts, TradingSession.OVERLAP)
        stats.update(100.0, ts, TradingSession.OVERLAP)
        stats.update(100.0, ts, TradingSession.OVERLAP)

        stats.reset()

        mean, std = stats.get_stats(TradingSession.OVERLAP)
        assert mean == 0.0
        assert std == 0.0


# ---------------------------------------------------------------------------
# ObservationNormalizer
# ---------------------------------------------------------------------------

class TestObservationNormalizerPassthrough:

    def test_passthrough_indices_unchanged(self):
        config = _simple_obs_config()
        normalizer = ObservationNormalizer(
            method="feature_specific",
            obs_config=config,
            min_samples=1,
        )

        # Build observation: 24 features (1 bar, features only)
        raw = np.zeros(24, dtype=np.float64)
        # Set PASSTHROUGH features (bb_percent_b=idx5, sessions=idx13-16, is_month_turn=idx19)
        passthrough_indices = []
        for i, col in enumerate(FEATURE_COLUMNS):
            if FEATURE_CLASS_MAP[col] == FeatureClass.PASSTHROUGH:
                raw[i] = 0.75  # arbitrary non-zero value
                passthrough_indices.append(i)

        ts = _make_timestamp(10)
        result = normalizer.normalize_observation(raw, ts)

        for i in passthrough_indices:
            assert result[i] == pytest.approx(0.75), f"Passthrough index {i} ({FEATURE_COLUMNS[i]}) changed"


class TestObservationNormalizerRescale:

    def test_rsi_midpoint(self):
        config = _simple_obs_config()
        normalizer = ObservationNormalizer(
            method="feature_specific", obs_config=config, min_samples=1,
        )
        raw = np.zeros(24, dtype=np.float64)
        rsi_14_idx = FEATURE_COLUMNS.index("rsi_14")
        raw[rsi_14_idx] = 50.0

        result = normalizer.normalize_observation(raw, _make_timestamp(10))
        assert result[rsi_14_idx] == pytest.approx(0.5)

    def test_rsi_boundaries(self):
        config = _simple_obs_config()
        normalizer = ObservationNormalizer(
            method="feature_specific", obs_config=config, min_samples=1,
        )
        rsi_idx = FEATURE_COLUMNS.index("rsi_14")

        raw_low = np.zeros(24, dtype=np.float64)
        raw_low[rsi_idx] = 0.0
        result_low = normalizer.normalize_observation(raw_low, _make_timestamp(10))
        assert result_low[rsi_idx] == pytest.approx(0.0)

        raw_high = np.zeros(24, dtype=np.float64)
        raw_high[rsi_idx] = 100.0
        result_high = normalizer.normalize_observation(raw_high, _make_timestamp(10))
        assert result_high[rsi_idx] == pytest.approx(1.0)

    def test_hour_of_day_rescale(self):
        config = _simple_obs_config()
        normalizer = ObservationNormalizer(
            method="feature_specific", obs_config=config, min_samples=1,
        )
        hour_idx = FEATURE_COLUMNS.index("hour_of_day")

        raw = np.zeros(24, dtype=np.float64)
        raw[hour_idx] = 12.0
        result = normalizer.normalize_observation(raw, _make_timestamp(10))
        assert result[hour_idx] == pytest.approx(12.0 / 23.0, abs=1e-6)


class TestObservationNormalizerVolAdjust:

    def test_vol_adjust_divides_by_prev_volatility(self):
        config = _simple_obs_config()
        normalizer = ObservationNormalizer(
            method="feature_specific", obs_config=config, min_samples=1,
        )
        return_idx = FEATURE_COLUMNS.index("return_1bar")

        raw = np.zeros(24, dtype=np.float64)
        raw[return_idx] = 0.01

        prev_fv = _make_feature_vector_mock(rolling_volatility_20=0.005)
        result = normalizer.normalize_observation(raw, _make_timestamp(10), prev_fv)

        # 0.01 / 0.005 = 2.0
        assert result[return_idx] == pytest.approx(2.0, abs=0.01)

    def test_vol_adjust_passthrough_without_prev_features(self):
        """Without prev_features, VOL_ADJUST returns should pass through raw."""
        config = _simple_obs_config()
        normalizer = ObservationNormalizer(
            method="feature_specific", obs_config=config, min_samples=100,
        )
        return_idx = FEATURE_COLUMNS.index("return_1bar")

        raw = np.zeros(24, dtype=np.float64)
        raw[return_idx] = 0.01

        # No prev_features → value stays raw (within clip range)
        result = normalizer.normalize_observation(raw, _make_timestamp(10), None)
        assert result[return_idx] == pytest.approx(0.01, abs=1e-6)

    def test_vol_adjust_uses_t_minus_1_not_current(self):
        """Verify VOL_ADJUST uses the PREVIOUS bar's volatility, not current."""
        config = _simple_obs_config()
        normalizer = ObservationNormalizer(
            method="feature_specific", obs_config=config, min_samples=1,
        )
        return_idx = FEATURE_COLUMNS.index("return_1bar")
        vol_idx = FEATURE_COLUMNS.index("rolling_volatility_20")

        raw = np.zeros(24, dtype=np.float64)
        raw[return_idx] = 0.01
        raw[vol_idx] = 0.010  # Current bar's volatility (should NOT be used)

        prev_fv = _make_feature_vector_mock(rolling_volatility_20=0.005)
        result = normalizer.normalize_observation(raw, _make_timestamp(10), prev_fv)

        # Should use prev (0.005) not current (0.010)
        assert result[return_idx] == pytest.approx(0.01 / 0.005, abs=0.1)


class TestObservationNormalizerLogZscore:

    def test_log_transform_applied(self):
        config = _simple_obs_config()
        normalizer = ObservationNormalizer(
            method="feature_specific", obs_config=config, min_samples=100,
        )
        vix_idx = FEATURE_COLUMNS.index("vix_level")

        raw = np.zeros(24, dtype=np.float64)
        raw[vix_idx] = 20.0

        # During warmup (min_samples=100), normalize returns the log-transformed value
        result = normalizer.normalize_observation(raw, _make_timestamp(10))
        assert result[vix_idx] == pytest.approx(math.log(20.0), abs=0.1)

    def test_log_floor_for_near_zero(self):
        """Zero values get floored to 1e-10 before log; result is clipped."""
        config = _simple_obs_config()
        normalizer = ObservationNormalizer(
            method="feature_specific", obs_config=config, min_samples=100,
            clip_range=(-50.0, 50.0),  # wide clip to see the raw log value
        )
        bb_width_idx = FEATURE_COLUMNS.index("bb_width")

        raw = np.zeros(24, dtype=np.float64)
        raw[bb_width_idx] = 0.0  # zero → floor at 1e-10

        result = normalizer.normalize_observation(raw, _make_timestamp(10))
        # log(1e-10) ≈ -23.03, which should survive the wide clip range
        assert result[bb_width_idx] == pytest.approx(math.log(1e-10), abs=0.1)


class TestObservationNormalizerClipping:

    def test_values_clipped_to_range(self):
        config = _simple_obs_config()
        normalizer = ObservationNormalizer(
            method="feature_specific", obs_config=config,
            min_samples=1, clip_range=(-5.0, 5.0),
        )

        # Put extreme values in RESCALE features (they don't clip by themselves,
        # but the final clip applies to all)
        raw = np.full(24, 1000.0, dtype=np.float64)
        result = normalizer.normalize_observation(raw, _make_timestamp(10))
        assert np.all(result <= 5.0)
        assert np.all(result >= -5.0)

    def test_custom_clip_range(self):
        config = ObservationConfig(
            windows=(TimeframeWindowConfig(Timeframe.M5, lookback=1),),
            include_regime=False,
            normalization=NormalizationConfig(
                method=NormalizationMethod.FEATURE_SPECIFIC,
                clip_range=(-2.0, 2.0),
            ),
        )
        normalizer = ObservationNormalizer(
            method="feature_specific", obs_config=config,
            min_samples=1, clip_range=(-2.0, 2.0),
        )

        raw = np.full(24, 500.0, dtype=np.float64)
        result = normalizer.normalize_observation(raw, _make_timestamp(10))
        assert np.all(result <= 2.0)


class TestObservationNormalizerOnlineZscore:

    def test_online_zscore_normalizes_non_passthrough(self):
        config = _simple_obs_config()
        normalizer = ObservationNormalizer(
            method="online_zscore", obs_config=config, min_samples=3,
        )

        ts = _make_timestamp(10)
        # Feed several observations to build stats
        for _ in range(10):
            raw = np.random.RandomState(42).normal(50.0, 5.0, size=24)
            normalizer.normalize_observation(raw, ts)

        # After warmup, passthrough features should still be ~original
        raw = np.full(24, 50.0, dtype=np.float64)
        bb_pct_idx = FEATURE_COLUMNS.index("bb_percent_b")
        raw[bb_pct_idx] = 0.7
        result = normalizer.normalize_observation(raw, ts)

        # bb_percent_b is PASSTHROUGH even in online_zscore
        assert result[bb_pct_idx] == pytest.approx(0.7, abs=0.01)


class TestObservationNormalizerIndexMap:

    def test_index_map_size_matches_observation(self):
        config = ObservationConfig(
            include_regime=True,
            normalization=NormalizationConfig(method=NormalizationMethod.FEATURE_SPECIFIC),
        )
        normalizer = ObservationNormalizer(
            method="feature_specific", obs_config=config,
        )
        assert len(normalizer._index_map) == config.observation_size

    def test_index_map_with_ohlcv(self):
        config = ObservationConfig(
            windows=(
                TimeframeWindowConfig(Timeframe.M5, lookback=2, include_ohlcv=True),
            ),
            include_regime=False,
            normalization=NormalizationConfig(method=NormalizationMethod.FEATURE_SPECIFIC),
        )
        normalizer = ObservationNormalizer(
            method="feature_specific", obs_config=config,
        )
        # 2 bars * (5 OHLCV + 24 features) = 58
        assert len(normalizer._index_map) == 58

    def test_regime_indices_are_passthrough(self):
        config = ObservationConfig(
            windows=(TimeframeWindowConfig(Timeframe.M5, lookback=1),),
            include_regime=True,
            normalization=NormalizationConfig(method=NormalizationMethod.FEATURE_SPECIFIC),
        )
        normalizer = ObservationNormalizer(
            method="feature_specific", obs_config=config,
        )
        # Last 4 indices should be regime (PASSTHROUGH)
        for i in range(24, 28):
            name, cls = normalizer._index_map[i]
            assert name == "regime"
            assert cls == FeatureClass.PASSTHROUGH


class TestObservationNormalizerReset:

    def test_reset_clears_stats(self):
        config = _simple_obs_config()
        normalizer = ObservationNormalizer(
            method="feature_specific", obs_config=config, min_samples=3,
        )

        ts = _make_timestamp(10)
        raw = np.random.RandomState(42).normal(50.0, 5.0, size=24)
        for _ in range(10):
            normalizer.normalize_observation(raw, ts)

        normalizer.reset()

        # After reset, stats should be back to zero
        for stats in normalizer._stats.values():
            mean, std = stats.get_stats()
            assert mean == 0.0
            assert std == 0.0


# ---------------------------------------------------------------------------
# NormalizationConfig
# ---------------------------------------------------------------------------

class TestNormalizationConfig:

    def test_defaults(self):
        c = NormalizationConfig()
        assert c.method == NormalizationMethod.NONE
        assert c.alpha == 0.97
        assert c.min_samples == 20
        assert c.winsorize_threshold == 5.0
        assert c.gap_threshold_minutes == 180
        assert c.session_aware is False
        assert c.clip_range == (-5.0, 5.0)

    def test_frozen_immutability(self):
        c = NormalizationConfig()
        with pytest.raises(AttributeError):
            c.method = NormalizationMethod.FEATURE_SPECIFIC

    def test_enum_values(self):
        assert NormalizationMethod.NONE.value == "none"
        assert NormalizationMethod.ONLINE_ZSCORE.value == "online_zscore"
        assert NormalizationMethod.FEATURE_SPECIFIC.value == "feature_specific"


# ---------------------------------------------------------------------------
# ObservationConfig backward compatibility
# ---------------------------------------------------------------------------

class TestObservationConfigCompat:

    def test_default_normalization_is_none(self):
        c = ObservationConfig()
        assert c.normalization.method == NormalizationMethod.NONE

    def test_observation_size_unchanged(self):
        c = ObservationConfig()
        assert c.observation_size == 844

    def test_observation_size_with_normalization(self):
        c = ObservationConfig(
            normalization=NormalizationConfig(
                method=NormalizationMethod.FEATURE_SPECIFIC,
            ),
        )
        assert c.observation_size == 844


# ---------------------------------------------------------------------------
# Trading-specific: no lookahead in normalization
# ---------------------------------------------------------------------------

class TestNoLookahead:

    def test_zscore_at_t_same_with_or_without_future(self):
        """Z-score at timestamp T should not change if future data is added."""
        config = _simple_obs_config()

        # Normalizer A: processes [v0, v1, v2]
        norm_a = ObservationNormalizer(
            method="feature_specific", obs_config=config, min_samples=1,
        )
        ts = _make_timestamp(10)
        macd_idx = FEATURE_COLUMNS.index("macd_line")

        obs_values = [1.0, 2.0, 3.0, 4.0]
        results_a = []
        for v in obs_values:
            raw = np.zeros(24, dtype=np.float64)
            raw[macd_idx] = v
            result = norm_a.normalize_observation(raw, ts)
            results_a.append(result[macd_idx])

        # Normalizer B: processes only [v0, v1] (no future)
        norm_b = ObservationNormalizer(
            method="feature_specific", obs_config=config, min_samples=1,
        )
        results_b = []
        for v in obs_values[:2]:
            raw = np.zeros(24, dtype=np.float64)
            raw[macd_idx] = v
            result = norm_b.normalize_observation(raw, ts)
            results_b.append(result[macd_idx])

        # The z-scores at positions 0 and 1 should be identical
        assert results_a[0] == pytest.approx(results_b[0])
        assert results_a[1] == pytest.approx(results_b[1])
