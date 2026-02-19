"""End-to-end integration tests for the data pipeline.

Tests the full flow: CSV -> Normalize -> (Store) -> Aggregate -> Features.
Uses the sample_bars.csv fixture and does not require external services.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from fxer.core.events import FeatureVector, NormalizedBar
from fxer.core.types import RawBar, Timeframe
from fxer.data.aggregator.bar_aggregator import BarAggregator, aggregate_batch
from fxer.data.loaders.csv_loader import CSVLoader
from fxer.data.normalizer.normalizer import BarNormalizer
from fxer.data.normalizer.validators import (
    validate_ohlc_consistency,
    validate_price_bounds,
    validate_volume,
)
from fxer.features.engine import FeatureEngine

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
SAMPLE_CSV = FIXTURES_DIR / "sample_bars.csv"


# ---------------------------------------------------------------------------
# Stage 1: CSV Loading
# ---------------------------------------------------------------------------


class TestCSVLoadingStage:
    """Verify that the CSV loader reads the fixture file correctly."""

    def test_load_sample_csv(self) -> None:
        loader = CSVLoader()
        stats = loader.load(SAMPLE_CSV, symbol="XAUUSD")

        assert stats.rows_loaded == 60
        assert stats.rows_skipped == 0
        assert stats.errors == []

    def test_loaded_bars_are_chronological(self) -> None:
        loader = CSVLoader()
        loader.load(SAMPLE_CSV, symbol="XAUUSD")
        bars = list(loader.iter_bars())

        timestamps = [b.timestamp for b in bars]
        assert timestamps == sorted(timestamps)

    def test_loaded_bars_have_correct_symbol(self) -> None:
        loader = CSVLoader()
        loader.load(SAMPLE_CSV, symbol="XAUUSD")
        bars = list(loader.iter_bars())

        assert all(b.symbol == "XAUUSD" for b in bars)

    def test_loaded_bars_have_all_fields(self) -> None:
        loader = CSVLoader()
        loader.load(SAMPLE_CSV, symbol="XAUUSD")
        bar = next(loader.iter_bars())

        assert bar.timestamp is not None
        assert bar.open is not None
        assert bar.high is not None
        assert bar.low is not None
        assert bar.close is not None
        assert bar.volume is not None


# ---------------------------------------------------------------------------
# Stage 2: Normalization
# ---------------------------------------------------------------------------


class TestNormalizationStage:
    """Verify that raw bars are correctly normalized."""

    @pytest.fixture
    def raw_bars(self) -> list[RawBar]:
        loader = CSVLoader()
        loader.load(SAMPLE_CSV, symbol="XAUUSD")
        return list(loader.iter_bars())

    def test_all_bars_normalize_successfully(self, raw_bars: list[RawBar]) -> None:
        normalizer = BarNormalizer(
            validators=[validate_price_bounds, validate_ohlc_consistency, validate_volume],
            default_symbol="XAUUSD",
            default_timeframe="5m",
        )

        normalized = []
        for raw in raw_bars:
            nbar = normalizer.normalize_bar(raw)
            normalized.append(nbar)

        assert len(normalized) == 60

    def test_normalized_bars_have_decimal_prices(self, raw_bars: list[RawBar]) -> None:
        normalizer = BarNormalizer(
            validators=[validate_price_bounds, validate_ohlc_consistency, validate_volume],
            default_symbol="XAUUSD",
            default_timeframe="5m",
        )

        nbar = normalizer.normalize_bar(raw_bars[0])
        assert isinstance(nbar.open, Decimal)
        assert isinstance(nbar.high, Decimal)
        assert isinstance(nbar.low, Decimal)
        assert isinstance(nbar.close, Decimal)
        assert isinstance(nbar.volume, Decimal)

    def test_normalized_bars_have_utc_timestamps(self, raw_bars: list[RawBar]) -> None:
        normalizer = BarNormalizer(
            validators=[validate_price_bounds, validate_ohlc_consistency, validate_volume],
            default_symbol="XAUUSD",
            default_timeframe="5m",
        )

        for raw in raw_bars:
            nbar = normalizer.normalize_bar(raw)
            assert nbar.timestamp.tzinfo is not None

    def test_normalized_bar_values_match_raw(self, raw_bars: list[RawBar]) -> None:
        normalizer = BarNormalizer(
            validators=[validate_price_bounds, validate_ohlc_consistency, validate_volume],
            default_symbol="XAUUSD",
            default_timeframe="5m",
        )

        raw = raw_bars[0]
        nbar = normalizer.normalize_bar(raw)

        assert nbar.open == Decimal(str(raw.open))
        assert nbar.high == Decimal(str(raw.high))
        assert nbar.low == Decimal(str(raw.low))
        assert nbar.close == Decimal(str(raw.close))
        assert nbar.symbol == "XAUUSD"
        assert nbar.timeframe == Timeframe.M5

    def test_ohlc_consistency_preserved(self, raw_bars: list[RawBar]) -> None:
        normalizer = BarNormalizer(
            validators=[validate_price_bounds, validate_ohlc_consistency, validate_volume],
            default_symbol="XAUUSD",
            default_timeframe="5m",
        )

        for raw in raw_bars:
            nbar = normalizer.normalize_bar(raw)
            assert nbar.low <= nbar.open <= nbar.high
            assert nbar.low <= nbar.close <= nbar.high


# ---------------------------------------------------------------------------
# Stage 3: Aggregation
# ---------------------------------------------------------------------------


class TestAggregationStage:
    """Verify that 5M bars aggregate to 15M and 1H correctly."""

    @pytest.fixture
    def normalized_5m_bars(self) -> list[NormalizedBar]:
        loader = CSVLoader()
        loader.load(SAMPLE_CSV, symbol="XAUUSD")
        normalizer = BarNormalizer(
            validators=[validate_price_bounds, validate_ohlc_consistency, validate_volume],
            default_symbol="XAUUSD",
            default_timeframe="5m",
        )
        return [normalizer.normalize_bar(raw) for raw in loader.iter_bars()]

    def test_batch_aggregate_to_15m(self, normalized_5m_bars: list[NormalizedBar]) -> None:
        bars_15m = aggregate_batch(normalized_5m_bars, Timeframe.M15)

        assert len(bars_15m) > 0
        for bar in bars_15m:
            assert bar.timeframe == Timeframe.M15

    def test_batch_aggregate_to_1h(self, normalized_5m_bars: list[NormalizedBar]) -> None:
        bars_1h = aggregate_batch(normalized_5m_bars, Timeframe.H1)

        assert len(bars_1h) > 0
        for bar in bars_1h:
            assert bar.timeframe == Timeframe.H1

    def test_aggregated_ohlc_consistency(
        self, normalized_5m_bars: list[NormalizedBar]
    ) -> None:
        bars_15m = aggregate_batch(normalized_5m_bars, Timeframe.M15)

        for bar in bars_15m:
            assert bar.low <= bar.open <= bar.high
            assert bar.low <= bar.close <= bar.high
            assert bar.volume >= 0

    def test_aggregated_volume_is_sum(
        self, normalized_5m_bars: list[NormalizedBar]
    ) -> None:
        """For complete aggregated bars, volume should be the sum of component bars."""
        bars_15m = aggregate_batch(normalized_5m_bars, Timeframe.M15)
        complete = [b for b in bars_15m if b.is_complete]

        # At least some bars should be complete
        assert len(complete) > 0
        for agg_bar in complete:
            # Find the 5M bars that contributed to this 15M bar
            period_end = agg_bar.timestamp + timedelta(minutes=15)
            src_bars = [
                b for b in normalized_5m_bars
                if agg_bar.timestamp <= b.timestamp < period_end
            ]
            if src_bars:
                expected_vol = sum(b.volume for b in src_bars)
                assert agg_bar.volume == expected_vol

    def test_streaming_aggregation(
        self, normalized_5m_bars: list[NormalizedBar]
    ) -> None:
        """Test BarAggregator streaming interface produces consistent results."""
        aggregator = BarAggregator(Timeframe.M5, Timeframe.M15)
        completed: list[NormalizedBar] = []

        for bar in normalized_5m_bars:
            result = aggregator.push(bar)
            if result is not None:
                completed.append(result)

        # Flush any remaining
        last = aggregator.flush()
        if last is not None:
            completed.append(last)

        assert len(completed) > 0
        for bar in completed:
            assert bar.timeframe == Timeframe.M15


# ---------------------------------------------------------------------------
# Stage 4: Feature Computation
# ---------------------------------------------------------------------------


class TestFeatureComputationStage:
    """Verify that features are computed correctly from normalized bars."""

    @pytest.fixture
    def normalized_5m_bars(self) -> list[NormalizedBar]:
        loader = CSVLoader()
        loader.load(SAMPLE_CSV, symbol="XAUUSD")
        normalizer = BarNormalizer(
            validators=[validate_price_bounds, validate_ohlc_consistency, validate_volume],
            default_symbol="XAUUSD",
            default_timeframe="5m",
        )
        return [normalizer.normalize_bar(raw) for raw in loader.iter_bars()]

    def test_feature_engine_processes_all_bars(
        self, normalized_5m_bars: list[NormalizedBar]
    ) -> None:
        engine = FeatureEngine()
        features = engine.compute_batch(normalized_5m_bars)

        assert len(features) == len(normalized_5m_bars)

    def test_feature_vectors_have_correct_symbol(
        self, normalized_5m_bars: list[NormalizedBar]
    ) -> None:
        engine = FeatureEngine()
        features = engine.compute_batch(normalized_5m_bars)

        for fv in features:
            assert fv.symbol == "XAUUSD"
            assert fv.timeframe == Timeframe.M5

    def test_feature_timestamps_match_bars(
        self, normalized_5m_bars: list[NormalizedBar]
    ) -> None:
        engine = FeatureEngine()
        features = engine.compute_batch(normalized_5m_bars)

        for bar, fv in zip(normalized_5m_bars, features):
            assert fv.timestamp == bar.timestamp

    def test_warmup_progression(
        self, normalized_5m_bars: list[NormalizedBar]
    ) -> None:
        engine = FeatureEngine()
        features = engine.compute_batch(normalized_5m_bars)

        # Early features should not have warmup_complete
        assert not features[0].warmup_complete

        # With 60 bars and max warmup of 35 (MACD), later features should be ready
        warmup_complete = [f for f in features if f.warmup_complete]
        assert len(warmup_complete) > 0

    def test_rsi_values_in_range(
        self, normalized_5m_bars: list[NormalizedBar]
    ) -> None:
        engine = FeatureEngine()
        features = engine.compute_batch(normalized_5m_bars)

        for fv in features:
            if fv.rsi_14 is not None:
                assert 0.0 <= fv.rsi_14 <= 100.0
            if fv.rsi_7 is not None:
                assert 0.0 <= fv.rsi_7 <= 100.0

    def test_atr_positive(
        self, normalized_5m_bars: list[NormalizedBar]
    ) -> None:
        engine = FeatureEngine()
        features = engine.compute_batch(normalized_5m_bars)

        for fv in features:
            if fv.atr_14 is not None:
                assert fv.atr_14 >= 0.0

    def test_session_flags_set(
        self, normalized_5m_bars: list[NormalizedBar]
    ) -> None:
        """The sample data includes bars from different sessions."""
        engine = FeatureEngine()
        features = engine.compute_batch(normalized_5m_bars)

        london_flags = [f.is_london_session for f in features]
        ny_flags = [f.is_ny_session for f in features]
        asian_flags = [f.is_asian_session for f in features]

        # Our sample data has bars at 00:xx (Asian), 07:xx (London),
        # 12:xx (London+NY overlap), 22:xx (Asian)
        assert any(london_flags)
        assert any(ny_flags)
        assert any(asian_flags)


# ---------------------------------------------------------------------------
# Full Pipeline: CSV -> Normalize -> Aggregate -> Features
# ---------------------------------------------------------------------------


class TestFullPipeline:
    """End-to-end test covering the complete pipeline without external services."""

    def test_csv_to_features_pipeline(self) -> None:
        """Full pipeline: load CSV, normalize, compute features."""
        # Step 1: Load
        loader = CSVLoader()
        stats = loader.load(SAMPLE_CSV, symbol="XAUUSD")
        assert stats.rows_loaded == 60

        # Step 2: Normalize
        normalizer = BarNormalizer(
            validators=[validate_price_bounds, validate_ohlc_consistency, validate_volume],
            default_symbol="XAUUSD",
            default_timeframe="5m",
        )
        normalized = [normalizer.normalize_bar(raw) for raw in loader.iter_bars()]
        assert len(normalized) == 60

        # Step 3: Compute features on 5M bars
        engine = FeatureEngine()
        features_5m = engine.compute_batch(normalized)
        assert len(features_5m) == 60

        # Step 4: Aggregate to 15M
        bars_15m = aggregate_batch(normalized, Timeframe.M15)
        assert len(bars_15m) > 0

        # Step 5: Compute features on 15M bars
        engine_15m = FeatureEngine()
        features_15m = engine_15m.compute_batch(bars_15m)
        assert len(features_15m) == len(bars_15m)

        # Verify data integrity across pipeline
        for fv in features_5m:
            assert fv.symbol == "XAUUSD"
            assert fv.timeframe == Timeframe.M5

        for fv in features_15m:
            assert fv.symbol == "XAUUSD"
            assert fv.timeframe == Timeframe.M15

    def test_csv_to_aggregation_data_integrity(self) -> None:
        """Verify numerical integrity through CSV -> Normalize -> Aggregate."""
        loader = CSVLoader()
        loader.load(SAMPLE_CSV, symbol="XAUUSD")

        normalizer = BarNormalizer(
            validators=[validate_price_bounds, validate_ohlc_consistency, validate_volume],
            default_symbol="XAUUSD",
            default_timeframe="5m",
        )
        bars_5m = [normalizer.normalize_bar(raw) for raw in loader.iter_bars()]

        # Aggregate to 15M
        bars_15m = aggregate_batch(bars_5m, Timeframe.M15)

        for agg in bars_15m:
            # OHLC consistency
            assert agg.low <= agg.open <= agg.high
            assert agg.low <= agg.close <= agg.high

            # Aggregated high should be >= any 5M high in the same period
            period_end = agg.timestamp + timedelta(minutes=15)
            period_bars = [
                b for b in bars_5m
                if agg.timestamp <= b.timestamp < period_end
            ]
            if period_bars:
                max_high = max(b.high for b in period_bars)
                min_low = min(b.low for b in period_bars)
                assert agg.high == max_high
                assert agg.low == min_low

    def test_feature_serialization_roundtrip(self) -> None:
        """Verify features can be serialized and deserialized."""
        loader = CSVLoader()
        loader.load(SAMPLE_CSV, symbol="XAUUSD")

        normalizer = BarNormalizer(
            validators=[validate_price_bounds, validate_ohlc_consistency, validate_volume],
            default_symbol="XAUUSD",
            default_timeframe="5m",
        )
        bars = [normalizer.normalize_bar(raw) for raw in loader.iter_bars()]

        engine = FeatureEngine()
        features = engine.compute_batch(bars)

        # Test serialization roundtrip on a feature with warmup complete
        warmup_features = [f for f in features if f.warmup_complete]
        if warmup_features:
            fv = warmup_features[0]
            d = fv.to_dict()
            restored = FeatureVector.from_dict(d)

            assert restored.symbol == fv.symbol
            assert restored.timestamp == fv.timestamp
            assert restored.timeframe == fv.timeframe
            assert restored.rsi_14 == fv.rsi_14
            assert restored.warmup_complete == fv.warmup_complete

    def test_bar_serialization_roundtrip(self) -> None:
        """Verify normalized bars can be serialized and deserialized."""
        loader = CSVLoader()
        loader.load(SAMPLE_CSV, symbol="XAUUSD")

        normalizer = BarNormalizer(
            validators=[validate_price_bounds, validate_ohlc_consistency, validate_volume],
            default_symbol="XAUUSD",
            default_timeframe="5m",
        )
        bars = [normalizer.normalize_bar(raw) for raw in loader.iter_bars()]

        for bar in bars[:5]:
            d = bar.to_dict()
            restored = NormalizedBar.from_dict(d)

            assert restored.symbol == bar.symbol
            assert restored.timeframe == bar.timeframe
            assert restored.open == bar.open
            assert restored.high == bar.high
            assert restored.low == bar.low
            assert restored.close == bar.close
            assert restored.volume == bar.volume
