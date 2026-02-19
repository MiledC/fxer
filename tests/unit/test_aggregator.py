"""Comprehensive tests for bar aggregation (batch and streaming)."""

import csv
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from fxer.core.events import NormalizedBar
from fxer.core.exceptions import DataValidationError
from fxer.core.types import Timeframe
from fxer.data.aggregator.bar_aggregator import (
    AGGREGATION_MAP,
    BarAggregator,
    _align_timestamp,
    aggregate,
    aggregate_batch,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


def _bar(
    ts: datetime | str,
    open_: str = "2000.00",
    high: str = "2010.00",
    low: str = "1990.00",
    close: str = "2005.00",
    volume: str = "100",
    symbol: str = "XAUUSD",
    timeframe: Timeframe = Timeframe.M5,
    is_complete: bool = True,
) -> NormalizedBar:
    """Build a NormalizedBar with sensible defaults."""
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
    elif ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return NormalizedBar(
        symbol=symbol,
        timeframe=timeframe,
        timestamp=ts,
        open=Decimal(open_),
        high=Decimal(high),
        low=Decimal(low),
        close=Decimal(close),
        volume=Decimal(volume),
        is_complete=is_complete,
    )


def _make_5m_bars(
    start: datetime, count: int, symbol: str = "XAUUSD"
) -> list[NormalizedBar]:
    """Generate a sequence of ascending 5M bars starting at *start*."""
    bars = []
    base_price = Decimal("2000.00")
    for i in range(count):
        ts = start + timedelta(minutes=5 * i)
        o = base_price + Decimal(i)
        bars.append(
            _bar(
                ts=ts,
                open_=str(o),
                high=str(o + Decimal("5")),
                low=str(o - Decimal("1")),
                close=str(o + Decimal("3")),
                volume=str(100 + i * 10),
                symbol=symbol,
            )
        )
    return bars


def _load_fixture_bars() -> list[NormalizedBar]:
    """Load sample_bars.csv from fixtures as NormalizedBar instances."""
    path = FIXTURES_DIR / "sample_bars.csv"
    bars: list[NormalizedBar] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = datetime.fromisoformat(row["datetime"]).replace(tzinfo=timezone.utc)
            bars.append(
                NormalizedBar(
                    symbol="XAUUSD",
                    timeframe=Timeframe.M5,
                    timestamp=ts,
                    open=Decimal(row["open"]),
                    high=Decimal(row["high"]),
                    low=Decimal(row["low"]),
                    close=Decimal(row["close"]),
                    volume=Decimal(row["volume"]),
                )
            )
    return bars


# ---------------------------------------------------------------------------
# _align_timestamp
# ---------------------------------------------------------------------------

class TestAlignTimestamp:
    def test_align_to_15m(self):
        ts = datetime(2024, 1, 2, 0, 5, 0, tzinfo=timezone.utc)
        assert _align_timestamp(ts, Timeframe.M15) == datetime(
            2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc
        )

    def test_align_to_15m_mid_period(self):
        ts = datetime(2024, 1, 2, 0, 37, 0, tzinfo=timezone.utc)
        assert _align_timestamp(ts, Timeframe.M15) == datetime(
            2024, 1, 2, 0, 30, 0, tzinfo=timezone.utc
        )

    def test_align_to_h1(self):
        ts = datetime(2024, 1, 2, 7, 35, 0, tzinfo=timezone.utc)
        assert _align_timestamp(ts, Timeframe.H1) == datetime(
            2024, 1, 2, 7, 0, 0, tzinfo=timezone.utc
        )

    def test_align_to_h4(self):
        ts = datetime(2024, 1, 2, 1, 55, 0, tzinfo=timezone.utc)
        assert _align_timestamp(ts, Timeframe.H4) == datetime(
            2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc
        )

    def test_align_to_h4_second_period(self):
        ts = datetime(2024, 1, 2, 7, 10, 0, tzinfo=timezone.utc)
        assert _align_timestamp(ts, Timeframe.H4) == datetime(
            2024, 1, 2, 4, 0, 0, tzinfo=timezone.utc
        )

    def test_already_aligned(self):
        ts = datetime(2024, 1, 2, 1, 0, 0, tzinfo=timezone.utc)
        assert _align_timestamp(ts, Timeframe.H1) == ts

    def test_naive_timestamp_treated_as_utc(self):
        ts = datetime(2024, 1, 2, 0, 7, 30)
        result = _align_timestamp(ts, Timeframe.M15)
        assert result.tzinfo == timezone.utc
        assert result == datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# aggregate() — single group
# ---------------------------------------------------------------------------

class TestAggregate:
    def test_5m_to_15m_complete(self):
        start = datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)
        bars = _make_5m_bars(start, 3)
        result = aggregate(bars, Timeframe.M15)
        assert result.timeframe == Timeframe.M15
        assert result.timestamp == start
        assert result.open == bars[0].open
        assert result.close == bars[-1].close
        assert result.high == max(b.high for b in bars)
        assert result.low == min(b.low for b in bars)
        assert result.volume == sum(b.volume for b in bars)
        assert result.is_complete is True

    def test_5m_to_h1_complete(self):
        start = datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)
        bars = _make_5m_bars(start, 12)
        result = aggregate(bars, Timeframe.H1)
        assert result.timeframe == Timeframe.H1
        assert result.is_complete is True
        assert result.symbol == "XAUUSD"

    def test_5m_to_h4_complete(self):
        start = datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)
        bars = _make_5m_bars(start, 48)
        result = aggregate(bars, Timeframe.H4)
        assert result.timeframe == Timeframe.H4
        assert result.is_complete is True
        assert result.volume == sum(b.volume for b in bars)

    def test_incomplete_bar_fewer_than_needed(self):
        """Fewer bars than required -> is_complete=False."""
        start = datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)
        bars = _make_5m_bars(start, 2)  # need 3 for 15M
        result = aggregate(bars, Timeframe.M15)
        assert result.is_complete is False

    def test_ohlc_correctness_manual(self):
        """Manual OHLC values to verify aggregation logic precisely."""
        bars = [
            _bar("2024-01-02T00:00:00", open_="100", high="110", low="95", close="105", volume="10"),
            _bar("2024-01-02T00:05:00", open_="105", high="115", low="98", close="108", volume="20"),
            _bar("2024-01-02T00:10:00", open_="108", high="112", low="100", close="103", volume="30"),
        ]
        result = aggregate(bars, Timeframe.M15)
        assert result.open == Decimal("100")
        assert result.high == Decimal("115")  # max of 110, 115, 112
        assert result.low == Decimal("95")    # min of 95, 98, 100
        assert result.close == Decimal("103")
        assert result.volume == Decimal("60")

    def test_empty_bars_raises(self):
        with pytest.raises(DataValidationError, match="empty"):
            aggregate([], Timeframe.M15)

    def test_mixed_symbols_raises(self):
        bars = [
            _bar("2024-01-02T00:00:00", symbol="XAUUSD"),
            _bar("2024-01-02T00:05:00", symbol="DXY"),
        ]
        with pytest.raises(DataValidationError, match="Mixed symbols"):
            aggregate(bars, Timeframe.M15)

    def test_mixed_timeframes_raises(self):
        bars = [
            _bar("2024-01-02T00:00:00", timeframe=Timeframe.M5),
            _bar("2024-01-02T00:05:00", timeframe=Timeframe.M1),
        ]
        with pytest.raises(DataValidationError, match="Mixed timeframes"):
            aggregate(bars, Timeframe.M15)

    def test_unsupported_aggregation_raises(self):
        bars = [_bar("2024-01-02T00:00:00", timeframe=Timeframe.H4)]
        with pytest.raises(DataValidationError, match="Unsupported aggregation"):
            aggregate(bars, Timeframe.M5)  # H4 -> M5 is downsampling

    def test_single_bar_aggregation(self):
        """A single bar still aggregates (incomplete)."""
        bar = _bar("2024-01-02T00:00:00")
        result = aggregate([bar], Timeframe.M15)
        assert result.open == bar.open
        assert result.close == bar.close
        assert result.is_complete is False


# ---------------------------------------------------------------------------
# aggregate_batch() — multiple groups
# ---------------------------------------------------------------------------

class TestAggregateBatch:
    def test_two_15m_periods(self):
        start = datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)
        bars = _make_5m_bars(start, 6)  # 2 complete 15M periods
        result = aggregate_batch(bars, Timeframe.M15)
        assert len(result) == 2
        assert result[0].timestamp == datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)
        assert result[1].timestamp == datetime(2024, 1, 2, 0, 15, tzinfo=timezone.utc)
        assert all(b.is_complete for b in result)

    def test_one_complete_one_incomplete(self):
        start = datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)
        bars = _make_5m_bars(start, 5)  # 3 + 2 -> 1 complete, 1 incomplete
        result = aggregate_batch(bars, Timeframe.M15)
        assert len(result) == 2
        assert result[0].is_complete is True
        assert result[1].is_complete is False

    def test_empty_input(self):
        assert aggregate_batch([], Timeframe.M15) == []

    def test_batch_h1(self):
        start = datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)
        bars = _make_5m_bars(start, 24)  # 2 complete hours
        result = aggregate_batch(bars, Timeframe.H1)
        assert len(result) == 2
        assert all(b.is_complete for b in result)
        assert result[0].timestamp.hour == 0
        assert result[1].timestamp.hour == 1

    def test_batch_preserves_sort_order(self):
        start = datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)
        bars = _make_5m_bars(start, 12)
        result = aggregate_batch(bars, Timeframe.M15)
        timestamps = [b.timestamp for b in result]
        assert timestamps == sorted(timestamps)

    def test_batch_with_gaps(self):
        """Bars that skip a period still produce correct groups."""
        bars = (
            _make_5m_bars(datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc), 3)
            + _make_5m_bars(datetime(2024, 1, 2, 1, 0, tzinfo=timezone.utc), 3)
        )
        result = aggregate_batch(bars, Timeframe.M15)
        assert len(result) == 2
        assert result[0].timestamp == datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)
        assert result[1].timestamp == datetime(2024, 1, 2, 1, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# BarAggregator — streaming
# ---------------------------------------------------------------------------

class TestBarAggregator:
    def test_init(self):
        agg = BarAggregator(Timeframe.M5, Timeframe.M15)
        assert agg.source_timeframe == Timeframe.M5
        assert agg.target_timeframe == Timeframe.M15
        assert agg.bars_needed == 3
        assert agg.buffer_size == 0

    def test_unsupported_pair_raises(self):
        with pytest.raises(DataValidationError, match="Unsupported"):
            BarAggregator(Timeframe.H4, Timeframe.M5)

    def test_push_wrong_timeframe_raises(self):
        agg = BarAggregator(Timeframe.M5, Timeframe.M15)
        bar = _bar("2024-01-02T00:00:00", timeframe=Timeframe.M1)
        with pytest.raises(DataValidationError, match="Expected 5m"):
            agg.push(bar)

    def test_complete_15m_bar(self):
        agg = BarAggregator(Timeframe.M5, Timeframe.M15)
        bars = _make_5m_bars(datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc), 3)
        results = [agg.push(b) for b in bars]
        # First two pushes return None, third completes the period
        assert results[0] is None
        assert results[1] is None
        assert results[2] is not None
        assert results[2].timeframe == Timeframe.M15
        assert results[2].is_complete is True

    def test_complete_h1_bar(self):
        agg = BarAggregator(Timeframe.M5, Timeframe.H1)
        bars = _make_5m_bars(datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc), 12)
        results = [agg.push(b) for b in bars]
        non_none = [r for r in results if r is not None]
        assert len(non_none) == 1
        assert non_none[0].is_complete is True
        assert non_none[0].timeframe == Timeframe.H1

    def test_callback_on_completion(self):
        agg = BarAggregator(Timeframe.M5, Timeframe.M15)
        received: list[NormalizedBar] = []
        agg.on_bar(received.append)

        bars = _make_5m_bars(datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc), 3)
        for b in bars:
            agg.push(b)

        assert len(received) == 1
        assert received[0].timeframe == Timeframe.M15

    def test_multiple_callbacks(self):
        agg = BarAggregator(Timeframe.M5, Timeframe.M15)
        list_a: list[NormalizedBar] = []
        list_b: list[NormalizedBar] = []
        agg.on_bar(list_a.append)
        agg.on_bar(list_b.append)

        bars = _make_5m_bars(datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc), 3)
        for b in bars:
            agg.push(b)

        assert len(list_a) == 1
        assert len(list_b) == 1

    def test_gap_flushes_incomplete(self):
        """When a bar from a new period arrives while old period is incomplete,
        the old period is flushed with is_complete=False."""
        agg = BarAggregator(Timeframe.M5, Timeframe.M15)
        received: list[NormalizedBar] = []
        agg.on_bar(received.append)

        # Push 2 bars in the 00:00 period (incomplete)
        bars_period1 = _make_5m_bars(datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc), 2)
        for b in bars_period1:
            agg.push(b)

        assert len(received) == 0  # not yet flushed

        # Push a bar from the 00:15 period -- gap triggers flush of period 1
        bar_period2 = _bar("2024-01-02T00:15:00")
        agg.push(bar_period2)

        assert len(received) == 1
        assert received[0].is_complete is False  # only 2/3 bars
        assert received[0].timestamp == datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)

    def test_large_gap_flushes(self):
        """A jump from hour 0 to hour 7 flushes the incomplete period."""
        agg = BarAggregator(Timeframe.M5, Timeframe.H1)
        received: list[NormalizedBar] = []
        agg.on_bar(received.append)

        bar1 = _bar("2024-01-02T00:05:00")
        bar2 = _bar("2024-01-02T07:00:00")
        agg.push(bar1)
        agg.push(bar2)

        assert len(received) == 1
        assert received[0].is_complete is False

    def test_flush_manually(self):
        agg = BarAggregator(Timeframe.M5, Timeframe.M15)
        bars = _make_5m_bars(datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc), 2)
        for b in bars:
            agg.push(b)

        result = agg.flush()
        assert result is not None
        assert result.is_complete is False
        assert agg.buffer_size == 0

    def test_flush_empty_returns_none(self):
        agg = BarAggregator(Timeframe.M5, Timeframe.M15)
        assert agg.flush() is None

    def test_reset(self):
        agg = BarAggregator(Timeframe.M5, Timeframe.M15)
        bar = _bar("2024-01-02T00:00:00")
        agg.push(bar)
        assert agg.buffer_size == 1
        agg.reset()
        assert agg.buffer_size == 0

    def test_multiple_complete_periods(self):
        agg = BarAggregator(Timeframe.M5, Timeframe.M15)
        received: list[NormalizedBar] = []
        agg.on_bar(received.append)

        bars = _make_5m_bars(datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc), 9)
        for b in bars:
            agg.push(b)

        assert len(received) == 3
        for r in received:
            assert r.is_complete is True
            assert r.timeframe == Timeframe.M15

    def test_buffer_clears_after_completion(self):
        agg = BarAggregator(Timeframe.M5, Timeframe.M15)
        bars = _make_5m_bars(datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc), 3)
        for b in bars:
            agg.push(b)
        assert agg.buffer_size == 0

    def test_h4_aggregation_streaming(self):
        agg = BarAggregator(Timeframe.M5, Timeframe.H4)
        bars = _make_5m_bars(datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc), 48)
        received: list[NormalizedBar] = []
        agg.on_bar(received.append)
        for b in bars:
            agg.push(b)
        assert len(received) == 1
        assert received[0].is_complete is True
        assert received[0].timeframe == Timeframe.H4


# ---------------------------------------------------------------------------
# Fixture-based tests (sample_bars.csv)
# ---------------------------------------------------------------------------

class TestFixtureData:
    @pytest.fixture()
    def fixture_bars(self) -> list[NormalizedBar]:
        return _load_fixture_bars()

    def test_fixture_loads(self, fixture_bars: list[NormalizedBar]):
        assert len(fixture_bars) > 0
        assert all(b.timeframe == Timeframe.M5 for b in fixture_bars)

    def test_batch_15m_from_fixture(self, fixture_bars: list[NormalizedBar]):
        result = aggregate_batch(fixture_bars, Timeframe.M15)
        assert len(result) > 0
        for bar in result:
            assert bar.timeframe == Timeframe.M15
            assert bar.symbol == "XAUUSD"

    def test_batch_h1_from_fixture(self, fixture_bars: list[NormalizedBar]):
        result = aggregate_batch(fixture_bars, Timeframe.H1)
        assert len(result) > 0
        for bar in result:
            assert bar.timeframe == Timeframe.H1

    def test_batch_h4_from_fixture(self, fixture_bars: list[NormalizedBar]):
        result = aggregate_batch(fixture_bars, Timeframe.H4)
        assert len(result) > 0
        for bar in result:
            assert bar.timeframe == Timeframe.H4

    def test_15m_first_group_values(self, fixture_bars: list[NormalizedBar]):
        """First 3 bars (00:00, 00:05, 00:10) -> 15M bar at 00:00."""
        result = aggregate_batch(fixture_bars, Timeframe.M15)
        first = result[0]
        assert first.timestamp == datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)
        # open = first bar's open
        assert first.open == Decimal("2062.50")
        # high = max(2063.20, 2064.10, 2064.50)
        assert first.high == Decimal("2064.50")
        # low = min(2061.80, 2062.50, 2063.00)
        assert first.low == Decimal("2061.80")
        # close = third bar's close
        assert first.close == Decimal("2063.20")
        # volume = 1250 + 1380 + 1120
        assert first.volume == Decimal("3750")

    def test_streaming_matches_batch(self, fixture_bars: list[NormalizedBar]):
        """Streaming aggregator produces same results as batch for 15M."""
        batch_result = aggregate_batch(fixture_bars, Timeframe.M15)

        agg = BarAggregator(Timeframe.M5, Timeframe.M15)
        streamed: list[NormalizedBar] = []
        agg.on_bar(streamed.append)
        for b in fixture_bars:
            agg.push(b)
        # Flush remaining incomplete
        remaining = agg.flush()
        if remaining is not None:
            streamed.append(remaining)

        assert len(streamed) == len(batch_result)
        for s, b in zip(streamed, batch_result):
            assert s.timestamp == b.timestamp
            assert s.open == b.open
            assert s.high == b.high
            assert s.low == b.low
            assert s.close == b.close
            assert s.volume == b.volume
            assert s.is_complete == b.is_complete

    def test_gap_handling_in_fixture(self, fixture_bars: list[NormalizedBar]):
        """Fixture has gaps (01:55 -> 07:00); H4 periods are incomplete."""
        result = aggregate_batch(fixture_bars, Timeframe.H4)
        # Each H4 period needs 48 bars but fixture has at most 24 per period
        incomplete = [b for b in result if not b.is_complete]
        assert len(incomplete) > 0


# ---------------------------------------------------------------------------
# AGGREGATION_MAP coverage
# ---------------------------------------------------------------------------

class TestAggregationMap:
    def test_all_supported_pairs_exist(self):
        expected_pairs = [
            (Timeframe.M5, Timeframe.M15),
            (Timeframe.M5, Timeframe.H1),
            (Timeframe.M5, Timeframe.H4),
        ]
        for pair in expected_pairs:
            assert pair in AGGREGATION_MAP

    def test_ratios_are_correct(self):
        for (src, tgt), count in AGGREGATION_MAP.items():
            assert tgt.minutes // src.minutes == count
