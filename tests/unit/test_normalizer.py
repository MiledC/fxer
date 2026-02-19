"""Comprehensive tests for data normalizer and validators."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from fxer.core.events import NormalizedBar
from fxer.core.exceptions import DataValidationError
from fxer.core.types import RawBar, Timeframe
from fxer.data.normalizer.normalizer import BarNormalizer
from fxer.data.normalizer.validators import (
    DEFAULT_VALIDATORS,
    LIVE_VALIDATORS,
    validate_ohlc_consistency,
    validate_price_bounds,
    validate_timestamp_not_future,
    validate_timestamp_not_stale,
    validate_volume,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _make_raw(
    timestamp: datetime | None = None,
    open_: float = 2000.0,
    high: float = 2010.0,
    low: float = 1990.0,
    close: float = 2005.0,
    volume: float = 100.0,
    symbol: str | None = "XAUUSD",
    timeframe: str | None = "1m",
) -> RawBar:
    """Build a valid RawBar with sensible defaults."""
    return RawBar(
        timestamp=timestamp or _utc_now(),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        symbol=symbol,
        timeframe=timeframe,
    )


# ---------------------------------------------------------------------------
# Timestamp validators
# ---------------------------------------------------------------------------

class TestTimestampNotFuture:
    def test_current_timestamp_passes(self):
        bar = _make_raw(timestamp=_utc_now())
        validate_timestamp_not_future(bar, "XAUUSD")  # no exception

    def test_past_timestamp_passes(self):
        bar = _make_raw(timestamp=_utc_now() - timedelta(seconds=30))
        validate_timestamp_not_future(bar, "XAUUSD")

    def test_future_timestamp_rejected(self):
        bar = _make_raw(timestamp=_utc_now() + timedelta(minutes=5))
        with pytest.raises(DataValidationError, match="future"):
            validate_timestamp_not_future(bar, "XAUUSD")

    def test_slight_clock_skew_allowed(self):
        """Up to 5 seconds ahead should be tolerated."""
        bar = _make_raw(timestamp=_utc_now() + timedelta(seconds=3))
        validate_timestamp_not_future(bar, "XAUUSD")

    def test_naive_timestamp_treated_as_utc(self):
        naive_now = datetime.now(timezone.utc).replace(tzinfo=None)
        bar = _make_raw(timestamp=naive_now)
        validate_timestamp_not_future(bar, "XAUUSD")

    def test_error_details(self):
        bar = _make_raw(timestamp=_utc_now() + timedelta(hours=1))
        with pytest.raises(DataValidationError) as exc_info:
            validate_timestamp_not_future(bar, "XAUUSD")
        assert exc_info.value.field == "timestamp"
        assert exc_info.value.rule == "timestamp_not_future"


class TestTimestampNotStale:
    def test_recent_timestamp_passes(self):
        bar = _make_raw(timestamp=_utc_now() - timedelta(seconds=10))
        validate_timestamp_not_stale(bar, "XAUUSD")

    def test_stale_timestamp_rejected(self):
        bar = _make_raw(timestamp=_utc_now() - timedelta(minutes=5))
        with pytest.raises(DataValidationError, match="stale"):
            validate_timestamp_not_stale(bar, "XAUUSD")

    def test_custom_staleness_threshold(self):
        bar = _make_raw(timestamp=_utc_now() - timedelta(seconds=300))
        # Should pass with a 600s threshold
        validate_timestamp_not_stale(bar, "XAUUSD", max_staleness_seconds=600)
        # Should fail with a 60s threshold
        with pytest.raises(DataValidationError):
            validate_timestamp_not_stale(bar, "XAUUSD", max_staleness_seconds=60)

    def test_error_details(self):
        bar = _make_raw(timestamp=_utc_now() - timedelta(hours=1))
        with pytest.raises(DataValidationError) as exc_info:
            validate_timestamp_not_stale(bar, "XAUUSD")
        assert exc_info.value.field == "timestamp"
        assert exc_info.value.rule == "timestamp_not_stale"


# ---------------------------------------------------------------------------
# Price bounds validators
# ---------------------------------------------------------------------------

class TestPriceBounds:
    def test_valid_xauusd_prices(self):
        bar = _make_raw(open_=2000, high=2010, low=1990, close=2005)
        validate_price_bounds(bar, "XAUUSD")

    def test_price_at_lower_bound(self):
        bar = _make_raw(open_=1000, high=1010, low=1000, close=1005)
        validate_price_bounds(bar, "XAUUSD")

    def test_price_at_upper_bound(self):
        bar = _make_raw(open_=9990, high=10000, low=9990, close=9995)
        validate_price_bounds(bar, "XAUUSD")

    def test_price_below_minimum(self):
        bar = _make_raw(open_=500, high=510, low=490, close=505)
        with pytest.raises(DataValidationError, match="outside"):
            validate_price_bounds(bar, "XAUUSD")

    def test_price_above_maximum(self):
        bar = _make_raw(open_=15000, high=15010, low=14990, close=15005)
        with pytest.raises(DataValidationError, match="outside"):
            validate_price_bounds(bar, "XAUUSD")

    def test_unknown_symbol_skips_check(self):
        bar = _make_raw(open_=99999, high=99999, low=99999, close=99999)
        # Should not raise for unknown symbols
        validate_price_bounds(bar, "UNKNOWN")

    def test_dxy_bounds(self):
        bar = _make_raw(open_=100, high=101, low=99, close=100.5)
        validate_price_bounds(bar, "DXY")

    def test_dxy_out_of_bounds(self):
        bar = _make_raw(open_=200, high=201, low=199, close=200)
        with pytest.raises(DataValidationError):
            validate_price_bounds(bar, "DXY")

    def test_error_contains_field_info(self):
        bar = _make_raw(open_=500, high=510, low=490, close=505)
        with pytest.raises(DataValidationError) as exc_info:
            validate_price_bounds(bar, "XAUUSD")
        assert exc_info.value.rule == "price_bounds"
        assert exc_info.value.field in ("open", "high", "low", "close")


# ---------------------------------------------------------------------------
# OHLC consistency validators
# ---------------------------------------------------------------------------

class TestOHLCConsistency:
    def test_valid_ohlc(self):
        bar = _make_raw(open_=2000, high=2010, low=1990, close=2005)
        validate_ohlc_consistency(bar, "XAUUSD")

    def test_open_equals_high_and_low(self):
        """All prices equal (flat bar) is valid."""
        bar = _make_raw(open_=2000, high=2000, low=2000, close=2000)
        validate_ohlc_consistency(bar, "XAUUSD")

    def test_low_greater_than_high(self):
        bar = _make_raw(open_=2000, high=1990, low=2010, close=2000)
        with pytest.raises(DataValidationError, match="low.*>.*high"):
            validate_ohlc_consistency(bar, "XAUUSD")

    def test_open_below_low(self):
        bar = _make_raw(open_=1980, high=2010, low=1990, close=2005)
        with pytest.raises(DataValidationError, match="open.*not in range"):
            validate_ohlc_consistency(bar, "XAUUSD")

    def test_open_above_high(self):
        bar = _make_raw(open_=2020, high=2010, low=1990, close=2005)
        with pytest.raises(DataValidationError, match="open.*not in range"):
            validate_ohlc_consistency(bar, "XAUUSD")

    def test_close_below_low(self):
        bar = _make_raw(open_=2000, high=2010, low=1990, close=1980)
        with pytest.raises(DataValidationError, match="close.*not in range"):
            validate_ohlc_consistency(bar, "XAUUSD")

    def test_close_above_high(self):
        bar = _make_raw(open_=2000, high=2010, low=1990, close=2020)
        with pytest.raises(DataValidationError, match="close.*not in range"):
            validate_ohlc_consistency(bar, "XAUUSD")

    def test_string_prices(self):
        """String values should be parsed and validated."""
        bar = RawBar(
            timestamp=_utc_now(),
            open="2000.50",
            high="2010.75",
            low="1990.25",
            close="2005.50",
            volume="100",
        )
        validate_ohlc_consistency(bar, "XAUUSD")


# ---------------------------------------------------------------------------
# Volume validator
# ---------------------------------------------------------------------------

class TestVolume:
    def test_positive_volume(self):
        bar = _make_raw(volume=100.0)
        validate_volume(bar, "XAUUSD")

    def test_zero_volume(self):
        bar = _make_raw(volume=0)
        validate_volume(bar, "XAUUSD")

    def test_none_volume(self):
        """None volume is permitted (treated as absent)."""
        bar = _make_raw(volume=None)
        validate_volume(bar, "XAUUSD")

    def test_negative_volume_rejected(self):
        bar = _make_raw(volume=-10)
        with pytest.raises(DataValidationError, match="negative"):
            validate_volume(bar, "XAUUSD")

    def test_error_details_on_negative_volume(self):
        bar = _make_raw(volume=-5)
        with pytest.raises(DataValidationError) as exc_info:
            validate_volume(bar, "XAUUSD")
        assert exc_info.value.field == "volume"
        assert exc_info.value.rule == "volume_non_negative"


# ---------------------------------------------------------------------------
# Normalizer
# ---------------------------------------------------------------------------

class TestBarNormalizer:
    def setup_method(self):
        self.normalizer = BarNormalizer(
            validators=[
                validate_timestamp_not_future,
                validate_price_bounds,
                validate_ohlc_consistency,
                validate_volume,
            ]
        )

    def test_normalize_valid_bar(self):
        raw = _make_raw()
        result = self.normalizer.normalize_bar(raw)
        assert isinstance(result, NormalizedBar)
        assert result.symbol == "XAUUSD"
        assert result.timeframe == Timeframe.M1
        assert isinstance(result.open, Decimal)
        assert isinstance(result.high, Decimal)
        assert isinstance(result.low, Decimal)
        assert isinstance(result.close, Decimal)
        assert isinstance(result.volume, Decimal)

    def test_prices_converted_to_decimal(self):
        raw = _make_raw(open_=2000.5, high=2010.75, low=1990.25, close=2005.5)
        result = self.normalizer.normalize_bar(raw)
        assert result.open == Decimal("2000.5")
        assert result.high == Decimal("2010.75")
        assert result.low == Decimal("1990.25")
        assert result.close == Decimal("2005.5")

    def test_string_prices_converted(self):
        raw = RawBar(
            timestamp=_utc_now(),
            open="2000.50",
            high="2010.75",
            low="1990.25",
            close="2005.50",
            volume="150",
            symbol="XAUUSD",
            timeframe="1m",
        )
        result = self.normalizer.normalize_bar(raw)
        assert result.open == Decimal("2000.50")
        assert result.volume == Decimal("150")

    def test_none_volume_defaults_to_zero(self):
        raw = _make_raw(volume=None)
        result = self.normalizer.normalize_bar(raw)
        assert result.volume == Decimal("0")

    def test_naive_timestamp_gets_utc(self):
        naive = datetime(2025, 6, 15, 12, 0, 0)
        raw = _make_raw(timestamp=naive)
        result = self.normalizer.normalize_bar(raw)
        assert result.timestamp.tzinfo is not None
        assert result.timestamp.tzinfo == timezone.utc

    def test_non_utc_timestamp_converted(self):
        from datetime import timezone as tz

        est = tz(timedelta(hours=-5))
        ts = datetime(2025, 6, 15, 12, 0, 0, tzinfo=est)
        raw = _make_raw(timestamp=ts)
        result = self.normalizer.normalize_bar(raw)
        assert result.timestamp.tzinfo == timezone.utc
        assert result.timestamp.hour == 17  # 12 EST = 17 UTC

    def test_default_symbol_used_when_none(self):
        raw = _make_raw(symbol=None)
        normalizer = BarNormalizer(
            validators=[validate_ohlc_consistency],
            default_symbol="XAUUSD",
        )
        result = normalizer.normalize_bar(raw)
        assert result.symbol == "XAUUSD"

    def test_default_timeframe_used_when_none(self):
        raw = _make_raw(timeframe=None)
        normalizer = BarNormalizer(
            validators=[validate_ohlc_consistency],
            default_timeframe="5m",
        )
        result = normalizer.normalize_bar(raw)
        assert result.timeframe == Timeframe.M5

    def test_symbol_uppercased(self):
        raw = _make_raw(symbol="xauusd")
        normalizer = BarNormalizer(validators=[validate_ohlc_consistency])
        result = normalizer.normalize_bar(raw)
        assert result.symbol == "XAUUSD"

    def test_validation_error_propagated(self):
        raw = _make_raw(open_=500, high=510, low=490, close=505)
        with pytest.raises(DataValidationError, match="price"):
            self.normalizer.normalize_bar(raw)

    def test_future_timestamp_error_propagated(self):
        raw = _make_raw(timestamp=_utc_now() + timedelta(hours=1))
        with pytest.raises(DataValidationError, match="future"):
            self.normalizer.normalize_bar(raw)

    def test_ohlc_consistency_error_propagated(self):
        raw = _make_raw(open_=2020, high=2010, low=1990, close=2005)
        with pytest.raises(DataValidationError, match="open.*not in range"):
            self.normalizer.normalize_bar(raw)

    def test_negative_volume_error_propagated(self):
        raw = _make_raw(volume=-100)
        with pytest.raises(DataValidationError, match="negative"):
            self.normalizer.normalize_bar(raw)

    def test_no_validators(self):
        """Normalizer with no validators just converts data."""
        normalizer = BarNormalizer(validators=[])
        raw = _make_raw(open_=500, high=510, low=490, close=505)
        result = normalizer.normalize_bar(raw)
        assert result.open == Decimal("500")

    def test_set_validators(self):
        normalizer = BarNormalizer(validators=[])
        normalizer.set_validators([validate_volume])
        assert len(normalizer.validators) == 1
        assert normalizer.validators[0] is validate_volume

    def test_add_validator(self):
        normalizer = BarNormalizer(validators=[validate_volume])
        normalizer.add_validator(validate_ohlc_consistency)
        assert len(normalizer.validators) == 2

    def test_different_timeframes(self):
        for tf_str, expected_tf in [
            ("1m", Timeframe.M1),
            ("5m", Timeframe.M5),
            ("15m", Timeframe.M15),
            ("1h", Timeframe.H1),
            ("4h", Timeframe.H4),
            ("1d", Timeframe.D1),
        ]:
            raw = _make_raw(timeframe=tf_str)
            normalizer = BarNormalizer(validators=[])
            result = normalizer.normalize_bar(raw)
            assert result.timeframe == expected_tf, f"Failed for {tf_str}"

    def test_decimal_input_preserved(self):
        raw = RawBar(
            timestamp=_utc_now(),
            open=Decimal("2000.12345"),
            high=Decimal("2010.12345"),
            low=Decimal("1990.12345"),
            close=Decimal("2005.12345"),
            volume=Decimal("500"),
            symbol="XAUUSD",
            timeframe="1m",
        )
        normalizer = BarNormalizer(validators=[validate_ohlc_consistency])
        result = normalizer.normalize_bar(raw)
        assert result.open == Decimal("2000.12345")


# ---------------------------------------------------------------------------
# Validator list defaults
# ---------------------------------------------------------------------------

class TestValidatorLists:
    def test_default_validators_do_not_include_staleness(self):
        validator_names = [v.__name__ for v in DEFAULT_VALIDATORS]
        assert "validate_timestamp_not_stale" not in validator_names

    def test_live_validators_include_staleness(self):
        validator_names = [v.__name__ for v in LIVE_VALIDATORS]
        assert "validate_timestamp_not_stale" in validator_names

    def test_default_validators_count(self):
        assert len(DEFAULT_VALIDATORS) == 4

    def test_live_validators_count(self):
        assert len(LIVE_VALIDATORS) == 5
