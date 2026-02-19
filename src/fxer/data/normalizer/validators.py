"""Validation rules for raw bar data before normalization."""

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Callable

from fxer.core.exceptions import DataValidationError
from fxer.core.types import RawBar, Symbol, SymbolSpec, SYMBOL_SPECS


# Type alias for validator functions
Validator = Callable[[RawBar, str], None]


def validate_timestamp_not_future(bar: RawBar, symbol: str) -> None:
    """Reject bars with timestamps in the future.

    Allows up to 5 seconds of clock skew tolerance.
    """
    now = datetime.now(timezone.utc)
    ts = bar.timestamp
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    delta = (ts - now).total_seconds()
    if delta > 5:
        raise DataValidationError(
            message=f"Timestamp is in the future by {delta:.1f}s: {bar.timestamp}",
            field="timestamp",
            value=bar.timestamp.isoformat(),
            rule="timestamp_not_future",
        )


def validate_timestamp_not_stale(
    bar: RawBar, symbol: str, max_staleness_seconds: int = 120
) -> None:
    """Reject bars with timestamps that are too old.

    Default staleness threshold is 120 seconds, appropriate for 1M bars
    with some buffer.
    """
    now = datetime.now(timezone.utc)
    ts = bar.timestamp
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    age = (now - ts).total_seconds()
    if age > max_staleness_seconds:
        raise DataValidationError(
            message=f"Timestamp is stale by {age:.1f}s (max {max_staleness_seconds}s): {bar.timestamp}",
            field="timestamp",
            value=bar.timestamp.isoformat(),
            rule="timestamp_not_stale",
        )


def _to_decimal(value: Decimal | float | str | None, field_name: str) -> Decimal:
    """Convert a price value to Decimal, raising on failure."""
    if value is None:
        raise DataValidationError(
            message=f"Field '{field_name}' is None",
            field=field_name,
            value=value,
            rule="not_none",
        )
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise DataValidationError(
            message=f"Cannot convert {field_name}={value!r} to Decimal",
            field=field_name,
            value=str(value),
            rule="decimal_conversion",
        ) from exc


def _get_symbol_spec(symbol: str) -> SymbolSpec | None:
    """Look up the SymbolSpec for a given symbol string."""
    try:
        sym = Symbol(symbol.upper())
    except ValueError:
        return None
    return SYMBOL_SPECS.get(sym)


def validate_price_bounds(bar: RawBar, symbol: str) -> None:
    """Validate that all price fields are within the symbol's allowed range."""
    spec = _get_symbol_spec(symbol)
    if spec is None:
        # Unknown symbol -- skip price bound checks
        return

    for field_name in ("open", "high", "low", "close"):
        raw_value = getattr(bar, field_name)
        price = _to_decimal(raw_value, field_name)
        if price < spec.min_price or price > spec.max_price:
            raise DataValidationError(
                message=(
                    f"{field_name} price {price} is outside "
                    f"[{spec.min_price}, {spec.max_price}] for {symbol}"
                ),
                field=field_name,
                value=str(price),
                rule="price_bounds",
            )


def validate_ohlc_consistency(bar: RawBar, symbol: str) -> None:
    """Validate OHLC relationship: low <= open,close <= high."""
    o = _to_decimal(bar.open, "open")
    h = _to_decimal(bar.high, "high")
    lo = _to_decimal(bar.low, "low")
    c = _to_decimal(bar.close, "close")

    if lo > h:
        raise DataValidationError(
            message=f"low ({lo}) > high ({h})",
            field="low",
            value=str(lo),
            rule="ohlc_consistency",
        )
    if not (lo <= o <= h):
        raise DataValidationError(
            message=f"open ({o}) not in range [{lo}, {h}]",
            field="open",
            value=str(o),
            rule="ohlc_consistency",
        )
    if not (lo <= c <= h):
        raise DataValidationError(
            message=f"close ({c}) not in range [{lo}, {h}]",
            field="close",
            value=str(c),
            rule="ohlc_consistency",
        )


def validate_volume(bar: RawBar, symbol: str) -> None:
    """Validate that volume is non-negative."""
    if bar.volume is None:
        # Volume may be absent for some sources; treat as zero
        return
    vol = _to_decimal(bar.volume, "volume")
    if vol < 0:
        raise DataValidationError(
            message=f"volume ({vol}) is negative",
            field="volume",
            value=str(vol),
            rule="volume_non_negative",
        )


# Default set of validators applied during normalization
DEFAULT_VALIDATORS: list[Validator] = [
    validate_timestamp_not_future,
    validate_price_bounds,
    validate_ohlc_consistency,
    validate_volume,
]

# Extended validators that include staleness check (for live data)
LIVE_VALIDATORS: list[Validator] = [
    validate_timestamp_not_future,
    validate_timestamp_not_stale,
    validate_price_bounds,
    validate_ohlc_consistency,
    validate_volume,
]
