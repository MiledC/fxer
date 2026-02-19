"""Data normalizer: converts RawBar to NormalizedBar with validation."""

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation

from fxer.core.events import NormalizedBar
from fxer.core.exceptions import DataValidationError
from fxer.core.types import RawBar, Timeframe
from fxer.data.normalizer.validators import (
    DEFAULT_VALIDATORS,
    Validator,
)


class BarNormalizer:
    """Normalizes raw bar data into the unified NormalizedBar format.

    Applies configurable validation rules, converts price fields to Decimal,
    and ensures timestamps are UTC.
    """

    def __init__(
        self,
        validators: list[Validator] | None = None,
        default_symbol: str = "XAUUSD",
        default_timeframe: str = "1m",
    ) -> None:
        self._validators = validators if validators is not None else list(DEFAULT_VALIDATORS)
        self._default_symbol = default_symbol
        self._default_timeframe = default_timeframe

    @property
    def validators(self) -> list[Validator]:
        """Return the current list of validators."""
        return self._validators

    def set_validators(self, validators: list[Validator]) -> None:
        """Replace the current validation rules."""
        self._validators = list(validators)

    def add_validator(self, validator: Validator) -> None:
        """Append an additional validator."""
        self._validators.append(validator)

    def normalize_bar(self, raw: RawBar) -> NormalizedBar:
        """Normalize a RawBar into a NormalizedBar.

        Steps:
        1. Resolve symbol and timeframe (use defaults if not provided).
        2. Run all configured validators.
        3. Convert price/volume fields to Decimal.
        4. Ensure timestamp is UTC.
        5. Construct and return NormalizedBar.

        Raises:
            DataValidationError: If any validation rule fails.
        """
        symbol = (raw.symbol or self._default_symbol).upper()
        tf_str = raw.timeframe or self._default_timeframe

        # Run validators
        for validator in self._validators:
            validator(raw, symbol)

        # Convert prices to Decimal
        open_price = self._to_decimal(raw.open, "open")
        high_price = self._to_decimal(raw.high, "high")
        low_price = self._to_decimal(raw.low, "low")
        close_price = self._to_decimal(raw.close, "close")
        volume = self._to_decimal(raw.volume, "volume") if raw.volume is not None else Decimal("0")

        # Ensure timestamp is UTC
        timestamp = self._ensure_utc(raw.timestamp)

        # Parse timeframe
        timeframe = Timeframe.from_string(tf_str)

        return NormalizedBar(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=timestamp,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume,
        )

    @staticmethod
    def _to_decimal(value: Decimal | float | str | None, field_name: str) -> Decimal:
        """Convert a value to Decimal."""
        if value is None:
            raise DataValidationError(
                message=f"Field '{field_name}' is None",
                field=field_name,
                value=value,
                rule="decimal_conversion",
            )
        if isinstance(value, Decimal):
            return value
        try:
            return Decimal(str(value))
        except (InvalidOperation, ValueError) as exc:
            raise DataValidationError(
                message=f"Cannot convert {field_name}={value!r} to Decimal",
                field=field_name,
                value=str(value),
                rule="decimal_conversion",
            ) from exc

    @staticmethod
    def _ensure_utc(ts: datetime) -> datetime:
        """Ensure a datetime is timezone-aware in UTC."""
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)
