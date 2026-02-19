"""Core type definitions for the fxEr trading system."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum


class Timeframe(Enum):
    """Supported bar timeframes."""

    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"

    @property
    def minutes(self) -> int:
        """Return the timeframe duration in minutes."""
        mapping = {
            Timeframe.M1: 1,
            Timeframe.M5: 5,
            Timeframe.M15: 15,
            Timeframe.H1: 60,
            Timeframe.H4: 240,
            Timeframe.D1: 1440,
        }
        return mapping[self]

    @classmethod
    def from_string(cls, value: str) -> "Timeframe":
        """Create Timeframe from string representation."""
        normalized = value.lower().strip()
        for tf in cls:
            if tf.value == normalized:
                return tf
        raise ValueError(f"Unknown timeframe: {value}")


@dataclass(frozen=True, slots=True)
class RawBar:
    """Raw bar data as received from a data source (broker, CSV, etc.).

    This is the input format before normalization. Fields may vary
    depending on the data source.
    """

    timestamp: datetime
    open: Decimal | float | str
    high: Decimal | float | str
    low: Decimal | float | str
    close: Decimal | float | str
    volume: Decimal | float | str | None = None
    symbol: str | None = None
    timeframe: str | None = None

    def to_decimal(self, value: Decimal | float | str | None) -> Decimal | None:
        """Convert a value to Decimal."""
        if value is None:
            return None
        if isinstance(value, Decimal):
            return value
        return Decimal(str(value))


class Symbol(Enum):
    """Supported trading symbols."""

    XAUUSD = "XAUUSD"
    DXY = "DXY"
    VIX = "VIX"


@dataclass(frozen=True, slots=True)
class SymbolSpec:
    """Specification for a trading symbol."""

    symbol: Symbol
    pip_size: Decimal
    min_price: Decimal
    max_price: Decimal
    description: str


# Symbol specifications
SYMBOL_SPECS: dict[Symbol, SymbolSpec] = {
    Symbol.XAUUSD: SymbolSpec(
        symbol=Symbol.XAUUSD,
        pip_size=Decimal("0.01"),
        min_price=Decimal("1000"),
        max_price=Decimal("10000"),
        description="Gold vs US Dollar",
    ),
    Symbol.DXY: SymbolSpec(
        symbol=Symbol.DXY,
        pip_size=Decimal("0.001"),
        min_price=Decimal("50"),
        max_price=Decimal("150"),
        description="US Dollar Index",
    ),
    Symbol.VIX: SymbolSpec(
        symbol=Symbol.VIX,
        pip_size=Decimal("0.01"),
        min_price=Decimal("5"),
        max_price=Decimal("100"),
        description="CBOE Volatility Index",
    ),
}
