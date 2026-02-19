"""Core event types for the fxEr trading system."""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

from fxer.core.types import Timeframe


@dataclass(frozen=True, slots=True)
class NormalizedBar:
    """Unified bar format used throughout the system.

    All bars are converted to this format after normalization,
    regardless of the original data source.
    """

    symbol: str
    timeframe: Timeframe
    timestamp: datetime  # Bar open time (UTC)
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    is_complete: bool = True

    def __post_init__(self) -> None:
        """Validate OHLC consistency."""
        if not (self.low <= self.open <= self.high):
            raise ValueError(
                f"Invalid OHLC: open={self.open} not in range [{self.low}, {self.high}]"
            )
        if not (self.low <= self.close <= self.high):
            raise ValueError(
                f"Invalid OHLC: close={self.close} not in range [{self.low}, {self.high}]"
            )

    @property
    def mid_price(self) -> Decimal:
        """Calculate mid price (high + low) / 2."""
        return (self.high + self.low) / 2

    @property
    def typical_price(self) -> Decimal:
        """Calculate typical price (high + low + close) / 3."""
        return (self.high + self.low + self.close) / 3

    @property
    def range(self) -> Decimal:
        """Calculate bar range (high - low)."""
        return self.high - self.low

    @property
    def body(self) -> Decimal:
        """Calculate bar body (close - open)."""
        return self.close - self.open

    @property
    def is_bullish(self) -> bool:
        """Check if bar is bullish (close > open)."""
        return self.close > self.open

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe.value,
            "timestamp": self.timestamp.isoformat(),
            "open": str(self.open),
            "high": str(self.high),
            "low": str(self.low),
            "close": str(self.close),
            "volume": str(self.volume),
            "is_complete": self.is_complete,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NormalizedBar":
        """Create from dictionary."""
        return cls(
            symbol=data["symbol"],
            timeframe=Timeframe.from_string(data["timeframe"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            open=Decimal(data["open"]),
            high=Decimal(data["high"]),
            low=Decimal(data["low"]),
            close=Decimal(data["close"]),
            volume=Decimal(data["volume"]),
            is_complete=data.get("is_complete", True),
        )


@dataclass(frozen=True, slots=True)
class FeatureVector:
    """Feature vector for a single bar.

    Phase 1a: Technical + Temporal features only.
    Phase 1b+: Cross-asset features will be added.
    """

    symbol: str
    timestamp: datetime
    timeframe: Timeframe

    # Technical indicators
    rsi_14: float | None = None
    rsi_7: float | None = None
    macd_line: float | None = None
    macd_signal: float | None = None
    macd_histogram: float | None = None
    bb_upper: float | None = None
    bb_middle: float | None = None
    bb_lower: float | None = None
    bb_width: float | None = None  # (upper - lower) / middle
    atr_14: float | None = None

    # Session flags
    is_london_session: bool = False  # 07:00-16:00 UTC
    is_ny_session: bool = False  # 12:00-21:00 UTC
    is_overlap_session: bool = False  # 12:00-16:00 UTC (peak liquidity)
    is_asian_session: bool = False  # 22:00-07:00 UTC

    # Temporal features
    hour_of_day: int = 0  # 0-23 UTC
    day_of_week: int = 0  # 0=Monday, 6=Sunday
    is_month_turn: bool = False  # First/last 2 trading days

    # Meta
    warmup_complete: bool = False  # All indicators have sufficient data

    # Phase 1b+ placeholders (cross-asset features)
    dxy_return_1h: float | None = None
    dxy_rsi_14: float | None = None
    vix_level: float | None = None
    vix_change: float | None = None
    tips_yield_10y: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "timeframe": self.timeframe.value,
            "rsi_14": self.rsi_14,
            "rsi_7": self.rsi_7,
            "macd_line": self.macd_line,
            "macd_signal": self.macd_signal,
            "macd_histogram": self.macd_histogram,
            "bb_upper": self.bb_upper,
            "bb_middle": self.bb_middle,
            "bb_lower": self.bb_lower,
            "bb_width": self.bb_width,
            "atr_14": self.atr_14,
            "is_london_session": self.is_london_session,
            "is_ny_session": self.is_ny_session,
            "is_overlap_session": self.is_overlap_session,
            "is_asian_session": self.is_asian_session,
            "hour_of_day": self.hour_of_day,
            "day_of_week": self.day_of_week,
            "is_month_turn": self.is_month_turn,
            "warmup_complete": self.warmup_complete,
            "dxy_return_1h": self.dxy_return_1h,
            "dxy_rsi_14": self.dxy_rsi_14,
            "vix_level": self.vix_level,
            "vix_change": self.vix_change,
            "tips_yield_10y": self.tips_yield_10y,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeatureVector":
        """Create from dictionary."""
        return cls(
            symbol=data["symbol"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            timeframe=Timeframe.from_string(data["timeframe"]),
            rsi_14=data.get("rsi_14"),
            rsi_7=data.get("rsi_7"),
            macd_line=data.get("macd_line"),
            macd_signal=data.get("macd_signal"),
            macd_histogram=data.get("macd_histogram"),
            bb_upper=data.get("bb_upper"),
            bb_middle=data.get("bb_middle"),
            bb_lower=data.get("bb_lower"),
            bb_width=data.get("bb_width"),
            atr_14=data.get("atr_14"),
            is_london_session=data.get("is_london_session", False),
            is_ny_session=data.get("is_ny_session", False),
            is_overlap_session=data.get("is_overlap_session", False),
            is_asian_session=data.get("is_asian_session", False),
            hour_of_day=data.get("hour_of_day", 0),
            day_of_week=data.get("day_of_week", 0),
            is_month_turn=data.get("is_month_turn", False),
            warmup_complete=data.get("warmup_complete", False),
            dxy_return_1h=data.get("dxy_return_1h"),
            dxy_rsi_14=data.get("dxy_rsi_14"),
            vix_level=data.get("vix_level"),
            vix_change=data.get("vix_change"),
            tips_yield_10y=data.get("tips_yield_10y"),
        )


@dataclass
class BarEvent:
    """Event wrapper for publishing bars via ZeroMQ."""

    bar: NormalizedBar
    event_time: datetime = field(default_factory=datetime.utcnow)
    source: str = "unknown"

    @property
    def topic(self) -> str:
        """Generate topic string for pub/sub routing."""
        return f"bar.{self.bar.symbol.lower()}.{self.bar.timeframe.value}"


@dataclass
class FeatureEvent:
    """Event wrapper for publishing features via ZeroMQ."""

    features: FeatureVector
    event_time: datetime = field(default_factory=datetime.utcnow)

    @property
    def topic(self) -> str:
        """Generate topic string for pub/sub routing."""
        return f"features.{self.features.symbol.lower()}"
