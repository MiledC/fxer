"""Regime classification types for the fxEr trading system."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class RegimeState(Enum):
    """Market regime states."""

    LOW_VOL_TREND = "low_vol_trend"
    HIGH_VOL_TREND = "high_vol_trend"
    RANGING = "ranging"


@dataclass(frozen=True, slots=True)
class RegimeDecision:
    """Regime classification result with trading implications."""

    state: RegimeState
    confidence: float  # 0.0-1.0
    position_multiplier: float  # 0.5, 1.0, 1.5
    should_trade: bool
    reason: str
    hmm_prob: float | None = None
    adx_value: float | None = None
    atr_expanding: bool = False
    session_bias: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "state": self.state.value,
            "confidence": self.confidence,
            "position_multiplier": self.position_multiplier,
            "should_trade": self.should_trade,
            "reason": self.reason,
            "hmm_prob": self.hmm_prob,
            "adx_value": self.adx_value,
            "atr_expanding": self.atr_expanding,
            "session_bias": self.session_bias,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RegimeDecision":
        """Create from dictionary."""
        return cls(
            state=RegimeState(data["state"]),
            confidence=data["confidence"],
            position_multiplier=data["position_multiplier"],
            should_trade=data["should_trade"],
            reason=data["reason"],
            hmm_prob=data.get("hmm_prob"),
            adx_value=data.get("adx_value"),
            atr_expanding=data.get("atr_expanding", False),
            session_bias=data.get("session_bias", 1.0),
        )


@dataclass(frozen=True, slots=True)
class RegimeEvent:
    """Event wrapper for publishing regime decisions via EventBus."""

    decision: RegimeDecision
    symbol: str
    timestamp: datetime
    event_time: datetime = field(default_factory=datetime.utcnow)

    @property
    def topic(self) -> str:
        """Generate topic string for pub/sub routing."""
        return f"regime.{self.symbol.lower()}"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "decision": self.decision.to_dict(),
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "event_time": self.event_time.isoformat(),
        }