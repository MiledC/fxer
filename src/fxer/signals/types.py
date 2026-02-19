"""Core types for the signal generation module."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class Direction(Enum):
    """Trade signal direction."""

    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


class HoldingPeriod(Enum):
    """Expected holding period for a trade signal."""

    MIN_15 = "15m"
    MIN_30 = "30m"
    HOUR_1 = "1h"
    HOUR_2 = "2h"

    @property
    def bars_5m(self) -> int:
        """Number of 5-minute bars in this holding period."""
        mapping = {
            HoldingPeriod.MIN_15: 3,
            HoldingPeriod.MIN_30: 6,
            HoldingPeriod.HOUR_1: 12,
            HoldingPeriod.HOUR_2: 24,
        }
        return mapping[self]

    @classmethod
    def from_bars(cls, bars: int) -> "HoldingPeriod":
        """Determine holding period from number of 5m bars."""
        if bars <= 3:
            return cls.MIN_15
        elif bars <= 6:
            return cls.MIN_30
        elif bars <= 12:
            return cls.HOUR_1
        else:
            return cls.HOUR_2


@dataclass(frozen=True, slots=True)
class ModelPrediction:
    """Raw prediction output from a single model."""

    prob_long: float
    prob_short: float
    raw_output: float

    @property
    def predicted_direction(self) -> Direction:
        """Direction based on higher probability."""
        if self.prob_long > self.prob_short:
            return Direction.LONG
        elif self.prob_short > self.prob_long:
            return Direction.SHORT
        return Direction.NEUTRAL


@dataclass(frozen=True, slots=True)
class TradeSignal:
    """Complete trade signal with ensemble predictions and metadata."""

    symbol: str
    timestamp: datetime
    direction: Direction
    confidence: float  # 0.0-1.0
    predicted_horizon: HoldingPeriod
    xgboost_prob: float | None
    lstm_prob: float | None
    ensemble_prob: float | None
    top_features: dict[str, float]  # Top-5 SHAP values
    warmup_complete: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for ZeroMQ transport."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "direction": self.direction.value,
            "confidence": self.confidence,
            "predicted_horizon": self.predicted_horizon.value,
            "xgboost_prob": self.xgboost_prob,
            "lstm_prob": self.lstm_prob,
            "ensemble_prob": self.ensemble_prob,
            "top_features": self.top_features,
            "warmup_complete": self.warmup_complete,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TradeSignal":
        """Deserialize from dictionary."""
        return cls(
            symbol=data["symbol"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            direction=Direction(data["direction"]),
            confidence=data["confidence"],
            predicted_horizon=HoldingPeriod(data["predicted_horizon"]),
            xgboost_prob=data.get("xgboost_prob"),
            lstm_prob=data.get("lstm_prob"),
            ensemble_prob=data.get("ensemble_prob"),
            top_features=data.get("top_features", {}),
            warmup_complete=data.get("warmup_complete", False),
        )
