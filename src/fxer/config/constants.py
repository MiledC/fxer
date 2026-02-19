"""Trading constants and session definitions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TradingSession:
    """Trading session definition with UTC hours."""

    name: str
    start_hour: int  # UTC hour (0-23)
    end_hour: int  # UTC hour (0-23)

    def is_active(self, hour: int) -> bool:
        """Check if the given UTC hour is within this session.

        Handles sessions that cross midnight (e.g., Asian session).
        """
        if self.start_hour <= self.end_hour:
            # Normal session (doesn't cross midnight)
            return self.start_hour <= hour < self.end_hour
        else:
            # Session crosses midnight
            return hour >= self.start_hour or hour < self.end_hour


# Trading session definitions (UTC)
LONDON_SESSION = TradingSession(name="London", start_hour=7, end_hour=16)
NY_SESSION = TradingSession(name="New York", start_hour=12, end_hour=21)
OVERLAP_SESSION = TradingSession(name="Overlap", start_hour=12, end_hour=16)
ASIAN_SESSION = TradingSession(name="Asian", start_hour=22, end_hour=7)


# Feature computation constants
INDICATOR_PARAMS = {
    "rsi_14": {"period": 14},
    "rsi_7": {"period": 7},
    "macd": {"fast": 12, "slow": 26, "signal": 9},
    "bb": {"period": 20, "std_dev": 2},
    "atr": {"period": 14},
}

# Minimum bars needed for each indicator to produce valid output
INDICATOR_WARMUP = {
    "rsi_14": 14,
    "rsi_7": 7,
    "macd": 26 + 9,  # slow period + signal period
    "bb": 20,
    "atr": 14,
}

# Maximum warmup needed across all indicators
MAX_WARMUP_BARS = max(INDICATOR_WARMUP.values())
