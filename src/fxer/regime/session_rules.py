"""Session-aware regime rules for gold trading."""

import logging
from datetime import datetime

from fxer.config.constants import (
    ASIAN_SESSION,
    LATE_NY_SESSION,
    LONDON_OPEN_SESSION,
    OVERLAP_SESSION,
)

logger = logging.getLogger(__name__)


class SessionRegimeRules:
    """Session-based regime expectations for XAUUSD.

    Asian (22:00-07:00 UTC)         → consolidation expected, bias=0.7
    London open (07:00-08:00 UTC)   → breakout expected, bias=1.2
    London-NY overlap (12:00-16:00) → directional moves, bias=1.3
    Late NY (after 17:00 UTC)       → avoid new entries, bias=0.3
    """

    # Session biases: multiplier for trading confidence
    _SESSION_BIAS = {
        "Asian": 0.7,
        "London Open": 1.2,
        "Overlap": 1.3,
        "Late NY": 0.3,
    }

    def get_session_bias(self, timestamp: datetime) -> tuple[float, str]:
        """Get the session-based trading bias for a timestamp.

        Args:
            timestamp: UTC datetime.

        Returns:
            Tuple of (bias multiplier, session description).
        """
        hour = timestamp.hour

        # Check sessions in priority order (most specific first)
        if LONDON_OPEN_SESSION.is_active(hour):
            return self._SESSION_BIAS["London Open"], "London open — breakout expected"

        if OVERLAP_SESSION.is_active(hour):
            return self._SESSION_BIAS["Overlap"], "London-NY overlap — peak liquidity"

        if LATE_NY_SESSION.is_active(hour):
            return self._SESSION_BIAS["Late NY"], "Late NY — declining liquidity"

        if ASIAN_SESSION.is_active(hour):
            return self._SESSION_BIAS["Asian"], "Asian session — consolidation expected"

        # Default: normal trading hours
        return 1.0, "Normal trading hours"

    def should_avoid_trading(self, timestamp: datetime) -> bool:
        """Check if trading should be avoided based on session.

        Args:
            timestamp: UTC datetime.

        Returns:
            True if new entries should be avoided.
        """
        hour = timestamp.hour
        return LATE_NY_SESSION.is_active(hour)