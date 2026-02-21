"""Real-time ADX/ATR-based intraday regime classification."""

import logging

from fxer.regime.types import RegimeState

logger = logging.getLogger(__name__)


class IntradayRegimeFilter:
    """Classifies intraday regime using ADX trend strength and ATR expansion.

    ADX < range_threshold → ranging (mean-reversion)
    ADX > trend_threshold → trending (momentum)
    ADX in dead zone → maintain previous state

    ATR expanding rapidly → volatility breakout (HIGH_VOL_TREND override)
    """

    def __init__(
        self,
        adx_range_threshold: float = 20.0,
        adx_trend_threshold: float = 25.0,
        atr_expansion_threshold: float = 1.5,
    ) -> None:
        self._adx_range_threshold = adx_range_threshold
        self._adx_trend_threshold = adx_trend_threshold
        self._atr_expansion_threshold = atr_expansion_threshold
        self._previous_state: RegimeState = RegimeState.RANGING

    def classify(
        self,
        adx: float | None,
        atr: float | None,
        prev_atr: float | None,
    ) -> tuple[RegimeState, float]:
        """Classify intraday regime based on ADX and ATR.

        Args:
            adx: Current ADX value, or None if not ready.
            atr: Current ATR value, or None if not ready.
            prev_atr: Previous ATR value for expansion detection.

        Returns:
            Tuple of (RegimeState, confidence 0.0-1.0).
        """
        # Check for ATR expansion (volatility breakout)
        atr_expanding = False
        if atr is not None and prev_atr is not None and prev_atr > 0:
            atr_ratio = atr / prev_atr
            if atr_ratio >= self._atr_expansion_threshold:
                atr_expanding = True

        if adx is None:
            # Not enough data for ADX; default to previous state
            return self._previous_state, 0.5

        if atr_expanding:
            state = RegimeState.HIGH_VOL_TREND
            confidence = min(adx / 50.0, 1.0)
            self._previous_state = state
            return state, confidence

        if adx < self._adx_range_threshold:
            state = RegimeState.RANGING
            # Lower ADX = higher confidence in ranging
            confidence = 1.0 - (adx / self._adx_range_threshold)
            confidence = max(0.5, min(confidence, 1.0))
        elif adx > self._adx_trend_threshold:
            state = RegimeState.LOW_VOL_TREND
            # Higher ADX = higher confidence in trending
            confidence = min(adx / 50.0, 1.0)
            confidence = max(0.5, confidence)
        else:
            # Dead zone (20-25): maintain previous state with reduced confidence
            state = self._previous_state
            confidence = 0.5

        self._previous_state = state
        return state, confidence

    def reset(self) -> None:
        """Reset to initial state."""
        self._previous_state = RegimeState.RANGING