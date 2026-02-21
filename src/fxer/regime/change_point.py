"""PELT change point detection with sliding window (no lookahead)."""

import logging

import numpy as np

from fxer.core.exceptions import RegimeClassificationError

logger = logging.getLogger(__name__)


class ChangePointDetector:
    """Detects structural breaks in returns/volatility using PELT algorithm.

    Uses a sliding window to avoid lookahead bias: only past observations
    within the window are used for detection. This accepts delayed detection
    as a tradeoff for causal correctness.
    """

    def __init__(self, window_size: int = 100, min_size: int = 10) -> None:
        self._window_size = window_size
        self._min_size = min_size

    def detect_change_points(self, returns: np.ndarray) -> list[int]:
        """Detect change points in a sliding window of returns.

        Args:
            returns: 1-D array of returns (only past data).

        Returns:
            List of change point indices within the window.
        """
        import ruptures

        if len(returns) < self._min_size * 2:
            return []

        # Use only the most recent window
        window = returns[-self._window_size :] if len(returns) > self._window_size else returns

        try:
            algo = ruptures.Pelt(model="rbf", min_size=self._min_size)
            change_points = algo.fit_predict(window, pen=3.0)
            # ruptures includes the last index; remove it
            change_points = [cp for cp in change_points if cp < len(window)]
            return change_points
        except Exception as exc:
            logger.warning("PELT detection failed: %s", exc)
            return []

    def has_recent_change(self, returns: np.ndarray, lookback: int = 20) -> bool:
        """Check if a structural break occurred recently.

        Args:
            returns: 1-D array of returns (only past data).
            lookback: Number of bars to consider as "recent".

        Returns:
            True if a change point was detected within the lookback window.
        """
        change_points = self.detect_change_points(returns)
        if not change_points:
            return False

        window_len = min(len(returns), self._window_size)
        # Check if any change point is within the last `lookback` bars of the window
        return any(cp > (window_len - lookback) for cp in change_points)