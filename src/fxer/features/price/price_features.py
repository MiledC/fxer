"""Price-derived features including returns, volatility, and momentum."""

from __future__ import annotations

import math
from collections import deque


class PriceFeatures:
    """Computes price-derived features: returns, rolling volatility, and momentum.

    Features computed:
    - return_1bar: 1-bar log return
    - return_5bar: 5-bar log return
    - return_12bar: 12-bar log return
    - rolling_volatility_20: std dev of 1-bar log returns over 20 bars
    - momentum_48: price change ratio over 48 bars
    """

    def __init__(self) -> None:
        """Initialize with fixed window sizes."""
        self._prices: deque[float] = deque(maxlen=49)  # Need 49 for 48-bar momentum
        self._returns_1bar: deque[float] = deque(maxlen=20)  # For rolling volatility
        self._count: int = 0

    @property
    def ready(self) -> bool:
        """True when all features can be computed (after 49 bars)."""
        return self._count >= 49

    @property
    def warmup_period(self) -> int:
        """Minimum bars needed before all features are valid."""
        return 49

    def update(self, close: float) -> tuple[float | None, float | None, float | None, float | None, float | None]:
        """Update with a new close price and compute features.

        Args:
            close: The closing price of the current bar.

        Returns:
            Tuple of (return_1bar, return_5bar, return_12bar, rolling_volatility_20, momentum_48).
            Returns None for features that cannot yet be computed.
        """
        self._prices.append(close)
        self._count += 1

        # Need at least 2 prices for 1-bar return
        if self._count < 2:
            return None, None, None, None, None

        # Compute 1-bar return
        return_1bar: float | None = None
        prev_close = self._prices[-2]
        if close > 0 and prev_close > 0:
            return_1bar = math.log(close / prev_close)
            self._returns_1bar.append(return_1bar)

        # Compute 5-bar return
        return_5bar: float | None = None
        if self._count >= 6:
            close_5bar_ago = self._prices[-6]  # -1 is current, -6 is 5 bars ago
            if close > 0 and close_5bar_ago > 0:
                return_5bar = math.log(close / close_5bar_ago)

        # Compute 12-bar return
        return_12bar: float | None = None
        if self._count >= 13:
            close_12bar_ago = self._prices[-13]  # -1 is current, -13 is 12 bars ago
            if close > 0 and close_12bar_ago > 0:
                return_12bar = math.log(close / close_12bar_ago)

        # Compute rolling volatility (20-bar std dev of 1-bar returns)
        rolling_volatility_20: float | None = None
        if len(self._returns_1bar) >= 20:
            # Compute population standard deviation manually
            mean = sum(self._returns_1bar) / len(self._returns_1bar)
            variance = sum((r - mean) ** 2 for r in self._returns_1bar) / len(self._returns_1bar)
            rolling_volatility_20 = math.sqrt(variance)

        # Compute 48-bar momentum
        momentum_48: float | None = None
        if len(self._prices) == 49:  # Deque is full
            close_48bar_ago = self._prices[0]  # Oldest entry is 48 bars ago
            if close_48bar_ago > 0:  # Guard against division by zero
                momentum_48 = (close - close_48bar_ago) / close_48bar_ago

        return return_1bar, return_5bar, return_12bar, rolling_volatility_20, momentum_48

    def compute_batch(
        self, closes: list[float]
    ) -> list[tuple[float | None, float | None, float | None, float | None, float | None]]:
        """Compute features for a batch of close prices in order.

        Args:
            closes: List of close prices in chronological order.

        Returns:
            List of feature tuples, one per close price.
        """
        return [self.update(close) for close in closes]

    def reset(self) -> None:
        """Reset all internal state."""
        self._prices.clear()
        self._returns_1bar.clear()
        self._count = 0