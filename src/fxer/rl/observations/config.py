"""Configuration for RL observations."""

from dataclasses import dataclass

from fxer.core.types import Timeframe
from fxer.signals.base import FEATURE_COLUMNS


@dataclass(frozen=True, slots=True)
class TimeframeWindowConfig:
    """Configuration for a single timeframe window in observations.

    Attributes:
        timeframe: The timeframe for this window.
        lookback: Number of bars to include in the window.
        include_ohlcv: Whether to include raw OHLCV data.
        include_features: Whether to include computed features.
    """

    timeframe: Timeframe
    lookback: int
    include_ohlcv: bool = False
    include_features: bool = True

    @property
    def features_per_bar(self) -> int:
        """Number of float values per bar in this window."""
        size = 0
        if self.include_ohlcv:
            size += 5  # open, high, low, close, volume
        if self.include_features:
            size += len(FEATURE_COLUMNS)  # 24
        return size

    @property
    def window_size(self) -> int:
        """Total float values in this window."""
        return self.lookback * self.features_per_bar


@dataclass(frozen=True, slots=True)
class ObservationConfig:
    """Configuration for building RL observations.

    Attributes:
        symbol: Trading symbol (default XAUUSD).
        primary_timeframe: Primary timeframe for the environment.
        windows: Tuple of timeframe window configurations.
        include_regime: Whether to include regime state in observations.
        regime_features: Number of regime-related features.
    """

    symbol: str = "XAUUSD"
    primary_timeframe: Timeframe = Timeframe.M5
    windows: tuple[TimeframeWindowConfig, ...] = (
        TimeframeWindowConfig(Timeframe.M5, 20, False, True),
        TimeframeWindowConfig(Timeframe.H1, 10, False, True),
        TimeframeWindowConfig(Timeframe.H4, 5, False, True),
    )
    include_regime: bool = True
    regime_features: int = 4  # 3 one-hot RegimeState + 1 confidence

    @property
    def observation_size(self) -> int:
        """Total floats in one observation vector."""
        size = sum(w.window_size for w in self.windows)
        if self.include_regime:
            size += self.regime_features
        return size