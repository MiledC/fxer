"""Core normalization module for RL observations."""

from __future__ import annotations

import logging
import math
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from fxer.signals.base import FEATURE_COLUMNS

if TYPE_CHECKING:
    from fxer.core.events import FeatureVector
    from fxer.rl.observations.config import ObservationConfig

logger = logging.getLogger(__name__)


class TradingSession(Enum):
    """Trading session classification."""

    ASIAN = "asian"      # 22:00-07:00 UTC
    LONDON = "london"    # 07:00-12:00 UTC
    OVERLAP = "overlap"  # 12:00-16:00 UTC
    NY = "ny"            # 16:00-22:00 UTC


def classify_session(timestamp: datetime) -> TradingSession:
    """Classify timestamp into trading session.

    Args:
        timestamp: UTC datetime to classify.

    Returns:
        TradingSession enum value.
    """
    hour = timestamp.hour

    # Asian session spans midnight (22:00-07:00)
    if hour >= 22 or hour < 7:
        return TradingSession.ASIAN
    elif 7 <= hour < 12:
        return TradingSession.LONDON
    elif 12 <= hour < 16:
        return TradingSession.OVERLAP
    else:  # 16 <= hour < 22
        return TradingSession.NY


class FeatureClass(Enum):
    """Feature normalization class."""

    PASSTHROUGH = "passthrough"
    RESCALE = "rescale"
    ZSCORE = "zscore"
    VOL_ADJUST = "vol_adjust"
    LOG_ZSCORE = "log_zscore"


# Map each feature column to its normalization class
FEATURE_CLASS_MAP: dict[str, FeatureClass] = {
    # PASSTHROUGH: boolean flags and already normalized
    "is_london_session": FeatureClass.PASSTHROUGH,
    "is_ny_session": FeatureClass.PASSTHROUGH,
    "is_overlap_session": FeatureClass.PASSTHROUGH,
    "is_asian_session": FeatureClass.PASSTHROUGH,
    "is_month_turn": FeatureClass.PASSTHROUGH,
    "bb_percent_b": FeatureClass.PASSTHROUGH,  # Already 0-1 range
    # RESCALE: bounded features with known ranges
    "rsi_14": FeatureClass.RESCALE,
    "rsi_7": FeatureClass.RESCALE,
    "dxy_rsi_14": FeatureClass.RESCALE,
    "hour_of_day": FeatureClass.RESCALE,
    "day_of_week": FeatureClass.RESCALE,
    # ZSCORE: unbounded features needing standardization
    "macd_line": FeatureClass.ZSCORE,
    "macd_signal": FeatureClass.ZSCORE,
    "macd_histogram": FeatureClass.ZSCORE,
    "atr_14": FeatureClass.ZSCORE,
    "vix_change": FeatureClass.ZSCORE,
    "dxy_return_1h": FeatureClass.ZSCORE,
    # VOL_ADJUST: returns scaled by volatility
    "return_1bar": FeatureClass.VOL_ADJUST,
    "return_5bar": FeatureClass.VOL_ADJUST,
    "return_12bar": FeatureClass.VOL_ADJUST,
    "momentum_48": FeatureClass.VOL_ADJUST,
    # LOG_ZSCORE: positive features needing log transform
    "bb_width": FeatureClass.LOG_ZSCORE,
    "rolling_volatility_20": FeatureClass.LOG_ZSCORE,
    "vix_level": FeatureClass.LOG_ZSCORE,
}

# Min/max ranges for RESCALE features
RESCALE_RANGES: dict[str, tuple[float, float]] = {
    "rsi_14": (0.0, 100.0),
    "rsi_7": (0.0, 100.0),
    "dxy_rsi_14": (0.0, 100.0),
    "hour_of_day": (0.0, 23.0),
    "day_of_week": (0.0, 6.0),
}


class OnlineStats:
    """Online mean/variance tracker using Welford's algorithm with EMA decay.

    Tracks running statistics with optional session awareness and gap handling.
    """

    def __init__(
        self,
        alpha: float = 0.97,
        min_samples: int = 20,
        winsorize_threshold: float = 5.0,
        gap_threshold_minutes: int = 180,
        session_aware: bool = False,
    ) -> None:
        """Initialize online statistics tracker.

        Args:
            alpha: EMA decay factor for stable state (0.9-0.99).
            min_samples: Minimum samples before switching to EMA.
            winsorize_threshold: Clip values beyond this many std devs.
            gap_threshold_minutes: Minutes to detect data gap.
            session_aware: Whether to track stats per trading session.
        """
        self.alpha = alpha
        self.min_samples = min_samples
        self.winsorize_threshold = winsorize_threshold
        self.gap_threshold_minutes = gap_threshold_minutes
        self.session_aware = session_aware

        # State storage
        if session_aware:
            # Per-session state
            self._states: dict[TradingSession, dict] = {
                session: {
                    "_mean": 0.0,
                    "_var": 0.0,
                    "_count": 0,
                    "_last_timestamp": None,
                }
                for session in TradingSession
            }
        else:
            # Global state
            self._mean: float = 0.0
            self._var: float = 0.0
            self._count: int = 0
            self._last_timestamp: datetime | None = None

    def _get_state(self, session: TradingSession | None) -> dict:
        """Get state dict for session or global."""
        if self.session_aware and session is not None:
            return self._states[session]
        elif self.session_aware:
            # Session-aware but no session provided, use global-like state
            if not hasattr(self, "_global_state"):
                self._global_state = {
                    "_mean": 0.0,
                    "_var": 0.0,
                    "_count": 0,
                    "_last_timestamp": None,
                }
            return self._global_state
        else:
            # Return self as dict-like for global state
            return {
                "_mean": self._mean,
                "_var": self._var,
                "_count": self._count,
                "_last_timestamp": self._last_timestamp,
            }

    def _set_state(self, session: TradingSession | None, state: dict) -> None:
        """Update state from dict."""
        if self.session_aware and session is not None:
            self._states[session] = state
        elif self.session_aware:
            self._global_state = state
        else:
            self._mean = state["_mean"]
            self._var = state["_var"]
            self._count = state["_count"]
            self._last_timestamp = state["_last_timestamp"]

    def update(
        self,
        value: float,
        timestamp: datetime,
        session: TradingSession | None = None,
    ) -> None:
        """Update statistics with new value.

        Args:
            value: New observation value.
            timestamp: Timestamp of the observation.
            session: Trading session (if session_aware).
        """
        state = self._get_state(session)
        mean = state["_mean"]
        var = state["_var"]
        count = state["_count"]
        last_timestamp = state["_last_timestamp"]

        if count < self.min_samples:
            # Expanding window (Welford's algorithm)
            count += 1
            delta = value - mean
            mean += delta / count
            delta2 = value - mean
            var = var + (delta * delta2 - var) / count
        else:
            # EMA mode with gap detection and winsorization
            alpha = self.alpha

            # Check for data gap
            if last_timestamp is not None:
                gap_minutes = (timestamp - last_timestamp).total_seconds() / 60
                if gap_minutes > self.gap_threshold_minutes:
                    # Dampen alpha for this update
                    alpha = alpha * 0.5
                    logger.debug(
                        f"Data gap detected ({gap_minutes:.0f} min), using dampened alpha={alpha:.3f}"
                    )

            # Winsorize extreme values
            std = math.sqrt(var) if var > 0 else 0.0
            if std > 1e-10:
                lower_bound = mean - self.winsorize_threshold * std
                upper_bound = mean + self.winsorize_threshold * std
                if value < lower_bound:
                    value = lower_bound
                elif value > upper_bound:
                    value = upper_bound

            # EMA update
            mean = alpha * mean + (1 - alpha) * value
            diff = value - mean
            var = alpha * var + (1 - alpha) * diff * diff
            count += 1

        # Update state
        new_state = {
            "_mean": mean,
            "_var": var,
            "_count": count,
            "_last_timestamp": timestamp,
        }
        self._set_state(session, new_state)

    def normalize(self, value: float, session: TradingSession | None = None) -> float:
        """Normalize value using current statistics.

        Args:
            value: Value to normalize.
            session: Trading session (if session_aware).

        Returns:
            Z-score normalized value, or original if insufficient samples.
        """
        state = self._get_state(session)
        if state["_count"] < self.min_samples:
            # Not enough samples, passthrough
            return value

        mean = state["_mean"]
        var = state["_var"]
        std = math.sqrt(var) if var > 0 else 0.0

        if std < 1e-10:
            # Zero variance, return 0
            return 0.0

        return (value - mean) / std

    def get_stats(self, session: TradingSession | None = None) -> tuple[float, float]:
        """Get current mean and standard deviation.

        Args:
            session: Trading session (if session_aware).

        Returns:
            Tuple of (mean, std).
        """
        state = self._get_state(session)
        mean = state["_mean"]
        var = state["_var"]
        std = math.sqrt(var) if var > 0 else 0.0
        return mean, std

    def reset(self) -> None:
        """Reset all statistics."""
        if self.session_aware:
            for session in TradingSession:
                self._states[session] = {
                    "_mean": 0.0,
                    "_var": 0.0,
                    "_count": 0,
                    "_last_timestamp": None,
                }
            if hasattr(self, "_global_state"):
                del self._global_state
        else:
            self._mean = 0.0
            self._var = 0.0
            self._count = 0
            self._last_timestamp = None


class ObservationNormalizer:
    """Normalizes observation vectors using configurable per-feature methods.

    Supports two normalization modes:
    - 'online_zscore': Treat all non-passthrough features as ZSCORE
    - 'feature_specific': Use FEATURE_CLASS_MAP for per-feature treatment
    """

    def __init__(
        self,
        method: str,
        obs_config: ObservationConfig,
        alpha: float = 0.97,
        min_samples: int = 20,
        winsorize_threshold: float = 5.0,
        gap_threshold_minutes: int = 180,
        session_aware: bool = False,
        clip_range: tuple[float, float] = (-5.0, 5.0),
        vol_adjust_floor: float = 1e-8,
    ) -> None:
        """Initialize observation normalizer.

        Args:
            method: Normalization method ('online_zscore' or 'feature_specific').
            obs_config: Observation configuration specifying layout.
            alpha: EMA decay factor for online stats.
            min_samples: Minimum samples before normalization.
            winsorize_threshold: Clip values beyond this many std devs.
            gap_threshold_minutes: Minutes to detect data gap.
            session_aware: Whether to track stats per trading session.
            clip_range: Final clipping range for all normalized values.
            vol_adjust_floor: Minimum volatility for VOL_ADJUST features.
        """
        self.method = method
        self.obs_config = obs_config
        self.alpha = alpha
        self.min_samples = min_samples
        self.winsorize_threshold = winsorize_threshold
        self.gap_threshold_minutes = gap_threshold_minutes
        self.session_aware = session_aware
        self.clip_range = clip_range
        self.vol_adjust_floor = vol_adjust_floor

        # Build index map for observation vector
        self._index_map: list[tuple[str, FeatureClass]] = []
        self._build_index_map()

        # Create OnlineStats instances for features that need them
        self._stats: dict[int, OnlineStats] = {}
        self._create_stats_instances()

        logger.debug(
            f"ObservationNormalizer initialized with method={method}, "
            f"obs_size={len(self._index_map)}, stats_count={len(self._stats)}"
        )

    def _build_index_map(self) -> None:
        """Build mapping from observation indices to (feature_name, class)."""
        idx = 0

        # Process each window
        for window in self.obs_config.windows:
            for _ in range(window.lookback):  # For each bar in the window
                # OHLCV features if included
                if window.include_ohlcv:
                    self._index_map.append(("open", FeatureClass.ZSCORE))
                    self._index_map.append(("high", FeatureClass.ZSCORE))
                    self._index_map.append(("low", FeatureClass.ZSCORE))
                    self._index_map.append(("close", FeatureClass.ZSCORE))
                    self._index_map.append(("volume", FeatureClass.LOG_ZSCORE))
                    idx += 5

                # Feature columns if included
                if window.include_features:
                    for feature_name in FEATURE_COLUMNS:
                        if self.method == "online_zscore":
                            # In online_zscore mode, treat non-passthrough as ZSCORE
                            feature_class = FEATURE_CLASS_MAP.get(
                                feature_name, FeatureClass.ZSCORE
                            )
                            if feature_class != FeatureClass.PASSTHROUGH:
                                feature_class = FeatureClass.ZSCORE
                        else:
                            # Use feature-specific class
                            feature_class = FEATURE_CLASS_MAP.get(
                                feature_name, FeatureClass.ZSCORE
                            )
                        self._index_map.append((feature_name, feature_class))
                        idx += 1

        # Regime features if included
        if self.obs_config.include_regime:
            # Regime is always passthrough (one-hot + confidence)
            for _ in range(self.obs_config.regime_features):
                self._index_map.append(("regime", FeatureClass.PASSTHROUGH))
                idx += 1

    def _create_stats_instances(self) -> None:
        """Create OnlineStats for indices that need statistical tracking."""
        for idx, (feature_name, feature_class) in enumerate(self._index_map):
            # Create stats for features that need them
            if feature_class in [
                FeatureClass.ZSCORE,
                FeatureClass.LOG_ZSCORE,
                FeatureClass.VOL_ADJUST,
            ]:
                self._stats[idx] = OnlineStats(
                    alpha=self.alpha,
                    min_samples=self.min_samples,
                    winsorize_threshold=self.winsorize_threshold,
                    gap_threshold_minutes=self.gap_threshold_minutes,
                    session_aware=self.session_aware,
                )

    def normalize_observation(
        self,
        raw_obs: np.ndarray,
        timestamp: datetime,
        prev_features: FeatureVector | None = None,
    ) -> np.ndarray:
        """Normalize raw observation vector.

        Args:
            raw_obs: Raw observation array to normalize.
            timestamp: Timestamp of the observation.
            prev_features: Previous features for VOL_ADJUST (optional).

        Returns:
            Normalized observation array.
        """
        # Copy to avoid mutation
        normalized = raw_obs.copy()

        # Classify trading session
        session = classify_session(timestamp) if self.session_aware else None

        # Get volatility for VOL_ADJUST features
        vol = None
        if prev_features and prev_features.rolling_volatility_20 is not None:
            vol = prev_features.rolling_volatility_20

        # Process each index
        for idx in range(len(normalized)):
            if idx >= len(self._index_map):
                # Should not happen if config is consistent
                logger.warning(f"Index {idx} beyond index map, skipping")
                continue

            feature_name, feature_class = self._index_map[idx]
            value = normalized[idx]

            # Apply normalization based on class
            if feature_class == FeatureClass.PASSTHROUGH:
                # No change
                pass

            elif feature_class == FeatureClass.RESCALE:
                # Linear rescaling to [0, 1]
                if feature_name in RESCALE_RANGES:
                    min_val, max_val = RESCALE_RANGES[feature_name]
                    if max_val > min_val:
                        normalized[idx] = (value - min_val) / (max_val - min_val)

            elif feature_class == FeatureClass.ZSCORE:
                # Update stats and normalize
                if idx in self._stats:
                    self._stats[idx].update(value, timestamp, session)
                    normalized[idx] = self._stats[idx].normalize(value, session)

            elif feature_class == FeatureClass.VOL_ADJUST:
                # Scale by volatility if available
                if vol is not None and vol > self.vol_adjust_floor:
                    normalized[idx] = value / vol
                # Also track stats for monitoring
                if idx in self._stats:
                    self._stats[idx].update(value, timestamp, session)

            elif feature_class == FeatureClass.LOG_ZSCORE:
                # Log transform then z-score
                log_value = math.log(max(value, 1e-10))
                if idx in self._stats:
                    self._stats[idx].update(log_value, timestamp, session)
                    normalized[idx] = self._stats[idx].normalize(log_value, session)

        # Final clipping
        normalized = np.clip(normalized, self.clip_range[0], self.clip_range[1])

        return normalized

    def reset(self) -> None:
        """Reset all online statistics."""
        for stats in self._stats.values():
            stats.reset()
        logger.debug("ObservationNormalizer statistics reset")