"""Pre-computes multi-timeframe observation vectors for RL environments."""

from __future__ import annotations

import bisect
import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np

from fxer.config.constants import MAX_WARMUP_BARS
from fxer.core.events import FeatureVector, NormalizedBar
from fxer.core.exceptions import ObservationError
from fxer.core.types import Timeframe
from fxer.features.engine import FeatureEngine
from fxer.regime.types import RegimeDecision, RegimeState
from fxer.rl.observations.config import NormalizationMethod, ObservationConfig
from fxer.rl.observations.normalization import ObservationNormalizer
from fxer.signals.base import feature_vector_to_array

if TYPE_CHECKING:
    from fxer.data.storage.questdb_client import QuestDBClient
    from fxer.regime.classifier import RegimeClassifier

logger = logging.getLogger(__name__)

# Pre-computed list of regime states for one-hot encoding
_REGIME_STATES = list(RegimeState)


def _encode_regime(decision: RegimeDecision | None, n_states: int = 3) -> np.ndarray:
    """Encode regime decision as one-hot vector + confidence.

    Args:
        decision: Regime decision to encode, or None for zeros.
        n_states: Number of regime states (default 3).

    Returns:
        Array of shape (n_states + 1,) with one-hot encoding and confidence.
    """
    if decision is None:
        return np.zeros(n_states + 1, dtype=np.float64)

    one_hot = np.zeros(n_states, dtype=np.float64)
    idx = _REGIME_STATES.index(decision.state)
    one_hot[idx] = 1.0
    return np.concatenate([one_hot, [decision.confidence]])


class ObservationBuilder:
    """Pre-computes multi-timeframe observation vectors for RL environments.

    Builds flat numpy arrays combining features from multiple timeframes
    and optional regime state, stored in a dict for O(1) lookup.
    """

    def __init__(
        self,
        config: ObservationConfig,
        db_client: QuestDBClient | None = None,
        regime_classifier: RegimeClassifier | None = None,
    ) -> None:
        """Initialize the observation builder.

        Args:
            config: Observation configuration specifying windows and features.
            db_client: QuestDB client for querying bars (optional if bars provided to build).
            regime_classifier: Regime classifier for adding regime features (optional).
        """
        self._config = config
        self._db_client = db_client
        self._regime_classifier = regime_classifier

        # Storage for pre-computed data
        self._observations: dict[datetime, np.ndarray] = {}
        self._tf_features: dict[Timeframe, dict[datetime, FeatureVector]] = {}
        self._tf_bars: dict[Timeframe, dict[datetime, NormalizedBar]] = {}
        self._regime_decisions: dict[datetime, RegimeDecision] = {}

        # Track build state
        self._is_built = False
        self._timestamps: list[datetime] = []

        # Normalization
        self._normalizer: ObservationNormalizer | None = None
        if config.normalization.method != NormalizationMethod.NONE:
            norm = config.normalization
            self._normalizer = ObservationNormalizer(
                method=norm.method.value,
                obs_config=config,
                alpha=norm.alpha,
                min_samples=norm.min_samples,
                winsorize_threshold=norm.winsorize_threshold,
                gap_threshold_minutes=norm.gap_threshold_minutes,
                session_aware=norm.session_aware,
                clip_range=norm.clip_range,
            )

    @property
    def observation_size(self) -> int:
        """Total floats per observation vector."""
        return self._config.observation_size

    @property
    def timestamps(self) -> list[datetime]:
        """Sorted list of all available primary-TF timestamps."""
        return self._timestamps

    @property
    def is_built(self) -> bool:
        """Whether build() has been called."""
        return self._is_built

    def build(
        self,
        start: datetime,
        end: datetime,
        bars: dict[Timeframe, list[NormalizedBar]] | None = None,
    ) -> None:
        """Pre-compute all observations for the given date range.

        Args:
            start: Start of episode range (inclusive).
            end: End of episode range (inclusive).
            bars: Optional pre-loaded bars per timeframe. If None, queries QuestDB.
                  Keys are Timeframe enums, values are lists of NormalizedBar sorted by timestamp.
                  Must include warmup bars BEFORE start.

        Raises:
            ObservationError: If neither bars nor db_client is provided.
        """
        if bars is None and self._db_client is None:
            raise ObservationError(
                "Either bars dict or db_client must be provided",
                timestamp=start,
            )

        logger.info(
            f"Building observations for {self._config.symbol} from {start} to {end}"
        )

        # Clear any existing data
        self._observations.clear()
        self._tf_features.clear()
        self._tf_bars.clear()
        self._regime_decisions.clear()
        self._timestamps.clear()

        # Step 1: Compute features for each timeframe
        for window in self._config.windows:
            tf = window.timeframe
            logger.debug(f"Processing timeframe {tf.value}")

            # Get bars for this timeframe
            if bars is not None:
                tf_bars = bars.get(tf, [])
            else:
                # Query from QuestDB with warmup
                warmup_start = start - timedelta(
                    minutes=(window.lookback + MAX_WARMUP_BARS) * tf.minutes
                )
                tf_bars = self._db_client.query_bars(
                    symbol=self._config.symbol,
                    timeframe=tf,
                    start=warmup_start,
                    end=end,
                )

            # Create fresh FeatureEngine and compute features
            engine = FeatureEngine()
            features = engine.compute_batch(tf_bars)

            # Store features and bars in dicts for O(1) lookup
            self._tf_features[tf] = {fv.timestamp: fv for fv in features}
            self._tf_bars[tf] = {bar.timestamp: bar for bar in tf_bars}

            logger.debug(
                f"Computed {len(features)} features for {tf.value}, "
                f"warmup complete: {engine.warmup_complete}"
            )

        # Step 2: Compute regime classifications if enabled
        if self._config.include_regime and self._regime_classifier is not None:
            primary_tf = self._config.primary_timeframe
            primary_features = self._tf_features.get(primary_tf, {})
            primary_bars = self._tf_bars.get(primary_tf, {})

            for timestamp, feature in primary_features.items():
                if feature.warmup_complete:
                    bar = primary_bars.get(timestamp)
                    if bar is not None:
                        decision = self._regime_classifier.classify(bar, feature)
                        self._regime_decisions[timestamp] = decision

            logger.debug(f"Computed {len(self._regime_decisions)} regime decisions")

        # Step 3: Build observation vectors
        primary_tf = self._config.primary_timeframe
        primary_features = self._tf_features.get(primary_tf, {})

        # Get valid timestamps (warmup complete and within range)
        valid_timestamps = [
            ts for ts, fv in primary_features.items()
            if fv.warmup_complete and start <= ts <= end
        ]
        valid_timestamps.sort()

        logger.debug(f"Building observations for {len(valid_timestamps)} timestamps")

        prev_timestamp = None
        for timestamp in valid_timestamps:
            obs_parts = []

            # Process each window. Uses for/else: if any window breaks
            # (insufficient features), the else clause is skipped and
            # this timestamp is dropped from the observation set.
            for window in self._config.windows:
                tf = window.timeframe
                tf_features = self._tf_features.get(tf, {})
                tf_bars = self._tf_bars.get(tf, {})

                # Find valid features for this window at this timestamp
                valid_features = self._get_valid_features_for_window(
                    timestamp, tf, tf_features, window.lookback
                )

                if len(valid_features) < window.lookback:
                    # Not enough features available, skip this timestamp
                    logger.debug(
                        f"Skipping timestamp {timestamp}: insufficient {tf.value} features "
                        f"(need {window.lookback}, have {len(valid_features)})"
                    )
                    break

                # Take the last lookback features
                recent_features = valid_features[-window.lookback:]

                # Convert each feature to array
                for fv in recent_features:
                    parts = []

                    # Add OHLCV if requested
                    if window.include_ohlcv:
                        bar = tf_bars.get(fv.timestamp)
                        if bar is not None:
                            ohlcv = np.array([
                                float(bar.open),
                                float(bar.high),
                                float(bar.low),
                                float(bar.close),
                                float(bar.volume),
                            ], dtype=np.float64)
                            parts.append(ohlcv)

                    # Add features if requested
                    if window.include_features:
                        feature_array = feature_vector_to_array(fv)
                        parts.append(feature_array)

                    if parts:
                        window_features = np.concatenate(parts)
                        obs_parts.append(window_features)
            else:
                # All windows processed successfully

                # Add regime encoding if enabled
                if self._config.include_regime:
                    regime_decision = self._regime_decisions.get(timestamp)
                    regime_encoding = _encode_regime(regime_decision)
                    obs_parts.append(regime_encoding)

                # Concatenate all parts into single observation
                observation = np.concatenate(obs_parts)

                # Apply normalization if configured
                if self._normalizer is not None:
                    prev_features = primary_features.get(prev_timestamp) if prev_timestamp else None
                    observation = self._normalizer.normalize_observation(
                        observation, timestamp, prev_features
                    )

                # Verify size matches config
                if len(observation) != self._config.observation_size:
                    raise ObservationError(
                        f"Observation size mismatch at {timestamp}: "
                        f"expected {self._config.observation_size}, got {len(observation)}",
                        timestamp=timestamp,
                    )

                self._observations[timestamp] = observation
                self._timestamps.append(timestamp)
                prev_timestamp = timestamp

        self._timestamps.sort()
        self._is_built = True

        logger.info(
            f"Built {len(self._observations)} observations for {self._config.symbol}"
        )

    def _get_valid_features_for_window(
        self,
        query_timestamp: datetime,
        timeframe: Timeframe,
        tf_features: dict[datetime, FeatureVector],
        lookback: int,
    ) -> list[FeatureVector]:
        """Get valid features for a window at a specific timestamp.

        For the primary timeframe: features at timestamps <= query_timestamp.
        For higher timeframes: features where bar completion time <= query_timestamp.

        Args:
            query_timestamp: The timestamp to query for.
            timeframe: The timeframe of the features.
            tf_features: Dict of timestamp -> FeatureVector for this timeframe.
            lookback: Maximum number of features to return.

        Returns:
            List of FeatureVector objects, sorted by timestamp.
        """
        # Get sorted timestamps for this timeframe
        tf_timestamps = sorted(tf_features.keys())

        if timeframe == self._config.primary_timeframe:
            # For primary TF: use features at timestamps <= query_timestamp
            idx = bisect.bisect_right(tf_timestamps, query_timestamp)
            valid_timestamps = tf_timestamps[:idx]
        else:
            # For higher TFs: use features where completion time <= query_timestamp
            valid_timestamps = []
            for ts in tf_timestamps:
                completion_time = ts + timedelta(minutes=timeframe.minutes)
                if completion_time <= query_timestamp:
                    valid_timestamps.append(ts)
                else:
                    # Since timestamps are sorted, we can break early
                    break

        # Get the features for valid timestamps
        valid_features = [tf_features[ts] for ts in valid_timestamps if ts in tf_features]

        # Return only features where warmup is complete
        return [fv for fv in valid_features if fv.warmup_complete]

    def get_observation(self, timestamp: datetime) -> np.ndarray | None:
        """Get pre-computed observation at exact timestamp. O(1) lookup.

        Args:
            timestamp: Exact timestamp to query.

        Returns:
            Observation array or None if not available.
        """
        return self._observations.get(timestamp)

    def get_bar(self, timestamp: datetime) -> NormalizedBar | None:
        """Get primary-timeframe bar at timestamp.

        Args:
            timestamp: Exact timestamp to query.

        Returns:
            NormalizedBar or None if not available.
        """
        primary_tf = self._config.primary_timeframe
        bars = self._tf_bars.get(primary_tf, {})
        return bars.get(timestamp)

    def get_regime(self, timestamp: datetime) -> RegimeDecision | None:
        """Get regime classification at timestamp.

        Args:
            timestamp: Exact timestamp to query.

        Returns:
            RegimeDecision or None if not available.
        """
        return self._regime_decisions.get(timestamp)

    def get_features(self, timestamp: datetime) -> FeatureVector | None:
        """Get primary-timeframe features at timestamp.

        Args:
            timestamp: Exact timestamp to query.

        Returns:
            FeatureVector or None if not available.
        """
        primary_tf = self._config.primary_timeframe
        features = self._tf_features.get(primary_tf, {})
        return features.get(timestamp)

    def reset_normalizer(self) -> None:
        """Reset normalization statistics.

        Call between RL episodes to clear accumulated statistics,
        ensuring each episode starts with fresh normalization state.
        """
        if self._normalizer is not None:
            self._normalizer.reset()