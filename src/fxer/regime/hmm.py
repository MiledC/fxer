"""Hidden Markov Model regime classifier using forward algorithm (no lookahead)."""

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fxer.core.exceptions import RegimeClassificationError
from fxer.regime.types import RegimeState

logger = logging.getLogger(__name__)

# Map HMM states to regime states based on volatility characteristics
_REGIME_ORDER = [RegimeState.LOW_VOL_TREND, RegimeState.HIGH_VOL_TREND, RegimeState.RANGING]


class HMMRegimeClassifier:
    """2-3 state Gaussian HMM for daily regime detection.

    Uses the forward algorithm via score_samples() for causal inference
    (no lookahead bias). The Viterbi algorithm (predict()) is NOT used
    because it performs global optimization using future observations.
    """

    def __init__(
        self,
        n_states: int = 3,
        covariance_type: str = "diag",
        n_iter: int = 1000,
        random_state: int = 42,
    ) -> None:
        self._n_states = n_states
        self._covariance_type = covariance_type
        self._n_iter = n_iter
        self._random_state = random_state
        self._model: Any | None = None
        self._state_map: dict[int, RegimeState] = {}

    @property
    def is_fitted(self) -> bool:
        """Whether the HMM has been trained."""
        return self._model is not None

    def fit(self, daily_df: pd.DataFrame) -> dict[str, Any]:
        """Train the HMM on daily OHLC data.

        Args:
            daily_df: DataFrame with columns 'close', 'high', 'low'.
                Must have at least 100 rows.

        Returns:
            Dictionary of training metrics.
        """
        from hmmlearn.hmm import GaussianHMM

        features = self._extract_features(daily_df)

        if len(features) < 50:
            raise RegimeClassificationError(
                f"Insufficient data for HMM training: {len(features)} rows (need >= 50)",
                regime_type="hmm",
            )

        model = GaussianHMM(
            n_components=self._n_states,
            covariance_type=self._covariance_type,
            n_iter=self._n_iter,
            random_state=self._random_state,
        )
        model.fit(features)
        self._model = model

        # Map HMM states to regime states by volatility of each state
        self._build_state_map(features)

        score = float(model.score(features))
        logger.info(
            "HMM fitted: n_states=%d, covariance=%s, log_likelihood=%.2f, n_samples=%d",
            self._n_states,
            self._covariance_type,
            score,
            len(features),
        )

        return {
            "n_states": self._n_states,
            "covariance_type": self._covariance_type,
            "log_likelihood": score,
            "n_samples": len(features),
        }

    def predict_regime(self, daily_df: pd.DataFrame) -> tuple[RegimeState, float]:
        """Predict current regime using forward algorithm (no lookahead).

        Uses score_samples() which runs the forward algorithm, providing
        posterior state probabilities using only data up to time T.

        Args:
            daily_df: DataFrame with columns 'close', 'high', 'low'.

        Returns:
            Tuple of (RegimeState, confidence probability).
        """
        if self._model is None:
            raise RegimeClassificationError(
                "HMM not fitted. Call fit() or load() first.",
                regime_type="hmm",
            )

        features = self._extract_features(daily_df)
        if len(features) == 0:
            raise RegimeClassificationError(
                "No valid features extracted from daily data.",
                regime_type="hmm",
            )

        # Forward algorithm: score_samples returns posterior probabilities
        # using only observations up to each time step (no lookahead)
        _, state_probs = self._model.score_samples(features)
        current_probs = state_probs[-1]

        # Most likely state at current time
        hmm_state = int(np.argmax(current_probs))
        confidence = float(current_probs[hmm_state])
        regime = self._state_map.get(hmm_state, RegimeState.RANGING)

        return regime, confidence

    def save(self, path: Path) -> None:
        """Persist HMM model to disk."""
        if self._model is None:
            raise RegimeClassificationError(
                "Cannot save: HMM not fitted.",
                regime_type="hmm",
            )
        path.mkdir(parents=True, exist_ok=True)
        model_file = path / "hmm_model.pkl"
        state_map_file = path / "hmm_state_map.pkl"
        with open(model_file, "wb") as f:
            pickle.dump(self._model, f)
        with open(state_map_file, "wb") as f:
            pickle.dump(self._state_map, f)
        logger.info("HMM model saved to %s", path)

    def load(self, path: Path) -> None:
        """Load HMM model from disk."""
        model_file = path / "hmm_model.pkl"
        state_map_file = path / "hmm_state_map.pkl"
        if not model_file.exists():
            raise RegimeClassificationError(
                f"HMM model file not found: {model_file}",
                regime_type="hmm",
            )
        with open(model_file, "rb") as f:
            self._model = pickle.load(f)  # noqa: S301
        if state_map_file.exists():
            with open(state_map_file, "rb") as f:
                self._state_map = pickle.load(f)  # noqa: S301
        else:
            self._state_map = {i: _REGIME_ORDER[i % len(_REGIME_ORDER)] for i in range(self._n_states)}
        logger.info("HMM model loaded from %s", path)

    def _extract_features(self, daily_df: pd.DataFrame) -> np.ndarray:
        """Extract HMM input features from daily OHLC data.

        Features: daily returns, volatility (rolling std of returns), daily range.
        """
        closes = daily_df["close"].astype(float)
        highs = daily_df["high"].astype(float)
        lows = daily_df["low"].astype(float)

        returns = closes.pct_change()
        volatility = returns.rolling(window=20, min_periods=5).std()
        daily_range = (highs - lows) / closes

        features_df = pd.DataFrame({
            "returns": returns,
            "volatility": volatility,
            "range": daily_range,
        }).dropna()

        return features_df.values

    def _build_state_map(self, features: np.ndarray) -> None:
        """Map HMM states to RegimeState by sorting on volatility characteristic.

        The state with lowest mean volatility → LOW_VOL_TREND,
        highest mean volatility → HIGH_VOL_TREND,
        middle → RANGING.
        """
        assert self._model is not None
        means = self._model.means_

        # Volatility is the second feature (index 1)
        vol_means = means[:, 1]
        sorted_indices = np.argsort(vol_means)

        if self._n_states == 3:
            self._state_map = {
                int(sorted_indices[0]): RegimeState.LOW_VOL_TREND,
                int(sorted_indices[1]): RegimeState.RANGING,
                int(sorted_indices[2]): RegimeState.HIGH_VOL_TREND,
            }
        elif self._n_states == 2:
            self._state_map = {
                int(sorted_indices[0]): RegimeState.LOW_VOL_TREND,
                int(sorted_indices[1]): RegimeState.HIGH_VOL_TREND,
            }
        else:
            self._state_map = {
                int(sorted_indices[i]): _REGIME_ORDER[i % len(_REGIME_ORDER)]
                for i in range(self._n_states)
            }