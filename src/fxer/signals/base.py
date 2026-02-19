"""Abstract base class for signal models."""

from abc import ABC, abstractmethod
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np

from fxer.core.events import FeatureVector
from fxer.signals.types import ModelPrediction

# Phase 1a features: stored in QuestDB features table today.
STORED_FEATURE_COLUMNS: list[str] = [
    "rsi_14",
    "rsi_7",
    "macd_line",
    "macd_signal",
    "macd_histogram",
    "bb_upper",
    "bb_middle",
    "bb_lower",
    "bb_width",
    "atr_14",
    "is_london_session",
    "is_ny_session",
    "is_overlap_session",
    "is_asian_session",
    "hour_of_day",
    "day_of_week",
    "is_month_turn",
]

# Phase 1b+ cross-asset features (not yet in QuestDB).
CROSS_ASSET_COLUMNS: list[str] = [
    "dxy_return_1h",
    "dxy_rsi_14",
    "vix_level",
    "vix_change",
    "tips_yield_10y",
]

# Full canonical column order for all model inputs.
FEATURE_COLUMNS: list[str] = STORED_FEATURE_COLUMNS + CROSS_ASSET_COLUMNS


def feature_vector_to_array(fv: FeatureVector) -> np.ndarray:
    """Convert a FeatureVector to a numpy array in canonical column order.

    None values are replaced with 0.0 (models must handle this).

    Args:
        fv: A FeatureVector instance.

    Returns:
        1-D float64 array of length len(FEATURE_COLUMNS).
    """
    values = []
    for col in FEATURE_COLUMNS:
        val = getattr(fv, col)
        if isinstance(val, bool):
            values.append(float(val))
        elif val is None:
            values.append(0.0)
        else:
            values.append(float(val))
    return np.array(values, dtype=np.float64)


class BaseSignalModel(ABC):
    """Abstract interface for all signal models.

    Subclasses must implement fit, predict, predict_batch, save, and load.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name."""

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Whether the model has been trained."""

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> dict[str, Any]:
        """Train the model.

        Args:
            X: Training features.
            y: Training labels (0 = short, 1 = long).
            validation_data: Optional (X_val, y_val) for early stopping.

        Returns:
            Dictionary of training metrics.
        """

    @abstractmethod
    def predict(self, features: np.ndarray) -> ModelPrediction:
        """Generate a single prediction.

        Args:
            features: 1-D feature array (from feature_vector_to_array).

        Returns:
            ModelPrediction with probability estimates.
        """

    @abstractmethod
    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Generate batch predictions.

        Args:
            X: 2-D feature array (n_samples, n_features).

        Returns:
            Array of shape (n_samples, 2) with [prob_short, prob_long].
        """

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist model to disk.

        Args:
            path: Directory to save model artifacts into.
        """

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load model from disk.

        Args:
            path: Directory containing saved model artifacts.
        """
