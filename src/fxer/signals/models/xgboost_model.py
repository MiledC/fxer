"""XGBoost signal model with SHAP explanations."""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import shap
import xgboost as xgb

from fxer.core.exceptions import ModelNotFittedError
from fxer.signals.base import FEATURE_COLUMNS, BaseSignalModel
from fxer.signals.types import ModelPrediction

logger = logging.getLogger(__name__)


class XGBoostSignalModel(BaseSignalModel):
    """XGBoost-based signal model (primary workhorse).

    Configuration per project.md:
        n_estimators: 300 (range 100-500)
        max_depth: 6 (range 3-10)
        learning_rate: 0.05 (range 0.01-0.1)
        Early stopping: 50 rounds on validation set
        scale_pos_weight: computed from label distribution
        SHAP TreeExplainer for top-5 feature importance
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        early_stopping_rounds: int = 50,
        random_state: int = 42,
    ) -> None:
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._learning_rate = learning_rate
        self._early_stopping_rounds = early_stopping_rounds
        self._random_state = random_state
        self._model: xgb.XGBClassifier | None = None
        self._explainer: shap.TreeExplainer | None = None

    @property
    def name(self) -> str:
        return "xgboost"

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> dict[str, Any]:
        """Train the XGBoost model.

        Args:
            X: Training features (n_samples, n_features).
            y: Training labels (0 = short, 1 = long).
            validation_data: Optional (X_val, y_val) for early stopping.

        Returns:
            Training metrics including best iteration and validation score.
        """
        # Compute scale_pos_weight for class imbalance
        n_short = np.sum(y == 0)
        n_long = np.sum(y == 1)
        scale_pos_weight = n_short / n_long if n_long > 0 else 1.0

        self._model = xgb.XGBClassifier(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            learning_rate=self._learning_rate,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=self._random_state,
            n_jobs=-1,
        )

        fit_params: dict[str, Any] = {}
        if validation_data is not None:
            fit_params["eval_set"] = [validation_data]
            fit_params["verbose"] = False

        self._model.fit(X, y, **fit_params)

        # Build SHAP explainer
        self._explainer = shap.TreeExplainer(self._model)

        metrics: dict[str, Any] = {
            "n_samples": len(y),
            "n_long": int(n_long),
            "n_short": int(n_short),
            "scale_pos_weight": scale_pos_weight,
        }

        if validation_data is not None:
            val_pred = self._model.predict(validation_data[0])
            val_acc = float(np.mean(val_pred == validation_data[1]))
            metrics["val_accuracy"] = val_acc
            try:
                best_iter = self._model.best_iteration
                if best_iter is not None:
                    metrics["best_iteration"] = best_iter
            except AttributeError:
                pass  # best_iteration only available with early stopping

        logger.info(
            "XGBoost trained: %d samples, scale_pos_weight=%.3f",
            len(y),
            scale_pos_weight,
        )
        return metrics

    def predict(self, features: np.ndarray) -> ModelPrediction:
        """Generate a single prediction with SHAP explanations.

        Args:
            features: 1-D feature array.

        Returns:
            ModelPrediction with class probabilities.
        """
        if self._model is None:
            raise ModelNotFittedError(
                "XGBoost model has not been fitted", model_name=self.name
            )

        X = features.reshape(1, -1)
        probs = self._model.predict_proba(X)[0]

        return ModelPrediction(
            prob_long=float(probs[1]),
            prob_short=float(probs[0]),
            raw_output=float(probs[1]),
        )

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Generate batch predictions.

        Args:
            X: 2-D feature array (n_samples, n_features).

        Returns:
            Array of shape (n_samples, 2) with [prob_short, prob_long].
        """
        if self._model is None:
            raise ModelNotFittedError(
                "XGBoost model has not been fitted", model_name=self.name
            )
        return self._model.predict_proba(X)

    def get_shap_values(self, features: np.ndarray, top_k: int = 5) -> dict[str, float]:
        """Get top-K SHAP feature importances for a single prediction.

        Args:
            features: 1-D feature array.
            top_k: Number of top features to return.

        Returns:
            Dictionary mapping feature names to SHAP values.
        """
        if self._explainer is None:
            return {}

        X = features.reshape(1, -1)
        shap_values = self._explainer.shap_values(X)

        # For binary classification, shap_values may be a list [class0, class1]
        if isinstance(shap_values, list):
            vals = shap_values[1][0]  # SHAP for class 1 (long)
        elif shap_values.ndim == 3:
            vals = shap_values[0, :, 1]
        else:
            vals = shap_values[0]

        # Get top-K by absolute value
        abs_vals = np.abs(vals)
        top_indices = np.argsort(abs_vals)[-top_k:][::-1]

        result = {}
        for idx in top_indices:
            if idx < len(FEATURE_COLUMNS):
                result[FEATURE_COLUMNS[idx]] = float(vals[idx])

        return result

    def save(self, path: Path) -> None:
        """Save model to disk.

        Args:
            path: Directory to save model artifacts.
        """
        if self._model is None:
            raise ModelNotFittedError(
                "Cannot save unfitted model", model_name=self.name
            )
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self._model.save_model(str(path / "xgboost_model.json"))

        # Save hyperparameters for reproducibility
        params = {
            "n_estimators": self._n_estimators,
            "max_depth": self._max_depth,
            "learning_rate": self._learning_rate,
            "early_stopping_rounds": self._early_stopping_rounds,
            "random_state": self._random_state,
        }
        with open(path / "xgboost_params.json", "w") as f:
            json.dump(params, f, indent=2)

        logger.info("Saved XGBoost model to %s", path)

    def load(self, path: Path) -> None:
        """Load model from disk.

        Args:
            path: Directory containing saved model artifacts.
        """
        path = Path(path)
        self._model = xgb.XGBClassifier()
        self._model.load_model(str(path / "xgboost_model.json"))
        self._explainer = shap.TreeExplainer(self._model)

        # Optionally restore hyperparameters
        params_path = path / "xgboost_params.json"
        if params_path.exists():
            with open(params_path) as f:
                params = json.load(f)
            self._n_estimators = params.get("n_estimators", self._n_estimators)
            self._max_depth = params.get("max_depth", self._max_depth)
            self._learning_rate = params.get("learning_rate", self._learning_rate)

        logger.info("Loaded XGBoost model from %s", path)
