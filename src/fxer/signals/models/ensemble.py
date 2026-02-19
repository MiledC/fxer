"""Stacking ensemble combining XGBoost and CNN-Bi-LSTM base models."""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb

from fxer.config.settings import Settings, settings as default_settings
from fxer.core.exceptions import ModelNotFittedError
from fxer.signals.base import FEATURE_COLUMNS, BaseSignalModel, feature_vector_to_array
from fxer.signals.models.cnn_lstm import CNNLSTMSignalModel
from fxer.signals.models.xgboost_model import XGBoostSignalModel
from fxer.signals.training.data_prep import FeatureScaler
from fxer.signals.types import Direction, HoldingPeriod, ModelPrediction, TradeSignal

logger = logging.getLogger(__name__)


class StackingEnsemble(BaseSignalModel):
    """Stacking ensemble with XGBoost meta-learner.

    Combines predictions from XGBoost and CNN-Bi-LSTM base models.
    The meta-learner is trained on **out-of-sample** predictions only
    (per project.md) to avoid information leakage.

    Training procedure:
        1. Split data chronologically: 70% train / 30% OOS
        2. Train XGBoost base model on train split
        3. Train CNN-Bi-LSTM on train split (with sequences)
        4. Generate OOS predictions from both base models
        5. Meta-features = [xgb_prob_short, xgb_prob_long,
                            lstm_prob_short, lstm_prob_long]
        6. Train meta-learner XGBoost (100 trees, depth 3) on
           meta-features + OOS labels

    Direction thresholds:
        prob > 0.70 → high confidence signal
        0.55 < prob ≤ 0.70 → standard signal
        prob ≤ 0.55 → NEUTRAL (no trade)
    """

    def __init__(
        self,
        xgb_model: XGBoostSignalModel | None = None,
        lstm_model: CNNLSTMSignalModel | None = None,
        scaler: FeatureScaler | None = None,
        settings: Settings | None = None,
    ) -> None:
        s = settings or default_settings
        self._xgb = xgb_model or XGBoostSignalModel()
        self._lstm = lstm_model or CNNLSTMSignalModel(lookback=s.signal_lookback_window)
        self._scaler = scaler or FeatureScaler()
        self._meta_learner: xgb.XGBClassifier | None = None
        self._neutral_threshold = s.signal_neutral_threshold
        self._high_conf_threshold = s.signal_high_confidence_threshold
        self._lookback = s.signal_lookback_window

    @property
    def name(self) -> str:
        return "stacking_ensemble"

    @property
    def is_fitted(self) -> bool:
        return (
            self._xgb.is_fitted
            and self._lstm.is_fitted
            and self._meta_learner is not None
        )

    @property
    def xgb_model(self) -> XGBoostSignalModel:
        return self._xgb

    @property
    def lstm_model(self) -> CNNLSTMSignalModel:
        return self._lstm

    @property
    def scaler(self) -> FeatureScaler:
        return self._scaler

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> dict[str, Any]:
        """Train the full stacking ensemble.

        This orchestrates the full training flow:
        1. Split data into train (70%) and OOS (30%)
        2. Train base models on train split
        3. Generate OOS predictions
        4. Train meta-learner on OOS predictions

        Args:
            X: Flat feature matrix (n_samples, n_features).
            y: Labels (n_samples,).
            validation_data: Ignored (ensemble manages its own splits).

        Returns:
            Training metrics for all components.
        """
        n = len(y)
        split_idx = int(n * 0.70)

        X_train, X_oos = X[:split_idx], X[split_idx:]
        y_train, y_oos = y[:split_idx], y[split_idx:]

        logger.info(
            "Ensemble split: %d train, %d OOS", len(y_train), len(y_oos)
        )

        # --- Step 1: Fit scaler on training data ---
        X_train_scaled = self._scaler.fit_transform(X_train)
        X_oos_scaled = self._scaler.transform(X_oos)

        # --- Step 2: Train XGBoost base model (uses unscaled features) ---
        xgb_val = (X_oos[:min(500, len(X_oos))], y_oos[:min(500, len(y_oos))])
        xgb_metrics = self._xgb.fit(X_train, y_train, validation_data=xgb_val)

        # --- Step 3: Train CNN-Bi-LSTM (uses scaled, sequenced features) ---
        from fxer.signals.training.data_prep import DatasetBuilder

        builder = DatasetBuilder.__new__(DatasetBuilder)
        X_train_seq, y_train_seq = self._build_sequences(
            X_train_scaled, y_train, self._lookback
        )
        X_oos_seq, y_oos_seq = self._build_sequences(
            X_oos_scaled, y_oos, self._lookback
        )

        lstm_val = None
        if len(X_oos_seq) > 0:
            lstm_val = (X_oos_seq[:min(500, len(X_oos_seq))],
                        y_oos_seq[:min(500, len(y_oos_seq))])
        lstm_metrics = self._lstm.fit(X_train_seq, y_train_seq, validation_data=lstm_val)

        # --- Step 4: Generate OOS predictions ---
        xgb_oos_probs = self._xgb.predict_batch(X_oos)  # (n_oos, 2)
        lstm_oos_probs = self._lstm.predict_batch(X_oos_seq)  # (n_oos_seq, 2)

        # Align lengths: LSTM sequences are shorter by (lookback - 1)
        offset = len(y_oos) - len(y_oos_seq)
        xgb_oos_aligned = xgb_oos_probs[offset:]
        y_oos_aligned = y_oos[offset:]

        # --- Step 5: Build meta-features ---
        meta_features = np.column_stack([
            xgb_oos_aligned,   # [prob_short, prob_long]
            lstm_oos_probs,     # [prob_short, prob_long]
        ])

        # --- Step 6: Train meta-learner ---
        self._meta_learner = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
        )
        self._meta_learner.fit(meta_features, y_oos_aligned, verbose=False)

        # Evaluate meta-learner
        meta_pred = self._meta_learner.predict(meta_features)
        meta_acc = float(np.mean(meta_pred == y_oos_aligned))

        logger.info("Ensemble trained: meta-learner accuracy=%.4f", meta_acc)

        return {
            "xgb": xgb_metrics,
            "lstm": lstm_metrics,
            "meta_accuracy": meta_acc,
            "meta_n_samples": len(y_oos_aligned),
        }

    def predict(self, features: np.ndarray) -> ModelPrediction:
        """Generate an ensemble prediction.

        Args:
            features: 1-D flat feature array (unscaled).

        Returns:
            ModelPrediction from the meta-learner.
        """
        if not self.is_fitted:
            raise ModelNotFittedError(
                "Ensemble has not been fully fitted", model_name=self.name
            )

        # XGBoost prediction (unscaled)
        xgb_pred = self._xgb.predict(features)

        # LSTM prediction (scaled, from history buffer)
        scaled = self._scaler.transform(features.reshape(1, -1))[0]
        lstm_pred = self._lstm.predict(scaled)

        # Meta-features
        meta_input = np.array([[
            xgb_pred.prob_short, xgb_pred.prob_long,
            lstm_pred.prob_short, lstm_pred.prob_long,
        ]])

        meta_probs = self._meta_learner.predict_proba(meta_input)[0]

        return ModelPrediction(
            prob_long=float(meta_probs[1]),
            prob_short=float(meta_probs[0]),
            raw_output=float(meta_probs[1]),
        )

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Generate batch predictions.

        Args:
            X: 2-D flat feature array (n_samples, n_features), unscaled.

        Returns:
            Array of shape (n_samples, 2) with [prob_short, prob_long].
        """
        if not self.is_fitted:
            raise ModelNotFittedError(
                "Ensemble has not been fully fitted", model_name=self.name
            )

        # XGBoost predictions
        xgb_probs = self._xgb.predict_batch(X)

        # LSTM predictions (need sequences)
        X_scaled = self._scaler.transform(X)
        X_seq, _ = self._build_sequences(
            X_scaled, np.zeros(len(X)), self._lookback
        )
        lstm_probs = self._lstm.predict_batch(X_seq)

        # Align lengths
        offset = len(X) - len(X_seq)
        xgb_aligned = xgb_probs[offset:]

        # Meta-features
        meta_features = np.column_stack([xgb_aligned, lstm_probs])
        return self._meta_learner.predict_proba(meta_features)

    def generate_signal(
        self,
        features: np.ndarray,
        symbol: str,
        timestamp: "datetime",
        horizon: HoldingPeriod,
    ) -> TradeSignal:
        """Generate a full TradeSignal from features.

        Args:
            features: 1-D flat feature array (unscaled).
            symbol: Trading symbol.
            timestamp: Signal timestamp.
            horizon: Expected holding period.

        Returns:
            TradeSignal with direction, confidence, and SHAP explanations.
        """
        from datetime import datetime as dt

        ensemble_pred = self.predict(features)
        xgb_pred = self._xgb.predict(features)

        # Determine direction and confidence
        prob_long = ensemble_pred.prob_long
        prob_short = ensemble_pred.prob_short
        max_prob = max(prob_long, prob_short)

        if max_prob <= self._neutral_threshold:
            direction = Direction.NEUTRAL
            confidence = 0.0
        elif prob_long > prob_short:
            direction = Direction.LONG
            confidence = prob_long
        else:
            direction = Direction.SHORT
            confidence = prob_short

        # Get SHAP explanations from XGBoost
        top_features = self._xgb.get_shap_values(features, top_k=5)

        return TradeSignal(
            symbol=symbol,
            timestamp=timestamp,
            direction=direction,
            confidence=confidence,
            predicted_horizon=horizon,
            xgboost_prob=xgb_pred.prob_long,
            lstm_prob=None,  # LSTM uses history buffer, not single-step
            ensemble_prob=prob_long,
            top_features=top_features,
            warmup_complete=True,
        )

    def save(self, path: Path) -> None:
        """Save all ensemble components to disk.

        Args:
            path: Directory to save all model artifacts.
        """
        if not self.is_fitted:
            raise ModelNotFittedError(
                "Cannot save unfitted ensemble", model_name=self.name
            )
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self._xgb.save(path / "xgboost")
        self._lstm.save(path / "cnn_lstm")
        self._scaler.save(path / "scaler.npz")

        # Save meta-learner
        self._meta_learner.save_model(str(path / "meta_learner.json"))

        # Save ensemble config
        config = {
            "neutral_threshold": self._neutral_threshold,
            "high_conf_threshold": self._high_conf_threshold,
            "lookback": self._lookback,
        }
        with open(path / "ensemble_config.json", "w") as f:
            json.dump(config, f, indent=2)

        logger.info("Saved ensemble to %s", path)

    def load(self, path: Path) -> None:
        """Load all ensemble components from disk.

        Args:
            path: Directory containing saved model artifacts.
        """
        path = Path(path)

        self._xgb.load(path / "xgboost")
        self._lstm.load(path / "cnn_lstm")
        self._scaler.load(path / "scaler.npz")

        self._meta_learner = xgb.XGBClassifier()
        self._meta_learner.load_model(str(path / "meta_learner.json"))

        config_path = path / "ensemble_config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            self._neutral_threshold = config.get(
                "neutral_threshold", self._neutral_threshold
            )
            self._high_conf_threshold = config.get(
                "high_conf_threshold", self._high_conf_threshold
            )
            self._lookback = config.get("lookback", self._lookback)

        logger.info("Loaded ensemble from %s", path)

    @staticmethod
    def _build_sequences(
        X: np.ndarray, y: np.ndarray, lookback: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build sliding-window sequences for LSTM input.

        Args:
            X: Flat feature matrix (n_samples, n_features).
            y: Labels (n_samples,).
            lookback: Sequence length.

        Returns:
            (X_seq, y_seq) with X_seq shape (n, lookback, features).
        """
        n = len(X)
        if n < lookback:
            return np.empty((0, lookback, X.shape[1])), np.empty((0,))

        n_seq = n - lookback + 1
        X_seq = np.zeros((n_seq, lookback, X.shape[1]), dtype=np.float64)
        for i in range(n_seq):
            X_seq[i] = X[i : i + lookback]

        y_seq = y[lookback - 1 :]
        return X_seq, y_seq
