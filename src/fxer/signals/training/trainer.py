"""Training orchestrator for the signal generation pipeline."""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from fxer.config.settings import Settings, settings as default_settings
from fxer.core.exceptions import InsufficientDataError
from fxer.core.types import Timeframe
from fxer.signals.models.cnn_lstm import CNNLSTMSignalModel
from fxer.signals.models.ensemble import StackingEnsemble
from fxer.signals.models.xgboost_model import XGBoostSignalModel
from fxer.signals.training.data_prep import DatasetBuilder, FeatureScaler
from fxer.signals.training.metrics import SignalMetrics, compute_metrics
from fxer.signals.training.validation import WalkForwardSplitter

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    """Results from a single walk-forward fold."""

    fold_index: int
    metrics: SignalMetrics
    train_size: int
    test_size: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold_index": self.fold_index,
            "metrics": self.metrics.to_dict(),
            "train_size": self.train_size,
            "test_size": self.test_size,
        }


@dataclass
class TrainResult:
    """Complete training result with metrics per fold."""

    symbol: str
    timeframe: str
    start: str
    end: str
    fold_results: list[FoldResult] = field(default_factory=list)
    final_metrics: SignalMetrics | None = None
    model_path: str | None = None

    @property
    def mean_sharpe(self) -> float:
        if not self.fold_results:
            return 0.0
        return float(np.mean([f.metrics.sharpe_ratio for f in self.fold_results]))

    @property
    def mean_win_rate(self) -> float:
        if not self.fold_results:
            return 0.0
        return float(np.mean([f.metrics.win_rate for f in self.fold_results]))

    @property
    def worst_drawdown(self) -> float:
        if not self.fold_results:
            return 0.0
        return float(np.max([f.metrics.max_drawdown for f in self.fold_results]))

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "start": self.start,
            "end": self.end,
            "fold_results": [f.to_dict() for f in self.fold_results],
            "final_metrics": self.final_metrics.to_dict() if self.final_metrics else None,
            "model_path": self.model_path,
            "mean_sharpe": self.mean_sharpe,
            "mean_win_rate": self.mean_win_rate,
            "worst_drawdown": self.worst_drawdown,
        }


class Trainer:
    """Orchestrates the full training pipeline.

    End-to-end flow:
        1. Build dataset via DatasetBuilder (query QuestDB)
        2. Generate labels via LabelGenerator
        3. Fit FeatureScaler on training portion
        4. Run walk-forward validation
        5. Train final ensemble on full data
        6. Save all artifacts (models, scaler, metadata)
        7. Return TrainResult with metrics per fold
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or default_settings
        self._builder = DatasetBuilder(self._settings)
        self._model_dir = Path(self._settings.signal_model_dir)

    def train(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
        n_folds: int = 5,
    ) -> TrainResult:
        """Run the full training pipeline.

        Args:
            symbol: Trading symbol (e.g. "XAUUSD").
            timeframe: Bar timeframe (e.g. Timeframe.M5).
            start: Training start datetime.
            end: Training end datetime.
            n_folds: Number of walk-forward validation folds.

        Returns:
            TrainResult with per-fold metrics and final model path.
        """
        result = TrainResult(
            symbol=symbol,
            timeframe=timeframe.value,
            start=start.isoformat(),
            end=end.isoformat(),
        )

        # --- Step 1: Build dataset ---
        logger.info("Building dataset for %s %s [%s, %s]", symbol, timeframe.value, start, end)
        X, y = self._builder.build_flat_dataset(symbol, timeframe, start, end)

        if len(y) < 200:
            raise InsufficientDataError(
                f"Need at least 200 labeled samples, got {len(y)}",
                required=200,
                available=len(y),
            )

        # We need bar returns for metric computation
        bars_returns = self._get_returns(symbol, timeframe, start, end, X)

        # --- Step 2: Walk-forward validation ---
        logger.info("Running %d-fold walk-forward validation", n_folds)
        splitter = WalkForwardSplitter(n_splits=n_folds, purge_gap=self._settings.signal_horizon_bars)
        splits = splitter.split(len(y))

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            fold_result = self._train_fold(
                X, y, bars_returns, train_idx, test_idx, fold_idx
            )
            result.fold_results.append(fold_result)
            logger.info(
                "Fold %d: sharpe=%.3f, win_rate=%.3f, drawdown=%.3f",
                fold_idx + 1,
                fold_result.metrics.sharpe_ratio,
                fold_result.metrics.win_rate,
                fold_result.metrics.max_drawdown,
            )

        # --- Step 3: Train final ensemble on full data ---
        logger.info("Training final ensemble on full dataset")
        ensemble = StackingEnsemble(settings=self._settings)
        if isinstance(X, np.ndarray):
            ensemble.fit(X, y)
        else:
            ensemble.fit(X.values, y)

        # --- Step 4: Save artifacts ---
        save_dir = self._model_dir / symbol.lower() / timeframe.value
        ensemble.save(save_dir)
        result.model_path = str(save_dir)

        # Save training metadata
        metadata = result.to_dict()
        with open(save_dir / "train_result.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(
            "Training complete: mean_sharpe=%.3f, mean_win_rate=%.3f",
            result.mean_sharpe,
            result.mean_win_rate,
        )
        return result

    def _train_fold(
        self,
        X: np.ndarray | Any,
        y: np.ndarray,
        returns: np.ndarray,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        fold_idx: int,
    ) -> FoldResult:
        """Train and evaluate a single walk-forward fold."""
        X_arr = X.values if hasattr(X, "values") else X

        X_train, y_train = X_arr[train_idx], y[train_idx]
        X_test, y_test = X_arr[test_idx], y[test_idx]
        test_returns = returns[test_idx] if len(returns) > max(test_idx) else np.zeros(len(test_idx))

        # Train a fresh ensemble for this fold
        ensemble = StackingEnsemble(settings=self._settings)
        ensemble.fit(X_train, y_train)

        # Generate predictions on test set
        # Use XGBoost-only predictions for fold metrics (LSTM needs sequences
        # which may not align perfectly with fold boundaries)
        xgb_probs = ensemble.xgb_model.predict_batch(X_test)
        predictions = (xgb_probs[:, 1] > 0.5).astype(int)

        metrics = compute_metrics(predictions, y_test.astype(int), test_returns)

        return FoldResult(
            fold_index=fold_idx,
            metrics=metrics,
            train_size=len(y_train),
            test_size=len(y_test),
        )

    def _get_returns(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
        X: Any,
    ) -> np.ndarray:
        """Query bar close prices and compute per-bar returns.

        Returns array aligned with X's index.
        """
        try:
            conn = self._builder._db._get_pg_connection()
            try:
                import pandas as pd

                df = pd.read_sql(
                    """
                    SELECT timestamp, close
                    FROM bars
                    WHERE symbol = %s AND timeframe = %s
                      AND timestamp >= %s AND timestamp <= %s
                    ORDER BY timestamp ASC
                    """,
                    conn,
                    params=(symbol, timeframe.value, start.isoformat(), end.isoformat()),
                    parse_dates=["timestamp"],
                )
                df = df.set_index("timestamp")
                df["close"] = df["close"].astype(float)
                df["return"] = df["close"].pct_change().shift(-1).fillna(0.0)

                # Align with X index
                if hasattr(X, "index"):
                    aligned = df["return"].reindex(X.index).fillna(0.0)
                    return aligned.values
                return df["return"].values[: len(X)]
            finally:
                conn.close()
        except Exception as exc:
            logger.warning("Could not compute returns: %s. Using zeros.", exc)
            n = len(X) if hasattr(X, "__len__") else X.shape[0]
            return np.zeros(n)
