"""Dataset preparation for signal model training."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from fxer.config.settings import Settings, settings as default_settings
from fxer.core.exceptions import InsufficientDataError
from fxer.core.types import Timeframe
from fxer.data.storage.questdb_client import QuestDBClient
from fxer.signals.base import FEATURE_COLUMNS

logger = logging.getLogger(__name__)


class LabelGenerator:
    """Create binary training labels from price data.

    Label = 1 (LONG) if future close > current close + threshold
    Label = 0 (SHORT) if future close < current close - threshold
    Rows within the dead zone (between thresholds) are marked as NaN
    and should be dropped before training.
    """

    def __init__(
        self,
        horizon_bars: int = 12,
        threshold_atr_mult: float = 0.5,
    ) -> None:
        self.horizon_bars = horizon_bars
        self.threshold_atr_mult = threshold_atr_mult

    def generate(self, bars_df: pd.DataFrame) -> pd.Series:
        """Generate labels from a DataFrame of bar data.

        When ``threshold_atr_mult == 0`` (sign-of-return mode), every bar
        with a valid forward return gets a label — no dead zone.  This
        maximises data utilisation and produces a cleaner learning target.

        When ``threshold_atr_mult > 0``, the original ATR-threshold
        behaviour is preserved for backward compatibility.

        Args:
            bars_df: DataFrame with columns 'close' and 'atr_14' (or computable).
                     Must be sorted by timestamp ascending.

        Returns:
            Series of labels: 1 (long), 0 (short), NaN (no future data).
            Same index as bars_df.
        """
        close = bars_df["close"].astype(float)
        future_close = close.shift(-self.horizon_bars)
        price_change = future_close - close

        if self.threshold_atr_mult == 0:
            # Sign-of-return labeling: no dead zone
            labels = pd.Series(np.nan, index=bars_df.index, dtype=float)
            valid = future_close.notna()
            labels[valid & (price_change > 0)] = 1.0   # LONG
            labels[valid & (price_change <= 0)] = 0.0   # SHORT
        else:
            # ATR-threshold labeling (original behaviour)
            if "atr_14" in bars_df.columns:
                atr = bars_df["atr_14"].astype(float)
            else:
                atr = self._compute_atr(bars_df, period=14)

            threshold = atr * self.threshold_atr_mult

            labels = pd.Series(np.nan, index=bars_df.index, dtype=float)
            labels[price_change > threshold] = 1.0   # LONG
            labels[price_change < -threshold] = 0.0   # SHORT

        n_long = (labels == 1.0).sum()
        n_short = (labels == 0.0).sum()
        n_dead = labels.isna().sum()
        logger.info(
            "Labels generated: %d long, %d short, %d dead zone (dropped)",
            n_long,
            n_short,
            n_dead,
        )
        return labels

    @staticmethod
    def _compute_atr(bars_df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Compute ATR from OHLC data."""
        high = bars_df["high"].astype(float)
        low = bars_df["low"].astype(float)
        close = bars_df["close"].astype(float)

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()


class DatasetBuilder:
    """Build training datasets from QuestDB data."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or default_settings
        self._db = QuestDBClient(self._settings)

    def build_flat_dataset(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Query features and bars from QuestDB, join, and build (X, y) matrices.

        Args:
            symbol: Trading symbol (e.g. "XAUUSD").
            timeframe: Bar timeframe (e.g. Timeframe.M5).
            start: Start datetime (inclusive).
            end: End datetime (inclusive).

        Returns:
            Tuple of (X DataFrame with FEATURE_COLUMNS, y array of labels).
            Rows in the dead zone are dropped.
        """
        features_df = self._query_features(symbol, timeframe, start, end)
        bars_df = self._query_bars(symbol, timeframe, start, end)

        if len(bars_df) < 100:
            raise InsufficientDataError(
                f"Need at least 100 bars, got {len(bars_df)}",
                required=100,
                available=len(bars_df),
            )

        # Merge features with bar close/ATR for label generation
        merged = features_df.join(
            bars_df[["close", "high", "low"]],
            how="inner",
        )

        label_gen = LabelGenerator(
            horizon_bars=self._settings.signal_horizon_bars,
            threshold_atr_mult=self._settings.signal_threshold_atr_mult,
        )
        labels = label_gen.generate(merged)

        # Drop dead-zone rows (NaN labels) and future-leaking tail
        valid_mask = labels.notna()
        merged = merged.loc[valid_mask]
        labels = labels.loc[valid_mask]

        # Extract feature columns — drop rows with any NaN features
        X = merged[FEATURE_COLUMNS]
        nan_rows = X.isna().any(axis=1).sum()
        if nan_rows > 0:
            logger.warning(
                "Dropping %d rows (%.1f%%) with NaN features",
                nan_rows,
                100 * nan_rows / len(X),
            )
            valid = X.notna().all(axis=1)
            X = X.loc[valid]
            labels = labels.loc[valid]
        y = labels.values.astype(np.float64)

        logger.info("Built flat dataset: X=%s, y=%s", X.shape, y.shape)
        return X, y

    def build_sequence_dataset(
        self,
        X_flat: pd.DataFrame | np.ndarray,
        y: np.ndarray,
        lookback: int = 48,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create sliding-window sequences for LSTM input.

        Args:
            X_flat: Flat feature matrix (n_samples, n_features).
            y: Label array (n_samples,).
            lookback: Number of timesteps per sequence.

        Returns:
            Tuple of (X_seq of shape (n, lookback, features), y_seq of shape (n,)).
        """
        if isinstance(X_flat, pd.DataFrame):
            X_arr = X_flat.values.astype(np.float64)
        else:
            X_arr = X_flat.astype(np.float64)

        n_samples = X_arr.shape[0]
        if n_samples < lookback:
            raise InsufficientDataError(
                f"Need at least {lookback} samples for sequences, got {n_samples}",
                required=lookback,
                available=n_samples,
            )

        n_features = X_arr.shape[1]
        n_seq = n_samples - lookback + 1

        X_seq = np.zeros((n_seq, lookback, n_features), dtype=np.float64)
        for i in range(n_seq):
            X_seq[i] = X_arr[i : i + lookback]

        # Labels correspond to the last timestep in each window
        y_seq = y[lookback - 1 :]

        logger.info("Built sequence dataset: X=%s, y=%s", X_seq.shape, y_seq.shape)
        return X_seq, y_seq

    def _query_features(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Query features table from QuestDB and return as DataFrame."""
        conn = self._db._get_pg_connection()
        try:
            query = """
                SELECT timestamp, {cols}
                FROM features
                WHERE symbol = %s
                  AND timeframe = %s
                  AND timestamp >= %s
                  AND timestamp <= %s
                  AND warmup_complete = true
                ORDER BY timestamp ASC
            """.format(cols=", ".join(FEATURE_COLUMNS))

            df = pd.read_sql(
                query,
                conn,
                params=(symbol, timeframe.value, start.isoformat(), end.isoformat()),
                parse_dates=["timestamp"],
            )
            df = df.set_index("timestamp")
            df = df[FEATURE_COLUMNS]

            # Fail loudly if any feature column is entirely NULL
            all_null = df.columns[df.isna().all()]
            if len(all_null) > 0:
                raise InsufficientDataError(
                    f"Feature columns are entirely NULL (never computed): "
                    f"{list(all_null)}. Run backfill_cross_asset or "
                    f"recompute features before training.",
                    required=len(FEATURE_COLUMNS),
                    available=len(FEATURE_COLUMNS) - len(all_null),
                )

            # Warn on columns with high NULL rates (>50%)
            null_pct = df.isna().mean()
            sparse = null_pct[null_pct > 0.5]
            if len(sparse) > 0:
                for col, pct in sparse.items():
                    logger.warning(
                        "Feature '%s' is %.0f%% NULL — check data coverage",
                        col,
                        pct * 100,
                    )

            return df
        finally:
            conn.close()

    def _query_bars(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Query bars table from QuestDB and return as DataFrame."""
        conn = self._db._get_pg_connection()
        try:
            df = pd.read_sql(
                """
                SELECT timestamp, open, high, low, close, volume
                FROM bars
                WHERE symbol = %s
                  AND timeframe = %s
                  AND timestamp >= %s
                  AND timestamp <= %s
                ORDER BY timestamp ASC
                """,
                conn,
                params=(symbol, timeframe.value, start.isoformat(), end.isoformat()),
                parse_dates=["timestamp"],
            )
            df = df.set_index("timestamp")
            return df
        finally:
            conn.close()


class FeatureScaler:
    """StandardScaler wrapper that persists alongside models.

    Required for CNN-Bi-LSTM (neural nets need normalized inputs).
    Optional for XGBoost (tree models are scale-invariant).
    """

    def __init__(self) -> None:
        self._scaler = StandardScaler()
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit_transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Fit the scaler and transform data in one step.

        Args:
            X: Feature matrix (n_samples, n_features).

        Returns:
            Scaled feature matrix.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        result = self._scaler.fit_transform(X).astype(np.float64)
        self._is_fitted = True
        return result

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Transform data using the fitted scaler.

        Args:
            X: Feature matrix (n_samples, n_features).

        Returns:
            Scaled feature matrix.
        """
        if not self._is_fitted:
            raise RuntimeError("FeatureScaler has not been fitted yet")
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self._scaler.transform(X).astype(np.float64)

    def save(self, path: Path) -> None:
        """Save the scaler to disk using numpy.

        Args:
            path: File path to save to (e.g. models/scaler.npz).
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted scaler")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            mean=self._scaler.mean_,
            scale=self._scaler.scale_,
            var=self._scaler.var_,
            n_samples_seen=np.array([self._scaler.n_samples_seen_]),
            n_features=np.array([self._scaler.n_features_in_]),
        )
        logger.info("Saved scaler to %s", path)

    def load(self, path: Path) -> None:
        """Load the scaler from disk.

        Args:
            path: File path to load from.
        """
        path = Path(path)
        data = np.load(path)
        self._scaler.mean_ = data["mean"]
        self._scaler.scale_ = data["scale"]
        self._scaler.var_ = data["var"]
        self._scaler.n_samples_seen_ = int(data["n_samples_seen"][0])
        self._scaler.n_features_in_ = int(data["n_features"][0])
        self._is_fitted = True
        logger.info("Loaded scaler from %s", path)
