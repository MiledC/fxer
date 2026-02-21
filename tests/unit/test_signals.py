"""Unit tests for the signal generation module."""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fxer.core.events import FeatureVector
from fxer.core.types import Timeframe
from fxer.signals.base import FEATURE_COLUMNS, feature_vector_to_array
from fxer.signals.models.cnn_lstm import CNNLSTMSignalModel
from fxer.signals.models.ensemble import StackingEnsemble
from fxer.signals.models.xgboost_model import XGBoostSignalModel
from fxer.signals.training.data_prep import FeatureScaler, LabelGenerator
from fxer.signals.training.metrics import SignalMetrics, compute_metrics
from fxer.signals.training.validation import CPCVSplitter, WalkForwardSplitter
from fxer.signals.types import Direction, HoldingPeriod, ModelPrediction, TradeSignal


# ---------- Fixtures ----------


def _make_feature_vector(**kwargs):
    """Create a FeatureVector with defaults."""
    defaults = {
        "symbol": "XAUUSD",
        "timestamp": datetime(2025, 1, 15, 12, 0, 0),
        "timeframe": Timeframe.M5,
        "rsi_14": 55.0,
        "rsi_7": 60.0,
        "macd_line": 0.5,
        "macd_signal": 0.3,
        "macd_histogram": 0.2,
        "bb_upper": 2050.0,
        "bb_middle": 2040.0,
        "bb_lower": 2030.0,
        "bb_width": 0.01,
        "atr_14": 5.0,
        "is_london_session": True,
        "is_ny_session": True,
        "is_overlap_session": True,
        "is_asian_session": False,
        "hour_of_day": 14,
        "day_of_week": 2,
        "is_month_turn": False,
        "warmup_complete": True,
        "dxy_return_1h": -0.001,
        "dxy_rsi_14": 48.0,
        "vix_level": 15.5,
        "vix_change": 0.3,
        # Price-derived features
        "return_1bar": 0.001,
        "return_5bar": 0.005,
        "return_12bar": 0.012,
        "rolling_volatility_20": 0.008,
        "momentum_48": 0.025,
    }
    defaults.update(kwargs)
    return FeatureVector(**defaults)


def _make_synthetic_data(n_samples=500, n_features=21):
    """Create synthetic data for model testing."""
    rng = np.random.RandomState(42)
    X = rng.randn(n_samples, n_features).astype(np.float64)
    # Create a signal: if feature 0 > 0, tend toward class 1
    prob = 1 / (1 + np.exp(-2 * X[:, 0]))
    y = (rng.rand(n_samples) < prob).astype(np.float64)
    return X, y


def _make_price_series(n_bars=500, start_price=2000.0, atr=5.0):
    """Create a synthetic price series DataFrame."""
    rng = np.random.RandomState(42)
    changes = rng.randn(n_bars) * atr
    close = np.cumsum(changes) + start_price
    high = close + np.abs(rng.randn(n_bars)) * atr * 0.5
    low = close - np.abs(rng.randn(n_bars)) * atr * 0.5
    open_ = close + rng.randn(n_bars) * atr * 0.2

    # Ensure OHLC consistency
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))

    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": rng.randint(100, 10000, size=n_bars).astype(float),
        "atr_14": np.full(n_bars, atr),
    })
    return df


# ---------- TradeSignal Tests ----------


class TestTradeSignal:
    """Test TradeSignal serialization and deserialization."""

    def test_to_dict_from_dict_roundtrip(self):
        ts = TradeSignal(
            symbol="XAUUSD",
            timestamp=datetime(2025, 1, 15, 14, 30, 0),
            direction=Direction.LONG,
            confidence=0.82,
            predicted_horizon=HoldingPeriod.HOUR_1,
            xgboost_prob=0.78,
            lstm_prob=0.85,
            ensemble_prob=0.82,
            top_features={"rsi_14": 0.15, "macd_line": 0.10},
            warmup_complete=True,
        )

        d = ts.to_dict()
        restored = TradeSignal.from_dict(d)

        assert restored.symbol == ts.symbol
        assert restored.timestamp == ts.timestamp
        assert restored.direction == ts.direction
        assert restored.confidence == pytest.approx(ts.confidence)
        assert restored.predicted_horizon == ts.predicted_horizon
        assert restored.xgboost_prob == pytest.approx(ts.xgboost_prob)
        assert restored.lstm_prob == pytest.approx(ts.lstm_prob)
        assert restored.ensemble_prob == pytest.approx(ts.ensemble_prob)
        assert restored.top_features == ts.top_features
        assert restored.warmup_complete == ts.warmup_complete

    def test_to_dict_contains_all_fields(self):
        ts = TradeSignal(
            symbol="XAUUSD",
            timestamp=datetime(2025, 1, 15, 14, 30, 0),
            direction=Direction.SHORT,
            confidence=0.65,
            predicted_horizon=HoldingPeriod.MIN_30,
            xgboost_prob=None,
            lstm_prob=None,
            ensemble_prob=0.65,
            top_features={},
            warmup_complete=False,
        )

        d = ts.to_dict()
        assert "symbol" in d
        assert "timestamp" in d
        assert "direction" in d
        assert d["direction"] == "short"
        assert d["xgboost_prob"] is None

    def test_from_dict_with_none_probs(self):
        d = {
            "symbol": "XAUUSD",
            "timestamp": "2025-01-15T14:30:00",
            "direction": "neutral",
            "confidence": 0.0,
            "predicted_horizon": "15m",
            "top_features": {},
            "warmup_complete": False,
        }
        ts = TradeSignal.from_dict(d)
        assert ts.direction == Direction.NEUTRAL
        assert ts.xgboost_prob is None
        assert ts.lstm_prob is None


# ---------- Direction & HoldingPeriod Tests ----------


class TestDirection:
    def test_values(self):
        assert Direction.LONG.value == "long"
        assert Direction.SHORT.value == "short"
        assert Direction.NEUTRAL.value == "neutral"


class TestHoldingPeriod:
    def test_bars_5m(self):
        assert HoldingPeriod.MIN_15.bars_5m == 3
        assert HoldingPeriod.MIN_30.bars_5m == 6
        assert HoldingPeriod.HOUR_1.bars_5m == 12
        assert HoldingPeriod.HOUR_2.bars_5m == 24

    def test_from_bars(self):
        assert HoldingPeriod.from_bars(3) == HoldingPeriod.MIN_15
        assert HoldingPeriod.from_bars(6) == HoldingPeriod.MIN_30
        assert HoldingPeriod.from_bars(12) == HoldingPeriod.HOUR_1
        assert HoldingPeriod.from_bars(24) == HoldingPeriod.HOUR_2


class TestModelPrediction:
    def test_predicted_direction_long(self):
        p = ModelPrediction(prob_long=0.8, prob_short=0.2, raw_output=0.8)
        assert p.predicted_direction == Direction.LONG

    def test_predicted_direction_short(self):
        p = ModelPrediction(prob_long=0.3, prob_short=0.7, raw_output=0.3)
        assert p.predicted_direction == Direction.SHORT

    def test_predicted_direction_neutral(self):
        p = ModelPrediction(prob_long=0.5, prob_short=0.5, raw_output=0.5)
        assert p.predicted_direction == Direction.NEUTRAL


# ---------- FeatureVector Conversion Tests ----------


class TestFeatureVectorConversion:
    def test_feature_columns_length_is_26(self):
        """Verify FEATURE_COLUMNS has exactly 26 entries."""
        assert len(FEATURE_COLUMNS) == 26

    def test_feature_columns_all_valid_attributes(self):
        """Verify every entry in FEATURE_COLUMNS is a valid FeatureVector attribute."""
        fv = _make_feature_vector()
        for col in FEATURE_COLUMNS:
            assert hasattr(fv, col), f"FeatureVector missing field: {col}"

    def test_feature_vector_to_array_shape(self):
        fv = _make_feature_vector()
        arr = feature_vector_to_array(fv)
        assert arr.shape == (len(FEATURE_COLUMNS),)
        assert arr.dtype == np.float64

    def test_feature_vector_to_array_values(self):
        fv = _make_feature_vector(rsi_14=55.0, atr_14=5.0)
        arr = feature_vector_to_array(fv)
        rsi_idx = FEATURE_COLUMNS.index("rsi_14")
        atr_idx = FEATURE_COLUMNS.index("atr_14")
        assert arr[rsi_idx] == pytest.approx(55.0)
        assert arr[atr_idx] == pytest.approx(5.0)

    def test_none_replaced_with_zero(self):
        fv = _make_feature_vector(dxy_return_1h=None)
        arr = feature_vector_to_array(fv)
        idx = FEATURE_COLUMNS.index("dxy_return_1h")
        assert arr[idx] == 0.0

    def test_bool_converted_to_float(self):
        fv = _make_feature_vector(is_london_session=True, is_asian_session=False)
        arr = feature_vector_to_array(fv)
        london_idx = FEATURE_COLUMNS.index("is_london_session")
        asian_idx = FEATURE_COLUMNS.index("is_asian_session")
        assert arr[london_idx] == 1.0
        assert arr[asian_idx] == 0.0


# ---------- LabelGenerator Tests ----------


class TestLabelGenerator:
    def test_generate_labels_from_known_series(self):
        """Verify labels from a simple, predictable price series."""
        # Price goes from 100 to 200 linearly
        df = pd.DataFrame({
            "close": np.linspace(100, 200, 100),
            "high": np.linspace(101, 201, 100),
            "low": np.linspace(99, 199, 100),
            "atr_14": np.full(100, 2.0),
        })

        gen = LabelGenerator(horizon_bars=5, threshold_atr_mult=0.5)
        labels = gen.generate(df)

        # Since prices monotonically increase, all labels in range should be LONG (1)
        valid = labels.dropna()
        assert (valid == 1.0).all()

    def test_generate_labels_short(self):
        """Verify SHORT labels from a declining price series."""
        df = pd.DataFrame({
            "close": np.linspace(200, 100, 100),
            "high": np.linspace(201, 101, 100),
            "low": np.linspace(199, 99, 100),
            "atr_14": np.full(100, 2.0),
        })

        gen = LabelGenerator(horizon_bars=5, threshold_atr_mult=0.5)
        labels = gen.generate(df)

        valid = labels.dropna()
        assert (valid == 0.0).all()

    def test_dead_zone_labels(self):
        """Flat price should produce NaN (dead zone) labels."""
        df = pd.DataFrame({
            "close": np.full(100, 100.0),
            "high": np.full(100, 101.0),
            "low": np.full(100, 99.0),
            "atr_14": np.full(100, 2.0),
        })

        gen = LabelGenerator(horizon_bars=5, threshold_atr_mult=0.5)
        labels = gen.generate(df)

        # With no price change and ATR-based threshold, most should be dead zone
        # (except last 5 which are NaN due to shift)
        assert labels.isna().sum() == 100

    def test_atr_computation_fallback(self):
        """When atr_14 column is missing, it should be computed."""
        rng = np.random.RandomState(42)
        n = 200
        close = np.cumsum(rng.randn(n)) + 2000
        high = close + np.abs(rng.randn(n)) * 2
        low = close - np.abs(rng.randn(n)) * 2

        df = pd.DataFrame({
            "close": close,
            "high": high,
            "low": low,
        })

        gen = LabelGenerator(horizon_bars=12, threshold_atr_mult=0.5)
        labels = gen.generate(df)

        # Should produce some valid labels
        assert labels.notna().sum() > 0


# ---------- FeatureScaler Tests ----------


class TestFeatureScaler:
    def test_fit_transform(self):
        X = np.random.randn(100, 10)
        scaler = FeatureScaler()
        X_scaled = scaler.fit_transform(X)

        assert X_scaled.shape == X.shape
        # Scaled data should have ~zero mean, ~unit std
        assert np.allclose(X_scaled.mean(axis=0), 0.0, atol=1e-10)
        assert np.allclose(X_scaled.std(axis=0, ddof=0), 1.0, atol=1e-10)

    def test_transform_requires_fit(self):
        scaler = FeatureScaler()
        with pytest.raises(RuntimeError, match="not been fitted"):
            scaler.transform(np.random.randn(10, 5))

    def test_save_load_roundtrip(self):
        X = np.random.randn(100, 10)
        scaler = FeatureScaler()
        X_scaled = scaler.fit_transform(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scaler.npz"
            scaler.save(path)

            scaler2 = FeatureScaler()
            scaler2.load(path)

            X_scaled2 = scaler2.transform(X)
            assert np.allclose(X_scaled, X_scaled2)

    def test_dataframe_input(self):
        df = pd.DataFrame(np.random.randn(50, 5), columns=list("abcde"))
        scaler = FeatureScaler()
        result = scaler.fit_transform(df)
        assert isinstance(result, np.ndarray)
        assert result.shape == (50, 5)


# ---------- XGBoost Model Tests ----------


class TestXGBoostSignalModel:
    def test_fit_and_predict(self):
        X, y = _make_synthetic_data(n_samples=300, n_features=21)
        model = XGBoostSignalModel(n_estimators=50, max_depth=3)

        metrics = model.fit(X, y)
        assert model.is_fitted
        assert "n_samples" in metrics

        pred = model.predict(X[0])
        assert isinstance(pred, ModelPrediction)
        assert 0 <= pred.prob_long <= 1
        assert 0 <= pred.prob_short <= 1
        assert pred.prob_long + pred.prob_short == pytest.approx(1.0, abs=1e-5)

    def test_predict_batch(self):
        X, y = _make_synthetic_data(n_samples=300, n_features=21)
        model = XGBoostSignalModel(n_estimators=50, max_depth=3)
        model.fit(X, y)

        probs = model.predict_batch(X[:10])
        assert probs.shape == (10, 2)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_shap_values(self):
        X, y = _make_synthetic_data(n_samples=300, n_features=21)
        model = XGBoostSignalModel(n_estimators=50, max_depth=3)
        model.fit(X, y)

        shap_vals = model.get_shap_values(X[0], top_k=5)
        assert isinstance(shap_vals, dict)
        assert len(shap_vals) <= 5

    def test_save_load_roundtrip(self):
        X, y = _make_synthetic_data(n_samples=300, n_features=21)
        model = XGBoostSignalModel(n_estimators=50, max_depth=3)
        model.fit(X, y)

        pred_before = model.predict(X[0])

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(Path(tmpdir))

            model2 = XGBoostSignalModel()
            model2.load(Path(tmpdir))

            pred_after = model2.predict(X[0])
            assert pred_after.prob_long == pytest.approx(pred_before.prob_long, abs=1e-5)

    def test_predict_unfitted_raises(self):
        model = XGBoostSignalModel()
        with pytest.raises(Exception):
            model.predict(np.zeros(21))

    def test_validation_data(self):
        X, y = _make_synthetic_data(n_samples=300, n_features=21)
        model = XGBoostSignalModel(n_estimators=50, max_depth=3)
        metrics = model.fit(X[:200], y[:200], validation_data=(X[200:], y[200:]))
        assert "val_accuracy" in metrics


# ---------- CNN-LSTM Model Tests ----------


class TestCNNLSTMSignalModel:
    def test_fit_and_predict_sequence(self):
        """Test fitting on 3D sequence data and predicting."""
        X, y = _make_synthetic_data(n_samples=300, n_features=21)
        # Build sequences
        lookback = 16  # Smaller for test speed
        n_seq = len(X) - lookback + 1
        X_seq = np.zeros((n_seq, lookback, 21))
        for i in range(n_seq):
            X_seq[i] = X[i : i + lookback]
        y_seq = y[lookback - 1 :]

        model = CNNLSTMSignalModel(
            lookback=lookback, max_epochs=3, patience=2, batch_size=32
        )
        metrics = model.fit(X_seq, y_seq)

        assert model.is_fitted
        assert "n_samples" in metrics

        # Predict with a full sequence
        pred = model.predict(X_seq[0])
        assert isinstance(pred, ModelPrediction)
        assert 0 <= pred.prob_long <= 1

    def test_predict_batch(self):
        lookback = 16
        X, y = _make_synthetic_data(n_samples=200, n_features=21)
        n_seq = len(X) - lookback + 1
        X_seq = np.zeros((n_seq, lookback, 21))
        for i in range(n_seq):
            X_seq[i] = X[i : i + lookback]
        y_seq = y[lookback - 1 :]

        model = CNNLSTMSignalModel(lookback=lookback, max_epochs=3, patience=2)
        model.fit(X_seq, y_seq)

        probs = model.predict_batch(X_seq[:10])
        assert probs.shape == (10, 2)

    def test_streaming_predict_with_history(self):
        """Test single-step streaming prediction with history buffer."""
        lookback = 8
        X, y = _make_synthetic_data(n_samples=100, n_features=21)
        n_seq = len(X) - lookback + 1
        X_seq = np.zeros((n_seq, lookback, 21))
        for i in range(n_seq):
            X_seq[i] = X[i : i + lookback]
        y_seq = y[lookback - 1 :]

        model = CNNLSTMSignalModel(lookback=lookback, max_epochs=3, patience=2)
        model.fit(X_seq, y_seq)

        # Feed features one at a time
        model.clear_history()
        for i in range(lookback - 1):
            pred = model.predict(X[i])
            # Not enough history yet — should return neutral
            assert pred.prob_long == pytest.approx(0.5)
            assert pred.prob_short == pytest.approx(0.5)

        # After feeding enough history, should get a real prediction
        pred = model.predict(X[lookback - 1])
        assert isinstance(pred, ModelPrediction)

    def test_save_load_roundtrip(self):
        lookback = 8
        X, y = _make_synthetic_data(n_samples=100, n_features=21)
        n_seq = len(X) - lookback + 1
        X_seq = np.zeros((n_seq, lookback, 21))
        for i in range(n_seq):
            X_seq[i] = X[i : i + lookback]
        y_seq = y[lookback - 1 :]

        model = CNNLSTMSignalModel(lookback=lookback, max_epochs=3, patience=2)
        model.fit(X_seq, y_seq)
        pred_before = model.predict(X_seq[0])

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(Path(tmpdir))

            model2 = CNNLSTMSignalModel()
            model2.load(Path(tmpdir))
            pred_after = model2.predict(X_seq[0])

            assert pred_after.prob_long == pytest.approx(pred_before.prob_long, abs=1e-4)

    def test_3d_input_required(self):
        model = CNNLSTMSignalModel()
        with pytest.raises(ValueError, match="3D input"):
            model.fit(np.zeros((100, 21)), np.zeros(100))


# ---------- Stacking Ensemble Tests ----------


class TestStackingEnsemble:
    def test_oos_only_meta_learner(self):
        """Verify meta-learner trains on OOS predictions only."""
        X, y = _make_synthetic_data(n_samples=500, n_features=21)

        ensemble = StackingEnsemble(
            xgb_model=XGBoostSignalModel(n_estimators=20, max_depth=3),
            lstm_model=CNNLSTMSignalModel(lookback=8, max_epochs=3, patience=2),
        )
        metrics = ensemble.fit(X, y)

        assert ensemble.is_fitted
        assert "meta_accuracy" in metrics
        assert "meta_n_samples" in metrics
        # Meta-learner should have been trained on ~30% of data (minus lookback)
        assert metrics["meta_n_samples"] < len(y) * 0.5

    def test_predict(self):
        X, y = _make_synthetic_data(n_samples=500, n_features=21)

        ensemble = StackingEnsemble(
            xgb_model=XGBoostSignalModel(n_estimators=20, max_depth=3),
            lstm_model=CNNLSTMSignalModel(lookback=8, max_epochs=3, patience=2),
        )
        ensemble.fit(X, y)

        pred = ensemble.predict(X[0])
        assert isinstance(pred, ModelPrediction)
        assert 0 <= pred.prob_long <= 1

    def test_save_load_roundtrip(self):
        X, y = _make_synthetic_data(n_samples=500, n_features=21)

        ensemble = StackingEnsemble(
            xgb_model=XGBoostSignalModel(n_estimators=20, max_depth=3),
            lstm_model=CNNLSTMSignalModel(lookback=8, max_epochs=3, patience=2),
        )
        ensemble.fit(X, y)
        pred_before = ensemble.predict(X[0])

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "ensemble"
            ensemble.save(save_path)

            ensemble2 = StackingEnsemble(
                xgb_model=XGBoostSignalModel(),
                lstm_model=CNNLSTMSignalModel(),
            )
            ensemble2.load(save_path)

            pred_after = ensemble2.predict(X[0])
            assert pred_after.prob_long == pytest.approx(
                pred_before.prob_long, abs=0.01
            )


# ---------- WalkForwardSplitter Tests ----------


class TestWalkForwardSplitter:
    def test_no_temporal_leakage(self):
        """Verify train indices always come before test indices."""
        splitter = WalkForwardSplitter(n_splits=5, purge_gap=1)
        splits = splitter.split(1000)

        for train_idx, test_idx in splits:
            # All train indices must be < all test indices (with purge gap)
            assert train_idx.max() < test_idx.min()

    def test_expanding_window(self):
        """Training window should expand with each fold."""
        splitter = WalkForwardSplitter(n_splits=5)
        splits = splitter.split(1000)

        train_sizes = [len(t) for t, _ in splits]
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] > train_sizes[i - 1]

    def test_purge_gap(self):
        """Verify purge gap between train and test."""
        splitter = WalkForwardSplitter(n_splits=3, purge_gap=5)
        splits = splitter.split(500)

        for train_idx, test_idx in splits:
            gap = test_idx.min() - train_idx.max()
            assert gap >= 5

    def test_insufficient_data_raises(self):
        splitter = WalkForwardSplitter(n_splits=5)
        with pytest.raises(Exception):
            splitter.split(10)

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            WalkForwardSplitter(n_splits=1)
        with pytest.raises(ValueError):
            WalkForwardSplitter(train_ratio=0.3)


# ---------- CPCVSplitter Tests ----------


class TestCPCVSplitter:
    def test_split_produces_combinations(self):
        splitter = CPCVSplitter(n_groups=5, n_test_groups=2)
        splits = splitter.split(500)

        # C(5, 2) = 10 combinations
        assert len(splits) == 10

    def test_no_overlap_train_test(self):
        splitter = CPCVSplitter(n_groups=5, n_test_groups=2, purge_gap=2)
        splits = splitter.split(500)

        for train_idx, test_idx in splits:
            overlap = set(train_idx) & set(test_idx)
            assert len(overlap) == 0

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            CPCVSplitter(n_groups=3, n_test_groups=3)


# ---------- Metrics Tests ----------


class TestMetrics:
    def test_compute_metrics_basic(self):
        rng = np.random.RandomState(42)
        n = 200
        predictions = rng.randint(0, 2, size=n)
        labels = rng.randint(0, 2, size=n)
        returns = rng.randn(n) * 0.001

        metrics = compute_metrics(predictions, labels, returns)
        assert isinstance(metrics, SignalMetrics)
        assert metrics.trade_count == n
        assert 0 <= metrics.win_rate <= 1
        assert 0 <= metrics.accuracy <= 1

    def test_perfect_predictions(self):
        n = 200
        labels = np.array([1, 0] * (n // 2))
        predictions = labels.copy()
        returns = np.where(labels == 1, 0.001, -0.001)

        metrics = compute_metrics(predictions, labels, returns)
        assert metrics.accuracy == 1.0
        assert metrics.win_rate == 1.0

    def test_empty_predictions(self):
        metrics = compute_metrics(np.array([]), np.array([]), np.array([]))
        assert metrics.trade_count == 0
        assert metrics.sharpe_ratio == 0.0

    def test_meets_minimum(self):
        m = SignalMetrics(
            sharpe_ratio=1.2,
            sortino_ratio=1.8,
            profit_factor=1.5,
            win_rate=0.45,
            max_drawdown=0.15,
            trade_count=150,
            total_return=0.1,
            mean_return=0.001,
            std_return=0.01,
            accuracy=0.55,
        )
        assert m.meets_minimum()

    def test_fails_minimum(self):
        m = SignalMetrics(
            sharpe_ratio=0.5,
            sortino_ratio=1.0,
            profit_factor=1.0,
            win_rate=0.35,
            max_drawdown=0.25,
            trade_count=50,
            total_return=0.01,
            mean_return=0.0001,
            std_return=0.01,
            accuracy=0.45,
        )
        assert not m.meets_minimum()


# ---------- Exceptions Tests ----------


class TestExceptions:
    def test_signal_generation_error(self):
        from fxer.core.exceptions import SignalGenerationError

        err = SignalGenerationError("test error", model="xgboost", symbol="XAUUSD")
        assert err.model == "xgboost"
        assert err.symbol == "XAUUSD"
        assert "test error" in str(err)

    def test_model_not_fitted_error(self):
        from fxer.core.exceptions import ModelNotFittedError

        err = ModelNotFittedError("not fitted", model_name="cnn_lstm")
        assert err.model_name == "cnn_lstm"

    def test_insufficient_data_error(self):
        from fxer.core.exceptions import InsufficientDataError

        err = InsufficientDataError("too few samples", required=100, available=50)
        assert err.required == 100
        assert err.available == 50
