"""CNN-Bi-LSTM signal model with attention mechanism."""

import json
import logging
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fxer.core.exceptions import ModelNotFittedError
from fxer.signals.base import BaseSignalModel
from fxer.signals.types import ModelPrediction

logger = logging.getLogger(__name__)


class Attention(nn.Module):
    """Simple additive attention over LSTM hidden states."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_output: torch.Tensor) -> torch.Tensor:
        """Apply attention and return weighted sum.

        Args:
            lstm_output: (batch, seq_len, hidden_size)

        Returns:
            Context vector: (batch, hidden_size)
        """
        weights = torch.softmax(self.attn(lstm_output), dim=1)  # (batch, seq, 1)
        return (weights * lstm_output).sum(dim=1)  # (batch, hidden)


class CNNBiLSTMNet(nn.Module):
    """CNN-Bi-LSTM network with attention.

    Architecture:
        Conv1D(64, k=3) -> BatchNorm -> ReLU -> Dropout ->
        Bi-LSTM(128) -> Bi-LSTM(64) -> Attention ->
        FC(64) -> FC(2)
    """

    def __init__(
        self,
        n_features: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        # CNN block
        self.conv = nn.Conv1d(
            in_channels=n_features,
            out_channels=64,
            kernel_size=3,
            padding=1,
        )
        self.bn = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.dropout_cnn = nn.Dropout(dropout)

        # Bi-LSTM layers
        self.lstm1 = nn.LSTM(
            input_size=64,
            hidden_size=128,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout_lstm1 = nn.Dropout(dropout)

        self.lstm2 = nn.LSTM(
            input_size=256,  # 128 * 2 (bidirectional)
            hidden_size=64,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout_lstm2 = nn.Dropout(dropout)

        # Attention over final LSTM output
        self.attention = Attention(128)  # 64 * 2 (bidirectional)

        # Classifier
        self.fc1 = nn.Linear(128, 64)
        self.fc_relu = nn.ReLU()
        self.fc_dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, lookback, n_features)

        Returns:
            Logits: (batch, 2)
        """
        # Conv1D expects (batch, channels, seq_len)
        x = x.permute(0, 2, 1)
        x = self.dropout_cnn(self.relu(self.bn(self.conv(x))))
        # Back to (batch, seq_len, channels)
        x = x.permute(0, 2, 1)

        # Bi-LSTM layers
        x, _ = self.lstm1(x)
        x = self.dropout_lstm1(x)
        x, _ = self.lstm2(x)
        x = self.dropout_lstm2(x)

        # Attention pooling
        x = self.attention(x)

        # Classifier
        x = self.fc_dropout(self.fc_relu(self.fc1(x)))
        return self.fc2(x)


class CNNLSTMSignalModel(BaseSignalModel):
    """CNN-Bi-LSTM signal model for temporal pattern recognition.

    Input: (batch, lookback=48, n_features) — requires scaled features.
    Maintains a history buffer for streaming prediction.
    """

    def __init__(
        self,
        lookback: int = 48,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        max_epochs: int = 100,
        patience: int = 15,
    ) -> None:
        self._lookback = lookback
        self._dropout = dropout
        self._lr = learning_rate
        self._batch_size = batch_size
        self._max_epochs = max_epochs
        self._patience = patience

        self._model: CNNBiLSTMNet | None = None
        self._n_features: int | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._history: deque[np.ndarray] = deque(maxlen=lookback)

    @property
    def name(self) -> str:
        return "cnn_lstm"

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> dict[str, Any]:
        """Train the CNN-Bi-LSTM model.

        Args:
            X: Training sequences (n_samples, lookback, n_features).
            y: Training labels (n_samples,).
            validation_data: Optional (X_val, y_val) for early stopping.

        Returns:
            Training metrics.
        """
        if X.ndim != 3:
            raise ValueError(
                f"CNN-LSTM expects 3D input (n, lookback, features), got shape {X.shape}"
            )

        self._n_features = X.shape[2]
        self._model = CNNBiLSTMNet(
            n_features=self._n_features,
            dropout=self._dropout,
        ).to(self._device)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self._lr)
        criterion = nn.CrossEntropyLoss()

        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X), torch.LongTensor(y.astype(int))
        )
        train_loader = DataLoader(
            train_dataset, batch_size=self._batch_size, shuffle=True
        )

        val_loader = None
        if validation_data is not None:
            val_dataset = TensorDataset(
                torch.FloatTensor(validation_data[0]),
                torch.LongTensor(validation_data[1].astype(int)),
            )
            val_loader = DataLoader(
                val_dataset, batch_size=self._batch_size, shuffle=False
            )

        # Training loop with early stopping
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(self._max_epochs):
            self._model.train()
            train_loss = 0.0
            n_batches = 0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self._device)
                batch_y = batch_y.to(self._device)

                optimizer.zero_grad()
                logits = self._model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                n_batches += 1

            avg_train_loss = train_loss / max(n_batches, 1)

            # Validation
            if val_loader is not None:
                val_loss = self._evaluate(val_loader, criterion)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {
                        k: v.cpu().clone() for k, v in self._model.state_dict().items()
                    }
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self._patience:
                    logger.info(
                        "Early stopping at epoch %d (val_loss=%.4f)",
                        epoch + 1,
                        best_val_loss,
                    )
                    break

                if (epoch + 1) % 10 == 0:
                    logger.debug(
                        "Epoch %d: train_loss=%.4f, val_loss=%.4f",
                        epoch + 1,
                        avg_train_loss,
                        val_loss,
                    )

        # Restore best model if we did validation
        if best_state is not None:
            self._model.load_state_dict(best_state)

        self._model.eval()

        metrics: dict[str, Any] = {
            "n_samples": len(y),
            "n_features": self._n_features,
            "epochs_trained": epoch + 1,
        }
        if val_loader is not None:
            metrics["best_val_loss"] = best_val_loss

        logger.info(
            "CNN-LSTM trained: %d samples, %d epochs, %d features",
            len(y),
            epoch + 1,
            self._n_features,
        )
        return metrics

    def predict(self, features: np.ndarray) -> ModelPrediction:
        """Generate prediction from a single feature array.

        For streaming use, call push_features() first to maintain history,
        then call predict_from_history(). For direct use with a pre-built
        sequence, reshape features to (lookback, n_features) before calling.

        Args:
            features: Either 1-D (single timestep, added to history buffer)
                      or 2-D (lookback, n_features) sequence.

        Returns:
            ModelPrediction with class probabilities.
        """
        if self._model is None:
            raise ModelNotFittedError(
                "CNN-LSTM model has not been fitted", model_name=self.name
            )

        if features.ndim == 1:
            # Single timestep: use history buffer
            self._history.append(features.copy())
            if len(self._history) < self._lookback:
                # Not enough history yet — return neutral prediction
                return ModelPrediction(
                    prob_long=0.5, prob_short=0.5, raw_output=0.5
                )
            seq = np.array(list(self._history), dtype=np.float32)
        elif features.ndim == 2:
            seq = features.astype(np.float32)
        else:
            raise ValueError(f"Expected 1-D or 2-D input, got {features.ndim}-D")

        X = torch.FloatTensor(seq).unsqueeze(0).to(self._device)

        self._model.eval()
        with torch.no_grad():
            logits = self._model(X)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        return ModelPrediction(
            prob_long=float(probs[1]),
            prob_short=float(probs[0]),
            raw_output=float(probs[1]),
        )

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Generate batch predictions.

        Args:
            X: 3-D array (n_samples, lookback, n_features).

        Returns:
            Array of shape (n_samples, 2) with [prob_short, prob_long].
        """
        if self._model is None:
            raise ModelNotFittedError(
                "CNN-LSTM model has not been fitted", model_name=self.name
            )

        self._model.eval()
        dataset = TensorDataset(torch.FloatTensor(X))
        loader = DataLoader(dataset, batch_size=self._batch_size, shuffle=False)

        all_probs: list[np.ndarray] = []
        with torch.no_grad():
            for (batch_X,) in loader:
                batch_X = batch_X.to(self._device)
                logits = self._model(batch_X)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs)

        return np.concatenate(all_probs, axis=0)

    def clear_history(self) -> None:
        """Clear the streaming history buffer."""
        self._history.clear()

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

        torch.save(self._model.state_dict(), path / "cnn_lstm_state.pt")

        params = {
            "lookback": self._lookback,
            "n_features": self._n_features,
            "dropout": self._dropout,
            "learning_rate": self._lr,
            "batch_size": self._batch_size,
            "max_epochs": self._max_epochs,
            "patience": self._patience,
        }
        with open(path / "cnn_lstm_params.json", "w") as f:
            json.dump(params, f, indent=2)

        logger.info("Saved CNN-LSTM model to %s", path)

    def load(self, path: Path) -> None:
        """Load model from disk.

        Args:
            path: Directory containing saved model artifacts.
        """
        path = Path(path)

        with open(path / "cnn_lstm_params.json") as f:
            params = json.load(f)

        self._lookback = params["lookback"]
        self._n_features = params["n_features"]
        self._dropout = params["dropout"]
        self._lr = params.get("learning_rate", self._lr)
        self._batch_size = params.get("batch_size", self._batch_size)

        self._model = CNNBiLSTMNet(
            n_features=self._n_features,
            dropout=self._dropout,
        ).to(self._device)

        state = torch.load(
            path / "cnn_lstm_state.pt",
            map_location=self._device,
            weights_only=True,
        )
        self._model.load_state_dict(state)
        self._model.eval()

        self._history = deque(maxlen=self._lookback)
        logger.info("Loaded CNN-LSTM model from %s", path)

    def _evaluate(
        self,
        loader: DataLoader,
        criterion: nn.Module,
    ) -> float:
        """Evaluate model on a data loader."""
        assert self._model is not None
        self._model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self._device)
                batch_y = batch_y.to(self._device)
                logits = self._model(batch_X)
                loss = criterion(logits, batch_y)
                total_loss += loss.item()
                n_batches += 1

        return total_loss / max(n_batches, 1)
