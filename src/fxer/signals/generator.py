"""Signal generator service with EventBus integration."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from fxer.config.settings import Settings, settings as default_settings
from fxer.core.events import FeatureVector
from fxer.core.exceptions import SignalGenerationError
from fxer.messaging.bus import EventBus
from fxer.signals.base import FEATURE_COLUMNS, feature_vector_to_array
from fxer.signals.models.ensemble import StackingEnsemble
from fxer.signals.types import Direction, HoldingPeriod, TradeSignal

logger = logging.getLogger(__name__)


class SignalGenerator:
    """Real-time signal generator that subscribes to feature events.

    Integrates with the EventBus (messaging/bus.py):
    - Subscribes to `features.*` topic prefix
    - Loads ensemble + scaler from disk
    - On each FeatureEvent:
        1. Converts FeatureVector → numpy → scale (for LSTM)
        2. Generates TradeSignal via ensemble
        3. Publishes to `signal.{symbol}` topic
    """

    def __init__(
        self,
        event_bus: EventBus,
        settings: Settings | None = None,
    ) -> None:
        self._bus = event_bus
        self._settings = settings or default_settings
        self._ensemble = StackingEnsemble(settings=self._settings)
        self._model_dir = Path(self._settings.signal_model_dir)
        self._running = False

    @property
    def is_loaded(self) -> bool:
        """Whether the ensemble model is loaded and ready."""
        return self._ensemble.is_fitted

    def load_model(self, symbol: str, timeframe: str = "5m") -> None:
        """Load a trained ensemble from disk.

        Args:
            symbol: Trading symbol (e.g. "XAUUSD").
            timeframe: Timeframe string (e.g. "5m").
        """
        model_path = self._model_dir / symbol.lower() / timeframe
        if not model_path.exists():
            raise SignalGenerationError(
                f"No model found at {model_path}",
                model="ensemble",
                symbol=symbol,
            )

        self._ensemble.load(model_path)
        logger.info("Loaded ensemble model for %s/%s from %s", symbol, timeframe, model_path)

    async def start(self) -> None:
        """Start the signal generator (subscribe to feature events)."""
        if self._running:
            return

        if not self._ensemble.is_fitted:
            raise SignalGenerationError(
                "Cannot start: ensemble model not loaded. Call load_model() first.",
                model="ensemble",
            )

        await self._bus.subscribe("features.", self._on_feature_event)
        self._running = True
        logger.info("SignalGenerator started, subscribed to features.*")

    async def stop(self) -> None:
        """Stop the signal generator."""
        if not self._running:
            return

        await self._bus.unsubscribe("features.")
        self._running = False
        logger.info("SignalGenerator stopped")

    async def _on_feature_event(
        self, topic: str, event_dict: dict[str, Any]
    ) -> None:
        """Handle incoming feature events from the EventBus.

        Args:
            topic: Event topic (e.g. "features.xauusd").
            event_dict: Serialized FeatureVector dict.
        """
        try:
            fv = FeatureVector.from_dict(event_dict)

            if not fv.warmup_complete:
                return

            signal = self._generate_signal(fv)

            if signal.direction != Direction.NEUTRAL:
                signal_topic = f"signal.{signal.symbol.lower()}"
                await self._bus.publish(signal_topic, signal.to_dict())
                logger.info(
                    "Published %s signal for %s: confidence=%.3f, horizon=%s",
                    signal.direction.value,
                    signal.symbol,
                    signal.confidence,
                    signal.predicted_horizon.value,
                )
            else:
                logger.debug(
                    "Neutral signal for %s (prob=%.3f), not publishing",
                    fv.symbol,
                    signal.ensemble_prob or 0.0,
                )

        except Exception as exc:
            logger.error("Error generating signal from %s: %s", topic, exc)

    def _generate_signal(self, fv: FeatureVector) -> TradeSignal:
        """Generate a trade signal from a feature vector.

        Args:
            fv: FeatureVector from the feature pipeline.

        Returns:
            TradeSignal with direction, confidence, and explanations.
        """
        features = feature_vector_to_array(fv)

        horizon = HoldingPeriod.from_bars(self._settings.signal_horizon_bars)

        return self._ensemble.generate_signal(
            features=features,
            symbol=fv.symbol,
            timestamp=fv.timestamp,
            horizon=horizon,
        )
