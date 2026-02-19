"""IBKR real-time bar streaming.

Streams live bars from Interactive Brokers and publishes them to the EventBus.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

from ib_insync import RealTimeBarList

from fxer.core.types import RawBar
from fxer.data.loaders.ibkr_client import IBKRClient
from fxer.messaging.bus import EventBus

logger = logging.getLogger(__name__)


@dataclass
class IBKRStreamer:
    """Stream real-time bars from IBKR to EventBus.

    Subscribes to real-time 5-second bars from IBKR and publishes
    them to the EventBus for downstream processing.

    Note: IBKR real-time bars are always 5-second bars. Higher timeframes
    need to be aggregated from these base bars.

    Usage::

        async with IBKRClient(settings) as client:
            bus = EventBus()
            await bus.start()

            streamer = IBKRStreamer(client, bus)
            await streamer.subscribe("XAUUSD")
            await streamer.subscribe("EURUSD")

            # Stream runs until cancelled
            try:
                await asyncio.Event().wait()
            finally:
                await streamer.unsubscribe_all()
                await bus.stop()
    """

    client: IBKRClient
    bus: EventBus
    what_to_show: str = "MIDPOINT"
    use_rth: bool = False

    _subscriptions: dict[str, RealTimeBarList] = field(
        default_factory=dict, init=False, repr=False
    )
    _bar_counts: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    async def subscribe(self, symbol: str) -> None:
        """Subscribe to real-time bars for a symbol.

        Args:
            symbol: Symbol name (e.g., "XAUUSD", "EURUSD").
        """
        if symbol in self._subscriptions:
            logger.warning("Already subscribed to %s", symbol)
            return

        contract = self.client.make_contract(symbol)

        # Qualify the contract
        try:
            contract = await self.client.qualify_contract(contract)
        except ValueError as exc:
            logger.error("Failed to qualify contract for %s: %s", symbol, exc)
            return

        logger.info("Subscribing to real-time bars for %s", symbol)

        # Request real-time bars (always 5-second bars)
        bars = self.client.ib.reqRealTimeBars(
            contract,
            barSize=5,  # Always 5-second bars from IBKR
            whatToShow=self.what_to_show,
            useRTH=self.use_rth,
        )

        # Register callback for new bars
        bars.updateEvent += self._make_bar_handler(symbol)

        self._subscriptions[symbol] = bars
        self._bar_counts[symbol] = 0
        logger.info("Subscribed to %s real-time bars", symbol)

    async def unsubscribe(self, symbol: str) -> None:
        """Unsubscribe from real-time bars for a symbol.

        Args:
            symbol: Symbol name to unsubscribe from.
        """
        if symbol not in self._subscriptions:
            logger.warning("Not subscribed to %s", symbol)
            return

        bars = self._subscriptions.pop(symbol)
        self.client.ib.cancelRealTimeBars(bars)
        del self._bar_counts[symbol]
        logger.info("Unsubscribed from %s", symbol)

    async def unsubscribe_all(self) -> None:
        """Unsubscribe from all symbols."""
        symbols = list(self._subscriptions.keys())
        for symbol in symbols:
            await self.unsubscribe(symbol)

    def _make_bar_handler(self, symbol: str) -> Callable[..., None]:
        """Create a bar update handler for a specific symbol.

        Args:
            symbol: Symbol name for the handler.

        Returns:
            Callback function for bar updates.
        """

        def on_bar_update(bars: RealTimeBarList, has_new_bar: bool) -> None:
            """Handle new bar from IBKR."""
            if not has_new_bar or not bars:
                return

            bar = bars[-1]
            self._bar_counts[symbol] = self._bar_counts.get(symbol, 0) + 1

            # Convert to RawBar format
            raw_bar = self._to_raw_bar(bar, symbol)

            # Create event dict for publishing
            event = {
                "symbol": symbol,
                "timeframe": "5s",  # Real-time bars are 5-second
                "timestamp": raw_bar.timestamp.isoformat(),
                "open": str(raw_bar.open),
                "high": str(raw_bar.high),
                "low": str(raw_bar.low),
                "close": str(raw_bar.close),
                "volume": str(raw_bar.volume) if raw_bar.volume else None,
            }

            # Publish to EventBus (fire and forget)
            topic = f"bar.{symbol.lower()}.5s"
            asyncio.create_task(self._publish_bar(topic, event))

            # Log periodically
            count = self._bar_counts[symbol]
            if count % 12 == 0:  # Every minute (12 * 5s = 60s)
                logger.debug(
                    "%s: Received bar %d at %s, close=%s",
                    symbol,
                    count,
                    raw_bar.timestamp,
                    raw_bar.close,
                )

        return on_bar_update

    async def _publish_bar(self, topic: str, event: dict[str, Any]) -> None:
        """Publish a bar event to the EventBus.

        Args:
            topic: Topic to publish to.
            event: Event data to publish.
        """
        try:
            await self.bus.publish(topic, event)
        except Exception as exc:
            logger.error("Failed to publish bar event to %s: %s", topic, exc)

    def _to_raw_bar(self, ibkr_bar: object, symbol: str) -> RawBar:
        """Convert an IBKR RealTimeBar to RawBar.

        Args:
            ibkr_bar: IBKR RealTimeBar object.
            symbol: Symbol name.

        Returns:
            RawBar instance.
        """
        # RealTimeBar has: time (Unix timestamp), open, high, low, close, volume, wap, count
        timestamp = datetime.fromtimestamp(ibkr_bar.time, tz=datetime.UTC)  # type: ignore[attr-defined]

        return RawBar(
            timestamp=timestamp,
            open=Decimal(str(ibkr_bar.open_)),  # type: ignore[attr-defined]
            high=Decimal(str(ibkr_bar.high)),  # type: ignore[attr-defined]
            low=Decimal(str(ibkr_bar.low)),  # type: ignore[attr-defined]
            close=Decimal(str(ibkr_bar.close)),  # type: ignore[attr-defined]
            volume=Decimal(str(ibkr_bar.volume)) if ibkr_bar.volume else None,  # type: ignore[attr-defined]
            symbol=symbol,
            timeframe="5s",
        )

    @property
    def subscribed_symbols(self) -> list[str]:
        """Get list of currently subscribed symbols."""
        return list(self._subscriptions.keys())

    def get_bar_count(self, symbol: str) -> int:
        """Get the number of bars received for a symbol.

        Args:
            symbol: Symbol name.

        Returns:
            Number of bars received, or 0 if not subscribed.
        """
        return self._bar_counts.get(symbol, 0)
