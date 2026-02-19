"""ZeroMQ-based event bus for inter-component communication.

Uses PUB/SUB pattern with msgpack serialization for efficient,
topic-based routing of BarEvent and FeatureEvent messages.
"""

import asyncio
import logging
from collections.abc import Callable
from typing import Any

import msgpack
import zmq
import zmq.asyncio

from fxer.config.settings import Settings, settings as default_settings

logger = logging.getLogger(__name__)


def _serialize(event_dict: dict[str, Any]) -> bytes:
    """Serialize an event dict to msgpack bytes."""
    return msgpack.packb(event_dict, use_bin_type=True)


def _deserialize(data: bytes) -> dict[str, Any]:
    """Deserialize msgpack bytes to an event dict."""
    return msgpack.unpackb(data, raw=False)


class EventBus:
    """ZeroMQ PUB/SUB event bus.

    Publishers send events to topics like:
        bar.xauusd.5m
        bar.xauusd.1h
        features.xauusd

    Subscribers register callbacks for topic prefixes using ZeroMQ's
    native subscription filtering.

    Usage::

        bus = EventBus()
        await bus.start()

        # Subscribe to all XAUUSD bar events
        await bus.subscribe("bar.xauusd", callback)

        # Publish a specific bar event
        await bus.publish("bar.xauusd.5m", bar_event.bar.to_dict())

        await bus.stop()
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or default_settings
        self._ctx: zmq.asyncio.Context | None = None
        self._pub_socket: zmq.asyncio.Socket | None = None
        self._sub_socket: zmq.asyncio.Socket | None = None
        self._callbacks: dict[str, list[Callable[[str, dict[str, Any]], Any]]] = {}
        self._listener_task: asyncio.Task[None] | None = None
        self._running = False

    @property
    def pub_address(self) -> str:
        return f"tcp://127.0.0.1:{self._settings.zmq_pub_port}"

    @property
    def sub_address(self) -> str:
        return f"tcp://127.0.0.1:{self._settings.zmq_sub_port}"

    async def start(self) -> None:
        """Start the event bus, binding PUB and connecting SUB sockets."""
        if self._running:
            return

        self._ctx = zmq.asyncio.Context()

        self._pub_socket = self._ctx.socket(zmq.PUB)
        self._pub_socket.bind(self.pub_address)

        self._sub_socket = self._ctx.socket(zmq.SUB)
        self._sub_socket.connect(self.pub_address)

        self._running = True
        self._listener_task = asyncio.create_task(self._listen())

        # Brief pause to allow ZeroMQ socket connections to settle
        # (the "slow joiner" problem).
        await asyncio.sleep(0.05)
        logger.info("EventBus started on %s", self.pub_address)

    async def stop(self) -> None:
        """Stop the event bus and clean up resources."""
        if not self._running:
            return

        self._running = False

        if self._listener_task is not None:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
            self._listener_task = None

        if self._sub_socket is not None:
            self._sub_socket.close(linger=0)
            self._sub_socket = None

        if self._pub_socket is not None:
            self._pub_socket.close(linger=0)
            self._pub_socket = None

        if self._ctx is not None:
            self._ctx.term()
            self._ctx = None

        logger.info("EventBus stopped")

    async def publish(self, topic: str, event: dict[str, Any]) -> None:
        """Publish an event to a topic.

        Args:
            topic: Dot-separated topic string (e.g. "bar.xauusd.5m").
            event: Event data as a dictionary (use event.to_dict()).
        """
        if self._pub_socket is None:
            raise RuntimeError("EventBus is not started")

        payload = _serialize(event)
        await self._pub_socket.send_multipart([topic.encode("utf-8"), payload])

    async def subscribe(
        self,
        topic: str,
        callback: Callable[[str, dict[str, Any]], Any],
    ) -> None:
        """Subscribe to events matching a topic prefix.

        The callback receives ``(topic, event_dict)`` and may be a
        coroutine or a regular function.

        Args:
            topic: Topic prefix to subscribe to. Use "" to receive all events.
            callback: Function called for each matching event.
        """
        if self._sub_socket is None:
            raise RuntimeError("EventBus is not started")

        if topic not in self._callbacks:
            self._callbacks[topic] = []
            self._sub_socket.subscribe(topic.encode("utf-8"))

        self._callbacks[topic].append(callback)

    async def unsubscribe(self, topic: str) -> None:
        """Remove all callbacks for a topic prefix."""
        if self._sub_socket is None:
            raise RuntimeError("EventBus is not started")

        if topic in self._callbacks:
            del self._callbacks[topic]
            self._sub_socket.unsubscribe(topic.encode("utf-8"))

    async def _listen(self) -> None:
        """Background task that receives messages and dispatches to callbacks."""
        assert self._sub_socket is not None

        while self._running:
            try:
                parts = await self._sub_socket.recv_multipart()
                if len(parts) != 2:
                    logger.warning("Malformed message with %d parts, skipping", len(parts))
                    continue

                topic_bytes, payload = parts
                topic_str = topic_bytes.decode("utf-8")
                event_dict = _deserialize(payload)

                await self._dispatch(topic_str, event_dict)

            except asyncio.CancelledError:
                break
            except zmq.ZMQError as exc:
                if not self._running:
                    break
                logger.error("ZMQ error in listener: %s", exc)
            except Exception:
                logger.exception("Unexpected error in listener")

    async def _dispatch(self, topic: str, event_dict: dict[str, Any]) -> None:
        """Dispatch an event to all matching callbacks."""
        for prefix, callbacks in self._callbacks.items():
            if topic.startswith(prefix):
                for cb in callbacks:
                    try:
                        result = cb(topic, event_dict)
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception:
                        logger.exception(
                            "Error in callback for topic %s (prefix %s)", topic, prefix
                        )
