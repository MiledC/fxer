"""Tests for the ZeroMQ EventBus."""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock

import msgpack
import pytest

from fxer.config.settings import Settings
from fxer.core.events import BarEvent, FeatureEvent, FeatureVector, NormalizedBar
from fxer.core.types import Timeframe
from fxer.messaging.bus import EventBus, _deserialize, _serialize


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bar(
    symbol: str = "XAUUSD",
    timeframe: Timeframe = Timeframe.M5,
) -> NormalizedBar:
    return NormalizedBar(
        symbol=symbol,
        timeframe=timeframe,
        timestamp=datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        open=Decimal("2050.50"),
        high=Decimal("2052.00"),
        low=Decimal("2049.00"),
        close=Decimal("2051.75"),
        volume=Decimal("1500"),
    )


def _make_feature_vector(symbol: str = "XAUUSD") -> FeatureVector:
    return FeatureVector(
        symbol=symbol,
        timestamp=datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        timeframe=Timeframe.M5,
        rsi_14=55.3,
        macd_line=0.25,
        warmup_complete=True,
    )


def _free_port_settings(pub_port: int, sub_port: int) -> Settings:
    """Create Settings with explicit ZMQ ports to avoid collisions."""
    return Settings(
        zmq_pub_port=pub_port,
        zmq_sub_port=sub_port,
    )


# ---------------------------------------------------------------------------
# Serialization tests
# ---------------------------------------------------------------------------

class TestSerialization:
    """Tests for msgpack serialization / deserialization."""

    def test_roundtrip_bar_dict(self) -> None:
        bar = _make_bar()
        original = bar.to_dict()
        packed = _serialize(original)
        restored = _deserialize(packed)
        assert restored == original

    def test_roundtrip_feature_dict(self) -> None:
        fv = _make_feature_vector()
        original = fv.to_dict()
        packed = _serialize(original)
        restored = _deserialize(packed)
        assert restored == original

    def test_serialize_produces_bytes(self) -> None:
        data = {"key": "value", "number": 42}
        result = _serialize(data)
        assert isinstance(result, bytes)

    def test_deserialize_returns_dict(self) -> None:
        data = {"key": "value", "number": 42}
        packed = msgpack.packb(data, use_bin_type=True)
        result = _deserialize(packed)
        assert isinstance(result, dict)
        assert result["key"] == "value"
        assert result["number"] == 42

    def test_none_values_preserved(self) -> None:
        data = {"rsi_14": None, "atr_14": None}
        assert _deserialize(_serialize(data)) == data

    def test_nested_structures(self) -> None:
        data = {"meta": {"source": "csv", "rows": 100}, "values": [1.0, 2.0, 3.0]}
        assert _deserialize(_serialize(data)) == data


# ---------------------------------------------------------------------------
# EventBus lifecycle tests
# ---------------------------------------------------------------------------

class TestEventBusLifecycle:
    """Tests for start/stop and error handling."""

    @pytest.fixture
    def bus(self) -> EventBus:
        return EventBus(settings=_free_port_settings(15555, 15556))

    async def test_start_and_stop(self, bus: EventBus) -> None:
        await bus.start()
        assert bus._running is True
        await bus.stop()
        assert bus._running is False

    async def test_double_start_is_idempotent(self, bus: EventBus) -> None:
        await bus.start()
        await bus.start()  # should not raise
        assert bus._running is True
        await bus.stop()

    async def test_double_stop_is_idempotent(self, bus: EventBus) -> None:
        await bus.start()
        await bus.stop()
        await bus.stop()  # should not raise
        assert bus._running is False

    async def test_publish_before_start_raises(self, bus: EventBus) -> None:
        with pytest.raises(RuntimeError, match="not started"):
            await bus.publish("test.topic", {"data": 1})

    async def test_subscribe_before_start_raises(self, bus: EventBus) -> None:
        with pytest.raises(RuntimeError, match="not started"):
            await bus.subscribe("test.topic", lambda t, e: None)


# ---------------------------------------------------------------------------
# Pub/Sub routing tests
# ---------------------------------------------------------------------------

class TestPubSubRouting:
    """Tests for publish/subscribe message routing."""

    @pytest.fixture
    async def bus(self):
        b = EventBus(settings=_free_port_settings(15557, 15558))
        await b.start()
        yield b
        await b.stop()

    async def test_basic_pubsub(self, bus: EventBus) -> None:
        received: list[tuple[str, dict]] = []
        callback = AsyncMock(side_effect=lambda t, e: received.append((t, e)))

        await bus.subscribe("test", callback)
        await asyncio.sleep(0.05)

        await bus.publish("test.hello", {"msg": "world"})
        await asyncio.sleep(0.1)

        assert len(received) == 1
        assert received[0][0] == "test.hello"
        assert received[0][1] == {"msg": "world"}

    async def test_bar_event_routing(self, bus: EventBus) -> None:
        received: list[tuple[str, dict]] = []
        callback = AsyncMock(side_effect=lambda t, e: received.append((t, e)))

        bar = _make_bar()
        event = BarEvent(bar=bar, source="test")

        await bus.subscribe("bar.xauusd", callback)
        await asyncio.sleep(0.05)

        await bus.publish(event.topic, bar.to_dict())
        await asyncio.sleep(0.1)

        assert len(received) == 1
        assert received[0][0] == "bar.xauusd.5m"
        restored = NormalizedBar.from_dict(received[0][1])
        assert restored.symbol == "XAUUSD"
        assert restored.close == Decimal("2051.75")

    async def test_feature_event_routing(self, bus: EventBus) -> None:
        received: list[tuple[str, dict]] = []
        callback = AsyncMock(side_effect=lambda t, e: received.append((t, e)))

        fv = _make_feature_vector()
        event = FeatureEvent(features=fv)

        await bus.subscribe("features", callback)
        await asyncio.sleep(0.05)

        await bus.publish(event.topic, fv.to_dict())
        await asyncio.sleep(0.1)

        assert len(received) == 1
        assert received[0][0] == "features.xauusd"
        restored = FeatureVector.from_dict(received[0][1])
        assert restored.rsi_14 == pytest.approx(55.3)
        assert restored.warmup_complete is True

    async def test_multiple_subscribers_same_topic(self, bus: EventBus) -> None:
        received_a: list[dict] = []
        received_b: list[dict] = []
        cb_a = AsyncMock(side_effect=lambda t, e: received_a.append(e))
        cb_b = AsyncMock(side_effect=lambda t, e: received_b.append(e))

        await bus.subscribe("data", cb_a)
        await bus.subscribe("data", cb_b)
        await asyncio.sleep(0.05)

        await bus.publish("data.tick", {"price": 100})
        await asyncio.sleep(0.1)

        assert len(received_a) == 1
        assert len(received_b) == 1

    async def test_sync_callback(self, bus: EventBus) -> None:
        received: list[str] = []

        def sync_cb(topic: str, event: dict) -> None:
            received.append(topic)

        await bus.subscribe("sync", sync_cb)
        await asyncio.sleep(0.05)

        await bus.publish("sync.test", {"v": 1})
        await asyncio.sleep(0.1)

        assert received == ["sync.test"]

    async def test_unsubscribe(self, bus: EventBus) -> None:
        received: list[dict] = []
        callback = AsyncMock(side_effect=lambda t, e: received.append(e))

        await bus.subscribe("unsub", callback)
        await asyncio.sleep(0.05)

        await bus.publish("unsub.a", {"n": 1})
        await asyncio.sleep(0.1)
        assert len(received) == 1

        await bus.unsubscribe("unsub")
        await asyncio.sleep(0.05)

        await bus.publish("unsub.b", {"n": 2})
        await asyncio.sleep(0.1)
        # Should still be 1 -- the second message should not arrive
        assert len(received) == 1


# ---------------------------------------------------------------------------
# Topic filtering tests
# ---------------------------------------------------------------------------

class TestTopicFiltering:
    """Tests for topic prefix-based filtering."""

    @pytest.fixture
    async def bus(self):
        b = EventBus(settings=_free_port_settings(15559, 15560))
        await b.start()
        yield b
        await b.stop()

    async def test_prefix_filter_matches_subtopics(self, bus: EventBus) -> None:
        received: list[str] = []
        callback = AsyncMock(side_effect=lambda t, e: received.append(t))

        await bus.subscribe("bar.xauusd", callback)
        await asyncio.sleep(0.05)

        await bus.publish("bar.xauusd.5m", {"tf": "5m"})
        await bus.publish("bar.xauusd.1h", {"tf": "1h"})
        await bus.publish("bar.xauusd.4h", {"tf": "4h"})
        await asyncio.sleep(0.15)

        assert sorted(received) == ["bar.xauusd.1h", "bar.xauusd.4h", "bar.xauusd.5m"]

    async def test_prefix_filter_excludes_non_matching(self, bus: EventBus) -> None:
        received: list[str] = []
        callback = AsyncMock(side_effect=lambda t, e: received.append(t))

        await bus.subscribe("bar.xauusd", callback)
        await asyncio.sleep(0.05)

        await bus.publish("bar.xauusd.5m", {"tf": "5m"})
        await bus.publish("bar.dxy.5m", {"tf": "5m"})  # different symbol
        await bus.publish("features.xauusd", {"rsi": 55})  # different event type
        await asyncio.sleep(0.15)

        assert received == ["bar.xauusd.5m"]

    async def test_empty_prefix_receives_all(self, bus: EventBus) -> None:
        received: list[str] = []
        callback = AsyncMock(side_effect=lambda t, e: received.append(t))

        await bus.subscribe("", callback)
        await asyncio.sleep(0.05)

        await bus.publish("bar.xauusd.5m", {})
        await bus.publish("features.xauusd", {})
        await bus.publish("other.topic", {})
        await asyncio.sleep(0.15)

        assert len(received) == 3

    async def test_exact_topic_match(self, bus: EventBus) -> None:
        received: list[str] = []
        callback = AsyncMock(side_effect=lambda t, e: received.append(t))

        await bus.subscribe("bar.xauusd.5m", callback)
        await asyncio.sleep(0.05)

        await bus.publish("bar.xauusd.5m", {"tf": "5m"})
        await bus.publish("bar.xauusd.1h", {"tf": "1h"})
        await asyncio.sleep(0.15)

        assert received == ["bar.xauusd.5m"]

    async def test_multiple_subscriptions_different_prefixes(self, bus: EventBus) -> None:
        bars: list[str] = []
        features: list[str] = []
        bar_cb = AsyncMock(side_effect=lambda t, e: bars.append(t))
        feat_cb = AsyncMock(side_effect=lambda t, e: features.append(t))

        await bus.subscribe("bar", bar_cb)
        await bus.subscribe("features", feat_cb)
        await asyncio.sleep(0.05)

        await bus.publish("bar.xauusd.5m", {})
        await bus.publish("features.xauusd", {})
        await bus.publish("bar.dxy.1h", {})
        await asyncio.sleep(0.15)

        assert sorted(bars) == ["bar.dxy.1h", "bar.xauusd.5m"]
        assert features == ["features.xauusd"]
