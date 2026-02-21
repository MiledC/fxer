"""Unit tests for storage clients (QuestDB and Redis) using mocks."""

import json
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from fxer.config.settings import Settings
from fxer.core.events import FeatureVector, NormalizedBar
from fxer.core.exceptions import StorageError
from fxer.core.types import Timeframe
from fxer.data.storage.questdb_client import QuestDBClient
from fxer.data.storage.redis_client import RedisClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def settings() -> Settings:
    return Settings(
        questdb_host="localhost",
        questdb_ilp_port=9009,
        questdb_pg_port=8812,
        questdb_pg_user="admin",
        questdb_pg_password="quest",
        redis_host="localhost",
        redis_port=6379,
        redis_db=0,
    )


@pytest.fixture
def sample_bar() -> NormalizedBar:
    return NormalizedBar(
        symbol="XAUUSD",
        timeframe=Timeframe.M5,
        timestamp=datetime(2025, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
        open=Decimal("2050.50"),
        high=Decimal("2052.00"),
        low=Decimal("2049.00"),
        close=Decimal("2051.75"),
        volume=Decimal("1200"),
        is_complete=True,
    )


@pytest.fixture
def sample_bars(sample_bar: NormalizedBar) -> list[NormalizedBar]:
    bar2 = NormalizedBar(
        symbol="XAUUSD",
        timeframe=Timeframe.M5,
        timestamp=datetime(2025, 1, 15, 10, 35, 0, tzinfo=timezone.utc),
        open=Decimal("2051.75"),
        high=Decimal("2053.00"),
        low=Decimal("2050.50"),
        close=Decimal("2052.50"),
        volume=Decimal("1100"),
        is_complete=True,
    )
    return [sample_bar, bar2]


@pytest.fixture
def sample_features() -> FeatureVector:
    return FeatureVector(
        symbol="XAUUSD",
        timestamp=datetime(2025, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
        timeframe=Timeframe.M5,
        rsi_14=55.3,
        macd_line=0.12,
        macd_signal=0.08,
        macd_histogram=0.04,
        atr_14=3.5,
        is_london_session=True,
        hour_of_day=10,
        day_of_week=2,
        warmup_complete=True,
    )


# ---------------------------------------------------------------------------
# QuestDB Client Tests
# ---------------------------------------------------------------------------

class TestQuestDBClient:
    """Tests for QuestDBClient with mocked ILP and PG connections."""

    @patch("fxer.data.storage.questdb_client.Sender")
    def test_insert_bar_calls_sender(
        self, mock_sender_cls: MagicMock, settings: Settings, sample_bar: NormalizedBar
    ) -> None:
        mock_sender = MagicMock()
        mock_sender_cls.from_conf.return_value.__enter__ = MagicMock(return_value=mock_sender)
        mock_sender_cls.from_conf.return_value.__exit__ = MagicMock(return_value=False)

        client = QuestDBClient(settings)
        client.insert_bar(sample_bar)

        mock_sender.row.assert_called_once()

    @patch("fxer.data.storage.questdb_client.Sender")
    def test_insert_bars_batch(
        self,
        mock_sender_cls: MagicMock,
        settings: Settings,
        sample_bars: list[NormalizedBar],
    ) -> None:
        mock_sender = MagicMock()
        mock_sender_cls.from_conf.return_value.__enter__ = MagicMock(return_value=mock_sender)
        mock_sender_cls.from_conf.return_value.__exit__ = MagicMock(return_value=False)

        client = QuestDBClient(settings)
        client.insert_bars(sample_bars)

        assert mock_sender.row.call_count == 2

    @patch("fxer.data.storage.questdb_client.Sender")
    def test_insert_bars_empty_list_noop(
        self, mock_sender_cls: MagicMock, settings: Settings
    ) -> None:
        client = QuestDBClient(settings)
        client.insert_bars([])
        mock_sender_cls.from_conf.assert_not_called()

    @patch("fxer.data.storage.questdb_client.Sender")
    def test_insert_bars_sender_error_raises_storage_error(
        self,
        mock_sender_cls: MagicMock,
        settings: Settings,
        sample_bar: NormalizedBar,
    ) -> None:
        mock_sender = MagicMock()
        mock_sender.row.side_effect = RuntimeError("ILP row failed")
        mock_sender_cls.from_conf.return_value.__enter__ = MagicMock(return_value=mock_sender)
        mock_sender_cls.from_conf.return_value.__exit__ = MagicMock(return_value=False)

        client = QuestDBClient(settings)
        with pytest.raises(StorageError, match="Failed to insert 1 bars"):
            client.insert_bars([sample_bar])

    @patch("fxer.data.storage.questdb_client.Sender")
    def test_insert_features(
        self,
        mock_sender_cls: MagicMock,
        settings: Settings,
        sample_features: FeatureVector,
    ) -> None:
        mock_sender = MagicMock()
        mock_sender_cls.from_conf.return_value.__enter__ = MagicMock(return_value=mock_sender)
        mock_sender_cls.from_conf.return_value.__exit__ = MagicMock(return_value=False)

        client = QuestDBClient(settings)
        client.insert_features(sample_features)

        mock_sender.row.assert_called_once()

    @patch("fxer.data.storage.questdb_client.psycopg2", create=True)
    def test_query_bars_returns_normalized_bars(
        self, mock_psycopg2: MagicMock, settings: Settings
    ) -> None:
        ts = datetime(2025, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            ("XAUUSD", "5m", ts, 2050.50, 2052.00, 2049.00, 2051.75, 1200.0, True),
        ]
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_psycopg2.connect.return_value = mock_conn

        # Patch the import inside _get_pg_connection
        with patch.dict("sys.modules", {"psycopg2": mock_psycopg2}):
            client = QuestDBClient(settings)
            bars = client.query_bars(
                symbol="XAUUSD",
                timeframe=Timeframe.M5,
                start=datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
                end=datetime(2025, 1, 15, 11, 0, 0, tzinfo=timezone.utc),
            )

        assert len(bars) == 1
        assert bars[0].symbol == "XAUUSD"
        assert bars[0].timeframe == Timeframe.M5
        assert bars[0].close == Decimal("2051.75")

    @patch("fxer.data.storage.questdb_client.psycopg2", create=True)
    def test_query_latest_bar_returns_none_when_empty(
        self, mock_psycopg2: MagicMock, settings: Settings
    ) -> None:
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_psycopg2.connect.return_value = mock_conn

        with patch.dict("sys.modules", {"psycopg2": mock_psycopg2}):
            client = QuestDBClient(settings)
            result = client.query_latest_bar("XAUUSD", Timeframe.M5)

        assert result is None

    @patch("fxer.data.storage.questdb_client.psycopg2", create=True)
    def test_init_tables(
        self, mock_psycopg2: MagicMock, settings: Settings
    ) -> None:
        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_psycopg2.connect.return_value = mock_conn

        with patch.dict("sys.modules", {"psycopg2": mock_psycopg2}):
            client = QuestDBClient(settings)
            client.init_tables()

        # 2 DDL + 4 cross-asset migrations + 5 price-feature migrations = 11
        assert mock_cursor.execute.call_count == 11

    def test_close_is_noop(self, settings: Settings) -> None:
        """close() is a no-op since Sender uses per-operation context managers."""
        client = QuestDBClient(settings)
        client.close()  # should not raise

    def test_context_manager(self, settings: Settings) -> None:
        """QuestDBClient supports the context manager protocol."""
        with QuestDBClient(settings) as client:
            assert isinstance(client, QuestDBClient)


# ---------------------------------------------------------------------------
# Redis Client Tests
# ---------------------------------------------------------------------------

class TestRedisClient:
    """Tests for RedisClient with mocked redis connection."""

    @patch("fxer.data.storage.redis_client.redis.Redis")
    def test_set_and_get_latest_bar(
        self,
        mock_redis_cls: MagicMock,
        settings: Settings,
        sample_bar: NormalizedBar,
    ) -> None:
        mock_redis = MagicMock()
        mock_redis_cls.return_value = mock_redis

        client = RedisClient(settings)
        # Override the lazy connection
        client._client = mock_redis

        client.set_latest_bar("XAUUSD", Timeframe.M5, sample_bar)
        mock_redis.set.assert_called_once()

        # Verify the serialized data round-trips
        call_args = mock_redis.set.call_args
        stored_json = call_args[0][1]
        recovered = NormalizedBar.from_dict(json.loads(stored_json))
        assert recovered.symbol == sample_bar.symbol
        assert recovered.close == sample_bar.close

    @patch("fxer.data.storage.redis_client.redis.Redis")
    def test_get_latest_bar_returns_none_on_miss(
        self, mock_redis_cls: MagicMock, settings: Settings
    ) -> None:
        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        mock_redis_cls.return_value = mock_redis

        client = RedisClient(settings)
        client._client = mock_redis

        result = client.get_latest_bar("XAUUSD", Timeframe.M5)
        assert result is None

    @patch("fxer.data.storage.redis_client.redis.Redis")
    def test_get_latest_bar_returns_bar_on_hit(
        self,
        mock_redis_cls: MagicMock,
        settings: Settings,
        sample_bar: NormalizedBar,
    ) -> None:
        mock_redis = MagicMock()
        mock_redis.get.return_value = json.dumps(sample_bar.to_dict())
        mock_redis_cls.return_value = mock_redis

        client = RedisClient(settings)
        client._client = mock_redis

        result = client.get_latest_bar("XAUUSD", Timeframe.M5)
        assert result is not None
        assert result.symbol == "XAUUSD"
        assert result.close == Decimal("2051.75")

    @patch("fxer.data.storage.redis_client.redis.Redis")
    def test_set_and_get_latest_features(
        self,
        mock_redis_cls: MagicMock,
        settings: Settings,
        sample_features: FeatureVector,
    ) -> None:
        mock_redis = MagicMock()
        mock_redis_cls.return_value = mock_redis

        client = RedisClient(settings)
        client._client = mock_redis

        client.set_latest_features("XAUUSD", Timeframe.M5, sample_features)
        mock_redis.set.assert_called_once()

        # Verify the serialized data round-trips
        call_args = mock_redis.set.call_args
        stored_json = call_args[0][1]
        recovered = FeatureVector.from_dict(json.loads(stored_json))
        assert recovered.symbol == sample_features.symbol
        assert recovered.rsi_14 == sample_features.rsi_14

    @patch("fxer.data.storage.redis_client.redis.Redis")
    def test_get_latest_features_returns_none_on_miss(
        self, mock_redis_cls: MagicMock, settings: Settings
    ) -> None:
        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        mock_redis_cls.return_value = mock_redis

        client = RedisClient(settings)
        client._client = mock_redis

        result = client.get_latest_features("XAUUSD", Timeframe.M5)
        assert result is None

    @patch("fxer.data.storage.redis_client.redis.Redis")
    def test_get_latest_features_returns_features_on_hit(
        self,
        mock_redis_cls: MagicMock,
        settings: Settings,
        sample_features: FeatureVector,
    ) -> None:
        mock_redis = MagicMock()
        mock_redis.get.return_value = json.dumps(sample_features.to_dict())
        mock_redis_cls.return_value = mock_redis

        client = RedisClient(settings)
        client._client = mock_redis

        result = client.get_latest_features("XAUUSD", Timeframe.M5)
        assert result is not None
        assert result.rsi_14 == 55.3
        assert result.warmup_complete is True

    @patch("fxer.data.storage.redis_client.redis.Redis")
    def test_delete_bar(
        self, mock_redis_cls: MagicMock, settings: Settings
    ) -> None:
        mock_redis = MagicMock()
        mock_redis_cls.return_value = mock_redis

        client = RedisClient(settings)
        client._client = mock_redis

        client.delete_bar("XAUUSD", Timeframe.M5)
        mock_redis.delete.assert_called_once_with("fxer:bar:latest:XAUUSD:5m")

    @patch("fxer.data.storage.redis_client.redis.Redis")
    def test_delete_features(
        self, mock_redis_cls: MagicMock, settings: Settings
    ) -> None:
        mock_redis = MagicMock()
        mock_redis_cls.return_value = mock_redis

        client = RedisClient(settings)
        client._client = mock_redis

        client.delete_features("XAUUSD", Timeframe.M5)
        mock_redis.delete.assert_called_once_with("fxer:feature:latest:XAUUSD:5m")

    @patch("fxer.data.storage.redis_client.redis.Redis")
    def test_redis_error_raises_storage_error(
        self, mock_redis_cls: MagicMock, settings: Settings
    ) -> None:
        import redis as redis_lib

        mock_redis = MagicMock()
        mock_redis.get.side_effect = redis_lib.RedisError("Connection lost")
        mock_redis_cls.return_value = mock_redis

        client = RedisClient(settings)
        client._client = mock_redis

        with pytest.raises(StorageError, match="Failed to get latest bar"):
            client.get_latest_bar("XAUUSD", Timeframe.M5)

    @patch("fxer.data.storage.redis_client.redis.Redis")
    def test_close_cleans_up(
        self, mock_redis_cls: MagicMock, settings: Settings
    ) -> None:
        mock_redis = MagicMock()
        mock_redis_cls.return_value = mock_redis

        client = RedisClient(settings)
        client._client = mock_redis
        client.close()

        mock_redis.close.assert_called_once()
        assert client._client is None

    @patch("fxer.data.storage.redis_client.redis.Redis")
    def test_context_manager(
        self, mock_redis_cls: MagicMock, settings: Settings
    ) -> None:
        mock_redis = MagicMock()
        mock_redis_cls.return_value = mock_redis

        with RedisClient(settings) as client:
            client._client = mock_redis

        mock_redis.close.assert_called_once()

    def test_bar_key_format(self) -> None:
        key = RedisClient._bar_key("XAUUSD", Timeframe.M5)
        assert key == "fxer:bar:latest:XAUUSD:5m"

    def test_feature_key_format(self) -> None:
        key = RedisClient._feature_key("XAUUSD", Timeframe.H1)
        assert key == "fxer:feature:latest:XAUUSD:1h"
