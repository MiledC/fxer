"""Integration tests for QuestDB and Redis storage clients.

These tests require running QuestDB and Redis instances.
They are skipped by default unless the services are available.

Run with Docker:
    docker run -d --name questdb -p 9009:9009 -p 8812:8812 questdb/questdb
    docker run -d --name redis -p 6379:6379 redis:7
"""

import time
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from fxer.config.settings import Settings
from fxer.core.events import FeatureVector, NormalizedBar
from fxer.core.types import Timeframe
from fxer.data.storage.questdb_client import QuestDBClient
from fxer.data.storage.redis_client import RedisClient


def _questdb_available(settings: Settings) -> bool:
    """Check if QuestDB is reachable via PG wire."""
    try:
        import psycopg2

        conn = psycopg2.connect(
            host=settings.questdb_host,
            port=settings.questdb_pg_port,
            user=settings.questdb_pg_user,
            password=settings.questdb_pg_password,
            database="qdb",
        )
        conn.close()
        return True
    except Exception:
        return False


def _redis_available(settings: Settings) -> bool:
    """Check if Redis is reachable."""
    try:
        import redis

        r = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
        )
        r.ping()
        r.close()
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def settings() -> Settings:
    return Settings()


@pytest.fixture
def sample_bar() -> NormalizedBar:
    return NormalizedBar(
        symbol="XAUUSD",
        timeframe=Timeframe.M5,
        timestamp=datetime(2025, 6, 15, 10, 30, 0, tzinfo=timezone.utc),
        open=Decimal("2050.50"),
        high=Decimal("2052.00"),
        low=Decimal("2049.00"),
        close=Decimal("2051.75"),
        volume=Decimal("1200"),
        is_complete=True,
    )


@pytest.fixture
def sample_features() -> FeatureVector:
    return FeatureVector(
        symbol="XAUUSD",
        timestamp=datetime(2025, 6, 15, 10, 30, 0, tzinfo=timezone.utc),
        timeframe=Timeframe.M5,
        rsi_14=55.3,
        atr_14=3.5,
        is_london_session=True,
        hour_of_day=10,
        day_of_week=2,
        warmup_complete=True,
    )


# ---------------------------------------------------------------------------
# QuestDB Integration Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestQuestDBIntegration:
    """Integration tests for QuestDB client. Requires a running QuestDB."""

    @pytest.fixture(autouse=True)
    def _skip_if_unavailable(self, settings: Settings) -> None:
        if not _questdb_available(settings):
            pytest.skip("QuestDB not available")

    def test_init_tables(self, settings: Settings) -> None:
        with QuestDBClient(settings) as client:
            client.init_tables()

    def test_insert_and_query_bar(
        self, settings: Settings, sample_bar: NormalizedBar
    ) -> None:
        with QuestDBClient(settings) as client:
            client.init_tables()
            client.insert_bar(sample_bar)

            # QuestDB WAL needs a moment to commit
            time.sleep(1)

            bars = client.query_bars(
                symbol="XAUUSD",
                timeframe=Timeframe.M5,
                start=datetime(2025, 6, 15, 10, 0, 0, tzinfo=timezone.utc),
                end=datetime(2025, 6, 15, 11, 0, 0, tzinfo=timezone.utc),
            )

            assert len(bars) >= 1
            matched = [b for b in bars if b.timestamp == sample_bar.timestamp]
            assert len(matched) == 1
            assert matched[0].close == sample_bar.close

    def test_insert_bars_batch(self, settings: Settings) -> None:
        bars = [
            NormalizedBar(
                symbol="XAUUSD",
                timeframe=Timeframe.M5,
                timestamp=datetime(2025, 6, 15, 12, i * 5, 0, tzinfo=timezone.utc),
                open=Decimal("2050.00"),
                high=Decimal("2052.00"),
                low=Decimal("2049.00"),
                close=Decimal("2051.00"),
                volume=Decimal("1000"),
                is_complete=True,
            )
            for i in range(5)
        ]

        with QuestDBClient(settings) as client:
            client.init_tables()
            client.insert_bars(bars)
            time.sleep(1)

            results = client.query_bars(
                symbol="XAUUSD",
                timeframe=Timeframe.M5,
                start=datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc),
                end=datetime(2025, 6, 15, 12, 30, 0, tzinfo=timezone.utc),
            )
            assert len(results) >= 5

    def test_query_latest_bar(
        self, settings: Settings, sample_bar: NormalizedBar
    ) -> None:
        with QuestDBClient(settings) as client:
            client.init_tables()
            client.insert_bar(sample_bar)
            time.sleep(1)

            latest = client.query_latest_bar("XAUUSD", Timeframe.M5)
            assert latest is not None
            assert latest.symbol == "XAUUSD"

    def test_insert_features(
        self, settings: Settings, sample_features: FeatureVector
    ) -> None:
        with QuestDBClient(settings) as client:
            client.init_tables()
            client.insert_features(sample_features)
            # If it doesn't raise, the insert succeeded


# ---------------------------------------------------------------------------
# Redis Integration Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestRedisIntegration:
    """Integration tests for Redis client. Requires a running Redis."""

    @pytest.fixture(autouse=True)
    def _skip_if_unavailable(self, settings: Settings) -> None:
        if not _redis_available(settings):
            pytest.skip("Redis not available")

    def test_set_and_get_latest_bar(
        self, settings: Settings, sample_bar: NormalizedBar
    ) -> None:
        with RedisClient(settings) as client:
            client.set_latest_bar("XAUUSD", Timeframe.M5, sample_bar)
            result = client.get_latest_bar("XAUUSD", Timeframe.M5)

            assert result is not None
            assert result.symbol == "XAUUSD"
            assert result.close == Decimal("2051.75")

            # Clean up
            client.delete_bar("XAUUSD", Timeframe.M5)

    def test_get_latest_bar_miss(self, settings: Settings) -> None:
        with RedisClient(settings) as client:
            # Use an unlikely key
            result = client.get_latest_bar("NONEXISTENT", Timeframe.D1)
            assert result is None

    def test_set_and_get_latest_features(
        self, settings: Settings, sample_features: FeatureVector
    ) -> None:
        with RedisClient(settings) as client:
            client.set_latest_features("XAUUSD", Timeframe.M5, sample_features)
            result = client.get_latest_features("XAUUSD", Timeframe.M5)

            assert result is not None
            assert result.rsi_14 == 55.3
            assert result.warmup_complete is True

            # Clean up
            client.delete_features("XAUUSD", Timeframe.M5)

    def test_delete_bar(
        self, settings: Settings, sample_bar: NormalizedBar
    ) -> None:
        with RedisClient(settings) as client:
            client.set_latest_bar("XAUUSD", Timeframe.M5, sample_bar)
            client.delete_bar("XAUUSD", Timeframe.M5)
            result = client.get_latest_bar("XAUUSD", Timeframe.M5)
            assert result is None

    def test_delete_features(
        self, settings: Settings, sample_features: FeatureVector
    ) -> None:
        with RedisClient(settings) as client:
            client.set_latest_features("XAUUSD", Timeframe.M5, sample_features)
            client.delete_features("XAUUSD", Timeframe.M5)
            result = client.get_latest_features("XAUUSD", Timeframe.M5)
            assert result is None
