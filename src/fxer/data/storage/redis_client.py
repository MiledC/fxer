"""Redis client for state caching of bars and features."""

import json
import logging
from datetime import datetime
from decimal import Decimal
from typing import Any

import redis

from fxer.config.settings import Settings
from fxer.core.events import FeatureVector, NormalizedBar
from fxer.core.exceptions import StorageError
from fxer.core.types import Timeframe

logger = logging.getLogger(__name__)

# Key prefix constants
_BAR_PREFIX = "fxer:bar:latest"
_FEATURE_PREFIX = "fxer:feature:latest"

# Default TTLs in seconds
_BAR_TTL = 3600  # 1 hour
_FEATURE_TTL = 3600  # 1 hour


class RedisClient:
    """Redis client for caching latest bars and features.

    Provides fast read/write access to the most recent market state
    used by downstream consumers (feature engine, signals, etc.).
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        self._client: redis.Redis | None = None

    def _get_client(self) -> redis.Redis:
        """Get or create the Redis client connection."""
        if self._client is None:
            try:
                self._client = redis.Redis(
                    host=self._settings.redis_host,
                    port=self._settings.redis_port,
                    db=self._settings.redis_db,
                    decode_responses=True,
                )
                # Verify connection
                self._client.ping()
            except redis.ConnectionError as exc:
                self._client = None
                raise StorageError(
                    f"Failed to connect to Redis: {exc}",
                    operation="connect",
                ) from exc
        return self._client

    @staticmethod
    def _bar_key(symbol: str, timeframe: Timeframe) -> str:
        """Build Redis key for a latest bar entry."""
        return f"{_BAR_PREFIX}:{symbol}:{timeframe.value}"

    @staticmethod
    def _feature_key(symbol: str, timeframe: Timeframe) -> str:
        """Build Redis key for a latest feature entry."""
        return f"{_FEATURE_PREFIX}:{symbol}:{timeframe.value}"

    @staticmethod
    def _serialize_bar(bar: NormalizedBar) -> str:
        """Serialize a NormalizedBar to JSON string."""
        return json.dumps(bar.to_dict())

    @staticmethod
    def _deserialize_bar(data: str) -> NormalizedBar:
        """Deserialize a JSON string to NormalizedBar."""
        return NormalizedBar.from_dict(json.loads(data))

    @staticmethod
    def _serialize_features(features: FeatureVector) -> str:
        """Serialize a FeatureVector to JSON string."""
        return json.dumps(features.to_dict())

    @staticmethod
    def _deserialize_features(data: str) -> FeatureVector:
        """Deserialize a JSON string to FeatureVector."""
        return FeatureVector.from_dict(json.loads(data))

    def set_latest_bar(
        self, symbol: str, timeframe: Timeframe, bar: NormalizedBar
    ) -> None:
        """Cache the latest bar for a symbol/timeframe pair.

        Args:
            symbol: Trading symbol (e.g. "XAUUSD").
            timeframe: Bar timeframe.
            bar: The NormalizedBar to cache.
        """
        client = self._get_client()
        key = self._bar_key(symbol, timeframe)
        try:
            client.set(key, self._serialize_bar(bar), ex=_BAR_TTL)
            logger.debug("Cached latest bar for %s/%s", symbol, timeframe.value)
        except redis.RedisError as exc:
            raise StorageError(
                f"Failed to set latest bar: {exc}",
                operation="set_latest_bar",
            ) from exc

    def get_latest_bar(
        self, symbol: str, timeframe: Timeframe
    ) -> NormalizedBar | None:
        """Retrieve the cached latest bar for a symbol/timeframe pair.

        Args:
            symbol: Trading symbol.
            timeframe: Bar timeframe.

        Returns:
            The cached NormalizedBar or None if not found.
        """
        client = self._get_client()
        key = self._bar_key(symbol, timeframe)
        try:
            data = client.get(key)
        except redis.RedisError as exc:
            raise StorageError(
                f"Failed to get latest bar: {exc}",
                operation="get_latest_bar",
            ) from exc

        if data is None:
            return None
        return self._deserialize_bar(data)

    def set_latest_features(
        self, symbol: str, timeframe: Timeframe, features: FeatureVector
    ) -> None:
        """Cache the latest feature vector for a symbol/timeframe pair.

        Args:
            symbol: Trading symbol.
            timeframe: Bar timeframe.
            features: The FeatureVector to cache.
        """
        client = self._get_client()
        key = self._feature_key(symbol, timeframe)
        try:
            client.set(key, self._serialize_features(features), ex=_FEATURE_TTL)
            logger.debug(
                "Cached latest features for %s/%s", symbol, timeframe.value
            )
        except redis.RedisError as exc:
            raise StorageError(
                f"Failed to set latest features: {exc}",
                operation="set_latest_features",
            ) from exc

    def get_latest_features(
        self, symbol: str, timeframe: Timeframe
    ) -> FeatureVector | None:
        """Retrieve the cached latest feature vector.

        Args:
            symbol: Trading symbol.
            timeframe: Bar timeframe.

        Returns:
            The cached FeatureVector or None if not found.
        """
        client = self._get_client()
        key = self._feature_key(symbol, timeframe)
        try:
            data = client.get(key)
        except redis.RedisError as exc:
            raise StorageError(
                f"Failed to get latest features: {exc}",
                operation="get_latest_features",
            ) from exc

        if data is None:
            return None
        return self._deserialize_features(data)

    def delete_bar(self, symbol: str, timeframe: Timeframe) -> None:
        """Remove a cached bar entry."""
        client = self._get_client()
        key = self._bar_key(symbol, timeframe)
        try:
            client.delete(key)
        except redis.RedisError as exc:
            raise StorageError(
                f"Failed to delete bar: {exc}",
                operation="delete_bar",
            ) from exc

    def delete_features(self, symbol: str, timeframe: Timeframe) -> None:
        """Remove a cached feature entry."""
        client = self._get_client()
        key = self._feature_key(symbol, timeframe)
        try:
            client.delete(key)
        except redis.RedisError as exc:
            raise StorageError(
                f"Failed to delete features: {exc}",
                operation="delete_features",
            ) from exc

    def close(self) -> None:
        """Close the Redis connection."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                logger.warning("Error closing Redis connection", exc_info=True)
            finally:
                self._client = None

    def __enter__(self) -> "RedisClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
