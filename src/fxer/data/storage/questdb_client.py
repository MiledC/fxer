"""QuestDB storage client for persisting bars and features."""

import logging
from datetime import datetime
from decimal import Decimal
from typing import Any

from questdb.ingress import Sender, TimestampNanos

from fxer.config.settings import Settings
from fxer.core.events import FeatureVector, NormalizedBar
from fxer.core.exceptions import StorageError
from fxer.core.types import Timeframe

logger = logging.getLogger(__name__)

# Table creation DDL for QuestDB (executed via PG wire protocol)
BARS_TABLE_DDL = """\
CREATE TABLE IF NOT EXISTS bars (
    symbol SYMBOL,
    timeframe SYMBOL,
    timestamp TIMESTAMP,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    volume DOUBLE,
    is_complete BOOLEAN
) timestamp(timestamp) PARTITION BY DAY WAL
DEDUP UPSERT KEYS(symbol, timeframe, timestamp);
"""

FEATURES_TABLE_DDL = """\
CREATE TABLE IF NOT EXISTS features (
    symbol SYMBOL,
    timeframe SYMBOL,
    timestamp TIMESTAMP,
    rsi_14 DOUBLE,
    rsi_7 DOUBLE,
    macd_line DOUBLE,
    macd_signal DOUBLE,
    macd_histogram DOUBLE,
    bb_upper DOUBLE,
    bb_middle DOUBLE,
    bb_lower DOUBLE,
    bb_width DOUBLE,
    bb_percent_b DOUBLE,
    atr_14 DOUBLE,
    is_london_session BOOLEAN,
    is_ny_session BOOLEAN,
    is_overlap_session BOOLEAN,
    is_asian_session BOOLEAN,
    hour_of_day INT,
    day_of_week INT,
    is_month_turn BOOLEAN,
    warmup_complete BOOLEAN,
    dxy_return_1h DOUBLE,
    dxy_rsi_14 DOUBLE,
    vix_level DOUBLE,
    vix_change DOUBLE,
    return_1bar DOUBLE,
    return_5bar DOUBLE,
    return_12bar DOUBLE,
    rolling_volatility_20 DOUBLE,
    momentum_48 DOUBLE
) timestamp(timestamp) PARTITION BY DAY WAL
DEDUP UPSERT KEYS(symbol, timeframe, timestamp);
"""


class QuestDBClient:
    """Client for QuestDB storage using ILP ingestion and PG wire queries.

    Uses the QuestDB Python client's Sender for high-performance ILP ingestion
    (port 9009) and psycopg2 via PG wire protocol for queries (port 8812).
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        self._host = self._settings.questdb_host
        self._ilp_port = self._settings.questdb_ilp_port
        self._pg_port = self._settings.questdb_pg_port
        self._pg_user = self._settings.questdb_pg_user
        self._pg_password = self._settings.questdb_pg_password

    @property
    def _ilp_conf(self) -> str:
        """Build the ILP connection configuration string."""
        # Use TCP protocol for ILP (more reliable than HTTP for batch inserts)
        return f"tcp::addr={self._host}:{self._ilp_port};"

    def _get_pg_connection(self) -> Any:
        """Create a new PG wire protocol connection for queries."""
        try:
            import psycopg2

            return psycopg2.connect(
                host=self._host,
                port=self._pg_port,
                user=self._pg_user,
                password=self._pg_password,
                database="qdb",
            )
        except Exception as exc:
            raise StorageError(
                f"Failed to connect to QuestDB PG wire: {exc}",
                operation="connect",
            ) from exc

    def init_tables(self) -> None:
        """Create tables if they don't exist using PG wire protocol.

        Also adds any new columns to existing tables (QuestDB's
        CREATE TABLE IF NOT EXISTS won't alter an existing schema).
        """
        conn = self._get_pg_connection()
        try:
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute(BARS_TABLE_DDL)
                cur.execute(FEATURES_TABLE_DDL)
                # Migrate: add cross-asset columns if missing
                for col in ("dxy_return_1h", "dxy_rsi_14", "vix_level", "vix_change"):
                    try:
                        cur.execute(f"ALTER TABLE features ADD COLUMN {col} DOUBLE")
                        logger.info("Added column features.%s", col)
                    except Exception:
                        pass  # column already exists
                # Migrate: add bb_percent_b if missing
                for col in ("bb_percent_b",):
                    try:
                        cur.execute(f"ALTER TABLE features ADD COLUMN {col} DOUBLE")
                        logger.info("Added column features.%s", col)
                    except Exception:
                        pass  # column already exists
                # Migrate: add price-derived features if missing
                for col in ("return_1bar", "return_5bar", "return_12bar",
                           "rolling_volatility_20", "momentum_48"):
                    try:
                        cur.execute(f"ALTER TABLE features ADD COLUMN {col} DOUBLE")
                        logger.info("Added column features.%s", col)
                    except Exception:
                        pass  # column already exists
            logger.info("QuestDB tables initialized")
        except Exception as exc:
            raise StorageError(
                f"Failed to initialize tables: {exc}",
                operation="init_tables",
            ) from exc
        finally:
            conn.close()

    def insert_bar(self, bar: NormalizedBar) -> None:
        """Insert a single bar via ILP.

        Args:
            bar: The normalized bar to insert.
        """
        self.insert_bars([bar])

    def insert_bars(self, bars: list[NormalizedBar]) -> None:
        """Insert multiple bars via ILP batch ingestion.

        Args:
            bars: List of normalized bars to insert.
        """
        if not bars:
            return

        try:
            with Sender.from_conf(self._ilp_conf) as sender:
                for bar in bars:
                    ts_nanos = TimestampNanos(int(bar.timestamp.timestamp() * 1_000_000_000))
                    sender.row(
                        "bars",
                        symbols={
                            "symbol": bar.symbol,
                            "timeframe": bar.timeframe.value,
                        },
                        columns={
                            "open": float(bar.open),
                            "high": float(bar.high),
                            "low": float(bar.low),
                            "close": float(bar.close),
                            "volume": float(bar.volume) if bar.volume else 0.0,
                            "is_complete": bar.is_complete,
                        },
                        at=ts_nanos,
                    )
            logger.debug("Inserted %d bars via ILP", len(bars))
        except Exception as exc:
            raise StorageError(
                f"Failed to insert {len(bars)} bars: {exc}",
                operation="insert_bars",
                table="bars",
            ) from exc

    def insert_features(self, features: FeatureVector) -> None:
        """Insert a feature vector via ILP.

        Args:
            features: The feature vector to insert.
        """
        try:
            with Sender.from_conf(self._ilp_conf) as sender:
                ts_nanos = TimestampNanos(
                    int(features.timestamp.timestamp() * 1_000_000_000)
                )

                columns: dict[str, Any] = {
                    "hour_of_day": features.hour_of_day,
                    "day_of_week": features.day_of_week,
                    "is_london_session": features.is_london_session,
                    "is_ny_session": features.is_ny_session,
                    "is_overlap_session": features.is_overlap_session,
                    "is_asian_session": features.is_asian_session,
                    "is_month_turn": features.is_month_turn,
                    "warmup_complete": features.warmup_complete,
                }
                # Only include non-None float columns
                for field_name in (
                    "rsi_14", "rsi_7", "macd_line", "macd_signal", "macd_histogram",
                    "bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_percent_b",
                    "atr_14",
                    "dxy_return_1h", "dxy_rsi_14", "vix_level", "vix_change",
                    "return_1bar", "return_5bar", "return_12bar",
                    "rolling_volatility_20", "momentum_48",
                ):
                    val = getattr(features, field_name)
                    if val is not None:
                        columns[field_name] = float(val)

                sender.row(
                    "features",
                    symbols={
                        "symbol": features.symbol,
                        "timeframe": features.timeframe.value,
                    },
                    columns=columns,
                    at=ts_nanos,
                )
            logger.debug("Inserted features for %s at %s", features.symbol, features.timestamp)
        except Exception as exc:
            raise StorageError(
                f"Failed to insert features: {exc}",
                operation="insert_features",
                table="features",
            ) from exc

    def query_bars(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> list[NormalizedBar]:
        """Query bars from QuestDB via PG wire protocol.

        Args:
            symbol: Trading symbol (e.g. "XAUUSD").
            timeframe: Bar timeframe.
            start: Start of time range (inclusive).
            end: End of time range (inclusive).

        Returns:
            List of NormalizedBar objects ordered by timestamp.
        """
        # QuestDB rejects psycopg2's ::timestamptz casts,
        # so pass timestamps as ISO-format strings.
        start_str = start.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        end_str = end.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        conn = self._get_pg_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT symbol, timeframe, timestamp,
                           open, high, low, close, volume, is_complete
                    FROM bars
                    WHERE symbol = %s
                      AND timeframe = %s
                      AND timestamp >= %s
                      AND timestamp <= %s
                    ORDER BY timestamp ASC
                    """,
                    (symbol, timeframe.value, start_str, end_str),
                )
                rows = cur.fetchall()
        except Exception as exc:
            raise StorageError(
                f"Failed to query bars: {exc}",
                operation="query_bars",
                table="bars",
            ) from exc
        finally:
            conn.close()

        bars: list[NormalizedBar] = []
        for row in rows:
            bars.append(
                NormalizedBar(
                    symbol=row[0],
                    timeframe=Timeframe.from_string(row[1]),
                    timestamp=row[2] if isinstance(row[2], datetime) else datetime.fromisoformat(str(row[2])),
                    open=Decimal(str(row[3])),
                    high=Decimal(str(row[4])),
                    low=Decimal(str(row[5])),
                    close=Decimal(str(row[6])),
                    volume=Decimal(str(row[7])),
                    is_complete=bool(row[8]),
                )
            )
        return bars

    def query_daily_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> list[NormalizedBar]:
        """Query daily aggregated bars using QuestDB SAMPLE BY.

        Aggregates 5m bars into daily OHLC using QuestDB's native
        SAMPLE BY 1d operator.

        Args:
            symbol: Trading symbol (e.g. "XAUUSD").
            start: Start of time range (inclusive).
            end: End of time range (inclusive).

        Returns:
            List of NormalizedBar objects with timeframe=D1, ordered by timestamp.
        """
        # QuestDB's SAMPLE BY parser rejects psycopg2's ::timestamptz casts,
        # so pass timestamps as ISO-format strings instead.
        start_str = start.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        end_str = end.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        conn = self._get_pg_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT timestamp, first(open) as open, max(high) as high,
                           min(low) as low, last(close) as close, sum(volume) as volume
                    FROM bars
                    WHERE symbol = %s AND timestamp >= %s AND timestamp <= %s
                    SAMPLE BY 1d
                    ORDER BY timestamp ASC
                    """,
                    (symbol, start_str, end_str),
                )
                rows = cur.fetchall()
        except Exception as exc:
            raise StorageError(
                f"Failed to query daily bars: {exc}",
                operation="query_daily_bars",
                table="bars",
            ) from exc
        finally:
            conn.close()

        bars: list[NormalizedBar] = []
        for row in rows:
            bars.append(
                NormalizedBar(
                    symbol=symbol,
                    timeframe=Timeframe.D1,
                    timestamp=row[0] if isinstance(row[0], datetime) else datetime.fromisoformat(str(row[0])),
                    open=Decimal(str(row[1])),
                    high=Decimal(str(row[2])),
                    low=Decimal(str(row[3])),
                    close=Decimal(str(row[4])),
                    volume=Decimal(str(row[5])) if row[5] is not None else Decimal("0"),
                    is_complete=True,
                )
            )
        return bars

    def query_latest_bar(
        self, symbol: str, timeframe: Timeframe
    ) -> NormalizedBar | None:
        """Query the most recent bar for a symbol/timeframe pair.

        Args:
            symbol: Trading symbol.
            timeframe: Bar timeframe.

        Returns:
            The latest NormalizedBar or None if no data exists.
        """
        conn = self._get_pg_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT symbol, timeframe, timestamp,
                           open, high, low, close, volume, is_complete
                    FROM bars
                    WHERE symbol = %s AND timeframe = %s
                    ORDER BY timestamp DESC
                    LIMIT 1
                    """,
                    (symbol, timeframe.value),
                )
                row = cur.fetchone()
        except Exception as exc:
            raise StorageError(
                f"Failed to query latest bar: {exc}",
                operation="query_latest_bar",
                table="bars",
            ) from exc
        finally:
            conn.close()

        if row is None:
            return None

        return NormalizedBar(
            symbol=row[0],
            timeframe=Timeframe.from_string(row[1]),
            timestamp=row[2] if isinstance(row[2], datetime) else datetime.fromisoformat(str(row[2])),
            open=Decimal(str(row[3])),
            high=Decimal(str(row[4])),
            low=Decimal(str(row[5])),
            close=Decimal(str(row[6])),
            volume=Decimal(str(row[7])),
            is_complete=bool(row[8]),
        )

    def close(self) -> None:
        """Close resources (no-op since Sender uses context manager per operation)."""
        pass

    def __enter__(self) -> "QuestDBClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
