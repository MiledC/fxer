"""IBKR historical data loader.

Fetches historical bar data from Interactive Brokers and provides
it through the DataLoader interface.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Generator
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from fxer.config.symbols import get_symbol_config
from fxer.core.types import RawBar, Timeframe
from fxer.data.loaders.base import LoadStats
from fxer.data.loaders.ibkr_client import IBKRClient

logger = logging.getLogger(__name__)

# Pagination constants
MAX_CHUNK_DAYS = 55  # Safe limit for 5m bars (IBKR max is ~60)
RATE_LIMIT_DELAY = 10.5  # Seconds between requests (stay under 60 req/10min)

# Mapping from Timeframe enum to IBKR bar size strings
TIMEFRAME_TO_IBKR: dict[Timeframe, str] = {
    Timeframe.M1: "1 min",
    Timeframe.M5: "5 mins",
    Timeframe.M15: "15 mins",
    Timeframe.H1: "1 hour",
    Timeframe.H4: "4 hours",
    Timeframe.D1: "1 day",
}


@dataclass
class IBKRLoader:
    """Load historical bar data from Interactive Brokers.

    Fetches historical bars using the IBKR API and provides them
    through an iterator interface compatible with the data pipeline.

    Automatically paginates large date ranges into ~55-day chunks to
    stay within IBKR's limits, with rate limiting between requests.

    Usage::

        async with IBKRClient(settings) as client:
            loader = IBKRLoader(client)
            start = datetime(2025, 1, 1, tzinfo=timezone.utc)
            stats = await loader.load("XAUUSD", Timeframe.M5, start_date=start)
            print(f"Loaded {stats.rows_loaded} bars")

            for bar in loader.iter_bars():
                # Process each bar
                pass
    """

    client: IBKRClient
    what_to_show: str | None = None  # If None, uses symbol config (MIDPOINT, BID, ASK, TRADES)
    use_rth: bool = False  # Include extended hours

    _bars: list[RawBar] = field(default_factory=list, init=False, repr=False)
    _stats: LoadStats | None = field(default=None, init=False, repr=False)

    async def load(
        self,
        symbol: str,
        timeframe: Timeframe,
        start_date: datetime,
        end_date: datetime | None = None,
    ) -> LoadStats:
        """Fetch historical bars from IBKR with automatic pagination.

        Chunks large date ranges into ~55-day requests to stay within
        IBKR's limits. Includes rate limiting between requests.

        Args:
            symbol: Symbol name (e.g., "XAUUSD", "EURUSD").
            timeframe: Bar timeframe (e.g., Timeframe.M5).
            start_date: Start date for the request.
            end_date: End date for the request. Defaults to now.

        Returns:
            LoadStats with information about the load operation.
        """
        end_date = end_date or datetime.now(timezone.utc)

        # Validate date range
        if start_date >= end_date:
            logger.error(
                "Invalid date range: start_date (%s) must be before end_date (%s)",
                start_date.date(),
                end_date.date(),
            )
            return LoadStats(
                rows_loaded=0,
                rows_skipped=0,
                start_date=None,
                end_date=None,
                errors=[f"Invalid date range: start ({start_date.date()}) >= end ({end_date.date()})"],
            )

        # Get what_to_show from symbol config if not explicitly set
        symbol_config = get_symbol_config(symbol)
        what_to_show = self.what_to_show or symbol_config.ibkr_what_to_show

        contract = self.client.make_contract(symbol)

        # Qualify the contract to ensure it's valid
        try:
            contract = await self.client.qualify_contract(contract)
        except ValueError as exc:
            logger.error("Failed to qualify contract for %s: %s", symbol, exc)
            return LoadStats(
                rows_loaded=0,
                rows_skipped=0,
                start_date=None,
                end_date=None,
                errors=[str(exc)],
            )

        bar_size = self._timeframe_to_ibkr(timeframe)

        # Calculate chunks (work backwards from end_date)
        chunks = self._calculate_chunks(start_date, end_date, MAX_CHUNK_DAYS)

        logger.info(
            "Fetching %s %s data for %s: %s to %s (%d chunks)",
            bar_size,
            what_to_show,
            symbol,
            start_date.date(),
            end_date.date(),
            len(chunks),
        )

        all_bars: list[RawBar] = []
        errors: list[str] = []

        for i, (chunk_start, chunk_end) in enumerate(chunks):
            logger.info(
                "Fetching chunk %d/%d: %s to %s",
                i + 1,
                len(chunks),
                chunk_start.date(),
                chunk_end.date(),
            )

            bars = await self._fetch_chunk(
                contract, symbol, timeframe, bar_size, what_to_show, chunk_end, chunk_start
            )
            if bars:
                all_bars.extend(bars)

            # Rate limit (skip delay after last chunk)
            if i < len(chunks) - 1:
                logger.debug("Rate limiting: sleeping %.1fs", RATE_LIMIT_DELAY)
                await asyncio.sleep(RATE_LIMIT_DELAY)

        if not all_bars:
            logger.warning("No historical data returned for %s", symbol)
            return LoadStats(
                rows_loaded=0,
                rows_skipped=0,
                start_date=None,
                end_date=None,
                errors=errors,
            )

        # Sort and dedupe by timestamp
        all_bars.sort(key=lambda b: b.timestamp)
        seen_timestamps: set[datetime] = set()
        unique_bars: list[RawBar] = []
        for bar in all_bars:
            if bar.timestamp not in seen_timestamps:
                seen_timestamps.add(bar.timestamp)
                unique_bars.append(bar)

        self._bars = unique_bars

        start = self._bars[0].timestamp if self._bars else None
        end = self._bars[-1].timestamp if self._bars else None

        self._stats = LoadStats(
            rows_loaded=len(self._bars),
            rows_skipped=len(all_bars) - len(unique_bars),
            start_date=start,
            end_date=end,
            errors=errors,
        )

        logger.info(
            "Loaded %d bars for %s from %s to %s",
            len(self._bars),
            symbol,
            start,
            end,
        )

        return self._stats

    def _calculate_chunks(
        self,
        start_date: datetime,
        end_date: datetime,
        chunk_days: int,
    ) -> list[tuple[datetime, datetime]]:
        """Calculate date chunks, working backwards from end_date.

        Args:
            start_date: Earliest date to fetch.
            end_date: Latest date to fetch.
            chunk_days: Maximum days per chunk.

        Returns:
            List of (chunk_start, chunk_end) tuples, ordered most recent first.
        """
        chunks: list[tuple[datetime, datetime]] = []
        current_end = end_date

        while current_end > start_date:
            chunk_start = max(start_date, current_end - timedelta(days=chunk_days))
            chunks.append((chunk_start, current_end))
            current_end = chunk_start

        return chunks  # Ordered from most recent to oldest

    async def _fetch_chunk(
        self,
        contract: object,
        symbol: str,
        timeframe: Timeframe,
        bar_size: str,
        what_to_show: str,
        end_date: datetime,
        start_date: datetime,
    ) -> list[RawBar]:
        """Fetch a single chunk of historical data.

        Args:
            contract: Qualified IBKR contract.
            symbol: Symbol name.
            timeframe: Bar timeframe.
            bar_size: IBKR bar size string.
            what_to_show: IBKR data type (MIDPOINT, TRADES, etc).
            end_date: End date for this chunk.
            start_date: Start date for this chunk.

        Returns:
            List of RawBar instances for this chunk.
        """
        duration_days = (end_date - start_date).days + 1
        duration_str = f"{duration_days} D"
        end_dt_str = end_date.strftime("%Y%m%d %H:%M:%S")

        try:
            bars = await self.client.ib.reqHistoricalDataAsync(
                contract,
                endDateTime=end_dt_str,
                durationStr=duration_str,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=self.use_rth,
                formatDate=1,  # Return as datetime objects
            )
        except Exception as exc:
            logger.error(
                "Failed to fetch chunk %s to %s: %s",
                start_date.date(),
                end_date.date(),
                exc,
            )
            return []

        if not bars:
            logger.debug("No data returned for chunk %s to %s", start_date, end_date)
            return []

        # Convert IBKR bars to RawBar format
        return [self._to_raw_bar(bar, symbol, timeframe.value) for bar in bars]

    def iter_bars(self) -> Generator[RawBar, None, None]:
        """Iterate over loaded bars in chronological order.

        Yields:
            RawBar instances sorted by timestamp ascending.
        """
        yield from self._bars

    def get_date_range(self) -> tuple[datetime, datetime] | None:
        """Return the date range of loaded data.

        Returns:
            Tuple of (start_date, end_date) or None if no data is loaded.
        """
        if not self._bars:
            return None
        return (self._bars[0].timestamp, self._bars[-1].timestamp)

    def _timeframe_to_ibkr(self, timeframe: Timeframe) -> str:
        """Convert Timeframe enum to IBKR bar size string.

        Args:
            timeframe: The timeframe to convert.

        Returns:
            IBKR bar size string (e.g., "5 mins").

        Raises:
            ValueError: If timeframe is not supported.
        """
        if timeframe not in TIMEFRAME_TO_IBKR:
            supported = ", ".join(tf.value for tf in TIMEFRAME_TO_IBKR)
            raise ValueError(
                f"Unsupported timeframe: {timeframe.value}. Supported: {supported}"
            )
        return TIMEFRAME_TO_IBKR[timeframe]

    def _to_raw_bar(self, ibkr_bar: object, symbol: str, timeframe: str) -> RawBar:
        """Convert an IBKR BarData to RawBar.

        Args:
            ibkr_bar: IBKR BarData object.
            symbol: Symbol name.
            timeframe: Timeframe string.

        Returns:
            RawBar instance.
        """
        # ib_insync BarData has: date, open, high, low, close, volume, etc.
        # IBKR returns -1 for volume when not available (common for CFDs)
        raw_volume = ibkr_bar.volume  # type: ignore[attr-defined]
        volume = None if raw_volume is None or raw_volume < 0 else Decimal(str(raw_volume))

        return RawBar(
            timestamp=ibkr_bar.date,  # type: ignore[attr-defined]
            open=Decimal(str(ibkr_bar.open)),  # type: ignore[attr-defined]
            high=Decimal(str(ibkr_bar.high)),  # type: ignore[attr-defined]
            low=Decimal(str(ibkr_bar.low)),  # type: ignore[attr-defined]
            close=Decimal(str(ibkr_bar.close)),  # type: ignore[attr-defined]
            volume=volume,
            symbol=symbol,
            timeframe=timeframe,
        )
