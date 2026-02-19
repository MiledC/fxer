"""Bar aggregation: combines lower-timeframe bars into higher timeframes.

Supports both batch aggregation (list of bars in, aggregated bars out)
and streaming aggregation (feed bars one-at-a-time with callbacks on completion).
"""

from collections import defaultdict
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from fxer.core.events import NormalizedBar
from fxer.core.exceptions import DataValidationError
from fxer.core.types import Timeframe

# Valid source -> target aggregation mappings and required bar counts.
AGGREGATION_MAP: dict[tuple[Timeframe, Timeframe], int] = {
    (Timeframe.M5, Timeframe.M15): 3,
    (Timeframe.M5, Timeframe.H1): 12,
    (Timeframe.M5, Timeframe.H4): 48,
    (Timeframe.M1, Timeframe.M5): 5,
    (Timeframe.M1, Timeframe.M15): 15,
    (Timeframe.M1, Timeframe.H1): 60,
    (Timeframe.M15, Timeframe.H1): 4,
    (Timeframe.M15, Timeframe.H4): 16,
    (Timeframe.H1, Timeframe.H4): 4,
}


def _validate_aggregation_pair(
    source: Timeframe, target: Timeframe
) -> int:
    """Validate that source -> target is a supported aggregation and return bar count."""
    key = (source, target)
    if key not in AGGREGATION_MAP:
        raise DataValidationError(
            message=f"Unsupported aggregation: {source.value} -> {target.value}",
            field="timeframe",
            value=f"{source.value}->{target.value}",
            rule="aggregation_pair",
        )
    return AGGREGATION_MAP[key]


def _align_timestamp(ts: datetime, target: Timeframe) -> datetime:
    """Align a timestamp to the start of its containing target-timeframe period.

    For example, 00:05 aligned to H1 -> 00:00, 00:35 aligned to M15 -> 00:30.
    """
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    minutes_since_midnight = ts.hour * 60 + ts.minute
    target_minutes = target.minutes
    aligned_minute = (minutes_since_midnight // target_minutes) * target_minutes
    return ts.replace(
        hour=aligned_minute // 60,
        minute=aligned_minute % 60,
        second=0,
        microsecond=0,
    )


def aggregate(
    bars: list[NormalizedBar],
    target_timeframe: Timeframe,
) -> NormalizedBar:
    """Aggregate a list of bars into a single higher-timeframe bar.

    Args:
        bars: Source bars, must be non-empty, same symbol and timeframe,
              sorted by timestamp ascending.
        target_timeframe: The desired output timeframe.

    Returns:
        A single NormalizedBar at the target timeframe.

    Raises:
        DataValidationError: If bars are empty, mixed symbols/timeframes,
            or the aggregation pair is unsupported.
    """
    if not bars:
        raise DataValidationError(
            message="Cannot aggregate empty bar list",
            field="bars",
            rule="non_empty",
        )

    source_tf = bars[0].timeframe
    symbol = bars[0].symbol
    bars_needed = _validate_aggregation_pair(source_tf, target_timeframe)

    # Validate homogeneity
    for bar in bars:
        if bar.symbol != symbol:
            raise DataValidationError(
                message=f"Mixed symbols in aggregation: {symbol} vs {bar.symbol}",
                field="symbol",
                value=bar.symbol,
                rule="homogeneous_symbol",
            )
        if bar.timeframe != source_tf:
            raise DataValidationError(
                message=f"Mixed timeframes in aggregation: {source_tf.value} vs {bar.timeframe.value}",
                field="timeframe",
                value=bar.timeframe.value,
                rule="homogeneous_timeframe",
            )

    is_complete = len(bars) == bars_needed
    aligned_ts = _align_timestamp(bars[0].timestamp, target_timeframe)

    return NormalizedBar(
        symbol=symbol,
        timeframe=target_timeframe,
        timestamp=aligned_ts,
        open=bars[0].open,
        high=max(b.high for b in bars),
        low=min(b.low for b in bars),
        close=bars[-1].close,
        volume=sum((b.volume for b in bars), Decimal("0")),
        is_complete=is_complete,
    )


def aggregate_batch(
    bars: list[NormalizedBar],
    target_timeframe: Timeframe,
) -> list[NormalizedBar]:
    """Aggregate a sequence of bars into multiple higher-timeframe bars.

    Bars are grouped by their aligned target-timeframe period, then each
    group is aggregated individually.

    Args:
        bars: Source bars (same symbol and timeframe), sorted by timestamp.
        target_timeframe: The desired output timeframe.

    Returns:
        List of aggregated NormalizedBar instances, sorted by timestamp.
    """
    if not bars:
        return []

    # Group bars by their aligned target-timeframe bucket
    groups: dict[datetime, list[NormalizedBar]] = defaultdict(list)
    for bar in bars:
        key = _align_timestamp(bar.timestamp, target_timeframe)
        groups[key].append(bar)

    result = []
    for key in sorted(groups):
        result.append(aggregate(groups[key], target_timeframe))
    return result


# Type alias for bar callbacks.
BarCallback = Callable[[NormalizedBar], None]


class BarAggregator:
    """Streaming bar aggregator with rolling buffer.

    Feed 5M (or other source-timeframe) bars one at a time; when enough
    bars accumulate to form a complete higher-timeframe bar, registered
    callbacks are invoked.

    Handles gaps: if a bar arrives that belongs to a new period while the
    previous period is incomplete, the incomplete period is flushed
    (with is_complete=False) before starting the new one.
    """

    def __init__(
        self,
        source_timeframe: Timeframe,
        target_timeframe: Timeframe,
    ) -> None:
        self._source_tf = source_timeframe
        self._target_tf = target_timeframe
        self._bars_needed = _validate_aggregation_pair(source_timeframe, target_timeframe)
        self._buffer: list[NormalizedBar] = []
        self._current_period: datetime | None = None
        self._callbacks: list[BarCallback] = []

    @property
    def source_timeframe(self) -> Timeframe:
        return self._source_tf

    @property
    def target_timeframe(self) -> Timeframe:
        return self._target_tf

    @property
    def bars_needed(self) -> int:
        return self._bars_needed

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    def on_bar(self, callback: BarCallback) -> None:
        """Register a callback invoked when a target-timeframe bar completes."""
        self._callbacks.append(callback)

    def push(self, bar: NormalizedBar) -> NormalizedBar | None:
        """Push a source-timeframe bar into the aggregator.

        Returns the completed aggregated bar if the push caused a flush,
        otherwise None.
        """
        if bar.timeframe != self._source_tf:
            raise DataValidationError(
                message=(
                    f"Expected {self._source_tf.value} bar, "
                    f"got {bar.timeframe.value}"
                ),
                field="timeframe",
                value=bar.timeframe.value,
                rule="source_timeframe",
            )

        period = _align_timestamp(bar.timestamp, self._target_tf)
        result: NormalizedBar | None = None

        # If we moved to a new period, flush the old buffer
        if self._current_period is not None and period != self._current_period:
            result = self._flush()

        self._current_period = period
        self._buffer.append(bar)

        # If buffer is full, flush a complete bar
        if len(self._buffer) == self._bars_needed:
            result = self._flush()

        return result

    def flush(self) -> NormalizedBar | None:
        """Manually flush the current buffer (e.g. at end of session).

        Returns the incomplete aggregated bar, or None if buffer is empty.
        """
        if not self._buffer:
            return None
        return self._flush()

    def reset(self) -> None:
        """Clear the buffer and period state."""
        self._buffer.clear()
        self._current_period = None

    def _flush(self) -> NormalizedBar:
        """Aggregate buffered bars, notify callbacks, and clear the buffer."""
        agg = aggregate(self._buffer, self._target_tf)
        self._buffer.clear()
        self._current_period = None
        for cb in self._callbacks:
            cb(agg)
        return agg
