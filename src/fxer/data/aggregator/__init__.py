"""Bar aggregation for creating higher timeframes."""

from fxer.data.aggregator.bar_aggregator import (
    AGGREGATION_MAP,
    BarAggregator,
    aggregate,
    aggregate_batch,
)

__all__ = [
    "AGGREGATION_MAP",
    "BarAggregator",
    "aggregate",
    "aggregate_batch",
]
