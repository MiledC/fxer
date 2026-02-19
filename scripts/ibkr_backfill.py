"""Backfill historical data from IBKR.

Fetches historical bar data from Interactive Brokers and stores it
in QuestDB, following the same pipeline as CSV data.

Usage:
    # Backfill 1 year of data
    python -m scripts.ibkr_backfill --symbols XAUUSD --start 2025-01-01

    # Backfill specific date range
    python -m scripts.ibkr_backfill --symbols XAUUSD --start 2025-01-01 --end 2025-06-01

    # Backfill with specific timeframe
    python -m scripts.ibkr_backfill --symbols XAUUSD --timeframe 5m --start 2025-01-01

    # Dry run (no QuestDB writes)
    python -m scripts.ibkr_backfill --symbols XAUUSD --start 2025-01-01 --dry-run

Prerequisites:
    - IB Gateway or TWS running on localhost
    - Market data subscriptions for desired instruments
    - API connections enabled in TWS/Gateway settings
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone

from fxer.config.settings import Settings
from fxer.config.symbols import get_symbol_config
from fxer.core.events import NormalizedBar
from fxer.core.exceptions import DataValidationError
from fxer.core.types import Timeframe
from fxer.data.aggregator.bar_aggregator import aggregate_batch
from fxer.data.loaders.ibkr_client import IBKRClient, IBKRConnectionError
from fxer.data.loaders.ibkr_loader import IBKRLoader
from fxer.data.normalizer.normalizer import BarNormalizer
from fxer.data.normalizer.validators import (
    validate_ohlc_consistency,
    validate_price_bounds,
    validate_volume,
)
from fxer.data.storage.questdb_client import QuestDBClient
from fxer.features.engine import FeatureEngine

logger = logging.getLogger(__name__)


@dataclass
class BackfillSummary:
    """Summary of a backfill operation for one symbol."""

    symbol: str
    bars_loaded: int = 0
    bars_normalized: int = 0
    normalize_errors: int = 0
    timeframes_stored: dict[str, int] | None = None
    features_stored: dict[str, int] | None = None

    def __post_init__(self) -> None:
        if self.timeframes_stored is None:
            self.timeframes_stored = {}
        if self.features_stored is None:
            self.features_stored = {}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill historical data from IBKR",
        prog="python -m scripts.ibkr_backfill",
    )
    parser.add_argument(
        "--symbols",
        help="Comma-separated symbols to backfill (default: from settings)",
    )
    parser.add_argument(
        "--start",
        "-s",
        type=lambda s: datetime.fromisoformat(s).replace(tzinfo=timezone.utc),
        required=True,
        help="Start date (ISO format: 2024-01-01)",
    )
    parser.add_argument(
        "--end",
        "-e",
        type=lambda s: datetime.fromisoformat(s).replace(tzinfo=timezone.utc),
        default=None,
        help="End date (ISO format, default: now)",
    )
    parser.add_argument(
        "--timeframe",
        "-t",
        default="5m",
        help='Base timeframe to fetch (e.g., "5m", "1h"). Default: "5m"',
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip QuestDB writes (fetch and normalize only)",
    )
    parser.add_argument(
        "--no-features",
        action="store_true",
        help="Skip feature computation",
    )
    parser.add_argument(
        "--no-aggregate",
        action="store_true",
        help="Skip aggregation to higher timeframes",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=500,
        help="Insert batch size (default: 500)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    return parser.parse_args(argv)


def normalize_bars(
    loader: IBKRLoader,
    symbol: str,
    timeframe: str,
    summary: BackfillSummary,
) -> list[NormalizedBar]:
    """Normalize raw bars from IBKR loader."""
    normalizer = BarNormalizer(
        validators=[validate_price_bounds, validate_ohlc_consistency, validate_volume],
        default_symbol=symbol,
        default_timeframe=timeframe,
    )

    normalized: list[NormalizedBar] = []
    errors: list[str] = []

    for i, raw_bar in enumerate(loader.iter_bars()):
        try:
            nbar = normalizer.normalize_bar(raw_bar)
            normalized.append(nbar)
        except DataValidationError as exc:
            errors.append(f"Bar {i + 1}: {exc.message}")

    summary.bars_normalized = len(normalized)
    summary.normalize_errors = len(errors)

    if errors:
        print(f"  Normalization errors: {len(errors)}")
        for err in errors[:3]:
            print(f"    - {err}")
        if len(errors) > 3:
            print(f"    ... and {len(errors) - 3} more")

    return normalized


def aggregate_and_store(
    client: QuestDBClient,
    bars_5m: list[NormalizedBar],
    summary: BackfillSummary,
    batch_size: int,
    no_aggregate: bool,
) -> dict[Timeframe, list[NormalizedBar]]:
    """Aggregate bars and store to QuestDB."""
    all_bars: dict[Timeframe, list[NormalizedBar]] = {Timeframe.M5: bars_5m}

    if not no_aggregate:
        # Aggregate to higher timeframes
        for target_tf in [Timeframe.M15, Timeframe.H1, Timeframe.H4]:
            aggregated = aggregate_batch(bars_5m, target_tf)
            all_bars[target_tf] = aggregated
            print(f"  5M -> {target_tf.value}: {len(bars_5m)} -> {len(aggregated)} bars")

    # Store all bars
    for tf, bars in all_bars.items():
        if not bars:
            continue

        total = len(bars)
        for start in range(0, total, batch_size):
            batch = bars[start : start + batch_size]
            client.insert_bars(batch)

        if summary.timeframes_stored is not None:
            summary.timeframes_stored[tf.value] = total
        print(f"  Stored {total} {tf.value} bars")

    return all_bars


def compute_features(
    client: QuestDBClient,
    all_bars: dict[Timeframe, list[NormalizedBar]],
    summary: BackfillSummary,
) -> None:
    """Compute and store features for all timeframes."""
    for tf, bars in all_bars.items():
        if not bars:
            continue

        engine = FeatureEngine()
        count = 0

        for bar in bars:
            fv = engine.compute_features(bar)
            client.insert_features(fv)
            count += 1

        if summary.features_stored is not None:
            summary.features_stored[tf.value] = count
        print(f"  Computed {count} {tf.value} features")


async def backfill_symbol(
    client: IBKRClient,
    questdb: QuestDBClient | None,
    symbol: str,
    timeframe: Timeframe,
    start_date: datetime,
    end_date: datetime,
    args: argparse.Namespace,
) -> BackfillSummary:
    """Backfill a single symbol."""
    summary = BackfillSummary(symbol=symbol)

    print(f"\n{'=' * 40}")
    print(f"Backfilling {symbol}")
    print(f"{'=' * 40}")

    # Fetch historical data
    loader = IBKRLoader(client)
    stats = await loader.load(symbol, timeframe, start_date=start_date, end_date=end_date)

    summary.bars_loaded = stats.rows_loaded
    print(f"  Loaded {stats.rows_loaded} bars")

    if stats.start_date and stats.end_date:
        print(f"  Date range: {stats.start_date} -> {stats.end_date}")

    if stats.rows_loaded == 0:
        print("  No data fetched, skipping")
        return summary

    # Normalize
    print(f"\nNormalizing {stats.rows_loaded} bars...")
    bars_5m = normalize_bars(loader, symbol, timeframe.value, summary)

    if not bars_5m:
        print("  No valid bars after normalization")
        return summary

    # Store and aggregate
    if questdb is not None and not args.dry_run:
        print("\nAggregating and storing bars...")
        all_bars = aggregate_and_store(
            questdb, bars_5m, summary, args.batch_size, args.no_aggregate
        )

        # Compute features
        if not args.no_features:
            print("\nComputing features...")
            compute_features(questdb, all_bars, summary)
    else:
        print("\n[DRY RUN] Skipping QuestDB operations")
        if summary.timeframes_stored is not None:
            summary.timeframes_stored[timeframe.value] = len(bars_5m)

    return summary


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    settings = Settings()

    # Determine symbols to backfill
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        symbols = settings.ibkr_symbols

    # Validate symbols
    for symbol in symbols:
        try:
            get_symbol_config(symbol)
        except ValueError as exc:
            print(f"Error: {exc}")
            return 1

    timeframe = Timeframe.from_string(args.timeframe)
    start_date = args.start
    end_date = args.end or datetime.now(timezone.utc)

    print("IBKR Historical Backfill")
    print(f"  Symbols:   {', '.join(symbols)}")
    print(f"  Timeframe: {timeframe.value}")
    print(f"  Range:     {start_date.date()} to {end_date.date()}")
    print(f"  Dry run:   {args.dry_run}")

    summaries: list[BackfillSummary] = []

    try:
        async with IBKRClient(settings) as ibkr:
            # Initialize QuestDB if not dry run
            questdb: QuestDBClient | None = None
            if not args.dry_run:
                questdb = QuestDBClient(settings)
                questdb.init_tables()

            try:
                for symbol in symbols:
                    summary = await backfill_symbol(
                        ibkr, questdb, symbol, timeframe, start_date, end_date, args
                    )
                    summaries.append(summary)
            finally:
                if questdb is not None:
                    questdb.close()

    except IBKRConnectionError as exc:
        print(f"\nConnection error: {exc}")
        print("\nMake sure:")
        print("  1. IB Gateway or TWS is running")
        print("  2. API connections are enabled")
        print(f"  3. Port {settings.ibkr_port} is configured correctly")
        return 1

    # Print final summary
    print("\n" + "=" * 40)
    print("BACKFILL SUMMARY")
    print("=" * 40)

    for s in summaries:
        print(f"\n{s.symbol}:")
        print(f"  Bars loaded:      {s.bars_loaded}")
        print(f"  Bars normalized:  {s.bars_normalized}")
        if s.normalize_errors:
            print(f"  Normalize errors: {s.normalize_errors}")
        if s.timeframes_stored:
            for tf, count in s.timeframes_stored.items():
                status = "stored" if not args.dry_run else "(dry run)"
                print(f"  {tf}: {count} bars {status}")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
