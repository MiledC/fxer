"""CLI script to load CSV bar data, normalize it, and insert into QuestDB.

Usage:
    python -m scripts.load_csv_data --file data/xauusd_5m.csv --symbol XAUUSD
    python -m scripts.load_csv_data --file data/xauusd_5m.csv --symbol XAUUSD --timeframe 5m
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from fxer.config.settings import Settings
from fxer.core.exceptions import DataValidationError
from fxer.data.loaders.csv_loader import CSVLoader, ColumnMapping
from fxer.data.normalizer.normalizer import BarNormalizer
from fxer.data.normalizer.validators import (
    validate_ohlc_consistency,
    validate_price_bounds,
    validate_volume,
)
from fxer.data.storage.questdb_client import QuestDBClient

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load CSV bar data into QuestDB",
        prog="python -m scripts.load_csv_data",
    )
    parser.add_argument(
        "--file", "-f",
        required=True,
        help="Path to the CSV file",
    )
    parser.add_argument(
        "--symbol", "-s",
        required=True,
        help="Trading symbol (e.g. XAUUSD)",
    )
    parser.add_argument(
        "--timeframe", "-t",
        default="5m",
        help="Bar timeframe (default: 5m)",
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=500,
        help="Number of bars per insert batch (default: 500)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and normalize without inserting into QuestDB",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    csv_path = Path(args.file)
    symbol = args.symbol.upper()
    timeframe = args.timeframe
    batch_size = args.batch_size

    # --- Load CSV ---
    print(f"Loading CSV: {csv_path}")
    loader = CSVLoader()
    stats = loader.load(csv_path, symbol=symbol)

    print(f"  Rows loaded:  {stats.rows_loaded}")
    print(f"  Rows skipped: {stats.rows_skipped}")
    if stats.start_date and stats.end_date:
        print(f"  Date range:   {stats.start_date} -> {stats.end_date}")

    if stats.errors:
        print(f"\n  Parse errors ({len(stats.errors)}):")
        for err in stats.errors[:10]:
            print(f"    - {err}")
        if len(stats.errors) > 10:
            print(f"    ... and {len(stats.errors) - 10} more")

    if stats.rows_loaded == 0:
        print("No data loaded. Exiting.")
        return 1

    # --- Normalize ---
    # Use validators that work for historical data (skip timestamp checks)
    normalizer = BarNormalizer(
        validators=[validate_price_bounds, validate_ohlc_consistency, validate_volume],
        default_symbol=symbol,
        default_timeframe=timeframe,
    )

    normalized_bars = []
    normalize_errors: list[str] = []

    print(f"\nNormalizing {stats.rows_loaded} bars...")
    for i, raw_bar in enumerate(loader.iter_bars()):
        try:
            nbar = normalizer.normalize_bar(raw_bar)
            normalized_bars.append(nbar)
        except DataValidationError as exc:
            normalize_errors.append(f"Bar {i + 1}: {exc.message}")

        # Progress reporting
        done = i + 1
        if done % 1000 == 0 or done == stats.rows_loaded:
            print(f"  Normalized {done}/{stats.rows_loaded} bars", end="\r")

    print(f"\n  Normalized: {len(normalized_bars)} bars")
    if normalize_errors:
        print(f"  Normalization errors: {len(normalize_errors)}")
        for err in normalize_errors[:10]:
            print(f"    - {err}")
        if len(normalize_errors) > 10:
            print(f"    ... and {len(normalize_errors) - 10} more")

    if not normalized_bars:
        print("No bars after normalization. Exiting.")
        return 1

    # --- Insert into QuestDB ---
    if args.dry_run:
        print("\n[DRY RUN] Skipping QuestDB insert.")
        print("Done.")
        return 0

    settings = Settings()
    print(f"\nConnecting to QuestDB at {settings.questdb_host}:{settings.questdb_ilp_port}...")

    with QuestDBClient(settings) as client:
        client.init_tables()

        total = len(normalized_bars)
        inserted = 0

        for start in range(0, total, batch_size):
            batch = normalized_bars[start : start + batch_size]
            client.insert_bars(batch)
            inserted += len(batch)
            print(f"  Inserted {inserted}/{total} bars", end="\r")

        print(f"\n  Inserted {inserted} bars into QuestDB.")

    # --- Summary ---
    print("\n--- Summary ---")
    print(f"  File:             {csv_path}")
    print(f"  Symbol:           {symbol}")
    print(f"  Timeframe:        {timeframe}")
    print(f"  CSV rows parsed:  {stats.rows_loaded}")
    print(f"  CSV rows skipped: {stats.rows_skipped}")
    print(f"  Bars normalized:  {len(normalized_bars)}")
    print(f"  Normalize errors: {len(normalize_errors)}")
    if not args.dry_run:
        print(f"  Bars inserted:    {inserted}")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
