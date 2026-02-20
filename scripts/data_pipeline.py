"""Unified data pipeline: CSV → Normalize → Aggregate → Store → Features

Complete pipeline that:
1. Reads 5M bars from CSV
2. Normalizes bars
3. Aggregates to 15M, 1H, 4H
4. Stores ALL bars (5M, 15M, 1H, 4H) to QuestDB
5. Computes features for ALL timeframes
6. Stores all features to QuestDB

Usage:
    python -m scripts.data_pipeline --file data/xauusd_5m.csv --symbol XAUUSD
    python -m scripts.data_pipeline --file data/xauusd_5m.csv --symbol XAUUSD --dry-run
    python -m scripts.data_pipeline --file data/xauusd_5m.csv --symbol XAUUSD --no-features
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from fxer.config.settings import Settings
from fxer.core.events import FeatureVector, NormalizedBar
from fxer.core.exceptions import DataValidationError
from fxer.core.types import Timeframe
from fxer.data.aggregator.bar_aggregator import aggregate_batch
from fxer.data.loaders.csv_loader import CSVLoader
from fxer.data.normalizer.normalizer import BarNormalizer
from fxer.data.normalizer.validators import (
    validate_ohlc_consistency,
    validate_price_bounds,
    validate_volume,
)
from fxer.data.storage.questdb_client import QuestDBClient
from fxer.features.cross_asset import CrossAssetEnricher
from fxer.features.engine import FeatureEngine

logger = logging.getLogger(__name__)


@dataclass
class TimeframeSummary:
    """Summary statistics for a single timeframe."""

    bars_count: int = 0
    features_count: int = 0
    warmup_bars: int = 0


@dataclass
class PipelineSummary:
    """Complete pipeline execution summary."""

    source_file: str
    symbol: str
    csv_rows_loaded: int = 0
    csv_rows_skipped: int = 0
    normalize_errors: int = 0
    timeframes: dict[Timeframe, TimeframeSummary] | None = None

    def __post_init__(self) -> None:
        if self.timeframes is None:
            self.timeframes = {}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified data pipeline: CSV → Normalize → Aggregate → Store → Features",
        prog="python -m scripts.data_pipeline",
    )
    parser.add_argument(
        "--file", "-f",
        required=True,
        help="Path to 5M CSV file",
    )
    parser.add_argument(
        "--symbol", "-s",
        required=True,
        help="Trading symbol (e.g. XAUUSD)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip QuestDB writes (normalize and aggregate only)",
    )
    parser.add_argument(
        "--no-features",
        action="store_true",
        help="Skip feature computation",
    )
    parser.add_argument(
        "--batch-size", "-b",
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


def load_and_normalize(
    csv_path: Path,
    symbol: str,
    summary: PipelineSummary,
) -> list[NormalizedBar]:
    """Load CSV and normalize bars to 5M timeframe."""
    print(f"Loading CSV: {csv_path}")
    loader = CSVLoader()
    stats = loader.load(csv_path, symbol=symbol)

    summary.csv_rows_loaded = stats.rows_loaded
    summary.csv_rows_skipped = stats.rows_skipped

    print(f"  Rows loaded:  {stats.rows_loaded}")
    print(f"  Rows skipped: {stats.rows_skipped}")
    if stats.start_date and stats.end_date:
        print(f"  Date range:   {stats.start_date} -> {stats.end_date}")

    if stats.errors:
        print(f"\n  Parse errors ({len(stats.errors)}):")
        for err in stats.errors[:5]:
            print(f"    - {err}")
        if len(stats.errors) > 5:
            print(f"    ... and {len(stats.errors) - 5} more")

    if stats.rows_loaded == 0:
        return []

    # Normalize to 5M bars
    normalizer = BarNormalizer(
        validators=[validate_price_bounds, validate_ohlc_consistency, validate_volume],
        default_symbol=symbol,
        default_timeframe="5m",
    )

    normalized_bars: list[NormalizedBar] = []
    normalize_errors: list[str] = []

    print(f"\nNormalizing {stats.rows_loaded} bars to 5M...")
    for i, raw_bar in enumerate(loader.iter_bars()):
        try:
            nbar = normalizer.normalize_bar(raw_bar)
            normalized_bars.append(nbar)
        except DataValidationError as exc:
            normalize_errors.append(f"Bar {i + 1}: {exc.message}")

        done = i + 1
        if done % 1000 == 0 or done == stats.rows_loaded:
            print(f"  Normalized {done}/{stats.rows_loaded} bars", end="\r")

    print(f"\n  Normalized: {len(normalized_bars)} bars")
    summary.normalize_errors = len(normalize_errors)

    if normalize_errors:
        print(f"  Normalization errors: {len(normalize_errors)}")
        for err in normalize_errors[:5]:
            print(f"    - {err}")
        if len(normalize_errors) > 5:
            print(f"    ... and {len(normalize_errors) - 5} more")

    return normalized_bars


def aggregate_timeframes(
    bars_5m: list[NormalizedBar],
) -> dict[Timeframe, list[NormalizedBar]]:
    """Aggregate 5M bars to higher timeframes."""
    print("\nAggregating to higher timeframes...")

    all_bars: dict[Timeframe, list[NormalizedBar]] = {Timeframe.M5: bars_5m}

    # Aggregate 5M -> 15M
    bars_15m = aggregate_batch(bars_5m, Timeframe.M15)
    all_bars[Timeframe.M15] = bars_15m
    print(f"  5M → 15M: {len(bars_5m)} → {len(bars_15m)} bars")

    # Aggregate 5M -> 1H
    bars_1h = aggregate_batch(bars_5m, Timeframe.H1)
    all_bars[Timeframe.H1] = bars_1h
    print(f"  5M → 1H:  {len(bars_5m)} → {len(bars_1h)} bars")

    # Aggregate 5M -> 4H
    bars_4h = aggregate_batch(bars_5m, Timeframe.H4)
    all_bars[Timeframe.H4] = bars_4h
    print(f"  5M → 4H:  {len(bars_5m)} → {len(bars_4h)} bars")

    return all_bars


def store_bars(
    client: QuestDBClient,
    all_bars: dict[Timeframe, list[NormalizedBar]],
    batch_size: int,
    summary: PipelineSummary,
) -> None:
    """Store all bars to QuestDB."""
    print("\nStoring bars to QuestDB...")

    for tf in [Timeframe.M5, Timeframe.M15, Timeframe.H1, Timeframe.H4]:
        bars = all_bars.get(tf, [])
        if not bars:
            continue

        total = len(bars)
        inserted = 0

        for start in range(0, total, batch_size):
            batch = bars[start : start + batch_size]
            client.insert_bars(batch)
            inserted += len(batch)
            print(f"  {tf.value}: Inserted {inserted}/{total} bars", end="\r")

        print(f"  {tf.value}: {inserted} bars stored")

        if summary.timeframes is not None:
            if tf not in summary.timeframes:
                summary.timeframes[tf] = TimeframeSummary()
            summary.timeframes[tf].bars_count = inserted


def compute_and_store_features(
    client: QuestDBClient,
    all_bars: dict[Timeframe, list[NormalizedBar]],
    summary: PipelineSummary,
    symbol: str = "",
) -> None:
    """Compute and store features for all timeframes."""
    print("\nComputing and storing features...")

    for tf in [Timeframe.M5, Timeframe.M15, Timeframe.H1, Timeframe.H4]:
        bars = all_bars.get(tf, [])
        if not bars:
            continue

        # Build cross-asset enricher for XAUUSD
        enricher: CrossAssetEnricher | None = None
        dxy_lookup: dict = {}
        vix_lookup: dict = {}

        if symbol == "XAUUSD":
            bar_start = bars[0].timestamp
            bar_end = bars[-1].timestamp
            dxy_bars = client.query_bars("DXY_SYNTH", tf, bar_start, bar_end)
            vix_bars = client.query_bars("VIX", tf, bar_start, bar_end)
            dxy_lookup = {b.timestamp: float(b.close) for b in dxy_bars}
            vix_lookup = {b.timestamp: float(b.close) for b in vix_bars}
            enricher = CrossAssetEnricher()
            print(f"  {tf.value}: Cross-asset bars — DXY: {len(dxy_lookup)}, VIX: {len(vix_lookup)}")

        # Fresh engine for each timeframe (independent indicator state)
        engine = FeatureEngine(cross_asset=enricher)
        warmup_bars = engine.get_warmup_bars()

        features_stored = 0
        total = len(bars)

        for i, bar in enumerate(bars):
            # Feed cross-asset closes aligned by timestamp
            if enricher is not None:
                ts = bar.timestamp
                if ts in dxy_lookup:
                    enricher.update_dxy(dxy_lookup[ts])
                if ts in vix_lookup:
                    enricher.update_vix(vix_lookup[ts])

            fv = engine.compute_features(bar)
            client.insert_features(fv)
            features_stored += 1

            if (i + 1) % 500 == 0 or (i + 1) == total:
                print(f"  {tf.value}: Computed {i + 1}/{total} features", end="\r")

        print(f"  {tf.value}: {features_stored} features stored (warmup: {warmup_bars})")

        if summary.timeframes is not None:
            if tf not in summary.timeframes:
                summary.timeframes[tf] = TimeframeSummary()
            summary.timeframes[tf].features_count = features_stored
            summary.timeframes[tf].warmup_bars = warmup_bars


def print_summary(summary: PipelineSummary, dry_run: bool, no_features: bool) -> None:
    """Print final pipeline summary."""
    print("\n" + "=" * 40)
    print("=== Data Pipeline Summary ===")
    print("=" * 40)
    print(f"Source: {summary.source_file}")
    print(f"Symbol: {summary.symbol}")
    print()

    print("Bars:")
    if summary.timeframes:
        for tf in [Timeframe.M5, Timeframe.M15, Timeframe.H1, Timeframe.H4]:
            if tf in summary.timeframes:
                ts = summary.timeframes[tf]
                status = "stored" if not dry_run else "(dry run)"
                print(f"  {tf.value:4s}: {ts.bars_count:6d} bars {status}")

    if not no_features:
        print()
        print("Features:")
        if summary.timeframes:
            for tf in [Timeframe.M5, Timeframe.M15, Timeframe.H1, Timeframe.H4]:
                if tf in summary.timeframes:
                    ts = summary.timeframes[tf]
                    if dry_run:
                        # In dry run, show expected feature count (same as bar count)
                        print(f"  {tf.value:4s}: {ts.bars_count:6d} features (dry run)")
                    else:
                        print(f"  {tf.value:4s}: {ts.features_count:6d} features (warmup: {ts.warmup_bars})")

    print()
    if dry_run:
        print("[DRY RUN] No data written to QuestDB.")
    print("Done.")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    csv_path = Path(args.file)
    symbol = args.symbol.upper()

    summary = PipelineSummary(
        source_file=str(csv_path),
        symbol=symbol,
    )

    # Step 1: Load and normalize CSV to 5M bars
    bars_5m = load_and_normalize(csv_path, symbol, summary)
    if not bars_5m:
        print("No bars after loading/normalization. Exiting.")
        return 1

    # Step 2: Aggregate to higher timeframes
    all_bars = aggregate_timeframes(bars_5m)

    # Initialize summary for all timeframes
    for tf, bars in all_bars.items():
        if summary.timeframes is not None:
            summary.timeframes[tf] = TimeframeSummary(bars_count=len(bars))

    # Step 3 & 4: Store bars and compute features
    if args.dry_run:
        print("\n[DRY RUN] Skipping QuestDB operations.")
        print_summary(summary, dry_run=True, no_features=args.no_features)
        return 0

    settings = Settings()
    print(f"\nConnecting to QuestDB at {settings.questdb_host}:{settings.questdb_ilp_port}...")

    with QuestDBClient(settings) as client:
        client.init_tables()

        # Store all bars
        store_bars(client, all_bars, args.batch_size, summary)

        # Compute and store features
        if not args.no_features:
            compute_and_store_features(client, all_bars, summary, symbol=symbol)

    print_summary(summary, dry_run=False, no_features=args.no_features)
    return 0


if __name__ == "__main__":
    sys.exit(main())
