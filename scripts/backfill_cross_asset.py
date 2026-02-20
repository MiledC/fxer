"""Backfill cross-asset features (DXY + VIX) for existing XAUUSD data.

Recomputes full feature vectors for XAUUSD bars that already exist in
QuestDB, this time including cross-asset enrichment from DXY_SYNTH and
VIX bars.  The QuestDB DEDUP UPSERT ensures rows are overwritten cleanly.

Usage:
    python -m scripts.backfill_cross_asset --symbol XAUUSD \
        --start 2025-01-01 --end 2026-01-01
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone

from fxer.config.settings import Settings
from fxer.core.events import FeatureVector
from fxer.core.types import Timeframe
from fxer.data.storage.questdb_client import QuestDBClient
from fxer.features.cross_asset import CrossAssetEnricher
from fxer.features.engine import FeatureEngine

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill cross-asset features for existing bars",
        prog="python -m scripts.backfill_cross_asset",
    )
    parser.add_argument(
        "--symbol", "-s",
        default="XAUUSD",
        help="Trading symbol (default: XAUUSD)",
    )
    parser.add_argument(
        "--timeframe", "-t",
        default="5m",
        help="Bar timeframe (default: 5m)",
    )
    parser.add_argument(
        "--start",
        required=True,
        help="Start date (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="End date (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    return parser.parse_args(argv)


def _parse_datetime(value: str) -> datetime:
    """Parse a date/datetime string into a timezone-aware datetime."""
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(value, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date: {value}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    symbol = args.symbol.upper()
    timeframe = Timeframe.from_string(args.timeframe)
    start = _parse_datetime(args.start).replace(tzinfo=None)
    end = _parse_datetime(args.end).replace(tzinfo=None)

    settings = Settings()

    print(f"Backfilling cross-asset features for {symbol} {args.timeframe}")
    print(f"  Range: {start} -> {end}")

    with QuestDBClient(settings) as client:
        # Ensure schema is up to date (adds new columns if needed)
        client.init_tables()

        # Step 1: Load XAUUSD bars
        print(f"\nQuerying {symbol} bars...")
        bars = client.query_bars(symbol, timeframe, start, end)
        print(f"  {symbol} bars: {len(bars)}")

        if not bars:
            print("No bars found. Exiting.")
            return 1

        # Step 2: Load DXY_SYNTH + VIX bars for the same range
        print("Querying cross-asset bars...")
        dxy_bars = client.query_bars("DXY_SYNTH", timeframe, start, end)
        vix_bars = client.query_bars("VIX", timeframe, start, end)

        dxy_lookup = {b.timestamp: float(b.close) for b in dxy_bars}
        vix_lookup = {b.timestamp: float(b.close) for b in vix_bars}
        print(f"  DXY_SYNTH bars: {len(dxy_lookup)}")
        print(f"  VIX bars:       {len(vix_lookup)}")

        # Step 3: Compute features with cross-asset enrichment
        print(f"\nComputing features for {len(bars)} bars...")
        enricher = CrossAssetEnricher()
        engine = FeatureEngine(cross_asset=enricher)

        features: list[FeatureVector] = []
        for i, bar in enumerate(bars):
            ts = bar.timestamp
            if ts in dxy_lookup:
                enricher.update_dxy(dxy_lookup[ts])
            if ts in vix_lookup:
                enricher.update_vix(vix_lookup[ts])

            fv = engine.compute_features(bar)
            features.append(fv)

            done = i + 1
            if done % 1000 == 0 or done == len(bars):
                print(f"  Computed {done}/{len(bars)} features", end="\r")

        warmup_count = sum(1 for f in features if f.warmup_complete)
        print(f"\n  Total features:  {len(features)}")
        print(f"  Warmup complete: {warmup_count}")

        # Step 4: Upsert features back to QuestDB
        print(f"\nStoring features to QuestDB...")
        for i, fv in enumerate(features):
            client.insert_features(fv)
            done = i + 1
            if done % 500 == 0 or done == len(features):
                print(f"  Stored {done}/{len(features)} features", end="\r")

        # Count features with cross-asset data
        has_cross = sum(
            1 for f in features
            if f.dxy_return_1h is not None or f.vix_level is not None
        )

        print(f"\n\n--- Backfill Summary ---")
        print(f"  Symbol:              {symbol}")
        print(f"  Timeframe:           {args.timeframe}")
        print(f"  Bars processed:      {len(bars)}")
        print(f"  Features stored:     {len(features)}")
        print(f"  With cross-asset:    {has_cross}")
        print(f"  Warmup complete:     {warmup_count}")
        print("Done.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
