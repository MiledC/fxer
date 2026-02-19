"""Calculate synthetic DXY from stored forex pairs.

Queries the 6 DXY component forex pairs from QuestDB, aligns them by
timestamp, calculates synthetic DXY, and stores the result.

Prerequisites:
    All 6 forex pairs must be backfilled first:
    python -m scripts.ibkr_backfill --symbols EURUSD,GBPUSD,USDJPY,USDCAD,USDSEK,USDCHF --start 2025-01-01

Usage:
    # Calculate DXY from stored data
    python -m scripts.dxy_synth_calculate --start 2025-01-01

    # Dry run (no QuestDB writes)
    python -m scripts.dxy_synth_calculate --start 2025-01-01 --dry-run

    # Skip aggregation to higher timeframes
    python -m scripts.dxy_synth_calculate --start 2025-01-01 --no-aggregate
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone

from fxer.config.settings import Settings
from fxer.core.events import NormalizedBar
from fxer.core.types import Timeframe
from fxer.data.aggregator.bar_aggregator import aggregate_batch
from fxer.data.storage.questdb_client import QuestDBClient
from fxer.data.synthetic.dxy_calculator import (
    DXY_COMPONENTS,
    calculate_dxy_bar,
)

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate synthetic DXY from stored forex pairs",
        prog="python -m scripts.dxy_synth_calculate",
    )
    parser.add_argument(
        "--start",
        "-s",
        type=lambda s: datetime.fromisoformat(s).replace(tzinfo=timezone.utc),
        required=True,
        help="Start date (ISO format: 2025-01-01)",
    )
    parser.add_argument(
        "--end",
        "-e",
        type=lambda s: datetime.fromisoformat(s).replace(tzinfo=timezone.utc),
        default=None,
        help="End date (ISO format, default: now)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip QuestDB writes (calculate only)",
    )
    parser.add_argument(
        "--no-aggregate",
        action="store_true",
        help="Skip aggregation to higher timeframes (15m, 1h, 4h)",
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


def load_forex_bars(
    client: QuestDBClient,
    start: datetime,
    end: datetime,
) -> dict[str, list[NormalizedBar]]:
    """Load all 6 forex pairs from QuestDB.

    Args:
        client: QuestDB client.
        start: Start timestamp.
        end: End timestamp.

    Returns:
        Dict of {symbol: list[NormalizedBar]} for all 6 DXY components.
    """
    # QuestDB doesn't support timestamptz - convert to naive UTC
    start_naive = start.replace(tzinfo=None) if start.tzinfo else start
    end_naive = end.replace(tzinfo=None) if end.tzinfo else end

    bars_by_symbol: dict[str, list[NormalizedBar]] = {}

    for symbol in DXY_COMPONENTS:
        bars = client.query_bars(symbol, Timeframe.M5, start_naive, end_naive)
        bars_by_symbol[symbol] = bars
        print(f"  {symbol}: {len(bars)} bars")

    return bars_by_symbol


def align_bars_by_timestamp(
    bars_by_symbol: dict[str, list[NormalizedBar]],
) -> list[dict[str, NormalizedBar]]:
    """Align bars by timestamp (inner join).

    Only returns timestamps where all 6 forex pairs have data.

    Args:
        bars_by_symbol: Dict of {symbol: list[NormalizedBar]}.

    Returns:
        List of dicts, each containing aligned bars for one timestamp.
    """
    # Index bars by timestamp for each symbol
    bars_by_ts: dict[str, dict[datetime, NormalizedBar]] = {}
    for symbol, bars in bars_by_symbol.items():
        bars_by_ts[symbol] = {bar.timestamp: bar for bar in bars}

    # Find timestamps present in all symbols (inner join)
    all_timestamps: set[datetime] | None = None
    for symbol in DXY_COMPONENTS:
        symbol_ts = set(bars_by_ts[symbol].keys())
        if all_timestamps is None:
            all_timestamps = symbol_ts
        else:
            all_timestamps &= symbol_ts

    if not all_timestamps:
        return []

    # Build aligned bar groups
    aligned: list[dict[str, NormalizedBar]] = []
    for ts in sorted(all_timestamps):
        bar_group = {symbol: bars_by_ts[symbol][ts] for symbol in DXY_COMPONENTS}
        aligned.append(bar_group)

    return aligned


def calculate_synthetic_dxy(
    aligned_bars: list[dict[str, NormalizedBar]],
) -> list[NormalizedBar]:
    """Calculate synthetic DXY for all aligned bar groups.

    Args:
        aligned_bars: List of aligned bar dicts from align_bars_by_timestamp.

    Returns:
        List of synthetic DXY bars.
    """
    dxy_bars: list[NormalizedBar] = []

    for bar_group in aligned_bars:
        try:
            dxy_bar = calculate_dxy_bar(bar_group, Timeframe.M5)
            dxy_bars.append(dxy_bar)
        except Exception as e:
            ts = next(iter(bar_group.values())).timestamp
            logger.warning("Failed to calculate DXY for %s: %s", ts, e)

    return dxy_bars


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    settings = Settings()
    start_date = args.start
    end_date = args.end or datetime.now(timezone.utc)

    print("Synthetic DXY Calculator")
    print(f"  Range:     {start_date.date()} to {end_date.date()}")
    print(f"  Dry run:   {args.dry_run}")
    print()

    client = QuestDBClient(settings)

    # Step 1: Load forex bars
    print("Loading forex pairs from QuestDB...")
    bars_by_symbol = load_forex_bars(client, start_date, end_date)

    # Check all pairs have data
    missing = [s for s in DXY_COMPONENTS if not bars_by_symbol.get(s)]
    if missing:
        print(f"\nError: Missing data for forex pairs: {', '.join(missing)}")
        print("\nRun the following to backfill:")
        print(f"  python -m scripts.ibkr_backfill --symbols {','.join(missing)} --start {start_date.date()}")
        return 1

    # Step 2: Align bars by timestamp
    print("\nAligning bars by timestamp...")
    aligned_bars = align_bars_by_timestamp(bars_by_symbol)
    print(f"  Aligned timestamps: {len(aligned_bars)}")

    if not aligned_bars:
        print("\nNo aligned data found. Ensure all forex pairs have overlapping data.")
        return 1

    # Show coverage stats
    total_bars = sum(len(bars) for bars in bars_by_symbol.values())
    print(f"  Total bars across pairs: {total_bars}")
    print(f"  Coverage: {len(aligned_bars) * 6} / {total_bars} ({100 * len(aligned_bars) * 6 / total_bars:.1f}%)")

    # Step 3: Calculate synthetic DXY
    print("\nCalculating synthetic DXY...")
    dxy_5m_bars = calculate_synthetic_dxy(aligned_bars)
    print(f"  Generated {len(dxy_5m_bars)} DXY_SYNTH 5m bars")

    if not dxy_5m_bars:
        print("\nNo DXY bars calculated.")
        return 1

    # Verify DXY values are in expected range
    closes = [float(b.close) for b in dxy_5m_bars]
    dxy_min, dxy_max = min(closes), max(closes)
    dxy_avg = sum(closes) / len(closes)
    print(f"  DXY range: {dxy_min:.3f} - {dxy_max:.3f} (avg: {dxy_avg:.3f})")

    if dxy_min < 50 or dxy_max > 150:
        print(f"  Warning: DXY values outside expected 50-150 range")

    # Step 4: Aggregate to higher timeframes
    all_bars: dict[Timeframe, list[NormalizedBar]] = {Timeframe.M5: dxy_5m_bars}

    if not args.no_aggregate:
        print("\nAggregating to higher timeframes...")
        for target_tf in [Timeframe.M15, Timeframe.H1, Timeframe.H4]:
            aggregated = aggregate_batch(dxy_5m_bars, target_tf)
            all_bars[target_tf] = aggregated
            print(f"  5m -> {target_tf.value}: {len(dxy_5m_bars)} -> {len(aggregated)} bars")

    # Step 5: Store in QuestDB
    if args.dry_run:
        print("\n[DRY RUN] Skipping QuestDB writes")
    else:
        print("\nStoring DXY_SYNTH bars in QuestDB...")
        for tf, bars in all_bars.items():
            if not bars:
                continue

            total = len(bars)
            for start_idx in range(0, total, args.batch_size):
                batch = bars[start_idx : start_idx + args.batch_size]
                client.insert_bars(batch)

            print(f"  Stored {total} {tf.value} bars")

    # Summary
    print("\n" + "=" * 40)
    print("SUMMARY")
    print("=" * 40)
    print(f"  Date range: {dxy_5m_bars[0].timestamp} to {dxy_5m_bars[-1].timestamp}")
    print(f"  DXY range:  {dxy_min:.3f} - {dxy_max:.3f}")
    for tf, bars in all_bars.items():
        status = "stored" if not args.dry_run else "(dry run)"
        print(f"  {tf.value}: {len(bars)} bars {status}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
