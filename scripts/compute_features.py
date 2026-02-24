"""Batch feature computation from stored bars.

Reads bars from QuestDB (or CSV), computes technical and temporal features,
and optionally stores them back to QuestDB or exports to CSV.

Usage:
    python -m scripts.compute_features --symbol XAUUSD --timeframe 5m \
        --start 2024-01-01 --end 2024-01-31
    python -m scripts.compute_features --symbol XAUUSD --timeframe 5m \
        --start 2024-01-01 --end 2024-01-31 --export features_out.csv
    python -m scripts.compute_features --csv data/xauusd_5m.csv --symbol XAUUSD
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from fxer.config.settings import Settings
from fxer.core.events import FeatureVector, NormalizedBar
from fxer.core.types import Timeframe
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

FEATURE_CSV_COLUMNS = [
    "symbol", "timeframe", "timestamp",
    "rsi_14", "rsi_7",
    "macd_line", "macd_signal", "macd_histogram",
    "bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_percent_b",
    "atr_14",
    "return_1bar", "return_5bar", "return_12bar",
    "rolling_volatility_20", "momentum_48",
    "is_london_session", "is_ny_session", "is_overlap_session", "is_asian_session",
    "hour_of_day", "day_of_week", "is_month_turn",
    "warmup_complete",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute features from stored bars",
        prog="python -m scripts.compute_features",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--csv",
        help="Load bars from a CSV file instead of QuestDB",
    )
    source.add_argument(
        "--start",
        help="Start date for QuestDB query (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)",
    )
    parser.add_argument(
        "--end",
        help="End date for QuestDB query (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)",
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
        "--export", "-e",
        help="Export features to CSV file",
    )
    parser.add_argument(
        "--store",
        action="store_true",
        help="Store computed features into QuestDB",
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


def _load_bars_from_csv(csv_path: str, symbol: str, timeframe: str) -> list[NormalizedBar]:
    """Load and normalize bars from a CSV file."""
    loader = CSVLoader()
    stats = loader.load(csv_path, symbol=symbol)
    print(f"  CSV: {stats.rows_loaded} rows loaded, {stats.rows_skipped} skipped")

    normalizer = BarNormalizer(
        validators=[validate_price_bounds, validate_ohlc_consistency, validate_volume],
        default_symbol=symbol,
        default_timeframe=timeframe,
    )

    bars = []
    for raw_bar in loader.iter_bars():
        try:
            bars.append(normalizer.normalize_bar(raw_bar))
        except Exception as exc:
            logger.debug("Skipping bar: %s", exc)
    return bars


def _load_bars_from_questdb(
    symbol: str, timeframe: Timeframe, start: datetime, end: datetime
) -> list[NormalizedBar]:
    """Load bars from QuestDB."""
    settings = Settings()
    with QuestDBClient(settings) as client:
        bars = client.query_bars(symbol, timeframe, start, end)
    return bars


def _export_to_csv(features: list[FeatureVector], path: str) -> None:
    """Write feature vectors to a CSV file."""
    p = Path(path)
    with p.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FEATURE_CSV_COLUMNS)
        writer.writeheader()
        for fv in features:
            row = {
                "symbol": fv.symbol,
                "timeframe": fv.timeframe.value,
                "timestamp": fv.timestamp.isoformat(),
                "rsi_14": fv.rsi_14,
                "rsi_7": fv.rsi_7,
                "macd_line": fv.macd_line,
                "macd_signal": fv.macd_signal,
                "macd_histogram": fv.macd_histogram,
                "bb_upper": fv.bb_upper,
                "bb_middle": fv.bb_middle,
                "bb_lower": fv.bb_lower,
                "bb_width": fv.bb_width,
                "bb_percent_b": fv.bb_percent_b,
                "atr_14": fv.atr_14,
                "return_1bar": fv.return_1bar,
                "return_5bar": fv.return_5bar,
                "return_12bar": fv.return_12bar,
                "rolling_volatility_20": fv.rolling_volatility_20,
                "momentum_48": fv.momentum_48,
                "is_london_session": fv.is_london_session,
                "is_ny_session": fv.is_ny_session,
                "is_overlap_session": fv.is_overlap_session,
                "is_asian_session": fv.is_asian_session,
                "hour_of_day": fv.hour_of_day,
                "day_of_week": fv.day_of_week,
                "is_month_turn": fv.is_month_turn,
                "warmup_complete": fv.warmup_complete,
            }
            writer.writerow(row)
    print(f"  Exported {len(features)} feature vectors to {p}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    symbol = args.symbol.upper()
    timeframe_str = args.timeframe
    timeframe = Timeframe.from_string(timeframe_str)

    # --- Load bars ---
    if args.csv:
        print(f"Loading bars from CSV: {args.csv}")
        bars = _load_bars_from_csv(args.csv, symbol, timeframe_str)
    else:
        start = _parse_datetime(args.start)
        if args.end:
            end = _parse_datetime(args.end)
        else:
            end = datetime.now(timezone.utc)
        print(f"Querying bars from QuestDB: {symbol} {timeframe_str} [{start} -> {end}]")
        bars = _load_bars_from_questdb(symbol, timeframe, start, end)

    print(f"  Loaded {len(bars)} bars")

    if not bars:
        print("No bars to process. Exiting.")
        return 1

    # --- Load cross-asset bars (DXY_SYNTH + VIX) if computing for XAUUSD ---
    enricher: CrossAssetEnricher | None = None
    dxy_lookup: dict[datetime, float] = {}
    vix_lookup: dict[datetime, float] = {}

    if symbol == "XAUUSD" and not args.csv:
        print("\nLoading cross-asset bars (DXY_SYNTH + VIX)...")
        bar_start = bars[0].timestamp
        bar_end = bars[-1].timestamp
        settings_ca = Settings()
        with QuestDBClient(settings_ca) as ca_client:
            dxy_bars = ca_client.query_bars("DXY_SYNTH", timeframe, bar_start, bar_end)
            vix_bars = ca_client.query_bars("VIX", timeframe, bar_start, bar_end)
        dxy_lookup = {b.timestamp: float(b.close) for b in dxy_bars}
        vix_lookup = {b.timestamp: float(b.close) for b in vix_bars}
        print(f"  DXY_SYNTH bars: {len(dxy_lookup)}")
        print(f"  VIX bars:       {len(vix_lookup)}")
        enricher = CrossAssetEnricher()

    # --- Compute features ---
    print(f"\nComputing features for {len(bars)} bars...")
    engine = FeatureEngine(cross_asset=enricher)
    features: list[FeatureVector] = []

    for i, bar in enumerate(bars):
        # Feed cross-asset closes aligned by timestamp
        if enricher is not None:
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
    print(f"  Warmup pending:  {len(features) - warmup_count}")

    # --- Store to QuestDB ---
    if args.store:
        settings = Settings()
        print(f"\nStoring features to QuestDB at {settings.questdb_host}...")
        with QuestDBClient(settings) as client:
            client.init_tables()
            for i, fv in enumerate(features):
                client.insert_features(fv)
                done = i + 1
                if done % 500 == 0 or done == len(features):
                    print(f"  Stored {done}/{len(features)} features", end="\r")
            print(f"\n  Stored {len(features)} feature vectors.")

    # --- Export to CSV ---
    if args.export:
        print(f"\nExporting features to {args.export}...")
        _export_to_csv(features, args.export)

    # --- Summary ---
    print("\n--- Summary ---")
    print(f"  Symbol:          {symbol}")
    print(f"  Timeframe:       {timeframe_str}")
    print(f"  Bars processed:  {len(bars)}")
    print(f"  Features:        {len(features)}")
    print(f"  Warmup complete: {warmup_count}")
    if args.store:
        print(f"  Stored to QuestDB: yes")
    if args.export:
        print(f"  Exported to: {args.export}")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
