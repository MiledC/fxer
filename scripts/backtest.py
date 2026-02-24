"""Backtest signal generator on historical data.

Usage:
    .venv/bin/python -m scripts.backtest --symbol XAUUSD --start 2024-06-01 --end 2025-01-01
    .venv/bin/python -m scripts.backtest --symbol XAUUSD --start 2024-06-01 --no-regime
    .venv/bin/python -m scripts.backtest --symbol XAUUSD --start 2024-06-01 --spread 0.40
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from fxer.backtesting import BacktestEngine
from fxer.config.settings import Settings
from fxer.core.exceptions import ModelLoadError, NoDataError
from fxer.core.types import Timeframe
from fxer.data.storage.questdb_client import QuestDBClient

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Backtest signal generator on historical data",
        prog="python -m scripts.backtest",
    )
    parser.add_argument(
        "--symbol",
        "-s",
        required=True,
        help="Trading symbol (e.g. XAUUSD)",
    )
    parser.add_argument(
        "--start",
        required=True,
        help="Start date (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)",
    )
    parser.add_argument(
        "--end",
        help="End date (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS). Defaults to now.",
    )
    parser.add_argument(
        "--timeframe",
        "-t",
        default="5m",
        help="Bar timeframe (default: 5m)",
    )
    parser.add_argument(
        "--spread",
        type=float,
        default=0.30,
        help="Spread cost in price units (default: 0.30)",
    )
    parser.add_argument(
        "--no-regime",
        action="store_true",
        help="Disable regime classification",
    )
    parser.add_argument(
        "--model-dir",
        help="Override model directory (default: from settings)",
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


def _format_number(value: float, decimals: int = 2, suffix: str = "") -> str:
    """Format a number for display."""
    if decimals == 0:
        return f"{int(value):,}{suffix}"
    return f"{value:.{decimals}f}{suffix}"


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    args = parse_args(argv)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Parse arguments
    symbol = args.symbol.upper()
    timeframe_str = args.timeframe
    timeframe = Timeframe.from_string(timeframe_str)

    start = _parse_datetime(args.start)
    end = _parse_datetime(args.end) if args.end else datetime.now(timezone.utc)

    # Initialize settings
    settings = Settings()
    if args.model_dir:
        settings.signal_model_dir = args.model_dir

    # Query historical bars
    print(f"Loading {symbol} {timeframe_str} bars [{start.date()} → {end.date()}]...")

    db_client = QuestDBClient(settings)
    try:
        bars = db_client.query_bars(symbol, timeframe, start, end)
    except Exception as exc:
        logger.error("Failed to query bars: %s", exc)
        print(f"Error: Failed to query bars from QuestDB: {exc}")
        return 1

    if not bars:
        print(f"Error: No bars found for {symbol} in the specified date range")
        return 1

    print(f"  Loaded {len(bars):,} bars")

    # Check for training data overlap (warn if backtesting on training period)
    train_result_path = Path(settings.signal_model_dir) / symbol.lower() / timeframe_str / "train_result.json"
    if train_result_path.exists():
        try:
            with open(train_result_path) as f:
                train_result = json.load(f)
            train_start = datetime.fromisoformat(train_result.get("train_start", ""))
            train_end = datetime.fromisoformat(train_result.get("train_end", ""))

            # Check for overlap
            if start <= train_end and end >= train_start:
                print(
                    f"  ⚠️  WARNING: Backtest period overlaps with training data "
                    f"[{train_start.date()} → {train_end.date()}]"
                )
        except Exception as exc:
            logger.warning("Could not check training overlap: %s", exc)

    # Run backtest
    print(f"\nRunning backtest...")
    print(f"  Spread: {args.spread:.2f} (per-trade)")
    print(f"  Regime filter: {'Enabled' if not args.no_regime else 'Disabled'}")

    try:
        engine = BacktestEngine(
            settings=settings,
            spread=args.spread,
            use_regime=not args.no_regime,
            model_dir=settings.signal_model_dir,
        )
        result = engine.run(bars, symbol, timeframe_str)
    except ModelLoadError as exc:
        print(f"Error: {exc}")
        print(f"  Make sure to train the model first using: python -m scripts.train_signals")
        return 1
    except Exception as exc:
        logger.exception("Backtest failed")
        print(f"Error: Backtest failed: {exc}")
        return 1

    # Print results
    print(f"\nBacktest: {symbol} {timeframe_str} [{start.date()} → {end.date()}]")
    print(f"  Bars processed: {result.bars_processed:,}")
    print(f"  Signals generated: {result.signals_generated:,}")
    print(f"  Trades executed: {result.trades_executed}")
    print(f"  Spread: {args.spread:.2f} (per-trade)")

    print("\n--- Performance ---")
    print(f"  Total Return:    {result.total_return:+.2f}%")
    print(f"  Sharpe Ratio:    {result.sharpe_ratio:.2f}")
    print(f"  Sortino Ratio:   {result.sortino_ratio:.2f}")
    print(f"  Profit Factor:   {result.profit_factor:.2f}")
    print(f"  Win Rate:        {result.win_rate:.1f}%")
    print(f"  Max Drawdown:    {result.max_drawdown:.1f}%")
    print(f"  Meets Minimum:   {'YES' if result.meets_minimum else 'NO'}")

    # Print direction breakdown
    print("\n--- Direction Breakdown ---")
    if result.long_metrics:
        lm = result.long_metrics
        print(
            f"  LONG   {lm.trade_count:4d} trades, "
            f"{lm.win_count:4d} wins ({lm.win_rate:5.1f}%), "
            f"PF {lm.profit_factor:5.2f}, Return {lm.total_return:+.2f}%"
        )
    else:
        print("  LONG      0 trades")

    if result.short_metrics:
        sm = result.short_metrics
        print(
            f"  SHORT  {sm.trade_count:4d} trades, "
            f"{sm.win_count:4d} wins ({sm.win_rate:5.1f}%), "
            f"PF {sm.profit_factor:5.2f}, Return {sm.total_return:+.2f}%"
        )
    else:
        print("  SHORT     0 trades")

    # Print regime breakdown if available
    if result.regime_breakdown:
        print("\n--- Regime Breakdown ---")
        for regime_name, metrics in result.regime_breakdown.items():
            print(
                f"  {regime_name:15s}  {metrics.trade_count:3d} trades, "
                f"{metrics.win_rate:5.1f}% win, Sharpe {metrics.sharpe_ratio:.2f}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())