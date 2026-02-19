"""Train the signal ensemble model on historical QuestDB data.

Runs walk-forward cross-validation and saves the final model artifacts.

Usage:
    python -m scripts.train_signals --symbol XAUUSD --start 2024-01-01 --end 2025-01-01
    python -m scripts.train_signals --symbol XAUUSD --start 2024-01-01 --folds 5 --log-level DEBUG
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone

from fxer.config.settings import Settings
from fxer.core.types import Timeframe
from fxer.signals.training.trainer import Trainer

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the signal ensemble model",
        prog="python -m scripts.train_signals",
    )
    parser.add_argument(
        "--symbol", "-s",
        required=True,
        help="Trading symbol (e.g. XAUUSD)",
    )
    parser.add_argument(
        "--start",
        required=True,
        help="Training start date (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)",
    )
    parser.add_argument(
        "--end",
        help="Training end date (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS). Defaults to now.",
    )
    parser.add_argument(
        "--timeframe", "-t",
        default="5m",
        help="Bar timeframe (default: 5m)",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of walk-forward CV folds (default: 5)",
    )
    parser.add_argument(
        "--model-dir",
        help="Override model save directory (default: from settings)",
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
    timeframe_str = args.timeframe
    timeframe = Timeframe.from_string(timeframe_str)

    start = _parse_datetime(args.start)
    end = _parse_datetime(args.end) if args.end else datetime.now(timezone.utc)

    settings = Settings()
    if args.model_dir:
        settings.signal_model_dir = args.model_dir

    print(f"Training {symbol} {timeframe_str} [{start.date()} \u2192 {end.date()}]")
    print(f"  Folds: {args.folds}")
    print(f"  Model dir: {settings.signal_model_dir}")

    try:
        trainer = Trainer(settings)
        result = trainer.train(symbol, timeframe, start, end, n_folds=args.folds)
    except Exception as exc:
        logger.exception("Training failed: %s", exc)
        print(f"\nTraining failed: {exc}")
        return 1

    # --- Per-fold metrics table ---
    print()
    header = f"{'Fold':>6}  {'Sharpe':>8}  {'Win Rate':>9}  {'Drawdown':>9}  {'Accuracy':>9}"
    print(header)
    print("-" * len(header))

    for fr in result.fold_results:
        m = fr.metrics
        print(
            f"{fr.fold_index + 1:>6}  "
            f"{m.sharpe_ratio:>8.2f}  "
            f"{m.win_rate:>9.2f}  "
            f"{m.max_drawdown:>9.2f}  "
            f"{m.accuracy:>9.2f}"
        )

    # --- Aggregate summary ---
    print()
    print("--- Summary ---")
    print(f"  Mean Sharpe:     {result.mean_sharpe:.2f}")
    print(f"  Mean Win Rate:   {result.mean_win_rate:.2f}")
    print(f"  Worst Drawdown:  {result.worst_drawdown:.2f}")
    print(f"  Model saved to:  {result.model_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
