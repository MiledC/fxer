"""Train the HMM regime classifier on historical daily OHLC data.

Queries 5m bars from QuestDB, aggregates to daily using SAMPLE BY 1d,
trains the HMM, evaluates regime distribution, and saves the model to disk.

Usage:
    python -m scripts.train_regime_hmm --symbol XAUUSD --start 2020-01-01
    python -m scripts.train_regime_hmm --symbol XAUUSD --start 2020-01-01 --end 2025-01-01 --n-states 3
    python -m scripts.train_regime_hmm --symbol XAUUSD --start 2020-01-01 --covariance diag --log-level DEBUG
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from fxer.config.settings import Settings
from fxer.data.storage.questdb_client import QuestDBClient
from fxer.regime.hmm import HMMRegimeClassifier
from fxer.regime.types import RegimeState

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the HMM regime classifier on daily OHLC data",
        prog="python -m scripts.train_regime_hmm",
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
        "--n-states",
        type=int,
        default=3,
        help="Number of HMM states (default: 3)",
    )
    parser.add_argument(
        "--covariance",
        default="diag",
        choices=["diag", "full", "spherical", "tied"],
        help="Covariance type (default: diag)",
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
    start = _parse_datetime(args.start)
    end = _parse_datetime(args.end) if args.end else datetime.now(timezone.utc)

    settings = Settings()
    model_dir = Path(args.model_dir) if args.model_dir else Path(settings.regime_model_dir)
    save_path = model_dir / symbol.lower() / "regime"

    print(f"Training HMM regime classifier for {symbol} [{start.date()} \u2192 {end.date()}]")
    print(f"  States: {args.n_states}, Covariance: {args.covariance}")

    # --- Query daily bars ---
    print("\nQuerying daily bars from QuestDB...")
    try:
        client = QuestDBClient(settings)
        daily_bars = client.query_daily_bars(symbol, start, end)
    except Exception as exc:
        logger.exception("Failed to query daily bars: %s", exc)
        print(f"\nFailed to query daily bars: {exc}")
        return 1

    if len(daily_bars) < 50:
        print(f"\nInsufficient data: {len(daily_bars)} daily bars (need >= 50)")
        return 1

    print(f"  Daily bars: {len(daily_bars):,}")
    print(f"  Date range: {daily_bars[0].timestamp.date()} \u2192 {daily_bars[-1].timestamp.date()}")

    # --- Convert to DataFrame ---
    daily_df = pd.DataFrame({
        "close": [float(b.close) for b in daily_bars],
        "high": [float(b.high) for b in daily_bars],
        "low": [float(b.low) for b in daily_bars],
    })

    # --- Train HMM ---
    print("\nTraining HMM...")
    try:
        hmm = HMMRegimeClassifier(
            n_states=args.n_states,
            covariance_type=args.covariance,
        )
        metrics = hmm.fit(daily_df)
    except Exception as exc:
        logger.exception("Training failed: %s", exc)
        print(f"\nTraining failed: {exc}")
        return 1

    print(f"  Log-likelihood: {metrics['log_likelihood']:.2f}")
    print(f"  Samples used: {metrics['n_samples']:,} (after feature warmup)")

    # --- Regime distribution on training data ---
    features = hmm._extract_features(daily_df)
    _, state_probs = hmm._model.score_samples(features)
    predicted_states = np.argmax(state_probs, axis=1)

    # Feature layout: [0]=realized_vol, [1]=autocorrelation
    print("\n--- Regime Distribution ---")
    header = f"  {'State':<20} {'Mean Vol':>10} {'Mean AC':>10} {'Count':>7} {'Pct':>7}"
    print(header)

    for hmm_state_idx in range(args.n_states):
        mask = predicted_states == hmm_state_idx
        count = int(mask.sum())
        pct = 100.0 * count / len(predicted_states) if len(predicted_states) > 0 else 0.0
        mean_vol = float(features[mask, 0].mean()) if count > 0 else 0.0
        mean_ac = float(features[mask, 1].mean()) if count > 0 else 0.0
        regime_label = hmm._state_map.get(hmm_state_idx, RegimeState.RANGING)
        print(f"  {regime_label.value:<20} {mean_vol:>10.4f} {mean_ac:>10.4f} {count:>7} {pct:>6.1f}%")

    # --- Transition matrix ---
    transmat = hmm._model.transmat_
    state_labels = []
    for i in range(args.n_states):
        regime = hmm._state_map.get(i, RegimeState.RANGING)
        state_labels.append(regime.value)

    col_width = max(len(label) for label in state_labels) + 2
    print("\n--- State Transition Matrix ---")
    header_row = f"  {'From \\\\ To':<{col_width}}"
    for label in state_labels:
        header_row += f" {label:>{col_width}}"
    print(header_row)

    for i in range(args.n_states):
        row_str = f"  {state_labels[i]:<{col_width}}"
        for j in range(args.n_states):
            row_str += f" {transmat[i, j]:>{col_width}.2f}"
        print(row_str)

    # --- Save model ---
    try:
        hmm.save(save_path)
    except Exception as exc:
        logger.exception("Failed to save model: %s", exc)
        print(f"\nFailed to save model: {exc}")
        return 1

    print(f"\nModel saved to: {save_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
