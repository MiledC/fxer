"""Run the signal generator as a live service on the EventBus.

Subscribes to feature events, generates trade signals via the ensemble
model, and publishes them back to the bus.

Usage:
    python -m scripts.generate_signals --symbol XAUUSD
    python -m scripts.generate_signals --symbol XAUUSD --timeframe 5m --log-level DEBUG
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys

from fxer.config.settings import Settings
from fxer.messaging.bus import EventBus
from fxer.signals.generator import SignalGenerator

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the signal generator service",
        prog="python -m scripts.generate_signals",
    )
    parser.add_argument(
        "--symbol", "-s",
        required=True,
        help="Trading symbol (e.g. XAUUSD)",
    )
    parser.add_argument(
        "--timeframe", "-t",
        default="5m",
        help="Model timeframe (default: 5m)",
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


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    symbol = args.symbol.upper()
    timeframe = args.timeframe

    settings = Settings()
    if args.model_dir:
        settings.signal_model_dir = args.model_dir

    model_path = f"{settings.signal_model_dir}/{symbol.lower()}/{timeframe}"
    print(f"Loading model from {model_path}")

    bus = EventBus(settings)
    generator = SignalGenerator(event_bus=bus, settings=settings)

    # Setup shutdown event
    shutdown_event = asyncio.Event()

    def handle_signal(sig: signal.Signals) -> None:
        print(f"\nReceived {sig.name}, shutting down...")
        shutdown_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, handle_signal, sig)

    try:
        generator.load_model(symbol, timeframe)

        await bus.start()
        print(f"EventBus started on {bus.pub_address}")

        await generator.start()
        print("SignalGenerator started, waiting for features...")
        print("[Ctrl+C to stop]")

        await shutdown_event.wait()

    except Exception as exc:
        logger.exception("Signal generator failed: %s", exc)
        print(f"\nError: {exc}")
        return 1

    finally:
        print("\nCleaning up...")
        await generator.stop()
        await bus.stop()

    print("Shutdown complete.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
