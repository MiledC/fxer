"""Stream live bars from IBKR.

Connects to Interactive Brokers and streams real-time bar data
to the EventBus for downstream processing.

Usage:
    # Stream all configured symbols
    python -m scripts.ibkr_stream

    # Stream specific symbols
    python -m scripts.ibkr_stream --symbols XAUUSD,EURUSD

    # Stream with verbose logging
    python -m scripts.ibkr_stream --symbols XAUUSD --log-level DEBUG

Prerequisites:
    - IB Gateway or TWS running on localhost
    - Market data subscriptions for desired instruments
    - API connections enabled in TWS/Gateway settings
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from datetime import datetime

from fxer.config.settings import Settings
from fxer.config.symbols import get_symbol_config
from fxer.data.loaders.ibkr_client import IBKRClient, IBKRConnectionError
from fxer.data.loaders.ibkr_streamer import IBKRStreamer
from fxer.messaging.bus import EventBus

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream live bars from IBKR",
        prog="python -m scripts.ibkr_stream",
    )
    parser.add_argument(
        "--symbols",
        "-s",
        help="Comma-separated symbols to stream (default: from settings)",
    )
    parser.add_argument(
        "--no-bus",
        action="store_true",
        help="Disable EventBus publishing (log only)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    return parser.parse_args(argv)


async def print_status(streamer: IBKRStreamer, interval: float = 60.0) -> None:
    """Periodically print streaming status."""
    start_time = datetime.now()

    while True:
        await asyncio.sleep(interval)

        elapsed = datetime.now() - start_time
        elapsed_min = elapsed.total_seconds() / 60

        symbols = streamer.subscribed_symbols
        counts = {s: streamer.get_bar_count(s) for s in symbols}

        print(f"\n[{elapsed_min:.1f}m] Streaming status:")
        for symbol, count in counts.items():
            rate = count / elapsed_min if elapsed_min > 0 else 0
            print(f"  {symbol}: {count} bars ({rate:.1f}/min)")


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    settings = Settings()

    # Determine symbols to stream
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

    print("IBKR Real-Time Streaming")
    print(f"  Symbols: {', '.join(symbols)}")
    print(f"  EventBus: {'disabled' if args.no_bus else 'enabled'}")

    # Setup shutdown event
    shutdown_event = asyncio.Event()

    def handle_signal(sig: signal.Signals) -> None:
        print(f"\nReceived {sig.name}, shutting down...")
        shutdown_event.set()

    # Register signal handlers
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, handle_signal, sig)

    bus: EventBus | None = None
    streamer: IBKRStreamer | None = None
    status_task: asyncio.Task[None] | None = None

    try:
        # Connect to IBKR
        print(f"\nConnecting to IBKR at {settings.ibkr_host}:{settings.ibkr_port}...")
        client = IBKRClient(settings)
        await client.connect()
        print("Connected to IBKR")

        # Start EventBus if enabled
        if not args.no_bus:
            bus = EventBus(settings)
            await bus.start()
            print(f"EventBus started on {bus.pub_address}")

        # Create streamer
        streamer = IBKRStreamer(
            client=client,
            bus=bus if bus else EventBus(settings),  # Dummy bus if disabled
        )

        # Subscribe to symbols
        print("\nSubscribing to symbols...")
        for symbol in symbols:
            await streamer.subscribe(symbol)

        print(f"\nStreaming {len(symbols)} symbol(s). Press Ctrl+C to stop.")

        # Start status reporting task
        status_task = asyncio.create_task(print_status(streamer))

        # Wait for shutdown signal
        await shutdown_event.wait()

    except IBKRConnectionError as exc:
        print(f"\nConnection error: {exc}")
        print("\nMake sure:")
        print("  1. IB Gateway or TWS is running")
        print("  2. API connections are enabled")
        print(f"  3. Port {settings.ibkr_port} is configured correctly")
        return 1

    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        return 1

    finally:
        # Cleanup
        print("\nCleaning up...")

        if status_task:
            status_task.cancel()
            try:
                await status_task
            except asyncio.CancelledError:
                pass

        if streamer:
            await streamer.unsubscribe_all()

        if bus:
            await bus.stop()

        if client and client.is_connected:
            await client.disconnect()

        # Print final stats
        if streamer:
            print("\nFinal statistics:")
            for symbol in symbols:
                count = streamer.get_bar_count(symbol)
                print(f"  {symbol}: {count} bars received")

    print("Shutdown complete.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
