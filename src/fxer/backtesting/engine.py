"""Main backtesting engine orchestrating the full replay loop."""

from __future__ import annotations

import logging
from pathlib import Path

from fxer.backtesting.metrics import compute_regime_breakdown, compute_trade_metrics
from fxer.backtesting.tracker import TradeTracker
from fxer.backtesting.types import BacktestResult
from fxer.config.settings import Settings, settings as default_settings
from fxer.core.events import NormalizedBar
from fxer.core.exceptions import ModelLoadError
from fxer.features.engine import FeatureEngine
from fxer.regime.classifier import RegimeClassifier
from fxer.signals.base import feature_vector_to_array
from fxer.signals.models.ensemble import StackingEnsemble
from fxer.signals.types import Direction, HoldingPeriod

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Orchestrates the full backtest replay loop.

    Replays historical bars through the signal pipeline:
    FeatureEngine → RegimeClassifier → StackingEnsemble → TradeSignal

    Execution model (no lookahead):
    - Signals are generated at bar[t] close using only data available at time t.
    - Trades are entered at bar[t+1] open (the next bar's opening price).
    - Positions are exited at bar close after predicted_horizon bars.
    - One position at a time; new signals are skipped while a position is open.

    Tracks realistic P&L with spread costs and position limits.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        spread: float = 0.30,
        use_regime: bool = True,
        model_dir: str | None = None,
    ) -> None:
        """Initialize the backtest engine.

        Args:
            settings: Application settings.
            spread: Spread cost in price units (e.g. 0.30 USD).
            use_regime: Whether to use regime classification.
            model_dir: Override for model directory.
        """
        self._settings = settings or default_settings
        self._spread = spread
        self._use_regime = use_regime
        self._model_dir = model_dir or self._settings.signal_model_dir

    def run(
        self,
        bars: list[NormalizedBar],
        symbol: str,
        timeframe: str = "5m",
    ) -> BacktestResult:
        """Run the full backtest replay loop.

        Args:
            bars: Historical bars to replay.
            symbol: Trading symbol.
            timeframe: Bar timeframe.

        Returns:
            Complete backtest results.

        Raises:
            ModelLoadError: If required models cannot be loaded.
        """
        if not bars:
            return BacktestResult(
                symbol=symbol,
                timeframe=timeframe,
                start_date=None,
                end_date=None,
                bars_processed=0,
                signals_generated=0,
                trades_executed=0,
                spread_pct=self._spread,
                total_return=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                profit_factor=0.0,
                win_rate=0.0,
                max_drawdown=0.0,
                meets_minimum=False,
                regime_breakdown={},
            )

        logger.info(
            "Starting backtest: %s %s, %d bars, spread=%.2f",
            symbol,
            timeframe,
            len(bars),
            self._spread,
        )

        # Initialize components
        feature_engine = FeatureEngine(cross_asset=None)  # No cross-asset for simplicity

        # Load ensemble model
        model_path = Path(self._model_dir) / symbol.lower() / timeframe
        if not model_path.exists():
            raise ModelLoadError(
                f"Model not found at {model_path}",
                model_name="StackingEnsemble",
            )

        ensemble = StackingEnsemble(settings=self._settings)
        ensemble.load(model_path)
        logger.info("Loaded ensemble model from %s", model_path)

        # Initialize regime classifier if requested
        regime_classifier = None
        if self._use_regime:
            regime_path = Path(self._settings.regime_model_dir) / symbol.lower() / "regime"
            if regime_path.exists():
                regime_classifier = RegimeClassifier(settings=self._settings)
                regime_classifier.load_hmm_model(str(regime_path))
                logger.info("Loaded regime classifier from %s", regime_path)
            else:
                logger.warning("No regime model found at %s, proceeding without regime gating", regime_path)
                self._use_regime = False

        # Initialize trade tracker
        # Use 12 bars (1 hour) as default horizon for 5m bars
        predicted_horizon = 12 if timeframe == "5m" else 6
        tracker = TradeTracker(spread=self._spread, predicted_horizon=predicted_horizon)

        # Track metrics
        signals_generated = 0

        # Main replay loop
        for i, bar in enumerate(bars):
            # Compute features
            features = feature_engine.compute_features(bar)

            # Update open position
            if tracker.has_open_position():
                tracker.update_bar()

                # Check for exit
                if tracker.should_exit():
                    trade = tracker.close_trade(bar.close, bar.timestamp)
                    logger.debug(
                        "Closed trade: %s %.2f%% after %d bars",
                        trade.direction.value,
                        trade.pnl_pct,
                        trade.bars_held,
                    )

            # Skip if warmup not complete
            if not feature_engine.warmup_complete:
                continue

            # Skip if position already open (one at a time)
            if tracker.has_open_position():
                continue

            # Check regime if enabled
            regime_state = None
            if self._use_regime and regime_classifier is not None:
                regime_decision = regime_classifier.classify(bar, features, daily_bars=None)
                regime_state = regime_decision.state

                if not regime_decision.should_trade:
                    logger.debug("Regime filter: %s", regime_decision.reason)
                    continue

            # Generate signal
            feature_array = feature_vector_to_array(features)
            signal = ensemble.generate_signal(
                feature_array,
                symbol,
                bar.timestamp,
                HoldingPeriod.HOUR_1,  # Default to 1 hour horizon
            )
            signals_generated += 1

            # Check for actionable signal
            if signal.direction == Direction.NEUTRAL:
                continue

            # Open trade at next bar's open (if available)
            if i + 1 < len(bars):
                next_bar = bars[i + 1]

                # Validate entry price
                if next_bar.open <= 0:
                    logger.warning("Skipping trade: invalid entry price %s", next_bar.open)
                    continue

                # Warn about weekend gaps (Friday → Monday)
                if (bar.timestamp.weekday() == 4
                        and next_bar.timestamp.weekday() == 0):
                    logger.warning(
                        "Trade entry spans weekend gap: %s → %s",
                        bar.timestamp.date(),
                        next_bar.timestamp.date(),
                    )

                tracker.open_trade(
                    symbol,
                    signal.direction,
                    next_bar.open,
                    next_bar.timestamp,
                    regime_state,
                )
                logger.debug(
                    "Opened %s trade at %.2f (signal confidence: %.2f)",
                    signal.direction.value,
                    next_bar.open,
                    signal.confidence,
                )

        # Force close any remaining position
        if bars and tracker.has_open_position():
            last_bar = bars[-1]
            trade = tracker.force_close_all(last_bar.close, last_bar.timestamp)
            if trade:
                logger.info("Force-closed final position: %.2f%%", trade.pnl_pct)

        # Compute metrics
        closed_trades = tracker.get_closed_trades()
        metrics = compute_trade_metrics(closed_trades)

        # Compute regime breakdown if used
        regime_breakdown = {}
        if self._use_regime:
            regime_breakdown = compute_regime_breakdown(closed_trades)

        # Check if meets minimum criteria (from project.md)
        # Minimum: Sharpe > 1.0, Sortino > 1.5, Profit Factor > 1.3, Win Rate > 40%
        meets_minimum = (
            metrics.sharpe_ratio >= 1.0
            and metrics.sortino_ratio >= 1.5
            and metrics.profit_factor >= 1.3
            and metrics.win_rate >= 40.0
        )

        # Build result
        result = BacktestResult(
            symbol=symbol,
            timeframe=timeframe,
            start_date=bars[0].timestamp if bars else None,
            end_date=bars[-1].timestamp if bars else None,
            bars_processed=len(bars),
            signals_generated=signals_generated,
            trades_executed=len(closed_trades),
            spread_pct=self._spread,
            total_return=metrics.total_return,
            sharpe_ratio=metrics.sharpe_ratio,
            sortino_ratio=metrics.sortino_ratio,
            profit_factor=metrics.profit_factor,
            win_rate=metrics.win_rate,
            max_drawdown=metrics.max_drawdown,
            meets_minimum=meets_minimum,
            regime_breakdown=regime_breakdown,
        )

        logger.info(
            "Backtest complete: %d trades, %.2f%% return, Sharpe %.2f",
            len(closed_trades),
            metrics.total_return,
            metrics.sharpe_ratio,
        )

        return result