"""Composite regime classifier combining all sub-components."""

import logging
from collections import deque
from datetime import datetime

import numpy as np

from fxer.config.settings import Settings, settings as default_settings
from fxer.core.events import FeatureVector, NormalizedBar
from fxer.features.technical.indicators import ADX
from fxer.regime.change_point import ChangePointDetector
from fxer.regime.hmm import HMMRegimeClassifier
from fxer.regime.intraday_filter import IntradayRegimeFilter
from fxer.regime.session_rules import SessionRegimeRules
from fxer.regime.types import RegimeDecision, RegimeState

logger = logging.getLogger(__name__)

# Position sizing multipliers per regime
_POSITION_MULTIPLIERS = {
    RegimeState.LOW_VOL_TREND: 1.5,
    RegimeState.HIGH_VOL_TREND: 0.5,
    RegimeState.RANGING: 1.0,
}


class RegimeClassifier:
    """Composite regime classifier combining HMM, ADX/ATR, session rules, and PELT.

    Produces a RegimeDecision that gates signal publishing and adjusts
    position sizing. Implements regime persistence to avoid whipsaws.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or default_settings

        # Sub-components
        self._hmm = HMMRegimeClassifier(
            n_states=self._settings.regime_hmm_states,
            covariance_type=self._settings.regime_hmm_covariance_type,
        )
        self._intraday_filter = IntradayRegimeFilter(
            adx_range_threshold=self._settings.regime_adx_range_threshold,
            adx_trend_threshold=self._settings.regime_adx_trend_threshold,
            atr_expansion_threshold=self._settings.regime_atr_expansion_threshold,
        )
        self._session_rules = SessionRegimeRules()
        self._change_detector = ChangePointDetector(
            window_size=self._settings.regime_pelt_window_size,
            min_size=self._settings.regime_pelt_min_size,
        )
        self._adx = ADX()

        # State tracking
        self._current_state: RegimeState = RegimeState.RANGING
        self._persistence_counter: int = 0
        self._persistence_bars: int = self._settings.regime_persistence_bars
        self._confidence_minimum: float = self._settings.regime_confidence_minimum
        self._atr_buffer: deque[float] = deque(maxlen=50)
        self._returns_buffer: deque[float] = deque(maxlen=self._settings.regime_pelt_window_size)
        self._prev_close: float | None = None

    @property
    def is_ready(self) -> bool:
        """Whether the classifier has enough data to produce decisions."""
        return len(self._atr_buffer) >= 2

    def load_hmm_model(self, path_or_symbol: str) -> None:
        """Load a pre-trained HMM model.

        Args:
            path_or_symbol: Path to model directory or symbol name
                (will look in regime_model_dir/symbol/regime/).
        """
        from pathlib import Path

        path = Path(path_or_symbol)
        if not path.exists():
            # Try as symbol name
            path = Path(self._settings.regime_model_dir) / path_or_symbol.lower() / "regime"
        self._hmm.load(path)

    def classify(
        self,
        bar: NormalizedBar,
        features: FeatureVector,
        daily_bars: list[NormalizedBar] | None = None,
    ) -> RegimeDecision:
        """Classify current market regime by combining all sub-components.

        Args:
            bar: Current price bar.
            features: Current feature vector (contains ATR, session flags).
            daily_bars: Optional daily bars for HMM (if available).

        Returns:
            RegimeDecision with state, confidence, position multiplier,
            and should_trade flag.
        """
        # Track returns for PELT
        close = float(bar.close)
        if self._prev_close is not None and self._prev_close > 0:
            ret = (close - self._prev_close) / self._prev_close
            self._returns_buffer.append(ret)
        self._prev_close = close

        # Track ATR history
        if features.atr_14 is not None:
            self._atr_buffer.append(features.atr_14)

        # Compute ADX from bar HLC data
        adx_value = self._adx.update(
            float(bar.high), float(bar.low), float(bar.close),
        )

        # 1. Intraday ADX/ATR classification
        prev_atr = self._atr_buffer[-2] if len(self._atr_buffer) >= 2 else None
        intraday_state, intraday_conf = self._intraday_filter.classify(
            adx=adx_value,
            atr=features.atr_14,
            prev_atr=prev_atr,
        )

        # 2. Session rules
        session_bias, session_reason = self._session_rules.get_session_bias(
            features.timestamp,
        )
        avoid_trading = self._session_rules.should_avoid_trading(features.timestamp)

        # 3. HMM (daily level, if available and fitted)
        hmm_state: RegimeState | None = None
        hmm_prob: float | None = None
        if self._hmm.is_fitted and daily_bars and len(daily_bars) >= 30:
            try:
                import pandas as pd

                daily_df = pd.DataFrame([
                    {
                        "close": float(b.close),
                        "high": float(b.high),
                        "low": float(b.low),
                    }
                    for b in daily_bars
                ])
                hmm_state, hmm_prob = self._hmm.predict_regime(daily_df)
            except Exception as exc:
                logger.warning("HMM prediction failed: %s", exc)

        # 4. Change point detection
        recent_change = False
        if len(self._returns_buffer) >= self._change_detector._min_size * 2:
            returns_array = np.array(self._returns_buffer)
            recent_change = self._change_detector.has_recent_change(returns_array)

        # Combine signals
        proposed_state, confidence = self._combine_signals(
            intraday_state=intraday_state,
            intraday_conf=intraday_conf,
            hmm_state=hmm_state,
            hmm_prob=hmm_prob,
            recent_change=recent_change,
        )

        # Apply regime persistence
        final_state = self._apply_persistence(proposed_state)

        # Determine position multiplier
        position_multiplier = _POSITION_MULTIPLIERS.get(final_state, 1.0)
        position_multiplier *= session_bias

        # Determine should_trade
        should_trade = True
        reason_parts: list[str] = []

        if avoid_trading:
            should_trade = False
            reason_parts.append(session_reason)

        if confidence < self._confidence_minimum:
            should_trade = False
            reason_parts.append(
                f"Low confidence ({confidence:.2f} < {self._confidence_minimum:.2f})"
            )

        if recent_change:
            reason_parts.append("Recent structural break detected")
            # Reduce confidence when regime is unstable
            confidence *= 0.8
            position_multiplier *= 0.7

        if not reason_parts:
            reason_parts.append(f"{final_state.value} regime — {session_reason}")

        atr_expanding = False
        if features.atr_14 is not None and prev_atr is not None and prev_atr > 0:
            atr_expanding = (
                features.atr_14 / prev_atr >= self._settings.regime_atr_expansion_threshold
            )

        return RegimeDecision(
            state=final_state,
            confidence=confidence,
            position_multiplier=round(position_multiplier, 3),
            should_trade=should_trade,
            reason="; ".join(reason_parts),
            hmm_prob=hmm_prob,
            adx_value=adx_value,
            atr_expanding=atr_expanding,
            session_bias=session_bias,
        )

    def _combine_signals(
        self,
        intraday_state: RegimeState,
        intraday_conf: float,
        hmm_state: RegimeState | None,
        hmm_prob: float | None,
        recent_change: bool,
    ) -> tuple[RegimeState, float]:
        """Combine sub-component signals into a single regime decision.

        Weighting:
        - Intraday (ADX/ATR): 60% weight (most relevant for 5m bars)
        - HMM (daily): 30% weight (broader context)
        - Change point: 10% (structural break indicator)
        """
        if hmm_state is not None and hmm_prob is not None:
            # Weighted vote
            intraday_weight = 0.6
            hmm_weight = 0.3
            change_weight = 0.1

            # Score each state
            state_scores: dict[RegimeState, float] = {s: 0.0 for s in RegimeState}
            state_scores[intraday_state] += intraday_weight * intraday_conf
            state_scores[hmm_state] += hmm_weight * hmm_prob

            # Change point boosts HIGH_VOL_TREND
            if recent_change:
                state_scores[RegimeState.HIGH_VOL_TREND] += change_weight

            best_state = max(state_scores, key=lambda s: state_scores[s])
            confidence = state_scores[best_state]

            # Normalize confidence to 0-1 range
            total = sum(state_scores.values())
            if total > 0:
                confidence = state_scores[best_state] / total

            return best_state, confidence
        else:
            # No HMM available — rely on intraday filter
            confidence = intraday_conf
            if recent_change:
                confidence *= 0.9  # Slight reduction due to instability
            return intraday_state, confidence

    def _apply_persistence(self, proposed_state: RegimeState) -> RegimeState:
        """Apply regime persistence to prevent whipsaws.

        Requires N consecutive bars proposing the same new state before
        switching.
        """
        if proposed_state != self._current_state:
            self._persistence_counter += 1
            if self._persistence_counter >= self._persistence_bars:
                self._current_state = proposed_state
                self._persistence_counter = 0
                logger.info("Regime changed to %s", self._current_state.value)
        else:
            self._persistence_counter = 0

        return self._current_state