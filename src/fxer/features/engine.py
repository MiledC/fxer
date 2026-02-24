"""Feature computation engine orchestrating all indicators and temporal features."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from fxer.config.constants import MAX_WARMUP_BARS
from fxer.core.events import FeatureVector, NormalizedBar
from fxer.features.price import PriceFeatures
from fxer.features.technical.indicators import ATR, MACD, RSI, BollingerBands
from fxer.features.temporal.time_features import compute_time_features

if TYPE_CHECKING:
    from fxer.features.cross_asset import CrossAssetEnricher


class FeatureEngine:
    """Orchestrates feature computation for a single symbol/timeframe stream.

    Maintains rolling state for all indicators. Feed bars sequentially
    via ``compute_features`` or in bulk via ``compute_batch``.
    """

    def __init__(self, cross_asset: CrossAssetEnricher | None = None) -> None:
        # Technical indicators
        self._rsi_14 = RSI(key="rsi_14")
        self._rsi_7 = RSI(key="rsi_7")
        self._macd = MACD()
        self._bb = BollingerBands()
        self._atr = ATR()

        # Price-derived features
        self._price_features = PriceFeatures()

        # Optional cross-asset enricher (DXY + VIX)
        self._cross_asset = cross_asset

        # Track bars received
        self._bar_count: int = 0
        self._warmup_complete: bool = False

        # Rolling buffer (keeps last MAX_WARMUP_BARS bars for potential recomputation)
        self._buffer: deque[NormalizedBar] = deque(maxlen=MAX_WARMUP_BARS + 10)

    @property
    def warmup_complete(self) -> bool:
        return self._warmup_complete

    @property
    def bar_count(self) -> int:
        return self._bar_count

    def get_warmup_bars(self) -> int:
        """Return minimum bars needed before all indicators produce valid output."""
        return MAX_WARMUP_BARS

    def compute_features(self, bar: NormalizedBar) -> FeatureVector:
        """Compute features for a single bar. Updates internal state."""
        self._buffer.append(bar)
        self._bar_count += 1

        close = float(bar.close)
        high = float(bar.high)
        low = float(bar.low)

        # Technical indicators
        rsi_14 = self._rsi_14.update(close)
        rsi_7 = self._rsi_7.update(close)
        macd_line, macd_signal, macd_histogram = self._macd.update(close)
        bb_upper, bb_middle, bb_lower, bb_width, bb_percent_b = self._bb.update(close)
        atr_14 = self._atr.update(high, low, close)

        # Price-derived features
        price_feats = self._price_features.update(close)
        return_1bar, return_5bar, return_12bar, rolling_vol_20, momentum_48 = price_feats

        # Check warmup
        all_ready = all([
            self._rsi_14.ready,
            self._rsi_7.ready,
            self._macd.ready,
            self._bb.ready,
            self._atr.ready,
            self._price_features.ready,
        ])
        if all_ready:
            self._warmup_complete = True

        # Temporal features
        time_feats = compute_time_features(bar.timestamp)

        fv = FeatureVector(
            symbol=bar.symbol,
            timestamp=bar.timestamp,
            timeframe=bar.timeframe,
            rsi_14=rsi_14,
            rsi_7=rsi_7,
            macd_line=macd_line,
            macd_signal=macd_signal,
            macd_histogram=macd_histogram,
            bb_upper=bb_upper,
            bb_middle=bb_middle,
            bb_lower=bb_lower,
            bb_width=bb_width,
            bb_percent_b=bb_percent_b,
            atr_14=atr_14,
            return_1bar=return_1bar,
            return_5bar=return_5bar,
            return_12bar=return_12bar,
            rolling_volatility_20=rolling_vol_20,
            momentum_48=momentum_48,
            is_london_session=time_feats["is_london_session"],
            is_ny_session=time_feats["is_ny_session"],
            is_overlap_session=time_feats["is_overlap_session"],
            is_asian_session=time_feats["is_asian_session"],
            hour_of_day=time_feats["hour_of_day"],
            day_of_week=time_feats["day_of_week"],
            is_month_turn=time_feats["is_month_turn"],
            warmup_complete=self._warmup_complete,
        )

        if self._cross_asset is not None:
            fv = self._cross_asset.enrich(fv)

        return fv

    def compute_batch(self, bars: list[NormalizedBar]) -> list[FeatureVector]:
        """Compute features for a batch of bars in order."""
        return [self.compute_features(bar) for bar in bars]

    def reset(self) -> None:
        """Reset all indicator state."""
        self._rsi_14.reset()
        self._rsi_7.reset()
        self._macd.reset()
        self._bb.reset()
        self._atr.reset()
        self._price_features.reset()
        self._bar_count = 0
        self._warmup_complete = False
        self._buffer.clear()
