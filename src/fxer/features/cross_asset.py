"""Cross-asset feature enrichment (DXY + VIX)."""

from __future__ import annotations

import dataclasses
from collections import deque

from fxer.core.events import FeatureVector
from fxer.features.technical.indicators import RSI


class CrossAssetEnricher:
    """Compute cross-asset features from DXY_SYNTH and VIX bar closes.

    Maintains rolling state for DXY and VIX closes.  Callers feed
    individual bar closes via :meth:`update_dxy` / :meth:`update_vix`
    and then call :meth:`enrich` to stamp the four cross-asset fields
    onto an existing :class:`FeatureVector`.

    Parameters
    ----------
    dxy_lookback:
        Number of 5-minute bars to look back for the 1-hour return
        calculation (default 12 = 12 x 5 min = 1 h).
    """

    def __init__(self, dxy_lookback: int = 12) -> None:
        self._dxy_lookback = dxy_lookback
        self._dxy_rsi = RSI(key="rsi_14")
        # +1 so the deque always contains the "current" close AND the one
        # `dxy_lookback` bars ago.
        self._dxy_closes: deque[float] = deque(maxlen=dxy_lookback + 1)
        self._vix_closes: deque[float] = deque(maxlen=dxy_lookback + 1)

    # ------------------------------------------------------------------
    # Feed methods — call before enrich()
    # ------------------------------------------------------------------

    def update_dxy(self, close: float) -> None:
        """Feed a DXY_SYNTH bar close."""
        self._dxy_closes.append(close)
        self._dxy_rsi.update(close)

    def update_vix(self, close: float) -> None:
        """Feed a VIX bar close."""
        self._vix_closes.append(close)

    # ------------------------------------------------------------------
    # Enrichment
    # ------------------------------------------------------------------

    def enrich(self, fv: FeatureVector) -> FeatureVector:
        """Set cross-asset fields on *fv* and return the updated copy.

        Because :class:`FeatureVector` is frozen, this creates a new
        instance via :func:`dataclasses.replace`.
        """
        dxy_return_1h: float | None = None
        dxy_rsi_14: float | None = None
        vix_level: float | None = None
        vix_change: float | None = None

        # DXY return over the lookback window
        if len(self._dxy_closes) > self._dxy_lookback:
            old = self._dxy_closes[0]
            cur = self._dxy_closes[-1]
            if old != 0.0:
                dxy_return_1h = (cur - old) / old

        # DXY RSI(14)
        if self._dxy_rsi.ready:
            # The RSI value was already computed in update_dxy; read it
            # from the internal state.  RSI.update returns the latest
            # value, but we already called it — the simplest way to get
            # the current value is to peek at the last computed result.
            # Re-derive from internal state:
            avg_gain = self._dxy_rsi._avg_gain
            avg_loss = self._dxy_rsi._avg_loss
            if avg_gain is not None and avg_loss is not None:
                if avg_loss == 0.0:
                    dxy_rsi_14 = 100.0
                else:
                    rs = avg_gain / avg_loss
                    dxy_rsi_14 = 100.0 - 100.0 / (1.0 + rs)

        # VIX level (latest close)
        if self._vix_closes:
            vix_level = self._vix_closes[-1]

        # VIX change over the lookback window
        if len(self._vix_closes) > self._dxy_lookback:
            old = self._vix_closes[0]
            cur = self._vix_closes[-1]
            if old != 0.0:
                vix_change = (cur - old) / old

        return dataclasses.replace(
            fv,
            dxy_return_1h=dxy_return_1h,
            dxy_rsi_14=dxy_rsi_14,
            vix_level=vix_level,
            vix_change=vix_change,
        )
