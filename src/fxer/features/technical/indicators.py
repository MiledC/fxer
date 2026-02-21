"""Technical indicators computed with numpy.

Each indicator class maintains rolling state and tracks its warmup period.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from fxer.config.constants import INDICATOR_PARAMS, INDICATOR_WARMUP


class RSI:
    """Relative Strength Index using exponential (Wilder) smoothing."""

    def __init__(self, period: int | None = None, *, key: str = "rsi_14") -> None:
        params = INDICATOR_PARAMS.get(key, {})
        self.period: int = period or params.get("period", 14)
        self.warmup_period: int = INDICATOR_WARMUP.get(key, self.period)
        self._prices: list[float] = []
        self._avg_gain: float | None = None
        self._avg_loss: float | None = None
        self._count: int = 0

    @property
    def ready(self) -> bool:
        return self._count >= self.warmup_period

    def update(self, close: float) -> float | None:
        """Feed one close price and return current RSI or None if warming up."""
        self._prices.append(close)
        self._count += 1

        if self._count < 2:
            return None

        change = close - self._prices[-2]
        gain = max(change, 0.0)
        loss = max(-change, 0.0)

        if self._count <= self.period:
            # Still accumulating initial window
            return None

        if self._count == self.period + 1:
            # First RSI: simple average of first `period` changes
            changes = np.diff(self._prices[: self.period + 1])
            self._avg_gain = float(np.mean(np.maximum(changes, 0.0)))
            self._avg_loss = float(np.mean(np.maximum(-changes, 0.0)))
        else:
            # Wilder smoothing
            assert self._avg_gain is not None
            assert self._avg_loss is not None
            self._avg_gain = (self._avg_gain * (self.period - 1) + gain) / self.period
            self._avg_loss = (self._avg_loss * (self.period - 1) + loss) / self.period

        assert self._avg_gain is not None
        assert self._avg_loss is not None

        if self._avg_loss == 0.0:
            return 100.0
        rs = self._avg_gain / self._avg_loss
        return 100.0 - 100.0 / (1.0 + rs)

    def compute_batch(self, closes: NDArray[np.floating]) -> NDArray[np.floating]:
        """Compute RSI for an array of close prices. Returns nan where not ready."""
        result = np.full(len(closes), np.nan)
        for i, c in enumerate(closes):
            val = self.update(float(c))
            if val is not None:
                result[i] = val
        return result

    def reset(self) -> None:
        self._prices.clear()
        self._avg_gain = None
        self._avg_loss = None
        self._count = 0


class MACD:
    """MACD (Moving Average Convergence Divergence).

    Uses EMA for fast, slow, and signal lines.
    """

    def __init__(
        self,
        fast: int | None = None,
        slow: int | None = None,
        signal: int | None = None,
    ) -> None:
        params = INDICATOR_PARAMS.get("macd", {})
        self.fast: int = fast or params.get("fast", 12)
        self.slow: int = slow or params.get("slow", 26)
        self.signal: int = signal or params.get("signal", 9)
        self.warmup_period: int = INDICATOR_WARMUP.get("macd", self.slow + self.signal)

        self._fast_ema: float | None = None
        self._slow_ema: float | None = None
        self._signal_ema: float | None = None
        self._fast_mult: float = 2.0 / (self.fast + 1)
        self._slow_mult: float = 2.0 / (self.slow + 1)
        self._signal_mult: float = 2.0 / (self.signal + 1)
        self._count: int = 0
        self._prices: list[float] = []

    @property
    def ready(self) -> bool:
        return self._count >= self.warmup_period

    def update(self, close: float) -> tuple[float | None, float | None, float | None]:
        """Feed one close price. Returns (macd_line, signal_line, histogram)."""
        self._prices.append(close)
        self._count += 1

        # Build fast EMA
        if self._count < self.fast:
            return None, None, None
        if self._count == self.fast:
            self._fast_ema = float(np.mean(self._prices[: self.fast]))
        else:
            assert self._fast_ema is not None
            self._fast_ema = (close - self._fast_ema) * self._fast_mult + self._fast_ema

        # Build slow EMA
        if self._count < self.slow:
            return None, None, None
        if self._count == self.slow:
            self._slow_ema = float(np.mean(self._prices[: self.slow]))
        else:
            assert self._slow_ema is not None
            self._slow_ema = (close - self._slow_ema) * self._slow_mult + self._slow_ema

        assert self._fast_ema is not None
        assert self._slow_ema is not None
        macd_line = self._fast_ema - self._slow_ema

        # Build signal EMA from MACD line values
        macd_values_count = self._count - self.slow + 1
        if macd_values_count < self.signal:
            return macd_line, None, None
        if macd_values_count == self.signal:
            # Need to compute first signal as SMA of the MACD values so far.
            # Recompute MACD values for the initial signal window.
            macd_vals = self._recompute_macd_values(self.signal)
            self._signal_ema = float(np.mean(macd_vals))
        else:
            assert self._signal_ema is not None
            self._signal_ema = (
                (macd_line - self._signal_ema) * self._signal_mult + self._signal_ema
            )

        assert self._signal_ema is not None
        histogram = macd_line - self._signal_ema
        return macd_line, self._signal_ema, histogram

    def _recompute_macd_values(self, count: int) -> list[float]:
        """Recompute the last `count` MACD line values from price history."""
        # We need prices starting from the slow period
        start = self.slow - 1  # index where slow EMA first exists
        prices = self._prices[: start + count]

        fast_ema = float(np.mean(prices[:self.fast]))
        slow_ema = float(np.mean(prices[:self.slow]))
        macd_vals = [fast_ema - slow_ema]

        for p in prices[self.slow:]:
            fast_ema = (p - fast_ema) * self._fast_mult + fast_ema
            slow_ema = (p - slow_ema) * self._slow_mult + slow_ema
            macd_vals.append(fast_ema - slow_ema)

        return macd_vals[-count:]

    def compute_batch(
        self, closes: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
        """Compute MACD for array. Returns (line, signal, histogram) arrays with nan."""
        n = len(closes)
        lines = np.full(n, np.nan)
        signals = np.full(n, np.nan)
        histograms = np.full(n, np.nan)
        for i, c in enumerate(closes):
            line, sig, hist = self.update(float(c))
            if line is not None:
                lines[i] = line
            if sig is not None:
                signals[i] = sig
            if hist is not None:
                histograms[i] = hist
        return lines, signals, histograms

    def reset(self) -> None:
        self._fast_ema = None
        self._slow_ema = None
        self._signal_ema = None
        self._count = 0
        self._prices.clear()


class BollingerBands:
    """Bollinger Bands using SMA and standard deviation."""

    def __init__(self, period: int | None = None, std_dev: float | None = None) -> None:
        params = INDICATOR_PARAMS.get("bb", {})
        self.period: int = period or params.get("period", 20)
        self.std_dev: float = std_dev or params.get("std_dev", 2.0)
        self.warmup_period: int = INDICATOR_WARMUP.get("bb", self.period)
        self._prices: list[float] = []
        self._count: int = 0

    @property
    def ready(self) -> bool:
        return self._count >= self.warmup_period

    def update(
        self, close: float
    ) -> tuple[float | None, float | None, float | None, float | None]:
        """Feed one close. Returns (upper, middle, lower, width) or all None."""
        self._prices.append(close)
        self._count += 1

        if self._count < self.period:
            return None, None, None, None

        window = self._prices[-self.period :]
        middle = float(np.mean(window))
        std = float(np.std(window, ddof=0))

        upper = middle + self.std_dev * std
        lower = middle - self.std_dev * std
        width = (upper - lower) / middle if middle != 0 else 0.0

        return upper, middle, lower, width

    def compute_batch(
        self, closes: NDArray[np.floating]
    ) -> tuple[
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
    ]:
        """Compute BB for array. Returns (upper, middle, lower, width) with nan."""
        n = len(closes)
        uppers = np.full(n, np.nan)
        middles = np.full(n, np.nan)
        lowers = np.full(n, np.nan)
        widths = np.full(n, np.nan)
        for i, c in enumerate(closes):
            u, m, l, w = self.update(float(c))
            if u is not None:
                uppers[i] = u
                middles[i] = m  # type: ignore[assignment]
                lowers[i] = l  # type: ignore[assignment]
                widths[i] = w  # type: ignore[assignment]
        return uppers, middles, lowers, widths

    def reset(self) -> None:
        self._prices.clear()
        self._count = 0


class ATR:
    """Average True Range using Wilder smoothing."""

    def __init__(self, period: int | None = None) -> None:
        params = INDICATOR_PARAMS.get("atr", {})
        self.period: int = period or params.get("period", 14)
        self.warmup_period: int = INDICATOR_WARMUP.get("atr", self.period)
        self._prev_close: float | None = None
        self._true_ranges: list[float] = []
        self._atr: float | None = None
        self._count: int = 0

    @property
    def ready(self) -> bool:
        return self._count >= self.warmup_period

    def update(self, high: float, low: float, close: float) -> float | None:
        """Feed one bar (high, low, close). Returns ATR or None."""
        self._count += 1

        if self._prev_close is None:
            # First bar: TR = high - low
            tr = high - low
        else:
            tr = max(
                high - low,
                abs(high - self._prev_close),
                abs(low - self._prev_close),
            )

        self._prev_close = close
        self._true_ranges.append(tr)

        if self._count < self.period:
            return None

        if self._count == self.period:
            self._atr = float(np.mean(self._true_ranges[: self.period]))
        else:
            assert self._atr is not None
            self._atr = (self._atr * (self.period - 1) + tr) / self.period

        return self._atr

    def compute_batch(
        self,
        highs: NDArray[np.floating],
        lows: NDArray[np.floating],
        closes: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Compute ATR for arrays. Returns array with nan where not ready."""
        n = len(highs)
        result = np.full(n, np.nan)
        for i in range(n):
            val = self.update(float(highs[i]), float(lows[i]), float(closes[i]))
            if val is not None:
                result[i] = val
        return result

    def reset(self) -> None:
        self._prev_close = None
        self._true_ranges.clear()
        self._atr = None
        self._count = 0


class ADX:
    """Average Directional Index for trend strength measurement.

    Uses Wilder smoothing for directional movement, true range, and ADX.
    Requires 2x period bars: period for DM/TR smoothing + period for DX->ADX smoothing.
    """

    def __init__(self, period: int | None = None, *, key: str = "adx_14") -> None:
        params = INDICATOR_PARAMS.get(key, {})
        self.period: int = period or params.get("period", 14)
        self.warmup_period: int = INDICATOR_WARMUP.get(key, self.period * 2)
        self._prev_high: float | None = None
        self._prev_low: float | None = None
        self._prev_close: float | None = None
        self._plus_dm_sum: float = 0.0
        self._minus_dm_sum: float = 0.0
        self._tr_sum: float = 0.0
        self._plus_dm_avg: float | None = None
        self._minus_dm_avg: float | None = None
        self._tr_avg: float | None = None
        self._dx_values: list[float] = []
        self._adx_avg: float | None = None
        self._count: int = 0

    @property
    def ready(self) -> bool:
        return self._count >= self.warmup_period

    def update(self, high: float, low: float, close: float) -> float | None:
        """Feed one bar (high, low, close). Returns ADX value or None during warmup."""
        self._count += 1

        if self._prev_high is None:
            self._prev_high = high
            self._prev_low = low
            self._prev_close = close
            return None

        # Directional Movement
        up_move = high - self._prev_high
        down_move = self._prev_low - low
        plus_dm = up_move if (up_move > down_move and up_move > 0) else 0.0
        minus_dm = down_move if (down_move > up_move and down_move > 0) else 0.0

        # True Range
        tr = max(
            high - low,
            abs(high - self._prev_close),
            abs(low - self._prev_close),
        )

        self._prev_high = high
        self._prev_low = low
        self._prev_close = close

        # Accumulate for initial smoothing
        if self._count <= self.period:
            self._plus_dm_sum += plus_dm
            self._minus_dm_sum += minus_dm
            self._tr_sum += tr
            if self._count == self.period:
                self._plus_dm_avg = self._plus_dm_sum / self.period
                self._minus_dm_avg = self._minus_dm_sum / self.period
                self._tr_avg = self._tr_sum / self.period
            return None

        # Wilder smoothing
        assert self._plus_dm_avg is not None
        assert self._minus_dm_avg is not None
        assert self._tr_avg is not None
        self._plus_dm_avg = (self._plus_dm_avg * (self.period - 1) + plus_dm) / self.period
        self._minus_dm_avg = (self._minus_dm_avg * (self.period - 1) + minus_dm) / self.period
        self._tr_avg = (self._tr_avg * (self.period - 1) + tr) / self.period

        if self._tr_avg == 0.0:
            return None

        plus_di = 100.0 * self._plus_dm_avg / self._tr_avg
        minus_di = 100.0 * self._minus_dm_avg / self._tr_avg
        di_sum = plus_di + minus_di

        if di_sum == 0.0:
            dx = 0.0
        else:
            dx = 100.0 * abs(plus_di - minus_di) / di_sum

        self._dx_values.append(dx)

        # ADX smoothing: need `period` DX values first
        if len(self._dx_values) < self.period:
            return None

        if len(self._dx_values) == self.period:
            self._adx_avg = float(np.mean(self._dx_values))
            return self._adx_avg

        assert self._adx_avg is not None
        self._adx_avg = (self._adx_avg * (self.period - 1) + dx) / self.period
        return self._adx_avg

    def compute_batch(
        self,
        highs: NDArray[np.floating],
        lows: NDArray[np.floating],
        closes: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Compute ADX for arrays. Returns array with nan where not ready."""
        n = len(highs)
        result = np.full(n, np.nan)
        for i in range(n):
            val = self.update(float(highs[i]), float(lows[i]), float(closes[i]))
            if val is not None:
                result[i] = val
        return result

    def reset(self) -> None:
        self._prev_high = None
        self._prev_low = None
        self._prev_close = None
        self._plus_dm_sum = 0.0
        self._minus_dm_sum = 0.0
        self._tr_sum = 0.0
        self._plus_dm_avg = None
        self._minus_dm_avg = None
        self._tr_avg = None
        self._dx_values.clear()
        self._adx_avg = None
        self._count = 0
