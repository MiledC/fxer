"""Unit tests for technical indicators."""

from __future__ import annotations

import numpy as np
import pytest

from fxer.features.technical.indicators import ADX, ATR, MACD, RSI, BollingerBands


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Sample close prices (50 bars of realistic XAUUSD-like data)
SAMPLE_CLOSES = [
    2000.0, 2002.5, 2001.0, 2004.0, 2003.5,
    2006.0, 2005.0, 2008.0, 2007.5, 2010.0,
    2009.0, 2012.0, 2011.0, 2014.0, 2013.5,
    2011.0, 2009.0, 2012.0, 2010.5, 2013.0,
    2015.0, 2014.0, 2016.5, 2015.5, 2018.0,
    2017.0, 2019.0, 2018.5, 2020.0, 2019.5,
    2017.0, 2015.5, 2018.0, 2016.5, 2019.0,
    2021.0, 2020.0, 2022.5, 2021.5, 2024.0,
    2023.0, 2025.0, 2024.5, 2026.0, 2025.5,
    2023.0, 2021.5, 2024.0, 2022.5, 2025.0,
]

SAMPLE_HIGHS = [p + 2.0 for p in SAMPLE_CLOSES]
SAMPLE_LOWS = [p - 2.0 for p in SAMPLE_CLOSES]


# ---------------------------------------------------------------------------
# RSI Tests
# ---------------------------------------------------------------------------

class TestRSI:
    def test_not_ready_during_warmup(self):
        rsi = RSI(period=14, key="rsi_14")
        for price in SAMPLE_CLOSES[:13]:
            result = rsi.update(price)
        assert not rsi.ready
        # Should return None until period+1 bars
        assert result is None

    def test_ready_after_warmup(self):
        rsi = RSI(period=14, key="rsi_14")
        for price in SAMPLE_CLOSES[:15]:
            result = rsi.update(price)
        assert rsi.ready
        assert result is not None

    def test_rsi_bounded_0_100(self):
        rsi = RSI(period=14, key="rsi_14")
        for price in SAMPLE_CLOSES:
            result = rsi.update(price)
            if result is not None:
                assert 0.0 <= result <= 100.0, f"RSI {result} out of [0,100]"

    def test_rsi_7_shorter_warmup(self):
        rsi7 = RSI(period=7, key="rsi_7")
        results = []
        for price in SAMPLE_CLOSES:
            val = rsi7.update(price)
            results.append(val)
        # RSI(7) should produce values earlier than RSI(14)
        first_non_none = next(i for i, v in enumerate(results) if v is not None)
        assert first_non_none == 7  # period+1 minus 1 (0-indexed)

    def test_constant_prices_rsi_is_none_or_stable(self):
        """Constant prices -> no gains or losses -> RSI = 100 (no losses)."""
        rsi = RSI(period=14, key="rsi_14")
        for _ in range(20):
            result = rsi.update(100.0)
        # With constant prices, avg_loss = 0, RSI = 100
        assert result == 100.0

    def test_monotonically_rising_gives_high_rsi(self):
        rsi = RSI(period=14, key="rsi_14")
        for i in range(30):
            result = rsi.update(100.0 + i)
        assert result is not None
        assert result == 100.0  # All gains, no losses

    def test_monotonically_falling_gives_low_rsi(self):
        rsi = RSI(period=14, key="rsi_14")
        for i in range(30):
            result = rsi.update(100.0 - i)
        assert result is not None
        assert result == 0.0  # All losses, no gains

    def test_compute_batch(self):
        rsi = RSI(period=14, key="rsi_14")
        closes = np.array(SAMPLE_CLOSES, dtype=np.float64)
        result = rsi.compute_batch(closes)
        assert len(result) == len(SAMPLE_CLOSES)
        # First period bars should be nan (index 0..13), first value at index 14
        assert all(np.isnan(result[:14]))
        assert not np.isnan(result[14])
        # After warmup, values should be valid
        assert all(not np.isnan(v) for v in result[15:])

    def test_reset(self):
        rsi = RSI(period=14, key="rsi_14")
        for price in SAMPLE_CLOSES:
            rsi.update(price)
        assert rsi.ready
        rsi.reset()
        assert not rsi.ready
        assert rsi._count == 0


# ---------------------------------------------------------------------------
# MACD Tests
# ---------------------------------------------------------------------------

class TestMACD:
    def test_not_ready_during_warmup(self):
        macd = MACD()
        for price in SAMPLE_CLOSES[:25]:
            line, sig, hist = macd.update(price)
        assert not macd.ready
        # Before slow period, no line
        assert sig is None

    def test_macd_line_available_before_signal(self):
        macd = MACD()
        results = []
        for price in SAMPLE_CLOSES:
            results.append(macd.update(price))
        # MACD line appears at bar index slow-1 (25)
        # Signal appears at bar index slow+signal-2 (34)
        line_at_26 = results[25]  # 0-indexed, 26th bar
        assert line_at_26[0] is not None  # line exists
        # Signal should appear later
        assert results[25][1] is None  # no signal yet at bar 26

    def test_ready_after_full_warmup(self):
        macd = MACD()
        for price in SAMPLE_CLOSES[:35]:
            macd.update(price)
        assert macd.ready

    def test_histogram_is_line_minus_signal(self):
        macd = MACD()
        for price in SAMPLE_CLOSES:
            line, sig, hist = macd.update(price)
        assert line is not None and sig is not None and hist is not None
        assert abs(hist - (line - sig)) < 1e-10

    def test_compute_batch(self):
        macd = MACD()
        closes = np.array(SAMPLE_CLOSES, dtype=np.float64)
        lines, signals, histograms = macd.compute_batch(closes)
        assert len(lines) == len(SAMPLE_CLOSES)
        # Check that signal appears later than line
        first_line = next(i for i in range(len(lines)) if not np.isnan(lines[i]))
        first_signal = next(i for i in range(len(signals)) if not np.isnan(signals[i]))
        assert first_signal > first_line

    def test_reset(self):
        macd = MACD()
        for price in SAMPLE_CLOSES:
            macd.update(price)
        assert macd.ready
        macd.reset()
        assert not macd.ready
        assert macd._count == 0


# ---------------------------------------------------------------------------
# Bollinger Bands Tests
# ---------------------------------------------------------------------------

class TestBollingerBands:
    def test_not_ready_during_warmup(self):
        bb = BollingerBands()
        for price in SAMPLE_CLOSES[:19]:
            u, m, l, w = bb.update(price)
        assert not bb.ready
        assert u is None

    def test_ready_after_warmup(self):
        bb = BollingerBands()
        for price in SAMPLE_CLOSES[:20]:
            u, m, l, w = bb.update(price)
        assert bb.ready
        assert u is not None

    def test_upper_above_middle_above_lower(self):
        bb = BollingerBands()
        for price in SAMPLE_CLOSES:
            u, m, l, w = bb.update(price)
        assert u is not None and m is not None and l is not None
        assert u > m > l

    def test_middle_is_sma(self):
        bb = BollingerBands(period=20)
        for price in SAMPLE_CLOSES[:20]:
            u, m, l, w = bb.update(price)
        expected_sma = np.mean(SAMPLE_CLOSES[:20])
        assert abs(m - expected_sma) < 1e-10

    def test_width_calculation(self):
        bb = BollingerBands()
        for price in SAMPLE_CLOSES:
            u, m, l, w = bb.update(price)
        assert u is not None and m is not None and l is not None and w is not None
        expected_width = (u - l) / m
        assert abs(w - expected_width) < 1e-10

    def test_constant_prices_zero_width(self):
        """Constant prices -> std=0 -> upper=middle=lower, width=0."""
        bb = BollingerBands(period=5)
        for _ in range(10):
            u, m, l, w = bb.update(100.0)
        assert u == m == l == 100.0
        assert w == 0.0

    def test_compute_batch(self):
        bb = BollingerBands()
        closes = np.array(SAMPLE_CLOSES, dtype=np.float64)
        uppers, middles, lowers, widths = bb.compute_batch(closes)
        assert len(uppers) == len(SAMPLE_CLOSES)
        assert all(np.isnan(uppers[:19]))
        assert not np.isnan(uppers[19])

    def test_reset(self):
        bb = BollingerBands()
        for price in SAMPLE_CLOSES:
            bb.update(price)
        assert bb.ready
        bb.reset()
        assert not bb.ready


# ---------------------------------------------------------------------------
# ATR Tests
# ---------------------------------------------------------------------------

class TestATR:
    def test_not_ready_during_warmup(self):
        atr = ATR()
        for i in range(13):
            result = atr.update(SAMPLE_HIGHS[i], SAMPLE_LOWS[i], SAMPLE_CLOSES[i])
        assert not atr.ready
        assert result is None

    def test_ready_after_warmup(self):
        atr = ATR()
        for i in range(14):
            result = atr.update(SAMPLE_HIGHS[i], SAMPLE_LOWS[i], SAMPLE_CLOSES[i])
        assert atr.ready
        assert result is not None

    def test_atr_positive(self):
        atr = ATR()
        for i in range(len(SAMPLE_CLOSES)):
            result = atr.update(SAMPLE_HIGHS[i], SAMPLE_LOWS[i], SAMPLE_CLOSES[i])
            if result is not None:
                assert result > 0.0

    def test_constant_range_atr_equals_range(self):
        """If every bar has same high-low range and no gaps, ATR = range."""
        atr = ATR(period=5)
        # Bars with high-low = 4.0, no gaps (close = open of next)
        for i in range(10):
            h = 100.0 + 2.0
            l = 100.0 - 2.0
            c = 100.0
            result = atr.update(h, l, c)
        assert result is not None
        assert abs(result - 4.0) < 1e-10

    def test_first_bar_tr_is_range(self):
        """First bar true range = high - low (no previous close)."""
        atr = ATR(period=1)
        result = atr.update(105.0, 95.0, 100.0)
        # Period=1 so first bar gives ATR = TR = 10.0
        assert result is not None
        assert abs(result - 10.0) < 1e-10

    def test_compute_batch(self):
        atr = ATR()
        highs = np.array(SAMPLE_HIGHS, dtype=np.float64)
        lows = np.array(SAMPLE_LOWS, dtype=np.float64)
        closes = np.array(SAMPLE_CLOSES, dtype=np.float64)
        result = atr.compute_batch(highs, lows, closes)
        assert len(result) == len(SAMPLE_CLOSES)
        assert all(np.isnan(result[:13]))
        assert not np.isnan(result[13])

    def test_reset(self):
        atr = ATR()
        for i in range(len(SAMPLE_CLOSES)):
            atr.update(SAMPLE_HIGHS[i], SAMPLE_LOWS[i], SAMPLE_CLOSES[i])
        assert atr.ready
        atr.reset()
        assert not atr.ready
        assert atr._count == 0


# ---------------------------------------------------------------------------
# ADX Tests
# ---------------------------------------------------------------------------

class TestADX:
    def test_adx_not_ready_during_warmup(self):
        """First 27 bars should not be ready (2x period - 1)."""
        adx = ADX(period=14)
        for i in range(27):
            result = adx.update(SAMPLE_HIGHS[i], SAMPLE_LOWS[i], SAMPLE_CLOSES[i])
        assert not adx.ready
        # Should return None until 2*period bars
        assert result is None

    def test_adx_ready_after_warmup(self):
        """28+ bars should be ready (2x period)."""
        adx = ADX(period=14)
        for i in range(28):
            result = adx.update(SAMPLE_HIGHS[i], SAMPLE_LOWS[i], SAMPLE_CLOSES[i])
        assert adx.ready
        assert result is not None

    def test_adx_bounded_0_100(self):
        """All ADX values should be in [0, 100] range."""
        adx = ADX(period=14)
        for i in range(len(SAMPLE_CLOSES)):
            result = adx.update(SAMPLE_HIGHS[i], SAMPLE_LOWS[i], SAMPLE_CLOSES[i])
            if result is not None:
                assert 0.0 <= result <= 100.0, f"ADX {result} out of [0,100]"

    def test_adx_trending_market(self):
        """Monotonically rising prices should produce higher ADX."""
        adx = ADX(period=14)
        # Create trending market data
        for i in range(50):
            high = 2000.0 + i * 2.0 + 1.0
            low = 2000.0 + i * 2.0 - 1.0
            close = 2000.0 + i * 2.0
            result = adx.update(high, low, close)

        assert result is not None
        # Strong trend should have ADX > 25
        assert result > 25.0

    def test_adx_ranging_market(self):
        """Alternating prices should produce lower ADX."""
        adx = ADX(period=14)
        # Create ranging market data (oscillating)
        for i in range(50):
            if i % 2 == 0:
                high = 2002.0
                low = 1998.0
                close = 2000.0
            else:
                high = 2002.0
                low = 1998.0
                close = 2001.0
            result = adx.update(high, low, close)

        assert result is not None
        # Ranging market should have ADX < 25
        assert result < 25.0

    def test_adx_compute_batch(self):
        """Batch mode should work correctly."""
        adx = ADX(period=14)
        highs = np.array(SAMPLE_HIGHS, dtype=np.float64)
        lows = np.array(SAMPLE_LOWS, dtype=np.float64)
        closes = np.array(SAMPLE_CLOSES, dtype=np.float64)

        result = adx.compute_batch(highs, lows, closes)

        assert len(result) == len(SAMPLE_CLOSES)
        # First 27 bars should be nan (2*period - 1)
        assert all(np.isnan(result[:27]))
        # Bar 28 should have a value
        assert not np.isnan(result[27])
        # All subsequent bars should have values
        assert all(not np.isnan(v) for v in result[28:])

    def test_adx_reset(self):
        """Reset should clear all state."""
        adx = ADX(period=14)
        for i in range(len(SAMPLE_CLOSES)):
            adx.update(SAMPLE_HIGHS[i], SAMPLE_LOWS[i], SAMPLE_CLOSES[i])
        assert adx.ready

        adx.reset()

        assert not adx.ready
        assert adx._count == 0
        assert adx._prev_high is None
        assert adx._prev_low is None
        assert adx._prev_close is None
        assert len(adx._dx_values) == 0
