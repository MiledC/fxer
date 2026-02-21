"""Unit tests for the price features module."""

import math

import pytest

from fxer.features.price.price_features import PriceFeatures


class TestPriceFeatures:
    """Test PriceFeatures calculation and state management."""

    def test_warmup_period_is_49(self):
        """Verify the warmup_period property returns 49."""
        pf = PriceFeatures()
        assert pf.warmup_period == 49

    def test_not_ready_before_49_bars(self):
        """Feed 48 bars and verify .ready is False."""
        pf = PriceFeatures()
        for i in range(48):
            pf.update(2000.0 + i)
        assert pf.ready is False

    def test_ready_after_49_bars(self):
        """Feed 49 bars and verify .ready is True."""
        pf = PriceFeatures()
        for i in range(49):
            pf.update(2000.0 + i)
        assert pf.ready is True

    def test_all_none_on_first_bar(self):
        """First bar should return all None values."""
        pf = PriceFeatures()
        result = pf.update(2000.0)
        assert result == (None, None, None, None, None)

    def test_return_1bar_calculation(self):
        """Feed 2 known prices, verify log return math."""
        pf = PriceFeatures()
        pf.update(100.0)  # First bar
        return_1bar, *_ = pf.update(110.0)  # Second bar: log(110/100)
        expected = math.log(110.0 / 100.0)
        assert return_1bar == pytest.approx(expected, abs=1e-10)

    def test_return_5bar_calculation(self):
        """Feed 6 known prices, verify log return math."""
        pf = PriceFeatures()
        prices = [100.0, 102.0, 104.0, 106.0, 108.0, 110.0]

        for i in range(5):
            pf.update(prices[i])

        # On the 6th bar, we should get 5-bar return
        _, return_5bar, *_ = pf.update(prices[5])
        expected = math.log(110.0 / 100.0)  # log(current / 5-bars-ago)
        assert return_5bar == pytest.approx(expected, abs=1e-10)

    def test_return_12bar_calculation(self):
        """Feed 13 known prices, verify log return math."""
        pf = PriceFeatures()
        prices = [100.0 + i * 2 for i in range(13)]  # 100, 102, 104, ..., 124

        for i in range(12):
            pf.update(prices[i])

        # On the 13th bar, we should get 12-bar return
        _, _, return_12bar, *_ = pf.update(prices[12])
        expected = math.log(124.0 / 100.0)  # log(current / 12-bars-ago)
        assert return_12bar == pytest.approx(expected, abs=1e-10)

    def test_rolling_volatility_20(self):
        """Feed 21+ known prices, verify std dev math."""
        pf = PriceFeatures()

        # Create a sequence with known volatility
        # Alternating between 100 and 110 for high volatility
        prices = []
        for i in range(21):
            if i % 2 == 0:
                prices.append(100.0)
            else:
                prices.append(110.0)

        # Feed 20 bars first
        for i in range(20):
            pf.update(prices[i])

        # On the 21st bar, we should get rolling volatility
        _, _, _, rolling_vol, _ = pf.update(prices[20])

        # Compute expected volatility manually
        returns = []
        for i in range(1, 21):
            returns.append(math.log(prices[i] / prices[i-1]))

        mean = sum(returns) / len(returns)
        variance = sum((r - mean) ** 2 for r in returns) / len(returns)
        expected = math.sqrt(variance)

        assert rolling_vol == pytest.approx(expected, abs=1e-10)

    def test_momentum_48_calculation(self):
        """Feed 49 known prices, verify (close - old) / old math."""
        pf = PriceFeatures()

        # Start at 100, increase by 1 each bar
        prices = [100.0 + i for i in range(49)]

        for i in range(48):
            pf.update(prices[i])

        # On the 49th bar, we should get momentum
        *_, momentum_48 = pf.update(prices[48])

        # Momentum = (current - 48-bars-ago) / 48-bars-ago
        expected = (148.0 - 100.0) / 100.0  # = 0.48
        assert momentum_48 == pytest.approx(expected, abs=1e-10)

    def test_zero_price_guard(self):
        """Feed a zero price, verify no crash and returns None for log features."""
        pf = PriceFeatures()
        pf.update(100.0)

        # Feed zero price - should not crash
        return_1bar, *_ = pf.update(0.0)
        assert return_1bar is None  # Can't compute log(0/100)

        # Feed another valid price after zero
        return_1bar, *_ = pf.update(110.0)
        assert return_1bar is None  # Can't compute log(110/0) - previous was 0

        # Feed one more valid price - should now work
        return_1bar, *_ = pf.update(120.0)
        assert return_1bar is not None  # Can compute log(120/110)
        expected = math.log(120.0 / 110.0)
        assert return_1bar == pytest.approx(expected, abs=1e-10)

    def test_zero_price_guard_comprehensive(self):
        """Test zero price handling more carefully."""
        pf = PriceFeatures()

        # Start with valid price
        pf.update(100.0)

        # Feed zero - should handle gracefully
        result = pf.update(0.0)
        return_1bar, *_ = result
        assert return_1bar is None  # Can't take log of zero

        # Feed another valid price
        result = pf.update(110.0)
        return_1bar, *_ = result
        # This computes log(110/0) which is invalid, so should be None
        assert return_1bar is None

    def test_reset_clears_state(self):
        """Feed bars, reset, verify not ready and count is 0."""
        pf = PriceFeatures()

        # Feed enough bars to be ready
        for i in range(50):
            pf.update(2000.0 + i)

        assert pf.ready is True
        assert pf._count == 50

        # Reset
        pf.reset()

        # Should be back to initial state
        assert pf.ready is False
        assert pf._count == 0
        assert len(pf._prices) == 0
        assert len(pf._returns_1bar) == 0

    def test_constant_prices_zero_volatility(self):
        """All same price → volatility = 0."""
        pf = PriceFeatures()

        # Feed 21 bars with constant price
        for _ in range(21):
            pf.update(100.0)

        # Get volatility on the 21st bar
        _, _, _, rolling_vol, _ = pf._prices[-1], pf._prices[-1], pf._prices[-1], 0.0, pf._prices[-1]

        # Need to actually get the result from update
        pf2 = PriceFeatures()
        for _ in range(20):
            pf2.update(100.0)
        _, _, _, rolling_vol, _ = pf2.update(100.0)

        # All returns are log(100/100) = 0, so std dev = 0
        assert rolling_vol == pytest.approx(0.0, abs=1e-10)

    def test_no_lookahead_bias(self):
        """Feature at time T must not use data from T+1 or later."""
        # The key insight: momentum_48 needs 49 bars total (48 bars ago + current)
        # So features are first available on the 49th update (index 48)

        # Create price series
        prices = [2000.0 + i * 0.5 for i in range(100)]

        # Test that features computed at any position match regardless of future data
        pf1 = PriceFeatures()
        pf2 = PriceFeatures()

        # Feed 60 bars to pf1 and save all features
        features_60 = []
        for i in range(60):
            features = pf1.update(prices[i])
            features_60.append(features)

        # Feed 100 bars to pf2 and compare features at same positions
        for i in range(100):
            features = pf2.update(prices[i])
            if i < 60:
                # Features at position i should be the same
                assert features == features_60[i], f"Lookahead bias detected at position {i}"

    def test_warmup_none_values(self):
        """Verify features are None during warmup period."""
        pf = PriceFeatures()

        # First bar - all None
        result = pf.update(2000.0)
        assert all(x is None for x in result)

        # Second bar - only 1-bar return should be available
        return_1bar, return_5bar, return_12bar, vol, momentum = pf.update(2001.0)
        assert return_1bar is not None
        assert return_5bar is None
        assert return_12bar is None
        assert vol is None
        assert momentum is None

        # After 5 bars, still no 5-bar return (need 6 bars)
        pf = PriceFeatures()
        for i in range(5):
            pf.update(2000.0 + i)
        _, return_5bar, *_ = pf.update(2005.0)
        assert return_5bar is not None  # Now we have 6 bars

        # After 12 bars, still no 12-bar return (need 13 bars)
        pf = PriceFeatures()
        for i in range(12):
            pf.update(2000.0 + i)
        _, _, return_12bar, *_ = pf.update(2012.0)
        assert return_12bar is not None  # Now we have 13 bars

        # After 20 bars, volatility should be available
        pf = PriceFeatures()
        for i in range(20):
            pf.update(2000.0 + i)
        _, _, _, vol, _ = pf.update(2020.0)
        assert vol is not None  # Now we have 21 bars with 20 returns

        # After 48 bars, still no momentum (need 49 bars)
        pf = PriceFeatures()
        for i in range(48):
            pf.update(2000.0 + i)
        *_, momentum = pf.update(2048.0)
        assert momentum is not None  # Now we have 49 bars

    def test_features_after_warmup(self):
        """Verify all features are populated after warmup."""
        pf = PriceFeatures()

        # Feed exactly 49 bars
        for i in range(49):
            result = pf.update(2000.0 + i * 0.5)

        # All features should be populated
        return_1bar, return_5bar, return_12bar, vol, momentum = result
        assert return_1bar is not None
        assert return_5bar is not None
        assert return_12bar is not None
        assert vol is not None
        assert momentum is not None

    def test_edge_case_single_price(self):
        """Test behavior with single price update."""
        pf = PriceFeatures()
        result = pf.update(2000.0)
        assert result == (None, None, None, None, None)
        assert pf._count == 1
        assert len(pf._prices) == 1

    def test_large_price_changes(self):
        """Test with large price movements."""
        pf = PriceFeatures()

        # Start at 1000, jump to 2000 (100% increase)
        pf.update(1000.0)
        return_1bar, *_ = pf.update(2000.0)
        expected = math.log(2.0)  # log(2000/1000)
        assert return_1bar == pytest.approx(expected, abs=1e-10)

        # Drop back to 500 (75% decrease)
        return_1bar, *_ = pf.update(500.0)
        expected = math.log(0.25)  # log(500/2000)
        assert return_1bar == pytest.approx(expected, abs=1e-10)

    def test_price_sequence_consistency(self):
        """Test that internal price buffer maintains correct sequence."""
        pf = PriceFeatures()

        prices = [2000.0 + i for i in range(60)]

        for price in prices[:49]:
            pf.update(price)

        # Check that deque is full and contains last 49 prices
        assert len(pf._prices) == 49
        assert pf._prices[0] == prices[0]
        assert pf._prices[-1] == prices[48]

        # Add more prices - deque should maintain maxlen
        for price in prices[49:55]:
            pf.update(price)

        assert len(pf._prices) == 49  # Still maxlen
        assert pf._prices[0] == prices[6]  # Oldest price shifted out
        assert pf._prices[-1] == prices[54]  # Newest price

    def test_negative_prices_rejected(self):
        """Negative prices should be handled gracefully."""
        pf = PriceFeatures()
        pf.update(100.0)

        # Negative price - log would be undefined
        return_1bar, *_ = pf.update(-50.0)
        assert return_1bar is None  # Can't take log of negative

    def test_floating_point_precision(self):
        """Test with high-precision floating point values."""
        pf = PriceFeatures()

        # Use prices with many decimal places
        price1 = 2062.123456789
        price2 = 2062.234567890

        pf.update(price1)
        return_1bar, *_ = pf.update(price2)

        expected = math.log(price2 / price1)
        assert return_1bar == pytest.approx(expected, abs=1e-10)

    def test_momentum_calculation_accuracy(self):
        """Verify momentum calculation with known values."""
        pf = PriceFeatures()

        # Create a price series where we know the exact momentum
        initial_price = 2000.0
        final_price = 2100.0  # 5% increase

        # Feed 48 bars with initial price
        for _ in range(48):
            pf.update(initial_price)

        # 49th bar with final price
        *_, momentum = pf.update(final_price)

        expected = (final_price - initial_price) / initial_price  # 0.05
        assert momentum == pytest.approx(expected, abs=1e-10)

    def test_volatility_with_alternating_prices(self):
        """Test volatility calculation with alternating high/low prices."""
        pf = PriceFeatures()

        # Alternating between two prices creates consistent returns
        price_low = 100.0
        price_high = 105.0

        for i in range(21):
            if i % 2 == 0:
                pf.update(price_low)
            else:
                pf.update(price_high)

        # The last update should give us volatility
        # Returns alternate between log(105/100) and log(100/105)
        return_up = math.log(price_high / price_low)
        return_down = math.log(price_low / price_high)

        # With 20 returns (10 up, 10 down for first 20 prices)
        # Actually we need to be more careful about the exact sequence
        pf2 = PriceFeatures()
        returns = []

        prices = []
        for i in range(21):
            if i % 2 == 0:
                prices.append(price_low)
            else:
                prices.append(price_high)

        for i, price in enumerate(prices):
            result = pf2.update(price)
            if i > 0:  # Can compute return
                prev_price = prices[i-1]
                if price > 0 and prev_price > 0:
                    ret = math.log(price / prev_price)
                    returns.append(ret)

        # On the 21st bar, we have 20 returns
        if len(returns) >= 20:
            last_20_returns = returns[-20:]
            mean = sum(last_20_returns) / 20
            variance = sum((r - mean) ** 2 for r in last_20_returns) / 20
            expected_vol = math.sqrt(variance)

            _, _, _, rolling_vol, _ = result
            if rolling_vol is not None:
                assert rolling_vol == pytest.approx(expected_vol, abs=1e-10)