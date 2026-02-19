"""Calculate synthetic DXY from component forex pairs.

The US Dollar Index (DXY) measures the value of the US dollar against a
basket of six foreign currencies. This module calculates a synthetic DXY
from the component forex pairs, avoiding the $150/month NYBOT subscription.

DXY Formula (official ICE):
    DXY = 50.14348112 × EURUSD^(-0.576) × USDJPY^(0.136) × GBPUSD^(-0.119)
                      × USDCAD^(0.091) × USDSEK^(0.042) × USDCHF^(0.036)
"""

from datetime import datetime
from decimal import Decimal

from fxer.core.events import NormalizedBar
from fxer.core.types import Timeframe

# DXY component weights (official ICE formula)
DXY_CONSTANT = Decimal("50.14348112")
DXY_WEIGHTS: dict[str, Decimal] = {
    "EURUSD": Decimal("-0.576"),  # Largest weight, inverted
    "USDJPY": Decimal("0.136"),
    "GBPUSD": Decimal("-0.119"),  # Inverted
    "USDCAD": Decimal("0.091"),
    "USDSEK": Decimal("0.042"),
    "USDCHF": Decimal("0.036"),
}

DXY_COMPONENTS = frozenset(DXY_WEIGHTS.keys())


def _calculate_dxy_value(rates: dict[str, Decimal]) -> Decimal:
    """Calculate raw DXY value from rates.

    Args:
        rates: Dict of {symbol: price} for all 6 pairs.

    Returns:
        Calculated DXY value.
    """
    result = DXY_CONSTANT
    for symbol, weight in DXY_WEIGHTS.items():
        result *= rates[symbol] ** weight
    return result


def calculate_dxy(
    rates: dict[str, Decimal],
    timestamp: datetime,
    timeframe: Timeframe,
) -> NormalizedBar:
    """Calculate DXY value from forex rates.

    Args:
        rates: Dict of {symbol: close_price} for all 6 pairs.
        timestamp: Bar timestamp.
        timeframe: Bar timeframe.

    Returns:
        NormalizedBar for synthetic DXY.

    Raises:
        ValueError: If not all 6 forex pairs are provided.
    """
    missing = DXY_COMPONENTS - set(rates.keys())
    if missing:
        raise ValueError(f"Missing rates for DXY calculation: {missing}")

    result = _calculate_dxy_value(rates)

    return NormalizedBar(
        symbol="DXY_SYNTH",
        timeframe=timeframe,
        timestamp=timestamp,
        open=result,
        high=result,
        low=result,
        close=result,
        volume=Decimal("0"),
        is_complete=True,
    )


def calculate_dxy_bar(
    bars: dict[str, NormalizedBar],
    timeframe: Timeframe,
) -> NormalizedBar:
    """Calculate DXY bar from component forex bars.

    Creates a proper OHLC bar by calculating DXY for each price point.
    High/low handling accounts for inverted weights (negative exponents
    mean inverse relationship).

    Args:
        bars: Dict of {symbol: NormalizedBar} for all 6 forex pairs.
        timeframe: Target timeframe for the synthetic bar.

    Returns:
        NormalizedBar for synthetic DXY with proper OHLC.

    Raises:
        ValueError: If not all 6 forex pairs are provided or timestamps differ.
    """
    missing = DXY_COMPONENTS - set(bars.keys())
    if missing:
        raise ValueError(f"Missing bars for DXY calculation: {missing}")

    timestamps = {b.timestamp for b in bars.values()}
    if len(timestamps) != 1:
        raise ValueError(f"Bars have different timestamps: {timestamps}")

    timestamp = next(iter(timestamps))

    # Calculate DXY for each OHLC price point
    dxy_open = _calculate_dxy_value({s: bars[s].open for s in DXY_WEIGHTS})
    dxy_close = _calculate_dxy_value({s: bars[s].close for s in DXY_WEIGHTS})

    # For high/low, we need to consider the weight direction:
    # - Positive weight (USDJPY, USDCAD, USDSEK, USDCHF): DXY rises when pair rises
    # - Negative weight (EURUSD, GBPUSD): DXY rises when pair falls
    # To find DXY high: use high for positive weights, low for negative weights
    # To find DXY low: use low for positive weights, high for negative weights
    rates_for_dxy_high = {}
    rates_for_dxy_low = {}
    for symbol, weight in DXY_WEIGHTS.items():
        bar = bars[symbol]
        if weight > 0:
            rates_for_dxy_high[symbol] = bar.high
            rates_for_dxy_low[symbol] = bar.low
        else:
            rates_for_dxy_high[symbol] = bar.low
            rates_for_dxy_low[symbol] = bar.high

    dxy_high_estimate = _calculate_dxy_value(rates_for_dxy_high)
    dxy_low_estimate = _calculate_dxy_value(rates_for_dxy_low)

    # Ensure OHLC consistency: actual high/low from all calculated values
    all_values = [dxy_open, dxy_close, dxy_high_estimate, dxy_low_estimate]
    actual_high = max(all_values)
    actual_low = min(all_values)

    return NormalizedBar(
        symbol="DXY_SYNTH",
        timeframe=timeframe,
        timestamp=timestamp,
        open=dxy_open,
        high=actual_high,
        low=actual_low,
        close=dxy_close,
        volume=Decimal("0"),
        is_complete=True,
    )
