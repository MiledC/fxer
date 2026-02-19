"""Symbol configuration and IBKR contract mapping.

Centralized configuration for tradeable and data symbols, including
their IBKR contract definitions and supported timeframes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ib_insync import Contract


@dataclass(frozen=True)
class SymbolConfig:
    """Configuration for a tradeable/data symbol.

    Attributes:
        name: Internal symbol name (e.g., XAUUSD, DXY, VIX).
        ibkr_symbol: IBKR symbol identifier.
        ibkr_exchange: IBKR exchange (e.g., SMART, IDEALPRO, CBOE).
        ibkr_sec_type: Security type (CFD, CASH, IND, FUT, CONTFUT).
        ibkr_currency: Currency for the contract.
        ibkr_what_to_show: IBKR historical data type (MIDPOINT, TRADES, BID, ASK).
        ibkr_include_expired: Include expired contracts (for historical futures data).
        timeframes: List of timeframes to fetch (e.g., ["5m", "1h", "4h"]).
        is_tradeable: True for tradeable symbols, False for data-only (DXY, VIX).
    """

    name: str
    ibkr_symbol: str
    ibkr_exchange: str
    ibkr_sec_type: str
    ibkr_currency: str = "USD"
    ibkr_what_to_show: str = "MIDPOINT"  # MIDPOINT for CFD/CASH, TRADES for IND/FUT
    ibkr_include_expired: bool = False  # For historical futures data
    timeframes: tuple[str, ...] = ("5m", "15m", "1h", "4h")
    is_tradeable: bool = True

    def make_contract(self) -> Contract:
        """Create an IBKR Contract object for this symbol.

        Returns:
            Configured ib_insync Contract object.
        """
        from ib_insync import Contract

        contract = Contract()
        contract.symbol = self.ibkr_symbol
        contract.secType = self.ibkr_sec_type
        contract.exchange = self.ibkr_exchange
        contract.currency = self.ibkr_currency
        if self.ibkr_include_expired:
            contract.includeExpired = True
        return contract


# Pre-defined symbol configurations
SYMBOL_CONFIGS: dict[str, SymbolConfig] = {
    "XAUUSD": SymbolConfig(
        name="XAUUSD",
        ibkr_symbol="XAUUSD",
        ibkr_exchange="SMART",
        ibkr_sec_type="CFD",
        ibkr_currency="USD",
        timeframes=("5m", "15m", "1h", "4h"),
        is_tradeable=True,
    ),
    # Note: Requires NYBOT market data subscription
    "DXY": SymbolConfig(
        name="DXY",
        ibkr_symbol="DX",
        ibkr_exchange="NYBOT",
        ibkr_sec_type="IND",
        ibkr_currency="USD",
        ibkr_what_to_show="TRADES",  # Indices only support TRADES
        timeframes=("5m", "1h"),
        is_tradeable=False,
    ),
    # DX continuous futures - use this if you don't have index data permissions
    "DXY_FUT": SymbolConfig(
        name="DXY_FUT",
        ibkr_symbol="DX",
        ibkr_exchange="NYBOT",
        ibkr_sec_type="CONTFUT",  # Continuous futures
        ibkr_currency="USD",
        ibkr_what_to_show="TRADES",
        ibkr_include_expired=True,  # Needed for historical continuous data
        timeframes=("5m", "1h"),
        is_tradeable=False,
    ),
    "VIX": SymbolConfig(
        name="VIX",
        ibkr_symbol="VIX",
        ibkr_exchange="CBOE",
        ibkr_sec_type="IND",
        ibkr_currency="USD",
        ibkr_what_to_show="TRADES",  # Indices only support TRADES
        timeframes=("5m", "1h"),
        is_tradeable=False,
    ),
    # VIX continuous futures - use this if you don't have CBOE index data permissions
    "VIX_FUT": SymbolConfig(
        name="VIX_FUT",
        ibkr_symbol="VIX",
        ibkr_exchange="CFE",  # CBOE Futures Exchange
        ibkr_sec_type="CONTFUT",  # Continuous futures
        ibkr_currency="USD",
        ibkr_what_to_show="TRADES",
        ibkr_include_expired=True,
        timeframes=("5m", "1h"),
        is_tradeable=False,
    ),
    "EURUSD": SymbolConfig(
        name="EURUSD",
        ibkr_symbol="EUR",
        ibkr_exchange="IDEALPRO",
        ibkr_sec_type="CASH",
        ibkr_currency="USD",
        timeframes=("5m", "15m", "1h", "4h"),
        is_tradeable=True,
    ),
    "GBPUSD": SymbolConfig(
        name="GBPUSD",
        ibkr_symbol="GBP",
        ibkr_exchange="IDEALPRO",
        ibkr_sec_type="CASH",
        ibkr_currency="USD",
        timeframes=("5m", "15m", "1h", "4h"),
        is_tradeable=True,
    ),
    # Additional forex pairs for DXY calculation (free via IDEALPRO)
    "USDJPY": SymbolConfig(
        name="USDJPY",
        ibkr_symbol="USD",
        ibkr_exchange="IDEALPRO",
        ibkr_sec_type="CASH",
        ibkr_currency="JPY",
        timeframes=("5m",),
        is_tradeable=False,
    ),
    "USDCAD": SymbolConfig(
        name="USDCAD",
        ibkr_symbol="USD",
        ibkr_exchange="IDEALPRO",
        ibkr_sec_type="CASH",
        ibkr_currency="CAD",
        timeframes=("5m",),
        is_tradeable=False,
    ),
    "USDSEK": SymbolConfig(
        name="USDSEK",
        ibkr_symbol="USD",
        ibkr_exchange="IDEALPRO",
        ibkr_sec_type="CASH",
        ibkr_currency="SEK",
        timeframes=("5m",),
        is_tradeable=False,
    ),
    "USDCHF": SymbolConfig(
        name="USDCHF",
        ibkr_symbol="USD",
        ibkr_exchange="IDEALPRO",
        ibkr_sec_type="CASH",
        ibkr_currency="CHF",
        timeframes=("5m",),
        is_tradeable=False,
    ),
    # Synthetic DXY calculated from forex pairs (no IBKR fetch)
    "DXY_SYNTH": SymbolConfig(
        name="DXY_SYNTH",
        ibkr_symbol="",  # Not fetched from IBKR
        ibkr_exchange="",
        ibkr_sec_type="SYNTH",  # Synthetic type
        ibkr_currency="USD",
        timeframes=("5m", "15m", "1h", "4h"),
        is_tradeable=False,
    ),
}


def get_symbol_config(symbol: str) -> SymbolConfig:
    """Get configuration for a symbol.

    Args:
        symbol: Symbol name (e.g., "XAUUSD", "DXY").

    Returns:
        SymbolConfig for the requested symbol.

    Raises:
        ValueError: If the symbol is not configured.
    """
    symbol_upper = symbol.upper()
    if symbol_upper not in SYMBOL_CONFIGS:
        available = ", ".join(sorted(SYMBOL_CONFIGS.keys()))
        raise ValueError(f"Unknown symbol: {symbol}. Available: {available}")
    return SYMBOL_CONFIGS[symbol_upper]


def list_symbols() -> list[str]:
    """Return list of all configured symbol names."""
    return list(SYMBOL_CONFIGS.keys())
