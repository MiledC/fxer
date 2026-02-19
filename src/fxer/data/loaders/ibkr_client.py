"""IBKR connection client for TWS/Gateway.

Manages connection lifecycle and provides contract creation utilities.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ib_insync import IB, Contract

from fxer.config.settings import Settings
from fxer.config.settings import settings as default_settings
from fxer.config.symbols import get_symbol_config

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class IBKRConnectionError(Exception):
    """Raised when connection to IBKR fails."""

    pass


class IBKRClient:
    """Connection manager for Interactive Brokers TWS/Gateway.

    Handles connection lifecycle and provides utilities for contract
    creation and validation.

    Usage::

        client = IBKRClient(settings)
        await client.connect()

        contract = client.make_contract("XAUUSD")
        # ... use contract for historical data or streaming

        await client.disconnect()
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize IBKR client.

        Args:
            settings: Application settings. Uses global settings if not provided.
        """
        self._settings = settings or default_settings
        self._ib = IB()

    @property
    def ib(self) -> IB:
        """Access the underlying ib_insync IB instance."""
        return self._ib

    @property
    def is_connected(self) -> bool:
        """Check if connected to TWS/Gateway."""
        return self._ib.isConnected()

    async def connect(self) -> None:
        """Connect to TWS/Gateway.

        Raises:
            IBKRConnectionError: If connection fails.
        """
        if self.is_connected:
            logger.debug("Already connected to IBKR")
            return

        try:
            await self._ib.connectAsync(
                host=self._settings.ibkr_host,
                port=self._settings.ibkr_port,
                clientId=self._settings.ibkr_client_id,
                readonly=self._settings.ibkr_readonly,
                timeout=self._settings.ibkr_timeout,
            )
            logger.info(
                "Connected to IBKR at %s:%d (clientId=%d, readonly=%s)",
                self._settings.ibkr_host,
                self._settings.ibkr_port,
                self._settings.ibkr_client_id,
                self._settings.ibkr_readonly,
            )
        except Exception as exc:
            raise IBKRConnectionError(
                f"Failed to connect to IBKR at {self._settings.ibkr_host}:"
                f"{self._settings.ibkr_port}: {exc}"
            ) from exc

    async def disconnect(self) -> None:
        """Disconnect from TWS/Gateway."""
        if self.is_connected:
            self._ib.disconnect()
            logger.info("Disconnected from IBKR")

    def make_contract(self, symbol: str) -> Contract:
        """Create an IBKR contract for the given symbol.

        Uses the centralized symbol configuration to create the
        appropriate contract type.

        Args:
            symbol: Symbol name (e.g., "XAUUSD", "EURUSD", "DXY").

        Returns:
            Configured IBKR Contract object.

        Raises:
            ValueError: If the symbol is not configured.
        """
        config = get_symbol_config(symbol)
        return config.make_contract()

    async def qualify_contract(self, contract: Contract) -> Contract:
        """Qualify a contract with IBKR to get full details.

        Args:
            contract: Contract to qualify.

        Returns:
            Qualified contract with full details from IBKR.

        Raises:
            ValueError: If contract cannot be qualified.
        """
        qualified = await self._ib.qualifyContractsAsync(contract)
        if not qualified:
            raise ValueError(f"Could not qualify contract: {contract}")
        return qualified[0]

    async def __aenter__(self) -> IBKRClient:
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, *args: object) -> None:
        """Async context manager exit."""
        await self.disconnect()
