"""Gymnasium environment configuration."""

from dataclasses import dataclass
from datetime import datetime

from fxer.rl.types import EpisodeMode


@dataclass(frozen=True, slots=True)
class GymConfig:
    """Configuration for the trading gym environment.

    Attributes:
        symbol: Trading symbol (default: XAUUSD).
        start_date: Start date for episode data selection.
        end_date: End date for episode data selection.
        mode: Episode mode (training or evaluation).
        initial_balance: Starting account balance in USD.
        leverage: Account leverage ratio.
        units_per_lot: Units per standard lot (100 oz for gold).
        commission_per_lot: Round-trip commission per lot in USD.
        episode_length: Number of bars per episode.
        lot_size: Trading lot size (0.01 = micro lot).
        margin_call_pct: Margin call threshold as percentage of balance.
    """

    symbol: str = "XAUUSD"
    start_date: datetime | None = None
    end_date: datetime | None = None
    mode: EpisodeMode = EpisodeMode.TRAINING
    initial_balance: float = 10_000.0
    leverage: float = 100.0
    units_per_lot: float = 100.0
    commission_per_lot: float = 7.0
    episode_length: int = 2016
    lot_size: float = 0.01
    margin_call_pct: float = 50.0

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.initial_balance <= 0:
            raise ValueError(f"initial_balance must be positive, got {self.initial_balance}")
        if self.leverage <= 0:
            raise ValueError(f"leverage must be positive, got {self.leverage}")
        if self.units_per_lot <= 0:
            raise ValueError(f"units_per_lot must be positive, got {self.units_per_lot}")
        if self.commission_per_lot < 0:
            raise ValueError(f"commission_per_lot cannot be negative, got {self.commission_per_lot}")
        if self.episode_length <= 0:
            raise ValueError(f"episode_length must be positive, got {self.episode_length}")
        if self.lot_size <= 0:
            raise ValueError(f"lot_size must be positive, got {self.lot_size}")
        if not 0 < self.margin_call_pct <= 100:
            raise ValueError(f"margin_call_pct must be between 0 and 100, got {self.margin_call_pct}")

    @property
    def margin_requirement(self) -> float:
        """Calculate margin requirement per lot."""
        # Assuming a typical gold price around $2000
        # This should be calculated dynamically based on current price
        typical_price = 2000.0
        return (self.units_per_lot * typical_price) / self.leverage

    @property
    def max_position_lots(self) -> float:
        """Calculate maximum position size in lots based on margin."""
        # Conservative estimate using typical price
        return (self.initial_balance * 0.95) / self.margin_requirement