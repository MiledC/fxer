"""Profit and loss based reward function."""

from __future__ import annotations

from fxer.rl.rewards.base import BaseRewardFunction
from fxer.rl.rewards.context import RewardContext
from fxer.rl.types import PositionState


class PnLReward(BaseRewardFunction):
    """Reward function based on realized and unrealized P&L.

    This reward function provides positive rewards for profitable trades
    and negative rewards for losses. It can optionally normalize rewards
    by ATR to account for varying market volatility.

    Attributes:
        realized_weight: Weight for realized P&L when trade closes.
        unrealized_weight: Weight for unrealized P&L while in position.
        hold_penalty: Small penalty for holding flat position.
        normalize_by_atr: Whether to normalize P&L by ATR.
        atr_multiplier: Multiplier for ATR normalization.
    """

    def __init__(
        self,
        realized_weight: float = 1.0,
        unrealized_weight: float = 0.5,
        hold_penalty: float = -0.0001,
        normalize_by_atr: bool = False,
        atr_multiplier: float = 0.1,
    ) -> None:
        """Initialize PnL reward function.

        Args:
            realized_weight: Weight for realized P&L (default 1.0).
            unrealized_weight: Weight for unrealized P&L (default 0.5).
            hold_penalty: Penalty for holding flat position (default -0.0001).
            normalize_by_atr: Whether to normalize by ATR (default False).
            atr_multiplier: ATR normalization multiplier (default 0.1).
        """
        self.realized_weight = realized_weight
        self.unrealized_weight = unrealized_weight
        self.hold_penalty = hold_penalty
        self.normalize_by_atr = normalize_by_atr
        self.atr_multiplier = atr_multiplier

    @property
    def name(self) -> str:
        """Human-readable name of the reward function."""
        return "PnL Reward"

    def compute(self, context: RewardContext) -> float:
        """Compute reward based on P&L.

        Args:
            context: Current state and market context.

        Returns:
            Reward value for this step.
        """
        reward: float = 0.0

        # Determine base reward
        if context.trade_just_closed:
            # Trade just closed - use realized P&L
            reward = context.realized_pnl * self.realized_weight
        elif context.position_state in (PositionState.LONG, PositionState.SHORT):
            # Currently in position - use unrealized P&L
            reward = context.unrealized_pnl * self.unrealized_weight
        else:
            # Flat position, no trade closed - apply hold penalty
            reward = self.hold_penalty

        # Normalize by ATR if requested and available
        if (
            self.normalize_by_atr
            and context.atr_14 is not None
            and context.atr_14 > 0
        ):
            # Only normalize the P&L component, not the hold penalty
            if context.trade_just_closed or context.position_state != PositionState.FLAT:
                reward = reward / (context.atr_14 * self.atr_multiplier)

        return reward

    def reset(self) -> None:
        """Reset internal state (no-op for stateless PnL reward)."""
        pass  # Stateless reward function