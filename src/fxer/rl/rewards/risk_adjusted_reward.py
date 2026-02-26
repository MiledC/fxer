"""Risk-adjusted composite reward function with regime awareness."""

from __future__ import annotations

from fxer.regime.types import RegimeState
from fxer.rl.rewards.base import BaseRewardFunction
from fxer.rl.rewards.context import RewardContext
from fxer.rl.rewards.pnl_reward import PnLReward
from fxer.rl.types import PositionState


class RiskAdjustedReward(BaseRewardFunction):
    """Composite reward function with multiple risk components.

    Combines a base reward function (default PnLReward) with additional
    risk-based adjustments including drawdown penalties, regime bonuses,
    overtrading penalties, and penalties for holding positions too long.

    Attributes:
        drawdown_threshold: Drawdown level before penalties apply.
        drawdown_penalty_scale: Scale factor for drawdown penalties.
        regime_bonus_scale: Scale factor for regime-aligned bonuses.
        overtrading_penalty: Penalty for changing positions (covers spread).
        max_hold_bars: Maximum bars to hold position before penalties.
        hold_penalty_per_bar: Penalty per bar beyond max_hold_bars.
    """

    def __init__(
        self,
        base_reward: BaseRewardFunction | None = None,
        drawdown_threshold: float = 0.02,
        drawdown_penalty_scale: float = 10.0,
        regime_bonus_scale: float = 0.1,
        overtrading_penalty: float = -0.02,
        max_hold_bars: int = 48,
        hold_penalty_per_bar: float = -0.0002,
    ) -> None:
        """Initialize risk-adjusted reward function.

        Args:
            base_reward: Base reward function to build upon (default PnLReward).
            drawdown_threshold: Drawdown level before penalties (default 0.02).
            drawdown_penalty_scale: Scale for drawdown penalties (default 10.0).
            regime_bonus_scale: Scale for regime bonuses (default 0.1).
            overtrading_penalty: Penalty for position changes (default -0.02).
            max_hold_bars: Max bars before hold penalties (default 48).
            hold_penalty_per_bar: Penalty per bar over max (default -0.0002).
        """
        self._base_reward = base_reward if base_reward is not None else PnLReward()
        self.drawdown_threshold = drawdown_threshold
        self.drawdown_penalty_scale = drawdown_penalty_scale
        self.regime_bonus_scale = regime_bonus_scale
        self.overtrading_penalty = overtrading_penalty
        self.max_hold_bars = max_hold_bars
        self.hold_penalty_per_bar = hold_penalty_per_bar

    @property
    def name(self) -> str:
        """Human-readable name of the reward function."""
        return "Risk-Adjusted Reward"

    def compute(self, context: RewardContext) -> float:
        """Compute risk-adjusted reward for the current step.

        Applies the following adjustments to the base reward:
        1. Drawdown penalty when exceeding threshold
        2. Regime bonus/penalty based on position alignment
        3. Overtrading penalty for position changes
        4. Hold penalty for positions held too long

        Args:
            context: Current state and market context.

        Returns:
            Risk-adjusted reward value for this step.
        """
        # Start with base reward
        reward = self._base_reward.compute(context)

        # Apply drawdown penalty if above threshold
        if context.drawdown_pct > self.drawdown_threshold:
            reward += -(context.drawdown_pct - self.drawdown_threshold) * self.drawdown_penalty_scale

        # Apply regime bonus/penalty if in position
        if (
            context.regime_state is not None
            and context.position_state in (PositionState.LONG, PositionState.SHORT)
        ):
            if context.regime_state in (RegimeState.LOW_VOL_TREND, RegimeState.HIGH_VOL_TREND):
                # Bonus for trading in trending regimes
                reward += self.regime_bonus_scale
            elif context.regime_state == RegimeState.RANGING:
                # Penalty for trading in ranging markets
                reward += -self.regime_bonus_scale

        # Apply overtrading penalty for position changes
        if context.position_state != context.previous_position_state:
            reward += self.overtrading_penalty

        # Apply hold penalty for positions held too long
        if context.bars_held > self.max_hold_bars:
            reward += (context.bars_held - self.max_hold_bars) * self.hold_penalty_per_bar

        return reward

    def reset(self) -> None:
        """Reset internal state including base reward function."""
        self._base_reward.reset()
