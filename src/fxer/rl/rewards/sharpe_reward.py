"""Differential Sharpe ratio reward function."""

from __future__ import annotations

import numpy as np

from fxer.rl.rewards.base import BaseRewardFunction
from fxer.rl.rewards.context import RewardContext


class SharpeReward(BaseRewardFunction):
    """Reward function based on differential Sharpe ratio.

    Implements the differential Sharpe ratio reward from Moody & Saffell (2001).
    This reward function encourages strategies that maximize risk-adjusted returns
    by computing the instantaneous change in Sharpe ratio.

    Attributes:
        eta: EMA learning rate for updating return statistics.
        scale: Output scaling factor for reward values.
        min_steps: Minimum steps before computing non-zero rewards (warmup period).
    """

    def __init__(
        self,
        eta: float = 0.01,
        scale: float = 100.0,
        min_steps: int = 20,
    ) -> None:
        """Initialize differential Sharpe ratio reward function.

        Args:
            eta: EMA learning rate (default 0.01).
            scale: Output scaling factor (default 100.0).
            min_steps: Warmup period before non-zero rewards (default 20).
        """
        self.eta = eta
        self.scale = scale
        self.min_steps = min_steps

        # Internal state
        self._A: float = 0.0  # EMA of returns
        self._B: float = 0.0  # EMA of squared returns
        self._prev_value: float = 0.0  # Previous portfolio balance
        self._step_count: int = 0

    @property
    def name(self) -> str:
        """Human-readable name of the reward function."""
        return "Sharpe Reward"

    def compute(self, context: RewardContext) -> float:
        """Compute differential Sharpe ratio reward.

        Args:
            context: Current state and market context.

        Returns:
            Clipped differential Sharpe ratio reward in [-1.0, 1.0].
        """
        # Step 1: First step initialization
        if self._prev_value == 0.0:
            self._prev_value = context.balance
            return 0.0

        # Step 2: Calculate return
        return_t = 0.0
        if self._prev_value > 0:
            return_t = (context.balance - self._prev_value) / self._prev_value

        # Step 3: Warmup period - update EMAs but return 0
        if self._step_count < self.min_steps:
            dA = return_t - self._A
            dB = return_t**2 - self._B
            self._A += self.eta * dA
            self._B += self.eta * dB
            self._prev_value = context.balance
            self._step_count += 1
            return 0.0

        # Step 4: Compute deltas using OLD values
        dA = return_t - self._A
        dB = return_t**2 - self._B

        # Step 5: Compute variance
        variance = self._B - self._A**2

        # Step 6: Check for minimum variance
        if variance < 1e-6:
            # Update EMAs even when variance is too small
            self._A += self.eta * dA
            self._B += self.eta * dB
            self._prev_value = context.balance
            self._step_count += 1
            return 0.0

        # Step 7: Compute denominator
        denom = variance ** 1.5 + 1e-8

        # Step 8: Compute differential Sharpe ratio
        dS = (self._B * dA - 0.5 * self._A * dB) / denom

        # Step 9: Update EMAs AFTER computing dS
        self._A += self.eta * dA
        self._B += self.eta * dB

        # Step 10: Update state
        self._prev_value = context.balance
        self._step_count += 1

        # Step 11: Return clipped and scaled reward
        return np.clip(dS * self.scale, -1.0, 1.0)

    def reset(self) -> None:
        """Reset internal state between episodes."""
        self._A = 0.0
        self._B = 0.0
        self._prev_value = 0.0
        self._step_count = 0
