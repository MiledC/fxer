"""Base class for reward functions."""

from __future__ import annotations

from abc import ABC, abstractmethod

from fxer.rl.rewards.context import RewardContext


class BaseRewardFunction(ABC):
    """Abstract base class for reward functions.

    All reward functions must implement the compute method to calculate
    rewards based on the current context, and reset method to clear any
    internal state between episodes.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the reward function."""
        ...

    @abstractmethod
    def compute(self, context: RewardContext) -> float:
        """Compute reward for the current step.

        Args:
            context: Current state and market context.

        Returns:
            Reward value for this step.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset any internal state between episodes."""
        ...