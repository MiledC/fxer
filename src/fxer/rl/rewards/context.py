"""Reward computation context for RL environments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fxer.regime.types import RegimeState
from fxer.rl.types import GymAction, PositionState


@dataclass(frozen=True, slots=True)
class RewardContext:
    """Context for computing rewards in the RL environment.

    Contains all necessary state information for reward functions to compute
    rewards based on the current step's outcome.

    Attributes:
        # Gym state
        action_taken: The action that was just executed.
        position_state: Current position state after the action.
        previous_position_state: Position state before the action.
        realized_pnl: Profit/loss from closed trades this step.
        unrealized_pnl: Current unrealized P&L if in position.
        unrealized_pnl_pct: Unrealized P&L as percentage.
        balance: Current account balance.
        initial_balance: Starting balance for episode.
        drawdown_pct: Current drawdown percentage.
        bars_held: Number of bars the current position has been held.
        trade_just_opened: Whether a new trade was opened this step.
        trade_just_closed: Whether a trade was closed this step.
        step_number: Current step number in the episode.
        total_trades: Total number of completed trades.

        # Market state
        current_price: Current market price.
        atr_14: 14-period Average True Range.
        rsi_14: 14-period Relative Strength Index.
        regime_state: Current market regime classification.
        regime_confidence: Confidence in regime classification (0-1).
        observation: Current observation vector.
    """

    # Gym state
    action_taken: GymAction
    position_state: PositionState
    previous_position_state: PositionState
    realized_pnl: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    balance: float
    initial_balance: float
    drawdown_pct: float
    bars_held: int
    trade_just_opened: bool
    trade_just_closed: bool
    step_number: int
    total_trades: int

    # Market state
    current_price: float
    atr_14: float | None = None
    rsi_14: float | None = None
    regime_state: RegimeState | None = None
    regime_confidence: float | None = None
    observation: np.ndarray | None = None