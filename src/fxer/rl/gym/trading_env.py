"""Trading Gymnasium environment for RL agents."""

from __future__ import annotations

import logging
import random
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import gymnasium
import numpy as np

from fxer.core.exceptions import GymError
from fxer.rl.gym.account import AccountState
from fxer.rl.gym.config import GymConfig
from fxer.rl.rewards.context import RewardContext
from fxer.rl.types import EpisodeMode, GymAction, PositionState

if TYPE_CHECKING:
    from fxer.rl.observations.builder import ObservationBuilder
    from fxer.rl.rewards.base import BaseRewardFunction

logger = logging.getLogger(__name__)


class TradingGymEnv(gymnasium.Env):
    """Gymnasium environment for trading XAUUSD with RL agents.

    Provides a standard Gym interface for training RL agents on historical
    gold trading data. Handles position management, reward computation, and
    episode management with proper lookahead bias prevention.

    Key design choices:
    - Observation at bar[t], action executes at bar[t+1] close price
    - Position and unrealized P&L appended to observation vector
    - Random episode start for training, sequential for evaluation
    - Normalizer reset between training episodes to prevent leakage
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: GymConfig,
        observation_builder: ObservationBuilder,
        reward_fn: BaseRewardFunction,
        seed: int | None = None,
    ) -> None:
        """Initialize the trading environment.

        Args:
            config: Environment configuration.
            observation_builder: Pre-built observation builder with data.
            reward_fn: Reward function for computing step rewards.
            seed: Random seed for reproducible episodes.
        """
        super().__init__()
        self._config = config
        self._obs_builder = observation_builder
        self._reward_fn = reward_fn

        # Initialize account state
        self._account = AccountState(
            initial_balance=config.initial_balance,
            leverage=config.leverage,
            units_per_lot=config.units_per_lot,
            commission_per_lot=config.commission_per_lot,
            margin_call_pct=config.margin_call_pct,
            lot_size=config.lot_size,
        )

        # Define action and observation spaces
        # Observation: base features + position_state + unrealized_pnl_pct
        obs_size = observation_builder.observation_size + 2
        self.action_space = gymnasium.spaces.Discrete(3)  # FLAT, BUY, SELL
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        # Episode state
        self._timestamps = observation_builder.timestamps  # Pre-computed valid timestamps
        self._current_idx = 0
        self._episode_start_idx = 0
        self._steps_in_episode = 0
        self._cumulative_realized_pnl = 0.0

        # Set random seed if provided
        if seed is not None:
            self.action_space.seed(seed)
            random.seed(seed)
            np.random.seed(seed)

        logger.info(
            f"Initialized TradingGymEnv: obs_size={obs_size}, "
            f"available_timestamps={len(self._timestamps)}, "
            f"episode_length={config.episode_length}"
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment for a new episode.

        Args:
            seed: Random seed for this episode.
            options: Additional reset options (unused).

        Returns:
            Tuple of (observation, info_dict).

        Raises:
            GymError: If insufficient timestamps for episode.
        """
        super().reset(seed=seed)

        # Reset account and reward function
        self._account.reset()
        self._reward_fn.reset()
        self._steps_in_episode = 0
        self._cumulative_realized_pnl = 0.0

        # Determine episode start index
        if self._config.mode == EpisodeMode.TRAINING:
            # Reset normalizer for training mode to prevent leakage
            self._obs_builder.reset_normalizer()

            # Random start position for training
            max_start = len(self._timestamps) - self._config.episode_length
            if max_start < 0:
                raise GymError(
                    f"Insufficient timestamps for episode: need {self._config.episode_length}, "
                    f"have {len(self._timestamps)}",
                    episode_step=0,
                )

            self._episode_start_idx = random.randint(0, max(0, max_start))
        else:
            # Sequential start for evaluation
            self._episode_start_idx = 0

        self._current_idx = self._episode_start_idx

        # Get initial observation
        timestamp = self._timestamps[self._current_idx]
        base_obs = self._obs_builder.get_observation(timestamp)

        if base_obs is None:
            # Should not happen with properly built observations, but handle gracefully
            logger.warning(f"No observation available at {timestamp}, using zeros")
            base_obs = np.zeros(self._obs_builder.observation_size, dtype=np.float64)

        # Append position state and unrealized P&L percentage
        position_features = np.array(
            [
                float(self._account.position_state),
                self._account.unrealized_pnl / self._config.initial_balance,
            ],
            dtype=np.float64,
        )
        observation = np.concatenate([base_obs, position_features])

        # Build info dict
        info = {
            "timestamp": timestamp,
            "balance": self._account.balance,
            "equity": self._account.equity,
            "position": self._account.position_state.name,
            "unrealized_pnl": self._account.unrealized_pnl,
            "realized_pnl": 0.0,
            "total_trades": self._account.total_trades,
            "drawdown_pct": self._account.drawdown_pct,
            "step": self._steps_in_episode,
        }

        return observation, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one environment step.

        Args:
            action: Action to take (0=FLAT, 1=BUY, 2=SELL).

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).

        Raises:
            GymError: If step called after episode end.
        """
        # Convert action to enum
        try:
            gym_action = GymAction(action)
        except ValueError:
            raise GymError(
                f"Invalid action: {action}. Must be 0 (FLAT), 1 (BUY), or 2 (SELL)",
                episode_step=self._steps_in_episode,
            )

        # Save previous position state for reward context
        prev_position = self._account.position_state

        # Advance to next bar (lookahead bias prevention)
        self._current_idx += 1
        self._steps_in_episode += 1

        # Check if we've run out of data
        if self._current_idx >= len(self._timestamps):
            raise GymError(
                "Step called after episode end",
                episode_step=self._steps_in_episode,
            )

        # Get current bar for execution
        timestamp = self._timestamps[self._current_idx]
        bar = self._obs_builder.get_bar(timestamp)

        if bar is None:
            raise GymError(
                f"No bar available at {timestamp}",
                episode_step=self._steps_in_episode,
            )

        # Execute action at current bar's close price
        execution_price = bar.close
        trade_opened, trade_closed, realized_pnl = self._execute_action(
            gym_action, execution_price
        )

        # Update cumulative realized P&L
        self._cumulative_realized_pnl += realized_pnl

        # Update account with current price for unrealized P&L and liquidation check
        self._account.update_price(execution_price, self._steps_in_episode)

        # Check termination conditions
        terminated = self._account.is_liquidated

        # Check truncation conditions
        if self._config.mode == EpisodeMode.TRAINING:
            truncated = self._steps_in_episode >= self._config.episode_length
        else:
            truncated = self._current_idx >= len(self._timestamps) - 1

        # Force close position if episode ending
        if (terminated or truncated) and self._account.position_state != PositionState.FLAT:
            close_pnl = self._account.close_position(execution_price)
            realized_pnl += close_pnl
            self._cumulative_realized_pnl += close_pnl
            trade_closed = True

        # Get observation from new timestamp
        base_obs = self._obs_builder.get_observation(timestamp)

        if base_obs is None:
            logger.warning(f"No observation available at {timestamp}, using zeros")
            base_obs = np.zeros(self._obs_builder.observation_size, dtype=np.float64)

        # Append position state and unrealized P&L percentage
        position_features = np.array(
            [
                float(self._account.position_state),
                self._account.unrealized_pnl / self._config.initial_balance,
            ],
            dtype=np.float64,
        )
        observation = np.concatenate([base_obs, position_features])

        # Build reward context
        features = self._obs_builder.get_features(timestamp)
        regime = self._obs_builder.get_regime(timestamp)

        context = RewardContext(
            action_taken=gym_action,
            position_state=self._account.position_state,
            previous_position_state=prev_position,
            realized_pnl=realized_pnl,
            unrealized_pnl=self._account.unrealized_pnl,
            unrealized_pnl_pct=self._account.unrealized_pnl_pct,
            balance=self._account.balance,
            initial_balance=self._config.initial_balance,
            drawdown_pct=self._account.drawdown_pct,
            bars_held=self._account.bars_held,
            trade_just_opened=trade_opened,
            trade_just_closed=trade_closed,
            step_number=self._steps_in_episode,
            total_trades=self._account.total_trades,
            current_price=float(bar.close),
            atr_14=features.atr_14 if features else None,
            rsi_14=features.rsi_14 if features else None,
            regime_state=regime.state if regime else None,
            regime_confidence=regime.confidence if regime else None,
            observation=observation,
        )

        # Compute reward
        reward = self._reward_fn.compute(context)

        # Build info dict
        info = {
            "timestamp": timestamp,
            "balance": self._account.balance,
            "equity": self._account.equity,
            "position": self._account.position_state.name,
            "unrealized_pnl": self._account.unrealized_pnl,
            "realized_pnl": realized_pnl,
            "total_trades": self._account.total_trades,
            "drawdown_pct": self._account.drawdown_pct,
            "step": self._steps_in_episode,
        }

        return observation, reward, terminated, truncated, info

    def _execute_action(
        self, action: GymAction, price: Decimal
    ) -> tuple[bool, bool, float]:
        """Execute trading action and return outcomes.

        Args:
            action: Action to execute.
            price: Execution price.

        Returns:
            Tuple of (trade_opened, trade_closed, realized_pnl).
        """
        current_pos = self._account.position_state
        trade_opened = False
        trade_closed = False
        realized_pnl = 0.0

        if action == GymAction.BUY:
            if current_pos == PositionState.FLAT:
                # Open long position
                self._account.open_position(PositionState.LONG, price, self._steps_in_episode)
                trade_opened = True
            elif current_pos == PositionState.SHORT:
                # Close short and open long
                realized_pnl = self._account.close_position(price)
                trade_closed = True
                self._account.open_position(PositionState.LONG, price, self._steps_in_episode)
                trade_opened = True
            # If already LONG: no-op (hold position)

        elif action == GymAction.SELL:
            if current_pos == PositionState.FLAT:
                # Open short position
                self._account.open_position(PositionState.SHORT, price, self._steps_in_episode)
                trade_opened = True
            elif current_pos == PositionState.LONG:
                # Close long and open short
                realized_pnl = self._account.close_position(price)
                trade_closed = True
                self._account.open_position(PositionState.SHORT, price, self._steps_in_episode)
                trade_opened = True
            # If already SHORT: no-op (hold position)

        elif action == GymAction.FLAT:
            if current_pos == PositionState.LONG:
                # Close long position
                realized_pnl = self._account.close_position(price)
                trade_closed = True
            elif current_pos == PositionState.SHORT:
                # Close short position
                realized_pnl = self._account.close_position(price)
                trade_closed = True
            # If already FLAT: no-op

        return trade_opened, trade_closed, realized_pnl

    def render(self) -> None:
        """Render the environment (not implemented)."""
        pass

    def close(self) -> None:
        """Clean up resources."""
        pass