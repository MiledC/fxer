"""Integration tests for the RL training pipeline."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import MagicMock

import numpy as np
import pytest

from fxer.core.events import FeatureVector
from fxer.regime.types import RegimeDecision, RegimeState
from fxer.rl.gym.account import AccountState
from fxer.rl.gym.config import GymConfig
from fxer.rl.gym.trading_env import TradingGymEnv
from fxer.rl.rewards.base import BaseRewardFunction
from fxer.rl.rewards.context import RewardContext
from fxer.rl.types import EpisodeMode, GymAction, PositionState
from tests.conftest import make_normalized_bar


class SimpleRewardFunction(BaseRewardFunction):
    """Simple reward function for integration testing."""

    @property
    def name(self) -> str:
        return "simple"

    def compute(self, context: RewardContext) -> float:
        """Simple P&L-based reward."""
        # Reward realized P&L
        if context.trade_just_closed:
            return context.realized_pnl / 100.0  # Scale down

        # Small penalty for holding
        if context.position_state != PositionState.FLAT:
            return -0.01

        return 0.0

    def reset(self) -> None:
        """No internal state to reset."""
        pass


def _create_integration_obs_builder(num_bars: int = 500) -> MagicMock:
    """Create a more realistic mock observation builder for integration tests."""
    mock = MagicMock()

    # Properties
    mock.observation_size = 20
    mock.is_built = True

    # Generate realistic price series
    np.random.seed(42)
    base_price = 2000.0
    prices = [base_price]

    for i in range(1, num_bars):
        # Random walk with slight upward drift
        change = np.random.normal(0.01, 2.0)  # Mean 0.01, std 2.0
        new_price = max(prices[-1] + change, 1500.0)  # Floor at 1500
        prices.append(new_price)

    # Create timestamps
    start = datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc)
    timestamps = [start + timedelta(minutes=5 * i) for i in range(num_bars)]
    mock.timestamps = timestamps

    # Create bars from prices
    bars = []
    for i, (ts, price) in enumerate(zip(timestamps, prices)):
        # Add some realistic OHLC variation
        open_p = price + np.random.uniform(-1, 1)
        high = max(open_p, price) + np.random.uniform(0, 2)
        low = min(open_p, price) - np.random.uniform(0, 2)
        bar = make_normalized_bar(
            timestamp=ts,
            open_=str(open_p),
            high=str(high),
            low=str(low),
            close=str(price),
            volume=str(np.random.randint(800, 2000))
        )
        bars.append(bar)

    # Mock methods
    mock.reset_normalizer = MagicMock()

    def get_observation(ts):
        idx = timestamps.index(ts) if ts in timestamps else 0
        # Create observation with some structure
        obs = np.zeros(20, dtype=np.float64)
        obs[0] = prices[idx] / 2000.0  # Normalized price
        obs[1] = (prices[idx] - prices[max(0, idx-1)]) / 100.0  # Return
        obs[2] = np.random.normal(0, 1)  # Random feature
        obs[3:10] = np.random.normal(0, 0.5, 7)  # More features
        return obs

    mock.get_observation = MagicMock(side_effect=get_observation)

    def get_bar(ts):
        idx = timestamps.index(ts) if ts in timestamps else 0
        return bars[idx]

    mock.get_bar = MagicMock(side_effect=get_bar)

    # Features with some correlation to price
    def get_features(ts):
        idx = timestamps.index(ts) if ts in timestamps else 0
        rsi = 50.0 + (prices[idx] - 2000.0) / 10.0  # RSI correlated with price
        rsi = max(20, min(80, rsi))  # Bound RSI
        return FeatureVector(
            symbol="XAUUSD",
            timestamp=ts,
            timeframe=None,
            rsi_14=rsi,
            atr_14=abs(np.random.normal(3.5, 0.5)),
            warmup_complete=idx > 50
        )

    mock.get_features = MagicMock(side_effect=get_features)

    # Regime based on price trend
    def get_regime(ts):
        idx = timestamps.index(ts) if ts in timestamps else 0
        if idx < 20:
            return None
        # Simple trend detection
        ma_short = np.mean(prices[max(0, idx-10):idx])
        ma_long = np.mean(prices[max(0, idx-50):idx])
        if ma_short > ma_long * 1.01:
            state = RegimeState.LOW_VOL_TREND
        elif ma_short < ma_long * 0.99:
            state = RegimeState.HIGH_VOL_TREND
        else:
            state = RegimeState.RANGING

        confidence = min(abs(ma_short - ma_long) / ma_long * 100, 1.0)
        return RegimeDecision(
            state=state,
            confidence=confidence,
            position_multiplier=1.0,
            should_trade=True,
            reason="Test regime"
        )

    mock.get_regime = MagicMock(side_effect=get_regime)

    return mock, bars, prices


class TestRLPipeline:
    """Integration tests for the complete RL pipeline."""

    def test_complete_episode_flow(self):
        """Test a complete episode from start to finish."""
        # Setup
        config = GymConfig(
            mode=EpisodeMode.TRAINING,
            episode_length=100,
            initial_balance=10_000.0,
            lot_size=0.01,
        )
        obs_builder, bars, prices = _create_integration_obs_builder(num_bars=500)
        reward_fn = SimpleRewardFunction()

        env = TradingGymEnv(config, obs_builder, reward_fn, seed=42)

        # Run episode
        obs, info = env.reset()
        assert info["balance"] == pytest.approx(10_000.0, abs=1)

        total_reward = 0.0
        actions_taken = []
        positions_held = []

        for step in range(config.episode_length):
            # Simple trading strategy for testing
            if step % 10 == 0:
                action = GymAction.BUY
            elif step % 10 == 5:
                action = GymAction.FLAT
            else:
                action = np.random.choice([GymAction.FLAT, GymAction.BUY, GymAction.SELL])

            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            actions_taken.append(action)
            positions_held.append(info["position"])

            if terminated or truncated:
                break

        # Verify episode completed
        assert step > 0
        assert len(actions_taken) > 0
        assert info["total_trades"] >= 0

        # Check that different positions were held
        unique_positions = set(positions_held)
        assert len(unique_positions) > 1  # Should have at least 2 different positions

    def test_multi_episode_training(self):
        """Test multiple training episodes with different random starts."""
        config = GymConfig(
            mode=EpisodeMode.TRAINING,
            episode_length=50,
            initial_balance=10_000.0,
            lot_size=0.01,
        )
        obs_builder, bars, prices = _create_integration_obs_builder(num_bars=500)
        reward_fn = SimpleRewardFunction()

        env = TradingGymEnv(config, obs_builder, reward_fn, seed=42)

        episode_starts = []
        episode_returns = []

        # Run multiple episodes
        for episode in range(5):
            obs, info = env.reset()
            episode_starts.append(env._episode_start_idx)

            episode_reward = 0.0
            for step in range(config.episode_length):
                # Random actions
                action = np.random.choice([GymAction.FLAT, GymAction.BUY, GymAction.SELL])
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward

                if terminated or truncated:
                    break

            episode_returns.append(episode_reward)

        # Verify different starting points
        assert len(set(episode_starts)) > 1  # Should have different starts

        # Verify returns vary
        assert len(set(episode_returns)) > 1  # Different episodes should have different returns

    def test_evaluation_mode_sequential(self):
        """Test evaluation mode processes data sequentially."""
        config = GymConfig(
            mode=EpisodeMode.EVALUATION,
            episode_length=1000,  # Large enough to not matter
            initial_balance=10_000.0,
            lot_size=0.01,
        )
        obs_builder, bars, prices = _create_integration_obs_builder(num_bars=100)
        reward_fn = SimpleRewardFunction()

        env = TradingGymEnv(config, obs_builder, reward_fn)

        # First episode
        obs, info = env.reset()
        assert env._episode_start_idx == 0

        steps_taken = 0
        for step in range(200):  # Try to go beyond available data
            action = GymAction.FLAT
            obs, reward, terminated, truncated, info = env.step(action)
            steps_taken += 1

            if terminated or truncated:
                break

        # Should truncate when data runs out
        assert truncated
        assert steps_taken < 100  # Can't exceed available data

    def test_position_management_integration(self):
        """Test position management with realistic price movements."""
        config = GymConfig(
            episode_length=100,
            initial_balance=10_000.0,
            lot_size=0.1,  # Larger position
            commission_per_lot=7.0,
        )
        obs_builder, bars, prices = _create_integration_obs_builder(num_bars=500)
        reward_fn = SimpleRewardFunction()

        env = TradingGymEnv(config, obs_builder, reward_fn)
        obs, info = env.reset()

        # Track P&L
        realized_pnls = []
        balances = []

        # Execute some trades
        # Buy
        obs, reward, terminated, truncated, info = env.step(GymAction.BUY)
        assert info["position"] == "LONG"
        balances.append(info["balance"])

        # Hold for a few steps
        for _ in range(5):
            obs, reward, terminated, truncated, info = env.step(GymAction.BUY)

        # Close
        obs, reward, terminated, truncated, info = env.step(GymAction.FLAT)
        assert info["position"] == "FLAT"
        realized_pnls.append(info["realized_pnl"])
        balances.append(info["balance"])

        # Short
        obs, reward, terminated, truncated, info = env.step(GymAction.SELL)
        assert info["position"] == "SHORT"

        # Hold
        for _ in range(3):
            obs, reward, terminated, truncated, info = env.step(GymAction.SELL)

        # Cover
        obs, reward, terminated, truncated, info = env.step(GymAction.FLAT)
        realized_pnls.append(info["realized_pnl"])
        balances.append(info["balance"])

        # Verify trades were executed
        assert len(realized_pnls) == 2
        assert info["total_trades"] == 2

        # Balance should change based on P&L
        assert balances[-1] != 10_000.0

    def test_liquidation_scenario(self):
        """Test that liquidation mechanism is integrated properly."""
        config = GymConfig(
            episode_length=100,
            initial_balance=10_000.0,
            lot_size=0.01,
            leverage=100.0,
            margin_call_pct=50.0,
        )

        obs_builder, bars, prices = _create_integration_obs_builder(num_bars=200)
        reward_fn = SimpleRewardFunction()

        env = TradingGymEnv(config, obs_builder, reward_fn)
        obs, info = env.reset()

        # Test that the liquidation mechanism exists and works
        # Force account into liquidated state
        env._account._is_liquidated = True

        # Next step should terminate
        obs, reward, terminated, truncated, info = env.step(GymAction.FLAT)
        assert terminated
        assert env._account.is_liquidated

    def test_reward_function_integration(self):
        """Test reward function receives correct context."""
        config = GymConfig(
            episode_length=50,
            initial_balance=10_000.0,
            lot_size=0.01,
        )
        obs_builder, bars, prices = _create_integration_obs_builder(num_bars=200)

        # Custom reward function that tracks calls
        reward_calls = []

        class TrackingRewardFunction(BaseRewardFunction):
            @property
            def name(self) -> str:
                return "tracking"

            def compute(self, context: RewardContext) -> float:
                reward_calls.append({
                    'action': context.action_taken,
                    'position': context.position_state,
                    'prev_position': context.previous_position_state,
                    'trade_opened': context.trade_just_opened,
                    'trade_closed': context.trade_just_closed,
                    'realized_pnl': context.realized_pnl,
                    'step': context.step_number,
                })
                return 0.1 if context.trade_just_closed else 0.0

            def reset(self) -> None:
                pass

        reward_fn = TrackingRewardFunction()

        env = TradingGymEnv(config, obs_builder, reward_fn)
        obs, info = env.reset()

        # Execute trades
        env.step(GymAction.BUY)  # Open long
        env.step(GymAction.FLAT)  # Close
        env.step(GymAction.SELL)  # Open short
        env.step(GymAction.BUY)  # Close short and open long

        # Verify reward function was called correctly
        assert len(reward_calls) == 4

        # First call - open long
        assert reward_calls[0]['action'] == GymAction.BUY
        assert reward_calls[0]['position'] == PositionState.LONG
        assert reward_calls[0]['prev_position'] == PositionState.FLAT
        assert reward_calls[0]['trade_opened'] is True
        assert reward_calls[0]['trade_closed'] is False

        # Second call - close long
        assert reward_calls[1]['action'] == GymAction.FLAT
        assert reward_calls[1]['position'] == PositionState.FLAT
        assert reward_calls[1]['prev_position'] == PositionState.LONG
        assert reward_calls[1]['trade_closed'] is True
        assert reward_calls[1]['realized_pnl'] != 0

    def test_observation_normalization_reset(self):
        """Test that normalizer is reset between training episodes."""
        config = GymConfig(
            mode=EpisodeMode.TRAINING,
            episode_length=20,
        )
        obs_builder, bars, prices = _create_integration_obs_builder(num_bars=200)
        reward_fn = SimpleRewardFunction()

        env = TradingGymEnv(config, obs_builder, reward_fn)

        # Run first episode
        env.reset()
        for _ in range(10):
            env.step(GymAction.FLAT)

        # Reset for second episode
        env.reset()

        # Normalizer should have been reset
        assert obs_builder.reset_normalizer.call_count == 2

    def test_action_space_and_observation_space(self):
        """Test Gym space definitions are correct."""
        config = GymConfig(episode_length=50)  # Reduce episode length to fit available data
        obs_builder, bars, prices = _create_integration_obs_builder(num_bars=100)
        reward_fn = SimpleRewardFunction()

        env = TradingGymEnv(config, obs_builder, reward_fn)

        # Action space
        assert env.action_space.n == 3
        assert env.action_space.contains(0)  # FLAT
        assert env.action_space.contains(1)  # BUY
        assert env.action_space.contains(2)  # SELL
        assert not env.action_space.contains(3)

        # Observation space
        obs, info = env.reset()
        assert env.observation_space.contains(obs)
        assert obs.dtype == np.float64
        assert obs.shape == (obs_builder.observation_size + 2,)

    def test_episode_with_features_and_regime(self):
        """Test episode with full features and regime information."""
        config = GymConfig(
            episode_length=30,
            initial_balance=10_000.0,
            lot_size=0.01,
        )
        obs_builder, bars, prices = _create_integration_obs_builder(num_bars=200)
        reward_fn = SimpleRewardFunction()

        env = TradingGymEnv(config, obs_builder, reward_fn)
        obs, info = env.reset()

        # Run episode and verify features are used
        for step in range(30):
            action = GymAction.FLAT if step % 5 == 0 else GymAction.BUY
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                break

        # Verify get_features and get_regime were called
        assert obs_builder.get_features.called
        assert obs_builder.get_regime.called

    def test_end_to_end_profitability(self):
        """Test that profitable trades increase balance."""
        config = GymConfig(
            episode_length=50,
            initial_balance=10_000.0,
            lot_size=0.1,
            commission_per_lot=7.0,
        )

        # Create predictable price series
        mock = MagicMock()
        mock.observation_size = 10
        mock.is_built = True

        start = datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc)
        timestamps = [start + timedelta(minutes=5 * i) for i in range(100)]
        mock.timestamps = timestamps

        # Predictable upward trend - create valid OHLC bars
        bars = []
        for i in range(100):
            base_price = 2000.0 + i * 0.5  # Steady increase
            bars.append(make_normalized_bar(
                open_=str(base_price - 0.5),
                high=str(base_price + 1.0),
                low=str(base_price - 1.0),
                close=str(base_price)
            ))

        mock.reset_normalizer = MagicMock()
        mock.get_observation = MagicMock(return_value=np.zeros(10, dtype=np.float64))
        mock.get_bar = MagicMock(side_effect=bars)
        mock.get_features = MagicMock(return_value=None)
        mock.get_regime = MagicMock(return_value=None)

        reward_fn = SimpleRewardFunction()

        env = TradingGymEnv(config, mock, reward_fn)
        obs, info = env.reset()

        initial_balance = info["balance"]

        # Buy early in uptrend
        obs, reward, terminated, truncated, info = env.step(GymAction.BUY)

        # Hold for several steps as price rises
        for _ in range(10):
            obs, reward, terminated, truncated, info = env.step(GymAction.BUY)

        # Sell after price has risen
        obs, reward, terminated, truncated, info = env.step(GymAction.FLAT)

        # Should have made profit (minus commission)
        final_balance = info["balance"]
        assert final_balance > initial_balance

        # Realized P&L should be positive
        assert info["realized_pnl"] > 0