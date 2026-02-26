"""Unit tests for the TradingGymEnv module."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import MagicMock, PropertyMock, call

import numpy as np
import pytest

from fxer.core.exceptions import GymError
from fxer.rl.gym.config import GymConfig
from fxer.rl.gym.trading_env import TradingGymEnv
from fxer.rl.types import EpisodeMode, GymAction, PositionState


def _create_mock_observation_builder(
    observation_size: int = 10,
    num_timestamps: int = 100,
) -> MagicMock:
    """Create a mock ObservationBuilder for testing."""
    mock = MagicMock()

    # Set properties
    mock.observation_size = observation_size
    mock.is_built = True

    # Create timestamps
    start = datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc)
    timestamps = [start + timedelta(minutes=5 * i) for i in range(num_timestamps)]
    mock.timestamps = timestamps

    # Mock methods
    mock.reset_normalizer = MagicMock()
    mock.get_observation = MagicMock(return_value=np.zeros(observation_size, dtype=np.float64))

    # Create sample bar
    from tests.conftest import make_normalized_bar
    sample_bar = make_normalized_bar()
    mock.get_bar = MagicMock(return_value=sample_bar)

    # Features and regime return None by default
    mock.get_features = MagicMock(return_value=None)
    mock.get_regime = MagicMock(return_value=None)

    return mock


def _create_mock_reward_function() -> MagicMock:
    """Create a mock BaseRewardFunction for testing."""
    mock = MagicMock()
    mock.compute = MagicMock(return_value=0.0)
    mock.reset = MagicMock()
    return mock


class TestTradingGymEnv:
    """Test TradingGymEnv gymnasium environment."""

    def test_initialization(self):
        """Verify environment initializes correctly."""
        config = GymConfig(episode_length=50)
        obs_builder = _create_mock_observation_builder(observation_size=10, num_timestamps=100)
        reward_fn = _create_mock_reward_function()

        env = TradingGymEnv(config, obs_builder, reward_fn, seed=42)

        # Check spaces
        assert env.action_space.n == 3  # FLAT, BUY, SELL
        assert env.observation_space.shape == (12,)  # 10 base + 2 position features
        assert env.observation_space.dtype == np.float64

        # Check internal state
        assert len(env._timestamps) == 100
        assert env._current_idx == 0
        assert env._episode_start_idx == 0
        assert env._steps_in_episode == 0

    def test_reset_training_mode(self):
        """Verify reset in training mode with random start."""
        config = GymConfig(mode=EpisodeMode.TRAINING, episode_length=20)
        obs_builder = _create_mock_observation_builder(observation_size=10, num_timestamps=100)
        reward_fn = _create_mock_reward_function()

        env = TradingGymEnv(config, obs_builder, reward_fn, seed=42)

        # Reset multiple times to check randomization
        start_indices = []
        for _ in range(5):
            obs, info = env.reset()
            start_indices.append(env._episode_start_idx)

            # Check observation shape
            assert obs.shape == (12,)
            assert obs.dtype == np.float64

            # Check info dict
            assert "timestamp" in info
            assert "balance" in info
            assert "equity" in info
            assert "position" in info
            assert info["step"] == 0

            # Normalizer should be reset in training
            obs_builder.reset_normalizer.assert_called()
            obs_builder.reset_normalizer.reset_mock()

        # Check that start indices vary (with seed, they should be deterministic but different)
        assert len(set(start_indices)) > 1

    def test_reset_evaluation_mode(self):
        """Verify reset in evaluation mode with sequential start."""
        config = GymConfig(mode=EpisodeMode.EVALUATION, episode_length=20)
        obs_builder = _create_mock_observation_builder(observation_size=10, num_timestamps=100)
        reward_fn = _create_mock_reward_function()

        env = TradingGymEnv(config, obs_builder, reward_fn)

        # Reset multiple times
        for _ in range(3):
            obs, info = env.reset()

            # Should always start at index 0 for evaluation
            assert env._episode_start_idx == 0

            # Normalizer should NOT be reset in evaluation
            obs_builder.reset_normalizer.assert_not_called()

    def test_reset_insufficient_timestamps(self):
        """Verify error when insufficient timestamps for episode."""
        config = GymConfig(mode=EpisodeMode.TRAINING, episode_length=150)
        obs_builder = _create_mock_observation_builder(observation_size=10, num_timestamps=100)
        reward_fn = _create_mock_reward_function()

        env = TradingGymEnv(config, obs_builder, reward_fn)

        with pytest.raises(GymError, match="Insufficient timestamps for episode"):
            env.reset()

    def test_step_buy_action_from_flat(self):
        """Verify BUY action opens long position from flat."""
        config = GymConfig(episode_length=50, lot_size=0.01)
        obs_builder = _create_mock_observation_builder(observation_size=10, num_timestamps=100)
        reward_fn = _create_mock_reward_function()

        env = TradingGymEnv(config, obs_builder, reward_fn)
        obs, info = env.reset()

        # Take BUY action
        obs, reward, terminated, truncated, info = env.step(GymAction.BUY)

        # Check position opened
        assert env._account.position_state == PositionState.LONG
        assert info["position"] == "LONG"
        assert env._steps_in_episode == 1

        # Reward function should have been called
        reward_fn.compute.assert_called_once()

    def test_step_sell_action_from_flat(self):
        """Verify SELL action opens short position from flat."""
        config = GymConfig(episode_length=50, lot_size=0.01)
        obs_builder = _create_mock_observation_builder(observation_size=10, num_timestamps=100)
        reward_fn = _create_mock_reward_function()

        env = TradingGymEnv(config, obs_builder, reward_fn)
        obs, info = env.reset()

        # Take SELL action
        obs, reward, terminated, truncated, info = env.step(GymAction.SELL)

        # Check position opened
        assert env._account.position_state == PositionState.SHORT
        assert info["position"] == "SHORT"

    def test_step_flat_action_closes_position(self):
        """Verify FLAT action closes open position."""
        config = GymConfig(episode_length=50, lot_size=0.01)
        obs_builder = _create_mock_observation_builder(observation_size=10, num_timestamps=100)
        reward_fn = _create_mock_reward_function()

        env = TradingGymEnv(config, obs_builder, reward_fn)
        obs, info = env.reset()

        # Open long position
        env.step(GymAction.BUY)

        # Close with FLAT
        obs, reward, terminated, truncated, info = env.step(GymAction.FLAT)

        # Check position closed
        assert env._account.position_state == PositionState.FLAT
        assert info["position"] == "FLAT"
        assert info["realized_pnl"] != 0  # Should have some P&L

    def test_step_reverse_position(self):
        """Verify BUY from SHORT and SELL from LONG reverse positions."""
        config = GymConfig(episode_length=50, lot_size=0.01)
        obs_builder = _create_mock_observation_builder(observation_size=10, num_timestamps=100)
        reward_fn = _create_mock_reward_function()

        env = TradingGymEnv(config, obs_builder, reward_fn)
        obs, info = env.reset()

        # Open long
        env.step(GymAction.BUY)
        assert env._account.position_state == PositionState.LONG

        # Reverse to short
        obs, reward, terminated, truncated, info = env.step(GymAction.SELL)
        assert env._account.position_state == PositionState.SHORT
        assert info["realized_pnl"] != 0  # Closed long

        # Reverse back to long
        obs, reward, terminated, truncated, info = env.step(GymAction.BUY)
        assert env._account.position_state == PositionState.LONG
        assert info["realized_pnl"] != 0  # Closed short

    def test_step_hold_position(self):
        """Verify repeating same action holds position (no-op)."""
        config = GymConfig(episode_length=50, lot_size=0.01)
        obs_builder = _create_mock_observation_builder(observation_size=10, num_timestamps=100)
        reward_fn = _create_mock_reward_function()

        env = TradingGymEnv(config, obs_builder, reward_fn)
        obs, info = env.reset()

        # Open long
        env.step(GymAction.BUY)
        initial_trades = env._account.total_trades

        # Try to buy again (should hold)
        obs, reward, terminated, truncated, info = env.step(GymAction.BUY)
        assert env._account.position_state == PositionState.LONG
        assert env._account.total_trades == initial_trades  # No new trade
        assert info["realized_pnl"] == 0  # No closing

    def test_step_invalid_action(self):
        """Verify error on invalid action value."""
        config = GymConfig(episode_length=50)
        obs_builder = _create_mock_observation_builder(observation_size=10, num_timestamps=100)
        reward_fn = _create_mock_reward_function()

        env = TradingGymEnv(config, obs_builder, reward_fn)
        env.reset()

        with pytest.raises(GymError, match="Invalid action: 99"):
            env.step(99)

    def test_step_after_episode_end(self):
        """Verify error when stepping after episode ends."""
        config = GymConfig(episode_length=50)
        obs_builder = _create_mock_observation_builder(observation_size=10, num_timestamps=51)
        reward_fn = _create_mock_reward_function()

        env = TradingGymEnv(config, obs_builder, reward_fn)
        env.reset()

        # Manually set to end of data
        env._current_idx = 50

        with pytest.raises(GymError, match="Step called after episode end"):
            env.step(GymAction.FLAT)

    def test_truncation_training_mode(self):
        """Verify truncation based on episode length in training mode."""
        config = GymConfig(mode=EpisodeMode.TRAINING, episode_length=10)
        obs_builder = _create_mock_observation_builder(observation_size=10, num_timestamps=100)
        reward_fn = _create_mock_reward_function()

        env = TradingGymEnv(config, obs_builder, reward_fn)
        env.reset()

        # Step through episode
        for i in range(9):
            obs, reward, terminated, truncated, info = env.step(GymAction.FLAT)
            assert not truncated
            assert not terminated

        # Last step should truncate
        obs, reward, terminated, truncated, info = env.step(GymAction.FLAT)
        assert truncated
        assert not terminated

    def test_truncation_evaluation_mode(self):
        """Verify truncation at end of data in evaluation mode."""
        config = GymConfig(mode=EpisodeMode.EVALUATION, episode_length=1000)
        obs_builder = _create_mock_observation_builder(observation_size=10, num_timestamps=20)
        reward_fn = _create_mock_reward_function()

        env = TradingGymEnv(config, obs_builder, reward_fn)
        env.reset()

        # Step through all available data
        for i in range(18):
            obs, reward, terminated, truncated, info = env.step(GymAction.FLAT)
            assert not truncated

        # Last step should truncate (at index 19, no more data)
        obs, reward, terminated, truncated, info = env.step(GymAction.FLAT)
        assert truncated

    def test_termination_on_liquidation(self):
        """Verify termination mechanism for liquidation exists."""
        config = GymConfig(
            episode_length=50,
            initial_balance=10_000.0,
            lot_size=0.01,
            margin_call_pct=50.0,
            leverage=100.0
        )
        obs_builder = _create_mock_observation_builder(observation_size=10, num_timestamps=100)
        reward_fn = _create_mock_reward_function()

        env = TradingGymEnv(config, obs_builder, reward_fn)
        env.reset()

        # Test that the liquidation check is in place
        # Manually set account to liquidated state
        env._account._is_liquidated = True

        # Step should return terminated=True
        obs, reward, terminated, truncated, info = env.step(GymAction.FLAT)
        assert terminated

    def test_force_close_on_episode_end(self):
        """Verify position is force-closed when episode ends."""
        config = GymConfig(mode=EpisodeMode.TRAINING, episode_length=5)
        obs_builder = _create_mock_observation_builder(observation_size=10, num_timestamps=100)
        reward_fn = _create_mock_reward_function()

        env = TradingGymEnv(config, obs_builder, reward_fn)
        env.reset()

        # Open position
        env.step(GymAction.BUY)
        assert env._account.position_state == PositionState.LONG

        # Step to end
        for i in range(3):
            env.step(GymAction.BUY)  # Hold position

        # Last step should force close
        obs, reward, terminated, truncated, info = env.step(GymAction.BUY)
        assert truncated
        assert env._account.position_state == PositionState.FLAT
        assert info["realized_pnl"] != 0  # Position was closed

    def test_observation_includes_position_features(self):
        """Verify observations include position state and unrealized P&L."""
        config = GymConfig(episode_length=50, initial_balance=10_000.0)
        obs_builder = _create_mock_observation_builder(observation_size=10, num_timestamps=100)
        reward_fn = _create_mock_reward_function()

        env = TradingGymEnv(config, obs_builder, reward_fn)

        # Initial observation (flat)
        obs, info = env.reset()
        assert obs[-2] == float(PositionState.FLAT)  # Position state
        assert obs[-1] == 0.0  # Unrealized P&L %

        # After opening long
        obs, reward, terminated, truncated, info = env.step(GymAction.BUY)
        assert obs[-2] == float(PositionState.LONG)

        # After opening short
        env.step(GymAction.SELL)  # Close long and open short
        obs, reward, terminated, truncated, info = env.step(GymAction.FLAT)  # Just to get obs
        assert obs[-2] == float(PositionState.FLAT)

    def test_reward_context_building(self):
        """Verify RewardContext is built correctly."""
        config = GymConfig(episode_length=50, lot_size=0.01)
        obs_builder = _create_mock_observation_builder(observation_size=10, num_timestamps=100)

        # Add features and regime
        from tests.conftest import make_normalized_bar
        from fxer.core.events import FeatureVector
        from fxer.regime.types import RegimeDecision, RegimeState

        features = FeatureVector(
            symbol="XAUUSD",
            timestamp=datetime.now(timezone.utc),
            timeframe=None,
            rsi_14=55.0,
            atr_14=3.5,
            warmup_complete=True
        )
        regime = RegimeDecision(
            state=RegimeState.LOW_VOL_TREND,
            confidence=0.85,
            position_multiplier=1.0,
            should_trade=True,
            reason="Test regime"
        )

        obs_builder.get_features = MagicMock(return_value=features)
        obs_builder.get_regime = MagicMock(return_value=regime)

        reward_fn = _create_mock_reward_function()

        env = TradingGymEnv(config, obs_builder, reward_fn)
        env.reset()

        # Take action to trigger reward computation
        env.step(GymAction.BUY)

        # Check reward function was called with proper context
        reward_fn.compute.assert_called_once()
        context = reward_fn.compute.call_args[0][0]

        assert context.action_taken == GymAction.BUY
        assert context.position_state == PositionState.LONG
        assert context.previous_position_state == PositionState.FLAT
        assert context.trade_just_opened is True
        assert context.trade_just_closed is False
        assert context.rsi_14 == 55.0
        assert context.atr_14 == 3.5
        assert context.regime_state == RegimeState.LOW_VOL_TREND
        assert context.regime_confidence == 0.85

    def test_cumulative_realized_pnl_tracking(self):
        """Verify cumulative realized P&L is tracked across trades."""
        config = GymConfig(episode_length=50, lot_size=0.01)
        obs_builder = _create_mock_observation_builder(observation_size=10, num_timestamps=100)

        # Mock bars with different prices
        from tests.conftest import make_normalized_bar
        bars = [
            make_normalized_bar(
                open_="2000.00",
                high="2010.00",
                low="1995.00",
                close="2000.00"
            ),
            make_normalized_bar(
                open_="2000.00",
                high="2015.00",
                low="2000.00",
                close="2010.00"
            ),  # Profit
            make_normalized_bar(
                open_="2010.00",
                high="2010.00",
                low="2000.00",
                close="2005.00"
            ),  # Loss
            make_normalized_bar(
                open_="2005.00",
                high="2020.00",
                low="2005.00",
                close="2015.00"
            ),  # Profit
        ]
        obs_builder.get_bar = MagicMock(side_effect=bars * 25)  # Repeat for multiple calls

        reward_fn = _create_mock_reward_function()

        env = TradingGymEnv(config, obs_builder, reward_fn)
        env.reset()

        # Trade 1: Buy and sell with profit
        env.step(GymAction.BUY)
        obs, reward, terminated, truncated, info = env.step(GymAction.FLAT)
        pnl1 = info["realized_pnl"]
        assert env._cumulative_realized_pnl == pnl1

        # Trade 2: Short and cover
        env.step(GymAction.SELL)
        obs, reward, terminated, truncated, info = env.step(GymAction.FLAT)
        pnl2 = info["realized_pnl"]
        assert env._cumulative_realized_pnl == pnl1 + pnl2

    def test_info_dict_contents(self):
        """Verify info dict contains expected keys and values."""
        config = GymConfig(episode_length=50, initial_balance=10_000.0)
        obs_builder = _create_mock_observation_builder(observation_size=10, num_timestamps=100)
        reward_fn = _create_mock_reward_function()

        env = TradingGymEnv(config, obs_builder, reward_fn)
        obs, info = env.reset()

        # Check reset info
        assert "timestamp" in info
        assert "balance" in info
        assert "equity" in info
        assert "position" in info
        assert "unrealized_pnl" in info
        assert "realized_pnl" in info
        assert "total_trades" in info
        assert "drawdown_pct" in info
        assert "step" in info

        assert info["balance"] == pytest.approx(10_000.0, abs=1e-2)
        assert info["equity"] == pytest.approx(10_000.0, abs=1e-2)
        assert info["position"] == "FLAT"
        assert info["unrealized_pnl"] == 0.0
        assert info["realized_pnl"] == 0.0
        assert info["total_trades"] == 0
        assert info["step"] == 0

        # Check step info
        obs, reward, terminated, truncated, info = env.step(GymAction.BUY)
        assert info["position"] == "LONG"
        assert info["step"] == 1

    def test_no_observation_fallback(self):
        """Verify graceful handling when observation is unavailable."""
        config = GymConfig(episode_length=50)
        obs_builder = _create_mock_observation_builder(observation_size=10, num_timestamps=100)

        # Make get_observation return None sometimes
        obs_builder.get_observation = MagicMock(
            side_effect=[None, np.ones(10, dtype=np.float64), None]
        )

        reward_fn = _create_mock_reward_function()

        env = TradingGymEnv(config, obs_builder, reward_fn)

        # Reset with missing observation
        obs, info = env.reset()
        assert obs.shape == (12,)
        assert np.all(obs[:10] == 0)  # Base observation should be zeros

        # Step with available observation
        obs, reward, terminated, truncated, info = env.step(GymAction.FLAT)
        assert np.all(obs[:10] == 1)  # Should use the provided observation

    def test_no_bar_data_error(self):
        """Verify error when bar data is unavailable."""
        config = GymConfig(episode_length=50)
        obs_builder = _create_mock_observation_builder(observation_size=10, num_timestamps=100)

        # Make get_bar return None
        obs_builder.get_bar = MagicMock(return_value=None)

        reward_fn = _create_mock_reward_function()

        env = TradingGymEnv(config, obs_builder, reward_fn)
        env.reset()

        with pytest.raises(GymError, match="No bar available"):
            env.step(GymAction.BUY)

    def test_seed_reproducibility(self):
        """Verify setting seed produces reproducible episodes."""
        config = GymConfig(mode=EpisodeMode.TRAINING, episode_length=20)
        obs_builder = _create_mock_observation_builder(observation_size=10, num_timestamps=100)
        reward_fn = _create_mock_reward_function()

        # First run with seed
        env1 = TradingGymEnv(config, obs_builder, reward_fn, seed=42)
        obs1, info1 = env1.reset()
        start1 = env1._episode_start_idx

        # Second run with same seed
        env2 = TradingGymEnv(config, obs_builder, reward_fn, seed=42)
        obs2, info2 = env2.reset()
        start2 = env2._episode_start_idx

        # Should get same start index
        assert start1 == start2

        # Third run with different seed
        env3 = TradingGymEnv(config, obs_builder, reward_fn, seed=123)
        obs3, info3 = env3.reset()
        start3 = env3._episode_start_idx

        # Should get different start index
        assert start1 != start3

    def test_render_and_close(self):
        """Verify render and close methods exist and don't error."""
        config = GymConfig()
        obs_builder = _create_mock_observation_builder()
        reward_fn = _create_mock_reward_function()

        env = TradingGymEnv(config, obs_builder, reward_fn)

        # Should not raise
        env.render()
        env.close()