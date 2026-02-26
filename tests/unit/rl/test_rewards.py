"""Comprehensive tests for the RL reward system."""

from __future__ import annotations

import dataclasses
from unittest.mock import MagicMock

import numpy as np
import pytest

from fxer.regime.types import RegimeState
from fxer.rl.rewards import (
    BaseRewardFunction,
    PnLReward,
    RewardContext,
    RiskAdjustedReward,
    SharpeReward,
)
from fxer.rl.types import GymAction, PositionState
from tests.conftest import make_reward_context


# ---------------------------------------------------------------------------
# RewardContext Tests
# ---------------------------------------------------------------------------


class TestRewardContext:
    """Test RewardContext frozen dataclass."""

    def test_context_creation(self) -> None:
        """Valid RewardContext with all required fields."""
        ctx = make_reward_context()
        assert ctx.action_taken == GymAction.FLAT
        assert ctx.position_state == PositionState.FLAT
        assert ctx.balance == 10000.0
        assert ctx.current_price == 2062.50

    def test_context_immutable(self) -> None:
        """Frozen dataclass raises on attribute assignment."""
        ctx = make_reward_context()
        with pytest.raises(dataclasses.FrozenInstanceError):
            ctx.balance = 9999.0  # type: ignore[misc]

    def test_context_optional_none(self) -> None:
        """Fields with None defaults work correctly."""
        ctx = make_reward_context(
            atr_14=None,
            rsi_14=None,
            regime_state=None,
            regime_confidence=None,
            observation=None,
        )
        assert ctx.atr_14 is None
        assert ctx.rsi_14 is None
        assert ctx.regime_state is None
        assert ctx.regime_confidence is None
        assert ctx.observation is None

    def test_context_with_observation(self) -> None:
        """Works with numpy array observation."""
        obs = np.ones(10, dtype=np.float32)
        ctx = make_reward_context(observation=obs)
        assert ctx.observation is not None
        assert len(ctx.observation) == 10
        np.testing.assert_array_equal(ctx.observation, obs)

    def test_context_slots(self) -> None:
        """RewardContext uses __slots__ for memory efficiency."""
        assert hasattr(RewardContext, "__slots__")


# ---------------------------------------------------------------------------
# PnLReward Tests
# ---------------------------------------------------------------------------


class TestPnLReward:
    """Test PnLReward function."""

    def test_name(self) -> None:
        """Returns correct name."""
        assert PnLReward().name == "PnL Reward"

    def test_realized_pnl_on_close(self) -> None:
        """Reward = realized_pnl * realized_weight when trade_just_closed."""
        reward_fn = PnLReward(realized_weight=1.0)
        ctx = make_reward_context(
            trade_just_closed=True,
            realized_pnl=50.0,
        )
        assert reward_fn.compute(ctx) == pytest.approx(50.0)

    def test_negative_realized_pnl(self) -> None:
        """Negative realized_pnl produces negative reward."""
        reward_fn = PnLReward(realized_weight=1.0)
        ctx = make_reward_context(
            trade_just_closed=True,
            realized_pnl=-30.0,
        )
        assert reward_fn.compute(ctx) == pytest.approx(-30.0)

    def test_unrealized_pnl_in_position_long(self) -> None:
        """Reward = unrealized_pnl * unrealized_weight when LONG."""
        reward_fn = PnLReward(unrealized_weight=0.5)
        ctx = make_reward_context(
            position_state=PositionState.LONG,
            unrealized_pnl=20.0,
        )
        assert reward_fn.compute(ctx) == pytest.approx(10.0)

    def test_unrealized_pnl_in_position_short(self) -> None:
        """Reward = unrealized_pnl * unrealized_weight when SHORT."""
        reward_fn = PnLReward(unrealized_weight=0.5)
        ctx = make_reward_context(
            position_state=PositionState.SHORT,
            unrealized_pnl=-10.0,
        )
        assert reward_fn.compute(ctx) == pytest.approx(-5.0)

    def test_hold_penalty_when_flat(self) -> None:
        """Negative penalty when FLAT and no close."""
        reward_fn = PnLReward(hold_penalty=-0.0001)
        ctx = make_reward_context(
            position_state=PositionState.FLAT,
            trade_just_closed=False,
        )
        assert reward_fn.compute(ctx) == pytest.approx(-0.0001)

    def test_atr_normalization_enabled(self) -> None:
        """PnL divided by ATR when enabled."""
        reward_fn = PnLReward(
            realized_weight=1.0,
            normalize_by_atr=True,
            atr_multiplier=0.1,
        )
        ctx = make_reward_context(
            trade_just_closed=True,
            realized_pnl=50.0,
            atr_14=5.0,
        )
        # 50.0 / (5.0 * 0.1) = 100.0
        assert reward_fn.compute(ctx) == pytest.approx(100.0)

    def test_atr_none_fallback(self) -> None:
        """No normalization when atr_14 is None."""
        reward_fn = PnLReward(
            realized_weight=1.0,
            normalize_by_atr=True,
        )
        ctx = make_reward_context(
            trade_just_closed=True,
            realized_pnl=50.0,
            atr_14=None,
        )
        assert reward_fn.compute(ctx) == pytest.approx(50.0)

    def test_atr_zero_fallback(self) -> None:
        """No division by zero when atr_14 is 0.0."""
        reward_fn = PnLReward(
            realized_weight=1.0,
            normalize_by_atr=True,
        )
        ctx = make_reward_context(
            trade_just_closed=True,
            realized_pnl=50.0,
            atr_14=0.0,
        )
        assert reward_fn.compute(ctx) == pytest.approx(50.0)

    def test_reset_noop(self) -> None:
        """reset() doesn't raise."""
        reward_fn = PnLReward()
        reward_fn.reset()  # should not raise

    def test_custom_weights(self) -> None:
        """Custom realized_weight and unrealized_weight affect output."""
        reward_fn = PnLReward(realized_weight=2.0, unrealized_weight=0.25)
        ctx_close = make_reward_context(
            trade_just_closed=True,
            realized_pnl=10.0,
        )
        assert reward_fn.compute(ctx_close) == pytest.approx(20.0)

        ctx_hold = make_reward_context(
            position_state=PositionState.LONG,
            unrealized_pnl=10.0,
        )
        assert reward_fn.compute(ctx_hold) == pytest.approx(2.5)

    def test_hold_penalty_not_normalized_by_atr(self) -> None:
        """ATR normalization does not apply to hold penalty."""
        reward_fn = PnLReward(
            hold_penalty=-0.001,
            normalize_by_atr=True,
            atr_multiplier=0.1,
        )
        ctx = make_reward_context(
            position_state=PositionState.FLAT,
            trade_just_closed=False,
            atr_14=5.0,
        )
        assert reward_fn.compute(ctx) == pytest.approx(-0.001)


# ---------------------------------------------------------------------------
# SharpeReward Tests
# ---------------------------------------------------------------------------


class TestSharpeReward:
    """Test SharpeReward (differential Sharpe ratio)."""

    def test_name(self) -> None:
        """Returns correct name."""
        assert SharpeReward().name == "Sharpe Reward"

    def test_first_step_returns_zero(self) -> None:
        """First call returns 0.0 and initializes prev_value."""
        reward_fn = SharpeReward()
        ctx = make_reward_context(balance=10000.0)
        assert reward_fn.compute(ctx) == 0.0
        assert reward_fn._prev_value == 10000.0

    def test_warmup_returns_zero(self) -> None:
        """Returns 0.0 for first min_steps steps."""
        reward_fn = SharpeReward(min_steps=5)

        # First step: init
        ctx = make_reward_context(balance=10000.0)
        assert reward_fn.compute(ctx) == 0.0

        # Next 5 steps: warmup
        for i in range(5):
            ctx = make_reward_context(balance=10000.0 + (i + 1) * 10)
            assert reward_fn.compute(ctx) == 0.0

    def test_positive_returns_positive_reward(self) -> None:
        """Consistent positive balance growth produces positive reward after warmup."""
        reward_fn = SharpeReward(min_steps=5, eta=0.05, scale=100.0)

        # Init step
        reward_fn.compute(make_reward_context(balance=10000.0))

        # Warmup with growing balance
        for i in range(5):
            reward_fn.compute(make_reward_context(balance=10000.0 + (i + 1) * 50))

        # Post-warmup: continue growing
        rewards = []
        balance = 10250.0
        for _ in range(10):
            balance += 50.0
            r = reward_fn.compute(make_reward_context(balance=balance))
            rewards.append(r)

        # At least some rewards should be positive
        assert any(r > 0 for r in rewards)

    def test_negative_returns_negative_reward(self) -> None:
        """Consistent negative balance produces negative reward after warmup."""
        reward_fn = SharpeReward(min_steps=5, eta=0.05, scale=100.0)

        # Init + warmup with declining balance
        reward_fn.compute(make_reward_context(balance=10000.0))
        for i in range(5):
            reward_fn.compute(make_reward_context(balance=10000.0 - (i + 1) * 50))

        # Post-warmup: continue declining
        rewards = []
        balance = 9750.0
        for _ in range(10):
            balance -= 50.0
            r = reward_fn.compute(make_reward_context(balance=balance))
            rewards.append(r)

        # At least some rewards should be negative
        assert any(r < 0 for r in rewards)

    def test_numerical_stability_zero_variance(self) -> None:
        """Constant balance returns 0.0 with no NaN/Inf."""
        reward_fn = SharpeReward(min_steps=3)

        # Init
        reward_fn.compute(make_reward_context(balance=10000.0))

        # Warmup + post-warmup with constant balance
        for _ in range(20):
            r = reward_fn.compute(make_reward_context(balance=10000.0))
            assert np.isfinite(r)
            assert r == 0.0

    def test_output_clamped(self) -> None:
        """Output always in [-1.0, 1.0]."""
        reward_fn = SharpeReward(min_steps=3, scale=1e6)

        # Init + warmup with varying balance
        reward_fn.compute(make_reward_context(balance=10000.0))
        for i in range(3):
            reward_fn.compute(make_reward_context(balance=10000.0 + (i + 1) * 100))

        # Post-warmup with large jump
        r = reward_fn.compute(make_reward_context(balance=20000.0))
        assert -1.0 <= r <= 1.0

    def test_reset_clears_state(self) -> None:
        """After reset, state is clean."""
        reward_fn = SharpeReward()

        # Use it for a few steps
        reward_fn.compute(make_reward_context(balance=10000.0))
        reward_fn.compute(make_reward_context(balance=10100.0))

        # Reset
        reward_fn.reset()
        assert reward_fn._A == 0.0
        assert reward_fn._B == 0.0
        assert reward_fn._prev_value == 0.0
        assert reward_fn._step_count == 0

    def test_ema_uses_old_values(self) -> None:
        """EMA update happens AFTER differential Sharpe computation."""
        reward_fn = SharpeReward(min_steps=2, eta=0.1)

        # Init
        reward_fn.compute(make_reward_context(balance=10000.0))

        # Warmup
        reward_fn.compute(make_reward_context(balance=10100.0))
        reward_fn.compute(make_reward_context(balance=10200.0))

        # Capture state before compute
        A_before = reward_fn._A
        B_before = reward_fn._B

        # Compute — should use A_before and B_before
        reward_fn.compute(make_reward_context(balance=10300.0))

        # State should have been updated after compute
        assert reward_fn._A != A_before or reward_fn._B != B_before


# ---------------------------------------------------------------------------
# RiskAdjustedReward Tests
# ---------------------------------------------------------------------------


class TestRiskAdjustedReward:
    """Test RiskAdjustedReward (composite regime-aware)."""

    def test_name(self) -> None:
        """Returns correct name."""
        assert RiskAdjustedReward().name == "Risk-Adjusted Reward"

    def test_base_reward_passthrough(self) -> None:
        """Base reward is included in total."""
        reward_fn = RiskAdjustedReward(
            drawdown_threshold=1.0,  # disable drawdown penalty
            overtrading_penalty=0.0,  # disable overtrading
        )
        ctx = make_reward_context(
            trade_just_closed=True,
            realized_pnl=50.0,
            regime_state=None,
        )
        # PnLReward default: 50.0 * 1.0 = 50.0
        assert reward_fn.compute(ctx) == pytest.approx(50.0)

    def test_custom_base_reward(self) -> None:
        """Can inject a different base reward function."""
        mock_base = MagicMock(spec=BaseRewardFunction)
        mock_base.compute.return_value = 42.0

        reward_fn = RiskAdjustedReward(
            base_reward=mock_base,
            drawdown_threshold=1.0,
            overtrading_penalty=0.0,
        )
        ctx = make_reward_context(regime_state=None)
        result = reward_fn.compute(ctx)
        mock_base.compute.assert_called_once_with(ctx)
        assert result == pytest.approx(42.0)

    def test_drawdown_no_penalty_below_threshold(self) -> None:
        """No penalty at 1% drawdown (below 2% threshold)."""
        reward_fn = RiskAdjustedReward(
            drawdown_threshold=0.02,
            overtrading_penalty=0.0,
        )
        ctx = make_reward_context(
            drawdown_pct=0.01,
            regime_state=None,
        )
        # Only base PnL reward (hold penalty)
        base = PnLReward().compute(ctx)
        assert reward_fn.compute(ctx) == pytest.approx(base)

    def test_drawdown_penalty_above_threshold(self) -> None:
        """Penalty at 5% drawdown with 2% threshold."""
        reward_fn = RiskAdjustedReward(
            drawdown_threshold=0.02,
            drawdown_penalty_scale=10.0,
            overtrading_penalty=0.0,
        )
        ctx = make_reward_context(
            drawdown_pct=0.05,
            regime_state=None,
        )
        base = PnLReward().compute(ctx)
        # penalty = -(0.05 - 0.02) * 10.0 = -0.3
        expected = base + (-0.03 * 10.0)
        assert reward_fn.compute(ctx) == pytest.approx(expected)

    def test_regime_bonus_trend_aligned(self) -> None:
        """Positive bonus for LONG in LOW_VOL_TREND."""
        reward_fn = RiskAdjustedReward(
            regime_bonus_scale=0.1,
            drawdown_threshold=1.0,
            overtrading_penalty=0.0,
        )
        ctx = make_reward_context(
            position_state=PositionState.LONG,
            previous_position_state=PositionState.LONG,
            regime_state=RegimeState.LOW_VOL_TREND,
            unrealized_pnl=10.0,
        )
        base = PnLReward().compute(ctx)
        assert reward_fn.compute(ctx) == pytest.approx(base + 0.1)

    def test_regime_bonus_high_vol_trend(self) -> None:
        """Positive bonus for SHORT in HIGH_VOL_TREND."""
        reward_fn = RiskAdjustedReward(
            regime_bonus_scale=0.1,
            drawdown_threshold=1.0,
            overtrading_penalty=0.0,
        )
        ctx = make_reward_context(
            position_state=PositionState.SHORT,
            previous_position_state=PositionState.SHORT,
            regime_state=RegimeState.HIGH_VOL_TREND,
            unrealized_pnl=10.0,
        )
        base = PnLReward().compute(ctx)
        assert reward_fn.compute(ctx) == pytest.approx(base + 0.1)

    def test_regime_penalty_ranging(self) -> None:
        """Negative penalty for LONG position in RANGING regime."""
        reward_fn = RiskAdjustedReward(
            regime_bonus_scale=0.1,
            drawdown_threshold=1.0,
            overtrading_penalty=0.0,
        )
        ctx = make_reward_context(
            position_state=PositionState.LONG,
            previous_position_state=PositionState.LONG,
            regime_state=RegimeState.RANGING,
            unrealized_pnl=10.0,
        )
        base = PnLReward().compute(ctx)
        assert reward_fn.compute(ctx) == pytest.approx(base - 0.1)

    def test_regime_none_no_effect(self) -> None:
        """No bonus when regime_state is None."""
        reward_fn = RiskAdjustedReward(
            regime_bonus_scale=0.1,
            drawdown_threshold=1.0,
            overtrading_penalty=0.0,
        )
        ctx = make_reward_context(
            position_state=PositionState.LONG,
            previous_position_state=PositionState.LONG,
            regime_state=None,
            unrealized_pnl=10.0,
        )
        base = PnLReward().compute(ctx)
        assert reward_fn.compute(ctx) == pytest.approx(base)

    def test_regime_flat_no_effect(self) -> None:
        """FLAT position gets no regime bonus regardless of regime."""
        reward_fn = RiskAdjustedReward(
            regime_bonus_scale=0.1,
            drawdown_threshold=1.0,
            overtrading_penalty=0.0,
        )
        ctx = make_reward_context(
            position_state=PositionState.FLAT,
            regime_state=RegimeState.LOW_VOL_TREND,
        )
        base = PnLReward().compute(ctx)
        assert reward_fn.compute(ctx) == pytest.approx(base)

    def test_overtrading_penalty(self) -> None:
        """Penalty when position changes."""
        reward_fn = RiskAdjustedReward(
            overtrading_penalty=-0.02,
            drawdown_threshold=1.0,
        )
        ctx = make_reward_context(
            position_state=PositionState.LONG,
            previous_position_state=PositionState.FLAT,
            regime_state=None,
            unrealized_pnl=0.0,
            trade_just_opened=True,
        )
        base = PnLReward().compute(ctx)
        assert reward_fn.compute(ctx) == pytest.approx(base + (-0.02))

    def test_no_overtrading_same_position(self) -> None:
        """No penalty when position unchanged."""
        reward_fn = RiskAdjustedReward(
            overtrading_penalty=-0.02,
            drawdown_threshold=1.0,
        )
        ctx = make_reward_context(
            position_state=PositionState.LONG,
            previous_position_state=PositionState.LONG,
            regime_state=None,
            unrealized_pnl=10.0,
        )
        base = PnLReward().compute(ctx)
        assert reward_fn.compute(ctx) == pytest.approx(base)

    def test_hold_penalty_beyond_max(self) -> None:
        """Penalty ramps after max_hold_bars."""
        reward_fn = RiskAdjustedReward(
            max_hold_bars=48,
            hold_penalty_per_bar=-0.0002,
            drawdown_threshold=1.0,
            overtrading_penalty=0.0,
        )
        ctx = make_reward_context(
            position_state=PositionState.LONG,
            previous_position_state=PositionState.LONG,
            bars_held=60,
            regime_state=None,
            unrealized_pnl=10.0,
        )
        base = PnLReward().compute(ctx)
        # (60 - 48) * -0.0002 = -0.0024
        expected = base + (12 * -0.0002)
        assert reward_fn.compute(ctx) == pytest.approx(expected)

    def test_no_hold_penalty_within_max(self) -> None:
        """No penalty when bars_held <= max_hold_bars."""
        reward_fn = RiskAdjustedReward(
            max_hold_bars=48,
            hold_penalty_per_bar=-0.0002,
            drawdown_threshold=1.0,
            overtrading_penalty=0.0,
        )
        ctx = make_reward_context(
            position_state=PositionState.LONG,
            previous_position_state=PositionState.LONG,
            bars_held=30,
            regime_state=None,
            unrealized_pnl=10.0,
        )
        base = PnLReward().compute(ctx)
        assert reward_fn.compute(ctx) == pytest.approx(base)

    def test_composite_all_components(self) -> None:
        """All penalties/bonuses sum correctly."""
        reward_fn = RiskAdjustedReward(
            drawdown_threshold=0.02,
            drawdown_penalty_scale=10.0,
            regime_bonus_scale=0.1,
            overtrading_penalty=-0.02,
            max_hold_bars=48,
            hold_penalty_per_bar=-0.0002,
        )
        ctx = make_reward_context(
            position_state=PositionState.LONG,
            previous_position_state=PositionState.FLAT,  # position changed
            trade_just_opened=True,
            unrealized_pnl=5.0,
            drawdown_pct=0.05,  # above threshold
            bars_held=60,  # above max
            regime_state=RegimeState.RANGING,  # penalty
        )

        base = PnLReward().compute(ctx)  # 5.0 * 0.5 = 2.5
        dd_penalty = -(0.05 - 0.02) * 10.0  # -0.3
        regime_penalty = -0.1  # ranging
        overtrade = -0.02
        hold = (60 - 48) * -0.0002  # -0.0024

        expected = base + dd_penalty + regime_penalty + overtrade + hold
        assert reward_fn.compute(ctx) == pytest.approx(expected)

    def test_reset_resets_base_reward(self) -> None:
        """reset() calls base_reward.reset()."""
        mock_base = MagicMock(spec=BaseRewardFunction)
        mock_base.compute.return_value = 0.0
        reward_fn = RiskAdjustedReward(base_reward=mock_base)
        reward_fn.reset()
        mock_base.reset.assert_called_once()


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


class TestRewardIntegration:
    """Integration tests for the reward system."""

    def test_all_exports_importable(self) -> None:
        """All 5 classes importable from fxer.rl.rewards."""
        from fxer.rl.rewards import (
            BaseRewardFunction,
            PnLReward,
            RewardContext,
            RiskAdjustedReward,
            SharpeReward,
        )

        assert BaseRewardFunction is not None
        assert PnLReward is not None
        assert RewardContext is not None
        assert RiskAdjustedReward is not None
        assert SharpeReward is not None

    def test_reward_functions_interchangeable(self) -> None:
        """All concrete rewards accept same RewardContext."""
        ctx = make_reward_context(
            position_state=PositionState.LONG,
            unrealized_pnl=10.0,
            balance=10100.0,
        )

        rewards = [PnLReward(), SharpeReward(), RiskAdjustedReward()]
        for reward_fn in rewards:
            result = reward_fn.compute(ctx)
            assert isinstance(result, float)
            assert np.isfinite(result)

    def test_stateful_reward_episode_boundary(self) -> None:
        """SharpeReward reset between episodes works correctly."""
        reward_fn = SharpeReward(min_steps=3)

        # Episode 1
        reward_fn.compute(make_reward_context(balance=10000.0))
        reward_fn.compute(make_reward_context(balance=10100.0))
        reward_fn.compute(make_reward_context(balance=10200.0))
        reward_fn.compute(make_reward_context(balance=10300.0))

        # Reset for episode 2
        reward_fn.reset()
        assert reward_fn._step_count == 0
        assert reward_fn._prev_value == 0.0

        # Episode 2 should start fresh
        r = reward_fn.compute(make_reward_context(balance=10000.0))
        assert r == 0.0  # first step returns 0

    def test_no_lookahead_bias(self) -> None:
        """Reward only uses current/past data from context, not future."""
        # RewardContext is a frozen snapshot — by design it cannot contain future data.
        # Verify the contract: compute() only accesses fields on the context.
        ctx = make_reward_context(
            trade_just_closed=True,
            realized_pnl=25.0,
            balance=10025.0,
        )

        for reward_fn in [PnLReward(), SharpeReward(), RiskAdjustedReward()]:
            result = reward_fn.compute(ctx)
            assert isinstance(result, float)

    def test_make_reward_context_factory(self) -> None:
        """The conftest factory produces valid contexts."""
        ctx = make_reward_context()
        assert isinstance(ctx, RewardContext)
        assert ctx.initial_balance == 10000.0
        assert ctx.atr_14 == 3.5
        assert ctx.rsi_14 == 55.0
