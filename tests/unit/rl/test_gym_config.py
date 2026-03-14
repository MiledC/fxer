"""Unit tests for the GymConfig module."""

from datetime import datetime

import pytest

from fxer.rl.gym.config import GymConfig
from fxer.rl.types import EpisodeMode


class TestGymConfig:
    """Test GymConfig behavior."""

    def test_default_initialization(self):
        """Verify default values are set correctly."""
        config = GymConfig()

        assert config.symbol == "XAUUSD"
        assert config.start_date is None
        assert config.end_date is None
        assert config.mode == EpisodeMode.TRAINING
        assert config.initial_balance == 10_000.0
        assert config.leverage == 100.0
        assert config.units_per_lot == 100.0
        assert config.commission_per_lot == 7.0
        assert config.episode_length == 2016
        assert config.lot_size == 0.01
        assert config.margin_call_pct == 50.0

    def test_custom_initialization(self):
        """Verify custom values are set correctly."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 12, 31)

        config = GymConfig(
            symbol="EURUSD",
            start_date=start,
            end_date=end,
            mode=EpisodeMode.EVALUATION,
            initial_balance=50_000.0,
            leverage=50.0,
            units_per_lot=1000.0,
            commission_per_lot=5.0,
            episode_length=1000,
            lot_size=0.1,
            margin_call_pct=25.0,
        )

        assert config.symbol == "EURUSD"
        assert config.start_date == start
        assert config.end_date == end
        assert config.mode == EpisodeMode.EVALUATION
        assert config.initial_balance == 50_000.0
        assert config.leverage == 50.0
        assert config.units_per_lot == 1000.0
        assert config.commission_per_lot == 5.0
        assert config.episode_length == 1000
        assert config.lot_size == 0.1
        assert config.margin_call_pct == 25.0

    def test_frozen_immutability(self):
        """Verify dataclass is frozen and immutable."""
        config = GymConfig()

        with pytest.raises(AttributeError, match="cannot assign to field"):
            config.initial_balance = 20_000.0

        with pytest.raises(AttributeError, match="cannot assign to field"):
            config.symbol = "GBPUSD"

    def test_validation_initial_balance(self):
        """Verify initial_balance validation."""
        # Zero balance
        with pytest.raises(ValueError, match="initial_balance must be positive"):
            GymConfig(initial_balance=0.0)

        # Negative balance
        with pytest.raises(ValueError, match="initial_balance must be positive"):
            GymConfig(initial_balance=-1000.0)

    def test_validation_leverage(self):
        """Verify leverage validation."""
        # Zero leverage
        with pytest.raises(ValueError, match="leverage must be positive"):
            GymConfig(leverage=0.0)

        # Negative leverage
        with pytest.raises(ValueError, match="leverage must be positive"):
            GymConfig(leverage=-10.0)

    def test_validation_units_per_lot(self):
        """Verify units_per_lot validation."""
        # Zero units
        with pytest.raises(ValueError, match="units_per_lot must be positive"):
            GymConfig(units_per_lot=0.0)

        # Negative units
        with pytest.raises(ValueError, match="units_per_lot must be positive"):
            GymConfig(units_per_lot=-100.0)

    def test_validation_commission_per_lot(self):
        """Verify commission_per_lot validation."""
        # Zero commission is valid
        config = GymConfig(commission_per_lot=0.0)
        assert config.commission_per_lot == 0.0

        # Negative commission
        with pytest.raises(ValueError, match="commission_per_lot cannot be negative"):
            GymConfig(commission_per_lot=-5.0)

    def test_validation_episode_length(self):
        """Verify episode_length validation."""
        # Zero length
        with pytest.raises(ValueError, match="episode_length must be positive"):
            GymConfig(episode_length=0)

        # Negative length
        with pytest.raises(ValueError, match="episode_length must be positive"):
            GymConfig(episode_length=-100)

    def test_validation_lot_size(self):
        """Verify lot_size validation."""
        # Zero lot size
        with pytest.raises(ValueError, match="lot_size must be positive"):
            GymConfig(lot_size=0.0)

        # Negative lot size
        with pytest.raises(ValueError, match="lot_size must be positive"):
            GymConfig(lot_size=-0.01)

    def test_validation_margin_call_pct(self):
        """Verify margin_call_pct validation."""
        # Zero percent
        with pytest.raises(ValueError, match="margin_call_pct must be between 0 and 100"):
            GymConfig(margin_call_pct=0.0)

        # Over 100 percent
        with pytest.raises(ValueError, match="margin_call_pct must be between 0 and 100"):
            GymConfig(margin_call_pct=101.0)

        # Negative percent
        with pytest.raises(ValueError, match="margin_call_pct must be between 0 and 100"):
            GymConfig(margin_call_pct=-10.0)

        # Valid boundaries
        config = GymConfig(margin_call_pct=0.1)
        assert config.margin_call_pct == 0.1

        config = GymConfig(margin_call_pct=100.0)
        assert config.margin_call_pct == 100.0

    def test_margin_per_lot_method(self):
        """Verify margin_per_lot calculation with dynamic price."""
        config = GymConfig(
            units_per_lot=100.0,
            leverage=100.0,
        )

        # At $2000/oz: margin = (100 * 2000) / 100 = $2000
        assert config.margin_per_lot(2000.0) == 2000.0

        # At $2500/oz: margin = (100 * 2500) / 100 = $2500
        assert config.margin_per_lot(2500.0) == 2500.0

    def test_different_leverage_affects_margin(self):
        """Verify leverage changes affect margin calculations."""
        config_low = GymConfig(leverage=50.0)
        config_high = GymConfig(leverage=200.0)

        price = 2000.0
        # Lower leverage = higher margin requirement
        assert config_low.margin_per_lot(price) > config_high.margin_per_lot(price)

    def test_equality_and_hashing(self):
        """Verify equality and hashing work correctly for frozen dataclass."""
        config1 = GymConfig(initial_balance=10_000.0)
        config2 = GymConfig(initial_balance=10_000.0)
        config3 = GymConfig(initial_balance=20_000.0)

        # Equal configs
        assert config1 == config2
        assert hash(config1) == hash(config2)

        # Different configs
        assert config1 != config3
        assert hash(config1) != hash(config3)

        # Can be used as dict key
        config_dict = {config1: "value1"}
        assert config_dict[config2] == "value1"