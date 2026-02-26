"""Shared RL enums for the fxEr trading system."""

from enum import IntEnum


class PositionState(IntEnum):
    """Current position state in the market."""

    FLAT = 0
    LONG = 1
    SHORT = 2


class GymAction(IntEnum):
    """Available actions in the Gym environment."""

    FLAT = 0
    BUY = 1
    SELL = 2


class EpisodeMode(IntEnum):
    """Episode mode for training vs evaluation."""

    TRAINING = 0
    EVALUATION = 1