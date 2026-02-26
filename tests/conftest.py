"""Shared pytest fixtures for fxEr tests."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock

import numpy as np
import pytest

from fxer.config.settings import Settings
from fxer.core.events import FeatureVector, NormalizedBar
from fxer.core.types import RawBar, Timeframe
from fxer.regime.types import RegimeState
from fxer.rl.rewards.context import RewardContext
from fxer.rl.types import GymAction, PositionState

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
SAMPLE_BARS_CSV = FIXTURES_DIR / "sample_bars.csv"


# ---------------------------------------------------------------------------
# Sample data factories
# ---------------------------------------------------------------------------


def make_raw_bar(
    timestamp: datetime | None = None,
    open_: str = "2062.50",
    high: str = "2063.20",
    low: str = "2061.80",
    close: str = "2062.90",
    volume: str = "1250",
    symbol: str = "XAUUSD",
    timeframe: str | None = None,
) -> RawBar:
    """Create a RawBar with sensible defaults."""
    return RawBar(
        timestamp=timestamp or datetime(2024, 1, 2, 0, 0, 0),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        symbol=symbol,
        timeframe=timeframe,
    )


def make_normalized_bar(
    timestamp: datetime | None = None,
    open_: str = "2062.50",
    high: str = "2063.20",
    low: str = "2061.80",
    close: str = "2062.90",
    volume: str = "1250",
    symbol: str = "XAUUSD",
    timeframe: Timeframe = Timeframe.M5,
    is_complete: bool = True,
) -> NormalizedBar:
    """Create a NormalizedBar with sensible defaults."""
    return NormalizedBar(
        symbol=symbol,
        timeframe=timeframe,
        timestamp=timestamp or datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc),
        open=Decimal(open_),
        high=Decimal(high),
        low=Decimal(low),
        close=Decimal(close),
        volume=Decimal(volume),
        is_complete=is_complete,
    )


def make_normalized_bar_series(
    count: int = 60,
    base_price: float = 2062.50,
    step_minutes: int = 5,
    symbol: str = "XAUUSD",
    timeframe: Timeframe = Timeframe.M5,
) -> list[NormalizedBar]:
    """Create a series of NormalizedBars with gradually increasing prices.

    Useful for testing feature computation which requires many bars.
    """
    import random

    random.seed(42)
    bars = []
    price = base_price

    start = datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc)

    for i in range(count):
        ts = start.replace(
            hour=(i * step_minutes) // 60,
            minute=(i * step_minutes) % 60,
        )
        change = random.uniform(-1.5, 1.5)
        open_p = round(price, 2)
        close_p = round(price + change, 2)
        high_p = round(max(open_p, close_p) + random.uniform(0.1, 1.0), 2)
        low_p = round(min(open_p, close_p) - random.uniform(0.1, 1.0), 2)
        vol = random.randint(800, 2000)

        bars.append(NormalizedBar(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=ts,
            open=Decimal(str(open_p)),
            high=Decimal(str(high_p)),
            low=Decimal(str(low_p)),
            close=Decimal(str(close_p)),
            volume=Decimal(str(vol)),
            is_complete=True,
        ))
        price = close_p

    return bars


def make_observation_config(
    symbol: str = "XAUUSD",
    primary_timeframe: Timeframe = Timeframe.M5,
    include_regime: bool = True,
) -> "ObservationConfig":
    """Create a default ObservationConfig for testing."""
    from fxer.rl.observations.config import ObservationConfig
    return ObservationConfig(symbol=symbol, primary_timeframe=primary_timeframe, include_regime=include_regime)


def make_reward_context(
    # Gym state
    action_taken: GymAction = GymAction.FLAT,
    position_state: PositionState = PositionState.FLAT,
    previous_position_state: PositionState = PositionState.FLAT,
    realized_pnl: float = 0.0,
    unrealized_pnl: float = 0.0,
    unrealized_pnl_pct: float = 0.0,
    balance: float = 10000.0,
    initial_balance: float = 10000.0,
    drawdown_pct: float = 0.0,
    bars_held: int = 0,
    trade_just_opened: bool = False,
    trade_just_closed: bool = False,
    step_number: int = 0,
    total_trades: int = 0,
    # Market state
    current_price: float = 2062.50,
    atr_14: float | None = 3.5,
    rsi_14: float | None = 55.0,
    regime_state: RegimeState | None = RegimeState.RANGING,
    regime_confidence: float | None = 0.75,
    observation: np.ndarray | None = None,
) -> RewardContext:
    """Create a RewardContext with sensible defaults for testing."""
    return RewardContext(
        action_taken=action_taken,
        position_state=position_state,
        previous_position_state=previous_position_state,
        realized_pnl=realized_pnl,
        unrealized_pnl=unrealized_pnl,
        unrealized_pnl_pct=unrealized_pnl_pct,
        balance=balance,
        initial_balance=initial_balance,
        drawdown_pct=drawdown_pct,
        bars_held=bars_held,
        trade_just_opened=trade_just_opened,
        trade_just_closed=trade_just_closed,
        step_number=step_number,
        total_trades=total_trades,
        current_price=current_price,
        atr_14=atr_14,
        rsi_14=rsi_14,
        regime_state=regime_state,
        regime_confidence=regime_confidence,
        observation=observation,
    )


def make_gym_config(
    symbol: str = "XAUUSD",
    mode: "EpisodeMode | None" = None,
    initial_balance: float = 10_000.0,
    leverage: float = 100.0,
    units_per_lot: float = 100.0,
    commission_per_lot: float = 7.0,
    episode_length: int = 2016,
    lot_size: float = 0.01,
    margin_call_pct: float = 50.0,
    **kwargs,
) -> "GymConfig":
    """Create a GymConfig with sensible defaults for testing."""
    from fxer.rl.gym.config import GymConfig
    from fxer.rl.types import EpisodeMode

    if mode is None:
        mode = EpisodeMode.TRAINING

    return GymConfig(
        symbol=symbol,
        mode=mode,
        initial_balance=initial_balance,
        leverage=leverage,
        units_per_lot=units_per_lot,
        commission_per_lot=commission_per_lot,
        episode_length=episode_length,
        lot_size=lot_size,
        margin_call_pct=margin_call_pct,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_bars_csv_path() -> Path:
    """Path to the sample_bars.csv fixture file."""
    return SAMPLE_BARS_CSV


@pytest.fixture
def raw_bar() -> RawBar:
    """A single RawBar fixture."""
    return make_raw_bar()


@pytest.fixture
def normalized_bar() -> NormalizedBar:
    """A single NormalizedBar fixture."""
    return make_normalized_bar()


@pytest.fixture
def sample_bar_series() -> list[NormalizedBar]:
    """A list of 60 NormalizedBars for feature computation testing."""
    return make_normalized_bar_series(count=60)


@pytest.fixture
def sample_feature_vector() -> FeatureVector:
    """A sample FeatureVector fixture."""
    return FeatureVector(
        symbol="XAUUSD",
        timestamp=datetime(2024, 1, 2, 5, 0, 0, tzinfo=timezone.utc),
        timeframe=Timeframe.M5,
        rsi_14=55.3,
        rsi_7=52.1,
        macd_line=0.45,
        macd_signal=0.38,
        macd_histogram=0.07,
        bb_upper=2070.0,
        bb_middle=2065.0,
        bb_lower=2060.0,
        bb_width=0.00484,
        atr_14=3.5,
        is_london_session=False,
        is_ny_session=False,
        is_overlap_session=False,
        is_asian_session=True,
        hour_of_day=5,
        day_of_week=1,
        is_month_turn=True,
        warmup_complete=True,
        # Price-derived features
        return_1bar=0.0012,
        return_5bar=0.0058,
        return_12bar=0.0142,
        rolling_volatility_20=0.0085,
        momentum_48=0.0325,
    )


@pytest.fixture
def settings() -> Settings:
    """Default application settings."""
    return Settings()


# ---------------------------------------------------------------------------
# Mock storage fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_questdb_client() -> MagicMock:
    """A mocked QuestDBClient for tests that should not hit real storage."""
    mock = MagicMock(spec=[
        "init_tables",
        "insert_bar",
        "insert_bars",
        "insert_features",
        "query_bars",
        "query_latest_bar",
        "close",
        "__enter__",
        "__exit__",
    ])
    mock.__enter__ = MagicMock(return_value=mock)
    mock.__exit__ = MagicMock(return_value=False)
    mock.insert_bars.return_value = None
    mock.insert_bar.return_value = None
    mock.insert_features.return_value = None
    mock.query_bars.return_value = []
    mock.query_latest_bar.return_value = None
    return mock


@pytest.fixture
def mock_redis_client() -> MagicMock:
    """A mocked RedisClient for tests that should not hit real storage."""
    mock = MagicMock(spec=[
        "set_latest_bar",
        "get_latest_bar",
        "set_latest_features",
        "get_latest_features",
        "delete_bar",
        "delete_features",
        "close",
        "__enter__",
        "__exit__",
    ])
    mock.__enter__ = MagicMock(return_value=mock)
    mock.__exit__ = MagicMock(return_value=False)
    mock.get_latest_bar.return_value = None
    mock.get_latest_features.return_value = None
    return mock
