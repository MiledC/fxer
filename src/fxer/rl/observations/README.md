# RL Observation Module

Pre-computes multi-timeframe observation vectors for RL environments. Each observation is a flat `np.ndarray` combining features from multiple timeframes and optional regime state, stored for O(1) lookup during episode stepping.

## Quick Start

```python
from datetime import datetime, timezone
from fxer.rl.observations import ObservationBuilder, ObservationConfig

# Default config: M5(20) + H1(10) + H4(5) features + regime = 844 floats
config = ObservationConfig()
builder = ObservationBuilder(config, db_client=questdb_client)

builder.build(
    start=datetime(2024, 6, 1, tzinfo=timezone.utc),
    end=datetime(2024, 6, 30, tzinfo=timezone.utc),
)

# O(1) lookup during episode stepping
obs = builder.get_observation(timestamp)   # np.ndarray (844,) or None
bar = builder.get_bar(timestamp)           # NormalizedBar or None
regime = builder.get_regime(timestamp)     # RegimeDecision or None
features = builder.get_features(timestamp) # FeatureVector or None
```

## Configuration

### ObservationConfig

| Parameter            | Type                              | Default     | Description                                      |
|----------------------|-----------------------------------|-------------|--------------------------------------------------|
| `symbol`             | `str`                             | `"XAUUSD"`  | Trading symbol                                   |
| `primary_timeframe`  | `Timeframe`                       | `M5`        | Stepping timeframe (determines observation timestamps) |
| `windows`            | `tuple[TimeframeWindowConfig,...]`| See below   | Multi-TF window definitions                      |
| `include_regime`     | `bool`                            | `True`      | Append regime state to each observation          |
| `regime_features`    | `int`                             | `4`         | Regime vector size (3 one-hot + 1 confidence)    |

### TimeframeWindowConfig

Each window defines one timeframe slot in the observation vector.

| Parameter          | Type        | Default | Description                                |
|--------------------|-------------|---------|--------------------------------------------|
| `timeframe`        | `Timeframe` | -       | M5, M15, H1, H4                           |
| `lookback`         | `int`       | -       | Number of bars in the window               |
| `include_ohlcv`    | `bool`      | `False` | Include raw OHLCV (5 floats/bar)           |
| `include_features` | `bool`      | `True`  | Include computed features (24 floats/bar)  |

**Floats per bar** depends on what you include:

| include_ohlcv | include_features | Floats/bar |
|:---:|:---:|:---:|
| False | True  | 24 (features only) |
| True  | True  | 29 (OHLCV + features) |
| True  | False | 5 (OHLCV only) |
| False | False | 0 (no data) |

### Default Windows

```python
windows = (
    TimeframeWindowConfig(Timeframe.M5, lookback=20, include_ohlcv=False, include_features=True),  # 480 floats
    TimeframeWindowConfig(Timeframe.H1, lookback=10, include_ohlcv=False, include_features=True),  # 240 floats
    TimeframeWindowConfig(Timeframe.H4, lookback=5,  include_ohlcv=False, include_features=True),  # 120 floats
)
# + 4 regime floats = 844 total
```

## Observation Vector Layout

The observation is a flat 1-D array with sections concatenated in window order:

```
[ window_0 features | window_1 features | ... | regime ]
```

Each window section contains `lookback` bars laid out oldest-to-newest, with each bar's data concatenated:

```
[ bar_t-19 | bar_t-18 | ... | bar_t-1 | bar_t ]
```

Per bar (if both OHLCV and features enabled): `[open, high, low, close, volume, feat_0, feat_1, ..., feat_23]`

Per bar (features only, default): `[feat_0, feat_1, ..., feat_23]` (24 values in FEATURE_COLUMNS order)

### Default Layout (844 floats)

| Section | Bars | Floats/bar | Total |
|---------|------|------------|-------|
| M5 features (lookback=20) | 20 | 24 | 480 |
| H1 features (lookback=10) | 10 | 24 | 240 |
| H4 features (lookback=5)  | 5  | 24 | 120 |
| Regime (one-hot + confidence) | - | - | 4 |
| **Total** | | | **844** |

### Regime Encoding (4 floats)

```
[low_vol_trend, high_vol_trend, ranging, confidence]
```

One-hot encoding of `RegimeState` + confidence float (0.0-1.0). If no regime classifier is provided, all zeros.

### Feature Order (FEATURE_COLUMNS)

Each bar's 24 features follow this order:

| Index | Feature | Index | Feature |
|:---:|---|:---:|---|
| 0 | rsi_14 | 12 | momentum_48 |
| 1 | rsi_7 | 13 | is_london_session |
| 2 | macd_line | 14 | is_ny_session |
| 3 | macd_signal | 15 | is_overlap_session |
| 4 | macd_histogram | 16 | is_asian_session |
| 5 | bb_percent_b | 17 | hour_of_day |
| 6 | bb_width | 18 | day_of_week |
| 7 | atr_14 | 19 | is_month_turn |
| 8 | return_1bar | 20 | dxy_return_1h |
| 9 | return_5bar | 21 | dxy_rsi_14 |
| 10 | return_12bar | 22 | vix_level |
| 11 | rolling_volatility_20 | 23 | vix_change |

## Configuration Examples

### Wider H4 window

```python
config = ObservationConfig(
    windows=(
        TimeframeWindowConfig(Timeframe.M5, lookback=20),
        TimeframeWindowConfig(Timeframe.H1, lookback=10),
        TimeframeWindowConfig(Timeframe.H4, lookback=10),  # doubled from 5
    ),
)
# observation_size = 20*24 + 10*24 + 10*24 + 4 = 964
```

### Include OHLCV on H4 only

```python
config = ObservationConfig(
    windows=(
        TimeframeWindowConfig(Timeframe.M5, lookback=20, include_ohlcv=False, include_features=True),
        TimeframeWindowConfig(Timeframe.H1, lookback=10, include_ohlcv=False, include_features=True),
        TimeframeWindowConfig(Timeframe.H4, lookback=5, include_ohlcv=True, include_features=True),
    ),
)
# observation_size = 20*24 + 10*24 + 5*29 + 4 = 869
```

### No regime, M5 only

```python
config = ObservationConfig(
    windows=(
        TimeframeWindowConfig(Timeframe.M5, lookback=30),
    ),
    include_regime=False,
)
# observation_size = 30*24 = 720
```

### Single timeframe with OHLCV only (no computed features)

```python
config = ObservationConfig(
    windows=(
        TimeframeWindowConfig(Timeframe.M5, lookback=50, include_ohlcv=True, include_features=False),
    ),
    include_regime=False,
)
# observation_size = 50*5 = 250
```

## Using Without QuestDB (Testing)

Pass bars directly as a dict keyed by `Timeframe`:

```python
from fxer.core.types import Timeframe

builder = ObservationBuilder(config)
builder.build(
    start=start_dt,
    end=end_dt,
    bars={
        Timeframe.M5: m5_bars_list,
        Timeframe.H1: h1_bars_list,
        Timeframe.H4: h4_bars_list,
    },
)
```

The bars lists must include warmup bars before `start`. Each timeframe needs `lookback + 49` (MAX_WARMUP_BARS) extra bars before `start` for indicator warmup.

## Timestamp Alignment (Lookahead Prevention)

Higher-timeframe bars are only used after they complete:

- An **H1 bar** timestamped at 14:00 contains data from 14:00-14:59. It completes at **15:00**.
- An **H4 bar** timestamped at 12:00 contains data from 12:00-15:59. It completes at **16:00**.

For an M5 observation at timestamp T, the builder uses the most recent higher-TF bar where:

```
bar.timestamp + timedelta(minutes=bar.timeframe.minutes) <= T
```

This guarantees no future data leaks into any observation.

## Memory Usage

Each observation stores 844 float64 values = ~6.7 KB per timestamp.

| Date Range | M5 Timestamps | Memory |
|---|---|---|
| 1 day | ~200 | ~1.3 MB |
| 1 week | ~1,000 | ~6.7 MB |
| 1 month | ~4,000 | ~27 MB |
| 1 year | ~50,000 | ~335 MB |

For long backtests, consider building in chunks.

## API Reference

### ObservationBuilder

| Method / Property | Returns | Description |
|---|---|---|
| `build(start, end, bars=None)` | `None` | Pre-compute observations for date range |
| `get_observation(timestamp)` | `np.ndarray \| None` | O(1) observation lookup |
| `get_bar(timestamp)` | `NormalizedBar \| None` | Primary-TF bar at timestamp |
| `get_regime(timestamp)` | `RegimeDecision \| None` | Regime classification at timestamp |
| `get_features(timestamp)` | `FeatureVector \| None` | Primary-TF features at timestamp |
| `observation_size` | `int` | Total floats per observation |
| `timestamps` | `list[datetime]` | All available timestamps (sorted) |
| `is_built` | `bool` | Whether `build()` has been called |
