"""Application settings using Pydantic."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="FXER_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # QuestDB settings
    questdb_host: str = "localhost"
    questdb_ilp_port: int = 9009
    questdb_pg_port: int = 8812
    questdb_pg_user: str = "admin"
    questdb_pg_password: str = "quest"

    # Redis settings
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0

    # ZeroMQ settings
    zmq_pub_port: int = 5555
    zmq_sub_port: int = 5556

    # Data settings
    data_dir: str = "./data"
    default_symbol: str = "XAUUSD"

    # Feature computation settings
    feature_warmup_bars: int = 50  # Bars needed before valid features

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"  # "json" or "console"

    # IBKR settings
    ibkr_host: str = "127.0.0.1"
    ibkr_port: int = 4001  # 4001/4002 for Gateway, 7496/7497 for TWS
    ibkr_client_id: int = 1
    ibkr_timeout: int = 30
    ibkr_readonly: bool = True  # Data only, no trading

    # Symbols to ingest (configurable via env FXER_IBKR_SYMBOLS="XAUUSD,DXY,VIX")
    ibkr_symbols: list[str] = ["XAUUSD", "DXY", "VIX"]

    # Signal generation settings
    signal_model_dir: str = "./models"
    signal_neutral_threshold: float = 0.55
    signal_high_confidence_threshold: float = 0.70
    signal_lookback_window: int = 48  # Bars for LSTM sequence input
    signal_horizon_bars: int = 12  # Look-ahead bars for labelling (12 × 5m = 1h)
    signal_threshold_atr_mult: float = 0.5  # Min move = 0.5 × ATR for label

    # Regime classification settings
    regime_hmm_states: int = 3
    regime_hmm_covariance_type: str = "diag"
    regime_adx_trend_threshold: float = 25.0
    regime_adx_range_threshold: float = 20.0
    regime_atr_expansion_threshold: float = 1.5
    regime_confidence_minimum: float = 0.60
    regime_persistence_bars: int = 3
    regime_model_dir: str = "./models"
    regime_pelt_window_size: int = 100
    regime_pelt_min_size: int = 10


# Global settings instance
settings = Settings()
