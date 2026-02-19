"""Custom exceptions for the fxEr trading system."""

from typing import Any


class FxerError(Exception):
    """Base exception for all fxEr errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class DataValidationError(FxerError):
    """Raised when data fails validation rules."""

    def __init__(
        self,
        message: str,
        field: str | None = None,
        value: Any = None,
        rule: str | None = None,
    ):
        details = {}
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = value
        if rule:
            details["rule"] = rule
        super().__init__(message, details)
        self.field = field
        self.value = value
        self.rule = rule


class StorageError(FxerError):
    """Raised when a storage operation fails."""

    def __init__(
        self,
        message: str,
        operation: str | None = None,
        table: str | None = None,
    ):
        details = {}
        if operation:
            details["operation"] = operation
        if table:
            details["table"] = table
        super().__init__(message, details)
        self.operation = operation
        self.table = table


class ConnectionError(FxerError):
    """Raised when a connection to an external service fails."""

    def __init__(
        self,
        message: str,
        service: str | None = None,
        host: str | None = None,
        port: int | None = None,
    ):
        details = {}
        if service:
            details["service"] = service
        if host:
            details["host"] = host
        if port:
            details["port"] = port
        super().__init__(message, details)
        self.service = service
        self.host = host
        self.port = port


class ConfigurationError(FxerError):
    """Raised when configuration is invalid or missing."""

    def __init__(
        self,
        message: str,
        config_key: str | None = None,
    ):
        details = {}
        if config_key:
            details["config_key"] = config_key
        super().__init__(message, details)
        self.config_key = config_key


class FeatureComputationError(FxerError):
    """Raised when feature computation fails."""

    def __init__(
        self,
        message: str,
        feature_name: str | None = None,
        indicator: str | None = None,
    ):
        details = {}
        if feature_name:
            details["feature_name"] = feature_name
        if indicator:
            details["indicator"] = indicator
        super().__init__(message, details)
        self.feature_name = feature_name
        self.indicator = indicator


class SignalGenerationError(FxerError):
    """Raised when signal generation fails."""

    def __init__(
        self,
        message: str,
        model: str | None = None,
        symbol: str | None = None,
    ):
        details = {}
        if model:
            details["model"] = model
        if symbol:
            details["symbol"] = symbol
        super().__init__(message, details)
        self.model = model
        self.symbol = symbol


class ModelNotFittedError(FxerError):
    """Raised when prediction is attempted on an unfitted model."""

    def __init__(self, message: str, model_name: str | None = None):
        details = {}
        if model_name:
            details["model_name"] = model_name
        super().__init__(message, details)
        self.model_name = model_name


class InsufficientDataError(FxerError):
    """Raised when there is not enough data for an operation."""

    def __init__(
        self,
        message: str,
        required: int | None = None,
        available: int | None = None,
    ):
        details = {}
        if required is not None:
            details["required"] = required
        if available is not None:
            details["available"] = available
        super().__init__(message, details)
        self.required = required
        self.available = available
