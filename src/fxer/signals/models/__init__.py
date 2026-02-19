"""Signal model implementations."""

# Lazy imports to avoid circular dependency with training subpackage.
# Use direct imports in code:
#   from fxer.signals.models.xgboost_model import XGBoostSignalModel

__all__ = [
    "XGBoostSignalModel",
    "CNNLSTMSignalModel",
    "StackingEnsemble",
]


def __getattr__(name: str):
    if name == "XGBoostSignalModel":
        from fxer.signals.models.xgboost_model import XGBoostSignalModel
        return XGBoostSignalModel
    if name == "CNNLSTMSignalModel":
        from fxer.signals.models.cnn_lstm import CNNLSTMSignalModel
        return CNNLSTMSignalModel
    if name == "StackingEnsemble":
        from fxer.signals.models.ensemble import StackingEnsemble
        return StackingEnsemble
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
