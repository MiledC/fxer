"""Training utilities for signal models."""

# Lazy imports to avoid circular dependency with signals.models.ensemble
# which imports from training.data_prep. Use direct imports in code:
#   from fxer.signals.training.data_prep import LabelGenerator
#   from fxer.signals.training.trainer import Trainer

__all__ = [
    "LabelGenerator",
    "DatasetBuilder",
    "FeatureScaler",
    "WalkForwardSplitter",
    "CPCVSplitter",
    "SignalMetrics",
    "compute_metrics",
    "Trainer",
    "TrainResult",
]


def __getattr__(name: str):
    if name in ("LabelGenerator", "DatasetBuilder", "FeatureScaler"):
        from fxer.signals.training.data_prep import (
            DatasetBuilder,
            FeatureScaler,
            LabelGenerator,
        )
        return locals()[name]
    if name in ("WalkForwardSplitter", "CPCVSplitter"):
        from fxer.signals.training.validation import CPCVSplitter, WalkForwardSplitter
        return locals()[name]
    if name in ("SignalMetrics", "compute_metrics"):
        from fxer.signals.training.metrics import SignalMetrics, compute_metrics
        return locals()[name]
    if name in ("Trainer", "TrainResult"):
        from fxer.signals.training.trainer import TrainResult, Trainer
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
