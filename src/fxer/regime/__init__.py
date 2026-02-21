"""Regime classification module for the fxEr trading system."""

from fxer.regime.change_point import ChangePointDetector
from fxer.regime.classifier import RegimeClassifier
from fxer.regime.hmm import HMMRegimeClassifier
from fxer.regime.intraday_filter import IntradayRegimeFilter
from fxer.regime.session_rules import SessionRegimeRules
from fxer.regime.types import RegimeDecision, RegimeEvent, RegimeState

__all__ = [
    "ChangePointDetector",
    "HMMRegimeClassifier",
    "IntradayRegimeFilter",
    "RegimeClassifier",
    "RegimeDecision",
    "RegimeEvent",
    "RegimeState",
    "SessionRegimeRules",
]