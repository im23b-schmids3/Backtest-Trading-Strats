"""Provider-independent Phase D futures and prop-account research."""

from .adapters import SyntheticTradeSignalAdapter, TradeSignalAdapter

__all__ = ["SyntheticTradeSignalAdapter", "TradeSignalAdapter"]

from .models import PropClassification, PropPhase
from .services import PropResearchService

__all__ = ["PropClassification", "PropPhase", "PropResearchService"]
