"""Provider-independent Phase D futures and prop-account research."""

from .adapters import AlertComplianceAdapter, SyntheticTradeSignalAdapter, TradeSignalAdapter

__all__ = ["AlertComplianceAdapter", "SyntheticTradeSignalAdapter", "TradeSignalAdapter"]

from .models import PropClassification, PropPhase
from .services import PropResearchService

__all__ = ["PropClassification", "PropPhase", "PropResearchService"]
