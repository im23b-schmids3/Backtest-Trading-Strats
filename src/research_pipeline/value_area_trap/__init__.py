"""Deterministic BTCUSDT aggregate-trade research components.

This package is deliberately research-only.  It has no authenticated exchange
client, no broker integration, and no live-order capability.
"""

from .data import AggregateTrade, AggregateTradeImporter, AggregateTradeManifest
from .profile import FiveMinuteBar, SessionProfile, build_five_minute_bars, build_session_profiles
from .strategy import ValueAreaTrapConfig, ValueAreaTrapResult, run_value_area_trap
from .alpha_zero import AlphaZeroScenarioResult, alpha_zero_25k_policy, import_usd_calendar, run_alpha_zero_scenario
from .reports import build_value_area_reports, fixed_ablation_diagnostics

__all__ = [
    "AggregateTrade", "AggregateTradeImporter", "AggregateTradeManifest",
    "FiveMinuteBar", "SessionProfile", "build_five_minute_bars", "build_session_profiles",
    "ValueAreaTrapConfig", "ValueAreaTrapResult", "run_value_area_trap",
    "AlphaZeroScenarioResult", "alpha_zero_25k_policy", "import_usd_calendar", "run_alpha_zero_scenario",
    "build_value_area_reports", "fixed_ablation_diagnostics",
]
