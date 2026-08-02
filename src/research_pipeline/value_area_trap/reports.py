from __future__ import annotations

from decimal import Decimal
from statistics import median
from typing import Any

from ..compliance import CalendarArtifact
from .alpha_zero import rolling_alpha_zero_scenarios, run_alpha_zero_scenario
from .profile import FiveMinuteBar, SessionProfile
from .strategy import ValueAreaTrapConfig, ValueAreaTrapResult, run_value_area_trap


def raw_strategy_metrics(result: ValueAreaTrapResult) -> dict[str, Any]:
    trades = result.trades; net = [Decimal(item["net_pnl"]) for item in trades]
    gains = sum((item for item in net if item > 0), Decimal()); losses = -sum((item for item in net if item < 0), Decimal())
    holding = [((__import__("datetime").datetime.fromisoformat(item["exit_timestamp"]) - __import__("datetime").datetime.fromisoformat(item["entry_timestamp"])).total_seconds() / 60) for item in trades]
    risk = [abs(Decimal(item["entry_price"]) - Decimal(item["initial_stop_price"])) * Decimal(item["quantity"]) for item in trades]
    rs = [net[index] / risk[index] for index in range(len(trades)) if risk[index] > 0]
    months = {item["entry_timestamp"][:7] for item in trades}
    return {"proposed_setups": result.proposed_setups, "significant_stop_runs": result.significant_stop_runs, "confirmed_divergences": result.confirmed_divergences, "return_triggers": result.return_triggers, "compliance_blocks": len(result.compliance_blocks), "executed_trades": len(trades), "long_trades": sum(item["direction"] == "LONG" for item in trades), "short_trades": sum(item["direction"] == "SHORT" for item in trades), "wins": sum(item > 0 for item in net), "losses": sum(item < 0 for item in net), "gross_pnl": str(result.gross_pnl), "fees": str(result.fees), "slippage": str(result.slippage_cost), "net_pnl": str(result.net_pnl), "expectancy": str(sum(net, Decimal()) / len(net)) if net else "0", "profit_factor": str(gains / losses) if losses else None, "average_r": str(sum(rs, Decimal()) / len(rs)) if rs else None, "median_r": str(median(rs)) if rs else None, "average_holding_minutes": sum(holding) / len(holding) if holding else None, "median_holding_minutes": median(holding) if holding else None, "forced_flat_count": result.forced_flat_count, "same_bar_ambiguity_count": result.same_bar_ambiguity_count, "trades_per_month": len(trades) / len(months) if months else 0, "zero_trade_months": 0, "policy_hash": result.policy_hash, "cost_model_hash": result.cost_model_hash}


def fixed_ablation_diagnostics(bars: list[FiveMinuteBar], profiles: dict, config: ValueAreaTrapConfig) -> dict[str, Any]:
    variants = {"A_VALUE_AREA_RETURN_ONLY": "VALUE_AREA_RETURN_ONLY", "B_VALUE_AREA_STOP_RUN": "VALUE_AREA_STOP_RUN", "C_VALUE_AREA_CVD_DIVERGENCE": "VALUE_AREA_CVD_DIVERGENCE", "D_FULL_VALUE_AREA_TRAP": "FULL"}
    values = {}
    for name, variant in variants.items():
        item = run_value_area_trap(bars, profiles, config.model_copy(update={"variant": variant}))
        values[name] = {"trade_count": len(item.trades), "expectancy": str(item.net_pnl / len(item.trades)) if item.trades else "0", "gross_pnl": str(item.gross_pnl), "net_pnl": str(item.net_pnl), "false_positive_reduction": None, "module_contribution": "diagnostic only; no parameter or winner selection"}
    return {"variants": values, "selection": "NONE; fixed ablations are explanatory diagnostics only"}


def build_value_area_reports(result: ValueAreaTrapResult, bars: list[FiveMinuteBar], profiles: dict, config: ValueAreaTrapConfig, *, calendar: CalendarArtifact | None = None) -> dict[str, Any]:
    evaluation = run_alpha_zero_scenario(result.trades, profile="ZERO_25K_EVALUATION")
    qualified = run_alpha_zero_scenario(result.trades, profile="ZERO_25K_QUALIFIED", calendar=calendar)
    return {"raw_strategy": raw_strategy_metrics(result), "alpha_zero_25k_evaluation": evaluation.model_dump(mode="json"), "alpha_zero_25k_qualified": qualified.model_dump(mode="json"), "risk_sensitivity": rolling_alpha_zero_scenarios(result.trades, profile="ZERO_25K_EVALUATION"), "ablations": fixed_ablation_diagnostics(bars, profiles, config), "warnings": ["BTCUSDT is not MES, MNQ, ES, or NQ.", "No report establishes CME performance or Alpha Futures compliance."]}
