from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Callable

from .artifacts import sha256_value
from .strategy import _ts
from .v3_strategy import (
    ACTIVE_STATES,
    TERMINAL_STATES,
    _trade_summary,
    run_imbalance_vwap_ride_v3,
    simulate_long_trade as _simulate_v3_long_trade,
)
from .v4_models import (
    COST_MODEL_VERSION,
    LOCKED_EVIDENCE,
    PHASE_A_MONTHS,
    PHASE_B_MONTHS,
    SELECTION_EVIDENCE,
    ImbalanceVWAPRideV4Config,
)


def _month(value: Any) -> str:
    if isinstance(value, datetime):
        return value.strftime("%Y-%m")
    return str(value)[:7]


def _phase_for_bars(bars: list[dict[str, Any]]) -> str:
    months = {
        str(item.get("month") or _month(_ts(item["bar_start_utc"])))
        for item in bars
    }
    if months and months <= set(PHASE_A_MONTHS):
        return "PHASE_A"
    if months and months <= set(PHASE_B_MONTHS):
        return "PHASE_B"
    return "FIXTURE"


def _fixed_subperiods(
    trades: list[dict[str, Any]], bars: list[dict[str, Any]], phase: str
) -> dict[str, Any]:
    if phase == "PHASE_A":
        periods = (("JAN_JUN_2023", PHASE_A_MONTHS[:6]), ("JUL_DEC_2023", PHASE_A_MONTHS[6:]))
    elif phase == "PHASE_B":
        periods = (("FEB_APR_2025", PHASE_B_MONTHS[:3]), ("MAY_JUL_2025", PHASE_B_MONTHS[3:]))
    else:
        present = tuple(sorted({str(item.get("month") or _month(item["bar_start_utc"])) for item in bars}))
        periods = (("FIXTURE_PERIOD", present),)
    output: dict[str, Any] = {}
    for label, months in periods:
        subset = [item for item in trades if _month(item["entry_timestamp"]) in months]
        output[label] = {
            "months": list(months),
            "five_minute_bar_count": sum(
                str(item.get("month") or _month(item["bar_start_utc"])) in months for item in bars
            ),
            **_trade_summary(subset),
        }
    return output


def simulate_long_trade(
    *,
    zone: dict[str, Any],
    signal_bar: dict[str, Any],
    entry_index: int,
    bars: list[dict[str, Any]],
    config: ImbalanceVWAPRideV4Config,
) -> tuple[str, dict[str, Any] | None]:
    """Run the fixed V4 long-only next-bar-open execution contract."""

    state, trade = _simulate_v3_long_trade(
        zone=zone,
        signal_bar=signal_bar,
        entry_index=entry_index,
        bars=bars,
        config=config,  # type: ignore[arg-type]
    )
    if trade is not None:
        trade["cost_model_version"] = COST_MODEL_VERSION
        trade["candidate_id"] = config.candidate_id
        trade["trade_id"] = sha256_value(
            {
                "kind": "trade-v4-long",
                "candidate_id": config.candidate_id,
                "zone_id": trade["zone_id"],
                "entry_timestamp": trade["entry_timestamp"],
                "frozen_parameters": config.parameter_payload(),
            }
        )[:24]
    return state, trade


def reconcile_funnel(funnel: dict[str, Any]) -> dict[str, Any]:
    proposed = int(funnel.get("proposed_setups", 0))
    components = sum(
        int(funnel.get(name, 0))
        for name in (
            "invalid_setups",
            "non_executable_setups",
            "compliance_blocks",
            "executed_trades",
        )
    )
    return {
        **funnel,
        "formula": (
            "proposed_setups = invalid_setups + non_executable_setups + "
            "compliance_blocks + executed_trades"
        ),
        "components_total": components,
        "reconciles": proposed == components,
    }


def run_imbalance_vwap_ride_v4(
    bars: list[dict[str, Any]],
    footprints: list[dict[str, Any]] | dict[datetime, list[dict[str, Any]]],
    config: ImbalanceVWAPRideV4Config = ImbalanceVWAPRideV4Config(),
    *,
    phase: str | None = None,
    compliance_check: Callable[[dict[str, Any], dict[str, Any]], tuple[bool, str | None]]
    | None = None,
) -> dict[str, Any]:
    """Execute one sealed V4 candidate without short or same-bar entry paths.

    The verified V3 lifecycle engine is reused read-only because V4 changes the
    candidate registry and temporal selection protocol, not the execution
    semantics. V4 identity, hashes, phase diagnostics, and trade IDs are then
    applied to the returned local result.
    """

    resolved_phase = phase or _phase_for_bars(bars)
    if resolved_phase not in {"PHASE_A", "PHASE_B", "FIXTURE"}:
        raise ValueError(f"unsupported V4 phase: {resolved_phase}")
    result = run_imbalance_vwap_ride_v3(
        bars,
        footprints,
        config,  # type: ignore[arg-type]
        compliance_check=compliance_check,
    )
    result["candidate_id"] = config.candidate_id
    result["variant_id"] = config.candidate_id
    result["phase"] = resolved_phase
    result["evidence_label"] = (
        LOCKED_EVIDENCE if resolved_phase == "PHASE_B" else SELECTION_EVIDENCE
    )
    result["confirmation_evidence"] = False
    result["optimization_claimed"] = False
    result["requires_external_live_or_contract_accurate_confirmation"] = True
    for trade in result["trades"]:
        trade["candidate_id"] = config.candidate_id
        trade["cost_model_version"] = COST_MODEL_VERSION
        trade["trade_id"] = sha256_value(
            {
                "kind": "trade-v4-long",
                "candidate_id": config.candidate_id,
                "zone_id": trade["zone_id"],
                "entry_timestamp": trade["entry_timestamp"],
                "frozen_parameters": config.parameter_payload(),
            }
        )[:24]
    result["funnel"] = reconcile_funnel(result["funnel"])
    if not result["funnel"]["reconciles"]:
        raise AssertionError("V4 funnel failed exact reconciliation")
    result["metrics"]["funnel_reconciliation"] = result["funnel"]
    result["metrics"]["cost_model_version"] = COST_MODEL_VERSION
    result["metrics"]["short_setups"] = 0
    result["metrics"]["short_orders"] = 0
    result["metrics"]["short_fills"] = 0
    result["metrics"]["short_pnl"] = "0"
    result["metrics"]["long_only_reconciliation"].update(
        {
            "short_setups": 0,
            "short_orders": 0,
            "short_fills": 0,
            "short_pnl": "0",
            "reconciles": all(item.get("direction") == "LONG" for item in result["trades"]),
        }
    )
    result["subperiods"] = _fixed_subperiods(result["trades"], bars, resolved_phase)
    if any(item.get("direction") != "LONG" for item in result["trades"] + result["zones"]):
        raise AssertionError("V4 emitted a non-long trade or zone")
    if any(item.get("direction") not in {None, "LONG"} for item in result["events"]):
        raise AssertionError("V4 emitted a short event")
    return result


__all__ = [
    "ACTIVE_STATES",
    "TERMINAL_STATES",
    "reconcile_funnel",
    "run_imbalance_vwap_ride_v4",
    "simulate_long_trade",
]
