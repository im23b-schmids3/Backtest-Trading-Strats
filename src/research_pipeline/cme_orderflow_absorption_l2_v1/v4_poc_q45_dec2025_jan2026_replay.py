"""Manual-only Dec-2025/Jan-2026 replay for the sealed L2 V4 Q45 challenger.

Importing this module performs no file reads and no execution.  The full replay
is available only through the explicit command-line entry point.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import median
from typing import Any, Callable, Iterable, Mapping, Sequence

from . import historical_runner as historical
from . import v3_poc_dec2025_jan2026_replay as parent
from .v2_quality050 import V2_CONFIG
from .v4_poc_q45 import (
    ELIGIBLE_STRUCTURAL_LEVELS,
    EVIDENCE_LABEL,
    PARENT_CONTRACT_SHA256,
    STRATEGY_ID,
    V4_CONFIG,
    v3_to_v4_contract_diff,
    v4_contract,
    v4_contract_sha256,
)


DATA_ROOT = parent.DATA_ROOT
AUDIT_REPORT = parent.AUDIT_OUTPUT
V3_ARTIFACT_ROOT = Path(
    "research_runs/CMEOrderflowAbsorption.ES_L2_V3_POC_ONLY_DEC2025_JAN2026_RETRO"
)
OUTPUT_ROOT = Path(
    "research_runs/CMEOrderflowAbsorption.ES_L2_V4_POC_ONLY_Q45_DEC2025_JAN2026_RETRO"
)
EXPECTED_SESSION_COUNT = parent.EXPECTED_SESSION_COUNT
QUALITY_V3 = 0.50
QUALITY_V4 = 0.45

V3_BENCHMARK: dict[str, Any] = {
    "source": "owner-supplied immutable published V3 benchmark",
    "sessions": 42,
    "trades": 65,
    "wins": 21,
    "losses": 44,
    "win_rate": 0.3230769230769231,
    "total_r": 9.383354585327483,
    "net_pnl_usd": 2136.50,
    "profit_factor": 1.2172233236744445,
    "max_cumulative_drawdown_r": -12.946378636204262,
    "es_trades": 46,
    "mes_trades": 19,
    "monthly": {
        "2025-12": {"trades": 38, "total_r": -3.790261594206621, "net_pnl_usd": -793.75},
        "2026-01": {"trades": 27, "total_r": 13.173616179534104, "net_pnl_usd": 2930.25},
    },
}


class V4ReplayError(RuntimeError):
    pass


def _validate_v4_contract() -> None:
    if parent.V3_CONTRACT_SHA256 != PARENT_CONTRACT_SHA256:
        raise V4ReplayError("frozen parent V3 hash literal changed")
    if parent.v3_contract_sha256() != PARENT_CONTRACT_SHA256:
        raise V4ReplayError("frozen parent V3 contract changed")
    diff = v3_to_v4_contract_diff()
    if diff["changed_strategy_fields"] != [
        {"field": "min_quality_score", "v3": QUALITY_V3, "v4": QUALITY_V4}
    ]:
        raise V4ReplayError("V4 contract diff is not the sealed one-field change")
    if parent.ELIGIBLE_STRUCTURAL_LEVELS != ELIGIBLE_STRUCTURAL_LEVELS:
        raise V4ReplayError("V4 POC eligibility differs from V3")
    if parent.v3_contract()["execution"] != v4_contract()["execution"]:
        raise V4ReplayError("V4 execution contract differs from V3")


def _quality_from_components(row: Mapping[str, Any], config: Any) -> float:
    raw = (
        float(row["aggression_score"]) * config.aggression_weight
        + float(row["restoration_score"]) * config.restoration_weight
        + float(row["price_resistance_score"]) * config.price_resistance_weight
        + float(row["persistence_score"]) * config.persistence_weight
        + float(row["multi_level_support_score"]) * config.multi_level_support_weight
        - float(row["false_refill_penalty"]) * config.false_refill_penalty_weight
    )
    return max(0.0, min(1.0, raw))


def decorate_pre_quality_interactions(runner: historical.HistoricalL2Runner) -> None:
    """Retain and label every POC interaction before either quality gate."""
    by_interaction: dict[str, dict[str, Any]] = {}
    for row in runner.interaction_ledger:
        v3_score = _quality_from_components(row, V2_CONFIG)
        v4_score = _quality_from_components(row, V4_CONFIG)
        emitted = float(row["l2_absorption_quality_score"])
        if v3_score != v4_score or abs(emitted - v4_score) > 1e-12:
            raise V4ReplayError("V3/V4 quality scores diverged despite identical score semantics")
        annotations = {
            "v3_quality_score": v3_score,
            "v4_recalculated_quality_score": v4_score,
            "quality_pass_0_50": v3_score >= QUALITY_V3,
            "quality_pass_0_45": v4_score >= QUALITY_V4,
            "incremental_q45_q50_band": QUALITY_V4 <= v4_score < QUALITY_V3,
        }
        row.update(annotations)
        by_interaction[str(row["interaction_id"])] = annotations
    for row in runner.setup_ledger:
        annotations = by_interaction.get(str(row["interaction_id"]))
        if annotations is None:
            raise V4ReplayError("setup ledger references an unknown pre-quality interaction")
        row.update(annotations)


def _execute_sessions(
    days: Sequence[str],
    preflight: Mapping[str, Any],
    data_root: Path,
    *,
    session_factory: Callable[..., historical.HistoricalL2Runner] = parent._run_session,
) -> list[historical.HistoricalL2Runner]:
    runners: list[historical.HistoricalL2Runner] = []
    for index, day in enumerate(days, start=1):
        print(f"=== V4 Q45 DEC/JAN RETRO {index:02d}/{len(days):02d} {day} ===", flush=True)
        runner = session_factory(
            day,
            preflight,
            data_root,
            config=V4_CONFIG,
            strategy_id=STRATEGY_ID,
            evidence_label=EVIDENCE_LABEL,
        )
        if any(runner is prior_runner for prior_runner in runners):
            raise V4ReplayError("V4 sessions must use independent chronological position state")
        decorate_pre_quality_interactions(runner)
        runners.append(runner)
    return runners


def _trade_key(row: Mapping[str, Any]) -> str:
    return f"{row['date']}|{row['interaction_id']}"


def _read_v3_trade_keys(path: Path) -> set[str]:
    if not path.is_file():
        raise V4ReplayError(f"immutable V3 trade ledger is missing: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"date", "interaction_id"}.issubset(reader.fieldnames):
            raise V4ReplayError("immutable V3 trade ledger lacks identity columns")
        return {_trade_key(row) for row in reader}


def _performance(trades: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(trades, key=lambda row: (int(row["exit_timestamp_ns"]), str(row["trade_id"])))
    r_values = [float(row.get("r_multiple") or 0.0) for row in ordered]
    pnl_values = [float(row["net_pnl_usd"]) for row in ordered]
    wins = sum(value > 0 for value in pnl_values)
    losses = sum(value < 0 for value in pnl_values)
    gross_profit = sum(value for value in pnl_values if value > 0)
    gross_loss = abs(sum(value for value in pnl_values if value < 0))
    equity = peak = drawdown = 0.0
    for value in r_values:
        equity += value
        peak = max(peak, equity)
        drawdown = min(drawdown, equity - peak)
    return {
        "trades": len(ordered),
        "wins": wins,
        "losses": losses,
        "win_rate": wins / len(ordered) if ordered else 0.0,
        "total_r": sum(r_values),
        "average_r": sum(r_values) / len(r_values) if r_values else 0.0,
        "median_r": float(median(r_values)) if r_values else None,
        "net_pnl_usd": sum(pnl_values),
        "profit_factor": gross_profit / gross_loss if gross_loss else None,
        "max_cumulative_drawdown_r": drawdown,
        "es_trades": sum(row["instrument"] == "ES" for row in ordered),
        "mes_trades": sum(row["instrument"] == "MES" for row in ordered),
        "stop_exits": sum(row["exit_reason"] == "STOP" for row in ordered),
        "target_exits": sum(row["exit_reason"] == "TARGET" for row in ordered),
        "hard_cutoff_exits": sum(str(row["exit_reason"]).startswith("HARD_") for row in ordered),
    }


def _breakdown(trades: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in trades:
        value = str(row["date"])[:7] if key == "month" else str(row[key])
        groups[value].append(row)
    return {name: _performance(rows) for name, rows in sorted(groups.items())}


def build_metrics(runners: Sequence[historical.HistoricalL2Runner]) -> dict[str, Any]:
    for runner in runners:
        runner.refresh_setup_ledger()
    interactions = [row for runner in runners for row in runner.interaction_ledger]
    setups = [row for runner in runners for row in runner.setup_ledger]
    trades = [row for runner in runners for row in runner.trade_ledger]
    accepted = [row for row in setups if bool(row["accepted"])]
    incremental_interactions = [row for row in interactions if bool(row["incremental_q45_q50_band"])]
    incremental_setup_ids = {
        str(row["setup_id"])
        for row in setups
        if bool(row["incremental_q45_q50_band"]) and bool(row["accepted"]) and row.get("setup_id")
    }
    incremental_setups = [row for row in setups if str(row.get("setup_id")) in incremental_setup_ids]
    incremental_trades = [row for row in trades if str(row["setup_id"]) in incremental_setup_ids]
    performance = _performance(trades)
    return {
        "completed_interactions": len(interactions),
        "accepted_poc_setups": len(accepted),
        "confirmations": sum(row.get("confirmation_timestamp_ns") is not None for row in accepted),
        "confirmation_expiries": sum(row.get("terminal_reason") == "CONFIRMATION_WINDOW_EXPIRED" for row in accepted),
        "active_position_blocks": sum(row.get("terminal_reason") == "COMPLIANCE_BLOCK_ACTIVE_POSITION" for row in accepted),
        "unresolved": sum(row.get("terminal_reason") is None for row in accepted),
        **performance,
        "breakdowns": {
            "day": _breakdown(trades, "date"),
            "month": _breakdown(trades, "month"),
            "direction": _breakdown(trades, "direction"),
            "instrument": _breakdown(trades, "instrument"),
        },
        "incremental_quality_band": {
            "range": "0.45 <= quality_score < 0.50",
            "interactions": len(incremental_interactions),
            "accepted_setups": len(incremental_setups),
            "confirmations": sum(row.get("confirmation_timestamp_ns") is not None for row in incremental_setups),
            **_performance(incremental_trades),
            "drawdown_interpretation": "standalone incremental-trade sequence; not a causal portfolio delta",
        },
    }


def build_comparison(
    metrics: Mapping[str, Any],
    v4_trades: Sequence[Mapping[str, Any]],
    v3_trade_keys: set[str],
) -> dict[str, Any]:
    v4_by_key = {_trade_key(row): row for row in v4_trades}
    v4_keys = set(v4_by_key)
    common = sorted(v4_keys & v3_trade_keys)
    incremental = sorted(
        key
        for key in v4_keys - v3_trade_keys
        if bool(v4_by_key[key].get("incremental_q45_q50_band"))
    )
    chronology_added = sorted(
        key
        for key in v4_keys - v3_trade_keys
        if not bool(v4_by_key[key].get("incremental_q45_q50_band"))
    )
    chronology_removed = sorted(v3_trade_keys - v4_keys)
    delta_fields = ("trades", "wins", "losses", "win_rate", "total_r", "net_pnl_usd", "profit_factor", "max_cumulative_drawdown_r")
    delta = {
        field: (
            None
            if metrics.get(field) is None or V3_BENCHMARK.get(field) is None
            else float(metrics[field]) - float(V3_BENCHMARK[field])
        )
        for field in delta_fields
    }
    monthly = {}
    for month, v3_row in V3_BENCHMARK["monthly"].items():
        v4_row = metrics["breakdowns"]["month"].get(month, _performance([]))
        monthly[month] = {
            "v3": v3_row,
            "v4": v4_row,
            "delta": {
                field: float(v4_row[field]) - float(v3_row[field])
                for field in ("trades", "total_r", "net_pnl_usd")
            },
        }
    return {
        "evidence_label": EVIDENCE_LABEL,
        "strict_chronological_oos": False,
        "v3": V3_BENCHMARK,
        "v4": dict(metrics),
        "delta": delta,
        "monthly": monthly,
        "trade_population_attribution": {
            "A_common_to_v3_and_v4": {"count": len(common), "trade_keys": common},
            "B_unique_to_v4_q45_q50_acceptance": {"count": len(incremental), "trade_keys": incremental},
            "C_independent_position_chronology": {
                "v4_parent_gate_trades_not_in_v3": chronology_added,
                "v3_trades_absent_from_v4": chronology_removed,
                "warning": "V4 is an independent chronological replay; it is not V3 plus appended trades.",
            },
        },
        "selection_or_optimization_performed": False,
    }


def _write_final_artifacts(
    output_root: Path,
    runners: list[historical.HistoricalL2Runner],
    metrics: Mapping[str, Any],
    comparison: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    historical.write_future_artifacts(output_root, runners, contract=v4_contract())
    summary = {
        "strategy_id": STRATEGY_ID,
        "parent_strategy_id": parent.STRATEGY_ID,
        "parent_contract_sha256": PARENT_CONTRACT_SHA256,
        "v4_contract_sha256": v4_contract_sha256(),
        "evidence_label": EVIDENCE_LABEL,
        "strict_chronological_oos": False,
        "quality_threshold": QUALITY_V4,
        "eligible_structural_levels": list(ELIGIBLE_STRUCTURAL_LEVELS),
        "contract_diff": v3_to_v4_contract_diff(),
        "adapter_audit": {"status": audit["status"], "totals": audit["totals"]},
        "metrics": dict(metrics),
        "pre_quality_interaction_features": {
            "all_completed_poc_interactions_retained": True,
            "below_0_45_retained": True,
            "v3_and_v4_scores_materialized": True,
            "full_dbn_rescan_needed_for_future_weight_research": False,
        },
        "phase_or_outcome_selection": False,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_root / "v3-v4-comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_root / "v3-v4-contract-diff.json").write_text(
        json.dumps(v3_to_v4_contract_diff(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_root / "diagnostic-report.md").write_text(
        "\n".join([
            f"# {STRATEGY_ID} — Dec 2025 / Jan 2026 retrospective research",
            "",
            f"Evidence classification: `{EVIDENCE_LABEL}`.",
            "",
            "This challenger was declared after the V3 Dec/Jan result and is not independent validation.",
            "The only strategy change is `min_quality_score: 0.50 -> 0.45`.",
            "No additional threshold, weight, filter, execution rule, or parameter was tested.",
            "",
            "`interaction-features.csv` retains every completed PRIOR_RTH_POC interaction before quality selection.",
            "V4 used independent chronological position state; results are not interpreted as V3 plus extra rows.",
            "",
        ]),
        encoding="utf-8",
    )
    return summary


def run(
    *,
    data_root: Path = DATA_ROOT,
    output_root: Path = OUTPUT_ROOT,
    audit_report: Path = AUDIT_REPORT,
    v3_artifact_root: Path = V3_ARTIFACT_ROOT,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"immutable V4 output already exists: {output_root}")
    _validate_v4_contract()
    preflight = parent.verify_data_preflight(data_root, verify_hashes=True)
    if preflight["status"] != "ALL_42_SESSIONS_DATA_SUFFICIENT":
        raise V4ReplayError("V4 replay blocked by incomplete V3 input data")
    audit = parent._load_passing_audit(audit_report, preflight)
    v3_trade_keys = _read_v3_trade_keys(v3_artifact_root / "trade-ledger.csv")
    days = list(preflight["base"]["target_sessions"])
    runners = _execute_sessions(days, preflight, data_root)
    if len(runners) != EXPECTED_SESSION_COUNT:
        raise V4ReplayError("V4 did not complete all 42 independent session runners")
    metrics = build_metrics(runners)
    trades = [row for runner in runners for row in runner.trade_ledger]
    interaction_annotations = {
        f"{row['date']}|{row['interaction_id']}": row
        for runner in runners
        for row in runner.interaction_ledger
    }
    for trade in trades:
        source = interaction_annotations[_trade_key(trade)]
        trade.update({
            "v3_quality_score": source["v3_quality_score"],
            "v4_recalculated_quality_score": source["v4_recalculated_quality_score"],
            "incremental_q45_q50_band": source["incremental_q45_q50_band"],
        })
    comparison = build_comparison(metrics, trades, v3_trade_keys)
    return _write_final_artifacts(output_root, runners, metrics, comparison, audit)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--audit-report", type=Path, default=AUDIT_REPORT)
    parser.add_argument("--v3-artifact-root", type=Path, default=V3_ARTIFACT_ROOT)
    args = parser.parse_args(argv)
    try:
        result = run(
            data_root=args.data_root,
            output_root=args.output_root,
            audit_report=args.audit_report,
            v3_artifact_root=args.v3_artifact_root,
        )
        print(json.dumps({
            "status": "V4_REPLAY_COMPLETE",
            "strategy_id": STRATEGY_ID,
            "v4_contract_sha256": v4_contract_sha256(),
            "output_root": str(args.output_root),
            "metrics": result["metrics"],
        }, indent=2, sort_keys=True))
    except (V4ReplayError, parent.DecJanReplayError, historical.HistoricalReplayError, FileExistsError, OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
