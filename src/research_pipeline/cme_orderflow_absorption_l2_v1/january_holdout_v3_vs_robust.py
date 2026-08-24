"""Strict January 2026 internal holdout for frozen V3 and one sealed challenger.

This module is an explicit local replay entry point over the completed causal
master tape. It has no DBN, acquisition, Databento, matrix-search, or candidate
selection path. The canonical V3 January replay must pass before the challenger
portfolio is evaluated.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import causal_master_tape as master
from . import historical_runner as historical
from . import weight_q_research as matrix
from .v3_poc_only import STRATEGY_ID as V3_STRATEGY_ID
from .v3_poc_only import v3_contract_sha256


STRATEGY_ID = "CMEOrderflowAbsorption.ES_L2_JAN2026_HOLDOUT_V3_VS_ROBUST_CANDIDATE"
EVIDENCE_LABEL = "JANUARY_2026_INTERNAL_HOLDOUT_AFTER_DECEMBER_SELECTION"
CHALLENGER_SELECTION_LABEL = "DECEMBER_SELECTED_ROBUST_STRESS_CANDIDATE_NOT_VALIDATED"
V3_CONTRACT_SHA256 = "a0ce94eeb78dcbf865cf4464bdf97ebc3f014a8ec5e2f559f99798534dfcbcb4"
JANUARY_START = date(2026, 1, 1)
JANUARY_END = date(2026, 1, 31)
MASTER_RELATIVE = Path("research_runs/CMEOrderflowAbsorption.ES_L2_CAUSAL_MASTER_DEC2025_JAN2026")
V3_REFERENCE_RELATIVE = Path("research_runs/CMEOrderflowAbsorption.ES_L2_V3_POC_ONLY_DEC2025_JAN2026_RETRO")
OUTPUT_RELATIVE = Path("research_runs/CMEOrderflowAbsorption.ES_L2_JAN2026_HOLDOUT_V3_VS_ROBUST_CANDIDATE")

EXPECTED_V3_JANUARY: dict[str, float | int] = {
    "completed_trades": 27,
    "wins": 11,
    "losses": 16,
    "total_r": 13.173616179534104,
    "net_pnl_usd": 2930.25,
    "profit_factor": 1.8357219251336898,
    "max_cumulative_drawdown_r": -7.0,
}


class JanuaryHoldoutError(RuntimeError):
    """The sealed January holdout cannot proceed without violating its contract."""


@dataclass(frozen=True)
class FrozenPortfolioContract:
    portfolio_id: str
    strategy_id: str
    config_id: str
    weights: tuple[tuple[str, Decimal], ...]
    quality_threshold: Decimal
    evidence_label: str

    def weight_mapping(self) -> dict[str, Decimal]:
        return dict(self.weights)

    def payload(self) -> dict[str, Any]:
        return {
            "portfolio_id": self.portfolio_id,
            "strategy_id": self.strategy_id,
            "config_id": self.config_id,
            "weights": {name: str(value) for name, value in self.weights},
            "quality_threshold": str(self.quality_threshold),
            "evidence_label": self.evidence_label,
            "eligible_structural_levels": ["PRIOR_RTH_POC"],
            "inherited_execution_contract_sha256": V3_CONTRACT_SHA256,
        }


V3_CONTRACT = FrozenPortfolioContract(
    portfolio_id="V3_BASELINE",
    strategy_id=V3_STRATEGY_ID,
    config_id="CMEOrderflowAbsorption.ES_L2_V3_POC_ONLY",
    weights=tuple(master.DEFAULT_WEIGHTS.items()),
    quality_threshold=Decimal("0.50"),
    evidence_label=EVIDENCE_LABEL,
)
CHALLENGER_CONTRACT = FrozenPortfolioContract(
    portfolio_id="ROBUST_STRESS_CANDIDATE",
    strategy_id=V3_STRATEGY_ID,
    config_id="W02-02-07-02-07-Q40",
    weights=(
        ("aggression_score", Decimal("0.10")),
        ("restoration_score", Decimal("0.10")),
        ("price_resistance_score", Decimal("0.35")),
        ("persistence_score", Decimal("0.10")),
        ("multi_level_support_score", Decimal("0.35")),
    ),
    quality_threshold=Decimal("0.40"),
    evidence_label=CHALLENGER_SELECTION_LABEL,
)
FROZEN_PORTFOLIOS = (V3_CONTRACT, CHALLENGER_CONTRACT)


@dataclass(frozen=True)
class JanuaryBundle:
    days: tuple[str, ...]
    interactions_by_day: Mapping[str, tuple[Mapping[str, Any], ...]]
    indexes: Mapping[str, Mapping[str, Any]]
    interaction_count: int
    source_to_master_id: Mapping[tuple[str, str], str]


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def challenger_contract_sha256() -> str:
    return _canonical_hash(CHALLENGER_CONTRACT.payload())


def _assert_frozen_contracts() -> None:
    if v3_contract_sha256() != V3_CONTRACT_SHA256:
        raise JanuaryHoldoutError("V3_CONTRACT_HASH_MISMATCH")
    if V3_CONTRACT.weight_mapping() != master.DEFAULT_WEIGHTS or V3_CONTRACT.quality_threshold != Decimal("0.50"):
        raise JanuaryHoldoutError("V3_FROZEN_PARAMETERS_MISMATCH")
    if CHALLENGER_CONTRACT.weight_mapping() != {
        "aggression_score": Decimal("0.10"),
        "restoration_score": Decimal("0.10"),
        "price_resistance_score": Decimal("0.35"),
        "persistence_score": Decimal("0.10"),
        "multi_level_support_score": Decimal("0.35"),
    } or CHALLENGER_CONTRACT.quality_threshold != Decimal("0.40"):
        raise JanuaryHoldoutError("CHALLENGER_FROZEN_PARAMETERS_MISMATCH")
    if sum(CHALLENGER_CONTRACT.weight_mapping().values()) != Decimal("1.00"):
        raise JanuaryHoldoutError("CHALLENGER_WEIGHT_SUM_MISMATCH")


def january_session_days(calendar: Mapping[str, Any]) -> tuple[str, ...]:
    raw = calendar.get("target_sessions")
    if not isinstance(raw, list):
        raise JanuaryHoldoutError("MASTER_CALENDAR_TARGET_SESSIONS_MISSING")
    selected: list[str] = []
    for value in raw:
        text = str(value)
        try:
            parsed = date.fromisoformat(text)
        except ValueError as exc:
            raise JanuaryHoldoutError(f"INVALID_MASTER_SESSION_DATE:{text}") from exc
        if JANUARY_START <= parsed <= JANUARY_END:
            selected.append(text)
    if not selected:
        raise JanuaryHoldoutError("NO_JANUARY_2026_SESSIONS_IN_MASTER")
    if selected != sorted(selected) or len(selected) != len(set(selected)):
        raise JanuaryHoldoutError("JANUARY_SESSION_CHRONOLOGY_INVALID")
    if any(not day.startswith("2026-01-") for day in selected):
        raise JanuaryHoldoutError("NON_JANUARY_SESSION_EXPOSED")
    return tuple(selected)


def _resolve_exact_paths(
    repository_root: Path, master_root: Path, v3_reference_root: Path, output_root: Path,
) -> tuple[Path, Path, Path, Path]:
    repository_root = repository_root.resolve()
    master_root = master_root.resolve()
    v3_reference_root = v3_reference_root.resolve()
    output_root = output_root.resolve()
    if master_root != (repository_root / MASTER_RELATIVE).resolve():
        raise JanuaryHoldoutError("UNSEALED_MASTER_ROOT")
    if v3_reference_root != (repository_root / V3_REFERENCE_RELATIVE).resolve():
        raise JanuaryHoldoutError("UNSEALED_V3_REFERENCE_ROOT")
    if output_root != (repository_root / OUTPUT_RELATIVE).resolve():
        raise JanuaryHoldoutError("UNSEALED_OUTPUT_ROOT")
    return repository_root, master_root, v3_reference_root, output_root


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise JanuaryHoldoutError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def _load_filtered_parquet(path: Path, days: Sequence[str]) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    table = pq.read_table(path, filters=[("session_date", "in", list(days))])
    try:
        rows = table.to_pylist()
    finally:
        del table
    allowed = set(days)
    if any(str(row.get("session_date")) not in allowed for row in rows):
        raise JanuaryHoldoutError("NON_JANUARY_ROW_EXPOSED_BY_PARQUET_FILTER")
    return rows


def load_january_bundle(master_root: Path, days: Sequence[str]) -> JanuaryBundle:
    interactions = _load_filtered_parquet(master_root / "interaction-master.parquet", days)
    index_rows = _load_filtered_parquet(master_root / "interaction-event-index.parquet", days)
    indexes = {str(row["interaction_id"]): row for row in index_rows}
    identities = {str(row["interaction_id"]) for row in interactions}
    if len(indexes) != len(index_rows) or identities != set(indexes):
        raise JanuaryHoldoutError("JANUARY_INTERACTION_INDEX_RECONCILIATION_FAILED")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    source_to_master: dict[tuple[str, str], str] = {}
    for row in interactions:
        day = str(row["session_date"])
        if day not in days:
            raise JanuaryHoldoutError("NON_JANUARY_INTERACTION_EXPOSED")
        grouped[day].append(row)
        key = (day, str(row["source_interaction_id"]))
        if key in source_to_master:
            raise JanuaryHoldoutError("DUPLICATE_SOURCE_INTERACTION_ID")
        source_to_master[key] = str(row["interaction_id"])
    frozen_grouped: dict[str, tuple[Mapping[str, Any], ...]] = {}
    for day in days:
        day_rows = grouped.get(day, [])
        day_rows.sort(key=lambda row: (int(row["interaction_end_ns"]), str(row["source_interaction_id"])))
        frozen_grouped[day] = tuple(day_rows)
    return JanuaryBundle(
        days=tuple(days), interactions_by_day=frozen_grouped, indexes=indexes,
        interaction_count=len(interactions), source_to_master_id=source_to_master,
    )


def _assert_expected_v3_metrics(metrics: Mapping[str, Any]) -> None:
    for field, expected in EXPECTED_V3_JANUARY.items():
        actual = metrics.get(field)
        if isinstance(expected, float):
            if actual is None or not math.isclose(float(actual), expected, rel_tol=0, abs_tol=1e-12):
                raise JanuaryHoldoutError(
                    f"V3_JANUARY_REPRODUCTION_FAILED:{field}:expected={expected!r}:actual={actual!r}"
                )
        elif actual != expected:
            raise JanuaryHoldoutError(
                f"V3_JANUARY_REPRODUCTION_FAILED:{field}:expected={expected!r}:actual={actual!r}"
            )


def _assert_expected_v3_gate(result: Mapping[str, Any], eligible_sessions: int) -> None:
    if result.get("dbn_files_opened") != 0 or result.get("network_calls") != 0:
        raise JanuaryHoldoutError("V3_GATE_USED_PROHIBITED_SOURCE")
    if int(result.get("sessions", -1)) != eligible_sessions:
        raise JanuaryHoldoutError("V3_JANUARY_SESSION_COUNT_MISMATCH")
    metrics = result.get("metrics")
    if not isinstance(metrics, Mapping):
        raise JanuaryHoldoutError("V3_JANUARY_METRICS_MISSING")
    _assert_expected_v3_metrics(metrics)


def _load_published_v3_reference(reference_root: Path) -> dict[str, Any]:
    summary_path = reference_root / "summary.json"
    ledger_path = reference_root / "trade-ledger.csv"
    if not summary_path.is_file() or not ledger_path.is_file():
        raise JanuaryHoldoutError("PUBLISHED_V3_REFERENCE_ARTIFACT_MISSING")
    summary = _read_json(summary_path)
    if summary.get("strategy_id") != V3_STRATEGY_ID or summary.get("v3_contract_sha256") != V3_CONTRACT_SHA256:
        raise JanuaryHoldoutError("PUBLISHED_V3_REFERENCE_CONTRACT_MISMATCH")
    january_summary = (
        summary.get("metrics", {}).get("breakdowns", {}).get("month", {}).get("2026-01")
    )
    if not isinstance(january_summary, Mapping):
        raise JanuaryHoldoutError("PUBLISHED_V3_JANUARY_SUMMARY_MISSING")
    for field in ("completed_trades", "wins", "losses", "total_r", "net_pnl_usd"):
        summary_field = "trades" if field == "completed_trades" else field
        actual, expected = january_summary.get(summary_field), EXPECTED_V3_JANUARY[field]
        if isinstance(expected, float):
            if actual is None or not math.isclose(float(actual), expected, rel_tol=0, abs_tol=1e-12):
                raise JanuaryHoldoutError(f"PUBLISHED_V3_JANUARY_SUMMARY_MISMATCH:{field}")
        elif actual != expected:
            raise JanuaryHoldoutError(f"PUBLISHED_V3_JANUARY_SUMMARY_MISMATCH:{field}")
    with ledger_path.open(newline="", encoding="utf-8-sig") as handle:
        january_trades = [row for row in csv.DictReader(handle) if str(row.get("date", "")).startswith("2026-01-")]
    metrics = historical._performance(january_trades)
    _assert_expected_v3_metrics(metrics)
    return {
        "metrics": metrics,
        "summary_sha256": master._sha256(summary_path),
        "trade_ledger_sha256": master._sha256(ledger_path),
        "january_trade_count": len(january_trades),
        "summary_january_metrics": dict(january_summary),
    }


def _assert_reference_consistency(canonical: Mapping[str, Any], reference: Mapping[str, Any]) -> None:
    canonical_metrics = canonical["metrics"]
    reference_metrics = reference["metrics"]
    for field in (
        "completed_trades", "wins", "losses", "win_rate", "total_r", "average_r", "median_r",
        "net_pnl_usd", "profit_factor", "max_cumulative_drawdown_r", "es_trades", "mes_trades",
        "target_exits", "stop_exits", "hard_cutoff_exits",
    ):
        left, right = canonical_metrics.get(field), reference_metrics.get(field)
        if left is None or right is None:
            if left != right:
                raise JanuaryHoldoutError(f"V3_PUBLISHED_REFERENCE_MISMATCH:{field}")
        elif isinstance(left, (float, int)) and isinstance(right, (float, int)):
            if not math.isclose(float(left), float(right), rel_tol=0, abs_tol=1e-12):
                raise JanuaryHoldoutError(f"V3_PUBLISHED_REFERENCE_MISMATCH:{field}")
        elif left != right:
            raise JanuaryHoldoutError(f"V3_PUBLISHED_REFERENCE_MISMATCH:{field}")


def _breakdown(
    trades: Sequence[Mapping[str, Any]], key: str, required_values: Sequence[str],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {value: [] for value in required_values}
    for trade in trades:
        value = str(trade["date"] if key == "day" else trade[key])
        grouped.setdefault(value, []).append(dict(trade))
    return {value: historical._performance(grouped[value]) for value in sorted(grouped)}


def _run_portfolio(master_root: Path, bundle: JanuaryBundle, contract: FrozenPortfolioContract) -> dict[str, Any]:
    terminal_outcomes: dict[str, str] = {}
    trades: list[dict[str, Any]] = []
    accepted_setups = confirmations = confirmation_expiries = active_blocks = unresolved = 0
    other_terminal: dict[str, int] = defaultdict(int)
    weights = contract.weight_mapping()
    for index, day in enumerate(bundle.days, start=1):
        print(f"JANUARY_{contract.portfolio_id} {index:02d}/{len(bundle.days):02d} {day}", flush=True)
        day_rows = list(bundle.interactions_by_day.get(day, ()))
        day_indexes = {str(row["interaction_id"]): bundle.indexes[str(row["interaction_id"])] for row in day_rows}
        accepted = [
            row for row in day_rows
            if master.interaction_is_accepted(
                row, threshold=contract.quality_threshold, weights=weights,
            )
        ]
        tape = matrix.SessionCausalTape.from_parquet(
            day, master_root / "causal-event-tape" / f"{day}.parquet"
        )
        session = matrix.simulate_independent_session(tape, accepted, day_indexes)
        accepted_setups += session.accepted_setups
        confirmations += session.confirmations
        confirmation_expiries += session.confirmation_expiries
        active_blocks += session.active_position_blocks
        unresolved += session.unresolved
        trades.extend(dict(row) for row in session.trades)
        for reason, count in session.other_terminal.items():
            other_terminal[reason] += count
        overlap = set(terminal_outcomes).intersection(session.terminal_outcomes)
        if overlap:
            raise JanuaryHoldoutError(f"DUPLICATE_TERMINAL_INTERACTION:{sorted(overlap)[0]}")
        terminal_outcomes.update(session.terminal_outcomes)
    trades.sort(key=lambda row: (int(row["exit_timestamp_ns"]), str(row["trade_id"])))
    if len({str(row["trade_id"]) for row in trades}) != len(trades):
        raise JanuaryHoldoutError("DUPLICATE_TRADE_ID")
    performance = historical._performance(trades)
    performance["hard_cutoff_exits"] = sum(str(row["exit_reason"]).startswith("HARD_") for row in trades)
    classified = confirmation_expiries + active_blocks + unresolved + sum(other_terminal.values()) + len(trades)
    if classified != accepted_setups or len(terminal_outcomes) != accepted_setups:
        raise JanuaryHoldoutError("PORTFOLIO_SETUP_RECONCILIATION_FAILED")
    return {
        "portfolio_id": contract.portfolio_id,
        "contract": contract.payload(),
        "contract_sha256": _canonical_hash(contract.payload()),
        "independent_chronological_portfolio_state": True,
        "eligible_sessions": len(bundle.days),
        "session_dates": list(bundle.days),
        "completed_interactions": bundle.interaction_count,
        "accepted_setups": accepted_setups,
        "confirmations": confirmations,
        "confirmation_expiries": confirmation_expiries,
        "active_position_blocks": active_blocks,
        **performance,
        "unresolved": unresolved,
        "other_terminal_outcomes": dict(sorted(other_terminal.items())),
        "terminal_outcomes": dict(sorted(terminal_outcomes.items())),
        "trades": trades,
        "breakdowns": {
            "day": _breakdown(trades, "day", bundle.days),
            "direction": _breakdown(trades, "direction", ("LONG", "SHORT")),
            "instrument": _breakdown(trades, "instrument", ("ES", "MES")),
        },
        "setup_reconciliation_pass": True,
        "dbn_files_opened": 0,
        "network_calls": 0,
    }


def _trade_master_ids(result: Mapping[str, Any], bundle: JanuaryBundle) -> dict[str, Mapping[str, Any]]:
    mapped: dict[str, Mapping[str, Any]] = {}
    for trade in result["trades"]:
        key = (str(trade["date"]), str(trade["interaction_id"]))
        master_id = bundle.source_to_master_id.get(key)
        if master_id is None:
            raise JanuaryHoldoutError(f"TRADE_INTERACTION_NOT_IN_MASTER:{key}")
        if master_id in mapped:
            raise JanuaryHoldoutError(f"DUPLICATE_TRADE_FOR_INTERACTION:{master_id}")
        mapped[master_id] = trade
    return mapped


def _comparison(v3: Mapping[str, Any], challenger: Mapping[str, Any], bundle: JanuaryBundle) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    v3_trades = _trade_master_ids(v3, bundle)
    candidate_trades = _trade_master_ids(challenger, bundle)
    common = sorted(set(v3_trades).intersection(candidate_trades))
    candidate_only = sorted(set(candidate_trades) - set(v3_trades))
    v3_only = sorted(set(v3_trades) - set(candidate_trades))
    rows: list[dict[str, Any]] = []
    all_ids = sorted(set(v3_trades).union(candidate_trades))
    for interaction_id in all_ids:
        v3_trade = v3_trades.get(interaction_id)
        candidate_trade = candidate_trades.get(interaction_id)
        v3_outcome = v3["terminal_outcomes"].get(interaction_id, "NOT_ACCEPTED")
        candidate_outcome = challenger["terminal_outcomes"].get(interaction_id, "NOT_ACCEPTED")
        chronology = "NONE"
        if candidate_trade is not None and v3_outcome == "COMPLIANCE_BLOCK_ACTIVE_POSITION":
            chronology = "CANDIDATE_TRADE_UNBLOCKED_RELATIVE_TO_V3"
        elif v3_trade is not None and candidate_outcome == "COMPLIANCE_BLOCK_ACTIVE_POSITION":
            chronology = "V3_TRADE_BLOCKED_IN_CANDIDATE"
        rows.append({
            "interaction_id": interaction_id,
            "date": str((candidate_trade or v3_trade)["date"]),
            "common_trade": v3_trade is not None and candidate_trade is not None,
            "v3_terminal_outcome": v3_outcome,
            "candidate_terminal_outcome": candidate_outcome,
            "one_position_chronology_difference": chronology,
            "v3_trade_id": "" if v3_trade is None else str(v3_trade["trade_id"]),
            "candidate_trade_id": "" if candidate_trade is None else str(candidate_trade["trade_id"]),
            "v3_r": "" if v3_trade is None else float(v3_trade["r_multiple"] or 0),
            "candidate_r": "" if candidate_trade is None else float(candidate_trade["r_multiple"] or 0),
            "v3_net_pnl_usd": "" if v3_trade is None else float(v3_trade["net_pnl_usd"]),
            "candidate_net_pnl_usd": "" if candidate_trade is None else float(candidate_trade["net_pnl_usd"]),
        })
    comparison = {
        "delta_convention": "candidate_minus_v3",
        "trade_count_delta": challenger["completed_trades"] - v3["completed_trades"],
        "win_rate_delta": challenger["win_rate"] - v3["win_rate"],
        "total_r_delta": challenger["total_r"] - v3["total_r"],
        "net_pnl_usd_delta": challenger["net_pnl_usd"] - v3["net_pnl_usd"],
        "profit_factor_delta": (
            None if challenger["profit_factor"] is None or v3["profit_factor"] is None
            else challenger["profit_factor"] - v3["profit_factor"]
        ),
        "max_cumulative_drawdown_r_delta": challenger["max_cumulative_drawdown_r"] - v3["max_cumulative_drawdown_r"],
        "common_trades": len(common),
        "common_interaction_ids": common,
        "trades_unique_to_candidate": len(candidate_only),
        "candidate_unique_interaction_ids": candidate_only,
        "v3_trades_lost": len(v3_only),
        "v3_lost_interaction_ids": v3_only,
        "one_position_chronology": {
            "v3_active_position_blocks": v3["active_position_blocks"],
            "candidate_active_position_blocks": challenger["active_position_blocks"],
            "active_position_block_delta": challenger["active_position_blocks"] - v3["active_position_blocks"],
            "candidate_trades_unblocked_relative_to_v3": sum(
                row["one_position_chronology_difference"] == "CANDIDATE_TRADE_UNBLOCKED_RELATIVE_TO_V3" for row in rows
            ),
            "v3_trades_blocked_in_candidate": sum(
                row["one_position_chronology_difference"] == "V3_TRADE_BLOCKED_IN_CANDIDATE" for row in rows
            ),
        },
    }
    return comparison, rows


def descriptive_classification(v3: Mapping[str, Any], challenger: Mapping[str, Any]) -> str:
    if float(challenger["total_r"]) <= 0:
        return "HOLDOUT_FAILURE"
    better = sum((
        float(challenger["total_r"]) > float(v3["total_r"]),
        challenger["profit_factor"] is not None and v3["profit_factor"] is not None
        and float(challenger["profit_factor"]) > float(v3["profit_factor"]),
        float(challenger["max_cumulative_drawdown_r"]) > float(v3["max_cumulative_drawdown_r"]),
    ))
    if better == 3:
        return "STRONG_HOLDOUT_SUCCESS"
    if better >= 2:
        return "MODERATE_HOLDOUT_SUCCESS"
    r_tolerance = max(1.0, abs(float(v3["total_r"])) * 0.10)
    if abs(float(challenger["total_r"]) - float(v3["total_r"])) <= r_tolerance:
        return "ROUGHLY_EQUIVALENT_TO_V3"
    return "WEAKER_THAN_V3_BUT_STILL_POSITIVE"


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_trade_comparison(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "interaction_id", "date", "common_trade", "v3_terminal_outcome", "candidate_terminal_outcome",
        "one_position_chronology_difference", "v3_trade_id", "candidate_trade_id", "v3_r", "candidate_r",
        "v3_net_pnl_usd", "candidate_net_pnl_usd",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _fmt_metric(value: object, digits: int = 6) -> str:
    return "N/A" if value is None else f"{float(value):.{digits}f}"


def _report(summary: Mapping[str, Any], v3: Mapping[str, Any], candidate: Mapping[str, Any], comparison: Mapping[str, Any]) -> str:
    return "\n".join([
        "# January 2026 internal holdout: V3 vs robust stress candidate", "",
        f"Evidence label: `{EVIDENCE_LABEL}`", "",
        f"Descriptive classification: **{summary['classification']}**. This classification does not accept, reject, mutate, or reselect a strategy.", "",
        "## Frozen contracts", "",
        f"- V3: weights 0.28/0.25/0.22/0.12/0.13, Q 0.50, contract `{V3_CONTRACT_SHA256}`.",
        "- Challenger: `W02-02-07-02-07-Q40`, weights 0.10/0.10/0.35/0.10/0.35, Q 0.40.", "",
        "## V3 reproduction gate", "",
        f"Status: `{summary['v3_reproduction_gate']['status']}`. Published PF and drawdown also reconciled before challenger execution.", "",
        "## Results", "",
        "| Portfolio | Sessions | Interactions | Accepted | Confirmed | Expired | Blocks | Trades | W/L | R | PnL | PF | DD R |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| V3 | {v3['eligible_sessions']} | {v3['completed_interactions']} | {v3['accepted_setups']} | {v3['confirmations']} | {v3['confirmation_expiries']} | {v3['active_position_blocks']} | {v3['completed_trades']} | {v3['wins']}/{v3['losses']} | {_fmt_metric(v3['total_r'])} | ${_fmt_metric(v3['net_pnl_usd'], 2)} | {_fmt_metric(v3['profit_factor'])} | {_fmt_metric(v3['max_cumulative_drawdown_r'])} |",
        f"| Candidate | {candidate['eligible_sessions']} | {candidate['completed_interactions']} | {candidate['accepted_setups']} | {candidate['confirmations']} | {candidate['confirmation_expiries']} | {candidate['active_position_blocks']} | {candidate['completed_trades']} | {candidate['wins']}/{candidate['losses']} | {_fmt_metric(candidate['total_r'])} | ${_fmt_metric(candidate['net_pnl_usd'], 2)} | {_fmt_metric(candidate['profit_factor'])} | {_fmt_metric(candidate['max_cumulative_drawdown_r'])} |", "",
        "## Direct comparison", "",
        f"Trade delta {comparison['trade_count_delta']:+d}; total-R delta {comparison['total_r_delta']:+.6f}; PnL delta ${comparison['net_pnl_usd_delta']:+.2f}; common trades {comparison['common_trades']}; candidate-only {comparison['trades_unique_to_candidate']}; V3 lost {comparison['v3_trades_lost']}.", "",
        "Both portfolios were recomputed independently from January-only master-tape rows. No December strategy result, DBN, network source, or additional weight/threshold was consulted.", "",
    ])


def run_holdout(
    *, repository_root: Path, master_root: Path, v3_reference_root: Path, output_root: Path,
) -> dict[str, Any]:
    repository_root, master_root, v3_reference_root, output_root = _resolve_exact_paths(
        repository_root, master_root, v3_reference_root, output_root,
    )
    _assert_frozen_contracts()
    if output_root.exists():
        raise FileExistsError(f"immutable January holdout output already exists: {output_root}")
    staging = output_root.with_name(output_root.name + ".building")
    if staging.exists():
        raise FileExistsError(f"January holdout staging output already exists: {staging}")

    master_validation = master.validate_building_root(master_root)
    calendar = _read_json(master_root / "calendar.json")
    days = january_session_days(calendar)

    canonical_v3 = master.replay_configuration(
        master_root, threshold=V3_CONTRACT.quality_threshold,
        weights=V3_CONTRACT.weight_mapping(), session_prefix="2026-01",
    )
    _assert_expected_v3_gate(canonical_v3, len(days))
    published_reference = _load_published_v3_reference(v3_reference_root)
    _assert_reference_consistency(canonical_v3, published_reference)
    print("V3_JANUARY_REPRODUCTION_GATE=PASS", flush=True)

    bundle = load_january_bundle(master_root, days)
    v3_result = _run_portfolio(master_root, bundle, V3_CONTRACT)
    compact_v3_metrics = matrix._assert_compact_v3(
        v3_result["trades"],
        canonical_v3,
        benchmark_label="JANUARY",
        expected_metrics=EXPECTED_V3_JANUARY,
    )
    _assert_expected_v3_metrics(v3_result)
    print("COMPACT_V3_JANUARY_IDENTITY_GATE=PASS", flush=True)

    # Challenger evaluation is intentionally below every V3 gate.
    candidate_result = _run_portfolio(master_root, bundle, CHALLENGER_CONTRACT)
    comparison, comparison_rows = _comparison(v3_result, candidate_result, bundle)
    classification = descriptive_classification(v3_result, candidate_result)
    summary = {
        "status": "JANUARY_INTERNAL_HOLDOUT_COMPLETE",
        "strategy_id": STRATEGY_ID,
        "evidence_label": EVIDENCE_LABEL,
        "classification": classification,
        "classification_is_descriptive_only": True,
        "automatic_acceptance_or_rejection": False,
        "automatic_reselection": False,
        "v5_created_or_declared": False,
        "evaluation_period": {"start_inclusive": "2026-01-01", "end_inclusive": "2026-01-31"},
        "eligible_sessions": list(days),
        "v3_reproduction_gate": {
            "status": "PASS",
            "expected": EXPECTED_V3_JANUARY,
            "canonical_metrics": canonical_v3["metrics"],
            "published_reference_metrics": published_reference["metrics"],
            "compact_metrics": compact_v3_metrics,
            "published_summary_sha256": published_reference["summary_sha256"],
            "published_trade_ledger_sha256": published_reference["trade_ledger_sha256"],
        },
        "v3_contract": V3_CONTRACT.payload(),
        "candidate_contract": CHALLENGER_CONTRACT.payload(),
        "candidate_contract_sha256": challenger_contract_sha256(),
        "v3_metrics": {key: value for key, value in v3_result.items() if key not in {"trades", "terminal_outcomes", "breakdowns"}},
        "candidate_metrics": {key: value for key, value in candidate_result.items() if key not in {"trades", "terminal_outcomes", "breakdowns"}},
        "comparison": comparison,
        "master_validation": master_validation,
        "independent_portfolio_states": True,
        "dbn_files_opened": 0,
        "databento_calls": 0,
        "network_calls": 0,
        "downloads": 0,
        "december_candidate_selection_rows_reused": 0,
    }

    staging.mkdir(parents=True)
    _write_json(staging / "summary.json", summary)
    _write_json(staging / "v3-results.json", v3_result)
    _write_json(staging / "candidate-results.json", candidate_result)
    _write_json(staging / "v3-vs-candidate.json", {**comparison, "classification": classification})
    _write_trade_comparison(staging / "trade-comparison.csv", comparison_rows)
    (staging / "diagnostic-report.md").write_text(
        _report(summary, v3_result, candidate_result, comparison), encoding="utf-8",
    )
    os.rename(staging, output_root)
    return {
        "status": summary["status"],
        "classification": classification,
        "output_root": str(output_root),
        "v3_reproduction_gate": "PASS",
        "candidate_id": CHALLENGER_CONTRACT.config_id,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--master-root", type=Path, required=True)
    parser.add_argument("--v3-reference-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = run_holdout(
            repository_root=args.repository_root,
            master_root=args.master_root,
            v3_reference_root=args.v3_reference_root,
            output_root=args.output_root,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", flush=True)
        return 1
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
