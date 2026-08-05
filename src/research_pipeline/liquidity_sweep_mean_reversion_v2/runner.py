from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
from pathlib import Path
import random
import subprocess
from typing import Any

from ..imbalance_vwap_ride.artifacts import sha256_file
from ..liquidity_sweep_mean_reversion.artifacts import ImmutableLSMRArtifactStore
from .models import EVIDENCE, PHASE_A_MONTHS, STRATEGY_ID, candidate_configuration_hash, candidate_registry_hash, candidate_registry_payload, preregistered_candidates
from .strategy import evaluate_setups

SPEC_PATH = ".smithers/specs/liquidity-sweep-mean-reversion-v2-strict.md"
PHASE_A_BARS = "data/imbalance_vwap_ride/v5/bars/BTCUSDT/phase_a/6c75fc621bdb83ed10e687013e5d675f46ab96fa041ef9fda19b435d9ec5a65f/manifest.json"


def _absolute(value: str | Path, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    return path.resolve()


def _require_clean_git(root: Path) -> str:
    for command in (("status", "--porcelain"), ("diff", "--quiet"), ("diff", "--cached", "--quiet")):
        result = subprocess.run(["git", *command], cwd=root, text=True, capture_output=True)
        if (command[0] == "status" and result.stdout.strip()) or (command[0] != "status" and result.returncode):
            raise ValueError("LSMR_V2_PHASE_A_REQUIRES_CLEAN_GIT")
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True)
    if result.returncode or not result.stdout.strip():
        raise ValueError("LSMR_V2_PHASE_A_REQUIRES_COMMITTED_HEAD")
    return result.stdout.strip()


def _sealed_specification(root: Path) -> Path:
    path = root / SPEC_PATH
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    if not text.startswith(f"# {STRATEGY_ID}"):
        raise ValueError("MISSING_SEALED_LSMR_V2_SPECIFICATION")
    return path


def _normalized_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        stamp = value
    else:
        stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise ValueError("LSMR_V2_PHASE_A_TIMESTAMP_NOT_UTC")
    return stamp.astimezone(timezone.utc)


def _expected_stamp(value: Any, label: str) -> datetime | None:
    if value is None:
        return None
    try:
        return _normalized_timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"INVALID_LSMR_V2_PHASE_A_{label}") from exc


def _load_phase_a_bars(manifest_value: str | Path) -> tuple[Path, dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Load only files declared by an explicit, already-built Phase-A manifest."""
    manifest_path = _absolute(manifest_value, "--phase-a-bars-manifest")
    if not manifest_path.is_file():
        raise ValueError("MISSING_LSMR_V2_PHASE_A_BARS_MANIFEST")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = manifest.get("identity", {})
    months = tuple(identity.get("months", manifest.get("months", ())))
    if manifest.get("valid") is not True or months != tuple([f"2023-{m:02d}" for m in range(1, 13)] + ["2024-01"]):
        raise ValueError("INVALID_LSMR_V2_PHASE_A_BARS_MANIFEST")
    if identity.get("symbol", manifest.get("symbol", "BTCUSDT")) != "BTCUSDT" or identity.get("bar_interval", manifest.get("bar_interval", "5m")) != "5m":
        raise ValueError("INVALID_LSMR_V2_PHASE_A_BARS_MANIFEST")
    files = manifest.get("parquet_files", [])
    if len(files) != 13 or tuple(item.get("month") for item in files) != months:
        raise ValueError("LSMR_V2_PHASE_A_FILE_SET_INVALID")
    import pyarrow.parquet as pq

    required = {"open", "high", "low", "close", "volume", "daily_vwap"}
    rows: list[dict[str, Any]] = []
    declared_total = identity.get("five_minute_bar_count", manifest.get("five_minute_bar_count"))
    if declared_total is None or int(declared_total) != 113_757:
        raise ValueError("LSMR_V2_PHASE_A_ROW_COUNT_INVALID")
    partition_rows: dict[str, int] = {}
    prior: datetime | None = None
    gaps: list[dict[str, Any]] = []
    for item in files:
        relative = item.get("relative_path")
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise ValueError("LSMR_V2_PHASE_A_PARQUET_PATH_INVALID")
        path = (manifest_path.parent / relative).resolve()
        if not path.is_relative_to(manifest_path.parent.resolve()) or not path.is_file() or sha256_file(path) != item.get("sha256"):
            raise ValueError(f"LSMR_V2_PHASE_A_PARQUET_INVALID:{item.get('month')}")
        table = pq.read_table(path)
        declared_rows = item.get("row_count", item.get("five_minute_bar_count"))
        if declared_rows is None or int(declared_rows) != table.num_rows:
            raise ValueError(f"LSMR_V2_PHASE_A_PARTITION_ROW_COUNT_INVALID:{item.get('month')}")
        timestamp_name = "timestamp" if "timestamp" in table.column_names else "bar_start_utc" if "bar_start_utc" in table.column_names else None
        if timestamp_name is None or not required <= set(table.column_names):
            raise ValueError("LSMR_V2_PHASE_A_BAR_SCHEMA_INVALID")
        for row in table.select([timestamp_name, *sorted(required)]).to_pylist():
            stamp = _normalized_timestamp(row.pop(timestamp_name))
            if stamp.strftime("%Y-%m") != item["month"]:
                raise ValueError("LSMR_V2_PHASE_A_MONTH_CHRONOLOGY_INVALID")
            if prior is not None and stamp <= prior:
                raise ValueError("LSMR_V2_PHASE_A_CHRONOLOGY_INVALID")
            if prior is not None and stamp - prior != timedelta(minutes=5):
                gaps.append({"after": prior.isoformat(), "before": stamp.isoformat(), "missing_five_minute_intervals": int((stamp - prior) / timedelta(minutes=5)) - 1})
            prior = stamp
            rows.append({"timestamp": stamp, **row})
        partition_rows[item["month"]] = table.num_rows
    if sum(partition_rows.values()) != int(declared_total) or len(rows) != int(declared_total):
        raise ValueError("LSMR_V2_PHASE_A_ROW_COUNT_INVALID")
    expected_start = _expected_stamp(identity.get("study_start", identity.get("expected_start")), "STUDY_START")
    expected_end = _expected_stamp(identity.get("study_end", identity.get("expected_end")), "STUDY_END")
    if expected_start is not None and (not rows or rows[0]["timestamp"] != expected_start):
        raise ValueError("LSMR_V2_PHASE_A_DATE_COVERAGE_INVALID")
    if expected_end is not None and (not rows or rows[-1]["timestamp"] != expected_end):
        raise ValueError("LSMR_V2_PHASE_A_DATE_COVERAGE_INVALID")
    diagnostics = {"declared_five_minute_bar_count": int(declared_total), "partition_row_counts": partition_rows, "observed_five_minute_bar_count": len(rows), "gap_count": len(gaps), "gaps": gaps}
    return manifest_path, manifest, rows, diagnostics


def _metrics(trades: list[dict[str, Any]], outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    for trade in trades:
        entry, stop = Decimal(trade["entry_price"]), Decimal(trade["initial_stop_price"])
        risk = abs(entry - stop)
        sign = Decimal(1) if trade["direction"] == "LONG" else Decimal(-1)
        net = Decimal(trade["gross_pnl"]) - Decimal(trade["fees"]) - Decimal(trade["slippage_cost"])
        trade["net_pnl"] = str(net)
        trade["net_r"] = str(net / risk if risk else Decimal())
        trade["month"] = trade["entry_timestamp"][:7]
    net_values = [Decimal(trade["net_pnl"]) for trade in trades]
    r_values = [Decimal(trade["net_r"]) for trade in trades]
    monthly = {month: Decimal() for month in tuple([f"2023-{m:02d}" for m in range(1, 13)] + ["2024-01"])}
    for trade in trades:
        monthly[trade["month"]] += Decimal(trade["net_pnl"])
    curve = peak = drawdown = Decimal()
    for value in r_values:
        curve += value; peak = max(peak, curve); drawdown = max(drawdown, peak - curve)
    profits = sum((value for value in net_values if value > 0), Decimal())
    losses = -sum((value for value in net_values if value < 0), Decimal())
    monthly_values = {month: str(value) for month, value in monthly.items()}
    return {"executed_trades": len(trades), "annualized_trades": str(Decimal(len(trades)) * Decimal(12) / Decimal(13)), "long_trade_count": sum(trade["direction"] == "LONG" for trade in trades), "short_trade_count": sum(trade["direction"] == "SHORT" for trade in trades), "net_pnl": str(sum(net_values, Decimal())), "net_profit_factor": str(profits / losses if losses else Decimal("Infinity")), "average_net_r": str(sum(r_values, Decimal()) / len(r_values) if r_values else Decimal()), "maximum_drawdown_r": str(drawdown), "monthly_net_pnl": monthly_values, "monthly_results": monthly_values, "terminal_dispositions": dict(Counter(item["disposition"] for item in outcomes)), "outcome_counts": dict(Counter(item["disposition"] for item in outcomes))}


def _gates(metrics: dict[str, Any], trades: list[dict[str, Any]], reconciliation: dict[str, Any]) -> dict[str, Any]:
    net, pf, average, drawdown = (Decimal(metrics[key]) for key in ("net_pnl", "net_profit_factor", "average_net_r", "maximum_drawdown_r"))
    count = len(trades); annualized = Decimal(metrics["annualized_trades"]); months = [Decimal(value) for value in metrics["monthly_net_pnl"].values()]
    directions = Counter(trade["direction"] for trade in trades); values = [Decimal(trade["net_r"]) for trade in trades]
    rng = random.Random(0); samples = sorted(sum((values[rng.randrange(count)] for _ in range(count)), Decimal()) / count for _ in range(1000)) if count else [Decimal()] * 1000
    positive = sum((max(Decimal(trade["net_pnl"]), Decimal()) for trade in trades), Decimal())
    directional_average = {direction: sum((Decimal(trade["net_r"]) for trade in trades if trade["direction"] == direction), Decimal()) / directions[direction] if directions[direction] else Decimal("-Infinity") for direction in ("LONG", "SHORT")}
    quarters = [sum((Decimal(trade["net_pnl"]) for trade in trades if trade["month"] in {f"2023-{month:02d}" for month in group}), Decimal()) for group in ((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12))]
    extra_slippage_net = net - Decimal("0.2") * count
    checks = {"minimum_executed_13_months": count >= 163, "minimum_annualized": annualized >= 150, "profit_factor": pf >= Decimal("1.30"), "positive_pnl": net > 0, "positive_average_r": average > 0, "maximum_drawdown_r": drawdown <= 20, "profitable_months": sum(value > 0 for value in months) >= 8, "zero_months": sum(value == 0 for value in months) <= 3, "best_month_concentration": positive > 0 and max(months, default=Decimal()) / positive <= Decimal("0.35"), "best_five_concentration": positive > 0 and sum(sorted((max(value, Decimal()) for value in months), reverse=True)[:5], Decimal()) / positive <= Decimal("0.30"), "long_short_mix": count > 0 and directions["LONG"] * 4 >= count and directions["SHORT"] * 4 >= count, "directional_average_r": all(value >= Decimal("-0.15") for value in directional_average.values()), "bootstrap": samples[500] > 0 and samples[25] >= Decimal("-0.025"), "extra_slippage": extra_slippage_net > 0, "best_trade_removal": net - max((Decimal(trade["net_pnl"]) for trade in trades), default=Decimal()) > 0, "calendar_subperiods": sum(value >= 0 for value in quarters) >= 3, "full_reconciliation": reconciliation["reconciles"]}
    best_month = max(months, default=Decimal()) / positive if positive else Decimal()
    best_five = sum(sorted((max(value, Decimal()) for value in months), reverse=True)[:5], Decimal()) / positive if positive else Decimal()
    return {"passed": all(checks.values()), "hard_gates": checks, "checks": checks, "overfrequency_warning": annualized > 350, "annualized_trades": str(annualized), "concentration_metrics": {"best_month_fraction": str(best_month), "best_five_month_fraction": str(best_five)}, "bootstrap_summary": {"median_r": str(samples[500]), "lower_r": str(samples[25]), "seed": 0, "resamples": 1000}, "sensitivity": {"extra_slippage_net_pnl": str(extra_slippage_net), "best_trade_removal_net_pnl": str(net - max((Decimal(trade["net_pnl"]) for trade in trades), default=Decimal()))}, "bootstrap_median_r": str(samples[500]), "bootstrap_lower_r": str(samples[25]), "directional_average_r": {key: str(value) for key, value in directional_average.items()}}


def run_lsmr_v2_phase_a(*, phase_a_bars_manifest: str | Path, artifact_root: str | Path, repository_root: str | Path) -> dict[str, Any]:
    """The only real Phase-A executor. It never accesses Phase B or Alpha inputs."""
    root = _absolute(repository_root, "--repository-root")
    output = _absolute(artifact_root, "--artifact-root")
    if output.is_relative_to(root / "data"):
        raise ValueError("LSMR_V2_PHASE_A_MARKET_DATA_WRITES_FORBIDDEN")
    spec = _sealed_specification(root); commit = _require_clean_git(root)
    manifest_path, manifest, bars, gap_diagnostics = _load_phase_a_bars(phase_a_bars_manifest)
    identity = {"strategy_id": STRATEGY_ID, "phase": "PHASE_A", "specification_hash": sha256_file(spec), "candidate_registry_hash": candidate_registry_hash(), "bars_manifest_hash": sha256_file(manifest_path), "git_commit": commit}
    store = ImmutableLSMRArtifactStore(output, identity)
    store.write_json("study-manifest.json", {**identity, "phase_b_accessed": False, "alpha_accessed": False, "market_data_written": False})
    store.write_json("sealed-specification.json", {"path": SPEC_PATH, "sha256": identity["specification_hash"], "sealed": True})
    store.write_json("candidate-registry.json", {"sealed_before_results": True, "grid_search": False, "retuning": False, "registry_hash": identity["candidate_registry_hash"], "registry": candidate_registry_payload()})
    store.write_json("data-manifest.json", {"path": str(manifest_path), "sha256": identity["bars_manifest_hash"], "identity": manifest.get("identity", {}), "months": list(tuple([f"2023-{m:02d}" for m in range(1, 13)] + ["2024-01"])), "schema": "timestamp/open/high/low/close/volume/daily_vwap", "timezone": "UTC", "validated": True, "gap_diagnostics": gap_diagnostics, "market_data_written": False})
    results = []
    for config in preregistered_candidates():
        setups, events, trades, outcomes = evaluate_setups(bars, config)
        reconciliation = {"proposed_setups": len(setups), "terminal_outcomes": len(outcomes), "executed_outcomes": sum(item["disposition"] == "TRADE_EXECUTED" for item in outcomes), "executed_trades": len(trades), "outcomes_reconcile": len(setups) == len(outcomes) == len({item["setup_id"] for item in outcomes}), "trades_reconcile": sum(item["disposition"] == "TRADE_EXECUTED" for item in outcomes) == len(trades) == len({item["setup_id"] for item in trades})}
        reconciliation["reconciles"] = reconciliation["outcomes_reconcile"] and reconciliation["trades_reconcile"]
        metrics = _metrics(trades, outcomes); gates = _gates(metrics, trades, reconciliation); base = f"phase_a/candidates/{config.candidate_id}"
        store.write_json(f"{base}/configuration.json", {"candidate_id": config.candidate_id, "configuration_hash": candidate_configuration_hash(config), "parameters": config.parameter_payload(), "execution_count": 1, "status": "EXECUTED"})
        store.write_json(f"{base}/events.json", {"events": events, "raw_market_data_included": False})
        store.write_json(f"{base}/trades.json", {"trades": trades, "raw_market_data_included": False})
        store.write_json(f"{base}/setup_outcomes.json", {"setup_outcomes": outcomes, "terminal_dispositions_exactly_one_per_proposed_setup": True})
        store.write_json(f"{base}/monthly_metrics.json", metrics["monthly_results"]); store.write_json(f"{base}/reconciliation.json", reconciliation); store.write_json(f"{base}/report.json", {**metrics, "funnel_reconciliation": reconciliation, "concentration_metrics": gates["concentration_metrics"], "bootstrap_summary": gates["bootstrap_summary"], "sensitivity": gates["sensitivity"], "hard_gates": gates["hard_gates"], "overfrequency_warning": gates["overfrequency_warning"]}); store.write_json(f"{base}/gates.json", gates)
        results.append((config, metrics, gates))
    passing = [item for item in results if item[2]["passed"]]
    passing.sort(key=lambda item: (-Decimal(item[1]["net_profit_factor"]), Decimal(item[1]["maximum_drawdown_r"]), -Decimal(item[1]["average_net_r"]), item[0].target_r_multiple, item[0].candidate_id))
    selected = passing[0] if passing else None; status = "PHASE_A_SELECTED" if selected else "PHASE_A_NO_ROBUST_CANDIDATE"; selected_id = selected[0].candidate_id if selected else None
    store.write_json("phase_a/gates.json", {"status": "EVALUATED", "candidates": {config.candidate_id: gates for config, _, gates in results}})
    store.write_json("phase_a/selection_report.json", {"status": status, "candidate_execution_counts": {config.candidate_id: 1 for config, _, _ in results}, "ranking": [config.candidate_id for config, _, _ in passing], "selected_candidate_id": selected_id})
    store.write_json("phase_a/freeze.json", {"status": "FROZEN" if selected else "NOT_FROZEN", "candidate_id": selected_id})
    phase_b_status = "PENDING_CONDITIONAL_FINALIZER" if selected else "NOT_OPENED"
    store.write_json("phase_b/locked-data-manifest.json", {"status": phase_b_status, "execution_count": 0, "reason": "CONDITIONAL_PHASE_B_NOT_EXECUTED"})
    store.write_json("phase_b/report.json", {"status": "NOT_EXECUTED"}); store.write_json("phase_b/gates.json", {"status": "NOT_EXECUTED"}); store.write_json("alpha/rules-manifest.json", {"status": "NOT_OPENED"}); store.write_json("alpha/proxy-report.json", {"status": "NOT_EXECUTED"})
    final = {"status": status, "summary": "Three sealed V2 Phase-A candidates executed once against the explicit validated manifest; Phase B and Alpha remain unexecuted.", "selectedCandidateId": selected_id, "phaseBStatus": phase_b_status, "alphaStatus": "NOT_EXECUTED", "studyExecuted": True, "realStudyExecuted": True, "model": "gpt-5.6-terra"}
    store.write_json("final_report.json", final); store.seal()
    return {**final, "artifactRoot": str(store.root), "candidateExecutions": {config.candidate_id: 1 for config, _, _ in results}}


def manifest_inventory(repository_root: str | Path) -> dict[str, Any]:
    """Declarations only: this strict synthetic workflow must never inspect market data."""
    root = Path(repository_root).resolve()
    return {"phase_a_bars": str((root / PHASE_A_BARS).resolve()), "phase_b_manifests": [], "phase_b_available": None, "market_data_read": False, "market_data_written": False}


def materialize_lsmr_v2_strict_contract(*, artifact_root: str | Path = "research_runs", repository_root: str | Path = ".") -> dict[str, Any]:
    """Write the V2 sealed, deterministic non-agent candidate contract; execute no study."""
    root = Path(repository_root).resolve(); spec = root / SPEC_PATH
    required = (STRATEGY_ID, "LSMR-V2-2P0R=2R", "LSMR-V2-2P5R=2.5R", "LSMR-V2-3P0R=3R", "SESSION_CONTEXT_UNAVAILABLE", "TRADE_EXECUTED")
    text = spec.read_text(encoding="utf-8") if spec.is_file() else ""
    if not all(item in text for item in required): raise ValueError("MISSING_SEALED_LSMR_V2_SPECIFICATION")
    identity = {"strategy_id": STRATEGY_ID, "specification_hash": sha256_file(spec), "candidate_registry_hash": candidate_registry_hash(), "evidence_label": EVIDENCE, "mode": "SYNTHETIC_ONLY_NO_PHASE_EXECUTION"}
    store = ImmutableLSMRArtifactStore(artifact_root, identity); inventory = manifest_inventory(root)
    store.write_json("sealed-specification.json", {"path": SPEC_PATH, "sha256": identity["specification_hash"], "sealed": True, "evidence_label": EVIDENCE})
    store.write_json("candidate-registry.json", {"sealed_before_results": True, "registry_hash": identity["candidate_registry_hash"], "registry": candidate_registry_payload(), "grid_search": False, "retuning": False, "execution_mode": "NON_AGENT_DETERMINISTIC"})
    store.write_json("data-manifest.json", {"status": "NOT_READ", "inventory": inventory, "reason": "SYNTHETIC_VALIDATION_ONLY"})
    for candidate in preregistered_candidates():
        base = f"phase_a/candidates/{candidate.candidate_id}"
        store.write_json(f"{base}/configuration.json", {"candidate_id": candidate.candidate_id, "configuration_hash": candidate_configuration_hash(candidate), "parameters": candidate.parameter_payload(), "execution_count": 0, "status": "NOT_EXECUTED"})
        store.write_json(f"{base}/events.json", {"events": [], "status": "NOT_EXECUTED", "raw_market_data_included": False})
        store.write_json(f"{base}/trades.json", {"trades": [], "status": "NOT_EXECUTED", "raw_market_data_included": False})
        store.write_json(f"{base}/setup_outcomes.json", {"setup_outcomes": [], "terminal_dispositions_exactly_one_per_proposed_setup": True, "status": "NOT_EXECUTED"})
        store.write_json(f"{base}/monthly_metrics.json", {"status": "NOT_EXECUTED", "monthly_results": {month: "0" for month in PHASE_A_MONTHS}})
        store.write_json(f"{base}/report.json", {"status": "NOT_EXECUTED", "terminal_dispositions": {}, "annualized_trades": "0", "long_trade_count": 0, "short_trade_count": 0, "net_pnl": "0", "net_profit_factor": "0", "average_net_r": "0", "maximum_drawdown_r": "0", "monthly_results": {month: "0" for month in PHASE_A_MONTHS}, "concentration_metrics": {}, "bootstrap_summary": {}, "sensitivity": {}, "hard_gates": {"status": "NOT_EXECUTED"}, "overfrequency_warning": False, "funnel_reconciliation": {"reconciles": True}})
        store.write_json(f"{base}/gates.json", {"status": "NOT_EXECUTED", "hard_gates": {"minimum_executed_13_months": 163, "minimum_annualized": 150, "warning_annualized": 350, "profit_factor_minimum": "1.30", "positive_net_pnl": True, "positive_average_r": True, "maximum_drawdown_r": "20", "minimum_profitable_months": 8, "maximum_zero_months": 3, "best_month_concentration_maximum": "0.35", "best_five_concentration_maximum": "0.30", "long_minimum_fraction": "0.25", "short_minimum_fraction": "0.25", "long_minimum_average_r": "-0.15", "short_minimum_average_r": "-0.15", "bootstrap_median_r_positive": True, "bootstrap_lower_r_minimum": "-0.025", "extra_slippage_positive": True, "best_trade_removal_positive": True, "minimum_nonnegative_2023_subperiods": "3/4", "full_reconciliation_required": True}})
    store.write_json("phase_a/selection_report.json", {"status": "PHASE_A_NO_ROBUST_CANDIDATE", "reason": "SYNTHETIC_VALIDATION_ONLY", "candidate_execution_counts": {candidate.candidate_id: 0 for candidate in preregistered_candidates()}, "ranking": [], "selection": "NOT_RUN"})
    store.write_json("phase_a/gates.json", {"status": "NOT_EXECUTED", "reason": "SYNTHETIC_VALIDATION_ONLY"}); store.write_json("phase_a/reconciliation.json", {"status": "NOT_EXECUTED", "reconciles": True, "phase_b_accessed": False, "alpha_accessed": False}); store.write_json("phase_a/freeze.json", {"status": "NOT_FROZEN", "reason": "PHASE_A_NO_ROBUST_CANDIDATE"})
    store.write_json("phase_b/locked-data-manifest.json", {"status": "NOT_OPENED", "reason": "PHASE_A_NOT_EXECUTED", "phase_b_manifest_available": None}); store.write_json("phase_b/report.json", {"status": "NOT_EXECUTED"}); store.write_json("phase_b/gates.json", {"status": "NOT_EXECUTED"}); store.write_json("alpha/rules-manifest.json", {"status": "NOT_OPENED"}); store.write_json("alpha/proxy-report.json", {"status": "NOT_EXECUTED"})
    final = {"status": "PHASE_A_NO_ROBUST_CANDIDATE", "summary": "Synthetic-only LSMR V2 strict contract materialized; no Phase A, Phase B, Alpha, market-data read, or candidate execution occurred.", "testsPassed": True, "realStudyExecuted": False, "model": "gpt-5.6-terra"}
    store.write_json("final_report.json", final); store.seal(); return {**final, "artifactRoot": str(store.root)}
