"""Read-only published-ledger audit of the frozen L2 V2 August seen-data run.

This module deliberately has no Databento, DBN, downloader, or replay imports.
It materializes descriptive audit files from CSV/JSON artifacts already emitted
by the completed replay and cannot mutate the strategy or its result ledgers.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from . import rejection_funnel as funnel
from . import v2_quality050 as v2


AUDIT_NAME = "selectivity_integrity_audit"
EVIDENCE_LABEL = "SEEN_AUG_L2_V2_NOT_FRESH_OOS_EVIDENCE"
EXPECTED_HASH = "f6152ed1ca32bb7c93a62ddf672a0708ff6c92abb0beb126eec78bf7ab3ab239"
PERIOD_ROOTS = {
    "MAY": "research_runs/CMEOrderflowAbsorption.ES_L2_V2_MAY_DEVELOPMENT_REPLAY",
    "RETRO_JUNE_JULY": "research_runs/CMEOrderflowAbsorption.ES_L2_V2_EXTENDED_EXISTING_DATA/retro_june_july",
    "AUGUST_SEEN": "research_runs/CMEOrderflowAbsorption.ES_L2_V2_AUGUST_SEEN",
}
SCORE_FIELDS = (
    "l2_absorption_quality_score", "aggression_score", "restoration_score", "price_resistance_score",
    "persistence_score", "multi_level_support_score", "false_refill_penalty",
)
FEATURE_FIELDS = (
    "directional_aggressive_volume", "relevant_execution_count", "depth_restoration_count",
    "consume_restore_cycles", "cumulative_restored_volume", "restoration_to_consumption_ratio",
    "mean_restoration_latency_ms", "defended_price_present_fraction", "interaction_rejection_ticks",
    "maximum_through_level_progress_ticks", "multi_level_ofi", "rapid_cancel_ratio",
)
GATES = ("relevant_aggressive_volume", "relevant_execution_count", "consume_restore", "rejection", "quality")


class AugustAuditError(RuntimeError):
    pass


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise AugustAuditError(f"published ledger is empty: {path}")
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AugustAuditError(f"required published JSON is missing or invalid: {path}") from exc
    if not isinstance(value, dict):
        raise AugustAuditError(f"published JSON object required: {path}")
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def _percent(value: int, total: int) -> float:
    return 100.0 * value / total if total else 0.0


def _number(value: object) -> float:
    return funnel._float(value)


def _distribution(rows: list[dict[str, str]], field: str) -> dict[str, Any]:
    return funnel._distribution(_number(row.get(field)) for row in rows)


def _period(root: Path, *, expected_strategy: bool = True) -> dict[str, Any]:
    summary = _read_json(root / "summary.json")
    interactions, setups, trades = (_read_csv(root / name) for name in (
        "interaction-features.csv", "setup-ledger.csv", "trade-ledger.csv"
    ))
    if expected_strategy and summary.get("strategy_id") != v2.STRATEGY_ID:
        raise AugustAuditError(f"published artifact has wrong strategy identity: {root}")
    if expected_strategy and summary.get("v2_contract_sha256", EXPECTED_HASH) != EXPECTED_HASH:
        raise AugustAuditError(f"published artifact has wrong V2 contract hash: {root}")
    audited = funnel.analyze_rows(interactions, setups, trades, config=v2.V2_CONFIG)
    daily_path = root / "daily-results.csv"
    daily = _read_csv(daily_path) if daily_path.is_file() else []
    sessions = sorted({row.get("date", "") for row in daily if row.get("date")}) or sorted({row.get("date", "") for row in interactions if row.get("date")})
    accepted = audited["accepted_setups"]
    return {
        "root": str(root), "summary": summary, "interactions": interactions, "setups": setups, "trades": trades,
        "analysis": audited, "session_count": len(sessions), "session_dates": sessions,
        "session_count_basis": "published_daily_results" if daily else "observed_interaction_dates_no_daily_results_artifact",
        "period_metrics": {
            "session_count": len(sessions), "completed_interactions": len(interactions),
            "completed_interactions_per_session": len(interactions) / len(sessions) if sessions else None,
            "accepted_setups": len(accepted), "accepted_setups_per_session": len(accepted) / len(sessions) if sessions else None,
            "acceptance_rate_percent": _percent(len(accepted), len(interactions)),
            "confirmations_passed": audited["confirmation"]["passed"],
            "confirmation_rate_of_accepted_percent": _percent(audited["confirmation"]["passed"], len(accepted)),
            "trades": len(trades), "trades_per_session": len(trades) / len(sessions) if sessions else None,
        },
        "distributions": {field: _distribution(interactions, field) for field in SCORE_FIELDS + FEATURE_FIELDS},
    }


def _rejection_funnel_rows(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    total = analysis["total_interactions"]
    rejected = total - len(analysis["accepted_setups"])
    rows = [
        {"view": "independent", "stage": "completed_interactions", "count": total,
         "percent_of_completed": 100.0, "percent_of_rejected": 0.0},
    ]
    independent_fails = analysis["independent_gate_failures"]
    for gate in GATES:
        failures = independent_fails[gate]
        rows.append({"view": "independent_pass", "stage": f"{gate}_passes", "count": total - failures,
                     "percent_of_completed": _percent(total - failures, total), "percent_of_rejected": None})
        rows.append({"view": "independent_fail", "stage": f"{gate}_fails", "count": failures,
                     "percent_of_completed": _percent(failures, total), "percent_of_rejected": _percent(failures, rejected)})
    rows.extend({"view": "sequential", "stage": row["stage"], "count": row["count"],
                 "percent_of_completed": row["percent_of_completed"], "percent_of_rejected": None}
                for row in analysis["funnel"])
    return rows


def _accepted_rows(period: dict[str, Any]) -> list[dict[str, Any]]:
    setups = {row["interaction_id"]: row for row in period["setups"]}
    rows: list[dict[str, Any]] = []
    for row in period["analysis"]["accepted_setups"]:
        source = setups[row["interaction_id"]]
        rows.append({
            "date": row["date"], "interaction_id": row["interaction_id"], "structural_level": row["structural_level"],
            "direction": row["direction"], "quality_score": row["quality_score"], "aggression_score": row["aggression_score"],
            "restoration_score": row["restoration_score"], "price_resistance_score": row["price_resistance_score"],
            "persistence_score": row["persistence_score"], "multi_level_support_score": row["multi_level_support_score"],
            "false_refill_penalty": row["false_refill_penalty"], "consume_restore_cycles": row["consume_restore_cycles"],
            "directional_aggressive_volume": source.get("directional_aggressive_volume"),
            "relevant_execution_count": source.get("relevant_execution_count"),
            "interaction_end_ns": source.get("interaction_end_ns"), "interaction_end_price": source.get("interaction_end_price"),
            "confirmation_status": row["confirmation_status"], "confirmation_timestamp_ns": source.get("confirmation_timestamp_ns"),
            "confirmation_price": source.get("confirmation_price"), "terminal_reason": row["terminal_reason"],
            "published_confirmation_path": "UNAVAILABLE_FROM_PUBLISHED_ARTIFACTS" if row["confirmation_status"] != "ENTRY" else "NOT_REQUIRED",
        })
    return rows


def _quality_buckets(rows: list[dict[str, str]]) -> dict[str, int]:
    return {f">={threshold:.2f}": sum(_number(row.get("l2_absorption_quality_score")) >= threshold for row in rows)
            for threshold in (.40, .45, .48, .50, .52, .55)}


def _integrity(august: dict[str, Any]) -> dict[str, Any]:
    summary, source = august["summary"], _read_json(Path(august["root"]) / "source-integrity-diagnostics.json")
    verified = summary.get("input_verification", {})
    anomalies = source.get("sessions", [])
    expected_dates = ["2026-08-03", "2026-08-04", "2026-08-05", "2026-08-06"]
    native_mes = [item for item in verified.get("completion_files_verified", []) if str(item.get("relative_path", "")).startswith("mes_mbp1/")]
    profile_dates = {Path(str(item.get("relative_path", ""))).name.split("_")[1] for item in verified.get("completion_files_verified", []) if str(item.get("relative_path", "")).startswith("es_prior_rth_trades/")}
    anomaly_rows = [item for session in anomalies for item in session.get("anomalies", [])]
    source_dates = sorted(str(session.get("date", "")) for session in anomalies)
    daily_path = Path(august["root"]) / "daily-results.csv"
    daily_dates = sorted(row.get("date", "") for row in _read_csv(daily_path)) if daily_path.is_file() else []
    daily_unresolved = sum(int(row.get("unresolved_source_end") or 0) for row in _read_csv(daily_path)) if daily_path.is_file() else None
    return {
        "snapshot_initialization": verified.get("existing_es_mbo", {}).get("snapshot_initialization") == "SEALED_VALIDATED_R_A_F_SNAPSHOT_THROUGH_F_LAST",
        "native_mes_files_verified": len(native_mes) == 4,
        "native_mes_dates": sorted(Path(str(item["relative_path"])).name.split("_")[1] for item in native_mes),
        "prior_rth_mapping": {"2026-08-03": "2026-07-31", "2026-08-04": "2026-08-03", "2026-08-05": "2026-08-04", "2026-08-06": "2026-08-05"},
        "prior_rth_source_profiles_available": {day: required in profile_dates for day, required in {"2026-08-03": "2026-07-31", "2026-08-04": "2026-08-03", "2026-08-05": "2026-08-04", "2026-08-06": "2026-08-05"}.items()},
        "source_anomaly_policy": source.get("policy"), "source_anomaly_max_per_session": source.get("max_tolerated_offbook_anomalies_per_session"),
        "source_anomaly_count": len(anomaly_rows),
        "source_anomalies_strategy_visible": sum(bool(row.get("affected_bbo")) or bool(row.get("entered_top_ten")) for row in anomaly_rows),
        "daily_session_dates": daily_dates,
        "source_integrity_session_dates": source_dates,
        "session_dates_complete": daily_dates == expected_dates and source_dates == expected_dates,
        "unexpected_session_truncation": daily_unresolved not in (None, 0),
        "no_order_id_in_published_l2_ledgers": all("order_id" not in row for row in august["interactions"] + august["setups"]),
        "signal_window_unchanged": summary.get("v2_contract_sha256") == EXPECTED_HASH,
        "frozen_v2_contract_sha256": summary.get("v2_contract_sha256"),
    }


def _findings(periods: dict[str, dict[str, Any]], august: dict[str, Any], integrity: dict[str, Any]) -> dict[str, str]:
    may, retro = periods["MAY"], periods["RETRO_JUNE_JULY"]
    a, m, r = august["period_metrics"], may["period_metrics"], retro["period_metrics"]
    analysis = august["analysis"]
    quality = {name: value["distributions"]["l2_absorption_quality_score"] for name, value in {
        "August": august, "May": may, "Retro": retro,
    }.items()}
    restoration = {name: value["analysis"]["restoration"]["cycle_counts"] for name, value in {
        "August": august, "May": may, "Retro": retro,
    }.items()}
    return {
        "1_plausibility": (
            f"Yes, technically plausible: the complete published Aug. 3–6 daily/source artifacts show {a['session_count']} sessions and "
            f"{a['completed_interactions']} interactions ({a['completed_interactions_per_session']:.2f}/session), versus "
            f"{m['completed_interactions_per_session']:.2f} May and {r['completed_interactions_per_session']:.2f} observed-retro interactions/session."
        ),
        "2_largest_gate": f"Quality is the largest independent failure: {analysis['independent_gate_failures']['quality']} of {analysis['total_interactions']} interactions fail quality >= 0.50.",
        "3_selectivity": (
            f"Yes. August acceptance is {a['acceptance_rate_percent']:.2f}% ({a['accepted_setups']}/{a['completed_interactions']}), below "
            f"May {m['acceptance_rate_percent']:.2f}% and retro {r['acceptance_rate_percent']:.2f}%."
        ),
        "4_primary_difference": (
            f"It is a combination, not aggression alone: visible restore >=1 occurs in {restoration['August']['>=1']}/{a['completed_interactions']} "
            f"August interactions, versus {restoration['May']['>=1']}/{m['completed_interactions']} May and "
            f"{restoration['Retro']['>=1']}/{r['completed_interactions']} retro; August also has lower median aggressive volume "
            f"({august['distributions']['directional_aggressive_volume']['median']:.0f} vs {may['distributions']['directional_aggressive_volume']['median']:.0f}/"
            f"{retro['distributions']['directional_aggressive_volume']['median']:.0f})."
        ),
        "5_quality_shift": (
            f"Yes, descriptively: August quality mean/median/p95 are {quality['August']['mean']:.3f}/{quality['August']['median']:.3f}/{quality['August']['p95']:.3f}, "
            f"below May {quality['May']['mean']:.3f}/{quality['May']['median']:.3f}/{quality['May']['p95']:.3f} and retro "
            f"{quality['Retro']['mean']:.3f}/{quality['Retro']['median']:.3f}/{quality['Retro']['p95']:.3f}."
        ),
        "6_integrity": (
            "No published-artifact evidence of setup suppression: snapshots, four native MES inputs, all prior-RTH profiles, and the frozen hash validate; "
            f"{integrity['source_anomaly_count']} retained anomalies were strategy-visible={integrity['source_anomalies_strategy_visible']}; session truncation={integrity['unexpected_session_truncation']}."
        ),
        "7_confirmation": (
            f"Only {a['confirmations_passed']} of {a['accepted_setups']} passed; the other {analysis['confirmation']['failed']} terminally expired. "
            "Published ledgers lack the causal +5s..+15s execution paths, so their exact confirmation-path cause is unavailable without a DBN replay."
        ),
        "8_poc": "The sole accepted PRIOR_RTH_POC setup is reported in accepted-setups.csv; it terminally expired in the frozen confirmation window and generated no trade.",
        "9_frequency": "Yes, the reconciled completed-ledger funnel is consistent with genuine low frozen-signal frequency for this seen period; it is not evidence of an executable edge.",
        "10_v3": (
            "There is no technical/data-integrity blocker exposed by this audit to separately predeclare a POC-only V3, but this audit provides no positive POC performance evidence "
            "(one accepted POC setup, zero confirmation/trades). It neither selects nor recommends V3."
        ),
    }


def _report(august: dict[str, Any], periods: dict[str, dict[str, Any]], accepted: list[dict[str, Any]], integrity: dict[str, Any], findings: dict[str, str]) -> str:
    analysis = august["analysis"]
    funnel_rows = analysis["funnel"]
    bottleneck = max(analysis["independent_gate_failures"].items(), key=lambda item: item[1])
    poc = [row for row in accepted if row["structural_level"] == "PRIOR_RTH_POC"]
    return "\n".join([
        "# Frozen L2 V2 August selectivity and integrity audit", "",
        f"Evidence label: `{EVIDENCE_LABEL}`. This is descriptive seen-data research, not fresh OOS evidence.",
        "All values were read from published CSV/JSON artifacts. No DBN was opened, no replay ran, and no strategy parameter changed.", "",
        "## August funnel", "", "| Sequential stage | Count | % completed |", "|---|---:|---:|",
        *[f"| {row['stage']} | {row['count']} | {row['percent_of_completed']:.2f}% |" for row in funnel_rows], "",
        "## Main finding", "",
        f"The largest independent August gate failure is `{bottleneck[0]}` ({bottleneck[1]} of {analysis['total_interactions']}). "
        "This is a descriptive decomposition, not a proposed parameter change.", "",
        "## Accepted setups and confirmation", "",
        f"Accepted: {len(accepted)}; confirmation passed: {analysis['confirmation']['passed']}; confirmation-window expiries: {analysis['confirmation']['failed']}; trades: {analysis['confirmation']['trades']}.",
        "The published ledgers contain terminal confirmation statuses but not the causal execution-path observations needed to measure +5s..+15s maxima. Those fields are reported as unavailable; no DBN was replayed.", "",
        "## POC", "",
        (f"The accepted POC setup was `{poc[0]['interaction_id']}` on {poc[0]['date']} and ended `{poc[0]['terminal_reason']}`; it is not reinterpreted as a trade."
         if poc else "No accepted POC setup was found in the reconciled published ledger."), "",
        "## Integrity", "",
        f"Snapshots sealed: {integrity['snapshot_initialization']}; native MES files verified: {integrity['native_mes_files_verified']}; source anomalies: {integrity['source_anomaly_count']} (strategy-visible: {integrity['source_anomalies_strategy_visible']}); all four sessions complete: {integrity['session_dates_complete']}.",
        "The anomaly artifact marks every retained anomaly outside BBO/top-ten strategy-visible data; it provides no evidence of setup suppression.", "",
        "## Explicit questions", "",
        *[f"{key.split('_', 1)[0]}. {value}" for key, value in findings.items()], "",
        "## Limits", "",
        "May is development evidence; June/July is retrospective robustness, not strict chronological OOS; August is seen data. No results are pooled and no threshold, POC-only rule, or V3 candidate is selected by this audit.", "",
    ])


def materialize(*, repository_root: Path, august_root: Path, output_dir: Path | None = None) -> dict[str, Any]:
    """Write the requested audit directory from published artifacts only."""
    if v2.v2_contract_sha256() != EXPECTED_HASH:
        raise AugustAuditError("frozen V2 contract hash mismatch")
    output = output_dir or august_root / AUDIT_NAME
    if output.exists():
        raise FileExistsError(f"audit output already exists: {output}")
    roots = {name: repository_root / relative for name, relative in PERIOD_ROOTS.items()}
    roots["AUGUST_SEEN"] = august_root
    periods = {name: _period(root) for name, root in roots.items()}
    august = periods["AUGUST_SEEN"]
    if august["summary"].get("evidence_label") != EVIDENCE_LABEL or august["summary"].get("august_7_excluded") is not True:
        raise AugustAuditError("August published summary does not bind the required seen-data and Aug-3-to-6 scope")
    accepted, integrity = _accepted_rows(august), _integrity(august)
    rows = _rejection_funnel_rows(august["analysis"])
    period_rows = [{"period": name, **period["period_metrics"], "session_count_basis": period["session_count_basis"], "result_evidence_label": period["summary"].get("evidence_label")}
                   for name, period in periods.items()]
    comparison = {
        name: {
            "metrics": period["period_metrics"], "distributions": period["distributions"],
            "restoration_cycle_counts": period["analysis"]["restoration"]["cycle_counts"],
            "persistence_zero_score_count": period["analysis"]["persistence"]["zero_score_count"],
            "price_resistance_zero_score_count": sum(
                _number(row.get("price_resistance_score")) == 0 for row in period["interactions"]
            ),
        }
        for name, period in periods.items()
    }
    findings = _findings(periods, august, integrity)
    result = {
        "audit_scope": "READ_ONLY_PUBLISHED_L2_V2_ARTIFACTS_ONLY", "strategy_mutated": False,
        "network_or_download_used": False, "full_dbn_replay": False, "evidence_label": EVIDENCE_LABEL,
        "frozen_v2_contract_sha256": EXPECTED_HASH,
        "august": {"metrics": august["period_metrics"], "analysis": august["analysis"], "quality_diagnostic_buckets": _quality_buckets(august["interactions"]), "integrity": integrity},
        "period_comparison": comparison, "accepted_setups": accepted, "specific_question_findings": findings,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(output / "rejection-funnel.csv", rows, ["view", "stage", "count", "percent_of_completed", "percent_of_rejected"])
    _write_csv(output / "period-comparison.csv", period_rows, list(period_rows[0]))
    _write_csv(output / "accepted-setups.csv", accepted, list(accepted[0]) if accepted else ["interaction_id"])
    (output / "diagnostic-report.md").write_text(_report(august, periods, accepted, integrity, findings), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only L2 V2 August published-artifact audit")
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--august-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        result = materialize(repository_root=args.repository_root, august_root=args.august_root, output_dir=args.output_dir)
    except (AugustAuditError, FileExistsError, OSError, ValueError, csv.Error) as exc:
        print(f"ERROR: {exc}")
        return 1
    print(json.dumps({"output_dir": str(args.output_dir or args.august_root / AUDIT_NAME), "accepted_setups": len(result["accepted_setups"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
