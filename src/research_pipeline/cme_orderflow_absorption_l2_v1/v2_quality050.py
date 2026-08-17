"""Frozen L2 V2 quality-0.50 variant and ledger-only selectivity check.

The selectivity path reads only published interaction feature rows.  It cannot
open a DBN, inspect a trade ledger, or use a strategy outcome.  The optional
future replay entry point delegates to the existing L2 engine with this one
explicit configuration difference; it is never invoked by this module's
selectivity command.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from .model import L2Config
from .rejection_funnel import gate_status


STRATEGY_ID = "CMEOrderflowAbsorption.ES_L2_V2"
VARIANT_LABEL = "L2_V2_MAY_DEVELOPMENT_QUALITY_0_50"
EVIDENCE_LABEL = "MAY_DEVELOPMENT_SELECTIVITY_ONLY_NOT_OOS_EVIDENCE"
V1_CONFIG = L2Config()
V2_CONFIG = replace(V1_CONFIG, min_quality_score=0.50)


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def v1_v2_config_diff() -> dict[str, Any]:
    v1, v2 = asdict(V1_CONFIG), asdict(V2_CONFIG)
    changed = [{"field": name, "v1": v1[name], "v2": v2[name]} for name in sorted(v1) if v1[name] != v2[name]]
    if changed != [{"field": "min_quality_score", "v1": 0.55, "v2": 0.50}]:
        raise RuntimeError("L2 V2 configuration must differ from L2 V1 only at min_quality_score")
    return {
        "comparison_scope": "strategy_configuration_fields_only",
        "v1_config_sha256": _canonical_hash(v1),
        "v2_config_sha256": _canonical_hash(v2),
        "changed_strategy_fields": changed,
        "unchanged_strategy_fields": [name for name in sorted(v1) if name != "min_quality_score"],
        "only_min_quality_score_differs": True,
    }


def v2_contract() -> dict[str, Any]:
    """The deterministic V2 contract; all non-config semantics are frozen literals."""
    return {
        "strategy_id": STRATEGY_ID,
        "parent_strategy_id": "CMEOrderflowAbsorption.ES_L2_V1",
        "variant_label": VARIANT_LABEL,
        "evidence_label": EVIDENCE_LABEL,
        "configuration": asdict(V2_CONFIG),
        "execution": {
            "confirmation_window_seconds_inclusive": [5.0, 15.0],
            "confirmation_favorable_ticks": 3,
            "entry_latency_ms": 2.0,
            "stop_buffer_ticks": 5,
            "target_r": 3.0,
            "risk_budget_usd": 250.0,
            "es_first": True,
            "mes_fallback": True,
            "max_es_contracts": 6,
            "max_mes_contracts": 60,
        },
        "strategy_configuration_diff": v1_v2_config_diff(),
        "development_only_parameters": ["min_quality_score=0.50 derived from May selectivity diagnostics"],
        "selection_prohibited": True,
        "outcome_parameter_selection": False,
    }


def v2_contract_sha256() -> str:
    return _canonical_hash(v2_contract())


def _read_feature_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "interaction_id" not in rows[0]:
        raise ValueError("published interaction-features.csv is missing or malformed")
    return rows


def _stage_counts(rows: list[dict[str, str]], config: L2Config) -> dict[str, int]:
    survivors = rows
    result = {"completed_interactions": len(rows)}
    for label, gate in (
        ("would_pass_aggressive_volume", "relevant_aggressive_volume"),
        ("would_pass_execution_count", "relevant_execution_count"),
        ("would_pass_consume_restore", "consume_restore"),
        ("would_pass_rejection", "rejection"),
        ("would_pass_quality", "quality"),
    ):
        survivors = [row for row in survivors if gate_status(row, config)[gate]]
        result[label] = len(survivors)
    result["final_accepted_v2_setups"] = len(survivors)
    return result


def selectivity_check(feature_rows: list[dict[str, str]]) -> dict[str, Any]:
    """Re-evaluate only qualification fields; no setup/trade outcome field is read."""
    v1_counts = _stage_counts(feature_rows, V1_CONFIG)
    v2_counts = _stage_counts(feature_rows, V2_CONFIG)
    if v1_counts["final_accepted_v2_setups"] != 8:
        raise ValueError("published V1 feature ledger does not reproduce the sealed eight V1 accepted setups")
    accepted = v2_counts["final_accepted_v2_setups"]
    guard = {"minimum_inclusive": 15, "maximum_inclusive": 150,
             "passes": 15 <= accepted <= 150,
             "status": "GO_PREPARE_REPLAY_COMMAND" if 15 <= accepted <= 150 else "NO_GO_DO_NOT_REPLAY"}
    return {
        "strategy_id": STRATEGY_ID,
        "variant_label": VARIANT_LABEL,
        "evidence_label": EVIDENCE_LABEL,
        "read_only_feature_ledger": True,
        "pnl_or_trade_outcomes_used": False,
        "v1_recomputed_counts": v1_counts,
        "v2_selectivity_counts": v2_counts,
        "accepted_setup_delta": accepted - v1_counts["final_accepted_v2_setups"],
        "go_no_go_guard": guard,
        "v2_contract_sha256": v2_contract_sha256(),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["variant", "completed_interactions", "would_pass_aggressive_volume", "would_pass_execution_count",
              "would_pass_consume_restore", "would_pass_rejection", "would_pass_quality", "final_accepted_setups"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def materialize_selectivity_check(v1_artifact_root: Path, output_root: Path) -> dict[str, Any]:
    """Write a new V2 development selectivity artifact without reading a DBN."""
    output = output_root / "selectivity_check"
    if output.exists():
        raise FileExistsError(f"immutable selectivity output already exists: {output}")
    rows = _read_feature_rows(v1_artifact_root / "interaction-features.csv")
    summary = selectivity_check(rows)
    diff = v1_v2_config_diff()
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "v1-v2-contract-diff.json").write_text(json.dumps({
        **diff, "v1_strategy_id": "CMEOrderflowAbsorption.ES_L2_V1", "v2_strategy_id": STRATEGY_ID,
        "v2_contract_sha256": v2_contract_sha256(),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(output / "accepted-setup-counts.csv", [
        {"variant": "L2_V1_QUALITY_0_55", **{key: value for key, value in summary["v1_recomputed_counts"].items() if key != "final_accepted_v2_setups"},
         "final_accepted_setups": summary["v1_recomputed_counts"]["final_accepted_v2_setups"]},
        {"variant": VARIANT_LABEL, **{key: value for key, value in summary["v2_selectivity_counts"].items() if key != "final_accepted_v2_setups"},
         "final_accepted_setups": summary["v2_selectivity_counts"]["final_accepted_v2_setups"]},
    ])
    report = "\n".join([
        "# L2 V2 May development selectivity check", "",
        f"Variant: `{VARIANT_LABEL}`", "",
        "This is a feature-ledger-only development diagnostic. It reads no DBN and no trade/PnL outcome fields.", "",
        f"V1 recomputed accepted setups: {summary['v1_recomputed_counts']['final_accepted_v2_setups']}",
        f"V2 quality-0.50 accepted setups: {summary['v2_selectivity_counts']['final_accepted_v2_setups']}",
        f"Guard: `{summary['go_no_go_guard']['status']}`", "",
        "The only strategy-configuration difference is `min_quality_score: 0.55 -> 0.50`; see `v1-v2-contract-diff.json`.",
        "No PnL or outcome-based parameter selection occurred.", "",
    ])
    (output / "diagnostic-report.md").write_text(report, encoding="utf-8")
    return summary


def run_v2_may_replay(*, data_root: Path, output_dir: Path) -> dict[str, Any]:
    """Future explicit V2 replay entry point. The caller, not this task, authorizes execution."""
    from .historical_runner import run_first_broad_may_2026
    return run_first_broad_may_2026(
        data_root=data_root, output_dir=output_dir, config=V2_CONFIG, strategy_id=STRATEGY_ID,
        evidence_label="MAY_DEVELOPMENT_V2_QUALITY_0_50_NOT_OOS_EVIDENCE",
        run_label=VARIANT_LABEL, contract=v2_contract(),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="L2 V2 quality-0.50 selectivity and explicit replay entry point")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--selectivity-check", action="store_true")
    group.add_argument("--replay-may-2026", action="store_true")
    parser.add_argument("--v1-artifact-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.selectivity_check:
            if args.v1_artifact_root is None:
                raise ValueError("--v1-artifact-root is required for --selectivity-check")
            result = materialize_selectivity_check(args.v1_artifact_root, args.output_root)
        else:
            if args.data_root is None:
                raise ValueError("--data-root is required for --replay-may-2026")
            result = run_v2_may_replay(data_root=args.data_root, output_dir=args.output_root)
        print(json.dumps(result, indent=2, sort_keys=True))
    except (FileExistsError, FileNotFoundError, ValueError, OSError, csv.Error) as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
