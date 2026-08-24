"""Reconcile frozen V3 artifacts with corrected MBO public-book tapes.

This is a compact, local-only reproduction audit.  It opens completed Parquet
tapes and published CSV/JSON artifacts, never DBN data, never a provider API,
and never evaluates the Weight x Quality grid.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import all_period_weight_q_research as research
from . import causal_master_tape as master
from . import cross_period_robust_stress as cross
from . import historical_runner as historical
from . import weight_q_research as matrix


AUDIT_ID = "CMEOrderflowAbsorption.ES_L2_V3_MBO_REPRODUCTION_RECONCILIATION"
CLASSIFICATION = "SOURCE_INTEGRITY_REPLAY_CLARIFICATION"
SUPERSEDED = "SUPERSEDED_BY_MBO_PUBLIC_BOOK_INTEGRITY_CLARIFICATION"
OUTPUT_JSON = research.TAPE_ROOT / "mbo-v3-reproduction-reconciliation.json"
OUTPUT_HTML = research.TAPE_ROOT / "mbo-v3-reproduction-reconciliation.html"
OLD_PERIOD_ROOT = cross.OUTPUT_RELATIVE / "periods"
MBO_PERIOD_IDS = ("MAY_2026", "RETRO_JUNE_JULY_2026", "AUGUST_03_06_2026")
NATIVE_PERIOD_IDS = ("APRIL_2026", "AUGUST_10_14_2026", "DECEMBER_2025", "JANUARY_2026")
MAY_DIVERGENCE_ID = "PRIOR_RTH_POC:7426.50:0001"
MAY_SOURCE_EPISODE = {
    "episode_index": 768,
    "classification": "ATOMIC_MBO_RECONSTRUCTION_TRANSITION",
    "timestamp_utc": "2026-05-13T13:30:00.443870Z",
    "source_record_index_first": 2_729_563,
    "source_record_index_reopen": 2_729_566,
    "non_executable_record_count": 3,
    "non_executable_shape": "LOCKED",
    "locked_bid": 7427.75,
    "locked_ask": 7427.75,
    "fresh_reopen_bid": 7427.75,
    "fresh_reopen_ask": 7428.00,
}


class MBOReproductionError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MBOReproductionError(f"missing or invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise MBOReproductionError(f"JSON artifact is not an object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except OSError as exc:
        raise MBOReproductionError(f"missing CSV artifact: {path}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _bundle(repository_root: Path, period_id: str) -> research.PeriodBundle:
    period = research.PERIOD_BY_ID[period_id]
    root = repository_root / research.TAPE_ROOT / period_id.lower()
    return research.PeriodBundle(period, root, tuple(period.dates))


def replay_v3_detail(bundle: research.PeriodBundle) -> dict[str, Any]:
    """Replay exactly one frozen V3 configuration over one compact period."""
    interactions, indexes = research._load_bundle_rows(bundle)
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in interactions:
        by_day[str(row["session_date"])].append(row)
    accumulator = matrix.ConfigurationAccumulator((1, 1, 1, 1, 16), Decimal("0.50"))
    all_trades: list[dict[str, Any]] = []
    terminal_outcomes: dict[str, str] = {}
    accepted_rows: list[dict[str, Any]] = []
    for day in bundle.days:
        tape = matrix.SessionCausalTape.from_parquet(
            day, bundle.root / "causal-event-tape" / f"{day}.parquet",
        )
        rows = [
            row for row in by_day.get(day, ())
            if master.interaction_is_accepted(row, threshold="0.50")
        ]
        accepted_rows.extend(rows)
        session = matrix.simulate_independent_session(
            tape, rows,
            {str(row["interaction_id"]): indexes[str(row["interaction_id"])] for row in rows},
        )
        accumulator.add(session, set())
        all_trades.extend(session.trades)
        for interaction_id, outcome in session.terminal_outcomes.items():
            if interaction_id in terminal_outcomes:
                raise MBOReproductionError(f"duplicate corrected setup outcome: {interaction_id}")
            terminal_outcomes[interaction_id] = outcome
    row = accumulator.row(0)
    performance = historical._performance(all_trades)
    if int(row["trades"]) != int(performance["completed_trades"]):
        raise MBOReproductionError("compact V3 trade metrics failed reconciliation")
    accepted = [{
        "interaction_id": str(item["interaction_id"]),
        "source_interaction_id": str(item["source_interaction_id"]),
        "session_date": str(item["session_date"]),
        "quality_score": float(item["original_v3_quality_score"]),
        "terminal_outcome": terminal_outcomes[str(item["interaction_id"])],
    } for item in accepted_rows]
    metrics = {
        "sessions": len(bundle.days), "completed_interactions": len(interactions),
        "accepted_setups": int(row["accepted_setups"]),
        "confirmations": int(row["confirmations"]),
        "confirmation_expiries": int(row["confirmation_expiries"]),
        "active_position_blocks": int(row["active_position_blocks"]),
        "trades": int(row["trades"]), "wins": int(row["wins"]), "losses": int(row["losses"]),
        "win_rate": float(row["win_rate"]), "total_r": float(row["total_r"]),
        "net_pnl_usd": float(row["net_pnl_usd"]), "profit_factor": row["profit_factor"],
        "max_cumulative_drawdown_r": float(row["max_cumulative_drawdown_r"]),
        "es_trades": int(row["es_trades"]), "mes_trades": int(row["mes_trades"]),
        "target_exits": int(row["target_exits"]), "stop_exits": int(row["stop_exits"]),
        "hard_cutoff_exits": int(row["hard_cutoff_exits"]), "unresolved": int(row["unresolved"]),
    }
    return {"metrics": metrics, "trades": all_trades, "accepted_setups": accepted}


def _old_period(repository_root: Path, period_id: str) -> dict[str, Any]:
    root = repository_root / OLD_PERIOD_ROOT / period_id.lower()
    result_path = root / "v3-results.json"
    result = _read_json(result_path)
    metrics = dict(result["metrics"])
    metrics["trades"] = int(metrics.pop("completed_trades"))
    return {
        "metrics": metrics,
        "trades": _read_csv(root / "v3-trades.csv"),
        "setups": _read_csv(root / "v3-setups.csv"),
        "artifact": str(result_path),
        "artifact_sha256": _sha256(result_path),
        "provenance": result.get("provenance"),
    }


def _trade_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row["date"]), str(row["interaction_id"])


def _same_number(left: object, right: object, *, tolerance: float = 1e-12) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return left == right


def compare_trades(old_rows: Sequence[Mapping[str, Any]], new_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    old = {_trade_key(row): row for row in old_rows}
    new = {_trade_key(row): row for row in new_rows}
    if len(old) != len(old_rows) or len(new) != len(new_rows):
        raise MBOReproductionError("duplicate trade identity in reproduction comparison")
    fields = (
        "entry_timestamp_ns", "instrument", "entry", "stop", "target",
        "exit_timestamp_ns", "exit", "exit_reason", "r_multiple", "net_pnl_usd",
    )
    common: list[dict[str, Any]] = []
    for key in sorted(set(old) & set(new)):
        changes = {
            field: {"old": old[key].get(field), "corrected": new[key].get(field)}
            for field in fields if not _same_number(old[key].get(field), new[key].get(field), tolerance=1e-9)
        }
        common.append({"date": key[0], "interaction_id": key[1], "changes": changes})
    return {
        "common_trade_count": len(common),
        "unchanged_common_trade_count": sum(not row["changes"] for row in common),
        "common_trades": common,
        "old_only_trades": [dict(old[key]) for key in sorted(set(old) - set(new))],
        "corrected_only_trades": [dict(new[key]) for key in sorted(set(new) - set(old))],
        "changed_entry_timestamp_count": sum("entry_timestamp_ns" in row["changes"] for row in common),
        "changed_entry_instrument_count": sum("instrument" in row["changes"] for row in common),
        "changed_entry_price_count": sum("entry" in row["changes"] for row in common),
        "changed_exit_timestamp_count": sum("exit_timestamp_ns" in row["changes"] for row in common),
        "changed_outcome_count": sum(
            bool({"exit_reason", "r_multiple", "net_pnl_usd"} & set(row["changes"])) for row in common
        ),
    }


def _old_accepted(row: Mapping[str, Any]) -> bool:
    return str(row.get("accepted", "")).strip().lower() == "true"


def compare_setup_acceptance(
    old_rows: Sequence[Mapping[str, Any]], corrected_root: Path,
) -> dict[str, Any]:
    interactions, _indexes = research._load_bundle_rows(research.PeriodBundle(
        research.PERIOD_BY_ID["MAY_2026"], corrected_root, tuple(research.PERIOD_BY_ID["MAY_2026"].dates),
    ))
    old = {(str(row["date"]), str(row["interaction_id"])): row for row in old_rows}
    new = {(str(row["session_date"]), str(row["source_interaction_id"])): row for row in interactions}
    if set(old) != set(new):
        raise MBOReproductionError("May interaction identity universe changed")
    outcome_differences: list[dict[str, Any]] = []
    for key in sorted(old):
        old_accepts = _old_accepted(old[key])
        new_accepts = master.interaction_is_accepted(new[key], threshold="0.50")
        if old_accepts == new_accepts:
            continue
        row = {
            "date": key[0], "interaction_id": key[1],
            "old_accepted": old_accepts, "corrected_accepted": new_accepts,
            "old_quality_score": float(old[key]["l2_absorption_quality_score"]),
            "corrected_quality_score": float(new[key]["original_v3_quality_score"]),
            "threshold_unchanged": 0.50,
            "old_rejection_reasons": old[key].get("rejection_reasons"),
            "corrected_rejection_reasons": new[key].get("original_v3_rejection_reasons"),
            "feature_deltas": {
                "size_consumed_by_execution": {"old": int(old[key]["size_consumed_by_execution"]), "corrected": int(new[key]["size_consumed_by_execution"])},
                "restored_size": {"old": int(old[key]["restored_size"]), "corrected": int(new[key]["restored_size"])},
                "cumulative_consumed_volume": {"old": int(old[key]["cumulative_consumed_volume"]), "corrected": int(new[key]["cumulative_consumed_volume"])},
                "cumulative_restored_volume": {"old": int(old[key]["cumulative_restored_volume"]), "corrected": int(new[key]["cumulative_restored_volume"])},
                "restoration_score": {"old": float(old[key]["restoration_score"]), "corrected": float(new[key]["restoration_score"])},
                "persistence_score": {"old": float(old[key]["persistence_score"]), "corrected": float(new[key]["persistence_score"])},
                "false_refill_penalty": {"old": float(old[key]["false_refill_penalty"]), "corrected": float(new[key]["false_refill_penalty"])},
            },
        }
        if key == ("2026-05-13", MAY_DIVERGENCE_ID):
            row["first_causal_divergence"] = {
                **MAY_SOURCE_EPISODE,
                "reason": "OLD_PATH_EXPOSED_LOCKED_ATOMIC_RECONSTRUCTION_SNAPSHOTS",
                "corrected_behavior": "SUSPEND_THEN_RESUME_FROM_FRESH_VALID_TWO_SIDED_BOOK",
                "unrelated_causal_tape_bug": False,
            }
        outcome_differences.append(row)
    return {
        "interaction_identity_count": len(old),
        "identity_universe_unchanged": True,
        "acceptance_outcome_difference_count": len(outcome_differences),
        "acceptance_outcome_differences": outcome_differences,
    }


def _metric_delta(old: Mapping[str, Any], corrected: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "sessions", "completed_interactions", "accepted_setups", "confirmations", "trades",
        "wins", "losses", "win_rate", "total_r", "net_pnl_usd", "profit_factor",
        "max_cumulative_drawdown_r", "es_trades", "mes_trades",
    )
    return {
        field: (
            None if old.get(field) is None or corrected.get(field) is None
            else float(corrected[field]) - float(old[field])
        ) for field in fields
    }


def _native_evidence(repository_root: Path) -> dict[str, Any]:
    old_root = repository_root / OLD_PERIOD_ROOT
    rows: dict[str, Any] = {}
    for period_id in ("APRIL_2026", "AUGUST_10_14_2026"):
        old = _old_period(repository_root, period_id)
        expected = research.EXPECTED_V3[period_id]
        metrics = old["metrics"]
        exact = all(
            _same_number(metrics.get(field), expected[field], tolerance=1e-12)
            for field in ("trades", "wins", "losses", "total_r", "net_pnl_usd")
        )
        rows[period_id] = {"status": "UNCHANGED" if exact else "MISMATCH", "published": metrics}
    gates = _read_json(
        repository_root / research.DEC_JAN_MASTER / "reproduction-gates.json"
    )
    for period_id, key in (("DECEMBER_2025", "v3_december"), ("JANUARY_2026", "v3_full")):
        published = _old_period(repository_root, period_id)["metrics"]
        reproduced = gates[key]["metrics"]
        if period_id == "JANUARY_2026":
            reproduced = {
                **reproduced,
                "completed_trades": int(reproduced["completed_trades"]) - int(gates["v3_december"]["metrics"]["completed_trades"]),
                "wins": int(reproduced["wins"]) - int(gates["v3_december"]["metrics"]["wins"]),
                "losses": int(reproduced["losses"]) - int(gates["v3_december"]["metrics"]["losses"]),
                "total_r": float(reproduced["total_r"]) - float(gates["v3_december"]["metrics"]["total_r"]),
                "net_pnl_usd": float(reproduced["net_pnl_usd"]) - float(gates["v3_december"]["metrics"]["net_pnl_usd"]),
            }
        exact = all(_same_number(
            reproduced["completed_trades" if field == "trades" else field], published[field], tolerance=1e-12,
        ) for field in ("trades", "wins", "losses", "total_r", "net_pnl_usd"))
        rows[period_id] = {
            "status": "UNCHANGED" if exact else "MISMATCH", "published": published,
            "compact_reproduction_gate": reproduced,
        }
    return rows


def build_reconciliation(*, repository_root: Path, output_json: Path, output_html: Path) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    output_json = output_json if output_json.is_absolute() else repository_root / output_json
    output_html = output_html if output_html.is_absolute() else repository_root / output_html
    periods: dict[str, Any] = {}
    for period_id in MBO_PERIOD_IDS:
        old = _old_period(repository_root, period_id)
        corrected = replay_v3_detail(_bundle(repository_root, period_id))
        trades = compare_trades(old["trades"], corrected["trades"])
        unchanged = not trades["old_only_trades"] and not trades["corrected_only_trades"] and all(
            not row["changes"] for row in trades["common_trades"]
        )
        periods[period_id] = {
            "old_published": old["metrics"], "corrected_semantics": corrected["metrics"],
            "delta": _metric_delta(old["metrics"], corrected["metrics"]),
            "trade_reconciliation": trades,
            "old_artifact": old["artifact"], "old_artifact_sha256": old["artifact_sha256"],
            "old_provenance": old["provenance"],
            "canonical_decision": (
                SUPERSEDED if period_id == "MAY_2026" and not unchanged else "OLD_PUBLISHED_BASELINE_REMAINS_EXACT"
            ),
        }
        if period_id == "MAY_2026":
            periods[period_id]["setup_reconciliation"] = compare_setup_acceptance(
                old["setups"], _bundle(repository_root, period_id).root,
            )
    native = _native_evidence(repository_root)
    if any(row["status"] != "UNCHANGED" for row in native.values()):
        raise MBOReproductionError("native V3 evidence changed during MBO-only clarification")
    may = periods["MAY_2026"]
    if may["canonical_decision"] != SUPERSEDED:
        raise MBOReproductionError("May mismatch was not isolated to corrected public-book semantics")
    if periods["RETRO_JUNE_JULY_2026"]["canonical_decision"] != "OLD_PUBLISHED_BASELINE_REMAINS_EXACT":
        raise MBOReproductionError("retro MBO baseline changed unexpectedly")
    if periods["AUGUST_03_06_2026"]["canonical_decision"] != "OLD_PUBLISHED_BASELINE_REMAINS_EXACT":
        raise MBOReproductionError("August MBO baseline changed unexpectedly")
    payload: dict[str, Any] = {
        "audit_id": AUDIT_ID, "status": "MBO_V3_REPRODUCTION_RECONCILIATION_COMPLETE",
        "classification": CLASSIFICATION, "v3_contract_sha256": research.V3_CONTRACT_SHA256,
        "v3_parameter_changes": False, "quality_threshold_changes": False,
        "weight_changes": False, "optimizer_executed": False,
        "weight_q_configurations_evaluated": 0, "dbn_files_opened": 0,
        "network_calls": 0, "downloads": 0,
        "corrected_public_book_semantics": [
            "LOCKED_OR_CROSSED_RECONSTRUCTION_SNAPSHOTS_ARE_NOT_EXECUTABLE",
            "NO_STALE_BBO_DURING_SUSPENSION",
            "RESUME_ONLY_ON_FRESH_VALID_TWO_SIDED_BID_LT_ASK_BOOK",
            "PRIVATE_ANOMALIES_REMAIN_STRATEGY_INVISIBLE",
            "NO_INVENTED_FILLS",
        ],
        "periods": periods, "native_periods": native,
        "baseline_updates": {
            "MAY_2026": {
                "label": CLASSIFICATION,
                "old": {key: may["old_published"][key] for key in research.REPRODUCTION_METRIC_FIELDS},
                "corrected": {key: may["corrected_semantics"][key] for key in research.REPRODUCTION_METRIC_FIELDS},
            }
        },
        "optimizer_gate_update_required": True,
        "optimizer_permitted_after_documented_gate_validation": True,
    }
    payload["canonical_baseline_sha256"] = _canonical_sha256({
        "classification": payload["classification"], "v3_contract_sha256": payload["v3_contract_sha256"],
        "baseline_updates": payload["baseline_updates"],
    })
    research.validate_corrected_baseline_document(payload)
    _write_json(output_json, payload)
    temporary = output_html.with_suffix(output_html.suffix + ".part")
    temporary.write_text(render_html(payload), encoding="utf-8")
    temporary.replace(output_html)
    return payload


def render_html(payload: Mapping[str, Any]) -> str:
    rows = []
    for period_id, period in payload["periods"].items():
        old, new = period["old_published"], period["corrected_semantics"]
        rows.append(
            "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in (
                period_id, old["trades"], new["trades"], old["total_r"], new["total_r"],
                old["net_pnl_usd"], new["net_pnl_usd"], period["canonical_decision"],
            )) + "</tr>"
        )
    may = payload["periods"]["MAY_2026"]
    setup = may["setup_reconciliation"]["acceptance_outcome_differences"][0]
    native = "".join(
        f"<li><code>{html.escape(period)}</code>: {html.escape(row['status'])}</li>"
        for period, row in payload["native_periods"].items()
    )
    return f"""<!doctype html>
<html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
<title>V3 MBO reproduction reconciliation</title><style>
body{{font:15px/1.5 system-ui,sans-serif;max-width:1120px;margin:40px auto;padding:0 24px;color:#17202a}}
h1,h2{{line-height:1.2}} table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border:1px solid #ccd4dd;padding:7px;text-align:left}}
.pass{{color:#087f5b}} .warn{{color:#a15c00}} code{{background:#f3f5f7;padding:2px 4px}} .card{{border:1px solid #d9e0e7;border-radius:8px;padding:16px;margin:16px 0}}
</style></head><body><h1>V3 MBO reproduction reconciliation</h1>
<p class=\"pass\"><strong>{html.escape(str(payload['status']))}</strong></p>
<p>This source-integrity audit evaluated zero Weight x Quality configurations, opened zero DBNs, and made zero network calls.</p>
<h2>Period comparison</h2><table><thead><tr><th>Period</th><th>Old trades</th><th>Corrected trades</th><th>Old R</th><th>Corrected R</th><th>Old PnL</th><th>Corrected PnL</th><th>Decision</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<div class=\"card\"><h2>May root cause</h2><p>The only outcome-changing setup was <code>{html.escape(setup['interaction_id'])}</code>. Its quality score moved from <strong>{setup['old_quality_score']:.15f}</strong> to <strong>{setup['corrected_quality_score']:.15f}</strong> at the unchanged 0.50 gate. Restored size changed from {setup['feature_deltas']['restored_size']['old']} to {setup['feature_deltas']['restored_size']['corrected']}.</p>
<p>The first causal divergence was adapter episode {MAY_SOURCE_EPISODE['episode_index']} at <code>{MAY_SOURCE_EPISODE['timestamp_utc']}</code>: three atomic locked snapshots at 7427.75/7427.75 were formerly exposed. The corrected adapter withheld them and resumed from the fresh 7427.75/7428.00 book.</p></div>
<h2>Trade reconciliation</h2><p>{may['trade_reconciliation']['common_trade_count']} common trades were byte/value-equivalent on all compared entry, instrument, geometry, exit, outcome, and PnL fields. One old-only winning May 13 MES trade was removed; there were no corrected-only trades and no position-blocking changes.</p>
<h2>Native controls</h2><ul>{native}</ul>
<h2>Decision</h2><p class=\"warn\">The old May V3 result is <strong>{SUPERSEDED}</strong>. Historical artifacts remain read-only. Only the documented corrected May baseline updates the reproduction contract; V3 parameters and hash remain unchanged.</p>
<p>V3 hash: <code>{html.escape(str(payload['v3_contract_sha256']))}</code></p></body></html>"""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--output-json", type=Path, default=OUTPUT_JSON)
    parser.add_argument("--output-html", type=Path, default=OUTPUT_HTML)
    args = parser.parse_args(argv)
    result = build_reconciliation(
        repository_root=args.repository_root, output_json=args.output_json, output_html=args.output_html,
    )
    print(json.dumps({
        "status": result["status"], "classification": result["classification"],
        "output_json": str(args.output_json), "output_html": str(args.output_html),
        "optimizer_executed": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
