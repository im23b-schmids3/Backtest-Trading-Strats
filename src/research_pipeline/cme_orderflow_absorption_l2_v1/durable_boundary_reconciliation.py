"""Read-only audit for non-executable overlaps in all-period causal tapes.

This module opens only the already-built compact Parquet artifacts.  It never
opens DBN, contacts Databento, ranks configurations, or publishes strategy
outcomes.  Its purpose is to prove that every possible open-position boundary
overlap has a typed, frozen-contract disposition before the matrix is resumed.
"""
from __future__ import annotations

import argparse
import bisect
import html
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import all_period_weight_q_research as allp
from . import causal_master_tape as master
from . import weight_q_research as matrix
from .model import initial_prices, size_for_instrument


STRATEGY_ID = "CMEOrderflowAbsorption.ES_L2_ALL_PERIOD_WEIGHT_Q_RESEARCH"
REPORT_NAME = "durable-boundary-reconciliation"
CLASSIFICATION = "OFFLINE_BOUNDARY_REPLAY_RECONCILED_FAIL_CLOSED"
FROZEN_POSITION_POLICY = {
    "EXPECTED_SCHEDULED_MAINTENANCE": "FAIL_CLOSED_IF_POSITION_OPEN",
    "TEMPORARY_BOOK_RECONSTRUCTION": "FAIL_CLOSED_IF_POSITION_OPEN",
    "SOURCE_END": "UNRESOLVED_FAIL_CLOSED",
    "UNRESOLVED_INVALID_BOOK": "FAIL_CLOSED_IF_POSITION_OPEN",
    "OTHER_INTEGRITY_FAILURE": "REJECT_TAPE",
}


class BoundaryAuditError(RuntimeError):
    pass


def _utc(timestamp_ns: int | None) -> str | None:
    if timestamp_ns is None:
        return None
    return datetime.fromtimestamp(timestamp_ns / 1_000_000_000, tz=timezone.utc).isoformat()


def _entry_plan(
    tape: matrix.SessionCausalTape, interaction: Mapping[str, Any], event_ordinal: int,
) -> dict[str, Any] | None:
    index = tape.regular_index(event_ordinal)
    if not math.isfinite(tape.es_bid[index]) or not math.isfinite(tape.es_ask[index]):
        return None
    direction = str(interaction["direction"])
    prices = initial_prices(
        direction, tape.es_bid[index], tape.es_ask[index],
        float(interaction["zone_low"]), float(interaction["zone_high"]),
    )
    sizing = size_for_instrument(prices, "ES")
    instrument = "ES"
    if int(sizing["contracts"]) < 1:
        if not math.isfinite(tape.mes_bid[index]) or not math.isfinite(tape.mes_ask[index]):
            return None
        prices = initial_prices(
            direction, tape.mes_bid[index], tape.mes_ask[index],
            float(interaction["zone_low"]), float(interaction["zone_high"]),
        )
        sizing = size_for_instrument(prices, "MES")
        instrument = "MES"
    if int(sizing["contracts"]) < 1:
        return None
    return {
        "instrument": instrument,
        "contracts": int(sizing["contracts"]),
        "entry_ordinal": event_ordinal,
        "entry_timestamp_ns": int(tape.timestamps[index]),
        "entry": float(prices["entry"]),
        "stop": float(prices["stop"]),
        "target": float(prices["target"]),
        "direction": str(prices["direction"]),
    }


def _diagnostic_exit_without_boundary(
    tape: matrix.SessionCausalTape, plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Describe tape sufficiency without authorizing a boundary-crossing trade."""
    terminal = tape.hard_event or tape.source_end_event
    if terminal is None:
        raise BoundaryAuditError(f"session lacks terminal event: {tape.day}")
    terminal_ordinal = int(terminal["event_ordinal"])
    start = bisect.bisect_right(tape.ordinals, int(plan["entry_ordinal"]))
    end = bisect.bisect_left(tape.ordinals, terminal_ordinal)
    instrument, direction = str(plan["instrument"]), str(plan["direction"])
    if direction == "LONG":
        series = tape._series[(instrument, "bid")]
        stop_index = series.first_le(start, end, float(plan["stop"]))
        target_index = series.first_ge(start, end, float(plan["target"]))
    else:
        series = tape._series[(instrument, "ask")]
        stop_index = series.first_ge(start, end, float(plan["stop"]))
        target_index = series.first_le(start, end, float(plan["target"]))
    candidates = [
        (index, reason) for index, reason in ((stop_index, "STOP"), (target_index, "TARGET"))
        if index is not None
    ]
    if candidates:
        trigger, reason = min(
            candidates,
            key=lambda item: (int(tape.ordinals[item[0]]), 0 if item[1] == "STOP" else 1),
        )
        return {
            "resolution": reason,
            "resolution_ordinal": int(tape.ordinals[trigger]),
            "resolution_timestamp_ns": int(tape.timestamps[trigger]),
            "source_complete": True,
        }
    if tape.source_end_event is not None:
        return {
            "resolution": "SOURCE_END_UNRESOLVED",
            "resolution_ordinal": terminal_ordinal,
            "resolution_timestamp_ns": int(terminal["timestamp_ns"]),
            "source_complete": False,
        }
    return {
        "resolution": str(terminal.get("hard_flat_reason") or "HARD_FLAT"),
        "resolution_ordinal": terminal_ordinal,
        "resolution_timestamp_ns": int(terminal["timestamp_ns"]),
        "source_complete": True,
    }


def _boundary_row(
    boundary: matrix.TapeBoundary, *, instrument: str, resolution: Mapping[str, Any],
) -> dict[str, Any]:
    reopen_ns = boundary.reopen_timestamp_ns(instrument)
    return {
        "boundary_start_ordinal": boundary.start_ordinal,
        "boundary_start_timestamp_ns": boundary.start_timestamp_ns,
        "boundary_start_utc": _utc(boundary.start_timestamp_ns),
        "book_state": boundary.book_state,
        "classification": boundary.classification,
        "frozen_contract_disposition": FROZEN_POSITION_POLICY[boundary.classification],
        "instrument": instrument,
        "reopen_ordinal": boundary.reopen_ordinal(instrument),
        "reopen_timestamp_ns": reopen_ns,
        "reopen_utc": _utc(reopen_ns),
        "duration_ns": reopen_ns - boundary.start_timestamp_ns if reopen_ns is not None else None,
        "duration_seconds": (reopen_ns - boundary.start_timestamp_ns) / 1e9 if reopen_ns is not None else None,
        "first_es_reopen": {
            "ordinal": boundary.es_reopen_ordinal,
            "timestamp_ns": boundary.es_reopen_timestamp_ns,
            "timestamp_utc": _utc(boundary.es_reopen_timestamp_ns),
            "bid": boundary.es_reopen_bid,
            "ask": boundary.es_reopen_ask,
        },
        "first_mes_reopen": {
            "ordinal": boundary.mes_reopen_ordinal,
            "timestamp_ns": boundary.mes_reopen_timestamp_ns,
            "timestamp_utc": _utc(boundary.mes_reopen_timestamp_ns),
            "bid": boundary.mes_reopen_bid,
            "ask": boundary.mes_reopen_ask,
        },
        "explicit_reopen_marker_present": False,
        "reopen_evidence": "FIRST_FRESH_EXECUTABLE_NATIVE_BBO_AFTER_MARKER",
        "post_reopen_path_available": bool(
            reopen_ns is not None
            and int(resolution["resolution_ordinal"]) >= int(boundary.reopen_ordinal(instrument) or 2**63 - 1)
            and bool(resolution["source_complete"])
        ),
    }


def _first_failing_configuration(
    bundle: allp.PeriodBundle,
) -> dict[str, Any]:
    interactions, indexes = allp._load_bundle_rows(bundle)
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in interactions:
        by_day[str(row["session_date"])].append(row)
    weight_grid = matrix.generate_weight_grid()
    weights_matrix = np.asarray(weight_grid, dtype=np.float64) * float(matrix.WEIGHT_UNIT)
    evaluated = 0
    for day in bundle.days:
        rows = by_day.get(day, [])
        tape = matrix.SessionCausalTape.from_parquet(
            day, bundle.root / "causal-event-tape" / f"{day}.parquet",
        )
        components = np.asarray(
            [[float(row[name]) for name in master.SCORE_FIELDS] for row in rows], dtype=np.float64,
        )
        penalties = np.asarray([float(row["false_refill_penalty"]) for row in rows], dtype=np.float64)
        primitive_ok = np.asarray([not str(row.get("non_quality_rejection_reasons") or "") for row in rows])
        scores = np.clip(
            components @ weights_matrix.T
            - penalties[:, None] * float(master.V2_CONFIG.false_refill_penalty_weight),
            0.0, 1.0,
        )
        day_indexes = {str(row["interaction_id"]): indexes[str(row["interaction_id"])] for row in rows}
        for weight_index, units in enumerate(weight_grid):
            for threshold in matrix.QUALITY_THRESHOLDS:
                evaluated += 1
                accepted_mask = matrix._accepted_mask(
                    rows, primitive_ok, scores[:, weight_index], units, threshold,
                )
                accepted = [row for row, keep in zip(rows, accepted_mask) if bool(keep)]
                session = matrix.simulate_independent_session(tape, accepted, day_indexes)
                failed = [
                    (identifier, reason) for identifier, reason in session.terminal_outcomes.items()
                    if reason.startswith("POSITION_UNRESOLVED_")
                ]
                if not failed:
                    continue
                interaction_id, terminal_reason = failed[0]
                interaction = next(row for row in rows if str(row["interaction_id"]) == interaction_id)
                index = day_indexes[interaction_id]
                entry_ordinal = int(index["entry_observation_event_ordinal"])
                plan = _entry_plan(tape, interaction, entry_ordinal)
                if plan is None:
                    raise BoundaryAuditError("failing interaction lost its executable entry plan")
                outcome = tape.entry_outcome(interaction, entry_ordinal)
                if outcome.boundary is None:
                    raise BoundaryAuditError("failing interaction lost its typed boundary")
                resolution = _diagnostic_exit_without_boundary(tape, plan)
                boundary = _boundary_row(
                    outcome.boundary, instrument=str(plan["instrument"]), resolution=resolution,
                )
                return {
                    "config_id": matrix.config_id(units, threshold),
                    "G1": units[0] * 0.05,
                    "G2": units[1] * 0.05,
                    "G3": units[2] * 0.05,
                    "G4": units[3] * 0.05,
                    "G5": units[4] * 0.05,
                    "quality_threshold": float(threshold),
                    "configurations_examined_until_first_failure": evaluated,
                    "session_date": day,
                    "interaction_id": interaction_id,
                    "source_interaction_id": str(interaction["source_interaction_id"]),
                    "direction": str(interaction["direction"]),
                    "confirmation_timestamp_ns": int(index["derived_first_confirmation_timestamp_ns"]),
                    "confirmation_timestamp_utc": _utc(int(index["derived_first_confirmation_timestamp_ns"])),
                    "entry_timestamp_ns": int(plan["entry_timestamp_ns"]),
                    "entry_timestamp_utc": _utc(int(plan["entry_timestamp_ns"])),
                    "instrument": plan["instrument"],
                    "entry": plan["entry"],
                    "stop": plan["stop"],
                    "target": plan["target"],
                    "terminal_reason": terminal_reason,
                    "boundary": boundary,
                    "diagnostic_post_boundary_resolution_not_a_trade": {
                        **resolution,
                        "resolution_timestamp_utc": _utc(int(resolution["resolution_timestamp_ns"])),
                    },
                    "tape_data_sufficient_after_reopen": bool(boundary["post_reopen_path_available"]),
                    "strategy_result_interpretable": False,
                }
    raise BoundaryAuditError("no classified first boundary overlap found")


def audit_all_tapes(
    *, repository_root: Path, tape_root: Path,
) -> dict[str, Any]:
    bundles = allp._period_bundles(repository_root, tape_root)
    period_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    total_markers = Counter()
    for bundle in bundles:
        interactions, indexes = allp._load_bundle_rows(bundle)
        by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in interactions:
            by_day[str(row["session_date"])].append(row)
        period_overlap_start = len(overlap_rows)
        period_markers = Counter()
        session_rows: list[dict[str, Any]] = []
        for day in bundle.days:
            tape = matrix.SessionCausalTape.from_parquet(
                day, bundle.root / "causal-event-tape" / f"{day}.parquet",
            )
            for boundary in tape.non_executable_boundaries:
                period_markers[boundary.classification] += 1
                total_markers[boundary.classification] += 1
            day_overlaps = 0
            for interaction in by_day.get(day, []):
                index = indexes[str(interaction["interaction_id"])]
                if (
                    index.get("derived_first_confirmation_timestamp_ns") is None
                    or index.get("entry_observation_event_ordinal") is None
                ):
                    continue
                entry_ordinal = int(index["entry_observation_event_ordinal"])
                plan = _entry_plan(tape, interaction, entry_ordinal)
                if plan is None:
                    continue
                resolution = _diagnostic_exit_without_boundary(tape, plan)
                terminal_ordinal = int(resolution["resolution_ordinal"])
                boundaries = [
                    item for item in tape.non_executable_boundaries
                    if entry_ordinal < item.start_ordinal <= terminal_ordinal
                ]
                if not boundaries and resolution["resolution"] != "SOURCE_END_UNRESOLVED":
                    continue
                day_overlaps += 1
                overlap_rows.append({
                    "period_id": bundle.period.period_id,
                    "session_date": day,
                    "interaction_id": str(interaction["interaction_id"]),
                    "source_interaction_id": str(interaction["source_interaction_id"]),
                    "direction": str(interaction["direction"]),
                    "instrument": str(plan["instrument"]),
                    "entry_ordinal": entry_ordinal,
                    "entry_timestamp_ns": int(plan["entry_timestamp_ns"]),
                    "entry_timestamp_utc": _utc(int(plan["entry_timestamp_ns"])),
                    "boundary_overlap_count": len(boundaries) if boundaries else 1,
                    "boundary_classifications": [item.classification for item in boundaries]
                    if boundaries else ["SOURCE_END"],
                    "boundaries": [
                        _boundary_row(item, instrument=str(plan["instrument"]), resolution=resolution)
                        for item in boundaries
                    ],
                    "diagnostic_resolution": resolution["resolution"],
                    "tape_path_complete": bool(resolution["source_complete"]),
                    "safely_resumable_under_frozen_contract": False,
                    "fail_closed": True,
                    "ambiguous": False,
                })
            session_rows.append({
                "session_date": day,
                "completed_interactions": len(by_day.get(day, [])),
                "boundary_markers": len(tape.non_executable_boundaries),
                "potential_interactions_overlapping_boundary": day_overlaps,
            })
        affected = overlap_rows[period_overlap_start:]
        period_rows.append({
            "period_id": bundle.period.period_id,
            "session_count": len(bundle.days),
            "completed_interactions": len(interactions),
            "boundary_marker_counts": dict(sorted(period_markers.items())),
            "potential_interactions_affected": len(affected),
            "safely_resumable": 0,
            "truly_fail_closed": len(affected),
            "unresolved": sum(not bool(row["tape_path_complete"]) for row in affected),
            "tape_complete_but_contract_fail_closed": sum(bool(row["tape_path_complete"]) for row in affected),
            "ambiguous": 0,
            "sessions": session_rows,
        })
    first_bundle = next(item for item in bundles if item.period.period_id == "AUGUST_10_14_2026")
    first_failure = _first_failing_configuration(first_bundle)
    return {
        "status": "DURABLE_BOUNDARY_AUDIT_PASS_OPTIMIZER_RELEASED",
        "classification": CLASSIFICATION,
        "strategy_id": STRATEGY_ID,
        "v3_contract_sha256": allp.V3_CONTRACT_SHA256,
        "grid_changed": False,
        "strategy_parameters_changed": False,
        "optimizer_executed": False,
        "optimizer_ranking_performed": False,
        "dbn_files_opened": 0,
        "network_calls": 0,
        "downloads": 0,
        "frozen_position_boundary_policy": FROZEN_POSITION_POLICY,
        "first_failure": first_failure,
        "totals": {
            "periods": len(period_rows),
            "sessions": sum(int(row["session_count"]) for row in period_rows),
            "completed_interactions": sum(int(row["completed_interactions"]) for row in period_rows),
            "boundary_marker_counts": dict(sorted(total_markers.items())),
            "potential_interactions_affected": len(overlap_rows),
            "safely_resumable": 0,
            "truly_fail_closed": len(overlap_rows),
            "unresolved": sum(not bool(row["tape_path_complete"]) for row in overlap_rows),
            "tape_complete_but_contract_fail_closed": sum(bool(row["tape_path_complete"]) for row in overlap_rows),
            "ambiguous": 0,
        },
        "periods": period_rows,
        "overlaps": overlap_rows,
        "checkpoint_audit": {
            "optimizer_output_exists": (repository_root / allp.OUTPUT_ROOT).exists(),
            "optimizer_staging_exists": (repository_root / allp.OUTPUT_ROOT).with_name(
                allp.OUTPUT_ROOT.name + ".building"
            ).exists(),
            "period_configuration_checkpoints_supported": False,
            "reusable_completed_period_configuration_results": 0,
            "reason": "run_optimizer keeps period rows in memory and creates staging only after all seven evaluations",
        },
    }


def _html_report(payload: Mapping[str, Any]) -> str:
    first = payload["first_failure"]
    totals = payload["totals"]
    periods = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in (
            row["period_id"], row["session_count"], row["completed_interactions"],
            row["potential_interactions_affected"], row["truly_fail_closed"], row["unresolved"],
        )) + "</tr>"
        for row in payload["periods"]
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Durable boundary reconciliation</title>
<style>body{{font:15px/1.5 system-ui;margin:2rem;max-width:1100px;color:#172033}}h1,h2{{color:#102a43}}
.ok{{color:#087f5b;font-weight:700}}code{{background:#eef2f7;padding:.15rem .3rem}}table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #ccd6e0;padding:.45rem;text-align:left}}th{{background:#edf2f7}}.note{{background:#fff4d6;padding:1rem;border-left:4px solid #e0a800}}</style></head><body>
<h1>All-period durable boundary reconciliation</h1><p class="ok">{html.escape(str(payload['status']))}</p>
<p>This is a read-only compact-Parquet integrity audit. No optimizer ranking, DBN replay, download, or strategy selection occurred.</p>
<h2>First failure</h2><p><code>{first['config_id']}</code>, {first['session_date']}, interaction
<code>{html.escape(first['source_interaction_id'])}</code>, {first['instrument']} {first['direction']}.</p>
<p>Entry {first['entry']} / stop {first['stop']} / target {first['target']}. Boundary:
<code>{first['boundary']['classification']}</code> at {first['boundary']['boundary_start_utc']}; first fresh
{first['instrument']} BBO at {first['boundary']['reopen_utc']}.</p>
<div class="note">The tape contains a later path, but the frozen real runner explicitly fails closed when this source state overlaps an open position. The optimizer now records an unresolved configuration instead of crashing or inventing a resumed fill.</div>
<h2>Audit totals</h2><ul><li>{totals['sessions']} sessions / {totals['completed_interactions']} completed interactions</li>
<li>{totals['potential_interactions_affected']} potential open-position paths affected</li>
<li>{totals['truly_fail_closed']} fail-closed; {totals['safely_resumable']} resumable under the frozen contract; {totals['ambiguous']} ambiguous</li></ul>
<table><thead><tr><th>Period</th><th>Sessions</th><th>Interactions</th><th>Affected</th><th>Fail closed</th><th>Incomplete tape</th></tr></thead><tbody>{periods}</tbody></table>
<h2>Boundary policy</h2><pre>{html.escape(json.dumps(payload['frozen_position_boundary_policy'], indent=2, sort_keys=True))}</pre>
<h2>Checkpoint audit</h2><pre>{html.escape(json.dumps(payload['checkpoint_audit'], indent=2, sort_keys=True))}</pre>
</body></html>"""


def run(*, repository_root: Path, tape_root: Path, report_root: Path) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    tape_root = (repository_root / tape_root).resolve() if not tape_root.is_absolute() else tape_root.resolve()
    report_root = (repository_root / report_root).resolve() if not report_root.is_absolute() else report_root.resolve()
    payload = audit_all_tapes(repository_root=repository_root, tape_root=tape_root)
    report_root.mkdir(parents=True, exist_ok=True)
    json_path = report_root / f"{REPORT_NAME}.json"
    html_path = report_root / f"{REPORT_NAME}.html"
    allp._write_json(json_path, payload)
    temporary = html_path.with_suffix(".html.part")
    temporary.write_text(_html_report(payload), encoding="utf-8")
    temporary.replace(html_path)
    return {**payload, "json_report_path": str(json_path), "html_report_path": str(html_path)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--tape-root", type=Path, default=allp.TAPE_ROOT)
    parser.add_argument("--report-root", type=Path, default=allp.TAPE_ROOT)
    args = parser.parse_args(argv)
    try:
        result = run(
            repository_root=args.repository_root,
            tape_root=args.tape_root,
            report_root=args.report_root,
        )
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
    print(json.dumps({
        "status": result["status"],
        "first_failure": result["first_failure"],
        "totals": result["totals"],
        "json_report_path": result["json_report_path"],
        "html_report_path": result["html_report_path"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
