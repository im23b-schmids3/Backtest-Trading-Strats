"""Local-only April 6--8 replay for the frozen native-L2 V3 POC-only contract.

The module deliberately has no Databento client, download path, or parameter
selection.  It accepts only the nine files bound by the acquired April
manifest and publishes results only after all three sessions complete.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import historical_runner as historical
from . import v3_poc_fresh_august_replay as native
from .model import StructuralLevel
from .v2_quality050 import V2_CONFIG
from .v3_poc_only import ELIGIBLE_STRUCTURAL_LEVELS, STRATEGY_ID, v3_contract, v3_contract_sha256


DATA_ROOT = Path("data/cme_orderflow_absorption_l2_v3/apr06_08_retro")
OUTPUT_ROOT = Path("research_runs/CMEOrderflowAbsorption.ES_L2_V3_POC_ONLY_APR06_08_RETRO")
TARGET_DATES = ("2026-04-06", "2026-04-07", "2026-04-08")
PRIOR_RTH = {"2026-04-06": "2026-04-02", "2026-04-07": "2026-04-06", "2026-04-08": "2026-04-07"}
EVIDENCE_LABEL = "APRIL_2026_RETROSPECTIVE_ROBUSTNESS_L2_V3_POC_ONLY"
V3_CONTRACT_SHA256 = "a0ce94eeb78dcbf865cf4464bdf97ebc3f014a8ec5e2f559f99798534dfcbcb4"


class V3AprilReplayError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def _expected_files() -> dict[str, dict[str, str]]:
    expected: dict[str, dict[str, str]] = {}
    for day in TARGET_DATES:
        expected[f"es_mbp10/ESM6_{day}_130000_224501_mbp10.dbn.zst"] = {
            "label": "ES_MBP10_USD", "schema": "mbp-10", "symbol": "ESM6", "session_date": day,
            "start": f"{day}T13:00:00Z", "end": f"{day}T22:45:01Z",
        }
        expected[f"mes_mbp1/MESM6_{day}_133000_224501_mbp1.dbn.zst"] = {
            "label": "MES_MBP1_USD", "schema": "mbp-1", "symbol": "MESM6", "session_date": day,
            "start": f"{day}T13:30:00Z", "end": f"{day}T22:45:01Z",
        }
        prior = PRIOR_RTH[day]
        expected[f"es_prior_rth_trades/ESM6_{prior}_133000_200000_trades.dbn.zst"] = {
            "label": "PRIOR_RTH_TRADES_USD", "schema": "trades", "symbol": "ESM6", "session_date": prior,
            "start": f"{prior}T13:30:00Z", "end": f"{prior}T20:00:00Z",
        }
    return expected


def verify_acquisition_manifest(data_root: Path) -> dict[str, Any]:
    """Verify manifest identities and hashes without parsing any DBN file."""
    if v3_contract_sha256() != V3_CONTRACT_SHA256:
        raise V3AprilReplayError("frozen V3 strategy contract hash mismatch")
    manifest_path = data_root / "acquisition-manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V3AprilReplayError("missing or unreadable sealed April acquisition manifest") from exc
    identity = manifest.get("request_identity", {})
    if (manifest.get("manifest_kind") != "APRIL_2026_L2_V3_POC_ONLY_RETROSPECTIVE_ACQUISITION" or
            manifest.get("data_acquired") is not True or manifest.get("strategy_replay_executed") is not False or
            manifest.get("outcomes_inspected") is not False or manifest.get("no_mbo_purchased") is not True or
            identity.get("strategy_id") != STRATEGY_ID or identity.get("v3_contract_sha256") != V3_CONTRACT_SHA256 or
            identity.get("evidence_label") != EVIDENCE_LABEL or identity.get("target_rth_dates") != list(TARGET_DATES) or
            identity.get("prior_rth_mapping") != PRIOR_RTH or identity.get("mbo_purchased") is not False or
            identity.get("strict_chronological_oos") is not False):
        raise V3AprilReplayError("April acquisition manifest does not bind the frozen retrospective V3 package")
    expected, files = _expected_files(), manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(expected):
        raise V3AprilReplayError("April acquisition manifest has missing, extra, or unordered declared inputs")
    verified: list[dict[str, Any]] = []
    for relative, required in expected.items():
        record, local = files.get(relative), data_root / relative
        if not isinstance(record, dict) or not local.is_file() or any(record.get(key) != value for key, value in required.items()):
            raise V3AprilReplayError(f"April input identity mismatch: {relative}")
        if local.stat().st_size <= 0 or local.stat().st_size != record.get("bytes") or _sha256(local) != record.get("sha256"):
            raise V3AprilReplayError(f"April input hash/size mismatch: {relative}")
        verified.append({"relative_path": relative, "bytes": local.stat().st_size, "sha256": record["sha256"]})
    counts = Counter(str(row["label"]) for row in files.values())
    if counts != Counter({"ES_MBP10_USD": 3, "MES_MBP1_USD": 3, "PRIOR_RTH_TRADES_USD": 3}):
        raise V3AprilReplayError("April acquisition manifest component cardinality is not 3/3/3")
    return {
        "preflight_only": True, "manifest_path": str(manifest_path), "manifest_sha256": _sha256(manifest_path),
        "files_verified": len(verified), "by_label": dict(counts), "target_dates": list(TARGET_DATES),
        "prior_rth_mapping": PRIOR_RTH, "v3_contract_sha256": V3_CONTRACT_SHA256,
        "strategy_or_outcomes_accessed": False, "files": verified,
    }


def _paths(data_root: Path, day: str) -> tuple[Path, Path, Path]:
    return (
        data_root / "es_mbp10" / f"ESM6_{day}_130000_224501_mbp10.dbn.zst",
        data_root / "mes_mbp1" / f"MESM6_{day}_133000_224501_mbp1.dbn.zst",
        data_root / "es_prior_rth_trades" / f"ESM6_{PRIOR_RTH[day]}_133000_200000_trades.dbn.zst",
    )


def _utc(timestamp_ns: int) -> str:
    return datetime.fromtimestamp(timestamp_ns / 1_000_000_000, tz=timezone.utc).isoformat()


def _record_action_side(record: object) -> tuple[str, str]:
    return native._code(getattr(record, "action", "")), native._code(getattr(record, "side", ""))


def _reopening_bbo(public: historical.PublicBookEvent) -> dict[str, float]:
    return {"bid": public.snapshot.bids[0].price, "ask": public.snapshot.asks[0].price}


def audit_native_mbp10(data_root: Path) -> dict[str, Any]:
    """Read-only full adapter audit; no strategy runner or outcome code exists here."""
    verification = verify_acquisition_manifest(data_root)
    sessions: list[dict[str, Any]] = []
    april7_transitions: list[dict[str, Any]] = []
    for day in TARGET_DATES:
        es_path, _, _ = _paths(data_root, day)
        adapter = native.NativeMBP10Adapter()
        episodes: list[dict[str, Any]] = []
        active_episode: dict[str, Any] | None = None
        records_processed = 0
        for record_number, record in enumerate(native._stream_native_mbp10_records(es_path), start=1):
            records_processed = record_number
            timestamp_ns = native._timestamp(record)
            action, side = _record_action_side(record)
            old_state = adapter.state
            expected_maintenance = native._in_maintenance_pause(timestamp_ns, day)
            public = adapter.feed(record, expected_non_executable=expected_maintenance)
            new_state = adapter.state
            if day == "2026-04-07" and 11_143_690 <= record_number <= 11_143_960:
                if record_number == 11_143_690 or old_state != new_state:
                    april7_transitions.append({
                        "record_number": record_number, "timestamp_utc": _utc(timestamp_ns),
                        "action": action, "side": side, "from_state": old_state, "to_state": new_state,
                    })
            if public is None and adapter.first_valid_book_ns is not None:
                if active_episode is None:
                    classification = (
                        "SCHEDULED_MAINTENANCE" if expected_maintenance else
                        "ORDINARY_TEMPORARY_RECONSTRUCTION" if new_state == "TEMPORARILY_NON_EXECUTABLE" else
                        "ORDINARY_WAITING_FOR_REOPEN_BOOK"
                    )
                    active_episode = {
                        "session": day, "start_record": record_number, "end_record": record_number,
                        "start_ts_recv": timestamp_ns, "end_ts_recv": timestamp_ns,
                        "initiating_action": action, "initiating_side": side,
                        "state_classification": classification,
                        "maintenance": classification == "SCHEDULED_MAINTENANCE",
                        "records": 0, "states_observed": [],
                    }
                active_episode["end_record"] = record_number
                active_episode["end_ts_recv"] = timestamp_ns
                active_episode["records"] += 1
                if new_state not in active_episode["states_observed"]:
                    active_episode["states_observed"].append(new_state)
            elif public is not None and active_episode is not None:
                active_episode["start_ts_recv_utc"] = _utc(int(active_episode["start_ts_recv"]))
                active_episode["end_ts_recv_utc"] = _utc(int(active_episode["end_ts_recv"]))
                active_episode["last_non_executable_span_ns"] = int(active_episode["end_ts_recv"]) - int(active_episode["start_ts_recv"])
                active_episode["first_valid_reopen_record"] = record_number
                active_episode["first_valid_reopen_ts_recv"] = timestamp_ns
                active_episode["first_valid_reopen_ts_recv_utc"] = _utc(timestamp_ns)
                active_episode["duration_ns"] = timestamp_ns - int(active_episode["start_ts_recv"])
                active_episode["first_valid_reopening_bbo"] = _reopening_bbo(public)
                episodes.append(active_episode)
                active_episode = None
            if record_number % 2_000_000 == 0:
                print(f"adapter-audit {day} records={record_number:,} episodes={len(episodes)}", file=sys.stderr, flush=True)
        if active_episode is not None:
            raise V3AprilReplayError(
                f"native MBP-10 source ended during unresolved episode: {day} record {active_episode['start_record']}"
            )
        adapter.assert_executable_at_boundary()
        sessions.append({
            "session": day, "records_processed": records_processed, "episodes": episodes,
            "adapter_completed_without_exception": True, "unresolved_at_source_end": False,
        })
    return {
        "audit_kind": "READ_ONLY_NATIVE_MBP10_ADAPTER_AUDIT",
        "strategy_id": STRATEGY_ID, "v3_contract_sha256": V3_CONTRACT_SHA256,
        "target_dates": list(TARGET_DATES), "manifest_verification": verification,
        "strategy_runner_invoked": False, "pnl_or_outcomes_accessed": False,
        "all_three_sessions_completed": len(sessions) == 3,
        "all_sessions_resolved_at_source_end": all(not row["unresolved_at_source_end"] for row in sessions),
        "april7_record_11143690_11143960_state_transitions": april7_transitions,
        "sessions": sessions,
    }


def calendar_contract() -> dict[str, Any]:
    return {
        "rule": "effective_hard_flat = min(frozen_22_45_UTC, scheduled_market_close)",
        "normal_hard_flat_utc": "22:45:00",
        "scheduled_close_by_date": {},
        "liquidation_window": "[effective_hard_flat - 1 second, effective_hard_flat] inclusive",
        "no_invented_post_close_bbo": True,
        "strategy_contract_changed": False,
    }


def _validated_prior_rth_poc(profile_path: Path) -> StructuralLevel:
    """Adapt the V3 helper's single-level POC interface explicitly.

    ``native._profile_poc`` deliberately returns one ``StructuralLevel``;
    ``HistoricalL2Runner`` separately accepts an iterable of levels.  Keeping
    those two interfaces distinct prevents an accidental multi-level path.
    """
    level = native._profile_poc(profile_path)
    if not isinstance(level, StructuralLevel):
        raise V3AprilReplayError("prior-RTH POC builder must return one StructuralLevel")
    if level.name != "PRIOR_RTH_POC":
        raise V3AprilReplayError("only PRIOR_RTH_POC may be eligible in the frozen V3 replay")
    return level


def _run_session(day: str, data_root: Path) -> historical.HistoricalL2Runner:
    es_path, mes_path, profile_path = _paths(data_root, day)
    level = _validated_prior_rth_poc(profile_path)
    runner = historical.HistoricalL2Runner(
        date=day, evidence_label=EVIDENCE_LABEL, levels=[level], config=V2_CONFIG,
        strategy_id=STRATEGY_ID, require_native_mes_for_fallback=True,
    )
    adapter = native.NativeMBP10Adapter()
    es_iter, mes_iter = iter(native._stream_native_mbp10_records(es_path)), iter(historical._stream_mes_quotes(mes_path))
    es, mes = historical._next(es_iter), historical._next(mes_iter)
    start_ns = historical._clock_ns(day, historical.RTH_START_SECONDS)
    cutoff_ns = historical._clock_ns(day, historical.HARD_CUTOFF_SECONDS)
    initialized, pause_active, closed, records = False, False, False, 0
    while es is not None or mes is not None:
        es_timestamp = native._timestamp(es) if es is not None else None
        mes_timestamp = mes[0] if mes is not None else None
        timestamp = min(value for value in (es_timestamp, mes_timestamp) if value is not None)
        if timestamp >= cutoff_ns:
            if not initialized:
                raise V3AprilReplayError("native MBP-10 initialization absent before RTH")
            adapter.assert_executable_at_boundary()
            runner.force_flat_from_last_causal_cutoff_quote(cutoff_ns)
            runner.finish(cutoff_ns); closed = True; break
        if native._in_maintenance_pause(timestamp, day):
            if not pause_active:
                native._begin_non_executable_state(runner, timestamp); pause_active = True
            if mes_timestamp is not None and mes_timestamp <= (es_timestamp if es_timestamp is not None else mes_timestamp):
                mes = historical._next(mes_iter)
            else:
                assert es is not None
                adapter.feed(es, expected_non_executable=True); es = historical._next(es_iter)
            continue
        if mes_timestamp is not None and mes_timestamp <= (es_timestamp if es_timestamp is not None else mes_timestamp):
            if getattr(runner, "_native_transient_started_ns", None) is None:
                runner.observe_mes_quote(*mes)
            mes = historical._next(mes_iter)
            continue
        assert es is not None
        public = adapter.feed(es)
        if es_timestamp < start_ns:
            initialized = adapter.first_valid_book_ns is not None and adapter.first_valid_book_ns < start_ns
            es = historical._next(es_iter); continue
        if not initialized:
            raise V3AprilReplayError("native MBP-10 initialization absent before first RTH event")
        if public is not None:
            native._resume_temporary_non_executable_state(runner, es_timestamp)
            runner.observe_public(public); records += 1
        elif adapter.state == "TEMPORARILY_NON_EXECUTABLE":
            native._begin_temporary_non_executable_state(runner, es_timestamp)
        else:
            native._begin_non_executable_state(runner, es_timestamp)
        if records and records % 1_000_000 == 0:
            print(f"  V3 April native MBP-10 {day} records={records:,}", flush=True)
        es = historical._next(es_iter)
    if not closed:
        if not initialized:
            raise V3AprilReplayError("native MBP-10 initialization absent before RTH")
        adapter.assert_executable_at_boundary()
        runner.force_flat_from_last_causal_cutoff_quote(cutoff_ns)
        runner.finish(cutoff_ns)
    return runner


def _rows(runners: list[historical.HistoricalL2Runner], attribute: str) -> list[dict[str, Any]]:
    return [row for runner in runners for row in getattr(runner, attribute)]


def _metrics(runners: list[historical.HistoricalL2Runner]) -> dict[str, Any]:
    setups, trades = _rows(runners, "setup_ledger"), _rows(runners, "trade_ledger")
    pending = [setup for runner in runners for setup in runner.signals.pending.values()]
    performance = historical._performance(trades)
    return {
        "completed_interactions": sum(len(runner.interaction_ledger) for runner in runners),
        "accepted_poc_setups": sum(bool(row["accepted"]) for row in setups),
        "rejected_interactions": sum(not bool(row["accepted"]) for row in setups),
        "confirmations_passed": sum(setup.confirmation_timestamp_ns is not None for setup in pending),
        "confirmation_expiries": sum(setup.terminal_reason == "CONFIRMATION_WINDOW_EXPIRED" for setup in pending),
        "active_position_blocks": sum(setup.terminal_reason == "COMPLIANCE_BLOCK_ACTIVE_POSITION" for setup in pending),
        "unresolved": sum(setup.terminal_reason in {"SOURCE_TAIL_UNRESOLVED", "SOURCE_NON_EXECUTABLE_BEFORE_ENTRY"} for setup in pending),
        **performance,
    }


def _cross_period_context(april_metrics: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"period": "May development POC subset", "poc_trades": 6, "total_r": 4.4091, "evidence_classification": "DEVELOPMENT_EVIDENCE_NOT_OOS"},
        {"period": "Retro June/July POC subset", "poc_trades": 8, "total_r": 2.6756, "unresolved": 0, "evidence_classification": "RETROSPECTIVE_ROBUSTNESS_NOT_STRICT_OOS"},
        {"period": "Seen Aug 3-6", "poc_trades": 0, "evidence_classification": "SEEN_DATA_NOT_FRESH_OOS"},
        {"period": "Fresh Aug 10-14", "poc_trades": 9, "total_r": 1.81197, "profit_factor": 1.4018, "evidence_classification": "FRESH_PREDECLARED_BLOCK"},
        {"period": "April Apr 6-8", "poc_trades": april_metrics["completed_trades"], "wins": april_metrics["wins"], "losses": april_metrics["losses"], "total_r": april_metrics["total_r"], "profit_factor": april_metrics["profit_factor"], "evidence_classification": EVIDENCE_LABEL},
    ]


def _write_results(output_root: Path, runners: list[historical.HistoricalL2Runner], verification: dict[str, Any]) -> dict[str, Any]:
    base = historical.write_future_artifacts(output_root, runners, contract=v3_contract())
    metrics = _metrics(runners)
    result = {
        **base, "strategy_id": STRATEGY_ID, "evidence_label": EVIDENCE_LABEL,
        "v3_contract_sha256": V3_CONTRACT_SHA256, "input_verification": verification,
        "execution_calendar": calendar_contract(), "metrics": metrics,
        "cross_period_context": _cross_period_context(metrics),
        "strategy_replay_completed_all_three_sessions": True,
        "strict_chronological_oos": False,
        "outcome_parameter_selection": False,
    }
    (output_root / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_root / "diagnostic-report.md").write_text(
        f"# {STRATEGY_ID} â€” April 2026 retrospective robustness replay\n\n"
        f"Evidence label: `{EVIDENCE_LABEL}`. This is not strict chronological OOS or fresh validation.\n\n"
        "Only the frozen `PRIOR_RTH_POC` level is eligible. No outcome-based parameter selection, pooling, or post-April strategy modification is permitted.\n\n"
        f"Completed interactions: {metrics['completed_interactions']}; accepted POC setups: {metrics['accepted_poc_setups']}; completed trades: {metrics['completed_trades']}; total R: {metrics['total_r']:.6f}.\n",
        encoding="utf-8",
    )
    return result


def run(*, data_root: Path, output_root: Path) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError("April V3 output directory already exists")
    verification = verify_acquisition_manifest(data_root)
    runners: list[historical.HistoricalL2Runner] = []
    for index, day in enumerate(TARGET_DATES, start=1):
        print(f"=== V3 APRIL RETRO {index:02d}/03 {day} ===", flush=True)
        runners.append(_run_session(day, data_root))
    return _write_results(output_root, runners, verification)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the sealed April native-L2 V3 POC-only replay locally")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--preflight-only", action="store_true", help="hash-verify inputs only; never parse DBNs or run strategy")
    parser.add_argument("--adapter-audit", action="store_true", help="read-only native MBP-10 state audit; never run strategy")
    args = parser.parse_args(argv)
    try:
        if args.preflight_only and args.adapter_audit:
            raise ValueError("--preflight-only and --adapter-audit are mutually exclusive")
        result = (
            verify_acquisition_manifest(args.data_root) if args.preflight_only else
            audit_native_mbp10(args.data_root) if args.adapter_audit else
            run(data_root=args.data_root, output_root=args.output_root)
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    except (V3AprilReplayError, historical.HistoricalReplayError, FileExistsError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
