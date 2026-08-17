"""One-pass local replay for the frozen L2 V2 August seen-data block.

This is intentionally a local DBN replay runner, not an acquisition client.
It validates the sealed ES MBO and nine completion inputs before processing and
reuses the frozen MBO-to-synthetic-MBP10 adapter without exposing order ids to
the strategy layer.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import historical_runner as historical
from . import v2_extended_existing_data as extended
from . import v2_quality050 as v2


EVIDENCE_LABEL = "SEEN_AUG_L2_V2_NOT_FRESH_OOS_EVIDENCE"
VARIANT_LABEL = "L2_V2_MAY_DEVELOPMENT_QUALITY_0_50"
TARGET_DATES = ("2026-08-03", "2026-08-04", "2026-08-05", "2026-08-06")
PRIOR_RTH = {"2026-08-03": "2026-07-31", "2026-08-04": "2026-08-03", "2026-08-05": "2026-08-04", "2026-08-06": "2026-08-05"}
ES_MBO_RELATIVE = "data/cme_orderflow_absorption_v1/oos_v1/ESU6/mbo/ESU6_2026-08-03_2026-08-08_mbo.dbn"
ES_MANIFEST_RELATIVE = "docs/research_pipeline/cme_orderflow_absorption_v1/oos-v1-data-manifest.json"
DEFAULT_COMPLETION_ROOT = Path("data/cme_orderflow_absorption_l2_v2/august_completion")
DEFAULT_OUTPUT_ROOT = Path("research_runs/CMEOrderflowAbsorption.ES_L2_V2_AUGUST_SEEN")


class AugustReplayError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def _expected_completion_files() -> dict[str, dict[str, str]]:
    expected: dict[str, dict[str, str]] = {}
    for day in TARGET_DATES:
        expected[f"mes_mbp1/MESU6_{day}_133000_224501_mbp1.dbn.zst"] = {
            "label": "MES_NATIVE_EXECUTION", "schema": "mbp-1", "symbol": "MESU6", "session_date": day,
            "start": f"{day}T13:30:00Z", "end": f"{day}T22:45:01Z",
        }
    # The sealed completion package contains the five previous-RTH profiles
    # July 31 through August 6. The Aug. 6 profile is verified as part of the
    # immutable nine-file package but is not consumed: Aug. 7 is excluded.
    for day in ("2026-07-31", "2026-08-03", "2026-08-04", "2026-08-05", "2026-08-06"):
        expected[f"es_prior_rth_trades/ESU6_{day}_133000_200000_trades.dbn.zst"] = {
            "label": "ES_PRIOR_RTH_PROFILE", "schema": "trades", "symbol": "ESU6", "session_date": day,
            "start": f"{day}T13:30:00Z", "end": f"{day}T20:00:00Z",
        }
    return expected


def verify_august_inputs(*, repository_root: Path, completion_root: Path) -> dict[str, Any]:
    """Hash-verify all local sources before any DBN stream is opened."""
    if v2.v2_contract_sha256() != "f6152ed1ca32bb7c93a62ddf672a0708ff6c92abb0beb126eec78bf7ab3ab239":
        raise AugustReplayError("frozen L2 V2 contract hash mismatch")
    try:
        es_manifest = json.loads((repository_root / ES_MANIFEST_RELATIVE).read_text(encoding="utf-8"))
        completion = json.loads((completion_root / "acquisition-manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AugustReplayError("required August manifest is missing or unreadable") from exc
    es_path = repository_root / ES_MBO_RELATIVE
    expected_es = es_manifest.get("proposed_acquisition", {})
    actual_es_sha256 = _sha256(es_path) if es_path.is_file() else ""
    if (not es_path.is_file() or expected_es.get("file_bytes") != es_path.stat().st_size or
            str(expected_es.get("file_sha256", "")).lower() != actual_es_sha256):
        raise AugustReplayError("existing sealed August ES MBO identity mismatch")
    provider = es_manifest.get("provider", {})
    instrument = es_manifest.get("symbol_and_instrument", {})
    chronology = es_manifest.get("chronology", {})
    if (es_manifest.get("status") != "OOS_DATA_ACQUIRED_AND_SEALED" or es_manifest.get("data_acquired") is not True or
            expected_es.get("record_count") != 61_106_259 or expected_es.get("integrity_pass") is not True or
            provider.get("dataset") != "GLBX.MDP3" or provider.get("schema") != "mbo" or
            instrument.get("resolved_raw_symbol") != "ESU6" or instrument.get("instrument_ids") != [42140870] or
            chronology.get("eligible_rth_dates") != ["2026-08-03", "2026-08-04", "2026-08-05", "2026-08-06", "2026-08-07"]):
        raise AugustReplayError("existing August ES MBO manifest does not prove the sealed source contract")
    identity = completion.get("request_identity", {})
    if (completion.get("manifest_kind") != "AUGUST_2026_L2_V2_MISSING_COMPONENT_ACQUISITION" or
            completion.get("data_acquired") is not True or completion.get("strategy_replay_occurred") is not False or
            identity.get("strategy_id") != v2.STRATEGY_ID or identity.get("evidence_label") != EVIDENCE_LABEL or
            identity.get("v2_contract_sha256") != v2.v2_contract_sha256()):
        raise AugustReplayError("August completion manifest does not bind the frozen L2 V2 package")
    expected = _expected_completion_files()
    if identity.get("components") != list(expected.values()) or set(completion.get("files", {})) != set(expected):
        raise AugustReplayError("August completion manifest has missing, extra, or reordered components")
    verified: list[dict[str, Any]] = []
    for relative, expected_identity in expected.items():
        record, path = completion["files"].get(relative), completion_root / relative
        if not isinstance(record, dict) or not path.is_file() or any(record.get(key) != value for key, value in expected_identity.items()):
            raise AugustReplayError(f"August completion file identity mismatch: {relative}")
        if path.stat().st_size <= 0 or path.stat().st_size != record.get("bytes") or _sha256(path) != record.get("sha256"):
            raise AugustReplayError(f"August completion file hash/size mismatch: {relative}")
        verified.append({"relative_path": relative, "bytes": path.stat().st_size, "sha256": record["sha256"]})
    return {"strategy_id": v2.STRATEGY_ID, "evidence_label": EVIDENCE_LABEL, "v2_contract_sha256": v2.v2_contract_sha256(),
            "existing_es_mbo": {"path": str(es_path), "bytes": es_path.stat().st_size, "sha256": actual_es_sha256,
                                "snapshot_initialization": "SEALED_VALIDATED_R_A_F_SNAPSHOT_THROUGH_F_LAST"},
            "completion_manifest": str(completion_root / "acquisition-manifest.json"), "completion_files_verified": verified,
            "target_dates": list(TARGET_DATES), "august_7_excluded": True}


def _date_from_ns(timestamp_ns: int) -> str:
    return datetime.fromtimestamp(timestamp_ns / 1_000_000_000, tz=timezone.utc).date().isoformat()


def _paths(completion_root: Path, day: str) -> tuple[Path, Path]:
    profile = completion_root / "es_prior_rth_trades" / f"ESU6_{PRIOR_RTH[day]}_133000_200000_trades.dbn.zst"
    mes = completion_root / "mes_mbp1" / f"MESU6_{day}_133000_224501_mbp1.dbn.zst"
    return profile, mes


def _run_once(*, repository_root: Path, completion_root: Path) -> list[historical.HistoricalL2Runner]:
    """Process the shared ES MBO source once and route records by UTC session."""
    runners: dict[str, historical.HistoricalL2Runner] = {}
    adapters: dict[str, historical.HistoricalMBOToMBP10Adapter] = {}
    mes_next: dict[str, tuple[int, float, float] | None] = {}
    mes_iters: dict[str, Any] = {}
    cutoffs = {day: historical._clock_ns(day, historical.HARD_CUTOFF_SECONDS) for day in TARGET_DATES}
    closed: set[str] = set()
    for day in TARGET_DATES:
        profile, mes = _paths(completion_root, day)
        runners[day] = historical.HistoricalL2Runner(date=day, evidence_label=EVIDENCE_LABEL,
            levels=historical._profile_levels_from_declared_trades(profile), config=v2.V2_CONFIG,
            strategy_id=v2.STRATEGY_ID, require_native_mes_for_fallback=True)
        adapters[day] = historical.HistoricalMBOToMBP10Adapter()
        mes_iters[day] = iter(historical._stream_mes_quotes(mes))
        mes_next[day] = historical._next(mes_iters[day])
    records = 0
    for record in historical._stream_private_mbo(repository_root / ES_MBO_RELATIVE):
        day = _date_from_ns(record.timestamp_ns)
        if day > TARGET_DATES[-1]:
            break
        if day not in runners or day in closed:
            continue
        runner = runners[day]
        while mes_next[day] is not None and mes_next[day][0] < record.timestamp_ns:
            runner.observe_mes_quote(*mes_next[day])
            mes_next[day] = historical._next(mes_iters[day])
        if record.timestamp_ns >= cutoffs[day]:
            runner.force_flat_from_last_causal_cutoff_quote(cutoffs[day])
            runner.finish(cutoffs[day])
            closed.add(day)
            continue
        records += 1
        try:
            public = adapters[day].feed(record, materialize_public=record.timestamp_ns >= historical._clock_ns(day, historical.RTH_START_SECONDS))
        except historical.L2ValidationError as exc:
            raise AugustReplayError(f"invalid August MBO record day={day} timestamp_ns={record.timestamp_ns}") from exc
        if public is not None and public.timestamp_ns >= historical._clock_ns(day, historical.RTH_START_SECONDS):
            runner.observe_public(public)
        if records % 5_000_000 == 0:
            print(f"  August ES MBO records={records:,} completed={sum(len(item.interaction_ledger) for item in runners.values()):,}", flush=True)
    if closed != set(TARGET_DATES):
        raise AugustReplayError("sealed August ES MBO did not reach every frozen 22:45 cutoff")
    for day in TARGET_DATES:
        adapters[day].finish()
        runners[day].source_integrity_diagnostics = adapters[day].source_integrity_diagnostics()
    return [runners[day] for day in TARGET_DATES]


def _rows(runners: list[historical.HistoricalL2Runner], attribute: str) -> list[dict[str, Any]]:
    return [row for runner in runners for row in getattr(runner, attribute)]


def _subset(setups: list[dict[str, Any]], trades: list[dict[str, Any]], levels: set[str]) -> dict[str, Any]:
    return extended._subset(setups, trades, levels=levels)


def _write_results(*, output_root: Path, runners: list[historical.HistoricalL2Runner], verification: dict[str, Any]) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"August L2 V2 output already exists: {output_root}")
    historical.write_future_artifacts(output_root, runners, contract=v2.v2_contract())
    interactions, setups, trades = _rows(runners, "interaction_ledger"), _rows(runners, "setup_ledger"), _rows(runners, "trade_ledger")
    performance = historical._performance(trades)
    metrics = {"completed_interactions": len(interactions), "accepted_setups": sum(bool(row["accepted"]) for row in setups),
               "rejected_setups": sum(not bool(row["accepted"]) for row in setups),
               "confirmations_passed": sum(row.get("confirmation_timestamp_ns") is not None for row in setups),
               "confirmation_window_expiries": sum(row.get("terminal_reason") == "CONFIRMATION_WINDOW_EXPIRED" for row in setups),
               "active_position_blocks": sum(row.get("terminal_reason") == "COMPLIANCE_BLOCK_ACTIVE_POSITION" for row in setups),
               "unresolved_trades": 0, **performance}
    result = {"strategy_id": v2.STRATEGY_ID, "variant_label": VARIANT_LABEL, "evidence_label": EVIDENCE_LABEL,
              "v2_contract_sha256": v2.v2_contract_sha256(), "august_7_excluded": True, "input_verification": verification,
              "metrics": metrics, "breakdowns": {"day": historical._breakdown(trades, "date"), "direction": historical._breakdown(trades, "direction"),
                  "structural_level": historical._breakdown(trades, "level"), "instrument": historical._breakdown(trades, "instrument")},
              "poc_val_descriptive_subsets": {"PRIOR_RTH_POC": _subset(setups, trades, {"PRIOR_RTH_POC"}),
                  "PRIOR_RTH_VAL": _subset(setups, trades, {"PRIOR_RTH_VAL"}), "POC_PLUS_VAL": _subset(setups, trades, {"PRIOR_RTH_POC", "PRIOR_RTH_VAL"})},
              "interpretation": "SEEN_AUGUST_DATA_NOT_FRESH_OOS_EVIDENCE; POC_VAL_POST_RUN_DESCRIPTIVE_SUBSETS_ONLY"}
    (output_root / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_root / "diagnostic-report.md").write_text(
        f"# {v2.STRATEGY_ID} — August seen-data replay\n\nEvidence: `{EVIDENCE_LABEL}`. Aug 7 is excluded. "
        "This is the unchanged all-level strategy; POC/VAL are post-run descriptive subsets and do not alter position blocking.\n",
        encoding="utf-8",
    )
    return result


def _read_summary(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _comparison_row(name: str, payload: dict[str, Any] | None, *, unavailable: str | None = None) -> dict[str, Any]:
    if payload is None:
        return {"period": name, "status": unavailable, "all_level_trades": None, "all_level_total_r": None, "all_level_pf": None,
                "poc_trades": None, "poc_total_r": None, "poc_pf": None, "val_trades": None, "val_total_r": None, "val_pf": None,
                "poc_val_trades": None, "poc_val_total_r": None, "poc_val_pf": None, "completeness": unavailable}
    metrics = payload.get("metrics", payload.get("performance", {}))
    subsets = payload.get("poc_val_descriptive_subsets", payload.get("poc_val_descriptive_subset", {}))
    def subset(name: str) -> dict[str, Any]: return subsets.get(name, {})
    def metric(item: dict[str, Any], key: str) -> Any: return item.get(key, item.get("completed_trades") if key == "trades" else None)
    poc, val, combined = subset("PRIOR_RTH_POC"), subset("PRIOR_RTH_VAL"), subset("POC_PLUS_VAL")
    if not combined:
        # The published May artifact has independent POC/VAL breakdowns. This is
        # a descriptive union of its existing ledger results, never a rerun.
        combined = {
            "completed_trades": int(metric(poc, "trades") or 0) + int(metric(val, "trades") or 0),
            "total_r": float(poc.get("total_r", 0.0)) + float(val.get("total_r", 0.0)),
            "profit_factor": None,
        }
    return {"period": name, "status": "AVAILABLE", "all_level_trades": metric(metrics, "completed_trades"), "all_level_total_r": metrics.get("total_r"),
            "all_level_pf": metrics.get("profit_factor"), "poc_trades": metric(poc, "trades"),
            "poc_total_r": poc.get("total_r"), "poc_pf": poc.get("profit_factor"),
            "val_trades": metric(val, "trades"), "val_total_r": val.get("total_r"),
            "val_pf": val.get("profit_factor"), "poc_val_trades": metric(combined, "trades"),
            "poc_val_total_r": combined.get("total_r"), "poc_val_pf": combined.get("profit_factor"),
            "completeness": metrics.get("result_completeness", "COMPLETE")}


def _write_cross_period(*, repository_root: Path, output_root: Path, august: dict[str, Any]) -> None:
    try:
        may = extended._published_may_reference(repository_root)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        may = None
    retro = _read_summary(repository_root / "research_runs/CMEOrderflowAbsorption.ES_L2_V2_EXTENDED_EXISTING_DATA/retro_june_july/summary.json")
    rows = [_comparison_row("MAY", may), _comparison_row("RETRO_JUNE_JULY", retro, unavailable="NO_PUBLISHED_EXTENDED_RESULT"),
            _comparison_row("AUGUST_SEEN", august)]
    (output_root / "cross-period-descriptive.json").write_text(json.dumps({"rows": rows, "pooled_oos_claim": False}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output_root / "cross-period-descriptive.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def run(*, repository_root: Path, completion_root: Path, output_root: Path) -> dict[str, Any]:
    verification = verify_august_inputs(repository_root=repository_root, completion_root=completion_root)
    runners = _run_once(repository_root=repository_root, completion_root=completion_root)
    result = _write_results(output_root=output_root, runners=runners, verification=verification)
    _write_cross_period(repository_root=repository_root, output_root=output_root, august=result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay sealed L2 V2 Aug 3-6 local inputs once")
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--completion-root", type=Path, default=DEFAULT_COMPLETION_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(run(repository_root=args.repository_root, completion_root=args.completion_root, output_root=args.output_root), indent=2, sort_keys=True))
    except (AugustReplayError, historical.HistoricalReplayError, FileExistsError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
