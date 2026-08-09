import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]
MANIFEST = ROOT / "docs/research_pipeline/cme_orderflow_absorption_v1/oos-v1-data-manifest.json"


def load_manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


F_SNAPSHOT = 32
F_LAST = 128
OOS_START_NS = 1785715200000000000
OOS_END_NS = 1786147200000000000
INSTRUMENT_ID = 42140870


def audit_start_snapshot_records(records, deterministic_snapshot_receives):
    """Audit initialization eligibility without reconstructing a price/depth book."""
    completed_receives = set()
    active_receive = None
    for record in records:
        is_snapshot = bool(record["flags"] & F_SNAPSHOT)
        if record["ts_event"] < OOS_START_NS:
            if not (
                is_snapshot
                and record["ts_recv"] in deterministic_snapshot_receives
                and record["action"] in {"R", "A"}
                and record["instrument_id"] == INSTRUMENT_ID
            ):
                raise ValueError("pre-start ts_event is not valid Databento snapshot history")
        if is_snapshot:
            if record["instrument_id"] != INSTRUMENT_ID or record["action"] not in {"R", "A"}:
                raise ValueError("invalid snapshot record")
            if record["action"] == "R":
                active_receive = record["ts_recv"]
            if active_receive != record["ts_recv"]:
                raise ValueError("snapshot add is not preceded by its reset")
            if record["flags"] & F_LAST:
                completed_receives.add(active_receive)
                active_receive = None
            continue
        if active_receive is not None:
            raise ValueError("ordinary incremental record precedes complete snapshot")
        record["oos_eligible"] = (
            record["ts_event"] >= OOS_START_NS
            and record["ts_event"] < OOS_END_NS
            and record["ts_recv"] >= OOS_START_NS
            and record["ts_recv"] < OOS_END_NS
            and bool(completed_receives)
        )
    if active_receive is not None:
        raise ValueError("snapshot lacks F_LAST completion")
    return completed_receives


def test_oos_manifest_is_json_and_seals_only_the_first_post_pilot_week():
    manifest = load_manifest()
    chronology = manifest["chronology"]
    assert chronology["oos_signal_and_performance_interval"] == {
        "start": "2026-08-03T00:00:00Z",
        "end": "2026-08-08T00:00:00Z",
    }
    assert chronology["eligible_rth_dates"] == [
        "2026-08-03", "2026-08-04", "2026-08-05", "2026-08-06", "2026-08-07"
    ]
    assert len(chronology["eligible_rth_dates"]) == 5
    assert chronology["pilot"]["role"] == "DEVELOPMENT_DESIGN_ONLY"
    assert chronology["july_31_warmup"]["oos_excluded"] is True


def test_oos_manifest_records_authorized_identity_and_fails_closed_when_acquisition_is_unavailable():
    manifest = load_manifest()
    assert manifest["status"] in {"OOS_DATA_ACQUIRED_AND_SEALED", "OOS_DATA_INTEGRITY_REPAIR_REQUIRED"}
    assert manifest["provider"]["dataset"] == "GLBX.MDP3"
    assert manifest["provider"]["schema"] == "mbo"
    assert manifest["provider"]["stype_in"] == "raw_symbol"
    evidence = manifest["provider"]["metadata_symbology_evidence"]
    assert evidence["provenance"] == "Owner-obtained official Databento evidence"
    assert evidence["result"] == {"status": "OK", "raw_symbol": "ESU6", "instrument_id": 42140870, "partial": [], "not_found": []}
    assert manifest["symbol_and_instrument"]["resolved_raw_symbol"] == "ESU6"
    assert manifest["symbol_and_instrument"]["instrument_ids"] == [42140870]
    assert manifest["rollover"]["resolved"] is True
    assert manifest["rollover"]["contracts"] == ["ESU6"]
    assert manifest["authorized_cost_quote"]["quote_count"] == 1
    assert manifest["authorized_cost_quote"]["estimated_usd"] == 5.736491556466
    assert manifest["proposed_acquisition"]["target_path"] == "data/cme_orderflow_absorption_v1/oos_v1/ESU6/mbo/ESU6_2026-08-03_2026-08-08_mbo.dbn"
    acquisition = manifest["proposed_acquisition"]
    assert manifest["status"] == "OOS_DATA_ACQUIRED_AND_SEALED"
    assert acquisition["integrity_pass"] is True
    assert acquisition["file_sha256"] == "BE4B56639E56DF9AACE81621E4E276463EA8AF889104F35F1744400310D53AA3"
    assert acquisition["file_bytes"] == 1068425668
    assert acquisition["record_count"] == 61106259
    assert acquisition["observed_instrument_ids"] == [42140870]
    assert acquisition["coverage_gaps"] == []
    assert manifest["data_acquired"] is True
    assert manifest["no_strategy_execution_or_pnl"] is True
    assert manifest["fixed_usd_risk_budget"] == 250.00
    assert manifest["stop_buffer_ticks"] == 5


def test_oos_manifest_preserves_frozen_plus_selection_without_refit():
    manifest = load_manifest()
    frozen = manifest["frozen_strategy"]
    assert frozen["absorption_p95"] == 0.7977986403366786
    assert frozen["replenishment_p95"] == 0.7785691162188411
    assert "no refit" in frozen["constraints"][0]
    assert "No strategy execution, backtest, PnL calculation" in frozen["constraints"][1]


def test_old_ts_event_with_snapshot_at_initial_receive_is_accepted():
    records = [
        {"ts_event": OOS_START_NS - 1, "ts_recv": OOS_START_NS, "action": "R", "flags": F_SNAPSHOT, "instrument_id": INSTRUMENT_ID},
        {"ts_event": OOS_START_NS - 1, "ts_recv": OOS_START_NS, "action": "A", "flags": F_SNAPSHOT | F_LAST, "instrument_id": INSTRUMENT_ID},
    ]
    assert audit_start_snapshot_records(records, {OOS_START_NS}) == {OOS_START_NS}
    assert "oos_eligible" not in records[0]
    assert "oos_eligible" not in records[1]


def test_old_ts_event_without_snapshot_is_rejected():
    records = [{"ts_event": OOS_START_NS - 1, "ts_recv": OOS_START_NS, "action": "A", "flags": 0, "instrument_id": INSTRUMENT_ID}]
    with pytest.raises(ValueError, match="pre-start"):
        audit_start_snapshot_records(records, {OOS_START_NS})


def test_snapshots_cannot_form_oos_eligibility_and_post_snapshot_incremental_is_gated():
    records = [
        {"ts_event": OOS_START_NS - 1, "ts_recv": OOS_START_NS, "action": "R", "flags": F_SNAPSHOT, "instrument_id": INSTRUMENT_ID},
        {"ts_event": OOS_START_NS - 1, "ts_recv": OOS_START_NS, "action": "A", "flags": F_SNAPSHOT | F_LAST, "instrument_id": INSTRUMENT_ID},
        {"ts_event": OOS_START_NS + 1, "ts_recv": OOS_START_NS + 1, "action": "A", "flags": 0, "instrument_id": INSTRUMENT_ID},
    ]
    audit_start_snapshot_records(records, {OOS_START_NS})
    assert all("oos_eligible" not in snapshot for snapshot in records[:2])
    assert records[2]["oos_eligible"] is True


def test_complete_snapshot_is_required_before_ordinary_incremental_data():
    records = [
        {"ts_event": OOS_START_NS - 1, "ts_recv": OOS_START_NS, "action": "R", "flags": F_SNAPSHOT, "instrument_id": INSTRUMENT_ID},
        {"ts_event": OOS_START_NS + 1, "ts_recv": OOS_START_NS + 1, "action": "A", "flags": 0, "instrument_id": INSTRUMENT_ID},
    ]
    with pytest.raises(ValueError, match="precedes complete snapshot"):
        audit_start_snapshot_records(records, {OOS_START_NS})
