from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from research_pipeline.cme_orderflow_absorption_v1 import may_2026_retrospective_holdout as may
from research_pipeline.cme_orderflow_absorption_v1 import v2_retro_holdout_runner as v2
from research_pipeline.cme_orderflow_absorption_v1 import v3_tick_trigger_target_matrix as v3


def _manifest_fixture(root: Path) -> None:
    files = {}
    for day in may.TARGET_DAYS:
        for label, relative in may._expected_relative_paths(day).items():
            path = root / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(relative.encode())
            files[relative] = {
                "label": label,
                "schema": {"ES_MBO_L3": "mbo", "MES_NATIVE_EXECUTION": "mbp-1", "ES_PRIOR_RTH_PROFILE": "trades"}[label],
                "session_date": day if label != "ES_PRIOR_RTH_PROFILE" else may.PRIOR_RTH[day],
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    (root / may.MANIFEST_NAME).write_text(json.dumps({"manifest_kind": "MAY_2026_ES_MES_COST_PROXY_ACQUISITION", "data_acquired": True,
        "request_identity": {"target_rth_dates": list(may.TARGET_DAYS)}, "files": files}), encoding="utf-8")


def test_may_manifest_requires_exact_15_by_3_verified_package(tmp_path):
    _manifest_fixture(tmp_path)
    verified = may.verify_manifest(tmp_path)
    assert len(verified["verified_files"]) == 45
    assert [item["relative_path"] for item in verified["verified_files"] if item["label"] == "ES_MBO_L3"] == sorted(
        may._expected_relative_paths(day)["ES_MBO_L3"] for day in may.TARGET_DAYS)
    first = tmp_path / may._expected_relative_paths(may.TARGET_DAYS[0])["ES_MBO_L3"]
    first.write_bytes(b"corrupt")
    with pytest.raises(may.MayHoldoutError, match="size/schema mismatch|SHA-256"):
        may.verify_manifest(tmp_path)


def test_v2_and_v3_specs_are_frozen_and_l2_minimum_delay_is_absent():
    assert may.EVIDENCE_LABEL == "UNSEEN_MAY_2026_RETROSPECTIVE_HOLDOUT"
    assert v2.CONFIRMATION_NS == 15_000_000_000 and v2.TARGET_R == 2.5 and v2.ENTRY_LATENCY_NS == 2_000_000
    cell = v3.Cell(v3.MatrixSpec(3, 3.0), may.EVIDENCE_LABEL, "NATIVE_MES_MBP1_FALLBACK")
    row = {"interaction_id": "x", "date": "2026-05-04", "interaction_end": 1_000_000_000,
           "end_price": 5_000_000_000_000, "direction": "BUYER_ABSORPTION", "level": "PRIOR_RTH_POC",
           "zone_low": 4_999_000_000_000, "zone_high": 5_001_000_000_000,
           "absorption_score": 1.0, "replenishment_score": 1.0}
    cell.add_signals([row]); cell.observe_execution(1_000_000_000, 5_000_750_000_000)
    assert cell.pending["x"].state == "AWAITING_ES_ENTRY"  # V3 has no L2 +5s delay.


def test_native_mes_path_and_single_point_conversion_are_preserved():
    cell = v3.Cell(v3.MatrixSpec(3, 3.0), may.EVIDENCE_LABEL, "NATIVE_MES_MBP1_FALLBACK")
    row = {"interaction_id": "x", "date": "2026-05-04", "interaction_end": 0, "end_price": 5_000_000_000_000,
           "direction": "BUYER_ABSORPTION", "level": "PRIOR_RTH_POC", "zone_low": 4_990_000_000_000,
           "zone_high": 5_001_000_000_000, "absorption_score": 1.0, "replenishment_score": 1.0}
    cell.add_signals([row]); cell.observe_execution(0, 5_000_750_000_000)
    cell.pending["x"].entry_ready_ns = 0
    cell.try_retro_es_entry(0, (5000.0, 5000.25))
    assert cell.pending["x"].state == "AWAITING_MES_ENTRY"
    cell.try_retro_mes_entry(1, (5000.0, 5000.25))
    assert cell.position is not None and cell.position.instrument == "MES"
    assert v2.interaction_zone_points(row) == (4990.0, 5001.0)


def test_hard_cutoff_closes_with_existing_adverse_fill_semantics():
    cell = v3.Cell(v3.MatrixSpec(3, 3.0), may.EVIDENCE_LABEL, "NATIVE_MES_MBP1_FALLBACK")
    row = {"interaction_id": "x", "date": "2026-05-04", "interaction_end": 0, "end_price": 5_000_000_000_000,
           "direction": "BUYER_ABSORPTION", "level": "PRIOR_RTH_POC", "zone_low": 4_999_000_000_000,
           "zone_high": 5_001_000_000_000, "absorption_score": 1.0, "replenishment_score": 1.0}
    cell.add_signals([row]); cell.observe_execution(0, 5_000_750_000_000); cell.pending["x"].entry_ready_ns = 0
    cell.try_retro_es_entry(0, (5000.0, 5000.25))
    assert cell.position is not None
    may._close_v3_at_cutoff(cell, 99, (5001.0, 5001.25))
    assert cell.position is None and cell.trades[0]["exit_reason"] == "CUTOFF_FORCED_FLAT"


def test_two_strategy_states_are_independent():
    first = v3.Cell(v3.MatrixSpec(3, 3.0), may.EVIDENCE_LABEL, "NATIVE_MES_MBP1_FALLBACK")
    second = v3.Cell(v3.MatrixSpec(3, 3.0), may.EVIDENCE_LABEL, "NATIVE_MES_MBP1_FALLBACK")
    assert first.pending is not second.pending and first.trades is not second.trades
