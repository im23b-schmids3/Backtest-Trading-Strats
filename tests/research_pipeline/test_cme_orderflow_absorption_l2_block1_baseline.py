from __future__ import annotations

from pathlib import Path

import pytest

from research_pipeline.cme_orderflow_absorption_l2_v1 import block1_baseline as block1


def test_block1_split_is_exact_and_disjoint() -> None:
    assert len(block1.TRAIN_DATES) == 38
    assert len(block1.VALIDATION_DATES) == 35
    assert "2025-12-24" not in block1.TRAIN_DATES
    assert "2025-12-26" not in block1.TRAIN_DATES
    assert "2025-12-31" not in block1.TRAIN_DATES
    assert set(block1.TRAIN_DATES).isdisjoint(block1.VALIDATION_DATES)
    assert not any(day.startswith("2026-09") for day in (*block1.TRAIN_DATES, *block1.VALIDATION_DATES))
    assert block1.validate_split("train")["count"] == 38
    assert block1.validate_split("validation")["count"] == 35


def test_block1_records_explicit_exclusions() -> None:
    exclusions = {row["date"]: row["reason"] for row in block1.validate_split("train")["excluded_dates"]}
    assert exclusions == {
        "2025-12-01": "PRIOR_ASIA_EUROPE_PROFILE_DEPENDENCY_INCOMPLETE",
        "2025-12-24": "ASIA_HARD_FLAT_FRESH_BBO_EVIDENCE_UNAVAILABLE",
        "2025-12-26": "ASIA_HARD_FLAT_FRESH_BBO_EVIDENCE_UNAVAILABLE",
        "2025-12-31": "ASIA_HARD_FLAT_FRESH_BBO_EVIDENCE_UNAVAILABLE",
    }
    assert block1.TRAIN_PROFILE_ONLY_CONTEXT_DATES == ("2025-12-24", "2025-12-26", "2025-12-31")


def test_block1_rejects_wrong_or_duplicate_split_dates() -> None:
    with pytest.raises(block1.Block1BaselineError):
        block1.validate_split("train", [block1.TRAIN_DATES[0], block1.TRAIN_DATES[0]])
    with pytest.raises(block1.Block1BaselineError):
        block1.validate_split("train", [*block1.TRAIN_DATES[:-1], "2026-09-01"])
    with pytest.raises(block1.Block1BaselineError):
        block1.validate_split("other")


def test_block1_config_integrity_preserves_distinct_session_contracts() -> None:
    report = block1._config_integrity()
    configs = report["configs"]
    assert configs["asia_w04"]["config_id"] == "W04-02-06-04-04-Q45"
    assert configs["europe_w04"]["config_id"] == "W04-02-06-04-04-Q45"
    assert configs["asia_w04"]["quality_threshold"] == "0.45"
    assert configs["europe_w04"]["quality_threshold"] == "0.45"
    assert configs["ny_berlin"]["contract_hash"]
    assert len(report["config_hashes"]) == 3


def test_block1_aggregate_uses_existing_accounting_without_strategy_logic() -> None:
    trades = [
        {"r_multiple": "2.0", "net_pnl_usd": "100", "exit_reason": "TARGET", "date": "2025-12-02", "exit_timestamp_ns": 2, "trade_id": "a", "instrument": "ES"},
        {"r_multiple": "-1.0", "net_pnl_usd": "-50", "exit_reason": "STOP", "date": "2025-12-03", "exit_timestamp_ns": 3, "trade_id": "b", "instrument": "MES"},
    ]
    sessions = [
        {"session_date": "2025-12-02", "total_r": 2.0},
        {"session_date": "2025-12-03", "total_r": -1.0},
    ]
    result = block1._aggregate([{"family": "synthetic"}], trades, sessions)
    assert result["sessions"] == 2
    assert result["trades"] == 2
    assert result["winners"] == 1
    assert result["losses"] == 1
    assert result["net_r"] == 1.0
    assert result["profitable_session_ratio"] == 0.5


def test_source_integrity_fails_closed_for_missing_block1_artifact(tmp_path: Path) -> None:
    with pytest.raises(block1.Block1BaselineError, match="required Block 1 source artifact"):
        block1.source_integrity(tmp_path)


def test_missing_ny_dates_are_manifest_bound_source_ready() -> None:
    dates = (
        "2025-12-05", "2025-12-18", "2025-12-19", "2025-12-22", "2025-12-29",
        "2026-01-02", "2026-01-05", "2026-01-06", "2026-01-08", "2026-01-14",
        "2026-01-15", "2026-01-16", "2026-01-20", "2026-01-21", "2026-01-22",
        "2026-01-26",
    )
    audit = block1.audit_ny_source_paths(Path.cwd(), dates)
    assert audit["status"] == "PASS"
    assert audit["classification_counts"] == {"NY_SOURCE_READY": 16}
    assert all(not row["failures"] for row in audit["dates"])


def test_active_contract_roll_is_deterministic() -> None:
    assert block1._active_contracts("2025-12-16") == ("ESZ5", "MESZ5")
    assert block1._active_contracts("2025-12-17") == ("ESH6", "MESH6")


def test_subset_loader_rejects_extra_or_missing_dates(tmp_path: Path) -> None:
    manifest = {
        "status": "BERLIN_TRAIN_DATE_SUBSET_COMPLETE",
        "execution_contract_sha256": block1.berlin.CONTRACT_SHA256,
        "dates": ["2025-12-05"],
    }
    (tmp_path / "manifest.json").write_text(__import__("json").dumps(manifest), encoding="utf-8")
    with pytest.raises(block1.berlin.CorrectedAllPeriodError, match="missing or invalid JSON"):
        block1.berlin.load_date_subset(Path.cwd(), tmp_path, ("2025-12-05",))
