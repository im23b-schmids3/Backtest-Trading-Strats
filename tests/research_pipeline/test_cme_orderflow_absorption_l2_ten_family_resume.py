from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from research_pipeline.cme_orderflow_absorption_l2_v1 import (
    mac_2025_candidate_tape as candidate_tape,
    ten_family_dec_jan_robustness as robustness,
)


def _make_artifact(tmp_path):
    day = "2025-12-02"
    tape_dir = tmp_path / "tapes"
    tape_dir.mkdir()
    tape_path = tape_dir / f"{day}-{candidate_tape.TAPE_FILENAME}"
    source_paths = [str(tmp_path / "source-a.dbn"), str(tmp_path / "source-b.dbn")]
    metadata = {
        "tape_version": candidate_tape.TAPE_VERSION,
        "date": day,
        "source_paths": source_paths,
        "source_sha256": "source-digest",
        "semantic_sha256": "semantic-digest",
        "feature_names": list(candidate_tape.FEATURE_NAMES),
        "candidate_count": 0,
        "event_count": 0,
        "session_order": ["ASIA", "EUROPE", "NY"],
        "completed_strategy_sessions": ["ASIA", "EUROPE", "NY"],
        "final_strategy_window_end_ns": 20,
        "source_last_timestamp_ns": 21,
        "last_strategy_timestamp_ns": 19,
        "book_state_at_last_strategy_record": "EXECUTABLE",
        "post_session_terminal_state_accepted": False,
        "available_families": list(robustness.EXPECTED_FAMILIES),
        "bbo_path_complete": True,
    }
    tape = candidate_tape.CandidateTape(
        metadata=metadata,
        candidates=(),
        events=np.zeros(0, dtype=candidate_tape.EVENT_DTYPE),
    )
    candidate_tape.write_tape(tape_path, tape)
    robustness._write_json(candidate_tape._tape_manifest_path(tape_path), metadata)
    job = {
        "day": day,
        "source_paths": source_paths,
        "source_sha256": "source-digest",
        "semantic_sha256": "semantic-digest",
        "tape_path": str(tape_path),
    }
    record = {
        "date": day,
        "tape_path": str(tape_path.relative_to(tmp_path)),
        "tape_sha256": robustness._sha256(tape_path),
        "candidate_count": 0,
        "event_count": 0,
        "source_sha256": "source-digest",
        "strict_route_equivalence": {"status": "PASS"},
    }
    return tape_path, job, record


def test_resume_audit_accepts_hash_bound_complete_tape(tmp_path):
    tape_path, job, record = _make_artifact(tmp_path)

    report = robustness._validate_existing_tape(tmp_path, job, record)

    assert report["date"] == "2025-12-02"
    assert report["tape_sha256"] == robustness._sha256(tape_path)
    assert report["candidate_count"] == 0
    assert report["event_count"] == 0
    assert report["strict_route_equivalence"] == {"status": "PASS"}


def test_resume_audit_rejects_recorded_tape_hash_mismatch(tmp_path):
    _, job, record = _make_artifact(tmp_path)
    record["tape_sha256"] = "not-the-file-hash"

    with pytest.raises(robustness.RobustnessRunError, match="SHA-256 mismatch"):
        robustness._validate_existing_tape(tmp_path, job, record)


def test_resume_inventory_identifies_missing_date_without_touching_valid_tape(tmp_path):
    tape_path, job, record = _make_artifact(tmp_path)
    progress_identity = {
        "config_sha256": "config", "semantic_sha256": "semantic-digest",
        "profile_contract_sha256": "profiles", "source_manifest_sha256": {"base": "b", "dec_jan_extension": "e"},
        "workers": 1,
    }
    robustness._write_json(tmp_path / "progress.json", {
        **progress_identity, "status": "BUILDING_CANDIDATE_TAPES", "completed_sessions": 1,
        "total_sessions": 2, "last_completed_date": "2025-12-02", "tapes": [record],
    })
    second_day = "2025-12-03"
    second_path = tape_path.with_name(f"{second_day}-{candidate_tape.TAPE_FILENAME}")
    jobs = {
        "2025-12-02": job,
        second_day: {
            **job, "day": second_day,
            "tape_path": str(second_path),
        },
    }

    reusable, invalid = robustness._resume_tape_inventory(
        tmp_path, tmp_path, ["2025-12-02", second_day], jobs, progress_identity,
    )

    assert list(reusable) == ["2025-12-02"]
    assert invalid == []
    assert tape_path.is_file()
    assert not second_path.exists()


def test_resume_rejects_unrecognized_existing_tape_file(tmp_path):
    tape_path, job, record = _make_artifact(tmp_path)
    progress_identity = {
        "config_sha256": "config", "semantic_sha256": "semantic-digest",
        "profile_contract_sha256": "profiles", "source_manifest_sha256": {"base": "b", "dec_jan_extension": "e"},
        "workers": 1,
    }
    robustness._write_json(tmp_path / "progress.json", {
        **progress_identity, "status": "BUILDING_CANDIDATE_TAPES", "completed_sessions": 1,
        "total_sessions": 1, "last_completed_date": record["date"], "tapes": [record],
    })
    (tape_path.parent / "unrecognized.part").write_bytes(b"stale")

    with pytest.raises(robustness.RobustnessRunError, match="unrecognized files"):
        robustness._resume_tape_inventory(
            tmp_path, tmp_path, [record["date"]], {record["date"]: job}, progress_identity,
        )


def test_scheduled_early_close_is_excluded_but_normal_full_date_is_eligible():
    early = robustness._classify_session_coverage(
        "2025-12-24",
        contract={
            "source_model": "NATIVE_MBP10", "raw_symbol": "ESH6",
            "requested_source_end": "2025-12-24T18:15:01Z",
            "scheduled_session_end": "2025-12-24T18:15:00Z",
            "scheduled_early_close": True,
        },
        source_last_timestamp_ns=robustness.baseline._ns("2025-12-24T18:15:00.900Z"),
        normal_required_final_session_end="2025-12-24T21:00:00Z",
    )
    normal = robustness._classify_session_coverage(
        "2025-12-26",
        contract={
            "source_model": "NATIVE_MBP10", "raw_symbol": "ESH6",
            "requested_source_end": "2025-12-26T22:45:01Z",
            "scheduled_session_end": "2025-12-26T22:45:00Z",
            "scheduled_early_close": False,
        },
        source_last_timestamp_ns=robustness.baseline._ns("2025-12-26T21:59:59Z"),
        normal_required_final_session_end="2025-12-26T21:00:00Z",
    )

    assert early["session_type"] == "SCHEDULED_EARLY_CLOSE"
    assert early["eligibility"] == "EXCLUDED"
    assert early["exclusion_reason"] == robustness.NON_STANDARD_SESSION_REASON
    assert normal["session_type"] == "NORMAL_FULL_SESSION"
    assert normal["eligibility"] == "ELIGIBLE"


def test_early_ending_normal_date_fails_closed_instead_of_being_excluded():
    result = robustness._classify_session_coverage(
        "2025-12-26",
        contract={
            "source_model": "NATIVE_MBP10", "raw_symbol": "ESH6",
            "requested_source_end": "2025-12-26T22:45:01Z",
            "scheduled_session_end": "2025-12-26T22:45:00Z",
            "scheduled_early_close": False,
        },
        source_last_timestamp_ns=robustness.baseline._ns("2025-12-26T20:59:59.999Z"),
        normal_required_final_session_end="2025-12-26T21:00:00Z",
    )

    assert result["session_type"] == "INCOMPLETE_NORMAL_SESSION"
    assert result["eligibility"] == "FAIL_CLOSED"
    assert result["failure_reason"] == "NORMAL_SESSION_SOURCE_ENDS_BEFORE_FROZEN_NY_WINDOW"


def test_dec24_remains_available_as_causal_profile_context_for_dec26():
    dates = ["2025-12-23", "2025-12-24", "2025-12-26"]
    profiles = {day: object() for day in dates}
    prior = robustness._prior_profiles_by_date(dates, profiles, object())

    assert prior["2025-12-26"] is profiles["2025-12-24"]


def test_frozen_calendar_marks_only_dec24_as_shortened_in_42_date_block():
    sessions = robustness.quote_contract.build_sessions(
        robustness.date(2025, 12, 1), robustness.date(2026, 1, 30),
    )

    assert len(sessions) == 42
    assert [item.session_date.isoformat() for item in sessions if item.shortened] == ["2025-12-24"]


def test_session_audit_uses_measured_source_end_and_preserves_evidence_paths(monkeypatch):
    dates = ["2025-12-24", "2025-12-26"]
    contracts = {
        "2025-12-24": {
            "source_model": "NATIVE_MBP10", "raw_symbol": "ESH6",
            "requested_source_end": "2025-12-24T18:15:01Z",
            "scheduled_session_end": "2025-12-24T18:15:00Z", "scheduled_early_close": True,
        },
        "2025-12-26": {
            "source_model": "NATIVE_MBP10", "raw_symbol": "ESH6",
            "requested_source_end": "2025-12-26T22:45:01Z",
            "scheduled_session_end": "2025-12-26T22:45:00Z", "scheduled_early_close": False,
        },
    }
    measured = {
        "2025-12-24": robustness.baseline._ns("2025-12-24T18:15:00Z"),
        "2025-12-26": robustness.baseline._ns("2025-12-26T21:59:59Z"),
    }
    monkeypatch.setattr(
        robustness, "_scan_source_last_timestamp_ns",
        lambda paths, *, expected_end: measured[Path(paths[0]).name[:10]],
    )
    sources = {
        day: (Path(f"{day}-pre.dbn"), Path(f"{day}-session.dbn")) for day in dates
    }
    manifest = {"session_contract_by_date": contracts}

    audit = robustness._session_eligibility_audit(dates, sources, manifest, {})

    assert audit["normal_full_session_dates"] == ["2025-12-26"]
    assert audit["non_standard_session_excluded_dates"] == ["2025-12-24"]
    assert audit["incomplete_normal_session_dates"] == []
    assert audit["per_date"][0]["source_last_timestamp"] == "2025-12-24T18:15:00Z"
    assert audit["per_date"][0]["source_paths"] == [str(p) for p in sources["2025-12-24"]]
