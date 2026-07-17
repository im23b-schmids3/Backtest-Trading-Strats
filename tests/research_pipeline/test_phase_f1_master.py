from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_pipeline.errors import SpecificationValidationError
from research_pipeline.phase_f1.models import FinalClassification, IntakeSpec, MasterRunStatus
from research_pipeline.phase_f1.service import MasterPipelineService


def write_intake(tmp_path: Path, *, name: str = "F1 synthetic strategy", market: str = "BTCUSDT", ambiguous: bool = False) -> Path:
    path = tmp_path / "intake.yaml"
    path.write_text(json.dumps({
        "strategy_name": name,
        "description": "A deterministic fictional strategy for the Phase F1 orchestration fixture.",
        "markets": [market],
        "timeframes": ["1h"],
        "entry_logic": ["Use the described fictional trend condition."],
        "exit_logic": ["Exit at the fictional target or stop."],
        "risk_model": "Fixed synthetic risk.",
        "position_sizing": "One synthetic unit.",
        "filters": ["Chronological timestamps only."],
        "optional_notes": "No live execution.",
        "unknown_fields": {},
        "confidence_flags": ["SYNTHETIC_FIXTURE_ONLY"],
        "ambiguities": ["entry timing is unclear"] if ambiguous else [],
        "confirmed_facts": ["This is a fixture."],
        "assumptions": ["Synthetic adapters provide evidence."],
        "missing_information": [],
    }), encoding="utf-8")
    return path


def service(tmp_path: Path) -> MasterPipelineService:
    return MasterPipelineService(tmp_path / "registry.sqlite3", Path.cwd())


def test_intake_schema_distinguishes_evidence_and_rejects_invalid_input():
    model = IntakeSpec(strategy_name="fixture", description="A sufficiently detailed fixture description.", markets=["TEST"], timeframes=["1h"])
    assert model.confirmed_facts
    with pytest.raises(ValueError):
        IntakeSpec.model_validate({"strategy_name": "x", "markets": ["TEST"], "timeframes": ["1h"]})


def test_run_stops_at_the_single_approval_gate_and_survives_restart(tmp_path: Path):
    intake = write_intake(tmp_path, name=f"F1 approval {tmp_path.name}")
    first = service(tmp_path).start(service(tmp_path).input_model(intake, Path.cwd(), registry_path=tmp_path / "registry.sqlite3"))
    assert first["outcome"] == MasterRunStatus.WAITING_FOR_APPROVAL.value
    assert first["approval_status"] == "PENDING"
    restarted = service(tmp_path).status(first["run_id"])
    assert restarted["run_id"] == first["run_id"]
    assert restarted["current_step"] == "APPROVAL"


def test_ambiguous_intake_requires_clarification_before_approval(tmp_path: Path):
    intake = write_intake(tmp_path, name=f"F1 ambiguous {tmp_path.name}", ambiguous=True)
    current = service(tmp_path)
    status = current.start(current.input_model(intake, Path.cwd(), registry_path=tmp_path / "registry.sqlite3"))
    assert status["outcome"] == MasterRunStatus.MANUAL_REVIEW_REQUIRED.value
    with pytest.raises(SpecificationValidationError):
        current.approve(status["run_id"])


def test_synthetic_end_to_end_generates_report_archive_and_hashes(tmp_path: Path):
    intake = write_intake(tmp_path, name=f"F1 end to end {tmp_path.name}")
    current = service(tmp_path)
    options = current.input_model(intake, Path.cwd(), registry_path=tmp_path / "registry.sqlite3", dry_run=True)
    pending = current.start(options)
    current.approve(pending["run_id"], note="fixture approval")
    completed = current.resume(pending["run_id"])
    assert completed["outcome"] == MasterRunStatus.SUCCESS.value
    assert completed["current_step"] == "COMPLETED"
    report = current.report(pending["run_id"])
    assert report["classification"] == FinalClassification.INSUFFICIENT_EVIDENCE.value
    assert report["artifacts"]
    assert all(item["sha256"] for item in report["artifacts"])
    assert Path(report["artifacts"][0]["path"]).is_file()
    assert (Path(completed["root_path"]) / "archive" / "manifest.json").is_file()


def test_resume_is_idempotent_after_completion(tmp_path: Path):
    intake = write_intake(tmp_path, name=f"F1 resume {tmp_path.name}")
    current = service(tmp_path); options = current.input_model(intake, Path.cwd(), registry_path=tmp_path / "registry.sqlite3")
    pending = current.start(options); current.approve(pending["run_id"])
    first = current.resume(pending["run_id"]); second = current.resume(pending["run_id"])
    assert first["outcome"] == second["outcome"] == MasterRunStatus.SUCCESS.value
    assert first["journal_entries"] == second["journal_entries"]


def test_rejection_aborts_without_implementation(tmp_path: Path):
    intake = write_intake(tmp_path, name=f"F1 rejected {tmp_path.name}")
    current = service(tmp_path); options = current.input_model(intake, Path.cwd(), registry_path=tmp_path / "registry.sqlite3")
    pending = current.start(options); rejected = current.approve(pending["run_id"], "REJECT", "fixture rejection")
    assert rejected["outcome"] == MasterRunStatus.ABORTED.value
    assert not any(item["phase"] == "IMPLEMENTATION" for item in rejected["phase_results"])


def test_cli_status_report_and_artifact_commands(tmp_path: Path, capsys):
    from research_pipeline.cli import main
    intake = write_intake(tmp_path, name=f"F1 cli {tmp_path.name}")
    registry = str(tmp_path / "registry.sqlite3")
    assert main(["--registry", registry, "run", str(intake), "--repository-root", str(Path.cwd())]) == 0
    pending = json.loads(capsys.readouterr().out)
    assert main(["--registry", registry, "status", pending["run_id"]]) == 0
    assert "APPROVAL" in capsys.readouterr().out
    assert main(["--registry", registry, "approve", pending["run_id"]]) == 0
    capsys.readouterr()
    assert main(["--registry", registry, "resume", pending["run_id"]]) == 0
    capsys.readouterr()
    assert main(["--registry", registry, "report", pending["run_id"]]) == 0
    assert "classification" in capsys.readouterr().out
    assert main(["--registry", registry, "artifacts", pending["run_id"]]) == 0
    assert "artifact_hash" in capsys.readouterr().out
