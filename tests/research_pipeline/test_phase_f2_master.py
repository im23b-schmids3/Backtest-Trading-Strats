from __future__ import annotations

from pathlib import Path

from research_pipeline.phase_f1.service import MasterPipelineService
from research_pipeline.phase_f1.models import FinalClassification


ROOT = Path(__file__).parents[2]
INTAKE = ROOT / "configs/research_pipeline/phase_f2_real_demo_intake.yaml"
SPEC = ROOT / "research_registry/spec_drafts/F2-real-breakout-demo_vphase-b-1.yaml"


def test_real_mode_is_explicit_and_approval_gated(tmp_path):
    service = MasterPipelineService(tmp_path / "registry.sqlite3", ROOT)
    options = service.input_model(INTAKE, ROOT, registry_path=tmp_path / "registry.sqlite3", dry_run=False, mode="real_run", allow_proxy_data=True)
    options = options.model_copy(update={"prebuilt_spec_path": str(SPEC)})
    started = service.start(options)
    assert started["current_step"] == "APPROVAL"
    assert started["approval_status"] == "PENDING"
    assert service.resume(started["run_id"])["current_step"] == "APPROVAL"


def test_real_demo_completes_without_synthetic_phase_adapter(tmp_path):
    service = MasterPipelineService(tmp_path / "registry.sqlite3", ROOT)
    options = service.input_model(INTAKE, ROOT, registry_path=tmp_path / "registry.sqlite3", dry_run=False, mode="real_run", allow_proxy_data=True)
    options = options.model_copy(update={"prebuilt_spec_path": str(SPEC)})
    started = service.start(options); run_id = started["run_id"]
    service.approve(run_id, "APPROVE", "bounded deterministic real-data fixture")
    final = service.resume(run_id)
    assert final["current_step"] == "COMPLETED"
    assert final["outcome"] == "SUCCESS"
    report = service.report(run_id)
    assert report["mode"] == "real_run"
    assert report["classification"] in {item.value for item in FinalClassification}
    assert report["confidence"] == "REAL_LOCAL_DATA_DEMONSTRATION"


def test_portfolio_deferral_does_not_override_standalone_precedence():
    assert MasterPipelineService._classification("ACCEPTED_STANDALONE", "PROP_ACCEPTED_STANDALONE", "INSUFFICIENT_EVIDENCE", real_mode=True) == FinalClassification.ACCEPTED_STANDALONE
