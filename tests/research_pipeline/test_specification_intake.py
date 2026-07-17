from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from research_pipeline.phase_b.models import CodexExecutionResult, WorkflowInput
from research_pipeline.phase_b.services import PhaseBService
from research_pipeline.validation.specification_semantics import SpecificationProvenance, semantic_validate


class SequenceCodex:
    def __init__(self, outputs: list[str]):
        self.outputs = list(outputs)
        self.calls = 0

    def run(self, prompt: str, cwd: str, *, sandbox: str, dry_run: bool) -> CodexExecutionResult:
        self.calls += 1
        output = self.outputs.pop(0) if self.outputs else "not a specification"
        return CodexExecutionResult(success=True, executed=True, command=["codex", "exec"], cwd=cwd,
                                    sandbox=sandbox, exit_code=0, stdout=output, stderr="", duration_ms=1,
                                    timed_out=False)


def workflow(root: Path, **changes: object) -> WorkflowInput:
    value: dict[str, object] = {
        "strategy_name": "intake-fixture",
        "natural_language_description": "A deterministic fictional strategy used to test specification intake.",
        "requested_markets": ["TEST"],
        "requested_timeframes": ["1h"],
        "repository_root": str(root),
        "dry_run": False,
        "implementation_enabled": False,
    }
    value.update(changes)
    return WorkflowInput.model_validate(value)


def valid_payload(tmp_path: Path, strategy_name: str = "intake-fixture") -> dict:
    service = PhaseBService(tmp_path / "seed.sqlite3")
    fixture = service._dry_spec(workflow(tmp_path, strategy_name=strategy_name, dry_run=True), strategy_name, "phase-b-1", [])
    return fixture.model_dump(mode="json")


def payload_text(payload: dict) -> str:
    return yaml.safe_dump(payload, sort_keys=False)


def test_valid_first_attempt_is_canonical_and_duplicate_call_is_cached(tmp_path: Path) -> None:
    payload = payload_text(valid_payload(tmp_path))
    runner = SequenceCodex([payload])
    service = PhaseBService(tmp_path / "registry.sqlite3", codex_runner=runner)
    request = workflow(tmp_path)
    generated = service.generate_spec(request)
    assert generated.attempt == 1
    assert service.validate_spec(generated).approval_ready
    again = service.generate_spec(request)
    assert again.specification_hash == generated.specification_hash
    assert runner.calls == 1
    run_id = service._intake_run_id(request, "intake-fixture")
    assert service.specification_status(run_id)["attempt_count"] == 1


@pytest.mark.parametrize("output", ["plain prose", "```yaml\na: 1\n```\n```yaml\nb: 2\n```", "a: ["])
def test_malformed_or_competing_structured_output_is_persisted_and_never_registered(tmp_path: Path, output: str) -> None:
    service = PhaseBService(tmp_path / "registry.sqlite3", codex_runner=SequenceCodex([output, output, output]))
    request = workflow(tmp_path, max_generation_attempts=3, max_repair_attempts=2)
    with pytest.raises(RuntimeError, match="SPECIFICATION_GENERATION_FAILURE"):
        service.generate_spec(request)
    run_id = service._intake_run_id(request, service._strategy_id(request.strategy_name))
    status = service.specification_status(run_id)
    assert status["attempt_count"] == 3
    assert status["failure"]["classification"] == "SPECIFICATION_GENERATION_FAILURE"
    assert service.specification_errors(run_id)[0]["error_code"] in {"STRUCTURED_OUTPUT_INVALID"}


def test_invalid_first_attempt_is_repaired_once_and_persisted(tmp_path: Path) -> None:
    payload = valid_payload(tmp_path)
    runner = SequenceCodex(["strategy_id: intake-fixture\n", payload_text(payload)])
    service = PhaseBService(tmp_path / "registry.sqlite3", codex_runner=runner)
    generated = service.generate_spec(workflow(tmp_path))
    assert generated.attempt == 2
    assert runner.calls == 2
    run_id = service._intake_run_id(workflow(tmp_path), "intake-fixture")
    attempts = service.specification_attempts(run_id)
    assert [item["status"] for item in attempts] == ["INVALID", "VALID"]
    assert Path(attempts[0]["validation_path"]).exists()
    assert Path(attempts[1]["repair_prompt_path"]).exists()


def test_exhausted_repair_is_durable_and_resume_can_continue_an_interrupted_run(tmp_path: Path) -> None:
    payload = payload_text(valid_payload(tmp_path))
    runner = SequenceCodex(["missing: fields", payload])
    service = PhaseBService(tmp_path / "registry.sqlite3", codex_runner=runner)
    request = workflow(tmp_path, max_generation_attempts=1, max_repair_attempts=0)
    with pytest.raises(RuntimeError, match="SPECIFICATION_GENERATION_FAILURE"):
        service.generate_spec(request)
    resumed = service.generate_spec(workflow(tmp_path, run_id=service._intake_run_id(request, "intake-fixture")))
    assert resumed.attempt == 2
    assert service.specification_status(service._intake_run_id(request, "intake-fixture"))["failure"] is None


@pytest.mark.parametrize(
    ("mutation", "error_code"),
    [
        (lambda p: p["baseline_parameters"].update(test_start_date="2026-01-01", test_end_date="2025-01-01"), "DATE_ORDER_INVALID"),
        (lambda p: p["baseline_parameters"].update(session_timezone="UTC") or p["session_assumptions"].append("America/New_York fixed UTC time"), "FIXED_UTC_SESSION"),
        (lambda p: p.update(description="A random strategy.", hypothesis="No edge.", entry_logic="Use a random coin flip without a seed."), "NONDETERMINISTIC_RANDOMNESS"),
        (lambda p: p.update(timeframes=["1h"], exit_logic="Exit exactly 10 minutes after entry."), "INCOMPATIBLE_HOLDING_PERIOD_DATA"),
        (lambda p: p["baseline_parameters"].update(equity_fraction=0.05) or p.update(invariants=["Allocate 5% of current equity; this is risk per trade."]), "EQUITY_ALLOCATION_MISREPRESENTED"),
        (lambda p: p.update(initial_stop_logic="No stop; use a default stop loss."), "INVENTED_STOP"),
        (lambda p: p.update(exit_logic="No target; use a default target."), "INVENTED_TARGET"),
        (lambda p: p.update(invariants=["No optimization."], parameter_families=[{**p["parameter_families"][0], "mutable": True, "maximum_rounds": 1}]), "INVENTED_OPTIMIZATION"),
    ],
)
def test_semantic_validation_rejects_material_intake_drift(tmp_path: Path, mutation, error_code: str) -> None:
    payload = valid_payload(tmp_path)
    mutation(payload)
    _, report, _ = semantic_validate(payload, provenance=SpecificationProvenance())
    assert error_code in {item.error_code for item in report.errors}


def test_missing_required_field_and_invalid_enum_are_structured(tmp_path: Path) -> None:
    payload = valid_payload(tmp_path)
    payload.pop("hypothesis")
    payload["status"] = "NOT_A_STATUS"
    runner = SequenceCodex([payload_text(payload)] * 3)
    service = PhaseBService(tmp_path / "registry.sqlite3", codex_runner=runner)
    with pytest.raises(RuntimeError):
        service.generate_spec(workflow(tmp_path))
    run_id = service._intake_run_id(workflow(tmp_path), "intake-fixture")
    codes = {item["error_code"] for item in service.specification_errors(run_id)}
    assert "PYDANTIC_VALIDATION_ERROR" in codes


def test_blocking_ambiguity_is_never_approval_ready(tmp_path: Path) -> None:
    service = PhaseBService(tmp_path / "registry.sqlite3")
    generated = service.generate_spec(workflow(tmp_path, dry_run=True, ambiguities=["The exit boundary is unspecified."]))
    validation = service.validate_spec(generated)
    assert generated.manual_review_required
    assert not validation.approval_ready
    assert any(item.error_code == "BLOCKING_AMBIGUITY" for item in validation.structured_errors)


def test_spy_requires_proxy_and_phase_d_mapping_disclosure(tmp_path: Path) -> None:
    payload = valid_payload(tmp_path)
    payload["markets"] = ["SPY"]
    payload["known_limitations"] = ["A market proxy is used."]
    _, report, _ = semantic_validate(payload)
    assert any(item.error_code == "PROXY_DISCLOSURE_MISSING" for item in report.errors)


def test_random_open_test_dry_fixture_is_approval_ready_without_manual_fallback(tmp_path: Path) -> None:
    service = PhaseBService(tmp_path / "registry.sqlite3")
    request = workflow(tmp_path, strategy_name="RandomOpenTest", dry_run=True,
                       natural_language_description="RandomOpenTest is a deterministic one-hour pipeline integration test using the SPY proxy.",
                       requested_markets=["SPY"], requested_timeframes=["1h"])
    generated = service.generate_spec(request)
    validation = service.validate_spec(generated)
    assert validation.approval_ready
    assert "1-hour repository-compatible test variant" in generated.approval_summary


def test_real_mode_does_not_reuse_manual_fixture_when_generation_fails(tmp_path: Path) -> None:
    service = PhaseBService(tmp_path / "registry.sqlite3", codex_runner=SequenceCodex(["bad"] * 3))
    request = workflow(tmp_path, strategy_name="RandomOpenTest", dry_run=False,
                       natural_language_description="RandomOpenTest deterministic integration test with a valid structured description.",
                       requested_markets=["SPY"], requested_timeframes=["1h"])
    (tmp_path / "research_registry" / "spec_drafts").mkdir(parents=True)
    (tmp_path / "research_registry" / "spec_drafts" / "RandomOpenTest_vphase-b-1.yaml").write_text("manual fixture", encoding="utf-8")
    with pytest.raises(RuntimeError, match="SPECIFICATION_GENERATION_FAILURE"):
        service.generate_spec(request)
    run_id = service._intake_run_id(request, "RandomOpenTest")
    assert service.specification_status(run_id)["approval_available"] is False
    assert service.specification_status(run_id)["attempt_count"] == 3
