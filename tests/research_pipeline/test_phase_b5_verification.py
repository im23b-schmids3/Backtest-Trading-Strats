from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from research_pipeline.enums import PipelineState
from research_pipeline.phase_b.services import PhaseBService
from research_pipeline.verification.fixtures import make_fixture
from research_pipeline.verification.services import VerificationService
from tests.research_pipeline.test_phase_b_core import git_repo, workflow_input


def verification_service(tmp_path: Path) -> tuple[VerificationService, str]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    root = git_repo(tmp_path)
    phase_b = PhaseBService(tmp_path / "registry.sqlite3")
    generated = phase_b.generate_spec(workflow_input(root, strategy_name="b5-fixture"))
    phase_b.register_generated(phase_b.validate_spec(generated))
    phase_b.approve(generated.strategy_id, "APPROVE")
    phase_b.controller.transition(generated.strategy_id, PipelineState.IMPLEMENTATION_VERIFICATION, "fixture implementation verified")
    return VerificationService(tmp_path / "registry.sqlite3"), generated.strategy_id


def test_schema_migration_and_correct_fixture_reach_baseline(tmp_path: Path) -> None:
    service, strategy_id = verification_service(tmp_path)
    manifest = make_fixture(tmp_path / "correct", strategy_id)
    result = service.run(strategy_id, manifest)
    assert result["outcome"] == "VERIFIED"
    assert service.registry.get_strategy(strategy_id)["current_phase"] == PipelineState.BASELINE_BACKTEST.value
    with sqlite3.connect(tmp_path / "registry.sqlite3") as connection:
        assert connection.execute("SELECT version FROM schema_version").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM verification_runs").fetchone()[0] == 1


@pytest.mark.parametrize("kind", ["missing-multiplier", "duplicate-fee", "partial-exit", "scaling", "lookahead", "terminal-flatten", "report-mismatch", "nondeterministic"])
def test_proven_diagnostic_defects_require_repair(tmp_path: Path, kind: str) -> None:
    service, strategy_id = verification_service(tmp_path)
    result = service.run(strategy_id, make_fixture(tmp_path / kind, strategy_id, kind=kind))
    assert result["outcome"] == "TECHNICAL_REPAIR_REQUIRED"
    assert service.registry.get_strategy(strategy_id)["current_phase"] == PipelineState.TECHNICAL_REPAIR_REQUIRED.value


def test_semantic_and_missing_data_outcomes(tmp_path: Path) -> None:
    service, strategy_id = verification_service(tmp_path)
    result = service.run(strategy_id, make_fixture(tmp_path / "ambiguous", strategy_id, kind="ambiguous-count"))
    assert result["outcome"] == "MANUAL_REVIEW_REQUIRED"

    service, strategy_id = verification_service(tmp_path / "missing")
    result = service.run(strategy_id, make_fixture(tmp_path / "missing-fixture", strategy_id, kind="missing-diagnostics"))
    assert result["outcome"] == "INSUFFICIENT_DIAGNOSTIC_DATA"


def test_duplicate_verification_run_is_idempotent(tmp_path: Path) -> None:
    service, strategy_id = verification_service(tmp_path)
    manifest = make_fixture(tmp_path / "correct", strategy_id)
    first = service.run(strategy_id, manifest)
    second = service.run(strategy_id, manifest)
    assert first["verification_run_id"] == second["verification_run_id"]
    assert len(service.registry.history(strategy_id)["transitions"]) == 5
