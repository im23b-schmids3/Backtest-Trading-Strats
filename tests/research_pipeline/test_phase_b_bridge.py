from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from research_pipeline.phase_b.models import WorkflowInput
from research_pipeline.phase_b.services import PhaseBService
from research_pipeline.workflow_bridge.bridge import PhaseBBridge


def test_bridge_rejects_unsupported_command() -> None:
    with pytest.raises(ValueError):
        PhaseBBridge().dispatch("not-a-command", {})


def test_cli_bridge_is_json_in_json_out_and_errors_are_nonzero(tmp_path: Path) -> None:
    root = tmp_path / "fixture"
    root.mkdir()
    payload = WorkflowInput(
        strategy_name="cli-fixture",
        natural_language_description="A fictional strategy used to verify the Phase B bridge.",
        requested_markets=["TEST"], requested_timeframes=["1h"], repository_root=str(root),
    ).model_dump(mode="json")
    env = {"PYTHONPATH": str(Path.cwd() / "src"), "RESEARCH_PIPELINE_REGISTRY": str(tmp_path / "registry.sqlite3")}
    generated = subprocess.run(
        [sys.executable, "-m", "research_pipeline", "workflow", "generate-spec", "--input-json", json.dumps(payload)],
        capture_output=True, text=True, env={**__import__("os").environ, **env}, shell=False,
    )
    assert generated.returncode == 0
    assert json.loads(generated.stdout)["strategy_id"] == "cli-fixture"

    failed = subprocess.run(
        [sys.executable, "-m", "research_pipeline", "workflow", "validate-spec", "--input-json", "{}"],
        capture_output=True, text=True, env={**__import__("os").environ, **env}, shell=False,
    )
    assert failed.returncode != 0
    assert "error" in failed.stderr


def test_registry_and_bridge_reconcile_after_final_status(tmp_path: Path) -> None:
    root = tmp_path / "fixture"
    root.mkdir()
    for command in (
        ["git", "init", "-q", str(root)],
        ["git", "-C", str(root), "config", "user.email", "phase-b@example.invalid"],
        ["git", "-C", str(root), "config", "user.name", "Phase B Tests"],
    ):
        subprocess.run(command, check=True, capture_output=True, text=True, shell=False)
    (root / "README.md").write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "README.md"], check=True, capture_output=True, text=True, shell=False)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "fixture"], check=True, capture_output=True, text=True, shell=False)
    service = PhaseBService(tmp_path / "registry.sqlite3")
    workflow = WorkflowInput(
        strategy_name="reconcile-fixture",
        natural_language_description="A fictional strategy used to verify registry reconciliation.",
        requested_markets=["TEST"], requested_timeframes=["1h"], repository_root=str(root),
    )
    generated = service.generate_spec(workflow)
    validation = service.validate_spec(generated)
    service.register_generated(validation)
    service.approve(generated.strategy_id, "APPROVE")
    plan = service.implementation_plan(generated.strategy_id, str(root))
    tests = service.run_tests(str(root), dry_run=True, worktree_path=plan.worktree_path)
    summary = service.final_status(generated.strategy_id, tests, implementation_executed=False, repair_attempts=0)
    assert summary.registry_reconciled is True
    assert service.registry.get_strategy(generated.strategy_id)["current_phase"] == "IMPLEMENTATION_VERIFICATION"
