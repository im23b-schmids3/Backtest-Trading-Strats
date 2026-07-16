from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from research_pipeline.phase_b.redaction import redact_secrets
from research_pipeline.runners.codex_runner import CodexRunner
from research_pipeline.runners.test_runner import DeterministicTestRunner
from research_pipeline.runners.worktree_manager import WorktreeError, WorktreeManager


def test_codex_dry_run_never_starts_a_subprocess() -> None:
    calls: list[object] = []

    def forbidden(*args: object, **kwargs: object) -> None:
        calls.append((args, kwargs))
        raise AssertionError("subprocess must not run in dry-run mode")

    result = CodexRunner(executable="codex", run_process=forbidden).run("safe prompt", ".", dry_run=True)
    assert result.success and not result.executed and calls == []


def test_codex_runner_missing_timeout_and_nonzero() -> None:
    missing = CodexRunner(executable=None, run_process=lambda *a, **k: None)
    missing.executable = None
    assert missing.run("prompt", ".", dry_run=False).error_type == "MISSING_EXECUTABLE"

    def timeout(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired(kwargs.get("timeout", 1), args)

    timed = CodexRunner(executable="codex", run_process=timeout).run("prompt", ".", dry_run=False)
    assert timed.timed_out and timed.error_type == "TIMEOUT"

    def failed(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 7, "1 failed", "compiler error")

    result = CodexRunner(executable="codex", run_process=failed).run("prompt", ".", dry_run=False)
    assert result.exit_code == 7 and not result.success and result.executed

    with pytest.raises(ValueError):
        CodexRunner(executable="codex").run("prompt", ".", sandbox="danger-full-access")


def test_secret_redaction_applies_to_outputs_and_commands() -> None:
    value = "Authorization: Bearer abcdefghijkl token=super-secret sk-test-secret"
    redacted = redact_secrets(value)
    assert "abcdefghijkl" not in redacted
    assert "super-secret" not in redacted
    assert "sk-test-secret" not in redacted


def test_test_runner_uses_process_exit_and_parses_counts() -> None:
    def passed(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 0, "12 passed, 2 skipped", "agent said tests passed")

    runner = DeterministicTestRunner()
    runner_process = subprocess.run
    try:
        subprocess.run = passed  # type: ignore[assignment]
        result = runner.run(".", ["python", "-m", "pytest"], dry_run=False)
    finally:
        subprocess.run = runner_process  # type: ignore[assignment]
    assert result.passed and result.parsed_passed == 12 and result.parsed_skipped == 2


def test_worktree_path_safety_and_duplicate_existing_path(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True, shell=False)
    subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True, shell=False)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True, shell=False)
    (root / "file.txt").write_text("x\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "file.txt"], check=True, shell=False)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "fixture"], check=True, shell=False)
    manager = WorktreeManager(root, tmp_path / "worktrees")
    plan = manager.plan("safe", "phase-b-1", dry_run=True)
    assert Path(plan.worktree_path).parent == (tmp_path / "worktrees").resolve()
    with pytest.raises(WorktreeError):
        WorktreeManager(root, root).plan("safe", "v1", dry_run=True)
    created = manager.create(plan.model_copy(update={"worktree_path": str((tmp_path / "worktrees" / "safe-phase-b-1").resolve())}), dry_run=False)
    assert Path(created.worktree_path).is_dir()
    assert manager.create(created, dry_run=False) == created
