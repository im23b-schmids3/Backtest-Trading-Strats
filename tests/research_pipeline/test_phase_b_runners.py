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
    assert timed.exit_code == -1 and timed.termination_method == "timeout_no_process_handle"

    def failed(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 7, "1 failed", "compiler error")

    result = CodexRunner(executable="codex", run_process=failed).run("prompt", ".", dry_run=False)
    assert result.exit_code == 7 and not result.success and result.executed

    with pytest.raises(ValueError):
        CodexRunner(executable="codex").run("prompt", ".", sandbox="danger-full-access")


class _FakePopen:
    def __init__(self, *, stdout: str = "", stderr: str = "", returncode: int = 0, timeout_once: bool = False):
        self.stdout = stdout
        self.stderr = stderr
        self.final_returncode = returncode
        self.returncode: int | None = None
        self.timeout_once = timeout_once
        self.communicate_calls = 0
        self.terminated = False
        self.killed = False
        self.wait_called = False

    def communicate(self, *, input: str | None = None, timeout: int | None = None):
        self.communicate_calls += 1
        if self.timeout_once and self.communicate_calls == 1:
            raise subprocess.TimeoutExpired("codex", timeout or 0, output="partial stdout", stderr="partial stderr")
        self.returncode = self.final_returncode
        return self.stdout, self.stderr

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True

    def wait(self) -> int:
        self.wait_called = True
        if self.returncode is None:
            self.returncode = self.final_returncode
        return self.returncode


def test_codex_runner_drains_output_and_records_terminal_exit_code() -> None:
    process = _FakePopen(stdout="x" * 250_000, stderr="diff --git a/file b/file\n" + "y" * 250_000)
    runner = CodexRunner(executable="codex", popen_factory=lambda *args, **kwargs: process)
    result = runner.run("prompt", ".", dry_run=False, timeout_seconds=2)
    assert result.success and result.exit_code == 0 and result.timed_out is False
    assert result.configured_timeout_seconds == 2 and process.wait_called
    assert len(result.stdout) == 250_000 and "diff --git" in result.stderr


def test_codex_runner_timeout_terminates_drains_and_records_exit_code() -> None:
    process = _FakePopen(stdout="final stdout", stderr="final stderr", returncode=-15, timeout_once=True)
    runner = CodexRunner(executable="codex", popen_factory=lambda *args, **kwargs: process)
    result = runner.run("prompt", ".", dry_run=False, timeout_seconds=1)
    assert result.timed_out and result.error_type == "TIMEOUT"
    assert result.exit_code == -15 and result.termination_method == "terminate"
    assert result.process_signal == 15 and process.terminated and process.wait_called
    assert result.stdout == "final stdout" and result.stderr == "final stderr"


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
