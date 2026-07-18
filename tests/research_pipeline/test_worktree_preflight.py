from __future__ import annotations

import subprocess
from pathlib import Path

from research_pipeline.repository.worktree_preflight import run_worktree_preflight


def _git_repo(root: Path, *, runtime_path: str | None = None) -> None:
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True, shell=False)
    subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True, shell=False)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "Preflight Test"], check=True, shell=False)
    (root / "src").mkdir()
    (root / "src" / "strategy.py").write_text("# fixture\n", encoding="utf-8")
    if runtime_path:
        path = root / Path(runtime_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("generated\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "."], check=True, shell=False)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "fixture"], check=True, shell=False)


def test_preflight_passes_and_probe_cleans_up(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _git_repo(root)
    report = run_worktree_preflight(root, probe=True, persist=False)
    assert report.safe_for_isolated_worktree
    assert report.probe_status == "PASSED"
    assert report.probe_path is not None
    assert not Path(report.probe_path).exists()


def test_preflight_reports_tracked_runtime_artifact_without_mutating_git(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _git_repo(root, runtime_path="research_runs/old/output.json")
    before = subprocess.check_output(["git", "-C", str(root), "ls-files"], text=True, shell=False).splitlines()
    report = run_worktree_preflight(root, persist=False)
    after = subprocess.check_output(["git", "-C", str(root), "ls-files"], text=True, shell=False).splitlines()
    assert not report.safe_for_isolated_worktree
    issue = next(item for item in report.issues if item.tracked_path == "research_runs/old/output.json")
    assert issue.error_code == "TRACKED_RUNTIME_ARTIFACT"
    assert "git rm --cached" in issue.recommended_remediation
    assert issue.automatic_remediation_safe is False
    assert before == after
