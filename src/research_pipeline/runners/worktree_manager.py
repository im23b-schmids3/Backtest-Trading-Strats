from __future__ import annotations

import re
import subprocess
import os
from pathlib import Path

from ..phase_b.models import ImplementationPlan


class WorktreeError(RuntimeError):
    pass


class WorktreeManager:
    def __init__(self, repository_root: str | Path, parent: str | Path | None = None):
        self.repository_root = Path(repository_root).resolve()
        self.parent = Path(parent).resolve() if parent else self.repository_root.parent / ".research_worktrees"

    def _git(self, args: list[str]) -> str:
        options = {"capture_output": True, "text": True, "shell": False}
        if os.name == "nt": options["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        result = subprocess.run(["git", "-C", str(self.repository_root), *args], **options)
        if result.returncode:
            raise WorktreeError(result.stderr.strip() or "git command failed")
        return result.stdout.strip()

    def plan(self, strategy_id: str, version: str, *, dry_run: bool = False, worktree_suffix: str | None = None) -> ImplementationPlan:
        if not self.repository_root.is_dir():
            raise WorktreeError(f"repository root does not exist: {self.repository_root}")
        top = Path(self._git(["rev-parse", "--show-toplevel"])).resolve()
        if top != self.repository_root:
            raise WorktreeError("repository_root must be the Git top-level directory")
        if not dry_run:
            status = self._git(["status", "--porcelain", "--untracked-files=all"])
            disallowed: list[str] = []
            for line in status.splitlines():
                path = line[3:].strip() if len(line) >= 4 else line.strip()
                # The pipeline has already persisted its own audit artifacts
                # before implementation starts. They are safe to leave in the
                # primary checkout; source/tracked changes are not.
                generated = path == "research_runs" or path.startswith("research_runs/") or path == "research_registry" or path.startswith("research_registry/")
                tracked_change = line[:2] != "??"
                if not generated or tracked_change:
                    disallowed.append(line)
            if disallowed:
                raise WorktreeError("primary repository must be clean before implementation worktree creation: " + " | ".join(disallowed))
        base_commit = self._git(["rev-parse", "HEAD"])
        if self.parent == self.repository_root:
            raise WorktreeError("worktree parent may not be the primary repository")
        suffix = f"-{worktree_suffix}" if worktree_suffix else ""
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", f"{strategy_id}-v{version}{suffix}").strip("-")
        branch = f"research-pipeline/{safe}"
        path = (self.parent / safe).resolve()
        if path == self.repository_root or self.parent not in path.parents:
            raise WorktreeError("unsafe worktree path")
        return ImplementationPlan(strategy_id=strategy_id, version=version, base_commit=base_commit, branch=branch,
            worktree_path=str(path), allowed_files=["src/", "tests/"],
            required_tests=[["python", "-m", "pytest", "-q", "tests/research_pipeline"]],
            invariants=[], prohibited_actions=["Do not run optimization or backtests", "Do not modify existing strategy behavior"], max_repair_attempts=3)

    def create(self, plan: ImplementationPlan, *, dry_run: bool = True) -> ImplementationPlan:
        path = Path(plan.worktree_path).resolve()
        if dry_run:
            return plan
        self.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            existing = self._git(["worktree", "list", "--porcelain"])
            worktree_paths = {
                Path(line.removeprefix("worktree ")).resolve()
                for line in existing.splitlines()
                if line.startswith("worktree ")
            }
            branches = {
                line.removeprefix("branch ").removeprefix("refs/heads/")
                for line in existing.splitlines()
                if line.startswith("branch ")
            }
            if path in worktree_paths and plan.branch.removeprefix("refs/heads/") in branches:
                return plan
            raise WorktreeError("worktree path already exists and is not the requested worktree")
        path.parent.mkdir(parents=True, exist_ok=True)
        options = {"capture_output": True, "text": True, "shell": False}
        if os.name == "nt": options["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        result = subprocess.run(["git", "-C", str(self.repository_root), "worktree", "add", "-b", plan.branch, str(path), plan.base_commit], **options)
        if result.returncode:
            raise WorktreeError(result.stderr.strip() or "git worktree add failed")
        return plan

    @staticmethod
    def current_commit(path: str) -> str | None:
        options = {"capture_output": True, "text": True, "shell": False}
        if os.name == "nt": options["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        result = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], **options)
        return result.stdout.strip() if result.returncode == 0 else None

    @staticmethod
    def changed_files(plan: ImplementationPlan) -> list[str]:
        options = {"capture_output": True, "text": True, "shell": False}
        if os.name == "nt": options["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        tracked = subprocess.run(["git", "-C", plan.worktree_path, "diff", "--name-only", plan.base_commit], **options)
        status = subprocess.run(["git", "-C", plan.worktree_path, "status", "--porcelain", "--untracked-files=all"], **options)
        paths = {line.strip() for line in tracked.stdout.splitlines() if line.strip()} if tracked.returncode == 0 else set()
        if status.returncode == 0:
            paths.update(line[3:].strip() for line in status.stdout.splitlines() if len(line) >= 4 and line[3:].strip())
        return sorted(paths)
