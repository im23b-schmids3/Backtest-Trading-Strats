from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import Field

from ..schemas.strategy_spec import StrictModel


class WorktreePreflightIssue(StrictModel):
    error_code: str
    tracked_path: str
    incompatibility_type: str
    explanation: str
    recommended_remediation: str
    automatic_remediation_safe: bool = False


class WorktreePreflightReport(StrictModel):
    repository_root: str
    git_head: str | None
    checked_at: datetime
    max_path_length: int = Field(ge=1)
    tracked_path_count: int = Field(ge=0)
    safe_for_isolated_worktree: bool
    issues: list[WorktreePreflightIssue] = Field(default_factory=list)
    remediation_plan: list[str] = Field(default_factory=list)
    probe_requested: bool = False
    probe_status: Literal["NOT_REQUESTED", "PASSED", "FAILED", "SKIPPED"] = "NOT_REQUESTED"
    probe_path: str | None = None
    probe_error: str | None = None
    report_path: str | None = None


_INVALID_WINDOWS_CHARS = re.compile(r'[<>:"|?*]')
_RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
_RUNTIME_PREFIXES = ("research_runs/", "research_registry/")


def _creationflags() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


def _git(repository_root: Path, args: list[str]) -> tuple[int, str, str]:
    result = subprocess.run(
        ["git", "-C", str(repository_root), *args],
        capture_output=True,
        text=True,
        shell=False,
        creationflags=_creationflags(),
    )
    return result.returncode, result.stdout, result.stderr


def _issue(code: str, path: str, kind: str, explanation: str, remediation: str, *, safe: bool = False) -> WorktreePreflightIssue:
    return WorktreePreflightIssue(
        error_code=code,
        tracked_path=path,
        incompatibility_type=kind,
        explanation=explanation,
        recommended_remediation=remediation,
        automatic_remediation_safe=safe,
    )


def _component_issues(path: str, root: Path, max_path_length: int) -> list[WorktreePreflightIssue]:
    issues: list[WorktreePreflightIssue] = []
    components = path.replace("\\", "/").split("/")
    for component in components:
        if not component:
            continue
        if _INVALID_WINDOWS_CHARS.search(component):
            issues.append(_issue(
                "INVALID_WINDOWS_FILENAME_CHARACTER", path, "invalid_character",
                f"Tracked component {component!r} contains a Windows-invalid filename character.",
                f"git rm --cached -- {path} (preserve the local file after confirming it is generated).",
            ))
        if component.endswith((" ", ".")):
            issues.append(_issue(
                "TRAILING_SPACE_OR_PERIOD", path, "trailing_character",
                f"Tracked component {component!r} ends with a space or period and cannot be safely checked out on Windows.",
                f"git rm --cached -- {path} (preserve the local file after confirming it is generated).",
            ))
        stem = component.split(".", 1)[0].upper()
        if stem in _RESERVED_NAMES:
            issues.append(_issue(
                "WINDOWS_RESERVED_DEVICE_NAME", path, "reserved_device_name",
                f"Tracked component {component!r} resolves to the reserved Windows device name {stem!r}.",
                f"Rename the source path, or git rm --cached -- {path} if it is generated.",
            ))
    anticipated = len(str(root / ".research_worktrees" / "preflight-probe" / path))
    if anticipated > max_path_length:
        issues.append(_issue(
            "WINDOWS_PATH_TOO_LONG", path, "path_length",
            f"The anticipated secondary-worktree path is {anticipated} characters, above the configured safe limit {max_path_length}.",
            f"Shorten the tracked path, or git rm --cached -- {path} for generated output while preserving the local file.",
        ))
    normalized = path.replace("\\", "/").casefold()
    if normalized.startswith(_RUNTIME_PREFIXES):
        issues.append(_issue(
            "TRACKED_RUNTIME_ARTIFACT", path, "tracked_runtime_artifact",
            "Generated registry or research-run output is tracked and can make secondary worktree checkout unsafe.",
            f"git rm --cached -- {path} (preserve the local file); add an appropriate ignore rule for future runtime artifacts.",
        ))
    return issues


def _probe(repository_root: Path, head: str, report: WorktreePreflightReport) -> WorktreePreflightReport:
    token = hashlib.sha256(f"{repository_root}|{head}".encode("utf-8")).hexdigest()[:16]
    probe = repository_root / ".research_pipeline" / "worktree-probes" / token
    report = report.model_copy(update={"probe_path": str(probe)})
    if probe.exists():
        return report.model_copy(update={"probe_status": "FAILED", "probe_error": "probe path already exists; it was not touched"})
    created = False
    result = report
    try:
        probe.parent.mkdir(parents=True, exist_ok=True)
        code, _, stderr = _git(repository_root, ["worktree", "add", "--detach", str(probe), head])
        if code:
            result = report.model_copy(update={"probe_status": "FAILED", "probe_error": stderr.strip() or "git worktree probe failed"})
        else:
            created = True
            result = report.model_copy(update={"probe_status": "PASSED"})
    finally:
        if created:
            code, _, stderr = _git(repository_root, ["worktree", "remove", "--force", str(probe)])
            if code:
                result = result.model_copy(update={"probe_status": "FAILED", "probe_error": f"probe cleanup failed: {stderr.strip()}"})
            elif probe.exists():
                try:
                    shutil.rmtree(probe, ignore_errors=False)
                except OSError as exc:
                    result = result.model_copy(update={"probe_status": "FAILED", "probe_error": f"probe cleanup failed: {exc}"})
    return result


def run_worktree_preflight(
    repository_root: str | Path,
    *,
    max_path_length: int = 240,
    probe: bool = False,
    persist: bool = True,
) -> WorktreePreflightReport:
    root = Path(repository_root).expanduser().resolve()
    code, head_out, head_err = _git(root, ["rev-parse", "HEAD"])
    head = head_out.strip() if code == 0 else None
    code, tracked_out, tracked_err = _git(root, ["ls-files", "-z"])
    paths = tracked_out.split("\x00") if code == 0 else []
    paths = sorted(path for path in paths if path)
    issues: list[WorktreePreflightIssue] = []
    if code:
        issues.append(_issue("GIT_TRACKED_PATH_SCAN_FAILED", "", "git_scan", tracked_err.strip() or "git ls-files failed", "Repair repository access before creating an implementation worktree."))
    for path in paths:
        issues.extend(_component_issues(path, root, max_path_length))
    folded: dict[str, list[str]] = {}
    for path in paths:
        folded.setdefault(path.replace("\\", "/").casefold(), []).append(path)
    for collision in folded.values():
        if len(collision) > 1:
            for path in collision:
                issues.append(_issue("CASE_INSENSITIVE_PATH_COLLISION", path, "case_collision", f"Tracked paths collide on a case-insensitive Windows filesystem: {collision!r}.", "Rename one path, or untrack generated artifacts with git rm --cached -- <path>."))
    remediation = sorted({item.recommended_remediation for item in issues})
    report = WorktreePreflightReport(
        repository_root=str(root), git_head=head, checked_at=datetime.now(timezone.utc),
        max_path_length=max_path_length, tracked_path_count=len(paths),
        safe_for_isolated_worktree=not issues, issues=issues, remediation_plan=remediation,
        probe_requested=probe, probe_status="NOT_REQUESTED",
    )
    if probe and head:
        report = _probe(root, head, report)
        if report.probe_status == "FAILED":
            report = report.model_copy(update={"safe_for_isolated_worktree": False})
    if persist:
        report_dir = root / "research_registry" / "worktree_preflight"
        report_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(report.model_dump_json().encode("utf-8")).hexdigest()[:16]
        report_path = report_dir / f"preflight-{digest}.json"
        report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        report = report.model_copy(update={"report_path": str(report_path.resolve())})
        report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return report
