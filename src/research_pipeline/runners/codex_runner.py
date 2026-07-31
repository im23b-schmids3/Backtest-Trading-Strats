from __future__ import annotations

import shutil
import subprocess
import time
import os
import sys
from pathlib import Path
from typing import Callable

from ..phase_b.models import CodexExecutionResult
from ..phase_b.redaction import redact_secrets
from .isolated_environment import build_isolated_environment


class CodexRunner:
    """Safe, argument-array Codex CLI adapter with a no-subprocess dry run."""

    def __init__(self, executable: str | None = None, run_process: Callable | None = None):
        self.executable = resolve_codex_executable(executable)
        self.run_process = run_process or subprocess.run

    def run(self, prompt: str, cwd: str | Path, *, sandbox: str = "read-only", timeout_seconds: int = 900,
            dry_run: bool = True, output_last_message: str | Path | None = None,
            environment: dict[str, str] | None = None,
            source_repository_root: str | Path | None = None) -> CodexExecutionResult:
        if sandbox not in {"read-only", "workspace-write"}:
            raise ValueError("Phase B Codex sandbox must be read-only or workspace-write")
        root = Path(cwd).resolve()
        process_environment, _ = build_isolated_environment(
            root,
            repository_root=source_repository_root or root,
            base_environment=environment,
        )
        command = [self.executable or "codex", "exec", "--sandbox", sandbox, "--cd", str(root)]
        if output_last_message:
            command.extend(["--output-last-message", str(Path(output_last_message).resolve())])
        safe_command = [redact_secrets(str(item)) for item in command]
        if dry_run:
            return CodexExecutionResult(success=True, executed=False, command=safe_command, cwd=str(root), sandbox=sandbox,
                exit_code=None, stdout="", stderr="", duration_ms=0, timed_out=False)
        if not self.executable:
            return CodexExecutionResult(success=False, executed=False, command=safe_command, cwd=str(root), sandbox=sandbox,
                exit_code=None, stdout="", stderr="Codex executable not found", duration_ms=0, timed_out=False, error_type="MISSING_EXECUTABLE")
        started = time.monotonic()
        try:
            process_options = dict(cwd=str(root), input=prompt, capture_output=True, text=True,
                                    encoding="utf-8", errors="replace", timeout=timeout_seconds, shell=False,
                                    env=process_environment)
            # Avoid opening a console window for the external executor on
            # Windows.  Test doubles often expose a narrower signature, so
            # only pass the platform-specific option to subprocess.run itself.
            if os.name == "nt" and self.run_process is subprocess.run:
                process_options["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            completed = self.run_process(command, **process_options)
            stdout, stderr = redact_secrets(completed.stdout or ""), redact_secrets(completed.stderr or "")
            return CodexExecutionResult(success=completed.returncode == 0, executed=True, command=safe_command, cwd=str(root), sandbox=sandbox,
                exit_code=completed.returncode, stdout=stdout, stderr=stderr, duration_ms=int((time.monotonic() - started) * 1000), timed_out=False)
        except subprocess.TimeoutExpired as exc:
            return CodexExecutionResult(success=False, executed=True, command=safe_command, cwd=str(root), sandbox=sandbox,
                exit_code=None, stdout=redact_secrets(str(exc.stdout or "")), stderr=redact_secrets(str(exc.stderr or "")),
                duration_ms=int((time.monotonic() - started) * 1000), timed_out=True, error_type="TIMEOUT")
        except OSError as exc:
            return CodexExecutionResult(success=False, executed=False, command=safe_command, cwd=str(root), sandbox=sandbox,
                exit_code=None, stdout="", stderr=redact_secrets(str(exc)), duration_ms=int((time.monotonic() - started) * 1000), timed_out=False, error_type="PROCESS_ERROR")


def _verified(path: str | None) -> str | None:
    if not path:
        return None
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        found = shutil.which(str(candidate))
        if found:
            candidate = Path(found)
    if not candidate.is_file():
        return None
    # PowerShell scripts require a shell and are deliberately not direct-spawned.
    if os.name == "nt" and candidate.suffix.lower() == ".ps1":
        return None
    return str(candidate.resolve())


def resolve_codex_executable(executable: str | None = None) -> str | None:
    """Resolve a directly spawnable Codex executable without shell parsing."""
    configured = executable or os.environ.get("CODEX_EXECUTABLE")
    candidates = [configured, shutil.which("codex"), shutil.which("codex.cmd")]
    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        if appdata:
            candidates.append(str(Path(appdata) / "npm" / "codex.cmd"))
    for candidate in candidates:
        resolved = _verified(candidate)
        if resolved:
            return resolved
    return None


def codex_tool_diagnostic() -> dict[str, object]:
    resolved = resolve_codex_executable()
    version_ok = False
    if resolved:
        try:
            options = dict(capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15, shell=False)
            if os.name == "nt":
                options["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            version_ok = subprocess.run([resolved, "--version"], **options).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            version_ok = False
    return {"python_executable": sys.executable, "codex_executable": resolved,
            "codex_version_command_success": version_ok}


def is_restricted_execution_failure(result: CodexExecutionResult) -> bool:
    """Recognize tenant/sandbox denial without weakening the security policy."""
    text = f"{result.stderr}\n{result.stdout}".lower()
    explicit = ("os error 10013", "tenant policy", "network access is disabled",
                "network access disabled", "sandbox policy", "restricted sandbox")
    return any(marker in text for marker in explicit)
