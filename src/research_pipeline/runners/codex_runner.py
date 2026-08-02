from __future__ import annotations

import shutil
import subprocess
import time
import os
import sys
from pathlib import Path
from typing import Any, Callable

from ..phase_b.models import CodexExecutionResult
from ..phase_b.redaction import redact_secrets
from .isolated_environment import build_isolated_environment


class CodexRunner:
    """Safe, argument-array Codex CLI adapter with a no-subprocess dry run."""

    DEFAULT_TIMEOUT_SECONDS = 1800
    MAX_TIMEOUT_SECONDS = 7200
    OUTPUT_DRAIN_TIMEOUT_SECONDS = 30

    def __init__(self, executable: str | None = None, run_process: Callable | None = None,
                 popen_factory: Callable[..., Any] | None = None):
        self.executable = resolve_codex_executable(executable)
        # ``run_process`` remains a compact compatibility seam for existing
        # unit tests. Production uses Popen so timeout handling can terminate,
        # drain both pipes, and record the terminal return code explicitly.
        self.run_process = run_process
        self.popen_factory = popen_factory or subprocess.Popen

    def run(self, prompt: str, cwd: str | Path, *, sandbox: str = "read-only", timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
            dry_run: bool = True, output_last_message: str | Path | None = None,
            environment: dict[str, str] | None = None,
            source_repository_root: str | Path | None = None) -> CodexExecutionResult:
        if sandbox not in {"read-only", "workspace-write"}:
            raise ValueError("Phase B Codex sandbox must be read-only or workspace-write")
        if not 1 <= timeout_seconds <= self.MAX_TIMEOUT_SECONDS:
            raise ValueError(f"Codex timeout must be between 1 and {self.MAX_TIMEOUT_SECONDS} seconds")
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
                exit_code=None, stdout="", stderr="", duration_ms=0, timed_out=False,
                configured_timeout_seconds=timeout_seconds)
        if not self.executable:
            return CodexExecutionResult(success=False, executed=False, command=safe_command, cwd=str(root), sandbox=sandbox,
                exit_code=None, stdout="", stderr="Codex executable not found", duration_ms=0, timed_out=False,
                configured_timeout_seconds=timeout_seconds, error_type="MISSING_EXECUTABLE")
        started = time.monotonic()
        process_options = dict(cwd=str(root), text=True, encoding="utf-8", errors="replace", shell=False,
                               env=process_environment)
        if os.name == "nt" and self.run_process is None:
            process_options["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        if self.run_process is not None:
            return self._run_compatibility_process(
                command, prompt, safe_command, root, sandbox, timeout_seconds, started, process_options
            )
        return self._run_managed_process(
            command, prompt, safe_command, root, sandbox, timeout_seconds, started, process_options
        )

    @staticmethod
    def _text(value: str | bytes | None) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return value or ""

    def _run_compatibility_process(self, command: list[str], prompt: str, safe_command: list[str], root: Path,
                                   sandbox: str, timeout_seconds: int, started: float,
                                   process_options: dict[str, Any]) -> CodexExecutionResult:
        """Compatibility path for injected subprocess.run-style test doubles."""

        try:
            completed = self.run_process(command, input=prompt, capture_output=True, timeout=timeout_seconds, **process_options)
            stdout, stderr = redact_secrets(completed.stdout or ""), redact_secrets(completed.stderr or "")
            return CodexExecutionResult(success=completed.returncode == 0, executed=True, command=safe_command, cwd=str(root), sandbox=sandbox,
                exit_code=completed.returncode, stdout=stdout, stderr=stderr, duration_ms=int((time.monotonic() - started) * 1000), timed_out=False,
                configured_timeout_seconds=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            return CodexExecutionResult(success=False, executed=True, command=safe_command, cwd=str(root), sandbox=sandbox,
                # A run-style test double has no process handle to terminate;
                # retain a non-null sentinel so timeout is never ambiguous.
                exit_code=-1, stdout=redact_secrets(self._text(exc.stdout)), stderr=redact_secrets(self._text(exc.stderr)),
                duration_ms=int((time.monotonic() - started) * 1000), timed_out=True,
                configured_timeout_seconds=timeout_seconds, termination_method="timeout_no_process_handle", error_type="TIMEOUT")
        except OSError as exc:
            return CodexExecutionResult(success=False, executed=False, command=safe_command, cwd=str(root), sandbox=sandbox,
                exit_code=None, stdout="", stderr=redact_secrets(str(exc)), duration_ms=int((time.monotonic() - started) * 1000), timed_out=False,
                configured_timeout_seconds=timeout_seconds, error_type="PROCESS_ERROR")

    def _run_managed_process(self, command: list[str], prompt: str, safe_command: list[str], root: Path,
                             sandbox: str, timeout_seconds: int, started: float,
                             process_options: dict[str, Any]) -> CodexExecutionResult:
        process = None
        try:
            process = self.popen_factory(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                **process_options,
            )
            stdout, stderr = process.communicate(input=prompt, timeout=timeout_seconds)
            # communicate() drains both pipes and waits. wait() is intentionally
            # repeated to make the terminal return-code contract explicit.
            exit_code = process.wait()
            return CodexExecutionResult(success=exit_code == 0, executed=True, command=safe_command, cwd=str(root), sandbox=sandbox,
                exit_code=exit_code, stdout=redact_secrets(self._text(stdout)), stderr=redact_secrets(self._text(stderr)),
                duration_ms=int((time.monotonic() - started) * 1000), timed_out=False,
                configured_timeout_seconds=timeout_seconds,
                process_signal=-exit_code if exit_code < 0 else None)
        except subprocess.TimeoutExpired:
            if process is None:
                return CodexExecutionResult(success=False, executed=False, command=safe_command, cwd=str(root), sandbox=sandbox,
                    exit_code=-1, stdout="", stderr="Codex process did not start before timeout", duration_ms=int((time.monotonic() - started) * 1000),
                    timed_out=True, configured_timeout_seconds=timeout_seconds, termination_method="process_start_timeout", error_type="TIMEOUT")
            termination_method = "terminate"
            process.terminate()
            try:
                stdout, stderr = process.communicate(timeout=self.OUTPUT_DRAIN_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                termination_method = "terminate_then_kill"
                process.kill()
                stdout, stderr = process.communicate()
            exit_code = process.wait()
            return CodexExecutionResult(success=False, executed=True, command=safe_command, cwd=str(root), sandbox=sandbox,
                exit_code=exit_code if exit_code is not None else -1, stdout=redact_secrets(self._text(stdout)), stderr=redact_secrets(self._text(stderr)),
                duration_ms=int((time.monotonic() - started) * 1000), timed_out=True,
                configured_timeout_seconds=timeout_seconds, termination_method=termination_method,
                process_signal=-exit_code if isinstance(exit_code, int) and exit_code < 0 else None, error_type="TIMEOUT")
        except OSError as exc:
            return CodexExecutionResult(success=False, executed=False, command=safe_command, cwd=str(root), sandbox=sandbox,
                exit_code=None, stdout="", stderr=redact_secrets(str(exc)), duration_ms=int((time.monotonic() - started) * 1000), timed_out=False,
                configured_timeout_seconds=timeout_seconds, error_type="PROCESS_ERROR")


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
