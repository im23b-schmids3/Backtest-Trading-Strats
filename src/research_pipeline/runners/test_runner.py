from __future__ import annotations

import re
import subprocess
import tempfile
import time
from pathlib import Path

from ..phase_b.models import TestResult
from ..phase_b.redaction import redact_secrets
from .isolated_environment import build_isolated_environment, pytest_command


PASSED_RE = re.compile(r"(\d+)\s+passed", re.IGNORECASE)
FAILED_RE = re.compile(r"(\d+)\s+failed", re.IGNORECASE)
SKIPPED_RE = re.compile(r"(\d+)\s+skipped", re.IGNORECASE)


class DeterministicTestRunner:
    def run(self, cwd: str | Path, command: list[str], *, timeout_seconds: int = 900, dry_run: bool = True,
            report_path: str | Path | None = None,
            source_repository_root: str | Path | None = None,
            environment: dict[str, str] | None = None,
            basetemp: str | Path | None = None) -> TestResult:
        root = Path(cwd).resolve()
        process_environment, _ = build_isolated_environment(
            root,
            repository_root=source_repository_root or root,
            base_environment=environment,
        )
        if dry_run:
            return TestResult(passed=True, command=command, exit_code=None, parsed_passed=1, parsed_failed=0, parsed_skipped=0,
                duration_ms=0, report_path=str(report_path) if report_path else None, failure_summary="mocked dry-run success", executed=False)
        started = time.monotonic()
        temp_root = Path(basetemp) if basetemp else Path(tempfile.gettempdir()) / "research_pipeline_pytest" / str(abs(hash(str(root))))
        temp_root = temp_root.resolve()
        temp_root.mkdir(parents=True, exist_ok=True)
        effective_command = pytest_command(command, temp_root)
        try:
            completed = subprocess.run(
                effective_command,
                cwd=str(root),
                env=process_environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
                shell=False,
            )
            output = redact_secrets((completed.stdout or "") + "\n" + (completed.stderr or ""))
            passed, failed, skipped = self._parse_summary(output)
            if report_path:
                Path(report_path).parent.mkdir(parents=True, exist_ok=True)
                Path(report_path).write_text(output[-100_000:], encoding="utf-8")
            return TestResult(passed=completed.returncode == 0 and failed == 0, command=effective_command, exit_code=completed.returncode,
                parsed_passed=passed, parsed_failed=failed, parsed_skipped=skipped, duration_ms=int((time.monotonic() - started) * 1000),
                report_path=str(report_path) if report_path else None, failure_summary="" if completed.returncode == 0 else output[-4000:], executed=True)
        except subprocess.TimeoutExpired as exc:
            return TestResult(passed=False, command=effective_command, exit_code=None, parsed_passed=0, parsed_failed=1, parsed_skipped=0,
                duration_ms=int((time.monotonic() - started) * 1000), report_path=str(report_path) if report_path else None,
                failure_summary=redact_secrets(str(exc)), executed=True)
        except OSError as exc:
            return TestResult(passed=False, command=effective_command, exit_code=None, parsed_passed=0, parsed_failed=1, parsed_skipped=0,
                duration_ms=int((time.monotonic() - started) * 1000), report_path=str(report_path) if report_path else None,
                failure_summary=redact_secrets(str(exc)), executed=False)

    @staticmethod
    def _parse_summary(output: str) -> tuple[int, int, int]:
        def last(pattern: re.Pattern[str]) -> int:
            matches = list(pattern.finditer(output))
            return int(matches[-1].group(1)) if matches else 0

        return last(PASSED_RE), last(FAILED_RE), last(SKIPPED_RE)
