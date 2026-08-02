from __future__ import annotations

import json
from pathlib import Path


def validate_report(path: str | Path, max_size_mb: float = 100) -> tuple[bool, list[str]]:
    report = Path(path)
    errors = []
    if not report.is_file():
        return False, [f"report does not exist: {report}"]
    if report.stat().st_size > max_size_mb * 1024 * 1024:
        errors.append(f"report exceeds {max_size_mb} MB")
    if report.suffix.lower() == ".json":
        try:
            json.loads(report.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"invalid JSON report: {exc}")
    return not errors, errors

