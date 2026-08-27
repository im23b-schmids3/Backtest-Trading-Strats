"""Bounded local checkpoint worker for the Asia W04 diagnostic.

The worker is a performance aid for independent daily source files.  It uses
the published baseline's prior-Asia POC and accepts a checkpoint only when the
same frozen state machine reproduces that day's published POC and funnel.
Shared multi-session DBNs remain on the authoritative sequential runner.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import asia_w04_diagnostic as diagnostic
from . import asia_w04_replay as asia
from . import historical_runner as historical


def _daily_baseline(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = {str(row["session_date"]): dict(row) for row in csv.DictReader(handle)}
    if len(rows) != asia.EXPECTED_ELIGIBLE_SESSIONS:
        raise diagnostic.AsiaW04DiagnosticError("published baseline daily ledger is incomplete")
    return rows


def _same_number(actual: Any, expected: str, *, tolerance: float = 1e-12) -> bool:
    if actual is None or expected in {"", "None"}:
        return actual is None and expected in {"", "None"}
    return abs(float(actual) - float(expected)) <= tolerance


def _validate_result(result: Mapping[str, Any], expected: Mapping[str, str]) -> None:
    day = str(result["session_date"])
    exact = {
        "raw_interactions": "raw_interactions",
        "accepted_setups": "accepted_setups",
        "confirmations_passed": "confirmations_passed",
        "confirmation_failures": "confirmation_failures",
        "unresolved": "unresolved",
    }
    mismatches = {
        key: (result[key], expected[column])
        for key, column in exact.items()
        if int(result[key]) != int(expected[column])
    }
    for key in ("prior_asia_poc", "session_poc"):
        if not _same_number(result[key], expected[key]):
            mismatches[key] = (result[key], expected[key])
    if mismatches:
        raise diagnostic.AsiaW04DiagnosticError(
            f"parallel diagnostic checkpoint does not reproduce baseline {day}: {mismatches}"
        )


def build_daily_checkpoints(
    *, repository_root: Path, staging: Path, days: Sequence[str],
    audit_root: Path = asia.AUDIT_ROOT, baseline_root: Path = diagnostic.BASELINE_ROOT,
) -> list[str]:
    repository_root = repository_root.resolve()
    staging = (staging if staging.is_absolute() else repository_root / staging).resolve()
    audit_root = (audit_root if audit_root.is_absolute() else repository_root / audit_root).resolve()
    baseline_root = (baseline_root if baseline_root.is_absolute() else repository_root / baseline_root).resolve()
    if not (staging / "diagnostic-contract.json").is_file():
        raise diagnostic.AsiaW04DiagnosticError("diagnostic staging contract is missing")
    sessions = asia.load_audit_sessions(audit_root)
    sessions_by_day = {row.day: row for row in sessions}
    bindings = {day: binding for binding in asia.source_bindings(repository_root, sessions) if not binding.shared for day in binding.days}
    baseline = _daily_baseline(baseline_root / "daily-results.csv")
    completed: list[str] = []
    for day in days:
        session = sessions_by_day.get(day)
        binding = bindings.get(day)
        expected = baseline.get(day)
        if session is None or binding is None or expected is None or not session.eligible:
            raise diagnostic.AsiaW04DiagnosticError(f"day is not an independent eligible daily binding: {day}")
        existing = asia._load_checkpoint(staging, session)
        if existing is not None:
            _validate_result(existing, expected)
            completed.append(day)
            continue
        prior_poc = float(expected["prior_asia_poc"])
        state = asia._AsiaMboState(asia._spec(session, binding, staging), prior_poc)
        next_progress = 5_000_000
        try:
            for record in historical._stream_private_mbo(binding.path):
                state.observe(record)
                if state.reached_cutoff:
                    break
                if state.decoded_records >= next_progress:
                    print(
                        f"ASIA_DIAGNOSTIC_WORKER {day} records={state.decoded_records:,} "
                        f"interactions={len(state.interactions):,}",
                        flush=True,
                    )
                    next_progress += 5_000_000
            result = state.finish()
            _validate_result(result, expected)
            asia._save_checkpoint(staging, result)
            completed.append(day)
            print(f"ASIA_DIAGNOSTIC_WORKER_COMPLETE {day}", flush=True)
        except BaseException:
            state.abort()
            raise
    return completed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--days", required=True, help="Comma-separated independent daily session dates")
    args = parser.parse_args(argv)
    try:
        completed = build_daily_checkpoints(
            repository_root=args.repository_root,
            staging=args.staging,
            days=tuple(day for day in args.days.split(",") if day),
        )
    except Exception as exc:
        print(f"ERROR: {exc}", flush=True)
        return 1
    print(f"ASIA_DIAGNOSTIC_WORKER_STATUS complete={len(completed)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
