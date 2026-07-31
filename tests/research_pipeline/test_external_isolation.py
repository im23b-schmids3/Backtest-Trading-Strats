from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

from research_pipeline.phase_f1.service import MasterPipelineService
from research_pipeline.runners.isolated_environment import (
    IsolationError,
    build_isolated_environment,
    import_origin_preflight,
    normalize_python_command,
    prepare_required_fixtures,
    validate_required_fixture_sources,
)
from research_pipeline.runners.test_runner import DeterministicTestRunner


ROOT = Path(__file__).parents[2].resolve()


def test_import_origin_resolves_inside_worktree_and_primary_src_is_removed(tmp_path: Path) -> None:
    worktree = tmp_path / "isolated worktree"
    (worktree / "src" / "research_pipeline").mkdir(parents=True)
    (worktree / "src" / "research_pipeline" / "__init__.py").write_text("WORKTREE_ONLY = True\n", encoding="utf-8")
    environment, report = build_isolated_environment(worktree, repository_root=ROOT, base_environment={"PYTHONPATH": str(ROOT / "src")})
    assert str(ROOT / "src") not in report["pythonpath"]
    origin = import_origin_preflight(worktree, environment=environment)
    assert Path(origin["origin"]).resolve().is_relative_to((worktree / "src").resolve())
    assert "WORKTREE_ONLY" not in (ROOT / "src" / "research_pipeline" / "__init__.py").read_text(encoding="utf-8")


def test_import_origin_mismatch_fails_closed(tmp_path: Path) -> None:
    environment, _ = build_isolated_environment(tmp_path, repository_root=ROOT, base_environment={"PYTHONPATH": str(ROOT / "src")})
    with pytest.raises(IsolationError, match="import-origin mismatch"):
        import_origin_preflight(tmp_path, environment=environment)


def test_fixture_preparation_copies_real_fixtures_and_missing_source_fails(tmp_path: Path) -> None:
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    report = prepare_required_fixtures(ROOT, worktree)
    assert report["valid"]
    assert (worktree / "research_registry" / "spec_drafts").is_dir()
    assert (worktree / "tests" / "fixtures").is_dir()
    with pytest.raises(IsolationError, match="required source fixtures"):
        validate_required_fixture_sources(tmp_path / "missing-primary")


def test_pytest_runner_uses_writable_basetemp_and_disables_cacheprovider(tmp_path: Path) -> None:
    worktree = tmp_path / "worktree"
    (worktree / "src" / "research_pipeline").mkdir(parents=True)
    (worktree / "src" / "research_pipeline" / "__init__.py").write_text("VALUE = 7\n", encoding="utf-8")
    (worktree / "tests").mkdir()
    (worktree / "tests" / "test_symbol.py").write_text(
        "from research_pipeline import VALUE\n\ndef test_worktree_symbol():\n    assert VALUE == 7\n",
        encoding="utf-8",
    )
    result = DeterministicTestRunner().run(
        worktree,
        ["python", "-m", "pytest", "-q", "tests/test_symbol.py"],
        dry_run=False,
        source_repository_root=tmp_path,
        basetemp=tmp_path / "job-temp" / "pytest",
    )
    assert result.passed
    assert str(sys.executable) == result.command[0]
    assert "no:cacheprovider" in result.command
    assert "--basetemp" in result.command
    assert (tmp_path / "job-temp" / "pytest").is_dir()


def test_windows_paths_use_one_python_executable(tmp_path: Path) -> None:
    command = normalize_python_command(["py", "-m", "pytest"])
    assert command == [sys.executable, "-m", "pytest"]
    assert " " in str(tmp_path / "path with spaces") or Path(tmp_path).exists()


def test_registry_discovery_uses_run_pointer_and_same_store(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite3"
    connection = sqlite3.connect(registry_path)
    connection.execute("CREATE TABLE master_runs (run_id TEXT PRIMARY KEY)")
    connection.execute("INSERT INTO master_runs(run_id) VALUES (?)", ("run-discovery",))
    connection.commit()
    connection.close()
    marker = tmp_path / "research_runs" / "Demo" / "run-discovery" / "run"
    marker.mkdir(parents=True)
    (marker / "registry.json").write_text(json.dumps({"run_id": "run-discovery", "registry_path": str(registry_path)}), encoding="utf-8")
    assert MasterPipelineService.discover_registry_path("run-discovery", tmp_path) == registry_path.resolve()
