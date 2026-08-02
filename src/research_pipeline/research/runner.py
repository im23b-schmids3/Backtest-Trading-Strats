from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import yaml

from ..enums import PipelineState
from ..schemas.splits import SplitDefinition
from ..schemas.strategy_spec import StrategySpec
from .models import ResearchArtifact


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class StrategyResearchAdapter(Protocol):
    def validate_environment(self) -> dict[str, Any]: ...
    def run_baseline(self, spec: StrategySpec, split: SplitDefinition, output_dir: Path) -> ResearchArtifact: ...
    def run_parameter_experiment(self, spec: StrategySpec, split: SplitDefinition, parameters: dict[str, Any], output_dir: Path, experiment_id: str) -> ResearchArtifact: ...
    def run_walk_forward(self, spec: StrategySpec, split: SplitDefinition, parameters: dict[str, Any], output_dir: Path) -> ResearchArtifact: ...
    def run_holdout(self, spec: StrategySpec, split: SplitDefinition, parameters: dict[str, Any], output_dir: Path) -> ResearchArtifact: ...
    def run_stress_test(self, spec: StrategySpec, split: SplitDefinition, parameters: dict[str, Any], output_dir: Path) -> ResearchArtifact: ...
    def run_throughput_analysis(self, spec: StrategySpec, split: SplitDefinition, parameters: dict[str, Any], output_dir: Path) -> ResearchArtifact: ...
    def generate_diagnostics(self, spec: StrategySpec, split: SplitDefinition, parameters: dict[str, Any], output_dir: Path) -> dict[str, Any]: ...
    def collect_metrics(self, artifact: ResearchArtifact) -> dict[str, Any]: ...


class GenericSubprocessAdapter:
    """Adapter for a declared executable; it never builds shell command strings."""

    def __init__(self, commands: dict[str, list[str]], repository_root: str | Path = ".", timeout_seconds: int = 3600):
        self.commands = {key: list(value) for key, value in commands.items()}
        self.repository_root = Path(repository_root).resolve()
        self.timeout_seconds = timeout_seconds

    def validate_environment(self) -> dict[str, Any]:
        if not self.commands:
            raise ValueError("at least one declared adapter command is required")
        return {"valid": True, "commands": sorted(self.commands)}

    def _run(self, phase: str, spec: StrategySpec, split: SplitDefinition, parameters: dict[str, Any], output_dir: Path, experiment_id: str) -> ResearchArtifact:
        command = self.commands.get(phase)
        if not command:
            raise ValueError(f"no declared command for phase {phase}")
        output_dir.mkdir(parents=True, exist_ok=True)
        input_path = output_dir / "input.json"
        metrics_path = output_dir / "metrics.json"
        payload = {"strategy_id": spec.strategy_id, "strategy_version": spec.version, "specification_hash": spec.specification_hash,
                   "split": split.model_dump(mode="json"), "split_hash": split.split_hash, "parameters": parameters, "experiment_id": experiment_id}
        input_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        started = datetime.now(timezone.utc).isoformat()
        began = time.monotonic()
        try:
            completed = subprocess.run(command + ["--input", str(input_path), "--output", str(metrics_path)], cwd=str(self.repository_root), capture_output=True, text=True, timeout=self.timeout_seconds, shell=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"research adapter {phase} failed: {exc}") from exc
        if completed.returncode != 0:
            raise RuntimeError(f"research adapter {phase} exited {completed.returncode}: {completed.stderr[-2000:]}")
        if not metrics_path.is_file():
            raise RuntimeError(f"research adapter {phase} did not write metrics.json")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        process_path = output_dir / "process_result.json"
        process_path.write_text(json.dumps({"command": command, "start_time": started, "end_time": datetime.now(timezone.utc).isoformat(), "duration_ms": int((time.monotonic() - began) * 1000), "exit_code": completed.returncode, "stdout": completed.stdout[-4000:], "stderr": completed.stderr[-4000:]}, indent=2, sort_keys=True), encoding="utf-8")
        return ResearchArtifact(experiment_id=experiment_id, strategy_id=spec.strategy_id, strategy_version=spec.version, phase=phase,
            experiment_dir=str(output_dir), input_path=str(input_path), metrics_path=str(metrics_path), dataset_hash=split.source_data_hash,
            split_hash=split.split_hash, command=command, status="COMPLETED", metrics=metrics, report_hashes={"metrics.json": hash_file(metrics_path), "process_result.json": hash_file(process_path)})

    def _attach_diagnostics(self, artifact: ResearchArtifact, output_dir: Path) -> ResearchArtifact:
        command = self.commands.get("diagnostics") or self.commands.get("baseline_diagnostics")
        if not command:
            return artifact
        manifest_path = output_dir / "manifest.yaml"
        started = datetime.now(timezone.utc).isoformat()
        try:
            completed = subprocess.run(command + ["--input", artifact.input_path, "--output", str(manifest_path)], cwd=str(self.repository_root), capture_output=True, text=True, timeout=self.timeout_seconds, shell=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"research adapter diagnostics failed: {exc}") from exc
        if completed.returncode != 0:
            raise RuntimeError(f"research adapter diagnostics exited {completed.returncode}: {completed.stderr[-2000:]}")
        if not manifest_path.is_file():
            raise RuntimeError("research adapter diagnostics did not write manifest.yaml")
        diagnostic = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        process_path = output_dir / "diagnostics_process_result.json"
        process_path.write_text(json.dumps({"command": command, "start_time": started, "end_time": datetime.now(timezone.utc).isoformat(), "exit_code": completed.returncode, "stdout": completed.stdout[-4000:], "stderr": completed.stderr[-4000:]}, indent=2, sort_keys=True), encoding="utf-8")
        hashes = dict(artifact.report_hashes)
        hashes["manifest.yaml"] = hash_file(manifest_path)
        hashes["diagnostics_process_result.json"] = hash_file(process_path)
        return artifact.model_copy(update={"diagnostic_manifest_path": str(manifest_path), "diagnostic_manifest": diagnostic, "report_hashes": hashes})

    def run_baseline(self, spec, split, output_dir):
        artifact = self._run("baseline", spec, split, spec.baseline_parameters, output_dir, f"baseline-{spec.strategy_id}-{spec.version}")
        return self._attach_diagnostics(artifact, output_dir)
    def run_parameter_experiment(self, spec, split, parameters, output_dir, experiment_id): return self._run("parameter_experiment", spec, split, parameters, output_dir, experiment_id)
    def run_walk_forward(self, spec, split, parameters, output_dir): return self._run("walk_forward", spec, split, parameters, output_dir, f"walk-forward-{spec.strategy_id}-{spec.version}")
    def run_holdout(self, spec, split, parameters, output_dir): return self._run("holdout", spec, split, parameters, output_dir, f"holdout-{spec.strategy_id}-{spec.version}")
    def run_stress_test(self, spec, split, parameters, output_dir): return self._run("stress", spec, split, parameters, output_dir, f"stress-{spec.strategy_id}-{spec.version}")
    def run_throughput_analysis(self, spec, split, parameters, output_dir): return self._run("throughput", spec, split, parameters, output_dir, f"throughput-{spec.strategy_id}-{spec.version}")
    def generate_diagnostics(self, spec, split, parameters, output_dir): return {}
    def collect_metrics(self, artifact): return artifact.metrics
