from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..schemas.splits import SplitDefinition
from ..schemas.strategy_spec import StrategySpec
from ..verification.fixtures import make_fixture
from .models import ResearchArtifact
from .runner import StrategyResearchAdapter, hash_file


class SyntheticFixtureAdapter:
    """Deterministic non-market fixture used for Phase C tests and dry runs."""

    def __init__(self, scenario: str = "strong-stable"):
        self.scenario = scenario

    def validate_environment(self) -> dict[str, Any]:
        return {"valid": True, "adapter": "synthetic", "scenario": self.scenario}

    def _metrics(self, phase: str, parameters: dict[str, Any] | None = None) -> dict[str, Any]:
        scenario = self.scenario
        if scenario == "no-edge": base = {"completed_trades": 50, "expectancy_r": -0.05, "profit_factor": 0.92, "max_drawdown": 0.35, "fee_share_of_gross_profit": 0.7, "executable_trades_per_month": 7.0, "median_days_between_trades": 4.0, "zero_trade_month_percentage": 0.1}
        elif scenario == "insufficient-trade": base = {"completed_trades": 8, "expectancy_r": 0.2, "profit_factor": 1.3, "max_drawdown": 0.15, "fee_share_of_gross_profit": 0.2, "executable_trades_per_month": 1.0, "median_days_between_trades": 35.0, "zero_trade_month_percentage": 0.65}
        elif scenario == "portfolio-component": base = {"completed_trades": 35, "expectancy_r": 0.24, "profit_factor": 1.35, "max_drawdown": 0.16, "fee_share_of_gross_profit": 0.2, "executable_trades_per_month": 1.8, "median_days_between_trades": 18.0, "zero_trade_month_percentage": 0.35}
        elif scenario == "stress-sensitive": base = {"completed_trades": 60, "expectancy_r": 0.23, "profit_factor": 1.35, "max_drawdown": 0.15, "fee_share_of_gross_profit": 0.2, "executable_trades_per_month": 8.0, "median_days_between_trades": 4.0, "zero_trade_month_percentage": 0.1}
        else: base = {"completed_trades": 60, "expectancy_r": 0.25, "profit_factor": 1.45, "max_drawdown": 0.12, "fee_share_of_gross_profit": 0.15, "executable_trades_per_month": 8.0, "median_days_between_trades": 4.0, "zero_trade_month_percentage": 0.08}
        if phase == "parameter_experiment" and parameters:
            value = next(iter(parameters.values()))
            if self.scenario == "isolated-maximum": base["expectancy_r"] = 0.45 if value == 5 else (-0.05 if value == 3 else 0.08)
            elif self.scenario == "stable-plateau": base["expectancy_r"] = 0.29 if value in {4, 5, 6} else 0.08
            elif isinstance(value, (int, float)): base["expectancy_r"] += max(-0.02, min(0.02, (float(value) - 5) * 0.002))
        if phase == "walk_forward" and self.scenario == "walk-forward-failure": base.update({"profitable_fold_ratio": 0.25, "validation_trades": 30, "validation_drawdown": 0.4, "validation_profit_factor": 0.8})
        else: base.update({"profitable_fold_ratio": 0.75, "validation_trades": max(30, int(base["completed_trades"] * .6)), "validation_drawdown": base["max_drawdown"], "validation_profit_factor": base["profit_factor"]})
        if phase == "holdout" and self.scenario == "holdout-failure": base.update({"holdout_trades": 30, "holdout_expectancy_r": -0.1, "holdout_drawdown": 0.4, "holdout_profit_factor": .8})
        else: base.update({"holdout_trades": max(30, int(base["completed_trades"] * .5)), "holdout_expectancy_r": base["expectancy_r"], "holdout_drawdown": base["max_drawdown"], "holdout_profit_factor": base["profit_factor"]})
        return base

    def _artifact(self, spec: StrategySpec, split: SplitDefinition, phase: str, output_dir: Path, parameters: dict[str, Any], experiment_id: str) -> ResearchArtifact:
        output_dir.mkdir(parents=True, exist_ok=True)
        input_path = output_dir / "input.json"
        metrics_path = output_dir / "metrics.json"
        payload = {"strategy_id": spec.strategy_id, "strategy_version": spec.version, "specification_hash": spec.specification_hash, "split_hash": split.split_hash, "dataset_hash": split.source_data_hash, "parameters": parameters, "experiment_id": experiment_id}
        input_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        metrics = self._metrics(phase, parameters)
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        manifest_path = None
        if phase == "baseline":
            manifest_path = make_fixture(output_dir / "b5", spec.strategy_id, spec.version)
        process = output_dir / "process_result.json"
        process.write_text(json.dumps({"command": ["synthetic-fixture", self.scenario, phase], "exit_code": 0, "status": "COMPLETED"}, indent=2, sort_keys=True), encoding="utf-8")
        return ResearchArtifact(experiment_id=experiment_id, strategy_id=spec.strategy_id, strategy_version=spec.version, phase=phase, experiment_dir=str(output_dir), input_path=str(input_path), metrics_path=str(metrics_path), diagnostic_manifest_path=str(manifest_path) if manifest_path else None, report_hashes={"metrics.json": hash_file(metrics_path), "process_result.json": hash_file(process)}, dataset_hash=split.source_data_hash, split_hash=split.split_hash, command=["synthetic-fixture", self.scenario, phase], status="COMPLETED", metrics=metrics)

    def run_baseline(self, spec, split, output_dir): return self._artifact(spec, split, "baseline", output_dir, spec.baseline_parameters, f"baseline-{spec.strategy_id}-{spec.version}")
    def run_parameter_experiment(self, spec, split, parameters, output_dir, experiment_id): return self._artifact(spec, split, "parameter_experiment", output_dir, parameters, experiment_id)
    def run_walk_forward(self, spec, split, parameters, output_dir): return self._artifact(spec, split, "walk_forward", output_dir, parameters, f"walk-forward-{spec.strategy_id}-{spec.version}")
    def run_holdout(self, spec, split, parameters, output_dir): return self._artifact(spec, split, "holdout", output_dir, parameters, f"holdout-{spec.strategy_id}-{spec.version}")
    def run_stress_test(self, spec, split, parameters, output_dir): return self._artifact(spec, split, "stress", output_dir, parameters, f"stress-{spec.strategy_id}-{spec.version}")
    def run_throughput_analysis(self, spec, split, parameters, output_dir): return self._artifact(spec, split, "throughput", output_dir, parameters, f"throughput-{spec.strategy_id}-{spec.version}")
    def generate_diagnostics(self, spec, split, parameters, output_dir): return {"scenario": self.scenario}
    def collect_metrics(self, artifact): return artifact.metrics
