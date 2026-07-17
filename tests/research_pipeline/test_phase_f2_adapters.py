from __future__ import annotations

from pathlib import Path

import pytest

from research_pipeline.adapters.compatibility import verify_implementation_scope
from research_pipeline.adapters.data import DataAvailabilityGate
from research_pipeline.adapters.errors import ImplementationScopeViolation, RealAdapterRequired
from research_pipeline.adapters.native_backtest import NativeRepositoryAdapter
from research_pipeline.adapters.parameter_families import declared_parameter_families
from research_pipeline.adapters.registry import default_adapter_registry
from research_pipeline.schemas.strategy_spec import calculate_specification_hash, load_strategy_spec


ROOT = Path(__file__).parents[2]
SPEC_PATH = ROOT / "research_registry" / "spec_drafts" / "F2-real-breakout-demo_vphase-b-1.yaml"
RANDOM_SPEC_PATH = ROOT / "research_registry" / "spec_drafts" / "RandomOpenTest_vphase-b-1.yaml"


def test_explicit_registry_and_health():
    spec = load_strategy_spec(SPEC_PATH)
    registry = default_adapter_registry()
    health = registry.inspect(spec, ROOT)
    assert health.healthy is True
    assert "f2_native_demo" in registry.list()
    assert registry.resolve(spec, ROOT).identity.specification_hash == spec.specification_hash


def test_missing_adapter_is_hard_failure():
    spec = load_strategy_spec(SPEC_PATH).model_copy(update={"strategy_family": "not-registered", "specification_hash": "pending"})
    spec = spec.model_copy(update={"specification_hash": calculate_specification_hash(spec)})
    with pytest.raises(RealAdapterRequired, match="REAL_ADAPTER_REQUIRED"):
        default_adapter_registry().resolve(spec, ROOT)


def test_data_gate_requires_declared_proxy():
    gate = DataAvailabilityGate(ROOT)
    assert gate.check("SPX", "1h", source_symbol="SPY", allow_proxy=False).classification.value == "MANUAL_MAPPING_REQUIRED"
    available = gate.check("SPX", "1h", source_symbol="SPY", allow_proxy=True)
    assert available.classification.value == "AVAILABLE_PROXY"
    assert available.dataset_hash


def test_parameter_families_are_bounded_and_declared():
    spec = load_strategy_spec(SPEC_PATH)
    assert declared_parameter_families(spec) == []


def test_forbidden_scope_is_blocked():
    with pytest.raises(ImplementationScopeViolation, match="IMPLEMENTATION_SCOPE_VIOLATION"):
        verify_implementation_scope(["src/fib_backtester/strategy/fibonacci.py"])


def test_native_demo_uses_real_data_and_hashable_outputs(tmp_path):
    spec = load_strategy_spec(SPEC_PATH)
    adapter = NativeRepositoryAdapter(spec, ROOT, {"SPX": "SPY"})
    availability = adapter.require_data(spec)
    assert availability[0].classification.value == "AVAILABLE_PROXY"
    from research_pipeline.phase_f1.service import MasterPipelineService
    split = MasterPipelineService._real_split(adapter, spec)
    first = adapter.run_baseline(spec, split, tmp_path / "one")
    second = adapter.run_baseline(spec, split, tmp_path / "two")
    assert first.metrics["completed_trades"] > 0
    assert first.metrics == second.metrics
    assert adapter.normalized_last_run() is not None
    assert first.diagnostic_manifest_path


def test_real_b5_runs_against_native_diagnostic(tmp_path):
    spec = load_strategy_spec(SPEC_PATH)
    from research_pipeline.config.defaults import DEFAULT_BUDGETS
    from research_pipeline.controller.pipeline_controller import PipelineController
    from research_pipeline.enums import PipelineState
    from research_pipeline.registry.database import Database
    from research_pipeline.registry.repositories import Registry
    from research_pipeline.phase_f1.service import MasterPipelineService
    from research_pipeline.verification.services import VerificationService
    registry = Registry(Database(tmp_path / "registry.sqlite3")); controller = PipelineController(registry)
    controller.register_strategy(spec, str(SPEC_PATH), DEFAULT_BUDGETS); controller.submit_specification(spec.strategy_id); controller.approve_specification(spec.strategy_id)
    controller.transition(spec.strategy_id, PipelineState.IMPLEMENTATION_VERIFICATION, "test native adapter")
    adapter = NativeRepositoryAdapter(spec, ROOT, {"SPX": "SPY"}); split = MasterPipelineService._real_split(adapter, spec)
    controller.create_split(spec.strategy_id, split)
    artifact = adapter.run_baseline(spec, split, tmp_path / "baseline")
    result = VerificationService(tmp_path / "registry.sqlite3").run(spec.strategy_id, artifact.diagnostic_manifest_path)
    assert result["outcome"] == "VERIFIED"


def test_random_open_test_is_dst_safe_deterministic_and_same_bar(tmp_path):
    spec = load_strategy_spec(RANDOM_SPEC_PATH)
    adapter = NativeRepositoryAdapter(spec, ROOT)
    from research_pipeline.phase_f1.service import MasterPipelineService
    from datetime import datetime
    split = MasterPipelineService._real_split(adapter, spec)
    first = adapter.run_baseline(spec, split, tmp_path / "random-one")
    second = adapter.run_baseline(spec, split, tmp_path / "random-two")
    assert first.metrics == second.metrics
    assert first.metrics["implementation_variant"] == "1-hour repository-compatible test variant"

    import json
    trades = [json.loads(line) for line in Path(first.experiment_dir, "trades.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(trades) == len({item["trade_id"] for item in trades})
    assert all(item["exit_reason"] == "same_hour_bar_close" for item in trades)
    assert all((datetime.fromisoformat(item["exit_time"]) - datetime.fromisoformat(item["entry_time"])).total_seconds() == 3600 for item in trades)
    assert all(item["exit_time"] > item["entry_time"] for item in trades)
    assert all(item["direction"] in {"long", "short"} for item in trades)
    first_entry = trades[0]
    assert first_entry["quantity"] == pytest.approx(10000 * 0.05 / first_entry["entry"])
    local_dates = []
    offsets = set()
    from zoneinfo import ZoneInfo
    timezone = ZoneInfo("America/New_York")
    for item in trades:
        entry = datetime.fromisoformat(item["entry_time"])
        local = entry.astimezone(timezone)
        assert local.hour == 9 and local.minute == 30
        local_dates.append(local.date())
        offsets.add(local.utcoffset())
    assert len(local_dates) == len(set(local_dates))
    assert min(local_dates).isoformat() >= "2025-01-01"
    assert max(local_dates).isoformat() < "2026-01-01"
    assert len(offsets) == 2
