from __future__ import annotations

from datetime import date, datetime, time, timezone

import pandas as pd
import pytest

from research_pipeline.compliance import EconomicEvent, NewsTradingPolicy, SessionPolicy
from research_pipeline.strategies.random_open_test import (
    RandomOpenTestConfig,
    default_random_open_cost_config,
    default_random_open_policy,
    generate_random_open_signals,
    run_random_open_test,
    stable_random_direction,
)


def bars() -> pd.DataFrame:
    index = pd.to_datetime(["2025-01-02T14:30:00Z", "2025-01-02T19:30:00Z", "2025-01-03T14:30:00Z", "2025-01-03T19:30:00Z"])
    return pd.DataFrame({"open": [100, 100.1, 101, 101.1], "high": [100.2, 100.3, 101.2, 101.3], "low": [99.8, 99.9, 100.8, 100.9], "close": [100.05, 100.15, 101.05, 101.15], "volume": [1, 1, 1, 1]}, index=index)


def test_direction_is_stable_and_does_not_use_process_hash() -> None:
    assert stable_random_direction("seed", "SPY", "2025-01-02") == stable_random_direction("seed", "SPY", date(2025, 1, 2))
    assert stable_random_direction("seed", "SPY", "2025-01-02") in {"LONG", "SHORT"}
    assert stable_random_direction("different-seed", "SPY", "2025-01-02") == stable_random_direction("different-seed", "SPY", "2025-01-02")


def test_one_signal_per_day_and_no_reentry() -> None:
    config = RandomOpenTestConfig(test_start_date=date(2025, 1, 1), test_end_date=date(2025, 1, 4), quantity=2, initial_stop_ticks=4, profit_target_ticks=8)
    signals = generate_random_open_signals(bars(), config)
    assert len(signals) == 2
    assert len({item.trading_date for item in signals}) == 2
    assert all(item.entry_timestamp.hour == 14 and item.entry_timestamp.minute == 30 for item in signals)
    assert all(item.quantity == 2 for item in signals)


def test_reference_run_applies_costs_and_records_forced_flat() -> None:
    config = RandomOpenTestConfig(forced_flat_time=time(15, 30), quantity=2, initial_stop_ticks=40, profit_target_ticks=80)
    result = run_random_open_test(bars(), config)
    assert result.proposed_entries == 2
    assert result.accepted_entries == 2
    assert result.blocked_entries == 0
    assert result.forced_flat_trade_count == 2
    assert all(item["exit_reason"] == "forced_flat" for item in result.trades)
    assert result.commissions > 0
    assert result.slippage_cost > 0
    assert result.net_pnl == pytest.approx(result.gross_pnl - result.commissions - result.fees - result.slippage_cost)
    assert result.policy_hash and result.execution_cost_configuration_hash
    assert result.diagnostics["average_holding_minutes"] > 0


def test_news_blocked_signal_is_persisted() -> None:
    config = RandomOpenTestConfig(forced_flat_time=time(15, 30))
    base = default_random_open_policy(config).model_dump(mode="python")
    base["news"] = NewsTradingPolicy(enabled=True, impact_levels=["HIGH"], minutes_before=5, minutes_after=5, applicable_instruments=["SPY"]).model_dump(mode="python")
    base["policy_hash"] = "pending"
    from research_pipeline.compliance import PropFirmPolicy, calculate_policy_hash
    candidate = PropFirmPolicy.model_validate(base, context={"skip_policy_hash_validation": True})
    base["policy_hash"] = calculate_policy_hash(candidate)
    event = EconomicEvent(event_id="e1", title="fixture event", timestamp=datetime(2025, 1, 2, 14, 30, tzinfo=timezone.utc), impact_level="HIGH", affected_instruments=["SPY"], source="fixture", retrieved_at=datetime(2025, 1, 2, 14, tzinfo=timezone.utc), source_data_hash="fixture")
    from research_pipeline.compliance import MarketState
    result = run_random_open_test(bars(), config, policy=PropFirmPolicy.model_validate(base), market_state=MarketState(calendar_source_hash="fixture-calendar", calendar_events=[event], calendar_retrieved_at=event.retrieved_at))
    assert result.blocked_entries == 1
    assert result.accepted_entries == 1
    assert result.blocked_signals[0]["decision_hash"]
    assert result.blocked_signals[0]["required_actions"] == []


def test_cost_configuration_is_instrument_specific() -> None:
    config = RandomOpenTestConfig()
    costs = default_random_open_cost_config(config)
    assert costs.instruments["SPY"].tick_size == .01
    assert costs.configuration_hash


def test_reference_intake_and_adapter_registration_are_normal_pipeline_paths(tmp_path) -> None:
    from research_pipeline.adapters.registry import default_adapter_registry
    from research_pipeline.phase_b.models import WorkflowInput
    from research_pipeline.phase_b.services import PhaseBService

    request = WorkflowInput(strategy_name="RandomOpenTest", natural_language_description="RandomOpenTest uses fixed quantity, an initial stop in ticks, a fixed profit target in ticks, and shared session forced-flat behavior for integration testing.", requested_markets=["SPY"], requested_timeframes=["1h"], repository_root=str(tmp_path), registry_path=str(tmp_path / "registry.sqlite3"), dry_run=True, implementation_enabled=False)
    generated = PhaseBService(tmp_path / "registry.sqlite3").generate_spec(request)
    assert generated.specification_hash
    from research_pipeline.schemas.strategy_spec import load_strategy_spec
    spec = load_strategy_spec(generated.specification_path)
    assert spec.strategy_family == "f2_random_open_reference"
    assert default_adapter_registry().inspect(spec, tmp_path).healthy


def test_reference_adapter_backtest_persists_compliance_and_cost_artifacts(tmp_path) -> None:
    from pathlib import Path
    from research_pipeline.adapters.registry import default_adapter_registry
    from research_pipeline.phase_f1.service import MasterPipelineService
    from research_pipeline.phase_b.models import WorkflowInput
    from research_pipeline.phase_b.services import PhaseBService
    from research_pipeline.schemas.strategy_spec import load_strategy_spec

    root = Path(__file__).parents[2]
    request = WorkflowInput(strategy_name="RandomOpenTest", natural_language_description="RandomOpenTest uses fixed quantity, an initial stop in ticks, a fixed profit target in ticks, and shared session forced-flat behavior for integration testing.", requested_markets=["SPY"], requested_timeframes=["1h"], repository_root=str(root), registry_path=str(tmp_path / "registry.sqlite3"), dry_run=True, implementation_enabled=False)
    generated = PhaseBService(tmp_path / "registry.sqlite3").generate_spec(request)
    spec = load_strategy_spec(generated.specification_path)
    adapter = default_adapter_registry().resolve(spec, root)
    split = MasterPipelineService._real_split(adapter, spec)
    artifact = adapter.run_baseline(spec, split, tmp_path / "reference-baseline")
    assert artifact.status == "COMPLETED"
    assert Path(artifact.experiment_dir, "random_open_compliance.json").is_file()
    assert artifact.metrics["proposed_entries"] >= artifact.metrics["accepted_entries"]
    assert artifact.metrics["execution_cost_configuration_hash"]

