from __future__ import annotations

import inspect
import math
from collections import Counter
from decimal import Decimal
from pathlib import Path

import pytest

from research_pipeline.cme_orderflow_absorption_l2_v1 import asia_w04_replay as asia
from research_pipeline.cme_orderflow_absorption_l2_v1 import asia_w04_structural_matrix as structural
from research_pipeline.cme_orderflow_absorption_l2_v1 import causal_master_tape as master
from research_pipeline.cme_orderflow_absorption_l2_v1 import weight_q_research as matrix
from research_pipeline.cme_orderflow_absorption_l2_v1.model import (
    ENTRY_LATENCY_NS,
    EXIT_RESET_NS,
    INACTIVITY_NS,
    MAX_CONFIRMATION_NS,
    MIN_CONFIRMATION_NS,
    RISK_BUDGET_USD,
    STOP_BUFFER_TICKS,
    TARGET_R,
    TICK,
    VICINITY_TICKS,
    Execution,
)


def _profile() -> dict[str, float]:
    return {"high": 5_002.0, "low": 4_998.0, "poc": 5_000.0, "vah": 5_001.0, "val": 4_999.0}


def test_exact_seven_level_registry_and_identifiers() -> None:
    assert structural.LEVEL_FAMILIES == (
        "PRIOR_ASIA_SESSION_POC",
        "PRIOR_ASIA_SESSION_HIGH",
        "PRIOR_ASIA_SESSION_LOW",
        "PRIOR_ASIA_SESSION_VAH",
        "PRIOR_ASIA_SESSION_VAL",
        "CURRENT_ASIA_HIGH_SWEEP",
        "CURRENT_ASIA_LOW_SWEEP",
    )
    assert len({cell.strategy_id for cell in structural.CELLS}) == 7


@pytest.mark.parametrize(
    ("family", "key"),
    [
        (structural.PRIOR_HIGH, "high"),
        (structural.PRIOR_LOW, "low"),
        (structural.PRIOR_POC, "poc"),
        (structural.PRIOR_VAH, "vah"),
        (structural.PRIOR_VAL, "val"),
    ],
)
def test_prior_asia_profile_level_is_exact(family: str, key: str) -> None:
    engine = structural._fixed_engine(structural.CELL_BY_FAMILY[family], _profile())
    assert len(engine.levels) == 1
    assert engine.levels[0].name == family
    assert engine.levels[0].price == _profile()[key]


def test_profile_poc_and_value_area_lower_price_ties() -> None:
    profile = structural.asia_volume_profile(Counter({20_000: 10, 19_999: 5, 20_001: 5}))
    assert profile == {
        "high": 5_000.25,
        "low": 4_999.75,
        "poc": 5_000.0,
        "vah": 5_000.0,
        "val": 4_999.75,
    }
    poc_tie = structural.asia_volume_profile(Counter({20_000: 10, 20_001: 10}))
    assert poc_tie["poc"] == 5_000.0


def test_profile_value_area_parity_with_canonical_research_implementation() -> None:
    values = Counter({19_998: 2, 19_999: 7, 20_000: 20, 20_001: 9, 20_002: 3})
    assert structural.assert_profile_parity(values) == structural.asia_volume_profile(values)


def test_current_asia_high_sweep_revises_one_active_lifecycle() -> None:
    engine = structural.CurrentAsiaSweepEngine(structural.CURRENT_HIGH, "HIGH")
    first = Execution(1, 5_000.0, 2, "SELL")
    engine.prepare_execution(first)
    engine.observe_execution(first)
    identifier = next(iter(engine.active))
    second = Execution(2, 5_001.25, 2, "SELL")
    engine.prepare_execution(second)
    engine.observe_execution(second)
    assert list(engine.active) == [identifier]
    assert engine.active[identifier].level.price == 5_001.25
    assert engine._sweep_sequence == 1


def test_current_asia_low_sweep_revises_one_active_lifecycle() -> None:
    engine = structural.CurrentAsiaSweepEngine(structural.CURRENT_LOW, "LOW")
    first = Execution(1, 5_000.0, 2, "BUY")
    engine.prepare_execution(first)
    engine.observe_execution(first)
    identifier = next(iter(engine.active))
    second = Execution(2, 4_998.75, 2, "BUY")
    engine.prepare_execution(second)
    engine.observe_execution(second)
    assert list(engine.active) == [identifier]
    assert engine.active[identifier].level.price == 4_998.75
    assert engine._sweep_sequence == 1


def test_current_sweeps_use_only_seen_execution_and_never_future_extreme() -> None:
    high = structural.CurrentAsiaSweepEngine(structural.CURRENT_HIGH, "HIGH")
    low = structural.CurrentAsiaSweepEngine(structural.CURRENT_LOW, "LOW")
    for price in (5_000.0, 4_999.0, 5_001.0):
        event = Execution(int(price * 100), price, 1, "SELL")
        high.prepare_execution(event)
        low.prepare_execution(event)
    assert high.current_extreme == 5_001.0
    assert low.current_extreme == 4_999.0


def test_same_execution_can_open_independent_high_and_low_cells() -> None:
    states = structural.build_cell_states(_profile())
    event = Execution(1, 5_000.0, 1, "SELL")
    for family in structural.SWEEP_FAMILIES:
        engine = states[family].engine
        assert isinstance(engine, structural.CurrentAsiaSweepEngine)
        engine.prepare_execution(event)
        engine.observe_execution(event)
    assert len(states[structural.CURRENT_HIGH].engine.active) == 1
    assert len(states[structural.CURRENT_LOW].engine.active) == 1
    assert states[structural.CURRENT_HIGH].engine is not states[structural.CURRENT_LOW].engine


def test_all_cells_have_independent_position_and_confirmation_state() -> None:
    states = structural.build_cell_states(_profile())
    assert len({id(state.engine) for state in states.values()}) == 7
    assert len({id(state.tracker) for state in states.values()}) == 7


def test_sweep_semantic_parity_assertion_passes() -> None:
    parity = structural.sweep_semantic_parity()
    assert parity["status"] == "PASS"
    assert parity["invariants"]["active_identity_excludes_revised_price"] is True
    assert parity["invariants"]["future_session_high_low_used"] is False


def test_exact_asia_window_inclusive_start_exclusive_end() -> None:
    start = asia._clock_ns("2026-01-02", 0)
    end = asia._clock_ns("2026-01-02", 8 * 3600)
    assert asia.in_asia_window("2026-01-02", start)
    assert asia.in_asia_window("2026-01-02", end - 1)
    assert not asia.in_asia_window("2026-01-02", end)


def test_prior_session_mapping_across_weekend_comes_from_frozen_audit() -> None:
    sessions = asia.load_audit_sessions(Path("research_runs/CMEOrderflowAbsorption.ES_L2_ASIA_DATA_COVERAGE_AUDIT"))
    monday = next(session for session in sessions if session.day == "2026-07-06")
    assert monday.prior_day == "2026-07-02"


def test_profile_only_seed_dates_remain_nontrading() -> None:
    sessions = asia.load_audit_sessions(Path("research_runs/CMEOrderflowAbsorption.ES_L2_ASIA_DATA_COVERAGE_AUDIT"))
    assert [session.day for session in sessions if not session.eligible] == ["2026-05-04", "2026-06-23"]
    assert sum(session.eligible for session in sessions) == 46


def test_frozen_w04_weights_and_q_unchanged() -> None:
    assert asia.W04_WEIGHTS == {
        "aggression_score": Decimal("0.20"),
        "restoration_score": Decimal("0.10"),
        "price_resistance_score": Decimal("0.30"),
        "persistence_score": Decimal("0.20"),
        "multi_level_support_score": Decimal("0.20"),
    }
    assert float(asia.QUALITY_THRESHOLD) == 0.45


def test_frozen_g_formula_implementation_is_reused() -> None:
    source = inspect.getsource(structural.interaction_row)
    assert "interaction.feature_inputs()" in source
    assert "interaction.component_scores()" in source
    assert "master.recompute_quality(row, asia.W04_WEIGHTS)" in source
    assert structural._fixed_engine(structural.CELL_BY_FAMILY[structural.PRIOR_POC], _profile()).config is asia.W04_CONFIG


def test_confirmation_semantics_are_unchanged() -> None:
    assert MIN_CONFIRMATION_NS == 5_000_000_000
    assert MAX_CONFIRMATION_NS == 15_000_000_000
    assert ENTRY_LATENCY_NS == 2_000_000
    source = inspect.getsource(master.CausalWindowTracker.observe_es_execution)
    assert "favorable >= 3" in source


def test_stop_target_risk_and_lifecycle_are_unchanged() -> None:
    assert VICINITY_TICKS == 4
    assert INACTIVITY_NS == 60_000_000_000
    assert EXIT_RESET_NS == 1_000_000_000
    assert STOP_BUFFER_TICKS == 5
    assert TARGET_R == 3.0
    assert RISK_BUDGET_USD == 250.0


def test_mes_proxy_execution_path_is_unchanged() -> None:
    assert issubclass(asia.AsiaProxySessionCausalTape, matrix.SessionCausalTape)
    source = inspect.getsource(asia.AsiaProxySessionCausalTape)
    assert "MES_PROXY_FROM_ES" in source
    assert "array(\"d\", self.es_bid)" in source


def test_semantic_document_allows_only_level_identity_source_differences() -> None:
    document = structural.semantic_diff_document()
    assert document["status"] == "PASS"
    assert document["unexpected_semantic_differences"] == []
    assert len(document["cells"]) == 7
    assert all(row["allowed_cell_differences"] == [
        "strategy_identifier", "structural_level_type", "structural_level_price/source",
    ] for row in document["cells"])


def test_quality_buckets_are_fixed_descriptive_only() -> None:
    assert structural.quality_bucket(0.2999) == "LOW_BELOW_0P30"
    assert structural.quality_bucket(0.30) == "MEDIUM_MISS_0P30_TO_0P40"
    assert structural.quality_bucket(0.40) == "NEAR_MISS_0P40_TO_0P45"
    assert structural.quality_bucket(0.45) == "SCORE_AT_OR_ABOVE_0P45"


def test_published_poc_baseline_headline_is_exact() -> None:
    summary = asia._read_json(Path("research_runs/CMEOrderflowAbsorption.ES_L2_W04_ASIA_POC_MES_PROXY/summary.json"))
    assert summary["raw_interactions"] == 742
    assert summary["accepted_setups"] == 9
    assert summary["confirmations_passed"] == 2
    assert summary["confirmations_failed"] == 7
    assert summary["performance"]["completed_trades"] == 2
    assert math.isclose(summary["performance"]["total_r"], -2.0)
    assert math.isclose(summary["performance"]["net_pnl_usd"], -491.0)


def test_no_network_databento_or_optimization_path() -> None:
    source = inspect.getsource(structural)
    assert "timeseries.get_range" not in source
    assert "metadata.get_cost" not in source
    assert "DATABENTO_API_KEY" not in source
    assert "Historical(" not in source
    assert "winner_selected\": False" in source
    assert "optimization_performed\": False" in source


def test_existing_matrix_output_collision_fails_before_source_read(tmp_path: Path) -> None:
    output = tmp_path / "exists"
    output.mkdir()
    with pytest.raises(FileExistsError, match="immutable Asia structural matrix output"):
        structural.run_matrix(
            repository_root=tmp_path,
            output_root=output,
            audit_root=tmp_path / "missing-audit",
            baseline_root=tmp_path / "missing-baseline",
        )


def test_existing_ny_and_asia_baseline_hashes_are_stable_during_semantic_audit() -> None:
    root = Path.cwd()
    baseline = root / structural.BASELINE_ROOT
    before = structural._root_snapshot(baseline)
    ny_before = asia._protected_ny_snapshot(root)
    structural.semantic_diff_document()
    assert structural._root_snapshot(baseline) == before
    assert asia._protected_ny_snapshot(root) == ny_before
