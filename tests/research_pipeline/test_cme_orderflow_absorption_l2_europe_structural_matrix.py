from __future__ import annotations

import inspect
from collections import Counter
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from research_pipeline.cme_orderflow_absorption_l2_v1 import causal_master_tape as master
from research_pipeline.cme_orderflow_absorption_l2_v1 import europe_w04_replay as europe
from research_pipeline.cme_orderflow_absorption_l2_v1 import europe_w04_structural_matrix as structural
from research_pipeline.cme_orderflow_absorption_l2_v1.model import (
    ENTRY_LATENCY_NS, MAX_CONFIRMATION_NS, MIN_CONFIRMATION_NS,
    RISK_BUDGET_USD, STOP_BUFFER_TICKS, TARGET_R, TICK, VICINITY_TICKS,
    Execution, initial_prices, size_for_instrument,
)


def _profile() -> dict[str, float]:
    return {"high": 5_002.0, "low": 4_998.0, "poc": 5_000.0, "vah": 5_001.0, "val": 4_999.0}


def _utc(value: int) -> str:
    return datetime.fromtimestamp(value / 1e9, tz=UTC).strftime("%H:%M:%S")


def _audit(tmp_path: Path) -> Path:
    root = tmp_path / "coverage"
    europe.build_europe_coverage_audit(Path.cwd(), root)
    return root


def test_01_europe_timezone_dst_conversion() -> None:
    summer = europe.europe_session_bounds("2026-07-06")
    winter = europe.europe_session_bounds("2026-01-05")
    assert (_utc(summer["start_ns"]), _utc(summer["signal_end_ns"]), _utc(summer["cutoff_ns"])) == ("07:00:00", "15:30:00", "15:55:00")
    assert (_utc(winter["start_ns"]), _utc(winter["signal_end_ns"]), _utc(winter["cutoff_ns"])) == ("08:00:00", "16:30:00", "16:55:00")


def test_02_canonical_session_boundaries_are_frozen() -> None:
    assert europe.EUROPE_TIMEZONE == "Europe/London"
    assert europe.EUROPE_START_LOCAL.isoformat() == "08:00:00"
    assert europe.EUROPE_SIGNAL_END_LOCAL.isoformat() == "16:30:00"
    assert europe.EUROPE_CUTOFF_LOCAL.isoformat() == "16:55:00"


def test_03_session_start_is_inclusive() -> None:
    start = europe.europe_session_bounds("2026-07-06")["start_ns"]
    assert europe.in_europe_window("2026-07-06", start)


def test_04_session_end_is_exclusive_and_blocks_new_interactions() -> None:
    end = europe.europe_session_bounds("2026-07-06")["signal_end_ns"]
    assert europe.in_europe_window("2026-07-06", end - 1)
    assert not europe.in_europe_window("2026-07-06", end)
    engine = structural._fixed_engine(structural.CELL_BY_FAMILY[structural.PRIOR_POC], _profile())
    europe.observe_execution_with_session_gate(engine, Execution(end, 5_000.0, 1, "SELL"), signal_end_ns=end)
    assert not engine.active


def test_05_weekend_previous_session_mapping(tmp_path: Path) -> None:
    sessions = europe.load_audit_sessions(_audit(tmp_path))
    monday = next(row for row in sessions if row.day == "2026-07-06")
    assert monday.prior_day == "2026-07-02"


def test_06_prior_europe_poc_is_exact_and_lower_tie_break() -> None:
    assert structural.europe_volume_profile(Counter({20_000: 10, 20_001: 10}))["poc"] == 5_000.0


def test_07_prior_europe_high_low_are_execution_extrema() -> None:
    profile = structural.europe_volume_profile(Counter({19_998: 2, 20_000: 10, 20_003: 1}))
    assert profile["low"] == 4_999.5
    assert profile["high"] == 5_000.75


def test_08_prior_europe_vah_val_match_canonical_profile() -> None:
    values = Counter({19_998: 2, 19_999: 7, 20_000: 20, 20_001: 9, 20_002: 3})
    assert structural.assert_profile_parity(values) == structural.europe_volume_profile(values)


def test_09_current_europe_high_sweep_revises_one_lifecycle() -> None:
    engine = structural.CurrentEuropeSweepEngine(structural.CURRENT_HIGH, "HIGH")
    for timestamp, price in ((1, 5_000.0), (2, 5_001.25)):
        event = Execution(timestamp, price, 2, "SELL")
        engine.prepare_execution(event)
        engine.observe_execution(event)
    assert engine.current_extreme == 5_001.25
    assert engine._sweep_sequence == 1


def test_10_current_europe_low_sweep_revises_one_lifecycle() -> None:
    engine = structural.CurrentEuropeSweepEngine(structural.CURRENT_LOW, "LOW")
    for timestamp, price in ((1, 5_000.0), (2, 4_998.75)):
        event = Execution(timestamp, price, 2, "BUY")
        engine.prepare_execution(event)
        engine.observe_execution(event)
    assert engine.current_extreme == 4_998.75
    assert engine._sweep_sequence == 1


def test_11_current_sweep_has_no_future_extreme_leakage() -> None:
    engine = structural.CurrentEuropeSweepEngine(structural.CURRENT_HIGH, "HIGH")
    first = Execution(1, 5_000.0, 1, "SELL")
    engine.prepare_execution(first)
    assert engine.current_extreme == 5_000.0
    assert structural.sweep_semantic_parity()["invariants"]["future_session_high_low_used"] is False


def test_12_w04_weights_are_unchanged() -> None:
    assert europe.W04_WEIGHTS == {
        "aggression_score": Decimal("0.20"), "restoration_score": Decimal("0.10"),
        "price_resistance_score": Decimal("0.30"), "persistence_score": Decimal("0.20"),
        "multi_level_support_score": Decimal("0.20"),
    }


def test_13_q_is_unchanged() -> None:
    assert europe.QUALITY_THRESHOLD == Decimal("0.45")
    assert europe.W04_CONFIG.min_quality_score == 0.45


def test_14_feature_formulas_are_reused() -> None:
    source = inspect.getsource(structural.interaction_row)
    assert "interaction.feature_inputs()" in source
    assert "interaction.component_scores()" in source
    assert "master.recompute_quality(row, europe.W04_WEIGHTS)" in source


def test_15_interaction_radius_is_unchanged() -> None:
    assert VICINITY_TICKS == 4
    assert TICK == 0.25


def test_16_confirmation_is_unchanged() -> None:
    assert (MIN_CONFIRMATION_NS, MAX_CONFIRMATION_NS) == (5_000_000_000, 15_000_000_000)
    assert "favorable >= 3" in inspect.getsource(master.CausalWindowTracker.observe_es_execution)


def test_17_latency_is_two_ms() -> None:
    assert ENTRY_LATENCY_NS == 2_000_000


def test_18_zone_stop_is_five_ticks() -> None:
    assert STOP_BUFFER_TICKS == 5
    assert initial_prices("BUYER_ABSORPTION", 99.75, 100.0, 100.0, 100.0)["stop"] == 98.75


def test_19_target_is_three_r() -> None:
    assert TARGET_R == 3.0


def test_20_risk_budget_is_250_usd() -> None:
    assert RISK_BUDGET_USD == 250.0


def test_21_es_is_first_when_one_contract_fits() -> None:
    prices = initial_prices("BUYER_ABSORPTION", 99.75, 100.0, 99.75, 100.0)
    assert size_for_instrument(prices, "ES")["contracts"] >= 1


def test_22_mes_fallback_is_explicit_proxy() -> None:
    source = inspect.getsource(europe.EuropeProxySessionCausalTape)
    assert "MES_PROXY_FROM_ES" in source
    assert 'array("d", self.es_bid)' in source


def test_23_level_cells_have_independent_position_state() -> None:
    states = structural.build_cell_states(_profile())
    assert len(states) == 7
    assert len({id(row.engine) for row in states.values()}) == 7
    assert len({id(row.tracker) for row in states.values()}) == 7


def test_24_no_network_or_databento_calls() -> None:
    source = inspect.getsource(structural) + inspect.getsource(europe)
    assert "timeseries.get_range" not in source
    assert "metadata.get_cost" not in source
    assert "DATABENTO_API_KEY" not in source


def test_25_exact_eligible_session_derivation(tmp_path: Path) -> None:
    sessions = europe.load_audit_sessions(_audit(tmp_path))
    assert len(sessions) == 48
    assert sum(row.eligible for row in sessions) == 46
    assert [row.day for row in sessions if not row.eligible] == ["2026-05-04", "2026-06-23"]


def test_26_no_ny_or_asia_artifact_modification() -> None:
    root = Path.cwd()
    asia_before = structural.protected_asia_snapshot(root)
    ny_before = europe._protected_ny_snapshot(root)
    structural.semantic_diff_document()
    assert structural.protected_asia_snapshot(root) == asia_before
    assert europe._protected_ny_snapshot(root) == ny_before


def test_27_europe_cutoff_exit_uses_namespaced_reason() -> None:
    trade = {
        "trade_id": "trade-europe-cutoff",
        "exit_timestamp_ns": 1,
        "instrument": "ES",
        "r_multiple": 0.25,
        "net_pnl_usd": 50.0,
        "gross_pnl_usd": 56.0,
        "exit_reason": europe.EUROPE_HARD_FLAT_REASON,
        "execution_model": "ES_NATIVE_SOURCE",
    }

    assert europe._performance([trade])["hard_cutoff_exits"] == 1
