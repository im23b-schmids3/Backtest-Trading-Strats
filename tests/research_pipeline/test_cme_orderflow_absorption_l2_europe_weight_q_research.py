from __future__ import annotations

import inspect
from decimal import Decimal
from pathlib import Path

import pytest

from research_pipeline.cme_orderflow_absorption_l2_v1 import causal_master_tape as master
from research_pipeline.cme_orderflow_absorption_l2_v1 import europe_w04_replay as europe
from research_pipeline.cme_orderflow_absorption_l2_v1 import europe_w04_structural_matrix as structural
from research_pipeline.cme_orderflow_absorption_l2_v1 import europe_weight_q_research as research
from research_pipeline.cme_orderflow_absorption_l2_v1 import weight_q_research as common


def _interaction(identifier: str, *, score: float = 0.5, primitive: str = "") -> dict[str, object]:
    return {
        "interaction_id": identifier, "source_interaction_id": identifier,
        "session_date": "2026-05-04", "interaction_start_ns": 1,
        "interaction_end_ns": 2, "direction": "BUYER_ABSORPTION",
        "level": "PRIOR_EUROPE_SESSION_POC", "level_price": 100.0,
        "interaction_end_price": 100.0, "zone_low": 99.0, "zone_high": 100.0,
        "aggression_score": score, "restoration_score": score,
        "price_resistance_score": score, "persistence_score": score,
        "multi_level_support_score": score, "false_refill_penalty": 0.0,
        "non_quality_rejection_reasons": primitive,
    }


def _trade(identifier: str, *, direction: str = "LONG", instrument: str = "MES_PROXY_FROM_ES") -> dict[str, object]:
    return {
        "trade_id": f"T:{identifier}", "direction": direction,
        "instrument": instrument, "execution_model": instrument,
        "exit_reason": "TARGET", "net_pnl_usd": 250.0, "r_multiple": 1.0,
    }


def _result(identifier: str, units: tuple[int, int, int, int, int], *, r: float = 1.0, trades: int = 20) -> dict[str, object]:
    family = research.FAMILIES[0]
    return {
        "evidence_label": research.EVIDENCE_LABEL, "level": family,
        "config_id": research.config_id(family, units, Decimal(".50")),
        **{f"G{index}_weight": value / 20 for index, value in enumerate(units, 1)},
        "quality_threshold": .5, "trades": trades, "total_r": r,
        "positive_month_count": 3, "worst_month_r": -.5,
        "profit_factor": 1.2, "max_drawdown_r": -2.0, "LOW_SAMPLE": trades < 15,
    }


def test_exact_weight_q_and_level_cardinalities() -> None:
    grid = research.generate_weight_grid()
    assert len(grid) == 3_876
    assert all(sum(row) == 20 and min(row) >= 1 for row in grid)
    assert research.QUALITY_THRESHOLDS == tuple(Decimal(value) for value in (".45", ".50", ".55", ".60", ".65"))
    assert len(research.FAMILIES) == 7
    assert len(grid) * 5 == 19_380
    assert len(research.configuration_registry()) == 135_660


def test_w04_vector_is_naturally_present_in_complete_grid() -> None:
    assert research.W04_UNITS == (4, 2, 6, 4, 4)
    assert research.W04_UNITS in research.generate_weight_grid()
    assert Decimal(".45") in research.QUALITY_THRESHOLDS


def test_primitive_rules_are_never_bypassed_and_q_is_inclusive() -> None:
    row = _interaction("accepted", score=.45)
    assert research.accepted(row, (4, 4, 4, 4, 4), Decimal(".45"))
    rejected = _interaction("rejected", score=1.0, primitive="NO_GENUINE_CONSUME_RESTORE")
    assert not research.accepted(rejected, (4, 4, 4, 4, 4), Decimal(".45"))


def test_same_cached_g_values_are_read_only_across_weight_calculations() -> None:
    row = _interaction("stable", score=.5)
    before = dict(row)
    research.accepted(row, (4, 2, 6, 4, 4), Decimal(".45"))
    research.accepted(row, (1, 1, 1, 1, 16), Decimal(".65"))
    assert row == before


def test_cache_semantic_hash_tampering_fails_closed() -> None:
    cache = research.make_cache([_interaction("cache")], source_sessions=[str(index) for index in range(46)])
    research.validate_cache(cache)
    cache["rows"][0]["aggression_score"] = .9
    with pytest.raises(research.EuropeWeightQError, match="cache version/hash"):
        research.validate_cache(cache)


def test_stage_b_has_no_source_reader_network_or_databento_path() -> None:
    source = inspect.getsource(research._run_family).lower()
    assert "_stream_private_mbo" not in source
    assert ".dbn" not in source
    assert "databento" not in source
    assert "get_range" not in source and "get_cost" not in source


def test_stage_b_simulates_each_unique_accepted_set_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, ...]] = []

    def simulate(_tape: object, selected: list[dict[str, object]], _indexes: object) -> common.SessionResult:
        calls.append(tuple(str(row["interaction_id"]) for row in selected))
        return common.SessionResult(accepted_setups=len(selected), unresolved=len(selected))

    monkeypatch.setattr(common, "simulate_independent_session", simulate)
    monkeypatch.setattr(europe, "_classify_europe_entry_cutoff", lambda _result, *, tape: None)
    rows = [_interaction("one"), _interaction("two")]
    groups = {(0,): (0, 1, 2), (0, 1): (3, 4), (): (5,)}
    results = research._simulate_selection_groups(
        tape=object(), rows=rows, indexes={}, groups=groups,
    )

    assert calls == [("one",), ("one", "two"), ()]
    assert [tuple(indexes) for indexes, _result in results] == [(0, 1, 2), (3, 4), (5,)]


def test_stage_a_retention_is_opt_in_and_default_w04_behavior_stays_disposable() -> None:
    signature = inspect.signature(structural.MatrixSessionState)
    assert signature.parameters["retain_event_tape"].default is False
    assert signature.parameters["event_tape_root"].default is None
    source = inspect.getsource(structural.MatrixSessionState.finish)
    assert "if self.retain_event_tape" in source
    assert "unlink()" in source


def test_frozen_confirmation_execution_and_risk_contracts_are_unchanged() -> None:
    assert europe.QUALITY_THRESHOLD == Decimal(".45")
    assert europe.W04_WEIGHTS == {
        "aggression_score": Decimal(".20"), "restoration_score": Decimal(".10"),
        "price_resistance_score": Decimal(".30"), "persistence_score": Decimal(".20"),
        "multi_level_support_score": Decimal(".20"),
    }
    assert europe.ENTRY_LATENCY_NS == 2_000_000
    assert europe.STOP_BUFFER_TICKS == 5
    assert europe.TARGET_R == 3.0
    assert europe.RISK_BUDGET_USD == 250.0
    assert "MES_PROXY_FROM_ES" in inspect.getsource(europe.EuropeProxySessionCausalTape)


def test_each_configuration_accumulator_has_independent_position_and_month_state() -> None:
    first = research.ResearchAccumulator((4, 4, 4, 4, 4), Decimal(".50"))
    second = research.ResearchAccumulator((4, 4, 4, 4, 4), Decimal(".55"))
    session = common.SessionResult(accepted_setups=1, confirmations=1, trades=[_trade("one")])
    session.terminal_outcomes["one"] = "TRADE_EXECUTED"
    first.add("2026-05-04", session)
    assert first.aggregate.trades == 1
    assert second.aggregate.trades == 0
    assert first.months["2026-05"].trades == 1


def test_result_reconciliation_month_direction_and_execution_breakdowns() -> None:
    accumulator = research.ResearchAccumulator((4, 4, 4, 4, 4), Decimal(".50"))
    session = common.SessionResult(accepted_setups=2, confirmations=2, active_position_blocks=1, trades=[_trade("one")])
    session.terminal_outcomes.update({"one": "TRADE_EXECUTED", "two": "COMPLIANCE_BLOCK_ACTIVE_POSITION"})
    accumulator.add("2026-06-04", session)
    row = accumulator.result_row(research.FAMILIES[0], raw_interactions=9, primitive_eligible=7)
    months, directions, executions = research._segment_rows(research.FAMILIES[0], accumulator)
    assert row["quality_accepted"] == 2 and row["trades"] == 1 and row["blocked_setups"] == 1
    assert sum(item["trades"] for item in months) == 1
    assert next(item for item in directions if item["direction"] == "LONG")["trades"] == 1
    assert next(item for item in executions if item["execution_model"] == "MES_PROXY_FROM_ES")["trades"] == 1


def test_neighbor_generation_uses_weight_transfer_and_adjacent_q_only() -> None:
    values = research.neighbors((4, 4, 4, 4, 4), Decimal(".50"))
    assert len(values) == 22
    assert all(sum(units) == 20 and min(units) >= 1 for units, _q in values)
    assert {q for units, q in values if units == (4, 4, 4, 4, 4)} == {Decimal(".45"), Decimal(".55")}


@pytest.mark.parametrize(
    ("threshold", "expected"),
    ((Decimal(".45"), {Decimal(".50")}), (Decimal(".65"), {Decimal(".60")})),
)
def test_neighbor_generation_stays_inside_sealed_q_boundaries(
    threshold: Decimal, expected: set[Decimal],
) -> None:
    units = (4, 4, 4, 4, 4)
    values = research.neighbors(units, threshold)
    assert {q for candidate, q in values if candidate == units} == expected
    assert all(q in research.QUALITY_THRESHOLDS for _candidate, q in values)


def test_plateau_connectivity_and_descriptive_buckets() -> None:
    a, b = (4, 4, 4, 4, 4), (3, 5, 4, 4, 4)
    rows = [_result("a", a), _result("b", b)]
    robust = [{
        "config_id": row["config_id"], "profitable_neighbor_fraction": 1.0,
        "neighbor_median_r": 1.0, "neighbor_worst_r": .5,
    } for row in rows]
    memberships, summaries = research.plateau_membership(rows, robust)
    buckets = research.descriptive_buckets(rows, robust)
    assert len(summaries) == 1 and summaries[0]["size"] == 2
    assert len(memberships) == 2
    assert len(buckets["A"]) == 2 and buckets["B"] == [] and buckets["C"] == []


@pytest.mark.parametrize(
    ("trades", "expected"),
    ((0, "LT_10"), (9, "LT_10"), (10, "10_TO_14"), (14, "10_TO_14"),
     (15, "15_TO_24"), (24, "15_TO_24"), (25, "25_TO_39"), (39, "25_TO_39"), (40, "40_PLUS")),
)
def test_low_sample_bands(trades: int, expected: str) -> None:
    assert research.sample_band(trades) == expected


def test_canonical_output_contract_names_are_present_in_stage_b_source() -> None:
    source = inspect.getsource(research.run_stage_b)
    for name in (
        "configuration-results.csv", "monthly-results.csv", "direction-results.csv",
        "execution-model-results.csv", "neighbor-robustness.csv", "plateau-membership.csv",
        "robustness-a.csv", "robustness-b.csv", "robustness-c.csv", "weight-grid.csv",
        "level-summary.csv", "q-summary.csv", "summary.json", "diagnostic-report.md",
    ):
        assert name in source


def test_no_production_selection_or_combined_level_strategy() -> None:
    assert research.NO_AUTOMATIC_SELECTION is True
    assert research.EVIDENCE_LABEL == "EUROPE_WEIGHT_Q_RETROSPECTIVE_RESEARCH_NOT_OOS"
    assert "No combined multi-level strategy" in inspect.getsource(research._report)


def test_causal_cache_validator_rejects_missing_manifest(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        research.validate_causal_cache(tmp_path / "missing")
