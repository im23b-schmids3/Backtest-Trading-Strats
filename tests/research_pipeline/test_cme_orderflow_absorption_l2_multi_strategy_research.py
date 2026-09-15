from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_pipeline.cme_orderflow_absorption_l2_v1 import causal_master_tape as master
from research_pipeline.cme_orderflow_absorption_l2_v1 import historical_runner as historical
from research_pipeline.cme_orderflow_absorption_l2_v1 import multi_strategy_research as multi
from research_pipeline.cme_orderflow_absorption_l2_v1 import weight_q_research as matrix


DAY = "2026-09-01"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _interaction(identifier: str, level: str) -> dict[str, object]:
    source = f"{level}:100.00:{identifier}"
    return {
        "interaction_id": f"{DAY}|{source}", "source_interaction_id": source,
        "session_date": DAY, "interaction_end_ns": 1, "direction": "BUYER_ABSORPTION",
        "level": level, "zone_low": 99.0, "zone_high": 100.0,
        "aggression_score": 0.60, "restoration_score": 0.60,
        "price_resistance_score": 0.60, "persistence_score": 0.60,
        "multi_level_support_score": 0.60, "false_refill_penalty": 0.0,
        "non_quality_rejection_reasons": "",
    }


def _event(ordinal: int, timestamp: int, *, hard: bool = False, bid: float = 100.0) -> dict[str, object]:
    return {
        "session_date": DAY, "event_ordinal": ordinal, "timestamp_ns": timestamp,
        "stream": "CALENDAR" if hard else "ES", "event_type": "HARD_FLAT" if hard else "ES_BBO",
        "es_bid": bid, "es_ask": bid + 0.25, "mes_bid": bid, "mes_ask": bid + 0.25,
        "book_state": "EXECUTABLE", "hard_flat_reason": "HARD_CUTOFF_TEST" if hard else None,
        "es_quote_timestamp_ns": timestamp, "mes_quote_timestamp_ns": timestamp,
    }


def _make_manifests(tmp_path: Path) -> tuple[Path, Path]:
    interactions = [_interaction("HIGH", "PRIOR_EUROPE_HIGH"), _interaction("POC", "PRIOR_RTH_POC")]
    indexes = [{"interaction_id": row["interaction_id"], "session_date": DAY,
                "derived_first_confirmation_timestamp_ns": 5, "entry_observation_event_ordinal": 0,
                "counterfactual_path_end_ns": 20} for row in interactions]
    artifacts = tmp_path / "artifacts"; artifacts.mkdir()
    master._write_small_parquet(artifacts / "interaction-master.parquet", interactions)
    master._write_small_parquet(artifacts / "interaction-index.parquet", indexes)
    master._write_small_parquet(artifacts / "day.parquet", [_event(0, 10), _event(1, 11, bid=110.0), _event(2, 12, hard=True, bid=110.0)])
    period = tmp_path / "period.json"
    period.write_text(json.dumps({"schema_version": 1, "period_id": "synthetic", "interaction_master": "artifacts/interaction-master.parquet",
                                  "interaction_index": "artifacts/interaction-index.parquet",
                                  "sessions": [{"date": DAY, "event_tape": "artifacts/day.parquet"}]}), encoding="utf-8")
    strategies = tmp_path / "strategies.yaml"
    strategies.write_text("""schema_version: 1
strategies:
  - strategy_id: Europe High
    session: EUROPE
    reference_level: PRIOR_EUROPE_HIGH
    long_short_behavior: shared_absorption_reversal
    uses_shared_absorption_engine: true
  - strategy_id: NY POC
    session: NEW_YORK_RTH
    reference_level: PRIOR_RTH_POC
    long_short_behavior: shared_absorption_reversal
    uses_shared_absorption_engine: true
""", encoding="utf-8")
    return strategies, period


def test_manifest_stage1_matrix_outputs_and_resume(tmp_path: Path):
    strategies, period = _make_manifests(tmp_path)
    summary = multi.run_stage1(strategies_path=strategies, period_path=period, output_root=tmp_path / "stage1")
    assert [item["strategy_id"] for item in summary["strategies"]] == ["Europe High", "NY POC"]
    assert summary["matrix_per_strategy"] == 15
    assert summary["stage1_quality_threshold"] == "0.50"
    assert summary["common_baseline_weights"] == {key: str(value) for key, value in multi.COMMON_BASELINE_WEIGHTS.items()}
    for item in summary["strategies"]:
        root = Path(item["root"])
        assert len((root / "stage1-matrix.csv").read_text(encoding="utf-8").splitlines()) == 16
        selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))
        assert set(selection["stage2_execution_configuration"]) == {"config_id", "rr", "stop_ticks"}
    resumed = multi.run_stage1(strategies_path=strategies, period_path=period, output_root=tmp_path / "stage1")
    assert {item["status"] for item in resumed["strategies"]} == {"REUSED"}


def test_stage2_uses_canonical_weight_generator_q_grid_and_separate_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    strategies, period = _make_manifests(tmp_path)
    multi.run_stage1(strategies_path=strategies, period_path=period, output_root=tmp_path / "stage1")
    legal = ((4, 2, 6, 4, 4), (3, 3, 6, 4, 4))
    monkeypatch.setattr(matrix, "generate_weight_grid", lambda: legal)

    def forbidden_dbn(*_args, **_kwargs):
        raise AssertionError("Stage 2 must not read DBN data")

    monkeypatch.setattr(historical, "_stream_private_mbo", forbidden_dbn)

    def small_neighbors(rows):
        common = [{"config_id": row["config_id"], "neighbor_count": 1, "worst_neighbor_total_r": row["total_r"],
                   "median_neighbor_total_r": row["total_r"], "proportion_neighbors_profitable": 1.0} for row in rows]
        return common, common, common

    monkeypatch.setattr(multi, "_stage2_robustness", small_neighbors)
    summary = multi.run_stage2(strategies_path=strategies, period_path=period, stage1_root=tmp_path / "stage1", output_root=tmp_path / "stage2")
    assert summary["quality_thresholds"] == [f"0.{value:02d}" for value in range(30, 80, 5)]
    assert summary["configuration_count_per_strategy"] == 20
    assert all(item["status"] == "COMPLETE" for item in summary["strategies"])
    roots = [Path(item["root"]) for item in summary["strategies"]]
    assert roots[0] != roots[1]
    assert all(len((root / "weight-q-results.csv").read_text(encoding="utf-8").splitlines()) == 21 for root in roots)
    assert all(len(list((root / "weight-q-checkpoints").glob("*.json"))) == 2 for root in roots)


def test_robust_stage1_selection_rejects_an_isolated_total_r_peak():
    rows = multi._stage1_neighbors([
        {"config_id": "stable-a", "rr": 2.0, "stop_ticks": 5, "total_r": 5.0, "expectancy_r_per_session": 1.0, "max_cumulative_drawdown_r": -1.0, "profit_factor": 1.5, "trades": 10},
        {"config_id": "stable-b", "rr": 2.5, "stop_ticks": 5, "total_r": 4.0, "expectancy_r_per_session": 0.9, "max_cumulative_drawdown_r": -1.0, "profit_factor": 1.4, "trades": 10},
        {"config_id": "peak", "rr": 4.0, "stop_ticks": 3, "total_r": 20.0, "expectancy_r_per_session": 3.0, "max_cumulative_drawdown_r": -5.0, "profit_factor": 2.0, "trades": 2},
        {"config_id": "peak-neighbor", "rr": 3.0, "stop_ticks": 3, "total_r": -8.0, "expectancy_r_per_session": -1.0, "max_cumulative_drawdown_r": -5.0, "profit_factor": 0.5, "trades": 2},
    ])
    selected = multi._select_stage1(rows)
    assert selected["raw_best"]["config_id"] == "peak"
    assert selected["stage2_execution_configuration"]["config_id"] != "peak"


def test_stop_geometry_is_the_existing_zone_stop_concept():
    prices = matrix.initial_prices("BUYER_ABSORPTION", 100.0, 100.25, 99.0, 100.0, stop_buffer_ticks=3, target_r=2.0)
    assert prices["stop"] == 98.25
    assert prices["target"] == pytest.approx(104.25)
    assert len(matrix.generate_weight_grid()) == 3_876


def test_explicit_candidate_universe_is_causal_and_requires_no_implicit_execution():
    universe = REPOSITORY_ROOT / "examples/research_pipeline/cme_l2_candidate_universe.example.yaml"
    specs = multi.load_strategy_manifest(universe)
    assert len(specs) == 51
    assert len({spec.strategy_id for spec in specs}) == 51
    same_session = [spec for spec in specs if spec.session == spec.source_session]
    cross_session = [spec for spec in specs if spec.session != spec.source_session]
    assert len(same_session) == 21
    assert len(cross_session) == 30
    assert all(not spec.level_resolver_required for spec in same_session)
    assert all(spec.level_resolver_required for spec in cross_session)
    assert all(spec.stage1_enabled and spec.stage2_enabled for spec in specs)
    assert all(spec.uses_shared_absorption_engine for spec in specs)
    assert all("BUYER_ABSORPTION_LONG" in spec.long_short_behavior for spec in specs)
    assert not {spec.strategy_id for spec in specs if spec.strategy_id.startswith("EU_PRIOR_ASIA_")}
    completed_europe = {spec.strategy_id: spec for spec in specs if spec.strategy_id.startswith("NY_COMPLETED_EU_")}
    assert set(completed_europe) == {"NY_COMPLETED_EU_POC", "NY_COMPLETED_EU_VAH", "NY_COMPLETED_EU_VAL"}
    assert all("16:30 London" in spec.causal_availability_rule for spec in completed_europe.values())


def test_focused_example_manifest_keeps_mapping_metadata_valid():
    specs = multi.load_strategy_manifest(REPOSITORY_ROOT / "examples/research_pipeline/cme_l2_multi_strategy.example.yaml")
    assert len(specs) == 4
    assert all(spec.baseline_quality == {"profile": "common_stage1_median_w04"} for spec in specs)
