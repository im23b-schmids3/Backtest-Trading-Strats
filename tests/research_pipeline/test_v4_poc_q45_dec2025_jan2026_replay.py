from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from research_pipeline.cme_orderflow_absorption_l2_v1 import historical_runner as historical
from research_pipeline.cme_orderflow_absorption_l2_v1 import v3_poc_dec2025_jan2026_replay as v3_replay
from research_pipeline.cme_orderflow_absorption_l2_v1 import v3_poc_only as v3
from research_pipeline.cme_orderflow_absorption_l2_v1 import v4_poc_q45 as v4
from research_pipeline.cme_orderflow_absorption_l2_v1 import v4_poc_q45_dec2025_jan2026_replay as replay
from research_pipeline.cme_orderflow_absorption_l2_v1.v2_quality050 import V2_CONFIG


def _feature_row(identifier: str, score: float, *, day: str = "2025-12-01") -> dict[str, object]:
    return {
        "interaction_id": identifier,
        "date": day,
        "level": "PRIOR_RTH_POC",
        "direction": "BUYER_ABSORPTION",
        "l2_absorption_quality_score": score,
        "aggression_score": score,
        "restoration_score": score,
        "price_resistance_score": score,
        "persistence_score": score,
        "multi_level_support_score": score,
        "false_refill_penalty": 0.0,
    }


def _runner(rows: list[dict[str, object]]) -> SimpleNamespace:
    setups = [
        {
            **row,
            "setup_id": f"S:{row['interaction_id']}",
            "accepted": float(row["l2_absorption_quality_score"]) >= 0.45,
            "confirmation_timestamp_ns": None,
            "terminal_reason": "CONFIRMATION_WINDOW_EXPIRED",
        }
        for row in rows
    ]
    return SimpleNamespace(interaction_ledger=rows, setup_ledger=setups, trade_ledger=[])


def test_v4_identity_parent_hash_and_exact_one_field_diff():
    assert v4.STRATEGY_ID == "CMEOrderflowAbsorption.ES_L2_V4_POC_ONLY_Q45"
    assert v4.STRATEGY_ID != v3.STRATEGY_ID
    assert v3.v3_contract_sha256() == v4.PARENT_CONTRACT_SHA256
    assert v4.PARENT_CONTRACT_SHA256 == "a0ce94eeb78dcbf865cf4464bdf97ebc3f014a8ec5e2f559f99798534dfcbcb4"
    assert v4.v3_to_v4_contract_diff()["changed_strategy_fields"] == [
        {"field": "min_quality_score", "v3": 0.50, "v4": 0.45}
    ]


def test_all_configuration_fields_except_quality_are_identical():
    parent, child = asdict(V2_CONFIG), asdict(v4.V4_CONFIG)
    assert parent["min_quality_score"] == 0.50
    assert child["min_quality_score"] == 0.45
    assert {key: value for key, value in parent.items() if key != "min_quality_score"} == {
        key: value for key, value in child.items() if key != "min_quality_score"
    }
    assert parent["weights_label"] == child["weights_label"]


def test_poc_confirmation_stop_target_risk_sizing_and_mes_are_inherited():
    parent, child = v3.v3_contract(), v4.v4_contract()
    assert child["eligible_structural_levels"] == ["PRIOR_RTH_POC"] == parent["eligible_structural_levels"]
    assert child["execution"] == parent["execution"]
    assert child["execution"] == {
        "confirmation_window_seconds_inclusive": [5.0, 15.0],
        "confirmation_favorable_ticks": 3,
        "entry_latency_ms": 2.0,
        "stop_buffer_ticks": 5,
        "target_r": 3.0,
        "risk_budget_usd": 250.0,
        "es_first": True,
        "mes_fallback": True,
        "max_es_contracts": 6,
        "max_mes_contracts": 60,
    }


def test_calendar_and_parent_replay_defaults_remain_frozen():
    assert v3_replay.V2_CONFIG.min_quality_score == 0.50
    assert v3_replay._effective_hard_flat.__module__ == v3_replay.__name__
    assert v3_replay._effective_hard_flat.__name__ == "_effective_hard_flat"
    replay._validate_v4_contract()


def test_pre_quality_rows_below_q45_are_retained_and_scores_match():
    runner = _runner([_feature_row("below", 0.44), _feature_row("band", 0.47), _feature_row("parent", 0.51)])
    replay.decorate_pre_quality_interactions(runner)
    assert [row["interaction_id"] for row in runner.interaction_ledger] == ["below", "band", "parent"]
    below, band, parent = runner.interaction_ledger
    assert below["quality_pass_0_45"] is False and below["quality_pass_0_50"] is False
    assert band["incremental_q45_q50_band"] is True
    assert parent["quality_pass_0_50"] is True
    assert all(row["v3_quality_score"] == row["v4_recalculated_quality_score"] for row in runner.interaction_ledger)


def test_interaction_feature_csv_keeps_below_gate_rows(tmp_path: Path):
    runner = _runner([_feature_row("below", 0.40), _feature_row("band", 0.49)])
    replay.decorate_pre_quality_interactions(runner)
    path = tmp_path / "interaction-features.csv"
    historical._rows_write(path, runner.interaction_ledger, ["interaction_id"])
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["interaction_id"] for row in rows] == ["below", "band"]
    assert rows[0]["quality_pass_0_45"] == "False"
    assert rows[1]["incremental_q45_q50_band"] == "True"


def test_incremental_band_metrics_are_reported():
    runner = _runner([_feature_row("below", 0.44), _feature_row("band", 0.47), _feature_row("parent", 0.55)])
    replay.decorate_pre_quality_interactions(runner)
    runner.refresh_setup_ledger = lambda: None
    metrics = replay.build_metrics([runner])
    assert metrics["completed_interactions"] == 3
    assert metrics["incremental_quality_band"]["range"] == "0.45 <= quality_score < 0.50"
    assert metrics["incremental_quality_band"]["interactions"] == 1
    assert metrics["incremental_quality_band"]["accepted_setups"] == 1


def test_v3_v4_comparison_distinguishes_common_incremental_and_chronology():
    metrics = {
        **replay._performance([]),
        "breakdowns": {"month": {}},
        "incremental_quality_band": {},
    }
    trades = [
        {"date": "2025-12-01", "interaction_id": "common", "incremental_q45_q50_band": False},
        {"date": "2025-12-01", "interaction_id": "new-band", "incremental_q45_q50_band": True},
        {"date": "2025-12-01", "interaction_id": "shifted", "incremental_q45_q50_band": False},
    ]
    result = replay.build_comparison(metrics, trades, {
        "2025-12-01|common", "2025-12-01|removed",
    })
    attribution = result["trade_population_attribution"]
    assert attribution["A_common_to_v3_and_v4"]["trade_keys"] == ["2025-12-01|common"]
    assert attribution["B_unique_to_v4_q45_q50_acceptance"]["trade_keys"] == ["2025-12-01|new-band"]
    assert attribution["C_independent_position_chronology"]["v4_parent_gate_trades_not_in_v3"] == ["2025-12-01|shifted"]
    assert attribution["C_independent_position_chronology"]["v3_trades_absent_from_v4"] == ["2025-12-01|removed"]


def test_v4_sessions_receive_distinct_independent_state():
    calls: list[dict[str, object]] = []

    def factory(day, preflight, data_root, **kwargs):
        calls.append({"day": day, **kwargs})
        return _runner([_feature_row(day, 0.46, day=day)])

    runners = replay._execute_sessions(
        ["2025-12-01", "2025-12-02"], {}, Path("unused"), session_factory=factory
    )
    assert runners[0] is not runners[1]
    assert all(call["config"] == v4.V4_CONFIG for call in calls)
    assert all(call["strategy_id"] == v4.STRATEGY_ID for call in calls)
    assert all(call["evidence_label"] == v4.EVIDENCE_LABEL for call in calls)


def test_contract_artifact_is_immutable(tmp_path: Path):
    path = tmp_path / "v4-contract.json"
    payload = v4.write_contract_artifact(path)
    assert payload["contract_sha256"] == v4.v4_contract_sha256()
    with pytest.raises(FileExistsError):
        v4.write_contract_artifact(path)


def test_existing_output_fails_before_data_or_outcome_access(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    output = tmp_path / "existing"
    output.mkdir()
    monkeypatch.setattr(replay.parent, "verify_data_preflight", lambda *_a, **_k: pytest.fail("data read"))
    monkeypatch.setattr(replay, "_read_v3_trade_keys", lambda *_a, **_k: pytest.fail("outcome read"))
    with pytest.raises(FileExistsError):
        replay.run(output_root=output)


def test_replay_preparation_has_no_network_download_or_threshold_search():
    source = Path(replay.__file__).read_text(encoding="utf-8")
    contract_source = Path(v4.__file__).read_text(encoding="utf-8")
    forbidden = ("databento", "get_range(", "metadata.get_cost", "timeseries", "download")
    assert all(term not in source.lower() for term in forbidden)
    assert all(term not in contract_source.lower() for term in forbidden)
    assert "0.40" not in contract_source and "0.46" not in contract_source
    assert "if __name__ == \"__main__\"" in source
