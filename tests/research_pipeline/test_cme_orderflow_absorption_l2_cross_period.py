from __future__ import annotations

import inspect
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from research_pipeline.cme_orderflow_absorption_l2_v1 import cross_period_robust_stress as cross
from research_pipeline.cme_orderflow_absorption_l2_v1 import v3_poc_april_retro_replay as april
from research_pipeline.cme_orderflow_absorption_l2_v1 import v3_poc_fresh_august_replay as fresh
from research_pipeline.cme_orderflow_absorption_l2_v1.model import StructuralLevel
from research_pipeline.cme_orderflow_absorption_l2_v1.v2_quality050 import V2_CONFIG


def _metrics(**updates: object) -> dict[str, object]:
    values: dict[str, object] = {
        "sessions": 1,
        "completed_interactions": 2,
        "accepted_setups": 1,
        "confirmations": 1,
        "confirmation_expiries": 0,
        "active_position_blocks": 0,
        "completed_trades": 1,
        "wins": 1,
        "losses": 0,
        "win_rate": 1.0,
        "total_r": 2.5,
        "average_r": 2.5,
        "median_r": 2.5,
        "net_pnl_usd": 200.0,
        "profit_factor": None,
        "max_cumulative_drawdown_r": 0.0,
        "es_trades": 1,
        "mes_trades": 0,
        "target_exits": 1,
        "stop_exits": 0,
        "hard_cutoff_exits": 0,
        "unresolved": 0,
    }
    values.update(updates)
    return values


def test_cross_period_frozen_hashes_weights_and_quality_are_exact():
    cross._assert_frozen_contracts()
    assert cross.V3_HASH == "a0ce94eeb78dcbf865cf4464bdf97ebc3f014a8ec5e2f559f99798534dfcbcb4"
    assert cross.CANDIDATE_HASH == "5d3d72cd378d0ac986670a3c48ee14344571e1f559746e3fa1f514429e80553a"
    assert cross.CANDIDATE_ID == "W02-02-07-02-07-Q40"
    assert (
        cross.V3_CONFIG.aggression_weight,
        cross.V3_CONFIG.restoration_weight,
        cross.V3_CONFIG.price_resistance_weight,
        cross.V3_CONFIG.persistence_weight,
        cross.V3_CONFIG.multi_level_support_weight,
        cross.V3_CONFIG.min_quality_score,
    ) == (0.28, 0.25, 0.22, 0.12, 0.13, 0.50)
    assert (
        cross.CANDIDATE_CONFIG.aggression_weight,
        cross.CANDIDATE_CONFIG.restoration_weight,
        cross.CANDIDATE_CONFIG.price_resistance_weight,
        cross.CANDIDATE_CONFIG.persistence_weight,
        cross.CANDIDATE_CONFIG.multi_level_support_weight,
        cross.CANDIDATE_CONFIG.min_quality_score,
    ) == (0.10, 0.10, 0.35, 0.10, 0.35, 0.40)
    changed = {
        key for key, value in asdict(cross.V3_CONFIG).items()
        if asdict(cross.CANDIDATE_CONFIG)[key] != value
    }
    assert changed == {
        "aggression_weight", "restoration_weight", "price_resistance_weight",
        "persistence_weight", "multi_level_support_weight", "min_quality_score", "weights_label",
    }


def test_cross_period_portfolios_have_independent_chronological_state():
    runners = cross._new_portfolio_runners(
        "2026-04-06", "SYNTHETIC_TEST", [StructuralLevel("PRIOR_RTH_POC", 5000.00)]
    )
    baseline, candidate = runners["V3"], runners[cross.CANDIDATE_ID]
    assert baseline is not candidate
    assert baseline.interactions is not candidate.interactions
    assert baseline.signals is not candidate.signals
    assert baseline.trade_ledger is not candidate.trade_ledger
    baseline.diagnostic_events.append({"event": "ONLY_BASELINE"})
    assert candidate.diagnostic_events == []
    assert baseline.config == cross.V3_CONFIG
    assert candidate.config == cross.CANDIDATE_CONFIG


def test_cross_period_poc_only_selection_accepts_one_poc_and_rejects_invalid_profiles():
    poc = StructuralLevel("PRIOR_RTH_POC", 5000.00)
    high = StructuralLevel("PRIOR_RTH_HIGH", 5010.00)
    assert cross._select_poc([high, poc]) == (poc,)
    with pytest.raises(cross.CrossPeriodError, match="EXACTLY_ONE_POC"):
        cross._select_poc([high])
    with pytest.raises(cross.CrossPeriodError, match="EXACTLY_ONE_POC"):
        cross._select_poc([poc, poc])


def test_cross_period_declares_native_and_mbo_sources_without_normalizing_them():
    periods = {period.period_id: period for period in cross.PERIODS}
    assert periods["APRIL_2026"].source_model == "NATIVE_DATABENTO_MBP10"
    assert periods["AUGUST_10_14_2026"].source_model == "NATIVE_DATABENTO_MBP10"
    assert periods["MAY_2026"].source_model == "MBO_DERIVED_SYNTHETIC_MBP10"
    assert periods["RETRO_JUNE_JULY_2026"].source_model == "MBO_DERIVED_SYNTHETIC_MBP10"
    assert periods["AUGUST_03_06_2026"].source_model == "MBO_DERIVED_SYNTHETIC_MBP10_SHARED_FILE"
    assert periods["AUGUST_03_06_2026"].dates == (
        "2026-08-03", "2026-08-04", "2026-08-05", "2026-08-06",
    )
    assert "2026-08-07" not in periods["AUGUST_03_06_2026"].dates


def test_cross_period_reuses_only_exact_v3_and_never_an_old_v2_poc_subset():
    periods = {period.period_id: period for period in cross.PERIODS}
    assert periods["APRIL_2026"].exact_v3_artifact == april.OUTPUT_ROOT
    assert periods["AUGUST_10_14_2026"].exact_v3_artifact == fresh.OUTPUT_ROOT
    for period_id in ("MAY_2026", "RETRO_JUNE_JULY_2026", "AUGUST_03_06_2026"):
        period = periods[period_id]
        assert period.exact_v3_artifact is None
        assert period.existing_non_v3_artifact is not None
        assert period.replay_mode.startswith("PAIRED_")


def test_cross_period_inventory_marks_old_v2_subset_non_exact(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    old = Path("research_runs/old-v2")
    exact = Path("research_runs/exact-v3")
    (tmp_path / old).mkdir(parents=True)
    (tmp_path / old / "summary.json").write_text("{}", encoding="utf-8")
    (tmp_path / exact).mkdir(parents=True)
    (tmp_path / exact / "summary.json").write_text(json.dumps({
        "strategy_id": "CMEOrderflowAbsorption.ES_L2_V3_POC_ONLY",
        "v3_contract_sha256": cross.V3_HASH,
    }), encoding="utf-8")
    native = replace(
        cross.PERIODS[0], dates=("2026-04-06",), source_root=Path("native"),
        exact_v3_artifact=exact, existing_non_v3_artifact=None,
    )
    mbo = replace(
        cross.PERIODS[1], dates=("2026-05-04",), source_root=Path("mbo"),
        exact_v3_artifact=None, existing_non_v3_artifact=old,
    )
    monkeypatch.setattr(cross, "PERIODS", (native, mbo))
    monkeypatch.setattr(cross, "_existing_files_for_period", lambda *_: [])
    inventory = cross.build_source_inventory(tmp_path)
    rows = {row["period_id"]: row for row in inventory["periods"]}
    assert rows["APRIL_2026"]["exact_v3_artifact_reusable"] is True
    assert rows["MAY_2026"]["exact_v3_artifact_reusable"] is False
    assert rows["MAY_2026"]["older_v2_poc_subset_is_exact_v3"] is False


def test_cross_period_missing_local_source_fails_inventory_without_download(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    period = replace(
        cross.PERIODS[0], dates=("2026-04-06",), source_root=Path("missing-source"),
        exact_v3_artifact=None,
    )
    monkeypatch.setattr(cross, "PERIODS", (period,))
    monkeypatch.setattr(cross, "_existing_files_for_period", lambda *_: [tmp_path / "absent.dbn"])
    inventory = cross.build_source_inventory(tmp_path)
    row = inventory["periods"][0]
    assert row["status"] == "INSUFFICIENT_EXISTING_LOCAL_DATA"
    assert row["candidate_evaluable_without_download"] is False
    assert inventory["downloads_permitted"] is False
    assert inventory["databento_calls"] == 0


def test_cross_period_evidence_labels_and_aggregate_are_never_all_oos():
    labels = {period.period_id: period.evidence_label for period in cross.PERIODS}
    assert labels["APRIL_2026"] == "RETROSPECTIVE_CROSS_PERIOD_STRESS_TEST"
    assert labels["MAY_2026"] == "DEVELOPMENT_SEEN_DATA_CROSS_PERIOD_STRESS_TEST"
    assert labels["AUGUST_03_06_2026"] == "SEEN_AUGUST_CROSS_PERIOD_STRESS_TEST"
    assert labels["AUGUST_10_14_2026"] == "PREVIOUSLY_SEEN_FOR_CANDIDATE_CROSS_PERIOD_STRESS_TEST"
    assert cross.NO_AGGREGATE_OOS_LABEL == "MIXED_EVIDENCE_CROSS_PERIOD_DESCRIPTIVE_NOT_OOS"


def test_cross_period_overlap_counts_common_unique_and_chronology_rows():
    rows = [
        {"record_type": "TRADE", "common_trade": True, "v3_trade_id": "v1", "candidate_trade_id": "c1",
         "one_position_chronology_difference": "NONE"},
        {"record_type": "TRADE", "common_trade": False, "v3_trade_id": "", "candidate_trade_id": "c2",
         "one_position_chronology_difference": "CANDIDATE_TRADE_UNBLOCKED_RELATIVE_TO_V3"},
        {"record_type": "TRADE", "common_trade": False, "v3_trade_id": "v3", "candidate_trade_id": "",
         "one_position_chronology_difference": "V3_TRADE_BLOCKED_IN_CANDIDATE"},
    ]
    assert cross._overlap_summary(rows) == {
        "common_trades": 1,
        "candidate_only_trades": 1,
        "v3_only_trades": 1,
        "one_position_chronology_differences": 2,
    }


def test_cross_period_period_checkpoint_is_atomic_reusable_and_contract_bound(tmp_path: Path):
    period = cross.PERIODS[0]
    root = tmp_path / "period"
    v3_result = {"metrics": _metrics(), "provenance": "TEST_V3", "trades": [], "setup_rows": []}
    candidate = {"metrics": _metrics(total_r=1.5), "provenance": "TEST_CANDIDATE", "trades": [], "setup_rows": []}
    overlap: list[dict[str, object]] = []
    cross._checkpoint_period(root, period, v3_result, candidate, overlap)
    loaded_v3, loaded_candidate, loaded_overlap = cross._load_period_checkpoint(root, period)
    assert loaded_v3["metrics"] == v3_result["metrics"]
    assert loaded_candidate["metrics"] == candidate["metrics"]
    assert loaded_overlap == []
    assert not root.with_name(root.name + ".building").exists()
    with pytest.raises(FileExistsError, match="checkpoint collision"):
        cross._checkpoint_period(root, period, v3_result, candidate, overlap)


def test_cross_period_native_session_helpers_keep_original_defaults_and_allow_frozen_candidate():
    for function in (april._run_session, fresh._run_session):
        signature = inspect.signature(function)
        assert signature.parameters["config"].default == V2_CONFIG
        assert signature.parameters["strategy_id"].default == "CMEOrderflowAbsorption.ES_L2_V3_POC_ONLY"
        assert signature.parameters["evidence_label"].kind is inspect.Parameter.KEYWORD_ONLY


def test_cross_period_source_has_no_search_download_or_network_execution_path():
    source = Path(cross.__file__).read_text(encoding="utf-8")
    assert "weight_q_research" not in source
    assert "get_range(" not in source
    assert "metadata.get_cost" not in source
    assert "timeseries" not in source
    assert "candidate_reselected\": False" in source
    assert "downloads\": 0" in source


def test_cross_period_resume_rejects_tampered_checkpoint_contract(tmp_path: Path):
    period = cross.PERIODS[0]
    root = tmp_path / "period"
    root.mkdir()
    (root / "checkpoint.json").write_text(json.dumps({"status": "PERIOD_CHECKPOINT_COMPLETE"}), encoding="utf-8")
    with pytest.raises(cross.CrossPeriodError, match="CHECKPOINT_CONTRACT_MISMATCH"):
        cross._load_period_checkpoint(root, period)


def test_cross_period_mocked_orchestration_writes_complete_non_oos_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    repository = tmp_path / "repo"
    repository.mkdir()
    output = repository / cross.OUTPUT_RELATIVE
    inventory = {
        "strategy_id": cross.STRATEGY_ID,
        "periods": [
            {"period_id": period.period_id, "status": "READY_FROM_EXISTING_LOCAL_DATA"}
            for period in cross.PERIODS
        ],
        "downloads_permitted": False,
        "databento_calls": 0,
    }
    monkeypatch.setattr(cross, "build_source_inventory", lambda _: inventory)

    calls: list[str] = []

    def fake_period(_repository: Path, period: cross.Period):
        calls.append(period.period_id)
        baseline = {"metrics": _metrics(), "trades": [], "setup_rows": [], "provenance": "MOCK_V3"}
        candidate = {
            "metrics": _metrics(total_r=3.0, average_r=3.0, net_pnl_usd=250.0),
            "trades": [], "setup_rows": [], "provenance": "MOCK_CANDIDATE",
        }
        return baseline, candidate, []

    monkeypatch.setattr(cross, "_run_period", fake_period)
    result = cross.run_all(repository_root=repository, output_root=output)
    assert result["status"] == "CROSS_PERIOD_STRESS_TEST_COMPLETE"
    assert calls == [period.period_id for period in cross.PERIODS]
    assert {path.name for path in output.iterdir()} >= {
        "source-inventory.json", "period-results.csv", "v3-vs-candidate.csv",
        "aggregate-comparison.json", "trade-overlap.csv", "summary.json", "diagnostic-report.md", "periods",
    }
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["aggregate_is_oos"] is False
    assert summary["candidate_reselected"] is False
    assert summary["candidate_promoted"] is False
    assert summary["aggregate_evidence_label"] == cross.NO_AGGREGATE_OOS_LABEL
    assert len(summary["periods"]) == 7
