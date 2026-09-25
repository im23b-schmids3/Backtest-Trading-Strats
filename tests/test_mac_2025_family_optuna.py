import json

import numpy as np

from research_pipeline.cme_orderflow_absorption_l2_v1 import mac_2025_family_optuna as family
from research_pipeline.cme_orderflow_absorption_l2_v1 import mac_2025_train_optuna as stage2a
from research_pipeline.cme_orderflow_absorption_l2_v1 import mac_2025_train_optimization as preparation


def _pool() -> stage2a.AllAOpportunityPool:
    names = (
        "directional_aggressive_volume", "relevant_execution_count", "consume_restore_cycles",
        "maximum_through_level_progress_ticks", "interaction_rejection_ticks",
        "aggressive_volume_imbalance", "executed_to_initial_displayed_ratio",
        "restoration_to_consumption_ratio", "restoration_supported_by_execution_ratio",
        "mean_restoration_latency_ms", "defended_price_present_fraction",
        "defended_depth_time_weighted_mean", "depth_imbalance_5", "multi_level_ofi",
        "unexecuted_add_volume", "rapid_cancel_ratio",
    )
    features = {name: np.asarray([10.0, 10.0]) for name in names}
    features["consume_restore_cycles"] = np.asarray([1.0, 1.0])
    features["maximum_through_level_progress_ticks"] = np.asarray([1.0, 1.0])
    features["interaction_rejection_ticks"] = np.asarray([1.0, 1.0])
    features["aggressive_volume_imbalance"] = np.asarray([0.5, 0.5])
    features["restoration_supported_by_execution_ratio"] = np.asarray([0.5, 0.5])
    return stage2a.AllAOpportunityPool(
        features=features, direction_sign=np.asarray([1.0, 1.0]),
        has_trade=np.asarray([True, True]), trade_r=np.asarray([2.0, -1.0]),
        date_index=np.asarray([0, 0], dtype=np.int16), session_index=np.asarray([0, 1], dtype=np.int8),
        family_index=np.asarray([0, 1], dtype=np.int16), family_ids=("A", "B"),
        trade_rows=(
            {"entry_timestamp_ns": 1, "exit_timestamp_ns": 2, "confirmation_index": 0,
             "confirmation_timestamp_ns": 1, "interaction_end_ns": 1, "r_multiple": 2.0,
             "date": "2025-03-03", "family_id": "A", "candidate_order": 0},
            {"entry_timestamp_ns": 3, "exit_timestamp_ns": 4, "confirmation_index": 1,
             "confirmation_timestamp_ns": 3, "interaction_end_ns": 3, "r_multiple": -1.0,
             "date": "2025-03-03", "family_id": "B", "candidate_order": 1},
        ),
    )


def test_family_fast_metrics_isolated() -> None:
    pool = _pool()
    params = {name: 0.2 for name in stage2a.candidate_tape.WEIGHT_NAMES}
    params.update({
        "min_quality_score": 0.0, "min_relevant_aggressive_volume": 0,
        "min_relevant_execution_count": 0, "min_consume_restore_cycles": 0,
        "max_through_level_progress_ticks": 4.0, "min_rejection_ticks": 0.25,
        "false_refill_penalty_weight": 0.0, "unexecuted_add_penalty_component_weight": 0.0,
        "rapid_cancel_penalty_component_weight": 0.0, "adverse_progress_penalty_component_weight": 0.0,
        "aggressive_volume_saturation": 250.0, "execution_count_saturation": 8.0,
        "restore_cycle_saturation": 3.0, "restoration_ratio_saturation": 1.0,
        "rejection_saturation_ticks": 3.0, "persistence_depth_saturation": 100.0,
        "restoration_latency_saturation_ms": 1000.0, "multi_level_ofi_saturation": 100.0,
    })
    qualified = stage2a.qualified_mask(pool, params)
    assert family._family_metrics(pool, qualified, "A", 1)["net_r"] == 2.0
    assert family._family_metrics(pool, qualified, "B", 1)["net_r"] == -1.0


def test_exact_family_result_excludes_other_family() -> None:
    pool = _pool()
    params = {name: 0.2 for name in stage2a.candidate_tape.WEIGHT_NAMES}
    params.update({
        "min_quality_score": 0.0, "min_relevant_aggressive_volume": 0,
        "min_relevant_execution_count": 0, "min_consume_restore_cycles": 0,
        "max_through_level_progress_ticks": 4.0, "min_rejection_ticks": 0.25,
        "false_refill_penalty_weight": 0.0, "unexecuted_add_penalty_component_weight": 0.0,
        "rapid_cancel_penalty_component_weight": 0.0, "adverse_progress_penalty_component_weight": 0.0,
        "aggressive_volume_saturation": 250.0, "execution_count_saturation": 8.0,
        "restore_cycle_saturation": 3.0, "restoration_ratio_saturation": 1.0,
        "rejection_saturation_ticks": 3.0, "persistence_depth_saturation": 100.0,
        "restoration_latency_saturation_ms": 1000.0, "multi_level_ofi_saturation": 100.0,
    })
    result = family._exact_family_result(pool, params, "A", ("2025-03-03",))
    assert result["net_r"] == 2.0
    assert result["trades"] == 1.0


def test_selected_families_are_exact_and_reported(tmp_path) -> None:
    report = {
        "families": {name: {} for name in family.SELECTED_FAMILIES},
        "consistently_strong": [family.SELECTED_FAMILIES[0], family.SELECTED_FAMILIES[2]],
        "parameter_sensitive": [family.SELECTED_FAMILIES[1]],
    }
    path = tmp_path / "family-report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    family._validate_selected_families(path)


def test_family_output_slug_is_deterministic() -> None:
    assert family._slug("NY|EUROPE|CURRENT|POC") == "ny-europe-current-poc"
