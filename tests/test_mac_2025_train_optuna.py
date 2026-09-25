import math

import numpy as np

from research_pipeline.cme_orderflow_absorption_l2_v1 import mac_2025_candidate_tape as candidate_tape
from research_pipeline.cme_orderflow_absorption_l2_v1 import mac_2025_train_optuna as stage2a


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
    features = {name: np.asarray([10.0, 2.0], dtype=float) for name in names}
    features["consume_restore_cycles"] = np.asarray([1.0, 0.0])
    features["maximum_through_level_progress_ticks"] = np.asarray([1.0, 5.0])
    features["interaction_rejection_ticks"] = np.asarray([1.0, 0.0])
    features["aggressive_volume_imbalance"] = np.asarray([0.5, -0.5])
    features["restoration_supported_by_execution_ratio"] = np.asarray([0.5, 0.5])
    return stage2a.AllAOpportunityPool(
        features=features, direction_sign=np.asarray([1.0, -1.0]),
        has_trade=np.asarray([True, True]), trade_r=np.asarray([2.0, -1.0]),
        date_index=np.asarray([0, 0], dtype=np.int16), session_index=np.asarray([0, 1], dtype=np.int8),
        family_index=np.asarray([0, 1], dtype=np.int16), family_ids=("A", "B"),
        trade_rows=(
            {"entry_timestamp_ns": 1, "exit_timestamp_ns": 2, "confirmation_index": 0,
             "confirmation_timestamp_ns": 1, "interaction_end_ns": 1, "r_multiple": 2.0,
             "date": "2025-03-03", "family_id": "A"},
            {"entry_timestamp_ns": 3, "exit_timestamp_ns": 4, "confirmation_index": 1,
             "confirmation_timestamp_ns": 3, "interaction_end_ns": 3, "r_multiple": -1.0,
             "date": "2025-03-03", "family_id": "B"},
        ),
    )


def _params() -> dict[str, float]:
    params = {name: 0.2 for name in candidate_tape.WEIGHT_NAMES}
    params.update({
        "min_quality_score": 0.0, "min_relevant_aggressive_volume": 0,
        "min_relevant_execution_count": 0, "min_consume_restore_cycles": 0,
        "max_through_level_progress_ticks": 4.0, "min_rejection_ticks": 0.25,
        "false_refill_penalty_weight": 0.25,
        "unexecuted_add_penalty_component_weight": 0.5,
        "rapid_cancel_penalty_component_weight": 0.3,
        "adverse_progress_penalty_component_weight": 0.2,
        "aggressive_volume_saturation": 250.0, "execution_count_saturation": 8.0,
        "restore_cycle_saturation": 3.0, "restoration_ratio_saturation": 1.0,
        "rejection_saturation_ticks": 3.0, "persistence_depth_saturation": 100.0,
        "restoration_latency_saturation_ms": 1000.0, "multi_level_ofi_saturation": 100.0,
    })
    return params


def test_search_space_covers_all_class_a_parameters() -> None:
    space = stage2a._search_space()
    assert len(stage2a.ALL_A_NAMES) == 23
    assert {f"raw_{name}" for name in candidate_tape.WEIGHT_NAMES} == set(space["weights"])
    assert set(stage2a.THRESHOLD_NAMES) == set(space["thresholds"])
    assert set(stage2a.PENALTY_NAMES) == set(space["penalty_weights"])
    assert set(stage2a.SATURATION_NAMES) == set(space["saturations"])


def test_class_a_quality_matches_scalar_candidate_semantics() -> None:
    pool = _pool()
    params = _params()
    vector = stage2a._quality(pool, params)
    row = {name: float(values[0]) for name, values in pool.features.items()}
    row["direction"] = "BUYER_ABSORPTION"
    scalar = candidate_tape._component_scores(row, {**params, "weights": params})
    scalar_quality = candidate_tape._quality(row, params, {**params, "weights": params})
    expected = (
        scalar["aggression_score"] * params["aggression_weight"]
        + scalar["restoration_score"] * params["restoration_weight"]
        + scalar["price_resistance_score"] * params["price_resistance_weight"]
        + scalar["persistence_score"] * params["persistence_weight"]
        + scalar["multi_level_support_score"] * params["multi_level_support_weight"]
        - scalar["false_refill_penalty"] * params["false_refill_penalty_weight"]
    )
    assert math.isclose(float(vector[0]), scalar_quality, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(float(vector[0]), expected, rel_tol=0.0, abs_tol=1e-12)


def test_class_a_thresholds_are_post_hoc_and_deterministic() -> None:
    pool = _pool()
    params = _params()
    first = stage2a.qualified_mask(pool, params)
    second = stage2a.qualified_mask(pool, params)
    assert np.array_equal(first, second)
    assert first.shape == (2,)
    params["min_relevant_aggressive_volume"] = 11
    assert not stage2a.qualified_mask(pool, params)[0]


def test_pool_cache_round_trip(tmp_path) -> None:
    pool = _pool()
    stage2a.save_all_a_pool(pool, output_root=tmp_path, train_dates=("2025-03-03",))
    restored = stage2a.load_all_a_pool(output_root=tmp_path, train_dates=("2025-03-03",))
    assert restored is not None
    assert restored.count == pool.count
    assert restored.family_ids == pool.family_ids
    assert np.array_equal(restored.has_trade, pool.has_trade)
    assert restored.trade_rows == pool.trade_rows


def test_fast_metrics_preserve_zero_trade_validity_and_finite_objective_inputs() -> None:
    pool = _pool()
    metrics = stage2a.fast_metrics(pool, np.asarray([False, False]), 1)
    assert metrics["total_trades"] == 0
    assert metrics["profit_factor"] == 0
    assert all(math.isfinite(float(metrics[name])) for name in (
        "net_r", "max_drawdown_r", "active_dates", "profitable_date_ratio",
        "median_date_r", "lower_quartile_date_r", "downside_tail",
        "median_session_date_r", "date_concentration", "family_concentration", "active_sessions",
    ))
