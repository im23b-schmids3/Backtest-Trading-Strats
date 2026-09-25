import math

import numpy as np
import pytest

from research_pipeline.cme_orderflow_absorption_l2_v1 import mac_2025_train_optimization as optimization


def _manifest(**overrides):
    payload = {
        "status": "COMPLETE",
        "train_dates": list(optimization.EXPECTED_TRAIN_DATES),
        "failed_dates": [],
        "validation_performance": False,
        "final_oos_accessed": False,
        "optimization_run": False,
        "reports": [{"date": day} for day in optimization.EXPECTED_TRAIN_DATES],
    }
    payload.update(overrides)
    return payload


def test_exact_35_date_allowlist_is_fail_closed() -> None:
    optimization.validate_train_manifest_metadata(_manifest())
    with pytest.raises(optimization.TrainOptimizationError):
        optimization.validate_train_manifest_metadata(_manifest(train_dates=list(optimization.EXPECTED_TRAIN_DATES[:-1])))


def test_validation_and_oos_flags_are_rejected() -> None:
    with pytest.raises(optimization.TrainOptimizationError):
        optimization.validate_train_manifest_metadata(_manifest(validation_performance=True))
    with pytest.raises(optimization.TrainOptimizationError):
        optimization.validate_train_manifest_metadata(_manifest(final_oos_accessed=True))


def test_duplicate_manifest_date_is_rejected() -> None:
    reports = [{"date": day} for day in optimization.EXPECTED_TRAIN_DATES]
    reports[-1] = {"date": optimization.EXPECTED_TRAIN_DATES[-2]}
    with pytest.raises(optimization.TrainOptimizationError):
        optimization.validate_train_manifest_metadata(_manifest(reports=reports))


def test_zero_trade_remains_valid_and_zero_loss_is_bounded() -> None:
    metrics = {
        "total_trades": 0, "active_dates": 0, "profitable_date_ratio": 0.0,
        "max_drawdown_r": 0.0, "date_concentration": 0.0, "family_concentration": 0.0,
    }
    assert optimization._technically_valid(metrics, weights=[0.2] * 5, min_q=0.55)
    raw, bounded = optimization._profit_factor([1.0, 2.0])
    assert raw is None
    assert bounded == optimization.PROFIT_FACTOR_CAP


def test_no_performance_rules_remain_in_constraints() -> None:
    rules = optimization.derive_constraints()["rules"]
    assert optimization.derive_constraints()["performance_constraints_removed"] is True
    assert not {"min_total_trades", "min_active_dates", "min_profitable_date_ratio",
                 "max_drawdown_r", "max_date_concentration_top5",
                 "max_family_concentration_top3"} & set(rules)


def test_stage1a_population_is_deterministic_and_weights_normalize() -> None:
    first = optimization._sample_population(128, seed=1234)
    second = optimization._sample_population(128, seed=1234)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert np.allclose(first[0].sum(axis=1), 1.0)
    assert np.all(first[0] > 0)


def test_objective_scaling_and_score_are_finite_and_deterministic() -> None:
    metrics = {name: np.asarray([1.0, 2.0, 3.0]) for name in optimization.OBJECTIVE_DEFINITION}
    metrics["active_sessions"] = np.asarray([1.0, 2.0, 3.0])
    metrics["total_trades"] = np.asarray([1.0, 2.0, 3.0])
    stage = {"feasible": np.asarray([True, True, True]), "metrics": metrics}
    scaling_a = optimization.build_objective_scaling(stage)
    scaling_b = optimization.build_objective_scaling(stage)
    assert scaling_a == scaling_b
    score = optimization.robust_objective({name: 2.0 for name in metrics}, scaling_a)
    assert all(math.isfinite(value) for value in score.values())
    assert math.isfinite(score["robust_score"])


def test_optimization_module_never_opens_dbn() -> None:
    source = open(optimization.__file__, encoding="utf-8").read()
    assert "dbn" not in source.lower() or "DBN" in source
