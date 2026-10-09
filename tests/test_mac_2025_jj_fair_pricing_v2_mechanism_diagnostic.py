from research_pipeline.cme_orderflow_absorption_l2_v1 import (
    mac_2025_jj_fair_pricing_v2_mechanism_diagnostic as diagnostic,
)


def test_same_candle_displacement_and_directional_bos_are_feasible_both_sides():
    result = diagnostic._combined_trigger_audit()
    assert result["classification"] == "IMPLEMENTATION_DEFECT"
    for side in ("bull", "bear"):
        case = result["synthetic_sequences"][side]
        assert case["displacement"] is True
        assert case["correct_directional_bos"] is True
        # The called V1 helper uses the opposite close-through convention.
        assert case["v2_called_v1_bos"] is False


def test_displacement_without_two_bar_directional_bos_is_not_combined():
    case = diagnostic._combined_trigger_audit()["synthetic_sequences"]["displacement_without_bos"]
    assert case["displacement"] is True
    assert case["correct_directional_bos"] is False


def test_control_decision_requires_positive_clustered_effect_in_both_periods():
    paired = {}
    for period in ("SPRING_2025", "OCTOBER_2025"):
        paired[f"OPENING_CONTINUATION|{period}"] = {
            "paired_signal_minus_control_net_ticks": {
                "300": {"date_clusters": 9, "mean": 1.0, "ci95": [0.1, 1.9]}
            }
        }
    assert diagnostic._classify(paired, "OPENING_CONTINUATION", positive_cells=False) == (
        "CONTINUATION_SIGNAL_HAS_PREDICTIVE_INFORMATION"
    )
    paired["OPENING_CONTINUATION|OCTOBER_2025"]["paired_signal_minus_control_net_ticks"]["300"]["ci95"] = [-0.1, 1.9]
    assert diagnostic._classify(paired, "OPENING_CONTINUATION", positive_cells=True) == (
        "CONTINUATION_POSITIVE_CELLS_EXPLAINED_BY_SELECTION"
    )
