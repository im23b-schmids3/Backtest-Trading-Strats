import json
import math
from pathlib import Path


ROOT = Path(__file__).parents[2]
CONTRACT = ROOT / "docs/research_pipeline/cme_orderflow_absorption_v1/backtest-contract.json"


def load_contract():
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def plus_only(contract, interaction):
    # Pure contract fixture: deliberately does not accept response fields.
    allowed = {"interaction_id", "interaction_end", "level", "absorption_score", "replenishment_score", "direction"}
    assert set(interaction) <= allowed
    return (interaction["level"] in contract["frozen_selection"]["mandatory_structural_levels"]
            and interaction["absorption_score"] >= contract["frozen_selection"]["absorption_p95"]
            and interaction["replenishment_score"] >= contract["frozen_selection"]["replenishment_p95"])


def synthetic_cutoff_result(observations, position_open=True):
    """Pure contract fixture; it neither loads data nor executes a backtest."""
    cutoff = 22 * 3600 + 45 * 60
    valid = [item for item in observations if cutoff - 1 <= item["timestamp"] <= cutoff and item["valid"]]
    result = {"pending_orders": 0, "position_open": False, "exit_timestamp": None, "outcome": None}
    if position_open and valid:
        result.update(exit_timestamp=valid[-1]["timestamp"], outcome="CUTOFF_FORCED_FLAT")
    elif position_open:
        result["outcome"] = "CUTOFF_EXECUTION_INTEGRITY_FAILURE"
    return result


def synthetic_contracts(budget, entry_fill, stop_exit_fill_assumption):
    one_contract_price_risk_usd = abs(entry_fill - stop_exit_fill_assumption) * 50.00
    round_trip_commission_risk_usd = 2 * 3.00
    one_contract_initial_risk_usd = one_contract_price_risk_usd + round_trip_commission_risk_usd
    return one_contract_price_risk_usd, round_trip_commission_risk_usd, one_contract_initial_risk_usd, math.floor(budget / one_contract_initial_risk_usd)


def test_frozen_thresholds_load_verbatim_and_validation_cannot_recompute():
    c = load_contract()
    assert c["frozen_selection"]["absorption_p95"] == 0.7977986403366786
    assert c["frozen_selection"]["replenishment_p95"] == 0.7785691162188411
    assert "must not compute a distribution, percentile, p95" in c["frozen_selection"]["loading_rule"]


def test_plus_only_causality_and_no_response_field_fixture():
    c = load_contract()
    assert plus_only(c, {"interaction_id": "x", "interaction_end": 10, "level": "PRIOR_RTH_POC", "absorption_score": 0.8, "replenishment_score": 0.8, "direction": "BUYER_ABSORPTION"})
    assert not plus_only(c, {"interaction_id": "y", "interaction_end": 10, "level": "PRIOR_RTH_POC", "absorption_score": 0.8, "replenishment_score": 0.7, "direction": "BUYER_ABSORPTION"})


def test_entry_is_strictly_after_interaction_end_and_cutoff_force_flat_are_sealed():
    c = load_contract()
    assert "strictly after interaction_end" in c["execution"]["entry_trigger"]
    assert c["execution"]["latency"]["decision_latency"] == "1 millisecond after interaction_end"
    assert c["session_and_deduplication"]["entry_cutoff_utc"] == "22:45:00.000000000 UTC"
    assert c["session_and_deduplication"]["cutoff_liquidation_window_utc"] == "22:44:59.000000000 through 22:45:00.000000000 UTC"
    assert "LAST valid executable ES market observation" in c["session_and_deduplication"]["rule"]


def test_cutoff_forced_exit_uses_last_valid_inclusive_window_observation_at_or_before_cutoff():
    result = synthetic_cutoff_result([{"timestamp": 81898.9, "valid": True}, {"timestamp": 81899.0, "valid": True}, {"timestamp": 81900.0, "valid": True}])
    assert result["outcome"] == "CUTOFF_FORCED_FLAT"
    assert result["exit_timestamp"] == 81900.0
    assert result["exit_timestamp"] <= 81900.0


def test_post_cutoff_forced_exit_is_rejected_and_state_is_flat():
    result = synthetic_cutoff_result([{"timestamp": 81900.000000001, "valid": True}])
    assert result == {"pending_orders": 0, "position_open": False, "exit_timestamp": None, "outcome": "CUTOFF_EXECUTION_INTEGRITY_FAILURE"}


def test_missing_valid_cutoff_window_observation_fails_closed():
    result = synthetic_cutoff_result([{"timestamp": 81900.0, "valid": False}])
    assert result["outcome"] == "CUTOFF_EXECUTION_INTEGRITY_FAILURE"
    assert result["exit_timestamp"] is None


def test_integer_es_sizing_and_insufficient_budget_have_no_fractional_contracts():
    price_risk, commissions, risk, contracts = synthetic_contracts(250.00, 5001.00, 4996.50)
    assert price_risk == 225.00
    assert commissions == 6.00
    assert risk == 231.00
    assert contracts == 1 and isinstance(contracts, int)
    assert contracts * risk <= 250.00
    _, _, insufficient_risk, insufficient_contracts = synthetic_contracts(230.99, 5001.00, 4996.50)
    assert insufficient_risk == risk
    assert insufficient_contracts == 0


def test_sizing_slippage_is_embedded_in_fills_and_never_double_counted():
    entry_reference_price, stop_price = 5000.75, 4990.75
    entry_fill, stop_exit_fill_assumption = 5001.00, 4990.50
    raw_price_risk_usd = abs(entry_reference_price - stop_price) * 50.00
    slippage_contribution_usd = (abs(entry_fill - entry_reference_price) + abs(stop_exit_fill_assumption - stop_price)) * 50.00
    price_risk, commissions, initial_risk, contracts = synthetic_contracts(1_062.00, entry_fill, stop_exit_fill_assumption)
    assert raw_price_risk_usd == 500.00
    assert slippage_contribution_usd == 25.00
    assert price_risk == raw_price_risk_usd + slippage_contribution_usd
    assert commissions == 6.00 and initial_risk == 531.00
    assert contracts == 2  # Adding the separately reported $25 slippage again would incorrectly size only one ES contract.
    c = load_contract()
    sizing = c["risk_and_exits"]["sizing"]
    assert "already includes ENTRY_SLIPPAGE=1 adverse ES tick" in sizing["entry_fill_accounting"]
    assert "already includes EXIT_SLIPPAGE=1 adverse ES tick" in sizing["stop_exit_fill_assumption_accounting"]
    assert "Do not add entry or exit slippage a second time" in sizing["anti_double_counting"]


def test_exact_one_tick_entry_and_exit_slippage_are_sealed():
    c = load_contract()
    assert c["execution"]["slippage_constants"] == {"ENTRY_SLIPPAGE": "1 adverse ES tick", "EXIT_SLIPPAGE": "1 adverse ES tick"}
    assert c["execution"]["entry_slippage"] == "ENTRY_SLIPPAGE=1 adverse ES tick. LONG entry fill = ask + 0.25; SHORT entry fill = bid - 0.25."
    assert c["execution"]["exit_slippage"] == "EXIT_SLIPPAGE=1 adverse ES tick. LONG exit fill = bid - 0.25; SHORT exit fill = ask + 0.25."


def test_fixed_250_risk_and_exact_five_tick_zone_stops_are_sealed():
    c = load_contract()
    sizing = c["risk_and_exits"]["sizing"]
    assert sizing["fixed_usd_risk_budget"] == 250.00
    assert sizing["budget_currency"] == "USD"
    stop = c["risk_and_exits"]["stop"]
    assert "stop_buffer_ticks=5" in stop
    assert "zone_low - (5 * 0.25) = zone_low - 1.25" in stop
    assert "zone_high + (5 * 0.25) = zone_high + 1.25" in stop
    assert "entry, level center, or NBBO" in stop


def test_causal_stop_target_ordering_and_reconciliation_invariants_are_sealed():
    c = load_contract()
    assert "chronological MBO market events" in c["risk_and_exits"]["fill_ordering"]
    assert "No OHLC collision rule" in c["risk_and_exits"]["fill_ordering"]
    invariants = c["required_reporting"]["reconciliation_invariants"]
    assert len(invariants) >= 7
    assert any("exactly one terminal exit reason" in x for x in invariants)
