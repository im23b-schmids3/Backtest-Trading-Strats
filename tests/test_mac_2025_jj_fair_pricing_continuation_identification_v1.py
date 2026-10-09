from research_pipeline.cme_orderflow_absorption_l2_v1.mac_2025_jj_fair_pricing_continuation_identification_v1 import (
    COHORTS,
    FEATURES,
    _canonical_hash,
    _contract,
    _match,
    _stds,
)
from research_pipeline.cme_orderflow_absorption_l2_v1.mac_2025_jj_fair_pricing_v2_1_corrected_bos import _synthetic_production_catalog


def _row(minute, *, cohort=None, direction=1, offset=0.0, signal_id="s"):
    cov = {
        "time_minute": float(minute),
        "prior_1m_directional_ticks": 1.0 + offset,
        "prior_2m_directional_ticks": 2.0 + offset,
        "log_prior_5m_range_ticks": 1.0 + offset,
        "anchor_distance_ticks": 4.0 + offset,
    }
    return {"date": "2025-03-03", "session": "NY_AM", "minute": minute,
            "timestamp_ns": minute * 60_000_000_000, "direction": direction,
            "covariates": cov, "cohort": cohort, "signal_id": signal_id}


def test_frozen_contract_hash_is_stable_and_covariate_only():
    contract = _contract()
    assert contract["status"].startswith("EXPLORATORY_INTERNAL")
    assert contract["matching"]["covariates"] == list(FEATURES)
    assert contract["outcomes"]["primary"].startswith("directional executable BBO net markout")
    assert _canonical_hash(contract) == _canonical_hash(contract)
    assert "markout" not in " ".join(contract["matching"]["covariates"])


def test_matching_is_deterministic_exact_stratum_and_without_replacement():
    treated = [_row(5, cohort="DISPLACEMENT_ONLY", signal_id="s1"),
               _row(6, cohort="DISPLACEMENT_ONLY", signal_id="s2")]
    controls = [_row(5, signal_id="c1"), _row(7, signal_id="c2"),
                {**_row(5, signal_id="wrong-date"), "date": "2025-03-04"},
                {**_row(5, signal_id="wrong-direction"), "direction": -1}]
    scales = _stds([*treated, *controls])
    first = _match(treated, controls, scales)
    second = _match(treated, controls, scales)
    assert [(x["signal"]["signal_id"], x["control"]["signal_id"]) for x in first] == [
        (x["signal"]["signal_id"], x["control"]["signal_id"]) for x in second]
    assert len(first) == 2
    assert len({x["control"]["minute"] for x in first}) == 2
    assert all(x["signal"]["date"] == x["control"]["date"] for x in first)
    assert all(x["signal"]["direction"] == x["control"]["direction"] for x in first)


def test_all_trigger_cohorts_are_disjoint_and_declared():
    assert COHORTS == ("DISPLACEMENT_ONLY", "BOS_ONLY", "COMBINED")
    assert len(set(COHORTS)) == 3


def test_actual_production_catalog_accepts_short_continuation():
    proof = _synthetic_production_catalog()
    assert proof["production_signal_catalog_all_direction_cases_pass"]
    short = proof["cases"]["short_continuation"]
    assert short["expected_direction"] == "SHORT"
    assert short["combined_accepted"]
    assert any(x["direction"] == "SHORT" and x["trigger"] == "BOS_PLUS_DISPLACEMENT" for x in short["signals"])
