from __future__ import annotations

import gzip
import json

import numpy as np
import pytest

from research_pipeline.cme_orderflow_absorption_l2_v1 import (
    mac_2025_absorption_mechanism_state as study,
    mac_2025_absorption_relative_normalization as norm,
)


def _compact(ts, *, mid=None, bid5=None, ask5=None, mlofi=None, denom=None):
    rows = np.zeros(len(ts), dtype=norm.COMPACT_DTYPE)
    rows["ts"] = ts
    rows["mid"] = mid if mid is not None else np.arange(len(ts), dtype=float) * .25 + 100
    rows["bid5"] = bid5 if bid5 is not None else 100
    rows["ask5"] = ask5 if ask5 is not None else 100
    rows["bid10"] = rows["bid5"] * 2
    rows["ask10"] = rows["ask5"] * 2
    rows["mlofi"] = mlofi if mlofi is not None else 0
    rows["denom"] = denom if denom is not None else 10
    return rows


def _path(ticks, *, start=10_000_000_000, step=1_000_000_000):
    # Include a strictly prior anchor and a complete 30-second path.
    times = np.arange(start - step, start + 31 * step, step, dtype=np.int64)
    moves = np.zeros(len(times), dtype=float)
    moves[1:1 + len(ticks)] = np.asarray(ticks, dtype=float) * study.TICK
    if len(ticks) < len(moves) - 1:
        moves[1 + len(ticks):] = moves[len(ticks)]
    out = np.zeros(len(times), dtype=[("timestamp_ns", "<i8"), ("bid", "<f8"), ("ask", "<f8")])
    out["timestamp_ns"] = times
    mid = 100 + moves
    out["bid"], out["ask"] = mid - .125, mid + .125
    return out


def _event(day="2025-03-03", family="EU_CURRENT_HIGH_SWEEP", markout=1.0, decile=0):
    p = decile / 10 + .01
    return {"date": day, "period": "SPRING_2025", "live_family": family,
            "session": "EUROPE", "interaction_start_ns": 1_000_000_000,
            "direction": "BUYER_ABSORPTION", "pre_event_features": {
                "feature_deciles": {name: decile for name in study.FEATURES},
                study.FEATURE_PERCENTILES["PRE_EVENT_RESILIENCY"]: p,
                study.FEATURE_PERCENTILES["NORMALIZED_MLOFI_PERSISTENCE"]: p,
                study.FEATURE_PERCENTILES["IMPACT_PER_FLOW"]: p,
                study.FEATURE_PERCENTILES["TREND_EFFICIENCY_5S"]: p,
                study.FEATURE_PERCENTILES["TREND_EFFICIENCY_30S"]: p,
                study.FEATURE_PERCENTILES["RV_30S"]: p},
            "markouts_ticks": {str(h): markout for h in study.HORIZONS_MS},
            "mfe_mae_ticks": {str(h): {"mfe": max(0, markout), "mae": max(0, -markout)} for h in study.PATH_HORIZONS_MS},
            "barriers": {f"+{u}/-{d}": "UP" for u, d in study.BARRIERS}}


def test_strict_pre_event_slices_exclude_equal_and_future_rows():
    ts = np.array([0, 10, 20, 30], dtype=np.int64)
    assert study._prior_strict_indices(ts, 20, 30) == (0, 2)
    assert ts[slice(*study._prior_strict_indices(ts, 20, 30))].tolist() == [0, 10]


def test_mlofi_persistence_uses_existing_latest_sign_bin_semantics_and_no_leakage():
    ts = np.arange(10, dtype=np.int64) * 250_000_000
    values = [1, 1, -1, -1, 2, 2, -3, -3, 999999, 999999]
    rows = _compact(ts, mlofi=values, denom=np.full(10, 10.0))
    got = study._pre_event_mlofi(rows, 2_000_000_000, 1.0)
    assert got["bin_signs"] == [1, 1, -1, -1, 1, 1, -1, -1]
    assert got["persistence"] == .5
    assert got["opposing_flow_persistence"] == .5
    changed = rows.copy(); changed[8:]["mlofi"] = -1e9
    assert study._pre_event_mlofi(changed, 2_000_000_000, 1.0) == got


def test_impact_per_flow_formula_and_strict_cutoff():
    ts = np.arange(0, 12, dtype=np.int64) * 1_000_000_000
    rows = _compact(ts, mid=np.arange(12) * .25 + 100, mlofi=np.ones(12), denom=np.full(12, 10.0))
    got = study._pre_event_impact(rows, 11_000_000_000, 1.0)
    assert got["mid_change_ticks"] == 9
    assert got["normalized_mlofi_integral"] == 1.0
    assert got["impact_per_flow"] == pytest.approx(9)
    changed = rows.copy(); changed[11]["mid"] = 10000; changed[11]["mlofi"] = 1e9
    assert study._pre_event_impact(changed, 11_000_000_000, 1.0) == got


def test_trend_efficiency_reuses_validated_path_efficiency_formula():
    ts = np.arange(0, 31, dtype=np.int64) * 1_000_000_000
    mid = 100 + np.arange(31, dtype=float) * .25
    rows = _compact(ts, mid=mid)
    result = norm._rolling_features(rows, np.asarray([30_000_000_000]), np.asarray([1.0]))
    assert result["ER_5S"][0] == pytest.approx(.8)  # validated as-of sample semantics include the lower-edge delta
    assert result["ER_30S"][0] == pytest.approx(1.0)


def test_resiliency_episodes_and_refill_latency_use_only_fully_prior_windows():
    grid = np.arange(41, dtype=np.int64) * study.ROLLING_TICK_NS
    depth = np.full(41, 100.0); depth[10:12] = 80; depth[12:] = 90
    bid, ask = depth / 2, depth / 2
    before = study._pre_event_resiliency(grid, bid, ask, int(grid[32]), "BID")
    changed = depth.copy(); changed[32:] = 10000
    after = study._pre_event_resiliency(grid, changed / 2, changed / 2, int(grid[32]), "BID")
    assert before == after
    assert before["episodes"]["500MS"]["n"] > 0
    assert before["refill_50pct_latency_ms"]["n"] > 0


def test_topology_precedence_and_path_statistics_are_deterministic():
    candidate = {"zone_low": 1.0, "zone_high": 200.0}
    event = {"interaction_start_ns": 10_000_000_000, "direction": "BUYER_ABSORPTION"}
    cases = [
        ([0, 0, -4], "A_IMMEDIATE_CONTINUATION_FAILURE"),
        ([0, 2, -4], "B_TEMPORARY_REVERSAL_THEN_FAILURE"),
        ([0, 0, 0, 0, 0, -4], "C_STAGNATION_OR_CHOP_THEN_FAILURE"),
        ([0, 0, 0, 0, 0, 0, 4], "D_DELAYED_SUCCESSFUL_REVERSAL"),
        ([0, 0, 4], "E_SUCCESSFUL_REVERSAL"),
        ([0, 0, 0], "F_UNCLASSIFIED_INSUFFICIENT_PATH"),
    ]
    for ticks, expected in cases:
        got = study._path_metrics(event, candidate, _path(ticks), 50_000_000_000)
        assert got["topology"] == expected
        assert got["path_sufficient"] is True
        assert "500" in got["markouts_ticks"]
        assert "+12/-6" in got["barriers"]
    assert study._path_metrics(event, candidate, _path([0, 0, 4]), 20_000_000_000)["topology"] == "F_UNCLASSIFIED_INSUFFICIENT_PATH"
    assert study._excursion_stats([{"mfe_mae_ticks": {"5000": {"mfe": 4.0, "mae": 2.0}}}], 5000)["mfe"]["mean"] == 4.0
    assert study._excursion_stats([{"mfe_mae_ticks": {"5000": {"mfe": 4.0, "mae": 2.0}}}], 5000)["p_mfe_at_least"]["4"] == 1.0


def test_full_path_uses_first_quote_at_or_after_30s_horizon():
    start = 10_250_000_000
    times = np.arange(0, 42, dtype=np.int64) * 1_000_000_000
    path = np.zeros(len(times), dtype=[("timestamp_ns", "<i8"), ("bid", "<f8"), ("ask", "<f8")])
    path["timestamp_ns"] = times
    path["bid"] = 100.0; path["ask"] = 100.25
    event = {"interaction_start_ns": start, "direction": "BUYER_ABSORPTION"}
    candidate = {"zone_low": 1.0, "zone_high": 200.0}
    got = study._path_metrics(event, candidate, path, 50_000_000_000)
    assert got["path_sufficient"] is True
    # The horizon falls between quote timestamps; the next quote is the endpoint.
    assert got["markouts_ticks"]["30000"] == 0


def test_path_never_uses_next_session_quote_to_fill_exact_boundary_horizon():
    times = np.asarray([0, 10_000_000_000, 39_000_000_000, 41_000_000_000], dtype=np.int64)
    path = np.zeros(len(times), dtype=[("timestamp_ns", "<i8"), ("bid", "<f8"), ("ask", "<f8")])
    path["timestamp_ns"] = times; path["bid"] = 100; path["ask"] = 100.25
    event = {"interaction_start_ns": 10_000_000_000, "direction": "BUYER_ABSORPTION"}
    candidate = {"zone_low": 1.0, "zone_high": 200.0}
    got = study._path_metrics(event, candidate, path, 40_000_000_000)
    assert got["path_sufficient"] is False
    assert got["topology_f_reason"] == "NO_QUOTE_AT_OR_AFTER_30S_HORIZON"


def test_history_percentiles_are_prior_date_only_and_deciles_fixed():
    first = _event("2025-03-03", decile=1)
    second = _event("2025-03-04", decile=8)
    first["pre_event_features"].update({
        "pre_event_recovery_ratio_500ms": .2, "opposing_mlofi_persistence_2s": .2,
        "pre_event_impact_per_flow_10s": 2.0})
    second["pre_event_features"].update({
        "pre_event_recovery_ratio_500ms": .8, "opposing_mlofi_persistence_2s": .8,
        "pre_event_impact_per_flow_10s": 8.0})
    for e in (first, second):
        start = study.baseline._session_windows(e["date"])["EUROPE"][0]
        e["interaction_start_ns"] = start + 60_000_000_000
    study._history_percentiles([second, first])
    assert first["pre_event_features"]["pre_event_recovery_ratio_500ms_historical_percentile"] is None
    assert second["pre_event_features"]["pre_event_recovery_ratio_500ms_historical_percentile"] == 1.0
    assert second["pre_event_features"]["feature_deciles"]["PRE_EVENT_RESILIENCY"] == 9
    assert second["pre_event_features"]["impact_per_flow_10s_historical_percentile"] == 1.0
    assert second["pre_event_features"]["feature_deciles"]["IMPACT_PER_FLOW"] == 9


def test_continuous_shape_and_daily_aggregation_keep_relationship_definition():
    events = []
    for d in range(10):
        for y in (float(d), float(d) + .2):
            e = _event(markout=y, decile=d)
            e["period"] = "SPRING_2025"
            events.append(e)
    shape = study._continuous_shape(events, "PRE_EVENT_RESILIENCY")
    assert shape["SPRING_2025"]["EU_CURRENT_HIGH_SWEEP"]["deciles"][9]["markouts"]["5000"]["n"] == 2
    daily = study._daily_rows(events, "PRE_EVENT_RESILIENCY", ["2025-03-03", "2025-03-04"])
    assert daily["dates"]["2025-03-03"]["markouts"]["5000"]["left"]["n"] == 4
    assert daily["dates"]["2025-03-03"]["markouts"]["5000"]["right"]["n"] == 4
    assert daily["dates"]["2025-03-04"]["event_count"] == 0
    assert daily["insufficient_dates"] == 1


def test_lodo_lowo_and_iso_week_omission_are_explicit():
    events = []
    for week, day in (("10", "03"), ("11", "10"), ("12", "17"), ("13", "24")):
        date_text = f"2025-03-{day}"
        for decile, value in ((0, -1), (1, -1), (8, 2), (9, 2)):
            e = _event(date_text, markout=value, decile=decile)
            e["period"] = "SPRING_2025"
            events.append(e)
    expected = [f"2025-03-{d}" for d in ("03", "10", "17", "24", "31")]
    result = study._lodo_lowo(events, ("PRE_EVENT_RESILIENCY",), expected)["PRE_EVENT_RESILIENCY"]
    assert result["LODO"]["groups_tested"] == 5
    assert result["weeks_tested"] == 5
    assert result["LOWO"]["groups_tested"] == 5


def test_interaction_construction_and_predeclared_good_bad_contrast():
    good = _event(decile=9)
    bad = _event(decile=0)
    for e, resil, mlofi, y in ((good, .9, .1, 2.0), (bad, .1, .9, -2.0)):
        f = e["pre_event_features"]
        f[study.FEATURE_PERCENTILES["PRE_EVENT_RESILIENCY"]] = resil
        f[study.FEATURE_PERCENTILES["NORMALIZED_MLOFI_PERSISTENCE"]] = mlofi
        e["markouts_ticks"]["5000"] = y
    assert study._interaction_labels(good, "RESILIENCY_X_OPPOSING_MLOFI") == ("HIGH", "LOW")
    assert study._interaction_labels(bad, "RESILIENCY_X_OPPOSING_MLOFI") == ("LOW", "HIGH")
    assert study._interaction_effect([good, good, bad, bad], "RESILIENCY_X_OPPOSING_MLOFI") == 4.0

    favorable, unfavorable = _event(markout=2), _event(markout=-2)
    favorable["pre_event_features"][study.FEATURE_PERCENTILES["IMPACT_PER_FLOW"]] = .1
    favorable["pre_event_features"][study.FEATURE_PERCENTILES["ER_30S"]] = .2
    unfavorable["pre_event_features"][study.FEATURE_PERCENTILES["IMPACT_PER_FLOW"]] = .9
    unfavorable["pre_event_features"][study.FEATURE_PERCENTILES["ER_30S"]] = .9
    assert study._interaction_labels(favorable, "IMPACT_PER_FLOW_X_ER") == ("LOW", "LOW")
    assert study._interaction_labels(unfavorable, "IMPACT_PER_FLOW_X_ER") == ("HIGH", "HIGH")
    assert study._interaction_effect([favorable, favorable, unfavorable, unfavorable], "IMPACT_PER_FLOW_X_ER") == 4.0


def test_neighbor_stability_uses_fixed_percentile_representations():
    events = []
    for i, p in enumerate([.05, .1, .15, .25, .35, .65, .75, .85, .9, .95]):
        e = _event(markout=(1 if p >= .65 else -1), decile=min(9, int(p * 10)))
        e["pre_event_features"][study.FEATURE_PERCENTILES["PRE_EVENT_RESILIENCY"]] = p
        events.append(e)
    result = study._neighbor_stability(events, ("PRE_EVENT_RESILIENCY",))["PRE_EVENT_RESILIENCY"]
    assert result["representations"]["deciles"]["high_n"] == 2
    assert result["representations"]["quintiles"]["low_n"] == 3
    assert result["representations"]["terciles"]["high_n"] == 4


def test_permutation_is_deterministic_for_primary_and_interaction_labels():
    events = []
    for i in range(60):
        within = i % 30
        is_good = within >= 15
        e = _event(markout=(1 if is_good else -1), decile=9 if is_good else 0)
        f = e["pre_event_features"]
        f[study.FEATURE_PERCENTILES["PRE_EVENT_RESILIENCY"]] = .9 if is_good else .1
        f[study.FEATURE_PERCENTILES["NORMALIZED_MLOFI_PERSISTENCE"]] = .1 if is_good else .9
        e["date"] = "2025-03-03" if i < 30 else "2025-03-04"
        e["markouts_ticks"]["5000"] = -1 if i < 30 else 1
        events.append(e)
    relationships = ("PRE_EVENT_RESILIENCY", "RESILIENCY_X_OPPOSING_MLOFI")
    a = study._permutation(events, relationships, repetitions=9)
    assert a == study._permutation(events, relationships, repetitions=9)
    assert a["PRE_EVENT_RESILIENCY"]["permutations"] == 9


def test_family_roles_and_checkpoint_resume_hash_invalidation(tmp_path):
    assert set(study.SPARSE) == {"EU_PRIOR_HIGH", "PRIOR_VAH"}
    assert study.DISCOVERY == "EU_CURRENT_HIGH_SWEEP" and study.REPLICATION == "NY_W04"
    tape = tmp_path / "tape.npz"; tape.write_bytes(b"tape")
    study.PRIOR_ROOT = tmp_path
    cp_path = tmp_path / "checkpoints" / "2025-03-03.json.gz"
    cp_path.parent.mkdir()
    payload = {"status": "COMPLETE", "source_sha256": "src", "tape_sha256": study._sha(tape), "config_sha256": "cfg"}
    with gzip.open(cp_path, "wt", encoding="utf-8") as f: json.dump(payload, f)
    assert study._load_cached_date("2025-03-03", "src", tape, "cfg")["status"] == "COMPLETE"
    with pytest.raises(study.MechanismStudyError): study._load_cached_date("2025-03-03", "wrong", tape, "cfg")
    with pytest.raises(study.MechanismStudyError): study._load_cached_date("2025-03-03", "src", tape, "wrong")
    tape.write_bytes(b"changed")
    with pytest.raises(study.MechanismStudyError): study._load_cached_date("2025-03-03", "src", tape, "cfg")


def test_current_study_checkpoint_resume_is_bound_to_version_and_all_input_hashes():
    cp = {"status": "COMPLETE", "date": "2025-03-03", "source_sha256": "s", "tape_sha256": "t",
          "config_sha256": "c", "study_version_hash": study.STUDY_VERSION_HASH, "events": []}
    assert study._valid_checkpoint_payload(cp, "2025-03-03", "s", "t", "c")
    for key, wrong in (("source_sha256", "bad"), ("tape_sha256", "bad"),
                       ("config_sha256", "bad"), ("study_version_hash", "bad")):
        altered = dict(cp); altered[key] = wrong
        assert not study._valid_checkpoint_payload(altered, "2025-03-03", "s", "t", "c")
    assert not study._valid_checkpoint_payload({**cp, "status": "RUNNING"}, "2025-03-03", "s", "t", "c")
    assert study._base_checkpoint_payload(cp, "2025-03-03", "s", "t", "c")
    assert not study._base_checkpoint_payload({**cp, "source_sha256": "bad"}, "2025-03-03", "s", "t", "c")
