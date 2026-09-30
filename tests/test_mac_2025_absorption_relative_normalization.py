from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

from research_pipeline.cme_orderflow_absorption_l2_v1 import mac_2025_absorption_relative_normalization as study


def _compact(ts, *, bid5=None, ask5=None, mid=None, action=None, side=None, size=None, mlofi=None, denom=None):
    n = len(ts)
    rows = np.zeros(n, dtype=study.COMPACT_DTYPE)
    rows["ts"] = ts
    rows["mid"] = mid if mid is not None else np.arange(n) * .25 + 100
    rows["bid5"] = bid5 if bid5 is not None else 100
    rows["ask5"] = ask5 if ask5 is not None else 100
    rows["bid10"] = rows["bid5"] * 2
    rows["ask10"] = rows["ask5"] * 2
    rows["action"] = action if action is not None else 0
    rows["side"] = side if side is not None else 0
    rows["size"] = size if size is not None else 0
    rows["mlofi"] = mlofi if mlofi is not None else 0
    rows["denom"] = denom if denom is not None else 10
    return rows


def test_context_is_asof_strictly_before_query_and_windows_are_causal():
    rows = _compact([0, 1_000_000_000, 2_000_000_000, 3_000_000_000],
                    mid=[100, 100.25, 100.5, 200], action=[0, 1, 1, 0], side=[0, 1, -1, 0], size=[0, 7, 11, 0])
    features = study._rolling_features(rows, np.asarray([3_000_000_000]), np.asarray([1.0]))
    assert features["_index"].tolist() == [2]  # row at query timestamp is excluded
    assert features["AGGRESSIVE_VOLUME_1S_RAW"][0] == 11  # sell pressure into a long reversal
    assert features["TRADES_PER_SECOND_5S"][0] == .4
    assert features["VELOCITY_2S_TICKS_PER_SECOND"][0] == .5


def test_aggression_depth_uses_attacked_passive_side_and_direction():
    rows = _compact([1, 2], bid5=[50, 50], ask5=[200, 200], action=[1, 1], side=[-1, 1], size=[20, 30])
    long = study._rolling_features(rows, np.asarray([3]), np.asarray([1.0]))
    short = study._rolling_features(rows, np.asarray([3]), np.asarray([-1.0]))
    assert long["AGGRESSION_TO_DEPTH_1S"][0] == 20 / 10
    assert short["AGGRESSION_TO_DEPTH_1S"][0] == 30 / 40


def test_mlofi_depth_normalization_and_persistence_are_pre_event():
    rows = _compact(np.arange(12) * 250_000_000 + 1,
                    mlofi=[1] * 8 + [-1, -1, -1, -1], denom=[10] * 12)
    result = study._rolling_features(rows, np.asarray([2_750_000_000]), np.asarray([1.0]))
    assert result["MLOFI_1S_RAW"][0] == -2
    assert result["MLOFI_1S_DEPTH_NORMALIZED"][0] == -.2
    # Late rows at/after the query timestamp do not influence either MLOFI feature.
    changed = rows.copy(); changed[11]["mlofi"] = -10_000
    result2 = study._rolling_features(changed, np.asarray([2_750_000_000]), np.asarray([1.0]))
    assert result2["MLOFI_1S_RAW"][0] == result["MLOFI_1S_RAW"][0]
    assert 0 <= result["MLOFI_PERSISTENCE_2S"][0] <= 1


def test_percentile_ranks_use_only_previous_dates_and_fixed_tod_bin():
    history_global = {"RV_30S_RAW": [1.0, 2.0]}
    history_tod = {"RV_30S_RAW": {"EUROPE:0": [1.0, 2.0]}}
    session_start = study.baseline._session_windows("2025-04-02")["EUROPE"][0]
    events = [{"interaction_start_ns": session_start + 1_000_000_000, "session": "EUROPE", "RV_30S_RAW": 1.5}]
    samples = [{"timestamp_ns": session_start + 2_000_000_000, "session": "EUROPE", "RV_30S_RAW": 1_000.0}]
    study._add_percentiles(events, samples, history_global, history_tod, "2025-04-02")
    assert events[0]["RV_30S_TOD_PERCENTILE"] == 1 / 2
    assert events[0]["RV_30S_GLOBAL_PERCENTILE"] == 1 / 2
    assert history_global["RV_30S_RAW"][-1] == 1_000.0  # appended after event ranking


def test_rank_empty_is_none_and_sorted_rank_boundaries():
    assert study._rank(1, []) is None
    assert study._rank_sorted(2.0, np.asarray([1.0, 2.0, 3.0])) == 2 / 3


def test_resiliency_fraction_is_capped_and_has_prior_only_episode_cutoff():
    # A 20-contract sampled depletion followed by a 10-contract recovery gives .5.
    ts = np.arange(0, 2_000_000_000, 100_000_000, dtype=np.int64) + 1
    depth = np.full(len(ts), 100.0); depth[3:] = 80; depth[5:] = 90
    rows = _compact(ts, bid5=depth / 2, ask5=depth / 2)
    result = study._resiliency(rows, np.asarray([700_000_000, 2_000_000_001]))
    assert result["RESILIENCY_MEDIAN_500MS_PRIOR_60S"][0] != result["RESILIENCY_MEDIAN_500MS_PRIOR_60S"][0]  # no fully matured 500ms episode yet
    assert result["RESILIENCY_MEDIAN_500MS_PRIOR_60S"][1] == .5


def test_markouts_mfe_mae_and_barriers_use_direction_normalized_public_path():
    events = np.zeros(5, dtype=[("timestamp_ns", "<i8"), ("bid", "<f8"), ("ask", "<f8"),
                                ("execution_price", "<f8"), ("execution_size", "<i8"), ("aggressor", "i1"), ("session", "i1")])
    events["timestamp_ns"] = [0, 1_000_000_000, 2_000_000_000, 3_000_000_000, 31_000_000_000]
    events["bid"] = [100, 100, 100.5, 99.5, 105]; events["ask"] = events["bid"] + .25
    candidate = {"interaction_start_ns": 500_000_000, "direction": "BUYER_ABSORPTION", "session": "NY"}
    out = study._path_outcomes(candidate, events, {"NY": (0, 40_000_000_000)})
    assert out["markouts_ticks"]["1000"] == 2
    assert out["mfe_mae_ticks"]["2000"]["mfe"] == 2
    assert out["mfe_mae_ticks"]["2000"]["mae"] == 0
    assert out["markouts_ticks"]["30000"] == 20  # first quote after 30s is included
    assert out["barriers"]["+1/-1"] == "UP"


def test_30s_markout_does_not_read_beyond_session_boundary():
    events = np.zeros(3, dtype=[("timestamp_ns", "<i8"), ("bid", "<f8"), ("ask", "<f8"),
                                ("execution_price", "<f8"), ("execution_size", "<i8"), ("aggressor", "i1"), ("session", "i1")])
    events["timestamp_ns"] = [0, 1_000_000_000, 31_000_000_000]
    events["bid"] = [100, 100, 105]; events["ask"] = events["bid"] + .25
    candidate = {"interaction_start_ns": 500_000_000, "direction": "BUYER_ABSORPTION", "session": "NY"}
    out = study._path_outcomes(candidate, events, {"NY": (0, 30_000_000_000)})
    assert out["markouts_ticks"]["30000"] is None


def test_checkpoint_path_outcome_refresh_uses_verified_tape_without_dbn(tmp_path):
    tape = tmp_path / "tape.npz"
    path_events = np.zeros(5, dtype=[("timestamp_ns", "<i8"), ("bid", "<f8"), ("ask", "<f8"),
                                     ("execution_price", "<f8"), ("execution_size", "<i8"),
                                     ("aggressor", "i1"), ("session", "i1")])
    path_events["timestamp_ns"] = [0, 1_000_000_000, 2_000_000_000, 3_000_000_000, 31_000_000_000]
    path_events["bid"] = [100, 100, 100.5, 99.5, 105]; path_events["ask"] = path_events["bid"] + .25
    metadata = {"date": "2025-04-02", "source_sha256": "source", "semantic_sha256": study.EXPECTED_TAPE_SEMANTIC_SHA}
    np.savez(tape, metadata_json=np.asarray(json.dumps(metadata)), events=path_events)
    checkpoint = {"source_sha256": "source", "events": [{"interaction_start_ns": 500_000_000,
                  "direction": "BUYER_ABSORPTION", "session": "NY", "markouts_ticks": {"30000": None}}]}
    assert study._refresh_checkpoint_path_outcomes(checkpoint, tape, "2025-04-02")
    assert checkpoint["events"][0]["markouts_ticks"]["30000"] == 20
    assert checkpoint["path_outcome_version"] == study.PATH_OUTCOME_VERSION
    assert not study._refresh_checkpoint_path_outcomes(checkpoint, tape, "2025-04-02")


def test_prototype_gate_decision_requires_cross_period_evidence_not_nonempty_families():
    gates = {}
    for family in ("F1", "F2"):
        for period in ("APRIL_2025", "OCTOBER_2025"):
            gates[f"{family}|{period}"] = {
                "BASELINE": {"markouts": {"5000": {"n": 50, "mean": -1.0}}},
                **{gate: {"markouts": {"5000": {"n": 30, "mean": 0.0 if family == "F1" else -2.0}}}
                   for gate in "ABCDEF"},
            }
    decision, evidence = study._prototype_gate_decision(gates)
    assert decision == "PROMISING_BUT_NEEDS_MORE_DATA"
    assert evidence["families_with_at_least_one_gate_improving_5s_mean_in_both_periods"] == {"F1": list("ABCDEF")}


def test_permutation_is_deterministic_and_bucket_counts_preserved():
    events = []
    for i in range(80):
        events.append({"date": "2025-04-01", "family": sorted(study.LIVE_TO_TAPE)[0],
                       "RV_30S_TOD_PERCENTILE": i / 80,
                       "AGGRESSION_TO_DEPTH_1S_TOD_PERCENTILE": i / 80,
                       "ER_5S_GLOBAL_PERCENTILE": i / 80,
                       "MLOFI_PERSISTENCE_DIRECTIONAL": (i % 9) / 9,
                       "markouts_ticks": {"5000": float(i % 7)}})
    a = study._permutation(events, repetitions=3); b = study._permutation(events, repetitions=3)
    assert a == b
    assert a["RV_30S_TOD_PERCENTILE"]["seed"] == 20250930


def test_daily_aggregation_and_family_groups_do_not_mix():
    events = []
    for family, move in [("F1", 1), ("F2", -1)]:
        for i in range(20):
            events.append({"date": "2025-04-01", "family": family, "RV_30S_TOD_PERCENTILE": i / 20,
                           "markouts_ticks": {"5000": move * i}})
    f1 = [e for e in events if e["family"] == "F1"]
    f2 = [e for e in events if e["family"] == "F2"]
    assert study._daily_stability(f1, "RV_30S_TOD_PERCENTILE")["positive_dates"] == 1
    assert study._daily_stability(f2, "RV_30S_TOD_PERCENTILE")["negative_dates"] == 1


def test_checkpoint_resume_requires_source_tape_and_config_hashes(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "OUT_ROOT", tmp_path)
    tape = tmp_path / "tape.npz"; tape.write_bytes(b"tape")
    cp = study._date_checkpoint_path("2025-04-01")
    study._write_gzip_json(cp, {"status": "COMPLETE", "source_sha256": "s",
                                "tape_sha256": study._sha(tape), "config_sha256": "c"})
    assert study._valid_checkpoint("2025-04-01", "s", tape, "c") is not None
    assert study._valid_checkpoint("2025-04-01", "bad", tape, "c") is None
    tape.write_bytes(b"changed")
    assert study._valid_checkpoint("2025-04-01", "s", tape, "c") is None


def test_source_semantic_identity_is_pinned():
    assert study.EXPECTED_STRATEGY_MANIFEST_SHA == "6c55756af201a20e11bfb87753142f9886ebfa26c796a37598a726f0ef66f56f"
    assert set(study.LIVE_TO_TAPE) == {"EUROPE|EUROPE|CURRENT|HIGH", "EUROPE|EUROPE|PRIOR|HIGH",
                                      "EUROPE|EUROPE|PRIOR|VAH", "NY|NY|PRIOR|POC"}
