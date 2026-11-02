from __future__ import annotations

import numpy as np

from research_pipeline.cme_orderflow_absorption_l2_v1 import mac_2025_hvn_context_interaction_research_v1 as study


def test_session_labels_are_iana_dst_aware():
    # 2025-03-10 13:30Z is 09:30 EDT; the same UTC hour on March 7 is 08:30 EST.
    assert study.session_label_et(1741613400 * study.NS) == "NY_OPEN"
    assert study.session_label_et(1741354200 * study.NS) == "NY_PREOPEN"


def test_feature_at_excludes_native_rows_after_target_and_reports_causal_window():
    target_ns = 1741613400 * study.NS
    q = {
        "idx": np.array([1, 2, 3]), "recv": target_ns + np.array([-10, 0, 10]) * study.NS,
        "mid": np.array([100.0, 101.0, 500.0]), "spread": np.ones(3),
        "top1b": np.ones(3), "top1a": np.ones(3), "top5b": np.ones(3),
        "top5a": np.ones(3), "top10b": np.ones(3), "top10a": np.ones(3),
    }
    t = {"idx": np.array([], dtype=np.int64), "recv": np.array([], dtype=np.int64),
         "side": np.array([], dtype="S1"), "size": np.array([], dtype=np.int64), "price": np.array([])}
    feature = study._feature_at({"entry_bid": 100.75, "entry_ask": 101.25, "session": 2}, 2,
                                target_ns, q, t, "2025-03-10", "TRADE_ENTRY")
    assert feature["quote_mid"] == 101.0
    assert feature["entry_native_index"] == 2
    assert feature["session_et"] == "NY_OPEN"


def test_date_bootstrap_resamples_dates_and_is_deterministic():
    treatment = [{"date": "d1", "net_r": 2.0}, {"date": "d2", "net_r": 1.0}]
    control = [{"date": "d1", "net_r": 0.0}, {"date": "d2", "net_r": -1.0}]
    a = study.date_bootstrap_difference(treatment, control, ["d1", "d2", "d3"], replicates=100, seed=3)
    b = study.date_bootstrap_difference(treatment, control, ["d1", "d2", "d3"], replicates=100, seed=3)
    assert a == b
    assert a["matched_dates"] == 2
    assert a["replicates_requested"] == 100


def test_bh_is_monotone_in_sorted_p_value_order():
    p = [0.04, 0.001, 0.02]
    q = study.benjamini_hochberg(p)
    ordered = sorted(zip(p, q))
    assert [v for _, v in ordered] == sorted(v for _, v in ordered)
