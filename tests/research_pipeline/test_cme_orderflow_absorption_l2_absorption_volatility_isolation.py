from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.research_pipeline.cme_orderflow_absorption_l2_v1 import absorption_regime_study as core
from src.research_pipeline.cme_orderflow_absorption_l2_v1 import absorption_volatility_isolation as study


def _event(day: str, i: int, *, family: str = "FAM", ts: int | None = None,
           bucket: str = "Q1", outcome: float = 0.0) -> dict:
    timestamp = ts if ts is not None else i * study.NS
    row = {"date": day, "family": family, "session": "NY", "direction": "BUYER_ABSORPTION",
           "event_id": f"{day}-{family}-{i}", "interaction_id": f"ix-{i}", "interaction_start_ns": timestamp,
           "interaction_end_ns": timestamp + 10, "interaction_end_price": 100.0, "level": "PRIOR_RTH_POC",
           "rv_30s_ticks_q5": bucket, "rv_30s_ticks_tercile": "MEDIUM",
           "tod_bucket": "CASH_OPEN", "rv_30s_ticks_expanding_pct": .5, "rv_30s_ticks": float(i + 1), "rv_120s_ticks": float(i + 2),
           "rv_10s_ticks": float(i), "rv_300s_ticks": float(i + 3), "tod_norm_rv_30s": 1.0,
           "tod_norm_rv_120s": 1.0, "rv_30s_within_tod_pct": .5, "rv_30s_tod_norm_pct": .5,
           "rv_30s_within_session_pct": .5, "rv_120s_within_tod_pct": .5,
           "rv_120s_tod_norm_pct": .5, "rv_120s_within_session_pct": .5,
           "markout_250ms_ticks": outcome, "markout_500ms_ticks": outcome,
           "markout_1000ms_ticks": outcome, "markout_2000ms_ticks": outcome,
           "markout_5000ms_ticks": outcome, "markout_10000ms_ticks": outcome, "markout_30000ms_ticks": outcome,
           "barrier_1_1_outcome": "UNRESOLVED", "barrier_1_1_seconds": None}
    for ms in (1000, 2000, 5000, 10000, 30000):
        row[f"mfe_{ms}ms_ticks"] = max(0.0, outcome)
        row[f"mae_{ms}ms_ticks"] = max(0.0, -outcome)
    for field in study.VOL_FEATURES:
        row[field] = 1.0
    for field in ("pre_30s_top5_depth_mean", "pre_30s_pressured_top5_depth_mean", "pre_event_spread_ticks",
                  "pre_5s_trades_per_second", "pre_30s_trades_per_second", "pre_5s_contracts_per_second",
                  "pre_30s_contracts_per_second", "pre_5s_book_updates_per_second", "pre_30s_book_updates_per_second",
                  "pre_5s_quote_records_per_second", "pre_30s_quote_records_per_second",
                  "velocity_ticks_per_second_5s", "velocity_ticks_per_second_30s",
                  "directional_velocity_ticks_per_second_5s", "directional_velocity_ticks_per_second_30s",
                  "trend_efficiency_5s", "trend_efficiency_30s"):
        row[field] = float(i + 1)
    return row


def test_realized_volatility_and_time_of_day_normalization_are_deterministic():
    assert study.realized_volatility_ticks([100.0, 100.25, 100.5]) == pytest.approx(np.sqrt(2))
    assert study.realized_volatility_ticks([100.0]) == 0
    assert study.tod_normalize(8.0, list(range(1, 21))) == pytest.approx(8 / 10.5)
    assert study.tod_normalize(8.0, list(range(19))) is None


def test_fixed_new_york_tod_and_europe_bucket_mapping():
    def stamp(hour: int, minute: int = 0) -> int:
        # Winter UTC-5: convert NY wall time to UTC.
        from datetime import datetime, timezone
        return int(datetime(2026, 1, 5, hour + 5, minute, tzinfo=timezone.utc).timestamp() * study.NS)
    assert study.tod_bucket(stamp(9, 45), "NY") == "CASH_OPEN"
    assert study.tod_bucket(stamp(10, 30), "NY") == "MORNING"
    assert study.tod_bucket(stamp(12), "NY") == "MIDDAY"
    assert study.tod_bucket(stamp(15, 45), "NY") == "CASH_CLOSE"
    assert study.tod_bucket(stamp(7), "EUROPE") == "EUROPE"
    assert study.tod_bucket(stamp(8, 30), "EUROPE") == "US_PREOPEN"
    assert study.tod_bucket(stamp(12), "ASIA") == "ASIA"


def test_velocity_efficiency_and_direction_are_pre_event_features():
    result = study.price_velocity_and_efficiency([100, 100.25, 100.5], 2, -1)
    assert result["velocity_ticks_per_second"] == pytest.approx(1)
    assert result["directional_velocity_ticks_per_second"] == pytest.approx(-1)
    assert result["trend_efficiency"] == pytest.approx(1)
    assert study.price_velocity_and_efficiency([100], 2, 1)["trend_efficiency"] is None


def test_depth_and_activity_terciles_use_prior_history_and_abnormal_proxy_is_frozen():
    values = list(map(float, range(1, 31)))
    assert study._tercile(1, values) == "LOW"
    assert study._tercile(20, values) == "MEDIUM"
    assert study._tercile(30, values) == "HIGH"
    assert study.abnormal_market_state("EXTREME", spread_abnormal=True,
                                       depth_abnormal=False, intensity_abnormal=True) == ("ABNORMAL_STRESS_STATE", 2)
    assert study.abnormal_market_state("HIGH", spread_abnormal=True,
                                       depth_abnormal=False, intensity_abnormal=False) == ("NORMAL_MARKET_STATE", 1)


def test_causal_buckets_do_not_use_same_date_or_future_date_to_calibrate_earlier_rows():
    date1 = [_event("2025-12-01", i, bucket="INSUFFICIENT_HISTORY") for i in range(25)]
    date2 = [_event("2025-12-02", i, ts=(i + 1) * study.NS) for i in range(4)]
    rows = {"2025-12-01": date1, "2025-12-02": date2}
    study._assign_causal_buckets(rows, list(rows))
    assert {r["rv_30s_ticks_q5"] for r in date1} == {"INSUFFICIENT_HISTORY"}  # no same-date calibration
    first_label = date2[0]["rv_30s_ticks_q5"]
    appended_future = {"2025-12-01": date1, "2025-12-02": date2,
                       "2025-12-03": [_event("2025-12-03", 1)]}
    study._assign_causal_buckets(appended_future, list(appended_future))
    assert date2[0]["rv_30s_ticks_q5"] == first_label
    assert date2[0]["rv_30s_ticks_expanding_pct"] is not None


def test_time_normalized_expanding_ranks_are_assigned_after_causal_normalization():
    by_date = {}
    for day_index, day in enumerate(("2025-12-01", "2025-12-02", "2025-12-03")):
        by_date[day] = []
        for i in range(30):
            row = _event(day, i, ts=(day_index * 100 + i) * study.NS)
            row["rv_30s_ticks"] = float(1 + i + day_index * 3)
            row["rv_120s_ticks"] = float(2 + i + day_index * 5)
            by_date[day].append(row)
    original = {day: [dict(row) for row in values] for day, values in by_date.items()}
    study._assign_causal_buckets(by_date, list(by_date))
    assert all(r["tod_norm_rv_30s"] is None for r in by_date["2025-12-01"])
    assert any(r["tod_norm_rv_30s"] is not None for r in by_date["2025-12-02"])
    labels = {r["tod_norm_rv_30s_q5"] for r in by_date["2025-12-03"]}
    assert labels <= {"Q1", "Q2", "Q3", "Q4", "Q5"}
    assert all(r["tod_norm_rv_30s_expanding_pct"] is not None for r in by_date["2025-12-03"])
    repaired = {day: [dict(row) for row in values] for day, values in original.items()}
    study._repair_tod_normalized_buckets(repaired, list(repaired))
    fields = ("tod_norm_rv_30s", "tod_norm_rv_120s", "tod_norm_rv_30s_baseline_median",
              "tod_norm_rv_120s_baseline_median", "rv_30s_within_tod_pct", "rv_120s_within_tod_pct",
              "rv_30s_tod_norm_pct", "rv_120s_tod_norm_pct", "tod_norm_rv_30s_expanding_pct",
              "tod_norm_rv_120s_expanding_pct", "tod_norm_rv_30s_q5", "tod_norm_rv_120s_q5",
              "tod_norm_rv_30s_tercile", "tod_norm_rv_120s_tercile", "tod_norm_rv_30s_state",
              "tod_norm_rv_120s_state")
    for day in by_date:
        for full_row, repaired_row in zip(by_date[day], repaired[day], strict=True):
            assert {key: full_row.get(key) for key in fields} == {key: repaired_row.get(key) for key in fields}


def test_date_safe_resiliency_uses_frozen_core_expanding_score_without_mutating_metric():
    values = []
    for i in range(30):
        row = _event("2025-12-01", i)
        row["resiliency_60s_score"] = i / 30
        values.append(row)
    history = {}
    core._assign_expanding_buckets(values, history)
    assert all(row["bucket_resiliency_60s_score"] == "INSUFFICIENT_HISTORY" for row in values)
    later = _event("2025-12-02", 1)
    later["resiliency_60s_score"] = .9
    core._assign_expanding_buckets([later], history)
    assert later["bucket_resiliency_60s_score"].startswith("Q")
    assert [row["resiliency_60s_score"] for row in values] == [i / 30 for i in range(30)]


def test_mfe_mae_and_first_touch_are_direction_normalized():
    assert core.mfe_mae(np.array([-1.0, 2.0, -3.0])) == (2.0, 3.0)
    ts = np.array([10, 20, 30], dtype=np.int64)
    result = core.first_touch(ts, np.array([1.0, -1.0, 2.0]), 1, 1)
    assert result["outcome"] == "FAVORABLE_FIRST"
    assert result["time_ns"] == 10


def test_daily_monthly_and_family_aggregation_keep_populations_separate():
    rows = []
    for i in range(12):
        rows.append(_event("2025-12-01", i, family="A", bucket="Q1", outcome=-1))
        rows.append(_event("2025-12-01", i + 20, family="A", bucket="Q5", outcome=1))
    for i in range(12):
        rows.append(_event("2026-01-02", i, family="B", bucket="Q1", outcome=-1))
        rows.append(_event("2026-01-02", i + 20, family="B", bucket="Q5", outcome=1))
    assert set(study._monthly_groups(rows)) == {"2025-12", "2026-01"}
    daily = study.daily_q5_q1_contrast(rows, "rv_30s_ticks_q5", 2000)
    assert daily["positive_dates"] == 2
    families = study._family_results(rows)
    assert set(families) == {"A", "B"}
    assert families["A"]["event_count"] == 24
    assert study._group_summary(rows)["mfe_mae"]["5000"]["probabilities"]["mfe_ge_ticks"]["1"] == .5


def test_clustered_bootstrap_and_permutation_are_reproducible():
    rows = []
    for day in range(8):
        for i in range(12):
            rows.append(_event(f"2026-01-{day + 1:02}", i, bucket="Q1", outcome=-1))
            rows.append(_event(f"2026-01-{day + 1:02}", i + 20, bucket="Q5", outcome=1))
    first = study.clustered_bootstrap(rows, "rv_30s_ticks_q5", "Q5_Q1", 2000, replicates=100, seed=11)
    second = study.clustered_bootstrap(rows, "rv_30s_ticks_q5", "Q5_Q1", 2000, replicates=100, seed=11)
    assert first == second
    perm1 = study.permutation_test(rows, "rv_30s_ticks_q5", 2000, replicates=100, seed=13)
    perm2 = study.permutation_test(rows, "rv_30s_ticks_q5", 2000, replicates=100, seed=13)
    assert perm1 == perm2
    assert study.permutation_test(rows, "rv_30s_state", 2000, replicates=20, contrast="HIGH_LOW")["contrast"] == "HIGH_LOW"


def test_optimized_permutation_matches_reference_label_shuffle_exactly():
    rows = []
    labels = ("Q1", "Q5", "Q2", "Q1", "Q5", "Q4", "Q1", "Q5")
    for i, label in enumerate(labels):
        rows.append(_event("2026-01-02" if i < 6 else "2026-01-05", i,
                           bucket=label, outcome=(i % 5) - 2))
        rows[-1]["session"] = "NY" if i % 2 else "EUROPE"
    replicates, seed = 50, 123
    actual = study.permutation_test(rows, "rv_30s_ticks_q5", 2000, seed=seed,
                                    replicates=replicates, contrast="Q5_Q1")
    values = np.asarray([float(r["markout_2000ms_ticks"]) for r in rows])
    original = np.asarray([r["rv_30s_ticks_q5"] for r in rows], dtype=object)
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault((row["date"], row["session"]), []).append(i)
    eligible = [np.asarray(ix, dtype=int) for ix in groups.values() if len(ix) >= 2]
    left = values[original == "Q5"].mean(); right = values[original == "Q1"].mean()
    observed = float(left - right); rng = np.random.default_rng(seed); extremes = 0
    for _ in range(replicates):
        shuffled = original.copy()
        for ix in eligible:
            shuffled[ix] = rng.permutation(shuffled[ix])
        a = values[shuffled == "Q5"]; b = values[shuffled == "Q1"]
        if len(a) and len(b) and abs(float(a.mean() - b.mean())) >= abs(observed):
            extremes += 1
    assert actual["extreme_replicates"] == extremes
    assert actual["two_sided_randomization_p"] == pytest.approx((extremes + 1) / (replicates + 1))


def test_event_overlap_density_and_deduplication_are_explicit():
    first = _event("2026-01-02", 1, family="A", bucket="Q1")
    duplicate = dict(first, event_id="other", family="B")
    rows = [first, duplicate]
    audit = study.event_overlap_audit(rows)
    assert audit["same_interaction_geometry"]["duplicate_excess_rows"] == 1
    assert audit["deduplicated_exact_geometry_count"] == 1
    assert len(study.deduplicate_event_clusters(rows)) == 1
    density = study._density(rows, "rv_30s_state")
    assert density["A"]["INSUFFICIENT_HISTORY"]["event_count"] == 1


def test_control_comparisons_use_volatility_quintiles_within_control_terciles():
    rows = []
    for control_i, control_label in enumerate(("LOW", "MEDIUM", "HIGH")):
        for bucket, outcome in (("Q1", 0.0), ("Q5", 1.0)):
            for i in range(study.MIN_CELL):
                row = _event(f"2026-01-{2 + (i % 3):02d}", control_i * 100 + i,
                             bucket=bucket, outcome=outcome)
                row["rv_30s_ticks_q5"] = bucket
                row["trend_efficiency_5s_tercile"] = control_label
                row["rv_30s_state"] = "HIGH" if bucket == "Q5" else "LOW"
                rows.append(row)

    within_tercile = study._control_results(
        rows, "trend_efficiency_5s_tercile", "rv_30s_ticks_q5")
    assert set(within_tercile) == {"LOW", "MEDIUM", "HIGH"}
    for cell in within_tercile.values():
        contrast = cell["q5_q1"]["2000"]
        assert contrast["q1_count"] == study.MIN_CELL
        assert contrast["q5_count"] == study.MIN_CELL
        assert contrast["q5_minus_q1"] == pytest.approx(1.0)
    assert study._control_survival(within_tercile)["classification"] == "SURVIVES"

    crosstab = study._state_control_crosstab(
        rows, "trend_efficiency_5s_tercile", "rv_30s_state")
    assert crosstab["LOW"]["LOW"]["event_count"] == study.MIN_CELL
    assert crosstab["LOW"]["HIGH"]["event_count"] == study.MIN_CELL
    assert crosstab["LOW"]["NORMAL"]["event_count"] == 0


def test_volatility_interaction_tercile_labels_are_translated_and_populated():
    expected = {"Q1": "LOW", "Q2": "MEDIUM", "Q3": "HIGH",
                "INSUFFICIENT_HISTORY": "INSUFFICIENT_HISTORY", None: "INSUFFICIENT_HISTORY"}
    assert {key: study.volatility_tercile_label(key) for key in expected} == expected
    rows = []
    for vol_bucket, vol_label in (("Q1", "LOW"), ("Q2", "MEDIUM"), ("Q3", "HIGH")):
        for resilience in ("SLOW", "NORMAL", "FAST"):
            row = _event("2026-01-05", len(rows), bucket="Q1", outcome=1.0)
            row.update({"rv_30s_tercile": vol_bucket,
                        "rv_30s_volatility_tercile_state": study.volatility_tercile_label(vol_bucket),
                        "resiliency_triplet_state": resilience,
                        "flow_state": "FLOW_SUPPORTS_REVERSAL"})
            rows.append(row)
    resilience = study._interaction_table(
        rows, "rv_30s_volatility_tercile_state", "resiliency_triplet_state",
        ("LOW", "MEDIUM", "HIGH"), ("SLOW", "NORMAL", "FAST"))
    flow = study._interaction_table(
        rows, "rv_30s_volatility_tercile_state", "flow_state",
        ("LOW", "MEDIUM", "HIGH"),
        ("FLOW_SUPPORTS_REVERSAL", "FLOW_NEUTRAL", "FLOW_OPPOSES_REVERSAL"))
    assert sum(cell["event_count"] for cell in resilience["cells"].values()) == 9
    assert sum(cell["event_count"] for cell in flow["cells"].values()) == 9


def test_date_checkpoint_resume_binds_source_event_config_and_code(tmp_path: Path):
    event_path = tmp_path / "date.jsonl.gz"
    study._atomic_jsonl_gz(event_path, [{"event_id": "x"}])
    source_records = [{"path": "source.dbn", "sha256": "abc"}]
    payload = {"date": "2026-01-02", "config_sha256": "cfg", "code_sha256": "code",
               "prior_event_sha256": "prior", "source_files": source_records,
               "output_sha256": study.sha256_file(event_path)}
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")
    kwargs = {"day": "2026-01-02", "source_records": source_records, "input_event_sha": "prior",
              "config_sha": "cfg", "code_sha": "code"}
    assert study._date_checkpoint_valid(checkpoint, event_path, **kwargs)
    assert not study._date_checkpoint_valid(checkpoint, event_path, **{**kwargs, "config_sha": "changed"})
    assert not study._date_checkpoint_valid(checkpoint, event_path, **{**kwargs, "source_records": []})
    event_path.write_bytes(b"changed")
    assert not study._date_checkpoint_valid(checkpoint, event_path, **kwargs)


def test_frozen_config_and_prior_inventory_identity_are_sealed():
    assert study.EXPECTED_CONFIG_SHA256 == "99c4af7f7b03cf6a255781524f7a6c2a32bd992d9785dbfb294db6a07fbc7448"
    assert study.RUN_ID == "CMEOrderflow_ABSORPTION_VOLATILITY_ISOLATION_DEC_JAN_V1"
    assert study.BOOTSTRAP_REPLICATES >= 1000
    assert study.PERMUTATION_REPLICATES >= 500


def test_synthetic_aggregation_emits_the_predeclared_artifact_family(tmp_path: Path):
    rows = []
    for day_i, day in enumerate(("2025-12-01", "2026-01-02")):
        for i in range(30):
            row = _event(day, i, bucket="Q1" if i < 10 else "Q5", outcome=-1 if i < 10 else 1)
            row.update({"rv_30s_state": "LOW" if i < 10 else "HIGH", "rv_120s_state": "NORMAL",
                        "tod_norm_rv_30s_state": "HIGH", "tod_norm_rv_120s_state": "NORMAL",
                        "rv_30s_ticks_q5": "Q1" if i < 10 else "Q5",
                        "rv_120s_ticks_q5": "Q1" if i < 10 else "Q5",
                        "tod_norm_rv_30s_q5": "Q1" if i < 10 else "Q5",
                        "tod_norm_rv_120s_q5": "Q1" if i < 10 else "Q5",
                        "rv_30s_tercile": "LOW" if i < 10 else "HIGH",
                        "resiliency_triplet_state": "SLOW" if i < 10 else "FAST",
                        "resiliency_speed_state": "SLOW" if i < 10 else "FAST",
                        "flow_state": "FLOW_SUPPORTS_REVERSAL",
                        "abnormal_state": "NORMAL_MARKET_STATE", "tod_bucket": "CASH_OPEN"})
            rows.append(row)
    result = study._run_aggregations(rows, tmp_path)
    required = {"volatility-buckets.json", "tod-normalized-volatility.json", "shape-analysis.json",
                "extreme-volatility.json", "session-control.json", "trend-control.json", "depth-control.json",
                "intensity-control.json", "abnormal-state.json", "daily-stability.json", "monthly-stability.json",
                "family-results.json", "mfe-mae-results.json", "barrier-results.json", "clustered-bootstrap.json",
                "permutation-results.json", "event-density.json", "overlap-audit.json", "volatility-x-resiliency.json",
                "volatility-x-flow-persistence.json", "prototype-gates.json", "prototype-gate-robustness.json"}
    assert required <= {p.name for p in tmp_path.iterdir()}
    assert result["overlap"]["raw_event_count"] == len(rows)
    norm = json.loads((tmp_path / "tod-normalized-volatility.json").read_text(encoding="utf-8"))
    rv30 = norm["tod_norm_rv_30s"]
    assert rv30["normalized_percentile"]["count"] == len(rows)
    assert set(rv30["within_tod_bucket"]) == set(study.TOD_BUCKETS)
    assert len(rv30["within_tod_bucket"]) <= 8  # fixed buckets, not one group per unique percentile
    months = {month: study._group_summary(members) for month, members in study._monthly_groups(rows).items()}
    months["COMBINED"] = study._group_summary(rows)
    summary = {"monthly": months, "event_count": len(rows), "primary_decision": "INSUFFICIENT_EVIDENCE",
               "prototype_decision": "NOT_READY_FOR_OOS", "tod_effect_label": "INSUFFICIENT",
               "family": study._family_results(rows), "bootstrap": {"rv_30s_ticks|Q5_Q1|2000": {}},
               "permutation": {"rv_30s_ticks|Q5_Q1|2000": {}}, "overlap": result["overlap"], "gates": {},
               "decision_checks": {"control_survival": {
                   key: {"classification": "INSUFFICIENT", "adequate_strata": 0, "positive_strata": 0}
                   for key in ("trend_velocity", "depth", "activity", "abnormal_state")},
                   "daily_q5_q1_2s": {"positive_dates": 0, "negative_dates": 0, "flat_dates": 0,
                                      "insufficient_dates": 2, "median_daily_effect": None,
                                      "p25_daily_effect": None, "p75_daily_effect": None},
                   "session_q5_q1": {}, "mfe_mae_support": False, "barrier_support": False,
                   "q5_mfe_5s_mean": None, "q1_mfe_5s_mean": None,
                   "q5_mae_5s_magnitude_mean": None, "q1_mae_5s_magnitude_mean": None}}
    report = study._render_report(summary, rows=rows, dates=("2025-12-01", "2026-01-02"),
                                  excluded=(), config_sha=study.EXPECTED_CONFIG_SHA256)
    assert "Descriptive regime diagnostic only" in report
    assert "within each independently defined control tercile" in report
    assert "Displayed depth | INSUFFICIENT" in report
    assert "## Predeclared volatility interactions" in report
