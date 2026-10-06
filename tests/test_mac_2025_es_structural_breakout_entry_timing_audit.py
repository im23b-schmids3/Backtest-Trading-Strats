"""Offline deterministic checks for the frozen structural entry-timing audit."""
from __future__ import annotations

import gzip
import json

import pytest

from research_pipeline.cme_orderflow_absorption_l2_v1 import mac_2025_es_structural_breakout_entry_timing_audit as audit


@pytest.mark.parametrize(("value", "expected"), [(1, "1_TICK"), (2, "2_TICKS"),
    (3, "3_TO_4_TICKS"), (4, "3_TO_4_TICKS"), (5, "5_TO_8_TICKS"), (8, "5_TO_8_TICKS"),
    (9, "9_PLUS_TICKS")])
def test_overshoot_buckets(value, expected):
    assert audit.overshoot_bucket(value) == expected


@pytest.mark.parametrize(("value", "expected"), [(-1, "LE_ZERO"), (0, "LE_ZERO"),
    (1, "ZERO_TO_1"), (2, "1_TO_2"), (4, "2_TO_4"), (5, "GT_4")])
def test_pre_move_buckets(value, expected):
    assert audit.pre_move_bucket(value) == expected


@pytest.mark.parametrize(("value", "expected"), [(2, "LE_2MS"), (5, "2_TO_5MS"),
    (10, "5_TO_10MS"), (25, "10_TO_25MS"), (50, "25_TO_50MS"), (51, "GT_50MS")])
def test_latency_buckets(value, expected):
    assert audit.latency_bucket(value) == expected


def test_invalid_overshoot_and_early_quote():
    with pytest.raises(audit.EntryAuditError): audit.overshoot_bucket(0)
    with pytest.raises(audit.EntryAuditError): audit.latency_bucket(1.9)


def test_adjacent_latency_merge_uses_only_counts():
    events = [{"latency_bucket": x} for x in
        ["LE_2MS"]*3+["2_TO_5MS"]*6+["5_TO_10MS"]*2+["10_TO_25MS"]*2]
    groups = audit.merge_latency_buckets(events)
    assert groups == [{"labels": ["LE_2MS", "2_TO_5MS", "5_TO_10MS",
                                  "10_TO_25MS", "25_TO_50MS", "GT_50MS"], "n": 13}]


def test_spearman_ties_and_insufficient():
    assert audit.spearman([1, 2, 3], [3, 2, 1])["rho"] == pytest.approx(-1)
    assert audit.spearman([1, 1, 1], [1, 2, 3])["status"] == "CONSTANT_INPUT"
    assert audit.spearman([1, 2], [3, 4])["status"] == "INSUFFICIENT_SAMPLE"


def test_checkpoint_hash_invalidation(tmp_path):
    path = tmp_path / "day.json.gz"
    row = {"version": audit.CHECKPOINT_VERSION, "status": "DATE_COMPLETE", "date": "2025-03-03",
        "source_sha256": "source", "tape_sha256": "tape", "parent_manifest_sha256": "parent",
        "config_sha256": audit.CONFIG_SHA256, "payload": {"events": []}}
    with gzip.open(path, "wt", encoding="utf-8") as f: json.dump(row, f)
    assert audit._read_checkpoint(path, "2025-03-03", "source", "tape", "parent") == row
    assert audit._read_checkpoint(path, "2025-03-03", "changed", "tape", "parent") is None
    assert audit._read_checkpoint(path, "2025-03-03", "source", "tape", "changed") is None
    row["config_sha256"] = "changed"
    with gzip.open(path, "wt", encoding="utf-8") as f: json.dump(row, f)
    assert audit._read_checkpoint(path, "2025-03-03", "source", "tape", "parent") is None


def test_week_and_split_governance():
    assert audit._week("2025-03-03") == "2025-W10"
    events = [{"period": "SPRING_2025", "direction": "LONG"},
              {"period": "OCTOBER_2025", "direction": "SHORT"}]
    assert len(audit._split(events, "SPRING_2025")) == 1
    assert len(audit._split(events, "SHORT")) == 1
    assert audit.CONFIG["no_l2_filter"] is True
    assert audit.CONFIG["no_optimization"] is True


def test_exact_loss_identity_and_direction_normalization():
    # Same directional price path after multiplying LONG/SHORT by sign.
    for sign in (1, -1):
        trade = 5000.0
        entry_mid = trade + sign*0.25
        entry_quote = entry_mid + sign*0.25
        fill = entry_quote + sign*0.25
        exit_side = trade + sign*0.75
        pre_entry = sign*(entry_quote-trade)/audit.TICK
        assert pre_entry == 2
        quote = sign*(exit_side-entry_quote)/audit.TICK
        actual = sign*(exit_side-fill)/audit.TICK
        assert quote == 1 and actual == 0
        assert actual == quote-1


def test_aggregate_loss_reconciles_horizon_shift():
    path = {"raw": 4.0, "quote": 1.0, "actual": 0.0,
            "horizon_shift": 0.5, "pre_entry_price_move": 1.5,
            "bid_ask_effect": -2.0}
    event = {"period": "SPRING_2025", "direction": "LONG",
             "paths": {str(h): path for h in audit.HORIZONS_MS}}
    block = audit._edge_loss([event])["ALL"]["10000"]
    assert block["raw_signal_edge"] == 4
    assert block["pre_entry_price_movement_effect"] == -1.5
    assert block["horizon_alignment_effect"] == .5
    assert block["bid_ask_execution_effect"] == -2
    assert block["adverse_entry_tick_effect"] == -1
    assert block["actual_fill_edge"] == 0


def test_period_classification_structural_first():
    period = {"SPRING_2025": {"raw": {"10000": 4.4}},
              "OCTOBER_2025": {"raw": {"10000": -.4}}}
    edge = {"SPRING_2025": {"10000": {"raw_to_executable_loss": -3.3}},
            "OCTOBER_2025": {"10000": {"raw_to_executable_loss": -1.75,
                                          "raw_to_actual_fill_loss": -2.75}}}
    result = audit._classification([], period, edge, {})
    assert result["primary_decision"] == "BOTH_STRUCTURE_AND_ENTRY_LIMITING"
    assert result["october_execution_worse_than_spring"] is False
