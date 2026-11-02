from __future__ import annotations

from research_pipeline.cme_orderflow_absorption_l2_v1 import mac_2025_hvn_native_mbp10_absorption_mechanism_v1 as base
from research_pipeline.cme_orderflow_absorption_l2_v1.mac_2025_hvn_causal_absorption_detection_research_v1 import (
    _depth_components,
    _analyse,
    _new_state,
    _origin,
    directional_price_progress,
    source_origin_match,
    source_touch_match,
    snapshot_is_valid,
    weak_impact_marker,
)


def _episode(*, heavy=False, heavy_ns=None, direction=1):
    return base.Episode(
        episode_id="fixture", date="2025-03-03", zone_id="zone", t0=100,
        direction=direction, low=100.0, high=100.25, poc=100.0, origin="fixture",
        v1_status="fixture", v1_heavy=heavy, v1_heavy_size=50 if heavy else None,
        v1_threshold=25, v1_heavy_ns=heavy_ns, v1_heavy_price=100.0 if heavy else None,
        touch_price=100.0, approach="fixture",
    )


def test_heavy_print_and_first_touch_origins_are_separate_and_deterministic():
    heavy = _episode(heavy=True, heavy_ns=200)
    no_heavy = _episode()
    assert _origin(heavy) == (200, "FIRST_SOURCE_SUPPORTED_HEAVY_PRINT")
    assert _origin(no_heavy) == (100, "FIRST_TOUCH_NO_HEAVY_PRINT")
    assert _new_state(heavy)["origin_ns"] == 200


def test_v1_receive_timestamp_maps_to_exact_native_trade_identity():
    ep = _episode(heavy=True, heavy_ns=200)
    # The native event may have an earlier ts_event; source identity is V1 ts_recv.
    assert source_origin_match(ep, 200, "T", 100_000_000_000, 50, "A")
    assert not source_origin_match(ep, 199, "T", 100_000_000_000, 50, "A")
    assert not source_origin_match(ep, 200, "T", 100_000_000_000, 49, "A")
    assert not source_origin_match(ep, 200, "A", 100_000_000_000, 50, "A")


def test_first_touch_maps_by_exact_receive_time_and_touch_price_without_trade_requirement():
    ep = _episode()
    row = {"action": b"A", "price": 100_000_000_000, "flags": 0}
    for i in range(10):
        row[f"bid_px_{i:02d}"] = (100_000_000_000 - i * 250_000_000) if i < 2 else 0
        row[f"bid_sz_{i:02d}"] = 10 if i < 2 else 0
        row[f"ask_px_{i:02d}"] = (100_250_000_000 + i * 250_000_000) if i < 2 else 0
        row[f"ask_sz_{i:02d}"] = 10 if i < 2 else 0
    assert source_touch_match(ep, row, 100)
    assert not source_touch_match(ep, row, 101)
    row["bid_px_00"] = 100_250_000_000
    assert not source_touch_match(ep, row, 100)


def test_zone_only_first_touch_can_use_exact_timestamp_when_v1_has_no_touch_price():
    ep = _episode()
    ep.origin = "ZONE_FIRST_TOUCH_NO_DIRECTION"
    ep.touch_price = None
    row = {"action": b"M"}
    assert source_touch_match(ep, row, 100)
    assert not source_touch_match(ep, row, 101)


def test_preentry_analysis_excludes_checkpoints_after_v1_entry():
    ep = _episode()
    ep.v1 = {"TE_ns": "150"}
    base_row = {"attack_volume_contracts": 1, "max_attack_direction_progress_ticks": 0.0,
        "attack_direction_price_impact_ticks_per_attack_contract": 0.0, "defended_price_depth": 1,
        "depth_reductions_contracts": 0, "depth_increases_contracts": 0,
        "directional_persistence_ratio": 1.0}
    rows = [
        {**base_row, "episode_id": ep.episode_id, "date": ep.date, "offset_ms": 250, "checkpoint_ns": 125,
         "snapshot_valid": True, "v1_confirmed": True, "rr1_class": "WINNER"},
        {**base_row, "episode_id": ep.episode_id, "date": ep.date, "offset_ms": 500, "checkpoint_ns": 175,
         "snapshot_valid": True, "v1_confirmed": True, "rr1_class": "WINNER"},
    ]
    result = _analyse(rows, [], [ep])
    assert result["fixed_snapshot_coverage"]["250"]["v1_entries"]["valid"] == 1
    assert result["fixed_snapshot_coverage"]["500"]["v1_entries"]["valid"] == 0
    assert result["winner_loser_by_offset"]["500"]["attack_volume_contracts"]["WINNER"]["valid_n"] == 0


def test_snapshot_requires_valid_prior_quote_freshness_and_continuous_source():
    assert snapshot_is_valid(True, 0, 0)
    assert snapshot_is_valid(True, 2_000_000_000, 0)
    assert not snapshot_is_valid(True, 2_000_000_001, 0)
    assert not snapshot_is_valid(True, -1, 0)  # future observation is lookahead
    assert not snapshot_is_valid(False, 0, 0)
    assert not snapshot_is_valid(True, 0, 1)
    assert not snapshot_is_valid(True, None, 0)


def test_attack_direction_price_progress_uses_es_tick_units():
    # Selling into long support: a lower mid is progress with the attack.
    assert directional_price_progress(99.75, 100.0, -1) == 1.0
    # Buying into short resistance: a higher mid is progress with the attack.
    assert directional_price_progress(100.25, 100.0, 1) == 1.0


def test_weak_impact_marker_requires_repeated_attack_execution_records():
    assert not weak_impact_marker(100, 25, 0.0, 1)
    assert weak_impact_marker(100, 25, 1.0, 2)
    assert not weak_impact_marker(100, 25, 1.01, 2)


def test_depth_response_uses_opposing_side_and_original_zone_prices():
    ep = _episode(direction=1)
    book = {
        "bids": [(100_000_000_000, 12), (99_750_000_000, 7)],
        "asks": [(100_250_000_000, 5), (100_500_000_000, 9)],
        "bid": 100.0, "ask": 100.25,
    }
    support = _depth_components(book, ep)
    assert support["defended_price_depth"] == 12
    assert support["top1_opposing_depth"] == 12
    assert support["top5_opposing_depth"] == 19
    resistance = _depth_components(book, _episode(direction=-1))
    assert resistance["defended_price_depth"] == 5
    assert resistance["top1_opposing_depth"] == 5
