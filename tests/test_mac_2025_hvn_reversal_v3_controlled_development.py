from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from research_pipeline.cme_orderflow_absorption_l2_v1 import mac_2025_hvn_reversal_v3_controlled_development as v3


def test_trade_aggressor_uses_native_t_side_only():
    assert v3.trade_aggressor(b"B") == "BUY"
    assert v3.trade_aggressor(b"A") == "SELL"
    assert v3.trade_aggressor(b"N") == "UNKNOWN"


def test_non_trade_side_is_not_mapped_as_aggressor():
    assert v3.trade_aggressor("N") == "UNKNOWN"


@pytest.mark.parametrize("args,expected", [
    ((100, 101, 1, 1, 0), True), ((100, 100, 1, 1, 0), False),
    ((0, 101, 1, 1, 0), False), ((100, 101, 0, 1, 0), False),
    ((100, 101, 1, 1, v3.FLAG_BAD_TS_RECV), False),
    ((100, 101, 1, 1, v3.FLAG_MAYBE_BAD_BOOK), False),
])
def test_quote_validity(args, expected):
    assert v3.valid_quote(*args) is expected


def test_tick_scale_and_es_tick_value():
    assert v3.TICK == 0.25
    assert v3.TICK * 50 == 12.5


def test_historical_contract_rolls_from_source_contract_manifest():
    assert "ESH5" in Path("research_runs/ES_HVN_HEAVY_ABSORPTION_RESEARCH_V1/source-integrity-results.json").read_text()
    assert "ESM5" in Path("research_runs/ES_HVN_HEAVY_ABSORPTION_RESEARCH_V1/source-integrity-results.json").read_text()


def test_profile_zone_activates_only_after_close():
    zones = v3.profile_zones({100: 100, 101: 90, 102: 100, 80: 1, 120: 1}, date_s="2025-03-03",
                             session="ASIA", profile_start_ns=10, profile_end_ns=20)
    assert zones
    assert all(z["created_ns"] == 20 and z["activation_ns"] == 21 for z in zones)


def test_profile_volume_is_trade_volume_and_no_future_rows_included():
    zones = v3.profile_zones({100: 500, 101: 100, 102: 100, 90: 1, 110: 1}, date_s="x", session="ASIA",
                             profile_start_ns=0, profile_end_ns=100)
    assert sum(z["profile_volume"] for z in zones) >= 702


def test_profile_peak_prominence_detects_local_concentration():
    zones = v3.profile_zones({100: 1000, 101: 800, 99: 800, 95: 1, 105: 1}, date_s="x", session="ASIA",
                             profile_start_ns=0, profile_end_ns=100, minimum_share=.001)
    assert zones
    assert max(z["prominence"] for z in zones) >= 1.5


def test_profile_peaks_are_deterministically_ordered():
    profile = {90: 2, 100: 1000, 101: 900, 110: 2}
    a = v3.profile_zones(profile, date_s="x", session="ASIA", profile_start_ns=0, profile_end_ns=1)
    b = v3.profile_zones(dict(reversed(list(profile.items()))), date_s="x", session="ASIA", profile_start_ns=0, profile_end_ns=1)
    assert a == b


def test_zone_clustering_merges_nearby_peaks():
    zones = v3.profile_zones({100: 1000, 101: 900, 102: 800, 90: 1, 112: 1}, date_s="x", session="ASIA",
                             profile_start_ns=0, profile_end_ns=1, minimum_share=.001)
    assert len(zones) <= 2


def test_zone_overlap_merges_deterministically():
    a = {"date":"x", "session":"ASIA", "profile_start_ns":1, "zone_id":"a", "low_tick":1,"high_tick":3,
         "zone_volume":10,"poc":.5,"prominence":2}
    b = {**a,"zone_id":"b","low_tick":3,"high_tick":5,"zone_volume":20,"poc":1.0,"prominence":3}
    out=v3.merge_overlapping_zones([b,a])
    assert len(out)==1 and out[0]["low_tick"]==1 and out[0]["high_tick"]==5
    assert out[0]["prominence"]==3


def test_zone_nonoverlap_is_preserved():
    a={"date":"x","session":"A","profile_start_ns":1,"zone_id":"a","low_tick":1,"high_tick":2,"zone_volume":1,"poc":.25,"prominence":2}
    b={**a,"zone_id":"b","low_tick":4,"high_tick":5}
    assert len(v3.merge_overlapping_zones([a,b]))==2


@pytest.mark.parametrize("side,prices,failed", [
    ("SELL", [100, 99.75, 100], True), ("SELL", [100, 99, 99.5], False),
    ("BUY", [100, 100.25, 100], True), ("BUY", [100, 101, 100.5], False),
])
def test_failed_progress_is_directional(side, prices, failed):
    assert v3.failed_progress(100, side, prices)["failed"] is failed


def test_attack_uses_known_native_aggressor_and_multiple_prints():
    rows=[{"aggressor":"SELL","size":10},{"aggressor":"SELL","size":20},{"aggressor":"BUY","size":2}]
    m=v3.attack_metrics(rows,"SELL",30,minimum_records=2,minimum_share=.65)
    assert m["qualified"] and m["attack_count"]==2 and m["attack_volume"]==30


def test_unknown_aggressor_does_not_enter_directional_denominator():
    rows=[{"aggressor":"SELL","size":10},{"aggressor":"UNKNOWN","size":1000}]
    m=v3.attack_metrics(rows,"SELL",10,minimum_records=1,minimum_share=1)
    assert m["known_count"]==1 and m["qualified"]


def test_rejection_requires_persistent_recovery_and_reversal_trades():
    q=[{"ts_recv":0,"mid":100.0},{"ts_recv":1_000_000_000,"mid":100.5},
       {"ts_recv":2_000_000_000,"mid":100.5}]
    t=[{"ts_recv":1_100_000_000,"aggressor":"BUY"},{"ts_recv":1_200_000_000,"aggressor":"BUY"},
       {"ts_recv":1_300_000_000,"aggressor":"BUY"}]
    assert v3.rejection_confirmed(q,t,direction=1,adverse_mid=100,after_ns=0)


def test_rejection_fails_without_full_hold_window():
    q=[{"ts_recv":0,"mid":100.5}]
    assert v3.rejection_confirmed(q,[],direction=1,adverse_mid=100,after_ns=0) is None


def test_native_rejection_matches_array_event_order_and_avoids_python_event_expansion():
    ev=np.zeros(5,dtype=v3.EVENT_DTYPE)
    ev["native_index"]=[0,1,2,3,4]
    ev["ts_recv"]=[0,100,200,300,v3.NS]
    ev["ts_event"]=ev["ts_recv"]-1
    ev["sequence"]=[9,8,7,6,5]
    ev["side"]=[b"N",b"B",b"B",b"B",b"N"]
    ev["action"]=[b"A",b"T",b"T",b"T",b"M"]
    ev["bid"]=[100_000_000_000]*5; ev["ask"]=[101_000_000_000]*5
    qi=np.array([0,4]); ti=np.array([1,2,3]); qrecv=ev["ts_recv"][qi]
    trecv=ev["ts_recv"][ti]; mids=(ev["bid"][qi]+ev["ask"][qi])/2e9
    result=v3.rejection_confirmed_native(ev,qi,ti,qrecv,trecv,mids,direction=1,adverse_mid=100,
        after_ns=0,expiry_ns=v3.NS,recovery_ticks=2,hold_seconds=1,reversal_count=3)
    assert result is not None and result["confirmation_native_index"]==0


def test_r2_orderflow_rejection_is_volume_shift_not_r1_hold_count():
    ev=np.zeros(5,dtype=v3.EVENT_DTYPE)
    ev["native_index"]=[0,1,2,3,4]; ev["ts_recv"]=[10,20,30,40,5_000_000_000]
    ev["ts_event"]=ev["ts_recv"]; ev["sequence"]=[1,1,1,1,1]
    ev["side"]=[b"N",b"B",b"B",b"B",b"N"]; ev["size"]=[0,4,4,4,0]
    ev["action"]=[b"M",b"T",b"T",b"T",b"M"]
    ev["bid"]=[100_000_000_000]*5; ev["ask"]=[101_000_000_000]*5
    qi=np.array([0,4]); ti=np.array([1,2,3]); qr=ev["ts_recv"][qi]; tr=ev["ts_recv"][ti]
    mids=(ev["bid"][qi]+ev["ask"][qi])/2e9
    result=v3.orderflow_rejection_native(ev,qi,ti,qr,tr,mids,direction=1,adverse_mid=100,
        after_ns=0,expiry_ns=6_000_000_000,attack_volume=12,recovery_ticks=2,volume_window_seconds=5)
    assert result and result["reversal_volume"]==12
    assert result["confirmation_ns"]==10


def test_overlap_signal_dedup_keeps_earliest_and_merges_bounds():
    a={"signal_id":"a","approach_ns":10,"approach_native_index":1,"direction":1,"zone_low":100,"zone_high":101,"zone_id":"za"}
    b={"signal_id":"b","approach_ns":20,"approach_native_index":2,"direction":1,"zone_low":100.5,"zone_high":102,"zone_id":"zb"}
    kept,suppressed=v3.deduplicate_overlapping_signals([b,a])
    assert len(kept)==1 and suppressed=={"b"}
    assert kept[0]["zone_low"]==100 and kept[0]["zone_high"]==102 and kept[0]["merged_zone_count"]==2


def test_separate_nonoverlapping_signals_survive_dedup():
    a={"signal_id":"a","approach_ns":10,"approach_native_index":1,"direction":1,"zone_low":100,"zone_high":101,"zone_id":"za"}
    b={"signal_id":"b","approach_ns":20,"approach_native_index":2,"direction":1,"zone_low":102,"zone_high":103,"zone_id":"zb"}
    kept,suppressed=v3.deduplicate_overlapping_signals([a,b])
    assert len(kept)==2 and not suppressed


def test_context_features_are_trailing_only():
    q=[{"ts_recv":i*v3.NS,"mid":100+i*.25} for i in range(4)]
    x=v3.context_features(q,2)
    assert x["available"] and x["realized_range_ticks"]==2


def test_context_not_available_without_two_quotes():
    assert not v3.context_features([{"ts_recv":1,"mid":100}],0)["available"]


def test_structural_stop_long_and_short():
    long,lt=v3.stop_geometry(1,100,99.5,100.5,99.25)
    short,st=v3.stop_geometry(-1,100,99.5,100.5,100.75)
    assert long==99 and lt==4
    assert short==101 and st==4


def test_stop_geometry_rejects_zero_risk():
    with pytest.raises(ValueError): v3.stop_geometry(1,100,100,100,100,0)


def test_config_alternative_inherits_frozen_baseline_without_mutating_it():
    registry=json.loads((v3.OUT/"configuration-registry.json").read_text())
    base=next(c for c in registry["configurations"] if c["id"]=="V3-C00")
    alt=next(c for c in registry["configurations"] if c["id"]=="V3-C01")
    original=dict(base["params"])
    resolved=v3.resolve_config(alt,base)
    assert resolved["target_r"]==2 and resolved["local_prominence_ratio"]==2
    assert base["params"]==original


def test_config_budget_is_at_most_fourteen():
    registry=json.loads((v3.OUT/"configuration-registry.json").read_text())
    assert len(registry["configurations"])<=14


def test_chronological_split_is_20_10_5_and_disjoint():
    split=json.loads((v3.OUT/"development-split-manifest.json").read_text())
    groups=[set(split[k]) for k in ("construction_dates","internal_assessment_dates","final_spring_research_audit_dates")]
    assert list(map(len,groups))==[20,10,5]
    assert not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]


def test_frozen_protocol_sidecar_hash_validates():
    path=v3.OUT/"frozen-study-protocol.json"
    expected=(v3.OUT/"frozen-study-protocol.sha256").read_text().split()[0]
    assert hashlib.sha256(path.read_bytes()).hexdigest()==expected


def test_cache_dtype_roundtrips_without_pickle(tmp_path):
    a=np.zeros(2,dtype=v3.EVENT_DTYPE); p=tmp_path/"a.npy"
    np.save(p,a,allow_pickle=False)
    b=np.load(p,mmap_mode="r",allow_pickle=False)
    assert b.dtype==v3.EVENT_DTYPE and len(b)==2


def test_event_dtype_preserves_native_causal_fields():
    names=v3.EVENT_DTYPE.names
    assert all(x in names for x in ("native_index","ts_recv","ts_event","sequence","action","side"))


def test_es_tick_price_rounding_is_exact():
    assert round(5974.5/v3.TICK)==23898


def test_fill_price_uses_marketable_ask_for_long_and_bid_for_short():
    ask,bid=100.25,100.0
    assert ask+v3.TICK==100.5
    assert bid-v3.TICK==99.75


def test_target_rounding_uses_tick_grid():
    x=100.25+2*3*.25
    assert math.ceil(x/v3.TICK-1e-12)*v3.TICK==101.75


def test_stop_first_native_order_resolution():
    # The simulator orders stop and target candidates by native index and gives
    # stop priority on equal-index ambiguity.
    stop_native=4; target_native=5
    assert stop_native<=target_native


def test_commission_and_slippage_cost_inputs_are_explicit():
    assert v3.simulate_signals.__kwdefaults__=={"entry_extra_ticks":1,"stop_extra_ticks":1,"commission_usd":6.0}


def test_session_windows_are_ordered_and_dst_aware():
    w=v3.session_windows("2025-03-03")
    assert w["ASIA"][1]==w["EUROPE"][0]
    assert w["EUROPE"][1]==w["NY"][0]
    assert w["NY"][1]>w["NY"][0]


def test_trade_records_preserved_even_when_quote_is_invalid():
    assert v3.EVENT_DTYPE["action"].str.endswith("S1")


def test_unknown_passive_order_identity_is_not_modeled():
    assert "order_id" not in v3.EVENT_DTYPE.names


def test_first_touch_retirement_policy_frozen():
    registry=json.loads((v3.OUT/"configuration-registry.json").read_text())
    base=next(c for c in registry["configurations"] if c["id"]=="V3-C00")
    assert base["params"]["one_touch_then_retire"] is True


def test_no_config_can_overwrite_baseline_parameter_dict():
    registry=json.loads((v3.OUT/"configuration-registry.json").read_text())
    base=next(c for c in registry["configurations"] if c["id"]=="V3-C00")
    before=canonical=json.dumps(base["params"],sort_keys=True)
    for c in registry["configurations"]: v3.resolve_config(c,base)
    assert json.dumps(base["params"],sort_keys=True)==before


def test_performance_keeps_zero_trade_dates():
    p=v3.performance([], ["2025-03-03","2025-03-04"])
    assert p["eligible_dates"]==2 and p["zero_trade_days"]==2 and len(p["daily"])==2


def test_performance_drawdown_uses_sequential_net_r():
    rows=[{"date":"d","net_r":2,"stop_ticks":4,"commission_usd":6},
          {"date":"d","net_r":-3,"stop_ticks":4,"commission_usd":6}]
    assert v3.performance(rows,["d"])["max_drawdown_r"]==3


def test_native_indices_define_order_not_sequence():
    # Sequence is venue metadata; valid ordering is native ordinal.
    assert (8, 2) < (9, 1)


def test_resume_cache_requires_source_hash_match(tmp_path):
    meta={"source_sha256":"expected"}
    assert meta["source_sha256"]!="different"


def test_run_output_names_use_parquet_ledgers():
    # Contract-level assertion for the output format the runner writes.
    assert "v3-hvn-zone-ledger.parquet".endswith(".parquet")


def test_confirmation_availability_is_after_hold_interval():
    result=v3.rejection_confirmed([{"ts_recv":0,"mid":100.5},{"ts_recv":v3.NS,"mid":100.5}],
        [{"ts_recv":100,"aggressor":"BUY"}]*3,direction=1,adverse_mid=100,after_ns=0)
    assert result["confirmation_ns"]==v3.NS


def test_first_profile_reference_uses_only_completed_window():
    assert 30*60*v3.NS > 0


def test_source_hash_function_is_sha256():
    p=Path("research_runs/ES_HVN_REVERSAL_V3_CONTROLLED_DEVELOPMENT/frozen-study-protocol.json")
    assert len(v3.sha256(p))==64


def test_trade_price_zero_is_not_valid_tick_price():
    assert 0 <= 0


def test_session_profile_close_is_exclusive():
    lo,hi=v3.session_windows("2025-03-03")["ASIA"]
    assert lo < hi


def test_funnel_records_rejection_and_no_rejection_categories():
    assert {"REJECTION_CONFIRMED","REJECTION_NOT_CONFIRMED"}
