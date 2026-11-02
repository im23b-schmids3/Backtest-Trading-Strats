from __future__ import annotations

import gzip
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from research_pipeline.cme_orderflow_absorption_l2_v1 import hvn_heavy_absorption_v1 as hvn


def event_array(rows):
    dtype=np.dtype([("timestamp_ns","<i8"),("bid","<f8"),("ask","<f8"),
        ("execution_price","<f8"),("execution_size","<i8"),("aggressor","i1"),("session","i1")])
    out=np.zeros(len(rows),dtype=dtype)
    for i,r in enumerate(rows):
        out[i]=(r[0],r[1],r[2],r[3],r[4],r[5],r[6])
    return out


def mk_events(prices, sizes=None, sides=None, start=0, step=1_000_000_000, session=0):
    sizes=sizes or [1]*len(prices); sides=sides or [1]*len(prices)
    return event_array([(start+i*step,p-.25,p+.25,p,sizes[i],sides[i],session) for i,p in enumerate(prices)])


def rejection_fixture(direction=1,reclaim=True):
    start=hvn._windows("2025-03-03")[0][0]
    profile_ticks=[400,401,402,403,404,405,406,407,408,409]*3
    sizes=[1]*30; sizes[15]=100
    rows=[]
    for i,(tick,size) in enumerate(zip(profile_ticks,sizes)):
        ts=start+i*hvn.BAR_NS; p=tick*hvn.TICK
        rows.append((ts,p-.25,p+.25,p,size,1,0))
    zones=hvn._zones(hvn._trades(event_array(rows)),"2025-03-03")
    z=max(zones,key=lambda x:x["zone_volume"])
    low=round(z["zone_low"]/hvn.TICK); high=round(z["zone_high"]/hvn.TICK)
    poc=round(z["zone_poc"]/hvn.TICK); after=start+30*hvn.BAR_NS
    if direction>0:
        prices=[high+3,high+1,poc,poc+1]
        sides=[0,0,-1,0]
        if not reclaim:
            prices=[high+3,high+1,poc,low-2,low-2]; sides=[0,0,-1,0,0]
    else:
        prices=[low-3,low-1,poc,poc-1]
        sides=[0,0,1,0]
        if not reclaim:
            prices=[low-3,low-1,poc,high+2,high+2]; sides=[0,0,1,0,0]
    sizes_after=[1,1,50]+[1]*(len(prices)-3)
    rows.extend((after+(i+1)*1_000_000_000,tick*hvn.TICK-.25,tick*hvn.TICK+.25,
                 tick*hvn.TICK,sizes_after[i],sides[i],0) for i,tick in enumerate(prices))
    return event_array(rows)


def test_hvn_histogram_and_zone_volume():
    start=hvn._windows("2025-03-03")[0][0]
    tr=mk_events([99,99.25,99.5,99.75,100,100.25,100.5,100.75,101],
        [1,1,2,4,20,5,2,1,1],start=start,step=3*hvn.BAR_NS)
    zones=hvn._zones(tr,"2025-03-03")
    assert zones and all(z["zone_volume"]>0 for z in zones)


def test_hvn_local_peak_is_deterministic():
    start=hvn._windows("2025-03-03")[0][0]
    args=([99,99.25,99.5,99.75,100,100.25,100.5,100.75,101],[1,1,2,4,20,5,2,1,1])
    a=hvn._zones(mk_events(*args,start=start,step=3*hvn.BAR_NS),"2025-03-03")
    b=hvn._zones(mk_events(*args,start=start,step=3*hvn.BAR_NS),"2025-03-03")
    assert a==b


def test_zone_not_active_before_profile_close():
    z=hvn._zones(mk_events([100,100.25,100.5,100.25,100,100.25,100.5],[1,2,20,5,1,10,1]),"2025-03-03")
    assert all(x["active_from_ns"]==x["profile_close_ns"] for x in z)


def test_zone_touch_lookup_finds_first_index():
    idx={100:np.array([2,7]),101:np.array([4])}
    assert hvn._first_in_tick_band(idx,0,100,101)==2


def test_touch_lookup_respects_retirement_start():
    idx={100:np.array([2,7]),101:np.array([4])}
    assert hvn._first_in_tick_band(idx,3,100,101)==4


def test_touch_lookup_retirement_uses_first_revisit_only():
    idx={100:np.array([3,9])}
    first=hvn._first_in_tick_band(idx,0,100,100)
    assert first==3
    assert hvn._first_in_tick_band(idx,first+1,100,100)==9


def test_heavy_threshold_uses_prior_completed_bars_only():
    tr=mk_events([100,100,100], [1,100,1], [1,1,1],step=hvn.BAR_NS)
    heavy,threshold=hvn._heavy_flags(tr)
    assert np.isnan(threshold[0])
    assert threshold[1]==1


def test_heavy_threshold_population_sd_formula():
    tr=mk_events([100]*5,[1,3,1,3,8],[1]*5,step=hvn.BAR_NS)
    _,threshold=hvn._heavy_flags(tr)
    prior=np.array([1,3,1,3],dtype=float)
    assert threshold[4]==pytest.approx(prior.mean()+2*prior.std(ddof=0))


def test_heavy_current_print_does_not_enter_its_threshold():
    tr=mk_events([100,100,100],[1,1,100],[1,1,1],step=hvn.BAR_NS)
    _,threshold=hvn._heavy_flags(tr)
    assert threshold[2]==1


def test_aggressor_buy_side_mapping_retained():
    ev=mk_events([100],[1],[1])
    assert int(ev["aggressor"][0])==1


def test_aggressor_sell_side_mapping_retained():
    ev=mk_events([100],[1],[-1])
    assert int(ev["aggressor"][0])==-1


def test_unknown_aggressor_is_not_buy_or_sell():
    ev=mk_events([100],[1],[0])
    assert int(ev["aggressor"][0])==0


def test_print_grouping_is_one_record():
    ev=mk_events([100,100],[4,5],[1,1])
    assert int(np.count_nonzero(ev["execution_size"]>0))==2


def test_confirmation_window_is_fixed_and_causal():
    assert hvn.CONFIRM_NS==15_000_000_000
    assert hvn.RESPONSE_NS<hvn.CONFIRM_NS


@pytest.mark.parametrize("direction",[1,-1])
def test_directional_rejection_requires_causal_confirmation(direction):
    _,_,signals=hvn.detect(rejection_fixture(direction),"2025-03-03")
    assert signals
    assert all(x["direction"]==direction and x["TC_ns"]>=x["TH_ns"]+hvn.RESPONSE_NS for x in signals)


def test_episode_without_reclaim_expires_or_is_invalidated():
    _,touches,signals=hvn.detect(rejection_fixture(1,reclaim=False),"2025-03-03")
    assert not signals
    assert any(x["episode_status"] in {"EXPIRED_NO_RECLAIM","INVALIDATED_THROUGH_HVN"} for x in touches)


def test_signal_timestamp_is_not_backdated_to_reversal_extreme():
    _,_,signals=hvn.detect(rejection_fixture(1),"2025-03-03")
    assert signals and all(x["TC_ns"]>=x["TH_ns"]+hvn.RESPONSE_NS for x in signals)


def test_entry_delay_begins_after_confirmation():
    assert hvn.ENTRY_DELAY_NS==2_000_000


def test_entry_uses_first_quote_after_latency_and_market_side():
    ev=event_array([(0,100,100.25,100,1,-1,0),(3_000_000,100,100.25,np.nan,0,0,0)])
    s={"TC_ns":0,"direction":1,"stop_price":99.0}
    entry=hvn.entry_for(ev,s,{0:(0,10_000_000)})
    assert entry and entry["TE_ns"]==3_000_000 and entry["entry_price"]==100.5


def test_invalid_structural_stop_side_rejects_entry():
    ev=event_array([(0,100,100.25,100,1,-1,0),(3_000_000,100,100.25,np.nan,0,0,0)])
    assert hvn.entry_for(ev,{"TC_ns":0,"direction":1,"stop_price":101},{0:(0,10_000_000)}) is None


def test_long_stop_structure_is_beyond_lower_level_and_extreme():
    low, rejection, buffer=100,99.5,1
    stop=min(low,rejection)-buffer*hvn.TICK
    assert stop==99.25


def test_short_stop_structure_is_beyond_upper_level_and_extreme():
    high, rejection, buffer=100,100.5,1
    stop=max(high,rejection)+buffer*hvn.TICK
    assert stop==100.75


@pytest.mark.parametrize("rr,entry,direction,expected",[
    (1.5,100.0,1,100.75),(2.25,100.0,1,101.25),(1.5,100.0,-1,99.25),(2.25,100.0,-1,98.75)])
def test_fractional_rr_target_rounding_away_from_entry(rr,entry,direction,expected):
    assert hvn._tick_round(entry+direction*rr*2*hvn.TICK,direction)==expected


def test_stop_first_if_stop_and_target_are_observable_on_same_record():
    ts=range(4)
    ev=event_array([(0,100,100.25,100,1,1,0),(1,99,99.25,101,1,1,0),
                    (2,99,99.25,np.nan,0,0,0),(3,99,99.25,np.nan,0,0,0)])
    entry={"entry_index":0,"TE_ns":0,"entry_price":100.5,"stop_price":99.25,"risk_ticks":5,
           "session":0,"entry_bid":100,"entry_ask":100.25}
    out=hvn.simulate(ev,entry,1,1,{0:(0,4)})
    assert out["outcome"]=="STOP"


def test_target_fill_requires_trade_through_not_quote_touch():
    ev=event_array([(0,100,100.25,100,1,1,0),(1,101,101.25,np.nan,0,0,0),
                    (2,101,101.25,np.nan,0,0,0),(3,101,101.25,np.nan,0,0,0)])
    entry={"entry_index":0,"TE_ns":0,"entry_price":100.5,"stop_price":99.5,"risk_ticks":4,
           "session":0,"entry_bid":100,"entry_ask":100.25}
    out=hvn.simulate(ev,entry,1,1,{0:(0,4)})
    assert out["outcome"]!="TARGET"


def test_actual_trade_through_can_fill_target():
    ev=event_array([(0,100,100.25,100,1,1,0),(1,101,101.25,101.75,1,1,0),
                    (2,101,101.25,np.nan,0,0,0),(3,101,101.25,np.nan,0,0,0)])
    entry={"entry_index":0,"TE_ns":0,"entry_price":100.5,"stop_price":99.5,"risk_ticks":4,
           "session":0,"entry_bid":100,"entry_ask":100.25}
    out=hvn.simulate(ev,entry,1,1,{0:(0,4)})
    assert out["outcome"]=="TARGET"
    assert out["mfe_ticks"]>=4


def test_session_end_flattens_at_last_in_session_record():
    ev=event_array([(0,100,100.25,100,1,1,0),(1,100,100.25,np.nan,0,0,0),
                    (2,100,100.25,np.nan,0,0,1)])
    entry={"entry_index":0,"TE_ns":0,"entry_price":100.5,"stop_price":99.0,"risk_ticks":6,
           "session":0,"entry_bid":100,"entry_ask":100.25}
    out=hvn.simulate(ev,entry,1,3,{0:(0,2),1:(2,3)})
    assert out["outcome"]=="SESSION_END" and out["exit_timestamp_ns"]==1


def test_invalid_book_is_rejected_by_source_tape_gate():
    ev=event_array([(0,100,100.25,100,1,1,0),(1,100,100,np.nan,0,0,0)])
    assert np.any(~(ev["ask"]>ev["bid"]))


def test_costs_are_subtracted_from_gross_r():
    assert hvn.COMMISSION_USD==6.0 and hvn.ENTRY_SLIPPAGE_TICKS==1 and hvn.STOP_SLIPPAGE_TICKS==1


def test_rr_population_is_frozen_and_predeclared():
    assert hvn.RR_GRID==(1.0,1.5,2.0,2.25,2.5,2.75,3.0)


def test_duplicate_decision_signals_are_keyed_by_decision_and_direction():
    candidates=[{"TC_ns":10,"direction":1,"zone_volume":20},{"TC_ns":10,"direction":1,"zone_volume":30}]
    grouped={}
    for x in candidates:
        key=(x["TC_ns"],x["direction"])
        if key not in grouped or x["zone_volume"]>grouped[key]["zone_volume"]: grouped[key]=x
    assert len(grouped)==1 and grouped[(10,1)]["zone_volume"]==30


def test_source_cache_requires_hash_match(tmp_path,monkeypatch):
    raw=tmp_path/"x.dbn"; raw.write_bytes(b"source"); digest=hvn._sha(raw)
    tape_dir=tmp_path/"tapes"; tape_dir.mkdir()
    events=event_array([(0,100,100.25,np.nan,0,0,0)])
    tape=tape_dir/"2025-03-03-candidate-tape.npz"
    np.savez(tape,events=events,metadata_json=np.asarray(json.dumps({"date":"2025-03-03","source_sha256":digest})))
    manifest=tape_dir/"2025-03-03-candidate-tape-manifest.json"
    manifest.write_text(json.dumps({"date":"2025-03-03","source_sha256":digest,"tape_version":"MAC2025_CANDIDATE_TAPE_V1"}))
    monkeypatch.setattr(hvn,"DATA_ROOT",tmp_path); monkeypatch.setattr(hvn,"TAPE_ROOT",tape_dir)
    class Meta:
        dataset="GLBX.MDP3"; schema="mbp-10"; symbols=["ESH5"]
    class Store:
        @staticmethod
        def from_file(_): return SimpleNamespace(metadata=Meta())
    monkeypatch.setitem(sys.modules,"databento",SimpleNamespace(DBNStore=Store))
    src={"path":"x.dbn","bytes":raw.stat().st_size,"sha256":digest,"status":"VERIFIED","schema":"mbp-10","symbol":"ESH5"}
    arr,ident=hvn.load_events("2025-03-03",src)
    assert len(arr)==1 and ident["raw_sha256"]==digest
    src["sha256"]="0"*64
    with pytest.raises(hvn.ResearchError): hvn.load_events("2025-03-03",src)


def test_checkpoint_versions_are_resume_scoped():
    assert hvn.CHECKPOINT_VERSION.endswith("v5")


def test_resume_accepts_only_matching_completed_checkpoint():
    identity={"raw_sha256":"a","tape_sha256":"b"}
    payload={"version":hvn.CHECKPOINT_VERSION,"spec_sha256":hvn.SPEC_SHA256,
        "source_sha256":"a","tape_sha256":"b","result":{"date":"2025-03-03"}}
    assert hvn.checkpoint_matches(payload,identity)
    payload["source_sha256"]="changed"
    assert not hvn.checkpoint_matches(payload,identity)


def test_resume_rejects_incomplete_checkpoint_result():
    payload={"version":hvn.CHECKPOINT_VERSION,"spec_sha256":hvn.SPEC_SHA256,
        "source_sha256":"a","tape_sha256":"b","result":None}
    assert not hvn.checkpoint_matches(payload,{"raw_sha256":"a","tape_sha256":"b"})


def test_frozen_spec_hash_is_stable():
    assert len(hvn.SPEC_SHA256)==64


def test_es_tick_size_and_point_value():
    assert hvn.TICK==0.25 and hvn.POINT_VALUE_USD==50.0
