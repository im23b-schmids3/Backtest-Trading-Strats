"""Deterministic unit checks for the frozen ES value migration event study."""
from __future__ import annotations

import gzip
import json

import numpy as np
import pytest

from research_pipeline.cme_orderflow_absorption_l2_v1 import mac_2025_es_value_area_migration_event_study_v1 as study


def _tape(day="2025-03-03", times=None, prices=None):
    start,_=study._session(day)
    times=times or [start,start+1_000_000_000,start+301_000_000_000,start+302_000_000_000,
                    start+601_000_000_000,start+602_000_000_000]
    prices=prices or [100.,100.25,100.5,100.75,100.75,101.]
    dtype=[("timestamp_ns","i8"),("execution_size","i8"),("execution_price","f8"),
        ("aggressor","i1"),("bid","f8"),("ask","f8")]
    a=np.zeros(len(times),dtype=dtype);a["timestamp_ns"]=times;a["execution_size"]=1
    a["execution_price"]=prices;a["aggressor"]=np.resize([1,-1,1,1,1,-1],len(times))
    a["bid"]=np.asarray(prices)-.125;a["ask"]=np.asarray(prices)+.125
    return a


def _compact(times):
    dtype=[("ts","i8"),("mid","f8"),("bid5","f8"),("ask5","f8"),
           ("bid10","f8"),("ask10","f8"),("action","i1"),("side","i1"),
           ("size","i4"),("mlofi","f8"),("denom","f8")]
    x=np.zeros(len(times),dtype=dtype);x["ts"]=times;x["bid5"]=80;x["ask5"]=90
    x["bid10"]=100;x["ask10"]=110;x["denom"]=10;x["mlofi"]=1
    return x


def test_rth_and_fixed_bar_profile_uses_actual_trade_prices():
    day="2025-03-03";start,end=study._session(day)
    tape=_tape(day,[start,start+60_000_000_000,start+300_000_000_000], [100.,100.25,100.5])
    bars=study._make_bars(day,tape,_compact([start]))
    assert bars[0]["start_ns"]==start and bars[0]["close_ns"]==start+study.BAR_NS
    assert bars[0]["open"]==100 and bars[0]["high"]==100.25 and bars[0]["low"]==100
    assert bars[0]["poc"]==100 and bars[0]["vah"]==100.25 and bars[0]["val"]==100
    assert bars[0]["total_volume"]==2 and bars[0]["delta"]==0
    assert bars[1]["total_volume"]==1


def test_profile_and_value_area_ties_are_deterministic():
    p=study._profile70({400:7,401:7,399:1})
    assert p["poc"]==100.0
    assert p["low"]<=p["poc"]<=p["high"]
    assert p["high"]==100.25
    # Empty levels cannot make the expansion walk past the observed extrema.
    sparse=study._profile70({4000:3,5000:5,6000:2})
    assert sparse["poc"]==1250.0 and sparse["val"]==1000.0 and sparse["vah"]==1250.0


def test_poc_migration_alignment_and_drift_streak_reset():
    day="2025-03-03";start,_=study._session(day)
    tape=_tape(day,[start+301_000_000_000,start+601_000_000_000,start+901_000_000_000], [100.,100.25,100.5])
    bars=[{"date":day,"period":"SPRING_2025","status":"COMPLETE","bar_index":i,"close_ns":start+(i+1)*study.BAR_NS,
           "migration":d,"migration_streak":s,"drift_bar_ordinal":s,"poc_shift_ticks":shift,
           "full_up_alignment":True,"full_down_alignment":False,"poc":100+i*.25,"vah":101+i*.25,"val":99+i*.25,
           "event_features":{}} for i,(d,s,shift) in enumerate([("UP",1,1),("UP",2,1),("FLAT",0,0),("DOWN",1,-1)])]
    migrations,onsets=study._migration_events(bars,tape,day)
    assert [e["is_drift_onset"] for e in migrations]==[True,False,True]
    assert len(onsets)==2 and onsets[-1]["direction"]=="SHORT"


def test_delta_and_price_alignment_semantics():
    bar={"migration":"UP","delta":10,"delta_ratio":.2,"price_bar_direction":1,
         "directional_delta":10,"aggressive_volume":10,"open":100.,"close":100.25,
         "forward_side_top5_depth":80.,"directional_top5_mlofi":2.,"delta_supportive":True,
         "price_aligns_with_migration":True,"delta_aligns_with_migration":True,"both_price_and_delta_align":True,
         "value_area_overlap_ratio":.6,"trend_efficiency_30s":.5,"directional_trend_efficiency_30s":.5,
         "rv_30s_ticks":2,"directional_price_impact_per_flow":.1,"ask5":80,"bid5":100,"mlofi":2}
    f=study._event_features(bar,{"LONG":100,"SHORT":100})
    assert f["directional_delta"]==10 and f["delta_supportive"]
    assert f["forward_side_depth_relative"]==.8
    assert f["directional_top5_mlofi"]==2
    assert f["effort_support_count"]==3


def test_event_paths_apply_delay_spread_and_adverse_fill():
    day="2025-03-03";start,_=study._session(day)
    times=[start+1_000_000_000+i*100_000_000 for i in range(100)]
    prices=[100.+i*.25 for i in range(100)]
    tape=_tape(day,times,prices)
    p=study._path(tape,times[0],1,start+20_000_000_000)
    assert p["entry_time_ns"]>=times[0]+study.ENTRY_DELAY_NS
    for h in study.HORIZONS_MS:
        x=p["paths"][str(h)]
        if x["quote"] is not None:assert x["actual"]==pytest.approx(x["quote"]-1)
    assert p["first_touch"]["2:-2"]["result"] in {"FAVORABLE_FIRST","ADVERSE_FIRST","NEITHER","UNAVAILABLE"}


def test_reload_only_emits_first_touch_per_continuous_drift():
    day="2025-03-03";start,_=study._session(day)
    times=[start+600_000_000_000,start+601_000_000_000,start+602_000_000_000]
    tape=_tape(day,times,[100.,100.25,100.5])
    bars=[{"migration":"UP","bar_index":1,"close_ns":start+300_000_000_000,"val":99.,"vah":101.},
          {"migration":"UP","bar_index":2,"close_ns":start+600_000_000_000,"val":99.5,"vah":101.5}]
    rows=study._reloads(bars,tape,day)
    assert len(rows)==1 and rows[0]["event_type"]=="RELOAD_TOUCH"
    assert rows[0]["active_value_bar_index"]==2


def test_tercile_sample_floor_is_explicit():
    group={"LOW":{"actual_markout":{"n":19,"mean":-1}},"MID":{"actual_markout":{"n":19,"mean":0}},
           "HIGH":{"actual_markout":{"n":19,"mean":1}}}
    assert study._shape(group,"actual_markout")=="INSUFFICIENT_BUCKET_SAMPLE"


def test_checkpoint_binds_study_source_and_config(tmp_path):
    path=tmp_path/"date.json.gz"
    record={"status":"DATE_COMPLETE","version":study.CHECKPOINT_VERSION,"date":"2025-03-03",
        "source_sha256":"s","tape_sha256":"t","config_sha256":study.CONFIG_SHA256,
        "study_sha256":study.STUDY_SHA256,"payload":{"date":"2025-03-03"}}
    with gzip.open(path,"wt") as f:json.dump(record,f)
    assert study._read_checkpoint(path,"2025-03-03","s","t")==record
    assert study._read_checkpoint(path,"2025-03-03","bad","t") is None
    assert study._read_checkpoint(path,"2025-03-03","s","bad") is None
    record["study_sha256"]="old"
    with gzip.open(path,"wt") as f:json.dump(record,f)
    assert study._read_checkpoint(path,"2025-03-03","s","t") is None
