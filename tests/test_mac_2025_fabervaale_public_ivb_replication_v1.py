"""Offline checks for the frozen public-model OR/profile/retrace convention."""
from __future__ import annotations

import gzip
import json

import numpy as np
import pytest

from research_pipeline.cme_orderflow_absorption_l2_v1 import mac_2025_fabervaale_public_ivb_replication_v1 as study


def _tape(day="2025-03-03"):
    start,or_end,_=study.parent._windows(day)
    times=[start,start+1,start+2,or_end+1,or_end+2,or_end+250_000_001,
           or_end+500_000_001,or_end+750_000_001,or_end+61_000_000_001]
    prices=[100.,100.25,99.75,100.5,100.,100.25,100.5,100.75,101.]
    dtype=[("timestamp_ns","i8"),("execution_size","i8"),("execution_price","f8"),
           ("aggressor","i1"),("bid","f8"),("ask","f8")]
    a=np.zeros(len(times),dtype=dtype)
    a["timestamp_ns"]=times;a["execution_price"]=prices
    a["execution_size"]=[10,5,5,1,2,3,4,5,1]
    a["aggressor"]=[1,-1,1,1,-1,1,1,1,1]
    a["bid"]=np.asarray(prices)-.125;a["ask"]=np.asarray(prices)+.125
    return a


def _frame_event(tape,direction="LONG",ix=3):
    return {"date":"2025-03-03","period":"SPRING_2025","direction":direction,
            "sign":1 if direction=="LONG" else -1,
            "timestamp_ns":int(tape["timestamp_ns"][ix]),"tape_index":ix,
            "trade_price":float(tape["execution_price"][ix]),
            "opening_range_high":100.25,"opening_range_low":99.75}


def test_actual_trade_profile_poc_tie_and_70pct():
    tape=_tape();p=study._profile("2025-03-03",tape)
    assert p["profile_total_volume"]==20
    assert p["or_high"]==100.25 and p["or_low"]==99.75
    assert p["poc"]==100 and p["val"]==99.75 and p["vah"]==100
    assert p["price_volume_by_tick"]=={399:5,400:10,401:5}
    # Equal POC volume resolves to the lower ES tick.
    assert study.profile_module.asia_volume_profile({399:10,400:10})["poc"]==99.75


def test_first_frame_only_and_zone():
    tape=_tape();p=study._profile("2025-03-03",tape)
    first=_frame_event(tape)
    later=dict(first,direction="SHORT",sign=-1,timestamp_ns=int(tape["timestamp_ns"][7]),
               tape_index=7,trade_price=99.5)
    frame=study._frame("2025-03-03",[later,first],p)
    assert frame["direction"]=="LONG" and frame["opposite_first_breakout_observed"]
    assert (frame["zone_low"],frame["zone_high"])==(100,100)
    assert frame["invalidation_reference"]==99.75
    short=study._frame("2025-03-03",[dict(first,direction="SHORT",sign=-1)],p)
    assert (short["zone_low"],short["zone_high"])==(99.75,100)
    assert short["invalidation_reference"]==100


def test_first_retrace_and_no_retrace():
    tape=_tape();p=study._profile("2025-03-03",tape)
    frame=study._frame("2025-03-03",[_frame_event(tape)],p)
    r=study._retrace(tape,frame,p)
    assert r["status"]=="RETRACE" and r["tape_index"]==4
    assert r["price"]==100 and r["location_bucket"]=="POC_ONLY"
    assert r["max_favorable_expansion_before_retrace_ticks"]==0
    tape["execution_size"][4:]=0
    assert study._retrace(tape,frame,p)["status"]=="NO_RETRACE"


def test_confirmation_window_excludes_later_trades():
    tape=_tape();p=study._profile("2025-03-03",tape)
    frame=study._frame("2025-03-03",[_frame_event(tape)],p)
    r=study._retrace(tape,frame,p);t=r["timestamp_ns"]
    dtype=[("ts","i8"),("mlofi","f8"),("denom","f8"),("bid5","f8"),("ask5","f8")]
    rows=np.zeros(3,dtype=dtype)
    rows["ts"]=[t,t+250_000_000,t+500_000_000]
    rows["mlofi"]=[1,2,3];rows["denom"]=[10,10,10]
    rows["bid5"]=[100,110,120];rows["ask5"]=[100,100,100]
    result=study._orderflow(tape,rows,r,frame,p)
    assert result["confirmation_timestamp_ns"]==t+500_000_000
    assert result["supportive_aggressive_volume"]==7  # +250ms and +500ms
    assert result["opposing_aggressive_volume"]==2  # retrace trade
    assert result["directional_top5_mlofi"]==pytest.approx(.6)
    assert result["passive_defense_ratio"]==pytest.approx(.2)
    assert result["support_count"]>=2


def test_quote_and_fill_decomposition():
    tape=_tape();t=int(tape["timestamp_ns"][4]);close=study.parent._windows("2025-03-03")[2]
    path=study._markouts(tape,t,1,close,executable=True,floor=4)
    valid=[p for p in path["paths"].values() if p["actual"] is not None]
    assert valid
    for p in valid:
        assert p["actual"]==pytest.approx(p["quote"]-1)
        assert p["actual"]==pytest.approx(p["raw"]+p["horizon_shift"]-
            p["pre_entry_price_move"]+p["bid_ask_effect"]-1)


def test_confirmed_anchor_excludes_equal_timestamp_observations():
    tape=_tape();close=study.parent._windows("2025-03-03")[2]
    end=int(tape["timestamp_ns"][6])
    path=study._markouts(tape,end+1,1,close,executable=True,floor=4,
        entry_reference_ns=end)
    assert path["raw_anchor_time_ns"]>end
    assert path["entry_time_ns"]>=end+2_000_000


def test_support_group_and_small_cells():
    assert study._group_metrics([])["status"]=="INSUFFICIENT_BUCKET_SAMPLE"
    assert study.CONFIG["support_groups"]=={"LOW":[0,1],"MID":[2],"HIGH":[3,4]}
    assert len(study.COMPONENTS)==4
    assert study.CONFIG["no_passive_fill"] is True


def test_checkpoint_rejects_source_and_config_changes(tmp_path):
    path=tmp_path/"day.json.gz"
    row={"status":"DATE_COMPLETE","version":study.CHECKPOINT_VERSION,"date":"2025-03-03",
         "source_sha256":"s","tape_sha256":"t","parent_manifest_sha256":"p",
         "entry_manifest_sha256":"e","config_sha256":study.CONFIG_SHA256,"payload":{}}
    with gzip.open(path,"wt") as f:json.dump(row,f)
    assert study._read_checkpoint(path,"2025-03-03","s","t","p","e")==row
    assert study._read_checkpoint(path,"2025-03-03","bad","t","p","e") is None
    row["config_sha256"]="bad"
    with gzip.open(path,"wt") as f:json.dump(row,f)
    assert study._read_checkpoint(path,"2025-03-03","s","t","p","e") is None
