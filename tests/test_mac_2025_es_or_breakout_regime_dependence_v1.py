"""Deterministic offline checks for the frozen OR breakout regime audit."""
from __future__ import annotations

import gzip
import json

import numpy as np
import pytest

from research_pipeline.cme_orderflow_absorption_l2_v1 import mac_2025_es_or_breakout_regime_dependence_v1 as study


def _fixture(sign: int):
    day="2025-03-03"
    _,or_end,_=study.parent._windows(day)
    t=or_end+1_000_000_000
    boundary=100.0
    prices=[boundary+sign*.25,boundary+sign*2,boundary,boundary]
    times=[t,t+100_000_000,t+200_000_000,t+61_000_000_000]
    dtype=[("timestamp_ns","i8"),("execution_size","f8"),("execution_price","f8"),
           ("bid","f8"),("ask","f8")]
    tape=np.zeros(4,dtype=dtype)
    tape["timestamp_ns"]=times
    tape["execution_size"]=1
    tape["execution_price"]=prices
    tape["bid"]=[p-.125 for p in prices]
    tape["ask"]=[p+.125 for p in prices]
    event={"date":day,"period":"SPRING_2025","direction":"LONG" if sign>0 else "SHORT",
           "sign":sign,"timestamp_ns":t,"tape_index":0,"boundary":boundary,
           "opening_range_high":boundary,"opening_range_low":boundary,
           "opening_range_width_ticks":0,"trade_price":prices[0],"overshoot_ticks":1,
           "raw_anchor_time_ns":t,"raw_anchor_mid":prices[0],
           "time_of_day_bucket":"10:00-10:30","topology":"RETURN_INSIDE_OPENING_RANGE",
           "paths":{str(h):{"raw":1.0,"quote":0.0,"actual":-1.0} for h in study.HORIZONS},
           "event_anchor_features":{"realized_volatility_10s":2.0,
               "trend_efficiency_10s":.5,"price_velocity_10s":.2,"mlofi_500ms":.1}}
    return tape,event


@pytest.mark.parametrize("sign",[1,-1])
def test_trade_return_expansion_good_topology_and_symmetry(sign):
    tape,event=_fixture(sign)
    lifecycle=study._lifecycle(tape,event)
    assert lifecycle["return_inside_ms"]==200
    assert lifecycle["favorable_expansion_before_return_ticks"]==8
    assert lifecycle["time_to_max_expansion_before_return_ms"]==100
    assert lifecycle["good_expansion_class"]=="GOOD_EXPANSION_EVENT"
    assert lifecycle["topology"]=="FAST_CONTINUATION"
    assert lifecycle["excursions"]["250"]["mfe_ticks"]==pytest.approx(7)
    assert lifecycle["opposite_or_break_before_close"] is False


def test_weak_and_unresolved_good_expansion():
    tape,event=_fixture(1)
    tape["execution_price"][1]=100.5
    assert study._lifecycle(tape,event)["good_expansion_class"]=="WEAK_OR_FAILED_EVENT"
    tape["execution_size"][2:]=0
    result=study._lifecycle(tape,event)
    assert result["return_status"]=="NO_RETURN_BEFORE_RTH_CLOSE"
    assert result["good_expansion_class"]=="UNRESOLVED"


def test_only_six_frozen_features_and_time():
    _,event=_fixture(1)
    features=study._features(event)
    assert tuple(features)==study.FEATURES
    assert features["minutes_after_10"]==pytest.approx(1/60)
    assert features["directional_top5_mlofi_500ms"]==.1
    assert study.CONFIG["price_features"].find("10-second")>=0


def test_spring_cuts_and_tercile_boundaries():
    spring=[{"features":{"or_width":x}} for x in (1.,2.,3.,4.,5.,6.)]
    # The full cutpoint function requires all six variables; test the exact
    # frozen inclusion convention independently.
    assert study._tercile(1,[2,4])=="LOW"
    assert study._tercile(2,[2,4])=="LOW"
    assert study._tercile(4,[2,4])=="MID"
    assert study._tercile(5,[2,4])=="HIGH"
    assert study._tercile(None,[2,4]) is None


def test_insufficient_cell_and_shape():
    assert study._feature_cell([])["status"]=="INSUFFICIENT_BUCKET_SAMPLE"
    cells={k:{"n":9,"raw_markouts":{"10000":{"mean":i}}}
           for i,k in enumerate(study.TERCILES)}
    assert study._shape(cells)=="INSUFFICIENT_SAMPLE"


def test_checkpoint_rejects_changed_source_config_or_parent(tmp_path):
    path=tmp_path/"day.json.gz"
    row={"status":"DATE_COMPLETE","version":study.CHECKPOINT_VERSION,"date":"2025-03-03",
         "source_sha256":"src","tape_sha256":"tape","parent_manifest_sha256":"parent",
         "entry_manifest_sha256":"entry","config_sha256":study.CONFIG_SHA256,"payload":{"events":[]}}
    with gzip.open(path,"wt") as f:json.dump(row,f)
    assert study._read_checkpoint(path,"2025-03-03","src","tape","parent","entry")==row
    assert study._read_checkpoint(path,"2025-03-03","other","tape","parent","entry") is None
    assert study._read_checkpoint(path,"2025-03-03","src","tape","other","entry") is None
    row["config_sha256"]="other"
    with gzip.open(path,"wt") as f:json.dump(row,f)
    assert study._read_checkpoint(path,"2025-03-03","src","tape","parent","entry") is None


def test_only_predeclared_interactions():
    assert len(study.CONFIG["interactions"])==2
    assert study.CONFIG["no_optimization"] is True
    assert study.CONFIG["no_2026"] is True


def test_chronological_prior_date_width_percentile():
    cuts={f:[2,4] for f in study.FEATURES}
    def row(day,width):
        return {"date":day,"features":{f:width for f in study.FEATURES}}
    rows=[row("2025-03-03",2),row("2025-03-04",4)]
    study._assign_features(rows,cuts)
    assert rows[0]["prior_date_or_width_percentile"] is None
    assert rows[1]["prior_date_or_width_percentile"]==100
    assert rows[0]["feature_terciles"]["or_width"]=="LOW"
    assert rows[1]["feature_terciles"]["or_width"]=="MID"


def test_daily_weekly_leaveout_and_spring_only():
    tape,event=_fixture(1)
    day=study.evaluate_date("2025-03-03",tape,[event])["events"][0]
    day["feature_terciles"]={f:"LOW" for f in study.FEATURES}
    daily,weekly=study._date_week_tables([day])
    assert daily["2025-03-03"]["n"]==1
    assert weekly["2025-W10"]["n"]==1
    loo=study._stability([day],"date")
    assert len(loo["rows"])==len(study.flow.SPRING_DATES)
    assert loo["rows"]["2025-03-03"]["n"]==0


def test_predeclared_permutation_is_seeded_and_direction_stratified():
    rows=[]
    for i in range(30):
        state="LOW" if i<10 else "MID" if i<20 else "HIGH"
        rows.append({"direction":"LONG" if i%2 else "SHORT",
            "feature_terciles":{f:state for f in study.FEATURES},
            "good_expansion_class":"GOOD_EXPANSION_EVENT" if i%3==0 else "WEAK_OR_FAILED_EVENT",
            "paths":{"10000":{"raw":float(i)}}})
    assert study._permutation(rows)==study._permutation(rows)
    assert len(study._permutation(rows))==4
