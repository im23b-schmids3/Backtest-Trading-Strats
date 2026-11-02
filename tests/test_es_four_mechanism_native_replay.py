from array import array

import numpy as np

NS=1_000_000_000

from research_pipeline.cme_orderflow_absorption_l2_v1.es_four_mechanism_native_replay import (
    BookState, DetectorSuite, NativeBookReplay, OnlineEmpiricalQuantile, _outcomes,
    _construction_outcomes,
)


def _dtype():
    fields=[("ts_recv","i8"),("ts_event","i8"),("flags","u1"),("action","S1"),
            ("side","S1"),("price","i8"),("size","i4")]
    for i in range(10):
        fields.extend([(f"bid_px_{i:02d}","i8"),(f"bid_sz_{i:02d}","i4"),
                       (f"ask_px_{i:02d}","i8"),(f"ask_sz_{i:02d}","i4")])
    return np.dtype(fields)


def _row(t, *, bid=100, ask=101, bid_size=10, ask_size=11, bid5=6,
         ask5=7, flags=0, action=b"M", side=b"N", price=0, size=0):
    r=np.zeros((),dtype=_dtype())
    r["ts_recv"]=t; r["ts_event"]=t-1; r["flags"]=flags
    r["action"]=action; r["side"]=side; r["price"]=price; r["size"]=size
    for i in range(10):
        r[f"bid_px_{i:02d}"]=(bid-i)*1_000_000_000
        r[f"ask_px_{i:02d}"]=(ask+i)*1_000_000_000
        r[f"bid_sz_{i:02d}"]=bid5 if i==4 else bid_size
        r[f"ask_sz_{i:02d}"]=ask5 if i==4 else ask_size
    return r


def _run(rows, replay=None):
    replay=replay or NativeBookReplay()
    seen=[]
    for row in rows:
        state,changed,depth_only=replay.process(row)
        seen.append((state,changed,depth_only))
    return replay,seen


def test_depth_only_change_and_price_migration_are_price_aligned():
    a=_row(1); b=_row(2,bid5=99); c=_row(3,bid=99)
    replay,seen=_run([a,b,c])
    assert seen[1][1] and seen[1][2]
    assert seen[2][1] and not seen[2][2]
    assert replay.state.bids[0][0]==99_000_000_000
    assert replay.state.bids[1][0]==98_000_000_000


def test_bid_and_ask_quantity_updates_change_state_without_price_migration():
    _,seen=_run([_row(1),_row(2,bid_size=20,ask_size=30)])
    assert seen[1][1]
    assert seen[1][0].bids[0]==(100_000_000_000,20)
    assert seen[1][0].asks[0]==(101_000_000_000,30)


def test_checkpoint_resume_is_exactly_equivalent_to_uninterrupted_replay():
    rows=[_row(1),_row(2,bid5=22),_row(3,bid=99,flags=32),_row(4,flags=4)]
    full,full_seen=_run(rows)
    partial,first=_run(rows[:2])
    resumed=NativeBookReplay.resume(partial.checkpoint())
    resumed,second=_run(rows[2:],resumed)
    assert full_seen==first+second
    assert full.checkpoint()==resumed.checkpoint()


def test_invalid_quote_gap_snapshot_and_timestamp_ties_are_preserved():
    rows=[_row(7),_row(7,bid=101,ask=101),_row(8,flags=8),_row(9,flags=32)]
    replay,seen=_run(rows)
    assert [x[0].ordinal for x in seen]==[0,1,2,3]
    assert seen[0][0].valid
    assert not seen[1][0].valid # locked
    assert seen[2][0].gap and not seen[2][0].valid
    assert seen[3][0].snapshot and seen[3][0].valid
    assert replay.validity["invalid"]==2


def test_unknown_aggressor_is_not_directionally_imputed():
    day="2025-03-03"; t=1_740_960_000_000_000_000
    state=BookState(0,t,t-1,tuple(((100-i)*1_000_000_000,10) for i in range(10)),
                    tuple(((101+i)*1_000_000_000,10) for i in range(10)),True,False,False)
    suite=DetectorSuite(day)
    suite.on_row(state,"T",100_000_000_000,9,"?")
    assert suite.validity["unknown_side_trades"]==1
    assert suite.validity["known_side_trades"]==0
    assert not suite.m1_bins


def test_trade_on_invalid_book_is_retained_for_flow_but_not_directionally_imputed():
    day="2025-03-03"; t=1_740_960_000_000_000_000
    state=BookState(0,t,t-1,tuple(((100-i)*1_000_000_000,10) for i in range(10)),
                    tuple(((101+i)*1_000_000_000,10) for i in range(10)),False,False,True)
    suite=DetectorSuite(day)
    suite.on_row(state,"T",100_000_000_000,7,"B")
    assert suite.validity["known_side_trades"]==1
    assert suite.m1_bins[(0,t//500_000_000)]==7


def test_m1_uses_canonical_b_side_as_aggressor_buy_and_prior_only_bins():
    day="2025-03-03"; base=1_740_960_000_000_000_000
    suite=DetectorSuite(day)
    state0=BookState(0,base,base,tuple(((100-i)*1_000_000_000,10) for i in range(10)),
                     tuple(((101+i)*1_000_000_000,10) for i in range(10)),True,False,False)
    for i in range(22):
        t=base+i*500_000_000
        state=BookState(i,t,t-1,state0.bids,state0.asks,True,False,False)
        suite.on_row(state,"T",101_000_000_000,100 if i==20 else 1,"B")
    # Close the large bin with a later receive-time event.
    t=base+22*500_000_000
    state=BookState(22,t,t-1,state0.bids,state0.asks,True,False,False)
    suite.on_row(state,"M",0,0,"N")
    assert suite.validity["known_side_trades"]==22
    qualified=[e for e in suite.events["M1"] if '"signed_volume":100.0' in e["features"]]
    assert len(qualified)==1 and qualified[0]["direction"]==1
    assert qualified[0]["ts_recv"]==base+21*500_000_000 and qualified[0]["source_ordinal"]==20


def test_detector_checkpoint_json_resume_preserves_online_state_and_events():
    import json
    day="2025-03-03"; base=1_740_960_000_000_000_000
    bids=tuple(((100-i)*1_000_000_000,10) for i in range(10))
    asks=tuple(((101+i)*1_000_000_000,10) for i in range(10))
    full=DetectorSuite(day); split=DetectorSuite(day)
    for i in range(3):
        state=BookState(i,base+i*50_000_000,base+i*50_000_000-1,bids,asks,True,False,False)
        full.on_row(state,"T",101_000_000_000,2,"B")
        if i<2: split.on_row(state,"T",101_000_000_000,2,"B")
    resumed=DetectorSuite.resume(json.loads(json.dumps(split.checkpoint(),allow_nan=False)))
    state=BookState(2,base+100_000_000,base+100_000_000-1,bids,asks,True,False,False)
    resumed.on_row(state,"T",101_000_000_000,2,"B")
    assert resumed.checkpoint()==full.checkpoint()


def test_m2_event_timestamp_is_qualification_time_not_candidate_start():
    from research_pipeline.cme_orderflow_absorption_l2_v1.es_four_mechanism_native_replay import _windows
    day="2025-03-03"; start=_windows(day)["ASIA"][0]
    t=start+2*NS; origin=start; target=origin+NS
    suite=DetectorSuite(day)
    warm=BookState(0,start,start-1,tuple(((100-i)*1_000_000_000,1) for i in range(10)),
                   tuple(((101+i)*1_000_000_000,1) for i in range(10)),True,False,False)
    suite.on_row(warm,"M",0,0,"N")
    suite.mid_grid_origin[0]=origin
    suite.mid_lookup[0][target]=100.25
    suite.next_mid_sample[0]=t+50_000_000
    suite.next_sample[0]=t
    suite.m2_medians[0]=(100.0,100.0)
    suite.m2_candidate[0]=(t-300_000_000,4,100.0,1,40)
    state=BookState(9,t,t-1,tuple(((100-i)*1_000_000_000,1) for i in range(10)),
                    tuple(((101+i)*1_000_000_000,1) for i in range(10)),True,False,False)
    suite.on_row(state,"M",0,0,"N")
    assert len(suite.events["M2"])==1
    event=suite.events["M2"][0]
    assert event["ts_recv"]==t and event["source_ordinal"]==9
    assert '"thinning_start_ts_recv":' in event["features"]


def test_midpoint_grid_uses_last_quote_available_at_each_sample_time():
    from research_pipeline.cme_orderflow_absorption_l2_v1.es_four_mechanism_native_replay import _windows
    day="2025-03-03"; start=_windows(day)["ASIA"][0]
    suite=DetectorSuite(day)
    bids=tuple(((100-i)*1_000_000_000,10) for i in range(10))
    asks=tuple(((101+i)*1_000_000_000,10) for i in range(10))
    suite.on_row(BookState(0,start,start-1,bids,asks,True,False,False),"M",0,0,"N")
    bids2=tuple(((101-i)*1_000_000_000,10) for i in range(10))
    asks2=tuple(((102+i)*1_000_000_000,10) for i in range(10))
    suite.on_row(BookState(1,start+100_000_000,start+99_999_999,bids2,asks2,True,False,False),"M",0,0,"N")
    suite.on_row(BookState(2,start+150_000_000,start+149_999_999,bids2,asks2,True,False,False),"M",0,0,"N")
    assert suite.mid_lookup[0][start]==100.5
    assert suite.mid_lookup[0][start+50_000_000]==100.5
    assert suite.mid_lookup[0][start+100_000_000]==101.5


def test_online_quantile_matches_numpy_linear_definition():
    q=OnlineEmpiricalQuantile()
    values=[0,1,1,2,4,9,9,10]
    for value in values:q.add(value)
    assert q.quantile(.9)==float(np.quantile(values,.9,method="linear"))


def test_executable_entry_uses_only_quote_available_by_latency_deadline():
    trigger=1_000_000_000
    ev={"date":"2025-03-03","mechanism":"M1","event_id":"e","ts_recv":trigger,
        "source_ordinal":1,"direction":1}
    # The later quote at +3ms must not be used for a +2ms simulated entry.
    quotes={"t":array("q",[trigger,trigger+3_000_000,trigger+10_002_000_000]),
            "ordinal":array("q",[1,2,3]),
            "bid":array("q",[100_000_000_000,101_000_000_000,102_000_000_000]),
            "ask":array("q",[101_000_000_000,102_000_000_000,103_000_000_000])}
    out=_outcomes([ev],quotes)[0]
    # Entry is from the trigger-time BBO, with the frozen one-tick adverse fill.
    assert out["net_executable_10s_ticks"] is not None


def test_trigger_without_fresh_anchor_is_emitted_as_censored_not_dropped():
    ev={"date":"2025-03-03","mechanism":"M2","event_id":"x","ts_recv":99_000_000_000,
        "source_ordinal":3,"direction":-1}
    quotes={"t":array("q",[1]),"ordinal":array("q",[1]),"bid":array("q",[100_000_000_000]),
            "ask":array("q",[101_000_000_000])}
    rows=_outcomes([ev],quotes)
    assert len(rows)==1 and rows[0]["source_censored"]


def test_source_gap_censors_first_passage_and_markout_windows():
    trigger=1_000_000_000
    ev={"date":"2025-03-03","mechanism":"M1","event_id":"g","ts_recv":trigger,
        "source_ordinal":1,"direction":1}
    quotes={"t":array("q",[trigger,trigger+1*NS,trigger+10*NS,trigger+30*NS]),
            "ordinal":array("q",[1,2,3,4]),
            "bid":array("q",[100_000_000_000,101_000_000_000,102_000_000_000,103_000_000_000]),
            "ask":array("q",[101_000_000_000,102_000_000_000,103_000_000_000,104_000_000_000])}
    out=_outcomes([ev],quotes,[(trigger+2*NS,trigger+8*NS)])[0]
    assert out["four_tick_first_passage_30s"]=="SOURCE_CENSORED"
    assert out["mid_markout_5s_ticks"] is None


def test_construction_outcomes_cover_registered_horizons_barriers_and_cost_stress():
    trigger=1_000_000_000
    ev={"date":"2025-03-03","mechanism":"M1","event_id":"full","ts_recv":trigger,
        "source_ordinal":1,"direction":1}
    times=[trigger,trigger+2_000_000,trigger+NS,trigger+5*NS,trigger+10*NS,
           trigger+30*NS,trigger+60*NS]
    mids=[100.5,100.5,101.0,101.5,102.0,102.0,102.0]
    quotes={"t":array("q",times),"ordinal":array("q",range(1,len(times)+1)),
            "bid":array("q",[int((x-.25)*1e9) for x in mids]),
            "ask":array("q",[int((x+.25)*1e9) for x in mids])}
    row=_construction_outcomes([ev],quotes)[0]
    assert row["barrier_2_first"]=="PREDICTED_FIRST"
    assert row["barrier_4_first"]=="PREDICTED_FIRST"
    assert row["mid_markout_60s_ticks"] is not None
    assert row["net_executable_10s_baseline_ticks"] > row["net_executable_10s_moderate_ticks"]
    assert row["net_executable_10s_moderate_ticks"] > row["net_executable_10s_severe_ticks"]


def test_same_timestamp_quote_after_trigger_uses_native_ordinal_order():
    trigger=1_000_000_000
    ev={"date":"2025-03-03","mechanism":"M1","event_id":"tie","ts_recv":trigger,
        "source_ordinal":1,"direction":1}
    quotes={"t":array("q",[trigger,trigger,trigger+NS,trigger+10*NS,trigger+30*NS]),
            "ordinal":array("q",[1,2,3,4,5]),
            "bid":array("q",[100_000_000_000,101_000_000_000,101_000_000_000,101_000_000_000,101_000_000_000]),
            "ask":array("q",[101_000_000_000,102_000_000_000,102_000_000_000,102_000_000_000,102_000_000_000])}
    row=_outcomes([ev],quotes)[0]
    assert row["anchor_mid"]==100.5
    assert row["four_tick_first_passage_30s"]=="PREDICTED_FIRST"
