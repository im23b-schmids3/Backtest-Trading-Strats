from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from research_pipeline.cme_orderflow_absorption_l2_v1 import mac_2025_sato_prior_day_liquidity_sweep_v1 as sato
from research_pipeline.cme_orderflow_absorption_l2_v1.mac_2025_candidate_tape import EVENT_DTYPE


def _tape(rows):
    out = np.zeros(len(rows), dtype=EVENT_DTYPE)
    for i, (ts, bid, ask, price, size, aggr) in enumerate(rows):
        out[i] = (ts, bid, ask, price, size, aggr, 2)
    return out


def test_previous_valid_mapping_skips_weekends_and_holiday_closure():
    got = sato.previous_valid_rth_map(
        ["2025-03-03", "2025-04-21", "2025-10-07"],
        ["2025-02-28", "2025-04-17", "2025-10-06"],
    )
    assert got == {"2025-03-03": "2025-02-28", "2025-04-21": "2025-04-17", "2025-10-07": "2025-10-06"}


def test_missing_previous_valid_rth_fails_closed():
    with pytest.raises(sato.SatoStudyError):
        sato.previous_valid_rth_map(["2025-03-03"], [])


def test_rth_bounds_are_dst_aware_and_half_open():
    start, end = sato._rth("2025-03-03")
    assert end - start == 6.5 * 60 * 60 * 1e9
    assert sato._rth("2025-03-03")[0] != sato._rth("2025-10-07")[0]


def test_early_close_uses_explicit_official_override(monkeypatch):
    monkeypatch.setitem(sato.OFFICIAL_RTH_EARLY_CLOSE_ET, "2025-03-03", (13, 0))
    start, end = sato._rth("2025-03-03")
    assert end-start == 3.5*60*60*1e9
    assert sato.OFFICIAL_RTH_EARLY_CLOSE_ET["2025-03-03"] == (13,0)


def test_actual_trade_levels_exclude_rth_end_timestamp():
    start, end = sato._rth("2025-03-03")
    tape = _tape([(start, np.nan, np.nan, 5000.0, 1, 1),
                  (end - 1, np.nan, np.nan, 5001.0, 1, -1),
                  (end, np.nan, np.nan, 9000.0, 1, 1)])
    ids = sato._trades(tape, start, end)
    assert tape["execution_price"][ids].tolist() == [5000.0, 5001.0]


def test_incomplete_previous_rth_request_is_rejected_before_level_construction(tmp_path):
    prev="2025-02-28"
    with pytest.raises(sato.SatoStudyError,match="request bounds incomplete"):
        sato._prior_levels("2025-03-03",prev,{prev:tmp_path/"unused.dbn"},
            {prev:{"start":"2025-02-28T15:00:00Z","end":"2025-02-28T21:00:00Z","sha256":"x"}})


def test_five_minute_bar_is_half_open_and_uses_actual_trades():
    start, _ = sato._rth("2025-03-03")
    tape = _tape([(start, np.nan, np.nan, 5000.0, 2, 1),
                  (start + sato.BAR_NS - 1, np.nan, np.nan, 5001.0, 3, -1),
                  (start + sato.BAR_NS, np.nan, np.nan, 4999.0, 1, 1)])
    bars = sato._bars("2025-03-03", tape, start_ns=start, end_ns=start + 2*sato.BAR_NS)
    assert len(bars) == 2
    assert bars[0]["open"] == 5000 and bars[0]["high"] == 5001 and bars[0]["close"] == 5001
    assert bars[0]["total_volume"] == 5 and bars[0]["delta"] == -1
    assert bars[1]["open"] == 4999


@pytest.mark.parametrize("equal_price,side", [(5000.0, "PDH"), (4990.0, "PDL")])
def test_equal_level_does_not_create_strict_breach(equal_price, side):
    day = "2025-03-03"; start, close = sato._rth(day)
    ts = np.array([start + 1_000_000_000], dtype=np.int64)
    px = np.array([equal_price]); sz = np.array([1]); ag = np.array([1], dtype=np.int8)
    bar = {"bar_index": 0, "start_ns": start, "close_ns": start+sato.BAR_NS,
           "open": equal_price, "high": equal_price, "low": equal_price, "close": equal_price,
           "total_volume": 1, "buy_volume": 1, "sell_volume": 0, "delta": 1}
    level = equal_price
    assert sato._make_candidate(day, side, level, np.empty(0, dtype=EVENT_DTYPE), [bar], [bar],
        {"pdh": 5000., "pdl": 4990., "midpoint": 4995., "range_ticks": 40.},
        np.array([0]), ts, px, sz, ag) is None


@pytest.mark.parametrize("closes,expected", [
    ([4999.75, 4999.5, 4999.25], 1),
    ([5000.25, 4999.75, 4999.5], 2),
    ([5000.25, 5000.25, 4999.75], 3),
])
def test_pdh_reclaim_can_only_be_same_next_or_third_bar(closes, expected):
    day = "2025-03-03"; start, _ = sato._rth(day); level = 5000.
    bars = []
    for i, close in enumerate(closes):
        bars.append({"bar_index": i, "start_ns": start+i*sato.BAR_NS, "close_ns": start+(i+1)*sato.BAR_NS,
                     "open": 5000.25, "high": 5000.5, "low": 4999.5, "close": close,
                     "total_volume": 100, "buy_volume": 60, "sell_volume": 40, "delta": 20})
    ts = np.array([start+1_000_000_000, start+sato.BAR_NS+1], dtype=np.int64)
    px = np.array([5000.25, 5000.5]); sz=np.array([10, 10]); ag=np.array([1, 1], dtype=np.int8)
    ev = sato._make_candidate(day, "PDH", level, np.empty(0, dtype=EVENT_DTYPE), bars, bars,
        {"pdh":5000.,"pdl":4990.,"midpoint":4995.,"range_ticks":40.}, np.arange(2),ts,px,sz,ag)
    assert ev["reclaim_bar_rank"] == expected
    assert ev["reclaim_timestamp_ns"] == start+expected*sato.BAR_NS


def test_no_reclaim_anchors_acceptance_at_third_bar_close():
    day = "2025-03-03"; start, _ = sato._rth(day)
    bars=[{"bar_index":i,"start_ns":start+i*sato.BAR_NS,"close_ns":start+(i+1)*sato.BAR_NS,
           "open":5000.25,"high":5000.5,"low":5000.0,"close":5000.25,
           "total_volume":50,"buy_volume":30,"sell_volume":20,"delta":10} for i in range(3)]
    ts=np.array([start+1_000_000_000],dtype=np.int64);px=np.array([5000.25]);sz=np.array([2]);ag=np.array([1],dtype=np.int8)
    ev=sato._make_candidate(day,"PDH",5000.,np.empty(0,dtype=EVENT_DTYPE),bars,bars,
       {"pdh":5000.,"pdl":4990.,"midpoint":4995.,"range_ticks":40.},np.array([0]),ts,px,sz,ag)
    assert ev["no_reclaim_acceptance_candidate"] is True
    assert ev["acceptance_timestamp_ns"] == start+3*sato.BAR_NS


def test_volume_spike_uses_frozen_150_percent_convention_and_six_prior_bars():
    day="2025-03-03";start,_=sato._rth(day)
    refs=[{"bar_index":i-6,"start_ns":start+(i-6)*sato.BAR_NS,"close_ns":start+(i-5)*sato.BAR_NS,
           "open":5000.,"high":5000.,"low":5000.,"close":5000.,"total_volume":100,
           "buy_volume":50,"sell_volume":50,"delta":0} for i in range(6)]
    current={"bar_index":0,"start_ns":start,"close_ns":start+sato.BAR_NS,"open":5000.25,"high":5000.5,
             "low":5000.,"close":4999.75,"total_volume":150,"buy_volume":100,"sell_volume":50,"delta":50}
    bars=refs+[current]
    ts=np.array([start+1_000_000_000],dtype=np.int64);px=np.array([5000.25]);sz=np.array([10]);ag=np.array([1],dtype=np.int8)
    ev=sato._make_candidate(day,"PDH",5000.,np.empty(0,dtype=EVENT_DTYPE),[current],bars,
        {"pdh":5000.,"pdl":4990.,"midpoint":4995.,"range_ticks":40.},np.array([0]),ts,px,sz,ag)
    assert ev["volume_ratio"] == 1.5
    assert ev["volume_spike"] is True
    assert ev["delta_divergence"] is True
    assert ev["public_stack_qualified"] is True  # reclaim is mandatory + volume + Delta


def test_checkpoint_identity_binds_source_config_and_study_versions():
    first=sato._checkpoint_key("2025-03-03","source-a","prior-b")
    assert first != sato._checkpoint_key("2025-03-03","source-a","prior-changed")
    assert first != sato._checkpoint_key("2025-03-03","source-changed","prior-b")


def test_frozen_execution_policy_is_two_ms_and_adverse_tick():
    assert sato.ENTRY_DELAY_NS == 2_000_000
    assert sato.TICK == 0.25


def test_path_uses_two_ms_delay_ask_for_long_and_one_tick_adverse_fill():
    signal=sato._ns("2025-03-03",10,0)
    rows=[]
    points=[signal,signal+1_000_000,signal+3_000_000]
    points += [signal+3_000_000+h*1_000_000 for h in sato.MARKOUT_MS]
    for i,ts in enumerate(sorted(set(points))):
        bid=5000.0+i*0.25
        rows.append((ts,bid,bid+0.25,np.nan,0,0))
    tape=_tape(rows)
    event={"reclaim_timestamp_ns":signal,"reversal_direction":"LONG","sweep_extreme":4998.0,
           "session_vwap_at_reclaim":5001.0,"prior_day_midpoint":5002.0,
           "prior_day_low":4990.0,"prior_day_high":5010.0}
    sato._path_event(event,tape,sato._ns("2025-03-03",16,0))
    assert event["path_status"] == "AVAILABLE"
    assert event["executable_entry_timestamp_ns"] == signal+3_000_000
    assert event["executable_entry_quote"] == tape["ask"][2]
    assert event["actual_fill_price"] == pytest.approx(event["executable_entry_quote"]+sato.TICK)
    assert event["raw_signal_best_bid"] == tape["bid"][1]
