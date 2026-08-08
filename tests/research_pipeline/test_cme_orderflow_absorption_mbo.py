import pytest
import json
from types import SimpleNamespace
from research_pipeline.cme_orderflow_absorption_v1.engine import CausalMBOBook, BookStateError
from research_pipeline.cme_orderflow_absorption_v1.analysis import (
    Diagnostics, RTH_START, TICK, DBN_FIXED_POINT_SCALE, previous_completed_rth, volume_profile,
)
from research_pipeline.cme_orderflow_absorption_v1.loader import MetadataSummary
from research_pipeline.cme_orderflow_absorption_v1.report import write_reports

def apply(b, a, s, p, z, oid, seq=1, ts=1): return b.apply(action=a,side=s,price=p,size=z,order_id=oid,sequence=seq,ts_recv=ts)
def test_add_cancel_modify_multiple_orders_queue_restoration_best_spread():
 b=CausalMBOBook(); apply(b,'A','B',100,3,1); apply(b,'A','B',100,2,2,2); apply(b,'A','A',101,4,3,3)
 assert b.depth['B'][100]==5 and b.best_bid()==100 and b.best_ask()==101 and b.spread()==1
 apply(b,'C','B',100,1,1,4); apply(b,'M','B',99,5,2,5); apply(b,'A','B',100,1,4,6)
 assert b.depth['B'][100]==3 and b.depth['B'][99]==5
def test_partial_and_full_fills():
 b=CausalMBOBook(); apply(b,'A','A',101,5,1); assert apply(b,'F','A',101,2,1,2).executed
 assert b.orders[1].size==3; apply(b,'F','A',101,3,1,3); assert 1 not in b.orders
def test_negative_depth_and_invalid_state_fail_closed():
 b=CausalMBOBook()
 with pytest.raises(BookStateError): apply(b,'C','B',100,1,1)
 apply(b,'A','B',100,1,1,2)
 with pytest.raises(BookStateError): apply(b,'F','B',100,2,1,3)
def test_sequence_ordering_fails_closed():
 b=CausalMBOBook(); apply(b,'A','B',100,1,1,2,10)
 with pytest.raises(BookStateError): apply(b,'A','A',101,1,2,1,10)
def test_replenishment_feature_raw_sequence():
 b=CausalMBOBook(); apply(b,'A','A',101,2,1); assert apply(b,'F','A',101,1,1,2).executed; apply(b,'A','A',101,1,2,3)
 assert b.depth['A'][101]==2

def test_previous_completed_rth_skips_weekend():
 assert previous_completed_rth("2026-07-27") == "2026-07-24"

def test_profile_poc_and_value_area_expand_one_tick_with_lower_ties():
 p = 5_000_000_000_000
 profile = volume_profile([(p - TICK, 25), (p, 50), (p + TICK, 25)])
 assert profile == {"high": p + TICK, "low": p - TICK, "poc": p, "vah": p, "val": p - TICK}
 assert volume_profile([(p, 10), (p + TICK, 10)])["poc"] == p

def _record(ts, action, side, price, size, order_id=1):
 return SimpleNamespace(ts_recv=ts, action=action, side=side, price=price, size=size, order_id=order_id)

def test_interaction_groups_timeout_exit_and_replenishment_absorption():
 d = Diagnostics(); day = "2026-07-21"; p = 5_000_000_000_000
 d.levels[day] = {"PRIOR_RTH_POC": p}; base = 1_784_000_000_000_000_000 + RTH_START * 1_000_000_000
 # Two F->A causal restores make the interaction absorption-labelled.
 d._touch(day, base, p, apply(CausalMBOBook(), 'T', 'A', p, 1, 99), TICK)
 for i in range(2):
  d._touch(day, base + (i + 1) * 1_000_000_000, p, SimpleNamespace(action='F', side='A', price=p, size=2, executed=True), TICK)
  d._touch(day, base + (i + 1) * 1_000_000_000 + 1, p, SimpleNamespace(action='A', side='A', price=p, size=2, executed=False), TICK)
 active = next(iter(d.active.values()))
 assert active.label() == "ABSORPTION_INTERACTION"
 d._touch(day, base + 3_000_000_000, p + 5 * TICK, SimpleNamespace(action='A', side='A', price=p + 5 * TICK, size=1, executed=False), TICK)
 d._touch(day, base + 5_000_000_000, p + 5 * TICK, SimpleNamespace(action='T', side='A', price=p + 5 * TICK, size=1, executed=True), TICK)
 d._touch(day, base + 6_000_000_000, p + 5 * TICK, SimpleNamespace(action='T', side='A', price=p + 5 * TICK, size=1, executed=True), TICK)
 assert d.completed[-1].termination == "VICINITY_EXIT_RESET"
 d._touch(day, base + 5_000_000_000, p, SimpleNamespace(action='T', side='A', price=p, size=1, executed=True), TICK)
 d._touch(day, base + 66_000_000_000, p, SimpleNamespace(action='T', side='A', price=p, size=1, executed=True), TICK)
 assert d.completed[-1].termination == "VICINITY_TIMEOUT"

def test_prior_profile_is_frozen_and_never_uses_current_or_future_rth():
 d = Diagnostics(); p = 5_000_000_000_000
 d.rth_prices["2026-07-24"][p] = 10
 d.rth_prices["2026-07-27"][p + 100 * TICK] = 999
 d.finish_day_context("2026-07-27")
 assert d.levels["2026-07-27"]["PRIOR_RTH_POC"] == p

def test_future_response_waits_for_causally_later_horizon_price():
 d = Diagnostics(); day = "2026-07-21"; p = 5_000_000_000_000; d.levels[day] = {"PRIOR_RTH_POC": p}
 base = 1_784_000_000_000_000_000 + RTH_START * 1_000_000_000
 d._touch(day, base, p, SimpleNamespace(action='T', side='A', price=p, size=1, executed=True), TICK)
 d._touch(day, base + 1, p + 5 * TICK, SimpleNamespace(action='T', side='A', price=p + 5 * TICK, size=1, executed=True), TICK)
 d._touch(day, base + 1_000_000_001, p + 5 * TICK, SimpleNamespace(action='T', side='A', price=p + 5 * TICK, size=1, executed=True), TICK)
 interaction = d.completed[-1]
 d._resolve_responses(base + 4_000_000_000, p + 9 * TICK)
 assert 5 not in interaction.responses
 d._resolve_responses(base + 6_000_000_002, p + 2 * TICK)
 assert interaction.responses[5] == -3

def test_continuous_multi_event_visit_is_one_interaction_and_book_events_do_not_restart():
 d = Diagnostics(); day = "2026-07-21"; p = 5_000_000_000_000
 d.levels[day] = {"PRIOR_RTH_POC": p}; base = 1_784_000_000_000_000_000 + RTH_START * 1_000_000_000
 for i, action in enumerate(("T", "A", "C", "M", "F")):
  executed = action in {"T", "F"}
  d._touch(day, base + i, p, SimpleNamespace(action=action, side="A", price=p, size=1, executed=executed), TICK)
 assert len(d.active) == 1 and next(iter(d.active.values())).events == 5

def test_current_sweep_revision_persists_one_lifecycle():
 d = Diagnostics(); day = "2026-07-21"; p = 5_000_000_000_000; base = 1_784_000_000_000_000_000 + RTH_START * 1_000_000_000
 d.current_extrema[day] = (p, p)
 d._touch(day, base, p, SimpleNamespace(action="T", side="A", price=p, size=1, executed=True), TICK)
 d.current_extrema[day] = (p, p + TICK)
 d._touch(day, base + 1, p + TICK, SimpleNamespace(action="T", side="A", price=p + TICK, size=1, executed=True), TICK)
 highs = [x for x in d.active.values() if x.level_name == "CURRENT_RTH_HIGH_SWEEP"]
 assert len(highs) == 1 and highs[0].level_price == p + TICK

def test_fixed_point_price_ticks_and_raw_price_isolation():
 d = Diagnostics(); p = 5_000_000_000_000
 assert d.es_price(p) == 5000.0 and DBN_FIXED_POINT_SCALE == 1_000_000_000
 assert d.response_ticks(p + TICK, p) == 1 and d.response_ticks(p - TICK, p) == -1
 day = "2026-07-21"; d.levels[day] = {"PRIOR_RTH_POC": p}
 d._touch(day, 100, p, SimpleNamespace(action="T", side="A", price=p, size=1, executed=True), TICK)
 d._touch(day, 101, p + 5*TICK, SimpleNamespace(action="T", side="A", price=p + 5*TICK, size=1, executed=True), TICK)
 d._touch(day, 1_000_000_101, p + 5*TICK, SimpleNamespace(action="T", side="A", price=p + 5*TICK, size=1, executed=True), TICK)
 interaction = d.completed[-1]
 d._resolve_responses(5_000_000_100, p + 999*TICK) # arbitrary order record cannot resolve responses
 assert 5 not in interaction.responses
 d._resolve_responses(6_000_000_102, p + 2*TICK)
 assert interaction.responses[5] == -3

def test_response_sanity_violation_is_explicit_and_excluded():
 d = Diagnostics(); day = "2026-07-21"; p = 5_000_000_000_000; d.levels[day] = {"PRIOR_RTH_POC": p}
 d._touch(day, 100, p, SimpleNamespace(action="T", side="A", price=p, size=1, executed=True), TICK)
 d._touch(day, 101, p + 5*TICK, SimpleNamespace(action="T", side="A", price=p + 5*TICK, size=1, executed=True), TICK)
 d._touch(day, 1_000_000_101, p + 5*TICK, SimpleNamespace(action="T", side="A", price=p + 5*TICK, size=1, executed=True), TICK)
 interaction = d.completed[-1]
 d._resolve_responses(6_000_000_102, p + 506*TICK)
 assert interaction.response_violations[5] == "RESPONSE_SANITY_VIOLATION" and 5 not in interaction.responses

def test_refined_report_emits_causal_example_schema_without_legacy_passive_side(tmp_path):
 d = Diagnostics(); day = "2026-07-21"; p = 5_000_000_000_000
 d.levels[day] = {"PRIOR_RTH_POC": p}
 d._touch(day, 1_784_000_000_000_000_000, p, SimpleNamespace(action="F", side="A", price=p, size=1, executed=True), TICK)
 d._touch(day, 1_784_000_001_000_000_000, p, SimpleNamespace(action="A", side="A", price=p, size=1, executed=False), TICK)
 d.finalize()
 write_reports(tmp_path, sha256="A" * 64, dbn_bytes=1,
               metadata=MetadataSummary("GLBX.MDP3", "mbo", "ESU6", 42140870, 1, 2),
               diagnostics=d, integrity="PASS")
 summary = json.loads((tmp_path / "refined-feature-diagnostic-summary.json").read_text())
 assert summary["tier_counts"]["RAW_INTERACTION"] == 1
 assert "passive_side" not in (tmp_path / "feature-diagnostic-report.md").read_text()
