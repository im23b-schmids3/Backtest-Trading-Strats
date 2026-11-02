from __future__ import annotations

import json
import unittest

import numpy as np

from research_pipeline.cme_orderflow_absorption_l2_v1.es_four_mechanism_intraday_horizon_v1 import (
    BlockExtrema, accounting_ticks, cost_decomposition, _event_horizons,
    _find_risk_paths, _has_gap, _union_record_schema,
    _episode_representative_ids, _apply_nonoverlap_schedule,
)


class AccountingTests(unittest.TestCase):
    def test_long_marketable_accounting_charges_spread_slippage_and_fee_once(self):
        # 1-tick spread at entry and exit: half-spreads sum to 1 tick.
        # Quote-to-quote gross midpoint change is +4 ticks; costs are 1 spread,
        # 2 slippage ticks, and 0.48 tick commission => +0.52 ticks net.
        out = cost_decomposition(1, 100.0, 100.0, 100.25, 101.0, 101.25)
        self.assertAlmostEqual(out["total_spread_ticks"], 1.0)
        self.assertAlmostEqual(out["entry_slippage_ticks"] + out["exit_slippage_ticks"], 2.0)
        self.assertAlmostEqual(out["commission_ticks"], 0.48)
        self.assertAlmostEqual(out["net_executable_ticks"], 0.52)
        self.assertAlmostEqual(out["net_executable_ticks"], out["reconciled_net_ticks"])
        self.assertAlmostEqual(out["net_executable_usd"], 6.5)

    def test_short_marketable_sign_and_spread(self):
        out = cost_decomposition(-1, 100.0, 100.0, 100.25, 99.0, 99.25)
        self.assertAlmostEqual(out["gross_from_entry_midpoint_ticks"], 4.0)
        self.assertAlmostEqual(out["net_executable_ticks"], 0.52)
        self.assertAlmostEqual(out["net_executable_ticks"], out["reconciled_net_ticks"])

    def test_zero_slippage_and_single_round_trip_commission(self):
        gross = accounting_ticks(1, 100.25, 101.0, 0, 0, commission_usd=0)
        with_fee = accounting_ticks(1, 100.25, 101.0, 0, 0, commission_usd=6)
        self.assertAlmostEqual(gross, 3.0)
        self.assertAlmostEqual(gross - with_fee, 0.48)
        self.assertAlmostEqual(accounting_ticks(-1, 100.0, 99.25, 0, 0, commission_usd=0), 3.0)

    def test_latency_is_separate_and_not_recharged_as_slippage(self):
        out = cost_decomposition(1, 100.0, 100.25, 100.50, 101.0, 101.25)
        self.assertAlmostEqual(out["latency_midpoint_ticks"], 1.5)
        # Changing the trigger midpoint changes only the reported latency move.
        other = cost_decomposition(1, 99.75, 100.25, 100.50, 101.0, 101.25)
        self.assertAlmostEqual(other["net_executable_ticks"], out["net_executable_ticks"])
        self.assertAlmostEqual(other["latency_midpoint_ticks"] - out["latency_midpoint_ticks"], 1.0)

    def test_invalid_direction_or_crossed_quote_fails_closed(self):
        with self.assertRaises(ValueError): accounting_ticks(0, 1, 2)
        with self.assertRaises(ValueError): cost_decomposition(1, 1, 2, 1, 1, 2)


class PathIndexTests(unittest.TestCase):
    def setUp(self):
        self.bid=np.asarray([100,101,99,102,98,103,97],dtype=np.int64)
        self.ask=self.bid+1
        self.idx=BlockExtrema(self.bid,self.ask)

    def test_range_extrema_and_first_crossing(self):
        self.assertEqual(self.idx.extrema("bid_min",1,5),98)
        self.assertEqual(self.idx.extrema("ask_max",1,5),104)
        self.assertEqual(self.idx.first_crossing("bid_min",1,6,98),4)
        self.assertEqual(self.idx.first_crossing("ask_max",1,6,103),3)
        self.assertIsNone(self.idx.first_crossing("bid_min",1,3,98))

    def test_entry_side_and_exit_side_formula_for_both_directions(self):
        # Marketable entry side is ask for long and bid for short; close side
        # is bid for long and ask for short, as represented by the accounting API.
        self.assertAlmostEqual(accounting_ticks(1, 100.25, 101.0, 0, 0, 0), 3.0)
        self.assertAlmostEqual(accounting_ticks(-1, 100.0, 99.25, 0, 0, 0), 3.0)

    def test_block_tree_matches_brute_force_across_multiple_blocks_and_ties(self):
        rng=np.random.default_rng(20261028)
        bid=rng.integers(100_000,110_000,size=900,dtype=np.int64)
        ask=bid+250
        bid[257]=bid[256]  # equal-time price value tie must retain earliest ordinal/index
        idx=BlockExtrema(bid,ask)
        for left,right in [(0,899),(1,255),(255,520),(256,512),(257,899),(511,777)]:
            self.assertEqual(idx.extrema("bid_min",left,right),int(np.min(bid[left:right+1])))
            self.assertEqual(idx.extrema("ask_max",left,right),int(np.max(ask[left:right+1])))
            lo=int(np.min(bid[left:right+1]))+1
            hits=np.flatnonzero(bid[left:right+1]<=lo)
            expected=None if not len(hits) else left+int(hits[0])
            self.assertEqual(idx.first_crossing("bid_min",left,right,lo),expected)


class FrozenEpisodeSelectionTests(unittest.TestCase):
    def test_only_frozen_episode_representatives_enter_deduplicated_population(self):
        rows=[
            {"date":"2025-03-03","mechanism":"M1","event_id":"e1","episode_id":"p1","is_episode_representative":True},
            {"date":"2025-03-03","mechanism":"M1","event_id":"e2","episode_id":"p1","is_episode_representative":False},
            {"date":"2025-03-03","mechanism":"M2","event_id":"e3","episode_id":"p2","is_episode_representative":True},
        ]
        selected=_episode_representative_ids(rows,["2025-03-03"])
        self.assertEqual(selected,{"2025-03-03":{"e1","e3"}})

    def test_episode_ledger_without_exactly_one_representative_fails_closed(self):
        rows=[{"date":"2025-03-03","event_id":"e1","episode_id":"p1","is_episode_representative":False}]
        with self.assertRaises(RuntimeError):
            _episode_representative_ids(rows,["2025-03-03"])

    def test_nonoverlap_schedule_uses_representatives_only_and_is_outcome_blind(self):
        rows=[
            {"event_id":"rep","mechanism":"M1","horizon_seconds":60,"trigger_ts_recv":1,"eligible":True,"entry_ts_target":10},
            {"event_id":"later","mechanism":"M1","horizon_seconds":60,"trigger_ts_recv":2,"eligible":True,"entry_ts_target":20},
            {"event_id":"nonrep","mechanism":"M1","horizon_seconds":60,"trigger_ts_recv":0,"eligible":True,"entry_ts_target":0},
        ]
        _apply_nonoverlap_schedule(rows,{"rep","later"})
        by_id={r["event_id"]:r for r in rows}
        self.assertTrue(by_id["rep"]["nonoverlap_selected"])
        self.assertFalse(by_id["later"]["nonoverlap_selected"])
        self.assertEqual(by_id["later"]["nonoverlap_exclusion_reason"],"OVERLAPS_EARLIER_SCHEDULED_HORIZON")
        self.assertFalse(by_id["nonrep"]["nonoverlap_selected"])
        self.assertEqual(by_id["nonrep"]["nonoverlap_exclusion_reason"],"NOT_EPISODE_REPRESENTATIVE")


class HorizonAndStopTests(unittest.TestCase):
    def _fixture(self, *, stop_first=True, gap=False):
        ns=1_000_000_000
        # Entry long at 100.50; a drop to 98.25 crosses the 8 tick stop.
        times=np.asarray([0,2_000_000,1*ns,60*ns,300*ns,900*ns,1800*ns,3600*ns,7200*ns],dtype=np.int64)
        later=101_000_000_000 if stop_first else 103_000_000_000
        bid=np.asarray([100_000_000_000,100_000_000_000,
                        98_250_000_000 if stop_first else later,
                        later,later,later,later,later,later],dtype=np.int64)
        ask=bid+250_000_000
        gaps=[(200_000_000_000,201_000_000_000,"TEST_GAP")] if gap else []
        market={"t":times,"ord":np.arange(len(times),dtype=np.int64),"bid":bid,"ask":ask,
                "gaps":gaps,"gap_starts":[x[0] for x in gaps]}
        event={"date":"2025-03-03","mechanism":"M1","event_id":"e1","ts_recv":0,
               "source_ordinal":0,"direction":1,"features":{}}
        return event,market,BlockExtrema(bid,ask)

    def test_fixed_horizon_uses_entry_clock_and_censors_source_gap(self):
        event,market,blocks=self._fixture(gap=True)
        rows,risk=_event_horizons(event,market,blocks,8000*1_000_000_000,8000*1_000_000_000)
        self.assertEqual(rows[0]["horizon_seconds"],60)
        self.assertEqual(rows[0]["eligible"],True)
        self.assertFalse(rows[2]["eligible"])  # 15m horizon crosses the injected gap
        self.assertEqual(rows[2]["censor_reason"],"INVALID_SOURCE_INTERVAL_OR_GAP")
        self.assertEqual(risk["entry_ts_target"],2_000_000)

    def test_stop_first_and_target_first_are_resolved_from_marketable_sides(self):
        event,market,blocks=self._fixture(stop_first=True)
        _,risk=_event_horizons(event,market,blocks,8000*1_000_000_000,8000*1_000_000_000)
        paths=_find_risk_paths(event,risk,market,blocks,8000*1_000_000_000)
        eight=next(x for x in paths if x["risk_framework"]=="FIXED_8T")
        self.assertIsNotNone(eight["stop_first_ts"])
        self.assertEqual(eight["first_passage_by_horizon_and_R"]["60:1R"],"STOP_FIRST")
        event,market,blocks=self._fixture(stop_first=False)
        _,risk=_event_horizons(event,market,blocks,8000*1_000_000_000,8000*1_000_000_000)
        paths=_find_risk_paths(event,risk,market,blocks,8000*1_000_000_000)
        eight=next(x for x in paths if x["risk_framework"]=="FIXED_8T")
        self.assertEqual(eight["first_passage_by_horizon_and_R"]["60:1R"],"TARGET_FIRST")

    def test_stop_exit_before_later_gap_remains_resolved(self):
        event,market,blocks=self._fixture(stop_first=True,gap=True)
        _,risk=_event_horizons(event,market,blocks,8000*1_000_000_000,8000*1_000_000_000)
        paths=_find_risk_paths(event,risk,market,blocks,8000*1_000_000_000)
        eight=next(x for x in paths if x["risk_framework"]=="FIXED_8T")
        self.assertTrue(eight["stop_constrained_by_horizon"]["300"]["eligible"])
        self.assertTrue(eight["stop_constrained_by_horizon"]["300"]["stop_first"])
        self.assertEqual(eight["first_passage_by_horizon_and_R"]["300:1R"],"STOP_FIRST")

    def test_gap_lookup_is_interval_aware(self):
        gaps=[(10,20,"G")];starts=[10]
        self.assertEqual(_has_gap(gaps,starts,15,30),"G")
        self.assertIsNone(_has_gap(gaps,starts,21,30))

    def test_sparse_censored_first_row_preserves_union_of_arrow_fields(self):
        rows=_union_record_schema([{"eligible":False,"reason":"GAP","nested":{"eligible":False,"reason":"GAP"}},
                                   {"eligible":True,"gross_ticks":1.25,"nested":{"eligible":True,"net_r":-1.0}}])
        self.assertIn("gross_ticks",rows[0]);self.assertIsNone(rows[0]["gross_ticks"])
        self.assertEqual(rows[1]["gross_ticks"],1.25)
        self.assertIn("net_r",rows[0]["nested"]);self.assertIsNone(rows[0]["nested"]["net_r"])


if __name__ == "__main__":
    unittest.main()
