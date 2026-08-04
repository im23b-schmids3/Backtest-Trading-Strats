from decimal import Decimal
from research_pipeline.imbalance_vwap_ride.v5_models import *
from research_pipeline.imbalance_vwap_ride.v5_data import annotate_price_scaled_bins,validate_scaled_footprints,aggregate_price_scaled_footprints,maximal_stacked_zones
from research_pipeline.imbalance_vwap_ride.v5_adapter import ImbalanceVWAPRideV5Adapter
from research_pipeline.imbalance_vwap_ride.v5_strategy import simulate_v5_long_trade
from research_pipeline.imbalance_vwap_ride.v5_runner import normalize_source_bar_timestamp,deterministic_setup_id,validate_v5_setup_audit
from research_pipeline.imbalance_vwap_ride import v5_runner
from datetime import datetime, timedelta, timezone
def test_v5_identity_registry_and_rounding():
 assert STRATEGY_ID=="ImbalanceVWAPRide.BTC_LONG_ONLY_V5_PRICE_SCALED_BINS"; assert [x.candidate_id for x in preregistered_candidates()]==[x[0] for x in CANDIDATE_REGISTRY]; assert scaled_bin_size("99999")==Decimal("100") and scaled_bin_size("100000")==Decimal("100") and scaled_bin_size("10000")==Decimal("20") and scaled_bin_size("123456")==Decimal("100")
def test_v5_preceding_close_only_and_local_capabilities():
 bars=[{"bar_start_utc":"a","close":"50000"},{"bar_start_utc":"b","close":"60000"}]; rows=annotate_price_scaled_bins(bars,[{"bar_start_utc":"a","price":"1"},{"bar_start_utc":"b","price":"50123"}]); assert len(rows)==1 and rows[0]["bin_size_usd"]==Decimal("50") and validate_scaled_footprints(rows)["valid"]; caps=ImbalanceVWAPRideV5Adapter().capabilities();assert not caps["live_orders"] and not caps["external_raw_trade_transmission"]

def test_v5_maximal_sequences_are_same_bar_and_same_bin_size():
 rows=[]
 for price in (100,120,140,160): rows.append({"bar_start_utc":"a","price_bin":Decimal(price),"bin_size_usd":Decimal("20"),"buy_volume_btc":Decimal("35"),"sell_volume_btc":Decimal(),"total_volume_btc":Decimal("35")})
 rows.append({"bar_start_utc":"a","price_bin":Decimal("180"),"bin_size_usd":Decimal("25"),"buy_volume_btc":Decimal("50"),"sell_volume_btc":Decimal(),"total_volume_btc":Decimal("50")})
 zones=maximal_stacked_zones(rows)
 assert zones==[{"source_bar_start_utc":"a","bottom":Decimal("100"),"top":Decimal("180"),"bin_size_usd":Decimal("20"),"stacked_bins":4}]

def test_v5_aggregation_preserves_buy_sell_and_delta():
 rows=[{"bar_start_utc":"a","price_bin":Decimal("100"),"bin_size_usd":Decimal("20"),"quantity":"2","is_buyer_maker":False},{"bar_start_utc":"a","price_bin":Decimal("100"),"bin_size_usd":Decimal("20"),"quantity":"1","is_buyer_maker":True}]
 footprint=aggregate_price_scaled_footprints(rows)[0]
 assert footprint["total_volume_btc"]==Decimal("3") and footprint["delta_btc"]==Decimal("1")

def test_v5_nonempty_zone_maps_json_timestamp_to_arrow_bar_without_drop():
 stamp=datetime(2023,1,1,0,0,tzinfo=timezone.utc)
 rows=[{"bar_start_utc":stamp,"price_bin":Decimal(p),"bin_size_usd":Decimal("20"),"buy_volume_btc":Decimal("35"),"sell_volume_btc":Decimal(),"total_volume_btc":Decimal("35")} for p in (100,120,140)]
 zone=maximal_stacked_zones(rows)[0]
 assert normalize_source_bar_timestamp(zone["source_bar_start_utc"]) == stamp

def test_v5_stop_uses_the_source_zone_bin_and_wins_ambiguous_bar():
 start=datetime(2023,1,1,tzinfo=timezone.utc)
 signal={"bar_start_utc":start,"bar_end_utc":start+timedelta(minutes=5),"open":"200","high":"200","low":"200","close":"200"}
 entry={"bar_start_utc":start+timedelta(minutes=5),"bar_end_utc":start+timedelta(minutes=10),"open":"200","high":"700","low":"99","close":"200"}
 state,trade=simulate_v5_long_trade(zone={"direction":"LONG","setup_id":"synthetic-setup","bottom":"150","top":"200","bin_size_usd":"25"},signal_bar=signal,entry_index=1,bars=[signal,entry],config=ImbalanceVWAPRideV5Config())
 assert state=="TRADE_EXECUTED" and trade["setup_id"]=="synthetic-setup" and trade["initial_stop_price"]=="100" and trade["exit_reason"]=="STOP_FIRST_AMBIGUITY" and trade["source_bar_bin_size_usd"]=="25"

def test_v5_setup_audit_rearming_does_not_duplicate_a_setup_and_all_candidates_reconcile():
 zone={"source_bar_start_utc":"2023-01-01T00:00:00+00:00","bottom":Decimal("100"),"top":Decimal("160"),"bin_size_usd":Decimal("20"),"stacked_bins":3}
 zone["setup_id"]=deterministic_setup_id(zone)
 assert deterministic_setup_id(zone)==zone["setup_id"]
 for candidate in preregistered_candidates():
  events=[{"event":"PROPOSED_SETUP","setup_id":zone["setup_id"],"candidate_id":candidate.candidate_id},{"event":"ARMED","setup_id":zone["setup_id"],"candidate_id":candidate.candidate_id},{"event":"RETEST_REGIME_REJECTED","setup_id":zone["setup_id"],"candidate_id":candidate.candidate_id},{"event":"REARMED","setup_id":zone["setup_id"],"candidate_id":candidate.candidate_id},{"event":"TRADE_EXECUTED","setup_id":zone["setup_id"],"candidate_id":candidate.candidate_id}]
  trades=[{"setup_id":zone["setup_id"],"candidate_id":candidate.candidate_id}]
  outcomes=[{"setup_id":zone["setup_id"],"disposition":"TRADE_EXECUTED","candidate_id":candidate.candidate_id}]
  validate_v5_setup_audit([zone],events,trades,outcomes)

def test_v5_setup_id_is_audit_only_and_does_not_change_trade_outcomes():
 start=datetime(2023,1,1,tzinfo=timezone.utc)
 signal={"bar_start_utc":start,"bar_end_utc":start+timedelta(minutes=5),"open":"200","high":"200","low":"200","close":"200"}
 entry={"bar_start_utc":start+timedelta(minutes=5),"bar_end_utc":start+timedelta(minutes=10),"open":"200","high":"200","low":"99","close":"200"}
 base={"direction":"LONG","setup_id":"baseline-audit-id","bottom":"150","top":"200","bin_size_usd":"25"}
 state,without_id=simulate_v5_long_trade(zone=base,signal_bar=signal,entry_index=1,bars=[signal,entry],config=ImbalanceVWAPRideV5Config())
 state_with,with_id=simulate_v5_long_trade(zone={**base,"setup_id":"audit-only"},signal_bar=signal,entry_index=1,bars=[signal,entry],config=ImbalanceVWAPRideV5Config())
 assert state==state_with and {k:v for k,v in with_id.items() if k!="setup_id"}=={k:v for k,v in without_id.items() if k!="setup_id"}

def test_v5_candidate_cli_requires_absolute_paths_before_any_execution(tmp_path):
 try:
  v5_runner.run_v5_candidate_cli(phase_a_manifest="relative.json",artifact_root=tmp_path)
 except ValueError as exc:
  assert "clean committed git tree" in str(exc)
 else:
  raise AssertionError("a dirty test worktree must prevent V5 execution")

def test_v5_phase_a_manifest_rejects_non_absolute_path():
 try:
  v5_runner.validate_pinned_v5_phase_a_manifest("relative.json")
 except ValueError as exc:
  assert "absolute path" in str(exc)
 else:
  raise AssertionError("relative manifests must be rejected")
