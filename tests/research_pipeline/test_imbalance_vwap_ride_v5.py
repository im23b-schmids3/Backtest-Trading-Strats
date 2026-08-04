from decimal import Decimal
from research_pipeline.imbalance_vwap_ride.v5_models import *
from research_pipeline.imbalance_vwap_ride.v5_data import annotate_price_scaled_bins,validate_scaled_footprints,aggregate_price_scaled_footprints,maximal_stacked_zones
from research_pipeline.imbalance_vwap_ride.v5_adapter import ImbalanceVWAPRideV5Adapter
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
