"""Sealed V5 price-scaled-bin contract (local research only)."""
from __future__ import annotations

import math
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

from pydantic import Field, model_validator

from ..schemas.strategy_spec import StrictModel
from .artifacts import sha256_value

STRATEGY_ID = "ImbalanceVWAPRide.BTC_LONG_ONLY_V5_PRICE_SCALED_BINS"
ADAPTER_ID = "imbalance-vwap-ride-btc-long-only-v5-1"
EVIDENCE = "POST_HOC_V5_PRICE_SCALED_BIN_RESEARCH"
SPEC_VERSION = ADAPTER_ID
PHASE_A_MONTHS = tuple([f"2023-{m:02d}" for m in range(1, 13)] + ["2024-01"])
PHASE_B_MONTHS = tuple(f"2024-{m:02d}" for m in range(2, 8))
CANDIDATE_REGISTRY = (("V5-A-SCALED-BIN-1P5R", Decimal("1.5")), ("V5-B-SCALED-BIN-2P0R", Decimal("2.0")), ("V5-C-SCALED-BIN-2P5R", Decimal("2.5")))

def scaled_bin_size(previous_close: Decimal | str | None) -> Decimal | None:
    if previous_close is None: return None
    value = Decimal(str(previous_close)) * Decimal("0.001") / 5
    return min(Decimal("100"), max(Decimal("20"), value.quantize(Decimal("1"), rounding=ROUND_HALF_UP) * 5))

class ImbalanceVWAPRideV5Config(StrictModel):
    candidate_id: str = CANDIDATE_REGISTRY[0][0]
    target_r_multiple: Decimal = Decimal("1.5")
    min_bin_volume_btc: Decimal = Decimal("35")
    min_imbalance_ratio: Decimal = Decimal("3")
    stacked_bins: int = 3
    move_away_bars: int = 1
    zone_expiry_bars: int = 36
    stop_buffer_bins: int = 2
    maximum_active_zones: int = 3
    maximum_trades_per_utc_day: int = 1
    vwap_slope_bars: int = 24
    direction: str = "LONG_ONLY"
    symbol: str = "BTCUSDT"
    quantity_btc: Decimal = Decimal("0.001")
    price_tick: Decimal = Decimal("0.1")
    quantity_step: Decimal = Decimal("0.001")
    taker_fee_rate: Decimal = Decimal("0.0005")
    market_slippage_ticks: int = 1
    stop_slippage_ticks: int = 2
    same_bar_policy: str = "STOP_FIRST"
    @model_validator(mode="after")
    def sealed(self):
        expected = dict(CANDIDATE_REGISTRY).get(self.candidate_id)
        if expected is None or self.target_r_multiple != expected: raise ValueError("sealed V5 candidate parameters do not match candidate_id")
        if self.direction != "LONG_ONLY" or self.symbol != "BTCUSDT" or self.min_bin_volume_btc != 35 or self.min_imbalance_ratio != 3 or self.stacked_bins != 3 or self.vwap_slope_bars != 24: raise ValueError("sealed V5 invariant violation")
        return self
    def parameter_payload(self) -> dict[str, Any]: return self.model_dump(mode="json")
    def frozen_payload(self) -> dict[str, Any]: return {"candidate_id":self.candidate_id,"parameters":self.parameter_payload(),"bin_formula":"clamp(20,100,round_half_up(previous_close*0.001/5)*5)","phase_a_months":list(PHASE_A_MONTHS),"phase_b_months":list(PHASE_B_MONTHS)}

def preregistered_candidates() -> list[ImbalanceVWAPRideV5Config]: return [ImbalanceVWAPRideV5Config(candidate_id=k,target_r_multiple=v) for k,v in CANDIDATE_REGISTRY]
def candidate_registry_payload() -> list[dict[str,str]]: return [{"candidate_id":k,"target_r_multiple":str(v)} for k,v in CANDIDATE_REGISTRY]
def candidate_registry_hash() -> str: return sha256_value(candidate_registry_payload())
def _d(v: Any) -> Decimal: return Decimal(str(v))
def _months(metrics: dict[str,Any], names: tuple[str,...]) -> dict[str,dict[str,Any]]: return {m:metrics.get("months",{}).get(m,{}) for m in names}
def _common(metrics: dict[str,Any]) -> bool:
    f=metrics.get("funnel_reconciliation",{}); l=metrics.get("long_only_reconciliation",{})
    return bool(f.get("reconciles")) and f.get("proposed_setups")==sum(int(f.get(k,0)) for k in ("invalid_setups","non_executable_setups","compliance_blocks","executed_trades")) and bool(l.get("reconciles")) and int(l.get("short_trades",0))==int(l.get("short_setups",0))==0 and _d(l.get("short_pnl",0))==0
def phase_a_gate(metrics: dict[str,Any]) -> dict[str,Any]:
    ms=_months(metrics,PHASE_A_MONTHS); n=lambda m:int(ms[m].get("executed_trades",0)); pos=[max(_d(x.get("net_pnl",0)),Decimal()) for x in ms.values()]; total=sum(pos,Decimal()); quarters=[sum((_d(ms[m].get("net_pnl",0)) for m in PHASE_A_MONTHS[i:i+3]),Decimal()) for i in range(0,12,3)]; halves=[sum((_d(ms[m].get("net_pnl",0)) for m in group),Decimal()) for group in (PHASE_A_MONTHS[:6],PHASE_A_MONTHS[6:12])]
    checks={"minimum_52_trades":int(metrics.get("executed_trades",0))>=52,"ten_active_months":sum(n(m)>0 for m in ms)>=10,"eight_months_with_four_trades":sum(n(m)>=4 for m in ms)>=8,"at_most_three_zero_months":sum(n(m)==0 for m in ms)<=3,"net_pnl_positive":_d(metrics.get("net_pnl",0))>0,"net_pf_above_1_10":_d(metrics.get("net_profit_factor",0))>Decimal("1.10"),"average_net_r_positive":_d(metrics.get("average_net_r",0))>0,"four_target_hits":int(metrics.get("target_hits",0))>=4,"positive_hit_rate":_d(metrics.get("target_hit_rate",0))>0,"mfe_1r_20pct":_d(metrics.get("mfe_at_least_1r_rate",0))>=Decimal(".2"),"finite_drawdown":math.isfinite(float(metrics.get("maximum_drawdown",float("nan")))),"funnel_long_only_costs_valid":_common(metrics),"best_month_60pct":not total or max(pos)/total<=Decimal(".6"),"best_three_85pct":not total or sum(sorted(pos,reverse=True)[:3])/total<=Decimal(".85"),"best_five_65pct":_d(metrics.get("best_five_positive_pnl_contribution",1))<Decimal(".65"),"three_nonnegative_2023_quarters":sum(x>=0 for x in quarters)>=3,"both_2023_halves_positive":all(x>0 for x in halves),"jan_2024_reported_and_limited":"2024-01" in ms and (not total or max(_d(ms["2024-01"].get("net_pnl",0)),Decimal())/total<=Decimal(".5"))}
    return {"passed":all(checks.values()),"checks":checks,"nonnegative_quarter_count":sum(x>=0 for x in quarters),"both_halves_positive":all(x>0 for x in halves),"best_month_concentration":str(max(pos,default=Decimal())/total if total else Decimal())}
def rank_phase_a_candidates(candidates):
    passed=[(c,m,phase_a_gate(m)) for c,m in candidates if phase_a_gate(m)["passed"]]
    return [{"rank":i,"candidate_id":c.candidate_id,"config":c,"metrics":m,"gate":g,"rank_trace":g} for i,(c,m,g) in enumerate(sorted(passed,key=lambda x:(-x[2]["nonnegative_quarter_count"],x[2]["best_month_concentration"],-_d(x[1].get("net_profit_factor",0)),_d(x[1].get("maximum_drawdown",0)),-_d(x[1].get("average_net_r",0)),-_d(x[1].get("target_hit_rate",0)),-int(x[1].get("executed_trades",0)),x[0].target_r_multiple,x[0].candidate_id)),1)]
def phase_b_gate(metrics):
    ms=_months(metrics,PHASE_B_MONTHS); n=lambda m:int(ms[m].get("executed_trades",0)); pnls=[sum((_d(ms[m].get("net_pnl",0)) for m in g),Decimal()) for g in (PHASE_B_MONTHS[:3],PHASE_B_MONTHS[3:])]; full=_d(metrics.get("net_pnl",0)); checks={"minimum_24_trades":int(metrics.get("executed_trades",0))>=24,"five_active_months":sum(n(m)>0 for m in ms)>=5,"four_months_with_three_trades":sum(n(m)>=3 for m in ms)>=4,"net_pnl_positive":full>0,"net_pf_above_1_05":_d(metrics.get("net_profit_factor",0))>Decimal("1.05"),"average_net_r_positive":_d(metrics.get("average_net_r",0))>0,"two_target_hits":int(metrics.get("target_hits",0))>=2,"mfe_1r_15pct":_d(metrics.get("mfe_at_least_1r_rate",0))>=Decimal(".15"),"integrity":_common(metrics) and bool(metrics.get("hashes_valid")) and bool(metrics.get("costs_valid")),"subperiods":any(x>0 for x in pnls) and all(x>=-(full/2) for x in pnls)}; return {"passed":all(checks.values()),"status":"LOCKED_TEST_PASSED" if all(checks.values()) else "LOCKED_TEST_FAILED","checks":checks}
