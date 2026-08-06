from __future__ import annotations

import hashlib
import json
from bisect import bisect_left, bisect_right
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable

SPEC_HASH = "16482273F7842FBB76BA022C07D3055A3126379CA789EDB7FF79F168EC8F5212"
CANDIDATES = {"HTFLFVG-V1-MIN2P5R": 2.5, "HTFLFVG-V1-MIN3P0R": 3.0, "HTFLFVG-V1-MIN4P0R": 4.0}
REQUIRED_ARTIFACTS = ("sealed-specification.json", "candidate-registry.json", "data-manifest.json", "derived-timeframe-manifest.json", "configuration.json", "levels.json", "events.json", "trades.json", "setup_outcomes.json", "monthly_metrics.json", "report.json", "gates.json", "selection_report.json", "freeze.json", "integrity-manifest.json", "final_report.json")

class Direction(StrEnum): LONG="LONG"; SHORT="SHORT"
class Bias(StrEnum): BULLISH="BULLISH"; BEARISH="BEARISH"; NEUTRAL="NEUTRAL"
class EventScope(StrEnum): SETUP="SETUP"; TRADE="TRADE"; LEVEL="LEVEL"; GLOBAL="GLOBAL"
class TerminalDisposition(StrEnum):
    TRADE_EXECUTED="TRADE_EXECUTED"; DAILY_BIAS_REJECTED="DAILY_BIAS_REJECTED"; NO_ACTIVE_HTF_LEVEL="NO_ACTIVE_HTF_LEVEL"; DUPLICATE_LEVEL_BLOCKED="DUPLICATE_LEVEL_BLOCKED"; SWEEP_DEPTH_REJECTED="SWEEP_DEPTH_REJECTED"; SWEEP_RECLAIM_REJECTED="SWEEP_RECLAIM_REJECTED"; SWEEP_WICK_REJECTED="SWEEP_WICK_REJECTED"; MSS_WINDOW_EXPIRED="MSS_WINDOW_EXPIRED"; MSS_STRUCTURE_REJECTED="MSS_STRUCTURE_REJECTED"; DISPLACEMENT_REJECTED="DISPLACEMENT_REJECTED"; FVG_NOT_FORMED="FVG_NOT_FORMED"; FVG_TOO_SMALL="FVG_TOO_SMALL"; PENDING_ORDER_EXPIRED="PENDING_ORDER_EXPIRED"; PRE_ENTRY_SWEEP_INVALIDATED="PRE_ENTRY_SWEEP_INVALIDATED"; BIAS_INVALIDATED_BEFORE_ENTRY="BIAS_INVALIDATED_BEFORE_ENTRY"; STOP_DISTANCE_REJECTED="STOP_DISTANCE_REJECTED"; PROJECTED_RR_REJECTED="PROJECTED_RR_REJECTED"; ACTIVE_POSITION_BLOCKED="ACTIVE_POSITION_BLOCKED"; ACTIVE_ORDER_BLOCKED="ACTIVE_ORDER_BLOCKED"; SESSION_ENTRY_BLOCKED="SESSION_ENTRY_BLOCKED"; SESSION_ENDED="SESSION_ENDED"; SUPERSEDED_BY_NEW_SWEEP="SUPERSEDED_BY_NEW_SWEEP"

@dataclass(frozen=True)
class Bar:
    time: datetime; open: float; high: float; low: float; close: float; volume: float=0.; id: str=""
    def __post_init__(self):
        if self.time.tzinfo is None or self.time.utcoffset() != timedelta(0): raise ValueError("bar timestamps must be UTC")
        if not self.low <= min(self.open,self.close) <= max(self.open,self.close) <= self.high: raise ValueError("invalid OHLC")
        if self.time.second or self.time.microsecond or self.time.minute % 5: raise ValueError("source bars must be UTC 5-minute interval starts")

@dataclass(frozen=True)
class Candidate: id: str; minimum_tp2_r: float
@dataclass
class Event:
    event_id: str; time: str; sequence: int; candidate_id: str; decision: str; setup_id: str|None=None; reason_code: str|None=None; inputs: dict[str,Any]=field(default_factory=dict); scope: EventScope=EventScope.GLOBAL; trade_id: str|None=None
@dataclass
class Level:
    level_id: str; side: str; price: float; source_time: datetime; confirmation_time: datetime; consumed: bool=False

def deterministic_id(kind: str, *parts: object) -> str:
    return f"{kind}_{hashlib.sha256('|'.join(map(str,parts)).encode()).hexdigest()[:20]}"

EVENT_SCOPES: dict[str, EventScope] = {
    "SETUP_PROPOSED": EventScope.SETUP, "MSS_CONFIRMED": EventScope.SETUP,
    "DISPLACEMENT_CONFIRMED": EventScope.SETUP, "FVG_CONFIRMED": EventScope.SETUP,
    "ORDER_ACTIVATED": EventScope.SETUP, "TERMINAL_DISPOSITION": EventScope.SETUP,
    "ENTRY_FILLED": EventScope.TRADE, "EXIT_FILLED": EventScope.TRADE,
    "LEVEL_CONFIRMED": EventScope.LEVEL, "AGGREGATION_CONFIRMED": EventScope.GLOBAL,
    "DAILY_BIAS_CONFIRMED": EventScope.GLOBAL,
}

def event_scope(decision: str) -> EventScope:
    try: return EVENT_SCOPES[decision]
    except KeyError as exc: raise ValueError(f"RECONCILIATION_ERROR: unknown event decision {decision}") from exc

class ClosedBarAggregator:
    """In-memory UTC aggregation. Gaps are legitimate; a gapped bucket is never emitted."""
    periods={"15m":3,"4h":48,"1d":288}
    def __init__(self): self._bars: list[Bar]=[]
    def push(self, bar: Bar) -> dict[str,Bar]:
        if self._bars and bar.time <= self._bars[-1].time: raise ValueError("source timestamps must be strictly increasing")
        self._bars.append(bar); emitted={}
        for name, count in self.periods.items():
            start = _bucket_start(bar.time,name); times=[start+timedelta(minutes=5*i) for i in range(count)]
            if bar.time != times[-1]: continue
            by_time={x.time:x for x in self._bars[-count:]}
            if any(t not in by_time for t in times): continue
            group=[by_time[t] for t in times]
            emitted[name]=Bar(start,group[0].open,max(x.high for x in group),min(x.low for x in group),group[-1].close,sum(x.volume for x in group),deterministic_id("derived",name,start.isoformat(),bar.time.isoformat()))
        return emitted

def _bucket_start(t:datetime, tf:str)->datetime:
    if tf=="15m": return t.replace(minute=t.minute-t.minute%15)
    if tf=="4h": return t.replace(hour=t.hour-t.hour%4,minute=0)
    return t.replace(hour=0,minute=0)

def wilder_atr_prior(bars:list[Bar], index:int, period:int=14)->float|None:
    """ATR at a bar index, excluding that bar and using only earlier confirmed bars."""
    if index < period+1: return None
    trs=[max(bars[i].high-bars[i].low,abs(bars[i].high-bars[i-1].close),abs(bars[i].low-bars[i-1].close)) for i in range(1,index)]
    atr=sum(trs[:period])/period
    for tr in trs[period:]: atr=(atr*(period-1)+tr)/period
    return atr

def ema_prior(closes:list[float], period:int=50)->list[float]:
    if not closes:return []
    a=2/(period+1); result=[closes[0]]
    for value in closes[1:]: result.append(value*a+result[-1]*(1-a))
    return result

def frequency_classification(executed_trades:int, days:float=365.)->tuple[float,str]:
    annualized=executed_trades*365/days if days else 0.
    return annualized, "UNDERFREQUENCY_FAIL" if annualized<50 else "TARGET_FREQUENCY" if annualized<=300 else "HIGH_FREQUENCY_WARNING" if annualized<=500 else "OVERFREQUENCY_FAIL"

def phase_a_hard_gates(metrics:dict[str,Any])->dict[str,bool]:
    """Pure Phase-A gate evaluator; callers supply precomputed candidate metrics."""
    freq=metrics.get("frequency_classification") or frequency_classification(int(metrics.get("executed_trades",0)),float(metrics.get("days",365)))[1]
    checks={"frequency_50_500":freq in {"TARGET_FREQUENCY","HIGH_FREQUENCY_WARNING"},"positive_net_pnl":metrics.get("net_pnl",0)>0,"profit_factor":metrics.get("profit_factor",0)>=1.30,"positive_average_net_r":metrics.get("average_net_r",0)>0,"max_drawdown":metrics.get("max_drawdown_r",float("inf"))<=20,"profitable_months":metrics.get("profitable_months",0)>=8,"zero_trade_months":metrics.get("zero_trade_months",99)<=3,"best_month_concentration":metrics.get("best_month_positive_pnl_share",1)<=.35,"best_five_concentration":metrics.get("best_five_positive_pnl_share",1)<=.30,"direction_mix":metrics.get("long_share",0)>=.25 and metrics.get("short_share",0)>=.25,"direction_returns":metrics.get("long_average_net_r",-99)>=-.15 and metrics.get("short_average_net_r",-99)>=-.15,"one_tick_stress":metrics.get("one_tick_stress_positive",False),"best_trade_removal":metrics.get("best_trade_removal_positive",False),"bootstrap":metrics.get("bootstrap_median_mean_net_r",0)>0 and metrics.get("bootstrap_lower_95_net_r",-99)>=-.025,"subperiods":metrics.get("nonnegative_2023_subperiods",0)>=3,"reconciliation":metrics.get("immutable_artifacts",False) and metrics.get("funnel_reconciled",False)}
    checks["passed"]=all(checks.values()); return checks

class HTFLevelLiquidityFVG:
    """Literal closed-bar, per-candidate implementation. It contains no data I/O."""
    def __init__(self,candidate_id:str,run_id:str="synthetic",tick:float=.1,quantity_step:float=.001,minimum_quantity:float=.001,fee_rate:float=.0004,slippage_ticks:int=1):
        if candidate_id not in CANDIDATES: raise ValueError("unsealed candidate")
        self.candidate=Candidate(candidate_id,CANDIDATES[candidate_id]); self.run_id=run_id; self.tick=tick; self.quantity_step=quantity_step; self.minimum_quantity=minimum_quantity; self.fee_rate=fee_rate; self.slippage_ticks=slippage_ticks
        self.aggregator=ClosedBarAggregator(); self.bars5=[]; self.bars15=[]; self.bars4h=[]; self.daily=[]; self.levels=[]; self.levels15=[]; self.events=[]; self.outcomes=[]; self.trades=[]; self._seq=0; self.setup=None; self.order=None; self.position=None; self._daily_bias_cache=(0,Bias.NEUTRAL); self._atr5_prior=[]; self._atr15_prior=[]; self._level_index_source=id(self.levels); self._level_index_count=0; self._available_levels={"SUPPORT":[],"RESISTANCE":[]}; self._available_level_prices={"SUPPORT":[],"RESISTANCE":[]}; self._nearest_indexes={}
    def _event(self,t:datetime,decision:str,setup_id:str|None=None,reason_code:str|None=None,trade_id:str|None=None,**inputs:Any)->None:
        scope=event_scope(decision)
        if scope in {EventScope.SETUP,EventScope.TRADE} and not setup_id: raise ValueError("setup/trade event requires setup_id")
        if scope in {EventScope.GLOBAL,EventScope.LEVEL} and setup_id is not None: raise ValueError("non-setup event cannot carry setup_id")
        if scope==EventScope.TRADE and not trade_id: raise ValueError("trade event requires trade_id")
        self._seq+=1; self.events.append(Event(deterministic_id("event",self.run_id,self.candidate.id,self._seq),t.isoformat(),self._seq,self.candidate.id,decision,setup_id,reason_code,inputs,scope,trade_id))
    def _finish(self,t:datetime,disposition:TerminalDisposition,reason:str|None=None)->None:
        if not self.setup:return
        self.setup["terminal_disposition"]=disposition.value; self.setup["event_history"].append("TERMINAL_DISPOSITION"); self._event(t,"TERMINAL_DISPOSITION",self.setup["setup_id"],reason_code=reason or disposition.value); self.outcomes.append(self.setup); self.setup=None; self.order=None
    def _supersede_setup(self,t:datetime,successor_setup_id:str,successor_sweep_id:str)->None:
        """Close only the replaced setup's audit lifecycle; execution state is untouched."""
        if not self.setup:return
        prior=self.setup
        prior["terminal_disposition"]=TerminalDisposition.SUPERSEDED_BY_NEW_SWEEP.value
        prior["superseded_by_setup_id"]=successor_setup_id
        prior["superseded_by_sweep_id"]=successor_sweep_id
        prior["event_history"].append("TERMINAL_DISPOSITION")
        self._event(t,"TERMINAL_DISPOSITION",prior["setup_id"],reason_code=TerminalDisposition.SUPERSEDED_BY_NEW_SWEEP.value,successor_setup_id=successor_setup_id,successor_sweep_id=successor_sweep_id)
        self.outcomes.append(prior)
    def _lifecycle(self,t:datetime,name:str,**data:Any)->None:
        self.setup["event_history"].append(name); self._event(t,name,self.setup["setup_id"],trade_id=self.setup.get("trade_id"),**data)
    def daily_bias(self)->Bias:
        if self._daily_bias_cache[0]==len(self.daily): return self._daily_bias_cache[1]
        if len(self.daily)<56: bias=Bias.NEUTRAL
        else:
            values=ema_prior([x.close for x in self.daily]); bias=Bias.BULLISH if self.daily[-1].close>values[-1] and values[-1]>values[-6] else Bias.BEARISH if self.daily[-1].close<values[-1] and values[-1]<values[-6] else Bias.NEUTRAL
        self._daily_bias_cache=(len(self.daily),bias); return bias
    def _refresh_available_levels(self)->None:
        if self._level_index_source==id(self.levels) and self._level_index_count==len(self.levels): return
        self._available_levels={side:sorted((level.price,level.level_id,level) for level in self.levels if level.side==side and not level.consumed) for side in ("SUPPORT","RESISTANCE")}
        self._available_level_prices={side:[value[0] for value in self._available_levels[side]] for side in ("SUPPORT","RESISTANCE")}
        self._level_index_source=id(self.levels); self._level_index_count=len(self.levels)
    def _available_level(self,side:str,price:float,t:datetime)->Level|None:
        self._refresh_available_levels(); values=self._available_levels[side]
        prices=self._available_level_prices[side]
        i=bisect_right(prices,price)-1 if side=="SUPPORT" else bisect_left(prices,price)
        step=-1 if side=="SUPPORT" else 1
        while 0<=i<len(values):
            candidate_price=values[i][0]; matches=[]
            while 0<=i<len(values) and values[i][0]==candidate_price:
                level=values[i][2]
                if level.confirmation_time<t: matches.append(level)
                i+=step
            if matches:return min(matches,key=lambda level:level.level_id)
        return None
    def _consume_level(self,level:Level)->None:
        level.consumed=True
        values=self._available_levels[level.side]
        index=next(index for index,value in enumerate(values) if value[2] is level)
        del values[index]; del self._available_level_prices[level.side][index]
    @staticmethod
    def _append_prior_atr(bars:list[Bar],cache:list[float|None])->None:
        index=len(bars)-1
        if index<15: value=None
        elif index==15: value=sum(max(bars[i].high-bars[i].low,abs(bars[i].high-bars[i-1].close),abs(bars[i].low-bars[i-1].close)) for i in range(1,15))/14
        else:
            previous=cache[-1]
            assert previous is not None
            bar=bars[index-1]; earlier=bars[index-2]
            true_range=max(bar.high-bar.low,abs(bar.high-earlier.close),abs(bar.low-earlier.close))
            value=(previous*13+true_range)/14
        cache.append(value)
    @staticmethod
    def _prior_atr(bars:list[Bar],cache:list[float|None])->float|None:
        return cache[-1] if len(cache)==len(bars) else wilder_atr_prior(bars,len(bars)-1)
    def feed(self,bar:Bar)->None:
        if self.bars5 and bar.time<=self.bars5[-1].time: raise ValueError("closed bars must be strictly ordered")
        self.bars5.append(bar); self._append_prior_atr(self.bars5,self._atr5_prior); derived=self.aggregator.push(bar)
        for name,target in (("15m",self.bars15),("4h",self.bars4h),("1d",self.daily)):
            if name in derived:
                target.append(derived[name]);
                if name=="15m": self._append_prior_atr(self.bars15,self._atr15_prior)
                self._event(bar.time,"AGGREGATION_CONFIRMED",timeframe=name,source_bar_id=bar.id); self._confirm_levels(name)
                if name=="15m": self._evaluate_sweep(target[-1])
        self._confirm_5m_fractals(); self._progress_setup(bar); self._progress_order_or_position(bar)
    def _confirm_levels(self,tf:str)->None:
        bars=self.bars4h if tf=="4h" else self.bars15
        right=3 if tf=="4h" else 2
        if len(bars)<2*right+1:return
        i=len(bars)-right-1; b=bars[i]; left=bars[i-right:i]; r=bars[i+1:i+right+1]
        low=b.low<min(x.low for x in left) and b.low<=min(x.low for x in r); high=b.high>max(x.high for x in left) and b.high>=max(x.high for x in r)
        if low or high:
            side="SUPPORT" if low else "RESISTANCE"; price=b.low if low else b.high; level=Level(deterministic_id("level",tf,side,b.time.isoformat(),price),side,price,b.time,bars[-1].time)
            (self.levels if tf=="4h" else self.levels15).append(level); self._event(level.confirmation_time,"LEVEL_CONFIRMED",level_id=level.level_id,timeframe=tf,side=side,price=price)
    def _evaluate_sweep(self,b:Bar)->None:
        atr=self._prior_atr(self.bars15,self._atr15_prior)
        if atr is None:return
        bias=self.daily_bias()
        if bias==Bias.NEUTRAL:return
        direction=Direction.LONG if bias==Bias.BULLISH else Direction.SHORT; level=self._available_level("SUPPORT" if direction==Direction.LONG else "RESISTANCE",b.close,b.time)
        if level is None:return
        depth=(b.low<=level.price-.1*atr) if direction==Direction.LONG else (b.high>=level.price+.1*atr)
        reclaim=(b.close>=level.price) if direction==Direction.LONG else (b.close<=level.price)
        wick=((min(b.open,b.close)-b.low)/(b.high-b.low) if b.high>b.low else 0) if direction==Direction.LONG else ((b.high-max(b.open,b.close))/(b.high-b.low) if b.high>b.low else 0)
        if not(depth and reclaim and wick>=.5):return
        self._consume_level(level)
        sid=deterministic_id("setup",self.run_id,self.candidate.id,level.level_id,b.time.isoformat()); sweep_id=deterministic_id("sweep",sid)
        # This preserves the historical replacement order and execution state:
        # only the displaced setup's otherwise-lost audit lifecycle is closed.
        self._supersede_setup(b.time,sid,sweep_id)
        self.setup={"setup_id":sid,"level_id":level.level_id,"sweep_id":sweep_id,"mss_id":None,"fvg_id":None,"direction":direction.value,"daily_bias_snapshot":bias.value,"level_snapshot":asdict(level),"sweep":asdict(b),"sweep_atr":atr,"sweep_extreme":b.low if direction==Direction.LONG else b.high,"sweep_5_index":len(self.bars5)-1,"event_history":["SETUP_PROPOSED"]}; self._event(b.time,"SETUP_PROPOSED",sid,level_id=level.level_id,sweep_id=self.setup["sweep_id"],prior_atr=atr,wick=wick)
    def _confirm_5m_fractals(self)->None: pass # confirmation is derived causally in _progress_setup
    def _progress_setup(self,b:Bar)->None:
        if not self.setup or self.order or self.position:return
        s=self.setup; elapsed=len(self.bars5)-1-s["sweep_5_index"]
        if elapsed>12:self._finish(b.time,TerminalDisposition.MSS_WINDOW_EXPIRED); return
        if self.daily_bias() != (Bias.BULLISH if s["direction"]=="LONG" else Bias.BEARISH): self._finish(b.time,TerminalDisposition.BIAS_INVALIDATED_BEFORE_ENTRY); return
        i=len(self.bars5)-1; atr=self._prior_atr(self.bars5,self._atr5_prior)
        if atr is None:return
        d=Direction(s["direction"])
        if s["mss_id"] is None and i>=4:
            pivot=self.bars5[i-3]; left=self.bars5[i-5:i-3]; right=self.bars5[i-2:i]
            fractal=(pivot.high>max(x.high for x in left+right) if d==Direction.LONG else pivot.low<min(x.low for x in left+right))
            structural=(b.close>pivot.high if d==Direction.LONG else b.close<pivot.low)
            if fractal and structural:
                bullish=d==Direction.LONG; displacement=(b.close>b.open if bullish else b.close<b.open) and b.high-b.low>=1.5*atr and (b.close>=b.low+.75*(b.high-b.low) if bullish else b.close<=b.high-.75*(b.high-b.low))
                if not displacement:self._finish(b.time,TerminalDisposition.DISPLACEMENT_REJECTED); return
                s["mss_id"]=deterministic_id("mss",s["setup_id"],b.time.isoformat()); s["displacement_id"]=deterministic_id("displacement",s["mss_id"]); s["mss_index"]=i; s["fractal_id"]=deterministic_id("fractal",pivot.time.isoformat()); self._lifecycle(b.time,"MSS_CONFIRMED",mss_id=s["mss_id"],fractal_id=s["fractal_id"],prior_atr=atr); self._lifecycle(b.time,"DISPLACEMENT_CONFIRMED",mss_id=s["mss_id"],displacement_id=s["displacement_id"])
        if s.get("mss_id") and i-s["mss_index"]<=2 and i>=2:
            a,c=self.bars5[i-2],self.bars5[i]; lower,upper=(a.high,c.low) if d==Direction.LONG else (c.high,a.low)
            valid=(c.low>a.high if d==Direction.LONG else c.high<a.low)
            if valid:
                width=upper-lower
                if width<.1*atr:self._finish(b.time,TerminalDisposition.FVG_TOO_SMALL); return
                s["fvg_id"]=deterministic_id("fvg",s["setup_id"],a.time.isoformat(),c.time.isoformat()); s["fvg"]={"bars":[x.id for x in self.bars5[i-2:i+1]],"lower":lower,"upper":upper,"width":width}; self._lifecycle(b.time,"FVG_CONFIRMED",fvg_id=s["fvg_id"],boundaries=s["fvg"]); self._activate_order(b)
        elif s.get("mss_id") and i-s["mss_index"]>2:self._finish(b.time,TerminalDisposition.FVG_NOT_FORMED)
    def _activate_order(self,b:Bar)->None:
        s=self.setup; d=Direction(s["direction"])
        if b.time.hour==23 and b.time.minute>=40:self._finish(b.time,TerminalDisposition.SESSION_ENTRY_BLOCKED); return
        entry=s["fvg"]["upper"] if d==Direction.LONG else s["fvg"]["lower"]; stop=s["sweep_extreme"]+(-.05 if d==Direction.LONG else .05)*s["sweep_atr"]; risk=abs(entry-stop)
        if not .0025<=risk/entry<=.02:self._finish(b.time,TerminalDisposition.STOP_DISTANCE_REJECTED); return
        tp1=self._nearest(self.levels15,"RESISTANCE" if d==Direction.LONG else "SUPPORT",entry,b.time,d); tp2=self._nearest(self.levels,"RESISTANCE" if d==Direction.LONG else "SUPPORT",entry,b.time,d)
        if not tp2:self._finish(b.time,TerminalDisposition.PROJECTED_RR_REJECTED); return
        projected=abs(tp2.price-entry)/risk
        if projected<self.candidate.minimum_tp2_r:self._finish(b.time,TerminalDisposition.PROJECTED_RR_REJECTED); return
        s.update({"entry_price":entry,"stop":stop,"tp1_level_id":tp1.level_id if tp1 else None,"tp1":tp1.price if tp1 else None,"tp2_level_id":tp2.level_id,"tp2":tp2.price,"projected_r":projected,"cost_assumptions":{"fee_rate":self.fee_rate,"slippage_ticks":self.slippage_ticks}}); self.order={"order_id":deterministic_id("order",s["setup_id"]),"activated_index":len(self.bars5)-1,"entry":entry}; self._lifecycle(b.time,"ORDER_ACTIVATED",order_id=self.order["order_id"],entry=entry)
    def _nearest(self,levels:list[Level],side:str,entry:float,t:datetime,d:Direction)->Level|None:
        key=(id(levels),side); cached=self._nearest_indexes.get(key)
        if cached is None or cached[0]!=len(levels):
            values=sorted((level.price,level.level_id,level) for level in levels if level.side==side)
            self._nearest_indexes[key]=(len(levels),values,[value[0] for value in values]); cached=self._nearest_indexes[key]
        _,values,prices=cached
        i=bisect_right(prices,entry) if d==Direction.LONG else bisect_left(prices,entry)-1
        step=1 if d==Direction.LONG else -1
        while 0<=i<len(values):
            candidate_price=values[i][0]; matches=[]
            while 0<=i<len(values) and values[i][0]==candidate_price:
                level=values[i][2]
                if level.confirmation_time<t: matches.append(level)
                i+=step
            if matches:return min(matches,key=lambda level:level.level_id)
        return None
    def _progress_order_or_position(self,b:Bar)->None:
        if self.order and not self.position:
            s=self.setup; d=Direction(s["direction"])
            if b.time.hour==23 and b.time.minute>=55:self._finish(b.time,TerminalDisposition.SESSION_ENDED); return
            if len(self.bars5)-1>s["sweep_5_index"]+12:self._finish(b.time,TerminalDisposition.PENDING_ORDER_EXPIRED); return
            if (b.low<s["sweep_extreme"] if d==Direction.LONG else b.high>s["sweep_extreme"]):self._finish(b.time,TerminalDisposition.PRE_ENTRY_SWEEP_INVALIDATED); return
            if len(self.bars5)-1<=self.order["activated_index"]:return
            touched=b.low<=self.order["entry"] if d==Direction.LONG else b.high>=self.order["entry"]
            if touched:
                price=self.order["entry"]+(self.slippage_ticks*self.tick if d==Direction.LONG else -self.slippage_ticks*self.tick); qty=max(self.minimum_quantity,1.0//self.quantity_step*self.quantity_step); self.position={"position_id":deterministic_id("position",self.order["order_id"]),"entry_index":len(self.bars5)-1,"entry":price,"qty":qty,"remaining":qty,"tp1_done":False}; s["trade_id"]=deterministic_id("trade",s["setup_id"]); self._lifecycle(b.time,"ENTRY_FILLED",fill_id=deterministic_id("fill",s["trade_id"],"entry"),quantity=qty,price=price)
        if not self.position:return
        s=self.setup; p=self.position; d=Direction(s["direction"]); stop=s["stop"] if not p["tp1_done"] else p["breakeven"]
        forced=(b.time.hour==23 and b.time.minute>=55) or len(self.bars5)-1-p["entry_index"]>=96
        stop_hit=b.low<=stop if d==Direction.LONG else b.high>=stop
        tp1_hit=not p["tp1_done"] and s["tp1"] is not None and (b.high>=s["tp1"] if d==Direction.LONG else b.low<=s["tp1"])
        tp2_hit=p["tp1_done"] and (b.high>=s["tp2"] if d==Direction.LONG else b.low<=s["tp2"])
        if stop_hit:self._exit(b,stop,p["remaining"],"TP1_STOPPED" if p["tp1_done"] else "STOPPED"); return
        if tp1_hit:
            q=p["qty"]*.5; self._exit(b,s["tp1"],q,"TP1",terminal=False); p["tp1_done"]=True; p["breakeven"]=p["entry"]+(2*self.fee_rate*p["entry"]+self.slippage_ticks*self.tick)*(1 if d==Direction.LONG else -1); return
        if tp2_hit:self._exit(b,s["tp2"],p["remaining"],"TP2_COMPLETED"); return
        if forced:self._exit(b,b.close,p["remaining"],"FORCED_TIME_EXIT")
    def _exit(self,b:Bar,price:float,qty:float,reason:str,terminal:bool=True)->None:
        p=self.position; s=self.setup; d=Direction(s["direction"]); gross=(price-p["entry"])*qty*(1 if d==Direction.LONG else -1); fees=(price+p["entry"])*qty*self.fee_rate; p["remaining"]-=qty; s.setdefault("exits",[]).append({"exit_id":deterministic_id("exit",s["trade_id"],len(s.get("exits",[]))),"quantity":qty,"price":price,"gross_pnl":gross,"fees":fees,"net_pnl":gross-fees,"reason":reason}); self._lifecycle(b.time,"EXIT_FILLED",reason=reason,quantity=qty,price=price)
        if terminal:
            trade={"trade_id":s["trade_id"],"setup_id":s["setup_id"],"direction":s["direction"],"quantity":p["qty"],"exits":s["exits"],"net_pnl":sum(x["net_pnl"] for x in s["exits"]),"exit_reason":reason}; self.trades.append(trade); self._finish(b.time,TerminalDisposition.TRADE_EXECUTED); self.position=None

def reconcile_events(events:Iterable[Event],outcomes:Iterable[dict],trades:Iterable[dict])->None:
    """Fail-closed, indexed event/outcome/trade reconciliation.

    All indexes are built in one pass; in particular, this must remain linear in
    the real audit volume rather than filtering the complete event list per setup.
    """
    es=list(events); os=list(outcomes); ts=list(trades)
    placeholders={"", "null", "none", "unknown", "placeholder"}
    def fail(detail:str)->None: raise ValueError(f"RECONCILIATION_ERROR: {detail}")
    def valid_id(value:object)->bool:
        return isinstance(value,str) and value.strip().lower() not in placeholders
    def add_identity(index:dict[str,object], value:object, label:str)->None:
        if not valid_id(value) or value in index: fail(f"duplicate or empty {label} id")
        index[value]=None

    event_ids:dict[str,object]={}; proposed:dict[str,Event]={}; lifecycle:dict[str,list[Event]]={}
    terminal_count:dict[str,int]={}; level_ids:dict[str,object]={}; sweep_ids:dict[str,object]={}
    mss_ids:dict[str,object]={}; fvg_ids:dict[str,object]={}
    for event in es:
        add_identity(event_ids,event.event_id,"event")
        if event.scope != event_scope(event.decision): fail("event scope contract")
        if event.scope in {EventScope.SETUP,EventScope.TRADE}:
            if not valid_id(event.setup_id): fail("empty or placeholder setup id")
            lifecycle.setdefault(event.setup_id,[]).append(event)
        elif event.setup_id is not None: fail("non-setup event has setup id")
        if event.scope==EventScope.LEVEL:
            if not valid_id(event.inputs.get("level_id")): fail("level event missing level id")
        if event.scope==EventScope.TRADE and not valid_id(event.trade_id): fail("trade event missing trade id")
        if event.decision=="SETUP_PROPOSED":
            if event.setup_id in proposed: fail("duplicate setup id")
            proposed[event.setup_id]=event; terminal_count[event.setup_id]=0
            add_identity(sweep_ids,event.inputs.get("sweep_id"),"sweep")
        elif event.decision=="LEVEL_CONFIRMED": add_identity(level_ids,event.inputs.get("level_id"),"level")
        elif event.decision=="MSS_CONFIRMED": add_identity(mss_ids,event.inputs.get("mss_id"),"MSS")
        elif event.decision=="FVG_CONFIRMED": add_identity(fvg_ids,event.inputs.get("fvg_id"),"FVG")
        elif event.decision=="TERMINAL_DISPOSITION":
            # The setup may be proposed earlier in the same append-only stream.
            if event.setup_id not in terminal_count: fail("orphan setup event")
            terminal_count[event.setup_id]+=1

    for setup_id in lifecycle:
        if setup_id not in proposed: fail("orphan setup event")
    if any(count!=1 for count in terminal_count.values()): fail("missing or duplicate terminal event")

    outcomes_by_setup:dict[str,dict]={}
    for outcome in os:
        setup_id=outcome.get("setup_id")
        if not valid_id(setup_id) or setup_id in outcomes_by_setup: fail("duplicate or empty setup outcome id")
        outcomes_by_setup[setup_id]=outcome
    if set(proposed)!=set(outcomes_by_setup): fail("proposed setup/outcome mismatch")

    trades_by_setup:dict[str,list[dict]]={setup_id:[] for setup_id in proposed}; trade_ids:dict[str,dict]={}
    for trade in ts:
        trade_id=trade.get("trade_id"); setup_id=trade.get("setup_id")
        if not valid_id(trade_id) or trade_id in trade_ids: fail("duplicate or empty trade id")
        if setup_id not in trades_by_setup: fail("trade without proposed setup")
        trade_ids[trade_id]=trade; trades_by_setup[setup_id].append(trade)

    for setup_id,outcome in outcomes_by_setup.items():
        setup_lifecycle=lifecycle.get(setup_id,[])
        decisions=[event.decision for event in setup_lifecycle]
        if decisions.count("SETUP_PROPOSED")!=1 or decisions.count("TERMINAL_DISPOSITION")!=1 or len(decisions)!=len(outcome.get("event_history",[])): fail("lifecycle")
        executed=outcome.get("terminal_disposition")==TerminalDisposition.TRADE_EXECUTED.value; linked=trades_by_setup[setup_id]
        trade_events=[event for event in setup_lifecycle if event.scope==EventScope.TRADE]
        if executed != (len(linked)==1): fail("executed setup/trade mismatch")
        if not executed and (linked or trade_events): fail("trade linked to non-executed setup")
        if executed:
            trade=linked[0]
            if outcome.get("trade_id") != trade["trade_id"]: fail("outcome trade id mismatch")
            if sum(event.decision=="ENTRY_FILLED" for event in trade_events)!=1: fail("executed setup entry lifecycle")
            if not trade_events or any(event.trade_id != trade["trade_id"] for event in trade_events): fail("trade event link")
            if abs(sum(exit["quantity"] for exit in trade["exits"])-trade["quantity"])>1e-9: fail("quantity")

def _write_json(path:Path,value:object)->None:path.write_text(json.dumps(value,sort_keys=True,indent=2,default=str)+"\n",encoding="utf-8")
def materialize_synthetic(artifact_root:Path,repository_root:Path)->dict:
    if artifact_root.exists():raise FileExistsError("immutable artifact collision")
    spec=repository_root/".smithers/specs/htf-level-liquidity-fvg-v1.md"
    if hashlib.sha256(spec.read_bytes()).hexdigest().upper()!=SPEC_HASH:raise RuntimeError("sealed specification hash mismatch")
    artifact_root.mkdir(parents=True); t=datetime(2023,1,1,tzinfo=timezone.utc); engines=[]
    # Each sealed candidate is actually fed closed synthetic bars.  No placeholder
    # artifact is emitted and no market-data path is consulted.
    for candidate_id in CANDIDATES:
        engine=HTFLevelLiquidityFVG(candidate_id,run_id=f"synthetic-{candidate_id}")
        for i in range(15):engine.feed(Bar(t+timedelta(minutes=5*i),100,101,99,100.5,1,f"synthetic-{candidate_id}-{i}"))
        engines.append(engine)
    common={"specification_hash":SPEC_HASH,"synthetic_only":True,"realStudyExecuted":False}; candidate_runs=[{"candidate_id":e.candidate.id,"event_count":len(e.events),"derived_confirmations":sum(x.decision=="AGGREGATION_CONFIRMED" for x in e.events)} for e in engines]; payloads={"sealed-specification.json":{"sha256":SPEC_HASH},"candidate-registry.json":{"candidates":[asdict(Candidate(k,v)) for k,v in CANDIDATES.items()]},"data-manifest.json":{**common,"source":"synthetic fixtures only"},"derived-timeframe-manifest.json":{"derived_only_in_memory":True,"candidate_runs":candidate_runs},"configuration.json":common,"levels.json":[],"events.json":[asdict(x) for e in engines for x in e.events],"trades.json":[],"setup_outcomes.json":[],"monthly_metrics.json":[],"report.json":{**common,"executed_candidate_flow":True,"candidate_runs":candidate_runs},"gates.json":{k:phase_a_hard_gates({"executed_trades":0,"immutable_artifacts":True,"funnel_reconciled":True}) for k in CANDIDATES},"selection_report.json":{**common,"status":"PHASE_A_NO_ROBUST_CANDIDATE"},"freeze.json":{**common,"status":"NOT_FROZEN"},"final_report.json":{**common,"status":"SYNTHETIC_MATERIALIZED","model":"gpt-5.6-terra"}}
    for name in REQUIRED_ARTIFACTS:
        if name!="integrity-manifest.json":_write_json(artifact_root/name,payloads[name])
    _write_json(artifact_root/"integrity-manifest.json",{"files":{n:hashlib.sha256((artifact_root/n).read_bytes()).hexdigest() for n in REQUIRED_ARTIFACTS if n!="integrity-manifest.json"}})
    return {"status":"SYNTHETIC_MATERIALIZED","artifactRoot":str(artifact_root),"realStudyExecuted":False,"candidates":list(CANDIDATES),"model":"gpt-5.6-terra"}
