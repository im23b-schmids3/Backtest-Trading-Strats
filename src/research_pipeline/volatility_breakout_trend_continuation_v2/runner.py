from __future__ import annotations

import hashlib, json, math, subprocess
from collections import Counter
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from pathlib import Path
from typing import Any

from ..imbalance_vwap_ride.models import COST_MODEL_VERSION, ImbalanceVWAPRideConfig

STUDY_ID = "VolatilityBreakoutTrendContinuation.BTC_LONG_SHORT_V2_STRICT_SELECTION"
SPEC_PATH = ".smithers/specs/volatility-breakout-trend-continuation-v2-strict.md"
SPEC_SHA256 = "ca6d66857e4388d7071ed3d56dee4628bdc7a353ca883db4eb9ac3a82c683e4f"
PHASE_A_MANIFEST_SHA256 = "9fb7228ca074fc5a3b90e6fe82181d07d49f8c62ebd68274e859d0361df3cd6e"
CANDIDATES = (("VBTC-V2-2P5R", Decimal("2.5")), ("VBTC-V2-3P0R", Decimal("3.0")), ("VBTC-V2-3P5R", Decimal("3.5")))
DISPOSITIONS=frozenset({"TREND_FILTER_REJECTED","EMA_SEPARATION_REJECTED","BREAKOUT_THRESHOLD_REJECTED","EXPANSION_FILTER_REJECTED","BREAKOUT_CLOSE_QUALITY_REJECTED","VOLUME_CONFIRMATION_REJECTED","DUPLICATE_STRUCTURE_BLOCKED","SESSION_ENDED","SESSION_ENTRY_BLOCKED","ACTIVE_POSITION_BLOCKED","DIRECTIONAL_COOLDOWN_BLOCKED","FALSE_BREAKOUT_INVALIDATED","STOP_DISTANCE_REJECTED","NO_EXECUTABLE_ENTRY","TRADE_EXECUTED"})
_cfg=ImbalanceVWAPRideConfig()
EXECUTION_ASSUMPTIONS={"cost_model_version":COST_MODEL_VERSION,"symbol":_cfg.symbol,"quantity_btc":str(_cfg.quantity_btc),"price_tick":str(_cfg.price_tick),"quantity_step":str(_cfg.quantity_step),"minimum_quantity":str(_cfg.minimum_quantity),"taker_fee_rate":str(_cfg.taker_fee_rate),"market_slippage_ticks":_cfg.market_slippage_ticks,"stop_slippage_ticks":_cfg.stop_slippage_ticks,"same_bar_policy":_cfg.same_bar_policy}

def _canon(x:Any)->bytes:return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False,default=str).encode()
def _hash(x:Any)->str:return hashlib.sha256(_canon(x)).hexdigest()
def _d(x:Any)->Decimal:return Decimal(str(x))
def _stamp(x:Any)->str:return _timestamp(x).isoformat().replace("+00:00","Z")
def _timestamp(x:Any)->datetime:
    d=datetime.fromisoformat(str(x).replace("Z","+00:00"))
    if d.tzinfo is None or d.utcoffset()!=timedelta(0):raise ValueError("VBTC_V2_TIMESTAMP_NOT_UTC")
    return d
def _file_hash(p:Path)->str:
    # Keep hashing streaming so synthetic materialization can audit its own
    # artifacts without treating their names as input data reads.
    digest=hashlib.sha256()
    with p.open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""): digest.update(block)
    return digest.hexdigest()
def verify_sealed_specification(repository_root:str|Path)->Path:
    p=Path(repository_root).resolve()/SPEC_PATH
    if not p.is_file() or _file_hash(p).lower()!=SPEC_SHA256:raise ValueError("MISSING_OR_CHANGED_SEALED_VBTC_V2_SPECIFICATION")
    return p
def _ema(v:list[Decimal],n:int)->list[Decimal|None]:
    o:[Decimal|None]=[None]*len(v)
    if len(v)>=n:
        o[n-1]=sum(v[:n])/n;a=Decimal(2)/(n+1)
        for i in range(n,len(v)):o[i]=v[i]*a+o[i-1]*(1-a) # type: ignore
    return o
def _atr(b:list[dict[str,Any]])->list[Decimal|None]:
    o:[Decimal|None]=[None]*len(b); trs=[]
    for i in range(1,len(b)):trs.append(max(_d(b[i]['high'])-_d(b[i]['low']),abs(_d(b[i]['high'])-_d(b[i-1]['close'])),abs(_d(b[i]['low'])-_d(b[i-1]['close']))))
    if len(b)>14:
        o[14]=sum(trs[:14])/14
        for i in range(15,len(b)):o[i]=(o[i-1]*13+trs[i-1])/14 # type: ignore
    return o
def _q(x:Decimal,mode:str)->Decimal:return x.quantize(_cfg.price_tick,rounding=mode)
def _event(structure_id:str,setup_id:str,timestamp:str,event_type:str,direction:str,price_rule:str)->dict[str,str]:
    ident=_hash({"structure_id":structure_id,"setup_id":setup_id,"event_timestamp":timestamp,"event_type":event_type,"bar_timestamp":timestamp,"direction":direction,"price_rule":price_rule})
    return {"event_id":ident,"setup_id":setup_id,"structure_id":structure_id,"timestamp":timestamp,"type":event_type,"direction":direction,"price_rule":price_rule}

def validate_synthetic_bars(bars:list[dict[str,Any]])->None:
    """Validate the in-memory test contract without opening a data source.

    The production reader performs the equivalent validation before handing bars
    to this evaluator.  Keeping it here makes synthetic tests exercise the same
    chronology and OHLCV invariants without ever naming a market-data path.
    """
    required={"timestamp","open","high","low","close","volume","daily_vwap"}
    previous:datetime|None=None
    for bar in bars:
        if not required.issubset(bar): raise ValueError("VBTC_V2_SYNTHETIC_BAR_SCHEMA_INVALID")
        stamp=_timestamp(bar["timestamp"])
        values={key:_d(bar[key]) for key in ("open","high","low","close","volume","daily_vwap")}
        if (not all(value.is_finite() for value in values.values()) or
            any(values[key]<=0 for key in ("open","high","low","close","daily_vwap")) or
            values["volume"]<0 or values["high"]<max(values["open"],values["close"]) or
            values["low"]>min(values["open"],values["close"]) or values["high"]<values["low"]):
            raise ValueError("VBTC_V2_SYNTHETIC_BAR_SCHEMA_INVALID")
        if previous is not None and (stamp<=previous or stamp-previous!=timedelta(minutes=5)):
            raise ValueError("VBTC_V2_SYNTHETIC_CHRONOLOGY_INVALID")
        previous=stamp

def evaluate_bars(bars:list[dict[str,Any]],candidate_id:str="VBTC-V2-2P5R",target_r:Decimal=Decimal('2.5')) -> tuple[list[dict[str,Any]],list[dict[str,Any]],list[dict[str,Any]]]:
    """Pure in-memory V2 evaluator.  It never opens manifests or market data."""
    if candidate_id not in dict(CANDIDATES):raise ValueError("VBTC_V2_UNREGISTERED_CANDIDATE")
    if target_r != dict(CANDIDATES)[candidate_id]: raise ValueError("VBTC_V2_TARGET_REGISTRY_MISMATCH")
    validate_synthetic_bars(bars)
    closes=[_d(b['close']) for b in bars]; e20,e50,atr=_ema(closes,20),_ema(closes,50),_atr(bars)
    setups=[];events=[];trades=[];claimed=set();active_until=-1;cooldown={"LONG":-1,"SHORT":-1}; qty=(_cfg.quantity_btc/_cfg.quantity_step).to_integral_value(rounding=ROUND_FLOOR)*_cfg.quantity_step
    for t in range(50,len(bars)-1):
        b=bars[t]; ts=_timestamp(b['timestamp']);a=atr[t-1]
        if a is None or e20[t-1] is None or e50[t-1] is None or e20[t-6] is None or e50[t-6] is None:continue
        rh=max(_d(x['high']) for x in bars[t-48:t]);rl=min(_d(x['low']) for x in bars[t-48:t]); rng=_d(b['high'])-_d(b['low']);tr=max(rng,abs(_d(b['high'])-closes[t-1]),abs(_d(b['low'])-closes[t-1])); med=sorted(_d(x['volume']) for x in bars[t-20:t]); ref=(med[9]+med[10])/2
        for direction in ('LONG','SHORT'):
            sign=Decimal(1) if direction=='LONG' else Decimal(-1); breakout=rh+Decimal('.20')*a if sign>0 else rl-Decimal('.20')*a
            structure=_hash({"study_id":STUDY_ID,"phase":"PHASE_A","symbol":"BTCUSDT","direction":direction,"range_start":_stamp(bars[t-48]['timestamp']),"range_end":_stamp(bars[t-1]['timestamp']),"range_high":str(rh),"range_low":str(rl),"range_bars":48,"trend_family":"EMA20_EMA50_DUAL_SLOPE","threshold_atr":"0.20","expansion_multiple":"1.50","ema_separation_atr":"0.20","volume_lookback_bars":20,"volume_multiplier":"1.25"})
            setup_id=_hash({"structure_id":structure,"signal_bar_timestamp":_stamp(b['timestamp']),"atr":str(a),"breakout_level":str(breakout),"breakout_close":str(_d(b['close'])),"true_range":str(tr),"reference_volume_median":str(ref)})
            base={"setup_id":setup_id,"structure_id":structure,"candidate_id":candidate_id,"direction":direction,"breakout_bar":_stamp(b['timestamp']),"entry_bar":_stamp(bars[t+1]['timestamp']),"range_high":str(rh),"range_low":str(rl),"atr":str(a)}
            events.append(_event(structure,setup_id,base['breakout_bar'],'PROPOSED_SETUP',direction,'COMPLETED_BAR_CLOSE'))
            trend=(e20[t-1]>e50[t-1] and e20[t-1]>e20[t-6] and e50[t-1]>e50[t-6]) if sign>0 else (e20[t-1]<e50[t-1] and e20[t-1]<e20[t-6] and e50[t-1]<e50[t-6])
            quality=(rng!=0 and ((_d(b['close'])-_d(b['low']))/rng>=Decimal('.75') if sign>0 else (_d(b['high'])-_d(b['close']))/rng>=Decimal('.75')))
            disp='TRADE_EXECUTED'; trade=None;stop=None;distance=None
            if not trend:disp='TREND_FILTER_REJECTED'
            elif abs(e20[t-1]-e50[t-1])<Decimal('.20')*a:disp='EMA_SEPARATION_REJECTED'
            elif not (_d(b['close'])>=breakout if sign>0 else _d(b['close'])<=breakout):disp='BREAKOUT_THRESHOLD_REJECTED'
            elif tr<Decimal('1.50')*a:disp='EXPANSION_FILTER_REJECTED'
            elif not quality:disp='BREAKOUT_CLOSE_QUALITY_REJECTED'
            elif _d(b['volume'])<Decimal('1.25')*ref:disp='VOLUME_CONFIRMATION_REJECTED'
            elif ts.hour==23 and ts.minute==55:disp='SESSION_ENDED'
            elif (ts.hour,ts.minute)<(0,5) or (ts.hour,ts.minute)>(23,35):disp='SESSION_ENTRY_BLOCKED'
            elif active_until>=t:disp='ACTIVE_POSITION_BLOCKED'
            elif t+1<=cooldown[direction]:disp='DIRECTIONAL_COOLDOWN_BLOCKED'
            elif structure in claimed:disp='DUPLICATE_STRUCTURE_BLOCKED'
            else:
                claimed.add(structure); op=_d(bars[t+1]['open'])
                if (sign>0 and op<=rh) or (sign<0 and op>=rl):disp='FALSE_BREAKOUT_INVALIDATED'
                else:
                    entry=_q(op+sign*_cfg.price_tick*_cfg.market_slippage_ticks,ROUND_CEILING if sign>0 else ROUND_FLOOR); raw=_d(b['low'])-Decimal('.10')*a if sign>0 else _d(b['high'])+Decimal('.10')*a;stop=_q(raw,ROUND_FLOOR if sign>0 else ROUND_CEILING);distance=sign*(entry-stop)
                    if qty<_cfg.minimum_quantity:disp='NO_EXECUTABLE_ENTRY'
                    elif distance<Decimal('.0020')*entry or distance>Decimal('.0125')*entry:disp='STOP_DISTANCE_REJECTED'
                    else:
                        target=_q(entry+sign*target_r*distance,ROUND_CEILING if sign>0 else ROUND_FLOOR);ei=None
                        for x in range(t+2,min(t+26,len(bars))):
                            z=bars[x];sh=_d(z['low'])<=stop if sign>0 else _d(z['high'])>=stop;th=_d(z['high'])>=target if sign>0 else _d(z['low'])<=target
                            if sh or th:ei=x;reason='STOP_FIRST_AMBIGUITY' if sh and th else ('STOP' if sh else 'TARGET');refexit=stop if sh else target;exitp=_q(refexit-sign*_cfg.price_tick*_cfg.stop_slippage_ticks,ROUND_FLOOR if sign>0 else ROUND_CEILING) if sh else target;break
                            if _timestamp(z['timestamp']).hour==23 and _timestamp(z['timestamp']).minute==55:ei=x;reason='SESSION_FLAT';refexit=_d(z['close']);exitp=_q(refexit-sign*_cfg.price_tick*_cfg.market_slippage_ticks,ROUND_FLOOR if sign>0 else ROUND_CEILING);break
                        if ei is None:ei=min(t+25,len(bars)-1);reason='TIME_STOP';refexit=_d(bars[ei]['close']);exitp=_q(refexit-sign*_cfg.price_tick*_cfg.market_slippage_ticks,ROUND_FLOOR if sign>0 else ROUND_CEILING)
                        ee=_event(structure,setup_id,base['entry_bar'],'ENTRY',direction,'NEXT_OPEN_MARKET');events.append(ee); gross=sign*(refexit-op)*qty;fees=_cfg.taker_fee_rate*(entry+exitp)*qty;slip=(abs(entry-op)+abs(exitp-refexit))*qty;net=gross-fees-slip;tid=_hash({"candidate_id":candidate_id,"structure_id":structure,"setup_id":setup_id,"entry_event_id":ee['event_id'],"target_r":str(target_r),"execution_assumption_hash":_hash(EXECUTION_ASSUMPTIONS)})
                        trade={"trade_id":tid,"setup_id":setup_id,"structure_id":structure,"entry_event_id":ee['event_id'],"entry_timestamp":base['entry_bar'],"exit_timestamp":_stamp(bars[ei]['timestamp']),"direction":direction,"entry":str(entry),"exit":str(exitp),"stop":str(stop),"target":str(target),"net_pnl":str(net),"net_r":str(net/(distance*qty)),"exit_reason":reason};trades.append(trade);active_until=ei;cooldown[direction]=t+13
            terminal_time=base['entry_bar'] if disp in {'FALSE_BREAKOUT_INVALIDATED','STOP_DISTANCE_REJECTED','NO_EXECUTABLE_ENTRY','TRADE_EXECUTED'} else base['breakout_bar'];events.append(_event(structure,setup_id,terminal_time,disp,direction,'NEXT_OPEN_MARKET' if terminal_time==base['entry_bar'] else 'COMPLETED_BAR_CLOSE'));base.update(terminal_disposition=disp,trade_id=trade['trade_id'] if trade else None,proposed_stop_or_null=str(stop) if stop is not None else None,stop_distance_or_null=str(distance) if distance is not None else None);setups.append(base)
    validate_reconciliation(setups,events,trades);return setups,events,trades

def reconciliation_summary(setups:list[dict[str,Any]],events:list[dict[str,Any]],trades:list[dict[str,Any]])->dict[str,Any]:
    ids=[x['setup_id'] for x in setups]; eids=[x['event_id'] for x in events]; tids=[x['trade_id'] for x in trades]
    if len(ids)!=len(set(ids)) or len(eids)!=len(set(eids)) or len(tids)!=len(set(tids)):raise ValueError('VBTC_V2_RECONCILIATION_FAILURE')
    known=set(ids)
    if any(x.get('setup_id') not in known for x in events+trades) or any(x.get('terminal_disposition') not in DISPOSITIONS for x in setups):raise ValueError('VBTC_V2_RECONCILIATION_FAILURE')
    by={i:[] for i in known}
    for e in events:by[e['setup_id']].append(e['type'])
    for s in setups:
        expected=Counter(['PROPOSED_SETUP',s['terminal_disposition']]+(['ENTRY'] if s['terminal_disposition']=='TRADE_EXECUTED' else []))
        if Counter(by[s['setup_id']])!=expected:raise ValueError('VBTC_V2_RECONCILIATION_FAILURE')
    executed={s['setup_id'] for s in setups if s['terminal_disposition']=='TRADE_EXECUTED'}
    event_ids=set(eids)
    if (executed!={t['setup_id'] for t in trades} or
        any(t.get('entry_event_id') not in event_ids for t in trades) or
        any(not any(e['event_id']==t['entry_event_id'] and e['type']=='ENTRY' and e['setup_id']==t['setup_id'] for e in events) for t in trades) or
        len(events)!=2*len(setups)+len(trades)):raise ValueError('VBTC_V2_RECONCILIATION_FAILURE')
    return {'formula':'events == 2 * proposed_setups + executed_trades','proposed_setups':len(setups),'executed_trades':len(trades),'expected_events':2*len(setups)+len(trades),'actual_events':len(events),'reconciles':True}
def validate_reconciliation(s,e,t)->None:reconciliation_summary(s,e,t)

def _synthetic_bars()->list[dict[str,Any]]:
    start=datetime(2023,1,1,tzinfo=timezone.utc);out=[]
    for i in range(80):
        p=Decimal('100')+Decimal(i)*Decimal('.1');out.append({'timestamp':_stamp(start+timedelta(minutes=5*i)),'open':str(p),'high':str(p+1),'low':str(p-1),'close':str(p+Decimal('.2')),'volume':'100','daily_vwap':str(p)})
    return out
def materialize_synthetic_contract(*,artifact_root:str|Path,repository_root:str|Path)->dict[str,Any]:
    root=Path(repository_root).resolve(); verify_sealed_specification(root)
    # This path intentionally never receives a manifest path and never imports parquet.
    store=_Store(Path(artifact_root).resolve(),"NOT_READ","SYNTHETIC")
    common={"schema_version":"VBTC-V2","study_id":STUDY_ID,"phase":"PHASE_A","specification_sha256":SPEC_SHA256,"data_manifest_sha256":"NOT_READ","created_at_utc":"SYNTHETIC_ONLY","phase_b":"NOT_OPENED","alpha":"NOT_OPENED"}
    store.write("sealed-specification.json",{**common,"text":(root/SPEC_PATH).read_text(encoding="utf-8"),"seal_status":"SEALED"}); store.write("candidate-registry.json",{**common,"candidates":[{"candidate_id":x,"target_r":str(y)} for x,y in CANDIDATES],"only_target_differs":True}); store.write("data-manifest.json",{**common,"status":"NOT_READ","market_data_read":False})
    results=[]
    for cid,target in CANDIDATES:
        setups,events,trades=evaluate_bars(_synthetic_bars(),cid,target); rec=reconciliation_summary(setups,events,trades)
        store.write("configuration.json",{**common,"candidate_id":cid,"target_r":str(target),"execution_assumption_hash":_hash(EXECUTION_ASSUMPTIONS)},cid)
        store.write("events.json",{**common,"candidate_id":cid,"events":events},cid); store.write("trades.json",{**common,"candidate_id":cid,"trades":trades},cid); store.write("setup_outcomes.json",{**common,"candidate_id":cid,"setup_outcomes":setups},cid); store.write("monthly_metrics.json",{**common,"candidate_id":cid,"status":"NOT_EXECUTED"},cid); store.write("report.json",{**common,"candidate_id":cid,"event_reconciliation":rec,"status":"SYNTHETIC_ONLY"},cid); store.write("gates.json",{**common,"candidate_id":cid,"status":"NOT_EXECUTED"},cid); results.append({"candidate_id":cid,"reconciliation":rec})
    store.write("selection_report.json",{**common,"status":"PHASE_A_NO_ROBUST_CANDIDATE","ranking":[]}); store.write("freeze.json",{**common,"status":"NOT_FROZEN"}); store.write("final_report.json",{**common,"status":"SYNTHETIC_MATERIALIZED","summary":"Synthetic-only VBTC V2 materialization; Phase B and Alpha remain unopened.","realStudyExecuted":False,"marketDataAccessed":False,"model":"gpt-5.6-terra","candidates":results}); store.seal()
    return {'status':'SYNTHETIC_MATERIALIZED','artifactRoot':str(store.root),'realStudyExecuted':False,'marketDataAccessed':False}
PHASE_A_MONTHS=tuple([f"2023-{m:02d}" for m in range(1,13)]+["2024-01"])
PHASE_A_START=datetime(2023,1,1,tzinfo=timezone.utc)
PHASE_A_LAST=datetime(2024,1,31,23,55,tzinfo=timezone.utc)

def _clean_git(root:Path)->str:
    status=subprocess.run(["git","status","--porcelain"],cwd=root,text=True,capture_output=True)
    head=subprocess.run(["git","rev-parse","HEAD"],cwd=root,text=True,capture_output=True)
    if status.returncode or status.stdout.strip() or head.returncode or not head.stdout.strip(): raise ValueError("VBTC_V2_PHASE_A_REQUIRES_CLEAN_COMMITTED_GIT")
    return head.stdout.strip()

class _Store:
    def __init__(self, root:Path, manifest_hash:str, revision:str):
        self.root=root/"research"/"volatility_breakout_trend_continuation"/"v2"/"phase_a"/f"study-{SPEC_SHA256}"
        # mkdir without exist_ok is the create-new primitive.  An intended run
        # directory is immutable even if it is empty or partially materialized.
        self.root.mkdir(parents=True, exist_ok=False); self.manifest_hash=manifest_hash; self.revision=revision
    def write(self,name:str,payload:Any,candidate:str|None=None)->None:
        path=self.root/(f"candidate-{candidate}" if candidate else "")/name
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists(): raise FileExistsError("VBTC_V2_IMMUTABLE_ARTIFACT_COLLISION")
        path.write_bytes(_canon(payload))
    def seal(self)->None:
        files=[]
        for p in sorted(self.root.rglob("*")):
            if p.is_file() and p.name!="integrity-manifest.json": files.append({"relative_path":p.relative_to(self.root).as_posix(),"byte_length":p.stat().st_size,"sha256":_file_hash(p),"specification_hash":SPEC_SHA256,"data_manifest_hash":self.manifest_hash,"execution_assumption_hash":_hash(EXECUTION_ASSUMPTIONS),"code_revision":self.revision,"phase":"PHASE_A","created_at_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z")})
        self.write("integrity-manifest.json",{"schema_version":"VBTC-V2","study_id":STUDY_ID,"phase":"PHASE_A","specification_sha256":SPEC_SHA256,"data_manifest_sha256":self.manifest_hash,"execution_assumption_hash":_hash(EXECUTION_ASSUMPTIONS),"created_at_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),"files":files})

def _assert_output_absent(artifact_root:Path)->None:
    intended=artifact_root/"research"/"volatility_breakout_trend_continuation"/"v2"/"phase_a"/f"study-{SPEC_SHA256}"
    if intended.exists(): raise FileExistsError("VBTC_V2_IMMUTABLE_OUTPUT_COLLISION")

def _load_phase_a_bars(manifest_path:Path)->tuple[dict[str,Any],list[dict[str,Any]]]:
    """Hash verification is performed by the caller before this function reads a bar."""
    try: manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError,json.JSONDecodeError) as exc: raise ValueError("VBTC_V2_PHASE_A_MANIFEST_SCHEMA_INVALID") from exc
    identity=manifest.get("identity")
    if not isinstance(identity,dict) or manifest.get("valid") is not True or identity.get("phase")!="PHASE_A" or identity.get("symbol")!="BTCUSDT" or identity.get("bar_interval")!="5m" or tuple(identity.get("months",()))!=PHASE_A_MONTHS: raise ValueError("VBTC_V2_PHASE_A_MANIFEST_SCHEMA_INVALID")
    files=[x for x in manifest.get("parquet_files",[]) if isinstance(x,dict) and x.get("kind","bars")=="bars"]
    if len(files)!=len(PHASE_A_MONTHS) or {x.get("month") for x in files}!=set(PHASE_A_MONTHS): raise ValueError("VBTC_V2_PHASE_A_CHRONOLOGY_INVALID")
    import pyarrow.parquet as pq
    rows=[]; required={"bar_start_utc","open","high","low","close","volume","daily_vwap"}
    for item in sorted(files,key=lambda x:x["month"]):
        rel=item.get("relative_path")
        if not isinstance(rel,str) or Path(rel).is_absolute() or ".." in Path(rel).parts: raise ValueError("VBTC_V2_PHASE_A_MANIFEST_SCHEMA_INVALID")
        path=(manifest_path.parent/rel).resolve()
        if not path.is_file() or _file_hash(path).lower()!=str(item.get("sha256","")).lower(): raise ValueError("VBTC_V2_PHASE_A_PARQUET_HASH_MISMATCH")
        table=pq.read_table(path)
        if not required.issubset(table.column_names): raise ValueError("VBTC_V2_PHASE_A_BAR_SCHEMA_INVALID")
        for row in table.select(sorted(required)).to_pylist():
            stamp=row.pop("bar_start_utc")
            if not isinstance(stamp,datetime) or stamp.tzinfo is None or stamp.utcoffset()!=timedelta(0): raise ValueError("VBTC_V2_TIMESTAMP_NOT_UTC")
            row["timestamp"]=_stamp(stamp)
            try: values={k:_d(row[k]) for k in ("open","high","low","close","volume","daily_vwap")}
            except Exception as exc: raise ValueError("VBTC_V2_PHASE_A_BAR_SCHEMA_INVALID") from exc
            if any(not x.is_finite() for x in values.values()) or any(values[k]<=0 for k in ("open","high","low","close","daily_vwap")) or values["volume"]<0 or values["high"]<max(values["open"],values["close"]) or values["low"]>min(values["open"],values["close"]) or values["high"]<values["low"]: raise ValueError("VBTC_V2_PHASE_A_BAR_SCHEMA_INVALID")
            rows.append(row)
    rows.sort(key=lambda x:_timestamp(x["timestamp"])); stamps=[_timestamp(x["timestamp"]) for x in rows]
    if not stamps or stamps[0]!=PHASE_A_START or stamps[-1]!=PHASE_A_LAST or any(a>=b or b-a!=timedelta(minutes=5) for a,b in zip(stamps,stamps[1:])): raise ValueError("VBTC_V2_PHASE_A_CHRONOLOGY_INVALID")
    return manifest,rows

def _metrics(trades:list[dict[str,Any]],setups:list[dict[str,Any]])->dict[str,Any]:
    pnl=[_d(x["net_pnl"]) for x in trades]; nr=[_d(x["net_r"]) for x in trades]; positive=sum((x for x in pnl if x>0),Decimal()); negative=-sum((x for x in pnl if x<0),Decimal())
    curve=peak=dd=Decimal()
    for x in nr: curve+=x; peak=max(peak,curve); dd=max(dd,peak-curve)
    directions=Counter(x["direction"] for x in trades); monthly={m:Decimal() for m in PHASE_A_MONTHS}; monthly_counts=Counter()
    for x in trades:
        month=x["entry_timestamp"][:7]; monthly[month]+=_d(x["net_pnl"]); monthly_counts[month]+=1
    direction_average={direction:str(sum((_d(x['net_r']) for x in trades if x['direction']==direction),Decimal())/directions[direction]) if directions[direction] else "0" for direction in ("LONG","SHORT")}
    return {"executed_trades":len(trades),"annualized_trades":str(Decimal(len(trades))*Decimal("365.25")/Decimal("396")),"net_pnl":str(sum(pnl,Decimal())),"net_profit_factor":str(positive/negative if negative else Decimal("Infinity")),"average_net_r":str(sum(nr,Decimal())/len(nr) if nr else Decimal()),"maximum_drawdown_r":str(dd),"monthly_net_pnl":{k:str(v) for k,v in monthly.items()},"monthly_trade_counts":dict(monthly_counts),"direction_counts":dict(directions),"direction_average_net_r":direction_average,"outcome_counts":dict(Counter(x["terminal_disposition"] for x in setups)),"net_r":[str(x) for x in nr]}

def _gates(metrics:dict[str,Any],trades:list[dict[str,Any]],reconciles:bool)->dict[str,Any]:
    # The named checks mirror the V1 hard-gate families; no candidate with an omitted check can rank.
    n=len(trades); frequency=Decimal(metrics["annualized_trades"]); net=Decimal(metrics["net_pnl"]); nr=[Decimal(x) for x in metrics["net_r"]]; dirs=Counter(x["direction"] for x in trades); months=[Decimal(x) for x in metrics["monthly_net_pnl"].values()]
    positive_total=sum((x for x in months if x>0),Decimal()); top_five=sorted((_d(x['net_pnl']) for x in trades),reverse=True)[:5]
    quarterly=[sum(months[i:i+3],Decimal()) for i in range(0,12,3)]
    checks={"integrity_reconciliation":reconciles,"frequency_100_to_500":Decimal(100)<=frequency<=Decimal(500),"positive_net_pnl":net>0,"net_pf":Decimal(metrics["net_profit_factor"])>=Decimal("1.30"),"positive_average_net_r":Decimal(metrics["average_net_r"])>0,"maximum_drawdown":Decimal(metrics["maximum_drawdown_r"])<=20,"profitable_months":sum(x>0 for x in months)>=8,"zero_trade_months":sum(metrics["monthly_trade_counts"].get(month,0)==0 for month in PHASE_A_MONTHS)<=3,"best_month_concentration":bool(positive_total) and max(months)<=Decimal('.35')*positive_total,"best_five_concentration":bool(positive_total) and sum(top_five,Decimal())<=Decimal('.30')*positive_total,"direction_mix":n>0 and dirs["LONG"]*4>=n and dirs["SHORT"]*4>=n,"direction_average":Decimal(metrics['direction_average_net_r']['LONG'])>=Decimal('-.15') and Decimal(metrics['direction_average_net_r']['SHORT'])>=Decimal('-.15'),"quarters_nonnegative":sum(x>=0 for x in quarterly)>=3}
    # deterministic robustness is deliberately still calculated for failures.
    import numpy as np
    values=np.array([float(x) for x in nr]); samples=np.random.Generator(np.random.PCG64(20240131)).choice(values,size=(10000,n),replace=True).mean(axis=1) if n else np.zeros(10000)
    checks.update({"bootstrap_median":float(np.median(samples))>0,"bootstrap_lower":float(np.percentile(samples,2.5))>=-.025,"extra_tick_cost":net-Decimal(n)*_cfg.price_tick*_cfg.quantity_btc*2>0,"best_trade_removal":net-max((_d(x["net_pnl"]) for x in trades),default=Decimal())>0})
    return {"passed":all(checks.values()),"checks":checks,"frequency_status":"UNDERFREQUENCY_FAIL" if frequency<100 else "OVERFREQUENCY_FAIL" if frequency>500 else "PASS","bootstrap":{"seed":20240131,"resamples":10000,"median_mean_net_r":float(np.median(samples)),"lower_2_5":float(np.percentile(samples,2.5))}}

def run_phase_a(*,phase_a_bars_manifest:str|Path,artifact_root:str|Path,repository_root:str|Path)->dict[str,Any]:
    manifest=Path(phase_a_bars_manifest); root=Path(repository_root); out=Path(artifact_root)
    if not manifest.is_absolute() or not out.is_absolute() or not root.is_absolute(): raise ValueError("VBTC_V2_ABSOLUTE_PATHS_REQUIRED")
    root=root.resolve(); verify_sealed_specification(root); revision=_clean_git(root)
    if not manifest.is_file() or _file_hash(manifest).lower()!=PHASE_A_MANIFEST_SHA256: raise ValueError("VBTC_V2_PHASE_A_MANIFEST_HASH_MISMATCH")
    _assert_output_absent(out.resolve())
    payload,bars=_load_phase_a_bars(manifest); store=_Store(out.resolve(),PHASE_A_MANIFEST_SHA256,revision); now=datetime.now(timezone.utc).isoformat().replace("+00:00","Z"); common={"schema_version":"VBTC-V2","study_id":STUDY_ID,"phase":"PHASE_A","specification_sha256":SPEC_SHA256,"data_manifest_sha256":PHASE_A_MANIFEST_SHA256,"execution_assumption_hash":_hash(EXECUTION_ASSUMPTIONS),"code_revision":revision,"created_at_utc":now,"phase_b":"NOT_OPENED","alpha":"NOT_OPENED"}
    store.write("sealed-specification.json",{**common,"text":(root/SPEC_PATH).read_text(encoding="utf-8"),"seal_status":"SEALED"}); store.write("candidate-registry.json",{**common,"candidates":[{"candidate_id":x,"target_r":str(y)} for x,y in CANDIDATES],"only_target_differs":True}); store.write("data-manifest.json",{**common,"identity":payload["identity"],"validated":True,"phase_b":"NOT_OPENED"})
    results=[]
    for cid,target in CANDIDATES:
        setups,events,trades=evaluate_bars(bars,cid,target); rec=reconciliation_summary(setups,events,trades); metrics=_metrics(trades,setups); gates=_gates(metrics,trades,rec["reconciles"]); config={**common,"candidate_id":cid,"target_r":str(target),"execution_assumption_hash":_hash(EXECUTION_ASSUMPTIONS),"code_revision":revision}
        for name,data in (("configuration.json",config),("events.json",{**common,"candidate_id":cid,"events":events}),("trades.json",{**common,"candidate_id":cid,"trades":trades}),("setup_outcomes.json",{**common,"candidate_id":cid,"setup_outcomes":setups}),("monthly_metrics.json",{**common,"candidate_id":cid,"monthly_net_pnl":metrics["monthly_net_pnl"]}),("report.json",{**common,"candidate_id":cid,**metrics,"event_reconciliation":rec}),("gates.json",{**common,"candidate_id":cid,**gates})): store.write(name,data,cid)
        results.append((cid,config,metrics,gates))
    passing=sorted((x for x in results if x[3]["passed"]),key=lambda x:(-Decimal(x[2]["average_net_r"]),-Decimal(x[2]["net_profit_factor"]),Decimal(x[2]["maximum_drawdown_r"]),x[0])); chosen=passing[0] if passing else None
    store.write("selection_report.json",{**common,"ranking":[x[0] for x in passing],"selected_candidate_id":chosen[0] if chosen else None,"status":"PHASE_A_SELECTED" if chosen else "PHASE_A_NO_ROBUST_CANDIDATE"}); store.write("freeze.json",{**common,"status":"FROZEN" if chosen else "NOT_FROZEN","candidate_id":chosen[0] if chosen else None,"phase_b":"NOT_OPENED","alpha":"NOT_OPENED"})
    final={**common,"status":"PHASE_A_SELECTED" if chosen else "PHASE_A_NO_ROBUST_CANDIDATE","summary":"Deterministic sealed VBTC V2 Phase A completed; Phase B and Alpha remain unopened.","realStudyExecuted":True,"marketDataAccessed":True,"phaseBStatus":"NOT_OPENED","alphaStatus":"NOT_OPENED","model":"gpt-5.6-terra"}; store.write("final_report.json",final); store.seal(); return {**final,"artifactRoot":str(store.root)}
