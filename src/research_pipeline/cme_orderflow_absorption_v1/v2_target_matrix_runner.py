"""One-pass target-matrix replay for seen Aug V2 research."""
from __future__ import annotations
import csv, json
from copy import deepcopy
from pathlib import Path
from typing import Any
from databento import DBNStore
from .oos_backtest_runner import COMMISSION, RAW_TICK, RISK_BUDGET, TICK, USD_PER_POINT, CausalMBOBook, _cutoff, _valid, day_and_seconds, digest, is_snapshot_record, load_frozen, prices, size_trade

ROOT=Path(__file__).resolve().parents[3]
DBN=ROOT/'data/cme_orderflow_absorption_v1/oos_v1/ESU6/mbo/ESU6_2026-08-03_2026-08-08_mbo.dbn'
RESEARCH_CSV=ROOT/'research_runs/CMEOrderflowAbsorption.ES_V2_RESEARCH/seen_15_rth/all-interactions.csv'
OUT=ROOT/'research_runs/CMEOrderflowAbsorption.ES_V2_RESEARCH/seen_aug_target_matrix'
EXPECTED_SOURCE_SHA='BE4B56639E56DF9AACE81621E4E276463EA8AF889104F35F1744400310D53AA3'
EXPECTED_PLUS=21
CONFIRM_NS=15_000_000_000
LATENCY_NS=2_000_000
MIN_FAVORABLE_TICKS=1.0
PROGRESS_EVERY=5_000_000
TARGET_MULTIPLES=(1.5,2.0,3.0,4.0)
MES_USD_PER_POINT=5.0
MES_COMMISSION_PER_SIDE=1.25

def action_value(rec:Any)->str:return str(getattr(rec.action,'value',rec.action))

def load_seen_aug_plus():
 rows=[]
 with RESEARCH_CSV.open(newline='',encoding='utf-8') as f:
  for r in csv.DictReader(f):
   if r['research_split']!='SEEN_OOS_AUG' or r['v1_plus']!='True':continue
   rows.append({'interaction_id':r['interaction_id'],'date':r['date'],'direction':r['direction'],'level':r['level'],'interaction_end':int(r['interaction_end']),'end_price':int(r['end_price']),'zone_low':int(r['zone_low']),'zone_high':int(r['zone_high']),'absorption_score':float(r['absorption_score']),'replenishment_score':float(r['replenishment_score'])})
 if len(rows)!=EXPECTED_PLUS:raise RuntimeError(f'seen-Aug V1 PLUS count={len(rows)}, expected {EXPECTED_PLUS}')
 return sorted(rows,key=lambda r:r['interaction_end'])

def size_trade_with_mes_fallback(p):
 es=size_trade(p)
 if es['contracts']>=1:return {**es,'instrument':'ES','usd_per_point':USD_PER_POINT,'commission_per_side':COMMISSION}
 mes_raw=abs(p['entry_reference']-p['stop'])*MES_USD_PER_POINT
 mes_slip=(abs(p['entry']-p['entry_reference'])+abs(p['stop_exit']-p['stop']))*MES_USD_PER_POINT
 mes_initial=abs(p['entry']-p['stop_exit'])*MES_USD_PER_POINT+2*MES_COMMISSION_PER_SIDE
 return {'raw_price_risk_usd':mes_raw,'slippage_contribution_usd':mes_slip,'one_contract_price_risk_usd':mes_raw+mes_slip,'one_contract_initial_risk_usd':mes_initial,'contracts':int(RISK_BUDGET//mes_initial),'instrument':'MES','usd_per_point':MES_USD_PER_POINT,'commission_per_side':MES_COMMISSION_PER_SIDE}

def target_for(pos,m):
 return pos['entry']+m*(pos['entry']-pos['stop']) if pos['direction']=='LONG' else pos['entry']-m*(pos['stop']-pos['entry'])

def close_trade(pos,ts,exit_fill,reason,m):
 sign=1 if pos['direction']=='LONG' else -1
 gross=(exit_fill-pos['entry'])*sign*pos['usd_per_point']*pos['contracts']
 fees=2*pos['commission_per_side']*pos['contracts']
 risk=pos['one_contract_initial_risk_usd']*pos['contracts']
 return {**pos,'target_multiple':m,'exit_timestamp':ts,'exit_fill':exit_fill,'exit_reason':reason,'gross_usd':gross,'commission_usd':fees,'net_usd':gross-fees,'r_multiple':gross/risk}

def main():
 _,manifest=load_frozen();source_sha=digest(DBN)
 if source_sha!=EXPECTED_SOURCE_SHA:raise RuntimeError(f'source SHA mismatch: {source_sha}')
 waiting=[{**r,'confirmation_due_ns':r['interaction_end']+CONFIRM_NS} for r in load_seen_aug_plus()]
 ready=[];book=CausalMBOBook();records=passed=failed=0
 positions={m:None for m in TARGET_MULTIPLES};trades={m:[] for m in TARGET_MULTIPLES};cutoff_quotes={m:None for m in TARGET_MULTIPLES}
 for rec in DBNStore.from_file(DBN):
  records+=1
  if is_snapshot_record(rec,manifest):
   book.apply(action=rec.action,side=rec.side,price=rec.price,size=rec.size,order_id=rec.order_id,sequence=rec.sequence,ts_recv=rec.ts_recv,channel_id=rec.channel_id,validate_sequence=False,mutate_execution=False);continue
  if action_value(rec)=='N':
   if records%PROGRESS_EVERY==0:print(f'[target matrix] records={records:,}',flush=True)
   continue
  applied=book.apply(action=rec.action,side=rec.side,price=rec.price,size=rec.size,order_id=rec.order_id,sequence=rec.sequence,ts_recv=rec.ts_recv,channel_id=rec.channel_id,validate_sequence=False,mutate_execution=False)
  day,_=day_and_seconds(rec.ts_recv);quote=_valid(book);cutoff=_cutoff(day)
  if quote and cutoff-1_000_000_000<=rec.ts_recv<=cutoff:
   for m in TARGET_MULTIPLES:cutoff_quotes[m]=(rec.ts_recv,quote)
  for m in TARGET_MULTIPLES:
   pos=positions[m]
   if pos is not None and rec.ts_recv>pos['cutoff_ns']:
    cq=cutoff_quotes[m]
    if cq is None:raise RuntimeError(f'CUTOFF_EXECUTION_INTEGRITY_FAILURE target={m}')
    ts,q=cq;ref=q[0] if pos['direction']=='LONG' else q[1];fill=ref-TICK if pos['direction']=='LONG' else ref+TICK
    trades[m].append(close_trade(pos,ts,fill,'CUTOFF_FORCED_FLAT',m));positions[m]=None;cutoff_quotes[m]=None
  waiting=[r for r in waiting if rec.ts_recv<_cutoff(r['date'])]
  if applied is not None and applied.executed:
   unresolved=[]
   for row in waiting:
    if row['confirmation_due_ns']<=rec.ts_recv:
     fav=(applied.price-row['end_price'])/RAW_TICK if row['direction']=='BUYER_ABSORPTION' else (row['end_price']-applied.price)/RAW_TICK
     if fav>=MIN_FAVORABLE_TICKS:
      passed+=1;ready.append({**row,'confirmation_timestamp':rec.ts_recv,'confirmation_price':applied.price/1e9,'confirmation_favorable_ticks':fav,'entry_ready_ns':rec.ts_recv+LATENCY_NS})
     else:failed+=1
    else:unresolved.append(row)
   waiting=unresolved;ready.sort(key=lambda r:(r['entry_ready_ns'],r['interaction_end'],r['interaction_id']))
  remaining=[]
  for row in ready:
   if row['entry_ready_ns']>rec.ts_recv:remaining.append(row);continue
   if rec.ts_recv>=_cutoff(row['date']) or quote is None:continue
   p=prices(row['direction'],quote[0],quote[1],row['zone_low']/1e9,row['zone_high']/1e9);s=size_trade_with_mes_fallback(p)
   if not s['contracts']:continue
   for m in TARGET_MULTIPLES:
    if positions[m] is not None:continue
    pos={**row,**p,**s,'entry_timestamp':rec.ts_recv,'cutoff_ns':_cutoff(row['date'])};pos['target']=target_for(pos,m);positions[m]=deepcopy(pos)
  ready=remaining
  if quote is not None:
   bid,ask=quote
   for m in TARGET_MULTIPLES:
    pos=positions[m]
    if pos is None:continue
    stop_hit=(pos['direction']=='LONG' and bid<=pos['stop']) or (pos['direction']=='SHORT' and ask>=pos['stop'])
    target_hit=(pos['direction']=='LONG' and bid>=pos['target']) or (pos['direction']=='SHORT' and ask<=pos['target'])
    if stop_hit or target_hit:
     ref=bid if pos['direction']=='LONG' else ask;fill=ref-TICK if pos['direction']=='LONG' else ref+TICK;reason='STOP' if stop_hit else 'TARGET'
     trades[m].append(close_trade(pos,rec.ts_recv,fill,reason,m));positions[m]=None
  if records%PROGRESS_EVERY==0:print(f'[target matrix] records={records:,}',flush=True)
 if any(positions[m] is not None for m in TARGET_MULTIPLES):raise RuntimeError('open position remained after source end')
 OUT.mkdir(parents=True,exist_ok=True);results=[]
 for m in TARGET_MULTIPLES:
  ts=trades[m];wins=[t for t in ts if t['net_usd']>0];losses=[t for t in ts if t['net_usd']<0];gp=sum(t['net_usd'] for t in wins);gl=abs(sum(t['net_usd'] for t in losses))
  results.append({'target_r':m,'trades':len(ts),'wins':len(wins),'losses':len(losses),'winrate_pct':100*len(wins)/len(ts) if ts else 0,'net_pnl_usd':sum(t['net_usd'] for t in ts),'profit_factor':gp/gl if gl else None,'total_r':sum(t['r_multiple'] for t in ts),'es_trades':sum(t['instrument']=='ES' for t in ts),'mes_trades':sum(t['instrument']=='MES' for t in ts)})
  fields=sorted({k for t in ts for k in t}) or ['interaction_id']
  with (OUT/f"trades_{str(m).replace('.','_')}R.csv").open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fields,lineterminator='\n');w.writeheader();w.writerows(ts)
 payload={'status':'TARGET_MATRIX_RESEARCH_READY','interpretation':'SEEN_AUG_DATA_NOT_FRESH_OOS_EVIDENCE','source_sha256':source_sha,'dbn_records_seen':records,'v1_plus_input':EXPECTED_PLUS,'confirmation_seconds':15,'minimum_favorable_ticks':1,'confirmations_passed':passed,'confirmations_failed':failed,'mes_commission_per_side':MES_COMMISSION_PER_SIDE,'mes_execution_model':'MES_PROXY_EXECUTION_FROM_ES_MBO','target_results':results}
 (OUT/'summary.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n',encoding='utf-8');print(json.dumps(payload,indent=2,sort_keys=True),flush=True);return 0

if __name__=='__main__':raise SystemExit(main())