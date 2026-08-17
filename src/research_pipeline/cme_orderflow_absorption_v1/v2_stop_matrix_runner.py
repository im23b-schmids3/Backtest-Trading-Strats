"""One-pass stop-matrix replay for seen Aug V2 research."""
from __future__ import annotations
import csv, json
from copy import deepcopy
from pathlib import Path
from typing import Any
from databento import DBNStore
from .oos_backtest_runner import COMMISSION, RAW_TICK, RISK_BUDGET, TICK, USD_PER_POINT, CausalMBOBook, _cutoff, _valid, day_and_seconds, digest, is_snapshot_record, load_frozen

ROOT = Path(__file__).resolve().parents[3]
DBN = ROOT / "data/cme_orderflow_absorption_v1/oos_v1/ESU6/mbo/ESU6_2026-08-03_2026-08-08_mbo.dbn"
RESEARCH_CSV = ROOT / "research_runs/CMEOrderflowAbsorption.ES_V2_RESEARCH/seen_15_rth/all-interactions.csv"
OUT = ROOT / "research_runs/CMEOrderflowAbsorption.ES_V2_RESEARCH/seen_aug_stop_matrix_3R"
EXPECTED_SOURCE_SHA = "BE4B56639E56DF9AACE81621E4E276463EA8AF889104F35F1744400310D53AA3"
EXPECTED_PLUS = 21
CONFIRM_NS = 15_000_000_000
LATENCY_NS = 2_000_000
MIN_FAVORABLE_TICKS = 1.0
PROGRESS_EVERY = 5_000_000
TARGET_R = 3.0
STOP_TICKS = (3,5,7,9)
MES_USD_PER_POINT = 5.0
MES_COMMISSION_PER_SIDE = 1.25

def action_value(rec: Any) -> str:
    return str(getattr(rec.action, "value", rec.action))

def load_seen_aug_plus():
    rows=[]
    with RESEARCH_CSV.open(newline="",encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["research_split"]!="SEEN_OOS_AUG" or r["v1_plus"]!="True":
                continue
            rows.append({
                "interaction_id":r["interaction_id"],"date":r["date"],"direction":r["direction"],"level":r["level"],
                "interaction_end":int(r["interaction_end"]),"end_price":int(r["end_price"]),
                "zone_low":int(r["zone_low"]),"zone_high":int(r["zone_high"])
            })
    if len(rows)!=EXPECTED_PLUS:
        raise RuntimeError(f"seen-Aug V1 PLUS count={len(rows)}, expected {EXPECTED_PLUS}")
    return sorted(rows,key=lambda r:r["interaction_end"])

def build_prices(direction,bid,ask,zone_low,zone_high,stop_ticks):
    if not bid>0 or not ask>bid:
        raise RuntimeError("invalid executable market")
    if direction=="BUYER_ABSORPTION":
        entry_reference=ask; entry=ask+TICK; stop=zone_low-stop_ticks*TICK; stop_exit=stop-TICK
        return {"entry_reference":entry_reference,"entry":entry,"stop":stop,"stop_exit":stop_exit,"target":entry+TARGET_R*(entry-stop),"direction":"LONG"}
    if direction=="SELLER_ABSORPTION":
        entry_reference=bid; entry=bid-TICK; stop=zone_high+stop_ticks*TICK; stop_exit=stop+TICK
        return {"entry_reference":entry_reference,"entry":entry,"stop":stop,"stop_exit":stop_exit,"target":entry-TARGET_R*(stop-entry),"direction":"SHORT"}
    raise RuntimeError("invalid direction")

def size_with_mes_fallback(p):
    raw=abs(p["entry_reference"]-p["stop"])*USD_PER_POINT
    slip=(abs(p["entry"]-p["entry_reference"])+abs(p["stop_exit"]-p["stop"]))*USD_PER_POINT
    es_initial=abs(p["entry"]-p["stop_exit"])*USD_PER_POINT+2*COMMISSION
    es_contracts=int(RISK_BUDGET//es_initial)
    if es_contracts>=1:
        return {"raw_price_risk_usd":raw,"slippage_contribution_usd":slip,"one_contract_price_risk_usd":raw+slip,
                "one_contract_initial_risk_usd":es_initial,"contracts":es_contracts,"instrument":"ES","execution_model":"ES_NATIVE",
                "usd_per_point":USD_PER_POINT,"commission_per_side":COMMISSION}
    mes_raw=abs(p["entry_reference"]-p["stop"])*MES_USD_PER_POINT
    mes_slip=(abs(p["entry"]-p["entry_reference"])+abs(p["stop_exit"]-p["stop"]))*MES_USD_PER_POINT
    mes_initial=abs(p["entry"]-p["stop_exit"])*MES_USD_PER_POINT+2*MES_COMMISSION_PER_SIDE
    return {"raw_price_risk_usd":mes_raw,"slippage_contribution_usd":mes_slip,"one_contract_price_risk_usd":mes_raw+mes_slip,
            "one_contract_initial_risk_usd":mes_initial,"contracts":int(RISK_BUDGET//mes_initial),"instrument":"MES",
            "execution_model":"MES_PROXY_EXECUTION_FROM_ES_MBO","usd_per_point":MES_USD_PER_POINT,"commission_per_side":MES_COMMISSION_PER_SIDE}

def close_trade(position,ts,exit_fill,reason,stop_ticks):
    sign=1 if position["direction"]=="LONG" else -1
    gross=(exit_fill-position["entry"])*sign*position["usd_per_point"]*position["contracts"]
    fees=2*position["commission_per_side"]*position["contracts"]
    risk=position["one_contract_initial_risk_usd"]*position["contracts"]
    return {**position,"stop_ticks":stop_ticks,"target_r":TARGET_R,"exit_timestamp":ts,"exit_fill":exit_fill,"exit_reason":reason,
            "gross_usd":gross,"commission_usd":fees,"net_usd":gross-fees,"r_multiple":gross/risk}

def main():
    _,manifest=load_frozen()
    source_sha=digest(DBN)
    if source_sha!=EXPECTED_SOURCE_SHA:
        raise RuntimeError(f"source SHA mismatch: {source_sha}")
    waiting=[{**r,"confirmation_due_ns":r["interaction_end"]+CONFIRM_NS} for r in load_seen_aug_plus()]
    ready=[]
    positions={s:None for s in STOP_TICKS}
    trades={s:[] for s in STOP_TICKS}
    cutoff_quotes={s:None for s in STOP_TICKS}
    passed=failed=records=0
    book=CausalMBOBook()

    for rec in DBNStore.from_file(DBN):
        records+=1
        if is_snapshot_record(rec,manifest):
            book.apply(action=rec.action,side=rec.side,price=rec.price,size=rec.size,order_id=rec.order_id,sequence=rec.sequence,ts_recv=rec.ts_recv,channel_id=rec.channel_id,validate_sequence=False,mutate_execution=False)
            continue
        if action_value(rec)=="N":
            if records%PROGRESS_EVERY==0: print(f"[stop matrix] records={records:,}",flush=True)
            continue

        applied=book.apply(action=rec.action,side=rec.side,price=rec.price,size=rec.size,order_id=rec.order_id,sequence=rec.sequence,ts_recv=rec.ts_recv,channel_id=rec.channel_id,validate_sequence=False,mutate_execution=False)
        day,_=day_and_seconds(rec.ts_recv); quote=_valid(book); cutoff=_cutoff(day)

        if quote and cutoff-1_000_000_000<=rec.ts_recv<=cutoff:
            for s in STOP_TICKS: cutoff_quotes[s]=(rec.ts_recv,quote)

        for s in STOP_TICKS:
            p=positions[s]
            if p is not None and rec.ts_recv>p["cutoff_ns"]:
                cq=cutoff_quotes[s]
                if cq is None: raise RuntimeError(f"CUTOFF_EXECUTION_INTEGRITY_FAILURE stop={s}")
                ts,q=cq; exit_ref=q[0] if p["direction"]=="LONG" else q[1]; exit_fill=exit_ref-TICK if p["direction"]=="LONG" else exit_ref+TICK
                trades[s].append(close_trade(p,ts,exit_fill,"CUTOFF_FORCED_FLAT",s)); positions[s]=None; cutoff_quotes[s]=None

        waiting=[r for r in waiting if rec.ts_recv<_cutoff(r["date"])]

        if applied is not None and applied.executed:
            unresolved=[]
            for row in waiting:
                if row["confirmation_due_ns"]<=rec.ts_recv:
                    fav=((applied.price-row["end_price"])/RAW_TICK) if row["direction"]=="BUYER_ABSORPTION" else ((row["end_price"]-applied.price)/RAW_TICK)
                    if fav>=MIN_FAVORABLE_TICKS:
                        passed+=1
                        ready.append({**row,"confirmation_timestamp":rec.ts_recv,"confirmation_price":applied.price/1e9,"confirmation_favorable_ticks":fav,"entry_ready_ns":rec.ts_recv+LATENCY_NS})
                    else:
                        failed+=1
                else:
                    unresolved.append(row)
            waiting=unresolved
            ready.sort(key=lambda r:(r["entry_ready_ns"],r["interaction_end"],r["interaction_id"]))

        remaining=[]
        for row in ready:
            if row["entry_ready_ns"]>rec.ts_recv:
                remaining.append(row); continue
            if rec.ts_recv>=_cutoff(row["date"]) or quote is None:
                continue
            for s in STOP_TICKS:
                if positions[s] is not None:
                    continue
                p=build_prices(row["direction"],quote[0],quote[1],row["zone_low"]/1e9,row["zone_high"]/1e9,s)
                sizing=size_with_mes_fallback(p)
                if not sizing["contracts"]:
                    continue
                positions[s]=deepcopy({**row,**p,**sizing,"entry_timestamp":rec.ts_recv,"cutoff_ns":_cutoff(row["date"])})
        ready=remaining

        if quote is not None:
            bid,ask=quote
            for s in STOP_TICKS:
                p=positions[s]
                if p is None: continue
                stop_hit=(p["direction"]=="LONG" and bid<=p["stop"]) or (p["direction"]=="SHORT" and ask>=p["stop"])
                target_hit=(p["direction"]=="LONG" and bid>=p["target"]) or (p["direction"]=="SHORT" and ask<=p["target"])
                if not (stop_hit or target_hit): continue
                exit_ref=bid if p["direction"]=="LONG" else ask
                exit_fill=exit_ref-TICK if p["direction"]=="LONG" else exit_ref+TICK
                trades[s].append(close_trade(p,rec.ts_recv,exit_fill,"STOP" if stop_hit else "TARGET",s))
                positions[s]=None

        if records%PROGRESS_EVERY==0: print(f"[stop matrix] records={records:,}",flush=True)

    if any(positions[s] is not None for s in STOP_TICKS):
        raise RuntimeError("open position remained after source end")

    OUT.mkdir(parents=True,exist_ok=True)
    results=[]
    for s in STOP_TICKS:
        ts=trades[s]
        wins=[t for t in ts if t["net_usd"]>0]; losses=[t for t in ts if t["net_usd"]<0]
        gp=sum(t["net_usd"] for t in wins); gl=abs(sum(t["net_usd"] for t in losses))
        results.append({"stop_ticks":s,"target_r":TARGET_R,"trades":len(ts),"wins":len(wins),"losses":len(losses),
                        "winrate_pct":100*len(wins)/len(ts) if ts else 0,"net_pnl_usd":sum(t["net_usd"] for t in ts),
                        "profit_factor":gp/gl if gl else None,"total_r":sum(t["r_multiple"] for t in ts),
                        "es_trades":sum(t["instrument"]=="ES" for t in ts),"mes_trades":sum(t["instrument"]=="MES" for t in ts)})
        fields=sorted({k for t in ts for k in t}) or ["interaction_id"]
        with (OUT/f"trades_stop_{s}ticks.csv").open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fields,lineterminator="\n"); w.writeheader(); w.writerows(ts)

    payload={"status":"STOP_MATRIX_RESEARCH_READY","interpretation":"SEEN_AUG_DATA_NOT_FRESH_OOS_EVIDENCE",
             "source_sha256":source_sha,"dbn_records_seen":records,"v1_plus_input":EXPECTED_PLUS,
             "confirmation_seconds":15,"minimum_favorable_ticks":1,"target_r":TARGET_R,
             "confirmations_passed":passed,"confirmations_failed":failed,
             "mes_commission_per_side":MES_COMMISSION_PER_SIDE,"mes_execution_model":"MES_PROXY_EXECUTION_FROM_ES_MBO",
             "stop_results":results}
    (OUT/"summary.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(payload,indent=2,sort_keys=True),flush=True)
    return 0

if __name__=="__main__":
    raise SystemExit(main())