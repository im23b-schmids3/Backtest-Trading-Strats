"""Frozen, causal ES HVN/heavy-aggression rejection study on Spring TRAIN."""
from __future__ import annotations

import argparse, csv, gzip, hashlib, json, math, os, statistics, time
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import mac_2025_es_only_train_baseline as baseline
from . import mac_2025_native_mbp_quote as quote_plan

RUN_ID = "ES_HVN_HEAVY_ABSORPTION_RESEARCH_V1"
OUT_ROOT = Path("research_runs") / RUN_ID
DATA_ROOT = baseline.DATA_ROOT
TAPE_ROOT = baseline.OUTPUT_ROOT / "candidate-tapes" / "tapes"
TICK = 0.25
BAR_NS = 60_000_000_000
PROFILE_NS = 30 * BAR_NS
RESPONSE_NS = 1_000_000_000
CONFIRM_NS = 15_000_000_000
ENTRY_DELAY_NS = 2_000_000
RR_GRID = (1.0, 1.5, 2.0, 2.25, 2.5, 2.75, 3.0)
STOP_BUFFER_TICKS = 1
ENTRY_SLIPPAGE_TICKS = 1
STOP_SLIPPAGE_TICKS = 1
COMMISSION_USD = 6.0
POINT_VALUE_USD = 50.0
CHECKPOINT_VERSION = "hvn-heavy-absorption-v1-date-v5"

# This entire definition is frozen before the first outcome is evaluated.
SPEC: dict[str, Any] = {
    "study_id": "ES_HVN_HEAVY_ABSORPTION_RESEARCH_V1",
    "source": "sealed source-hash-bound canonical native ES GLBX.MDP3 mbp-10 event tape",
    "profile": {
        "period": "30 minutes", "anchor": "canonical session open",
        "smoothing": "centered 5-tick box sum over executed volume, zero-filled untraded ticks",
        "peak": "smoothed local maximum against adjacent ticks; equal plateaus choose lower tick",
        "screen": "peak >= 90th percentile of positive smoothed tick values in that profile",
        "clustering": "peaks within 2 ticks grouped; bounds extend one tick around cluster extrema",
        "poc": "highest smoothed tick, lower-tick tie; zone volume sums raw executed volume in zone",
        "availability": "only after source 30-minute profile closes",
        "retirement": "first causal trade in zone +/- 1 tick",
    },
    "heavy_print": {
        "lookback": "100 completed one-minute bars", "threshold": "mean + 2 population standard deviations",
        "history": "prior bars only; all available completed bars during warmup; at least one completed bar",
        "print_unit": "one positive-size native T record; no inferred parent-order grouping",
        "aggressor": "T side B=buyer initiated; A=seller initiated; other side unknown",
        "unknown_side": "included in size history, excluded from directional signal",
        "minimum_volume_floor": "disabled",
    },
    "approach_touch": "one first touch per zone, after approach from above (long) or below (short); trade enters zone +/- 1 tick",
    "long_confirmation": "first SELL heavy print within zone +/-2 ticks and 15s of touch; wait 1s for failure evidence; invalidate below zone_low-2 ticks; confirm first subsequent trade >= POC+1 tick within 15s of heavy print",
    "short_confirmation": "directional mirror of long confirmation",
    "stop": "long=min(zone_low,rejection_low)-1 tick; short=max(zone_high,rejection_high)+1 tick",
    "entry": "first valid BBO at/after confirmation+2ms; buy ask+1 tick, sell bid-1 tick",
    "target": "predeclared RR grid; target rounded away from entry to next valid ES tick",
    "target_fill": "require actual trade-through by >=1 tick; limit filled at target; BBO-mid touches do not fill",
    "stop_fill": "trigger on executable exit-side BBO; market exit adverse one tick; stop wins same-record ambiguity",
    "session_end": "flatten at last valid executable BBO before the containing canonical session closes",
    "costs": {"commission_round_trip_usd": 6, "entry_slippage_ticks": 1, "stop_slippage_ticks": 1,
              "spread": "paid through executable ask/bid"},
    "no_optimization": True, "no_october": True, "no_protected_oos": True, "no_download": True,
}
SPEC_SHA256 = hashlib.sha256(json.dumps(SPEC, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class ResearchError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _json_default(v: Any) -> Any:
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    raise TypeError(type(v).__name__)


def _atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, sort_keys=True, indent=2, default=_json_default, allow_nan=False) + "\n")
    os.replace(tmp, path)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _write_gz_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
            for row in rows:
                gz.write((json.dumps(row, sort_keys=True, separators=(",", ":"), default=_json_default,
                                     allow_nan=False) + "\n").encode())
    os.replace(tmp, path)


def checkpoint_matches(payload: Mapping[str, Any], identity: Mapping[str, Any]) -> bool:
    return (payload.get("version") == CHECKPOINT_VERSION
        and payload.get("spec_sha256") == SPEC_SHA256
        and payload.get("source_sha256") == identity.get("raw_sha256")
        and payload.get("tape_sha256") == identity.get("tape_sha256")
        and isinstance(payload.get("result"), dict))


def _sources() -> tuple[list[str], dict[str, dict[str, Any]]]:
    path = DATA_ROOT / baseline.MANIFEST_NAME
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ResearchError(f"native MBP-10 manifest unreadable: {path}") from exc
    if manifest.get("status") != "COMPLETE":
        raise ResearchError("native MBP-10 manifest is not COMPLETE")
    dates = list(baseline.TRAIN_DATES)
    rows = {str(r.get("session_date")): r for r in manifest.get("requests", {}).values()
            if isinstance(r, dict) and r.get("category") == "TRAIN" and r.get("schema") == "mbp-10"}
    if len(dates) != 35 or set(rows) != set(dates):
        raise ResearchError(f"expected exactly 35 TRAIN source rows, got {len(rows)}")
    return dates, rows


def load_events(day: str, source: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    raw = DATA_ROOT / str(source.get("path", ""))
    if not raw.is_file() or raw.stat().st_size != int(source.get("bytes", -1)):
        raise ResearchError(f"missing/size-mismatched DBN source: {raw}")
    digest = _sha(raw)
    if digest != source.get("sha256") or source.get("status") != "VERIFIED":
        raise ResearchError(f"DBN integrity/status failure: {raw}")
    if source.get("schema") != "mbp-10" or not source.get("symbol"):
        raise ResearchError(f"invalid native source identity for {day}")
    expected_symbol=quote_plan.contract_for(date.fromisoformat(day))[0]
    if source.get("symbol")!=expected_symbol:
        raise ResearchError(f"contract mismatch for {day}: got {source.get('symbol')}, expected {expected_symbol}")
    try:
        from databento import DBNStore
        dbn_meta=DBNStore.from_file(raw).metadata
    except Exception as exc:
        raise ResearchError(f"DBN header cannot be read for {day}: {exc}") from exc
    if dbn_meta.dataset!="GLBX.MDP3" or dbn_meta.schema!="mbp-10" or expected_symbol not in dbn_meta.symbols:
        raise ResearchError(f"DBN header identity mismatch for {day}: dataset/schema/symbol")
    tm_path = TAPE_ROOT / f"{day}-candidate-tape-manifest.json"
    tape_path = TAPE_ROOT / f"{day}-candidate-tape.npz"
    try:
        tm = json.loads(tm_path.read_text())
        with np.load(tape_path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            events = np.asarray(archive["events"])
    except Exception as exc:
        raise ResearchError(f"sealed canonical event tape unavailable/unreadable for {day}: {exc}") from exc
    if (tm.get("date") != day or tm.get("source_sha256") != digest or
            metadata.get("date") != day or metadata.get("source_sha256") != digest):
        raise ResearchError(f"source hash/date mismatch in canonical tape for {day}")
    if tm.get("tape_version") not in {"MAC2025_CANDIDATE_TAPE_V1", "MAC2025_CANDIDATE_TAPE_V2_BBO_COMPLETE"}:
        raise ResearchError(f"unsupported sealed tape version for {day}")
    if not len(events) or np.any(np.diff(events["timestamp_ns"].astype(np.int64)) < 0):
        raise ResearchError(f"empty or unordered tape for {day}")
    valid = np.isfinite(events["bid"]) & np.isfinite(events["ask"]) & (events["ask"] > events["bid"])
    if not np.all(valid):
        raise ResearchError(f"non-executable BBO in sealed tape for {day}")
    identity = {
        "date": day, "raw_path": str(raw), "raw_sha256": digest, "raw_bytes": raw.stat().st_size,
        "dataset": "GLBX.MDP3", "schema": "mbp-10", "symbol": source["symbol"],
        "tape_path": str(tape_path), "tape_sha256": _sha(tape_path), "tape_version": tm["tape_version"],
        "event_count": len(events), "trade_records": int(np.count_nonzero(events["execution_size"] > 0)),
        "unknown_side_trades": int(np.count_nonzero((events["execution_size"] > 0) & (events["aggressor"] == 0))),
        "first_ns": int(events["timestamp_ns"][0]), "last_ns": int(events["timestamp_ns"][-1]),
        "equal_timestamp_order": "sealed source order preserved",
        "ts_event_and_sequence_retained": False,
    }
    return events, identity


def _windows(day: str) -> dict[int, tuple[int, int]]:
    raw = baseline._session_windows(day)
    return {i: tuple(map(int, raw[name])) for i, name in enumerate(("ASIA", "EUROPE", "NY"))}


def _trades(events: np.ndarray) -> np.ndarray:
    rows = events[(events["execution_size"] > 0) & np.isfinite(events["execution_price"])]
    ticks = rows["execution_price"].astype(np.float64) / TICK
    if np.any(np.abs(ticks - np.rint(ticks)) > 1e-7):
        raise ResearchError("off-tick native ES trade")
    return rows


def _zones(trades: np.ndarray, day: str) -> list[dict[str, Any]]:
    ts = trades["timestamp_ns"].astype(np.int64)
    px = np.rint(trades["execution_price"].astype(np.float64) / TICK).astype(np.int64)
    qty = trades["execution_size"].astype(np.int64)
    windows = _windows(day)
    first = min(x[0] for x in windows.values())
    last = max(x[1] for x in windows.values())
    zones = []
    for sid, (session_lo, session_hi) in windows.items():
        edges = np.arange(session_lo, session_hi + PROFILE_NS, PROFILE_NS, dtype=np.int64)
        for pidx, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            a, b = np.searchsorted(ts, (lo, hi), side="left")
            if b <= a:
                continue
            u, inv = np.unique(px[a:b], return_inverse=True)
            vol = np.bincount(inv, weights=qty[a:b]).astype(np.int64)
            sm = np.convolve(np.pad(vol.astype(float),(2,2)), np.ones(5), mode="valid")
            pos = sm[sm > 0]
            if len(pos) < 3:
                continue
            cutoff = float(np.quantile(pos, .90))
            peaks = [i for i in range(1, len(u)-1)
                     if sm[i] >= cutoff and sm[i] >= sm[i-1] and sm[i] > sm[i+1]]
            groups: list[list[int]] = []
            for i in peaks:
                t = int(u[i])
                if groups and t - groups[-1][-1] <= 2:
                    groups[-1].append(t)
                else:
                    groups.append([t])
            for ci, group in enumerate(groups):
                low, high = min(group)-1, max(group)+1
                mask = (u >= low) & (u <= high)
                poc = min(group, key=lambda t: (-sm[int(np.searchsorted(u, t))], t))
                zones.append({"date": day, "zone_id": f"{day}:{sid}:{pidx}:{ci}", "session": sid,
                              "profile_id": pidx, "profile_start_ns": int(lo), "profile_close_ns": int(hi),
                              "active_from_ns": int(hi), "zone_low": low*TICK, "zone_high": high*TICK,
                              "zone_poc": poc*TICK, "zone_volume": int(vol[mask].sum()),
                              "cluster_peak_count": len(group), "prominence_threshold": cutoff})
    return zones


def _heavy_flags(trades: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not len(trades):
        return np.zeros(0, bool), np.zeros(0)
    ts = trades["timestamp_ns"].astype(np.int64)
    anchor = (int(ts[0]) // BAR_NS) * BAR_NS
    bars = ((ts-anchor)//BAR_NS).astype(np.int64)
    nbar = int(bars[-1])+1
    size = trades["execution_size"].astype(np.float64)
    count = np.bincount(bars, minlength=nbar).astype(float)
    total = np.bincount(bars, weights=size, minlength=nbar)
    sq = np.bincount(bars, weights=size*size, minlength=nbar)
    cc, cs, cq = (np.r_[0., np.cumsum(x)] for x in (count,total,sq))
    bar_threshold = np.full(nbar, np.nan)
    for b in range(1,nbar):
        lo=max(0,b-100); n=cc[b]-cc[lo]
        if n:
            mean=(cs[b]-cs[lo])/n
            var=max(0.,(cq[b]-cq[lo])/n-mean*mean)
            bar_threshold[b]=mean+2*math.sqrt(var)
    threshold=bar_threshold[bars]
    return np.isfinite(threshold)&(size>=threshold),threshold


def _first_in_tick_band(by_tick: Mapping[int,np.ndarray], start: int, lo: int, hi: int) -> int|None:
    found=None
    for t in range(lo,hi+1):
        ix=by_tick.get(t)
        if ix is None: continue
        p=int(np.searchsorted(ix,start))
        if p<len(ix) and (found is None or int(ix[p])<found): found=int(ix[p])
    return found


def detect(events: np.ndarray, day: str) -> tuple[list[dict[str,Any]],list[dict[str,Any]],list[dict[str,Any]]]:
    tr=_trades(events)
    if not len(tr): return [],[],[]
    ts=tr["timestamp_ns"].astype(np.int64); px=tr["execution_price"].astype(float)
    ticks=np.rint(px/TICK).astype(np.int64); sizes=tr["execution_size"].astype(np.int64)
    sides=tr["aggressor"].astype(np.int8)
    heavy,threshold=_heavy_flags(tr); hix=np.flatnonzero(heavy)
    by_tick={int(t):np.flatnonzero(ticks==t) for t in np.unique(ticks)}
    zones=_zones(tr,day); touches=[]; signals=[]
    for z in zones:
        z["first_touch_ns"]=None; z["retired_on_first_touch"]=False
        low=int(round(z["zone_low"]/TICK)); high=int(round(z["zone_high"]/TICK))
        active=int(np.searchsorted(ts,z["active_from_ns"]))
        ix=_first_in_tick_band(by_tick,active,low-1,high+1)
        if ix is None: continue
        z["first_touch_ns"]=int(ts[ix]); z["retired_on_first_touch"]=True
        prev=int(ticks[ix-1]) if ix else None
        if ix and _session_of(int(ts[ix-1]),_windows(day)) != int(z["session"]):
            prev=None
        direction=1 if prev is not None and prev>high+1 else -1 if prev is not None and prev<low-1 else 0
        if not direction: continue
        row={**z,"touch_index":ix,"T0_ns":int(ts[ix]),"T0_price":float(px[ix]),"direction":direction,
             "approach":"ABOVE" if direction>0 else "BELOW","heavy_print":False,"confirmation":False,
             "episode_status":"TOUCH_NO_HEAVY"}
        touches.append(row)
        wanted=-1 if direction>0 else 1
        end=int(np.searchsorted(ts,ts[ix]+CONFIRM_NS,side="right"))
        hp=hix[(hix>=ix)&(hix<end)]
        hp=[int(j) for j in hp if sides[j]==wanted and low-2<=ticks[j]<=high+2]
        if not hp: continue
        j=hp[0]; row.update({"heavy_print":True,"TH_ns":int(ts[j]),"TH_price":float(px[j]),
                             "heavy_size":int(sizes[j]),"heavy_threshold":float(threshold[j]),
                             "heavy_aggressor":"SELL" if wanted<0 else "BUY","episode_status":"HEAVY_AWAITING_RESPONSE"})
        evidence=int(ts[j]+RESPONSE_NS)
        ei=int(np.searchsorted(ts,evidence,side="left"))
        if ei>=len(ts) or ts[ei]>ts[j]+CONFIRM_NS:
            row["episode_status"]="EXPIRED_BEFORE_FAILURE_EVIDENCE"; continue
        poc=int(round(z["zone_poc"]/TICK)); target=poc+(1 if direction>0 else -1)
        deadline=int(np.searchsorted(ts,ts[j]+CONFIRM_NS,side="right"))
        search=np.arange(ei,min(deadline,len(ts)))
        cond=ticks[search]>=target if direction>0 else ticks[search]<=target
        candidates=search[cond]
        if not len(candidates):
            row["episode_status"]="EXPIRED_NO_RECLAIM"; continue
        ci=int(candidates[0])
        broken=(np.any(ticks[j:ci+1]<low-2) if direction>0 else np.any(ticks[j:ci+1]>high+2))
        if broken:
            row["episode_status"]="INVALIDATED_THROUGH_HVN"; continue
        segment=ticks[j:ci+1]
        reject=int(np.min(segment) if direction>0 else np.max(segment))
        stop=(min(low,reject)-STOP_BUFFER_TICKS if direction>0 else max(high,reject)+STOP_BUFFER_TICKS)
        row.update({"TF_ns":evidence,"TC_ns":int(ts[ci]),"TC_price":float(px[ci]),"rejection_extreme":reject*TICK,
                    "stop_price":stop*TICK,"episode_status":"CONFIRMED","confirmation":True})
        signals.append(row)
    # Overlapping historical zones cannot create duplicate entries at an identical decision time.
    by_key={}
    for row in signals:
        key=(row["TC_ns"],row["direction"])
        if key not in by_key or row["zone_volume"]>by_key[key]["zone_volume"]: by_key[key]=row
    return zones,touches,sorted(by_key.values(),key=lambda r:(r["TC_ns"],r["direction"],r["zone_id"]))


def _session_of(ts: int, windows: Mapping[int,tuple[int,int]]) -> int|None:
    return next((s for s,(lo,hi) in windows.items() if lo<=ts<hi),None)


def entry_for(events: np.ndarray, signal: Mapping[str,Any], windows: Mapping[int,tuple[int,int]]) -> dict[str,Any]|None:
    ts=events["timestamp_ns"].astype(np.int64); d=int(signal["direction"])
    ix=int(np.searchsorted(ts,int(signal["TC_ns"])+ENTRY_DELAY_NS))
    while ix<len(events):
        sess=_session_of(int(ts[ix]),windows)
        if sess is not None and int(events["session"][ix])==sess:
            bid,ask=float(events["bid"][ix]),float(events["ask"][ix])
            if math.isfinite(bid) and math.isfinite(ask) and ask>bid:
                price=ask+TICK if d>0 else bid-TICK
                stop=float(signal["stop_price"])
                if (d>0 and stop>=price) or (d<0 and stop<=price):
                    return None
                risk=abs(price-stop)/TICK
                if risk<=0 or not math.isfinite(risk): return None
                return {"entry_index":ix,"TE_ns":int(ts[ix]),"entry_bid":bid,"entry_ask":ask,"entry_price":price,
                        "entry_spread_ticks":(ask-bid)/TICK,"entry_slippage_ticks":1,"stop_price":signal["stop_price"],
                        "risk_ticks":risk,"risk_usd_per_contract":risk*TICK*POINT_VALUE_USD,"session":sess,
                        "confirmation_to_entry_ms":(int(ts[ix])-int(signal["TC_ns"]))/1e6}
        ix+=1
    return None


def _tick_round(price: float,direction: int)->float:
    x=price/TICK
    return (math.ceil(x-1e-12) if direction>0 else math.floor(x+1e-12))*TICK


def simulate(events: np.ndarray, entry: Mapping[str,Any], direction: int, rr: float,
             windows: Mapping[int,tuple[int,int]])->dict[str,Any]:
    d=int(direction); ei=int(entry["entry_index"]); sess=int(entry["session"])
    stop=float(entry["stop_price"]); risk=float(entry["risk_ticks"]); ep=float(entry["entry_price"])
    target=_tick_round(ep+d*rr*risk*TICK,d); effective=d*(target-ep)/(risk*TICK)
    ts=events["timestamp_ns"].astype(np.int64); end=windows[sess][1]
    last=max(ei,min(int(np.searchsorted(ts,end))-1,len(events)-1))
    outcome="SESSION_END"; fill=None; mfe=mae=0.
    for j in range(ei,last+1):
        if int(events["session"][j])!=sess: continue
        bid,ask=float(events["bid"][j]),float(events["ask"][j])
        if not(math.isfinite(bid) and math.isfinite(ask) and ask>bid): continue
        favorable=d*((bid if d>0 else ask)-ep)/TICK
        mfe=max(mfe,favorable); mae=max(mae,-favorable)
        # Executions are part of the source price path too. In particular, a
        # trade-through can fill a resting target while the executable BBO has
        # not yet advanced; include that observed trade price in excursions.
        trade_price=float(events["execution_price"][j]); trade_size=int(events["execution_size"][j])
        if trade_size>0 and math.isfinite(trade_price):
            trade_move=d*(trade_price-ep)/TICK
            mfe=max(mfe,trade_move); mae=max(mae,-trade_move)
        stop_hit=bid<=stop if d>0 else ask>=stop
        p=trade_price; q=trade_size
        through=q>0 and math.isfinite(p) and (p>=target+TICK if d>0 else p<=target-TICK)
        if stop_hit:
            outcome="STOP"; fill=min(bid,stop)-TICK if d>0 else max(ask,stop)+TICK; last=j; break
        if through:
            outcome="TARGET"; fill=target; last=j; break
    if fill is None:
        bid,ask=float(events["bid"][last]),float(events["ask"][last]); fill=bid if d>0 else ask
    gross=d*(fill-ep)/(risk*TICK); fee=COMMISSION_USD/(risk*TICK*POINT_VALUE_USD)
    return {"rr":rr,"target_price":target,"effective_rr":effective,"outcome":outcome,
            "exit_timestamp_ns":int(ts[last]),"holding_seconds":(int(ts[last])-int(entry["TE_ns"]))/1e9,
            "gross_r":gross,"net_r":gross-fee,"commission_r":fee,"mfe_ticks":mfe,"mae_ticks":mae,
            "max_favorable_excursion_r":mfe/risk,"max_adverse_excursion_r":mae/risk,
            "target_first":outcome=="TARGET","stop_first":outcome=="STOP","session_end_exit":outcome=="SESSION_END"}


def evaluate_date(day: str,events: np.ndarray,identity: Mapping[str,Any])->dict[str,Any]:
    zones,touches,signals=detect(events,day); windows=_windows(day); ledgers=[]; rr_rows=[]; marks=[]
    # Every qualifying print used by a touch episode is preserved with its
    # threshold and directional role; no non-source print aggregation occurs.
    trades=_trades(events); heavy_mask,heavy_threshold=_heavy_flags(trades)
    trade_times=trades["timestamp_ns"].astype(np.int64); trade_prices=trades["execution_price"].astype(float)
    associated_times={int(ep["TH_ns"]) for ep in touches if ep.get("heavy_print")}
    heavy_ledger=[]
    for j in np.flatnonzero(heavy_mask):
        side=int(trades["aggressor"][j])
        heavy_ledger.append({"date":day,"timestamp_ns":int(trades["timestamp_ns"][j]),
            "price":float(trades["execution_price"][j]),"size":int(trades["execution_size"][j]),
            "aggressor":"BUY" if side>0 else "SELL" if side<0 else "UNKNOWN",
            "threshold":float(heavy_threshold[j]),"associated_with_hvn_touch":int(trades["timestamp_ns"][j]) in associated_times})
    for n,s in enumerate(signals):
        ent=entry_for(events,s,windows)
        if ent is None:
            ledgers.append({**s,"date":day,"signal_id":f"{day}:{n}","entry_status":"NO_EXECUTABLE_ENTRY"}); continue
        base={**s,"date":day,"signal_id":f"{day}:{n}","entry_status":"EXECUTABLE",**ent,
              "T0_to_TH_ms":(s["TH_ns"]-s["T0_ns"])/1e6,
              "TH_to_failure_evidence_ms":(s["TF_ns"]-s["TH_ns"])/1e6,
              "TH_to_confirmation_ms":(s["TC_ns"]-s["TH_ns"])/1e6,
              "confirmation_to_entry_ms":ent["confirmation_to_entry_ms"],
              "TH_to_TC_directional_ticks":s["direction"]*(s["TC_price"]-s["TH_price"])/TICK,
              "TC_to_TE_directional_ticks":s["direction"]*(ent["entry_price"]-s["TC_price"])/TICK,
              "entry_hvn_boundary_distance_ticks":min(abs(ent["entry_price"]-s["zone_low"]),abs(ent["entry_price"]-s["zone_high"]))/TICK,
              "entry_rejection_extreme_distance_ticks":abs(ent["entry_price"]-s["rejection_extreme"])/TICK}
        def landmark(label: str, timestamp: int) -> dict[str,Any]:
            qix=int(np.searchsorted(events["timestamp_ns"],timestamp,side="left"))
            if qix>=len(events): return {f"{label}_{k}":None for k in ("quote_timestamp_ns","bid","ask","quote_age_ms")}
            qb,qa=float(events["bid"][qix]),float(events["ask"][qix])
            if not(math.isfinite(qb) and math.isfinite(qa) and qa>qb):
                return {f"{label}_{k}":None for k in ("quote_timestamp_ns","bid","ask","quote_age_ms")}
            qts=int(events["timestamp_ns"][qix])
            return {f"{label}_quote_timestamp_ns":qts,f"{label}_bid":qb,f"{label}_ask":qa,
                    f"{label}_quote_age_ms":(qts-timestamp)/1e6}
        for label,key in (("T0","T0_ns"),("TH","TH_ns"),("TF","TF_ns"),("TC","TC_ns")):
            base.update(landmark(label,int(s[key])))
        tf_ix=int(np.searchsorted(trade_times,int(s["TF_ns"]),side="left"))
        base["TF_first_trade_timestamp_ns"]=int(trade_times[tf_ix]) if tf_ix<len(trade_times) else None
        base["TF_first_trade_price"]=float(trade_prices[tf_ix]) if tf_ix<len(trade_times) else None
        prior_ix=int(np.searchsorted(trade_times,int(ent["TE_ns"])-30_000_000_000,side="left"))
        entry_ix=int(np.searchsorted(trade_times,int(ent["TE_ns"]),side="left"))
        local=trade_prices[prior_ix:entry_ix]
        base["nearby_prior_30s_range_ticks"]=float((np.max(local)-np.min(local))/TICK) if len(local)>1 else None
        base["stop_distance_to_nearby_30s_range_ratio"]=(float(ent["risk_ticks"])/base["nearby_prior_30s_range_ticks"] if base["nearby_prior_30s_range_ticks"] else None)
        # Retrospective-only diagnostic: estimate how much of the maximum
        # favorable reversal from TH had already occurred by confirmation.
        th=int(s["TH_ns"]); close=windows[int(ent["session"])][1]
        ta=int(np.searchsorted(events["timestamp_ns"],th,side="left"))
        tb=int(np.searchsorted(events["timestamp_ns"],close,side="left"))
        segment=events[ta:tb]
        mids=(segment["bid"].astype(float)+segment["ask"].astype(float))/2
        finite=np.isfinite(mids)
        denom=max(0.,float(np.max(int(s["direction"])*(mids[finite]-float(s["TH_price"])))) if np.any(finite) else 0.)
        tc_ix=int(np.searchsorted(events["timestamp_ns"],int(s["TC_ns"]),side="right"))
        pre=events[ta:min(tc_ix,tb)]
        pre_mid=(pre["bid"].astype(float)+pre["ask"].astype(float))/2
        pre_finite=np.isfinite(pre_mid)
        before=max(0.,float(np.max(int(s["direction"])*(pre_mid[pre_finite]-float(s["TH_price"])))) if np.any(pre_finite) else 0.)
        base["fraction_reversal_before_confirmation"]=min(1.,before/denom) if denom>0 else None
        ledgers.append(base)
        # Forward quote-mid markouts begin after the executable decision, not
        # at the earlier touch or heavy-print timestamp.
        for horizon in (5_000,30_000,120_000,300_000):
            mark_ix=int(np.searchsorted(events["timestamp_ns"],int(ent["TE_ns"])+horizon*1_000_000,side="left"))
            mid=None
            if mark_ix<len(events) and int(events["session"][mark_ix])==int(ent["session"]):
                bid=float(events["bid"][mark_ix]); ask=float(events["ask"][mark_ix])
                if math.isfinite(bid) and math.isfinite(ask) and ask>bid: mid=(bid+ask)/2
            marks.append({"date":day,"signal_id":base["signal_id"],"kind":"CONFIRMED_EXECUTABLE_ENTRY",
                "decision_timestamp_ns":int(ent["TE_ns"]),"horizon_ms":horizon,"direction":int(s["direction"]),
                "decision_mid":(ent["entry_bid"]+ent["entry_ask"])/2,
                "forward_mid":mid,"directional_markout_ticks":None if mid is None else int(s["direction"])*(mid-(ent["entry_bid"]+ent["entry_ask"])/2)/TICK})
        for rr in RR_GRID: rr_rows.append({**base,**simulate(events,ent,int(s["direction"]),rr,windows)})
    counts={"zones":len(zones),"first_touches":len(touches),"heavy_touch_episodes":sum(bool(x["heavy_print"]) for x in touches),
            "confirmed":len(signals),"entries":sum(x.get("entry_status")=="EXECUTABLE" for x in ledgers),
            "long":sum(x["direction"]>0 for x in signals),"short":sum(x["direction"]<0 for x in signals),
            "heavy_print_events":int(np.count_nonzero(heavy_mask)),
            "source_events":len(events),"source_trades":identity["trade_records"]}
    # Decision-time controls are descriptive and outcome-blind in construction:
    # touch-without-heavy at T0 and heavy-without-confirmation at TH.
    sig_keys={(x.get("zone_id"),x.get("T0_ns")) for x in signals}
    confirmed_touch={(x.get("zone_id"),x.get("T0_ns")) for x in signals}
    for ep in touches:
        if ep.get("heavy_print") and not ep.get("confirmation"):
            at=int(ep["TH_ns"]); kind="HVN_HEAVY_WITHOUT_CONFIRMATION"; direction=int(ep["direction"])
        elif not ep.get("heavy_print"):
            at=int(ep["T0_ns"]); kind="HVN_TOUCH_WITHOUT_HEAVY"; direction=int(ep["direction"])
        else:
            continue
        ix=int(np.searchsorted(events["timestamp_ns"],at+ENTRY_DELAY_NS,side="left"))
        if ix>=len(events): continue
        sess=_session_of(int(events["timestamp_ns"][ix]),windows)
        if sess is None or int(events["session"][ix])!=sess: continue
        b,a=float(events["bid"][ix]),float(events["ask"][ix])
        if not(math.isfinite(b) and math.isfinite(a) and a>b): continue
        anchor=(b+a)/2
        for horizon in (5_000,30_000,120_000,300_000):
            qi=int(np.searchsorted(events["timestamp_ns"],at+ENTRY_DELAY_NS+horizon*1_000_000,side="left"))
            mid=None
            if qi<len(events) and int(events["session"][qi])==sess:
                qb,qa=float(events["bid"][qi]),float(events["ask"][qi])
                if math.isfinite(qb) and math.isfinite(qa) and qa>qb: mid=(qb+qa)/2
            marks.append({"date":day,"signal_id":None,"kind":kind,"decision_timestamp_ns":at,
                "horizon_ms":horizon,"direction":direction,"decision_mid":anchor,"forward_mid":mid,
                "directional_markout_ticks":None if mid is None else direction*(mid-anchor)/TICK})
    return {"date":day,"source_identity":dict(identity),"zones":zones,"touches":touches,"signals":ledgers,
            "rr":rr_rows,"markouts":marks,"heavy_prints":heavy_ledger,"counts":counts}


def _pf(values: Sequence[float])->float|None:
    win=sum(x for x in values if x>0); loss=-sum(x for x in values if x<0)
    return win/loss if loss else (math.inf if win else None)


def _dd(values: Sequence[float])->float:
    eq=peak=draw=0.
    for x in values: eq+=x; peak=max(peak,eq); draw=max(draw,peak-eq)
    return draw


def summarize_rr(rr_rows: Sequence[Mapping[str,Any]],dates: Sequence[str])->tuple[list[dict[str,Any]],list[dict[str,Any]],dict[str,Any]]:
    summaries=[]; daily=[]; robustness={}
    for rr in RR_GRID:
        selected=[r for r in rr_rows if float(r["rr"])==rr]
        vals=[float(r["net_r"]) for r in selected]; gross=[float(r["gross_r"]) for r in selected]
        date_r=[]
        for day in dates:
            rows=[r for r in selected if r["date"]==day]; net=sum(float(r["net_r"]) for r in rows); date_r.append(net)
            daily.append({"date":day,"rr":rr,"trades":len(rows),"net_r":net,"gross_r":sum(float(r["gross_r"]) for r in rows),
                          "wins":sum(float(r["net_r"])>0 for r in rows),"losses":sum(float(r["net_r"])<0 for r in rows),
                          "target_first":sum(bool(r["target_first"]) for r in rows),"stop_first":sum(bool(r["stop_first"]) for r in rows)})
        positive_sum=sum(max(0,x) for x in date_r)
        summaries.append({"rr":rr,"trades":len(selected),"target_first":sum(bool(r["target_first"]) for r in selected),
            "stop_first":sum(bool(r["stop_first"]) for r in selected),"unresolved_session_end":sum(bool(r["session_end_exit"]) for r in selected),
            "win_rate":sum(x>0 for x in vals)/len(vals) if vals else None,"gross_r":sum(gross),"net_r":sum(vals),
            "avg_r_per_trade":statistics.mean(vals) if vals else None,"median_r":statistics.median(vals) if vals else None,
            "profit_factor":_pf(vals),"max_drawdown_r":_dd(vals),
            "avg_time_to_target_seconds":statistics.mean([r["holding_seconds"] for r in selected if r["target_first"]]) if any(r["target_first"] for r in selected) else None,
            "avg_time_to_stop_seconds":statistics.mean([r["holding_seconds"] for r in selected if r["stop_first"]]) if any(r["stop_first"] for r in selected) else None,
            "median_holding_seconds":statistics.median([r["holding_seconds"] for r in selected]) if selected else None,
            "max_mae_r":max((float(r["max_adverse_excursion_r"]) for r in selected),default=None),
            "max_mfe_r":max((float(r["max_favorable_excursion_r"]) for r in selected),default=None),
            "mfe_reached_target_count":sum(float(r["mfe_ticks"])>=float(rr)*float(r["risk_ticks"]) for r in selected),
            "mfe_reached_target_fraction":(sum(float(r["mfe_ticks"])>=float(rr)*float(r["risk_ticks"]) for r in selected)/len(selected) if selected else None),
            "target_first_rate":sum(bool(r["target_first"]) for r in selected)/len(selected) if selected else None,
            "positive_session_frequency":sum(x>0 for x in date_r)/len(dates) if dates else None,
            "active_dates":sum(any(r["date"]==d for r in selected) for d in dates),"date_count":len(dates),
            "net_r_ex_best_day":sum(date_r)-max(date_r,default=0),
            "net_r_ex_top_three_days":sum(date_r)-sum(sorted(date_r,reverse=True)[:3]),
            "largest_positive_day_share":max((max(0,x) for x in date_r),default=0)/positive_sum if positive_sum else None})
        bootstrap=None
        if len(date_r)>=2:
            rng=np.random.default_rng(20251025+int(round(rr*100)))
            sample=rng.choice(np.asarray(date_r,dtype=float),size=(5000,len(date_r)),replace=True).mean(axis=1)
            bootstrap={"method":"5000 trading-date bootstrap resamples; percentile CI for mean daily net R",
                "mean_daily_net_r":float(np.mean(date_r)),"ci95_low":float(np.quantile(sample,.025)),
                "ci95_high":float(np.quantile(sample,.975))}
        robustness[str(rr)]={"date_net_r":date_r,"mean_date_r":statistics.mean(date_r) if date_r else None,
            "positive_date_fraction":sum(x>0 for x in date_r)/len(date_r) if date_r else None,
            "leave_one_best_day_net_r":summaries[-1]["net_r_ex_best_day"],
            "leave_three_best_days_net_r":summaries[-1]["net_r_ex_top_three_days"],
            "largest_positive_day_share":summaries[-1]["largest_positive_day_share"],
            "date_clustered_uncertainty":bootstrap}
    return summaries,daily,robustness


def _aggregate(mode: str,dates: Sequence[str],results: Mapping[str,Any],out: Path,
               identities: Sequence[Mapping[str,Any]],started: float)->dict[str,Any]:
    zones=[]; touches=[]; signals=[]; rr=[]; daily=[]
    for day in dates:
        r=results[day]; zones+=r["zones"]; touches+=r["touches"]; signals+=r["signals"]; rr+=r["rr"]; daily.append({"date":day,**r["counts"]})
    _atomic_csv(out/"hvn-zone-ledger.csv",zones); _atomic_csv(out/"hvn-touch-episodes.csv",touches)
    _atomic_csv(out/"confirmed-signal-ledger.csv",signals); _atomic_csv(out/"rr-grid-trade-results.csv",rr)
    _atomic_csv(out/"spring-train-date-results.csv",daily)
    _atomic_csv(out/"entry-timing-analysis.csv",[r for r in signals if r.get("entry_status")=="EXECUTABLE"])
    _atomic_csv(out/"structural-stop-analysis.csv",[{k:r.get(k) for k in ("date","signal_id","direction","zone_low","zone_high","rejection_extreme","entry_price","stop_price","risk_ticks","risk_usd_per_contract","entry_spread_ticks")} for r in signals if r.get("entry_status")=="EXECUTABLE"])
    _atomic_csv(out/"post-entry-excursion-analysis.csv",[{k:r.get(k) for k in ("date","signal_id","rr","outcome","mfe_ticks","mae_ticks","max_favorable_excursion_r","max_adverse_excursion_r")} for r in rr])
    marks=[x for day in dates for x in results[day].get("markouts",[])]
    _atomic_csv(out/"forward-markouts.csv",marks)
    heavy_prints=[x for day in dates for x in results[day].get("heavy_prints",[])]
    _atomic_csv(out/"heavy-print-ledger.csv",heavy_prints)
    _write_gz_jsonl(out/"failed-aggression-state-traces.jsonl.gz",
                    [{"date":d,"zones":results[d]["zones"],"touches":results[d]["touches"],"signals":results[d]["signals"]} for d in dates])
    rr_summary,daily_rr,robust=summarize_rr(rr,dates)
    _atomic_csv(out/"rr-grid-daily-results.csv",daily_rr)
    _atomic_json(out/"rr-grid-family-summary.json",{"dates":list(dates),"rr_grid":rr_summary})
    _atomic_json(out/"rr-grid-robustness.json",robust)
    # Predictive evidence is descriptive: all groups share horizon units and
    # are explicitly keyed by their actual decision timestamp.
    signal_quality={}
    for kind in ("CONFIRMED_EXECUTABLE_ENTRY","HVN_TOUCH_WITHOUT_HEAVY","HVN_HEAVY_WITHOUT_CONFIRMATION"):
        group=[x for x in marks if x["kind"]==kind]
        signal_quality[kind]={str(h):{
            "n":sum(x["horizon_ms"]==h and x["directional_markout_ticks"] is not None for x in group),
            "mean_directional_markout_ticks":(statistics.mean(x["directional_markout_ticks"] for x in group if x["horizon_ms"]==h and x["directional_markout_ticks"] is not None)
                if any(x["horizon_ms"]==h and x["directional_markout_ticks"] is not None for x in group) else None),
            "positive_fraction":(sum(x["directional_markout_ticks"]>0 for x in group if x["horizon_ms"]==h and x["directional_markout_ticks"] is not None)/
                sum(x["horizon_ms"]==h and x["directional_markout_ticks"] is not None for x in group)
                if any(x["horizon_ms"]==h and x["directional_markout_ticks"] is not None for x in group) else None)}
            for h in (5_000,30_000,120_000,300_000)}
    _atomic_json(out/"signal-quality-analysis.json",{
        "decision_time_matched_horizons_ms":[5000,30000,120000,300000],"groups":signal_quality,
        "incremental_hvn_heavy_absorption":"INCONCLUSIVE: observational groups are not risk-set matched and differ in decision time",
        "independent_edge_validation":"NOT_ESTABLISHED"})
    _atomic_json(out/"execution-cost-analysis.json",{
        "commission_usd_round_trip":COMMISSION_USD,"entry_market_slippage_ticks":ENTRY_SLIPPAGE_TICKS,
        "stop_market_slippage_ticks":STOP_SLIPPAGE_TICKS,"spread_paid_through_executable_side":True,
        "target_limit_fill_requires_trade_through_ticks":1,
        "median_entry_spread_ticks":statistics.median([r["entry_spread_ticks"] for r in signals if r.get("entry_status")=="EXECUTABLE"]) if any(r.get("entry_status")=="EXECUTABLE" for r in signals) else None,
        "target_queue_position":"unobservable; conservative trade-through proxy used",
        "same_entry_population_all_rr":True})
    entries=[r for r in signals if r.get("entry_status")=="EXECUTABLE"]
    census={k:sum(int(results[d]["counts"].get(k,0)) for d in dates) for k in results[dates[0]]["counts"]} if dates else {}
    heavy_delay=[r["TH_to_confirmation_ms"] for r in entries]
    entry_delay=[r["confirmation_to_entry_ms"] for r in entries]
    risks=[r["risk_ticks"] for r in entries]
    markout_5s=[x["directional_markout_ticks"] for x in marks if x["kind"]=="CONFIRMED_EXECUTABLE_ENTRY" and x["horizon_ms"]==5000 and x["directional_markout_ticks"] is not None]
    rr_counts={str(x):sum(float(r["rr"])==x for r in rr) for x in RR_GRID}
    reference_mfe=[r["mfe_ticks"] for r in rr if float(r["rr"])==3.0]
    reference_mae=[r["mae_ticks"] for r in rr if float(r["rr"])==3.0]
    cost_share=[]
    for r in entries:
        available=max(0.0,float(next((x["mfe_ticks"] for x in rr if x["signal_id"]==r["signal_id"] and float(x["rr"])==3.0),0.0)))
        modeled_cost=float(r["entry_spread_ticks"])+ENTRY_SLIPPAGE_TICKS+COMMISSION_USD/(TICK*POINT_VALUE_USD)
        if available>0: cost_share.append(modeled_cost/available)
    summary={"study_id":RUN_ID,"spec_sha256":SPEC_SHA256,"mode":mode,"dates":list(dates),
        "pilot_dates_completed":list(dates[:5]) if mode=="pilot" else list(baseline.TRAIN_DATES[:5]),
        "full_spring_dates_completed":list(dates) if mode=="full" else [],
        "full_spring_complete":mode=="full" and len(dates)==35,"census_totals":census,"rr_summary":rr_summary,
        "median_heavy_to_confirmation_ms":statistics.median(heavy_delay) if heavy_delay else None,
        "median_confirmation_to_entry_ms":statistics.median(entry_delay) if entry_delay else None,
        "median_entry_to_stop_ticks":statistics.median(risks) if risks else None,
        # Use the longest predeclared target as a consistent reference path;
        # pooling seven counterfactual exits would overweight each signal and
        # mix different path truncation points.
        "post_entry_excursion_reference_rr":3.0,
        "median_post_entry_mfe_ticks":statistics.median(reference_mfe) if reference_mfe else None,
        "median_post_entry_mae_ticks":statistics.median(reference_mae) if reference_mae else None,
        # Keep path excursion reach separate from executable target fills. MFE
        # can touch a target without satisfying the conservative trade-through
        # fill rule, so those rates must not be conflated in the summary.
        "post_entry_reachability":{str(x):next((z["mfe_reached_target_fraction"] for z in rr_summary if float(z["rr"])==x),None) for x in (1.0,1.5,2.0,3.0)},
        "post_entry_executable_target_fill_rate":{str(x):next((z["target_first_rate"] for z in rr_summary if float(z["rr"])==x),None) for x in (1.0,1.5,2.0,3.0)},
        "fraction_reversal_before_confirmation":statistics.median([r["fraction_reversal_before_confirmation"] for r in entries if r.get("fraction_reversal_before_confirmation") is not None]) if any(r.get("fraction_reversal_before_confirmation") is not None for r in entries) else None,
        "fraction_without_5s_directional_continuation":sum(x<=0 for x in markout_5s)/len(markout_5s) if markout_5s else None,
        "fraction_cost_exceeds_25pct_of_observed_post_entry_mfe":sum(x>=.25 for x in cost_share)/len(cost_share) if cost_share else None,
        "fraction_cost_exceeds_50pct_of_observed_post_entry_mfe":sum(x>=.50 for x in cost_share)/len(cost_share) if cost_share else None,
        "rr_trade_counts":rr_counts,
        "rr_grid_same_entry_population":len({r["signal_id"] for r in rr if float(r["rr"])==RR_GRID[0]})==len({r["signal_id"] for r in rr if float(r["rr"])==RR_GRID[-1]}),
        "rr_grid_fully_evaluated":all(rr_counts[str(x)]==sum(int(results[d]["counts"]["entries"]) for d in dates) for x in RR_GRID),
        "source_identities":list(identities),"runtime_seconds":time.perf_counter()-started,
        "october_accessed":False,"protected_oos_accessed":False,"data_downloaded":False}
    _atomic_json(out/"summary.json",summary)
    if summary["full_spring_complete"]:
        lines=["# ES HVN + Heavy Absorption Strategy Research V1","","Status: FULL SPRING TRAIN; no parameter optimization.","",
               f"Frozen specification SHA-256: {SPEC_SHA256}","","## Funnel and timing","",
               f"Census: {json.dumps(census,sort_keys=True)}",
               f"Median heavy-to-confirmation (ms): {summary['median_heavy_to_confirmation_ms']}",
               f"Median confirmation-to-entry (ms): {summary['median_confirmation_to_entry_ms']}",
               f"Median entry-to-stop (ticks): {summary['median_entry_to_stop_ticks']}","",
               "## Frozen RR grid","","| RR | Trades | Win rate | Gross R | Net R | Avg R/trade | PF | Max DD | Target-first rate |",
               "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for x in rr_summary:
            lines.append("| "+" | ".join(str(x[k]) for k in ("rr","trades","win_rate","gross_r","net_r","avg_r_per_trade","profit_factor","max_drawdown_r","target_first_rate"))+" |")
        lines+=["","## Date-level census","","| Date | Zones | Touches | Heavy-touch | Confirmed | Entries |","|---|---:|---:|---:|---:|---:|"]
        for x in daily: lines.append(f"| {x['date']} | {x['zones']} | {x['first_touches']} | {x['heavy_touch_episodes']} | {x['confirmed']} | {x['entries']} |")
        lines+=["","All results are exploratory Spring TRAIN evidence. No RR winner is selected. Predictive value and executable net tradability are separate questions.",""]
        (out/"report.md").write_text("\n".join(lines))
    doc="# Indicator documentation and assumptions\n\nThe frozen specification records all numerical choices. HVNs use completed 30-minute executed-volume profiles, fixed 5-tick smoothing and a profile-local 90th-percentile peak screen. A zone is not available before profile close and retires at its first revisit. Historical volume is not interpreted as current resting liquidity.\n\nHeavy print means one native T record meeting the threshold from the prior 100 completed 1-minute bars. No parent-order aggregation or undocumented floor is applied. These are research assumptions, not exact ATAS reproductions.\n"
    (out/"indicator-documentation-and-assumptions.md").write_text(doc)
    sem={"dataset":"GLBX.MDP3","schema":"mbp-10","T_action":"execution record","aggressor":"T side B=BUY, A=SELL per repository native adapter","print_grouping":SPEC["heavy_print"]["print_unit"],"event_timestamp":"sealed canonical tape retains ts_recv; ts_event and sequence not present","BBO":"executable best bid/ask in source order","top10":"present in raw DBN, not retained in compact tape; not needed by this detector"}
    _atomic_json(out/"source-and-print-semantics.json",sem)
    _atomic_json(out/"source-integrity-results.json",{
        "status":"PASS" if len(dates)==35 else "PILOT_PASS","date_count":len(dates),
        "sources":[{"date":x["date"],"raw_path":x["raw_path"],"raw_sha256":x["raw_sha256"],
            "raw_bytes":x["raw_bytes"],"dataset":x["dataset"],"schema":x["schema"],"symbol":x["symbol"],
            "tape_path":x["tape_path"],"tape_sha256":x["tape_sha256"],"event_count":x["event_count"],
            "trade_records":x["trade_records"],"unknown_side_trades":x["unknown_side_trades"],
            "ts_event_sequence_retained":x["ts_event_and_sequence_retained"]} for x in identities],
        "method":"verified acquisition manifest status, raw bytes/SHA, sealed tape date/source SHA/event ordering/executable BBO"})
    total_touches=sum(int(results[d]["counts"]["first_touches"]) for d in dates)
    total_heavy=sum(int(results[d]["counts"]["heavy_touch_episodes"]) for d in dates)
    total_confirmed=sum(int(results[d]["counts"]["confirmed"]) for d in dates)
    total_entries=sum(int(results[d]["counts"]["entries"]) for d in dates)
    _atomic_json(out/"indicator-contribution-analysis.json",{
        "funnel":{"hvn_first_touches":total_touches,"heavy_print_touch_episodes":total_heavy,
            "failed_aggression_confirmations":total_confirmed,"executable_entries":total_entries},
        "rates":{"heavy_given_touch":total_heavy/total_touches if total_touches else None,
            "confirmation_given_heavy_touch":total_confirmed/total_heavy if total_heavy else None,
            "entry_given_confirmation":total_entries/total_confirmed if total_confirmed else None},
        "incremental_hvn_heavy_absorption":"INCONCLUSIVE without risk-set-matched comparisons"})
    (out/"research-limitations.md").write_text("# Research limitations\n\nThe ATAS proprietary smoothing/prominence/cluster details remain unverified. T records may represent fragments of larger parent orders; this study does not infer parent grouping. The reused sealed tape omits raw ts_event, sequence, and unchanged top-ten states, so exact event-vs-receive-time sequencing cannot be independently audited. Target-limit queue position is unknown; require one-tick trade-through. Outcomes are TRAIN discovery only; independent edge validation is not established.\n")
    (out/"next-research-recommendation.md").write_text("# Next research recommendation\n\nDo not select the highest in-sample RR as a final configuration. Review timing, cost, date concentration, and signal-quality controls. If source order/print grouping is material, prepare a compact source tape retaining ts_event, ts_recv, sequence, raw action/side and top-ten state before a V2 model. Do not tune V1 thresholds to outcomes.\n")
    return summary


def run(mode: str="pilot",out_root: Path=OUT_ROOT)->dict[str,Any]:
    dates,sources=_sources(); target=dates[:5] if mode=="pilot" else dates
    out_root.mkdir(parents=True,exist_ok=True)
    spec_file=out_root/"frozen-v1-specification.json"
    if spec_file.exists() and json.loads(spec_file.read_text()).get("sha256")!=SPEC_SHA256:
        raise ResearchError("frozen study specification changed")
    _atomic_json(spec_file,{"specification":SPEC,"sha256":SPEC_SHA256})
    (out_root/"frozen-v1-specification.sha256").write_text(SPEC_SHA256+"  frozen-v1-specification.json\n")
    started=time.perf_counter(); results={}; identities=[]
    cp_root=out_root/"checkpoints"; cp_root.mkdir(exist_ok=True)
    for n,day in enumerate(target,1):
        events,ident=load_events(day,sources[day]); identities.append(ident)
        cp=cp_root/f"{day}.json.gz"; result=None
        if cp.exists():
            try:
                with gzip.open(cp,"rt") as f: cached=json.load(f)
                if checkpoint_matches(cached,ident):
                    result=cached["result"]
                    print(f"RESUME {day} checkpoint valid",flush=True)
            except (OSError,json.JSONDecodeError,KeyError): result=None
        if result is None:
            result=evaluate_date(day,events,ident)
            tmp=cp.with_name(f".{cp.name}.{os.getpid()}.tmp")
            with tmp.open("wb") as raw:
                with gzip.GzipFile(fileobj=raw,mode="wb",mtime=0) as gz:
                    gz.write(json.dumps({"version":CHECKPOINT_VERSION,"spec_sha256":SPEC_SHA256,
                        "source_sha256":ident["raw_sha256"],"tape_sha256":ident["tape_sha256"],
                        "result":result},sort_keys=True,default=_json_default,allow_nan=False).encode())
                raw.flush(); os.fsync(raw.fileno())
            os.replace(tmp,cp)
        results[day]=result
        print(f"DATE_PROGRESS {n}/{len(target)} {day} zones={len(result['zones'])} touches={len(result['touches'])} heavy={result['counts']['heavy_touch_episodes']} confirmed={result['counts']['confirmed']} entries={result['counts']['entries']} elapsed_s={time.perf_counter()-started:.1f}",flush=True)
    summary=_aggregate(mode,target,results,out_root,identities,started)
    hashes={str(p.relative_to(out_root)):_sha(p) for p in sorted(out_root.rglob("*")) if p.is_file() and p.name not in {"artifact-hashes.json","run-manifest.json"}}
    _atomic_json(out_root/"artifact-hashes.json",hashes)
    _atomic_json(out_root/"run-manifest.json",{"study_id":RUN_ID,"spec_sha256":SPEC_SHA256,"mode":mode,
        "completed_dates":list(target),"complete":summary["full_spring_complete"],"checkpoint_version":CHECKPOINT_VERSION,
        "artifact_hash_count":len(hashes)})
    return summary


def main(argv: Sequence[str]|None=None)->int:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("mode",choices=("pilot","full"))
    p.add_argument("--out-root",type=Path,default=OUT_ROOT); args=p.parse_args(argv)
    try: result=run(args.mode,args.out_root)
    except ResearchError as exc: print(f"RESEARCH_ERROR: {exc}"); return 2
    print(json.dumps({"mode":result["mode"],"dates":result["dates"],"full_spring_complete":result["full_spring_complete"],"census_totals":result["census_totals"],"rr_summary":result["rr_summary"],"runtime_seconds":result["runtime_seconds"]},indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
