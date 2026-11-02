"""Native MBP-10 replay and frozen four-mechanism detector pilot.

The module consumes one authorized DBN file in native order. It never sorts by
exchange sequence and uses receive time as the information clock.
"""
from __future__ import annotations

from collections import deque
from array import array
from bisect import insort
from functools import lru_cache
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable
from collections.abc import Mapping

import numpy as np

from . import es_four_mechanism_comparative_research_v1 as audit
from . import mac_2025_es_only_train_baseline as baseline

NS = 1_000_000_000
TICK_RAW = 250_000_000
BAD_RECV = 8
MAYBE_BAD_BOOK = 4
SNAPSHOT = 32


@dataclass(frozen=True)
class BookState:
    ordinal: int
    ts_recv: int
    ts_event: int
    bids: tuple[tuple[int, int], ...]
    asks: tuple[tuple[int, int], ...]
    valid: bool
    snapshot: bool
    gap: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BookState":
        v = dict(value)
        v["bids"] = tuple(tuple(map(int, x)) for x in v["bids"])
        v["asks"] = tuple(tuple(map(int, x)) for x in v["asks"])
        return cls(**v)


class NativeBookReplay:
    """MBP-10 full-state replay with deterministic state checkpoints."""

    def __init__(self, state: BookState | None = None, next_ordinal: int = 0):
        self.state = state
        self.next_ordinal = int(next_ordinal)
        self.validity = {"rows": 0, "valid": 0, "invalid": 0, "crossed_locked": 0,
                         "bad_recv_flag": 0, "maybe_bad_book_flag": 0, "snapshots": 0,
                         "gaps": 0, "depth_only_updates": 0, "depth_updates": 0}

    def process(self, row: Any) -> tuple[BookState, bool, bool]:
        ordinal = self.next_ordinal
        self.next_ordinal += 1
        bids = tuple((int(row[f"bid_px_{i:02d}"]), int(row[f"bid_sz_{i:02d}"])) for i in range(10))
        asks = tuple((int(row[f"ask_px_{i:02d}"]), int(row[f"ask_sz_{i:02d}"])) for i in range(10))
        flags = int(row["flags"])
        snap = bool(flags & SNAPSHOT)
        elapsed_gap=self.state is not None and int(row["ts_recv"])-self.state.ts_recv>2*NS
        gap = bool(flags & (BAD_RECV | MAYBE_BAD_BOOK)) or elapsed_gap
        good_bid = bids[0][0] > 0 and bids[0][1] > 0
        good_ask = asks[0][0] > 0 and asks[0][1] > 0
        ordered_bid = all(not bids[i][0] or not bids[i + 1][0] or bids[i][0] > bids[i + 1][0] for i in range(9))
        ordered_ask = all(not asks[i][0] or not asks[i + 1][0] or asks[i][0] < asks[i + 1][0] for i in range(9))
        crossed = good_bid and good_ask and bids[0][0] >= asks[0][0]
        valid = good_bid and good_ask and ordered_bid and ordered_ask and not crossed and not gap
        old = self.state
        changed = old is None or old.bids != bids or old.asks != asks or snap
        same_bbo = old is not None and old.bids[0] == bids[0] and old.asks[0] == asks[0]
        self.validity["rows"] += 1
        self.validity["valid"] += int(valid)
        self.validity["invalid"] += int(not valid)
        self.validity["crossed_locked"] += int(crossed)
        self.validity["bad_recv_flag"] += int(bool(flags & BAD_RECV))
        self.validity["maybe_bad_book_flag"] += int(bool(flags & MAYBE_BAD_BOOK))
        self.validity["snapshots"] += int(snap)
        self.validity["gaps"] += int(gap)
        self.validity["depth_updates"] += int(changed)
        self.validity["depth_only_updates"] += int(changed and same_bbo)
        state = BookState(ordinal, int(row["ts_recv"]), int(row["ts_event"]), bids, asks, bool(valid), snap, gap)
        # MBP-10 is an absolute aggregate state per row. A snapshot resets the
        # prior state; non-snapshot rows replace it with the complete state too.
        self.state = state
        return state, bool(changed), bool(changed and same_bbo)

    def checkpoint(self) -> dict[str, Any]:
        return {"version": 1, "next_ordinal": self.next_ordinal,
                "state": None if self.state is None else self.state.as_dict(),
                "validity": dict(self.validity)}

    @classmethod
    def resume(cls, checkpoint: dict[str, Any]) -> "NativeBookReplay":
        if checkpoint.get("version") != 1:
            raise ValueError("unsupported checkpoint version")
        state = checkpoint.get("state")
        obj = cls(None if state is None else BookState.from_dict(state), checkpoint["next_ordinal"])
        obj.validity.update(checkpoint.get("validity", {}))
        return obj


def _session_code(day: str, t: int) -> int:
    for i, name in enumerate(("ASIA", "EUROPE", "NY")):
        start, end = _windows(day)[name]
        if start <= t < end:
            return i
    return -1


@lru_cache(maxsize=64)
def _windows(day:str):
    return baseline._session_windows(day)


def _event_id(date: str, mechanism: str, ts: int, ordinal: int) -> str:
    return f"{date}:{mechanism}:{ts}:{ordinal}"


def _empty_event(mechanism: str, date: str, ts: int, ordinal: int, direction: int,
                 features: dict[str, Any], reason: str = "QUALIFIED") -> dict[str, Any]:
    return {"date": date, "mechanism": mechanism, "event_id": _event_id(date, mechanism, ts, ordinal),
            "ts_recv": ts, "source_ordinal": ordinal, "direction": direction,
            "direction_reason": "SIGNED_FEATURE" if direction else reason,
            "features": json.dumps(features, sort_keys=True, separators=(",", ":")),
            "source_valid": True, "expired": False}


def _quantile(history: deque[float], q: float) -> float | None:
    return None if not history else float(np.quantile(np.fromiter(history, dtype=np.float64), q, method="linear"))


class OnlineEmpiricalQuantile:
    """Exact linear empirical quantiles with persistent cumulative counts."""
    def __init__(self):
        self.values: list[float]=[]
        self.counts: dict[float,int]={}
        self.n=0
    def add(self,value:float):
        value=float(value)
        if value not in self.counts:
            insort(self.values,value)
            self.counts[value]=0
        self.counts[value]+=1
        self.n+=1
    def _order(self,k:int)->float:
        cumulative=0
        for value in self.values:
            cumulative+=self.counts[value]
            if cumulative>k:return value
        raise IndexError(k)
    def quantile(self,q:float)->float|None:
        if not self.n:return None
        rank=q*(self.n-1); lower=int(rank); upper=min(self.n-1,lower+1)
        a=self._order(lower); b=self._order(upper)
        return float(a+(rank-lower)*(b-a))


def _pack_checkpoint_value(value: Any) -> Any:
    """JSON-safe, type-preserving encoding for detector checkpoint state."""
    if isinstance(value, OnlineEmpiricalQuantile):
        return {"__type__":"online_quantile", "values":_pack_checkpoint_value(value.values),
                "counts":_pack_checkpoint_value(value.counts), "n":value.n}
    if isinstance(value, array):
        return {"__type__":"array", "typecode":value.typecode, "values":value.tolist()}
    if isinstance(value, deque):
        return {"__type__":"deque", "maxlen":value.maxlen,
                "values":[_pack_checkpoint_value(v) for v in value]}
    if isinstance(value, tuple):
        return {"__type__":"tuple", "values":[_pack_checkpoint_value(v) for v in value]}
    if isinstance(value, Mapping):
        return {"__type__":"mapping", "items":[[_pack_checkpoint_value(k),_pack_checkpoint_value(v)] for k,v in value.items()]}
    if isinstance(value, list):
        return [_pack_checkpoint_value(v) for v in value]
    if value is None or isinstance(value,(str,int,float,bool)):
        return value
    raise TypeError(f"unsupported detector checkpoint value: {type(value).__name__}")


def _unpack_checkpoint_value(value: Any) -> Any:
    if isinstance(value,list):
        return [_unpack_checkpoint_value(v) for v in value]
    if not isinstance(value,dict) or "__type__" not in value:
        return value
    kind=value["__type__"]
    if kind=="online_quantile":
        obj=OnlineEmpiricalQuantile()
        obj.values=_unpack_checkpoint_value(value["values"])
        obj.counts=_unpack_checkpoint_value(value["counts"])
        obj.n=int(value["n"])
        return obj
    if kind=="array":
        return array(value["typecode"],value["values"])
    if kind=="deque":
        return deque((_unpack_checkpoint_value(v) for v in value["values"]),maxlen=value["maxlen"])
    if kind=="tuple":
        return tuple(_unpack_checkpoint_value(v) for v in value["values"])
    if kind=="mapping":
        return {_unpack_checkpoint_value(k):_unpack_checkpoint_value(v) for k,v in value["items"]}
    raise ValueError(f"unknown detector checkpoint type: {kind}")


class DetectorSuite:
    """Deterministic online detector state machines matching the frozen appendix."""

    def __init__(self, day: str):
        self.day = day
        self.events: dict[str, list[dict[str, Any]]] = {f"M{i}": [] for i in range(1, 5)}
        self.transitions = {f"M{i}": {} for i in range(1, 5)}
        self.validity = {"rows_seen": 0, "valid_book_rows": 0, "invalid_book_rows": 0,
                         "unknown_side_trades": 0, "known_side_trades": 0}
        self.last_session = -1
        self.session_start: dict[int,int] = {}
        self.last_mid_by_session: dict[int, tuple[int, float]] = {}
        self.quote_rows = {"t": array("q"), "ordinal": array("q"), "bid": array("q"), "ask": array("q")}
        self.last_quote_state: tuple[int,int] | None = None
        self.m1_bins: dict[tuple[int, int], float] = {}
        self.m1_bin_ord: dict[tuple[int, int], int] = {}
        self.m1_next_bin: dict[int,int] = {}
        self.m1_hist: dict[int, OnlineEmpiricalQuantile] = {i: OnlineEmpiricalQuantile() for i in range(3)}
        self.m1_last_trigger: dict[int, int] = {}
        self.m2_samples: dict[int, deque[tuple[int, tuple[int,int]]]] = {i: deque(maxlen=100) for i in range(3)}
        self.m2_medians: dict[int, tuple[float,float] | None] = {i:None for i in range(3)}
        self.last_book_sample_by_session: dict[int,tuple[int,int,int]] = {}
        self.m2_candidate: dict[int, tuple[int, int, float, int, int] | None] = {i: None for i in range(3)}
        self.m2_last_emit: dict[int, int] = {}
        self.trade_window: deque[tuple[int, int]] = deque()
        self.m3_highs: deque[tuple[int,int]] = deque()
        self.m3_lows: deque[tuple[int,int]] = deque()
        self.m3_candidate: tuple[int, int, int, int, int] | None = None
        self.m3_seen_range: tuple[int, int] | None = None
        self.range_5m: dict[tuple[int, int], tuple[float, float]] = {}
        self.range5_history: dict[int, deque[float]] = {i: deque(maxlen=500) for i in range(3)}
        self.range5s_history: dict[int, deque[float]] = {i: deque(maxlen=360) for i in range(3)}
        self.m4_baseline5: dict[int,float|None] = {i:None for i in range(3)}
        self.m4_compressed: dict[int, bool] = {i: False for i in range(3)}
        self.m4_last_emit: dict[int, int] = {}
        self.mid_samples: dict[int, deque[tuple[int, float]]] = {i: deque(maxlen=110) for i in range(3)}
        self.mid_lookup: dict[int,dict[int,float]] = {i:{} for i in range(3)}
        self.mid_grid_origin: dict[int,int] = {}
        self.next_sample: dict[int, int] = {}
        self.next_mid_sample: dict[int, int] = {}
        self.last_m4_5s_bin: dict[int, int] = {}
        self.current5s: dict[int,tuple[int,float,float]|None] = {i:None for i in range(3)}
        self.mid_min: dict[int,deque[tuple[int,float]]] = {i:deque() for i in range(3)}
        self.mid_max: dict[int,deque[tuple[int,float]]] = {i:deque() for i in range(3)}

    def checkpoint(self) -> dict[str, Any]:
        return {"version":1,"day":self.day,
                "state":{key:_pack_checkpoint_value(value) for key,value in self.__dict__.items() if key!="day"}}

    @classmethod
    def resume(cls, checkpoint: dict[str, Any]) -> "DetectorSuite":
        if checkpoint.get("version")!=1:
            raise ValueError("unsupported detector checkpoint version")
        obj=cls.__new__(cls)
        obj.day=str(checkpoint["day"])
        for key,value in checkpoint["state"].items():
            setattr(obj,key,_unpack_checkpoint_value(value))
        return obj

    def _transition(self, mech: str, name: str):
        self.transitions[mech][name] = self.transitions[mech].get(name, 0) + 1

    def _close_m1_bins(self, session:int, end_ts:int):
        end_bin=end_ts//500_000_000
        next_bin=self.m1_next_bin.setdefault(session,self.session_start.get(session,end_ts)//500_000_000)
        while next_bin<end_bin:
            b=next_bin; key=(session,b); delta=self.m1_bins.pop(key,0.0)
            hist=self.m1_hist[session]
            if hist.n>=20 and delta:
                threshold=hist.quantile(.90); stamp=(b+1)*500_000_000
                if threshold is not None and abs(delta)>=threshold and stamp-self.m1_last_trigger.get(session,-10**30)>=2_000_000_000:
                    last_ordinal=self.m1_bin_ord.get(key)
                    if last_ordinal is None:
                        # Empty bins cannot qualify because delta is zero.
                        last_ordinal=0
                    self.events["M1"].append(_empty_event("M1",self.day,stamp,last_ordinal,int(np.sign(delta)),
                        {"signed_volume":delta,"prior_bin_abs_delta_q90":threshold,"session":session,
                         "completed_bin_start_ts_recv":b*500_000_000,
                         "completed_bin_end_ts_recv":stamp,
                         "completed_bin_last_record_ordinal":last_ordinal}))
                    self.m1_last_trigger[session]=stamp
                    self._transition("M1","PRESSURE_QUALIFIED")
            hist.add(abs(delta))
            self.m1_bin_ord.pop(key,None)
            next_bin+=1
        self.m1_next_bin[session]=next_bin

    def on_row(self, state: BookState, action: str, price: int, size: int, side: str):
        self.validity["rows_seen"] += 1
        t, ordinal = state.ts_recv, state.ordinal
        session = _session_code(self.day, t)
        valid = state.valid and session >= 0
        self.validity["valid_book_rows" if valid else "invalid_book_rows"] += 1
        if session != self.last_session:
            old_session=self.last_session
            if old_session>=0:
                old_name=("ASIA","EUROPE","NY")[old_session]
                self._close_m1_bins(old_session,min(t,_windows(self.day)[old_name][1]))
            self.last_session = session
            # Candidate state does not cross session boundaries.
            self.m2_candidate[session] = None if session >= 0 else None
            self.m3_candidate = None
            self.trade_window.clear(); self.m3_highs.clear(); self.m3_lows.clear()
            if session >= 0:
                self.next_sample[session] = t
                self.next_mid_sample[session] = t
                self.mid_grid_origin[session]=t
                self.session_start[session]=_windows(self.day)[("ASIA","EUROPE","NY")[session]][0]
        if session>=0:
            bin_id=t//500_000_000
            self.m1_next_bin.setdefault(session,self.session_start[session]//500_000_000)
            # Every native source row is passed in the exact construction run;
            # this records the frozen appendix's last-record ordinal per bin.
            self.m1_bin_ord[(session,bin_id)]=ordinal
        if not valid:
            if session>=0:
                # An invalid state breaks the causal baseline and censors any
                # candidate; do not carry stale quote/depth into later samples.
                self.m2_samples[session].clear()
                self.m2_medians[session]=None
                self.m2_candidate[session]=None
                self.last_mid_by_session.pop(session,None)
                self.last_book_sample_by_session.pop(session,None)
                self.mid_samples[session].clear(); self.mid_lookup[session].clear()
                self.mid_min[session].clear(); self.mid_max[session].clear()
                self.range5s_history[session].clear(); self.current5s[session]=None
                self.m4_baseline5[session]=None; self.m4_compressed[session]=False
            # Executions remain valid market observations when the book is
            # flagged invalid; retain their side/volume and trade-price range.
            if session>=0 and action=="T" and size>0:
                if side not in ("A","B"):
                    self.validity["unknown_side_trades"]+=1
                else:
                    self.validity["known_side_trades"]+=1
                    b=t//500_000_000; key=(session,b)
                    self.m1_bins[key]=self.m1_bins.get(key,0.0)+(size if side=="B" else -size)
                    self.m1_bin_ord[key]=ordinal
                if price>0:
                    self.trade_window.append((t,price))
                    while self.m3_highs and self.m3_highs[-1][1]<=price:self.m3_highs.pop()
                    self.m3_highs.append((t,price))
                    while self.m3_lows and self.m3_lows[-1][1]>=price:self.m3_lows.pop()
                    self.m3_lows.append((t,price))
            while self.trade_window and self.trade_window[0][0]<=t-1800*NS:self.trade_window.popleft()
            while self.m3_highs and self.m3_highs[0][0]<=t-1800*NS:self.m3_highs.popleft()
            while self.m3_lows and self.m3_lows[0][0]<=t-1800*NS:self.m3_lows.popleft()
            self._transition("M2", "INVALID_OR_OUT_OF_SESSION")
            if self.m3_candidate is not None:
                self._transition("M3", "CENSORED_INVALID_BOOK")
                self.m3_candidate = None
            return

        bid = state.bids[0][0] / 1e9
        ask = state.asks[0][0] / 1e9
        mid = (bid + ask) / 2
        quote_state=(state.bids[0],state.asks[0])
        if quote_state!=self.last_quote_state:
            self.quote_rows["t"].append(t); self.quote_rows["ordinal"].append(ordinal)
            self.quote_rows["bid"].append(state.bids[0][0]); self.quote_rows["ask"].append(state.asks[0][0])
            self.last_quote_state=quote_state
        # Causal 50ms as-of midpoint grid bounds memory and supplies fixed
        # sampling for the appendix's one-second momentum and range features.
        mid_clock=self.next_mid_sample.get(session,t)
        sampled_mid=False
        previous_mid=self.last_mid_by_session.get(session)
        while mid_clock<t:
            sample_value=None
            if previous_mid is not None and previous_mid[0]<=mid_clock and mid_clock-previous_mid[0]<=2*NS:
                sample_value=previous_mid[1]
            if sample_value is None:
                self.next_mid_sample[session]=mid_clock+50_000_000
                mid_clock+=50_000_000
                continue
            while self.mid_samples[session] and self.mid_samples[session][0][0] < mid_clock-5*NS:
                old_t,_=self.mid_samples[session].popleft()
                self.mid_lookup[session].pop(old_t,None)
            self.mid_samples[session].append((mid_clock,sample_value))
            self.mid_lookup[session][mid_clock]=sample_value
            sampled_mid=True
            b5=mid_clock//(5*NS)
            current5=self.current5s[session]
            if current5 is not None and b5>current5[0]:
                self.range5s_history[session].append(current5[2]-current5[1])
                if len(self.range5s_history[session])>=360:
                    self.m4_baseline5[session]=float(np.median(np.fromiter(self.range5s_history[session],dtype=np.float64)))
                current5=None
            if current5 is None:self.current5s[session]=(b5,sample_value,sample_value)
            else:self.current5s[session]=(b5,min(current5[1],sample_value),max(current5[2],sample_value))
            lowq=self.mid_min[session]; highq=self.mid_max[session]
            while lowq and lowq[-1][1]>=sample_value: lowq.pop()
            lowq.append((mid_clock,sample_value))
            while highq and highq[-1][1]<=sample_value: highq.pop()
            highq.append((mid_clock,sample_value))
            cutoff=mid_clock-5*NS
            while lowq and lowq[0][0]<=cutoff: lowq.popleft()
            while highq and highq[0][0]<=cutoff: highq.popleft()
            self.last_m4_5s_bin[session]=b5
            self.next_mid_sample[session]=mid_clock+50_000_000
            mid_clock+=50_000_000
        self.last_mid_by_session[session]=(t,mid)

        # Complete prior bins before incorporating the current source record.
        self._close_m1_bins(session,t)
        if action == "T" and size > 0:
            if side not in ("A", "B"):
                self.validity["unknown_side_trades"] += 1
            else:
                self.validity["known_side_trades"] += 1
                # Existing canonical execution semantics: B is aggressor buy,
                # A is aggressor sell (see baseline adapter).
                signed = size if side == "B" else -size
                key = (session, bin_id)
                self.m1_bins[key] = self.m1_bins.get(key, 0.0) + signed
                self.m1_bin_ord[key] = ordinal
            # M3 rolling trade range/volume.

        # M2: sample as-of book at 50ms grid; assess every native valid row.
        top_bid = sum(s for _, s in state.bids[:5])
        top_ask = sum(s for _, s in state.asks[:5])
        sample_clock = self.next_sample.get(session, t)
        previous_book=self.last_book_sample_by_session.get(session)
        while sample_clock < t:
            sample_depth=None
            if previous_book is not None and previous_book[0]<=sample_clock and sample_clock-previous_book[0]<=2*NS:
                sample_depth=(previous_book[1],previous_book[2])
            if sample_depth is not None:
                samples0=self.m2_samples[session]
                if len(samples0)>=100:
                    self.m2_medians[session]=(float(np.median([x[1][0] for x in samples0])),
                                              float(np.median([x[1][1] for x in samples0])))
                samples0.append((sample_clock,sample_depth))
            self.next_sample[session] = sample_clock + 50_000_000
            sample_clock += 50_000_000
        self.last_book_sample_by_session[session]=(t,top_bid,top_ask)
        origin=self.mid_grid_origin[session]
        target_index=(t-origin-NS)//50_000_000
        target=origin+target_index*50_000_000
        past_mid=self.mid_lookup[session].get(target) if target_index>=0 else None
        direction = 0 if past_mid is None else int(np.sign(mid - past_mid))
        opposing = top_ask if direction > 0 else top_bid if direction < 0 else 0
        medians=self.m2_medians[session]
        base=None if medians is None or not direction else medians[1 if direction>0 else 0]
        # Cached 100-sample median is updated only on the 50ms sample clock.
        candidate = self.m2_candidate.get(session)
        if direction and base is not None:
            below = base > 0 and opposing < .5 * base
            if below:
                if candidate is None:
                    self.m2_candidate[session] = (t, ordinal, base, direction, opposing)
                    self._transition("M2", "RELATIVE_THINNING_OBSERVED")
                elif direction == candidate[3] and t - candidate[0] >= 250_000_000:
                    if t - self.m2_last_emit.get(session, -10**30) >= 2 * NS:
                        # Emit at qualification availability, not retroactively
                        # at the first observation of the thinning candidate.
                        self.events["M2"].append(_empty_event("M2", self.day, t, ordinal, direction,
                            {"opposing_top5_size": candidate[4], "trailing_5s_median": candidate[2],
                             "ratio": candidate[4] / candidate[2], "qualification_ts_recv": t,
                             "thinning_start_ts_recv": candidate[0],
                             "thinning_start_source_ordinal": candidate[1],
                             "persistence_ns": t-candidate[0], "price_mid": mid}))
                        self.m2_last_emit[session] = t
                        self._transition("M2", "THIN_BOOK_QUALIFIED")
                    self.m2_candidate[session] = None
            else:
                if candidate is not None:
                    self._transition("M2", "THINNING_RESET")
                self.m2_candidate[session] = None
        elif candidate is not None:
            self._transition("M2", "CENSORED_OR_INSUFFICIENT_BASELINE")
            self.m2_candidate[session] = None

        # M3: trailing prior trade range; crossing versus acceptance are separate.
        if (len(self.trade_window) >= 2 and self.m3_highs and self.m3_lows and
                t-self.session_start.get(session,t)>=1800*NS):
            hi, lo = self.m3_highs[0][1] / 1e9, self.m3_lows[0][1] / 1e9
            if self.m3_candidate is None:
                if mid > hi:
                    initial_volume=(size if action=="T" and size>0 and price>0 and price>self.m3_highs[0][1] else 0)
                    self.m3_candidate = (t, ordinal, 1, int(self.m3_highs[0][1]), initial_volume)
                    self._transition("M3", "BOUNDARY_CROSSED_UP")
                elif mid < lo:
                    initial_volume=(size if action=="T" and size>0 and price>0 and price<self.m3_lows[0][1] else 0)
                    self.m3_candidate = (t, ordinal, -1, int(self.m3_lows[0][1]), initial_volume)
                    self._transition("M3", "BOUNDARY_CROSSED_DOWN")
            else:
                start, start_ord, d, boundary_raw, volume = self.m3_candidate
                boundary = boundary_raw / 1e9
                outside = mid > boundary if d > 0 else mid < boundary
                if not outside:
                    self._transition("M3", "CROSSING_RESET")
                    self.m3_candidate = None
                else:
                    if action == "T" and size > 0 and price > 0 and ((price / 1e9 > boundary) if d > 0 else (price / 1e9 < boundary)):
                        volume += size
                    self.m3_candidate = (start, start_ord, d, boundary_raw, volume)
                    if t - start >= NS and volume >= 100:
                        self.events["M3"].append(_empty_event("M3", self.day, t, ordinal, d,
                            {"boundary": boundary, "acceptance_elapsed_ns": t-start,
                             "outside_executed_volume": volume, "range_high": hi, "range_low": lo}))
                        self._transition("M3", "ACCEPTANCE_QUALIFIED")
                        self.m3_candidate = None

        # Add this trade only after M3's prior-range boundary was evaluated.
        # That preserves strict prior-source-order availability.
        if action == "T" and size > 0 and price > 0:
            self.trade_window.append((t,price))
            while self.m3_highs and self.m3_highs[-1][1]<=price: self.m3_highs.pop()
            self.m3_highs.append((t,price))
            while self.m3_lows and self.m3_lows[-1][1]>=price: self.m3_lows.pop()
            self.m3_lows.append((t,price))
        while self.trade_window and self.trade_window[0][0] <= t - 1800 * NS:
            self.trade_window.popleft()
        while self.m3_highs and self.m3_highs[0][0] <= t-1800*NS: self.m3_highs.popleft()
        while self.m3_lows and self.m3_lows[0][0] <= t-1800*NS: self.m3_lows.popleft()

        # M4: update completed 5-minute midpoint ranges and 5-second ranges.
        five_min = t // (300 * NS)
        range_key=(session,five_min)
        previous_range=self.range_5m.get(range_key)
        self.range_5m[range_key]=(mid,mid) if previous_range is None else (min(previous_range[0],mid),max(previous_range[1],mid))
        # Completed 5m ranges become available on first event in next bucket.
        old_keys = [k for k in self.range_5m if k[0]==session and k[1] < five_min]
        for k in old_keys:
            old_sess, old_bucket = k
            low_high = self.range_5m.pop(k)
            hist = self.range5_history[old_sess]
            if len(hist) >= 20:
                q20 = _quantile(hist, .20)
                if q20 is not None and low_high[1]-low_high[0] <= q20:
                    self.m4_compressed[old_sess] = True
                    self._transition("M4", "COMPRESSION_QUALIFIED")
            hist.append(low_high[1]-low_high[0])
        # completed 5s range from midpoint samples; baseline is prior completed ranges.
        if sampled_mid and self.m4_compressed[session] and len(self.range5s_history[session]) >= 360:
            baseline5 = self.m4_baseline5[session] or 0.0
            mid_range=self.mid_max[session][0][1]-self.mid_min[session][0][1] if self.mid_max[session] and self.mid_min[session] else 0.0
            if baseline5 > 0 and mid_range >= 2*baseline5 and direction:
                if t - self.m4_last_emit.get(session, -10**30) >= 2*NS:
                    self.events["M4"].append(_empty_event("M4", self.day, t, ordinal, direction,
                        {"trailing_5s_range": mid_range, "prior_30m_median_5s_range": baseline5,
                         "compression_qualified": True, "momentum_1s_ticks": (mid-(past_mid or mid))/(TICK_RAW/1e9)}))
                    self._transition("M4", "EXPANSION_DIRECTIONAL_QUALIFIED")
                    self.m4_last_emit[session] = t
                    self.m4_compressed[session] = False


def _outcomes(events: list[dict[str, Any]], quotes: dict[str,array],
              source_gaps: list[tuple[int,int]] | None=None) -> list[dict[str, Any]]:
    if not events:
        return []
    if not quotes or not len(quotes["t"]):
        return [{"date":e["date"],"mechanism":e["mechanism"],"event_id":e["event_id"],
                 "trigger_ts_recv":e["ts_recv"],"source_ordinal":e["source_ordinal"],
                 "direction":e["direction"],"source_censored":True,
                 "censor_reason":"NO_VALID_QUOTES_FOR_DATE"} for e in events]
    qt=np.frombuffer(quotes["t"],dtype=np.int64)
    qo=np.frombuffer(quotes["ordinal"],dtype=np.int64)
    bidraw=np.frombuffer(quotes["bid"],dtype=np.int64)
    askraw=np.frombuffer(quotes["ask"],dtype=np.int64)
    gaps=source_gaps or []
    def gap_between(start:int,end:int)->bool:
        return any(gend>=start and gstart<=end for gstart,gend in gaps)
    def quote_asof(target: int, max_ordinal: int | None=None) -> int:
        j=int(np.searchsorted(qt,target,side="right")-1)
        if max_ordinal is not None:
            while j>=0 and int(qt[j])==target and int(qo[j])>max_ordinal:
                j-=1
        return j
    output=[]
    for ev in events:
        t=int(ev["ts_recv"]); d=int(ev["direction"])
        idx=quote_asof(t,int(ev["source_ordinal"]))
        if idx<0 or t-int(qt[idx])>2*NS:
            output.append({"date":ev["date"],"mechanism":ev["mechanism"],"event_id":ev["event_id"],
                "trigger_ts_recv":t,"source_ordinal":ev["source_ordinal"],"direction":d,
                "anchor_mid":None,"mid_markout_1s_ticks":None,"mid_markout_5s_ticks":None,
                "mid_markout_10s_ticks":None,"four_tick_first_passage_30s":"SOURCE_CENSORED",
                "mfe_30s_ticks":None,"mae_30s_ticks":None,"net_executable_10s_ticks":None,
                "source_censored":True,"censor_reason":"NO_FRESH_VALID_TRIGGER_QUOTE"})
            continue
        bid0=float(bidraw[idx])/1e9; ask0=float(askraw[idx])/1e9
        anchor=(bid0+ask0)/2
        row={"date":ev["date"],"mechanism":ev["mechanism"],"event_id":ev["event_id"],"trigger_ts_recv":t,"source_ordinal":ev["source_ordinal"],"direction":d,"anchor_mid":anchor}
        for sec in (1,5,10):
            target=t+sec*NS
            j=quote_asof(target)
            value=None
            if j>=0 and target-int(qt[j])<=2*NS and not gap_between(t,target):
                mid=(float(bidraw[j])+float(askraw[j]))/(2e9)
                value=d*(mid-anchor)/(TICK_RAW/1e9)
            row[f"mid_markout_{sec}s_ticks"]=value
        # Path begins strictly after trigger; first directional/opposite +/-4 tick barrier.
        lo=int(np.searchsorted(qt,t,side="left"))
        while lo<len(qt) and int(qt[lo])==t and int(qo[lo])<=int(ev["source_ordinal"]):lo+=1
        hi=int(np.searchsorted(qt,t+30*NS,side="right"))
        mids=(bidraw[lo:hi].astype(np.float64)+askraw[lo:hi].astype(np.float64))/(2e9)
        signed=d*(mids-anchor)/(TICK_RAW/1e9)
        up=np.flatnonzero(signed>=4); down=np.flatnonzero(signed<=-4)
        if gap_between(t,t+30*NS) or int(qt[-1])<t+30*NS: first="SOURCE_CENSORED"
        elif len(up) and len(down) and up[0]==down[0]: first="TIE_OR_ORDER_AMBIGUOUS"
        elif len(up) and (not len(down) or up[0]<down[0]): first="PREDICTED_FIRST"
        elif len(down): first="OPPOSITE_FIRST"
        else: first="UNRESOLVED"
        row["four_tick_first_passage_30s"]=first
        path_censored=gap_between(t,t+30*NS) or int(qt[-1])<t+30*NS
        row["mfe_30s_ticks"]=float(np.max(signed)) if len(signed) and not path_censored else None
        row["mae_30s_ticks"]=float(np.min(signed)) if len(signed) and not path_censored else None
        # Actual marketable entry at first BBO at/after trigger+2ms.
        entry_target=t+2_000_000
        ei=quote_asof(entry_target)
        executable=None
        if ei>=0 and entry_target-int(qt[ei])<=2*NS and not gap_between(t,entry_target):
            entry=(float(askraw[ei])/1e9+1*(TICK_RAW/1e9)) if d>0 else (float(bidraw[ei])/1e9-1*(TICK_RAW/1e9))
            # As-of valid liquidation BBO at the 10s checkpoint, no future quote fill.
            exit_target=entry_target+10*NS
            xi=quote_asof(exit_target)
            if xi>=ei and exit_target-int(qt[xi])<=2*NS and not gap_between(entry_target,exit_target):
                exit_price=(float(bidraw[xi])/1e9-1*(TICK_RAW/1e9)) if d>0 else (float(askraw[xi])/1e9+1*(TICK_RAW/1e9))
                commission_ticks=6.0/12.5
                executable=d*(exit_price-entry)/(TICK_RAW/1e9)-commission_ticks
        row["net_executable_10s_ticks"]=executable
        row["source_censored"]=bool(executable is None or row["mid_markout_10s_ticks"] is None or path_censored)
        if row["source_censored"]:
            row["censor_reason"]="SOURCE_GAP_OR_INSUFFICIENT_FUTURE_VALID_QUOTE_COVERAGE"
        else:
            row["censor_reason"]=None
        output.append(row)
    return output


def _construction_outcomes(events:list[dict[str,Any]], quotes:dict[str,array],
                           source_gaps:list[tuple[int,int]]|None=None)->list[dict[str,Any]]:
    """Frozen construction outcomes at all registered horizons/barriers/costs."""
    if not events:return []
    import bisect
    if not quotes or not len(quotes["t"]):
        return [{"date":e["date"],"mechanism":e["mechanism"],"event_id":e["event_id"],
                 "trigger_ts_recv":e["ts_recv"],"source_ordinal":e["source_ordinal"],
                 "direction":e["direction"],"anchor_mid":None,
                 **{f"mid_markout_{s}s_ticks":None for s in (1,5,10,30,60)},
                 **{f"barrier_{b}_first": "SOURCE_CENSORED" for b in (2,4,6,8)},
                 "first_favorable_4_tick_ns":None,"first_adverse_4_tick_ns":None,
                 "mfe_30s_ticks":None,"mae_30s_ticks":None,
                 **{f"net_executable_10s_{c.lower()}_ticks":None for c in ("BASELINE","MODERATE","SEVERE")},
                 "source_censored":True,"censor_reason":"NO_VALID_QUOTES_FOR_DATE"} for e in events]
    qt=np.frombuffer(quotes["t"],dtype=np.int64); qo=np.frombuffer(quotes["ordinal"],dtype=np.int64)
    bid=np.frombuffer(quotes["bid"],dtype=np.int64); ask=np.frombuffer(quotes["ask"],dtype=np.int64)
    gaps=sorted(source_gaps or [])
    gap_starts=[x[0] for x in gaps]; gap_prefix_ends=[]; max_end=-2**63
    for _,end in gaps:
        max_end=max(max_end,end); gap_prefix_ends.append(max_end)
    def has_gap(a:int,b:int)->bool:
        j=bisect.bisect_right(gap_starts,b)-1
        return j>=0 and gap_prefix_ends[j]>=a
    def asof(target:int,max_ordinal:int|None=None)->int:
        j=int(np.searchsorted(qt,target,side="right")-1)
        if max_ordinal is not None:
            while j>=0 and int(qt[j])==target and int(qo[j])>max_ordinal:j-=1
        return j
    output=[]
    for ev in events:
        t=int(ev["ts_recv"]); d=int(ev["direction"]); ordinal=int(ev["source_ordinal"])
        anchor_i=asof(t,ordinal)
        base={"date":ev["date"],"mechanism":ev["mechanism"],"event_id":ev["event_id"],
              "trigger_ts_recv":t,"source_ordinal":ordinal,"direction":d}
        if anchor_i<0 or t-int(qt[anchor_i])>2*NS:
            output.append({**base,"anchor_mid":None,
                **{f"mid_markout_{s}s_ticks":None for s in (1,5,10,30,60)},
                **{f"barrier_{b}_first":"SOURCE_CENSORED" for b in (2,4,6,8)},
                "first_favorable_4_tick_ns":None,"first_adverse_4_tick_ns":None,
                "mfe_30s_ticks":None,"mae_30s_ticks":None,
                **{f"net_executable_10s_{c.lower()}_ticks":None for c in ("BASELINE","MODERATE","SEVERE")},
                "source_censored":True,"censor_reason":"NO_FRESH_VALID_TRIGGER_QUOTE"})
            continue
        bid0=int(bid[anchor_i]); ask0=int(ask[anchor_i]); anchor=(bid0+ask0)/(2e9)
        row={**base,"anchor_mid":anchor}
        for sec in (1,5,10,30,60):
            target=t+sec*NS; j=asof(target)
            row[f"mid_markout_{sec}s_ticks"]=(d*((int(bid[j])+int(ask[j]))/(2e9)-anchor)/(TICK_RAW/1e9)
                if j>=0 and target-int(qt[j])<=2*NS and not has_gap(t,target) else None)
        lo=int(np.searchsorted(qt,t,side="left"))
        while lo<len(qt) and int(qt[lo])==t and int(qo[lo])<=ordinal:lo+=1
        end30=t+30*NS; hi=int(np.searchsorted(qt,end30,side="right"))
        path_mid=(bid[lo:hi].astype(np.float64)+ask[lo:hi].astype(np.float64))/(2e9)
        signed=d*(path_mid-anchor)/(TICK_RAW/1e9)
        path_t=qt[lo:hi]
        first_hits={}
        for barrier in (2,4,6,8):
            favorable=np.flatnonzero(signed>=barrier); adverse=np.flatnonzero(signed<=-barrier)
            fi=int(favorable[0]) if len(favorable) else None
            ai=int(adverse[0]) if len(adverse) else None
            first=min([x for x in (fi,ai) if x is not None],default=None)
            gap_before=has_gap(t,int(path_t[first])) if first is not None else has_gap(t,end30)
            if gap_before:
                label="SOURCE_CENSORED"
            elif first is None:
                label="SOURCE_CENSORED" if int(qt[-1])<end30 else "UNRESOLVED"
            elif fi is not None and ai is not None and fi==ai:
                label="TIE_OR_ORDER_AMBIGUOUS"
            elif fi is not None and (ai is None or fi<ai):label="PREDICTED_FIRST"
            else:label="OPPOSITE_FIRST"
            row[f"barrier_{barrier}_first"]=label
            first_hits[barrier]=(fi,ai)
        f4,a4=first_hits[4]
        row["first_favorable_4_tick_ns"]=None if f4 is None else int(path_t[f4])-t
        row["first_adverse_4_tick_ns"]=None if a4 is None else int(path_t[a4])-t
        window_censored=has_gap(t,end30) or int(qt[-1])<end30
        row["mfe_30s_ticks"]=float(np.max(signed)) if len(signed) and not window_censored else None
        row["mae_30s_ticks"]=float(np.min(signed)) if len(signed) and not window_censored else None
        entry_t=t+2_000_000; ei=asof(entry_t); exit_t=entry_t+10*NS; xi=asof(exit_t)
        for scenario,entry_slip,exit_slip in (("BASELINE",1.0,1.0),("MODERATE",1.5,1.5),("SEVERE",3.0,3.0)):
            value=None
            if (ei>=0 and entry_t-int(qt[ei])<=2*NS and not has_gap(t,entry_t) and xi>=ei
                and exit_t-int(qt[xi])<=2*NS and not has_gap(entry_t,exit_t)):
                entry=(int(ask[ei])/1e9+entry_slip*.25) if d>0 else (int(bid[ei])/1e9-entry_slip*.25)
                exit_price=(int(bid[xi])/1e9-exit_slip*.25) if d>0 else (int(ask[xi])/1e9+exit_slip*.25)
                value=d*(exit_price-entry)/.25-6.0/12.5
            row[f"net_executable_10s_{scenario.lower()}_ticks"]=value
        row["source_censored"]=bool(row["mid_markout_10s_ticks"] is None or
            row["barrier_4_first"]=="SOURCE_CENSORED" or row["net_executable_10s_baseline_ticks"] is None)
        row["censor_reason"]="SOURCE_GAP_OR_INSUFFICIENT_FUTURE_VALID_QUOTE_COVERAGE" if row["source_censored"] else None
        output.append(row)
    return output


def _write_parquet(path: Path, rows: list[dict[str, Any]], mechanism: str | None = None, *, outcomes: bool=False,
                   construction_outcomes:bool=False):
    import pyarrow as pa
    import pyarrow.parquet as pq
    if construction_outcomes:
        fields=[("date",pa.string()),("mechanism",pa.string()),("event_id",pa.string()),
            ("trigger_ts_recv",pa.int64()),("source_ordinal",pa.int64()),("direction",pa.int8()),
            ("anchor_mid",pa.float64())]
        fields.extend((f"mid_markout_{s}s_ticks",pa.float64()) for s in (1,5,10,30,60))
        fields.extend((f"barrier_{b}_first",pa.string()) for b in (2,4,6,8))
        fields.extend([("first_favorable_4_tick_ns",pa.int64()),("first_adverse_4_tick_ns",pa.int64()),
            ("mfe_30s_ticks",pa.float64()),("mae_30s_ticks",pa.float64())])
        fields.extend((f"net_executable_10s_{c}_ticks",pa.float64()) for c in ("baseline","moderate","severe"))
        fields.extend([("source_censored",pa.bool_()),("censor_reason",pa.string())])
        table=pa.Table.from_pylist(rows,schema=pa.schema(fields))
    elif outcomes:
        schema=pa.schema([("date",pa.string()),("mechanism",pa.string()),("event_id",pa.string()),
            ("trigger_ts_recv",pa.int64()),("source_ordinal",pa.int64()),("direction",pa.int8()),
            ("anchor_mid",pa.float64()),("mid_markout_1s_ticks",pa.float64()),
            ("mid_markout_5s_ticks",pa.float64()),("mid_markout_10s_ticks",pa.float64()),
            ("four_tick_first_passage_30s",pa.string()),("mfe_30s_ticks",pa.float64()),
            ("mae_30s_ticks",pa.float64()),("net_executable_10s_ticks",pa.float64()),
            ("source_censored",pa.bool_()),("censor_reason",pa.string())])
        table=pa.Table.from_pylist(rows,schema=schema)
    elif rows:
        table=pa.Table.from_pylist(rows)
    else:
        schema=pa.schema([("date",pa.string()),("mechanism",pa.string()),("event_id",pa.string()),
                          ("ts_recv",pa.int64()),("source_ordinal",pa.int64()),("direction",pa.int8()),
                          ("direction_reason",pa.string()),("features",pa.string()),("source_valid",pa.bool_()),("expired",pa.bool_())])
        table=pa.Table.from_pylist([],schema=schema)
    pq.write_table(table,path,compression="zstd")


def source_window_checkpoint_equivalence(repo:Path, source:dict[str,Any], count:int=10_000)->dict[str,Any]:
    """Compare book and detector state across resume on a bounded real DBN prefix."""
    from databento import DBNStore
    store=DBNStore.from_file(str(repo/source["raw_path"]))
    if store.metadata.dataset!="GLBX.MDP3" or str(store.metadata.schema)!="mbp-10":
        raise ValueError("source-window identity/schema mismatch")
    batches=[]; remaining=count
    for batch in store.to_ndarray(count=min(count,10_000)):
        take=min(remaining,len(batch)); batches.append(batch[:take].copy()); remaining-=take
        if remaining<=0:break
    close=getattr(store,"close",None)
    if callable(close):close()
    if not batches:return {"status":"FAIL","records":0}
    rows=np.concatenate(batches)[:count]
    split=len(rows)//2
    day=str(source["date"])
    def apply(book:NativeBookReplay, detectors:DetectorSuite, subset):
        for row in subset:
            state,_,_=book.process(row)
            action=bytes(row["action"]).decode("ascii","replace")
            side=bytes(row["side"]).decode("ascii","replace")
            detectors.on_row(state,action,int(row["price"]),int(row["size"]),side)
    full_book=NativeBookReplay(); full_detectors=DetectorSuite(day)
    apply(full_book,full_detectors,rows)
    left_book=NativeBookReplay(); left_detectors=DetectorSuite(day)
    apply(left_book,left_detectors,rows[:split])
    serialized=json.loads(json.dumps({"book":left_book.checkpoint(),"detectors":left_detectors.checkpoint()},allow_nan=False))
    resumed_book=NativeBookReplay.resume(serialized["book"])
    resumed_detectors=DetectorSuite.resume(serialized["detectors"])
    apply(resumed_book,resumed_detectors,rows[split:])
    equivalent=(full_book.checkpoint()==resumed_book.checkpoint() and
                full_detectors.checkpoint()==resumed_detectors.checkpoint())
    return {"status":"PASS" if equivalent else "FAIL","date":source["date"],"source_sha256":source["raw_sha256"],
            "records":len(rows),"split_after_record":split,"uninterrupted_vs_resumed_equal":equivalent,
            "book_state_equivalent":full_book.checkpoint()==resumed_book.checkpoint(),
            "detector_state_equivalent":full_detectors.checkpoint()==resumed_detectors.checkpoint(),
            "native_order_preserved":True,"test_type":"bounded source-backed DBN prefix; full row replay with detector checkpoint/resume"}


def run_one_date(repo: Path, out: Path, day: str, chunk_rows: int=100_000, *,
                 construction_outcomes:bool=False,
                 checkpoint_equivalence:dict[str,Any]|None=None) -> dict[str, Any]:
    from databento import DBNStore
    sources=audit.source_rows(repo)
    source=next((x for x in sources if x["date"]==day),None)
    if source is None: raise ValueError(f"date not in authorized source manifest: {day}")
    raw=repo/source["raw_path"]
    if audit.sha256(raw)!=source["raw_sha256"]: raise ValueError("source hash mismatch")
    equivalence=checkpoint_equivalence or source_window_checkpoint_equivalence(repo,source)
    if equivalence["status"]!="PASS":raise ValueError("source-backed checkpoint replay equivalence failed")
    store=DBNStore.from_file(str(raw)); md=store.metadata
    if md.dataset!="GLBX.MDP3" or str(md.schema)!="mbp-10" or source["symbol"] not in md.symbols:
        raise ValueError("DBN identity/schema mismatch")
    replay=NativeBookReplay(); suite=DetectorSuite(day); ordinal=0
    started=time.monotonic(); records=depth_updates=depth_only=0
    prior_bid_px=prior_bid_sz=prior_ask_px=prior_ask_sz=None
    last_recv=None; next_grid=None; previous_session=-99; source_gaps=[]
    windows=_windows(day)
    for batch in store.to_ndarray(count=chunk_rows):
        if np.any(batch["rtype"]!=10): raise ValueError("non-MBP-10 record in input")
        n=len(batch); ts=batch["ts_recv"].astype(np.int64,copy=False)
        bp=np.column_stack([batch[f"bid_px_{i:02d}"] for i in range(10)])
        bs=np.column_stack([batch[f"bid_sz_{i:02d}"] for i in range(10)])
        ap=np.column_stack([batch[f"ask_px_{i:02d}"] for i in range(10)])
        ass=np.column_stack([batch[f"ask_sz_{i:02d}"] for i in range(10)])
        flags=batch["flags"].astype(np.uint8,copy=False)
        bid_ok=(bp[:,0]>0)&(bs[:,0]>0); ask_ok=(ap[:,0]>0)&(ass[:,0]>0)
        crossed=bid_ok&ask_ok&(bp[:,0]>=ap[:,0])
        bid_order=np.all((bp[:,:-1]==0)|(bp[:,1:]==0)|(bp[:,:-1]>bp[:,1:]),axis=1)
        ask_order=np.all((ap[:,:-1]==0)|(ap[:,1:]==0)|(ap[:,:-1]<ap[:,1:]),axis=1)
        bad=(flags & (BAD_RECV|MAYBE_BAD_BOOK))!=0
        gapmask=np.zeros(n,dtype=bool)
        if last_recv is not None and n: gapmask[0]=int(ts[0])-last_recv>2*NS
        if n>1: gapmask[1:]=ts[1:]-ts[:-1]>2*NS
        for gi in np.flatnonzero(gapmask):
            prev=int(ts[gi-1]) if gi>0 else int(last_recv or ts[gi])
            source_gaps.append((prev,int(ts[gi])))
        for gi in np.flatnonzero(bad): source_gaps.append((int(ts[gi]),int(ts[gi])))
        valid=bid_ok&ask_ok&bid_order&ask_order&~crossed&~bad&~gapmask
        session_codes=np.full(n,-1,dtype=np.int8)
        for code,name in enumerate(("ASIA","EUROPE","NY")):
            start,end=windows[name]; session_codes[(ts>=start)&(ts<end)]=code
        same_bbo=np.zeros(n,dtype=bool); changed=np.zeros(n,dtype=bool)
        if prior_bid_px is not None and n:
            changed[0]=not(np.array_equal(bp[0],prior_bid_px) and np.array_equal(bs[0],prior_bid_sz) and np.array_equal(ap[0],prior_ask_px) and np.array_equal(ass[0],prior_ask_sz))
            same_bbo[0]=bp[0,0]==prior_bid_px[0] and bs[0,0]==prior_bid_sz[0] and ap[0,0]==prior_ask_px[0] and ass[0,0]==prior_ask_sz[0]
        if n>1:
            changed[1:]=np.any(bp[1:]!=bp[:-1],axis=1)|np.any(bs[1:]!=bs[:-1],axis=1)|np.any(ap[1:]!=ap[:-1],axis=1)|np.any(ass[1:]!=ass[:-1],axis=1)
            same_bbo[1:]=(bp[1:,0]==bp[:-1,0])&(bs[1:,0]==bs[:-1,0])&(ap[1:,0]==ap[:-1,0])&(ass[1:,0]==ass[:-1,0])
        snap=(flags&SNAPSHOT)!=0
        changed|=snap
        depth_updates+=int(np.count_nonzero(changed)); depth_only+=int(np.count_nonzero(changed&same_bbo))
        bbo_changed=np.zeros(n,dtype=bool)
        if n>1: bbo_changed[1:]=~same_bbo[1:]
        if prior_bid_px is not None and n: bbo_changed[0]=not same_bbo[0]
        is_trade=(batch["action"]==b"T")&(batch["size"]>0)
        # Replay every source state vectorized; send only detector-relevant
        # transitions plus as-of 50ms observations into the Python state machines.
        # M1's frozen bin-end timestamp requires the final native ordinal in
        # every completed bin; passing all source rows also makes detector-state
        # checkpoint equivalence directly testable against native record order.
        relevant=np.ones(n,dtype=bool)
        top5changed=np.zeros(n,dtype=bool)
        if n>1:
            top5changed[1:]=np.any(bp[1:,:5]!=bp[:-1,:5],axis=1)|np.any(bs[1:,:5]!=bs[:-1,:5],axis=1)|np.any(ap[1:,:5]!=ap[:-1,:5],axis=1)|np.any(ass[1:,:5]!=ass[:-1,:5],axis=1)
        if prior_bid_px is not None and n:
            top5changed[0]=not(np.array_equal(bp[0,:5],prior_bid_px[:5]) and np.array_equal(bs[0,:5],prior_bid_sz[:5]) and np.array_equal(ap[0,:5],prior_ask_px[:5]) and np.array_equal(ass[0,:5],prior_ask_sz[:5]))
        relevant|=top5changed
        if n:
            if next_grid is None: next_grid=((int(ts[0])+49_999_999)//50_000_000)*50_000_000
            while next_grid<=int(ts[-1]):
                ix=int(np.searchsorted(ts,next_grid,side="right")-1)
                if ix>=0: relevant[ix]=True
                next_grid+=50_000_000
            # Preserve first record entering each registered session.
            entered=np.zeros(n,dtype=bool)
            if n: entered[0]=int(session_codes[0])!=previous_session
            if n>1: entered[1:]=session_codes[1:]!=session_codes[:-1]
            relevant|=entered
            if n: previous_session=int(session_codes[-1])
            indices=np.flatnonzero(relevant)
            for ix in indices:
                row=batch[ix]
                state=BookState(ordinal+int(ix),int(row["ts_recv"]),int(row["ts_event"]),
                    tuple((int(bp[ix,j]),int(bs[ix,j])) for j in range(10)),
                    tuple((int(ap[ix,j]),int(ass[ix,j])) for j in range(10)),bool(valid[ix]),bool(snap[ix]),bool(bad[ix] or gapmask[ix]))
                action=bytes(row["action"]).decode("ascii","replace"); side=bytes(row["side"]).decode("ascii","replace")
                suite.on_row(state,action,int(row["price"]),int(row["size"]),side)
            prior_bid_px=bp[-1].copy(); prior_bid_sz=bs[-1].copy(); prior_ask_px=ap[-1].copy(); prior_ask_sz=ass[-1].copy()
            last_recv=int(ts[-1]); records+=n; ordinal+=n
            replay.next_ordinal=ordinal
            replay.state=BookState(ordinal-1,int(ts[-1]),int(batch["ts_event"][-1]),
                tuple((int(bp[-1,j]),int(bs[-1,j])) for j in range(10)),
                tuple((int(ap[-1,j]),int(ass[-1,j])) for j in range(10)),bool(valid[-1]),bool(snap[-1]),bool(bad[-1] or gapmask[-1]))
            replay.validity["rows"]+=n; replay.validity["valid"]+=int(np.count_nonzero(valid)); replay.validity["invalid"]+=int(n-np.count_nonzero(valid))
            replay.validity["crossed_locked"]+=int(np.count_nonzero(crossed)); replay.validity["bad_recv_flag"]+=int(np.count_nonzero(flags&BAD_RECV))
            replay.validity["maybe_bad_book_flag"]+=int(np.count_nonzero(flags&MAYBE_BAD_BOOK)); replay.validity["snapshots"]+=int(np.count_nonzero(snap))
            replay.validity["gaps"]+=int(np.count_nonzero(gapmask|bad)); replay.validity["depth_updates"]+=int(np.count_nonzero(changed)); replay.validity["depth_only_updates"]+=int(np.count_nonzero(changed&same_bbo))
    close=getattr(store,"close",None)
    if callable(close):close()
    all_events=[event for mech in ("M1","M2","M3","M4") for event in suite.events[mech]]
    outcomes=(_construction_outcomes(all_events,suite.quote_rows,source_gaps) if construction_outcomes
              else _outcomes(all_events,suite.quote_rows,source_gaps))
    out.mkdir(parents=True,exist_ok=True)
    for mech in ("M1","M2","M3","M4"):
        _write_parquet(out/f"{mech.lower()}-events.parquet",suite.events[mech],mech)
    outcomes_name="construction-outcomes.parquet" if construction_outcomes else "one-date-forward-outcomes.parquet"
    _write_parquet(out/outcomes_name,outcomes,outcomes=not construction_outcomes,
                   construction_outcomes=construction_outcomes)
    checkpoint={"date":day,"source_sha256":source["raw_sha256"],"next_ordinal":replay.next_ordinal,
                "book_state":None if replay.state is None else replay.state.as_dict(),"book_validity":replay.validity}
    (out/"final-book-checkpoint.json").write_text(json.dumps(checkpoint,sort_keys=True,separators=(",",":"))+"\n")
    result={"status":"PASS","date":day,"source_path":source["raw_path"],"source_sha256":source["raw_sha256"],
            "dataset":md.dataset,"schema":str(md.schema),"symbol":source["symbol"],"source_records_processed":records,
            "depth_updates":depth_updates,"depth_only_updates":depth_only,"book_validity":replay.validity,
            "detector_source_validity":suite.validity,"transition_counts":suite.transitions,
            "source_gap_intervals_count":len(source_gaps),
            "source_window_checkpoint_equivalence":equivalence,
            "trigger_counts":{m:len(suite.events[m]) for m in suite.events},"outcome_rows":len(outcomes),
            "outcome_rows_by_mechanism":{m:sum(r["mechanism"]==m for r in outcomes) for m in suite.events},
            "source_censored_outcomes":sum(bool(x["source_censored"]) for x in outcomes),
            "runtime_seconds":time.monotonic()-started,"events_paths":{m:f"{m.lower()}-events.parquet" for m in suite.events},
            "outcomes_path":outcomes_name,"checkpoint_path":"final-book-checkpoint.json",
            "checkpoint_replay_equivalence":"SYNTHETIC_TESTED; source run checkpoint saved at EOF"}
    (out/"source-validity.json").write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    (out/"native-depth-replay-equivalence.json").write_text(json.dumps(equivalence,indent=2,sort_keys=True)+"\n")
    return result
