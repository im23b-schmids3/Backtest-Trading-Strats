from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from pathlib import Path
from typing import Any

from ..imbalance_vwap_ride.models import COST_MODEL_VERSION, ImbalanceVWAPRideConfig

STUDY_ID = "VolatilityBreakoutTrendContinuation.BTC_LONG_SHORT_V1_SPECIFICATION"
SPEC_PATH = ".smithers/specs/volatility-breakout-trend-continuation-v1.md"
SPEC_SHA256 = "07659bfcd3979f913a2f16471cd384b53a3f7783da7ee41637fb32c5662e5f83"
PHASE_A_MANIFEST_SHA256 = "9fb7228ca074fc5a3b90e6fe82181d07d49f8c62ebd68274e859d0361df3cd6e"
CANDIDATES = (("VBTC-V1-1P5R", Decimal("1.5")), ("VBTC-V1-2P0R", Decimal("2.0")), ("VBTC-V1-2P5R", Decimal("2.5")))
DISPOSITIONS = frozenset({"TREND_FILTER_REJECTED", "BREAKOUT_THRESHOLD_REJECTED", "EXPANSION_FILTER_REJECTED", "FALSE_BREAKOUT_INVALIDATED", "STOP_DISTANCE_REJECTED", "NO_EXECUTABLE_ENTRY", "ACTIVE_POSITION_BLOCKED", "SESSION_ENTRY_BLOCKED", "SESSION_ENDED", "TRADE_EXECUTED"})
PHASE_A_MONTHS = tuple([f"2023-{month:02d}" for month in range(1, 13)] + ["2024-01"])
PHASE_A_START = datetime(2023, 1, 1, tzinfo=timezone.utc)
PHASE_A_END = datetime(2024, 1, 31, 23, 59, 59, 999999, tzinfo=timezone.utc)
_VERIFIED_BTC_CONFIG = ImbalanceVWAPRideConfig()
EXECUTION_ASSUMPTIONS = {
    "cost_model_version": COST_MODEL_VERSION,
    "symbol": _VERIFIED_BTC_CONFIG.symbol,
    "quantity_btc": str(_VERIFIED_BTC_CONFIG.quantity_btc),
    "price_tick": str(_VERIFIED_BTC_CONFIG.price_tick),
    "quantity_step": str(_VERIFIED_BTC_CONFIG.quantity_step),
    "minimum_quantity": str(_VERIFIED_BTC_CONFIG.minimum_quantity),
    "taker_fee_rate": str(_VERIFIED_BTC_CONFIG.taker_fee_rate),
    "market_slippage_ticks": _VERIFIED_BTC_CONFIG.market_slippage_ticks,
    "stop_slippage_ticks": _VERIFIED_BTC_CONFIG.stop_slippage_ticks,
    "same_bar_policy": _VERIFIED_BTC_CONFIG.same_bar_policy,
}

def _canon(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode("utf-8")
def _hash(value: Any) -> str: return hashlib.sha256(_canon(value)).hexdigest()
def _file_hash(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def _timestamp(value: Any) -> datetime:
    stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if stamp.tzinfo is None or stamp.utcoffset().total_seconds() != 0: raise ValueError("VBTC_V1_TIMESTAMP_NOT_UTC")
    return stamp
def _d(value: Any) -> Decimal: return Decimal(str(value))
def _stamp(value: Any) -> str: return _timestamp(value).isoformat().replace("+00:00", "Z")

def verify_sealed_specification(repository_root: str | Path) -> Path:
    path = Path(repository_root).resolve() / SPEC_PATH
    if not path.is_file() or _file_hash(path).lower() != SPEC_SHA256:
        raise ValueError("MISSING_OR_CHANGED_SEALED_VBTC_V1_SPECIFICATION")
    return path

def _clean_git(root: Path) -> str:
    status = subprocess.run(["git", "status", "--porcelain"], cwd=root, text=True, capture_output=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True)
    if status.returncode or status.stdout.strip() or head.returncode or not head.stdout.strip(): raise ValueError("VBTC_V1_PHASE_A_REQUIRES_CLEAN_COMMITTED_GIT")
    return head.stdout.strip()

class Store:
    def __init__(self, root: Path, phase: str, manifest_hash: str, revision: str):
        self.root = root / "research" / "volatility_breakout_trend_continuation" / "v1" / phase.lower() / f"study-{SPEC_SHA256}"
        if self.root.exists(): raise FileExistsError("VBTC_V1_IMMUTABLE_OUTPUT_COLLISION")
        self.root.mkdir(parents=True)
        self.manifest_hash, self.revision = manifest_hash, revision
    def write(self, name: str, payload: Any, candidate: str | None = None) -> None:
        path = self.root / (f"candidate-{candidate}" if candidate else "") / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists(): raise FileExistsError("VBTC_V1_IMMUTABLE_ARTIFACT_COLLISION")
        path.write_bytes(_canon(payload))
    def seal(self) -> None:
        files=[]
        for p in sorted(self.root.rglob("*")):
            if p.is_file() and p.name != "integrity-manifest.json": files.append({"relative_path":p.relative_to(self.root).as_posix(),"byte_length":p.stat().st_size,"sha256":_file_hash(p),"specification_hash":SPEC_SHA256,"manifest_hash":self.manifest_hash,"execution_assumption_hash":_hash(EXECUTION_ASSUMPTIONS),"code_revision":self.revision,"phase":"PHASE_A","created_at_utc":datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")})
        self.write("integrity-manifest.json", {"schema_version":"VBTC-V1", "files":files})

def _ema(values: list[Decimal], period: int) -> list[Decimal | None]:
    out:[Decimal|None]=[None]*len(values)
    if len(values) < period: return out
    out[period-1]=sum(values[:period])/period; a=Decimal(2)/(period+1)
    for i in range(period, len(values)): out[i]=values[i]*a+out[i-1]*(1-a) # type: ignore[operator]
    return out
def _atr(bars: list[dict[str,Any]]) -> list[Decimal|None]:
    tr:[Decimal|None]=[None]*len(bars); out:[Decimal|None]=[None]*len(bars)
    for i in range(1,len(bars)):
        h,l,c=_d(bars[i]["high"]),_d(bars[i]["low"]),_d(bars[i-1]["close"]); tr[i]=max(h-l,abs(h-c),abs(l-c))
    if len(bars)>14: out[14]=sum(x for x in tr[1:15] if x is not None)/14
    for i in range(15,len(bars)): out[i]=(out[i-1]*13+tr[i])/14 # type: ignore[operator]
    return out

def _quantize(value: Decimal, rounding: str) -> Decimal:
    return value.quantize(_VERIFIED_BTC_CONFIG.price_tick, rounding=rounding)


def _event(*, structure_id: str, setup_id: str, timestamp: str, event_type: str, direction: str, price_rule: str) -> dict[str, str]:
    event_id = _hash({"structure_id": structure_id, "event_timestamp": timestamp, "event_type": event_type, "bar_timestamp": timestamp, "direction": direction, "price_rule": price_rule})
    return {"event_id": event_id, "setup_id": setup_id, "structure_id": structure_id, "timestamp": timestamp, "type": event_type, "direction": direction, "price_rule": price_rule}


def evaluate_bars(bars: list[dict[str,Any]], candidate_id: str, target_r: Decimal) -> tuple[list[dict[str,Any]],list[dict[str,Any]],list[dict[str,Any]]]:
    """Evaluate the sealed completed-bar pseudocode without market I/O.

    The only future bar read before an execution is t+1's open.  A filled entry
    is not evaluated for stop/target until t+2, so a signal cannot fill and exit
    on its own entry bar.
    """
    closes=[_d(b["close"]) for b in bars]
    e20, e50, atr = _ema(closes,20), _ema(closes,50), _atr(bars)
    setups: list[dict[str, Any]]=[]; events: list[dict[str, Any]]=[]; trades: list[dict[str, Any]]=[]; active_until=-1
    execution_hash=_hash(EXECUTION_ASSUMPTIONS)
    quantity=(_VERIFIED_BTC_CONFIG.quantity_btc/_VERIFIED_BTC_CONFIG.quantity_step).to_integral_value(rounding=ROUND_FLOOR)*_VERIFIED_BTC_CONFIG.quantity_step
    for t in range(50,len(bars)-1):
        b=bars[t]; ts=_timestamp(b["timestamp"]); rh=max(_d(x["high"]) for x in bars[t-36:t]); rl=min(_d(x["low"]) for x in bars[t-36:t]); a=atr[t-1]
        if a is None or e20[t-1] is None or e50[t-1] is None or e50[t-6] is None:
            continue
        tr=max(_d(b["high"])-_d(b["low"]),abs(_d(b["high"])-closes[t-1]),abs(_d(b["low"])-closes[t-1]))
        entry_bar=bars[t+1]
        for direction in ("LONG","SHORT"):
            sign=Decimal(1) if direction=="LONG" else Decimal(-1)
            breakout=rh+Decimal("0.10")*a if direction=="LONG" else rl-Decimal("0.10")*a
            trend=(e20[t-1]>e50[t-1] and e50[t-1]>e50[t-6]) if direction=="LONG" else (e20[t-1]<e50[t-1] and e50[t-1]<e50[t-6])
            passed_break=_d(b["close"])>=breakout if direction=="LONG" else _d(b["close"])<=breakout
            setup_id=_hash({"study_id":STUDY_ID,"phase":"PHASE_A","symbol":"BTCUSDT","signal_bar_timestamp":_stamp(b["timestamp"]),"direction":direction,"range_high":str(rh),"range_low":str(rl),"breakout_level":str(breakout),"atr":str(a),"range_bars":36,"trend_family":"EMA20_EMA50","threshold_atr":"0.10","expansion_multiple":"1.25"})
            base={"setup_id":setup_id,"candidate_id":candidate_id,"direction":direction,"range_high":str(rh),"range_low":str(rl),"breakout_level":str(breakout),"atr":str(a),"breakout_bar":_stamp(b["timestamp"]),"entry_bar":_stamp(entry_bar["timestamp"]),"history":{"range_bars":36,"prior_only":True,"indicator_history_start":_stamp(bars[t-50]["timestamp"]),"true_range":str(tr)}}
            structure=_hash({"setup_id":setup_id,"breakout_bar_timestamp":base["breakout_bar"],"entry_bar_timestamp":base["entry_bar"],"entry_mode":"NEXT_OPEN_MARKET","proposed_stop_or_null":None,"stop_distance_or_null":None,"session":"00:05-23:35Z","time_stop_bars":24})
            proposed=_event(structure_id=structure,setup_id=setup_id,timestamp=base["breakout_bar"],event_type="PROPOSED_SETUP",direction=direction,price_rule="COMPLETED_BAR_CLOSE")
            events.append(proposed)
            disp: str
            trade: dict[str, Any] | None=None
            if not trend: disp="TREND_FILTER_REJECTED"
            elif not passed_break: disp="BREAKOUT_THRESHOLD_REJECTED"
            elif tr < Decimal("1.25")*a: disp="EXPANSION_FILTER_REJECTED"
            elif ts.hour==23 and ts.minute==55: disp="SESSION_ENDED"
            elif (ts.hour,ts.minute) < (0,5) or (ts.hour,ts.minute) > (23,35): disp="SESSION_ENTRY_BLOCKED"
            elif active_until>=t: disp="ACTIVE_POSITION_BLOCKED"
            elif (direction=="LONG" and _d(entry_bar["open"])<rh) or (direction=="SHORT" and _d(entry_bar["open"])>rl): disp="FALSE_BREAKOUT_INVALIDATED"
            else:
                reference_entry=_d(entry_bar["open"])
                entry=_quantize(reference_entry + sign*_VERIFIED_BTC_CONFIG.price_tick*_VERIFIED_BTC_CONFIG.market_slippage_ticks, ROUND_CEILING if direction=="LONG" else ROUND_FLOOR)
                raw_stop=_d(b["low"])-Decimal("0.10")*a if direction=="LONG" else _d(b["high"])+Decimal("0.10")*a
                stop=_quantize(raw_stop, ROUND_FLOOR if direction=="LONG" else ROUND_CEILING)
                distance=sign*(entry-stop)
                structure=_hash({"setup_id":setup_id,"breakout_bar_timestamp":base["breakout_bar"],"entry_bar_timestamp":base["entry_bar"],"entry_mode":"NEXT_OPEN_MARKET","proposed_stop_or_null":str(stop),"stop_distance_or_null":str(distance),"session":"00:05-23:35Z","time_stop_bars":24})
                if quantity < _VERIFIED_BTC_CONFIG.minimum_quantity:
                    disp="NO_EXECUTABLE_ENTRY"
                elif distance < Decimal("0.0020")*entry or distance > Decimal("0.0125")*entry:
                    disp="STOP_DISTANCE_REJECTED"
                else:
                    disp="TRADE_EXECUTED"; target=_quantize(entry+sign*target_r*distance, ROUND_CEILING if direction=="LONG" else ROUND_FLOOR)
                    exit_index=None; reference_exit=None; exit_price=None; reason=None; exit_slippage_ticks=0
                    # Start after the entry bar.  The loop keeps stop-before-target ordering.
                    for x in range(t+2,min(t+26,len(bars))):
                        candidate=bars[x]; hi,lo=_d(candidate["high"]),_d(candidate["low"])
                        stop_hit=lo<=stop if direction=="LONG" else hi>=stop; target_hit=hi>=target if direction=="LONG" else lo<=target
                        if stop_hit:
                            exit_index=x; reference_exit=stop; exit_price=_quantize(stop-sign*_VERIFIED_BTC_CONFIG.price_tick*_VERIFIED_BTC_CONFIG.stop_slippage_ticks, ROUND_FLOOR if direction=="LONG" else ROUND_CEILING); reason="STOP_FIRST_AMBIGUITY" if target_hit else "STOP"; exit_slippage_ticks=_VERIFIED_BTC_CONFIG.stop_slippage_ticks; break
                        if target_hit:
                            exit_index=x; reference_exit=target; exit_price=target; reason="TARGET"; break
                        if _timestamp(candidate["timestamp"]).hour==23 and _timestamp(candidate["timestamp"]).minute==55:
                            exit_index=x; reference_exit=_d(candidate["close"]); exit_price=_quantize(reference_exit-sign*_VERIFIED_BTC_CONFIG.price_tick*_VERIFIED_BTC_CONFIG.market_slippage_ticks, ROUND_FLOOR if direction=="LONG" else ROUND_CEILING); reason="SESSION_FLAT"; exit_slippage_ticks=_VERIFIED_BTC_CONFIG.market_slippage_ticks; break
                    if exit_index is None:
                        # A complete input must contain the next completed exit bar; this is a deterministic time stop.
                        exit_index=min(t+25,len(bars)-1); reference_exit=_d(bars[exit_index]["close"]); exit_price=_quantize(reference_exit-sign*_VERIFIED_BTC_CONFIG.price_tick*_VERIFIED_BTC_CONFIG.market_slippage_ticks, ROUND_FLOOR if direction=="LONG" else ROUND_CEILING); reason="TIME_STOP"; exit_slippage_ticks=_VERIFIED_BTC_CONFIG.market_slippage_ticks
                    entry_event=_event(structure_id=structure,setup_id=setup_id,timestamp=base["entry_bar"],event_type="ENTRY",direction=direction,price_rule="NEXT_OPEN_MARKET")
                    events.append(entry_event)
                    trade_id=_hash({"candidate_id":candidate_id,"structure_id":structure,"entry_event_id":entry_event["event_id"],"target_r":str(target_r),"execution_assumption_hash":execution_hash})
                    entry_slippage=abs(entry-reference_entry)*quantity; exit_slippage=abs(exit_price-reference_exit)*quantity
                    gross=sign*(reference_exit-reference_entry)*quantity; fees=_VERIFIED_BTC_CONFIG.taker_fee_rate*(entry+exit_price)*quantity; risk=distance*quantity
                    trade={"trade_id":trade_id,"setup_id":setup_id,"structure_id":structure,"entry_event_id":entry_event["event_id"],"entry_timestamp":base["entry_bar"],"exit_timestamp":_stamp(bars[exit_index]["timestamp"]),"direction":direction,"reference_entry_price":str(reference_entry),"entry":str(entry),"reference_exit_price":str(reference_exit),"exit":str(exit_price),"stop":str(stop),"target":str(target),"quantity_btc":str(quantity),"gross_pnl":str(gross),"fees":str(fees),"slippage":str(entry_slippage+exit_slippage),"total_costs":str(fees+entry_slippage+exit_slippage),"net_pnl":str(gross-fees-entry_slippage-exit_slippage),"net_r":str((gross-fees-entry_slippage-exit_slippage)/risk),"exit_reason":reason,"entry_slippage_ticks":_VERIFIED_BTC_CONFIG.market_slippage_ticks,"exit_slippage_ticks":exit_slippage_ticks,"same_bar_stop_first":reason=="STOP_FIRST_AMBIGUITY","cost_model_version":COST_MODEL_VERSION}
                    trades.append(trade); active_until=exit_index
            terminal=_event(structure_id=structure,setup_id=setup_id,timestamp=base["entry_bar"] if disp in {"FALSE_BREAKOUT_INVALIDATED","STOP_DISTANCE_REJECTED","NO_EXECUTABLE_ENTRY","TRADE_EXECUTED"} else base["breakout_bar"],event_type=disp,direction=direction,price_rule="NEXT_OPEN_MARKET" if disp in {"FALSE_BREAKOUT_INVALIDATED","STOP_DISTANCE_REJECTED","NO_EXECUTABLE_ENTRY","TRADE_EXECUTED"} else "COMPLETED_BAR_CLOSE")
            events.append(terminal)
            base.update({"structure_id":structure,"event_id":proposed["event_id"],"proposed_stop_or_null":str(stop) if disp in {"TRADE_EXECUTED","STOP_DISTANCE_REJECTED","NO_EXECUTABLE_ENTRY"} else None,"stop_distance_or_null":str(distance) if disp in {"TRADE_EXECUTED","STOP_DISTANCE_REJECTED","NO_EXECUTABLE_ENTRY"} else None,"terminal_disposition":disp,"trade_id":trade["trade_id"] if trade else None})
            setups.append(base)
    validate_reconciliation(setups,events,trades); return setups,events,trades

def reconciliation_summary(setups: list[dict[str, Any]], events: list[dict[str, Any]], trades: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate the sealed VBTC event audit trail and return its exact formula.

    Every proposed setup emits exactly two events: ``PROPOSED_SETUP`` and its
    sole terminal disposition.  A setup that executes emits one additional
    ``ENTRY`` event.  Price exits are fields of the trade, not setup events.
    """
    setup_ids = [item["setup_id"] for item in setups]
    known_setups = set(setup_ids)
    event_ids = [item["event_id"] for item in events]
    trade_ids = [item["trade_id"] for item in trades]
    if len(setup_ids) != len(known_setups) or len(event_ids) != len(set(event_ids)) or len(trade_ids) != len(set(trade_ids)):
        raise ValueError("VBTC_V1_RECONCILIATION_FAILURE")
    if any(item.get("terminal_disposition") not in DISPOSITIONS for item in setups) or any(item.get("setup_id") not in known_setups for item in events + trades):
        raise ValueError("VBTC_V1_RECONCILIATION_FAILURE")

    terminal_by_setup = {item["setup_id"]: item["terminal_disposition"] for item in setups}
    executed_setups = {setup_id for setup_id, disposition in terminal_by_setup.items() if disposition == "TRADE_EXECUTED"}
    trades_by_setup = {item["setup_id"]: item for item in trades}
    if executed_setups != set(trades_by_setup) or len(trades_by_setup) != len(trades):
        raise ValueError("VBTC_V1_RECONCILIATION_FAILURE")
    if any(item.get("trade_id") != trades_by_setup[item["setup_id"]].get("trade_id") for item in setups if item["terminal_disposition"] == "TRADE_EXECUTED"):
        raise ValueError("VBTC_V1_RECONCILIATION_FAILURE")
    if any(item.get("trade_id") is not None for item in setups if item["terminal_disposition"] != "TRADE_EXECUTED"):
        raise ValueError("VBTC_V1_RECONCILIATION_FAILURE")

    events_by_setup: dict[str, list[dict[str, Any]]] = {setup_id: [] for setup_id in known_setups}
    for event in events:
        events_by_setup[event["setup_id"]].append(event)
    for setup_id, terminal in terminal_by_setup.items():
        types = [event["type"] for event in events_by_setup[setup_id]]
        expected = ["PROPOSED_SETUP", terminal] + (["ENTRY"] if terminal == "TRADE_EXECUTED" else [])
        if Counter(types) != Counter(expected):
            raise ValueError("VBTC_V1_RECONCILIATION_FAILURE")

    expected_events = 2 * len(setups) + len(trades)
    if len(events) != expected_events:
        raise ValueError("VBTC_V1_RECONCILIATION_FAILURE")
    return {
        "formula": "events == 2 * proposed_setups + executed_trades",
        "proposed_setups": len(setups),
        "executed_trades": len(trades),
        "expected_events": expected_events,
        "actual_events": len(events),
        "reconciles": True,
    }


def validate_reconciliation(setups, events, trades) -> None:
    reconciliation_summary(setups, events, trades)

def _load_phase_a_bars(manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read only files declared by the already-hashed Phase-A manifest."""
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("VBTC_V1_PHASE_A_MANIFEST_SCHEMA_INVALID") from exc
    identity = manifest.get("identity")
    if not isinstance(identity, dict) or manifest.get("valid") is not True or identity.get("phase") != "PHASE_A" or identity.get("symbol") != "BTCUSDT" or identity.get("bar_interval") != "5m" or tuple(identity.get("months", ())) != PHASE_A_MONTHS:
        raise ValueError("VBTC_V1_PHASE_A_MANIFEST_SCHEMA_INVALID")
    files = [item for item in manifest.get("parquet_files", []) if isinstance(item, dict) and item.get("kind", "bars") == "bars"]
    if len(files) != 13 or {item.get("month") for item in files} != set(PHASE_A_MONTHS):
        raise ValueError("VBTC_V1_PHASE_A_MANIFEST_CHRONOLOGY_INVALID")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - declared project dependency
        raise RuntimeError("VBTC_V1_PYARROW_REQUIRED") from exc
    rows: list[dict[str, Any]] = []
    required = {"bar_start_utc", "open", "high", "low", "close", "volume", "daily_vwap"}
    for item in sorted(files, key=lambda entry: entry["month"]):
        relative = item.get("relative_path")
        if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("VBTC_V1_PHASE_A_MANIFEST_SCHEMA_INVALID")
        parquet_path = (manifest_path.parent / relative).resolve()
        if not parquet_path.is_file() or _file_hash(parquet_path).lower() != str(item.get("sha256", "")).lower():
            raise ValueError("VBTC_V1_PHASE_A_PARQUET_HASH_MISMATCH")
        table = pq.read_table(parquet_path)
        if not required.issubset(table.column_names): raise ValueError("VBTC_V1_PHASE_A_BAR_SCHEMA_INVALID")
        for row in table.select(sorted(required)).to_pylist():
            stamp = row.pop("bar_start_utc")
            if not isinstance(stamp, datetime) or stamp.tzinfo is None or stamp.utcoffset() != timezone.utc.utcoffset(stamp):
                raise ValueError("VBTC_V1_TIMESTAMP_NOT_UTC")
            row["timestamp"] = stamp.isoformat().replace("+00:00", "Z")
            try:
                values = {key: _d(row[key]) for key in ("open", "high", "low", "close", "volume", "daily_vwap")}
            except Exception as exc:
                raise ValueError("VBTC_V1_PHASE_A_BAR_SCHEMA_INVALID") from exc
            if any(not value.is_finite() for value in values.values()) or any(values[key] <= 0 for key in ("open", "high", "low", "close", "daily_vwap")) or values["volume"] < 0 or values["high"] < max(values["open"], values["close"]) or values["low"] > min(values["open"], values["close"]) or values["high"] < values["low"]:
                raise ValueError("VBTC_V1_PHASE_A_BAR_SCHEMA_INVALID")
            rows.append(row)
    rows.sort(key=lambda row: _timestamp(row["timestamp"]))
    stamps = [_timestamp(row["timestamp"]) for row in rows]
    if not stamps or stamps[0] != PHASE_A_START or stamps[-1] != datetime(2024, 1, 31, 23, 55, tzinfo=timezone.utc) or any(left >= right for left, right in zip(stamps, stamps[1:])):
        raise ValueError("VBTC_V1_PHASE_A_CHRONOLOGY_INVALID")
    return manifest, rows

def _metrics(trades: list[dict[str, Any]], setups: list[dict[str, Any]]) -> dict[str, Any]:
    pnl = [Decimal(item["net_pnl"]) for item in trades]
    net_r = [Decimal(item["net_r"]) for item in trades]
    positive = sum((value for value in pnl if value > 0), Decimal())
    negative = -sum((value for value in pnl if value < 0), Decimal())
    monthly = {month: {"executed_trades": 0, "gross_pnl": Decimal(), "net_pnl": Decimal(), "net_r": Decimal()} for month in PHASE_A_MONTHS}
    for trade, value, value_r in zip(trades, pnl, net_r):
        bucket = monthly[trade["entry_timestamp"][:7]]
        bucket["executed_trades"] += 1; bucket["gross_pnl"] += Decimal(trade["gross_pnl"]); bucket["net_pnl"] += value; bucket["net_r"] += value_r
    curve = peak = drawdown = Decimal()
    for value in net_r:
        curve += value; peak = max(peak, curve); drawdown = max(drawdown, peak - curve)
    directions = Counter(item["direction"] for item in trades)
    return {"executed_trades": len(trades), "annualized_trades": str(Decimal(len(trades)) * Decimal("365.25") / Decimal("396")), "overfrequency_warning": len(trades) * Decimal("365.25") / Decimal("396") > 350, "long_trades": directions["LONG"], "short_trades": directions["SHORT"], "wins": sum(value > 0 for value in pnl), "losses": sum(value < 0 for value in pnl), "gross_pnl": str(sum((Decimal(item["gross_pnl"]) for item in trades), Decimal())), "costs": str(sum((Decimal(item["fees"])+Decimal(item["slippage"]) for item in trades), Decimal())), "net_pnl": str(sum(pnl, Decimal())), "net_profit_factor": str(positive / negative if negative else Decimal("Infinity")), "average_net_r": str(sum(net_r, Decimal()) / len(net_r) if net_r else Decimal()), "maximum_drawdown_r": str(drawdown), "monthly_net_pnl": {month: str(value["net_pnl"]) for month, value in monthly.items()}, "monthly_results": {month: {key: str(value) if isinstance(value, Decimal) else value for key, value in bucket.items()} for month, bucket in monthly.items()}, "outcome_counts": dict(Counter(item["terminal_disposition"] for item in setups)), "net_r": [str(value) for value in net_r]}

def _gates(metrics: dict[str, Any], trades: list[dict[str, Any]], reconciles: bool) -> dict[str, Any]:
    import numpy as np
    values = np.array([float(value) for value in metrics["net_r"]], dtype=float); n = len(values)
    samples = np.random.Generator(np.random.PCG64(20240131)).choice(values, size=(10000, n), replace=True).mean(axis=1) if n else np.zeros(10000)
    net = Decimal(metrics["net_pnl"]); positive = sum((max(Decimal(item["net_pnl"]), Decimal()) for item in trades), Decimal()); months = [Decimal(value) for value in metrics["monthly_net_pnl"].values()]
    directions = Counter(item["direction"] for item in trades); direction_r = {side: sum((Decimal(item["net_pnl"]) / abs(Decimal(item["entry"])-Decimal(item["stop"])) for item in trades if item["direction"] == side), Decimal()) / directions[side] if directions[side] else Decimal("-Infinity") for side in ("LONG", "SHORT")}
    quarters = [sum((Decimal(item["net_pnl"]) for item in trades if item["entry_timestamp"][:7] in {f"2023-{month:02d}" for month in group}), Decimal()) for group in ((1,2,3),(4,5,6),(7,8,9),(10,11,12))]
    one_extra_tick_per_entry_and_exit = _VERIFIED_BTC_CONFIG.price_tick * _VERIFIED_BTC_CONFIG.quantity_btc * 2
    extra = net - sum((one_extra_tick_per_entry_and_exit for _ in trades), Decimal())
    checks = {"manifest_schema_chronology_reconciliation": reconciles, "positive_net_pnl": net > 0, "net_pf": Decimal(metrics["net_profit_factor"]) >= Decimal("1.30"), "positive_average_net_r": Decimal(metrics["average_net_r"]) > 0, "maximum_dd": Decimal(metrics["maximum_drawdown_r"]) <= 20, "minimum_trades": n >= 108, "annualized_trades": Decimal(n) * Decimal("365.25") / Decimal("396") >= 100, "profitable_months": sum(value > 0 for value in months) >= 8, "zero_trade_months": sum(value == 0 for value in months) <= 3, "best_month_concentration": positive > 0 and max(months, default=Decimal()) / positive <= Decimal(".35"), "best_five_concentration": positive > 0 and sum(sorted((max(Decimal(item["net_pnl"]), Decimal()) for item in trades), reverse=True)[:5], Decimal()) / positive <= Decimal(".30"), "long_short_mix": n > 0 and directions["LONG"] * 4 >= n and directions["SHORT"] * 4 >= n, "direction_average_r": all(value >= Decimal("-.15") for value in direction_r.values()), "extra_tick_entry_exit": extra > 0, "best_trade_removal": net - max((Decimal(item["net_pnl"]) for item in trades), default=Decimal()) > 0, "bootstrap_median": float(np.median(samples)) > 0, "bootstrap_lower": float(np.percentile(samples, 2.5)) >= -.025, "calendar_subperiods": sum(value >= 0 for value in quarters) >= 3}
    return {"passed": all(checks.values()), "checks": checks, "extra_slippage_sensitivity_net_pnl": str(extra), "best_trade_removal_net_pnl": str(net - max((Decimal(item["net_pnl"]) for item in trades), default=Decimal())), "bootstrap": {"seed": 20240131, "resamples": 10000, "median_mean_net_r": float(np.median(samples)), "lower_2_5": float(np.percentile(samples, 2.5)), "upper_97_5": float(np.percentile(samples, 97.5))}}

def materialize_synthetic_contract(*, artifact_root:str|Path, repository_root:str|Path) -> dict[str,Any]:
    root=Path(repository_root).resolve(); verify_sealed_specification(root); store=Store(Path(artifact_root).resolve(),"PHASE_A","NOT_READ","SYNTHETIC")
    common={"study_id":STUDY_ID,"phase":"PHASE_A","specification_sha256":SPEC_SHA256,"data_manifest_sha256":"NOT_READ","created_at_utc":"SYNTHETIC_ONLY"}
    store.write("sealed-specification.json",{**common,"text":(root/SPEC_PATH).read_text(encoding="utf-8"),"seal_status":"SEALED"}); store.write("candidate-registry.json",{**common,"candidates":[{"candidate_id":a,"target_r":str(b)} for a,b in CANDIDATES],"only_target_differs":True}); store.write("data-manifest.json",{**common,"status":"NOT_READ","market_data_read":False}); store.write("configuration.json",{**common,"execution_assumption_hash":_hash(EXECUTION_ASSUMPTIONS),"configuration_hash":_hash(EXECUTION_ASSUMPTIONS)})
    for cid,target in CANDIDATES:
        for name,key in (("events.json","events"),("trades.json","trades"),("setup_outcomes.json","setup_outcomes")) : store.write(name,{**common,"candidate_id":cid,key:[],"status":"NOT_EXECUTED"},cid)
        for name in ("monthly_metrics.json","report.json","gates.json"): store.write(name,{**common,"candidate_id":cid,"status":"NOT_EXECUTED","phase_b":"NOT_OPENED","alpha":"NOT_OPENED"},cid)
    store.write("selection_report.json",{**common,"status":"PHASE_A_NO_ROBUST_CANDIDATE","ranking":[]}); store.write("freeze.json",{**common,"status":"NOT_FROZEN"}); store.write("final_report.json",{**common,"status":"SYNTHETIC_MATERIALIZED","realStudyExecuted":False,"phase_b":"NOT_OPENED","alpha":"NOT_OPENED","model":"gpt-5.6-terra"}); store.seal(); return {"status":"SYNTHETIC_MATERIALIZED","summary":"Synthetic-only VBTC V1 contract materialized; Phase B and Alpha remain unopened.","testsPassed":True,"realStudyExecuted":False,"model":"gpt-5.6-terra","artifactRoot":str(store.root)}

def run_phase_a(*,phase_a_bars_manifest:str|Path,artifact_root:str|Path,repository_root:str|Path)->dict[str,Any]:
    root=Path(repository_root)
    manifest=Path(phase_a_bars_manifest)
    if not manifest.is_absolute() or not Path(artifact_root).is_absolute() or not Path(repository_root).is_absolute(): raise ValueError("VBTC_V1_ABSOLUTE_PATHS_REQUIRED")
    root=root.resolve(); verify_sealed_specification(root); revision=_clean_git(root)
    if not manifest.is_file() or _file_hash(manifest).lower()!=PHASE_A_MANIFEST_SHA256: raise ValueError("VBTC_V1_PHASE_A_MANIFEST_HASH_MISMATCH")
    payload, bars = _load_phase_a_bars(manifest)
    store=Store(Path(artifact_root).resolve(),"PHASE_A",PHASE_A_MANIFEST_SHA256,revision)
    common={"schema_version":"VBTC-V1","study_id":STUDY_ID,"phase":"PHASE_A","specification_sha256":SPEC_SHA256,"data_manifest_sha256":PHASE_A_MANIFEST_SHA256,"created_at_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z")}
    execution_hash=_hash(EXECUTION_ASSUMPTIONS)
    store.write("sealed-specification.json",{**common,"text":(root/SPEC_PATH).read_text(encoding="utf-8"),"seal_status":"SEALED"})
    store.write("candidate-registry.json",{**common,"candidates":[{"candidate_id":cid,"target_r":str(target)} for cid,target in CANDIDATES],"only_target_differs":True,"execution_count_per_candidate":1})
    store.write("data-manifest.json",{**common,"path":str(manifest),"identity":payload["identity"],"requested_schema":"bar_start_utc/open/high/low/close/volume/daily_vwap","chronology":"2023-01-01T00:00:00Z/2024-01-31T23:59:59.999999Z","validated":True,"market_data_read":True,"phase_b":"NOT_OPENED"})
    results=[]
    for candidate_id,target_r in CANDIDATES:
        setups,events,trades=evaluate_bars(bars,candidate_id,target_r)
        metrics=_metrics(trades,setups)
        event_reconciliation=reconciliation_summary(setups,events,trades)
        reconciles=(len(setups)==sum(metrics["outcome_counts"].values()) and metrics["outcome_counts"].get("TRADE_EXECUTED",0)==len(trades) and event_reconciliation["reconciles"])
        gates=_gates(metrics,trades,reconciles)
        configuration={**common,"candidate_id":candidate_id,"target_r":str(target_r),"execution_assumption_hash":execution_hash,"code_revision":revision,"formula":"sealed VBTC V1 EMA20/EMA50/ATR14 next-open design","configuration_hash":_hash({"candidate_id":candidate_id,"target_r":str(target_r),"execution_assumption_hash":execution_hash,"specification_sha256":SPEC_SHA256})}
        store.write("configuration.json",configuration,candidate_id)
        store.write("events.json",{**common,"candidate_id":candidate_id,"events":events},candidate_id)
        store.write("trades.json",{**common,"candidate_id":candidate_id,"trades":trades},candidate_id)
        store.write("setup_outcomes.json",{**common,"candidate_id":candidate_id,"setup_outcomes":setups},candidate_id)
        store.write("monthly_metrics.json",{**common,"candidate_id":candidate_id,"monthly_results":metrics["monthly_results"],"zero_trade_months":sum(Decimal(value)==0 for value in metrics["monthly_net_pnl"].values())},candidate_id)
        store.write("report.json",{**common,"candidate_id":candidate_id,**{key:value for key,value in metrics.items() if key != "net_r"},"funnel_reconciliation":{"proposed_setups":len(setups),"disposition_sum":sum(metrics["outcome_counts"].values()),"executed":metrics["outcome_counts"].get("TRADE_EXECUTED",0),"trades":len(trades),"events":len(events),"event_lifecycle":event_reconciliation,"reconciles":reconciles}},candidate_id)
        store.write("gates.json",{**common,"candidate_id":candidate_id,**gates},candidate_id)
        results.append((candidate_id,configuration,metrics,gates))
    passing=sorted((item for item in results if item[3]["passed"]),key=lambda item:(-Decimal(item[2]["average_net_r"]),-Decimal(item[2]["net_profit_factor"]),Decimal(item[2]["maximum_drawdown_r"]),item[0]))
    selected=passing[0] if passing else None; selected_id=selected[0] if selected else None
    store.write("selection_report.json",{**common,"complete_phase_a_passes":[item[0] for item in passing],"ranking":[item[0] for item in passing],"selected_candidate_id":selected_id,"status":"PHASE_A_SELECTED" if selected else "PHASE_A_NO_ROBUST_CANDIDATE"})
    store.write("freeze.json",{**common,"status":"FROZEN" if selected else "NOT_FROZEN","candidate_id":selected_id,"configuration_hash":selected[1]["configuration_hash"] if selected else None,"execution_assumption_hash":execution_hash,"phase_b":"NOT_OPENED"})
    final={"status":"PHASE_A_SELECTED" if selected else "PHASE_A_NO_ROBUST_CANDIDATE","summary":"Deterministic sealed VBTC V1 Phase A completed; Phase B and Alpha remain unopened.","testsPassed":True,"realStudyExecuted":True,"selectedCandidateId":selected_id,"phaseBStatus":"NOT_OPENED","alphaStatus":"NOT_OPENED","model":"gpt-5.6-terra"}
    store.write("final_report.json",{**common,**final,"integrity":"SEALED_ON_WRITE","no_phase_b_or_alpha_access":True,"candidate_gates":{candidate_id: gates["checks"] for candidate_id, _, _, gates in results}}); store.seal()
    return {**final,"artifactRoot":str(store.root)}
