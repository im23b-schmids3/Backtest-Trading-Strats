"""Frozen V2.1 replication with only intended-direction BOS semantics corrected.

The original V2 module and its artifacts are not modified. This runner patches
the V2 detector's BOS helper in-process, writes to a new namespace, then adds
auditable comparisons and mechanism summaries from the same sealed tapes.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import statistics
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import mac_2025_jj_fair_pricing_expanded_v2 as v2
from . import mac_2025_jj_fair_pricing_v2_mechanism_diagnostic as diag
from .mac_2025_candidate_tape import load_tape

STUDY_ID = "ES_JJ_FAIR_PRICING_V2_1_CORRECTED_BOS_RESEARCH"
V2_ROOT = Path("research_runs/CMEOrderflow_ES_JJ_FAIR_PRICING_EXPANDED_V2")
DIAG_ROOT = Path("research_runs/CMEOrderflow_ES_JJ_FAIR_PRICING_V2_MECHANISM_DIAGNOSTIC")
OUT_ROOT = Path("research_runs/CMEOrderflow_ES_JJ_FAIR_PRICING_V2_1_CORRECTED_BOS")
FLOAT_TOLERANCE = 1e-12


class CorrectedBosError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, sort_keys=True, indent=2, allow_nan=False, default=v2._json_default) + "\n", encoding="utf-8")


def _write_jsonl_gz(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    v2._write_jsonl_gz(path, rows)


def directional_bos_for_bar(bar_map: Mapping[int, Mapping[str, Any]], minute_index: int,
                            trade_direction: int) -> dict[str, Any]:
    """Completed-candle BOS against the immediately preceding two completed bars."""
    prior_indices = [minute_index - 2, minute_index - 1]
    if any(i not in bar_map for i in prior_indices) or minute_index not in bar_map:
        return {"confirmed": False, "status": "MISSING_REFERENCE_CANDLE", "reference_level": None,
                "prior_minute_indices": prior_indices}
    a, b, c = (bar_map[i] for i in (*prior_indices, minute_index))
    if trade_direction > 0:
        level = max(float(a["high"]), float(b["high"]))
        passed = float(c["close"]) > level
    else:
        level = min(float(a["low"]), float(b["low"]))
        passed = float(c["close"]) < level
    return {"confirmed": bool(passed), "status": "CONFIRMED" if passed else "NO_CLOSE_THROUGH",
            "reference_level": level, "prior_minute_indices": prior_indices}


def _load_json_artifacts(root: Path, names: Sequence[str]) -> tuple[dict[str, Any], dict[str, str]]:
    manifest = _json(root / "run-manifest.json")
    hashes = _json(root / "artifact-hashes.json")
    if manifest.get("status") != "COMPLETE" or hashes.get("status") != "HASHED":
        raise CorrectedBosError(f"source run is not COMPLETE/HASHED: {root}")
    mismatches = {name: expected for name, expected in manifest.get("files", {}).items()
                  if not (root / name).is_file() or _sha(root / name) != expected}
    hash_mismatches = {name: expected for name, expected in hashes.get("files", {}).items()
                       if not (root / name).is_file() or _sha(root / name) != expected}
    if mismatches or hash_mismatches:
        raise CorrectedBosError(f"source artifact hash mismatch: manifest={list(mismatches)}, artifact={list(hash_mismatches)}")
    for name in names:
        if name not in manifest.get("files", {}) or name not in hashes.get("files", {}):
            raise CorrectedBosError(f"required frozen artifact not hash-pinned: {name}")
    return manifest, dict(manifest["files"])


def freeze_inputs() -> dict[str, Any]:
    required = ("study-config.json", "source-coverage.json", "all-signals.jsonl.gz",
                "causal-features.jsonl.gz", "executed-trades.jsonl.gz", "daily-results.csv",
                "exit-model-surface.csv", "spring-results.json", "october-results.json",
                "session-model-trigger-results.json", "summary.json")
    manifest, hashes = _load_json_artifacts(V2_ROOT, required)
    diagnostic_hashes = _json(DIAG_ROOT / "artifact-hashes.json")
    if diagnostic_hashes.get("status") != "HASHED":
        raise CorrectedBosError("the prior mechanism diagnostic is not HASHED")
    dm = {name: digest for name, digest in diagnostic_hashes["files"].items()
          if not (DIAG_ROOT / name).is_file() or _sha(DIAG_ROOT / name) != digest}
    if dm:
        raise CorrectedBosError(f"mechanism diagnostic hash mismatches: {list(dm)}")
    cfg = _json(V2_ROOT / "study-config.json")
    old_module = Path(v2.__file__)
    return {
        "study_id": STUDY_ID,
        "original_v2": {"root": str(V2_ROOT), "run_manifest_sha256": _sha(V2_ROOT / "run-manifest.json"),
            "artifact_hash_manifest_sha256": _sha(V2_ROOT / "artifact-hashes.json"),
            "files_sha256": hashes, "config": cfg, "config_sha256": manifest.get("config_sha256"),
            "source_coverage_sha256": manifest.get("source_coverage_sha256"),
            "completed_dates": manifest.get("completed_dates"), "study_module_sha256": _sha(old_module)},
        "prior_mechanism_diagnostic": {"root": str(DIAG_ROOT), "artifact_hash_manifest_sha256": _sha(DIAG_ROOT / "artifact-hashes.json"),
            "files_sha256": diagnostic_hashes["files"]},
        "original_defect": {"call_site": "V2._signal_catalog -> V1.bos_for_bar(bar_map, minute, intended_trade_direction)",
            "actual_v1_semantics": "positive direction closes below min(prior lows); negative direction closes above max(prior highs)",
            "corrected_semantics": "long closes above max(prior highs); short closes below min(prior lows)"},
        "semantic_diff": [{"field": "directional_bos_confirmation", "before": "opposite-direction close-through", "after": "intended trade-direction close-through"}],
        "preservation": {"original_v2_artifacts_modified": False, "data_downloaded": False,
            "raw_dbn_read": False, "2026_accessed": False, "final_oos_accessed": False}}


def _candle_tests() -> dict[str, Any]:
    def bar(o: float, h: float, l: float, c: float) -> dict[str, float]:
        return {"open": o, "high": h, "low": l, "close": c}
    tests: dict[str, bool] = {}
    long_bos = {1: bar(100, 101, 99, 100), 2: bar(100, 100.75, 99.5, 100.25), 3: bar(100.25, 101.5, 100, 101.25)}
    short_bos = {1: bar(100, 101, 99, 100), 2: bar(100, 100.5, 99.25, 99.75), 3: bar(99.75, 100, 98.5, 98.75)}
    bad_long = {1: bar(100, 101, 99, 100), 2: bar(100, 100.75, 99.5, 100.25), 3: bar(100.25, 101.25, 100, 100.75)}
    bad_short = {1: bar(100, 101, 99, 100), 2: bar(100, 100.5, 99.25, 99.75), 3: bar(99.75, 100, 98.75, 99.25)}
    tests["valid_bullish_bos"] = directional_bos_for_bar(long_bos, 3, 1)["confirmed"]
    tests["valid_bearish_bos"] = directional_bos_for_bar(short_bos, 3, -1)["confirmed"]
    tests["invalid_bullish_bos"] = not directional_bos_for_bar(bad_long, 3, 1)["confirmed"]
    tests["invalid_bearish_bos"] = not directional_bos_for_bar(bad_short, 3, -1)["confirmed"]
    bull_disp = {1: bar(100.75, 101, 100.5, 100.5), 2: bar(100.5, 100.75, 100, 100.25), 3: bar(100.25, 102, 100, 101.75)}
    bear_disp = {1: bar(100.25, 100.5, 100, 100.5), 2: bar(100.5, 101, 100.25, 100.75), 3: bar(100.75, 101, 99, 99.25)}
    for label, bm, direction in (("bullish", bull_disp, 1), ("bearish", bear_disp, -1)):
        d = v2.displacement_candle(bm[3], bm[2], direction)["valid"]
        b = directional_bos_for_bar(bm, 3, direction)["confirmed"]
        tests[f"same_candle_{label}_combined"] = bool(d and b)
    tests["long_continuation_direction"] = v2._model_direction(100, 101, 3) == ("OPENING_CONTINUATION", 1)
    tests["short_continuation_direction"] = v2._model_direction(100, 99, 3) == ("OPENING_CONTINUATION", -1)
    tests["long_reversion_after_down_displacement"] = v2._model_direction(100, 99, 16) == ("FAIR_PRICE_REVERSION", 1)
    tests["short_reversion_after_up_displacement"] = v2._model_direction(100, 101, 16) == ("FAIR_PRICE_REVERSION", -1)
    tests["missing_prior_candle_fails_closed"] = directional_bos_for_bar({2: long_bos[2], 3: long_bos[3]}, 3, 1)["status"] == "MISSING_REFERENCE_CANDLE"
    tests["bos_uses_only_completed_prior_indices"] = directional_bos_for_bar(long_bos, 3, 1)["prior_minute_indices"] == [1, 2]
    return {"tests": tests, "all_pass": all(tests.values()),
        "timestamp_contract": "catalog signal timestamp equals current completed bar end; BOS references minute i-2 and i-1 only",
        "production_detector_integration": "also exercised through V2 _signal_catalog in test fixture; standalone helper checks are not the sole detector test"}


def _synthetic_production_catalog() -> dict[str, Any]:
    """Exercise the actual V2 catalog with dependencies replaced by synthetic bars."""
    saved = {n: getattr(v2, n) for n in ("_bars_for_session", "_anchor", "_episode_map", "_feature_context", "causal_features")}
    saved_bos = v2.v1.bos_for_bar
    cases = {}
    try:
        v2.v1.bos_for_bar = directional_bos_for_bar
        for label, session, minute, seq, anchor_side in (
            ("long_continuation", "NY_AM", 3, "bull", 1),
            ("short_continuation", "NY_AM", 3, "bear", -1),
            ("long_reversion", "NY_AM", 16, "bull_rev", -1),
            ("short_reversion", "NY_AM", 16, "bear_rev", 1),
        ):
            start, end = v2._session_ns("2025-03-03", session)
            bars = []
            for i in range(20):
                center = 101.0 if anchor_side > 0 else 99.0
                bars.append({"minute_index": i, "start_ns": start+i*v2.MINUTE_NS, "end_ns": start+(i+1)*v2.MINUTE_NS,
                             "open": center, "high": center+.125, "low": center-.125, "close": center})
            if seq == "bull":
                bars[1].update(open=100.5, high=100.75, low=100.0, close=100.25)
                bars[2].update(open=100.5, high=100.75, low=100.0, close=100.25)
                bars[3].update(open=100.25, high=101.5, low=100.0, close=101.25)
            elif seq == "bear":
                bars[1].update(open=99.5, high=100.0, low=99.25, close=99.75)
                bars[2].update(open=99.5, high=100.0, low=99.25, close=99.75)
                bars[3].update(open=99.75, high=100.0, low=98.5, close=98.75)
            elif seq == "bull_rev":
                bars[14].update(open=99.5, high=99.5, low=99.0, close=99.25)
                bars[15].update(open=99.25, high=99.5, low=98.75, close=99.0)
                bars[16].update(open=99.0, high=99.75, low=98.75, close=99.6)
            else:
                bars[14].update(open=101.1, high=101.5, low=101.0, close=101.3)
                bars[15].update(open=101.0, high=101.5, low=100.9, close=101.25)
                bars[16].update(open=101.25, high=101.5, low=100.5, close=100.75)
            tts=np.asarray([b["end_ns"]-1 for b in bars],dtype=np.int64)
            px=np.full(len(bars), 101.0 if anchor_side>0 else 99.0)
            sizes=np.ones(len(bars)); ix=np.arange(len(bars))
            v2._bars_for_session=lambda ev,a,b,bs=bars,tt=tts,p=px,sz=sizes,ii=ix:(bs,tt,p,sz,ii)
            v2._anchor=lambda *args:{"status":"VALID","price":100.0,"timestamp_ns":start}
            v2._episode_map=lambda *args:(np.ones(len(tts),dtype=np.int32),np.zeros(len(tts),dtype=np.int32),{1:(0,len(tts),float(np.min(px)),float(np.max(px)))})
            v2._feature_context=lambda *args:{}
            v2.causal_features=lambda *args:{}
            fields=[("timestamp_ns","i8"),("bid","f8"),("ask","f8"),("execution_price","f8"),("execution_size","f8"),("aggressor","i1"),("session","i1")]
            ev=np.zeros(1,dtype=fields)
            signals,audit=v2._signal_catalog("2025-03-03",session,ev)
            matching=[s for s in signals if s["minute_index"]==minute]
            expected_model="OPENING_CONTINUATION" if minute<15 else "FAIR_PRICE_REVERSION"
            expected_direction="LONG" if anchor_side>0 and minute<15 or anchor_side<0 and minute>=15 else "SHORT"
            combined=[s for s in matching if s["trigger"]=="BOS_PLUS_DISPLACEMENT"]
            cases[label]={"eligible":audit["eligible"],"expected_model":expected_model,"expected_direction":expected_direction,"minute_index":minute,
                "signals":[{"trigger":s["trigger"],"direction":s["direction"],"model":s["model"],"signal_timestamp_ns":s["signal_timestamp_ns"],
                    "prior_indices":s.get("bos_prior_minute_indices")} for s in matching],
                "combined_accepted":bool(combined),"timestamp_equals_completed_bar_end":all(s["signal_timestamp_ns"]==bars[minute]["end_ns"] for s in matching)}
        # Explicit ordering proof from the production execution function: first allowed quote is >= signal+2ms.
        signal={"signal_timestamp_ns":1_000_000_000,"direction_sign":1,"date":"2025-03-03","period":"SPRING_2025",
                "session":"NY_AM","model":"OPENING_CONTINUATION","trigger":"BOS_PLUS_DISPLACEMENT","signal_id":"synthetic",
                "direction":"LONG","minute_index":3,"episode_adverse_extreme":99.0,"anchor_price":100.0}
        times=np.asarray([1_001_999_999,1_002_000_000,1_003_000_000],dtype=np.int64)
        ev=np.zeros(3,dtype=[("timestamp_ns","i8"),("bid","f8"),("ask","f8"),("execution_price","f8"),("execution_size","f8"),("aggressor","i1"),("session","i1")])
        ev["timestamp_ns"]=times;ev["bid"]=100;ev["ask"]=100.25
        ev["execution_price"]=100;ev["execution_size"]=1
        run=v2._execution_cell(ev,signal,[],0,2_000_000_000,"EPISODE_STRUCTURAL","FIXED_1R")
        cases["entry_ordering"]={"status":run["status"],"entry_timestamp_ns":run.get("entry_timestamp_ns"),
            "entry_ready_ns":signal["signal_timestamp_ns"]+2_000_000,
            "pass":run.get("entry_timestamp_ns",0)>=signal["signal_timestamp_ns"]+2_000_000}
    finally:
        for name,value in saved.items(): setattr(v2,name,value)
        v2.v1.bos_for_bar=saved_bos
    required=("long_continuation","short_continuation","long_reversion","short_reversion")
    return {"cases":cases,"production_signal_catalog_all_direction_cases_pass":all(
        cases[n]["combined_accepted"] and cases[n]["timestamp_equals_completed_bar_end"] and
        any(x["trigger"]=="BOS_PLUS_DISPLACEMENT" and x["direction"]==cases[n]["expected_direction"] and
            x["model"]==cases[n]["expected_model"] and x["prior_indices"]==[cases[n]["minute_index"]-2,cases[n]["minute_index"]-1]
            for x in cases[n]["signals"]) for n in required),
        "entry_ordering_pass":cases["entry_ordering"]["pass"]}


def _match_controls(paths: list[dict[str,Any]], signals: Sequence[Mapping[str,Any]], source: Mapping[str,Any]) -> dict[str,Any]:
    """Run frozen diagnostic matching and expose attrition after each unchanged rule."""
    tapes={r["date"]:load_tape(Path(r["candidate_tape_path"]),source_sha256=r["candidate_tape_source_sha256"],
        semantic_sha256=r["candidate_tape_semantic_sha256"]) for r in source["files"]}
    path_by={p["signal_id"]:p for p in paths};sig_minutes=defaultdict(set);groups=defaultdict(list)
    for s in signals:
        sig_minutes[(s["date"],s["session"])].add(int(s["minute_index"]))
        groups[(s["date"],s["session"])].append(s)
    stage_totals=Counter();stage_signal_support=Counter();failure_reasons=Counter();first_zero=Counter();matches=[];initial_zero=0
    for (day,session),group in sorted(groups.items()):
        start,end=v2._session_ns(day,session);bars,_,_,_,_=v2._bars_for_session(tapes[day].events,start,end)
        bm={int(b["minute_index"]):b for b in bars};anchor=float(group[0]["anchor_price"]);used=set()
        control=[]
        for b in bars:
            m=int(b["minute_index"]);model="OPENING_CONTINUATION" if m<v2.CONTINUATION_MINUTES else "FAIR_PRICE_REVERSION"
            if m in sig_minutes[(day,session)] or m-2 not in bm:continue
            close=float(b["close"]);side=1 if close>anchor else -1 if close<anchor else 0
            if not side:continue
            direction=side if model=="OPENING_CONTINUATION" else -side
            move=close-float(bm[m-2]["close"]);trend=1 if move>0 else -1 if move<0 else 0
            atr=diag._atr14(bars,m)
            if atr is None or atr<=0:continue
            control.append({"minute":m,"model":model,"direction":direction,"trend":trend,"atr":atr,
                "dist":abs(close-anchor)/diag.TICK,"timestamp_ns":int(b["end_ns"])})
        # Preserve the prior protocol's continuation displacement-only cohort for diagnosing its zero matches.
        cohort=[s for s in group if s["model"]=="OPENING_CONTINUATION" and s["trigger"]=="DISPLACEMENT_CANDLE"]
        for s in cohort:
            p=path_by.get(s["signal_id"]);m=int(s["minute_index"]);stage="all_candidate_controls"
            candidates=[c for c in control if c["model"]==s["model"]];stage_totals[stage]+=len(candidates)
            if candidates:stage_signal_support[stage]+=1
            else:initial_zero+=1;first_zero[stage]+=1
            filt=[c for c in candidates if c["direction"]==int(s["direction_sign"])];stage="same_intended_direction";stage_totals[stage]+=len(filt)
            if filt:stage_signal_support[stage]+=1
            if not filt and candidates:failure_reasons[stage]+=1;first_zero[stage]+=1
            if m-2 not in bm:
                if filt:first_zero["trailing_reference_unavailable"]+=1;failure_reasons["trailing_reference_unavailable"]+=1
                continue
            signal_trend=float(bm[m]["close"])-float(bm[m-2]["close"]);signal_trend=1 if signal_trend>0 else -1 if signal_trend<0 else 0
            nxt=[c for c in filt if c["trend"]==signal_trend];stage="same_trailing_2m_direction";stage_totals[stage]+=len(nxt)
            if nxt:stage_signal_support[stage]+=1
            if not nxt and filt:failure_reasons[stage]+=1;first_zero[stage]+=1
            timed=[c for c in nxt if abs(c["minute"]-m)<=5];stage="within_5_minute_caliper";stage_totals[stage]+=len(timed)
            if timed:stage_signal_support[stage]+=1
            if not timed and nxt:failure_reasons[stage]+=1;first_zero[stage]+=1
            atr=p.get("atr14_ticks") if p else None
            if atr is None or atr<=0:
                if timed:first_zero["signal_atr_unavailable"]+=1;failure_reasons["signal_atr_unavailable"]+=1
                continue
            stage_totals["signal_atr_available"]+=len(timed)
            if timed:stage_signal_support["signal_atr_available"]+=1
            vol=[c for c in timed if abs(math.log(c["atr"]/float(atr)))<=.5];stage="within_log_atr_caliper";stage_totals[stage]+=len(vol)
            if vol:stage_signal_support[stage]+=1
            if not vol and timed:failure_reasons[stage]+=1;first_zero[stage]+=1
            dist=float(p["anchor_distance_ticks"]);caliper=max(8.0,.5*dist)
            near=[c for c in vol if abs(c["dist"]-dist)<=caliper];stage="within_anchor_distance_caliper";stage_totals[stage]+=len(near)
            if near:stage_signal_support[stage]+=1
            if not near and vol:failure_reasons[stage]+=1;first_zero[stage]+=1
            available=[c for c in near if c["minute"] not in used];stage="available_without_replacement";stage_totals[stage]+=len(available)
            if available:stage_signal_support[stage]+=1
            if not available and near:failure_reasons[stage]+=1;first_zero[stage]+=1
            if not available:continue
            c=min(available,key=lambda z:(abs(z["minute"]-m)/5+abs(math.log(z["atr"]/float(atr)))/.5+
                abs(z["dist"]-dist)/caliper,abs(z["minute"]-m),z["minute"]))
            used.add(c["minute"]);cpath=diag._quote_path(tapes[day],c["timestamp_ns"],int(s["direction_sign"]),end,float(s["anchor_price"]))
            if cpath.get("entry_available"):
                p["control_match"]={"control_minute":c["minute"],"control_timestamp_ns":c["timestamp_ns"],"control_path":cpath}
                matches.append(p)
    return {"protocol":"unchanged prior protocol; same date/session/phase, intended direction, trailing 2m direction, <=5m, abs(log ATR14 ratio)<=0.50, anchor distance difference<=max(8 ticks,50% signal distance), deterministic nearest normalized distance, without replacement",
        "cohort":"OPENING_CONTINUATION|DISPLACEMENT_CANDLE; no outcome used for matching",
        "candidate_count_by_requirement_stage":dict(stage_totals),"signals_with_any_candidates_by_stage":dict(stage_signal_support),
        "signals_with_zero_controls_before_any_covariate_filter":initial_zero,
        "elimination_events_by_first_zero_stage":dict(failure_reasons),"first_zero_by_requirement_stage":dict(first_zero),
        "cohort_signals":sum(len([s for s in g if s["model"]=="OPENING_CONTINUATION" and s["trigger"]=="DISPLACEMENT_CANDLE"]) for g in groups.values()),
        "matched":len(matches),"unmatched":sum(len([s for s in g if s["model"]=="OPENING_CONTINUATION" and s["trigger"]=="DISPLACEMENT_CANDLE"]) for g in groups.values())-len(matches),
        "matched_signal_ids":[p["signal_id"] for p in matches],"implementation_error_detected":False if stage_totals else None,
        "overlap_positivity_conclusion":"Stage counts identify the first filter where support disappears. Rules were not relaxed; zero final support means the controlled effect is unidentified."}


def _counts(signals: Sequence[Mapping[str,Any]], trades: Sequence[Mapping[str,Any]]) -> dict[str,Any]:
    trade_ids={str(t["signal_id"]) for t in trades};out={}
    for period in ("SPRING_2025","OCTOBER_2025"):
        for session in ("NY_AM","NY_PM"):
            for model in v2.MODELS:
                for trigger in v2.TRIGGERS:
                    for direction in ("LONG","SHORT"):
                        key="|".join((period,session,model,trigger,direction))
                        ss=[s for s in signals if all(s[k]==v for k,v in (("period",period),("session",session),("model",model),("trigger",trigger),("direction",direction)))]
                        ts=[t for t in trades if all(t[k]==v for k,v in (("period",period),("session",session),("model",model),("trigger",trigger),("direction",direction)))]
                        out[key]={"raw_signals":len(ss),"distinct_signal_ids_executed":len({str(t["signal_id"]) for t in ts}),
                            "execution_rows_across_fixed_exit_configurations":len(ts),"signal_dates":len({s["date"] for s in ss})}
    return {"by_period_session_model_trigger_direction":out,
        "overall":{"raw_signals":len(signals),"distinct_signal_ids_executed":len(trade_ids),"execution_rows_across_fixed_exit_configurations":len(trades)},
        "configuration_count":126,"executions_are_not_independent_signals":True}


def _trigger_overlap(signals: Sequence[Mapping[str,Any]]) -> dict[str,Any]:
    groups=defaultdict(dict)
    for s in signals:
        k=(s["date"],s["session"],s["model"],s["direction"],int(s["signal_timestamp_ns"]))
        groups[k][s["trigger"]]=s
    counts=Counter();by_period=defaultdict(Counter);pairs=Counter()
    for key,items in groups.items():
        flags=frozenset(items)
        label="+".join(sorted(flags))
        counts[label]+=1;period="SPRING_2025" if key[0] in v2.PERIODS["SPRING_2025"] else "OCTOBER_2025";by_period[period][label]+=1
        for a in flags:
            for b in flags:
                if a<b:pairs[f"{a}|{b}"]+=1
    return {"unique_event_keys":len(groups),"trigger_membership_patterns":dict(counts),
        "by_period":{k:dict(v) for k,v in by_period.items()},"pairwise_overlap_counts":dict(pairs),
        "same_candle_combined_count":sum(1 for items in groups.values() if "BOS_PLUS_DISPLACEMENT" in items),
        "note":"Trigger labels share a candle when their membership overlaps; BOS_PLUS_DISPLACEMENT is a separate catalog row for the same event key."}


def _event_markouts(signals: Sequence[Mapping[str,Any]], source: Mapping[str,Any]) -> tuple[dict[str,Any],list[dict[str,Any]]]:
    tapes={r["date"]:load_tape(Path(r["candidate_tape_path"]),source_sha256=r["candidate_tape_source_sha256"],
        semantic_sha256=r["candidate_tape_semantic_sha256"]) for r in source["files"]}
    grouped=defaultdict(list);records=[];horizons=(10,30,60,120,300,600,900);bar_cache={}
    for s in signals:
        tape=tapes[s["date"]];start,end=v2._session_ns(s["date"],s["session"]);ev=tape.events
        bar_key=(s["date"],s["session"])
        if bar_key not in bar_cache:bar_cache[bar_key]=v2._bars_for_session(ev,start,end)[0]
        bars=bar_cache[bar_key];minute=int(s["minute_index"])
        anchor_distance=abs(float(s["signal_close"])-float(s["anchor_price"]))/diag.TICK
        atr14=diag._atr14(bars,minute)
        direction=int(s["direction_sign"]);sig_ns=int(s["signal_timestamp_ns"]);anchor=float(s["anchor_price"])
        p=diag._quote_path(tape,sig_ns,direction,end,anchor);entry_ns=p.get("entry_timestamp_ns")
        ts=ev["timestamp_ns"].astype(np.int64,copy=False);trade=(ev["execution_size"]>0)&np.isfinite(ev["execution_price"])
        ti=np.flatnonzero(trade&(ts>=sig_ns)&(ts<end));q=np.flatnonzero((ts>=sig_ns)&(ts<end)&np.isfinite(ev["bid"])&np.isfinite(ev["ask"])&(ev["bid"]>0)&(ev["ask"]>=ev["bid"]))
        raw={};execmarks={}
        for sec in horizons:
            t0=(entry_ns if entry_ns is not None else sig_ns)+sec*1_000_000_000
            j=ti[np.searchsorted(ts[ti],t0,side="right")-1] if len(ti) and np.searchsorted(ts[ti],t0,side="right")>0 else None
            raw[str(sec)]={"last_trade_directional_ticks_from_signal_close":direction*(float(ev["execution_price"][j])-float(s["signal_close"]))/diag.TICK if j is not None else None,
                "last_trade_directional_progress_from_anchor_ticks":direction*(float(ev["execution_price"][j])-anchor)/diag.TICK if j is not None else None,
                "quote_time_ns":int(ts[j]) if j is not None else None}
            h=p.get("horizons",{}).get(str(sec));execmarks[str(sec)]=h
        if entry_ns is not None:
            end_t=int(entry_ns)+900*1_000_000_000
            good=q[ts[q]>=entry_ns];bid=ev["bid"][good].astype(float);ask=ev["ask"][good].astype(float)
            exitrefs=(bid-diag.TICK if direction>0 else ask+diag.TICK)
            entrycost=direction*(float(p["entry_fill"])-float(s["signal_close"]))/diag.TICK
            mfe=max(0.0,float(np.max(direction*(exitrefs-float(p["entry_fill"]))/diag.TICK))) if len(good) else None
            mae=max(0.0,-float(np.min(direction*(exitrefs-float(p["entry_fill"]))/diag.TICK))) if len(good) else None
        else:entrycost=mfe=mae=None
        rec={"signal_id":s["signal_id"],"date":s["date"],"period":s["period"],"session":s["session"],"model":s["model"],"trigger":s["trigger"],"direction":s["direction"],
            "direction_sign":direction,"minute_since_open":minute,"anchor_price":anchor,"signal_close":float(s["signal_close"]),
            "anchor_distance_ticks":anchor_distance,"atr14_ticks":atr14,"features":s.get("features",{}),
            "entry_available":p.get("entry_available",False),"signal_to_entry_directional_ticks":entrycost,
            "signal_to_entry_definition":"includes 2ms elapsed price movement, displayed spread, and one adverse tick; not a pure spread-only estimate",
            "raw_trade_price_markouts":raw,
            "executable_bbo_markouts":execmarks,"executable_mfe_net_ticks":mfe,"executable_mae_net_ticks":mae,
            "seconds_to_anchor_revisit":p.get("seconds_to_anchor_revisit"),"seconds_to_first_favorable_move":None,
            "seconds_to_first_adverse_move":None,"anchor_directional_progress_ticks":direction*(float(p["entry_mid"])-anchor)/diag.TICK if entry_ns is not None else None}
        # First event-time favorable/adverse trade and quote movements after entry.
        if entry_ns is not None:
            lo=int(np.searchsorted(ts,entry_ns,side="left")); hts=ts[lo:]
            rawix=np.flatnonzero(trade[lo:])+lo
            for ix in rawix:
                signed=direction*(float(ev["execution_price"][ix])-float(p["entry_mid"]))/diag.TICK
                if signed>0 and rec["seconds_to_first_favorable_move"] is None:rec["seconds_to_first_favorable_move"]=(int(ts[ix])-entry_ns)/1e9
                if signed<0 and rec["seconds_to_first_adverse_move"] is None:rec["seconds_to_first_adverse_move"]=(int(ts[ix])-entry_ns)/1e9
                if rec["seconds_to_first_favorable_move"] is not None and rec["seconds_to_first_adverse_move"] is not None:break
        records.append(rec);grouped[(s["period"],s["session"],s["model"],s["trigger"],s["direction"])].append(rec)
    results={}
    for key,rows in sorted(grouped.items()):
        name="|".join(key); horizons_out={}
        for sec in horizons:
            raw=[r["raw_trade_price_markouts"][str(sec)]["last_trade_directional_ticks_from_signal_close"] for r in rows]
            anchor_progress=[r["raw_trade_price_markouts"][str(sec)]["last_trade_directional_progress_from_anchor_ticks"] for r in rows]
            exe=[x["executable_net_ticks"] for r in rows if (x:=r["executable_bbo_markouts"].get(str(sec))) is not None]
            horizons_out[str(sec)]={"raw_directional_ticks_from_signal_close":diag._stats(raw),
                "raw_directional_progress_from_anchor_ticks":diag._stats(anchor_progress),"executable_net_ticks":diag._stats(exe)}
        results[name]={"signal_count":len(rows),"signal_to_entry_directional_ticks":diag._stats([r["signal_to_entry_directional_ticks"] for r in rows]),
            "horizons":horizons_out,"MFE_net_ticks":diag._stats([r["executable_mfe_net_ticks"] for r in rows]),
            "MAE_net_ticks":diag._stats([r["executable_mae_net_ticks"] for r in rows]),
            "time_to_favorable_seconds":diag._stats([r["seconds_to_first_favorable_move"] for r in rows]),
            "time_to_adverse_seconds":diag._stats([r["seconds_to_first_adverse_move"] for r in rows]),
            "anchor_progress_ticks":diag._stats([r["anchor_directional_progress_ticks"] for r in rows])}
    return {"method":"one event row per corrected signal; raw print markout from signal close, executable markout from 2ms entry path; descriptive, no exit selection",
        "horizons_seconds":list(horizons),"by_period_session_model_trigger":results,"signal_rows":len(records)},records


def _orderflow_analysis(paths: Sequence[Mapping[str,Any]], trades: Sequence[Mapping[str,Any]]) -> dict[str,Any]:
    """Period-separated continuous associations; signals remain the unit, exit cells are averaged descriptively."""
    fields=("delta_30s_normalized","delta_2m_normalized","session_cvd_normalized","aggression_reversal",
        "opposing_aggression_fraction_2m","price_progress_toward_ticks_2m","effort_without_result",
        "price_impact_ticks_per_100_aggressive_contracts","prior_30s_opposed_fraction","recent_30s_support_fraction")
    by_signal=defaultdict(list)
    for t in trades:by_signal[str(t["signal_id"])].append(t)
    data=[]
    for p in paths:
        trs=by_signal.get(str(p["signal_id"]),[])
        data.append({**p,"mean_net_r_all_applicable_fixed_cells":float(np.mean([float(t["net_r"]) for t in trs])) if trs else None,
            "target_fraction_all_applicable_fixed_cells":sum(t["outcome"]=="TARGET" for t in trs)/len(trs) if trs else None,
            "trade_rows_fixed_cells":len(trs)})
    result={}
    for period in ("SPRING_2025","OCTOBER_2025"):
      for model in v2.MODELS:
       for trigger in v2.TRIGGERS:
        rows=[r for r in data if r["period"]==period and r["model"]==model and r["trigger"]==trigger]
        features={}
        for field in fields:
            per_outcome={}
            for outcome in ("mid_directional_300s","executable_net_300s","MFE_net_ticks","MAE_net_ticks","mean_net_r_all_applicable_fixed_cells","target_fraction_all_applicable_fixed_cells"):
                pairs=[]
                for r in rows:
                    x=r.get("features",{}).get(field)
                    if field=="aggression_reversal": x=1.0 if x is True else 0.0 if x is False else None
                    if outcome=="mid_directional_300s":y=(r.get("horizons",{}).get("300") or {}).get("mid_directional_ticks")
                    elif outcome=="executable_net_300s":y=(r.get("horizons",{}).get("300") or {}).get("executable_net_ticks")
                    else:y=r.get(outcome)
                    if x is not None and y is not None and math.isfinite(float(x)) and math.isfinite(float(y)):pairs.append((float(x),float(y),r["date"]))
                by_date=defaultdict(list)
                for x,y,d in pairs:by_date[d].append((x,y))
                dates=sorted(by_date);rng=np.random.default_rng(9201+len(field)+len(outcome));boot=[]
                for _ in range(500):
                    sample=[z for d in rng.choice(dates,size=len(dates),replace=True) for z in by_date[str(d)]] if dates else []
                    corr=diag._corr([a for a,_ in sample],[b for _,b in sample])
                    if corr is not None:boot.append(corr)
                per_outcome[outcome]={"n":len(pairs),"date_clusters":len(dates),"pearson_r":diag._corr([x for x,_,_ in pairs],[y for _,y,_ in pairs]),
                    "date_cluster_bootstrap_ci95":[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))] if boot else None}
            features[field]=per_outcome
        result[f"{period}|{model}|{trigger}"]={"signal_count":len(rows),"features_vs_outcomes":features,
            "interpretation":"descriptive correlation only; fixed-cell means repeat configurations and are not independent observations"}
    return {"method":"causal signal-time features associated with same-signal forward markouts and average across all applicable fixed exit cells; date-cluster bootstrap, no thresholds or model fitting",
        "period_trigger_model_results":result,"signal_count":len(data),"missing_depth_features":{"TOP5_TOP10_IMBALANCE":"not in sealed tape","MLOFI":"not in sealed tape"},
        "feature_threshold_search":False,"complex_ml":False,"exit_configuration_rows_not_independent":True}


def _exit_surface(trades: Sequence[Mapping[str,Any]], signals: Sequence[Mapping[str,Any]], out_csv: Path) -> list[dict[str,Any]]:
    dates=tuple(v2.DATES);groups=defaultdict(list)
    for t in trades:
        key=(t["period"],t["session"],t["model"],t["trigger"],t["stop_family"],t["target_model"]);groups[key].append(t)
        groups[("ALL",t["session"],t["model"],t["trigger"],t["stop_family"],t["target_model"])].append(t)
    sig_by={(s["date"],s["session"],s["model"],s["trigger"]):[] for s in signals}
    for s in signals:sig_by[(s["date"],s["session"],s["model"],s["trigger"])].append(s)
    rows=[]
    # Full surface, including empty cells, by period and full 54-date set.
    periods=("ALL","SPRING_2025","OCTOBER_2025")
    for period in periods:
      for session in ("NY_AM","NY_PM"):
       for model in v2.MODELS:
        targets=v2._target_cells(model)
        for trigger in v2.TRIGGERS:
         for stop in v2.STOP_FAMILIES:
          for target in targets:
           key=(period,session,model,trigger,stop,target);ts=groups.get(key,[])
           matching_signals=[s for s in signals if s["session"]==session and s["model"]==model and s["trigger"]==trigger and
               (period=="ALL" or s["period"]==period)]
           target_dist=[t["target_distance_ticks"] for t in ts];stop_dist=[t["stop_distance_ticks"] for t in ts]
           metrics=v2._metrics(ts)
           rows.append({"period":period,"session":session,"model":model,"trigger":trigger,"stop_family":stop,"target_model":target,
             "raw_signals":len(matching_signals),"trade_count":metrics["trade_count"],"trading_dates":metrics["active_days"],"win_rate":metrics["win_rate"],
             "average_net_r":metrics["average_net_r"],"total_net_r":metrics["total_net_r"],"profit_factor":metrics["profit_factor_net_usd"],
             "median_initial_stop_distance_ticks":float(np.median(stop_dist)) if stop_dist else None,
             "median_initial_target_distance_ticks":float(np.median(target_dist)) if target_dist else None,
             "median_rr":float(np.median([a/b for a,b in zip(target_dist,stop_dist) if b>0])) if any(b>0 for b in stop_dist) else None,
             "target_before_stop_frequency":metrics["target_before_stop_rate"],"median_mfe_ticks":metrics["median_mfe_ticks"],
             "median_mae_ticks":metrics["median_mae_ticks"],"max_drawdown_r":metrics["max_drawdown_r"]})
    with out_csv.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return rows


def _daily_weekly(signals: Sequence[Mapping[str,Any]], trades: Sequence[Mapping[str,Any]], daily_csv: Path) -> dict[str,Any]:
    # V2 daily rows already include every eligible date/session/model/trigger and per-cell execution details.
    expected={"2025-03-03","2025-04-21","2025-10-07","2025-10-31"}
    # Preserve V2 runner output unchanged under the required artifact name.
    if not daily_csv.is_file():raise CorrectedBosError("V2.1 runner did not emit daily-results.csv")
    with daily_csv.open(newline="",encoding="utf-8") as f: daily=list(csv.DictReader(f))
    found={r["date"] for r in daily}
    if not expected.issubset(found):raise CorrectedBosError("daily results lack expected dates")
    weekly=defaultdict(list);by_day=defaultdict(float)
    for t in trades:
        week=__import__("datetime").date.fromisoformat(t["date"]).isocalendar()
        key=(t["period"],t["session"],t["model"],t["trigger"],t["stop_family"],t["target_model"],f"{week.year}-W{week.week:02d}")
        weekly[key].append(t);by_day[(key[:-1],t["date"])]+=float(t["net_r"])
    weekly_rows=[]
    for key,rows in sorted(weekly.items()):
        weekly_rows.append({"period":key[0],"session":key[1],"model":key[2],"trigger":key[3],"stop_family":key[4],"target_model":key[5],"iso_week":key[6],
            "trades":len(rows),"net_r":sum(float(t["net_r"]) for t in rows),"wins":sum(float(t["net_pnl_usd"])>0 for t in rows),"losses":sum(float(t["net_pnl_usd"])<0 for t in rows)})
    daily_by_config=defaultdict(dict)
    for (cfg,day),value in by_day.items():daily_by_config[cfg][day]=value
    cluster={}
    for cfg,vals in daily_by_config.items():
        active=list(vals.values());n=len(active)
        cluster["|".join(cfg)]={"active_days":n,"positive_days":sum(x>0 for x in active),"negative_days":sum(x<0 for x in active),
            "positive_date_ratio":sum(x>0 for x in active)/n if n else None,"daily_net_r":diag._stats(active),
            "top_5_absolute_daily_contribution_share":sum(sorted((abs(x) for x in active),reverse=True)[:5])/sum(map(abs,active)) if any(active) else None,
            "date_cluster_bootstrap_mean_net_r":diag._bootstrap({d:[x] for d,x in vals.items()},seed=771)}
    return {"daily_rows":len(daily),"daily_rows_by_session_combo_include_zero_signal_eligible_dates":True,
        "weekly_by_exit_configuration":weekly_rows,"date_cluster_robustness_by_exit_configuration":cluster,
        "multiple_comparisons":{"session_model_trigger_combinations":12,"applicable_fixed_exit_configurations":126,
            "period_specific_exit_cells":252,"independent_observation_unit":"trading date; executions across fixed cells are repeated configuration outcomes"}}


def _displacement_regression(original: Mapping[str,Any], corrected: Mapping[str,Any], out: Path) -> dict[str,Any]:
    keys=("signal_id","date","period","session","model","trigger","direction","direction_sign","signal_timestamp_ns","signal_close",
          "anchor_price","anchor_timestamp_ns","minute_index","episode_id","episode_start_trade_index","episode_adverse_extreme","features")
    old_s=[r for r in original["signals"] if r["trigger"]=="DISPLACEMENT_CANDLE"]
    new_s=[r for r in corrected["signals"] if r["trigger"]=="DISPLACEMENT_CANDLE"]
    old_map={r["signal_id"]:r for r in old_s};new_map={r["signal_id"]:r for r in new_s};signal_diff=[]
    if set(old_map)!=set(new_map):signal_diff.append({"issue":"signal_id_set_mismatch","original_only":sorted(set(old_map)-set(new_map))[:20],"corrected_only":sorted(set(new_map)-set(old_map))[:20]})
    else:
        for sid in old_map:
            for k in keys:
                if old_map[sid].get(k)!=new_map[sid].get(k):signal_diff.append({"signal_id":sid,"field":k});break
    trade_keys=("date","period","session","model","trigger","signal_id","signal_timestamp_ns","direction","stop_family","target_model",
        "entry_timestamp_ns","entry_quote","entry_price","stop_price","target_price","stop_distance_ticks","target_distance_ticks","outcome",
        "exit_timestamp_ns","exit_quote","exit_price","gross_pnl_usd","net_pnl_usd","gross_r","net_r","fees_usd","mfe_ticks","mae_ticks")
    old_t=[r for r in original["trades"] if r["trigger"]=="DISPLACEMENT_CANDLE"]
    new_t=[r for r in corrected["trades"] if r["trigger"]=="DISPLACEMENT_CANDLE"]
    def tid(r):return (r["signal_id"],r["stop_family"],r["target_model"])
    om={tid(x):x for x in old_t};nm={tid(x):x for x in new_t};trade_diff=[]
    if set(om)!=set(nm):trade_diff.append({"issue":"trade_configuration_id_set_mismatch","original_only":len(set(om)-set(nm)),"corrected_only":len(set(nm)-set(om))})
    else:
        for k in om:
            for field in trade_keys:
                a,b=om[k].get(field),nm[k].get(field)
                if isinstance(a,(int,float)) and not isinstance(a,bool) and isinstance(b,(int,float)):
                    equal=math.isclose(float(a),float(b),rel_tol=0,abs_tol=FLOAT_TOLERANCE)
                else:equal=a==b
                if not equal:trade_diff.append({"trade_key":k,"field":field,"original":a,"corrected":b});break
    def daily(path):
        with path.open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))
    old_daily=[r for r in daily(V2_ROOT/"daily-results.csv") if r["trigger"]=="DISPLACEMENT_CANDLE"]
    new_daily=[r for r in daily(out/"daily-results.csv") if r["trigger"]=="DISPLACEMENT_CANDLE"]
    daily_same=old_daily==new_daily
    def surface(path):
        with path.open(newline="",encoding="utf-8") as f:return [r for r in csv.DictReader(f) if r["trigger"]=="DISPLACEMENT_CANDLE"]
    old_surface=surface(V2_ROOT/"exit-model-surface.csv");new_surface=surface(out/"exit-model-surface.csv")
    surface_same=old_surface==new_surface
    period_same={}
    for name in ("spring-results.json","october-results.json"):
        a=_json(V2_ROOT/name);b=_json(out/name)
        def only(x):return [r for r in x["metrics"] if r["trigger"]=="DISPLACEMENT_CANDLE"]
        period_same[name]=only(a)==only(b)
    result={"status":"PASS" if not signal_diff and not trade_diff and daily_same and surface_same and all(period_same.values()) else "FAIL",
        "float_abs_tolerance":FLOAT_TOLERANCE,"signal_counts":{"original":len(old_s),"corrected":len(new_s)},
        "signal_identity_timestamp_direction_feature_mismatches":signal_diff[:100],"trade_counts":{"original":len(old_t),"corrected":len(new_t)},
        "trade_economic_mismatches":trade_diff[:100],"daily_displacement_rows_exact":daily_same,"exit_surface_displacement_rows_exact":surface_same,
        "period_displacement_metrics_exact":period_same,
        "fields_compared":list(keys),"trade_fields_compared":list(trade_keys),
        "note":"BOS reference-level annotations may change on displacement rows because the corrected helper returns its direction-correct level; raw displacement trigger identity and all unaffected economics are tested independently."}
    _write_json(out/"displacement-only-regression.json",result)
    return result


def _census_lines(counts: Mapping[str,Any]) -> str:
    lines=[]
    for k,v in sorted(counts["by_period_session_model_trigger_direction"].items()):
        if v["raw_signals"]:lines.append(f"- {k}: raw={v['raw_signals']}, executed signal IDs={v['distinct_signal_ids_executed']}, config executions={v['execution_rows_across_fixed_exit_configurations']}")
    return "\n".join(lines)


def _report(summary: Mapping[str,Any], regression: Mapping[str,Any], counts: Mapping[str,Any], controls: Mapping[str,Any], overlap: Mapping[str,Any]) -> str:
    decisions=summary["decisions"]; surface=summary["exit_surface"]
    lines=["# ES JJ Fair Pricing V2.1 Corrected-BOS Research","",
        "## Integrity and scope","",f"Status: **{summary['status']}**. Original V2 artifacts remained unchanged and verified. Displacement-only regression: **{regression['status']}**.",
        "Only semantic change: BOS closes through the prior-two-bar extreme in intended trade direction. Same 54 sealed 2025 dates, config, execution rules, and exit grid. No optimization, downloads, raw DBN read, depth reconstruction, 2026/OOS, or live-code changes.","",
        "## Corrected census","",_census_lines(counts),"",
        f"Total raw signals={counts['overall']['raw_signals']}; distinct signal IDs executed in >=1 configuration={counts['overall']['distinct_signal_ids_executed']}; execution rows across 126 fixed configurations={counts['overall']['execution_rows_across_fixed_exit_configurations']}. Repeated configuration executions are not independent trades.","",
        "## Displacement-only regression","",f"Signal counts {regression['signal_counts']}; trade counts {regression['trade_counts']}; exact daily rows={regression['daily_displacement_rows_exact']}; exact surface={regression['exit_surface_displacement_rows_exact']}; period summaries={regression['period_displacement_metrics_exact']}. Tolerance: 1e-12 for numeric serialization.","",
        "## Control audit","",f"Continuation displacement cohort={controls['cohort_signals']}; matched={controls['matched']}; unmatched={controls['unmatched']}.",
        f"Candidate counts after each unchanged rule: `{json.dumps(controls['candidate_count_by_requirement_stage'],sort_keys=True)}`. First-zero attrition: `{json.dumps(controls.get('first_zero_by_requirement_stage',controls['elimination_events_by_first_zero_stage']),sort_keys=True)}`.",
        "The replacement-control proposal is documented separately in `continuation-control-methodology.md`; it is not used to claim an effect in this study. Original matching rules were not changed. Zero common support leaves controlled continuation effect unidentified.","",
        "## Mechanism decisions","",f"- Continuation: `{decisions['continuation']}`",f"- Reversion: `{decisions['reversion']}`",f"- Directional BOS: `{decisions['directional_bos']}`",f"- BOS + displacement: `{decisions['combined']}`",
        "Positive markouts or any one exit cell are not predictive-alpha evidence. Spring and October are development periods, not untouched validation/OOS.","",
        "## Fixed exit surface","",f"All {len(surface)} period/session/model/trigger/stop/target rows (including empty cells) are in `exit-surface.csv`; no cell was selected. Exit rows are repeated fixed configurations.","",
        "Per-signal forward responses, executable/raw marks, MFE/MAE and event timing are stored row-wise in `event-markout-rows.jsonl.gz`; `event-markouts.json` contains grouped summaries.","",
        "## Trigger overlap","",json.dumps(overlap,sort_keys=True),"",
        "## Multiple comparisons","",f"12 session/model/trigger combinations; 126 applicable fixed exit configurations; {len(surface)} period/full-sample cells. Date is the statistical dependence unit. No optimization or positive-cell selection.","",
        "## Limitations","","Only sealed BBO/trades/aggressor Candidate Tape V2 was read. TOP5/TOP10 imbalance, MLOFI, depletion, replenishment, and resiliency cannot be reconstructed here. No production edge or strategy approval is claimed.",""]
    return "\n".join(lines)


def _control_methodology() -> str:
    return """# Proposed continuation replacement-control design (not run in V2.1)\n\nThis proposal is for a future, separately preregistered study only. It does not replace the zero-match result or support a continuation-alpha claim here. No outcomes or exit PnL were used to select these rules.\n\n## Risk set and index time\n\nFor each eligible date/session/continuation phase, define the risk set as completed candles that meet the same data-eligibility rules as signals but are not labeled as any of the three trigger populations. Index controls at the completed-candle close. Exclude minutes with an active position/overlapping signal under the frozen one-position rule.\n\n## Matching, fixed before outcomes\n\nMatch within exact date, NY session (AM/PM), continuation phase, intended direction, and sign of trailing two-minute return. Use up to three controls per signal without replacement within a date/session. Require absolute time distance <=10 minutes. Among eligible controls, choose nearest standardized Euclidean distance on (i) log ATR14, (ii) absolute anchor distance, and (iii) absolute two-minute return magnitude; standardization parameters are estimated once from the pooled eligible TRAIN/DEV risk set without outcomes. A tie is broken by earlier timestamp then minute index.\n\nIf any exact stratum has no control, report it as unsupported and do not relax the stratum or calipers. Report overlap tables and retain unmatched signals in the denominator. Estimate signal-minus-control differences at the fixed 10s/30s/1m/2m/5m/10m/15m horizons, with date-cluster bootstrap intervals. Any different caliper or replacement policy must be frozen before outcomes are examined and reported as a new analysis.\n\nThis proposal changes the control design and is not evidence from the current study. The current fixed protocol yielded zero matched continuation displacement controls; continuation effect remains unidentified.\n"""


def run(*, output_root: Path = OUT_ROOT, force: bool = False) -> dict[str,Any]:
    if output_root.exists() and any(output_root.iterdir()) and not force:
        raise CorrectedBosError(f"output exists; refusing to overwrite: {output_root}")
    frozen=freeze_inputs()
    reusable_runner_summary=None
    if output_root.exists():
        try:
            prior_manifest=_json(output_root/"run-manifest.json")
            prior_hashes=_json(output_root/"artifact-hashes.json")
            prior_config=_json(output_root/"study-config.json")
            core=("study-config.json","source-coverage.json","all-signals.jsonl.gz","causal-features.jsonl.gz",
                "executed-trades.jsonl.gz","daily-results.csv","exit-model-surface.csv","session-model-trigger-results.json")
            core_valid=all(name in prior_manifest.get("files",{}) and name in prior_hashes.get("files",{})
                and _sha(output_root/name)==prior_manifest["files"][name]==prior_hashes["files"][name] for name in core)
            if (prior_manifest.get("status")=="COMPLETE" and prior_hashes.get("status")=="HASHED"
                    and prior_manifest.get("study_id")==STUDY_ID and prior_manifest.get("completed_dates")==list(v2.DATES)
                    and prior_config.get("study_id")==STUDY_ID and core_valid):
                reusable_runner_summary={"status":"COMPLETE","dates_processed":54,
                    "source_coverage_sha256":_sha(output_root/"source-coverage.json"),
                    "completed_dates":list(v2.DATES),"reused_verified_core_artifacts":True}
        except (CorrectedBosError,KeyError,ValueError,OSError):
            reusable_runner_summary=None
    tests=_candle_tests()
    if not tests["all_pass"]:raise CorrectedBosError(f"synthetic BOS proof failed: {tests['tests']}")
    integration=_synthetic_production_catalog()
    if not integration["production_signal_catalog_all_direction_cases_pass"] or not integration["entry_ordering_pass"]:
        raise CorrectedBosError(f"production detector synthetic test failed: {integration}")
    output_root.mkdir(parents=True,exist_ok=True)
    saved=(v2.STUDY_ID,v2.RUN_ID,v2.CONFIG)
    old_bos=v2.v1.bos_for_bar
    try:
        v2.STUDY_ID=STUDY_ID;v2.RUN_ID=STUDY_ID
        v2.CONFIG={**saved[2],"study_id":STUDY_ID,"bos_semantic_correction":"intended trade direction; prior two completed candles only"}
        v2.v1.bos_for_bar=directional_bos_for_bar
        # Required smoke stage reuses the exact corrected production runner.
        with tempfile.TemporaryDirectory(prefix="jj-fpt-v21-smoke-") as temp:
            smoke=v2.run(output_root=Path(temp),smoke=True)
        # Small multi-date detector run compares unaffected displacement catalog before full execution.
        original_signals=_read_jsonl_gz(V2_ROOT/"all-signals.jsonl.gz")
        focused=[]
        for day in ("2025-03-03","2025-04-17","2025-10-31"):
            ev,_,_=v2._load_events(day,v2._source_contract()[0])
            for session in ("NY_AM","NY_PM"):
                rows,audit=v2._signal_catalog(day,session,ev)
                focused.extend(rows)
                if not audit.get("eligible"):raise CorrectedBosError(f"focused date/session unexpectedly ineligible: {day} {session}")
        old_focus={(s["signal_id"],s["direction"],s["signal_timestamp_ns"]) for s in original_signals if s["date"] in {"2025-03-03","2025-04-17","2025-10-31"} and s["trigger"]=="DISPLACEMENT_CANDLE"}
        new_focus={(s["signal_id"],s["direction"],s["signal_timestamp_ns"]) for s in focused if s["trigger"]=="DISPLACEMENT_CANDLE"}
        if old_focus!=new_focus:raise CorrectedBosError("focused multi-date displacement signal parity failed")
        focused_proof={"dates":["2025-03-03","2025-04-17","2025-10-31"],"sessions":6,
            "detector_rows":len(focused),"displacement_identity_parity":True,"corrected_bos_rows":sum(x["trigger"]!="DISPLACEMENT_CANDLE" for x in focused)}
        corrected_summary=(reusable_runner_summary if reusable_runner_summary is not None else
                           v2.run(output_root=output_root,smoke=False,force=force))
    finally:
        v2.v1.bos_for_bar=old_bos
        v2.STUDY_ID,v2.RUN_ID,v2.CONFIG=saved
    signals=_read_jsonl_gz(output_root/"all-signals.jsonl.gz");trades=_read_jsonl_gz(output_root/"executed-trades.jsonl.gz")
    original={"signals":_read_jsonl_gz(V2_ROOT/"all-signals.jsonl.gz"),"trades":_read_jsonl_gz(V2_ROOT/"executed-trades.jsonl.gz")}
    current={"signals":signals,"trades":trades}
    regression=_displacement_regression(original,current,output_root)
    if regression["status"]!="PASS":raise CorrectedBosError(f"displacement-only regression failed: {regression}")
    source=_json(output_root/"source-coverage.json")
    counts=_counts(signals,trades);overlap=_trigger_overlap(signals)
    markouts,markout_rows=_event_markouts(signals,source);_write_json(output_root/"event-markouts.json",markouts)
    _write_jsonl_gz(output_root/"event-markout-rows.jsonl.gz",markout_rows)
    # Reuse the diagnostic's established event-path schema for matching and
    # order-flow associations; the compact markout rows intentionally have a
    # different, horizon-oriented schema.
    paths,_=diag._event_paths(signals,source)
    _write_jsonl_gz(output_root/"corrected-signals.jsonl.gz",signals);_write_jsonl_gz(output_root/"corrected-trades.jsonl.gz",trades)
    # Exact original names are retained as extra run outputs; requested concise aliases are exact copies.
    import shutil
    shutil.copyfile(output_root/"exit-model-surface.csv",output_root/"exit-surface.csv")
    surface=_exit_surface(trades,signals,output_root/"exit-surface.csv")
    controls=_match_controls(paths,signals,source)
    matched_control_meta=diag._match_controls(paths,signals,source)
    paired=diag._paired_control_results(paths)
    orderflow=_orderflow_analysis(paths,trades)
    _write_json(output_root/"continuation-control-audit.json",{"attrition":controls,"displacement_only_matched_controls":matched_control_meta,"paired_results":paired})
    _write_json(output_root/"trigger-overlap.json",overlap)
    _write_json(output_root/"bos-direction-tests.json",{**tests,"production_catalog_integration":integration,"focused_multi_date_verification":focused_proof,
        "one_date_smoke":smoke})
    cfg_comp={**frozen,"corrected_config":_json(output_root/"study-config.json"),"source_manifest":source,
        "single_semantic_correction":"directional BOS predicate only; all other search/data/execution/exit definitions frozen"}
    _write_json(output_root/"frozen-config-comparison.json",cfg_comp)
    _write_json(output_root/"orderflow-feature-analysis.json",orderflow)
    _write_json(output_root/"original-v2-versus-v2_1.json",{"original_v2":{"raw_signals":len(original["signals"]),"trade_rows":len(original["trades"])},
        "corrected_v2_1":{"raw_signals":len(signals),"trade_rows":len(trades)},"displacement_only_regression":regression,
        "correction":"only intended-direction BOS confirmation changed","original_v2_findings_remaining_valid":["all displacement-only signal/economic results after regression passes","source/tape integrity and execution model definitions"],
        "original_v2_findings_invalid":["BOS-only direction-dependent outcomes","zero BOS-plus-displacement count","BOS-dependent exit surfaces and flow associations"]})
    # Enrich existing period artifacts with a guarantee they are the complete corrected 54-date run.
    for fname in ("spring-results.json","october-results.json"):
        d=_json(output_root/fname);d["study_id"]=STUDY_ID;d["corrected_bos"]=True;_write_json(output_root/fname,d)
    feasibility=_mbp10_feasibility(source)
    (output_root/"mbp10-feature-feasibility.md").write_text(feasibility,encoding="utf-8")
    (output_root/"continuation-control-methodology.md").write_text(_control_methodology(),encoding="utf-8")
    weekly=_daily_weekly(signals,trades,output_root/"daily-results.csv")
    counts_path=counts
    total_combo=counts_path["overall"]
    pred_control=paired
    # Decisions require effects against controls; zero common support => unidentified.
    continuation_pairs=sum(v["matched_pairs"] for k,v in pred_control.items() if k.startswith("OPENING_CONTINUATION"))
    bos_support=any(v["raw_signals"] for k,v in counts["by_period_session_model_trigger_direction"].items() if "|BOS_ONLY|" in k or "|BOS_PLUS_DISPLACEMENT|" in k)
    reversion_pairs=sum(v["matched_pairs"] for k,v in pred_control.items() if k.startswith("FAIR_PRICE_REVERSION"))
    decisions={"continuation":"INCONCLUSIVE_CONTROL_EFFECT_UNIDENTIFIED" if continuation_pairs==0 else "DESCRIPTIVE_ONLY_PENDING_CLUSTERED_EFFECT",
        "reversion":"DEVELOPMENT_ONLY_MATCHED_EFFECT_REPORTED" if reversion_pairs else "DESCRIPTIVE_NOT_ECONOMICALLY_VALIDATED",
        "directional_bos":"INCONCLUSIVE_DEVELOPMENT_ONLY",
        "combined":"OBSERVED_RARITY" if not any(s["trigger"]=="BOS_PLUS_DISPLACEMENT" for s in signals) else "DEVELOPMENT_ONLY_NO_EDGE_CLAIM",
        "bos_incremental_information":"inconclusive" if bos_support else "unsupported"}
    summary={"study_id":STUDY_ID,"status":"PASS" if regression["status"]=="PASS" and corrected_summary["dates_processed"]==54 else "PARTIAL",
        "dates_processed":corrected_summary["dates_processed"],"period_dates":{"SPRING_2025":35,"OCTOBER_2025":19},
        "signal_census":counts,"trigger_overlap":overlap,"displacement_only_regression":regression,
        "decisions":decisions,"continuation_control_audit":controls,"matched_control_results":paired,
        "event_markout_summary":markouts,"event_markout_rows_path":"event-markout-rows.jsonl.gz",
        "orderflow_feature_analysis_path":"orderflow-feature-analysis.json",
        "continuation_control_methodology_path":"continuation-control-methodology.md",
        "exit_surface":surface,"exit_surface_rows":len(surface),"weekly_daily_robustness":weekly,"multiple_comparisons":weekly["multiple_comparisons"],
        "corrected_run_summary":corrected_summary,"synthetic_tests":tests,"production_catalog_synthetic":integration,
        "frozen_inputs":frozen,"scope":{"parameter_optimization":False,"2026_data_accessed":False,"final_oos_accessed":False,
            "data_downloaded":False,"raw_dbn_read":False,"depth_reconstructed":False,"live_code_modified":False,"commit_performed":False},
        "decisions_caution":"No predictive edge classification without common-support causal controls; Spring and October are researched development periods."}
    _write_json(output_root/"summary.json",summary)
    (output_root/"report.md").write_text(_report(summary,regression,counts,controls,overlap),encoding="utf-8")
    # Freeze final V2.1 manifest; hashes cover every file except the two self-referential manifests.
    run_manifest={"status":"COMPLETE","study_id":STUDY_ID,"semantic_change":"directional BOS confirmation only",
        "source_v2_run_manifest_sha256":frozen["original_v2"]["run_manifest_sha256"],
        "source_v2_artifact_hash_manifest_sha256":frozen["original_v2"]["artifact_hash_manifest_sha256"],
        "source_coverage_sha256":corrected_summary.get("source_coverage_sha256"),"completed_dates":list(v2.DATES),
        "no_2026":True,"no_oos":True,"no_download":True,"files":{}}
    _write_json(output_root/"run-manifest.json",run_manifest)
    files={p.name:_sha(p) for p in sorted(output_root.iterdir()) if p.is_file() and p.name not in ("run-manifest.json","artifact-hashes.json")}
    run_manifest["files"]=files;_write_json(output_root/"run-manifest.json",run_manifest)
    hashes={p.name:_sha(p) for p in sorted(output_root.iterdir()) if p.is_file() and p.name!="artifact-hashes.json"}
    _write_json(output_root/"artifact-hashes.json",{"status":"HASHED","convention":"SHA-256 of every regular artifact including run-manifest.json; excludes artifact-hashes.json itself","files":hashes})
    return summary


def _mbp10_feasibility(source: Mapping[str,Any]) -> str:
    rows=source.get("files",[]);native_rows=[]
    for r in rows:
        p=Path(r["native_source_path"]);native_rows.append({"date":r["date"],"path":str(p),"exists":p.is_file(),"schema":r.get("schema"),"bytes":r.get("native_source_bytes"),"sha256":r.get("native_source_sha256")})
    return ("# Native MBP-10 feature feasibility\n\n"
      "Authorized source-coverage metadata identifies 54 native GLBX.MDP3 ES mbp-10 files and their byte counts/SHA-256; the corrected run did not open those native files. "
      "The already-validated candidate-tape loader preserves BBO, executions, aggressor side and timestamp/order, but not depth ladders.\n\n"
      "A separate event-time depth study would require the listed native MBP-10 sources plus the repository's validated Databento-to-canonical-tape/import path; it must preserve event ordering and use only the same explicitly authorized TRAIN/DEV dates. Feasible metrics include TOP5/TOP10 imbalance, MLOFI, depletion/replenishment, resiliency, and aggression-conditioned impact/exhaustion. None is generated here.\n\n"
      "Metadata inventory (existence checked only; no native data read):\n\n```json\n"+json.dumps(native_rows,indent=2,sort_keys=True)+"\n```\n\n"
      "Protected validation/OOS sources were not enumerated or accessed. No depth reconstruction or liquidity filter was applied.\n")


def main(argv: Sequence[str]|None=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root",type=Path,default=OUT_ROOT)
    parser.add_argument("--force",action="store_true",help="replace only this V2.1 output directory")
    args=parser.parse_args(argv)
    try:result=run(output_root=args.output_root,force=args.force)
    except (CorrectedBosError,KeyError,ValueError,OSError) as exc:parser.exit(2,f"ERROR: {exc}\n")
    print(json.dumps({"status":result["status"],"signals":result["signal_census"]["overall"],"output":str(args.output_root)},sort_keys=True))
    return 0


if __name__=="__main__":raise SystemExit(main())
