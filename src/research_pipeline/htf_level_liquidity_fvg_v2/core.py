"""V2-only deltas over the sealed V1 closed-bar engine.

This module deliberately reuses V1's unchanged mechanics and overrides only the
three sealed V2 differences plus version-namespaced identities.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from research_pipeline.htf_level_liquidity_fvg.core import (
    Bar, Bias, Candidate, ClosedBarAggregator, Direction, Event, EventScope, Level,
    TerminalDisposition, REQUIRED_ARTIFACTS, HTFLevelLiquidityFVG as V1Engine,
    event_scope, phase_a_hard_gates, reconcile_events,
)

SPEC_HASH = "A33DF7968CA6277B66C23F59CBC50AC8945DC8C446715E8AC6724C527F930E9C"
CANDIDATES = {"HTFLFVG-V2-MIN1P5R": 1.5, "HTFLFVG-V2-MIN2P0R": 2.0, "HTFLFVG-V2-MIN2P5R": 2.5}


def deterministic_id(kind: str, *parts: object) -> str:
    return f"v2_{kind}_{hashlib.sha256(('HTFLFVG-V2|' + '|'.join(map(str, parts))).encode()).hexdigest()[:20]}"


class V2ClosedBarAggregator(ClosedBarAggregator):
    """The unchanged aggregation rules with V2-derived-bar identities."""
    def push(self, bar: Bar) -> dict[str, Bar]:
        emitted = super().push(bar)
        return {name: Bar(value.time, value.open, value.high, value.low, value.close, value.volume,
                          deterministic_id("derived", name, value.time.isoformat(), bar.time.isoformat()))
                for name, value in emitted.items()}


class HTFLevelLiquidityFVG(V1Engine):
    """V1 execution matrix with the sealed V2 MSS window and swing rule."""
    def __init__(self, candidate_id: str, **kwargs: Any):
        if candidate_id not in CANDIDATES:
            raise ValueError("unsealed V2 candidate")
        # V1 validates its private registry; replace only the candidate descriptor.
        super().__init__("HTFLFVG-V1-MIN2P5R", **kwargs)
        self.candidate = Candidate(candidate_id, CANDIDATES[candidate_id])
        self.aggregator = V2ClosedBarAggregator()

    def _event(self, t: datetime, decision: str, setup_id: str | None = None,
               reason_code: str | None = None, trade_id: str | None = None, **inputs: Any) -> None:
        scope = event_scope(decision)
        if scope in {EventScope.SETUP, EventScope.TRADE} and not setup_id:
            raise ValueError("setup/trade event requires setup_id")
        if scope in {EventScope.GLOBAL, EventScope.LEVEL} and setup_id is not None:
            raise ValueError("non-setup event cannot carry setup_id")
        if scope == EventScope.TRADE and not trade_id:
            raise ValueError("trade event requires trade_id")
        self._seq += 1
        self.events.append(Event(deterministic_id("event", self.run_id, self.candidate.id, self._seq),
            t.isoformat(), self._seq, self.candidate.id, decision, setup_id, reason_code,
            inputs, scope, trade_id))

    def _confirm_levels(self, tf: str) -> None:
        bars = self.bars4h if tf == "4h" else self.bars15
        right = 3 if tf == "4h" else 2
        if len(bars) < 2 * right + 1:
            return
        i = len(bars) - right - 1; pivot = bars[i]
        left, right_bars = bars[i-right:i], bars[i+1:i+right+1]
        low = pivot.low < min(x.low for x in left) and pivot.low <= min(x.low for x in right_bars)
        high = pivot.high > max(x.high for x in left) and pivot.high >= max(x.high for x in right_bars)
        if low or high:
            side, price = ("SUPPORT", pivot.low) if low else ("RESISTANCE", pivot.high)
            level = Level(deterministic_id("level", tf, side, pivot.time.isoformat(), price), side, price,
                          pivot.time, bars[-1].time)
            (self.levels if tf == "4h" else self.levels15).append(level)
            self._event(level.confirmation_time, "LEVEL_CONFIRMED", level_id=level.level_id,
                        timeframe=tf, side=side, price=price)

    def _evaluate_sweep(self, b: Bar) -> None:
        # Exact V1 sweep logic, with V2-namespaced setup and sweep IDs.
        atr = self._prior_atr(self.bars15, self._atr15_prior)
        if atr is None or self.daily_bias() == Bias.NEUTRAL:
            return
        bias = self.daily_bias(); direction = Direction.LONG if bias == Bias.BULLISH else Direction.SHORT
        level = self._available_level("SUPPORT" if direction == Direction.LONG else "RESISTANCE", b.close, b.time)
        if level is None:
            return
        depth = b.low <= level.price - .1 * atr if direction == Direction.LONG else b.high >= level.price + .1 * atr
        reclaim = b.close >= level.price if direction == Direction.LONG else b.close <= level.price
        wick = ((min(b.open,b.close)-b.low)/(b.high-b.low) if b.high > b.low else 0) if direction == Direction.LONG else ((b.high-max(b.open,b.close))/(b.high-b.low) if b.high > b.low else 0)
        if not (depth and reclaim and wick >= .5):
            return
        self._consume_level(level)
        sid = deterministic_id("setup", self.run_id, self.candidate.id, level.level_id, b.time.isoformat()); sweep_id = deterministic_id("sweep", sid)
        self._supersede_setup(b.time, sid, sweep_id)
        self.setup = {"setup_id": sid, "level_id": level.level_id, "sweep_id": sweep_id,
            "mss_id": None, "fvg_id": None, "direction": direction.value, "daily_bias_snapshot": bias.value,
            "level_snapshot": asdict(level), "sweep": asdict(b), "sweep_atr": atr,
            "sweep_extreme": b.low if direction == Direction.LONG else b.high,
            "sweep_5_index": len(self.bars5)-1, "event_history": ["SETUP_PROPOSED"]}
        self._event(b.time, "SETUP_PROPOSED", sid, level_id=level.level_id, sweep_id=sweep_id, prior_atr=atr, wick=wick)

    def _progress_setup(self, b: Bar) -> None:
        if not self.setup or self.order or self.position:
            return
        s = self.setup; elapsed = len(self.bars5)-1-s["sweep_5_index"]
        if elapsed > 24:
            self._finish(b.time, TerminalDisposition.MSS_WINDOW_EXPIRED); return
        if self.daily_bias() != (Bias.BULLISH if s["direction"] == "LONG" else Bias.BEARISH):
            self._finish(b.time, TerminalDisposition.BIAS_INVALIDATED_BEFORE_ENTRY); return
        i = len(self.bars5)-1; atr = self._prior_atr(self.bars5, self._atr5_prior)
        if atr is None:
            return
        d = Direction(s["direction"])
        # Candidate pivot is i-2: two completed left bars, one completed right bar (i-1).
        # Therefore i-1 closes strictly before the MSS close at i; no same-bar/future leak.
        if s["mss_id"] is None and i >= 3:
            pivot = self.bars5[i-2]; left = self.bars5[i-4:i-2]; right = self.bars5[i-1]
            swing = (pivot.high > max(x.high for x in left) and pivot.high >= right.high) if d == Direction.LONG else (pivot.low < min(x.low for x in left) and pivot.low <= right.low)
            structural = b.close > pivot.high if d == Direction.LONG else b.close < pivot.low
            if swing and structural:
                displacement = (b.close > b.open if d == Direction.LONG else b.close < b.open) and b.high-b.low >= 1.5*atr and (b.close >= b.low+.75*(b.high-b.low) if d == Direction.LONG else b.close <= b.high-.75*(b.high-b.low))
                if not displacement:
                    self._finish(b.time, TerminalDisposition.DISPLACEMENT_REJECTED); return
                s["mss_id"] = deterministic_id("mss", s["setup_id"], b.time.isoformat()); s["displacement_id"] = deterministic_id("displacement", s["mss_id"]); s["mss_index"] = i
                swing_id = deterministic_id("swing", pivot.time.isoformat())
                self._lifecycle(b.time, "MSS_CONFIRMED", mss_id=s["mss_id"], fractal_id=swing_id,
                    left_bar_ids=[x.id for x in left], right_bar_id=right.id, right_bar_close_time=right.time.isoformat(), prior_atr=atr)
                self._lifecycle(b.time, "DISPLACEMENT_CONFIRMED", mss_id=s["mss_id"], displacement_id=s["displacement_id"])
        # Unchanged V1 FVG order/window; boundary extends only because MSS may be bar 24.
        if s.get("mss_id") and i-s["mss_index"] <= 2 and i >= 2:
            a, c = self.bars5[i-2], self.bars5[i]; lower, upper = (a.high,c.low) if d == Direction.LONG else (c.high,a.low)
            valid = c.low > a.high if d == Direction.LONG else c.high < a.low
            if valid:
                width = upper-lower
                if width < .1*atr:
                    self._finish(b.time, TerminalDisposition.FVG_TOO_SMALL); return
                s["fvg_id"] = deterministic_id("fvg",s["setup_id"],a.time.isoformat(),c.time.isoformat()); s["fvg"] = {"bars":[x.id for x in self.bars5[i-2:i+1]],"lower":lower,"upper":upper,"width":width}
                self._lifecycle(b.time,"FVG_CONFIRMED",fvg_id=s["fvg_id"],boundaries=s["fvg"]); self._activate_order(b)
        elif s.get("mss_id") and i-s["mss_index"] > 2:
            self._finish(b.time, TerminalDisposition.FVG_NOT_FORMED)

    def _progress_order_or_position(self, b: Bar) -> None:
        """Keep V1 execution mechanics while applying V2's activation-based expiry.

        The sealed order lifetime begins at order activation, not at the earlier
        sweep.  Handling pending orders here also makes the required pre-fill
        daily-bias invalidation explicit; positions continue through the
        unchanged V1 exit matrix.
        """
        if self.order and not self.position:
            s = self.setup
            assert s is not None
            direction = Direction(s["direction"])
            index = len(self.bars5) - 1
            if b.time.hour == 23 and b.time.minute >= 55:
                self._finish(b.time, TerminalDisposition.SESSION_ENDED)
                return
            if index > self.order["activated_index"] + 12:
                self._finish(b.time, TerminalDisposition.PENDING_ORDER_EXPIRED)
                return
            expected_bias = Bias.BULLISH if direction == Direction.LONG else Bias.BEARISH
            if self.daily_bias() != expected_bias:
                self._finish(b.time, TerminalDisposition.BIAS_INVALIDATED_BEFORE_ENTRY)
                return
            if (b.low < s["sweep_extreme"] if direction == Direction.LONG else b.high > s["sweep_extreme"]):
                self._finish(b.time, TerminalDisposition.PRE_ENTRY_SWEEP_INVALIDATED)
                return
            if index <= self.order["activated_index"]:
                return
            touched = b.low <= self.order["entry"] if direction == Direction.LONG else b.high >= self.order["entry"]
            if not touched:
                return
            price = self.order["entry"] + (self.slippage_ticks * self.tick if direction == Direction.LONG else -self.slippage_ticks * self.tick)
            quantity = max(self.minimum_quantity, 1.0 // self.quantity_step * self.quantity_step)
            self.position = {"position_id": deterministic_id("position", self.order["order_id"]), "entry_index": index,
                             "entry": price, "qty": quantity, "remaining": quantity, "tp1_done": False}
            s["trade_id"] = deterministic_id("trade", s["setup_id"])
            self._lifecycle(b.time, "ENTRY_FILLED", fill_id=deterministic_id("fill", s["trade_id"], "entry"), quantity=quantity, price=price)
            return
        if self.position:
            super()._progress_order_or_position(b)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2, default=str) + "\n", encoding="utf-8")


def materialize_synthetic(artifact_root: Path, repository_root: Path) -> dict[str, Any]:
    if artifact_root.exists():
        raise FileExistsError("immutable artifact collision")
    spec = repository_root / ".smithers/specs/htf-level-liquidity-fvg-v2-relaxed-mss.md"
    if hashlib.sha256(spec.read_bytes()).hexdigest().upper() != SPEC_HASH:
        raise RuntimeError("sealed specification hash mismatch")
    artifact_root.mkdir(parents=True); now = datetime(2023,1,1,tzinfo=timezone.utc); engines = []
    for candidate_id in CANDIDATES:
        engine = HTFLevelLiquidityFVG(candidate_id, run_id=f"synthetic-v2-{candidate_id}")
        for i in range(15): engine.feed(Bar(now+timedelta(minutes=5*i),100,101,99,100.5,1,f"synthetic-v2-{i}"))
        engines.append(engine)
    common = {"specification_hash": SPEC_HASH, "synthetic_only": True, "realStudyExecuted": False, "phaseBExecuted": False, "alphaExecuted": False}
    runs = [{"candidate_id": e.candidate.id, "event_count":len(e.events)} for e in engines]
    payloads = {"sealed-specification.json":{"sha256":SPEC_HASH}, "candidate-registry.json":{"candidates":[asdict(Candidate(k,v)) for k,v in CANDIDATES.items()]}, "data-manifest.json":{**common,"source":"synthetic fixtures only"}, "derived-timeframe-manifest.json":{"derived_only_in_memory":True,"candidate_runs":runs}, "configuration.json":common, "levels.json":[], "events.json":[asdict(x) for e in engines for x in e.events], "trades.json":[], "setup_outcomes.json":[], "monthly_metrics.json":[], "report.json":{**common,"candidate_runs":runs}, "gates.json":{k:phase_a_hard_gates({"executed_trades":0,"immutable_artifacts":True,"funnel_reconciled":True}) for k in CANDIDATES}, "selection_report.json":{**common,"status":"PHASE_A_NO_ROBUST_CANDIDATE"}, "freeze.json":{**common,"status":"NOT_FROZEN"}, "final_report.json":{**common,"status":"SYNTHETIC_MATERIALIZED"}}
    for name in REQUIRED_ARTIFACTS:
        if name != "integrity-manifest.json": _write_json(artifact_root/name, payloads[name])
    _write_json(artifact_root/"integrity-manifest.json", {"files": {n: hashlib.sha256((artifact_root/n).read_bytes()).hexdigest() for n in REQUIRED_ARTIFACTS if n != "integrity-manifest.json"}})
    return {"status":"SYNTHETIC_MATERIALIZED","artifactRoot":str(artifact_root),"realStudyExecuted":False,"candidates":list(CANDIDATES)}
