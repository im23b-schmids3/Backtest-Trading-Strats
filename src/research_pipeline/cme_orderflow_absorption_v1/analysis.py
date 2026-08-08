"""Non-optimized structural context and raw diagnostics."""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, date, timedelta
from statistics import median

from .engine import Applied

RTH_START, RTH_END = 13 * 3600 + 30 * 60, 20 * 3600
PILOT_START_NS = 1784505600000000000
NS_PER_DAY = 86_400_000_000_000
_DATES = tuple((date(2026, 7, 20) + timedelta(days=i)).isoformat() for i in range(12))
def day_and_seconds(ns: int) -> tuple[str, int]:
    offset = ns - PILOT_START_NS
    day_index, within_day = divmod(offset, NS_PER_DAY)
    if 0 <= day_index < len(_DATES):
        return _DATES[day_index], within_day // 1_000_000_000
    dt = datetime.fromtimestamp(ns / 1_000_000_000, timezone.utc)
    return dt.date().isoformat(), dt.hour * 3600 + dt.minute * 60 + dt.second

@dataclass
class DayMetrics:
    events: int = 0; executions: int = 0; adds: int = 0; cancels: int = 0; modifies: int = 0
    resets: int = 0; unknown_aggressor: int = 0; sequence_regressions: int = 0
    probable_replenishment: int = 0; absorption_candidates: int = 0; no_clear_replenishment: int = 0
    rth_events: int = 0; structural_tags: int = 0

@dataclass
class Diagnostics:
    days: dict[str, DayMetrics] = field(default_factory=lambda: defaultdict(DayMetrics))
    action_counts: Counter = field(default_factory=Counter)
    issues: Counter = field(default_factory=Counter)
    examples: deque = field(default_factory=lambda: deque(maxlen=5))
    executions: int = 0; events: int = 0
    # prior completed RTH extrema/volume-at-price; current day's levels are frozen before RTH.
    rth_prices: dict[str, list[tuple[int,int]]] = field(default_factory=lambda: defaultdict(list))
    levels: dict[str, set[int]] = field(default_factory=dict)
    pending_execs: dict[tuple[str,int,str], int] = field(default_factory=lambda: defaultdict(int))

    def finish_day_context(self, date: str) -> None:
        prior = (datetime.fromisoformat(date).date().toordinal() - 1)
        for d, vals in self.rth_prices.items():
            if datetime.fromisoformat(d).date().toordinal() == prior and vals:
                prices = [p for p,_ in vals]; by_price = Counter()
                for p,s in vals: by_price[p] += s
                poc = max(by_price, key=lambda p:(by_price[p], -p))
                self.levels[date] = {min(prices), max(prices), poc}
                return
        self.levels[date] = set()

    def observe(self, rec: object, applied: Applied | None) -> None:
        date, second = day_and_seconds(rec.ts_recv); m = self.days[date]
        self.events += 1; m.events += 1; self.action_counts[rec.action] += 1
        if RTH_START <= second < RTH_END: m.rth_events += 1
        if rec.action == "R": m.resets += 1; return
        if applied is None: return
        if applied.action == "A": m.adds += 1
        elif applied.action == "C": m.cancels += 1
        elif applied.action == "M": m.modifies += 1
        if applied.executed:
            self.executions += 1; m.executions += 1
            if applied.action == "T":
                # Transaction records do not identify an active displayed order
                # in this reconstruction; do not force an aggressor label.
                m.unknown_aggressor += 1
            key = (date, applied.price, applied.side)
            self.pending_execs[key] += applied.size
            if RTH_START <= second < RTH_END:
                self.rth_prices[date].append((applied.price, applied.size))
            if applied.price in self.levels.get(date, set()): m.structural_tags += 1
        elif applied.action == "A":
            key = (date, applied.price, applied.side)
            prior = self.pending_execs.get(key, 0)
            if prior:
                # Raw, non-threshold diagnostic: an add after any execution at identical price/side.
                m.probable_replenishment += 1
                if len(self.examples) < self.examples.maxlen:
                    self.examples.append({"date":date,"price":applied.price,"passive_side":applied.side,
                                          "executed_size_before_add":prior,"label":"PROBABLE_REPLENISHMENT"})
                self.pending_execs[key] = 0
        # Candidates retain all execution evidence; no threshold is selected.
        if applied.executed:
            m.absorption_candidates += 1

    def finalize(self) -> None:
        for date, m in self.days.items():
            m.no_clear_replenishment = max(0, m.executions - m.probable_replenishment)
