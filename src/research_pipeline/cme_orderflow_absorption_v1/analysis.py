"""Causal structural-interaction study helpers (no signal or trade generation)."""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
import heapq
from statistics import median

from .engine import Applied

RTH_START, RTH_END = 13 * 3600 + 30 * 60, 20 * 3600
TICK = 250_000_000  # DBN fixed-point ES price; 0.25 points at 1e-9 scale.
DBN_FIXED_POINT_SCALE = 1_000_000_000
ES_TICK_SIZE = 0.25
VICINITY_TICKS, TIMEOUT_NS, RESET_NS = 4, 60_000_000_000, 1_000_000_000
PILOT_START_NS, NS_PER_DAY = 1784505600000000000, 86_400_000_000_000
_DATES = tuple((date(2026, 7, 20) + timedelta(days=i)).isoformat() for i in range(12))


def day_and_seconds(ns: int) -> tuple[str, int]:
    offset = ns - PILOT_START_NS
    day_index, within_day = divmod(offset, NS_PER_DAY)
    if 0 <= day_index < len(_DATES):
        return _DATES[day_index], within_day // 1_000_000_000
    dt = datetime.fromtimestamp(ns / 1_000_000_000, timezone.utc)
    return dt.date().isoformat(), dt.hour * 3600 + dt.minute * 60 + dt.second


def previous_completed_rth(day: str) -> str:
    """Return the preceding weekday RTH, never a calendar-day shortcut."""
    cursor = date.fromisoformat(day) - timedelta(days=1)
    while cursor.weekday() >= 5:
        cursor -= timedelta(days=1)
    return cursor.isoformat()


def volume_profile(rows: list[tuple[int, int]] | Counter[int]) -> dict[str, int] | None:
    """Sealed 70% profile: lower-price ties for POC and VA expansion."""
    if not rows:
        return None
    by_price: Counter[int] = Counter(rows) if isinstance(rows, Counter) else Counter()
    if not isinstance(rows, Counter):
        for price, size in rows:
            by_price[price] += size
    poc = min(by_price, key=lambda price: (-by_price[price], price))
    total = sum(by_price.values())
    included = by_price[poc]
    low = high = poc
    # Expand one ES tick at a time. Equal outside volumes select the lower tick.
    while included * 100 < total * 70:
        below, above = low - TICK, high + TICK
        below_volume, above_volume = by_price.get(below, 0), by_price.get(above, 0)
        if below_volume >= above_volume:
            low = below; included += below_volume
        else:
            high = above; included += above_volume
    return {"high": max(by_price), "low": min(by_price), "poc": poc, "vah": high, "val": low}


@dataclass
class DayMetrics:
    events: int = 0; executions: int = 0; adds: int = 0; cancels: int = 0; modifies: int = 0
    resets: int = 0; unknown_aggressor: int = 0; rth_events: int = 0; structural_tags: int = 0


@dataclass
class Interaction:
    interaction_id: str; date: str; level_name: str; level_price: int; start_ns: int
    end_ns: int | None = None; end_price: int | None = None; termination: str | None = None
    events: int = 0; executions: int = 0; execution_volume: int = 0; buy_aggressor_volume: int = 0
    sell_aggressor_volume: int = 0; unknown_aggressor_volume: int = 0; adds: int = 0; cancels: int = 0
    modifies: int = 0; replenishment_count: int = 0; replenished_volume: int = 0
    cancel_replace_ambiguity: int = 0; spreads: list[int] = field(default_factory=list)
    pending_executed: dict[tuple[int, str], int] = field(default_factory=lambda: defaultdict(int))
    # Values are signed ES ticks, never DBN raw-price units.  Violations are
    # retained for audit but deliberately omitted from `responses`.
    responses: dict[int, int] = field(default_factory=dict)
    response_violations: dict[int, str] = field(default_factory=dict)
    last_vicinity_trade_ns: int | None = None
    exit_started_ns: int | None = None

    def summary(self) -> dict[str, object]:
        spread = self.spreads
        return {"interaction_id": self.interaction_id, "date": self.date, "level": self.level_name,
          "level_price": self.level_price, "start_ns": self.start_ns, "end_ns": self.end_ns, "end_price": self.end_price,
          "termination": self.termination, "events": self.events, "executions": self.executions,
          "execution_volume": self.execution_volume, "buy_aggressor_volume": self.buy_aggressor_volume,
          "sell_aggressor_volume": self.sell_aggressor_volume, "unknown_aggressor_volume": self.unknown_aggressor_volume,
          "aggressive_imbalance": self.buy_aggressor_volume - self.sell_aggressor_volume,
          "adds": self.adds, "cancels": self.cancels, "modifies": self.modifies,
          "replenishment_count": self.replenishment_count, "replenished_volume": self.replenished_volume,
          "cancel_replace_ambiguity": self.cancel_replace_ambiguity,
          "spread_min_ticks": min(spread) // TICK if spread else None,
          "spread_median_ticks": median(spread) / TICK if spread else None,
          "responses_signed_ticks": dict(self.responses),
          "response_sanity_violations": dict(self.response_violations),
          "label": self.label()}

    def label(self) -> str:
        # Qualitative, causal labels only: execution precedes same-price restore;
        # "absorption" requires repeated (two or more) such restores.
        if self.replenishment_count >= 2 and self.executions > 0:
            return "ABSORPTION_INTERACTION"
        if self.replenishment_count and self.executions > 0:
            return "PROBABLE_REPLENISHMENT_INTERACTION"
        return "UNLABELED_INTERACTION"


@dataclass
class Diagnostics:
    days: dict[str, DayMetrics] = field(default_factory=lambda: defaultdict(DayMetrics))
    action_counts: Counter = field(default_factory=Counter); issues: Counter = field(default_factory=Counter)
    examples: deque = field(default_factory=lambda: deque(maxlen=5)); executions: int = 0; events: int = 0
    rth_prices: dict[str, Counter[int]] = field(default_factory=lambda: defaultdict(Counter))
    levels: dict[str, dict[str, int]] = field(default_factory=dict)
    active: dict[tuple[str, str, int | None], Interaction] = field(default_factory=dict)
    completed: list[Interaction] = field(default_factory=list)
    current_extrema: dict[str, tuple[int, int]] = field(default_factory=dict)
    interaction_sequence: Counter = field(default_factory=Counter)
    pending_responses: list[tuple[int, int, Interaction, int]] = field(default_factory=list)
    response_sequence: int = 0
    response_sanity_violations: list[dict[str, object]] = field(default_factory=list)
    last_market: dict[str, tuple[int, int]] = field(default_factory=dict)

    @staticmethod
    def es_price(raw_price: int) -> float:
        """Convert a Databento fixed-point price to ES points exactly once."""
        return raw_price / DBN_FIXED_POINT_SCALE

    @staticmethod
    def response_ticks(future_raw_price: int, reference_raw_price: int) -> int:
        """Signed future-minus-reference response in 0.25-point ES ticks."""
        delta = future_raw_price - reference_raw_price
        if delta % TICK:
            raise ValueError("non-integral ES tick response")
        return delta // TICK

    def finish_day_context(self, day: str) -> None:
        prior = previous_completed_rth(day)
        profile = volume_profile(self.rth_prices.get(prior, []))
        self.levels[day] = {} if profile is None else {"PRIOR_RTH_HIGH": profile["high"], "PRIOR_RTH_LOW": profile["low"], "PRIOR_RTH_POC": profile["poc"], "PRIOR_RTH_VAH": profile["vah"], "PRIOR_RTH_VAL": profile["val"]}

    def _close(self, key: tuple[str, str, int | None], interaction: Interaction, reason: str, ts: int, price: int) -> None:
        interaction.end_ns, interaction.end_price, interaction.termination = ts, price, reason
        self.completed.append(interaction); self.active.pop(key, None)
        for horizon in (5, 15, 30, 60, 120):
            self.response_sequence += 1
            heapq.heappush(self.pending_responses, (ts + horizon * 1_000_000_000, self.response_sequence, interaction, horizon))

    def _resolve_responses(self, ts: int, price: int) -> None:
        """Use the first subsequent ES execution at/after each horizon only."""
        while self.pending_responses and self.pending_responses[0][0] <= ts:
            _, _, interaction, horizon = heapq.heappop(self.pending_responses)
            if interaction.end_price is not None:
                value = self.response_ticks(price, interaction.end_price)
                if abs(value) > 500:
                    interaction.response_violations[horizon] = "RESPONSE_SANITY_VIOLATION"
                    self.response_sanity_violations.append({"interaction_id": interaction.interaction_id,
                        "horizon_seconds": horizon, "response_ticks": value,
                        "reference_price_es": self.es_price(interaction.end_price),
                        "future_price_es": self.es_price(price), "classification": "RESPONSE_SANITY_VIOLATION"})
                else:
                    interaction.responses[horizon] = value

    def _levels_for(self, day: str, price: int) -> dict[str, int]:
        levels = dict(self.levels.get(day, {}))
        if day in self.current_extrema:
            low, high = self.current_extrema[day]
            levels.update({"CURRENT_RTH_LOW_SWEEP": low, "CURRENT_RTH_HIGH_SWEEP": high})
        return levels

    def _key(self, day: str, name: str, level: int) -> tuple[str, str, int | None]:
        # A current RTH sweep is one lifecycle while its extreme is revised;
        # fixed prior levels retain their immutable price in the identity.
        return (day, name, None if name.startswith("CURRENT_RTH_") else level)

    def _touch(self, day: str, ts: int, price: int, applied: Applied | None, spread: int | None,
               *, market_observation: bool | None = None) -> None:
        if market_observation is None:
            market_observation = bool(applied and applied.executed)
        # Lifecycle transitions are driven only by executed ES trades.  Order
        # adds, cancels, depth and side changes can enrich an open interaction,
        # but can neither close it nor create/restart it.
        for key, interaction in list(self.active.items()):
            if key[0] != day: continue
            if market_observation:
                if interaction.last_vicinity_trade_ns is not None and ts - interaction.last_vicinity_trade_ns >= TIMEOUT_NS:
                    self._close(key, interaction, "VICINITY_TIMEOUT", ts, price); continue
                if abs(price - interaction.level_price) > VICINITY_TICKS * TICK:
                    if interaction.exit_started_ns is None:
                        interaction.exit_started_ns = ts
                    elif ts - interaction.exit_started_ns >= RESET_NS:
                        self._close(key, interaction, "VICINITY_EXIT_RESET", ts, price); continue
                else:
                    interaction.exit_started_ns = None
                    interaction.last_vicinity_trade_ns = ts
        if not market_observation:
            # Attribute book events only to an already-open lifecycle; they do
            # not establish a price visit.
            candidates = [(name, level) for name, level in self._levels_for(day, price).items()
                          if self._key(day, name, level) in self.active]
        else:
            candidates = [(name, level) for name, level in self._levels_for(day, price).items()
                          if abs(price - level) <= VICINITY_TICKS * TICK]
        for name, level in candidates:
                key = self._key(day, name, level)
                if key not in self.active:
                    self.interaction_sequence[key] += 1
                    self.active[key] = Interaction(f"{day}:{name}:{level}:{self.interaction_sequence[key]:04d}", day, name, level, ts)
                interaction = self.active[key]
                # Current sweep price revisions are supersessions within the
                # same lifecycle, never an automatic new interaction.
                if name.startswith("CURRENT_RTH_"):
                    interaction.level_price = level
                interaction.events += 1
                if spread is not None: interaction.spreads.append(spread)
                if applied is None: continue
                if applied.action == "A":
                    interaction.adds += 1; pending = interaction.pending_executed.pop((applied.price, applied.side), 0)
                    if pending:
                        interaction.replenishment_count += 1; interaction.replenished_volume += applied.size
                        self.examples.append({"date": day, "interaction_id": interaction.interaction_id, "level": name, "price": applied.price, "label": "PROBABLE_REPLENISHMENT_INTERACTION"})
                elif applied.action == "C": interaction.cancels += 1; interaction.cancel_replace_ambiguity += 1
                elif applied.action == "M": interaction.modifies += 1; interaction.cancel_replace_ambiguity += 1
                if applied.executed:
                    interaction.executions += 1; interaction.execution_volume += applied.size
                    interaction.pending_executed[(applied.price, applied.side)] += applied.size
                    if market_observation: interaction.last_vicinity_trade_ns = ts
                    if applied.action == "F":
                        if applied.side == "A": interaction.buy_aggressor_volume += applied.size
                        elif applied.side == "B": interaction.sell_aggressor_volume += applied.size
                    else: interaction.unknown_aggressor_volume += applied.size

    def observe(self, rec: object, applied: Applied | None, spread: int | None = None) -> None:
        day, second = day_and_seconds(rec.ts_recv); metric = self.days[day]
        self.events += 1; metric.events += 1; self.action_counts[rec.action] += 1
        if RTH_START <= second < RTH_END: metric.rth_events += 1
        if rec.action == "R": metric.resets += 1; return
        if applied is None: return
        if applied.action == "A": metric.adds += 1
        elif applied.action == "C": metric.cancels += 1
        elif applied.action == "M": metric.modifies += 1
        if applied.executed:
            self.executions += 1; metric.executions += 1
            if applied.action == "T": metric.unknown_aggressor += 1
            if RTH_START <= second < RTH_END:
                self.rth_prices[day][applied.price] += applied.size
                low, high = self.current_extrema.get(day, (applied.price, applied.price))
                self.current_extrema[day] = min(low, applied.price), max(high, applied.price)
        if RTH_START <= second < RTH_END:
            # Only an ES execution is a valid market observation for visits,
            # closure references, and future response lookup.  Raw order
            # prices are never allowed into response arithmetic.
            market_observation = applied.executed
            if market_observation:
                self.last_market[day] = (rec.ts_recv, applied.price)
                self._resolve_responses(rec.ts_recv, applied.price)
            self._touch(day, rec.ts_recv, applied.price, applied, spread, market_observation=market_observation)
            metric.structural_tags += sum(abs(applied.price - p) <= VICINITY_TICKS * TICK for p in self._levels_for(day, applied.price).values())

    def finalize(self) -> None:
        for key, interaction in list(self.active.items()):
            ts, price = self.last_market.get(interaction.date, (interaction.start_ns, interaction.level_price))
            self._close(key, interaction, "RTH_END", ts, price)

    def interaction_rows(self) -> list[dict[str, object]]:
        return [item.summary() for item in self.completed]
