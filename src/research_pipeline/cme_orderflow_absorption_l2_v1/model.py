"""Independent, causal Level-2 absorption / latent-liquidity research model.

The strategy layer in this module only accepts an MBP-10-equivalent public
view: aggregate price, displayed size, and aggregate order count for each
level, plus executions with aggressor side.  ``MBOToMBP10View`` may reconstruct
that view from historical MBO events, but order ids remain private to that
adapter and are never present in any object consumed by the L2 engine.

There is deliberately no Databento client, DBN reader, L3 signal import, or
outcome/PnL calibration here.  All thresholds are predeclared research values.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from math import floor, isfinite
from statistics import median
from typing import Iterable, Literal


TICK = 0.25
VICINITY_TICKS = 4
VICINITY_POINTS = VICINITY_TICKS * TICK
INACTIVITY_NS = 60_000_000_000
EXIT_RESET_NS = 1_000_000_000
MIN_CONFIRMATION_NS = 5_000_000_000
MAX_CONFIRMATION_NS = 15_000_000_000
ENTRY_LATENCY_NS = 2_000_000
STOP_BUFFER_TICKS = 5
TARGET_R = 3.0
RISK_BUDGET_USD = 250.0
ES_POINT_VALUE, MES_POINT_VALUE = 50.0, 5.0
ES_COMMISSION, MES_COMMISSION = 3.0, 1.25
ES_CAP, MES_CAP = 6, 60
RECOVERY_WINDOWS_NS = (100_000_000, 250_000_000, 500_000_000, 1_000_000_000)
LEVEL_NAMES = (
    "PRIOR_RTH_HIGH", "PRIOR_RTH_LOW", "PRIOR_RTH_POC", "PRIOR_RTH_VAH",
    "PRIOR_RTH_VAL", "CURRENT_RTH_HIGH_SWEEP", "CURRENT_RTH_LOW_SWEEP",
)


class L2ValidationError(ValueError):
    """The public L2 contract could not be satisfied without inference."""


def _point_price(value: float) -> float:
    """Require normalized positive points; reject raw Databento fixed-point values.

    The L2 book may legitimately contain far-away resting prices (including
    100.00) during a valid snapshot.  Plausibility relative to the current ES
    market is not a unit-conversion test, so it must not discard that source
    depth before MBP-10 aggregation.
    """
    number = float(value)
    if not 0.0 < number < 100_000.0:
        raise L2ValidationError("L2 strategy requires normalized ES point prices exactly once")
    return number


def _private_book_price(value: float) -> float:
    """Accept a finite normalized private-book price without exposing it to L2."""
    number = float(value)
    if not isfinite(number) or not 0.0 < number < 1_000_000_000.0:
        raise L2ValidationError("private MBO book requires a finite normalized point price")
    return number


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe_ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator <= 0 else numerator / denominator


@dataclass(frozen=True)
class StructuralLevel:
    name: str
    price: float

    def __post_init__(self) -> None:
        if self.name not in LEVEL_NAMES:
            raise L2ValidationError("unknown structural level")
        object.__setattr__(self, "price", _point_price(self.price))


@dataclass(frozen=True)
class Execution:
    """A public trade observation; no resting-order or queue identity exists."""

    timestamp_ns: int
    price: float
    size: int
    aggressor: Literal["BUY", "SELL", "UNKNOWN"]

    def __post_init__(self) -> None:
        object.__setattr__(self, "price", _point_price(self.price))
        if self.size <= 0:
            raise L2ValidationError("execution size must be positive")


@dataclass(frozen=True)
class MBPLevel:
    """One aggregate displayed L2 level, intentionally without an order id."""

    price: float
    size: int
    order_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "price", _point_price(self.price))
        if self.size < 0 or self.order_count < 0:
            raise L2ValidationError("MBP displayed size and order count must be non-negative")


@dataclass(frozen=True)
class MBP10Snapshot:
    """MBP-10-equivalent public book: exactly aggregate top-ten information."""

    timestamp_ns: int
    bids: tuple[MBPLevel, ...]
    asks: tuple[MBPLevel, ...]

    def __post_init__(self) -> None:
        if len(self.bids) > 10 or len(self.asks) > 10:
            raise L2ValidationError("MBP-10 view cannot expose more than ten levels per side")
        if tuple(sorted((item.price for item in self.bids), reverse=True)) != tuple(item.price for item in self.bids):
            raise L2ValidationError("bids must be descending")
        if tuple(sorted(item.price for item in self.asks)) != tuple(item.price for item in self.asks):
            raise L2ValidationError("asks must be ascending")

    def level_at(self, side: Literal["B", "A"], price: float) -> MBPLevel | None:
        target = _point_price(price)
        rows = self.bids if side == "B" else self.asks
        return next((item for item in rows if item.price == target), None)

    def depth(self, side: Literal["B", "A"], levels: int) -> int:
        rows = self.bids if side == "B" else self.asks
        return sum(item.size for item in rows[:levels])

    @staticmethod
    def _field(rows: tuple[MBPLevel, ...], name: Literal["price", "size", "order_count"]) -> tuple[float | int | None, ...]:
        values: list[float | int | None] = [getattr(item, name) for item in rows]
        return tuple(values + [None] * (10 - len(values)))

    @property
    def bid_px(self) -> tuple[float | int | None, ...]: return self._field(self.bids, "price")
    @property
    def bid_sz(self) -> tuple[float | int | None, ...]: return self._field(self.bids, "size")
    @property
    def bid_ct(self) -> tuple[float | int | None, ...]: return self._field(self.bids, "order_count")
    @property
    def ask_px(self) -> tuple[float | int | None, ...]: return self._field(self.asks, "price")
    @property
    def ask_sz(self) -> tuple[float | int | None, ...]: return self._field(self.asks, "size")
    @property
    def ask_ct(self) -> tuple[float | int | None, ...]: return self._field(self.asks, "order_count")


@dataclass(frozen=True)
class MBP10Update:
    """Aggregate public book change accompanying an MBP-10 snapshot.

    ``size_delta`` and ``order_count_delta`` deliberately express only an
    aggregate change.  They cannot identify an individual order or queue.
    """

    timestamp_ns: int
    side: Literal["B", "A"]
    price: float
    size_delta: int
    order_count_delta: int
    kind: Literal["ADD", "CANCEL", "MODIFY", "FILL", "RESET"]

    def __post_init__(self) -> None:
        object.__setattr__(self, "price", _point_price(self.price))


@dataclass(frozen=True)
class MBOEvent:
    """Private adapter input.  Only this adapter type contains ``order_id``."""

    timestamp_ns: int
    action: Literal["A", "C", "M", "F", "R", "T"]
    side: Literal["B", "A"]
    price: float
    size: int
    order_id: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "price", _private_book_price(self.price))


@dataclass(frozen=True)
class _RestingOrder:
    side: Literal["B", "A"]
    price: float
    size: int


class MBOToMBP10View:
    """Private order-id reconstruction which exports MBP-10-safe aggregate data."""

    def __init__(self) -> None:
        self._orders: dict[int, _RestingOrder] = {}
        self._depth: dict[str, dict[float, int]] = {"B": {}, "A": {}}
        self._counts: dict[str, dict[float, int]] = {"B": {}, "A": {}}

    def _adjust(self, side: Literal["B", "A"], price: float, size_delta: int, count_delta: int) -> None:
        size = self._depth[side].get(price, 0) + size_delta
        count = self._counts[side].get(price, 0) + count_delta
        if size < 0 or count < 0:
            raise L2ValidationError("MBO adapter would create negative aggregate depth")
        if size:
            self._depth[side][price] = size
        else:
            self._depth[side].pop(price, None)
        if count:
            self._counts[side][price] = count
        else:
            self._counts[side].pop(price, None)

    def snapshot(self, timestamp_ns: int) -> MBP10Snapshot:
        bids = tuple(MBPLevel(price, self._depth["B"][price], self._counts["B"][price])
                     for price in sorted(self._depth["B"], reverse=True)[:10])
        asks = tuple(MBPLevel(price, self._depth["A"][price], self._counts["A"][price])
                     for price in sorted(self._depth["A"])[:10])
        return MBP10Snapshot(timestamp_ns, bids, asks)

    def order(self, order_id: int) -> _RestingOrder | None:
        """Private adapter inspection only; no identity escapes the L2 model."""
        return self._orders.get(order_id)

    def is_price_in_top_ten(self, side: Literal["B", "A"], price: float) -> bool:
        prices = sorted(self._depth[side], reverse=side == "B")[:10]
        return price in prices

    def is_price_best(self, side: Literal["B", "A"], price: float) -> bool:
        prices = self._depth[side]
        if not prices:
            return False
        return price == (max(prices) if side == "B" else min(prices))

    def apply(
        self,
        event: MBOEvent,
        *,
        materialize_snapshot: bool = True,
        materialize_update: bool = True,
    ) -> tuple[MBP10Snapshot | None, MBP10Update | None]:
        """Apply one private MBO event and optionally render aggregate L2 output.

        Snapshot initialization can contain millions of private order updates.
        The historical adapter exposes no public book until ``F_LAST``, so it
        can safely defer the equivalent MBP-10 rendering until that boundary.
        """
        def result(update: MBP10Update | None) -> tuple[MBP10Snapshot | None, MBP10Update | None]:
            return (
                self.snapshot(event.timestamp_ns) if materialize_snapshot else None,
                update if materialize_update else None,
            )
        if event.action == "R":
            self._orders.clear(); self._depth = {"B": {}, "A": {}}; self._counts = {"B": {}, "A": {}}
            return result(MBP10Update(event.timestamp_ns, event.side, event.price, 0, 0, "RESET"))
        old = self._orders.get(event.order_id)
        if event.action == "T":
            # A trade-only MBO message contains execution evidence but does not
            # identify a documented resting-book transition.  The caller emits
            # its public ``Execution`` separately; the aggregate book is
            # unchanged and no order identity leaves this adapter.
            return result(None)
        if event.action == "A":
            if old is not None or event.size <= 0:
                raise L2ValidationError("invalid MBO add for MBP view")
            self._orders[event.order_id] = _RestingOrder(event.side, event.price, event.size)
            self._adjust(event.side, event.price, event.size, 1)
            update = (
                MBP10Update(event.timestamp_ns, event.side, event.price, event.size, 1, "ADD")
                if materialize_update else None
            )
        elif event.action in {"C", "F"}:
            if old is None or event.size <= 0 or event.size > old.size or old.side != event.side or old.price != event.price:
                raise L2ValidationError("invalid MBO reduction for MBP view")
            remaining = old.size - event.size
            if remaining:
                self._orders[event.order_id] = _RestingOrder(old.side, old.price, remaining)
                count_delta = 0
            else:
                del self._orders[event.order_id]
                count_delta = -1
            self._adjust(old.side, old.price, -event.size, count_delta)
            update = (
                MBP10Update(
                    event.timestamp_ns, old.side, old.price, -event.size, count_delta,
                    "CANCEL" if event.action == "C" else "FILL",
                )
                if materialize_update else None
            )
        elif event.action == "M":
            if old is None or event.size <= 0:
                raise L2ValidationError("invalid MBO modify for MBP view")
            self._adjust(old.side, old.price, -old.size, -1)
            self._orders[event.order_id] = _RestingOrder(event.side, event.price, event.size)
            self._adjust(event.side, event.price, event.size, 1)
            update = (
                MBP10Update(event.timestamp_ns, event.side, event.price, event.size, 1, "MODIFY")
                if materialize_update else None
            )
        else:
            raise L2ValidationError("unsupported MBO action for MBP view")
        return result(update)


@dataclass(frozen=True)
class L2Config:
    """Single predeclared L2-V1 threshold and weight contract.

    The values are monotonic research defaults, not fitted to L3 labels or
    trading outcomes.  They are frozen as ``L2_V1_PREDECLARED_RESEARCH_WEIGHTS``.
    """

    min_relevant_aggressive_volume: int = 50
    min_relevant_execution_count: int = 2
    min_consume_restore_cycles: int = 1
    max_through_level_progress_ticks: float = 4.0
    min_rejection_ticks: float = 0.25
    min_quality_score: float = 0.55
    aggressive_volume_saturation: float = 250.0
    execution_count_saturation: float = 8.0
    restore_cycle_saturation: float = 3.0
    restoration_ratio_saturation: float = 1.0
    rejection_saturation_ticks: float = 3.0
    persistence_depth_saturation: float = 100.0
    restoration_latency_saturation_ms: float = 1_000.0
    multi_level_ofi_saturation: float = 100.0
    rapid_cancel_ns: int = 250_000_000
    aggression_weight: float = 0.28
    restoration_weight: float = 0.25
    price_resistance_weight: float = 0.22
    persistence_weight: float = 0.12
    multi_level_support_weight: float = 0.13
    false_refill_penalty_weight: float = 0.25
    unexecuted_add_penalty_component_weight: float = 0.50
    rapid_cancel_penalty_component_weight: float = 0.30
    adverse_progress_penalty_component_weight: float = 0.20
    weights_label: str = "L2_V1_PREDECLARED_RESEARCH_WEIGHTS"


@dataclass
class _ConsumeRestoreCycle:
    start_ns: int
    depth_before: int
    order_count_before: int
    relevant_execution_volume: int = 0
    minimum_depth: int | None = None
    minimum_order_count: int | None = None
    restored: bool = False
    restored_size: int = 0
    restored_orders: int = 0
    restoration_latency_ns: int | None = None


@dataclass
class L2Interaction:
    interaction_id: str
    level: StructuralLevel
    start_ns: int
    direction: Literal["BUYER_ABSORPTION", "SELLER_ABSORPTION"]
    config: L2Config
    end_ns: int | None = None
    end_price: float | None = None
    termination: str | None = None
    execution_count: int = 0
    relevant_execution_count: int = 0
    directional_aggressive_volume: int = 0
    opposite_aggressive_volume: int = 0
    executed_volume_at_defended_price: int = 0
    executed_volume_within_1_tick: int = 0
    executed_volume_within_2_ticks: int = 0
    first_execution_ns: int | None = None
    last_vicinity_execution_ns: int | None = None
    exit_started_ns: int | None = None
    min_price: float = 0.0
    max_price: float = 0.0
    zone_low: float = 0.0
    zone_high: float = 0.0
    initial_displayed_depth_at_price: int | None = None
    displayed_depth_history: list[int] = field(default_factory=list)
    displayed_order_count_history: list[int] = field(default_factory=list)
    _last_snapshot_ns: int | None = None
    _last_defended_depth: int | None = None
    _last_defended_order_count: int | None = None
    _depth_weighted_sum: float = 0.0
    _depth_weighted_duration_ns: int = 0
    _active_cycle: _ConsumeRestoreCycle | None = None
    _recovery_pending: list[tuple[int, int, int]] = field(default_factory=list)
    depth_restoration_count: int = 0
    restored_depth_volume: int = 0
    restoration_latencies_ns: list[int] = field(default_factory=list)
    restoration_timestamps_ns: list[int] = field(default_factory=list)
    consume_restore_cycles: int = 0
    cumulative_consumed_volume: int = 0
    cumulative_restored_volume: int = 0
    defended_order_count_before_consumption: int | None = None
    minimum_order_count_after_consumption: int | None = None
    restored_order_count: int = 0
    order_count_restoration_cycles: int = 0
    recovery_ratio_by_window: dict[str, float | None] = field(default_factory=lambda: {str(window // 1_000_000): None for window in RECOVERY_WINDOWS_NS})
    unexecuted_add_volume: int = 0
    rapid_cancel_volume: int = 0
    restoration_supported_by_execution_volume: int = 0
    restoration_away_from_defended_price_volume: int = 0
    _passive_adds: list[tuple[int, int]] = field(default_factory=list)
    _previous_snapshot: MBP10Snapshot | None = None
    multi_level_ofi: float = 0.0
    bid_depth_1: int = 0
    ask_depth_1: int = 0
    bid_depth_3: int = 0
    ask_depth_3: int = 0
    bid_depth_5: int = 0
    ask_depth_5: int = 0

    def __post_init__(self) -> None:
        self.min_price = self.max_price = self.zone_low = self.zone_high = self.level.price

    @property
    def defended_side(self) -> Literal["B", "A"]:
        return "B" if self.direction == "BUYER_ABSORPTION" else "A"

    def _is_relevant(self, event: Execution) -> bool:
        return ((self.direction == "BUYER_ABSORPTION" and event.aggressor == "SELL") or
                (self.direction == "SELLER_ABSORPTION" and event.aggressor == "BUY"))

    def observe_execution(self, event: Execution) -> None:
        self.execution_count += 1
        self.first_execution_ns = self.first_execution_ns or event.timestamp_ns
        self.last_vicinity_execution_ns = event.timestamp_ns
        self.min_price = min(self.min_price, event.price); self.max_price = max(self.max_price, event.price)
        self.zone_low = min(self.zone_low, event.price); self.zone_high = max(self.zone_high, event.price)
        if not self._is_relevant(event):
            self.opposite_aggressive_volume += event.size
            return
        self.directional_aggressive_volume += event.size
        self.relevant_execution_count += 1
        distance_ticks = abs(event.price - self.level.price) / TICK
        if distance_ticks == 0:
            self.executed_volume_at_defended_price += event.size
        if distance_ticks <= 1:
            self.executed_volume_within_1_tick += event.size
        if distance_ticks <= 2:
            self.executed_volume_within_2_ticks += event.size
        if event.price == self.level.price and self._last_defended_depth is not None:
            # An execution makes a potential cycle eligible; a subsequent drop
            # must still be observed before any restoration can be counted.
            if self._active_cycle is None or self._active_cycle.restored:
                self._active_cycle = _ConsumeRestoreCycle(
                    event.timestamp_ns, self._last_defended_depth, self._last_defended_order_count or 0,
                )
            self._active_cycle.relevant_execution_volume += event.size

    @staticmethod
    def _ofi_side(previous: tuple[MBPLevel, ...], current: tuple[MBPLevel, ...], *, bid: bool) -> float:
        total = 0.0
        for old, new in zip(previous[:5], current[:5]):
            if bid:
                total += new.size if new.price > old.price else (new.size - old.size if new.price == old.price else -old.size)
            else:
                total += -new.size if new.price < old.price else (-(new.size - old.size) if new.price == old.price else old.size)
        return total

    def _observe_depth_context(self, snapshot: MBP10Snapshot) -> None:
        self.bid_depth_1, self.ask_depth_1 = snapshot.depth("B", 1), snapshot.depth("A", 1)
        self.bid_depth_3, self.ask_depth_3 = snapshot.depth("B", 3), snapshot.depth("A", 3)
        self.bid_depth_5, self.ask_depth_5 = snapshot.depth("B", 5), snapshot.depth("A", 5)
        if self._previous_snapshot is not None:
            self.multi_level_ofi += self._ofi_side(self._previous_snapshot.bids, snapshot.bids, bid=True)
            self.multi_level_ofi += self._ofi_side(self._previous_snapshot.asks, snapshot.asks, bid=False)
        self._previous_snapshot = snapshot

    def _resolve_recoveries(self, timestamp_ns: int, depth: int) -> None:
        pending: list[tuple[int, int, int]] = []
        for target_ns, base_depth, window_ns in self._recovery_pending:
            if timestamp_ns >= target_ns:
                self.recovery_ratio_by_window[str(window_ns // 1_000_000)] = _safe_ratio(depth, base_depth)
            else:
                pending.append((target_ns, base_depth, window_ns))
        self._recovery_pending = pending

    def observe_snapshot(self, snapshot: MBP10Snapshot, update: MBP10Update | None = None) -> None:
        self._observe_depth_context(snapshot)
        level = snapshot.level_at(self.defended_side, self.level.price)
        depth, count = (level.size, level.order_count) if level is not None else (0, 0)
        if self._last_snapshot_ns is not None and self._last_defended_depth is not None:
            duration = max(0, snapshot.timestamp_ns - self._last_snapshot_ns)
            self._depth_weighted_sum += self._last_defended_depth * duration
            self._depth_weighted_duration_ns += duration
        self._resolve_recoveries(snapshot.timestamp_ns, depth)
        if self.initial_displayed_depth_at_price is None:
            self.initial_displayed_depth_at_price = depth
        self.displayed_depth_history.append(depth); self.displayed_order_count_history.append(count)
        previous_depth, previous_count = self._last_defended_depth, self._last_defended_order_count
        self._last_snapshot_ns, self._last_defended_depth, self._last_defended_order_count = snapshot.timestamp_ns, depth, count
        if previous_depth is None:
            return
        cycle = self._active_cycle
        if cycle is not None and not cycle.restored:
            if cycle.minimum_depth is None and depth < previous_depth:
                cycle.minimum_depth, cycle.minimum_order_count = depth, count
                self.cumulative_consumed_volume += max(cycle.relevant_execution_volume, previous_depth - depth)
                self.defended_order_count_before_consumption = cycle.order_count_before
                self.minimum_order_count_after_consumption = count
                for window_ns in RECOVERY_WINDOWS_NS:
                    self._recovery_pending.append((cycle.start_ns + window_ns, max(1, cycle.depth_before), window_ns))
            elif cycle.minimum_depth is not None and depth > cycle.minimum_depth:
                cycle.restored = True
                cycle.restored_size = depth - cycle.minimum_depth
                cycle.restored_orders = max(0, count - (cycle.minimum_order_count or 0))
                cycle.restoration_latency_ns = snapshot.timestamp_ns - cycle.start_ns
                self.depth_restoration_count += 1; self.consume_restore_cycles += 1
                self.restored_depth_volume += cycle.restored_size; self.cumulative_restored_volume += cycle.restored_size
                self.restoration_supported_by_execution_volume += cycle.restored_size
                self.restored_order_count += cycle.restored_orders
                if cycle.restored_orders:
                    self.order_count_restoration_cycles += 1
                self.restoration_latencies_ns.append(cycle.restoration_latency_ns)
                self.restoration_timestamps_ns.append(snapshot.timestamp_ns)
        if update is None or update.side != self.defended_side or update.price != self.level.price:
            if update is not None and update.side == self.defended_side and update.size_delta > 0:
                self.restoration_away_from_defended_price_volume += update.size_delta
            return
        if update.size_delta > 0 and not (cycle and cycle.restored):
            self.unexecuted_add_volume += update.size_delta
            self._passive_adds.append((update.timestamp_ns, update.size_delta))
        elif update.size_delta < 0:
            still_pending: list[tuple[int, int]] = []
            for add_ns, volume in self._passive_adds:
                if update.timestamp_ns - add_ns <= self.config.rapid_cancel_ns:
                    self.rapid_cancel_volume += min(volume, -update.size_delta)
                else:
                    still_pending.append((add_ns, volume))
            self._passive_adds = still_pending

    def _progress(self) -> tuple[float, float, float]:
        if self.direction == "BUYER_ABSORPTION":
            maximum = max(0.0, (self.level.price - self.min_price) / TICK)
            final = max(0.0, (self.level.price - (self.end_price or self.level.price)) / TICK)
            rejection = max(0.0, ((self.end_price or self.level.price) - self.min_price) / TICK)
        else:
            maximum = max(0.0, (self.max_price - self.level.price) / TICK)
            final = max(0.0, ((self.end_price or self.level.price) - self.level.price) / TICK)
            rejection = max(0.0, (self.max_price - (self.end_price or self.level.price)) / TICK)
        return maximum, final, rejection

    def feature_inputs(self) -> dict[str, float | int | None]:
        maximum, final, rejection = self._progress()
        median_depth = float(median(self.displayed_depth_history)) if self.displayed_depth_history else 0.0
        max_depth = float(max(self.displayed_depth_history, default=0))
        depth_mean = _safe_ratio(self._depth_weighted_sum, self._depth_weighted_duration_ns)
        present_fraction = _safe_ratio(sum(depth > 0 for depth in self.displayed_depth_history), len(self.displayed_depth_history))
        imbalance = _safe_ratio(self.directional_aggressive_volume - self.opposite_aggressive_volume,
                                self.directional_aggressive_volume + self.opposite_aggressive_volume)
        initial = float(self.initial_displayed_depth_at_price or 0)
        rapid_cancel_ratio = _safe_ratio(self.rapid_cancel_volume, self.unexecuted_add_volume)
        restoration_ratio = _safe_ratio(self.cumulative_restored_volume, self.cumulative_consumed_volume)
        return {
            "directional_aggressive_volume": self.directional_aggressive_volume,
            "opposite_aggressive_volume": self.opposite_aggressive_volume,
            "aggressive_volume_imbalance": imbalance,
            "execution_count": self.execution_count,
            "relevant_execution_count": self.relevant_execution_count,
            "executed_volume_at_defended_price": self.executed_volume_at_defended_price,
            "executed_volume_within_1_tick": self.executed_volume_within_1_tick,
            "executed_volume_within_2_ticks": self.executed_volume_within_2_ticks,
            "execution_rate": _safe_ratio(self.relevant_execution_count, ((self.last_vicinity_execution_ns or self.start_ns) - self.start_ns) / 1_000_000_000),
            "aggressive_volume_rate": _safe_ratio(self.directional_aggressive_volume, ((self.last_vicinity_execution_ns or self.start_ns) - self.start_ns) / 1_000_000_000),
            "displayed_size_before_consumption": self._active_cycle.depth_before if self._active_cycle else None,
            "size_consumed_by_execution": self.cumulative_consumed_volume,
            "minimum_displayed_size_after_consumption": self._active_cycle.minimum_depth if self._active_cycle else None,
            "restored_size": self.cumulative_restored_volume,
            "restoration_timestamp_ns": self.restoration_timestamps_ns[-1] if self.restoration_timestamps_ns else None,
            "depth_restoration_count": self.depth_restoration_count,
            "restored_depth_volume": self.restored_depth_volume,
            "mean_restoration_latency_ms": _safe_ratio(sum(self.restoration_latencies_ns) / 1_000_000, len(self.restoration_latencies_ns)),
            "median_restoration_latency_ms": median(self.restoration_latencies_ns) / 1_000_000 if self.restoration_latencies_ns else None,
            "fastest_restoration_latency_ms": min(self.restoration_latencies_ns) / 1_000_000 if self.restoration_latencies_ns else None,
            "consume_restore_cycles": self.consume_restore_cycles,
            "cumulative_consumed_volume": self.cumulative_consumed_volume,
            "cumulative_restored_volume": self.cumulative_restored_volume,
            "restoration_to_consumption_ratio": restoration_ratio,
            "cumulative_executed_at_price": self.executed_volume_at_defended_price,
            "initial_displayed_depth_at_price": initial,
            "median_displayed_depth_at_price": median_depth,
            "max_displayed_depth_at_price": max_depth,
            "executed_to_initial_displayed_ratio": _safe_ratio(self.executed_volume_at_defended_price, initial),
            "executed_to_median_displayed_ratio": _safe_ratio(self.executed_volume_at_defended_price, median_depth),
            "maximum_through_level_progress_ticks": maximum,
            "final_through_level_progress_ticks": final,
            "interaction_rejection_ticks": rejection,
            "adverse_progress_per_100_aggressive_contracts": _safe_ratio(maximum * 100, self.directional_aggressive_volume),
            "aggressive_contracts_per_adverse_tick": self.directional_aggressive_volume if maximum == 0 else _safe_ratio(self.directional_aggressive_volume, maximum),
            "defended_price_present_fraction": present_fraction,
            "fraction_of_interaction_with_nonzero_defended_depth": present_fraction,
            "defended_depth_time_weighted_mean": depth_mean,
            "defended_depth_time_weighted_median": median_depth,
            "defended_order_count_before_consumption": self.defended_order_count_before_consumption,
            "minimum_order_count_after_consumption": self.minimum_order_count_after_consumption,
            "restored_order_count": self.restored_order_count,
            "order_count_restoration_cycles": self.order_count_restoration_cycles,
            "bid_depth_1": self.bid_depth_1, "ask_depth_1": self.ask_depth_1,
            "bid_depth_3": self.bid_depth_3, "ask_depth_3": self.ask_depth_3,
            "bid_depth_5": self.bid_depth_5, "ask_depth_5": self.ask_depth_5,
            "depth_imbalance_1": _safe_ratio(self.bid_depth_1 - self.ask_depth_1, self.bid_depth_1 + self.ask_depth_1),
            "depth_imbalance_3": _safe_ratio(self.bid_depth_3 - self.ask_depth_3, self.bid_depth_3 + self.ask_depth_3),
            "depth_imbalance_5": _safe_ratio(self.bid_depth_5 - self.ask_depth_5, self.bid_depth_5 + self.ask_depth_5),
            "multi_level_ofi": self.multi_level_ofi,
            "depth_recovery_100ms": self.recovery_ratio_by_window["100"],
            "depth_recovery_250ms": self.recovery_ratio_by_window["250"],
            "depth_recovery_500ms": self.recovery_ratio_by_window["500"],
            "depth_recovery_1s": self.recovery_ratio_by_window["1000"],
            "unexecuted_add_volume": self.unexecuted_add_volume,
            "rapid_cancel_volume": self.rapid_cancel_volume,
            "rapid_cancel_ratio": rapid_cancel_ratio,
            "restoration_supported_by_execution_ratio": _safe_ratio(self.restoration_supported_by_execution_volume, self.cumulative_restored_volume),
            "restoration_away_from_defended_price_volume": self.restoration_away_from_defended_price_volume,
        }

    def component_scores(self) -> dict[str, float]:
        f = self.feature_inputs()
        directional = float(f["directional_aggressive_volume"])
        opposite = float(f["opposite_aggressive_volume"])
        maximum = float(f["maximum_through_level_progress_ticks"])
        rejection = float(f["interaction_rejection_ticks"])
        imbalance = _clamp01((float(f["aggressive_volume_imbalance"]) + 1.0) / 2.0)
        aggression = (min(1.0, directional / self.config.aggressive_volume_saturation) +
                      min(1.0, float(f["relevant_execution_count"]) / self.config.execution_count_saturation) +
                      imbalance + min(1.0, float(f["executed_to_initial_displayed_ratio"]))) / 4.0
        restoration = (min(1.0, float(f["consume_restore_cycles"]) / self.config.restore_cycle_saturation) +
                       min(1.0, float(f["restoration_to_consumption_ratio"]) / self.config.restoration_ratio_saturation) +
                       _clamp01(float(f["restoration_supported_by_execution_ratio"])) +
                       (1.0 - min(1.0, float(f["mean_restoration_latency_ms"]) / self.config.restoration_latency_saturation_ms))) / 4.0
        resistance = ((1.0 - min(1.0, maximum / self.config.max_through_level_progress_ticks)) +
                      min(1.0, rejection / self.config.rejection_saturation_ticks)) / 2.0
        persistence = (_clamp01(float(f["defended_price_present_fraction"])) +
                       min(1.0, float(f["defended_depth_time_weighted_mean"]) / self.config.persistence_depth_saturation)) / 2.0
        directional_book = float(f["depth_imbalance_5"]) if self.direction == "BUYER_ABSORPTION" else -float(f["depth_imbalance_5"])
        directional_ofi = float(f["multi_level_ofi"]) if self.direction == "BUYER_ABSORPTION" else -float(f["multi_level_ofi"])
        multi = (_clamp01((directional_book + 1.0) / 2.0) + min(1.0, max(0.0, directional_ofi) / self.config.multi_level_ofi_saturation)) / 2.0
        penalty = _clamp01(self.config.unexecuted_add_penalty_component_weight * _safe_ratio(float(f["unexecuted_add_volume"]), directional + 1.0) +
                           self.config.rapid_cancel_penalty_component_weight * float(f["rapid_cancel_ratio"]) +
                           self.config.adverse_progress_penalty_component_weight * min(1.0, maximum / self.config.max_through_level_progress_ticks))
        return {"aggression_score": aggression, "restoration_score": restoration,
                "price_resistance_score": resistance, "persistence_score": persistence,
                "multi_level_support_score": multi, "false_refill_penalty": penalty}

    def quality(self) -> dict[str, float]:
        scores = self.component_scores()
        raw = (scores["aggression_score"] * self.config.aggression_weight +
               scores["restoration_score"] * self.config.restoration_weight +
               scores["price_resistance_score"] * self.config.price_resistance_weight +
               scores["persistence_score"] * self.config.persistence_weight +
               scores["multi_level_support_score"] * self.config.multi_level_support_weight -
               scores["false_refill_penalty"] * self.config.false_refill_penalty_weight)
        return {**scores, "l2_absorption_quality_score": _clamp01(raw)}

    def qualification(self) -> tuple[bool, tuple[str, ...]]:
        features = self.feature_inputs(); quality = self.quality()["l2_absorption_quality_score"]
        reasons: list[str] = []
        if self.directional_aggressive_volume < self.config.min_relevant_aggressive_volume or self.relevant_execution_count < self.config.min_relevant_execution_count:
            reasons.append("INSUFFICIENT_RELEVANT_AGGRESSION")
        if self.consume_restore_cycles < self.config.min_consume_restore_cycles:
            reasons.append("NO_GENUINE_CONSUME_RESTORE")
        if float(features["maximum_through_level_progress_ticks"]) > self.config.max_through_level_progress_ticks and float(features["interaction_rejection_ticks"]) < self.config.min_rejection_ticks:
            reasons.append("PRICE_PROGRESS_NOT_RESISTED")
        if quality < self.config.min_quality_score:
            reasons.append("L2_QUALITY_BELOW_THRESHOLD")
        return not reasons, tuple(reasons)


class L2InteractionEngine:
    """L2-native structural interaction lifecycle driven only by executions."""

    def __init__(self, levels: list[StructuralLevel], config: L2Config = L2Config()) -> None:
        self.levels, self.config = tuple(levels), config
        self.active: dict[str, L2Interaction] = {}
        self.completed: list[L2Interaction] = []
        self._sequence: dict[str, int] = {}

    @staticmethod
    def _direction_for(event: Execution) -> Literal["BUYER_ABSORPTION", "SELLER_ABSORPTION"]:
        if event.aggressor == "SELL": return "BUYER_ABSORPTION"
        if event.aggressor == "BUY": return "SELLER_ABSORPTION"
        raise L2ValidationError("unknown aggressor cannot create an L2 interaction")

    def _open(self, level: StructuralLevel, event: Execution) -> L2Interaction:
        key = f"{level.name}:{level.price:.2f}"; ordinal = self._sequence.get(key, 0) + 1; self._sequence[key] = ordinal
        interaction = L2Interaction(f"{key}:{ordinal:04d}", level, event.timestamp_ns, self._direction_for(event), self.config)
        self.active[interaction.interaction_id] = interaction
        return interaction

    def observe_snapshot(self, snapshot: MBP10Snapshot, update: MBP10Update | None = None) -> None:
        for interaction in self.active.values(): interaction.observe_snapshot(snapshot, update)

    def _close(self, interaction: L2Interaction, timestamp_ns: int, price: float, reason: str) -> None:
        interaction.end_ns, interaction.end_price, interaction.termination = timestamp_ns, _point_price(price), reason
        self.completed.append(interaction); self.active.pop(interaction.interaction_id, None)

    def advance(self, timestamp_ns: int, *, rth_end: bool = False) -> None:
        for interaction in list(self.active.values()):
            if rth_end:
                self._close(interaction, timestamp_ns, interaction.max_price, "RTH_END")
            elif interaction.last_vicinity_execution_ns is not None and timestamp_ns - interaction.last_vicinity_execution_ns >= INACTIVITY_NS:
                self._close(interaction, timestamp_ns, interaction.max_price, "VICINITY_TIMEOUT")

    def observe_execution(self, event: Execution) -> None:
        self.advance(event.timestamp_ns); observed: set[str] = set()
        for interaction in list(self.active.values()):
            if abs(event.price - interaction.level.price) > VICINITY_POINTS:
                if interaction.exit_started_ns is None: interaction.exit_started_ns = event.timestamp_ns
                elif event.timestamp_ns - interaction.exit_started_ns >= EXIT_RESET_NS:
                    self._close(interaction, event.timestamp_ns, event.price, "VICINITY_EXIT_RESET")
                continue
            interaction.exit_started_ns = None; interaction.observe_execution(event); observed.add(interaction.interaction_id)
        for level in self.levels:
            if abs(event.price - level.price) <= VICINITY_POINTS:
                current = next((item for item in self.active.values() if item.level == level), None)
                if current is None: current = self._open(level, event)
                if current.interaction_id not in observed: current.observe_execution(event)

    def finish_rth(self, timestamp_ns: int) -> None:
        self.advance(timestamp_ns, rth_end=True)


def initial_prices(direction: str, bid: float, ask: float, zone_low: float, zone_high: float) -> dict[str, float | str]:
    """Frozen 5-tick zone stop, 3R target, and one-tick adverse fills."""
    bid, ask, zone_low, zone_high = map(_point_price, (bid, ask, zone_low, zone_high))
    if ask <= bid: raise L2ValidationError("invalid executable BBO")
    if direction == "BUYER_ABSORPTION":
        entry, stop = ask + TICK, zone_low - STOP_BUFFER_TICKS * TICK
        return {"direction": "LONG", "entry_reference": ask, "entry": entry, "stop": stop, "stop_exit": stop - TICK, "target": entry + TARGET_R * (entry - stop)}
    if direction == "SELLER_ABSORPTION":
        entry, stop = bid - TICK, zone_high + STOP_BUFFER_TICKS * TICK
        return {"direction": "SHORT", "entry_reference": bid, "entry": entry, "stop": stop, "stop_exit": stop + TICK, "target": entry - TARGET_R * (stop - entry)}
    raise L2ValidationError("unknown absorption direction")


def size_for_instrument(prices: dict[str, float | str], instrument: Literal["ES", "MES"]) -> dict[str, float | int | str]:
    point_value, commission, cap = (ES_POINT_VALUE, ES_COMMISSION, ES_CAP) if instrument == "ES" else (MES_POINT_VALUE, MES_COMMISSION, MES_CAP)
    one_contract = abs(float(prices["entry"]) - float(prices["stop_exit"])) * point_value + 2 * commission
    risk_based = floor(RISK_BUDGET_USD / one_contract); contracts = min(risk_based, cap)
    return {"instrument": instrument, "contracts": contracts, "risk_based_contracts": risk_based, "account_max_contracts": cap,
            "one_contract_initial_risk_usd": one_contract, "estimated_initial_risk_usd": contracts * one_contract}


@dataclass
class L2Setup:
    setup_id: str
    interaction: L2Interaction
    state: str = "WAIT_MIN_CONFIRMATION_TIME"
    confirmation_timestamp_ns: int | None = None
    confirmation_price: float | None = None
    entry_ready_ns: int | None = None
    terminal_reason: str | None = None


@dataclass
class L2Position:
    setup: L2Setup
    instrument: str
    contracts: int
    prices: dict[str, float | str]
    entry_timestamp_ns: int


class L2SignalEngine:
    """Predeclared L2 setup qualification and frozen causal confirmation gate."""

    def __init__(self, config: L2Config = L2Config()) -> None:
        self.config = config; self.pending: dict[str, L2Setup] = {}; self.position: L2Position | None = None
        self.events: list[dict[str, object]] = []; self.rejections: list[dict[str, object]] = []

    def register_completed(self, interaction: L2Interaction) -> L2Setup | None:
        if interaction.end_ns is None or interaction.end_price is None: raise L2ValidationError("only completed interactions can enter L2 qualification")
        accepted, reasons = interaction.qualification(); quality = interaction.quality()
        if not accepted:
            self.rejections.append({"interaction_id": interaction.interaction_id, "reasons": reasons, **quality}); return None
        setup = L2Setup(f"L2:{interaction.interaction_id}", interaction); self.pending[setup.setup_id] = setup
        self.events.append({"setup_id": setup.setup_id, "state": setup.state, "timestamp_ns": interaction.end_ns, **quality, "weights_label": self.config.weights_label})
        return setup

    def observe_execution(self, event: Execution) -> None:
        for setup in self.pending.values():
            if setup.terminal_reason is not None or setup.state == "CONFIRMED": continue
            end = setup.interaction.end_ns or 0; age = event.timestamp_ns - end
            if age > MAX_CONFIRMATION_NS:
                setup.state, setup.terminal_reason = "FAILED", "CONFIRMATION_WINDOW_EXPIRED"; continue
            if age < MIN_CONFIRMATION_NS: continue
            favorable = ((event.price - float(setup.interaction.end_price)) / TICK if setup.interaction.direction == "BUYER_ABSORPTION" else (float(setup.interaction.end_price) - event.price) / TICK)
            if favorable >= 3:
                setup.state = "CONFIRMED"; setup.confirmation_timestamp_ns, setup.confirmation_price = event.timestamp_ns, event.price; setup.entry_ready_ns = event.timestamp_ns + ENTRY_LATENCY_NS
                self.events.append({"setup_id": setup.setup_id, "state": "CONFIRMED", "timestamp_ns": event.timestamp_ns, "favorable_ticks": favorable})

    def advance(self, timestamp_ns: int) -> None:
        for setup in self.pending.values():
            if setup.terminal_reason is None and setup.state != "CONFIRMED" and timestamp_ns > (setup.interaction.end_ns or 0) + MAX_CONFIRMATION_NS:
                setup.state, setup.terminal_reason = "FAILED", "CONFIRMATION_WINDOW_EXPIRED"

    def try_enter(self, setup_id: str, *, timestamp_ns: int, es_bid: float, es_ask: float, mes_bid: float | None = None, mes_ask: float | None = None) -> L2Position | None:
        setup = self.pending[setup_id]
        if setup.state != "CONFIRMED" or setup.entry_ready_ns is None or timestamp_ns < setup.entry_ready_ns: return None
        if self.position is not None:
            setup.state, setup.terminal_reason = "FAILED", "COMPLIANCE_BLOCK_ACTIVE_POSITION"; return None
        prices = initial_prices(setup.interaction.direction, es_bid, es_ask, setup.interaction.zone_low, setup.interaction.zone_high); sizing = size_for_instrument(prices, "ES"); instrument = "ES"
        if int(sizing["contracts"]) < 1 and mes_bid is not None and mes_ask is not None:
            prices = initial_prices(setup.interaction.direction, mes_bid, mes_ask, setup.interaction.zone_low, setup.interaction.zone_high); sizing = size_for_instrument(prices, "MES"); instrument = "MES"
        if int(sizing["contracts"]) < 1:
            setup.state, setup.terminal_reason = "FAILED", "INSUFFICIENT_RISK_BUDGET_FOR_ONE_CONTRACT"; return None
        self.position = L2Position(setup, instrument, int(sizing["contracts"]), prices, timestamp_ns)
        setup.state, setup.terminal_reason = "ENTRY", "ENTRY"; self.events.append({"setup_id": setup.setup_id, "state": "ENTRY", "timestamp_ns": timestamp_ns, "instrument": instrument, "contracts": sizing["contracts"]})
        return self.position


def public_l2_field_names() -> dict[str, tuple[str, ...]]:
    """Audit helper proving the strategy-facing objects have no order identity."""
    return {name: tuple(item.name for item in fields(cls)) for name, cls in {"execution": Execution, "mbp_level": MBPLevel, "mbp_snapshot": MBP10Snapshot, "mbp_update": MBP10Update}.items()}
