"""Causal, order-id keyed MBO book reconstruction; no market-data mutation."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


class BookStateError(ValueError):
    """A record cannot be applied without inventing book state."""


@dataclass(frozen=True)
class Order:
    side: str
    price: int
    size: int


@dataclass(frozen=True)
class Applied:
    action: str
    side: str
    price: int
    size: int
    order_id: int
    executed: bool = False


class CausalMBOBook:
    """A strict L3 book. Modify is interpreted as an absolute replacement size."""

    def __init__(self) -> None:
        self.orders: dict[int, Order] = {}
        self.depth: dict[str, dict[int, int]] = {"B": defaultdict(int), "A": defaultdict(int)}
        # Databento sequence numbers are channel-scoped, not a single global
        # monotone counter. Raw iterator order remains the applied order.
        self.last_order: dict[int, tuple[int, int]] = {}

    def _sequence(self, channel_id: int, sequence: int, ts_recv: int) -> None:
        previous = self.last_order.get(channel_id)
        if previous is not None and ts_recv < previous[1]:
            raise BookStateError("timestamp regressed in source ordering")
        if previous is not None and ts_recv == previous[1] and sequence < previous[0]:
            raise BookStateError("sequence regressed within receive timestamp")
        self.last_order[channel_id] = (sequence, ts_recv)

    def _adjust(self, side: str, price: int, delta: int) -> None:
        value = self.depth[side][price] + delta
        if value < 0:
            raise BookStateError("negative displayed depth")
        if value:
            self.depth[side][price] = value
        else:
            self.depth[side].pop(price, None)

    def apply(self, *, action: str, side: str, price: int, size: int, order_id: int,
              sequence: int, ts_recv: int, channel_id: int = 0,
              validate_sequence: bool = True, mutate_execution: bool = True) -> Applied | None:
        if validate_sequence:
            self._sequence(channel_id, sequence, ts_recv)
        if action == "R":
            # Provider clear/reset marker: causally discard prior displayed state.
            self.orders.clear(); self.depth = {"B": defaultdict(int), "A": defaultdict(int)}
            return None
        if action not in {"A", "C", "M", "T", "F"}:
            raise BookStateError(f"unsupported MBO action {action!r}")
        if action == "T":
            # A trade record reports an execution but does not, by itself,
            # document a resting-order lifecycle transition. Its order_id is
            # therefore not required to be an active displayed order.
            return Applied(action, side, price, size, order_id, executed=True)
        if order_id <= 0 or size < 0:
            raise BookStateError("invalid order fields")
        if action in {"A", "C", "M"} and side not in {"A", "B"}:
            raise BookStateError("invalid displayed-order side")
        old = self.orders.get(order_id)
        if action == "A":
            if old is not None:
                raise BookStateError("duplicate active order add")
            if size <= 0:
                raise BookStateError("non-positive add")
            self.orders[order_id] = Order(side, price, size); self._adjust(side, price, size)
            return Applied(action, side, price, size, order_id)
        if action == "F" and not mutate_execution:
            # Feed F can coexist with a later cancel of the same displayed
            # quantity. In that declared mode it is execution evidence only.
            return Applied(action, side, price, size, order_id, executed=True)
        if side not in {"A", "B"}:
            raise BookStateError("invalid fill side")
        if old is None:
            raise BookStateError("operation for unknown active order")
        if side != old.side:
            raise BookStateError("side transition is unsupported")
        if action == "M":
            if size <= 0:
                raise BookStateError("non-positive modify")
            self._adjust(old.side, old.price, -old.size)
            self._adjust(side, price, size)
            self.orders[order_id] = Order(side, price, size)
            return Applied(action, side, price, size, order_id)
        if price != old.price:
            raise BookStateError("cancel/fill price does not match active order")
        if size <= 0 or size > old.size:
            raise BookStateError("invalid cancel/fill size")
        self._adjust(old.side, old.price, -size)
        remaining = old.size - size
        if remaining:
            self.orders[order_id] = Order(old.side, old.price, remaining)
        else:
            del self.orders[order_id]
        return Applied(action, side, price, size, order_id, executed=action in {"T", "F"})

    def best_bid(self) -> int | None:
        return max(self.depth["B"], default=None)

    def best_ask(self) -> int | None:
        return min(self.depth["A"], default=None)

    def spread(self) -> int | None:
        bid, ask = self.best_bid(), self.best_ask()
        return None if bid is None or ask is None else ask - bid
