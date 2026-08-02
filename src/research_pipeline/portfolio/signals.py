from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from typing import Protocol

from .models import ConflictPolicy, PortfolioMember, PortfolioSignalEvent


class PortfolioSignalAdapter(Protocol):
    def signals(self, candidate_id: str, members: Sequence[PortfolioMember], scenario: str) -> Sequence[PortfolioSignalEvent]: ...


class SyntheticPortfolioSignalAdapter:
    """Fixture-only causal events; it never runs a strategy or backtest."""

    def signals(self, candidate_id: str, members: Sequence[PortfolioMember], scenario: str) -> Sequence[PortfolioSignalEvent]:
        start = datetime(2020, 1, 2, 14, tzinfo=timezone.utc)
        events: list[PortfolioSignalEvent] = []
        for member_index, member in enumerate(members):
            count = 24
            if scenario in {"low-frequency", "insufficient-history"}: count = 5 if member_index else 12
            for index in range(count):
                if scenario in {"low-frequency", "insufficient-history"} and (index + member_index) % len(members) != 0: continue
                day = start + timedelta(days=index)
                if scenario in {"complementary", "payout-improving"} and member_index:
                    day = day + timedelta(hours=2)
                if scenario in {"correlated", "redundant", "duplicate-signal", "stress-fragile"}:
                    day = start + timedelta(days=index)
                if scenario == "opposite-conflicts" and member_index % 2: direction = "SHORT"
                else: direction = "LONG"
                if scenario in {"harmful-member", "negative-economics"} and member_index > 0:
                    direction = "SHORT"
                entry = 10000
                exit_price = 11250 if direction == "LONG" else 8750
                if scenario in {"complementary", "payout-improving"} and member_index and index % 2:
                    exit_price = 11000
                if scenario in {"harmful-member", "negative-economics"} and member_index > 0:
                    exit_price = 9000 if direction == "LONG" else 11000
                if scenario == "correlated-loss-shock": exit_price = 9000
                signal_suffix = "" if scenario != "duplicate-signal" else "-duplicate"
                events.append(PortfolioSignalEvent(signal_id=f"{candidate_id}-{member.strategy_id}-{index:03d}{signal_suffix}", strategy_id=member.strategy_id, market=member.markets[0], timeframe=member.timeframes[0], direction=direction, setup_timestamp=day - timedelta(minutes=15), entry_timestamp=day, exit_timestamp=day + timedelta(minutes=30), stop=9500 if direction == "LONG" else 10500, targets=[exit_price], entry_price=entry, exit_price=exit_price, quantity_intent=1, candidate_hash=member.candidate_hash, source_data_classification=member.data_source_classification, duplicate_exposure_group="crypto_btc" if member.markets[0].startswith("BTC") else member.markets[0], regime=["trending", "ranging", "high_volatility", "low_volatility"][index % 4]))
        return sorted(events, key=lambda item: (item.entry_timestamp, item.strategy_id, item.signal_id))


def conflict_group(events: Sequence[PortfolioSignalEvent], window_minutes: int) -> list[list[PortfolioSignalEvent]]:
    groups: list[list[PortfolioSignalEvent]] = []
    for event in sorted(events, key=lambda item: (item.entry_timestamp, item.signal_id)):
        placed = False
        for group in groups:
            anchor = group[0]
            if anchor.market == event.market and abs((event.entry_timestamp - anchor.entry_timestamp).total_seconds()) <= window_minutes * 60:
                group.append(event); placed = True; break
        if not placed: groups.append([event])
    return groups


def apply_conflict_policy(events: Sequence[PortfolioSignalEvent], members: Sequence[PortfolioMember], policy: ConflictPolicy, window_minutes: int = 60) -> tuple[list[PortfolioSignalEvent], dict[str, int]]:
    member_by_id = {member.strategy_id: member for member in members}
    accepted: list[PortfolioSignalEvent] = []
    counts = {"exact_duplicates": 0, "same_direction_overlaps": 0, "opposite_signal_conflicts": 0, "conflict_skips": 0}
    for group in conflict_group(events, window_minutes):
        if len(group) == 1: accepted.extend(group); continue
        unique = {(item.market, item.entry_timestamp, item.direction, item.entry_price, item.exit_price) for item in group}
        counts["exact_duplicates"] += max(0, len(group) - len(unique))
        if len({item.direction for item in group}) == 1:
            counts["same_direction_overlaps"] += len(group) - 1
            if policy == ConflictPolicy.ALLOW_INDEPENDENT: accepted.extend(group)
            elif policy == ConflictPolicy.HIGHEST_CONFIDENCE: accepted.append(max(group, key=lambda item: (member_by_id[item.strategy_id].confidence_score, -member_by_id[item.strategy_id].priority)))
            elif policy == ConflictPolicy.STRATEGY_PRIORITY: accepted.append(min(group, key=lambda item: (member_by_id[item.strategy_id].priority, item.strategy_id)))
            else: accepted.append(min(group, key=lambda item: (item.entry_timestamp, item.strategy_id)))
        else:
            counts["opposite_signal_conflicts"] += 1
            if policy == ConflictPolicy.FIRST_SIGNAL_WINS: accepted.append(min(group, key=lambda item: (item.entry_timestamp, item.strategy_id)))
            elif policy == ConflictPolicy.HIGHEST_CONFIDENCE: accepted.append(max(group, key=lambda item: member_by_id[item.strategy_id].confidence_score))
            elif policy == ConflictPolicy.STRATEGY_PRIORITY: accepted.append(min(group, key=lambda item: (member_by_id[item.strategy_id].priority, item.strategy_id)))
            elif policy == ConflictPolicy.NET_EXPOSURE and len({item.direction for item in group}) == 1: accepted.append(group[0])
            elif policy == ConflictPolicy.ALLOW_INDEPENDENT: accepted.extend(group)
            else: counts["conflict_skips"] += len(group)
    return sorted(accepted, key=lambda item: (item.entry_timestamp, item.strategy_id, item.signal_id)), counts
