from __future__ import annotations

import hashlib
import json
import csv
from datetime import datetime, time, timedelta
from decimal import Decimal, ROUND_DOWN
from typing import Any, Literal
from zoneinfo import ZoneInfo

from pydantic import Field

from ..compliance import CalendarArtifact, EconomicEvent, save_calendar_artifact
from ..schemas.strategy_spec import StrictModel

NY = ZoneInfo("America/New_York")
OFFICIAL_EVIDENCE = [
    "https://alpha-futures.com/ (Zero 25K product table; verified 2026-07-31)",
    "https://help.alpha-futures.com/en/articles/9491999-maximum-loss-limit-mll (EOD trailing MLL; verified 2026-07-31)",
    "https://help.alpha-futures.com/en/articles/9492014-daily-loss-guard (daily guard; verified 2026-07-31)",
    "https://help.alpha-futures.com/en/articles/9492051-payout-policy (Zero Qualified 40% consistency; verified 2026-07-31)",
]


class AlphaZero25KPolicy(StrictModel):
    profile: Literal["ZERO_25K_EVALUATION", "ZERO_25K_QUALIFIED"]
    policy_version: str = "alpha-zero-25k-scenario-2026-07-31"
    verification_date: str = "2026-07-31"
    starting_balance: Decimal = Decimal("25000")
    profit_target: Decimal | None = Decimal("1500")
    maximum_loss_distance: Decimal = Decimal("1000")
    daily_loss_guard: Decimal = Decimal("500")
    reset_timezone: str = "America/New_York"
    reset_time: time = time(18, 0)
    maximum_position_reference: str = "1 mini / 10 micros (NOT_COMPARABLE_TO_BINANCE)"
    qualified_news_window_minutes: int = 2
    consistency_limit: Decimal | None = None
    minimum_trading_days: int = 1
    official_evidence_references: list[str] = Field(default_factory=lambda: list(OFFICIAL_EVIDENCE))
    policy_hash: str = "pending"


class AlphaZeroScenarioResult(StrictModel):
    profile: str
    outcome: str
    ending_balance: Decimal
    account_survives: bool
    payout_eligibility: str
    policy_hash: str
    calendar_artifact_hash: str | None = None
    trades: list[dict[str, Any]] = Field(default_factory=list)
    mll_threshold_history: list[dict[str, Any]] = Field(default_factory=list)
    daily_locks: int = 0
    forced_liquidations: int = 0
    skipped_trades: int = 0
    news_blocks: int = 0
    news_violations: int = 0
    breach_count: int = 0
    days_to_pass: int | None = None
    closest_distance_to_mll: Decimal
    maximum_intraday_loss: Decimal
    consistency_percentage: Decimal | None = None
    maximum_contracts: str = "NOT_COMPARABLE_TO_BINANCE"
    warnings: list[str] = Field(default_factory=list)


def _hash_policy(policy: AlphaZero25KPolicy) -> str:
    payload = policy.model_dump(mode="json"); payload.pop("policy_hash", None)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def alpha_zero_25k_policy(profile: Literal["ZERO_25K_EVALUATION", "ZERO_25K_QUALIFIED"]) -> AlphaZero25KPolicy:
    raw = {"profile": profile, "profit_target": Decimal("1500") if profile == "ZERO_25K_EVALUATION" else None, "consistency_limit": None if profile == "ZERO_25K_EVALUATION" else Decimal("0.40"), "minimum_trading_days": 1 if profile == "ZERO_25K_EVALUATION" else 5, "policy_hash": "pending"}
    candidate = AlphaZero25KPolicy.model_validate(raw); raw["policy_hash"] = _hash_policy(candidate)
    return AlphaZero25KPolicy.model_validate(raw)


def _trading_day(value: datetime) -> str:
    local = value.astimezone(NY)
    if local.timetz().replace(tzinfo=None) < time(18):
        local -= timedelta(days=1)
    return local.date().isoformat()


def _risk_quantity(trade: dict[str, Any], risk: Decimal, minimum: Decimal, step: Decimal) -> Decimal | None:
    entry, stop = Decimal(str(trade["entry_price"])), Decimal(str(trade["initial_stop_price"]))
    distance = abs(entry - stop)
    if distance <= 0:
        return None
    result = (risk / distance / step).to_integral_value(rounding=ROUND_DOWN) * step
    return result if result >= minimum else None


def _news_blocked(timestamp: datetime, calendar: CalendarArtifact) -> bool:
    for event in calendar.events:
        if "USD" not in event.affected_currencies or event.impact_level != "HIGH":
            continue
        if event.timestamp - timedelta(minutes=2) <= timestamp <= event.timestamp + timedelta(minutes=2):
            return True
    return False


def run_alpha_zero_scenario(
    trades: list[dict[str, Any]],
    *,
    profile: Literal["ZERO_25K_EVALUATION", "ZERO_25K_QUALIFIED"],
    risk_per_trade_usd: Decimal = Decimal("100"),
    minimum_quantity: Decimal = Decimal("0.001"),
    quantity_step: Decimal = Decimal("0.001"),
    calendar: CalendarArtifact | None = None,
) -> AlphaZeroScenarioResult:
    policy = alpha_zero_25k_policy(profile)
    if profile == "ZERO_25K_QUALIFIED" and calendar is None:
        return AlphaZeroScenarioResult(profile=profile, outcome="NEWS_DATA_UNAVAILABLE", ending_balance=policy.starting_balance, account_survives=True, payout_eligibility="MANUAL_REVIEW_REQUIRED", policy_hash=policy.policy_hash, closest_distance_to_mll=policy.maximum_loss_distance, maximum_intraday_loss=Decimal(), warnings=["Qualified scenario cannot claim news-rule compliance without a historical calendar artifact."])
    balance = policy.starting_balance; eod_high = balance; threshold = balance - policy.maximum_loss_distance
    lock_day: str | None = None; current_day: str | None = None; day_start = balance; daily_net = Decimal(); best_day = Decimal(); pass_day = None
    rows: list[dict[str, Any]] = []; history: list[dict[str, Any]] = []; locks = forced = skipped = news_blocks = breaches = 0; max_intraday_loss = Decimal(); closest = balance - threshold
    for source in sorted(trades, key=lambda item: item["entry_timestamp"]):
        entry_time = datetime.fromisoformat(str(source["entry_timestamp"])); day = _trading_day(entry_time)
        if current_day is not None and day != current_day:
            eod_high = max(eod_high, balance); threshold = min(policy.starting_balance, eod_high - policy.maximum_loss_distance)
            history.append({"trading_day": current_day, "eod_balance": str(balance), "eod_high": str(eod_high), "mll_threshold": str(threshold)})
            current_day, day_start, daily_net, lock_day = day, balance, Decimal(), None
        elif current_day is None:
            current_day = day
        if lock_day == day:
            skipped += 1; rows.append({"trade_id": source.get("trade_id"), "status": "SKIPPED_DAILY_LOCK"}); continue
        if profile == "ZERO_25K_QUALIFIED" and calendar and _news_blocked(entry_time, calendar):
            news_blocks += 1; skipped += 1; rows.append({"trade_id": source.get("trade_id"), "status": "BLOCKED_NEWS_WINDOW"}); continue
        quantity = _risk_quantity(source, risk_per_trade_usd, minimum_quantity, quantity_step)
        if quantity is None:
            skipped += 1; rows.append({"trade_id": source.get("trade_id"), "status": "REJECTED_QUANTITY"}); continue
        source_quantity = Decimal(str(source["quantity"])); scale = quantity / source_quantity
        gross = Decimal(str(source["gross_pnl"])) * scale
        costs = (Decimal(str(source.get("fees", 0))) + Decimal(str(source.get("slippage_cost", 0)))) * scale
        net = gross - costs; balance += net; daily_net += net; best_day = max(best_day, daily_net)
        row = {"trade_id": source.get("trade_id"), "trading_day": day, "quantity": str(quantity), "gross_pnl": str(gross), "costs": str(costs), "net_pnl": str(net), "balance": str(balance), "mll_threshold": str(threshold), "status": "EXECUTED"}
        rows.append(row); closest = min(closest, balance - threshold); max_intraday_loss = min(max_intraday_loss, daily_net)
        if balance <= threshold:
            breaches += 1; row["status"] = "FAILED_MLL"; break
        if daily_net <= -policy.daily_loss_guard:
            locks += 1; forced += 1; lock_day = day; row["status"] = "DAILY_LOSS_LOCK"
        if profile == "ZERO_25K_EVALUATION" and balance - policy.starting_balance >= (policy.profit_target or Decimal("Infinity")) and pass_day is None:
            pass_day = len({item.get("trading_day") for item in rows if item.get("trading_day")})
    if current_day is not None:
        eod_high = max(eod_high, balance); threshold = min(policy.starting_balance, eod_high - policy.maximum_loss_distance)
        history.append({"trading_day": current_day, "eod_balance": str(balance), "eod_high": str(eod_high), "mll_threshold": str(threshold)})
    total_profit = max(Decimal(), balance - policy.starting_balance)
    consistency = (best_day / total_profit) if total_profit > 0 else None
    survives = breaches == 0
    if not survives:
        outcome, payout = "FAILED_MLL", "NOT_ELIGIBLE"
    elif profile == "ZERO_25K_EVALUATION":
        outcome, payout = ("PASSED", "NOT_APPLICABLE") if pass_day is not None else ("IN_PROGRESS", "NOT_APPLICABLE")
    elif len({item.get("trading_day") for item in rows if item.get("status") in {"EXECUTED", "DAILY_LOSS_LOCK"}}) < policy.minimum_trading_days:
        outcome, payout = "SURVIVES", "PAYOUT_NOT_YET_ELIGIBLE"
    elif consistency is not None and consistency <= (policy.consistency_limit or Decimal("1")):
        outcome, payout = "SURVIVES", "PAYOUT_ELIGIBLE"
    else:
        outcome, payout = "SURVIVES", "PAYOUT_NOT_YET_ELIGIBLE"
    return AlphaZeroScenarioResult(profile=profile, outcome=outcome, ending_balance=balance, account_survives=survives, payout_eligibility=payout, policy_hash=policy.policy_hash, calendar_artifact_hash=calendar.artifact_hash if calendar else None, trades=rows, mll_threshold_history=history, daily_locks=locks, forced_liquidations=forced, skipped_trades=skipped, news_blocks=news_blocks, breach_count=breaches, days_to_pass=pass_day, closest_distance_to_mll=closest, maximum_intraday_loss=max_intraday_loss, consistency_percentage=consistency, warnings=["BTCUSDT quantities are not CME mini or micro contracts.", "Scenario output is not Alpha Futures account compliance."])


def rolling_alpha_zero_scenarios(trades: list[dict[str, Any]], *, profile: Literal["ZERO_25K_EVALUATION", "ZERO_25K_QUALIFIED"], risk_levels: tuple[Decimal, ...] = (Decimal("50"), Decimal("100"), Decimal("150"), Decimal("200")), calendar: CalendarArtifact | None = None) -> dict[str, Any]:
    """Fresh-account rolling starts; results are comparisons, never winner selection."""
    ordered = sorted(trades, key=lambda item: item["entry_timestamp"]); results: dict[str, Any] = {}
    starts = sorted({str(item["entry_timestamp"])[:7] for item in ordered})
    for risk in risk_levels:
        windows = [run_alpha_zero_scenario([item for item in ordered if str(item["entry_timestamp"])[:7] >= start], profile=profile, risk_per_trade_usd=risk, calendar=calendar) for start in starts]
        results[str(risk)] = {"window_count": len(windows), "pass_rate": sum(item.outcome == "PASSED" for item in windows) / len(windows) if windows else 0, "mll_breach_rate": sum(not item.account_survives for item in windows) / len(windows) if windows else 0, "daily_lock_rate": sum(item.daily_locks > 0 for item in windows) / len(windows) if windows else 0, "ending_balances": [str(item.ending_balance) for item in windows], "results": [item.model_dump(mode="json") for item in windows]}
    return {"profile": profile, "risk_levels": results, "selection": "NONE; fixed sensitivity comparison only"}


def import_usd_calendar(path: str, output_path: str) -> CalendarArtifact:
    """Import a manually supplied JSON/CSV historical USD-event artifact.

    BTCUSDT mapping is deliberately labelled as a research assumption by the
    caller/report; this routine only preserves the supplied source data.
    """
    from pathlib import Path
    source = Path(path)
    if source.suffix.lower() == ".json":
        raw = json.loads(source.read_text(encoding="utf-8"))
        rows = raw.get("events", raw) if isinstance(raw, dict) else raw
    else:
        with source.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    events = []
    for index, row in enumerate(rows):
        timestamp = datetime.fromisoformat(str(row.get("timestamp") or row.get("timestamp_utc")).replace("Z", "+00:00"))
        events.append(EconomicEvent(event_id=str(row.get("event_id") or f"manual-{index}"), title=str(row.get("title") or "manual USD event"), timestamp=timestamp, impact_level=str(row.get("impact_level") or "HIGH").upper(), affected_currencies=["USD"], affected_instruments=["BTCUSDT"], source=str(row.get("source") or "manual_calendar_import"), retrieved_at=datetime.now(timestamp.tzinfo), source_data_hash=hashlib.sha256(json.dumps(row, sort_keys=True, default=str).encode()).hexdigest()))
    if not events:
        raise ValueError("calendar import contains no events")
    return save_calendar_artifact(min(event.timestamp for event in events), max(event.timestamp for event in events), events, output_path, source="manual_historical_calendar")
