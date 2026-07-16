from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from .models import PropRuleSet

ALPHA_ZERO_SOURCE = "https://help.alpha-futures.com/en/articles/11771813-zero-account-overview"
ALPHA_PAYOUT_SOURCE = "https://help.alpha-futures.com/en/articles/9492051-payout-policy"
ALPHA_DLG_SOURCE = "https://help.alpha-futures.com/en/articles/9492014-daily-loss-guard"


def default_alpha_zero_rules() -> dict[str, PropRuleSet]:
    verified = datetime(2026, 7, 16, tzinfo=timezone.utc)
    common = dict(provider="Alpha Futures", version="zero-official-2026-07-16", activation_fee=0, drawdown_behavior="trailing MLL", drawdown_type="marked equity threshold", intraday_or_end_of_day="intraday DLG; provider MLL behavior must be replayed", consistency_rule=None, winning_day_requirements={"qualified": True, "minimum_winning_days": 5, "minimum_profit_per_winning_day": 200}, minimum_trading_days=0, payout_frequency="up to four times per month after eligibility", payout_minimum=200, payout_split=.90, payout_cycle_reset_behavior="retain provider-defined MLL and reset payout-cycle profit", account_cancellation_behavior="trader may cancel evaluation subscription/account", billing_after_failure="subscription continues until cancellation under the account terms", billing_after_pass="no evaluation subscription after pass; qualified terms apply", session_close_requirements="flatten/settle before terminal boundary", prohibited_practices=["account manipulation", "prohibited recovery behavior", "rule circumvention"], automation_restrictions=["automation eligibility must follow current provider terms and platform rules"], maximum_account_allocation=5, rule_source_urls=[ALPHA_ZERO_SOURCE, ALPHA_PAYOUT_SOURCE, ALPHA_DLG_SOURCE], verification_date=verified, unresolved_ambiguities=[], official_verified=True)
    return {
        "Alpha Futures Zero 25K": PropRuleSet(product="Zero 25K", account_size=25000, evaluation_price=79, monthly_subscription=79, reset_fee=69, profit_target=1500, maximum_loss_limit=1000, daily_loss_guard=500, contract_limits={"mini": 1, "micro": 10}, payout_maximum=1000, source_hash="pending", **common),
        "Alpha Futures Zero 50K": PropRuleSet(product="Zero 50K", account_size=50000, evaluation_price=119, monthly_subscription=119, reset_fee=109, profit_target=3000, maximum_loss_limit=2000, daily_loss_guard=1000, contract_limits={"mini": 3, "micro": 30}, payout_maximum=1500, source_hash="pending", **common),
    }


def rule_hash(rule: PropRuleSet) -> str:
    payload = rule.model_dump(mode="json", exclude={"source_hash"})
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verified_rule_registry() -> dict[str, PropRuleSet]:
    result = {}
    for key, rule in default_alpha_zero_rules().items(): result[key] = rule.model_copy(update={"source_hash": rule_hash(rule)})
    return result


def verify_rules(rule: PropRuleSet, *, max_age_days: int = 120, now: datetime | None = None) -> list[str]:
    now = now or datetime.now(timezone.utc)
    errors: list[str] = []
    if not rule.official_verified: errors.append("official rules are not verified")
    if rule.unresolved_ambiguities: errors.extend(f"unresolved ambiguity: {item}" for item in rule.unresolved_ambiguities)
    if rule_hash(rule) != rule.source_hash: errors.append("rule hash mismatch")
    age = (now - rule.verification_date).days
    if age > max_age_days: errors.append(f"rule version is stale: {age} days")
    if not rule.rule_source_urls: errors.append("no official rule source URL")
    return errors
