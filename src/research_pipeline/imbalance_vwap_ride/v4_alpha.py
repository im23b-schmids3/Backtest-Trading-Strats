from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .alpha_proxy import is_mbt_entry_available, run_alpha_proxy
from .artifacts import sha256_value
from .v4_models import PHASE_B_MONTHS, alpha_eligibility

MINIMUM_BOOTSTRAP_PATHS = 20_000


def validate_alpha_rules_artifact(
    rules_artifact: dict[str, Any] | None,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    artifact = rules_artifact if isinstance(rules_artifact, dict) else {}
    content = {key: value for key, value in artifact.items() if key != "content_hash"}
    content_hash_valid = bool(artifact.get("content_hash")) and artifact.get("content_hash") == sha256_value(content)
    probe = {**artifact, "consistent": bool(artifact.get("consistent")) and content_hash_valid}
    eligibility = alpha_eligibility(
        locked_test_status="LOCKED_TEST_PASSED",
        phase_b_execution_count=1,
        frozen_candidate_valid=True,
        proxy_eligible_trade_count=30,
        rules_artifact=probe,
        now=now,
    )
    return {
        "valid": eligibility["checks"]["official_alpha_futures_rules"]
        and eligibility["checks"]["rules_fresh"]
        and eligibility["checks"]["rules_consistent"],
        "content_hash_valid": content_hash_valid,
        "checks": {
            key: eligibility["checks"][key]
            for key in (
                "official_alpha_futures_rules",
                "rules_fresh",
                "rules_consistent",
            )
        },
    }


def proxy_eligible_phase_b_trades(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for trade in trades:
        if trade.get("direction") != "LONG":
            raise ValueError("V4 Alpha input contains a non-long trade")
        month = str(trade.get("entry_timestamp", ""))[:7]
        if month not in PHASE_B_MONTHS:
            raise ValueError("V4 Alpha input must contain Phase-B-only trades")
        if is_mbt_entry_available(trade["entry_timestamp"]):
            output.append(trade)
    return output


def run_v4_alpha_proxy(
    trades: list[dict[str, Any]],
    bars: list[dict[str, Any]],
    *,
    locked_test_status: str,
    phase_b_execution_count: int,
    frozen_candidate_valid: bool,
    rules_artifact: dict[str, Any] | None,
    paths: int = MINIMUM_BOOTSTRAP_PATHS,
    seed: int | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Run the limited one-MBT proxy only after every V4 eligibility gate."""

    if paths < MINIMUM_BOOTSTRAP_PATHS:
        raise ValueError("V4 Alpha proxy requires at least 20,000 bootstrap paths")
    eligible_trades = proxy_eligible_phase_b_trades(trades)
    rules_validation = validate_alpha_rules_artifact(rules_artifact, now=now)
    rules = dict(rules_artifact or {})
    if not rules_validation["valid"]:
        rules["consistent"] = False
    eligibility = alpha_eligibility(
        locked_test_status=locked_test_status,
        phase_b_execution_count=phase_b_execution_count,
        frozen_candidate_valid=frozen_candidate_valid,
        proxy_eligible_trade_count=len(eligible_trades),
        rules_artifact=rules,
        now=now or datetime.now(timezone.utc),
    )
    if not eligibility["eligible"]:
        return {
            "status": "NOT_EXECUTED",
            "reason": eligibility["reason"],
            "eligibility": eligibility,
            "rules_validation": rules_validation,
            "alpha_executed": False,
            "confirmation_evidence": False,
            "contract_accurate_fills_claimed": False,
        }
    report = run_alpha_proxy(eligible_trades, bars, paths=paths, seed=seed)
    return {
        "status": "EXECUTED_LIMITED_ONE_MBT_PROXY",
        "alpha_executed": True,
        "eligibility": eligibility,
        "official_rules_artifact": rules_artifact,
        "rules_validation": rules_validation,
        "instrument_mapping": "BTCUSDT_SIGNAL_TO_ONE_MBT_0P1_BTC_PROXY",
        "phase_b_only": True,
        "paths": paths,
        "chronological_report": report["evaluation"]["chronological"],
        "daily_block_bootstrap_report": report["evaluation"],
        "evaluation_report": report["evaluation"],
        "qualified_report": report["qualified"],
        "payout_report": {
            "total_gross_withdrawals": report["qualified"]["total_gross_withdrawals"],
            "total_trader_share": report["qualified"]["total_trader_share"],
            "second_payout_probability": report["qualified"]["second_payout_probability"],
            "minimum_payout_buffer": report["qualified"]["minimum_payout_buffer"],
        },
        "sensitivity_report": report["sensitivities"],
        "mapping_report": report["mapping"],
        "confirmation_evidence": False,
        "contract_accurate_fills_claimed": False,
        "requires_external_live_or_contract_accurate_confirmation": True,
    }


__all__ = [
    "MINIMUM_BOOTSTRAP_PATHS",
    "proxy_eligible_phase_b_trades",
    "run_v4_alpha_proxy",
    "validate_alpha_rules_artifact",
]
