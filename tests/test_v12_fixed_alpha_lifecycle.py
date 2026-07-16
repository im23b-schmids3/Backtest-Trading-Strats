import pandas as pd
import pytest

from fib_backtester.research.v12_fixed_alpha_lifecycle import (
    ACCOUNT_SPECS,
    CANONICAL_PROXIES,
    _censoring_summary,
    _run_path,
    _run_stage,
    _transition,
)


def _trade(day, gross, market="ETH", contracts=1, entry=100.0):
    entry_time = pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=10)
    exit_time = entry_time + pd.Timedelta(hours=1)
    return {
        "market": market,
        "entry_timestamp": entry_time,
        "exit_timestamp": exit_time,
        "setup_id": f"test-{day}-{gross}",
        "side": "long",
        "contracts": contracts,
        "entry": entry,
        "entry_fee": 0.0,
        "legs": [{"timestamp": str(exit_time), "reason": "stop", "price": entry + gross / 0.1, "quantity": contracts, "gross": gross, "fee": 0.0, "net": gross}],
    }


def _run_eval(trades, end="2025-02-01"):
    return _run_stage(trades, pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp(end, tz="UTC"), ACCOUNT_SPECS["25K Zero"], "EVALUATION", "test-eval", "test-run")


def _run_qualified(trades, account="25K Zero", end="2025-03-01"):
    return _run_stage(trades, pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp(end, tz="UTC"), ACCOUNT_SPECS[account], "QUALIFIED", "test-qualified", "test-run", initial_balance=ACCOUNT_SPECS[account].account_size)


def test_evaluation_stops_immediately_on_pass_and_cannot_fail_later():
    result = _run_eval([_trade("2025-01-01", 1500), _trade("2025-01-10", -1200)])
    assert result["status"] == "PASSED"
    assert result["passed"] is True
    assert result["failed"] is False
    assert result["lifetime_days"] < 2
    assert all("EVALUATION_FAILED" not in transition["state_after"] for transition in result["transitions"])


def test_evaluation_billing_stops_on_pass_and_qualified_has_no_billing():
    trades = [_trade("2025-01-01", 1500), _trade("2025-01-10", 300)]
    path = _run_path(trades, pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2025-02-01", tz="UTC"), ACCOUNT_SPECS["25K Zero"], "B_REBILL_AFTER_BREACH", "P", "1h", 1, "test-path")
    assert path["evaluations"][0]["status"] == "PASSED"
    assert path["qualified"][0]["qualified_subscription_cost"] == 0.0
    assert path["qualified"][0]["activation_fee"] == 0.0
    assert path["evaluation_subscription_cost_total"] == pytest.approx(79.0)


def test_rebill_is_charged_once_when_failed_evaluation_is_replaced():
    path = _run_path(
        [_trade("2025-01-01", -1000)],
        pd.Timestamp("2025-01-01", tz="UTC"),
        pd.Timestamp("2025-03-01", tz="UTC"),
        ACCOUNT_SPECS["25K Zero"],
        "B_REBILL_AFTER_BREACH",
        "P",
        "1h",
        1,
        "rebill-boundary",
    )
    assert path["evaluation_subscription_cost_total"] == pytest.approx(79.0 * 3)


def test_daily_loss_guard_locks_without_permanent_failure():
    result = _run_eval([_trade("2025-01-01", -500)])
    assert result["failed"] is False
    assert result["status"] == "CENSORED_END_OF_DATA"
    assert any(t["state_after"] == "EVALUATION_DAILY_LOCKED" for t in result["transitions"])


def test_mll_is_permanent_failure():
    result = _run_eval([_trade("2025-01-01", -1000)])
    assert result["failed"] is True
    assert result["status"] == "FAILED"
    assert any(t["state_after"] == "EVALUATION_FAILED" for t in result["transitions"])


def test_five_winning_days_and_consistency_enable_payout():
    trades = [_trade(f"2025-01-{day:02d}", 200) for day in range(1, 6)]
    result = _run_qualified(trades)
    assert result["payout_count"] == 1
    assert result["withdrawal_requested"] == pytest.approx(500.0)
    assert result["trader_payout"] == pytest.approx(450.0)


def test_consistency_blocks_payout_when_largest_day_exceeds_40_percent():
    trades = [_trade("2025-01-01", 1000)] + [_trade(f"2025-01-{day:02d}", 200) for day in range(2, 6)]
    result = _run_qualified(trades)
    assert result["payout_count"] == 0


def test_payout_cap_and_cycle_reset():
    trades = [_trade(f"2025-01-{day:02d}", 800) for day in range(1, 11)]
    result = _run_qualified(trades, account="50K Zero", end="2025-03-01")
    assert result["payout_count"] == 2
    assert result["withdrawal_requested"] == pytest.approx(3000.0)
    assert result["trader_payout"] == pytest.approx(2700.0)


def test_censored_accounts_are_excluded_from_uncensored_lifetime_statistics():
    frame = pd.DataFrame([
        {"lifecycle_stage": "EVALUATION", "account": "25K Zero", "position_size": 2, "status": "FAILED", "censored": False, "lifetime_days": 10, "payout_count": 0, "pass_timestamp": "", "first_payout_timestamp": "", "start": "2025-01-01"},
        {"lifecycle_stage": "EVALUATION", "account": "25K Zero", "position_size": 2, "status": "CENSORED_END_OF_DATA", "censored": True, "lifetime_days": 100, "payout_count": 0, "pass_timestamp": "", "first_payout_timestamp": "", "start": "2025-01-01"},
    ])
    result = _censoring_summary(frame)
    assert result.iloc[0]["median_uncensored_lifetime_days"] == pytest.approx(10)


def test_shared_portfolio_chronology_and_event_balance_are_deterministic():
    trades = [_trade("2025-01-01", 1500)]
    first = _run_eval(trades)
    second = _run_eval(trades)
    assert first["transitions"] == second["transitions"]
    assert first["ending_balance"] == pytest.approx(26_500)
    assert first["ending_balance"] - first["starting_balance"] == pytest.approx(first["net_trading_pnl"])


def test_duplicate_proxy_exposure_is_blocked():
    products = [spec.alpha_product for spec in CANONICAL_PROXIES.values()]
    assert len(products) == len(set(products))
    assert "SPY" not in CANONICAL_PROXIES


def test_daily_loss_guard_unlocks_on_next_trading_session():
    result = _run_eval([_trade("2025-01-01", -500), _trade("2025-01-02", 100)])
    assert any(t["state_after"] == "EVALUATION_DAILY_LOCKED" for t in result["transitions"])
    assert any(
        t["state_before"] == "EVALUATION_DAILY_LOCKED" and t["state_after"] == "EVALUATION_ACTIVE"
        for t in result["transitions"]
    )


def test_qualified_lifecycle_starts_from_passed_evaluation_balance():
    path = _run_path(
        [_trade("2025-01-01", 1500)],
        pd.Timestamp("2025-01-01", tz="UTC"),
        pd.Timestamp("2025-02-01", tz="UTC"),
        ACCOUNT_SPECS["25K Zero"],
        "B_REBILL_AFTER_BREACH",
        "P",
        "1h",
        1,
        "separate-ledgers",
    )
    evaluation = path["evaluations"][0]
    qualified = path["qualified"][0]
    assert evaluation["status"] == "PASSED"
    assert qualified["starting_balance"] == pytest.approx(evaluation["ending_balance"])
    assert qualified["starting_balance"] != pytest.approx(ACCOUNT_SPECS["25K Zero"].account_size)


def test_qualified_zero_has_no_subscription_or_activation_fee():
    result = _run_qualified([_trade("2025-01-01", 200)] * 5)
    assert result.get("qualified_subscription_cost", 0.0) == 0.0
    assert result.get("activation_fee", 0.0) == 0.0


def test_impossible_passed_evaluation_transition_is_rejected():
    with pytest.raises(ValueError):
        _transition([], "invalid", pd.Timestamp("2025-01-01", tz="UTC"), "EVALUATION_PASSED", "EVALUATION_FAILED", "invalid")


def test_event_ledger_reconciles_balance_and_payout_cashflow():
    result = _run_qualified([_trade(f"2025-01-{day:02d}", 200) for day in range(1, 6)])
    balance_delta = sum(float(event["balance_after"]) - float(event["balance_before"]) for event in result["events"])
    event_cashflow = sum(float(event["net_pnl"]) - float(event["payout_request"]) for event in result["events"])
    assert balance_delta == pytest.approx(result["ending_balance"] - result["starting_balance"])
    assert event_cashflow == pytest.approx(result["ending_balance"] - result["starting_balance"])


def test_payout_cycle_counters_reset_after_each_payout():
    result = _run_qualified([_trade(f"2025-01-{day:02d}", 800) for day in range(1, 11)], account="50K Zero")
    payout_events = [event for event in result["events"] if event["event_type"] == "PAYOUT_REQUEST_FILLED"]
    assert len(payout_events) == 2
    assert [event["payout_request"] for event in payout_events] == [1500.0, 1500.0]
    assert pd.Timestamp(payout_events[1]["timestamp"]) > pd.Timestamp(payout_events[0]["timestamp"])


def test_all_recorded_state_transitions_are_marked_valid():
    result = _run_eval([_trade("2025-01-01", 100)])
    assert result["transitions"]
    assert all(transition["valid_transition"] is True for transition in result["transitions"])
