import pandas as pd

from fib_backtester.research import v13_risk_managed as v13
from fib_backtester.research import v12_fixed_alpha_lifecycle as fixed


T0 = pd.Timestamp("2025-01-02 12:00:00", tz="UTC")
T1 = pd.Timestamp("2025-01-03 12:00:00", tz="UTC")


def _replay(price=100.0, timeline=None, session_closes=None):
    timeline = timeline or [T0, T1]
    marks = {"BTC": {timestamp: price for timestamp in timeline}}
    return v13.Replay([], timeline, T1 + pd.Timedelta(hours=4), {"BTC": price, "ETH": price, "Gold": price}, marks, timeline, session_closes or [])


def _account(replay):
    return replay._buy_evaluation(T0, "test")


def _position(account, price=100.0, quantity=1, side="long", entry_fee=0.0):
    trade = {
        "market": "BTC", "alpha_product": "MBT", "side": side, "entry": 100.0,
        "initial_stop": 90.0, "contracts": quantity, "stop_distance": 10.0,
        "stop_ticks": 10.0, "tick_size": 0.01, "tick_value": 0.1,
        "dollar_risk_per_contract": 1.0, "initial_risk": float(quantity),
        "entry_timestamp": T0, "setup_timestamp": str(T0), "entry_fee": entry_fee,
        "legs": [], "balance_before": account.balance,
    }
    account.positions[1] = v13.Position(trade, 1, quantity, price, entry_fee=entry_fee)
    return trade


def _signal(account, trade):
    return {"signal_id": 2, "entry_timestamp": T0, "market": "BTC", "contracts": trade["contracts"], "stop_distance": 10.0, "dollar_risk_per_contract": 1.0, "raw": {"side": trade["side"]}, "trade": trade}


def test_open_position_is_marked_to_equity():
    replay = _replay(105.0)
    account = _account(replay)
    _position(account)
    equity, _ = replay._mark_account(account, T0)
    multiplier = fixed.CANONICAL_PROXIES["BTC"].multiplier
    assert equity > account.balance
    assert equity == account.balance + 5.0 * multiplier - 105.0 * multiplier * 0.001


def test_equity_mll_breach_fails_before_terminal():
    replay = _replay(-5000.0)
    account = _account(replay)
    account.mll = account.balance - 1.0
    _position(account)
    replay._check_lifecycle(account, T0, "mark")
    assert account.state == "EVALUATION_FAILED"
    assert account.failure_timestamp == T0


def test_dlg_breach_locks_before_terminal():
    replay = _replay(-5000.0)
    account = _account(replay)
    account.mll = -1_000_000_000.0
    _position(account)
    replay._check_lifecycle(account, T0, "mark")
    assert account.state == "EVALUATION_DAILY_LOCKED"
    assert not account.positions
    assert account.dlg_breach_count == 1


def test_daily_locked_account_resumes_next_day():
    replay = _replay()
    account = _account(replay)
    account.state = "EVALUATION_DAILY_LOCKED"
    account.current_session = v13._session(T0)
    replay._advance(T1)
    assert account.state == "EVALUATION_ACTIVE"


def test_normal_session_cutoff_closes_positions_during_replay():
    replay = _replay(101.0)
    account = _account(replay)
    _position(account)
    replay._flatten(account, T0, "session_forced_liquidation")
    assert not account.positions
    assert len(replay.trades) == 1
    assert replay.trades[0]["exit_reason"] == "session_forced_liquidation"


def test_session_close_pnl_triggers_pass_check():
    replay = _replay(110.0)
    account = _account(replay)
    account.balance = account.initial_balance + fixed.ACCOUNT_SPECS["25K Zero"].target - 0.5
    _position(account)
    replay._flatten(account, T0, "session_forced_liquidation")
    replay._check_lifecycle(account, T0, "forced_session_close")
    assert account.pass_timestamp == T0


def test_evaluation_pass_is_at_earliest_valid_timestamp():
    replay = _replay()
    account = _account(replay)
    account.balance = account.initial_balance + fixed.ACCOUNT_SPECS["25K Zero"].target
    replay._check_lifecycle(account, T0, "realized")
    assert account.pass_timestamp == T0


def test_passed_evaluation_stops_billing():
    replay = _replay()
    account = _account(replay)
    paid = account.subscription_paid
    account.balance = account.initial_balance + fixed.ACCOUNT_SPECS["25K Zero"].target
    replay._check_lifecycle(account, T0, "target")
    replay._charge_subscriptions(account, T1 + pd.DateOffset(months=2))
    assert account.subscription_paid == paid


def test_failed_evaluation_stops_billing():
    replay = _replay()
    account = _account(replay)
    paid = account.subscription_paid
    account.mll = account.balance + 1.0
    replay._check_lifecycle(account, T0, "mll")
    replay._charge_subscriptions(account, T1 + pd.DateOffset(months=2))
    assert account.subscription_paid == paid


def test_failed_accounts_are_removed_from_routing():
    replay = _replay()
    account = _account(replay)
    account.state = "EVALUATION_FAILED"
    trade = _position(account)
    replay._accept(account, _signal(account, trade), T0)
    assert replay.skipped[-1]["reason"] == "inactive account"


def test_failure_schedules_one_replacement():
    replay = _replay()
    account = _account(replay)
    account.mll = account.balance + 1.0
    replay._check_lifecycle(account, T0, "mll")
    replay._fail(account, T0, "second failure attempt")
    next_time = replay.trading_times[1]
    assert replay.replacements[next_time] == ["replacement Evaluation after failure"]


def test_two_loss_rule_resets_next_day():
    replay = _replay()
    account = _account(replay)
    account.current_session = v13._session(T0)
    account.daily_losses = 2
    account.daily_stop = True
    replay._advance(T1)
    assert account.daily_losses == 0
    assert not account.daily_stop


def test_marked_equity_uses_position_quantity():
    replay = _replay(110.0)
    account = _account(replay)
    _position(account, quantity=2)
    one, _ = replay._mark_account(account, T0)
    account.positions[1].remaining = 1
    two, _ = replay._mark_account(account, T0)
    assert one - two > 0


def test_terminal_flatten_does_not_create_delayed_transition():
    replay = _replay()
    account = _account(replay)
    account.mll = account.balance + 1.0
    replay._check_lifecycle(account, T0, "mll")
    assert account.failure_timestamp == T0
    assert not any("terminal" in str(event["event_reason"]).lower() for event in replay.events)


def test_subscription_totals_follow_lifecycle_dates():
    replay = _replay()
    account = _account(replay)
    account.billing_next = T0 + pd.DateOffset(months=1)
    replay._charge_subscriptions(account, T0 + pd.DateOffset(months=2))
    assert account.subscription_paid == 3 * fixed.ACCOUNT_SPECS["25K Zero"].subscription


def test_no_account_remains_active_after_mll_breach():
    replay = _replay()
    account = _account(replay)
    account.mll = account.balance + 1.0
    replay._check_lifecycle(account, T0, "mll")
    assert account.state.endswith("FAILED")


def test_each_month_has_one_purchase_marker():
    times = [pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2025-01-15", tz="UTC"), pd.Timestamp("2025-02-01", tz="UTC")]
    replay = _replay(timeline=times)
    assert len(replay.month_first) == 2
    assert replay.month_first["2025-01"] == times[0]


def test_forced_session_balance_reconciles():
    replay = _replay(110.0)
    account = _account(replay)
    trade = _position(account)
    before = account.balance
    replay._flatten(account, T0, "session_forced_liquidation")
    leg = replay.trades[0]
    assert account.balance == before + leg["net_pnl"]
    assert leg["quantity_reconciles"]
