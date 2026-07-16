import pandas as pd

from fib_backtester.research import v13_1_risk_account_exit_policy as research
from fib_backtester.research import v13_risk_managed as v13
from fib_backtester.research import v12_fixed_alpha_lifecycle as fixed


T0 = pd.Timestamp("2025-01-02 12:00:00", tz="UTC")
T1 = pd.Timestamp("2025-01-03 12:00:00", tz="UTC")


def _replay(policy="A_FIXED_RISK", risk_cap=250.0, price=100.0):
    marks = {"BTC": {T0: price, T1: price}, "ETH": {T0: price, T1: price}, "Gold": {T0: price, T1: price}}
    return research.PolicyReplay([], [T0, T1], T1 + pd.Timedelta(hours=4), {m: price for m in v13.MEMBERS}, marks, [T0, T1], [], risk_cap=risk_cap, policy=policy)


def _account(replay):
    return replay._buy_evaluation(T0, "test")


def _row(risk=100.0):
    return {"market": "BTC", "entry_timestamp": T0, "raw": {"signal_timestamp": T0, "fill_timestamp": T0, "entry_price": 100.0, "initial_stop": 90.0, "side": "long", "exit_events": [], "average_exit_price": 100.0, "exit_timestamp": T1, "setup_id": "S1"}, "stop": 90.0, "stop_distance": 10.0, "stop_ticks": 2.0, "tick_size": 5.0, "tick_value": 0.1, "dollar_risk_per_contract": risk, "signal_id": 1}


def test_contract_count_rounds_down_for_each_cap(monkeypatch):
    monkeypatch.setattr(research.fixed, "_prepare_trade", lambda raw, market, contracts, context=None: {"market": market, "contracts": contracts})
    rows = [_row(100.0)]
    assert [research.build_signals(rows, {}, cap)[0]["contracts"] for cap in (150.0, 200.0, 250.0, 300.0)] == [1, 2, 2, 3]


def test_frozen_stop_price_never_changes(monkeypatch):
    monkeypatch.setattr(research.fixed, "_prepare_trade", lambda raw, market, contracts, context=None: {"market": market, "contracts": contracts})
    signal = research.build_signals([_row(100.0)], {}, 250.0)[0]
    assert signal["stop"] == 90.0


def test_risk_tier_changes_contract_count_only(monkeypatch):
    monkeypatch.setattr(research.fixed, "_prepare_trade", lambda raw, market, contracts, context=None: {"market": market, "contracts": contracts})
    low = research.build_signals([_row(100.0)], {}, 150.0)[0]
    high = research.build_signals([_row(100.0)], {}, 300.0)[0]
    assert low["stop"] == high["stop"] == 90.0
    assert low["raw"]["entry_price"] == high["raw"]["entry_price"] == 100.0
    assert low["contracts"] != high["contracts"]


def test_drawdown_zones_use_marked_equity():
    assert research._zone(24600.0, 25000.0, 24000.0) == "GREEN"
    assert research._zone(24500.0, 25000.0, 24000.0) == "YELLOW"
    assert research._zone(24300.0, 25000.0, 24000.0) == "RED"
    assert research._zone(24200.0, 25000.0, 24000.0) == "CRITICAL"


def test_policy_c_cancels_at_800_exactly():
    replay = _replay("C_CANCEL_AT_800", price=100.0)
    account = _account(replay)
    account.balance = 24200.0
    replay._check_lifecycle(account, T0, "mark")
    assert account.voluntary_cancel_timestamp == T0
    assert account.state == "CLOSED"


def test_cancelled_account_stops_billing_immediately():
    replay = _replay("C_CANCEL_AT_800")
    account = _account(replay)
    paid = account.subscription_paid
    replay._cancel_evaluation(account, T0)
    replay._charge_subscriptions(account, T1 + pd.DateOffset(months=2))
    assert account.subscription_paid == paid


def test_replacement_is_scheduled_once():
    replay = _replay("C_CANCEL_AT_800")
    account = _account(replay)
    replay._cancel_evaluation(account, T0)
    replay._cancel_evaluation(account, T0)
    assert sum(len(values) for values in replay.replacements.values()) == 1


def test_policy_d_allows_one_recovery_attempt():
    replay = _replay("D_ONE_RECOVERY", risk_cap=250.0)
    account = _account(replay)
    account.recovery_pending = True
    signal = {"signal_id": 1, "entry_timestamp": T0, "market": "BTC", "raw": {"signal_timestamp": T0, "fill_timestamp": T0, "entry_price": 100.0, "initial_stop": 90.0, "side": "long", "exit_events": [], "average_exit_price": 100.0, "exit_timestamp": T1, "setup_id": "S1"}, "risk_per_contract": 50.0, "conversion_context": None, "stop_distance": 10.0, "dollar_risk_per_contract": 50.0}
    replay._accept(account, signal, T0)
    assert account.recovery_attempts == 1
    assert not account.recovery_pending


def test_policy_e_respects_remaining_mll_buffer():
    replay = _replay("E_BUFFER_LIMITED_RECOVERY", risk_cap=300.0)
    account = _account(replay)
    account.mll = 24000.0
    account.balance = 24250.0
    account.recovery_pending = True
    signal = {"signal_id": 1, "entry_timestamp": T0, "market": "BTC", "raw": {"signal_timestamp": T0, "fill_timestamp": T0, "entry_price": 100.0, "initial_stop": 90.0, "side": "long", "exit_events": [], "average_exit_price": 100.0, "exit_timestamp": T1, "setup_id": "S1"}, "risk_per_contract": 100.0, "conversion_context": None, "stop_distance": 10.0, "dollar_risk_per_contract": 100.0}
    replay.signals.append(signal)
    replay._accept(account, signal, T0)
    assert account.recovery_attempts == 1
    assert account.policy_risk_caps[-1] <= 250.0


def test_failed_account_is_not_billed_or_routed():
    replay = _replay()
    account = _account(replay)
    paid = account.subscription_paid
    account.mll = account.balance + 1.0
    replay._check_lifecycle(account, T0, "mll")
    replay._charge_subscriptions(account, T1 + pd.DateOffset(months=2))
    assert account.subscription_paid == paid
    assert account.state.endswith("FAILED")


def test_daily_two_loss_lock_resets():
    replay = _replay()
    account = _account(replay)
    account.current_session = research.v13._session(T0)
    account.daily_losses = 2
    account.daily_stop = True
    replay._advance(T1)
    assert account.daily_losses == 0
    assert not account.daily_stop


def test_mll_breach_leaves_no_active_account():
    replay = _replay()
    account = _account(replay)
    account.mll = account.balance + 1.0
    replay._check_lifecycle(account, T0, "mll")
    assert account.state == "EVALUATION_FAILED"


def test_qualified_payout_split_is_unchanged():
    assert fixed.legacy.PAYOUT_SPLIT == 0.90


def test_policy_metrics_external_cashflow_reconciles():
    replay = _replay()
    account = _account(replay)
    df = research._account_summary(replay, 250.0, "A_FIXED_RISK")
    metrics = research._metrics(replay, df, 250.0, "A_FIXED_RISK", T1)
    assert metrics["net_external_cashflow"] == -metrics["subscription_cost"]
