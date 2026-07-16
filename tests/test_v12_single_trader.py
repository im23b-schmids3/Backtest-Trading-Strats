import pandas as pd
import pytest

from fib_backtester.research.v12_single_trader import SingleTrader


def _trader(mode="MIRRORED"):
    return SingleTrader("25K Zero", "ETH only", "1h", 2, mode, [])


def test_only_one_evaluation_can_be_active():
    trader = _trader()
    trader._buy_evaluation(pd.Timestamp("2025-01-01", tz="UTC"), "test")
    with pytest.raises(AssertionError):
        trader._buy_evaluation(pd.Timestamp("2025-01-01", tz="UTC"), "duplicate")


def test_failed_evaluation_gets_one_replacement():
    trader = _trader()
    trader._buy_evaluation(pd.Timestamp("2025-01-01", tz="UTC"), "test")
    trader._handle_eval_failure(pd.Timestamp("2025-01-02", tz="UTC"), "MLL")
    assert trader.total_evaluations_purchased == 2
    assert trader.eval is not None


def test_passed_evaluation_becomes_qualified():
    trader = _trader()
    trader._buy_evaluation(pd.Timestamp("2025-01-01", tz="UTC"), "test")
    trader.eval.balance += trader.spec.target
    trader._handle_eval_pass(pd.Timestamp("2025-01-02", tz="UTC"))
    assert trader.total_qualified_started == 1
    assert trader.qualified is not None


def test_pass_creates_new_evaluation_for_mirrored_mode():
    trader = _trader("MIRRORED")
    trader._buy_evaluation(pd.Timestamp("2025-01-01", tz="UTC"), "test")
    trader.eval.balance += trader.spec.target
    trader._handle_eval_pass(pd.Timestamp("2025-01-02", tz="UTC"))
    assert trader.eval is not None
    assert trader.qualified is not None


def test_at_most_one_qualified_account_is_active():
    trader = _trader()
    trader._buy_evaluation(pd.Timestamp("2025-01-01", tz="UTC"), "test")
    assert trader._start_qualified(pd.Timestamp("2025-01-02", tz="UTC"), 25_000)
    assert not trader._start_qualified(pd.Timestamp("2025-01-03", tz="UTC"), 25_000)
    assert trader.pending_passes == 1


def test_evaluation_billing_stops_after_pass():
    trader = _trader()
    trader._buy_evaluation(pd.Timestamp("2025-01-01", tz="UTC"), "test")
    trader.eval.balance += trader.spec.target
    trader._handle_eval_pass(pd.Timestamp("2025-01-02", tz="UTC"))
    before = sum(row["related_cost_or_payout"] for row in trader.transitions if row["reason"] == "Evaluation monthly subscription rebill")
    trader._charge_subscription(pd.Timestamp("2025-03-01", tz="UTC"))
    after = sum(row["related_cost_or_payout"] for row in trader.transitions if row["reason"] == "Evaluation monthly subscription rebill")
    # The additional charge belongs to the replacement Evaluation's new
    # billing lifecycle; the passed Evaluation itself receives no rebill.
    assert after - before == pytest.approx(trader.spec.subscription)


def test_new_evaluation_starts_new_billing_lifecycle():
    trader = _trader()
    trader._buy_evaluation(pd.Timestamp("2025-01-01", tz="UTC"), "test")
    trader._handle_eval_failure(pd.Timestamp("2025-01-02", tz="UTC"), "MLL")
    charges = [row for row in trader.transitions if row["reason"] == "Evaluation monthly subscription rebill"]
    assert len(charges) == 2


def test_qualified_and_evaluation_balances_are_separate():
    trader = _trader()
    trader._buy_evaluation(pd.Timestamp("2025-01-01", tz="UTC"), "test")
    trader.eval.balance += trader.spec.target
    trader._handle_eval_pass(pd.Timestamp("2025-01-02", tz="UTC"))
    assert trader.eval is not trader.qualified
    assert trader.qualified.balance != trader.eval.balance


def test_monthly_cashflow_reconciles_to_year_summary():
    monthly, yearly, _ = _trader().run()
    assert monthly["net_monthly_cashflow"].sum() == pytest.approx(yearly.iloc[0]["net_external_cashflow"])


def test_single_trader_run_has_one_year_summary_not_many_paths():
    monthly, yearly, _ = _trader().run()
    assert len(yearly) == 1
    assert len(monthly) == 12
