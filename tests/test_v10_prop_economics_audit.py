from types import SimpleNamespace

from fib_backtester.research.v10_prop_economics_audit import ACCOUNT_SPECS, _costed


def test_official_zero_account_economics_are_distinct_by_size():
    assert ACCOUNT_SPECS["25k"]["subscription"] == 79.0
    assert ACCOUNT_SPECS["50k"]["subscription"] == 119.0
    assert ACCOUNT_SPECS["100k"]["subscription"] == 239.0
    assert ACCOUNT_SPECS["25k"]["payout_max"] == 1000.0
    assert ACCOUNT_SPECS["100k"]["payout_max"] == 2500.0


def test_subscription_stops_when_account_closes_after_first_payout():
    row = SimpleNamespace(
        account_size="25k",
        lifetime_days=365.25,
        payout_amounts="900|900",
        payout_offsets="30|60",
        subscription_cost=79.0,
        challenge_fees=0.0,
        reset_fees=0.0,
        commissions=0.0,
    )
    result = _costed(row, "B", 1)
    assert result["closure_days"] == 30.0
    assert result["subscription_cost"] == 79.0
    assert result["net_profit"] == 821.0


def test_natural_end_subscription_uses_evaluation_lifetime_cost_model():
    row = SimpleNamespace(
        account_size="25k",
        lifetime_days=365.25,
        payout_amounts="900",
        payout_offsets="30",
        subscription_cost=79.0,
        challenge_fees=0.0,
        reset_fees=0.0,
        commissions=0.0,
    )
    result = _costed(row, "A")
    assert result["closure_days"] == 365.25
    assert result["subscription_cost"] == 948.0
