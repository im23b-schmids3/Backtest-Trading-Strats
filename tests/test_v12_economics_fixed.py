import pandas as pd

from fib_backtester.research import v12_economics_fixed as economics
from fib_backtester.research import v12_fixed_alpha_lifecycle as fixed


def _trade(day, gross):
    entry = pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=10)
    exit_time = entry + pd.Timedelta(hours=1)
    return {
        "market": "ETH",
        "entry_timestamp": entry,
        "exit_timestamp": exit_time,
        "setup_id": f"econ-{day}-{gross}",
        "side": "long",
        "contracts": 1,
        "entry": 100.0,
        "entry_fee": 0.0,
        "legs": [{"timestamp": str(exit_time), "reason": "stop", "price": 100 + gross / 0.1, "quantity": 1, "gross": gross, "fee": 0.0, "net": gross}],
    }


def test_canonical_proxy_mapping_has_no_btc_eth_or_spx_spy_collision():
    assert fixed.CANONICAL_PROXIES["BTC"].alpha_product == "MBT"
    assert fixed.CANONICAL_PROXIES["ETH"].alpha_product == "MET"
    assert fixed.CANONICAL_PROXIES["S&P proxy"].alpha_product == "MES"
    assert "SPY" not in fixed.CANONICAL_PROXIES
    assert len({proxy.alpha_product for proxy in fixed.CANONICAL_PROXIES.values()}) == len(fixed.CANONICAL_PROXIES)


def test_evaluation_passes_and_stage_stops():
    result = economics._run_stage(
        [_trade("2025-01-01", 1500), _trade("2025-01-10", -1200)],
        pd.Timestamp("2025-01-01", tz="UTC"),
        pd.Timestamp("2025-02-01", tz="UTC"),
        economics.ACCOUNT_SPECS["25K Zero"],
        "EVALUATION",
        "MAX_30_DAYS",
    )
    assert result["status"] == "PASSED"
    assert result["terminal"] < pd.Timestamp("2025-01-10", tz="UTC")


def test_fixed_policy_cancels_unresolved_evaluation():
    result = economics._run_stage(
        [],
        pd.Timestamp("2025-01-01", tz="UTC"),
        pd.Timestamp("2025-01-31", tz="UTC"),
        economics.ACCOUNT_SPECS["25K Zero"],
        "EVALUATION",
        "MAX_30_DAYS",
    )
    assert result["status"] == "VOLUNTARY_CANCEL"


def test_first_qualified_payout_ends_qualified_stage():
    result = economics._run_stage(
        [_trade(f"2025-01-{day:02d}", 200) for day in range(1, 6)],
        pd.Timestamp("2025-01-01", tz="UTC"),
        pd.Timestamp("2025-03-01", tz="UTC"),
        economics.ACCOUNT_SPECS["25K Zero"],
        "QUALIFIED",
    )
    assert result["status"] == "FIRST_PAYOUT"
    assert result["payout_received"] == 450.0


def test_subscription_is_calendar_month_based():
    start = pd.Timestamp("2025-01-01", tz="UTC")
    assert economics._billing_months(start, start) == 1
    assert economics._billing_months(start, pd.Timestamp("2025-02-01", tz="UTC")) == 2
