import pandas as pd

from fib_backtester.research.v9_alpha_risk_engine import (
    DEFAULT_FORCED_LIQUIDATION,
    DEFAULT_SESSION_CUTOFF,
    _apply_session_rules,
    _parse_time,
    _risk_decision,
    _spec,
)


def _trade(entry_timestamp):
    return {
        "entry_timestamp": entry_timestamp,
        "side": "long",
        "entry": 100.0,
        "stop": 110.0,
        "risk": 2.0,
        "entry_fee": 0.02,
        "contracts": 2,
        "contract_type": "micro",
        "multiplier": 0.10,
        "fee_rate": 0.001,
        "asset": "ETH",
        "legs": [{"timestamp": "2024-01-02T00:00:00+00:00", "reason": "stop", "price": 99.0, "quantity": 2, "gross": -0.2, "fee": 0.2, "net": -0.4}],
    }


def test_default_session_times_are_configurable_values():
    assert _parse_time(DEFAULT_SESSION_CUTOFF).hour == 22
    assert _parse_time(DEFAULT_FORCED_LIQUIDATION).minute == 30


def test_session_cutoff_rejects_entries_at_or_after_cutoff():
    bars = pd.DataFrame({"open": [100.0], "close": [100.0]}, index=pd.DatetimeIndex(["2024-01-01T22:00:00Z"]))
    result, skipped = _apply_session_rules(_trade("2024-01-01T21:30:00+00:00"), bars, _parse_time("22:20"), _parse_time("22:30"), "Europe/Berlin")
    assert result is None
    assert skipped is True


def test_forced_liquidation_replaces_later_exit_causally():
    bars = pd.DataFrame(
        {"open": [100.0, 101.0], "close": [100.0, 101.0]},
        index=pd.DatetimeIndex(["2024-01-01T20:00:00Z", "2024-01-01T22:00:00Z"]),
    )
    result, skipped = _apply_session_rules(_trade("2024-01-01T20:00:00+00:00"), bars, _parse_time("22:20"), _parse_time("22:30"), "Europe/Berlin")
    assert skipped is False
    assert result["forced_exit"] is True
    assert result["legs"][-1]["reason"] == "session_forced_exit"
    assert result["legs"][-1]["timestamp"] == "2024-01-01 22:00:00+00:00"


def test_risk_policies_reduce_or_block_only_at_documented_thresholds():
    spec = _spec("ETH", "micros", 10)
    assert _risk_decision("A", spec, 100.0, 1000.0, False, False)[0] == 10
    assert _risk_decision("B", spec, 249.0, 1000.0, False, False)[0] == 5
    assert _risk_decision("C", spec, 174.0, 1000.0, False, False)[0] == 1
    assert _risk_decision("D", spec, 124.0, 1000.0, False, False)[0] == 0
    assert _risk_decision("E", spec, 500.0, 299.0, False, False)[0] == 5
    assert _risk_decision("F", spec, 500.0, 199.0, False, False)[0] == 0
