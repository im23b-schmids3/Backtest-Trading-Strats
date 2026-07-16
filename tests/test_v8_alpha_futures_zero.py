import pandas as pd

from fib_backtester.research.v8_alpha_futures_zero import (
    ACCOUNT_SIZE,
    CONTRACT_SIZES,
    SIZE_CASES,
    SPECS,
    _allocate_contracts,
    _feasible_starts,
    _spec,
)


def test_official_contract_specs_are_asset_specific():
    assert SPECS[("ETH", "micros")].multiplier == 0.10
    assert SPECS[("ETH", "micros")].tick_value == 0.05
    assert SPECS[("SOL", "micros")].multiplier == 25.0
    assert SPECS[("SOL", "micros")].tick_value == 1.25
    assert _spec("ETH", 10).contracts == 10
    assert _spec("ETH", 1).contract_type == "mini"
    assert _spec("SOL", 1).multiplier == 500.0
    assert SIZE_CASES[-1] == ("1 Mini", "mini", 1)


def test_contract_sizes_never_exceed_zero_25k_limit():
    assert CONTRACT_SIZES == (2, 3, 5, 7, 10)
    assert max(CONTRACT_SIZES) <= SPECS[("ETH", "micros")].max_position
    assert _spec("ETH", 1).contracts <= SPECS[("ETH", "mini")].max_position


def test_integer_contract_allocation_preserves_position_size():
    for size in CONTRACT_SIZES:
        allocation = _allocate_contracts(size, (.30, .25, .20, .15, .10))
        assert sum(allocation) == size
        assert all(value >= 0 for value in allocation)


def test_feasible_starts_require_historical_runway():
    index = pd.date_range("2022-01-01", "2024-12-31", freq="D", tz="UTC")
    starts = _feasible_starts(index)
    assert starts
    assert starts[0] >= index[0] + pd.Timedelta(days=180)
    assert starts[-1] <= index[-1] - pd.Timedelta(days=180)
    assert ACCOUNT_SIZE == 25_000
