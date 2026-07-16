from fib_backtester.strategy.swings import confirmed_swings
from conftest import ohlcv


def test_high_is_confirmed_only_n_bars_later():
    bars = ohlcv([1, 2, 3, 10, 3, 2, 1], [0, 1, 2, 4, 2, 1, 0])
    swings = confirmed_swings(bars, 2)
    high = next(s for s in swings if s.kind == "high")
    assert high.pivot_index == 3
    assert high.confirmation_index == 5
    assert high.confirmation_time == bars.index[5]


def test_equal_highs_are_not_strict_swings():
    bars = ohlcv([1, 2, 5, 5, 2, 1], [0, 1, 2, 2, 1, 0])
    assert not [s for s in confirmed_swings(bars, 2) if s.kind == "high"]


def test_final_unconfirmable_pivot_never_emitted():
    bars = ohlcv([1, 2, 3, 10], [0, 1, 2, 4])
    assert not confirmed_swings(bars, 2)
