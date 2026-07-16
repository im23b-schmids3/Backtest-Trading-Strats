import pandas as pd

from fib_backtester.strategy.signals import setup_from_swings
from fib_backtester.strategy.swings import Swing


def test_inverted_alternating_pivots_do_not_make_nonsensical_setup():
    t = pd.Timestamp("2025-01-01", tz="UTC")
    low = Swing("low", 0, 2, 20, t, t)
    high = Swing("high", 10, 12, 10, t, t)
    assert setup_from_swings(low, high, 5) is None
