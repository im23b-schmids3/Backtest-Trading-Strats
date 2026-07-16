import pandas as pd

from fib_backtester.research.v11_5_proxy_market_expansion import (
    FROZEN_DISTANCE,
    FROZEN_MIN_MOVE,
    PROXIES,
    _splits,
)


def test_v11_5_uses_documented_proxy_set_and_intraday_only():
    assert len(PROXIES) == 9
    assert PROXIES["ETH"]["ticker"] == "ETH-USD"
    assert PROXIES["QQQ"]["alpha_market"] == "MNQ/NQ"
    assert PROXIES["EURUSD"]["proxy_type"] == "FX spot"


def test_v11_5_keeps_frozen_global_parameters():
    assert FROZEN_DISTANCE == 4
    assert FROZEN_MIN_MOVE == 0.0025


def test_v11_5_split_is_chronological_60_20_20():
    start = pd.Timestamp("2024-07-15T00:00:00Z")
    end = pd.Timestamp("2026-07-14T00:00:00Z")
    splits = _splits(start, end)
    assert [row[0] for row in splits] == ["training", "validation", "holdout"]
    assert splits[0][1] == start
    assert splits[0][2] < splits[1][1] < splits[1][2] < splits[2][1] < splits[2][2]

