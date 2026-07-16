from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd


@dataclass(frozen=True)
class Swing:
    kind: Literal["high", "low"]
    pivot_index: int
    confirmation_index: int
    price: float
    pivot_time: pd.Timestamp
    confirmation_time: pd.Timestamp


def confirmed_swings(ohlcv: pd.DataFrame, n: int, tie_policy: Literal["strict", "allow"] = "strict") -> list[Swing]:
    """Emit each pivot only on its confirmation bar (i + n), preventing look-ahead."""
    if n < 1:
        raise ValueError("n must be positive")
    result: list[Swing] = []
    highs, lows = ohlcv["high"].to_numpy(), ohlcv["low"].to_numpy()
    for i in range(n, len(ohlcv) - n):
        left_h, right_h = highs[i - n:i], highs[i + 1:i + n + 1]
        left_l, right_l = lows[i - n:i], lows[i + 1:i + n + 1]
        if tie_policy == "strict":
            high_ok = highs[i] > max(left_h.max(), right_h.max())
            low_ok = lows[i] < min(left_l.min(), right_l.min())
        else:
            high_ok = highs[i] >= max(left_h.max(), right_h.max())
            low_ok = lows[i] <= min(left_l.min(), right_l.min())
        if high_ok:
            result.append(Swing("high", i, i + n, float(highs[i]), ohlcv.index[i], ohlcv.index[i + n]))
        if low_ok:
            result.append(Swing("low", i, i + n, float(lows[i]), ohlcv.index[i], ohlcv.index[i + n]))
    return sorted(result, key=lambda swing: (swing.confirmation_index, swing.pivot_index))
