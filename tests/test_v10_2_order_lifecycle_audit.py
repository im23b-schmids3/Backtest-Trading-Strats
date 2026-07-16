import pandas as pd

from fib_backtester.research.v10_2_order_lifecycle_audit import _termination_category, _timestamp_distance_to_close


def test_lifecycle_termination_categories_are_explicit():
    assert _termination_category("active_swing_extreme_updated").startswith("New higher")
    assert _termination_category("anchor_max_age") == "Anchor invalidation"
    assert _termination_category("anchor_low_broken") == "New swing invalidation"
    assert _termination_category("conflicting_open_position") == "Position conflict"
    assert _termination_category("session_close") == "Daily session close cancelled the order"


def test_session_close_window_uses_europe_berlin():
    timestamp = pd.Timestamp("2024-01-01 21:30:00+00:00")
    assert _timestamp_distance_to_close(timestamp) == 0
