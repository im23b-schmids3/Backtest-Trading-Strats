import pytest

from fib_backtester.strategy.fibonacci import levels


@pytest.mark.parametrize(
    ("side", "entry", "stop", "tp1", "post"),
    [("long", 11.18, 9.8, 12.14, 11.2), ("short", 18.82, 20.2, 17.86, 18.8)],
)
def test_all_fib_prices_follow_explicit_convention(side, entry, stop, tp1, post):
    fib = levels(side, 10, 20)
    assert fib.entry == pytest.approx(entry)
    assert fib.stop == pytest.approx(stop)
    assert fib.targets == pytest.approx((tp1, 13.82 if side == "long" else 16.18, 15 if side == "long" else 15, 17.64 if side == "long" else 12.36, 19.5 if side == "long" else 10.5))
    assert fib.post_tp1_stop == pytest.approx(post)


def test_invalid_fib_range_is_rejected():
    with pytest.raises(ValueError):
        levels("long", 20, 10)
