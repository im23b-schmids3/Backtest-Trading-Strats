import json

import pytest

from fib_backtester.research.v12_contract_registry import CONTRACTS, PROXY_TO_CONTRACT, mapped_price, resolve_proxy_symbol, round_to_tick
from fib_backtester.research.v12_fixed_alpha_lifecycle import _prepare_trade


def test_btc_and_eth_have_one_canonical_mapping():
    assert resolve_proxy_symbol("BTCUSDT") == "MBT"
    assert resolve_proxy_symbol("ETHUSDT") == "MET"
    assert PROXY_TO_CONTRACT["BTC"] == "MBT"
    assert PROXY_TO_CONTRACT["BTC"] != "MET"


def test_legacy_runner_reads_the_same_btc_mapping():
    from fib_backtester.research.v12_binance_proxy_prop_simulation import PROXY_SPECS

    assert PROXY_SPECS["BTC"].alpha_product == "MBT"
    assert PROXY_SPECS["ETH"].alpha_product == "MET"
    assert PROXY_SPECS["Silver"].tick_size == pytest.approx(0.01)
    assert PROXY_SPECS["Silver"].tick_value == pytest.approx(10.0)


def test_silver_contract_metadata_and_tick_pnl():
    spec = CONTRACTS["SIL"]
    assert spec.multiplier == pytest.approx(1000.0)
    assert spec.tick_size == pytest.approx(0.01)
    assert spec.tick_value == pytest.approx(10.0)
    assert (round_to_tick(25.01, spec.tick_size) - round_to_tick(25.00, spec.tick_size)) * spec.multiplier == pytest.approx(10.0)
    assert (round_to_tick(25.10, spec.tick_size) - round_to_tick(25.00, spec.tick_size)) * spec.multiplier == pytest.approx(100.0)


@pytest.mark.parametrize("contracts", [2, 5, 7, 10])
@pytest.mark.parametrize("side,sign", [("long", 1), ("short", -1)])
def test_silver_direction_and_contract_scaling(contracts, side, sign):
    move = 0.10
    expected = sign * move * CONTRACTS["SIL"].multiplier * contracts
    assert expected == pytest.approx(sign * 100.0 * contracts)


def test_return_based_mes_mapping_and_tick_rounding():
    context = {"mode": "SYNTHETIC_RETURN_MAPPED_PROXY", "proxy_anchor": 1.0, "mapped_anchor": 6000.0}
    assert mapped_price(1.01, "S&P proxy", context) == pytest.approx(6060.0)
    assert round_to_tick(mapped_price(1.01, "S&P proxy", context), CONTRACTS["MES"].tick_size) == pytest.approx(6060.0)

    one_contract = (6060.0 - 6000.0) * CONTRACTS["MES"].multiplier
    assert one_contract == pytest.approx(300.0)
    assert one_contract * 2 == pytest.approx(600.0)
    assert (-60.0) * CONTRACTS["MES"].multiplier == pytest.approx(-300.0)
    assert (-1) * (-60.0) * CONTRACTS["MES"].multiplier == pytest.approx(300.0)


def test_index_conversion_applies_to_partial_exit_legs():
    raw = {
        "entry_price": 1.0,
        "initial_stop": 0.98,
        "fill_timestamp": "2025-01-01T10:00:00Z",
        "exit_timestamp": "2025-01-01T12:00:00Z",
        "side": "long",
        "setup_id": "test-index",
        "average_exit_price": 1.01,
        "exit_reason": "tp2",
        "exit_events": json.dumps([
            {"reason": "tp1", "fill_price": 1.005, "timestamp": "2025-01-01T11:00:00Z"},
            {"reason": "tp2", "fill_price": 1.01, "timestamp": "2025-01-01T12:00:00Z"},
        ]),
    }
    trade = _prepare_trade(raw, "S&P proxy", 2, {"mode": "SYNTHETIC_RETURN_MAPPED_PROXY", "proxy_anchor": 1.0, "mapped_anchor": 6000.0})
    assert sum(leg["quantity"] for leg in trade["legs"]) == 2
    assert trade["gross_pnl"] == pytest.approx((30.0 * 5.0) + (60.0 * 5.0))


def test_silver_partial_exit_quantities_and_direction():
    raw = {
        "entry_price": 25.00,
        "fill_timestamp": "2025-01-01T10:00:00Z",
        "exit_timestamp": "2025-01-01T12:00:00Z",
        "side": "short",
        "setup_id": "test-silver",
        "average_exit_price": 24.90,
        "exit_reason": "tp2",
        "exit_events": json.dumps([
            {"reason": "tp1", "fill_price": 24.99, "timestamp": "2025-01-01T11:00:00Z"},
            {"reason": "tp2", "fill_price": 24.90, "timestamp": "2025-01-01T12:00:00Z"},
        ]),
    }
    trade = _prepare_trade(raw, "Silver", 2)
    assert sum(leg["quantity"] for leg in trade["legs"]) == 2
    assert trade["gross_pnl"] == pytest.approx((0.01 * 1000) + (0.10 * 1000))
