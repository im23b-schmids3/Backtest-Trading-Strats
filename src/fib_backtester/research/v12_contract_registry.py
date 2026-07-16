"""Single canonical contract and proxy registry for the V12 research harness.

This module contains metadata/conversion infrastructure only. It does not alter
signal generation or trading rules.
"""

from __future__ import annotations

from dataclasses import dataclass


VERIFICATION_DATE = "2026-07-15"


@dataclass(frozen=True)
class ContractSpec:
    symbol: str
    contract_unit: str
    multiplier: float
    tick_size: float
    tick_value: float
    dollar_value_per_point: float
    micro_mini_relationship: str
    official_source: str
    verification_date: str = VERIFICATION_DATE


CONTRACTS = {
    "MBT": ContractSpec("MBT", "0.10 BTC", 0.10, 5.00, 0.50, 0.10, "1/50 of BTC", "CME Micro Bitcoin fact card"),
    "MET": ContractSpec("MET", "0.10 ETH", 0.10, 0.50, 0.05, 0.10, "1/500 of standard Ether (50 ETH)", "CME Micro Ether FAQ"),
    "MGC": ContractSpec("MGC", "10 troy ounces", 10.0, 0.10, 1.00, 10.0, "1/10 of GC", "CME Micro Metals specifications"),
    "SIL": ContractSpec("SIL", "1,000 troy ounces", 1000.0, 0.01, 10.00, 1000.0, "1/5 of SI", "CME Micro Metals specifications"),
    "MES": ContractSpec("MES", "1 index contract", 5.0, 0.25, 1.25, 5.0, "1/10 of ES", "CME Micro E-mini FAQ"),
    "MNQ": ContractSpec("MNQ", "1 index contract", 2.0, 0.25, 0.50, 2.0, "1/10 of NQ", "CME Micro E-mini FAQ"),
}


PROXY_TO_CONTRACT = {
    "BTC": "MBT",
    "ETH": "MET",
    "Gold": "MGC",
    "Silver": "SIL",
    "QQQ": "MNQ",
    "S&P proxy": "MES",
}

PROXY_SYMBOLS = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "Gold": "PAXGUSDT",
    "Silver": "XAGUSDT",
    "QQQ": "QQQUSDT",
    "S&P proxy": "SPXUSDT",
}

INDEX_REFERENCE_LEVELS = {"MES": 6000.0, "MNQ": 20_000.0}
INDEX_PROXY_MODE = "SYNTHETIC_RETURN_MAPPED_PROXY"


def contract_for_proxy(market: str) -> ContractSpec:
    return CONTRACTS[PROXY_TO_CONTRACT[market]]


def resolve_proxy_symbol(proxy_symbol: str) -> str:
    normalized = proxy_symbol.upper().replace("-", "")
    if normalized in {"BTC", "BTCUSDT", "BITCOIN", "BITCOINUSDT"}:
        return "MBT"
    if normalized in {"ETH", "ETHUSDT", "ETHER", "ETHERUSDT"}:
        return "MET"
    raise KeyError(f"unsupported proxy symbol: {proxy_symbol}")


def round_to_tick(price: float, tick_size: float) -> float:
    return round(float(price) / tick_size) * tick_size


def mapped_price(proxy_price: float, market: str, context: dict | None = None) -> float:
    contract = contract_for_proxy(market)
    if contract.symbol not in INDEX_REFERENCE_LEVELS:
        return float(proxy_price)
    if not context or context.get("mode") != INDEX_PROXY_MODE:
        raise ValueError(f"{market} requires {INDEX_PROXY_MODE} conversion context")
    anchor_proxy = float(context["proxy_anchor"])
    anchor_futures = float(context["mapped_anchor"])
    if anchor_proxy <= 0:
        raise ValueError("proxy anchor must be positive")
    return anchor_futures * (float(proxy_price) / anchor_proxy)


def build_synthetic_context(first_proxy_prices: dict[str, float]) -> dict[str, dict]:
    contexts = {}
    for market, proxy_anchor in first_proxy_prices.items():
        contract = contract_for_proxy(market)
        if contract.symbol in INDEX_REFERENCE_LEVELS:
            contexts[market] = {"mode": INDEX_PROXY_MODE, "proxy_anchor": float(proxy_anchor), "mapped_anchor": INDEX_REFERENCE_LEVELS[contract.symbol]}
        else:
            contexts[market] = {"mode": "DIRECT_PRICE_LEVEL", "proxy_anchor": float(proxy_anchor), "mapped_anchor": float(proxy_anchor)}
    return contexts


def registry_rows() -> list[dict]:
    rows = []
    for market, product in PROXY_TO_CONTRACT.items():
        spec = CONTRACTS[product]
        rows.append({"proxy_market": market, "proxy_symbol": PROXY_SYMBOLS[market], "mapped_futures_symbol": product, "contract_unit": spec.contract_unit, "contract_multiplier": spec.multiplier, "minimum_tick": spec.tick_size, "tick_value": spec.tick_value, "dollar_value_per_point": spec.dollar_value_per_point, "micro_mini_relationship": spec.micro_mini_relationship, "official_source": spec.official_source, "verification_date": spec.verification_date, "conversion_mode": INDEX_PROXY_MODE if product in INDEX_REFERENCE_LEVELS else "DIRECT_PRICE_LEVEL"})
    return rows
