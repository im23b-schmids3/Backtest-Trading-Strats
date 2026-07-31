from __future__ import annotations

import hashlib
import json

from .models import ConfidenceClass, MarketMapping


def default_market_mappings() -> list[MarketMapping]:
    return [
        MarketMapping(strategy_market="BTCUSDT", source_symbol="BTCUSDT", source_provider="Binance USD-M Futures", target_futures_contract="MBT", mapping_method="cross-venue crypto perpetual proxy", native_or_proxy="proxy", reference_price_method="close", duplicate_exposure_group="crypto_btc", date_range={"start": "2019-01-01", "end": "open"}, confidence_level=ConfidenceClass.CRYPTO_PERPETUAL_PROXY, limitations=["Binance BTCUSDT perpetual is not a CME MBT contract", "cross-venue transferability is unverified"]),
        MarketMapping(strategy_market="ETHUSDT", source_symbol="ETHUSDT", source_provider="declared_strategy_source", target_futures_contract="MET", mapping_method="native symbol family", native_or_proxy="native", reference_price_method="close", duplicate_exposure_group="crypto_eth", date_range={"start": "2019-01-01", "end": "open"}, confidence_level=ConfidenceClass.NATIVE_FUTURES_SUPPORTED, limitations=["continuous-contract construction must be verified"]),
        MarketMapping(strategy_market="PAXGUSDT", source_symbol="PAXGUSDT", source_provider="declared_strategy_source", target_futures_contract="MGC", mapping_method="gold proxy", native_or_proxy="proxy", reference_price_method="close", duplicate_exposure_group="gold", date_range={"start": "2019-01-01", "end": "open"}, confidence_level=ConfidenceClass.PROXY_EXPLORATORY, limitations=["PAXG is not native COMEX gold"]),
        MarketMapping(strategy_market="XAGUSDT", source_symbol="XAGUSDT", source_provider="declared_strategy_source", target_futures_contract="SIL", mapping_method="silver proxy", native_or_proxy="proxy", reference_price_method="close", duplicate_exposure_group="silver", date_range={"start": "2019-01-01", "end": "open"}, confidence_level=ConfidenceClass.PROXY_EXPLORATORY, limitations=["spot silver is not native futures"]),
        MarketMapping(strategy_market="SPX", source_symbol="SPX", source_provider="declared_strategy_source", target_futures_contract="MES", mapping_method="index proxy", native_or_proxy="proxy", reference_price_method="close", duplicate_exposure_group="us_equity_index", date_range={"start": "2010-01-01", "end": "open"}, confidence_level=ConfidenceClass.PROXY_EXPLORATORY, limitations=["index proxy is not native CME validation"]),
        MarketMapping(strategy_market="QQQ", source_symbol="QQQ", source_provider="declared_strategy_source", target_futures_contract="MNQ", mapping_method="ETF proxy", native_or_proxy="proxy", reference_price_method="close", duplicate_exposure_group="us_equity_index", date_range={"start": "2010-01-01", "end": "open"}, confidence_level=ConfidenceClass.PROXY_EXPLORATORY, limitations=["ETF proxy is not native CME validation"]),
    ]


def mapping_hash(mappings: list[MarketMapping]) -> str:
    return hashlib.sha256(json.dumps([item.model_dump(mode="json") for item in sorted(mappings, key=lambda value: (value.strategy_market, value.target_futures_contract))], sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_mappings(mappings: list[MarketMapping]) -> list[str]:
    errors: list[str] = []
    for item in mappings:
        if item.strategy_market.upper().startswith("BTC") and item.target_futures_contract == "MET": errors.append("BTC must never map to MET")
        if item.strategy_market.upper().startswith("ETH") and item.target_futures_contract == "MBT": errors.append("ETH must never map to MBT")
        if item.native_or_proxy != "native" and item.confidence_level == ConfidenceClass.NATIVE_FUTURES_SUPPORTED: errors.append(f"proxy mapping is incorrectly labelled native: {item.strategy_market}")
        if item.native_or_proxy != "native" and not item.limitations: errors.append(f"proxy mapping needs limitations: {item.strategy_market}")
    groups: dict[str, list[MarketMapping]] = {}
    for item in mappings: groups.setdefault(item.duplicate_exposure_group, []).append(item)
    for group, values in groups.items():
        if group and len(values) > 1 and group == "us_equity_index" and len({item.target_futures_contract for item in values}) == 1:
            errors.append("duplicate proxy exposure group maps multiple sources to one futures contract")
    return errors
