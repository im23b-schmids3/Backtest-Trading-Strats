from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from .models import ContractSpec


def default_contract_registry() -> dict[str, ContractSpec]:
    verified = datetime(2026, 7, 16, tzinfo=timezone.utc)
    source = "https://www.cmegroup.com/markets.html"
    return {
        "MBT": ContractSpec(exchange="CME", symbol="MBT", product_name="Micro Bitcoin Futures", contract_unit="0.1 BTC", minimum_tick=5.0, tick_value=0.50, point_value=0.10, currency="USD", session_definition="CME crypto session", micro_mini_relationship="1 MBT = 1/10 BTC futures contract", official_source=source, verification_date=verified, data_quality="REFERENCE_ONLY", active=True),
        "MET": ContractSpec(exchange="CME", symbol="MET", product_name="Micro Ether Futures", contract_unit="0.1 ETH", minimum_tick=0.25, tick_value=0.025, point_value=0.10, currency="USD", session_definition="CME crypto session", micro_mini_relationship="1 MET = 1/10 Ether futures contract", official_source=source, verification_date=verified, data_quality="REFERENCE_ONLY", active=True),
        "MGC": ContractSpec(exchange="COMEX", symbol="MGC", product_name="Micro Gold Futures", contract_unit="10 troy oz", minimum_tick=0.10, tick_value=1.0, point_value=10.0, currency="USD", session_definition="CME metals session", micro_mini_relationship="1 MGC = 1/10 GC", official_source=source, verification_date=verified, data_quality="REFERENCE_ONLY", active=True),
        "SIL": ContractSpec(exchange="COMEX", symbol="SIL", product_name="Micro Silver Futures", contract_unit="1,000 troy oz", minimum_tick=0.005, tick_value=5.0, point_value=1000.0, currency="USD", session_definition="CME metals session", micro_mini_relationship="micro silver product", official_source=source, verification_date=verified, data_quality="REFERENCE_ONLY", active=True),
        "MES": ContractSpec(exchange="CME", symbol="MES", product_name="Micro E-mini S&P 500", contract_unit="$5 x S&P 500 Index", minimum_tick=0.25, tick_value=1.25, point_value=5.0, currency="USD", session_definition="CME equity index session", micro_mini_relationship="1 MES = 1/10 ES", official_source=source, verification_date=verified, data_quality="REFERENCE_ONLY", active=True),
        "MNQ": ContractSpec(exchange="CME", symbol="MNQ", product_name="Micro E-mini Nasdaq-100", contract_unit="$2 x Nasdaq-100 Index", minimum_tick=0.25, tick_value=0.50, point_value=2.0, currency="USD", session_definition="CME equity index session", micro_mini_relationship="1 MNQ = 1/10 NQ", official_source=source, verification_date=verified, data_quality="REFERENCE_ONLY", active=True),
        "MCL": ContractSpec(exchange="NYMEX", symbol="MCL", product_name="Micro WTI Crude Oil", contract_unit="100 barrels", minimum_tick=0.01, tick_value=1.0, point_value=100.0, currency="USD", session_definition="CME energy session", micro_mini_relationship="1 MCL = 1/10 CL", official_source=source, verification_date=verified, data_quality="REFERENCE_ONLY", active=True),
    }


def contract_registry_hash(registry: dict[str, ContractSpec] | None = None) -> str:
    registry = registry or default_contract_registry()
    payload = {key: value.model_dump(mode="json") for key, value in sorted(registry.items())}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def get_contract(symbol: str, registry: dict[str, ContractSpec] | None = None) -> ContractSpec:
    registry = registry or default_contract_registry()
    try:
        contract = registry[symbol.upper()]
    except KeyError as exc:
        raise ValueError(f"unsupported futures contract: {symbol}") from exc
    if not contract.active:
        raise ValueError(f"inactive futures contract: {symbol}")
    return contract
