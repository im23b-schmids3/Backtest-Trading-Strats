from __future__ import annotations

from .contracts import contract_registry_hash, get_contract
from .mappings import mapping_hash
from .models import FuturesTradeReconciliation, MarketMapping, TradeSignal


def reconcile_trade(trade: TradeSignal, mapping: MarketMapping, quantity: int, contracts: dict) -> FuturesTradeReconciliation:
    contract = get_contract(mapping.target_futures_contract, contracts)
    direction = trade.direction.upper()
    multiplier = 1 if direction == "LONG" else -1
    gross = ((trade.exit_price - trade.entry_price) / contract.minimum_tick) * contract.tick_value * quantity * multiplier
    fees = max(0.0, trade.fees) * quantity
    slippage = max(0.0, trade.slippage) * quantity
    return FuturesTradeReconciliation(trade_id=trade.trade_id, source_market=trade.source_market, source_entry=trade.entry_price, source_exit=trade.exit_price, source_return=trade.source_return, mapped_entry=trade.entry_price, mapped_exit=trade.exit_price, direction=direction, contract=contract.symbol, tick_size=contract.minimum_tick, tick_value=contract.tick_value, point_value=contract.point_value, quantity=quantity, gross_pnl=gross, fees=fees, slippage=slippage, net_pnl=gross - fees - slippage, mapping_hash=mapping_hash([mapping]), contract_registry_hash=contract_registry_hash(contracts))
