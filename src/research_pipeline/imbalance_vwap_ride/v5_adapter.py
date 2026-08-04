from __future__ import annotations
from typing import Any
from .v5_models import *
class ImbalanceVWAPRideV5Adapter:
    adapter_id=ADAPTER_ID; strategy_id=STRATEGY_ID
    def capabilities(self)->dict[str,Any]: return {"adapter_id":ADAPTER_ID,"strategy_id":STRATEGY_ID,"direction":"LONG_ONLY","local_parquet_processing":True,"live_orders":False,"external_raw_trade_transmission":False,"secrets_required":False,"confirmation_evidence":False,"external_confirmation_required":True}
