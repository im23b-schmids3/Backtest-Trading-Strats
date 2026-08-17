"""Native MBP-10 source contract for future L2 V3 acquisition.

This is a schema/mapping contract only.  It neither reads DBN data nor runs the
strategy.  It keeps native aggregate depth free of MBO order identities.
"""
from __future__ import annotations

from typing import Any


NATIVE_MBP10_REQUIRED_FIELDS = {
    "timestamps": ("ts_event", "ts_recv"),
    "event": ("action", "side", "price", "size", "flags", "sequence"),
    "top_ten_bid": ("bid_px_00..09", "bid_sz_00..09", "bid_ct_00..09"),
    "top_ten_ask": ("ask_px_00..09", "ask_sz_00..09", "ask_ct_00..09"),
}


def native_mbp10_adapter_contract() -> dict[str, Any]:
    """Fields and fail-closed conditions for a future native MBP-10 adapter."""
    return {
        "schema": "mbp-10",
        "source": "GLBX.MDP3",
        "book_view": "aggregate_top_ten_price_levels",
        "required_fields": NATIVE_MBP10_REQUIRED_FIELDS,
        "order_id_permitted": False,
        "execution_records_required": True,
        "execution_aggressor_mapping": {
            "native_trade_side_B": "BUY",
            "native_trade_side_A": "SELL",
            "unknown_side": "UNKNOWN",
        },
        "normalization": "Databento fixed-point prices divided by 1e9 exactly once at adapter boundary",
        "initialization_gate": (
            "Before 13:30 UTC the native adapter must observe a valid two-sided top-ten book; "
            "otherwise the session fails closed and cannot start interactions."
        ),
        "signal_scope": "No interaction is eligible before RTH 13:30 UTC.",
        "mbo_required": False,
    }


def validate_native_mbp10_field_mapping(fields: dict[str, Any]) -> None:
    """Reject incomplete mappings or any accidental private order identity."""
    flattened = {str(item) for values in fields.values() for item in values}
    required = {item for values in NATIVE_MBP10_REQUIRED_FIELDS.values() for item in values}
    if not required.issubset(flattened):
        raise ValueError("native MBP-10 mapping omits required aggregate L2 fields")
    if any("order_id" in item.lower() for item in flattened):
        raise ValueError("native MBP-10 mapping must not expose order identity")
