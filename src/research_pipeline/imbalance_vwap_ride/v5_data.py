"""V5 price-scaled footprint helpers; no network or raw-row exports."""
from __future__ import annotations
from decimal import Decimal
from typing import Any
from .v5_models import scaled_bin_size

def annotate_price_scaled_bins(bars: list[dict[str,Any]], trades: list[dict[str,Any]]) -> list[dict[str,Any]]:
    """Assign each trade its source bar's frozen, preceding-close bin size."""
    by_bar={str(x["bar_start_utc"]):x for x in bars}; ordered=sorted(bars,key=lambda x:str(x["bar_start_utc"])); previous={str(b["bar_start_utc"]):scaled_bin_size(ordered[i-1]["close"]) if i else None for i,b in enumerate(ordered)}; output=[]
    for raw in trades:
        row=dict(raw); key=str(row["bar_start_utc"]); size=previous.get(key)
        if size is None: continue
        price=Decimal(str(row["price"])); row["bin_size_usd"]=size; row["price_bin"]= (price//size)*size; output.append(row)
    return output

def validate_scaled_footprints(rows: list[dict[str,Any]]) -> dict[str,Any]:
    valid=all(Decimal(str(x["bin_size_usd"])) in {Decimal("20"),Decimal("25"),Decimal("30"),Decimal("35"),Decimal("40"),Decimal("45"),Decimal("50"),Decimal("55"),Decimal("60"),Decimal("65"),Decimal("70"),Decimal("75"),Decimal("80"),Decimal("85"),Decimal("90"),Decimal("95"),Decimal("100")} and Decimal(str(x["price_bin"]))==Decimal(str(x["price"]))//Decimal(str(x["bin_size_usd"]))*Decimal(str(x["bin_size_usd"])) for x in rows)
    return {"valid":valid,"footprint_rows":len(rows),"raw_aggregate_rows_transmitted":False}
