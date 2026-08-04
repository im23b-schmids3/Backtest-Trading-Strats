"""V5 price-scaled footprint helpers; no network or raw-row exports."""
from __future__ import annotations
from decimal import Decimal
from typing import Any
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from .v5_models import scaled_bin_size
from .v5_models import PHASE_A_MONTHS, PHASE_B_MONTHS
from .artifacts import sha256_file, sha256_value
from .v4_data import validate_v4_source_manifest
from .footprint import RAW_COLUMNS, _batch_table, _utc_from_epoch_minute

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

def aggregate_price_scaled_footprints(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate local normalized trades without emitting raw trade rows."""
    buckets: dict[tuple[str, Decimal, Decimal], dict[str, Any]] = {}
    for row in rows:
        key=(str(row["bar_start_utc"]),Decimal(str(row["price_bin"])),Decimal(str(row["bin_size_usd"])))
        item=buckets.setdefault(key,{"bar_start_utc":key[0],"price_bin":key[1],"bin_size_usd":key[2],"buy_volume_btc":Decimal(),"sell_volume_btc":Decimal()})
        qty=Decimal(str(row.get("quantity",row.get("quantity_btc",0))))
        item["sell_volume_btc" if bool(row.get("is_buyer_maker",False)) else "buy_volume_btc"] += qty
    result=[]
    for item in buckets.values():
        item["total_volume_btc"]=item["buy_volume_btc"]+item["sell_volume_btc"]; item["delta_btc"]=item["buy_volume_btc"]-item["sell_volume_btc"]; result.append(item)
    return sorted(result,key=lambda x:(x["bar_start_utc"],x["price_bin"]))

def maximal_stacked_zones(footprints: list[dict[str, Any]], *, minimum_volume: Decimal=Decimal("35"), imbalance_ratio: Decimal=Decimal("3"), stacked_bins: int=3) -> list[dict[str, Any]]:
    """Create non-overlapping maximal sequences and never combine adaptive bin sizes."""
    groups: dict[tuple[str,Decimal],list[dict[str,Any]]]={}
    for row in footprints: groups.setdefault((str(row["bar_start_utc"]),Decimal(str(row["bin_size_usd"]))),[]).append(row)
    zones=[]
    for (bar,size),values in sorted(groups.items()):
        values.sort(key=lambda x:Decimal(str(x["price_bin"]))); run=[]
        def emit():
            if len(run)>=stacked_bins: zones.append({"source_bar_start_utc":bar,"bottom":Decimal(str(run[0]["price_bin"])),"top":Decimal(str(run[-1]["price_bin"]))+size,"bin_size_usd":size,"stacked_bins":len(run)})
        for item in values:
            buy,sell,total=(Decimal(str(item.get(k,0))) for k in ("buy_volume_btc","sell_volume_btc","total_volume_btc")); qualifies=total>=minimum_volume and ((sell==0 and buy>=minimum_volume) or buy>=imbalance_ratio*sell); adjacent=bool(run) and Decimal(str(item["price_bin"]))==Decimal(str(run[-1]["price_bin"]))+size
            if qualifies and (not run or adjacent): run.append(item)
            else: emit(); run=[item] if qualifies else []
        emit()
    return zones

V5_DATA_BUILDER_VERSION = "imbalance-vwap-ride-v5-scaled-streaming-2"

def build_v5_scaled_dataset(source_manifest_path: str | Path, bars_manifest_path: str | Path, cache_root: str | Path, *, phase: str, batch_size: int = 1_000_000) -> dict[str, Any]:
    """Build the V5 price-scaled footprints locally from a validated official source.

    This consumes Arrow batches only; aggregate-trade rows never leave the process
    and every month is staged then atomically published.
    """
    expected_months = PHASE_A_MONTHS if phase == "PHASE_A" else PHASE_B_MONTHS if phase == "PHASE_B" else None
    if expected_months is None:
        raise ValueError("V5 phase must be PHASE_A or PHASE_B")
    source_validation = validate_v4_source_manifest(
        source_manifest_path,
        phase=phase,
        expected_months=expected_months,
        verify_archives=True,
    )
    if not source_validation["valid"]:
        raise ValueError("invalid official Phase A source: " + "; ".join(source_validation["errors"]))
    source = source_validation["manifest"]
    bars_manifest = json.loads(Path(bars_manifest_path).read_text(encoding="utf-8"))
    if bars_manifest.get("source_row_count") != source.row_count or not bars_manifest.get("valid"):
        raise ValueError("validated 5m bar source does not reconcile to normalized source")
    identity = {"builder_version": V5_DATA_BUILDER_VERSION, "phase": phase, "symbol": "BTCUSDT", "months": [p.month for p in source.partitions], "normalized_dataset_hash": source.normalized_dataset_hash, "source_manifest_hash": source.manifest_hash, "bin_formula": "clamp(20,100,round_half_up(previous_close*0.001/5)*5)", "bar_interval": "5m", "aggressor_rule": "BUY_IFF_BUYER_IS_MAKER_FALSE", "daily_vwap_reset": "UTC_MIDNIGHT"}
    dataset_id = sha256_value(identity)
    root = Path(cache_root).resolve() / "BTCUSDT" / phase.lower() / dataset_id
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("identity") != identity: raise ValueError("immutable V5 dataset identity collision")
        if existing.get("valid") and all(sha256_file(root / x["relative_path"]) == x["sha256"] for x in existing["parquet_files"]): return existing
        raise ValueError("existing V5 dataset is invalid")
    root.mkdir(parents=True, exist_ok=True)
    source_root = Path(source_manifest_path).resolve().parent
    bars_root = Path(bars_manifest_path).resolve().parent
    bar_files = {x["month"]: x for x in bars_manifest["parquet_files"] if x["kind"] == "bars"}
    output_files=[]; total_footprints=total_trades=0; previous_close=None; monthly=[]
    for partition in source.partitions:
        month=partition.month; month_opening_previous_close=previous_close; bar_table=pq.read_table(bars_root / bar_files[month]["relative_path"])
        bar_rows=bar_table.to_pylist(); size_by_bucket={int(row["bar_start_utc"].timestamp()//60): scaled_bin_size(previous_close if i == 0 else bar_rows[i-1]["close"]) for i,row in enumerate(bar_rows)}
        if bar_rows: previous_close=bar_rows[-1]["close"]
        target=root / "footprints" / f"{month}.parquet"
        checkpoint=target.with_suffix(".checkpoint.json")
        checkpoint_identity={"month":month,"source_partition_hash":partition.parquet_hash,"bars_sha256":bar_files[month].get("sha256"),"previous_close":None if month_opening_previous_close is None else str(month_opening_previous_close)}
        if target.exists() and checkpoint.exists():
            prior=json.loads(checkpoint.read_text(encoding="utf-8"))
            if prior.get("identity") == checkpoint_identity and sha256_file(target) == prior.get("sha256"):
                item=prior["file"]
                output_files.append(item); monthly.append(prior["monthly"])
                total_footprints += int(item["row_count"]); total_trades += int(item["trade_count"])
                continue
            raise ValueError(f"immutable V5 monthly checkpoint collision: {month}")
        accum: dict[tuple[int,Decimal,Decimal],list[Any]] = {}; trade_total=0
        parquet=pq.ParquetFile(source_root / partition.file_name)
        for batch in parquet.iter_batches(batch_size=batch_size, columns=list(RAW_COLUMNS), use_threads=True):
            table=_batch_table(batch); buckets=table["bucket"].combine_chunks().to_numpy(zero_copy_only=False); prices=table["raw_price"].combine_chunks(); quantities=table["quantity"].combine_chunks(); makers=table["buyer_is_maker"].combine_chunks()
            for i,bucket in enumerate(buckets):
                size=size_by_bucket.get(int(bucket))
                if size is None: continue
                price=Decimal(str(prices[i].as_py())); key=(int(bucket),(price//size)*size,size); current=accum.setdefault(key,[Decimal(),Decimal(),0]); qty=Decimal(str(quantities[i].as_py())); current[1 if bool(makers[i].as_py()) else 0]+=qty; current[2]+=1; trade_total+=1
        rows=[]
        for (bucket,floor,size),(buy,sell,count) in sorted(accum.items()):
            start=_utc_from_epoch_minute(bucket); rows.append({"bar_start_utc":start,"bar_end_utc":_utc_from_epoch_minute(bucket+5),"month":month,"bin_size_usd":size,"bin_floor":floor,"bin_upper_exclusive":floor+size,"buy_volume_btc":buy,"sell_volume_btc":sell,"total_volume_btc":buy+sell,"delta_btc":buy-sell,"trade_count":count})
        if sum(x["trade_count"] for x in rows) != trade_total: raise AssertionError("V5 footprint trade reconciliation failed")
        target.parent.mkdir(parents=True,exist_ok=True); staging=target.with_suffix(".parquet.part"); pq.write_table(pa.Table.from_pylist(rows),staging,compression="zstd"); os.replace(staging,target)
        item={"kind":"footprints","month":month,"relative_path":target.relative_to(root).as_posix(),"row_count":len(rows),"trade_count":trade_total,"sha256":sha256_file(target)}
        month_report={"month":month,"bars":len(bar_rows),"footprint_rows":len(rows),"footprint_trade_count":trade_total,"bin_size_distribution":{str(k):sum(1 for x in rows if x["bin_size_usd"]==k) for k in sorted({x["bin_size_usd"] for x in rows})}}
        checkpoint_staging=checkpoint.with_suffix(".json.part")
        checkpoint_staging.write_text(json.dumps({"identity":checkpoint_identity,"sha256":item["sha256"],"file":item,"monthly":month_report},sort_keys=True,indent=2)+"\n",encoding="utf-8")
        os.replace(checkpoint_staging,checkpoint)
        output_files.append(item); monthly.append(month_report); total_footprints+=len(rows); total_trades+=trade_total
    content_hash=sha256_value({x["relative_path"]:x["sha256"] for x in output_files})
    out={"identity":identity,"dataset_hash":source.normalized_dataset_hash,"footprint_dataset_hash":content_hash,"source_manifest_path":str(Path(source_manifest_path).resolve()),"bars_manifest_path":str(Path(bars_manifest_path).resolve()),"source_row_count":source.row_count,"footprint_trade_count":total_trades,"footprint_row_count":total_footprints,"parquet_files":output_files,"monthly_diagnostics":monthly,"raw_aggregate_rows_transmitted":False,"valid":True,"resumable":True}
    staging=manifest_path.with_suffix(".json.part"); staging.write_text(json.dumps(out,sort_keys=True,indent=2,default=str)+"\n",encoding="utf-8"); os.replace(staging,manifest_path)
    return out


def build_v5_phase_a_scaled_dataset(source_manifest_path: str | Path, bars_manifest_path: str | Path, cache_root: str | Path, *, batch_size: int = 1_000_000) -> dict[str, Any]:
    """Compatibility entry point for the sealed Phase-A footprint build."""
    return build_v5_scaled_dataset(source_manifest_path, bars_manifest_path, cache_root, phase="PHASE_A", batch_size=batch_size)


def build_v5_phase_b_scaled_dataset(source_manifest_path: str | Path, bars_manifest_path: str | Path, cache_root: str | Path, *, batch_size: int = 1_000_000) -> dict[str, Any]:
    """Build the separately content-addressed sealed Phase-B footprint dataset."""
    return build_v5_scaled_dataset(source_manifest_path, bars_manifest_path, cache_root, phase="PHASE_B", batch_size=batch_size)
