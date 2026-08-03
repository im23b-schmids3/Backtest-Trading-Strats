"""Explicitly exploratory TheSnowGuru ES tick-rule CVD pilot.

This module is intentionally separate from the exact-CVD import gate.
"""
from __future__ import annotations

import csv, hashlib, json, shutil, tempfile, zipfile
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .thesnowguru import TheSnowGuruAuditResult, _column_map, _timestamp_from_row
from .data import AggregateTrade
from .profile import FiveMinuteBar, UTC_SESSION_LABEL, build_session_profiles, session_date_for
from .strategy import ValueAreaTrapConfig, run_value_area_trap

LABELS = {"evidence_label": "EXPLORATORY_TICK_RULE_CVD", "exact_cvd": False, "tick_rule_approximated_cvd": True, "confirmation_evidence": False, "optimization_claimed": False, "requires_future_holdout": True}
START = datetime(2018, 1, 1, tzinfo=timezone.utc)
END = datetime(2018, 12, 31, 23, 59, 59, 999999, tzinfo=timezone.utc)

def tick_rule_direction(price: Decimal, previous_price: Decimal | None, last_nonzero: int) -> int:
    """Tick rule: up=buy, down=sell, unchanged carries; initially neutral."""
    if previous_price is None: return 0
    if price > previous_price: return 1
    if price < previous_price: return -1
    return last_nonzero

def retained_duplicate_counts(rows: list[tuple[datetime, int, Decimal, Decimal]]) -> tuple[int, int]:
    """Count adjacent timestamp and exact normalized-row duplicates safely."""
    duplicate_timestamps = exact_duplicates = 0; previous = None
    for stamp, _source_row_index, price, volume in sorted(rows, key=lambda item: (item[0], item[1])):
        normalized = (stamp, price, volume)
        if previous is not None and previous[0] == stamp: duplicate_timestamps += 1
        if previous == normalized: exact_duplicates += 1
        previous = normalized
    return duplicate_timestamps, exact_duplicates

def _write_once(path: Path, value: Any) -> None:
    raw = json.dumps(value, sort_keys=True, indent=2, default=str).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != raw: raise ValueError(f"immutable pilot artifact differs: {path}")
    if not path.exists(): path.write_bytes(raw)

BAR_SCHEMA = pa.schema([
    pa.field("timestamp_utc", pa.timestamp("us", tz="UTC")), pa.field("open", pa.decimal128(20, 8)), pa.field("high", pa.decimal128(20, 8)), pa.field("low", pa.decimal128(20, 8)), pa.field("close", pa.decimal128(20, 8)),
    pa.field("total_volume", pa.decimal128(20, 8)), pa.field("buy_volume", pa.decimal128(20, 8)), pa.field("sell_volume", pa.decimal128(20, 8)), pa.field("neutral_volume", pa.decimal128(20, 8)), pa.field("delta", pa.decimal128(20, 8)), pa.field("cumulative_delta", pa.decimal128(24, 8)), pa.field("trade_count", pa.int64()),
])

def _bar_table(bars: dict[datetime, list[Any]]) -> pa.Table:
    cumulative=Decimal(); data=[]
    for stamp, value in sorted(bars.items()):
        delta=value[5]-value[6]; cumulative += delta
        data.append((stamp, value[0],value[1],value[2],value[3],value[4],value[5],value[6],value[7],delta,cumulative,value[8]))
    return pa.Table.from_pylist([dict(zip(BAR_SCHEMA.names, row)) for row in data], schema=BAR_SCHEMA)

def _validate_bars(path: Path, count: int) -> None:
    table=pq.read_table(path); rows=table.to_pylist()
    if len(rows)!=count or not rows: raise ValueError("five-minute bar read-back row count is invalid")
    previous=None; cumulative=Decimal()
    for row in rows:
        stamp=row["timestamp_utc"]
        if previous is not None and stamp<=previous: raise ValueError("five-minute bar timestamps are not strictly monotonic")
        previous=stamp; o,h,l,c=row["open"],row["high"],row["low"],row["close"]
        if min(o,h,l,c)<=0 or h<max(o,c,l) or l>min(o,c,h): raise ValueError("invalid five-minute bar OHLC")
        component_total=row["buy_volume"]+row["sell_volume"]+row["neutral_volume"]; difference=row["total_volume"]-component_total
        if min(row["total_volume"],row["buy_volume"],row["sell_volume"],row["neutral_volume"])<0 or difference != 0 or row["delta"] != row["buy_volume"]-row["sell_volume"] or row["trade_count"]<=0:
            raise ValueError(f"five-minute bar volume reconciliation failed timestamp={stamp} total_volume={row['total_volume']} buy_volume={row['buy_volume']} sell_volume={row['sell_volume']} neutral_volume={row['neutral_volume']} component_total={component_total} difference={difference}")
        cumulative += row["delta"]
        if row["cumulative_delta"] != cumulative: raise ValueError("five-minute bar cumulative delta reconciliation failed")

def import_es_tick_rule(*, audit_path: str | Path, cache_root: str | Path = "data/value_area_trap") -> dict[str, Any]:
    audit_file = Path(audit_path).resolve(); audit = TheSnowGuruAuditResult.model_validate_json(audit_file.read_text())
    candidate = next((c for c in audit.candidates if c.relative_path.endswith("s&p-tick.zip::SP.csv") and c.classification == "VERIFIED_FUTURES_TICK" and c.cvd_support == "CVD_REQUIRES_TICK_RULE_APPROXIMATION"), None)
    if candidate is None: raise ValueError("pilot requires the audited S&P futures tick candidate with tick-rule-only CVD support")
    archive = Path(candidate.source_path)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != candidate.source_sha256: raise ValueError("source archive hash changed after audit")
    rows=[]; rejected=Counter(); scanned=0
    with zipfile.ZipFile(archive) as zf, zf.open(candidate.relative_path.split("::",1)[1]) as binary:
        reader=csv.DictReader((line.decode("utf-8") for line in binary)); columns=_column_map(reader.fieldnames or [])
        for index,row in enumerate(reader,1):
            scanned += 1; parsed=_timestamp_from_row(row,columns)
            if parsed is None: rejected["parse_failures"]+=1; continue
            stamp=parsed[0].replace(tzinfo=timezone.utc)
            if stamp < START: continue
            if stamp > END: break
            try: price=Decimal(row[columns["price"]]); volume=Decimal(row[columns["volume"]])
            except Exception: rejected["null_or_parse"]+=1; continue
            if price<=0: rejected["nonpositive_price"]+=1; continue
            if volume<=0: rejected["zero_or_negative_volume"]+=1; continue
            rows.append((stamp,index,price,volume))
    # Source row index is deliberately retained as a sort tie-breaker.  It is
    # never a positional index into this filtered list: source rows before
    # 2018 and rejected rows make the two sequences different lengths.
    rows.sort(key=lambda item:(item[0],item[1])); previous=None; last=0; bars={}; duplicates, exact = retained_duplicate_counts(rows); weekend=0; neutral=buy=sell=0
    for stamp,source_row_index,price,volume in rows:
        weekend += stamp.weekday()>=5; direction=tick_rule_direction(price,previous,last); previous=price
        if direction: last=direction
        neutral += direction==0; buy += direction>0; sell += direction<0
        bucket=stamp.replace(minute=stamp.minute-stamp.minute%5,second=0,microsecond=0); bar=bars.setdefault(bucket,[price,price,price,price,Decimal(),Decimal(),Decimal(),Decimal(),0]); bar[1]=max(bar[1],price);bar[2]=min(bar[2],price);bar[3]=price;bar[4]+=volume;bar[5]+=volume if direction>0 else Decimal();bar[6]+=volume if direction<0 else Decimal();bar[7]+=volume if direction==0 else Decimal();bar[8]+=1
    table=_bar_table(bars)
    with tempfile.TemporaryDirectory() as temporary:
        staged=Path(temporary)/"five-minute-bars.parquet"; pq.write_table(table, staged, compression="zstd", compression_level=3, use_dictionary=False, write_statistics=True)
        _validate_bars(staged, len(bars)); bar_hash=hashlib.sha256(staged.read_bytes()).hexdigest()
        payload={**LABELS,"strategy_id":"ValueAreaTrap.ES_TICK_RULE_CVD","source_archive":str(archive),"source_archive_hash":candidate.source_sha256,"source_rows_scanned":scanned,"rows_retained":len(rows),"rejected_rows":dict(rejected),"duplicate_timestamps":duplicates,"exact_duplicate_rows":exact,"weekend_rows":weekend,"neutral_rows":neutral,"buy_rows":buy,"sell_rows":sell,"bar_count":len(bars),"period_start":START,"period_end":END,"tick_rule":"up=buy; down=sell; unchanged=carry last non-zero; before first non-zero=neutral","bar_data_sha256":bar_hash}
        digest=hashlib.sha256(json.dumps(payload,sort_keys=True,default=str).encode()).hexdigest(); root=Path(cache_root)/"normalized"/"ES_THESNOWGURU_TICK_RULE"/digest; root.mkdir(parents=True,exist_ok=True); destination=root/"five-minute-bars.parquet"
        if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest()!=bar_hash: raise ValueError("immutable bar artifact differs")
        if not destination.exists(): shutil.copyfile(staged,destination)
    manifest={**LABELS,"dataset_hash":digest,"bar_data_path":"five-minute-bars.parquet","bar_data_format":"parquet","bar_data_sha256":bar_hash,"bar_count":len(bars),"schema":str(BAR_SCHEMA),"period_start":START,"period_end":END}
    _validate_bars(destination, len(bars)); _write_once(root/"source-manifest.json",payload); _write_once(root/"cleaning-report.json",payload); _write_once(root/"tick-rule-report.json",payload); _write_once(root/"five-minute-bar-manifest.json",manifest)
    return {**payload,"dataset_hash":digest,"artifact_root":str(root),"status":"IMPORTED_EXPLORATORY_TICK_RULE_CVD"}

def validate_es_tick_rule_dataset(dataset_root: str | Path) -> dict[str, Any]:
    root=Path(dataset_root).resolve(); manifest=json.loads((root/"five-minute-bar-manifest.json").read_text()); path=root/manifest["bar_data_path"]
    _validate_bars(path, int(manifest["bar_count"]))
    if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest["bar_data_sha256"]: raise ValueError("five-minute bar artifact hash mismatch")
    return {"path":str(root),"valid":True,"bar_count":manifest["bar_count"],"dataset_hash":manifest["dataset_hash"],"errors":[]}

def run_es_tick_rule_pilot(*, dataset_root: str | Path, artifact_root: str | Path, repository_root: str | Path = ".") -> dict[str, Any]:
    validated=validate_es_tick_rule_dataset(dataset_root); root=Path(dataset_root).resolve(); table=pq.read_table(root/"five-minute-bars.parquet")
    bars=[]; profile_trades=[]
    for index,row in enumerate(table.to_pylist(),1):
        stamp=row["timestamp_utc"]; day=session_date_for(stamp, UTC_SESSION_LABEL)
        bars.append(FiveMinuteBar(start_utc=stamp,end_utc=stamp.replace(second=0,microsecond=0)+__import__('datetime').timedelta(minutes=5),start_new_york=stamp,end_new_york=stamp,open=row['open'],high=row['high'],low=row['low'],close=row['close'],total_volume=row['total_volume'],aggressive_buy_volume=row['buy_volume'],aggressive_sell_volume=row['sell_volume'],bar_delta=row['delta'],cumulative_volume_delta=row['cumulative_delta'],trade_count=row['trade_count'],vwap=row['close'],session_date=day,session_label=UTC_SESSION_LABEL))
        profile_trades.append(AggregateTrade(event_time_utc=stamp,trade_time_utc=stamp,aggregate_trade_id=index,price=row['close'],quantity_base=row['total_volume'],notional_quote=row['close']*row['total_volume'],buyer_is_maker=False,aggressor_side='BUY',signed_quantity=row['total_volume'],source='thesnowguru_tick_rule',source_file=str(root/'five-minute-bars.parquet'),source_hash=validated['dataset_hash']))
    config=ValueAreaTrapConfig(symbol='ES_THESNOWGURU_TICK_RULE',session_definition=UTC_SESSION_LABEL,price_tick=Decimal('0.25'),quantity=Decimal('1'),minimum_quantity=Decimal('1'),quantity_step=Decimal('1'),enforce_symbol_filters=True,record_compliance_events=True)
    result=run_value_area_trap(bars,build_session_profiles(profile_trades,bucket_size=Decimal('0.25'),source_dataset_hash=validated['dataset_hash'],session_definition=UTC_SESSION_LABEL),config)
    run_id=hashlib.sha256(f"ValueAreaTrap.ES_TICK_RULE_CVD|{validated['dataset_hash']}|1".encode()).hexdigest()[:16]; output=Path(artifact_root)/'ValueAreaTrap.ES_TICK_RULE_CVD'/run_id
    report={**LABELS,"execution_realism":False,"contract_accurate_pnl":False,"run_id":run_id,"strategy_id":"ValueAreaTrap.ES_TICK_RULE_CVD","dataset_hash":validated['dataset_hash'],"session_count":len({b.session_date for b in bars}),"bar_count":len(bars),"significant_stop_runs":result.significant_stop_runs,"confirmed_divergences":result.confirmed_divergences,"return_triggers":result.return_triggers,"proposed_setups":result.proposed_setups,"compliance_blocks":len(result.compliance_blocks),"executed_trades":len(result.trades),"gross_points":str(result.gross_pnl),"net_points":str(result.net_pnl),"total_costs_points":str(result.fees+result.slippage_cost),"warnings":["tick-rule CVD is an approximation","source is sparse relative to a complete ES trade feed","no contract code or rollover metadata","equal timestamps and exact duplicates retained","exploratory evidence only"]}
    _write_once(output/'report.json',report); _write_once(output/'trades.json',result.trades)
    return {**report,"report_path":str(output/'report.json')}
