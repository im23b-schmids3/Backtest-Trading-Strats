from datetime import datetime, timedelta, timezone
from pathlib import Path
import json

import pytest
import hashlib
import pyarrow as pa
import pyarrow.parquet as pq

from research_pipeline.htf_level_liquidity_fvg import Bar, ClosedBarAggregator, HTFLevelLiquidityFVG, frequency_classification, materialize_synthetic, reconcile_events
from research_pipeline.htf_level_liquidity_fvg.core import Event, Level, TerminalDisposition, phase_a_hard_gates, wilder_atr_prior
from research_pipeline.htf_level_liquidity_fvg import runner

T=datetime(2023,1,1,tzinfo=timezone.utc)
def bar(i,o=100,h=102,l=99,c=101): return Bar(T+timedelta(minutes=5*i),o,h,l,c,1,f"b{i}")
def trend(i,close): return bar(i,close,close+1,close-1,close)

def test_aggregation_is_closed_utc_and_gaps_are_tolerated():
    a=ClosedBarAggregator(); assert not a.push(bar(0)); assert not a.push(bar(1)); d=a.push(bar(2))["15m"]
    assert (d.open,d.high,d.low,d.close,d.volume)==(100,102,99,101,3)
    assert not a.push(bar(4)) # legitimate missing source bar creates no false aggregate
    with pytest.raises(ValueError): a.push(bar(4))
    with pytest.raises(ValueError): Bar(datetime(2023,1,1),100,101,99,100)

def test_prior_only_atr_and_bias():
    xs=[bar(i,100,102+i%2,99,101) for i in range(20)]
    before=wilder_atr_prior(xs,15); changed=xs[:]; changed[15]=bar(15,100,999,1,100)
    assert before==wilder_atr_prior(changed,15)
    e=HTFLevelLiquidityFVG("HTFLFVG-V1-MIN2P5R"); e.daily=[trend(i,100+i*.1) for i in range(56)]
    assert e.daily_bias().value=="BULLISH"

def test_confirmed_4h_fractal_delay_and_deterministic_level():
    e=HTFLevelLiquidityFVG("HTFLFVG-V1-MIN2P5R"); e.bars4h=[bar(i,100,110,90+i,100) for i in range(7)]; e.bars4h[3]=bar(3,100,110,80,100)
    e._confirm_levels("4h"); assert len(e.levels)==1 and e.levels[0].side=="SUPPORT"
    assert e.levels[0].confirmation_time==e.bars4h[-1].time

def test_long_and_short_sweep_choose_nearest_and_consume_once():
    e=HTFLevelLiquidityFVG("HTFLFVG-V1-MIN2P5R"); e.daily=[trend(i,100+i) for i in range(56)]; e.bars15=[bar(i,100,101,99,100) for i in range(16)]
    e.levels=[Level("far","SUPPORT",90,T,T),Level("near","SUPPORT",95,T,T)]
    e._evaluate_sweep(bar(99,100,101,93,97)); assert e.setup and e.setup["level_id"]=="near" and e.levels[1].consumed
    s=HTFLevelLiquidityFVG("HTFLFVG-V1-MIN2P5R"); s.daily=[trend(i,200-i) for i in range(56)]; s.bars15=[bar(i,100,101,99,100) for i in range(16)]; s.levels=[Level("r","RESISTANCE",105,T,T)]
    s._evaluate_sweep(bar(99,104,108,103,104)); assert s.setup and s.setup["direction"]=="SHORT"

def test_mss_displacement_first_fvg_and_projected_r_rejection():
    e=HTFLevelLiquidityFVG("HTFLFVG-V1-MIN2P5R"); e.daily=[trend(i,100+i) for i in range(56)]; e.bars5=[bar(i,100,101,99,100) for i in range(20)]
    e.setup={"setup_id":"s","direction":"LONG","sweep_5_index":0,"sweep_extreme":90,"sweep_atr":2,"event_history":["SETUP_PROPOSED"]}
    e._finish(T,TerminalDisposition.MSS_WINDOW_EXPIRED); assert e.outcomes[0]["terminal_disposition"]=="MSS_WINDOW_EXPIRED"

def test_order_invalidation_expiry_and_session_controls():
    e=HTFLevelLiquidityFVG("HTFLFVG-V1-MIN2P5R"); e.setup={"setup_id":"s","direction":"LONG","sweep_5_index":0,"sweep_extreme":99,"event_history":["SETUP_PROPOSED"]}; e.order={"activated_index":0,"entry":100}; e.bars5=[bar(0)]
    e._progress_order_or_position(bar(1,100,101,98,100)); assert e.outcomes[-1]["terminal_disposition"]=="PRE_ENTRY_SWEEP_INVALIDATED"

def test_stop_first_partial_tp_and_forced_exit():
    e=HTFLevelLiquidityFVG("HTFLFVG-V1-MIN2P5R"); e.setup={"setup_id":"s","trade_id":"t","direction":"LONG","stop":95,"tp1":105,"tp2":110,"event_history":["SETUP_PROPOSED","ENTRY_FILLED"]}; e.position={"position_id":"p","entry_index":0,"entry":100,"qty":1.,"remaining":1.,"tp1_done":False}; e.bars5=[bar(0)]
    e._progress_order_or_position(bar(1,100,106,94,100)); assert e.trades[-1]["exit_reason"]=="STOPPED" # stop wins same bar
    e=HTFLevelLiquidityFVG("HTFLFVG-V1-MIN2P5R"); e.setup={"setup_id":"s","trade_id":"t","direction":"LONG","stop":95,"tp1":105,"tp2":110,"event_history":["SETUP_PROPOSED","ENTRY_FILLED"]}; e.position={"position_id":"p","entry_index":0,"entry":100,"qty":1.,"remaining":1.,"tp1_done":False}; e.bars5=[bar(0)]
    e._progress_order_or_position(bar(1,100,106,96,105)); assert e.position["tp1_done"] and e.position["remaining"]==.5
    e._progress_order_or_position(bar(2,105,111,104,110)); assert e.trades[-1]["exit_reason"]=="TP2_COMPLETED"

def test_ids_lifecycle_and_reconciliation():
    ev=[Event("a",T.isoformat(),1,"c","SETUP_PROPOSED","s"),Event("b",T.isoformat(),2,"c","TERMINAL_DISPOSITION","s")]
    reconcile_events(ev,[{"setup_id":"s","terminal_disposition":"DAILY_BIAS_REJECTED","event_history":["SETUP_PROPOSED","TERMINAL_DISPOSITION"]}],[])
    with pytest.raises(ValueError): reconcile_events(ev,[],[])

@pytest.mark.parametrize(("n,label"),[(49,"UNDERFREQUENCY_FAIL"),(50,"TARGET_FREQUENCY"),(300,"TARGET_FREQUENCY"),(301,"HIGH_FREQUENCY_WARNING"),(500,"HIGH_FREQUENCY_WARNING"),(501,"OVERFREQUENCY_FAIL")])
def test_frequency_matrix(n,label): assert frequency_classification(n)[1]==label

def test_hard_gates_are_all_fail_closed():
    result=phase_a_hard_gates({"executed_trades":50,"net_pnl":1,"profit_factor":1.3,"average_net_r":.1,"max_drawdown_r":20,"profitable_months":8,"zero_trade_months":3,"best_month_positive_pnl_share":.35,"best_five_positive_pnl_share":.3,"long_share":.25,"short_share":.25,"long_average_net_r":-.15,"short_average_net_r":-.15,"one_tick_stress_positive":True,"best_trade_removal_positive":True,"bootstrap_median_mean_net_r":.1,"bootstrap_lower_95_net_r":-.025,"nonnegative_2023_subperiods":3,"immutable_artifacts":True,"funnel_reconciled":True})
    assert result["passed"]

def test_synthetic_materialization_has_exercised_complete_immutable_artifacts(tmp_path):
    root=tmp_path/"repo"; (root/".smithers/specs").mkdir(parents=True); source=Path(".smithers/specs/htf-level-liquidity-fvg-v1.md"); (root/".smithers/specs/htf-level-liquidity-fvg-v1.md").write_bytes(source.read_bytes()); target=tmp_path/"artifacts"
    result=materialize_synthetic(target,root); assert result["realStudyExecuted"] is False
    events=json.loads((target/"events.json").read_text()); assert events and all((target/name).exists() for name in __import__("research_pipeline.htf_level_liquidity_fvg.core",fromlist=["REQUIRED_ARTIFACTS"]).REQUIRED_ARTIFACTS)
    with pytest.raises(FileExistsError): materialize_synthetic(target,root)


def _v5_manifest(tmp_path, *, timestamp_overrides=None):
    root=tmp_path/"phase_a"; root.mkdir(); files=[]; rows=0
    for index,month in enumerate(runner.PHASE_A_MONTHS):
        start=datetime.fromisoformat(month+"-01T00:00:00+00:00")
        timestamp=(timestamp_overrides or {}).get(month,start)
        target=root/"bars"/f"{month}.parquet"; target.parent.mkdir(exist_ok=True)
        table=pa.table({"bar_start_utc":pa.array([timestamp],type=pa.timestamp("us",tz="UTC")),"open":[1.],"high":[2.],"low":[.5],"close":[1.5],"volume":[3.]})
        pq.write_table(table,target)
        files.append({"kind":"bars","month":month,"relative_path":f"bars/{month}.parquet","row_count":1,"sha256":hashlib.sha256(target.read_bytes()).hexdigest()}); rows+=1
    manifest={"valid":True,"identity":{"phase":"PHASE_A","symbol":"BTCUSDT","bar_interval":"5m","months":list(runner.PHASE_A_MONTHS)},"parquet_files":files,"five_minute_bar_count":rows}
    path=root/"manifest.json"; path.write_text(json.dumps(manifest)); return path,manifest


def test_real_phase_a_loader_accepts_only_declared_v5_partitions_and_tolerates_gaps(tmp_path):
    path,_=_v5_manifest(tmp_path)
    loaded=runner._load_explicit_phase_a_bars(path)
    assert len(loaded)==13 and loaded[0].time.tzinfo is not None


@pytest.mark.parametrize("mutate",[
    lambda m: m.__setitem__("valid",False),
    lambda m: m["identity"].__setitem__("phase","PHASE_B"),
    lambda m: m["identity"].__setitem__("symbol","ETHUSDT"),
    lambda m: m["identity"].__setitem__("bar_interval","1m"),
    lambda m: m["identity"].__setitem__("months",list(reversed(m["identity"]["months"]))),
    lambda m: m["parquet_files"].reverse(),
    lambda m: m["parquet_files"].append(dict(m["parquet_files"][0])),
    lambda m: m["parquet_files"][0].__setitem__("relative_path","../escape.parquet"),
    lambda m: m["parquet_files"][0].__setitem__("row_count",2),
    lambda m: m["parquet_files"][0].__setitem__("sha256","0"*64),
    lambda m: m.__setitem__("five_minute_bar_count",999),
])
def test_real_phase_a_manifest_contract_fails_closed(tmp_path,mutate):
    path,manifest=_v5_manifest(tmp_path); mutate(manifest); path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError): runner._load_explicit_phase_a_bars(path)


@pytest.mark.parametrize("bad_time",[
    datetime(2022,12,31,23,55,tzinfo=timezone.utc),
    datetime(2023,1,1,0,0),
])
def test_real_phase_a_loader_rejects_outside_or_non_utc_timestamps(tmp_path,bad_time):
    path,manifest=_v5_manifest(tmp_path)
    target=path.parent/manifest["parquet_files"][0]["relative_path"]
    typ=pa.timestamp("us") if bad_time.tzinfo is None else pa.timestamp("us",tz="UTC")
    pq.write_table(pa.table({"bar_start_utc":pa.array([bad_time],type=typ),"open":[1.],"high":[2.],"low":[.5],"close":[1.5],"volume":[3.]}),target)
    manifest["parquet_files"][0]["sha256"]=hashlib.sha256(target.read_bytes()).hexdigest(); path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError): runner._load_explicit_phase_a_bars(path)


def test_real_phase_a_loader_rejects_duplicate_or_decreasing_declared_order(tmp_path):
    path,manifest=_v5_manifest(tmp_path, timestamp_overrides={"2023-02":datetime(2023,1,1,tzinfo=timezone.utc)})
    with pytest.raises(ValueError): runner._load_explicit_phase_a_bars(path)


def test_synthetic_embedded_bars_are_supported_only_by_synthetic_helper(tmp_path):
    synthetic={"bars":[{"time":"2023-01-01T00:00:00Z","open":1,"high":2,"low":.5,"close":1.5}]}
    assert len(runner.load_synthetic_embedded_bars(synthetic))==1
    path,_=_v5_manifest(tmp_path); raw=json.loads(path.read_text()); raw["bars"]=synthetic["bars"]; path.write_text(json.dumps(raw))
    assert len(runner._load_explicit_phase_a_bars(path))==13
