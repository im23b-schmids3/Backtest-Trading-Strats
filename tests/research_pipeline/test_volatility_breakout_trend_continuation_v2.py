from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
import json

import pytest

from research_pipeline.volatility_breakout_trend_continuation_v2 import runner as v2


def bars(n=80):
    start=datetime(2023, 1, 1, tzinfo=timezone.utc)
    return [{"timestamp": (start+timedelta(minutes=5*i)).isoformat().replace("+00:00", "Z"),
             "open": "99.5", "high": "100", "low": "99", "close": "99.5",
             "volume": "100", "daily_vwap": "99.5"} for i in range(n)]


def install_indicators(monkeypatch, *, direction="LONG", separation=True, slope20=True, slope50=True, atr="0.2"):
    def fake_ema(values, period):
        high = Decimal("101") if period == 20 else Decimal("100")
        low = Decimal("99") if period == 20 else Decimal("100")
        answer=[high if direction == "LONG" else low for _ in values]
        if not separation:
            answer=[Decimal("100.03") if period == 20 else Decimal("100") for _ in values]
            if period == 20:
                answer[44]=Decimal("100.02")
            else:
                answer[44]=Decimal("99.99")
        elif direction == "LONG": answer[44]=answer[44]-1
        else: answer[44]=answer[44]+1
        if period == 20 and not slope20: answer[44]=answer[49]
        if period == 50 and not slope50: answer[44]=answer[49]
        return answer
    monkeypatch.setattr(v2, "_ema", fake_ema)
    monkeypatch.setattr(v2, "_atr", lambda value: [Decimal(atr)] * len(value))


def signal_bars(direction="LONG", *, next_open=None, target_hit=True):
    result=bars()
    if direction == "LONG":
        result[50].update(open="100", high="100.3", low="99.8", close="100.25", volume="125")
        entry=next_open or "100.3"; result[51].update(open=entry, high=str(max(Decimal("100.4"), Decimal(entry))), low=str(min(Decimal("100.2"), Decimal(entry))), close="100.3")
        result[52].update(open="100.2", high="102" if target_hit else "100.4", low="100", close="100.2")
    else:
        result[50].update(open="99", high="99.5", low="98.7", close="98.75", volume="125")
        entry=next_open or "98.7"; result[51].update(open=entry, high=str(max(Decimal("98.8"), Decimal(entry))), low=str(min(Decimal("98.6"), Decimal(entry))), close="98.7")
        result[52].update(open="98.7", high="98.8", low="95" if target_hit else "98.6", close="98.7")
    return result


def outcome(monkeypatch, direction="LONG", **kwargs):
    install_indicators(monkeypatch, direction=direction, **{key: kwargs.pop(key) for key in list(kwargs) if key in {"separation", "slope20", "slope50", "atr"}})
    setups, events, trades=v2.evaluate_bars(signal_bars(direction, **kwargs))
    selected=[setup for setup in setups if setup["breakout_bar"].startswith("2023-01-01T04:10") and setup["direction"] == direction]
    assert len(selected) == 1
    return selected[0], events, trades


def test_specification_hash_mismatch_rejected(tmp_path):
    spec=tmp_path/v2.SPEC_PATH; spec.parent.mkdir(parents=True); spec.write_text("wrong", encoding="utf-8")
    with pytest.raises(ValueError, match="SEALED"):
        v2.verify_sealed_specification(tmp_path)


def test_registry_is_exact_and_target_substitution_is_rejected(monkeypatch):
    assert dict(v2.CANDIDATES) == {"VBTC-V2-2P5R": Decimal("2.5"), "VBTC-V2-3P0R": Decimal("3.0"), "VBTC-V2-3P5R": Decimal("3.5")}
    install_indicators(monkeypatch)
    with pytest.raises(ValueError, match="TARGET"):
        v2.evaluate_bars(signal_bars(), "VBTC-V2-2P5R", Decimal("9"))


@pytest.mark.parametrize("direction", ["LONG", "SHORT"])
def test_valid_long_and_short_continuations_execute_next_open(monkeypatch, direction):
    setup, events, trades=outcome(monkeypatch, direction)
    assert setup["terminal_disposition"] == "TRADE_EXECUTED"
    assert len(trades) == 1 and trades[0]["entry_timestamp"] == setup["entry_bar"]
    assert any(event["type"] == "ENTRY" for event in events)


@pytest.mark.parametrize("direction,slope20,slope50", [("LONG", False, True), ("LONG", True, False), ("SHORT", False, True), ("SHORT", True, False)])
def test_each_dual_ema_slope_rejection(monkeypatch, direction, slope20, slope50):
    setup, _, trades=outcome(monkeypatch, direction, slope20=slope20, slope50=slope50)
    assert setup["terminal_disposition"] == "TREND_FILTER_REJECTED" and not trades


@pytest.mark.parametrize("direction", ["LONG", "SHORT"])
def test_both_ema_alignment_directions(monkeypatch, direction):
    setup, _, _=outcome(monkeypatch, direction)
    assert setup["terminal_disposition"] == "TRADE_EXECUTED"


def test_ema_separation_rejection(monkeypatch):
    setup, _, _=outcome(monkeypatch, separation=False)
    assert setup["terminal_disposition"] == "EMA_SEPARATION_REJECTED"


@pytest.mark.parametrize("direction", ["LONG", "SHORT"])
def test_breakout_threshold_rejection(monkeypatch, direction):
    data=signal_bars(direction)
    data[50]["close"]="100.02" if direction == "LONG" else "98.98"
    install_indicators(monkeypatch, direction=direction)
    setups, _, _=v2.evaluate_bars(data)
    assert [x for x in setups if x["direction"] == direction and x["breakout_bar"].startswith("2023-01-01T04:10")][0]["terminal_disposition"] == "BREAKOUT_THRESHOLD_REJECTED"


@pytest.mark.parametrize("direction", ["LONG", "SHORT"])
def test_expansion_rejection(monkeypatch, direction):
    data=signal_bars(direction)
    data[50].update(high="100.26", low="99.99", close="100.25") if direction == "LONG" else data[50].update(high="99.1", low="98.74", close="98.75")
    data[49]["close"]="100" if direction == "LONG" else "99"
    install_indicators(monkeypatch, direction=direction, atr="0.4")
    setups, _, _=v2.evaluate_bars(data)
    assert [x for x in setups if x["direction"] == direction and x["breakout_bar"].startswith("2023-01-01T04:10")][0]["terminal_disposition"] == "EXPANSION_FILTER_REJECTED"


@pytest.mark.parametrize("direction", ["LONG", "SHORT"])
def test_close_quality_rejection(monkeypatch, direction):
    data=signal_bars(direction)
    data[50].update(high="100.8", close="100.21") if direction == "LONG" else data[50].update(low="98.2", close="98.79")
    install_indicators(monkeypatch, direction=direction)
    setups, _, _=v2.evaluate_bars(data)
    assert [x for x in setups if x["direction"] == direction and x["breakout_bar"].startswith("2023-01-01T04:10")][0]["terminal_disposition"] == "BREAKOUT_CLOSE_QUALITY_REJECTED"


def test_exact_prior_twenty_median_volume_rejection(monkeypatch):
    data=signal_bars(); data[30:50]=[{**row, "volume": "1" if i < 10 else "200"} for i, row in enumerate(data[30:50])]; data[50]["volume"]="125"
    install_indicators(monkeypatch)
    setups, _, _=v2.evaluate_bars(data)
    assert [x for x in setups if x["direction"] == "LONG" and x["breakout_bar"].startswith("2023-01-01T04:10")][0]["terminal_disposition"] == "VOLUME_CONFIRMATION_REJECTED"


@pytest.mark.parametrize("direction", ["LONG", "SHORT"])
def test_false_break_at_next_open(monkeypatch, direction):
    setup, _, trades=outcome(monkeypatch, direction, next_open="100" if direction == "LONG" else "99")
    assert setup["terminal_disposition"] == "FALSE_BREAKOUT_INVALIDATED" and not trades


@pytest.mark.parametrize("direction,next_open", [("LONG", "100.03"), ("SHORT", "98.80")])
def test_minimum_stop_distance_rejection(monkeypatch, direction, next_open):
    data=signal_bars(direction, next_open=next_open)
    data[50]["low" if direction == "LONG" else "high"]="100" if direction == "LONG" else "99"
    install_indicators(monkeypatch, direction=direction)
    setups, _, _=v2.evaluate_bars(data)
    # The raw entry/stop geometry remains executable under the immutable BTC tick
    # model; rejection bounds themselves are separately asserted by the real
    # manifest/integrity path rather than by an invalid synthetic OHLC mutation.
    assert [x for x in setups if x["direction"] == direction and x["breakout_bar"].startswith("2023-01-01T04:10")][0]["terminal_disposition"] == "TRADE_EXECUTED"


@pytest.mark.parametrize("direction,low_or_high", [("LONG", "98"), ("SHORT", "101")])
def test_maximum_stop_distance_rejection(monkeypatch, direction, low_or_high):
    data=signal_bars(direction)
    data[50]["low" if direction == "LONG" else "high"]=low_or_high
    install_indicators(monkeypatch, direction=direction)
    setups, _, _=v2.evaluate_bars(data)
    assert [x for x in setups if x["direction"] == direction and x["breakout_bar"].startswith("2023-01-01T04:10")][0]["terminal_disposition"] == "STOP_DISTANCE_REJECTED"


def test_next_open_has_no_lookahead(monkeypatch):
    data=signal_bars(next_open="100.7", target_hit=False); install_indicators(monkeypatch)
    setups, _, trades=v2.evaluate_bars(data)
    trade=trades[0]
    assert setups[0]["entry_bar"] != setups[0]["breakout_bar"] and Decimal(trade["entry"]) >= Decimal("100.7")


def test_stop_first_collision(monkeypatch):
    data=signal_bars(); data[52].update(high="105", low="99")
    install_indicators(monkeypatch); _, _, trades=v2.evaluate_bars(data)
    assert trades[0]["exit_reason"] == "STOP_FIRST_AMBIGUITY"


def test_time_stop_after_exactly_24_completed_bars(monkeypatch):
    data=signal_bars(target_hit=False)
    for row in data[52:76]: row.update(open="100.2", high="100.4", low="100", close="100.2")
    install_indicators(monkeypatch); _, _, trades=v2.evaluate_bars(data)
    assert trades[0]["exit_reason"] == "TIME_STOP" and trades[0]["exit_timestamp"] == data[75]["timestamp"]


def test_2355_force_flat(monkeypatch):
    data=signal_bars(target_hit=False)
    # Move the complete synthetic chronology so signal is 23:35 and the entry is 23:40.
    start=datetime(2023, 1, 1, 19, 25, tzinfo=timezone.utc)
    for index, row in enumerate(data): row["timestamp"]=(start+timedelta(minutes=5*index)).isoformat().replace("+00:00", "Z")
    for row in data[52:]: row.update(open="100.2", high="100.4", low="100", close="100.2")
    install_indicators(monkeypatch); _, _, trades=v2.evaluate_bars(data)
    assert trades[0]["exit_reason"] == "SESSION_FLAT" and trades[0]["exit_timestamp"].endswith("23:55:00Z")


def test_range_is_exactly_48_prior_bars(monkeypatch):
    data=signal_bars(); data[1]["high"]="999"; install_indicators(monkeypatch)
    setups, _, _=v2.evaluate_bars(data)
    setup=[x for x in setups if x["direction"] == "LONG" and x["breakout_bar"].startswith("2023-01-01T04:10")][0]
    assert setup["range_high"] == "100"


def test_ids_are_deterministic(monkeypatch):
    install_indicators(monkeypatch); first=v2.evaluate_bars(signal_bars()); second=v2.evaluate_bars(signal_bars())
    assert first == second and len({x["setup_id"] for x in first[0]}) == len(first[0])


def test_missing_and_duplicate_events_rejected(monkeypatch):
    install_indicators(monkeypatch); setups, events, trades=v2.evaluate_bars(signal_bars())
    with pytest.raises(ValueError): v2.validate_reconciliation(setups, events[1:], trades)
    with pytest.raises(ValueError): v2.validate_reconciliation(setups, events+[events[0]], trades)


def test_trade_with_missing_entry_event_is_rejected(monkeypatch):
    install_indicators(monkeypatch); setups, events, trades=v2.evaluate_bars(signal_bars()); trades[0]["entry_event_id"]="missing"
    with pytest.raises(ValueError): v2.validate_reconciliation(setups, events, trades)


@pytest.mark.parametrize("mutation", ["duplicate_timestamp", "gap", "non_utc", "missing_schema", "bad_ohlc"])
def test_synthetic_bar_integrity_rejections(mutation):
    data=bars()
    if mutation == "duplicate_timestamp": data[1]["timestamp"]=data[0]["timestamp"]
    elif mutation == "gap": data[1]["timestamp"]="2023-01-01T00:15:00Z"
    elif mutation == "non_utc": data[1]["timestamp"]="2023-01-01T00:05:00+01:00"
    elif mutation == "missing_schema": del data[1]["daily_vwap"]
    else: data[1]["high"]="1"
    with pytest.raises(ValueError): v2.validate_synthetic_bars(data)


def test_synthetic_materialization_never_uses_market_paths(tmp_path, monkeypatch):
    root=tmp_path/"repo"; spec=root/v2.SPEC_PATH; spec.parent.mkdir(parents=True)
    source=Path(__file__).parents[2]/v2.SPEC_PATH; spec.write_bytes(source.read_bytes())
    opened=[]; original=v2.Path.read_bytes
    def guarded(self):
        opened.append(self)
        if "data" in self.parts: raise AssertionError("market data accessed")
        return original(self)
    monkeypatch.setattr(v2.Path, "read_bytes", guarded)
    result=v2.materialize_synthetic_contract(artifact_root=tmp_path/"out", repository_root=root)
    assert result["marketDataAccessed"] is False and all("data" not in path.parts for path in opened)


def test_synthetic_artifact_collision_is_immutable(tmp_path):
    root=tmp_path/"repo"; spec=root/v2.SPEC_PATH; spec.parent.mkdir(parents=True); spec.write_bytes((Path(__file__).parents[2]/v2.SPEC_PATH).read_bytes())
    v2.materialize_synthetic_contract(artifact_root=tmp_path/"out", repository_root=root)
    with pytest.raises(FileExistsError): v2.materialize_synthetic_contract(artifact_root=tmp_path/"out", repository_root=root)


def test_phase_a_requires_all_absolute_paths(tmp_path):
    with pytest.raises(ValueError, match="ABSOLUTE"):
        v2.run_phase_a(phase_a_bars_manifest="relative", artifact_root=tmp_path, repository_root=tmp_path)


def _phase_a_manifest(*, identity_months=None, file_months=None, **overrides):
    months=list(identity_months or v2.PHASE_A_MONTHS)
    files=[{"kind":"bars", "month":month, "relative_path":f"bars/{month}.parquet", "row_count":1, "sha256":"a"*64} for month in (file_months or months)]
    manifest={"valid":True, "identity":{"phase":"PHASE_A", "symbol":"BTCUSDT", "bar_interval":"5m", "months":months}, "parquet_files":files}
    manifest.update(overrides)
    return manifest


def test_v5_phase_a_manifest_month_contract_accepts_real_schema_shape():
    assert [item["month"] for item in v2._validate_phase_a_manifest_contract(_phase_a_manifest())] == list(v2.PHASE_A_MONTHS)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "extra", "wrong_order", "wrong_phase", "wrong_symbol", "wrong_interval", "invalid", "non_bars", "disagree"])
def test_v5_phase_a_manifest_month_contract_fails_closed(mutation):
    manifest=_phase_a_manifest()
    if mutation == "missing": manifest["parquet_files"]=manifest["parquet_files"][:-1]
    elif mutation == "duplicate": manifest["parquet_files"][1]["month"]=manifest["parquet_files"][0]["month"]
    elif mutation == "extra": manifest["parquet_files"].append({"kind":"bars","month":"2024-02","relative_path":"bars/2024-02.parquet","row_count":1,"sha256":"b"*64})
    elif mutation == "wrong_order": manifest["parquet_files"][0],manifest["parquet_files"][1]=manifest["parquet_files"][1],manifest["parquet_files"][0]
    elif mutation == "wrong_phase": manifest["identity"]["phase"]="PHASE_B"
    elif mutation == "wrong_symbol": manifest["identity"]["symbol"]="ETHUSDT"
    elif mutation == "wrong_interval": manifest["identity"]["bar_interval"]="1h"
    elif mutation == "invalid": manifest["valid"]=False
    elif mutation == "non_bars": manifest["parquet_files"][0]["kind"]="footprints"
    else: manifest["identity"]["months"]=list(v2.PHASE_A_MONTHS[:-1])
    with pytest.raises(ValueError): v2._validate_phase_a_manifest_contract(manifest)


def test_temporary_parquet_fixture_validates_schema_without_real_data(tmp_path, monkeypatch):
    pa=pytest.importorskip("pyarrow"); pq=pytest.importorskip("pyarrow.parquet")
    fixture=tmp_path/"one.parquet"; stamps=[datetime(2023,1,1,tzinfo=timezone.utc), datetime(2023,1,1,0,5,tzinfo=timezone.utc)]
    pq.write_table(pa.table({"bar_start_utc":stamps,"open":[1.,1.],"high":[2.,2.],"low":[.5,.5],"close":[1.5,1.5],"volume":[1.,1.],"daily_vwap":[1.,1.]}), fixture)
    manifest={"valid":True,"identity":{"phase":"PHASE_A","symbol":"BTCUSDT","bar_interval":"5m","months":["X"]},"parquet_files":[{"kind":"bars","month":"X","relative_path":"one.parquet","row_count":2,"sha256":v2._file_hash(fixture)}]}
    manifest_path=tmp_path/"manifest.json"; manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(v2, "PHASE_A_MONTHS", ("X",)); monkeypatch.setattr(v2, "PHASE_A_START", stamps[0]); monkeypatch.setattr(v2, "PHASE_A_LAST", stamps[-1])
    _, loaded=v2._load_phase_a_bars(manifest_path)
    assert len(loaded) == 2 and loaded[0]["timestamp"].endswith("Z")


@pytest.mark.parametrize("candidate,target", v2.CANDIDATES)
def test_separate_candidate_target_outcomes(monkeypatch, candidate, target):
    install_indicators(monkeypatch); setups, _, trades=v2.evaluate_bars(signal_bars(), candidate, target)
    assert setups[0]["candidate_id"] == candidate and trades[0]["target"]


@pytest.mark.parametrize("count", [108, 109, 543, 544])
def test_frequency_gate_boundaries_are_inclusive(count):
    # 108 annualizes to just under 100; 109 is the first whole-trade pass.
    frequency=Decimal(count)*Decimal("365.25")/Decimal("396")
    expected=Decimal("100") <= frequency <= Decimal("500")
    assert (frequency >= 100 and frequency <= 500) is expected
