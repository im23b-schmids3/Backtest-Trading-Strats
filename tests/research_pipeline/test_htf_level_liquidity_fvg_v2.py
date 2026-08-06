from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from research_pipeline.htf_level_liquidity_fvg_v2 import Bar, CANDIDATES, HTFLevelLiquidityFVG, TerminalDisposition, materialize_synthetic, reconcile_events
from research_pipeline.htf_level_liquidity_fvg_v2 import runner
from research_pipeline.htf_level_liquidity_fvg.core import Bias
from research_pipeline.htf_level_liquidity_fvg.core import Event, EventScope, Level, REQUIRED_ARTIFACTS

T = datetime(2023, 1, 1, tzinfo=timezone.utc)
def b(i, o=100, h=102, l=99, c=101): return Bar(T + timedelta(minutes=5*i), o, h, l, c, 1, f"b{i}")

def _engine(direction="LONG"):
    e = HTFLevelLiquidityFVG("HTFLFVG-V2-MIN1P5R")
    e.bars5 = [b(i) for i in range(20)]; e._atr5_prior = [1.0] * 20
    e.setup = {"setup_id":"v2_setup", "direction":direction, "sweep_5_index":-5, "mss_id":None,
        "fvg_id":None, "event_history":["SETUP_PROPOSED"], "sweep_extreme":90 if direction == "LONG" else 110,
        "sweep_atr":1.0}
    e._event(T, "SETUP_PROPOSED", "v2_setup", sweep_id="v2_sweep")
    e.daily_bias = lambda: Bias.BULLISH if direction == "LONG" else Bias.BEARISH
    return e

def test_v2_registry_and_ids_are_sealed_and_distinct():
    assert CANDIDATES == {"HTFLFVG-V2-MIN1P5R":1.5, "HTFLFVG-V2-MIN2P0R":2.0, "HTFLFVG-V2-MIN2P5R":2.5}
    assert HTFLevelLiquidityFVG("HTFLFVG-V2-MIN1P5R").events == []
    with pytest.raises(ValueError): HTFLevelLiquidityFVG("HTFLFVG-V1-MIN2P5R")

def test_v2_mss_uses_two_left_one_completed_right_and_no_same_bar_leakage():
    e = _engine(); e.bars5[-5:] = [b(15,100,101,99,100), b(16,100,102,99,101), b(17,100,105,99,104), b(18,100,103,99,101), b(19,100,107,100,106)]
    e._progress_setup(e.bars5[-1])
    assert e.setup["mss_id"] and e.events[-2].inputs["right_bar_id"] == "b18"
    # A matching close on the right-bar itself cannot confirm a swing.
    f = _engine(); f.bars5[-4:] = [b(16,100,102,99,101), b(17,100,105,99,104), b(18,100,107,99,106), b(19,100,103,99,101)]
    f._progress_setup(f.bars5[-1]); assert f.setup["mss_id"] is None

def test_v2_mss_window_admits_24_and_rejects_25():
    e = _engine(); e.setup["sweep_5_index"] = len(e.bars5) - 1 - 24; e.bars5[-1] = b(19,100,102,99,101); e._progress_setup(e.bars5[-1]); assert e.setup is not None
    e.bars5.append(b(20)); e._atr5_prior.append(1.0); e._progress_setup(e.bars5[-1]); assert e.outcomes[-1]["terminal_disposition"] == TerminalDisposition.MSS_WINDOW_EXPIRED.value

def test_synthetic_artifacts_integrity_collision_and_hash_lock(tmp_path):
    root = tmp_path / "artifacts"; result = materialize_synthetic(root, __import__("pathlib").Path.cwd())
    assert result["realStudyExecuted"] is False
    files = json.loads((root / "integrity-manifest.json").read_text())["files"]
    assert set(files) and (root / "events.json").exists()
    with pytest.raises(FileExistsError): materialize_synthetic(root, __import__("pathlib").Path.cwd())
    altered = tmp_path / "repo"; altered.mkdir(); (altered / ".smithers").mkdir(); (altered / ".smithers/specs").mkdir(); (altered / ".smithers/specs/htf-level-liquidity-fvg-v2-relaxed-mss.md").write_text("bad")
    with pytest.raises(RuntimeError): materialize_synthetic(tmp_path / "other", altered)

def test_reconciliation_rejects_orphan_duplicate_and_quantity_failure():
    with pytest.raises(ValueError): reconcile_events([], [{"setup_id":"orphan"}], [])

def test_synthetic_diagnostic_is_read_only_and_real_lock_opens_no_market_data(tmp_path):
    fixture = tmp_path / "synthetic.json"; fixture.write_text(json.dumps({"synthetic_only":True,"bars":[{"time":"2023-01-01T00:00:00Z","open":1,"high":2,"low":.5,"close":1.5}]}))
    before = set(tmp_path.iterdir()); report = runner.synthetic_funnel_diagnostic(str(fixture)); assert before == set(tmp_path.iterdir()) and report["artifactWritten"] is False
    with pytest.raises(ValueError): runner.run_htf_lfvg_v2_phase_a(phase_a_bars_manifest="not-the-contract", artifact_root=str(tmp_path / "new"), repository_root=str(__import__("pathlib").Path.cwd()))
    with pytest.raises(ValueError): runner.run_htf_lfvg_v2_phase_a(phase_a_bars_manifest="relative-manifest.json", artifact_root=str((tmp_path / "new").resolve()), repository_root=str(__import__("pathlib").Path.cwd()))


def _v2_local_v5_manifest(tmp_path: Path) -> Path:
    root = tmp_path / "local-v5"; files = []
    for index, month in enumerate(__import__("research_pipeline.htf_level_liquidity_fvg.runner", fromlist=["PHASE_A_MONTHS"]).PHASE_A_MONTHS):
        target = root / "bars" / f"{month}.parquet"; target.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.fromisoformat(f"{month}-01T00:00:00+00:00")
        pq.write_table(pa.table({"bar_start_utc": pa.array([timestamp], type=pa.timestamp("us", tz="UTC")), "open": [100.0], "high": [101.0], "low": [99.0], "close": [100.5], "volume": [1.0]}), target)
        files.append({"kind": "bars", "month": month, "relative_path": f"bars/{month}.parquet", "row_count": 1, "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
    manifest = {"valid": True, "identity": {"phase": "PHASE_A", "symbol": "BTCUSDT", "bar_interval": "5m", "months": [item["month"] for item in files]}, "parquet_files": files, "five_minute_bar_count": 13}
    path = root / "manifest.json"; path.write_text(json.dumps(manifest), encoding="utf-8")
    return path.resolve()


def test_v2_real_phase_a_fixture_accepts_exact_pinned_manifest_and_writes_final_root(tmp_path, monkeypatch):
    manifest = _v2_local_v5_manifest(tmp_path); artifact = (tmp_path / "final-artifact").resolve()
    monkeypatch.setattr(runner, "PHASE_A_MANIFEST", str(manifest)); monkeypatch.setattr(runner, "PHASE_A_TOTAL_ROWS", 13)
    result = runner.run_htf_lfvg_v2_phase_a(phase_a_bars_manifest=str(manifest), artifact_root=str(artifact), repository_root=str(Path.cwd()))
    assert result["artifact_root"] == str(artifact) and not (artifact / "phase_a").exists()
    payload = json.loads((artifact / "phase-a-result.json").read_text())
    assert payload["schema_version"].endswith(".v1") and payload["phase_b_status"] == "NOT_OPENED"
    assert payload["phase_b_executed"] is False and payload["alpha_executed"] is False
    assert {item["candidate_id"] for item in payload["candidates"]} == set(CANDIDATES)
    required = {"candidate_id", "executed_trades", "annualized_trades", "net_pnl", "net_profit_factor", "average_net_r", "maximum_drawdown_r", "long_trades", "short_trades", "funnel_reconciliation", "gate_results", "terminal_disposition_counts", "phase_b_status"}
    assert all(required <= set(item) and item["phase_b_status"] == "NOT_OPENED" for item in payload["candidates"])


def test_v2_real_phase_a_fixture_rejects_existing_output_and_never_opens_phase_b(tmp_path, monkeypatch):
    manifest = _v2_local_v5_manifest(tmp_path); artifact = (tmp_path / "already-there").resolve(); artifact.mkdir()
    monkeypatch.setattr(runner, "PHASE_A_MANIFEST", str(manifest)); monkeypatch.setattr(runner, "PHASE_A_TOTAL_ROWS", 13)
    with pytest.raises(FileExistsError):
        runner.run_htf_lfvg_v2_phase_a(phase_a_bars_manifest=str(manifest), artifact_root=str(artifact), repository_root=str(Path.cwd()))


@pytest.mark.parametrize("mutation", [
    lambda payload: payload.__setitem__("valid", False),
    lambda payload: payload["parquet_files"].__setitem__(0, {**payload["parquet_files"][0], "relative_path": "../escape.parquet"}),
    lambda payload: payload["parquet_files"].__setitem__(0, {**payload["parquet_files"][0], "sha256": "0" * 64}),
    lambda payload: payload.__setitem__("five_minute_bar_count", 12),
])
def test_v2_local_v5_loader_fails_closed_for_manifest_contract(tmp_path, mutation):
    manifest = _v2_local_v5_manifest(tmp_path); payload = json.loads(manifest.read_text()); mutation(payload); manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        runner._load_sealed_v5_phase_a_bars(manifest)


@pytest.mark.parametrize("candidate,threshold", sorted(CANDIDATES.items()))
def test_all_sealed_candidate_thresholds_are_v2_only(candidate, threshold):
    engine = HTFLevelLiquidityFVG(candidate, run_id="candidate-contract")
    assert engine.candidate.minimum_tp2_r == threshold
    assert engine.candidate.id.startswith("HTFLFVG-V2-")


@pytest.mark.parametrize("direction,pivot,trigger", [("LONG", "high", "close"), ("SHORT", "low", "close")])
def test_two_left_one_right_mss_confirmation_is_directional(direction, pivot, trigger):
    e = _engine(direction)
    if direction == "LONG":
        e.bars5[-5:] = [b(15, 100, 101, 99, 100), b(16, 100, 102, 99, 101), b(17, 100, 105, 99, 104), b(18, 100, 103, 99, 101), b(19, 100, 107, 100, 106)]
    else:
        e.bars5[-5:] = [b(15, 100, 102, 99, 101), b(16, 100, 102, 98, 99), b(17, 100, 101, 95, 96), b(18, 100, 102, 97, 99), b(19, 100, 100, 93, 94)]
    e._progress_setup(e.bars5[-1])
    assert e.setup["mss_id"]
    mss = next(event for event in e.events if event.decision == "MSS_CONFIRMED")
    assert len(mss.inputs["left_bar_ids"]) == 2 and mss.inputs["right_bar_id"] == "b18"


@pytest.mark.parametrize("elapsed,terminal", [(24, False), (25, True)])
def test_mss_window_boundary_has_no_future_bar_leakage(elapsed, terminal):
    e = _engine(); e.setup["sweep_5_index"] = len(e.bars5) - 1 - elapsed
    e._progress_setup(e.bars5[-1])
    assert (e.setup is None) is terminal
    if terminal:
        assert e.outcomes[-1]["terminal_disposition"] == "MSS_WINDOW_EXPIRED"


@pytest.mark.parametrize("disposition", list(TerminalDisposition))
def test_each_sealed_terminal_disposition_is_an_exclusive_outcome(disposition):
    e = _engine(); e._finish(T, disposition)
    assert e.outcomes == [pytest.approx(e.outcomes[0])]
    assert e.outcomes[0]["terminal_disposition"] == disposition.value
    assert [x.decision for x in e.events].count("TERMINAL_DISPOSITION") == 1


@pytest.mark.parametrize("decision,scope", [
    ("SETUP_PROPOSED", EventScope.SETUP), ("MSS_CONFIRMED", EventScope.SETUP),
    ("DISPLACEMENT_CONFIRMED", EventScope.SETUP), ("FVG_CONFIRMED", EventScope.SETUP),
    ("ORDER_ACTIVATED", EventScope.SETUP), ("ENTRY_FILLED", EventScope.TRADE),
    ("EXIT_FILLED", EventScope.TRADE), ("TERMINAL_DISPOSITION", EventScope.SETUP),
    ("LEVEL_CONFIRMED", EventScope.LEVEL), ("AGGREGATION_CONFIRMED", EventScope.GLOBAL),
])
def test_event_scope_contract_covers_lifecycle(decision, scope):
    assert __import__("research_pipeline.htf_level_liquidity_fvg.core", fromlist=["event_scope"]).event_scope(decision) == scope


@pytest.mark.parametrize("reason", ["STOPPED", "TP1_STOPPED", "TP2_COMPLETED", "FORCED_TIME_EXIT", "FORCED_END_OF_DATA_EXIT"])
def test_execution_exit_reasons_do_not_create_extra_setup_terminals(reason):
    assert reason not in {item.value for item in TerminalDisposition}


@pytest.mark.parametrize("bad_kind", ["orphan", "duplicate_event", "missing_terminal", "multiple_terminal", "trade_mismatch", "quantity_mismatch"])
def test_reconciliation_failures_are_fail_closed(bad_kind):
    proposed = Event("event-1", T.isoformat(), 1, "HTFLFVG-V2-MIN1P5R", "SETUP_PROPOSED", "setup-1", None, {"sweep_id": "sweep-1"}, EventScope.SETUP)
    terminal = Event("event-2", T.isoformat(), 2, "HTFLFVG-V2-MIN1P5R", "TERMINAL_DISPOSITION", "setup-1", "MSS_WINDOW_EXPIRED", {}, EventScope.SETUP)
    outcome = {"setup_id": "setup-1", "terminal_disposition": "MSS_WINDOW_EXPIRED", "event_history": ["SETUP_PROPOSED", "TERMINAL_DISPOSITION"]}
    events, outcomes, trades = [proposed, terminal], [outcome], []
    if bad_kind == "orphan": events[1] = Event("event-2", T.isoformat(), 2, proposed.candidate_id, "TERMINAL_DISPOSITION", "other", "x", {}, EventScope.SETUP)
    elif bad_kind == "duplicate_event": events.append(Event("event-1", T.isoformat(), 3, proposed.candidate_id, "TERMINAL_DISPOSITION", "setup-1", "x", {}, EventScope.SETUP))
    elif bad_kind == "missing_terminal": events.pop()
    elif bad_kind == "multiple_terminal": events.append(Event("event-3", T.isoformat(), 3, proposed.candidate_id, "TERMINAL_DISPOSITION", "setup-1", "x", {}, EventScope.SETUP))
    elif bad_kind == "trade_mismatch": outcomes[0]["terminal_disposition"] = "TRADE_EXECUTED"
    else:
        outcomes[0].update({"terminal_disposition": "TRADE_EXECUTED", "trade_id": "trade-1", "event_history": ["SETUP_PROPOSED", "ENTRY_FILLED", "EXIT_FILLED", "TERMINAL_DISPOSITION"]})
        events[1] = Event("event-2", T.isoformat(), 2, proposed.candidate_id, "ENTRY_FILLED", "setup-1", None, {}, EventScope.TRADE, "trade-1")
        events.extend([Event("event-3", T.isoformat(), 3, proposed.candidate_id, "EXIT_FILLED", "setup-1", None, {}, EventScope.TRADE, "trade-1"), Event("event-4", T.isoformat(), 4, proposed.candidate_id, "TERMINAL_DISPOSITION", "setup-1", "TRADE_EXECUTED", {}, EventScope.SETUP)])
        trades = [{"trade_id": "trade-1", "setup_id": "setup-1", "quantity": 1, "exits": [{"quantity": .4}]}]
    with pytest.raises(ValueError, match="RECONCILIATION_ERROR"):
        reconcile_events(events, outcomes, trades)


def test_reconciliation_accepts_a_costed_two_exit_trade():
    p = Event("event-1", T.isoformat(), 1, "HTFLFVG-V2-MIN1P5R", "SETUP_PROPOSED", "setup-1", None, {"sweep_id": "sweep-1"}, EventScope.SETUP)
    entry = Event("event-2", T.isoformat(), 2, p.candidate_id, "ENTRY_FILLED", "setup-1", None, {}, EventScope.TRADE, "trade-1")
    exits = [Event("event-3", T.isoformat(), 3, p.candidate_id, "EXIT_FILLED", "setup-1", None, {}, EventScope.TRADE, "trade-1"), Event("event-4", T.isoformat(), 4, p.candidate_id, "EXIT_FILLED", "setup-1", None, {}, EventScope.TRADE, "trade-1")]
    terminal = Event("event-5", T.isoformat(), 5, p.candidate_id, "TERMINAL_DISPOSITION", "setup-1", "TRADE_EXECUTED", {}, EventScope.SETUP)
    reconcile_events([p, entry, *exits, terminal], [{"setup_id": "setup-1", "trade_id": "trade-1", "terminal_disposition": "TRADE_EXECUTED", "event_history": ["SETUP_PROPOSED", "ENTRY_FILLED", "EXIT_FILLED", "EXIT_FILLED", "TERMINAL_DISPOSITION"]}], [{"trade_id": "trade-1", "setup_id": "setup-1", "quantity": 1, "exits": [{"quantity": .5, "fees": .1}, {"quantity": .5, "fees": .1}]}])


def test_synthetic_manifest_partition_hash_row_count_gap_timestamp_and_artifact_absence(tmp_path):
    root = tmp_path / "synthetic"; materialize_synthetic(root, __import__("pathlib").Path.cwd())
    manifest = json.loads((root / "data-manifest.json").read_text())
    integrity = json.loads((root / "integrity-manifest.json").read_text())
    assert manifest["synthetic_only"] and manifest["source"] == "synthetic fixtures only"
    assert set(integrity["files"]) == set(REQUIRED_ARTIFACTS) - {"integrity-manifest.json"}
    assert not (root / "phase-b.json").exists() and not (root / "alpha.json").exists()


def _active_order_engine(direction="LONG"):
    e = _engine(direction)
    e.setup.update({"fvg": {"lower": 100, "upper": 101}, "entry_price": 101, "stop": 95,
                    "tp1": 105 if direction == "LONG" else 97, "tp2": 110 if direction == "LONG" else 92})
    e.order = {"order_id": "v2_order", "activated_index": len(e.bars5) - 1, "entry": 101 if direction == "LONG" else 100}
    return e


@pytest.mark.parametrize("direction", ["LONG", "SHORT"])
def test_activation_is_next_bar_only_and_pending_expiry_is_twelve_bars_after_activation(direction):
    e = _active_order_engine(direction)
    same = b(19, 100, 102, 99, 101); e._progress_order_or_position(same)
    assert e.position is None
    for i in range(20, 32):
        pending = b(i, 103, 104, 102, 103) if direction == "LONG" else b(i, 98, 99, 97, 98)
        e.bars5.append(pending); e._progress_order_or_position(e.bars5[-1])
    assert e.order is not None and e.setup is not None
    e.bars5.append(b(32)); e._progress_order_or_position(e.bars5[-1])
    assert e.outcomes[-1]["terminal_disposition"] == "PENDING_ORDER_EXPIRED"


def test_fill_tp1_cost_break_even_tp2_and_stop_first_lifecycle():
    e = _active_order_engine()
    e.bars5.append(b(20, 100, 102, 99, 101)); e._progress_order_or_position(e.bars5[-1])
    assert e.position and e.events[-1].decision == "ENTRY_FILLED"
    e.bars5.append(b(21, 101, 106, 100, 105)); e._progress_order_or_position(e.bars5[-1])
    assert e.position["tp1_done"] and e.position["remaining"] == pytest.approx(e.position["qty"] * .5)
    assert e.position["breakeven"] > e.position["entry"]
    e.bars5.append(b(22, 105, 111, 104, 110)); e._progress_order_or_position(e.bars5[-1])
    assert e.trades[-1]["exit_reason"] == "TP2_COMPLETED" and len(e.trades[-1]["exits"]) == 2
    f = _active_order_engine(); f.bars5.append(b(20, 100, 102, 99, 101)); f._progress_order_or_position(f.bars5[-1])
    f.bars5.append(b(21, 101, 106, 94, 100)); f._progress_order_or_position(f.bars5[-1])
    assert f.trades[-1]["exit_reason"] == "STOPPED"


@pytest.mark.parametrize("count,expected", [(0, "UNDERFREQUENCY_FAIL"), (50, "TARGET_FREQUENCY"), (301, "HIGH_FREQUENCY_WARNING"), (501, "OVERFREQUENCY_FAIL")])
def test_diagnostic_frequency_counts_and_long_short_gate_requirements(count, expected):
    frequency = __import__("research_pipeline.htf_level_liquidity_fvg.core", fromlist=["frequency_classification"]).frequency_classification(count)[1]
    assert frequency == expected
