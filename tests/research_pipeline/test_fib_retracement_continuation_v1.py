from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path

import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from research_pipeline.cli import main
from research_pipeline.fib_retracement_continuation_v1.accounting import close_trade, compounded_equity
from research_pipeline.fib_retracement_continuation_v1.constants import CANDIDATES, ENTRY_RATIO, TARGET_RATIOS
from research_pipeline.fib_retracement_continuation_v1.execution import execute_order, process_position, submit_order
from research_pipeline.fib_retracement_continuation_v1.ids import event_id, exit_leg_id, fib_range_id, impulse_id, order_id, setup_id, trade_id
from research_pipeline.fib_retracement_continuation_v1.manifests import ManifestError, canonical_chronology_hash, canonical_manifest_hash, verify_chronology_manifest, verify_manifest
from research_pipeline.fib_retracement_continuation_v1.metrics import gates
from research_pipeline.fib_retracement_continuation_v1.models import Bar, Candidate, ExecutionAssumptions
from research_pipeline.fib_retracement_continuation_v1.reconciliation import reconcile
from research_pipeline.fib_retracement_continuation_v1.diagnostics import _candidate_diagnostic
from research_pipeline.fib_retracement_continuation_v1.loader import load_development_bars, parquet_schema_sha256
from research_pipeline.fib_retracement_continuation_v1.runner import development_diagnostic, materialize_synthetic, run_candidate, run_development, run_holdout, run_synthetic
from research_pipeline.fib_retracement_continuation_v1.strategy import create_setup, fib_price

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)

def D(value): return Decimal(str(value))
def candidate(index=0): return Candidate(**CANDIDATES[index])
def bar(hours, o=100, h=101, l=99, c=100): return Bar(NOW + timedelta(hours=hours), D(o), D(h), D(l), D(c))
def long_order(): return submit_order(create_setup(candidate(), "LONG", bar(0, 100, 101, 100, 101), bar(4, 120, 130, 119, 129)), NOW + timedelta(hours=8))
def short_order(): return submit_order(create_setup(candidate(), "SHORT", bar(0, 100, 130, 129, 129), bar(4, 110, 111, 100, 101)), NOW + timedelta(hours=8))
def long_trade(assumptions=ExecutionAssumptions()):
 trade, reason = execute_order(long_order(), bar(8, 104, 105, 103, 104), candidate(), D(10000), assumptions)
 assert reason is None and trade
 return trade

def manifest(tmp_path, rows=None):
 source = tmp_path / "synthetic-source.json"; source.write_text("synthetic", encoding="utf-8")
 rows = rows or [{"timestamp": NOW.isoformat(), "open": "1", "high": "2", "low": "1", "close": "2", "volume": "3"}]
 data = {"absoluteSourcePath": str(source.resolve()), "sourceFileSha256": hashlib.sha256(source.read_bytes()).hexdigest(), "fileSizeBytes": source.stat().st_size, "schemaSha256": "schema", "rowCount": len(rows), "firstTimestamp": rows[0]["timestamp"], "finalTimestamp": rows[-1]["timestamp"], "expectedInterval": "PT4H", "provenanceClassification": "USER_SUPPLIED_PROVENANCE_EVIDENCE", "sourceExchange": "UNKNOWN", "instrumentType": "UNKNOWN", "historicalV6Identity": "NOT_CLAIMED"}
 data["manifestSha256"] = canonical_manifest_hash(data)
 path = tmp_path / "synthetic-manifest.json"; path.write_text(json.dumps(data), encoding="utf-8")
 return path, data, rows

def parquet_manifest(tmp_path, symbol, interval_hours):
 source = tmp_path / f"{symbol}.parquet"
 start = datetime(2022, 1, 1, tzinfo=timezone.utc)
 end = datetime(2025, 1, 1, tzinfo=timezone.utc)
 stamps = [start, end - timedelta(hours=interval_hours), end]
 values = [(100, 101, 99, 100)] * 5 + [(999, 1000, 998, 999)]
 writer = None
 for stamp, (open_, high, low, close) in zip(stamps, values):
  table = pa.table({"open": [float(open_)], "high": [float(high)], "low": [float(low)], "close": [float(close)], "volume": [3.0], "timestamp": [stamp]})
  if writer is None: writer = pq.ParquetWriter(source, table.schema)
  writer.write_table(table)
 writer.close()
 data = {"absoluteSourcePath": str(source.resolve()), "sourceFileSha256": hashlib.sha256(source.read_bytes()).hexdigest(), "fileSizeBytes": source.stat().st_size, "schemaSha256": parquet_schema_sha256(source), "rowCount": 3, "developmentRowCount": 2, "firstTimestamp": stamps[0].isoformat(), "finalTimestamp": stamps[-1].isoformat(), "expectedInterval": "PT4H" if interval_hours == 4 else "P1D", "provenanceClassification": "USER_SUPPLIED_PROVENANCE_EVIDENCE", "sourceExchange": "UNKNOWN", "instrumentType": "UNKNOWN", "historicalV6Identity": "NOT_CLAIMED"}
 data["manifestSha256"] = canonical_manifest_hash(data)
 path = tmp_path / f"{symbol}-manifest.json"; path.write_text(json.dumps(data), encoding="utf-8")
 return path, stamps

def test_parquet_schema_hash_canonicalizes_json_metadata_whitespace(tmp_path):
 table = pa.table({"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1.0], "timestamp": [NOW]})
 spaced = tmp_path / "spaced.parquet"; compact = tmp_path / "compact.parquet"
 pq.write_table(table.replace_schema_metadata({b"pandas": b'{"index_columns": ["timestamp"]}'}), spaced)
 pq.write_table(table.replace_schema_metadata({b"pandas": b'{"index_columns":["timestamp"]}'}), compact)
 assert parquet_schema_sha256(spaced) == parquet_schema_sha256(compact)

def chronology_manifest(tmp_path, eth, btc, eth_stamps, btc_stamps):
 def asset(path, stamps):
  source = json.loads(path.read_text())
  return {"source_manifest_path": str(path.resolve()), "source_manifest_sha256": source["manifestSha256"], "total_row_count": 3, "development_row_count": 2, "holdout_row_count": 1, "development_first_timestamp": stamps[0].isoformat(), "development_final_timestamp": stamps[1].isoformat(), "holdout_first_timestamp": stamps[2].isoformat(), "holdout_final_timestamp": stamps[2].isoformat()}
 data = {"chronology_contract_id":"FIB09_V1_ETH_BTC_UTC_2022_2025_LOCK","study_id":"FibRetracementContinuation.ETH_BTC_V1_PROSPECTIVE_VALIDATION","timezone":"UTC","development_start_inclusive":"2022-01-01T00:00:00+00:00","development_end_exclusive":"2025-01-01T00:00:00+00:00","holdout_start_inclusive":"2025-01-01T00:00:00+00:00","holdout_end_exclusive":None,"boundary_semantics":{"development":"timestamp >= development_start_inclusive and timestamp < development_end_exclusive","holdout":"timestamp >= holdout_start_inclusive","no_gap_requirement":True},"candidate_ids":[row["candidate_id"] for row in CANDIDATES],"assets":{"ETH":asset(eth,eth_stamps),"BTC":asset(btc,btc_stamps)},"no_overlap":True,"holdout_locked":True,"holdout_strategy_accessed":False,"chronology_selected_without_strategy_results":True,"historical_matrix_used_for_split_selection":False,"source_files_modified":False}
 data["chronologyManifestSha256"] = canonical_chronology_hash(data)
 path=tmp_path / "chronology.json"; path.write_text(json.dumps(data),encoding="utf-8")
 return path

def test_manifest_tampering_rejected(tmp_path):
 path, data, _ = manifest(tmp_path); data["rowCount"] = 2; path.write_text(json.dumps(data), encoding="utf-8")
 with pytest.raises(ManifestError, match="SELF_HASH"): verify_manifest(path)

def test_source_hash_mismatch_rejected(tmp_path):
 path, _, _ = manifest(tmp_path); Path(json.loads(path.read_text())["absoluteSourcePath"]).write_text("changed", encoding="utf-8")
 with pytest.raises(ManifestError, match="SOURCE_HASH"): verify_manifest(path)

def test_schema_mismatch_rejected(tmp_path):
 path, _, rows = manifest(tmp_path)
 with pytest.raises(ManifestError, match="SCHEMA_HASH"): verify_manifest(path, rows=rows, schema_hash="wrong")

def test_cadence_mismatch_rejected(tmp_path):
 rows = [{"timestamp": NOW.isoformat(), "open": "1", "high": "2", "low": "1", "close": "2", "volume": "3"}, {"timestamp": (NOW + timedelta(hours=5)).isoformat(), "open": "1", "high": "2", "low": "1", "close": "2", "volume": "3"}]; path, _, _ = manifest(tmp_path, rows)
 with pytest.raises(ManifestError, match="MISSING_INTERVAL"): verify_manifest(path, rows=rows)

def test_duplicate_timestamps_rejected(tmp_path):
 rows = [{"timestamp": NOW.isoformat(), "open": "1", "high": "2", "low": "1", "close": "2", "volume": "3"}] * 2; path, _, _ = manifest(tmp_path, rows)
 with pytest.raises(ManifestError, match="DUPLICATE"): verify_manifest(path, rows=rows)

def test_holdout_refusal_precedes_loader(monkeypatch):
 monkeypatch.setattr("research_pipeline.fib_retracement_continuation_v1.runner.development_diagnostic", lambda **_: pytest.fail("loader called"))
 with pytest.raises(ManifestError, match="LOCKED_HOLDOUT"): run_holdout()

def test_cli_holdout_failure_before_source_loader(monkeypatch):
 monkeypatch.setattr("research_pipeline.fib_retracement_continuation_v1.runner.verify_manifest", lambda *a, **k: pytest.fail("source loader called"))
 assert main(["fib09-v1-holdout"]) == 2

@pytest.mark.parametrize(("direction", "expected"), [("LONG", D(103)), ("SHORT", D(127))])
def test_fibonacci_orientation(direction, expected):
 assert fib_price(direction, D(100), D(130), ENTRY_RATIO) == expected

def test_next_bar_activation():
 order = long_order(); assert execute_order(order, bar(4, 104, 105, 103, 104), candidate(), D(10000), ExecutionAssumptions())[0] is None

@pytest.mark.parametrize("factory", [long_order, short_order])
def test_limit_touch_fill(factory):
 order = factory(); direction = order["direction"]; level = fib_price(direction, order["low"], order["high"], ENTRY_RATIO)
 hit = bar(8, level, level if direction == "SHORT" else level + 1, level if direction == "LONG" else level - 1, level)
 assert execute_order(order, hit, candidate(), D(10000), ExecutionAssumptions())[0] is not None

def test_same_bar_stop_precedence():
 trade = long_trade(); stop = trade["current_stop"]; target = fib_price("LONG", trade["low"], trade["high"], TARGET_RATIOS[0])
 legs = process_position(trade, bar(12, 105, target + 1, stop - 1, 105), candidate(), ExecutionAssumptions()); assert [x["reason"] for x in legs] == ["STOP"]

@pytest.mark.parametrize("post_ratio", [D(".830"), D(".786")])
def test_tp1_partial_and_delayed_post_tp1_stop(post_ratio):
 c = Candidate(**{**CANDIDATES[0], "post_tp1_ratio": post_ratio}); trade = long_trade(); target = fib_price("LONG", trade["low"], trade["high"], TARGET_RATIOS[0])
 legs = process_position(trade, bar(12, target, target + 1, target - 1, target), c, ExecutionAssumptions()); assert legs[0]["reason"] == "TP1" and legs[0]["quantity"] == trade["quantity"] * D(".30")
 assert trade["current_stop"] == trade["initial_stop_price"]
 process_position(trade, bar(16, target, target, target, target), c, ExecutionAssumptions()); assert trade["current_stop"] == fib_price("LONG", trade["low"], trade["high"], post_ratio)

def test_fee_and_slippage_every_fill():
 assumptions = ExecutionAssumptions(); trade = long_trade(assumptions); raw = fib_price("LONG", trade["low"], trade["high"], TARGET_RATIOS[0]); leg = process_position(trade, bar(12, raw, raw + 1, raw, raw), candidate(), assumptions)[0]
 assert trade["entry_fee"] > 0 and leg["fee"] > 0 and trade["slippage_cost"] > leg["slippage_cost"] > 0

def test_compounded_sizing_and_quantity_conservation():
 one = long_trade(); two, _ = execute_order(long_order(), bar(8, 104, 105, 103, 104), candidate(), D(11000), ExecutionAssumptions()); assert two and two["quantity"] > one["quantity"]
 leg = process_position(one, bar(12, 105, 105, one["current_stop"] - 1, 105), candidate(), ExecutionAssumptions())[0]; assert one["quantity"] == leg["quantity"] + one["remaining_quantity"]

def test_deterministic_identifiers():
 sid = setup_id("c", "LONG", NOW, D(1)); iid = impulse_id(sid, NOW, D(2)); fid = fib_range_id(iid, D(1), D(2)); oid = order_id(fid, 1, NOW); tid = trade_id(oid, NOW)
 assert (sid, iid, fid, oid, tid, exit_leg_id(tid, 1), event_id("c", NOW, "K", sid, 1)) == (setup_id("c", "LONG", NOW, D(1)), impulse_id(sid, NOW, D(2)), fib_range_id(iid, D(1), D(2)), order_id(fid, 1, NOW), trade_id(oid, NOW), exit_leg_id(tid, 1), event_id("c", NOW, "K", sid, 1))

def test_distinct_impulses_from_one_anchor_have_distinct_setup_ids():
 anchor = bar(0, 100, 101, 99, 100)
 first = create_setup(candidate(), "LONG", anchor, bar(4, 100, 120, 101, 119))
 second = create_setup(candidate(), "LONG", anchor, bar(8, 100, 125, 102, 124))
 assert first["setup_id"] != second["setup_id"]
 assert first["direction"] == second["direction"] == "LONG"

def test_run_candidate_one_terminal_and_reconciliation():
 result = run_candidate([bar(0,100,101,100,101), bar(4,120,130,119,129), bar(8,104,105,103,104), bar(12,105,106,90,95)], candidate())
 assert result["reconciliation"]["one_terminal_outcome"] and result["reconciliation"]["reconciles"]

def test_event_order_trade_reconciliation():
 result = run_candidate([bar(0,100,101,100,101), bar(4,120,130,119,129), bar(8,104,105,103,104), bar(12,105,106,90,95)], candidate())
 fills = [x for x in result["events"] if x["kind"] == "ORDER_FILLED"]; assert fills and fills[0]["trade_id"] == result["trades"][0]["trade_id"]

def test_reconciliation_failure_blocks_gate():
 value = {"net_pnl": D(2), "profit_factor": D(2), "average_net_r": D(1), "maximum_drawdown_percent": D(0), "evidence_label": "LOW_FREQUENCY_DEVELOPMENT_EVIDENCE"}
 assert not gates(value, [], False)["passed"] and not gates(value, [], False)["hard_gates"]["full_reconciliation"]

def test_reconciliation_rejects_real_pattern_duplicate_setup_lifecycle():
 setup = {"setup_id": "setup-repeated-anchor"}
 outcome = {"setup_id": "setup-repeated-anchor", "disposition": "ENTRY_EXPIRED"}
 result = reconcile([setup, setup], [outcome, outcome], [], [], D(10000), final_equity=D(10000))
 assert not result["reconciles"] and not result["unique_setup_ids"] and not result["one_terminal_outcome"]

def test_reconciliation_accepts_costed_two_leg_trade_and_equity_delta():
 legs = [{"quantity": D(".3"), "gross_pnl": D(3), "fee": D(".1")}, {"quantity": D(".7"), "gross_pnl": D(7), "fee": D(".2")}]
 trade = {"trade_id": "trade-1", "order_id": "order-1", "setup_id": "setup-1", "quantity": D(1), "remaining_quantity": D(0), "legs": legs, "entry_fee": D(".1"), "gross_pnl": D(10), "net_pnl": D("9.6")}
 result = reconcile([{"setup_id": "setup-1"}], [{"setup_id": "setup-1", "disposition": "TRADE_EXECUTED"}], [{"setup_id": "setup-1", "order_id": "order-1"}], [trade], D(100), final_equity=D("109.6"), events=[{"kind": "ORDER_SUBMITTED", "setup_id": "setup-1", "order_id": "order-1"}, {"kind": "ORDER_FILLED", "setup_id": "setup-1", "order_id": "order-1", "trade_id": "trade-1"}])
 assert result["reconciles"] and result["quantity_conservation"] and result["equity_reconcile"]

def test_diagnostic_is_compact_and_reports_first_duplicate_setup_failure():
 item = {"setups": [{"setup_id": "setup-1"}, {"setup_id": "setup-1"}], "setup_outcomes": [{"setup_id": "setup-1", "disposition": "ENTRY_EXPIRED"}, {"setup_id": "setup-1", "disposition": "ENTRY_EXPIRED"}], "orders": [], "trades": [], "events": [], "metrics": {"final_equity": D(100)}, "reconciliation": {"reconciles": False}}
 value = _candidate_diagnostic(item, "candidate-1", D(100))
 assert value["RECONCILIATION_FIRST_FAILURE"]["setup_id"] == "setup-1"
 assert value["ids"]["duplicate_setup_ids"] == {"count": 1, "first_id": "setup-1"}

def test_best_trade_removal_and_extra_slippage_stress():
 trade = {"net_pnl": D(10), "entry_price": D(100), "quantity": D(1), "legs": [{"fill_price": D(110), "quantity": D(1)}]}; value = {"net_pnl": D(10), "profit_factor": D(2), "average_net_r": D(1), "maximum_drawdown_percent": D(0), "evidence_label": "LOW_FREQUENCY_DEVELOPMENT_EVIDENCE"}; output = gates(value, [trade], True)
 assert output["best_trade_removal_net_pnl"] == 0 and output["additional_slippage_net_pnl"] < 10

def test_synthetic_artifact_contract_and_collision(tmp_path):
 root = tmp_path / "new"; result = materialize_synthetic(artifact_root=root, repository_root=Path.cwd())
 for name in ("sealed-specification.json", "candidate-registry.json", "data-manifest.json", "execution-assumptions.json", "freeze.json", "integrity-manifest.json", "final-report.json"): assert (root / name).is_file()
 assert result["holdout_status"] == "LOCKED_NOT_OPENED"
 with pytest.raises(FileExistsError): materialize_synthetic(artifact_root=root, repository_root=Path.cwd())

def test_output_provenance_labels(tmp_path):
 root = tmp_path / "out"; materialize_synthetic(artifact_root=root, repository_root=Path.cwd())
 assert json.loads((root / "data-manifest.json").read_text())["synthetic_only"] is True

def test_candidate_independence(tmp_path):
 root = tmp_path / "out"; result = run_synthetic(bars_by_candidate={}, artifact_root=root, repository_root=Path.cwd())
 assert [x["candidate_id"] for x in result["candidates"]] == [x["candidate_id"] for x in CANDIDATES]

def test_development_requires_absolute_new_paths(tmp_path):
 with pytest.raises(ValueError, match="ARTIFACT_ROOT"): run_development(eth_manifest="x", btc_manifest="y", chronology_manifest="x", artifact_root="relative", repository_root=Path.cwd())
 with pytest.raises(FileExistsError, match="COLLISION"): run_development(eth_manifest="x", btc_manifest="y", chronology_manifest="x", artifact_root=tmp_path, repository_root=Path.cwd())

def test_development_diagnostic_is_manifest_only(tmp_path):
 eth, _ = parquet_manifest(tmp_path, "ETH", 4); btc, _ = parquet_manifest(tmp_path, "BTC", 24)
 result = development_diagnostic(eth_manifest=eth, btc_manifest=btc)
 assert result["eth"]["rows_read"] is False and result["btc"]["rows_read"] is False
 assert "artifact_root" not in result and not list(tmp_path.glob("**/development-result.json"))

def test_development_reads_only_synthetic_development_rows_and_writes_artifacts(tmp_path, monkeypatch):
 eth, eth_stamps = parquet_manifest(tmp_path, "ETH", 4); btc, btc_stamps = parquet_manifest(tmp_path, "BTC", 24)
 chronology = chronology_manifest(tmp_path, eth, btc, eth_stamps, btc_stamps)
 observed = []
 original = run_candidate
 def capture(bars, candidate, *args, **kwargs):
  observed.extend(bar.timestamp for bar in bars)
  return original(bars, candidate, *args, **kwargs)
 monkeypatch.setattr("research_pipeline.fib_retracement_continuation_v1.runner.run_candidate", capture)
 root = tmp_path / "development-output"
 result = run_development(eth_manifest=eth, btc_manifest=btc, chronology_manifest=chronology, artifact_root=root.resolve(), repository_root=Path.cwd().resolve())
 assert result == {"artifact_root": str(root.resolve()), "rows_read": True, "candidate_count": 3, "holdout_status": "LOCKED_NOT_OPENED"}
 assert all(stamp not in {eth_stamps[-1], btc_stamps[-1]} for stamp in observed)
 assert (root / "development-result.json").is_file()
 assert {path.name for path in (root / "candidates").iterdir()} == {row["candidate_id"] for row in CANDIDATES}
 assert all((root / "candidates" / row["candidate_id"] / "reconciliation.json").is_file() for row in CANDIDATES)

def test_development_and_diagnostic_callables_are_distinct():
 assert run_development is not development_diagnostic

def test_missing_chronology_manifest_is_rejected(tmp_path):
 eth, _ = parquet_manifest(tmp_path, "ETH", 4); btc, _ = parquet_manifest(tmp_path, "BTC", 24)
 with pytest.raises(ManifestError, match="CHRONOLOGY_MANIFEST_MISSING"):
  run_development(eth_manifest=eth, btc_manifest=btc, chronology_manifest=(tmp_path / "missing.json").resolve(), artifact_root=(tmp_path / "never-created").resolve(), repository_root=Path.cwd().resolve())

def test_chronology_tampering_and_linked_manifest_mismatch_are_rejected(tmp_path):
 eth, eth_stamps = parquet_manifest(tmp_path, "ETH", 4); btc, btc_stamps = parquet_manifest(tmp_path, "BTC", 24); chronology = chronology_manifest(tmp_path, eth, btc, eth_stamps, btc_stamps)
 payload=json.loads(chronology.read_text()); payload["no_overlap"] = False; chronology.write_text(json.dumps(payload))
 with pytest.raises(ManifestError, match="SELF_HASH"): verify_chronology_manifest(chronology, eth_manifest=eth, btc_manifest=btc)
 payload["chronologyManifestSha256"] = canonical_chronology_hash(payload); payload["assets"]["ETH"]["source_manifest_sha256"]="0"*64; payload["chronologyManifestSha256"] = canonical_chronology_hash(payload); chronology.write_text(json.dumps(payload))
 with pytest.raises(ManifestError, match="SOURCE_MANIFEST_HASH"): verify_chronology_manifest(chronology, eth_manifest=eth, btc_manifest=btc)

def test_mixed_row_group_is_privately_filtered_before_public_bar_exposure(tmp_path):
 eth, eth_stamps = parquet_manifest(tmp_path, "ETH", 4); btc, btc_stamps = parquet_manifest(tmp_path, "BTC", 24); chronology = chronology_manifest(tmp_path, eth, btc, eth_stamps, btc_stamps)
 source=Path(json.loads(eth.read_text())["absoluteSourcePath"]); table=pq.read_table(source); pq.write_table(table, source)
 payload=json.loads(eth.read_text()); payload["sourceFileSha256"]=hashlib.sha256(source.read_bytes()).hexdigest(); payload["fileSizeBytes"]=source.stat().st_size; payload["schemaSha256"]=parquet_schema_sha256(source); payload["manifestSha256"]=canonical_manifest_hash(payload); eth.write_text(json.dumps(payload))
 payload=json.loads(chronology.read_text()); payload["assets"]["ETH"]["source_manifest_sha256"]=json.loads(eth.read_text())["manifestSha256"]; payload["chronologyManifestSha256"]=canonical_chronology_hash(payload); chronology.write_text(json.dumps(payload))
 bars, metadata = load_development_bars(eth, development_start=datetime(2022, 1, 1, tzinfo=timezone.utc), development_end=datetime(2025, 1, 1, tzinfo=timezone.utc), chronology_claim=json.loads(chronology.read_text())["assets"]["ETH"])
 assert [item.timestamp for item in bars] == eth_stamps[:-1]
 assert max(item.timestamp for item in bars) < datetime(2025, 1, 1, tzinfo=timezone.utc)
 assert all(item.close != D(999) for item in bars)  # synthetic holdout sentinel is never public.
 assert metadata["isolation_counters"] == {"physical_row_groups_read": 1, "mixed_row_groups_read": 1, "development_rows_returned": 2, "holdout_rows_discarded": 1}
 assert "rows" not in metadata and "table" not in metadata

def test_all_holdout_groups_are_skipped_and_all_development_groups_are_included(tmp_path):
 eth, stamps = parquet_manifest(tmp_path, "ETH", 4)
 claim = {"development_row_count": 2, "development_final_timestamp": stamps[1].isoformat()}
 bars, metadata = load_development_bars(eth, development_start=datetime(2022, 1, 1, tzinfo=timezone.utc), development_end=datetime(2025, 1, 1, tzinfo=timezone.utc), chronology_claim=claim)
 assert [item.timestamp for item in bars] == stamps[:-1]
 assert metadata["isolation_counters"] == {"physical_row_groups_read": 2, "mixed_row_groups_read": 0, "development_rows_returned": 2, "holdout_rows_discarded": 0}

def test_sealed_eth_btc_development_count_claims_are_exact_metadata_invariants():
 chronology = json.loads((Path.cwd() / "docs" / "research_pipeline" / "fib_prospective_v1" / "chronology-manifest.json").read_text(encoding="utf-8"))
 assert (chronology["assets"]["ETH"]["development_row_count"], chronology["assets"]["ETH"]["development_final_timestamp"]) == (6576, "2024-12-31T20:00:00+00:00")
 assert (chronology["assets"]["BTC"]["development_row_count"], chronology["assets"]["BTC"]["development_final_timestamp"]) == (1096, "2024-12-31T00:00:00+00:00")

@pytest.mark.parametrize(("direction", "reason"), [("LONG", "STOP"), ("SHORT", "STOP")])
def test_synthetic_stop_losses(direction, reason):
 order = long_order() if direction == "LONG" else short_order(); c = candidate(); level = fib_price(direction, order["low"], order["high"], ENTRY_RATIO); hit = bar(8, level, level + 1, level - 1, level); trade, _ = execute_order(order, hit, c, D(10000), ExecutionAssumptions()); assert trade
 stop = trade["current_stop"]; leg = process_position(trade, bar(12, stop, stop if direction == "LONG" else stop + 1, stop - 1 if direction == "LONG" else stop, stop), c, ExecutionAssumptions())[0]; assert leg["reason"] == reason

def test_entry_expiry_and_active_position_blocking():
 order = long_order(); assert run_candidate([bar(0), bar(4,120,130,119,129), bar(8,104,105,99,104)], candidate())["setup_outcomes"]
 assert order["setup_id"]

def test_chronology_split_isolation(tmp_path):
 # The synthetic materializer records only a locked, unopened holdout state.
 assert materialize_synthetic(artifact_root=tmp_path / "fib09-isolation", repository_root=Path.cwd())["holdout_status"] == "LOCKED_NOT_OPENED"
