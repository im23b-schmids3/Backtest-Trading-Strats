from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
import subprocess

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from research_pipeline.cli import main
from research_pipeline.liquidity_sweep_mean_reversion.artifacts import validate_artifact_tree
from research_pipeline.liquidity_sweep_mean_reversion_v2.models import LSMRV2Config, TERMINAL_DISPOSITIONS, preregistered_candidates
from research_pipeline.liquidity_sweep_mean_reversion_v2.runner import _load_phase_a_bars, materialize_lsmr_v2_strict_contract, run_lsmr_v2_phase_a
from research_pipeline.liquidity_sweep_mean_reversion_v2.strategy import detect_setups, terminal_disposition, validate_setup_audit


def _bars(count=70, start=None):
    start = start or datetime(2023, 1, 1, 0, 0, tzinfo=timezone.utc)
    return [{"timestamp": start + timedelta(minutes=5 * index), "open": "100", "high": "101", "low": "99", "close": "100", "volume": "10", "daily_vwap": "100"} for index in range(count)]


def _setup(index=24): return {"setup_id": "setup", "structure_id": "structure", "direction": "LONG", "reference": "99", "extreme": "99.0", "sweep_index": index, "event_history": [{"event": "PROPOSED_SETUP"}]}
def _reclaim(rows, index=24): rows[index].update(open="98", high="101", low="99", close="100", volume="20", daily_vwap="100")


def test_v2_registry_and_sealed_configuration():
    assert [(c.candidate_id, c.target_r_multiple) for c in preregistered_candidates()] == [("LSMR-V2-2P0R", Decimal("2.0")), ("LSMR-V2-2P5R", Decimal("2.5")), ("LSMR-V2-3P0R", Decimal("3.0"))]
    with pytest.raises(ValueError, match="invariant"): LSMRV2Config(reference_bars=12)


def test_v2_reference_penetration_floor_and_repeat_extreme_history():
    rows = _bars(); rows[24]["low"] = "98.5"
    setups, events = detect_setups(rows, LSMRV2Config())
    assert setups[0]["reference"] == "99" and setups[0]["event_history"][0]["event"] == "PROPOSED_SETUP"
    assert events[0]["setup_id"] == setups[0]["setup_id"]


@pytest.mark.parametrize(("change", "expected"), [(lambda rows: rows[24].update(open="99.9", close="100"), "CANDLE_REJECTED"), (lambda rows: rows[24].update(volume="14.9"), "VOLUME_REJECTED"), (lambda rows: rows[0].update(daily_vwap="99"), "REGIME_REJECTED"), (lambda rows: [row.update(daily_vwap="101") for row in rows], "VWAP_PROXIMITY_REJECTED")])
def test_v2_reclaim_filters_have_exact_terminal_dispositions(change, expected):
    rows = _bars(); _reclaim(rows); change(rows)
    assert terminal_disposition(_setup(), rows, LSMRV2Config())[0] == expected


def test_v2_session_context_requires_24_same_session_bars():
    rows = _bars(start=datetime(2023, 1, 1, 22, 0, tzinfo=timezone.utc)); _reclaim(rows)
    assert terminal_disposition(_setup(), rows, LSMRV2Config())[0] == "SESSION_CONTEXT_UNAVAILABLE"


def test_v2_stop_limits_next_open_and_two_bar_reclaim_window():
    rows = _bars(); rows[24]["close"] = rows[25]["close"] = "99"; _reclaim(rows, 26)
    assert terminal_disposition(_setup(), rows, LSMRV2Config())[0] == "RECLAIM_WINDOW_EXPIRED"
    rows = _bars(); _reclaim(rows); rows[25]["open"] = "100"
    disposition, trade = terminal_disposition(_setup(), rows, LSMRV2Config())
    assert disposition == "TRADE_EXECUTED" and trade and trade["entry_price"] == "100" and trade["trade_id"]


def test_v2_terminal_audit_is_exact_and_requires_trade_id():
    setup = _setup(); trade = {"setup_id": "setup", "trade_id": "trade"}
    validate_setup_audit([setup], [], [trade], [{"setup_id": "setup", "disposition": "TRADE_EXECUTED"}])
    with pytest.raises(AssertionError): validate_setup_audit([setup], [], [], [{"setup_id": "setup", "disposition": "TRADE_EXECUTED"}])
    assert "COOLDOWN_BLOCKED" in TERMINAL_DISPOSITIONS and "DUPLICATE_REFERENCE_SUPPRESSED" in TERMINAL_DISPOSITIONS


def test_v2_synthetic_cli_is_deterministic_and_never_executes_a_study(tmp_path, capsys):
    root = tmp_path / "repo"; spec = root / ".smithers/specs/liquidity-sweep-mean-reversion-v2-strict.md"; spec.parent.mkdir(parents=True)
    spec.write_text("\n".join(["LiquiditySweepMeanReversion.BTC_LONG_SHORT_V2_STRICT_SELECTION", "LSMR-V2-2P0R=2R LSMR-V2-2P5R=2.5R LSMR-V2-3P0R=3R", "SESSION_CONTEXT_UNAVAILABLE TRADE_EXECUTED"]), encoding="utf-8")
    result = materialize_lsmr_v2_strict_contract(repository_root=root, artifact_root=tmp_path / "runs")
    assert result["realStudyExecuted"] is False and validate_artifact_tree(result["artifactRoot"])["valid"]
    assert main(["lsmr-v2-strict-materialize", "--repository-root", str(root), "--artifact-root", str(tmp_path / "cli-runs")]) == 0
    assert '"realStudyExecuted": false' in capsys.readouterr().out


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _phase_a_manifest(tmp_path):
    """Synthetic 5-minute OHLCV/VWAP data only; no repository market data is read."""
    root = tmp_path / "bars"; root.mkdir()
    start = datetime(2023, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 31, 23, 55, tzinfo=timezone.utc)
    rows_by_month = {}
    stamp = start
    while stamp <= end:
        rows_by_month.setdefault(stamp.strftime("%Y-%m"), []).append({"timestamp": stamp, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 10.0, "daily_vwap": 100.0})
        stamp += timedelta(minutes=5)
    # Synthetic fixture mirrors the authorized count and documented source gaps:
    # September has 8,637 rows, but timestamps remain strictly increasing.
    rows_by_month["2023-09"] = [row for index, row in enumerate(rows_by_month["2023-09"]) if index not in {100, 200, 300}]
    rows_by_month["2023-01"] = [row for index, row in enumerate(rows_by_month["2023-01"]) if index not in set(range(100, 388))]
    files = []
    for month, rows in rows_by_month.items():
        path = root / f"{month}.parquet"; pq.write_table(pa.Table.from_pylist(rows), path)
        files.append({"month": month, "relative_path": path.name, "row_count": len(rows), "sha256": _sha256(path)})
    manifest = {"valid": True, "identity": {"phase": "PHASE_A", "symbol": "BTCUSDT", "bar_interval": "5m", "months": list(rows_by_month), "five_minute_bar_count": 113757, "study_start": start.isoformat(), "study_end": end.isoformat()}, "parquet_files": files}
    path = root / "manifest.json"; path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _sealed_clean_repository(tmp_path):
    root = tmp_path / "repo"; spec = root / ".smithers/specs/liquidity-sweep-mean-reversion-v2-strict.md"; spec.parent.mkdir(parents=True)
    spec.write_text("# LiquiditySweepMeanReversion.BTC_LONG_SHORT_V2_STRICT_SELECTION\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "tests@example.invalid"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Tests"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True); subprocess.run(["git", "commit", "-m", "sealed spec"], cwd=root, check=True, capture_output=True)
    return root


def test_v2_phase_a_manifest_is_gap_tolerant_but_strictly_ordered(tmp_path):
    manifest = _phase_a_manifest(tmp_path)
    _, _, bars, diagnostics = _load_phase_a_bars(manifest.resolve())
    assert len(bars) == 113757 and diagnostics["partition_row_counts"]["2023-09"] == 8637
    assert diagnostics["gap_count"] == 4 and all(bars[index]["timestamp"] < bars[index + 1]["timestamp"] for index in range(len(bars) - 1))


@pytest.mark.parametrize("mutation, error", [
    (lambda payload: payload["parquet_files"].pop(), "FILE_SET_INVALID"),
    (lambda payload: payload["parquet_files"].append(dict(payload["parquet_files"][-1])), "FILE_SET_INVALID"),
    (lambda payload: payload["parquet_files"].__setitem__(0, {**payload["parquet_files"][0], "month": "2022-12"}), "FILE_SET_INVALID"),
    (lambda payload: payload["parquet_files"].reverse(), "FILE_SET_INVALID"),
    (lambda payload: payload["parquet_files"].__setitem__(0, {**payload["parquet_files"][0], "sha256": "0" * 64}), "PARQUET_INVALID"),
])
def test_v2_phase_a_manifest_rejects_missing_duplicate_extra_and_hash_invalid_partitions(tmp_path, mutation, error):
    manifest = _phase_a_manifest(tmp_path); payload = json.loads(manifest.read_text())
    mutation(payload); manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=error): _load_phase_a_bars(manifest.resolve())


def test_v2_phase_a_workflow_validates_the_true_runtime_result_without_launching_it():
    workflow = (__import__("pathlib").Path(__file__).parents[2] / ".smithers/workflows/liquidity-sweep-mean-reversion-v2-strict-phase-a.tsx").read_text(encoding="utf-8")
    assert "realStudyExecuted: z.literal(true)" in workflow and "return result.parse(JSON.parse(stdout))" in workflow
    assert "candidateExecutions: z.record(z.string(), z.literal(1))" in workflow and "Bun.spawn" in workflow


def test_v2_synthetic_contract_has_required_unexecuted_artifacts_and_immutable_collision(tmp_path):
    root = tmp_path / "repo"; spec = root / ".smithers/specs/liquidity-sweep-mean-reversion-v2-strict.md"; spec.parent.mkdir(parents=True)
    spec.write_text("\n".join(["LiquiditySweepMeanReversion.BTC_LONG_SHORT_V2_STRICT_SELECTION", "LSMR-V2-2P0R=2R LSMR-V2-2P5R=2.5R LSMR-V2-3P0R=3R", "SESSION_CONTEXT_UNAVAILABLE TRADE_EXECUTED"]), encoding="utf-8")
    result = materialize_lsmr_v2_strict_contract(repository_root=root, artifact_root=tmp_path / "runs")
    artifact = __import__("pathlib").Path(result["artifactRoot"])
    report = json.loads((artifact / "phase_a/candidates/LSMR-V2-2P0R/report.json").read_text())
    assert result["realStudyExecuted"] is False and report["status"] == "NOT_EXECUTED"
    assert {"terminal_dispositions", "annualized_trades", "long_trade_count", "short_trade_count", "monthly_results", "concentration_metrics", "bootstrap_summary", "sensitivity", "hard_gates", "overfrequency_warning"} <= set(report)
    assert json.loads((artifact / "phase_b/locked-data-manifest.json").read_text())["status"] == "NOT_OPENED"
    with pytest.raises(FileExistsError): materialize_lsmr_v2_strict_contract(repository_root=root, artifact_root=tmp_path / "runs")


def test_v2_phase_a_rejects_relative_paths_and_manifest_tampering(tmp_path):
    manifest = _phase_a_manifest(tmp_path)
    with pytest.raises(ValueError, match="absolute path"):
        _load_phase_a_bars("relative.json")
    payload = json.loads(manifest.read_text()); payload["parquet_files"][0]["sha256"] = "0" * 64; manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="PARQUET_INVALID"):
        _load_phase_a_bars(manifest.resolve())
