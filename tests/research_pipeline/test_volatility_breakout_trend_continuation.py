from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_CEILING
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from research_pipeline.cli import main
from research_pipeline.volatility_breakout_trend_continuation import runner


def _bars(direction: str = "LONG", count: int = 100, signal: int = 60) -> list[dict[str, str]]:
    """A deterministic, entirely in-memory completed-five-minute-bar fixture."""
    start = datetime(2023, 1, 1, 0, 5, tzinfo=timezone.utc)
    sign = Decimal("1") if direction == "LONG" else Decimal("-1")
    rows = []
    for index in range(count):
        close = Decimal("100") + sign * Decimal(index) * Decimal("0.1")
        rows.append({"timestamp": (start + timedelta(minutes=5 * index)).isoformat().replace("+00:00", "Z"), "open": str(close - sign * Decimal("0.05")), "high": str(close + Decimal("0.2")), "low": str(close - Decimal("0.2")), "close": str(close), "volume": "10", "daily_vwap": str(close)})
    if direction == "LONG":
        rows[signal].update(open="106.5", high="108", low="106.8", close="107.8")
        rows[signal + 1].update(open="107.9", high="108.1", low="107.7", close="108")
        rows[signal + 2].update(open="108", high="112", low="107.9", close="111")
    else:
        rows[signal].update(open="92.8", high="92.9", low="91.8", close="92.2")
        rows[signal + 1].update(open="92.1", high="92.3", low="91.9", close="92")
        rows[signal + 2].update(open="92", high="92.1", low="88", close="89")
    return rows


def _executed(rows, target=Decimal("1.5")):
    setups, events, trades = runner.evaluate_bars(rows, "VBTC-V1-1P5R", target)
    assert trades
    return setups, events, trades[0]


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / ".smithers/specs").mkdir(parents=True)
    (root / runner.SPEC_PATH).write_bytes((Path.cwd() / runner.SPEC_PATH).read_bytes())
    for command in (("init", "-q"), ("config", "user.email", "test@example.test"), ("config", "user.name", "test"), ("add", "."), ("commit", "-qm", "fixture")):
        subprocess.run(["git", *command], cwd=root, check=True, capture_output=True)
    return root


def _manifest(tmp_path: Path) -> Path:
    root = tmp_path / "bars"; root.mkdir()
    files = []
    for index, month in enumerate(runner.PHASE_A_MONTHS):
        stamp = datetime(2023, 1, 1, tzinfo=timezone.utc) if index == 0 else datetime.fromisoformat(f"{month}-01T00:00:00+00:00")
        if month == "2024-01": stamp = datetime(2024, 1, 31, 23, 55, tzinfo=timezone.utc)
        path = root / f"{month}.parquet"
        pq.write_table(pa.table({"bar_start_utc": [stamp], "open": [100.0], "high": [101.0], "low": [99.0], "close": [100.0], "volume": [1.0], "daily_vwap": [100.0]}), path)
        files.append({"kind": "bars", "month": month, "relative_path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    path = root / "manifest.json"
    path.write_text(json.dumps({"valid": True, "identity": {"phase": "PHASE_A", "symbol": "BTCUSDT", "bar_interval": "5m", "months": list(runner.PHASE_A_MONTHS)}, "parquet_files": files}), encoding="utf-8")
    return path


def test_01_sealed_specification_and_registry_are_exact(tmp_path):
    assert runner.verify_sealed_specification(_repo(tmp_path)).is_file()
    assert [(candidate_id, target) for candidate_id, target in runner.CANDIDATES] == [("VBTC-V1-1P5R", Decimal("1.5")), ("VBTC-V1-2P0R", Decimal("2.0")), ("VBTC-V1-2P5R", Decimal("2.5"))]
    assert runner.PHASE_A_MONTHS == tuple([f"2023-{month:02d}" for month in range(1, 13)] + ["2024-01"])


def test_02_long_breakout_continuation_executes_at_next_open_only():
    setups, events, trade = _executed(_bars("LONG"))
    setup = next(item for item in setups if item["setup_id"] == trade["setup_id"])
    assert setup["direction"] == trade["direction"] == "LONG"
    assert trade["entry_timestamp"] == setup["entry_bar"]
    assert trade["exit_timestamp"] > trade["entry_timestamp"]
    assert any(item["type"] == "PROPOSED_SETUP" and item["setup_id"] == setup["setup_id"] for item in events)


def test_03_short_breakout_continuation_executes():
    _, _, trade = _executed(_bars("SHORT"))
    assert trade["direction"] == "SHORT" and Decimal(trade["target"]) < Decimal(trade["entry"])


def test_04_range_is_exactly_thirty_six_prior_completed_bars():
    rows = _bars(); setups, _, _ = runner.evaluate_bars(rows, "VBTC-V1-1P5R", Decimal("1.5"))
    item = next(setup for setup in setups if setup["breakout_bar"] == rows[60]["timestamp"] and setup["direction"] == "LONG")
    assert item["history"]["range_bars"] == 36 and item["history"]["prior_only"] is True
    assert Decimal(item["range_high"]) == max(Decimal(row["high"]) for row in rows[24:60])


@pytest.mark.parametrize(("direction", "expected"), [("LONG", "TRADE_EXECUTED"), ("SHORT", "TRADE_EXECUTED")])
def test_05_ema20_ema50_alignment_accepts_the_correct_direction(direction, expected):
    setups, _, _ = runner.evaluate_bars(_bars(direction), "VBTC-V1-1P5R", Decimal("1.5"))
    assert any(item["direction"] == direction and item["terminal_disposition"] == expected for item in setups)


@pytest.mark.parametrize(("mutation", "disposition"), [
    (lambda rows: rows[60].update(close="106.7", high="107.0", low="106.6"), "BREAKOUT_THRESHOLD_REJECTED"),
    (lambda rows: rows[60].update(high="107.9", low="107.7", close="107.8"), "EXPANSION_FILTER_REJECTED"),
    (lambda rows: rows[61].update(open="106.0"), "FALSE_BREAKOUT_INVALIDATED"),
    (lambda rows: (rows[60].update(open="108.2", high="109", low="108.15", close="108.5"), rows[61].update(open="108.2")), "STOP_DISTANCE_REJECTED"),
    (lambda rows: rows[60].update(low="100"), "STOP_DISTANCE_REJECTED"),
])
def test_06_long_filter_and_preentry_terminal_dispositions(mutation, disposition):
    rows = _bars(); mutation(rows)
    setups, _, _ = runner.evaluate_bars(rows, "VBTC-V1-1P5R", Decimal("1.5"))
    assert any(item["direction"] == "LONG" and item["terminal_disposition"] == disposition for item in setups)


def test_07_ema_slope_rejection_is_terminal_without_a_trade():
    rows = _bars()
    for index, row in enumerate(rows):
        row.update(open="100", high="100.2", low="99.8", close="100")
    setups, _, trades = runner.evaluate_bars(rows, "VBTC-V1-1P5R", Decimal("1.5"))
    assert not trades and any(item["terminal_disposition"] == "TREND_FILTER_REJECTED" for item in setups)


@pytest.mark.parametrize("target", [Decimal("1.5"), Decimal("2.0"), Decimal("2.5")])
def test_08_sealed_targets_are_distinct_and_fixed(target):
    _, _, trade = _executed(_bars(), target)
    risk = Decimal(trade["entry"]) - Decimal(trade["stop"])
    assert Decimal(trade["target"]) == (Decimal(trade["entry"]) + target * risk).quantize(Decimal("0.1"), rounding=ROUND_CEILING)


def test_09_long_stop_and_same_bar_stop_first():
    rows = _bars(); rows[62].update(low="106", high="112")
    _, _, trade = _executed(rows)
    assert trade["exit_reason"] == "STOP_FIRST_AMBIGUITY" and Decimal(trade["exit"]) < Decimal(trade["stop"])


def test_10_short_stop_is_adversely_slipped():
    rows = _bars("SHORT"); rows[62].update(high="114", low="89")
    _, _, trade = _executed(rows)
    assert trade["exit_reason"] == "STOP_FIRST_AMBIGUITY" and Decimal(trade["exit"]) > Decimal(trade["stop"])


def test_11_time_stop_uses_the_twenty_fourth_completed_bar_after_entry():
    rows = _bars()
    for row in rows[62:86]: row.update(open="108", high="109", low="107", close="108")
    _, _, trade = _executed(rows)
    assert trade["exit_reason"] == "TIME_STOP"
    assert datetime.fromisoformat(trade["exit_timestamp"].replace("Z", "+00:00")) == datetime.fromisoformat(trade["entry_timestamp"].replace("Z", "+00:00")) + timedelta(minutes=5 * 24)


def test_12_utc_2355_forces_flat_after_entry():
    rows = _bars(); start = datetime(2023, 1, 1, 18, 25, tzinfo=timezone.utc)
    for index, row in enumerate(rows): row["timestamp"] = (start + timedelta(minutes=5 * index)).isoformat().replace("+00:00", "Z")
    for row in rows[62:]: row.update(open="108", high="109", low="107", close="108")
    _, _, trade = _executed(rows)
    assert trade["exit_reason"] == "SESSION_FLAT"


def test_13_one_active_position_blocks_an_overlapping_setup():
    setups, _, trades = runner.evaluate_bars(_bars(), "VBTC-V1-1P5R", Decimal("1.5"))
    assert len(trades) == 1 and any(item["terminal_disposition"] == "ACTIVE_POSITION_BLOCKED" for item in setups)


def test_14_quantity_tick_fee_and_slippage_contracts_are_bound_to_existing_btc_model():
    _, _, trade = _executed(_bars())
    assert trade["quantity_btc"] == "0.001" and Decimal(trade["fees"]) > 0 and Decimal(trade["slippage"]) > 0
    assert trade["cost_model_version"] == runner.COST_MODEL_VERSION


def test_15_deterministic_setup_structure_event_and_trade_ids():
    first = _executed(_bars()); second = _executed(_bars())
    assert [item["setup_id"] for item in first[0]] == [item["setup_id"] for item in second[0]]
    assert [item["event_id"] for item in first[1]] == [item["event_id"] for item in second[1]]
    assert first[2]["trade_id"] == second[2]["trade_id"]


def test_16_every_proposed_setup_has_one_terminal_disposition_and_reconciles():
    setups, events, trades = _executed(_bars())
    assert len({item["setup_id"] for item in setups}) == len(setups)
    assert Counter(item["terminal_disposition"] for item in setups).total() == len(setups)
    runner.validate_reconciliation(setups, events, [trades])
    metrics = runner._metrics([trades], setups)
    assert metrics["monthly_results"]["2023-01"]["executed_trades"] == 1 and metrics["overfrequency_warning"] is False


def test_17_every_event_and_trade_reference_a_valid_setup():
    setups, events, trades = _executed(_bars())
    known = {item["setup_id"] for item in setups}
    assert all(item["setup_id"] in known for item in events + [trades])
    assert all(item["terminal_disposition"] == "TRADE_EXECUTED" for item in setups if item["trade_id"])


def test_18_long_and_short_accounting_have_signed_pnl():
    long_trade = _executed(_bars("LONG"))[2]; short_trade = _executed(_bars("SHORT"))[2]
    assert Decimal(long_trade["net_pnl"]) != 0 and Decimal(short_trade["net_pnl"]) != 0


def test_19_synthetic_materialization_is_data_free_immutable_and_phase_b_alpha_unopened(tmp_path):
    repo = _repo(tmp_path); result = runner.materialize_synthetic_contract(artifact_root=tmp_path / "out", repository_root=repo)
    root = Path(result["artifactRoot"]); final = json.loads((root / "final_report.json").read_text())
    assert result["realStudyExecuted"] is False and final["phase_b"] == final["alpha"] == "NOT_OPENED"
    assert json.loads((root / "data-manifest.json").read_text())["market_data_read"] is False
    inventory = json.loads((root / "integrity-manifest.json").read_text())["files"]
    assert inventory and all(hashlib.sha256((root / item["relative_path"]).read_bytes()).hexdigest() == item["sha256"] for item in inventory)
    with pytest.raises(FileExistsError, match="COLLISION"):
        runner.materialize_synthetic_contract(artifact_root=tmp_path / "out", repository_root=repo)


def test_20_specification_hash_mismatch_fails_closed(tmp_path):
    repo = _repo(tmp_path); (repo / runner.SPEC_PATH).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="SEALED"):
        runner.verify_sealed_specification(repo)


def test_21_phase_a_interface_requires_absolute_paths_and_a_clean_committed_repository(tmp_path):
    with pytest.raises(ValueError, match="ABSOLUTE"):
        runner.run_phase_a(phase_a_bars_manifest="relative", artifact_root="relative", repository_root="relative")
    repo = _repo(tmp_path); (repo / "dirty.txt").write_text("dirty", encoding="utf-8")
    with pytest.raises(ValueError, match="CLEAN"):
        runner.run_phase_a(phase_a_bars_manifest=(tmp_path / "none.json").resolve(), artifact_root=(tmp_path / "artifacts").resolve(), repository_root=repo.resolve())


def test_22_phase_a_interface_rejects_manifest_hash_without_reading_rows(tmp_path, monkeypatch):
    repo = _repo(tmp_path); manifest = _manifest(tmp_path)
    with pytest.raises(ValueError, match="MANIFEST_HASH"):
        runner.run_phase_a(phase_a_bars_manifest=manifest.resolve(), artifact_root=(tmp_path / "artifacts").resolve(), repository_root=repo.resolve())
    assert not (tmp_path / "artifacts").exists()


def test_23_synthetic_cli_emits_json_only_and_never_executes_real_phase_a(tmp_path, capsys):
    repo = _repo(tmp_path)
    assert main(["vbtc-v1-synthetic-materialize", "--artifact-root", str((tmp_path / "out").resolve()), "--repository-root", str(repo.resolve())]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["realStudyExecuted"] is False and payload["status"] == "SYNTHETIC_MATERIALIZED"


def test_24_workflow_is_non_agent_and_invokes_only_the_deterministic_cli():
    workflow = (Path.cwd() / ".smithers/workflows/volatility-breakout-trend-continuation-v1-phase-a.tsx").read_text(encoding="utf-8")
    assert "vbtc-v1-phase-a" in workflow and "Bun.spawn" in workflow and "agent=" not in workflow
    assert "realStudyExecuted: z.literal(true)" in workflow and "NOT_OPENED" in workflow
