from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from research_pipeline.cli import main
from research_pipeline.liquidity_sweep_mean_reversion.artifacts import validate_artifact_tree
from research_pipeline.liquidity_sweep_mean_reversion.models import LSMRConfig, TERMINAL_DISPOSITIONS, preregistered_candidates
from research_pipeline.liquidity_sweep_mean_reversion.runner import PHASE_A_BARS, materialize_lsmr_v1_contract, run_lsmr_v1_phase_a
from research_pipeline.imbalance_vwap_ride.artifacts import sha256_file
from research_pipeline.liquidity_sweep_mean_reversion.strategy import detect_setups, simulate_trade, terminal_disposition, validate_setup_audit


def _bars(count=64, start=None):
    start = start or datetime(2023, 1, 1, tzinfo=timezone.utc)
    return [{"timestamp": start + timedelta(minutes=5 * i), "open": "100", "high": "101", "low": "99", "close": "100", "volume": "10", "daily_vwap": "100"} for i in range(count)]


def _long_setup(index=24, extreme="99.1"):
    return {"setup_id": "long-setup", "direction": "LONG", "sweep_index": index, "reference_level": "99", "sweep_extreme": extreme}


def _short_setup(index=24, extreme="102"):
    return {"setup_id": "short-setup", "direction": "SHORT", "sweep_index": index, "reference_level": "101", "sweep_extreme": extreme}


def _long_reclaim(rows, index=24):
    rows[index].update(open="98", high="101", low="99", close="100", volume="20", daily_vwap="100")


def test_01_registry_has_only_the_three_sealed_candidates():
    assert [(c.candidate_id, c.target_r_multiple) for c in preregistered_candidates()] == [("LSMR-V1-1P5R", Decimal("1.5")), ("LSMR-V1-2P0R", Decimal("2.0")), ("LSMR-V1-2P5R", Decimal("2.5"))]


def test_02_config_rejects_any_unsealed_parameter_change():
    with pytest.raises(ValueError, match="sealed LSMR invariant"):
        LSMRConfig(price_tick=Decimal("0.01"))


def test_03_config_rejects_candidate_target_mismatch():
    with pytest.raises(ValueError, match="candidate parameters"):
        LSMRConfig(candidate_id="LSMR-V1-1P5R", target_r_multiple=Decimal("2.0"))


def test_04_detects_long_sweep_and_source_structure():
    rows = _bars(); rows[24].update(low="98.8")
    setups, events = detect_setups(rows, LSMRConfig())
    assert setups[0]["direction"] == "LONG" and setups[0]["source_structure"]["reference_start_index"] == 12 and events[0]["setup_id"] == setups[0]["setup_id"]


def test_05_detects_short_sweep():
    rows = _bars(); rows[24].update(high="101.2")
    assert any(s["direction"] == "SHORT" for s in detect_setups(rows, LSMRConfig())[0])


def test_06_penetration_uses_two_tick_floor():
    rows = _bars(); rows[24].update(low="98.81")
    assert not any(s["direction"] == "LONG" and s["sweep_index"] == 24 for s in detect_setups(rows, LSMRConfig())[0])


def test_07_reference_range_excludes_current_bar():
    rows = _bars(); rows[24].update(low="98.8")
    setup = next(s for s in detect_setups(rows, LSMRConfig())[0] if s["direction"] == "LONG")
    assert setup["reference_level"] == "99"


def test_08_reclaim_window_includes_the_sweep_bar():
    rows = _bars(); _long_reclaim(rows)
    assert terminal_disposition(_long_setup(), rows, LSMRConfig())[0] == "EXECUTED"


def test_09_reclaim_window_expires_after_three_completed_bars():
    rows = _bars(); _long_reclaim(rows, 27)
    assert terminal_disposition(_long_setup(), rows, LSMRConfig())[0] == "RECLAIM_WINDOW_EXPIRED"


def test_10_reclaim_requires_reversal_direction_close():
    rows = _bars(); rows[24].update(open="100", high="101", low="98", close="99.5", volume="20")
    assert terminal_disposition(_long_setup(), rows, LSMRConfig())[0] == "RECLAIM_WINDOW_EXPIRED"


def test_11_reclaim_requires_half_range_body():
    rows = _bars(); rows[24].update(open="99.5", high="101", low="98", close="100", volume="20")
    assert terminal_disposition(_long_setup(), rows, LSMRConfig())[0] == "RECLAIM_WINDOW_EXPIRED"


def test_12_volume_confirmation_uses_prior_ten_completed_bars():
    rows = _bars(); _long_reclaim(rows); rows[14:24] = [{**row, "volume": "10"} for row in rows[14:24]]; rows[24]["volume"] = "11.9"
    assert terminal_disposition(_long_setup(), rows, LSMRConfig())[0] == "RECLAIM_WINDOW_EXPIRED"


def test_13_even_volume_median_is_the_average_of_middle_values():
    rows = _bars(); _long_reclaim(rows)
    for i, value in enumerate(["1", "1", "1", "1", "1", "19", "19", "19", "19", "19"], 14): rows[i]["volume"] = value
    rows[24]["volume"] = "12"
    assert terminal_disposition(_long_setup(), rows, LSMRConfig())[0] == "EXECUTED"


def test_14_regime_filter_rejects_excessive_vwap_slope():
    rows = _bars(); _long_reclaim(rows); rows[0]["daily_vwap"] = "99"
    assert terminal_disposition(_long_setup(), rows, LSMRConfig())[0] == "REGIME_REJECTED"


def test_15_entry_is_next_bar_open_after_reclaim():
    rows = _bars(); _long_reclaim(rows); rows[25]["open"] = "99.9"
    _, trade = terminal_disposition(_long_setup(), rows, LSMRConfig())
    assert trade and trade["entry_price"] == "99.9"


def test_16_stop_distance_minimum_is_enforced():
    rows = _bars(); _long_reclaim(rows); rows[25]["open"] = "98.9"
    assert terminal_disposition(_long_setup(extreme="99"), rows, LSMRConfig())[0] == "STOP_DISTANCE_REJECTED"


def test_17_stop_distance_maximum_is_enforced():
    rows = _bars(); _long_reclaim(rows); rows[25]["open"] = "100"
    assert terminal_disposition(_long_setup(extreme="98.4"), rows, LSMRConfig())[0] == "STOP_DISTANCE_REJECTED"


def test_18_stop_uses_extreme_across_reclaim_window():
    rows = _bars(); rows[24].update(open="98", close="98.5", low="98", high="99", volume="20"); _long_reclaim(rows, 25); rows[25]["low"] = "97.5"; rows[26]["open"] = "98.5"
    _, trade = terminal_disposition(_long_setup(), rows, LSMRConfig())
    assert trade and trade["initial_stop_price"] == "97.3"


def test_19_missing_next_bar_is_not_executable_entry():
    rows = _bars(25); _long_reclaim(rows)
    assert terminal_disposition(_long_setup(), rows, LSMRConfig())[0] == "NO_EXECUTABLE_ENTRY"


def test_20_utc_day_boundary_is_session_ended():
    rows = _bars(start=datetime(2023, 1, 1, 21, 55, tzinfo=timezone.utc)); _long_reclaim(rows)
    rows[25]["timestamp"] = datetime(2023, 1, 2, tzinfo=timezone.utc)
    assert terminal_disposition(_long_setup(), rows, LSMRConfig())[0] == "SESSION_ENDED"


def test_21_force_flat_reclaim_is_session_ended():
    rows = _bars(start=datetime(2023, 1, 1, 21, 55, tzinfo=timezone.utc)); _long_reclaim(rows)
    assert rows[24]["timestamp"].hour == 23 and rows[24]["timestamp"].minute == 55
    assert terminal_disposition(_long_setup(), rows, LSMRConfig())[0] == "SESSION_ENDED"


def test_22_target_is_r_multiple_of_initial_risk():
    rows = _bars(); _long_reclaim(rows)
    _, trade = terminal_disposition(_long_setup(), rows, LSMRConfig())
    assert trade and trade["target_price"] == "101.80"


def test_23_same_bar_ambiguity_stops_first():
    rows = _bars(); rows[25].update(open="100", low="97", high="104")
    status, trade = simulate_trade(setup=_long_setup(), reclaim_index=24, bars=rows, config=LSMRConfig())
    assert status == "EXECUTED" and trade["exit_reason"] == "STOP_FIRST_AMBIGUITY"


def test_24_target_exit_is_recorded_when_stop_is_not_hit():
    rows = _bars(); rows[25].update(open="100", low="99", high="104")
    _, trade = simulate_trade(setup=_long_setup(), reclaim_index=24, bars=rows, config=LSMRConfig())
    assert trade and trade["exit_reason"] == "TARGET"


def test_25_force_flat_exit_is_recorded():
    rows = _bars(start=datetime(2023, 1, 1, 21, 55, tzinfo=timezone.utc)); rows[24].update(open="100", low="99", high="101")
    _, trade = simulate_trade(setup=_long_setup(), reclaim_index=23, bars=rows, config=LSMRConfig())
    assert trade and trade["exit_reason"] == "UTC_FORCE_FLAT"


def test_26_audit_requires_exactly_one_valid_terminal_outcome():
    setup = _long_setup()
    with pytest.raises(AssertionError, match="one terminal"):
        validate_setup_audit([setup], [], [], [])


def test_27_invalid_sealed_specification_fails_closed(tmp_path):
    root = tmp_path / "repo"; (root / ".smithers/specs").mkdir(parents=True); (root / ".smithers/specs/liquidity-sweep-mean-reversion-v1.md").write_text("not sealed")
    with pytest.raises(ValueError, match="MISSING_SEALED_LSMR_SPECIFICATION"):
        materialize_lsmr_v1_contract(repository_root=root, artifact_root=tmp_path / "runs")


def test_28_synthetic_materialization_is_immutable_and_cli_safe(tmp_path, capsys):
    root = tmp_path / "repo"; spec = root / ".smithers/specs/liquidity-sweep-mean-reversion-v1.md"; spec.parent.mkdir(parents=True)
    spec.write_text("LiquiditySweepMeanReversion.BTC_LONG_SHORT_V1_SPECIFICATION\nsealed")
    result = materialize_lsmr_v1_contract(repository_root=root, artifact_root=tmp_path / "runs")
    assert result["status"] == "PHASE_A_NO_ROBUST_CANDIDATE" and result["phaseBManifest"] is None and validate_artifact_tree(result["artifactRoot"])["valid"]
    with pytest.raises(FileExistsError): materialize_lsmr_v1_contract(repository_root=root, artifact_root=tmp_path / "runs")
    assert main(["lsmr-v1-materialize", "--repository-root", str(root), "--artifact-root", str(tmp_path / "cli-runs")]) == 0
    assert '"studyExecuted": false' in capsys.readouterr().out


def _sealed_spec(root):
    path = root / ".smithers/specs/liquidity-sweep-mean-reversion-v1.md"; path.parent.mkdir(parents=True)
    path.write_text("\n".join(["LiquiditySweepMeanReversion.BTC_LONG_SHORT_V1_SPECIFICATION", "Phase A: 2023-01-01T00:00:00Z", "Phase B: 2024-02-01T00:00:00Z", "Same-bar ambiguity Stop first", "LSMR-V1-1P5R LSMR-V1-2P0R LSMR-V1-2P5R"]), encoding="utf-8")


def _pinned_synthetic_phase_a_manifest(root):
    import json
    import pyarrow as pa
    import pyarrow.parquet as pq
    manifest_path = root / PHASE_A_BARS; manifest_path.parent.mkdir(parents=True)
    files=[]
    months=[f"2023-{m:02d}" for m in range(1, 13)]+["2024-01"]
    for index, month in enumerate(months):
        target=manifest_path.parent / "bars" / f"{month}.parquet"; target.parent.mkdir(exist_ok=True)
        stamp=datetime(2023 + (index//12), (index%12)+1, 1, tzinfo=timezone.utc)
        if month == "2024-01": stamp=datetime(2024,1,31,23,55,tzinfo=timezone.utc)
        pq.write_table(pa.Table.from_pylist([{"bar_start_utc":stamp,"open":100,"high":101,"low":99,"close":100,"volume":10,"daily_vwap":100}]), target)
        files.append({"kind":"bars","month":month,"relative_path":target.relative_to(manifest_path.parent).as_posix(),"sha256":sha256_file(target)})
    manifest={"valid":True,"identity":{"phase":"PHASE_A","symbol":"BTCUSDT","bar_interval":"5m","months":months},"parquet_files":files}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _clean_git_repo(root):
    import subprocess
    subprocess.run(["git","init"],cwd=root,check=True,capture_output=True)
    subprocess.run(["git","config","user.email","synthetic@example.test"],cwd=root,check=True)
    subprocess.run(["git","config","user.name","Synthetic"],cwd=root,check=True)
    subprocess.run(["git","add","."],cwd=root,check=True); subprocess.run(["git","commit","-m","sealed fixture"],cwd=root,check=True,capture_output=True)


def test_29_real_cli_contract_executes_exactly_three_only_on_synthetic_pinned_fixture(tmp_path):
    root=tmp_path/"repo"; root.mkdir(); _sealed_spec(root); _pinned_synthetic_phase_a_manifest(root); _clean_git_repo(root)
    result=run_lsmr_v1_phase_a(repository_root=root,artifact_root=tmp_path/"runs")
    assert result["status"] == "PHASE_A_NO_ROBUST_CANDIDATE" and result["phaseBStatus"] == "NOT_OPENED" and result["alphaStatus"] == "NOT_EXECUTED"
    assert validate_artifact_tree(result["artifactRoot"])["valid"]
    output_root=__import__("pathlib").Path(result["artifactRoot"])
    assert {p.name for p in (output_root/"phase_a/candidates").iterdir()} == {"LSMR-V1-1P5R","LSMR-V1-2P0R","LSMR-V1-2P5R"}
    assert all(__import__("json").loads((p/"configuration.json").read_text())["execution_count"] == 1 for p in (output_root/"phase_a/candidates").iterdir())


def test_30_real_cli_contract_fails_closed_on_dirty_git_before_market_data_read(tmp_path):
    root=tmp_path/"repo"; root.mkdir(); _sealed_spec(root); _pinned_synthetic_phase_a_manifest(root); _clean_git_repo(root); (root/"dirty.txt").write_text("x")
    with pytest.raises(ValueError, match="LSMR_PHASE_A_REQUIRES_CLEAN_GIT"):
        run_lsmr_v1_phase_a(repository_root=root,artifact_root=tmp_path/"runs")
