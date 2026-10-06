from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from research_pipeline.cme_orderflow_absorption_l2_v1.mac_2025_sato_overnight_ohlcv1h_quote_plan import (
    DEFAULT_NATIVE_MANIFEST,
    DEFAULT_SATO_DIR,
    QuotePlanError,
    build_plan,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/quote_sato_missing_overnight_ohlcv1h.py"
ET = ZoneInfo("America/New_York")


def _plan():
    return build_plan(ROOT)


def test_dst_conversion_derives_est_and_edt_offsets_from_timezone_database():
    before = datetime.combine(date(2025, 3, 2), time(18), tzinfo=ET)
    after = datetime.combine(date(2025, 3, 9), time(18), tzinfo=ET)
    assert before.utcoffset().total_seconds() == -5 * 3600
    assert after.utcoffset().total_seconds() == -4 * 3600
    assert before.astimezone(ZoneInfo("UTC")).isoformat() == "2025-03-02T23:00:00+00:00"
    assert after.astimezone(ZoneInfo("UTC")).isoformat() == "2025-03-09T22:00:00+00:00"


def test_plan_has_exact_dates_dst_hours_and_hour_bar_starts():
    plan = _plan()
    rows = plan["requests"]
    assert len(plan["current_rth_dates"]) == 54
    assert len(plan["spring_dates"]) == 35
    assert len(plan["october_dates"]) == 19
    one_hour = [row for row in rows if row["EXPECTED_HOURLY_BAR_COUNT"] == 1]
    two_hour = [row for row in rows if row["EXPECTED_HOURLY_BAR_COUNT"] == 2]
    assert len(one_hour) == 5
    assert len(two_hour) == 49
    assert plan["total_missing_hours"] == 103
    assert all(row["CURRENT_RTH_DATE"] in {"2025-03-03", "2025-03-04", "2025-03-05", "2025-03-06", "2025-03-07"} for row in one_hour)
    assert all(row["PATCH_START_UTC"].endswith("T23:00:00Z") for row in one_hour)
    assert all(row["PATCH_START_UTC"].endswith("T22:00:00Z") for row in two_hour)
    assert all(row["PATCH_END_UTC"].endswith("T00:00:00Z") for row in rows)
    oct_rows = [row for row in rows if row["CURRENT_RTH_DATE"].startswith("2025-10")]
    assert len(oct_rows) == 19
    assert all(row["EXPECTED_HOURLY_BAR_COUNT"] == 2 for row in oct_rows)
    assert all(len(row["EXPECTED_HOURLY_BAR_STARTS_UTC"]) == row["EXPECTED_HOURLY_BAR_COUNT"] for row in rows)


def test_plan_binds_symbols_to_verified_native_source_and_sato_hashes():
    plan = _plan()
    by_date = {row["CURRENT_RTH_DATE"]: row for row in plan["requests"]}
    assert by_date["2025-03-03"]["RAW_ES_SYMBOL"] == "ESH5"
    assert by_date["2025-03-14"]["RAW_ES_SYMBOL"] == "ESH5"
    assert by_date["2025-03-17"]["RAW_ES_SYMBOL"] == "ESM5"
    assert by_date["2025-03-18"]["RAW_ES_SYMBOL"] == "ESM5"
    assert by_date["2025-04-21"]["RAW_ES_SYMBOL"] == "ESM5"
    assert by_date["2025-10-07"]["RAW_ES_SYMBOL"] == "ESZ5"
    assert plan["distinct_raw_es_symbols"] == ["ESH5", "ESM5", "ESZ5"]
    assert plan["ambiguous_symbol_sessions"] == []
    assert all(row["STATUS"] == "APPROVED_FOR_COST_QUOTE" for row in plan["requests"])
    assert all(row["SOURCE_FILE_SHA256"] for row in plan["requests"])


def test_every_patch_is_hour_aligned_ends_exactly_at_native_start_and_has_no_overlap():
    for row in _plan()["requests"]:
        start = datetime.fromisoformat(row["PATCH_START_UTC"].replace("Z", "+00:00"))
        end = datetime.fromisoformat(row["PATCH_END_UTC"].replace("Z", "+00:00"))
        assert start < end
        assert start.minute == start.second == 0
        assert end.minute == end.second == 0
        assert int((end - start).total_seconds()) % 3600 == 0
        assert row["PATCH_END_UTC"] == row["SOURCE_FILE_START_UTC"]
        assert row["PATCH_END_UTC"] == f"{row['CURRENT_RTH_DATE']}T00:00:00Z"
        assert row["MISSING_DURATION_MINUTES"] == row["EXPECTED_HOURLY_BAR_COUNT"] * 60


def test_plan_fails_closed_on_ambiguous_or_non_outright_native_symbol(tmp_path):
    manifest = json.loads((ROOT / DEFAULT_NATIVE_MANIFEST).read_text())
    request = next(value for value in manifest["requests"].values() if value["session_date"] == "2025-03-03")
    request["symbol"] = "ES.FUT"
    request["verification"]["symbols"] = ["ES.FUT"]
    bad_manifest = tmp_path / "bad-manifest.json"
    bad_manifest.write_text(json.dumps(manifest))
    with pytest.raises(QuotePlanError, match="non-outright"):
        build_plan(ROOT, native_manifest_path=bad_manifest.relative_to(ROOT) if bad_manifest.is_relative_to(ROOT) else bad_manifest)


def test_quote_script_has_no_download_or_record_retrieval_implementation():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "import databento as db" in source
    assert "QUOTE_ONLY = True" in source
    assert "THIS SCRIPT MUST NEVER DOWNLOAD DATA" in source
    assert "client.metadata.get_cost(" in source
    for forbidden in ("timeseries.get_range", "batch.submit_job", "batch.download", "symbology.resolve", "metadata.get_dataset_range"):
        assert forbidden not in source


def test_show_plan_needs_no_key_and_runs_without_network():
    env = os.environ.copy()
    env.pop("DATABENTO_API_KEY", None)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--show-plan"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "REQUESTS_IN_PLAN = 54" in result.stdout
    assert "EXPECTED_HOURLY_BARS = 103" in result.stdout
    assert "NETWORK_CALLS = 0" in result.stdout
    assert "2025-03-03 ESH5 2025-03-02T23:00:00Z" in result.stdout


def test_quote_mode_with_empty_api_key_exits_before_importing_sdk_or_network(monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location("sato_quote_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
    assert module.main([]) == 2
    assert "DATABENTO_API_KEY is not set" in capsys.readouterr().out
