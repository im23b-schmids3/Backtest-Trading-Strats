from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from research_pipeline.cme_orderflow_absorption_l2_v1 import mac_2025_sato_overnight_ohlcv1h_download as dl


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / dl.QUOTE_PLAN_PATH


@pytest.fixture
def plan():
    return dl.load_and_validate_plan(PLAN_PATH)


def test_frozen_plan_count_bar_total_and_constants(plan):
    assert len(plan["current_rth_dates"]) == dl.EXPECTED_PLAN_REQUESTS == 54
    assert len(plan["requests"]) == 54
    assert sum(row["EXPECTED_HOURLY_BAR_COUNT"] for row in plan["requests"]) == 103
    assert all((row["DATASET"], row["SCHEMA"], row["STYPE_IN"]) == ("GLBX.MDP3", "ohlcv-1h", "raw_symbol") for row in plan["requests"])
    assert dl.request_set_sha256(plan["requests"]) == dl.FROZEN_REQUEST_SET_SHA256


def test_api_request_exactly_matches_frozen_quote_fields(plan):
    row = plan["requests"][0]
    assert dl.api_request(row) == {
        "dataset": row["DATASET"],
        "symbols": [row["RAW_ES_SYMBOL"]],
        "schema": row["SCHEMA"],
        "stype_in": row["STYPE_IN"],
        "start": row["PATCH_START_UTC"],
        "end": row["PATCH_END_UTC"],
    }
    assert set(dl.api_request(row)) == {"dataset", "symbols", "schema", "stype_in", "start", "end"}


def test_plan_mutation_fails_before_any_client_is_used(plan, tmp_path):
    altered = dict(plan)
    altered["requests"] = [dict(row) for row in plan["requests"]]
    altered["requests"][0]["RAW_ES_SYMBOL"] = "ES.FUT"
    path = tmp_path / "mutated-plan.json"
    path.write_text(json.dumps(altered))
    with pytest.raises(dl.DownloadError, match="QUOTE_PLAN_MISMATCH"):
        dl.load_and_validate_plan(path)


def _request(plan, day):
    return next(row for row in plan["requests"] if row["CURRENT_RTH_DATE"] == day)


def _valid_bar(timestamp, symbol):
    return {"ts_event": timestamp, "symbol": symbol, "open": "5000.00", "high": "5000.50",
            "low": "4999.75", "close": "5000.25", "volume": 12}


def test_one_hour_and_two_hour_requests_require_exact_timestamp_boundaries(plan):
    one = _request(plan, "2025-03-03")
    two = _request(plan, "2025-03-10")
    assert len(dl.validate_bars([_valid_bar(one["EXPECTED_HOURLY_BAR_STARTS_UTC"][0], one["RAW_ES_SYMBOL"])], one)) == 1
    bars = [_valid_bar(ts, two["RAW_ES_SYMBOL"]) for ts in two["EXPECTED_HOURLY_BAR_STARTS_UTC"]]
    assert len(dl.validate_bars(bars, two)) == 2
    with pytest.raises(dl.DownloadError, match="bar count mismatch"):
        dl.validate_bars([], one)
    with pytest.raises(dl.DownloadError, match="bar count mismatch"):
        dl.validate_bars(bars[:1], two)


@pytest.mark.parametrize("mutation, message", [
    (lambda r: r.update(ts_event=datetime(2025, 3, 2, 22, tzinfo=timezone.utc)), "timestamp"),
    (lambda r: r.update(symbol="ESM5"), "unexpected symbol"),
    (lambda r: r.update(open="5000.10"), "off the 0.25 tick grid"),
    (lambda r: r.update(high="4999.75"), "OHLC high invariant"),
    (lambda r: r.update(low="5000.25"), "OHLC low invariant"),
    (lambda r: r.update(volume=-1), "negative or non-integral volume"),
])
def test_bars_reject_bad_timestamps_symbols_ohlc_tick_and_volume(plan, mutation, message):
    request = _request(plan, "2025-03-03")
    row = _valid_bar(request["EXPECTED_HOURLY_BAR_STARTS_UTC"][0], request["RAW_ES_SYMBOL"])
    mutation(row)
    with pytest.raises(dl.DownloadError, match=message):
        dl.validate_bars([row], request)


def test_duplicate_and_subhour_precision_timestamps_are_rejected(plan):
    request = _request(plan, "2025-03-03")
    ts = request["EXPECTED_HOURLY_BAR_STARTS_UTC"][0]
    with pytest.raises(dl.DownloadError, match="bar count mismatch"):
        dl.validate_bars([_valid_bar(ts, request["RAW_ES_SYMBOL"])] * 2, request)
    fractional = _valid_bar("2025-03-02T23:00:00.000000001Z", request["RAW_ES_SYMBOL"])
    with pytest.raises(dl.DownloadError, match="unexpected hourly timestamp"):
        dl.validate_bars([fractional], request)


def test_dbn_frame_requires_metadata_and_maps_exact_raw_symbol(plan):
    request = _request(plan, "2025-03-03")

    class Metadata:
        partial = []
        not_found = []

    class FakeStore:
        dataset = "GLBX.MDP3"
        schema = "ohlcv-1h"
        symbols = [request["RAW_ES_SYMBOL"]]
        stype_in = "raw_symbol"
        start = pd.Timestamp(request["PATCH_START_UTC"])
        end = pd.Timestamp(request["PATCH_END_UTC"])
        metadata = Metadata()

        @staticmethod
        def to_df(**kwargs):
            assert kwargs == {"price_type": "fixed", "pretty_ts": True, "map_symbols": True}
            ts = pd.Timestamp(request["EXPECTED_HOURLY_BAR_STARTS_UTC"][0])
            return pd.DataFrame([{"ts_event": ts, "symbol": request["RAW_ES_SYMBOL"], "open": 5_000_000_000_000,
                                  "high": 5_000_500_000_000, "low": 4_999_750_000_000,
                                  "close": 5_000_250_000_000, "volume": 12}])

    bars = dl._frame_rows(FakeStore(), request)
    normalized = dl.validate_bars(bars, request)
    assert normalized[0]["symbol"] == request["RAW_ES_SYMBOL"]
    assert normalized[0]["open"] == "5000"


def test_resume_skips_verified_files_without_a_second_download(plan, tmp_path, monkeypatch):
    request = _request(plan, "2025-03-03")
    small_plan = {**plan, "requests": [request]}
    ts = request["EXPECTED_HOURLY_BAR_STARTS_UTC"][0]
    bars = [_valid_bar(ts, request["RAW_ES_SYMBOL"])]
    monkeypatch.setattr(dl, "validate_dbn_file", lambda path, req: dl.validate_bars(bars, req))

    class Response:
        @staticmethod
        def to_file(path, compression):
            assert compression == "zstd"
            Path(path).write_bytes(b"fake dbn payload")

    class Timeseries:
        calls = 0

        def get_range(self, **kwargs):
            self.calls += 1
            assert kwargs == dl.api_request(request)
            return Response()

    class Client:
        timeseries = Timeseries()

    client = Client()
    manifest_path = tmp_path / "manifest.json"
    data_root = tmp_path / "data"
    first = dl.execute_downloads(small_plan, client, data_root=data_root, manifest_path=manifest_path)
    assert client.timeseries.calls == 1
    assert first["requests"][0]["STATUS"] == "VERIFIED"
    assert Path(first["requests"][0]["LOCAL_FILE"]).is_file()
    original_hash = dl.sha256_file(Path(first["requests"][0]["LOCAL_FILE"]))
    second = dl.execute_downloads(small_plan, client, data_root=data_root, manifest_path=manifest_path)
    assert client.timeseries.calls == 1
    assert second["requests_skipped_already_verified_this_run"] == 1
    assert second["requests"][0]["SHA256"] == original_hash


def test_resume_redownloads_only_a_manifest_hash_mismatch(plan, tmp_path, monkeypatch):
    request = _request(plan, "2025-03-03")
    small_plan = {**plan, "requests": [request]}
    bars = [_valid_bar(request["EXPECTED_HOURLY_BAR_STARTS_UTC"][0], request["RAW_ES_SYMBOL"])]
    monkeypatch.setattr(dl, "validate_dbn_file", lambda path, req: dl.validate_bars(bars, req))

    class Response:
        @staticmethod
        def to_file(path, compression):
            Path(path).write_bytes(b"different valid response bytes")

    class Timeseries:
        calls = 0

        def get_range(self, **kwargs):
            self.calls += 1
            return Response()

    class Client:
        timeseries = Timeseries()

    client = Client()
    manifest_path = tmp_path / "manifest.json"
    data_root = tmp_path / "data"
    first = dl.execute_downloads(small_plan, client, data_root=data_root, manifest_path=manifest_path)
    raw = Path(first["requests"][0]["LOCAL_FILE"])
    raw.write_bytes(b"corrupted after checkpoint")
    second = dl.execute_downloads(small_plan, client, data_root=data_root, manifest_path=manifest_path)
    assert client.timeseries.calls == 2
    assert second["requests_downloaded_this_run"] == 1
    assert second["requests"][0]["STATUS"] == "VERIFIED"


def test_resume_promotes_valid_part_without_network(plan, tmp_path, monkeypatch):
    request = _request(plan, "2025-03-03")
    small_plan = {**plan, "requests": [request]}
    bars = [_valid_bar(request["EXPECTED_HOURLY_BAR_STARTS_UTC"][0], request["RAW_ES_SYMBOL"])]
    monkeypatch.setattr(dl, "validate_dbn_file", lambda path, req: dl.validate_bars(bars, req))
    raw, part, _ = dl._filenames(request, tmp_path / "data")
    part.parent.mkdir(parents=True)
    part.write_bytes(b"complete DBN interrupted before promotion")
    result = dl.execute_downloads(small_plan, None, data_root=tmp_path / "data", manifest_path=tmp_path / "manifest.json")
    assert not part.exists()
    assert raw.is_file()
    assert result["requests"][0]["STATUS"] == "RECOVERED_VALID_PART"
    assert result["requests_downloaded_this_run"] == 0
    assert result["requests_skipped_already_verified_this_run"] == 1


def test_missing_api_key_fails_before_instantiating_or_using_databento(plan, tmp_path):
    request = _request(plan, "2025-03-03")
    with pytest.raises(dl.DownloadError, match="DATABENTO_API_KEY is not set"):
        dl.execute_downloads({**plan, "requests": [request]}, None, data_root=tmp_path / "data", manifest_path=tmp_path / "manifest.json")


def test_cli_missing_api_key_returns_before_network_and_mentions_no_request(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
    monkeypatch.setattr(dl, "MANIFEST_PATH", tmp_path / "download" / "manifest.json")
    monkeypatch.setattr(dl, "DATA_ROOT", tmp_path / "isolated-data")
    assert dl.main(["--plan", str(PLAN_PATH)]) == 1
    assert "DATABENTO_API_KEY is not set; no download request was made" in capsys.readouterr().out
    assert not (tmp_path / "download" / "manifest.json").exists()


def test_download_code_contains_only_authorized_historical_data_method():
    source = (ROOT / "src/research_pipeline/cme_orderflow_absorption_l2_v1/mac_2025_sato_overnight_ohlcv1h_download.py").read_text()
    assert source.count("client.timeseries.get_range(**api_request(request))") == 1
    for forbidden in ("batch.submit_job", "batch.download", "metadata.get_cost", "symbology.resolve", "metadata.get_dataset_range"):
        assert forbidden not in source
