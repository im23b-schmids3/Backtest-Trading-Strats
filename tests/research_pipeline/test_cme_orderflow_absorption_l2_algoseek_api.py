from __future__ import annotations

import gzip
import io
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse

import pytest

from research_pipeline.cme_orderflow_absorption_l2_v1 import algoseek_api as api
from research_pipeline.cme_orderflow_absorption_l2_v1 import algoseek_adapter as adapter
from research_pipeline.cme_orderflow_absorption_l2_v1 import multi_strategy_research


class FakeResponse(io.BytesIO):
    def __init__(self, body: bytes, headers: dict[str, str] | None = None, status: int = 200) -> None:
        super().__init__(body); self.headers = headers or {}; self.status = status

    def getcode(self) -> int:
        return self.status


def _gzip_csv(*, ticker: str, trade_date: str, timestamp: str, header: bool = True) -> bytes:
    columns = ["TradeDate", "EventDateTime", "Ticker", "BaseSymbol", "SecurityID", "EventType", "Price", "Quantity", "Flags", "TypeMask"]
    row = [trade_date, timestamp, ticker, "MES" if ticker.startswith("MES") else "ES", "101", "QUOTE BID", "4000.00", "0", "T", "QUOTE BID"]
    text = ((",".join(columns) + "\n") if header else "") + ",".join(row) + "\n"
    return gzip.compress(text.encode())


def _client(opener, *, sleep=lambda _: None) -> api.AlgoseekAPIClient:
    return api.AlgoseekAPIClient(api_key="not-a-real-key", base_url="https://example.test/api/v1", opener=opener, sleep=sleep)


def _market_opener(request, timeout):
    parsed = urlparse(request.full_url); query = parse_qs(parsed.query)
    if parsed.path.endswith("/account/my/quotas"):
        return FakeResponse(b'{"quotas_usage": {}, "quotas_limit": {}}')
    parts = parsed.path.split("/")
    trade_date, ticker = parts[-2:]
    if "EventDateTime.lt" in query:
        translated_end = datetime.fromisoformat(query["EventDateTime.lt"][0]).replace(tzinfo=api.PROVIDER_FILTER_TIMEZONE)
        timestamp = (translated_end.astimezone(api.SOURCE_TIMEZONE) - timedelta(seconds=1)).strftime("%Y-%m-%d %H:%M:%S")
    else:
        timestamp = f"{trade_date} 17:00:00"
    assert query["response_format"] == ["csv_gzip"]
    return FakeResponse(_gzip_csv(ticker=ticker, trade_date=trade_date, timestamp=timestamp), {"X-Request-ID": "request-1"})


def test_missing_key_fails_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ALGOSEEK_API_KEY", raising=False)
    with pytest.raises(api.AlgoseekAuthenticationError, match="ALGOSEEK_API_KEY"):
        api.AlgoseekAPIClient()


def test_production_base_constructs_all_required_endpoints() -> None:
    client = api.AlgoseekAPIClient(api_key="not-a-real-key", opener=lambda *_args, **_kwargs: None)
    assert client.base_url == "https://api.algoseek.com/v1"
    assert client._url(api.ACCOUNT_IDENTITY_ENDPOINT) == "https://api.algoseek.com/v1/account/my"
    assert client._url(api.ACCOUNT_QUOTAS_ENDPOINT) == "https://api.algoseek.com/v1/account/my/quotas"
    assert client._url(api.ACCOUNT_ACCESS_RULES_ENDPOINT) == "https://api.algoseek.com/v1/account/my/data-access-rules"
    assert client._url(api.MY_DATASETS_ENDPOINT) == "https://api.algoseek.com/v1/meta/datasets/my"
    assert client._url(api.FUTURES_DEPTH_ENDPOINT.format(trade_date="2023-01-03", ticker="ESH3")) == "https://api.algoseek.com/v1/data/us-futures/multiple-depth/2023-01-03/ESH3"
    assert client._url(api.FUTURES_TAQ_ENDPOINT.format(trade_date="2023-01-03", ticker="MESH3")) == "https://api.algoseek.com/v1/data/us-futures/taq/2023-01-03/MESH3"


def test_api_base_environment_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api.os, "environ", api.os.environ.copy() | {"ALGOSEEK_API_BASE_URL": "https://sandbox.example/v9/"})
    client = api.AlgoseekAPIClient(api_key="not-a-real-key", opener=lambda *_args, **_kwargs: None)
    assert client.base_url == "https://sandbox.example/v9"


def test_contract_map_and_closed_calendar_are_explicit() -> None:
    assert api.contract_map_for(date(2023, 3, 12)).es == "ESH3"
    assert api.contract_map_for(date(2023, 3, 13)).mes == "MESM3"
    assert api.is_cme_closed_session(date(2023, 1, 2))
    assert api.is_cme_closed_session(date(2023, 1, 7))
    assert not api.is_cme_closed_session(date(2023, 1, 3))


def test_logical_session_uses_inclusive_1700_and_exclusive_1600() -> None:
    window = api.logical_session_window(date(2023, 1, 3))
    assert window.start.strftime("%F %T") == "2023-01-02 17:00:00"
    assert window.end.strftime("%F %T") == "2023-01-03 16:00:00"
    assert [(day.isoformat(), start.strftime("%F %T"), end.strftime("%F %T")) for day, start, end in window.provider_trade_date_windows] == [
        ("2023-01-02", "2023-01-02 17:00:00", "2023-01-03 00:00:00"),
        ("2023-01-03", "2023-01-03 00:00:00", "2023-01-03 16:00:00"),
    ]


def test_filter_translation_uses_named_timezone_frames_across_dst() -> None:
    cases = [
        (datetime(2023, 1, 2, 23), datetime(2023, 1, 3), date(2023, 1, 2), ("2023-01-03 00:00:00", "2023-01-03 01:00:00")),
        (datetime(2023, 3, 10, 23), datetime(2023, 3, 11), date(2023, 3, 10), ("2023-03-11 00:00:00", "2023-03-11 01:00:00")),
        (datetime(2023, 3, 12, 15), datetime(2023, 3, 12, 16), date(2023, 3, 12), ("2023-03-12 16:00:00", "2023-03-12 17:00:00")),
        (datetime(2023, 3, 13, 15), datetime(2023, 3, 13, 16), date(2023, 3, 13), ("2023-03-13 16:00:00", "2023-03-13 17:00:00")),
    ]
    for start, end, provider_date, expected in cases:
        aware_start = start.replace(tzinfo=api.SOURCE_TIMEZONE)
        aware_end = end.replace(tzinfo=api.SOURCE_TIMEZONE)
        assert api.api_filter_bounds_for_local_window(aware_start, aware_end, provider_date) == expected


def test_hourly_partitions_cover_session_without_omissions() -> None:
    partitions = api.hourly_session_partitions(api.logical_session_window(date(2023, 1, 3)))
    assert len(partitions) == 23
    assert partitions[0][1].strftime("%F %T") == "2023-01-02 17:00:00"
    assert partitions[-1][2].strftime("%F %T") == "2023-01-03 16:00:00"
    assert all(left[2] == right[1] for left, right in zip(partitions, partitions[1:]))


def test_partition_completeness_fails_closed_when_an_hour_is_missing(tmp_path: Path) -> None:
    session_date = date(2023, 1, 3)
    manifest_path = tmp_path / session_date.isoformat() / "algoseek-download-manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(api._new_manifest(session_date, api.contract_map_for(session_date), tmp_path, 80_000)))
    with pytest.raises(api.AlgoseekDownloadError, match="partition completeness mismatch"):
        api._validate_complete_partition_set(tmp_path, session_date)


def test_provider_trade_date_mismatch_fails_closed() -> None:
    with pytest.raises(api.AlgoseekDownloadError, match="does not own local partition"):
        api.api_filter_bounds_for_local_window(
            datetime(2023, 1, 3, 0, tzinfo=api.SOURCE_TIMEZONE),
            datetime(2023, 1, 3, 1, tzinfo=api.SOURCE_TIMEZONE),
            date(2023, 1, 2),
        )


def test_normalized_multiset_ignores_csv_formatting_and_chunk_boundaries(tmp_path: Path) -> None:
    header = "TradeDate,EventDateTime,Ticker,Price,Quantity,Orders,Depth,Flags,TypeMask\n"
    first = header + "2023-01-03,2023-01-03 15:00:00.100000000,ESH3,3900.5000,2.0,3.0,10,0.0000,1\n"
    second = header + "2023-01-03T15:00:00.100000-06:00,ESH3,3900.5,2,3,10,0,1\n"
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    left.write_text(first)
    right.write_text(second)
    assert api.normalized_row_multiset_fingerprint([left]) == api.normalized_row_multiset_fingerprint([right])


def test_tiny_probe_validates_gzip_schema_ticker_and_trade_date() -> None:
    result = api.tiny_probe(_client(_market_opener))
    assert result["ready"] and [entry["ticker"] for entry in result["results"]] == ["ESH3", "ESH3", "MESH3"]
    assert [entry["dataset_id"] for entry in result["results"]] == ["US6002", "US6011", "US6011"]
    assert result["results"][1]["request_url"].startswith("https://example.test/api/v1/data/us-futures/taq/2023-01-03/ESH3?")


def test_tiny_probe_http_error_identifies_feed_dataset_url_and_safe_body() -> None:
    def opener(request, timeout):
        if "/multiple-depth/" in request.full_url:
            return FakeResponse(_gzip_csv(ticker="ESH3", trade_date="2023-01-03", timestamp="2023-01-03 15:59:59"))
        raise HTTPError(request.full_url, 404, "Not Found", {}, io.BytesIO(b'{"detail":"unknown TAQ route"}'))
    with pytest.raises(api.AlgoseekDownloadError) as raised:
        api.tiny_probe(_client(opener))
    message = str(raised.value)
    assert "feed=es-trade-and-quote" in message and "dataset_id=US6011" in message
    assert "method=GET" in message and "/data/us-futures/taq/2023-01-03/ESH3" in message
    assert "status=404" in message and "unknown TAQ route" in message


def test_corrupt_gzip_fails_without_publishing(tmp_path: Path) -> None:
    response = api.Response(FakeResponse(b"not gzip"), {}, 200)
    with pytest.raises(api.AlgoseekDownloadError, match="valid gzip"):
        api._validate_and_write_page(response=response, destination=tmp_path / "bad.csv.gz", expected_ticker="ESH3",
            expected_trade_date=date(2023, 1, 3), start=api.logical_session_window(date(2023, 1, 3)).start,
            end=api.logical_session_window(date(2023, 1, 3)).end, expected_header=None)
    assert not list(tmp_path.glob("*.csv.gz")) and not list(tmp_path.glob("*.part"))


def test_pagination_uses_next_offset_and_only_first_response_has_header(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[tuple[str, str, dict[str, list[str]]]] = []
    def opener(request, timeout):
        parsed = urlparse(request.full_url)
        if parsed.path.endswith("/account/my/quotas"):
            return FakeResponse(b'{"quotas_usage": {}, "quotas_limit": {}}')
        query = parse_qs(parsed.query); offset = int(query["offset"][0]); requests.append((parsed.path, parsed.path.split("/")[-2], query))
        trade_date, ticker = parsed.path.split("/")[-2:]
        translated_end = datetime.fromisoformat(query["EventDateTime.lt"][0]).replace(tzinfo=api.PROVIDER_FILTER_TIMEZONE)
        timestamp = (translated_end.astimezone(api.SOURCE_TIMEZONE) - timedelta(seconds=1)).strftime("%Y-%m-%d %H:%M:%S")
        return FakeResponse(_gzip_csv(ticker=ticker, trade_date=trade_date, timestamp=timestamp, header=offset == 0),
                            {"X-Pagination-Next-Offset": "2"} if offset == 0 else {})
    monkeypatch.setattr(api, "_run_session_audit", lambda *_: {"ok": True})
    result = api.download_session(client=_client(opener), logical_session_date=date(2023, 1, 3), output_root=tmp_path, page_limit=2)
    assert result["session_complete"] and [item[2]["offset"][0] for item in requests] == ["0", "2"] * 69
    expected_windows = {
        "2023-01-02": ("2023-01-02 17:00:00", "2023-01-03 00:00:00"),
        "2023-01-03": ("2023-01-03 00:00:00", "2023-01-03 16:00:00"),
    }
    assert {path.split("/")[-3] for path, _, _ in requests} == {"multiple-depth", "taq"}
    assert {(path.split("/")[-3], path.split("/")[-1]) for path, _, _ in requests} == {
        ("multiple-depth", "ESH3"), ("taq", "ESH3"), ("taq", "MESH3"),
    }
    for _, trade_date, query in requests:
        assert "EventDateTime.gte" not in query
        assert query["EventDateTime.ge"] != [expected_windows[trade_date][0]]
        assert query["EventDateTime.lt"] != [expected_windows[trade_date][1]]
        assert datetime.fromisoformat(query["EventDateTime.ge"][0]).hour in {0, 1, 16, 17, 18, 19, 20, 21, 22, 23}
    manifest = json.loads((tmp_path / "2023-01-03" / "algoseek-download-manifest.json").read_text())
    assert len(manifest["pages"]) == 138 and {page["pagination_offset"] for page in manifest["pages"]} == {0, 2}
    assert len(manifest["partitions"]) == 69
    assert {page["feed"] for page in manifest["pages"]} == {api.DEPTH, api.ES_TAQ, api.MES_TAQ}
    assert all(path.name.endswith(".csv.gz") for path in (tmp_path / "2023-01-03").rglob("*.csv.gz"))
    assert (tmp_path / "2023-01-03" / "algoseek-complete-session.json").is_file()


def test_resume_verifies_hash_and_prevents_requests_when_complete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api, "_run_session_audit", lambda *_: {"ok": True})
    api.download_session(client=_client(_market_opener), logical_session_date=date(2023, 1, 3), output_root=tmp_path, page_limit=2)
    calls: list[str] = []
    def unexpected(request, timeout):
        calls.append(request.full_url); raise AssertionError("completed resume must not fetch")
    result = api.download_session(client=_client(unexpected), logical_session_date=date(2023, 1, 3), output_root=tmp_path, page_limit=2, resume=True)
    assert result["session_complete"] and not calls
    page = next((tmp_path / "2023-01-03").rglob("*.csv.gz")); page.write_bytes(b"changed")
    with pytest.raises(api.AlgoseekDownloadError, match="hash mismatch"):
        api.download_session(client=_client(unexpected), logical_session_date=date(2023, 1, 3), output_root=tmp_path, page_limit=2, resume=True)


def test_wrong_ticker_and_trade_date_fail_closed(tmp_path: Path) -> None:
    window = api.logical_session_window(date(2023, 1, 3))
    for ticker, trade_date, match in (("ESM3", "2023-01-03", "ticker"), ("ESH3", "2023-01-04", "TradeDate")):
        with pytest.raises(api.AlgoseekDownloadError, match=match):
            api._validate_and_write_page(response=api.Response(FakeResponse(_gzip_csv(ticker=ticker, trade_date=trade_date, timestamp="2023-01-03 15:59:00")), {}, 200),
                destination=tmp_path / f"{ticker}.csv.gz", expected_ticker="ESH3", expected_trade_date=date(2023, 1, 3),
                start=window.start, end=window.end, expected_header=None)


def test_auth_entitlement_and_retry_classification() -> None:
    def unauthorized(request, timeout):
        raise HTTPError(request.full_url, 401, "bad", {}, None)
    with pytest.raises(api.AlgoseekAuthenticationError):
        _client(unauthorized).open("/x")
    def forbidden(request, timeout):
        raise HTTPError(request.full_url, 403, "forbidden", {}, None)
    with pytest.raises(api.AlgoseekEntitlementError):
        _client(forbidden).open("/x")
    sleeps: list[float] = []; attempts = 0
    def throttled(request, timeout):
        nonlocal attempts; attempts += 1
        if attempts == 1:
            raise HTTPError(request.full_url, 429, "slow", {"Retry-After": "3"}, None)
        return FakeResponse(b"{}")
    response = _client(throttled, sleep=sleeps.append).open("/x")
    assert response.status == 200 and sleeps == [3.0] and attempts == 2
    attempts = 0; sleeps.clear()
    def unavailable(request, timeout):
        nonlocal attempts; attempts += 1
        if attempts == 1:
            raise HTTPError(request.full_url, 503, "temporary", {}, None)
        return FakeResponse(b"{}")
    assert _client(unavailable, sleep=sleeps.append).open("/x").status == 200
    assert attempts == 2 and sleeps == [1.0]


def test_preflight_parses_available_dataset_names_without_market_request() -> None:
    paths: list[str] = []
    values = {
        "/account/my": {"id": "account"}, "/account/my/quotas": {"monthly": {"remaining": 1}},
        "/account/my/data-access-rules": [
            {"dataset_id": "US6002", "dataset_name": "US Futures Multiple Depth", "start_date": "2023-01-01", "end_date": "2023-03-31", "universe_identifiers": []},
            {"dataset_id": "US6011", "dataset_name": "US Futures Trade and Quote", "start_date": "2023-01-01", "end_date": "2023-03-31", "universe_identifiers": []},
        ],
    }
    def opener(request, timeout):
        path = urlparse(request.full_url).path.removeprefix("/api/v1"); paths.append(path)
        return FakeResponse(json.dumps(values[path]).encode())
    result = api.preflight(_client(opener))
    assert result == {
        "status": "ALGOSEEK_API_PREFLIGHT_READY", "ready": True, "q1_2023_access": "accessible",
        "quota": {"monthly": {"remaining": 1}},
        "required_datasets": {
            "US6002": {"dataset_id": "US6002", "required_dataset": "US Futures Multiple Depth", "provider_dataset_name": "US Futures Multiple Depth",
                       "match": "dataset_id", "access": "accessible", "date_range": {"start_date": "2023-01-01", "end_date": "2023-03-31"}, "universe_access": "accessible"},
            "US6011": {"dataset_id": "US6011", "required_dataset": "US Futures Trade and Quote", "provider_dataset_name": "US Futures Trade and Quote",
                       "match": "dataset_id", "access": "accessible", "date_range": {"start_date": "2023-01-01", "end_date": "2023-03-31"}, "universe_access": "accessible"},
        },
    }
    assert paths == list(values)


def test_access_rules_report_q1_dates_and_ticker_universes() -> None:
    rules = [
        {"dataset_name": "US Futures Multiple Depth", "start_date": "2020-01-01", "end_date": None, "universe_identifiers": ["ESH3", "ESM3"]},
        {"dataset_name": "US Futures Trade & Quote", "start_date": "2020-01-01", "end_date": None,
         "universe_identifiers": ["ESH3", "MESH3", "ESM3", "MESM3"]},
    ]
    access = api._required_entitlement_access(api._access_rule_records(rules))
    assert access["US6002"]["access"] == "accessible"
    assert access["US6011"]["access"] == "accessible"
    assert access["US6011"]["match"] == "normalized_name"


def test_dataset_id_overrides_display_name_and_restrictions_fail_closed() -> None:
    rules = [
        {"dataset_id": "US6002", "dataset_name": "unexpected display name", "start_date": "2023-01-01", "end_date": "2023-03-31", "universe_identifiers": []},
        {"dataset_id": "US6011", "dataset_name": "US Futures Trade and Quote", "start_date": "2023-01-02", "end_date": "2023-03-30", "universe_identifiers": ["ESH3"]},
    ]
    access = api._required_entitlement_access(api._access_rule_records(rules))
    assert access["US6002"]["match"] == "dataset_id" and access["US6002"]["access"] == "accessible"
    assert access["US6011"]["match"] == "dataset_id" and access["US6011"]["access"] == "restricted"


def test_range_plan_skips_weekends_and_q1_holidays() -> None:
    plan = api.range_plan(date(2023, 1, 1), date(2023, 1, 3), Path("separate"))
    assert plan["skipped_closed_dates"] == ["2023-01-01", "2023-01-02"]
    assert [entry["logical_session_date"] for entry in plan["sessions"]] == ["2023-01-03"]


def test_main_dispatches_download_range_dry_run_without_an_api_key() -> None:
    assert multi_strategy_research.main([
        "algoseek-download-range", "--start-date", "2023-01-01", "--end-date", "2023-01-03",
        "--output-root", "/private/tmp/algoseek-api-dry-run", "--dry-run",
    ]) == 0


def test_range_stops_at_the_first_failed_session_audit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[date] = []
    def fail_first(**kwargs):
        calls.append(kwargs["logical_session_date"])
        raise api.AlgoseekDownloadError("audit failed")
    monkeypatch.setattr(api, "download_session", fail_first)
    result = api.download_range(client=object(), start_date=date(2023, 1, 3), end_date=date(2023, 1, 4), output_root=tmp_path)
    assert result["status"] == "ALGOSEEK_API_RANGE_FAILED" and calls == [date(2023, 1, 3)]


def test_csv_gzip_remains_a_streaming_adapter_input(tmp_path: Path) -> None:
    path = tmp_path / "taq.csv.gz"
    path.write_bytes(_gzip_csv(ticker="ESH3", trade_date="2023-01-03", timestamp="2023-01-03 15:59:00"))
    row = next(adapter.iter_taq(path, instrument="ES"))
    assert row.ticker == "ESH3"
