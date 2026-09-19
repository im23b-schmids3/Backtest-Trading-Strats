from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
from decimal import Decimal

import pytest
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import HTTPError
from databento.common.error import BentoError

sys.path.insert(0, str(Path(__file__).parents[1]))
import databento_replay_repair_downloader as downloader


START = 1_000_000_000
END = 2_000_000_000


def request() -> downloader.Request:
    return downloader.Request(
        "mes-mbp-1-test", "2026-09-07", "MES_BBO", "mbp-1", "MESU6",
        downloader.Window(START, END, "test"), 1,
    )


def test_mbp1_ts_recv_is_authoritative_when_ts_event_is_inside():
    record = SimpleNamespace(ts_event=START + 1, ts_recv=END + 1)
    with pytest.raises(downloader.PlanError, match="range_field=ts_recv"):
        downloader.validate_record_bounds(record, request(), record_index=0, path=Path("x"))


def test_ts_event_before_start_is_allowed_when_receive_time_is_in_range():
    record = SimpleNamespace(ts_event=START - 1, ts_recv=START)
    assert downloader.validate_record_bounds(record, request(), record_index=0, path=Path("x")) == START


def test_genuine_receive_timestamp_before_start_is_rejected():
    record = SimpleNamespace(ts_event=START + 1, ts_recv=START - 1)
    with pytest.raises(downloader.PlanError, match="ts_recv=999999999"):
        downloader.validate_record_bounds(record, request(), record_index=0, path=Path("x"))


def test_genuine_ts_event_at_or_after_end_is_rejected():
    for timestamp in (END, END + 1):
        record = SimpleNamespace(ts_event=timestamp, ts_recv=timestamp)
        with pytest.raises(downloader.PlanError, match=f"ts_recv={timestamp}"):
            downloader.validate_record_bounds(record, request(), record_index=0, path=Path("x"))


def test_valid_part_is_promoted_without_download(tmp_path, monkeypatch):
    item = request()
    partial = downloader._partial_path(tmp_path, item)
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"valid-part")
    monkeypatch.setattr(downloader, "validate_dbn", lambda path, item: {"record_count": 1})

    class NeverDownload:
        class Timeseries:
            def get_range(self, **kwargs):
                raise AssertionError("a valid .part must not be downloaded again")

        timeseries = Timeseries()

    manifest = {"requests": {}}
    progress = {"completed": {}}
    downloader._download(NeverDownload(), (item,), tmp_path, manifest, progress, {})

    destination = downloader._target_path(tmp_path, item)
    assert destination.is_file()
    assert not partial.exists()
    assert manifest["requests"][item.request_id]["verification"]["status"] == "PROMOTED_VERIFIED_PART"


def test_resume_does_not_redownload_verified_file(tmp_path, monkeypatch):
    item = request()
    monkeypatch.setattr(downloader, "validate_dbn", lambda path, item: {"record_count": 1})
    calls = []

    class FakeClient:
        class Timeseries:
            def get_range(self, **kwargs):
                calls.append(kwargs)
                Path(kwargs["path"]).write_bytes(b"downloaded")

        timeseries = Timeseries()

    manifest = {"requests": {}}
    progress = {"completed": {}}
    downloader._download(FakeClient(), (item,), tmp_path, manifest, progress, {item.request_id: 1})
    downloader._download(FakeClient(), (item,), tmp_path, manifest, progress, {item.request_id: 1})
    assert len(calls) == 1


def test_quote_retries_three_transient_failures_then_succeeds():
    item = request()
    calls = []

    class Metadata:
        def get_cost(self, **kwargs):
            calls.append(kwargs)
            if len(calls) <= 3:
                raise RequestsConnectionError("remote disconnected")
            return "0.125000"

    sleeps = []
    client = SimpleNamespace(metadata=Metadata())
    assert downloader._quote(
        client, item, sleep_fn=sleeps.append, random_fn=lambda: 0,
    ) == Decimal("0.125000")
    assert len(calls) == 4
    assert sleeps == [2, 4, 8]


def test_download_retries_three_transient_failures_then_succeeds(tmp_path, monkeypatch):
    item = request()
    monkeypatch.setattr(downloader, "validate_dbn", lambda path, item: {"record_count": 1})
    calls = []

    class FakeClient:
        class Timeseries:
            def get_range(self, **kwargs):
                calls.append(kwargs)
                if len(calls) <= 3:
                    raise RequestsConnectionError("connection reset")
                Path(kwargs["path"]).write_bytes(b"downloaded")

        timeseries = Timeseries()

    sleeps = []
    downloader._download(
        FakeClient(), (item,), tmp_path, {"requests": {}}, {"completed": {}},
        {item.request_id: Decimal("0.125000")}, sleep_fn=sleeps.append, random_fn=lambda: 0,
    )
    assert len(calls) == 4
    assert sleeps == [2, 4, 8]
    assert downloader._target_path(tmp_path, item).is_file()


def test_permanent_4xx_fails_without_retry():
    item = request()
    calls = []

    class Response:
        status_code = 400

    class Metadata:
        def get_cost(self, **kwargs):
            calls.append(kwargs)
            raise HTTPError("invalid schema", response=Response())

    with pytest.raises(HTTPError):
        downloader._quote(
            SimpleNamespace(metadata=Metadata()), item,
            sleep_fn=lambda _: pytest.fail("permanent 4xx was retried"),
        )
    assert len(calls) == 1


def test_verified_file_is_not_requoted_or_redownloaded_after_restart(tmp_path, monkeypatch):
    item = request()
    monkeypatch.setattr(downloader, "validate_dbn", lambda path, item: {"record_count": 1})
    download_calls = []
    quote_calls = []

    class Client:
        class Metadata:
            def get_cost(self, **kwargs):
                quote_calls.append(kwargs)
                return "0.125000"

        class Timeseries:
            def get_range(self, **kwargs):
                download_calls.append(kwargs)
                Path(kwargs["path"]).write_bytes(b"downloaded")

        metadata = Metadata()
        timeseries = Timeseries()

    manifest = {"requests": {}}
    progress = {"completed": {}}
    downloader._download(Client(), (item,), tmp_path, manifest, progress, {item.request_id: Decimal("0.125000")})
    downloader._quote_pending(Client(), (item,), tmp_path, manifest)
    downloader._download(Client(), (item,), tmp_path, manifest, progress, {item.request_id: Decimal("0.125000")})
    assert len(download_calls) == 1
    assert quote_calls == []


def test_bento_error_wrapped_read_timeout_retries_then_succeeds(tmp_path, monkeypatch, capsys):
    item = request()
    monkeypatch.setattr(downloader, "validate_dbn", lambda path, item: {"record_count": 1})
    calls = []

    class FakeClient:
        class Timeseries:
            def get_range(self, **kwargs):
                calls.append(kwargs)
                if len(calls) <= 3:
                    raise BentoError(
                        "Error streaming response: HTTPSConnectionPool: Read timed out."
                    )
                Path(kwargs["path"]).write_bytes(b"downloaded")

        timeseries = Timeseries()

    sleeps = []
    downloader._download(
        FakeClient(), (item,), tmp_path, {"requests": {}}, {"completed": {}},
        {item.request_id: Decimal("0.125000")}, sleep_fn=sleeps.append, random_fn=lambda: 0,
    )
    assert len(calls) == 4
    assert sleeps == [2, 4, 8]
    output = capsys.readouterr().out
    assert "exception=BentoError" in output
    assert "Read timed out" in output


def test_bento_error_wrapped_remote_disconnect_retries():
    sleeps = []
    calls = []

    def callback():
        calls.append(1)
        if len(calls) <= 3:
            raise BentoError("Error streaming response: remote disconnected")
        return "ok"

    assert downloader._retry_call(
        operation="download", label="remote-disconnect", callback=callback,
        sleep_fn=sleeps.append, random_fn=lambda: 0,
    ) == "ok"
    assert len(calls) == 4
    assert sleeps == [2, 4, 8]


def test_deterministic_bento_error_fails_immediately():
    calls = []

    def callback():
        calls.append(1)
        raise BentoError("invalid schema mbp-1")

    with pytest.raises(BentoError):
        downloader._retry_call(
            operation="download", label="bad-schema", callback=callback,
            sleep_fn=lambda _: pytest.fail("deterministic BentoError was retried"),
        )
    assert len(calls) == 1


def test_interrupted_part_preserves_previous_manifest_state(tmp_path, monkeypatch):
    item = request()
    previous = downloader.Request(
        "previous", item.date, "MES_BBO", "mbp-1", "MESU6",
        downloader.Window(START + 10, END - 10, "previous"), 1,
    )
    monkeypatch.setattr(downloader, "validate_dbn", lambda path, item: {"record_count": 1})
    previous_path = downloader._target_path(tmp_path, previous)
    previous_path.parent.mkdir(parents=True)
    previous_path.write_bytes(b"previous-verified")
    manifest = {"requests": {previous.request_id: {
        "size": previous_path.stat().st_size,
        "sha256": downloader.sha256_file(previous_path),
    }}}
    progress = {"completed": {previous.request_id: {"status": "VERIFIED"}}}

    class AlwaysTimeout:
        class Timeseries:
            def get_range(self, **kwargs):
                Path(kwargs["path"]).write_bytes(b"incomplete")
                raise BentoError("Error streaming response: Read timed out")

        timeseries = Timeseries()

    with pytest.raises(BentoError):
        downloader._download(
            AlwaysTimeout(), (item,), tmp_path, manifest, progress,
            {item.request_id: Decimal("0.125000")}, sleep_fn=lambda _: None, random_fn=lambda: 0,
        )
    assert previous_path.read_bytes() == b"previous-verified"
    assert previous.request_id in manifest["requests"]
    assert item.request_id not in manifest["requests"]
    assert downloader._partial_path(tmp_path, item).is_file()


def test_practical_candidate_selection_prefers_fewest_requests_within_budget():
    first = request()
    second = downloader.Request(
        "second", first.date, first.purpose, first.schema, first.symbol,
        downloader.Window(START + 2, END - 2, "test"), 2,
    )
    third = downloader.Request(
        "third", first.date, first.purpose, first.schema, first.symbol,
        downloader.Window(START + 3, END - 3, "test"), 3,
    )
    candidates = {86400: (first,), 3600: (second, third)}

    class Metadata:
        def get_cost(self, **kwargs):
            return "1.000000" if kwargs["start"].endswith("000000000Z") else "0.400000"

    selected_gap, selected, quotes, total = downloader.choose_practical_plan(
        SimpleNamespace(metadata=Metadata()), candidates,
        remaining_budget=Decimal("0.90"),
    )
    assert selected_gap == 3600
    assert selected == (second, third)
    assert total == Decimal("0.800000")
    assert set(quotes) == {second.request_id, third.request_id}
