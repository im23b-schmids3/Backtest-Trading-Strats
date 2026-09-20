from __future__ import annotations

from datetime import date
import json
from pathlib import Path

import pytest

from research_pipeline.cme_orderflow_absorption_l2_v1 import historical_completion_acquisition as acquisition


QUOTE = Path("research_runs/L2_HISTORICAL_COMPLETION_QUOTE/quote.json")


class FakeClient:
    def __init__(self) -> None:
        self.symbology = self
        self.metadata = self
        self.cost_calls: list[dict[str, object]] = []
        payload = json.loads(QUOTE.read_text(encoding="utf-8"))
        self.costs = {
            (str(row["schema"]), str(row["symbol"]), str(row["start"]), str(row["end"])): str(row["cost_usd"])
            for row in payload["requests"]
        }

    def resolve(self, **request: object) -> dict[str, object]:
        symbols = request["symbols"]
        symbol = str(symbols[0])  # type: ignore[index]
        start = str(request["start_date"])
        end = str(request["end_date"])
        if request["stype_in"] == "continuous":
            return {"status": 0, "partial": [], "not_found": [], "result": {"ES.v.0": [
                {"d0": "2025-11-01", "d1": "2025-12-17", "s": "100"},
                {"d0": "2025-12-17", "d1": "2026-08-01", "s": "200"},
            ]}}
        if request["stype_in"] == "instrument_id":
            raw = "ESZ5" if symbol == "100" else "ESH6" if start < "2026-06-01" else "ESU6"
            return {"status": 0, "partial": [], "not_found": [], "result": {symbol: [{"d0": start, "d1": end, "s": raw}]}}
        mes = symbol.startswith("MES")
        assert mes
        return {"status": 0, "partial": [], "not_found": [], "result": {symbol: [{"d0": start, "d1": end, "s": "300"}]}}

    def get_cost(self, **request: object) -> str:
        self.cost_calls.append(request)
        return self.costs[(str(request["schema"]), str(request["symbols"][0]), str(request["start"]), str(request["end"]))]  # type: ignore[index]

    @property
    def timeseries(self):  # pragma: no cover - proving this path is not touched
        raise AssertionError("download API must not be touched during preflight")


def test_approved_quote_binds_exact_family_cardinalities_and_separate_paths() -> None:
    _, items, quote_hash = acquisition._load_approved_quote(QUOTE)
    assert len(items) == 79 and len(quote_hash) == 64
    assert sum(item.family == acquisition.JUN_JUL_ROOT_NAME for item in items) == 36
    assert sum(item.family == acquisition.DEC_JAN_ROOT_NAME for item in items) == 43
    assert all(item.relative_path.startswith(item.family + "/") for item in items)
    assert all(item.relative_path.endswith(".dbn.zst") for item in items)
    assert any(item.symbol == "ESZ5" and item.session_date == "2025-11-28" for item in items)


def test_preflight_requotes_but_never_downloads(tmp_path: Path) -> None:
    client = FakeClient()
    result = acquisition.run(client=client, output_root=tmp_path)
    assert result["status"] == "PREFLIGHT_QUOTE_COMPLETE_DOWNLOAD_NOT_REQUESTED"
    assert result["request_count"] == 79
    assert result["download_api_invoked"] is False
    assert len(client.cost_calls) == 79
    assert not list(tmp_path.rglob("*"))


def test_download_is_explicit_and_cost_guard_is_not_bypassable(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient()
    monkeypatch.setattr(acquisition, "APPROVED_TOTAL_USD", acquisition.APPROVED_TOTAL_USD + 1)
    with pytest.raises(acquisition.AcquisitionError, match="approved quote total"):
        acquisition.run(client=client)


def test_exact_utc_windows_and_schema_families_are_frozen() -> None:
    _, items, _ = acquisition._load_approved_quote(QUOTE)
    june = [item for item in items if item.family == acquisition.JUN_JUL_ROOT_NAME]
    assert all(item.start[11:] == "16:00:00Z" and item.end[11:] == "22:45:01Z" for item in june)
    assert sum(item.schema == "mbo" for item in june) == 18
    assert sum(item.schema == "mbp-1" for item in june) == 18
    dec = [item for item in items if item.component == "DEC_JAN_ES_MBP10"]
    assert len(dec) == 42 and all(item.start[11:] == "00:00:00Z" and item.end[11:] == "13:00:00Z" for item in dec)
    dependency = next(item for item in items if item.component == "NOV28_PROFILE_MBP10")
    assert (dependency.start, dependency.end) == ("2025-11-28T00:00:00Z", "2025-11-28T16:30:00Z")
    assert dependency.symbol == "ESZ5"
