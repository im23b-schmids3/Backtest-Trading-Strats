from __future__ import annotations

import json
from datetime import date
from decimal import Decimal
from pathlib import Path

from research_pipeline.cme_orderflow_absorption_l2_v1 import historical_completion_quote as quote


class FakeClient:
    def __init__(self) -> None:
        self.symbology = self
        self.metadata = self
        self.resolve_calls: list[dict[str, object]] = []
        self.cost_calls: list[dict[str, object]] = []

    def resolve(self, **request: object) -> dict[str, object]:
        self.resolve_calls.append(request)
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
        assert request["stype_out"] == "instrument_id"
        return {"status": 0, "partial": [], "not_found": [], "result": {symbol: [{"d0": start, "d1": end, "s": "300"}]}}

    def get_cost(self, **request: object) -> str:
        self.cost_calls.append(request)
        return {"mbo": "2.0", "mbp-1": "1.0", "mbp-10": "3.0", "trades": "0.5"}[str(request["schema"])]


def test_request_inventory_is_exact_and_chronological() -> None:
    requests = quote.build_requests()
    assert len(requests) == 79
    assert sum(row.group == "JUN_JUL" for row in requests) == 36
    assert sum(row.group == "DEC_JAN" for row in requests) == 42
    assert sum(row.group == "NOV28_DEPENDENCY" for row in requests) == 1
    assert len(quote._dec_jan_dates()) == 42
    assert quote.NOV28 == date(2025, 11, 28)
    assert all(row.start.endswith("T16:00:00Z") and row.end.endswith("T22:45:01Z")
               for row in requests if row.group == "JUN_JUL")
    assert all(row.start.endswith("T00:00:00Z") and row.end.endswith("T13:00:00Z")
               for row in requests if row.group == "DEC_JAN")
    dependency = next(row for row in requests if row.group == "NOV28_DEPENDENCY")
    assert (dependency.start, dependency.end) == ("2025-11-28T00:00:00Z", "2025-11-28T16:30:00Z")


def test_quote_uses_metadata_only_and_reconciles_totals() -> None:
    client = FakeClient()
    payload = quote.quote(client)
    assert len(client.cost_calls) == 79
    assert payload["market_data_downloaded"] is False
    assert payload["strategy_outcomes_run"] is False
    assert payload["june_july_es_mbo_total_usd"] == "36.0"
    assert payload["june_july_mes_mbp1_total_usd"] == "18.0"
    assert payload["dec_jan_mbp10_total_usd"] == "126.0"
    assert payload["nov28_profile_dependency_usd"] == "3.0"
    assert payload["grand_total_usd"] == "183.0"
    source = Path(quote.__file__).read_text(encoding="utf-8")
    assert "timeseries.get_range(" not in source and "DBNStore" not in source


def test_quote_records_rolls_and_per_request_windows() -> None:
    payload = quote.quote(FakeClient())
    rows = payload["requests"]
    assert any(row["symbol"] == "ESZ5" for row in rows)
    assert any(row["symbol"] == "ESH6" for row in rows)
    assert any(row["symbol"] == "ESU6" for row in rows)
    assert all(row["schema"] in {"mbo", "mbp-1", "mbp-10"} for row in rows)
    assert all(row["start"].endswith("Z") and row["end"].endswith("Z") for row in rows)


def test_plan_only_does_not_require_api_key_or_write_output(tmp_path: Path, capsys) -> None:
    output = tmp_path / "quote.json"
    assert quote.main(["--output", str(output)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "PLAN_ONLY_QUOTE_NOT_EXECUTED"
    assert payload["request_count"] == 79
    assert not output.exists()
