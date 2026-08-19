from __future__ import annotations

import json
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from research_pipeline.cme_orderflow_absorption_l2_v1 import dec2025_feb2026_quote as quote


class FakeClient:
    def __init__(self) -> None:
        self.resolve_calls: list[dict[str, object]] = []
        self.cost_calls: list[dict[str, object]] = []
        self.symbology = self
        self.metadata = self

    def resolve(self, **request: object) -> dict[str, object]:
        self.resolve_calls.append(request)
        symbols = request["symbols"]
        symbol = symbols[0]  # type: ignore[index]
        start = str(request["start_date"])
        end = str(request["end_date"])
        stype_in = request["stype_in"]
        if stype_in == "continuous":
            return {
                "status": 0, "partial": [], "not_found": [],
                "result": {"ES.v.0": [
                    {"d0": "2025-11-01", "d1": "2025-12-10", "s": "100"},
                    {"d0": "2025-12-10", "d1": "2026-03-01", "s": "200"},
                ]},
            }
        if stype_in == "instrument_id":
            raw = "ESZ5" if int(symbol) == 100 else "ESH6"
            return {"status": 0, "partial": [], "not_found": [],
                    "result": {str(symbol): [{"d0": start, "d1": end, "s": raw}]}}
        assert stype_in == "raw_symbol"
        instrument_id = 300 if str(symbol) == "MESZ5" else 400
        return {"status": 0, "partial": [], "not_found": [],
                "result": {str(symbol): [{"d0": start, "d1": end, "s": str(instrument_id)}]}}

    def get_cost(self, **request: object) -> str:
        self.cost_calls.append(request)
        return {"mbp-10": "2.000000000001", "mbp-1": "1.000000000002", "trades": "0.500000000003"}[
            str(request["schema"])
        ]


def test_calendar_is_chronological_maps_weekends_and_excludes_closed_holidays():
    sessions = quote.build_sessions()
    dates = [item.session_date for item in sessions]
    assert dates == sorted(dates) and len(dates) == len(set(dates))
    assert date(2025, 12, 25) not in dates
    assert date(2026, 1, 1) not in dates
    assert date(2026, 1, 19) not in dates
    assert date(2026, 2, 16) not in dates
    dec1 = next(item for item in sessions if item.session_date == date(2025, 12, 1))
    jan20 = next(item for item in sessions if item.session_date == date(2026, 1, 20))
    assert dec1.previous_rth_date == date(2025, 11, 28)
    assert jan20.previous_rth_date == date(2026, 1, 16)


def test_shortened_session_and_request_windows_are_exact():
    session = next(item for item in quote.build_sessions() if item.session_date == date(2025, 12, 24))
    assert session.shortened is True
    symbols = quote.ResolvedSymbols(
        {date(2025, 12, 24): "ESZ5", date(2025, 12, 23): "ESZ5"},
        {date(2025, 12, 24): "MESZ5"},
        {date(2025, 12, 24): 1, date(2025, 12, 23): 1},
        {date(2025, 12, 24): 2},
    )
    es, mes, prior = quote._component_requests(session, symbols)
    assert (es["start"], es["end"]) == ("2025-12-24T13:00:00Z", "2025-12-24T18:15:01Z")
    assert (mes["start"], mes["end"]) == ("2025-12-24T13:30:00Z", "2025-12-24T18:15:01Z")
    assert (prior["start"], prior["end"]) == ("2025-12-23T13:30:00Z", "2025-12-23T20:00:00Z")


def test_symbology_rolls_mechanically_and_matching_mes_is_validated():
    sessions = quote.build_sessions(date(2025, 12, 9), date(2025, 12, 11))
    client = FakeClient()
    resolved = quote.resolve_symbols(client, sessions)
    assert resolved.es_by_date[date(2025, 12, 9)] == "ESZ5"
    assert resolved.mes_by_date[date(2025, 12, 9)] == "MESZ5"
    assert resolved.es_by_date[date(2025, 12, 10)] == "ESH6"
    assert resolved.mes_by_date[date(2025, 12, 10)] == "MESH6"
    assert all(call["stype_out"] in {"instrument_id", "raw_symbol"} for call in client.resolve_calls)


def test_quote_uses_cost_metadata_only_and_preserves_unrounded_cumulative_cost(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(quote, "v3_contract_sha256", lambda: quote.EXPECTED_V3_HASH)
    sessions = quote.build_sessions(date(2025, 12, 8), date(2025, 12, 9))
    client = FakeClient()
    result = quote.quote(client, sessions)
    assert len(client.cost_calls) == 6
    assert result["market_data_downloaded"] is False
    assert result["strategy_outcomes_run"] is False
    expected = Decimal("7.000000000012")
    assert Decimal(result["grand_total"]["component_totals_usd"]["TOTAL_USD"]) == expected
    source = Path(quote.__file__).read_text(encoding="utf-8")
    assert "timeseries" not in source and "get_range" not in source and "batch." not in source


def test_full_block_monthly_totals_and_budget_scenarios_are_deterministic(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(quote, "v3_contract_sha256", lambda: quote.EXPECTED_V3_HASH)
    client = FakeClient()
    result = quote.quote(client)
    assert len(result["sessions"]) == 61
    assert len(client.cost_calls) == 61 * 3
    assert {month: summary["session_count"] for month, summary in result["monthly_totals"].items()} == {
        "2025-12": 22,
        "2026-01": 20,
        "2026-02": 19,
    }
    expected = Decimal("3.500000000006") * 61
    assert Decimal(result["grand_total"]["component_totals_usd"]["TOTAL_USD"]) == expected
    scenarios = {item["name"]: item for item in result["scenarios"]}
    for name in (
        "LONGEST_PREFIX_FROM_2025_12_01_UNDER_115",
        "LONGEST_PREFIX_FROM_2026_01_02_UNDER_115",
    ):
        scenario = scenarios[name]
        assert scenario["session_count"] == 32
        assert Decimal(scenario["component_totals_usd"]["TOTAL_USD"]) <= Decimal("115")
        assert Decimal(scenario["remaining_budget_usd"]) >= 0
    assert scenarios["LONGEST_PREFIX_FROM_2025_12_01_UNDER_115"]["start_date"] == "2025-12-01"
    assert scenarios["LONGEST_PREFIX_FROM_2026_01_02_UNDER_115"]["start_date"] == "2026-01-02"


def test_longest_prefix_under_budget_is_consecutive_and_never_cherry_picks():
    rows = [
        {"target_session": f"2025-12-0{index}", "session_total_usd": value,
         "costs_usd": {"ES_MBP10_USD": value, "MES_MBP1_USD": "0", "PRIOR_RTH_TRADES_USD": "0"}}
        for index, value in enumerate(("40", "80", "1"), start=1)
    ]
    selected = quote.longest_prefix(rows, Decimal("115"))
    assert [row["target_session"] for row in selected] == ["2025-12-01"]
    assert rows[2] not in selected  # The cheap third day cannot be cherry-picked.


def test_plan_only_cli_never_initializes_databento_or_writes_quote(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    assert quote.main(["--output-root", str(tmp_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "PLAN_ONLY_QUOTE_NOT_EXECUTED"
    assert payload["quote_flag_required"] is True
    assert list(tmp_path.iterdir()) == []


def test_fail_closed_symbology_and_frozen_v3_hash(monkeypatch: pytest.MonkeyPatch):
    bad = FakeClient()
    bad.resolve = lambda **_kwargs: {"status": 1, "not_found": ["ES.v.0"], "result": {}}  # type: ignore[method-assign]
    with pytest.raises(quote.QuotePlanError, match="status"):
        quote.resolve_symbols(bad, quote.build_sessions(date(2025, 12, 8), date(2025, 12, 8)))
    monkeypatch.setattr(quote, "v3_contract_sha256", lambda: "0" * 64)
    with pytest.raises(quote.QuotePlanError, match="V3 contract hash"):
        quote.quote(FakeClient(), quote.build_sessions(date(2025, 12, 8), date(2025, 12, 8)))
    assert quote.EXPECTED_V3_HASH == "a0ce94eeb78dcbf865cf4464bdf97ebc3f014a8ec5e2f559f99798534dfcbcb4"
