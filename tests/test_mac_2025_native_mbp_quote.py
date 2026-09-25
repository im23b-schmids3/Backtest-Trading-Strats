from __future__ import annotations

from datetime import date

import pytest

from research_pipeline.cme_orderflow_absorption_l2_v1 import mac_2025_native_mbp_quote as plan


@pytest.mark.parametrize(
    ("day", "expected"),
    [
        (date(2025, 3, 14), ("ESH5", "MESH5")),
        (date(2025, 3, 17), ("ESM5", "MESM5")),
        (date(2025, 9, 12), ("ESU5", "MESU5")),
        (date(2025, 9, 15), ("ESZ5", "MESZ5")),
        (date(2025, 9, 26), ("ESZ5", "MESZ5")),
        (date(2025, 9, 29), ("ESZ5", "MESZ5")),
        (date(2025, 10, 31), ("ESZ5", "MESZ5")),
    ],
)
def test_contract_roll_boundaries(day: date, expected: tuple[str, str]) -> None:
    assert plan.contract_for(day) == expected


def test_frozen_plan_has_expected_dates_and_native_requests() -> None:
    requests = plan.build_requests()
    assert len(requests) == 112
    assert len({item.session_date for item in requests if item.category == "TRAIN"}) == 35
    assert len({item.session_date for item in requests if item.category == "VALIDATION"}) == 19
    assert len({item.session_date for item in requests if item.category == "DEPENDENCY"}) == 2
    assert {item.session_date for item in requests if item.category == "DEPENDENCY"} == {"2025-02-28", "2025-10-06"}
    assert {item.schema for item in requests} == {"mbp-10", "mbp-1"}
    assert all("U5" not in item.symbol for item in requests if item.session_date >= "2025-09-15")
    assert all(item.session_date <= "2025-04-21" for item in requests if item.category == "TRAIN")
    assert all(item.session_date >= "2025-10-07" for item in requests if item.category == "VALIDATION")


def test_quote_aggregates_only_metadata_costs() -> None:
    class Metadata:
        calls = 0

        def get_cost(self, **kwargs: object) -> str:
            self.calls += 1
            assert kwargs["schema"] in {"mbp-10", "mbp-1"}
            assert kwargs["symbols"] in (["ESZ5"], ["MESZ5"], ["ESM5"], ["MESM5"], ["ESH5"], ["MESH5"])
            return "1.000000"

    class Client:
        metadata = Metadata()

    result = plan.quote(Client())
    assert len(result["requests"]) == 112
    assert Client.metadata.calls == 112
    assert result["grand_total_usd"] == "112.000000"


def test_zero_quote_fails_closed() -> None:
    class Metadata:
        def get_cost(self, **kwargs: object) -> str:
            return "0.000000"

    class Client:
        metadata = Metadata()

    with pytest.raises(plan.PlanError, match="invalid Databento quote"):
        plan.quote(Client())


def test_es_only_plan_reuses_dates_coverage_and_has_no_mes_or_trades() -> None:
    requests = plan.build_es_only_requests()
    full = plan.build_requests()
    assert len(requests) == 56
    assert {item.schema for item in requests} == {"mbp-10"}
    assert all(item.symbol.startswith("ES") for item in requests)
    assert all("MES" not in item.symbol and item.schema != "trades" for item in requests)
    assert [(item.category, item.session_date, item.start, item.end) for item in requests] == [
        (item.category, item.session_date, item.start, item.end)
        for item in full if item.schema == "mbp-10"
    ]
    assert plan.es_only_plan_artifact()["format_version"] != plan.plan_artifact()["format_version"]


def test_es_only_quote_calls_metadata_exactly_once_per_es_request() -> None:
    class Metadata:
        calls = 0

        def get_cost(self, **kwargs: object) -> str:
            self.calls += 1
            assert kwargs["schema"] == "mbp-10"
            assert kwargs["symbols"][0].startswith("ES")  # type: ignore[index]
            return "2.000000"

    class Client:
        metadata = Metadata()

    result = plan.quote_es_only(Client())
    assert len(result["requests"]) == 56
    assert Client.metadata.calls == 56
    assert result["grand_total_usd"] == "112.000000"
    assert result["mes_market_data_included"] is False
    assert result["execution_policy"] == plan.ES_ONLY_EXECUTION_POLICY


def test_es_only_zero_quote_fails_closed() -> None:
    class Metadata:
        def get_cost(self, **kwargs: object) -> str:
            return "0.000000"

    class Client:
        metadata = Metadata()

    with pytest.raises(plan.PlanError, match="invalid Databento quote"):
        plan.quote_es_only(Client())
