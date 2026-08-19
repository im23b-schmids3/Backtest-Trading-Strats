from __future__ import annotations

import inspect
import hashlib
import json
from datetime import date, time
from decimal import Decimal
from pathlib import Path

import pytest

from research_pipeline.cme_orderflow_absorption_l2_v1 import dec2025_feb2026_quote as calendar
from research_pipeline.cme_orderflow_absorption_l2_v1 import v3_poc_dec2025_jan2026_calendar_audit as audit


def _verification() -> dict:
    sessions = calendar.build_sessions(date(2025, 12, 1), date(2026, 1, 30))
    inputs = {}
    for row in sessions:
        day, prior = row.session_date.isoformat(), row.previous_rth_date.isoformat()
        target_es = "ESZ5" if row.session_date <= date(2025, 12, 16) else "ESH6"
        target_mes = "MESZ5" if row.session_date <= date(2025, 12, 16) else "MESH6"
        prior_es = "ESZ5" if row.previous_rth_date <= date(2025, 12, 16) else "ESH6"
        end = "18:15:01" if row.session_date == date(2025, 12, 24) else "22:45:01"
        prior_end = "18:15:00" if row.previous_rth_date in audit.EARLY_CLOSE_DATES else "20:00:00"
        inputs[day] = {
            "ES_MBP10": {"raw_symbol": target_es, "start_utc": f"{day}T13:00:00Z", "end_utc": f"{day}T{end}Z", "fresh_component_cost_usd": "2"},
            "MES_MBP1": {"raw_symbol": target_mes, "start_utc": f"{day}T13:30:00Z", "end_utc": f"{day}T{end}Z", "fresh_component_cost_usd": "1"},
            "PRIOR_RTH_TRADES": {"raw_symbol": prior_es, "prior_rth_date": prior, "start_utc": f"{prior}T13:30:00Z", "end_utc": f"{prior}T{prior_end}Z", "fresh_component_cost_usd": "0.65"},
        }
    return {"target_sessions": [row.session_date.isoformat() for row in sessions],
            "session_inputs": inputs, "manifest_sha256": "a" * 64}


def test_new_york_dst_and_standard_rth_mapping() -> None:
    assert tuple(value.hour for value in audit.rth_window(date(2026, 7, 15))) == (13, 20)
    assert tuple(value.hour for value in audit.rth_window(date(2025, 12, 15))) == (14, 21)
    assert audit.timezone_state(date(2026, 7, 15)) == "DST"
    assert audit.timezone_state(date(2025, 12, 15)) == "STANDARD"


def test_chicago_and_new_york_represent_same_maintenance_instants() -> None:
    day = date(2025, 12, 15)
    ny = audit.maintenance_window(day)
    chicago = (
        audit._utc(day, time(16), audit.CHICAGO),
        audit._utc(day, time(17), audit.CHICAGO),
    )
    assert ny == chicago
    assert tuple(value.hour for value in ny) == (22, 23)


def test_summer_and_winter_maintenance_and_hard_flat() -> None:
    assert tuple(value.hour for value in audit.maintenance_window(date(2026, 8, 10))) == (21, 22)
    assert audit.effective_hard_flat(date(2026, 8, 10)).time() == time(22, 45)
    assert tuple(value.hour for value in audit.maintenance_window(date(2025, 12, 1))) == (22, 23)
    assert audit.effective_hard_flat(date(2025, 12, 1)).time() == time(22, 0)


def test_dst_transition_is_resolved_by_iana_rules() -> None:
    assert audit.rth_window(date(2026, 3, 6))[0].hour == 14
    assert audit.rth_window(date(2026, 3, 9))[0].hour == 13
    assert audit.maintenance_window(date(2026, 3, 6))[0].hour == 22
    assert audit.maintenance_window(date(2026, 3, 9))[0].hour == 21


def test_shortened_sessions_and_nov28_profile() -> None:
    nov28 = audit.rth_window(date(2025, 11, 28))
    dec24 = audit.rth_window(date(2025, 12, 24))
    assert (nov28[0].hour, nov28[0].minute, nov28[1].hour) == (14, 30, 18)
    assert dec24 == (audit._parse("2025-12-24T14:30:00Z"), audit._parse("2025-12-24T18:00:00Z"))
    assert audit.effective_hard_flat(date(2025, 12, 24)) == audit._parse("2025-12-24T18:15:00Z")


def test_calendar_audit_has_42_sessions_and_only_40_prior_tails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audit, "verify_acquisition_manifest", lambda *_args, **_kwargs: _verification())
    result = audit.build_calendar_audit(Path("unused"))
    assert result["target_session_count"] == 42
    assert result["affected_session_count"] == 40
    assert result["component_sufficiency"] == {"ES_MBP10": True, "MES_MBP1": True, "PRIOR_RTH_TRADES": False}
    assert result["supplemental_plan"]["schemas"] == {"trades": 40, "mbp-10": 0, "mbp-1": 0, "mbo": 0}
    assert all(item["start_utc"].endswith("T20:00:00Z") and item["end_utc"].endswith("T21:00:00Z")
               for item in result["supplemental_plan"]["requests"])


def test_prior_completed_rth_and_contract_roll_are_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audit, "verify_acquisition_manifest", lambda *_args, **_kwargs: _verification())
    result = audit.build_calendar_audit(Path("unused"))
    rows = {row["session_date"]: row for row in result["sessions"]}
    assert rows["2025-12-01"]["prior_rth_date"] == "2025-11-28"
    assert rows["2025-12-17"]["es_contract"] == "ESH6"
    assert rows["2025-12-17"]["prior_rth_contract"] == "ESZ5"
    assert rows["2026-01-20"]["prior_rth_date"] == "2026-01-16"


def test_nov28_remains_usable_and_dec1_needs_no_supplement(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audit, "verify_acquisition_manifest", lambda *_args, **_kwargs: _verification())
    result = audit.build_calendar_audit(Path("unused"))
    dec1 = result["sessions"][0]
    assert result["nov28"]["decision"] == "USABLE_WITH_DEGRADED_SOURCE_WARNING"
    assert result["nov28"]["correct_shortened_rth_fully_covered"] is True
    assert dec1["prior_rth_missing_ranges"] == []
    assert dec1["acquisition_sufficient"] is True


def test_no_stale_bbo_semantics_and_no_strategy_or_outcome_access(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audit, "verify_acquisition_manifest", lambda *_args, **_kwargs: _verification())
    result = audit.build_calendar_audit(Path("unused"))
    assert result["canonical_semantics"]["source_end"].endswith("actual executable BBO required in liquidation window")
    assert result["strategy_runner_invoked"] is False
    assert result["outcomes_accessed"] is False
    assert result["dbn_files_opened"] is False
    source = inspect.getsource(audit.build_calendar_audit)
    assert "HistoricalL2Runner" not in source and "trade_ledger" not in source and "pnl" not in source.lower()


def test_v3_hash_and_quality_contract_are_unchanged() -> None:
    assert audit.V3_CONTRACT_SHA256 == "a0ce94eeb78dcbf865cf4464bdf97ebc3f014a8ec5e2f559f99798534dfcbcb4"
    assert audit.v3_contract_sha256() == audit.V3_CONTRACT_SHA256


class _Metadata:
    def __init__(self) -> None:
        self.calls = []

    def get_cost(self, **kwargs):
        self.calls.append(kwargs)
        return "0.125"


class _Client:
    def __init__(self) -> None:
        self.metadata = _Metadata()


def test_quote_is_metadata_only_and_sums_without_rounding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audit, "verify_acquisition_manifest", lambda *_args, **_kwargs: _verification())
    plan = audit.build_calendar_audit(Path("unused"))["supplemental_plan"]
    client = _Client()
    quoted = audit.quote_supplement(client, plan["requests"])
    assert len(client.metadata.calls) == 40
    assert quoted["symbol_subtotals_usd"] == {"ESZ5": str(Decimal("1.500")), "ESH6": str(Decimal("3.500"))}
    assert quoted["total_usd"] == str(Decimal("5.000"))
    assert quoted["status"] == "SUPPLEMENT_QUOTE_COMPLETE_DOWNLOAD_NOT_REQUESTED"
    assert all(call["schema"] == "trades" and call["stype_in"] == "raw_symbol" for call in client.metadata.calls)
    assert not hasattr(client, "timeseries")


def _write_frozen_audit(path: Path, monkeypatch: pytest.MonkeyPatch) -> bytes:
    monkeypatch.setattr(audit, "verify_acquisition_manifest", lambda *_args, **_kwargs: _verification())
    payload = audit.build_calendar_audit(Path("unused"))
    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return content


def test_real_quote_path_reads_frozen_audit_and_writes_separate_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit_path = tmp_path / "calendar-audit.json"
    quote_path = tmp_path / "supplement-quote.json"
    original = _write_frozen_audit(audit_path, monkeypatch)
    client = _Client()
    factory_calls: list[str] = []

    def factory(key: str) -> _Client:
        factory_calls.append(key)
        return client

    result = audit.run_supplement_quote(
        audit_path=audit_path,
        quote_path=quote_path,
        environment={"DATABENTO_API_KEY": "secret-not-for-output"},
        client_factory=factory,
    )
    assert factory_calls == ["secret-not-for-output"]
    assert len(client.metadata.calls) == 40
    assert len([call for call in client.metadata.calls if call["symbols"] == ["ESZ5"]]) == 12
    assert len([call for call in client.metadata.calls if call["symbols"] == ["ESH6"]]) == 28
    assert all(call["start"].endswith("T20:00:00Z") and call["end"].endswith("T21:00:00Z")
               for call in client.metadata.calls)
    assert audit_path.read_bytes() == original
    assert result == json.loads(quote_path.read_text(encoding="utf-8"))
    assert result["calendar_audit_sha256"] == hashlib.sha256(original).hexdigest()
    assert "secret-not-for-output" not in quote_path.read_text(encoding="utf-8")


def test_missing_api_key_fails_before_client_or_api_use(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    audit_path = tmp_path / "calendar-audit.json"
    quote_path = tmp_path / "supplement-quote.json"
    _write_frozen_audit(audit_path, monkeypatch)
    factory_called = False

    def factory(_key: str):
        nonlocal factory_called
        factory_called = True
        raise AssertionError("client must not be created")

    with pytest.raises(audit.CalendarAuditError, match="DATABENTO_API_KEY"):
        audit.run_supplement_quote(
            audit_path=audit_path, quote_path=quote_path, environment={}, client_factory=factory,
        )
    assert factory_called is False
    assert not quote_path.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [("raw_symbol", "ESM6"), ("schema", "mbp-10"), ("start_utc", "2025-12-01T19:59:59Z")],
)
def test_frozen_quote_request_mismatch_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: str,
) -> None:
    audit_path = tmp_path / "calendar-audit.json"
    quote_path = tmp_path / "supplement-quote.json"
    _write_frozen_audit(audit_path, monkeypatch)
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    payload["supplemental_plan"]["requests"][0][field] = value
    audit_path.write_text(json.dumps(payload), encoding="utf-8")
    factory_called = False

    def factory(_key: str):
        nonlocal factory_called
        factory_called = True
        return _Client()

    with pytest.raises(audit.CalendarAuditError):
        audit.run_supplement_quote(
            audit_path=audit_path,
            quote_path=quote_path,
            environment={"DATABENTO_API_KEY": "secret"},
            client_factory=factory,
        )
    assert factory_called is False
    assert not quote_path.exists()


def test_quote_implementation_has_no_download_api() -> None:
    source = inspect.getsource(audit.quote_supplement) + inspect.getsource(audit.run_supplement_quote)
    assert ".timeseries." not in source
    assert ".get_range(" not in source
    assert ".batch." not in source
    assert ".metadata.get_cost(" in source


def test_report_and_json_materialization_are_strategy_free(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audit, "verify_acquisition_manifest", lambda *_args, **_kwargs: _verification())
    output, report = tmp_path / "calendar-audit.json", tmp_path / "audit.md"
    assert audit.main(["--data-root", str(tmp_path), "--output", str(output), "--report", str(report)]) == 0
    payload = output.read_text(encoding="utf-8")
    assert '"target_session_count": 42' in payload
    assert "Per-session audit" in report.read_text(encoding="utf-8")
