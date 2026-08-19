from __future__ import annotations

import csv
import json
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from research_pipeline.cme_orderflow_absorption_l2_v1 import dec2025_feb2026_quote as quote
from research_pipeline.cme_orderflow_absorption_l2_v1 import dec2025_jan2026_acquisition as acquire


def _write_quote_fixture(root: Path) -> Path:
    sessions = quote.build_sessions(date(2025, 12, 1), date(2026, 1, 30))
    rows = []
    running = Decimal(0)
    for session in sessions:
        day, prior = session.session_date, session.previous_rth_date
        es, mes, es_id, mes_id = (
            ("ESZ5", "MESZ5", 294973, 42004164) if day <= date(2025, 12, 16)
            else ("ESH6", "MESH6", 42140878, 42003800)
        )
        prior_es = "ESZ5" if prior <= date(2025, 12, 16) else "ESH6"
        costs = {"ES_MBP10_USD": Decimal(0), "MES_MBP1_USD": Decimal(0),
                 "PRIOR_RTH_TRADES_USD": Decimal(0)}
        if day == date(2025, 12, 31):
            costs = {"ES_MBP10_USD": Decimal("28.604654915630"),
                     "MES_MBP1_USD": Decimal("16.210373818874"),
                     "PRIOR_RTH_TRADES_USD": Decimal("7.171603560450")}
        if day == date(2026, 1, 30):
            costs = {"ES_MBP10_USD": Decimal("31.569380082188"),
                     "MES_MBP1_USD": Decimal("18.862187102439"),
                     "PRIOR_RTH_TRADES_USD": Decimal("7.631710052489")}
        total = sum(costs.values(), Decimal(0))
        running += total
        rows.append({
            "target_session": day.isoformat(),
            "es_raw_symbol": es,
            "es_instrument_id": es_id,
            "mes_raw_symbol": mes,
            "mes_instrument_id": mes_id,
            "prior_rth_date": prior.isoformat(),
            "prior_rth_es_raw_symbol": prior_es,
            "effective_session_end_utc": f"{day.isoformat()}T{session.effective_end.isoformat()}Z",
            "shortened_session": session.shortened,
            "costs_usd": {key: str(value) for key, value in costs.items()},
            "session_total_usd": str(total),
            "session_total_usd_display": str(total.quantize(Decimal("0.01"))),
            "cumulative_total_usd": str(running),
            "cumulative_total_usd_display": str(running.quantize(Decimal("0.01"))),
        })
    payload = {
        "status": "QUOTE_COMPLETE_NO_DATA_ACQUIRED",
        "strategy_id": acquire.STRATEGY_ID,
        "v3_contract_sha256": acquire.EXPECTED_V3_HASH,
        "market_data_downloaded": False,
        "strategy_outcomes_run": False,
        "sessions": rows,
        "scenarios": [{
            "name": "DEC_PLUS_JAN",
            "session_count": 42,
            "start_date": "2025-12-01",
            "end_date": "2026-01-30",
            "component_totals_usd": {"TOTAL_USD": str(acquire.APPROVED_TOTAL_USD)},
        }],
    }
    path = root / "quote.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    fields = (
        "target_session", "es_raw_symbol", "mes_raw_symbol", "prior_rth_date",
        "prior_rth_es_raw_symbol", "effective_session_end_utc", "shortened_session",
        "ES_MBP10_USD", "MES_MBP1_USD", "PRIOR_RTH_TRADES_USD",
        "session_total_usd", "cumulative_total_usd",
    )
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {**row, **row["costs_usd"]}
            writer.writerow({field: flat[field] for field in fields})
    return path


class FakeClient:
    def __init__(self, costs: list[str] | None = None, *, fail_first_download: bool = False) -> None:
        self.symbology = self
        self.metadata = self
        self.timeseries = self
        self.costs = list(costs or [])
        self.resolve_calls: list[dict[str, object]] = []
        self.cost_calls: list[dict[str, object]] = []
        self.download_calls: list[dict[str, object]] = []
        self.fail_first_download = fail_first_download

    def resolve(self, **request: object) -> dict[str, object]:
        self.resolve_calls.append(request)
        symbol = request["symbols"][0]  # type: ignore[index]
        start, end = str(request["start_date"]), str(request["end_date"])
        if request["stype_in"] == "continuous":
            return {"status": 0, "partial": [], "not_found": [], "result": {"ES.v.0": [
                {"d0": "2025-11-01", "d1": "2025-12-17", "s": "294973"},
                {"d0": "2025-12-17", "d1": "2026-02-01", "s": "42140878"},
            ]}}
        if request["stype_in"] == "instrument_id":
            raw = "ESZ5" if int(symbol) == 294973 else "ESH6"
            return {"status": 0, "partial": [], "not_found": [],
                    "result": {str(symbol): [{"d0": start, "d1": end, "s": raw}]}}
        assert request["stype_in"] == "raw_symbol"
        instrument_id = 42004164 if symbol == "MESZ5" else 42003800
        return {"status": 0, "partial": [], "not_found": [],
                "result": {str(symbol): [{"d0": start, "d1": end, "s": str(instrument_id)}]}}

    def get_cost(self, **request: object) -> str:
        self.cost_calls.append(request)
        return self.costs.pop(0)

    def get_range(self, **request: object) -> None:
        self.download_calls.append(request)
        if self.fail_first_download and len(self.download_calls) == 1:
            Path(str(request["path"])).write_bytes(b"partial")
            raise OSError("synthetic transient provider failure")
        Path(str(request["path"])).write_bytes(b"synthetic DBN payload")


def _client_for(frozen: acquire.FrozenQuote, items: tuple[acquire.Component, ...], **kwargs: object) -> FakeClient:
    return FakeClient([str(item.approved_cost_usd) for item in items], **kwargs)


def test_frozen_quote_has_exact_42_sessions_calendar_roll_and_prior_mapping(tmp_path: Path):
    frozen = acquire.load_frozen_quote(_write_quote_fixture(tmp_path))
    assert len(frozen.rows) == 42
    assert frozen.rows[0].target_session == date(2025, 12, 1)
    assert frozen.rows[-1].target_session == date(2026, 1, 30)
    assert frozen.rows[0].prior_rth_date == date(2025, 11, 28)
    assert all(row.target_session.weekday() < 5 and row.target_session.month != 2 for row in frozen.rows)
    dec16 = next(row for row in frozen.rows if row.target_session == date(2025, 12, 16))
    dec17 = next(row for row in frozen.rows if row.target_session == date(2025, 12, 17))
    assert (dec16.es_raw_symbol, dec16.mes_raw_symbol) == ("ESZ5", "MESZ5")
    assert (dec17.es_raw_symbol, dec17.mes_raw_symbol) == ("ESH6", "MESH6")
    assert dec17.prior_rth_es_raw_symbol == "ESZ5"


def test_components_are_exact_126_no_mbo_no_february_and_calendar_aware(tmp_path: Path):
    frozen = acquire.load_frozen_quote(_write_quote_fixture(tmp_path))
    items = acquire.components(frozen)
    assert len(items) == 126
    assert sum(item.schema == "mbp-10" for item in items) == 42
    assert sum(item.schema == "mbp-1" for item in items) == 42
    assert sum(item.schema == "trades" for item in items) == 42
    assert all(item.schema != "mbo" and item.target_session.month != 2 for item in items)
    dec24 = [item for item in items if item.target_session == date(2025, 12, 24)]
    assert {item.end for item in dec24 if item.schema != "trades"} == {"2025-12-24T18:15:01Z"}
    first_prior = next(item for item in items if item.target_session == date(2025, 12, 1)
                       and item.purpose == "PRIOR_RTH_TRADES")
    assert first_prior.start == "2025-11-28T13:30:00Z"
    assert first_prior.end == "2025-11-28T18:15:00Z"


def test_quote_only_preflight_calls_metadata_but_zero_download_apis(tmp_path: Path):
    path = _write_quote_fixture(tmp_path)
    frozen = acquire.load_frozen_quote(path)
    items = acquire.components(frozen)
    client = _client_for(frozen, items)
    output = tmp_path / "not-created"
    result = acquire.run(client=client, quote_json=path, output_root=output, download=False)
    assert result["target_session_count"] == 42 and result["component_count"] == 126
    assert result["fresh_quote"]["total_usd"] == str(acquire.APPROVED_TOTAL_USD)
    assert len(client.cost_calls) == 126 and client.download_calls == []
    assert result["strategy_replay_executed"] is False and result["outcomes_inspected"] is False
    assert not output.exists()


@pytest.mark.parametrize("total, error", [
    (Decimal("115.000000000001"), "hard cap"),
    (Decimal("111.049909532071"), "deviation"),
])
def test_fresh_quote_hard_cap_and_one_dollar_deviation_fail_closed(total: Decimal, error: str):
    with pytest.raises(acquire.AcquisitionError, match=error):
        acquire._validate_fresh_total(total)


def test_download_uses_part_retries_atomically_and_resumes_by_hash(tmp_path: Path):
    path = _write_quote_fixture(tmp_path)
    frozen = acquire.load_frozen_quote(path)
    items = acquire.components(frozen)
    fresh = {
        "cost_by_relative_path_usd": {item.relative_path: str(item.approved_cost_usd) for item in items},
        "monthly_totals_usd": {}, "total_usd": str(acquire.APPROVED_TOTAL_USD),
        "approved_total_usd": str(acquire.APPROVED_TOTAL_USD), "hard_cap_usd": "115.00", "deviation_usd": "0",
    }
    manifest = acquire._initial_manifest(frozen, items, fresh)
    root = tmp_path / "data"
    acquire._write_manifest(root, manifest)
    client = FakeClient(fail_first_download=True)
    item = items[0]
    assert acquire._download_one(
        client=client, root=root, item=item, manifest=manifest,
        fresh_cost=str(item.approved_cost_usd),
    ) == "DOWNLOADED_VERIFIED"
    destination = root / item.relative_path
    assert destination.is_file() and not destination.with_suffix(destination.suffix + ".part").exists()
    assert len(client.download_calls) == 2
    assert acquire._download_one(
        client=client, root=root, item=item, manifest=manifest,
        fresh_cost=str(item.approved_cost_usd),
    ) == "SKIPPED_VERIFIED"
    assert len(client.download_calls) == 2
    destination.write_bytes(b"tampered")
    with pytest.raises(acquire.AcquisitionError, match="hash/size mismatch"):
        acquire._verified_existing(root, item, manifest)


def test_stale_part_and_unknown_files_fail_closed(tmp_path: Path):
    path = _write_quote_fixture(tmp_path)
    frozen = acquire.load_frozen_quote(path)
    items = acquire.components(frozen)
    root = tmp_path / "tree"
    root.mkdir()
    (root / "unknown.dbn").write_bytes(b"x")
    with pytest.raises(acquire.AcquisitionError, match="unknown existing file"):
        acquire._assert_output_tree(root, items, None)
    (root / "unknown.dbn").unlink()
    part = root / items[0].relative_path
    part.parent.mkdir(parents=True)
    part.with_suffix(part.suffix + ".part").write_bytes(b"partial")
    with pytest.raises(acquire.AcquisitionError, match="stale partial"):
        acquire._assert_output_tree(root, items, None)


def test_v3_identity_and_source_have_no_strategy_or_outcome_execution():
    assert acquire.EXPECTED_V3_HASH == "a0ce94eeb78dcbf865cf4464bdf97ebc3f014a8ec5e2f559f99798534dfcbcb4"
    assert quote.v3_contract_sha256() == acquire.EXPECTED_V3_HASH
    source = Path(acquire.__file__).read_text(encoding="utf-8")
    assert "historical_runner" not in source
    assert "strategy_runner" not in source
