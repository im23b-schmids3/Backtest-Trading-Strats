from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path

import pytest

from research_pipeline.cme_orderflow_absorption_l2_v1 import dec2025_jan2026_supplement_acquisition as acquire
from research_pipeline.cme_orderflow_absorption_l2_v1.v3_poc_dec2025_jan2026_calendar_audit import (
    _canonical_sha256,
)


def _requests() -> list[dict[str, str]]:
    rows = []
    for index in range(40):
        month = 12 if index < 20 else 1
        year = 2025 if month == 12 else 2026
        day = index + 1 if month == 12 else index - 19
        date_text = f"{year:04d}-{month:02d}-{day:02d}"
        rows.append({
            "target_session": f"target-{index:02d}",
            "prior_rth_date": date_text,
            "dataset": "GLBX.MDP3",
            "schema": "trades",
            "stype_in": "raw_symbol",
            "raw_symbol": "ESZ5" if index < 12 else "ESH6",
            "start_utc": f"{date_text}T20:00:00Z",
            "end_utc": f"{date_text}T21:00:00Z",
        })
    return rows


def _write_contract(root: Path) -> tuple[Path, Path]:
    requests = _requests()
    audit_payload = {
        "strategy_id": acquire.STRATEGY_ID,
        "v3_contract_sha256": acquire.V3_CONTRACT_SHA256,
        "target_session_count": 42,
        "affected_session_count": 40,
        "supplemental_plan": {"request_count": 40, "requests": requests},
    }
    audit_path = root / "calendar-audit.json"
    audit_path.write_text(json.dumps(audit_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    costs = []
    for index in range(40):
        if index == 0:
            costs.append(acquire.APPROVED_ESZ5_USD)
        elif index == 12:
            costs.append(acquire.APPROVED_ESH6_USD)
        else:
            costs.append(Decimal(0))
    quote_payload = {
        "status": "SUPPLEMENT_QUOTE_COMPLETE_DOWNLOAD_NOT_REQUESTED",
        "calendar_audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        "request_spec_sha256": _canonical_sha256(requests),
        "v3_contract_sha256": acquire.V3_CONTRACT_SHA256,
        "request_count": 40,
        "dataset": "GLBX.MDP3",
        "schema": "trades",
        "symbol_subtotals_usd": {
            "ESZ5": str(acquire.APPROVED_ESZ5_USD),
            "ESH6": str(acquire.APPROVED_ESH6_USD),
        },
        "total_usd": str(acquire.APPROVED_TOTAL_USD),
        "requests": [dict(row, estimated_usd=str(cost)) for row, cost in zip(requests, costs, strict=True)],
    }
    quote_path = root / "supplement-quote.json"
    quote_path.write_text(json.dumps(quote_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit_path, quote_path


class FakeClient:
    def __init__(self, costs: list[str], *, fail_first_download: bool = False) -> None:
        self.metadata = self
        self.timeseries = self
        self.costs = list(costs)
        self.cost_calls: list[dict[str, object]] = []
        self.download_calls: list[dict[str, object]] = []
        self.fail_first_download = fail_first_download

    def get_cost(self, **request: object) -> str:
        self.cost_calls.append(request)
        return self.costs.pop(0)

    def get_range(self, **request: object) -> None:
        self.download_calls.append(request)
        path = Path(str(request["path"]))
        path.write_bytes(b"partial" if self.fail_first_download and len(self.download_calls) == 1 else b"dbn")
        if self.fail_first_download and len(self.download_calls) == 1:
            raise OSError("synthetic transient failure")


def _approved_costs() -> list[str]:
    return [
        str(acquire.APPROVED_ESZ5_USD if index == 0 else
            acquire.APPROVED_ESH6_USD if index == 12 else Decimal(0))
        for index in range(40)
    ]


def test_frozen_supplement_is_exactly_40_trade_tails(tmp_path: Path) -> None:
    audit_path, quote_path = _write_contract(tmp_path)
    frozen = acquire.load_frozen_supplement(audit_path, quote_path)
    assert len(frozen.requests) == 40
    assert sum(row.raw_symbol == "ESZ5" for row in frozen.requests) == 12
    assert sum(row.raw_symbol == "ESH6" for row in frozen.requests) == 28
    assert all(row.start_utc.endswith("T20:00:00Z") and row.end_utc.endswith("T21:00:00Z")
               for row in frozen.requests)
    assert all(row.api_request()["schema"] == "trades" for row in frozen.requests)
    assert all(token not in {row.api_request()["schema"] for row in frozen.requests}
               for token in ("mbp-10", "mbp-1", "mbo"))
    assert all("143000" not in row.filename and "200000_210000" in row.filename for row in frozen.requests)


def test_preflight_requotes_40_requests_and_never_downloads(tmp_path: Path) -> None:
    audit_path, quote_path = _write_contract(tmp_path)
    client = FakeClient(_approved_costs())
    output = tmp_path / "not-created"
    result = acquire.run(
        client=client, audit_path=audit_path, quote_path=quote_path, output_root=output, download=False,
    )
    assert result["status"] == "SUPPLEMENT_PREFLIGHT_QUOTE_COMPLETE_DOWNLOAD_NOT_REQUESTED"
    assert result["fresh_quote"]["total_usd"] == str(acquire.APPROVED_TOTAL_USD)
    assert len(client.cost_calls) == 40 and client.download_calls == []
    assert all(call["schema"] == "trades" for call in client.cost_calls)
    assert result["strategy_executed"] is False and result["outcomes_inspected"] is False
    assert not output.exists()


@pytest.mark.parametrize("total, message", [
    (Decimal("3.000000000001"), "hard cap"),
    (Decimal("1.896373689176"), "deviation"),
])
def test_quote_safety_limits_fail_closed(total: Decimal, message: str) -> None:
    with pytest.raises(acquire.SupplementalAcquisitionError, match=message):
        acquire._validate_fresh_total(total)


def test_quote_artifact_or_request_tampering_fails_closed(tmp_path: Path) -> None:
    audit_path, quote_path = _write_contract(tmp_path)
    quote = json.loads(quote_path.read_text(encoding="utf-8"))
    quote["requests"][0]["start_utc"] = "2025-12-01T19:30:00Z"
    quote_path.write_text(json.dumps(quote), encoding="utf-8")
    with pytest.raises(acquire.SupplementalAcquisitionError, match="differs from frozen request"):
        acquire.load_frozen_supplement(audit_path, quote_path)


def test_atomic_retry_hash_resume_and_unknown_files_fail_closed(tmp_path: Path) -> None:
    audit_path, quote_path = _write_contract(tmp_path)
    frozen = acquire.load_frozen_supplement(audit_path, quote_path)
    quote = acquire.fresh_quote(FakeClient(_approved_costs()), frozen)
    root = tmp_path / "supplement"
    manifest = acquire._initial_manifest(frozen, quote)
    acquire._atomic_write_manifest(root, manifest)
    item = frozen.requests[0]
    client = FakeClient([], fail_first_download=True)
    result = acquire._download_one(
        client=client, root=root, item=item, manifest=manifest,
        fresh_cost=quote["cost_by_filename_usd"][item.filename],
    )
    assert result == "DOWNLOADED_VERIFIED" and len(client.download_calls) == 2
    destination = root / item.filename
    assert destination.is_file() and not destination.with_suffix(destination.suffix + ".part").exists()
    assert acquire._download_one(
        client=client, root=root, item=item, manifest=manifest,
        fresh_cost=quote["cost_by_filename_usd"][item.filename],
    ) == "SKIPPED_VERIFIED"
    assert len(client.download_calls) == 2
    destination.write_bytes(b"tampered")
    with pytest.raises(acquire.SupplementalAcquisitionError, match="hash/size mismatch"):
        acquire._verified_existing(root, item, manifest)
    destination.write_bytes(b"dbn")
    (root / "unknown.dbn.zst").write_bytes(b"unknown")
    with pytest.raises(acquire.SupplementalAcquisitionError, match="unknown existing file"):
        acquire._assert_output_tree(root, frozen.requests, manifest)


def test_stale_part_fails_closed(tmp_path: Path) -> None:
    audit_path, quote_path = _write_contract(tmp_path)
    frozen = acquire.load_frozen_supplement(audit_path, quote_path)
    root = tmp_path / "supplement"
    root.mkdir()
    (root / (frozen.requests[0].filename + ".part")).write_bytes(b"stale")
    with pytest.raises(acquire.SupplementalAcquisitionError, match="stale partial"):
        acquire._assert_output_tree(root, frozen.requests, None)


def test_v3_contract_and_strategy_isolation_are_unchanged() -> None:
    assert acquire.V3_CONTRACT_SHA256 == "a0ce94eeb78dcbf865cf4464bdf97ebc3f014a8ec5e2f559f99798534dfcbcb4"
    source = Path(acquire.__file__).read_text(encoding="utf-8")
    assert "historical_runner" not in source
    assert "strategy_runner" not in source
    assert "get_range" in source and "metadata.get_cost" in source
