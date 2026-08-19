from __future__ import annotations

import hashlib
import json
import sys
import types
from datetime import date, datetime, time, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from research_pipeline.cme_orderflow_absorption_l2_v1 import dec2025_feb2026_quote as quote
from research_pipeline.cme_orderflow_absorption_l2_v1 import v3_poc_dec2025_jan2026_replay as replay
from research_pipeline.cme_orderflow_absorption_l2_v1 import v3_poc_fresh_august_replay as native
from research_pipeline.cme_orderflow_absorption_l2_v1.model import StructuralLevel


def _ns(day: date, clock: time) -> int:
    return int(datetime.combine(day, clock, tzinfo=timezone.utc).timestamp() * 1_000_000_000)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_fixture(root: Path) -> Path:
    sessions = quote.build_sessions(replay.FIRST_TARGET, replay.LAST_TARGET)
    components = []
    files = {}
    prior_map = {}
    for session in sessions:
        day, prior = session.session_date, session.previous_rth_date
        es, mes = (("ESZ5", "MESZ5") if day <= date(2025, 12, 16) else ("ESH6", "MESH6"))
        prior_es = "ESZ5" if prior <= date(2025, 12, 16) else "ESH6"
        target_end_ns = _ns(day, session.effective_end) + 1_000_000_000
        target_end = replay._utc(target_end_ns).replace(".000000000Z", "Z")
        prior_end = min(quote.PROFILE_END, quote._session_end(prior)).isoformat()
        definitions = (
            ("ES_MBP10", "mbp-10", es, f"{day}T13:00:00Z", target_end,
             f"es_mbp10/{es}_{day}_130000_mbp-10.dbn", None),
            ("MES_MBP1", "mbp-1", mes, f"{day}T13:30:00Z", target_end,
             f"mes_mbp1/{mes}_{day}_133000_mbp-1.dbn", None),
            ("PRIOR_RTH_TRADES", "trades", prior_es, f"{prior}T13:30:00Z", f"{prior}T{prior_end}Z",
             f"es_prior_rth_trades/{prior_es}_{prior}_trades.dbn", str(prior)),
        )
        prior_map[str(day)] = str(prior)
        for purpose, schema, symbol, start, end, relative, prior_value in definitions:
            component = {
                "target_session": str(day), "prior_rth_date": prior_value, "purpose": purpose,
                "dataset": "GLBX.MDP3", "schema": schema, "raw_symbol": symbol,
                "start_utc": start, "end_utc": end, "local_path": relative,
            }
            local = root / relative
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(f"fixture:{relative}".encode())
            components.append(component)
            files[relative] = {**component, "status": "DOWNLOADED_VERIFIED", "bytes": local.stat().st_size,
                               "sha256": _hash(local)}
    payload = {
        "status": "ACQUISITION_COMPLETE_VERIFIED",
        "strategy_id": replay.STRATEGY_ID,
        "v3_contract_sha256": replay.V3_CONTRACT_SHA256,
        "evidence_classification": replay.EVIDENCE_LABEL,
        "target_session_count": 42,
        "first_target_session": "2025-12-01",
        "last_target_session": "2026-01-30",
        "prior_rth_by_target_session": prior_map,
        "constraints": {"no_mbo": True, "no_strategy_replay": True, "no_outcomes_inspected": True,
                        "february_excluded": True},
        "components": components,
        "files": files,
    }
    path = root / "acquisition-manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    _supplement_fixture(root, sessions)
    return path


def _supplement_fixture(root: Path, sessions: tuple[object, ...]) -> None:
    supplement_root = root / replay.SUPPLEMENT_DIRECTORY
    supplement_root.mkdir()
    bindings = root.parent / f"{root.name}-contract-bindings"
    bindings.mkdir()
    audit_path, quote_path = bindings / "calendar-audit.json", bindings / "supplement-quote.json"
    audit_path.write_text("{}", encoding="utf-8")
    quote_path.write_text("{}", encoding="utf-8")
    requests, files = [], {}
    for session in sessions:
        prior = session.previous_rth_date
        if prior in replay.EARLY_RTH_CLOSE_DATES:
            continue
        index = len(requests)
        symbol = "ESZ5" if prior <= date(2025, 12, 16) else "ESH6"
        relative = f"{symbol}_{prior}_200000_210000_trades.dbn.zst"
        request = {
            "index": index, "target_session": str(session.session_date), "prior_rth_date": str(prior),
            "dataset": "GLBX.MDP3", "schema": "trades", "stype_in": "raw_symbol",
            "raw_symbol": symbol, "start_utc": f"{prior}T20:00:00Z", "end_utc": f"{prior}T21:00:00Z",
            "local_path": relative,
        }
        local = supplement_root / relative
        local.write_bytes(f"fixture:{relative}".encode())
        requests.append(request)
        files[relative] = {**request, "status": "DOWNLOADED_VERIFIED", "bytes": local.stat().st_size,
                           "sha256": _hash(local)}
    assert len(requests) == 40
    manifest = {
        "status": "SUPPLEMENT_ACQUISITION_COMPLETE_VERIFIED", "strategy_id": replay.STRATEGY_ID,
        "v3_contract_sha256": replay.V3_CONTRACT_SHA256, "dataset": "GLBX.MDP3", "schema": "trades",
        "request_count": 40, "verified_file_count": 40, "symbol_counts": {"ESZ5": 12, "ESH6": 28},
        "approved_quote": {"path": str(quote_path), "sha256": _hash(quote_path),
                           "total_usd": replay.EXPECTED_SUPPLEMENT_TOTAL_USD},
        "calendar_audit": {"path": str(audit_path), "sha256": _hash(audit_path)},
        "constraints": {"supplemental_tails_only": True, "no_overlapping_1430_2000_purchase": True,
                        "no_mbp_10": True, "no_mbp_1": True, "no_mbo": True,
                        "strategy_executed": False, "outcomes_inspected": False},
        "requests": requests, "files": files,
    }
    (supplement_root / replay.SUPPLEMENT_MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")


def _level(bid: int = 5000_000_000_000, ask: int = 5000_250_000_000) -> SimpleNamespace:
    return SimpleNamespace(bid_px=bid, bid_sz=10, bid_ct=1, ask_px=ask, ask_sz=10, ask_ct=1)


def _native_record(timestamp: int, *, action: str = "R", side: str = "N",
                   bid: int = 5000_000_000_000, ask: int = 5000_250_000_000) -> SimpleNamespace:
    return SimpleNamespace(ts_recv=timestamp, ts_event=timestamp, action=action, side=side,
                           price=bid, size=1, levels=(_level(bid, ask),))


def test_manifest_exact_42_sessions_roll_prior_mapping_and_hashes(tmp_path: Path):
    _manifest_fixture(tmp_path)
    result = replay.verify_acquisition_manifest(tmp_path)
    assert result["files_verified"] == 126
    assert result["by_purpose"] == {"ES_MBP10": 42, "MES_MBP1": 42, "PRIOR_RTH_TRADES": 42}
    assert result["target_sessions"][0] == "2025-12-01" and result["target_sessions"][-1] == "2026-01-30"
    assert result["prior_rth_mapping"]["2025-12-01"] == "2025-11-28"
    roll = result["session_inputs"]["2025-12-17"]
    assert roll["ES_MBP10"]["raw_symbol"] == "ESH6"
    assert roll["MES_MBP1"]["raw_symbol"] == "MESH6"
    assert roll["PRIOR_RTH_TRADES"]["raw_symbol"] == "ESZ5"


def test_supplement_and_all_42_semantic_sufficiency(tmp_path: Path):
    _manifest_fixture(tmp_path)
    supplement = replay.verify_supplemental_manifest(tmp_path)
    assert supplement["files_verified"] == 40
    assert supplement["symbol_counts"] == {"ESZ5": 12, "ESH6": 28}
    preflight = replay.verify_data_preflight(tmp_path)
    assert preflight["status"] == "ALL_42_SESSIONS_DATA_SUFFICIENT"
    assert preflight["sufficient_session_count"] == 42 and preflight["excluded_sessions"] == []
    normal = next(row for row in preflight["sufficiency_matrix"] if row["session_date"] == "2025-12-02")
    assert normal["required_prior_rth_utc"] == {
        "start_utc": "2025-12-01T14:30:00.000000000Z", "end_utc": "2025-12-01T21:00:00.000000000Z",
    }
    assert normal["base_covered_range"]["start_utc"].endswith("T14:30:00.000000000Z")
    assert normal["supplement_covered_range"]["start_utc"].endswith("T20:00:00.000000000Z")
    dec1 = preflight["sufficiency_matrix"][0]
    assert dec1["prior_rth_date"] == "2025-11-28" and dec1["supplement_covered_range"] is None
    assert dec1["required_prior_rth_utc"]["start_utc"].endswith("T14:30:00.000000000Z")
    assert dec1["required_prior_rth_utc"]["end_utc"].endswith("T18:00:00.000000000Z")


def test_manifest_sha_and_size_tampering_fail_closed(tmp_path: Path):
    _manifest_fixture(tmp_path)
    target = next((tmp_path / "es_mbp10").iterdir())
    target.write_bytes(b"tampered")
    with pytest.raises(replay.DecJanReplayError, match="size mismatch|SHA-256 mismatch"):
        replay.verify_acquisition_manifest(tmp_path)


def test_frozen_v3_semantics_quality_poc_confirmation_execution_and_risk():
    replay._validate_frozen_strategy()
    contract = replay.v3_contract()
    assert replay.ELIGIBLE_STRUCTURAL_LEVELS == ("PRIOR_RTH_POC",)
    assert replay.V2_CONFIG.min_quality_score == 0.50
    assert contract["execution"] == {
        "confirmation_window_seconds_inclusive": [5.0, 15.0],
        "confirmation_favorable_ticks": 3,
        "entry_latency_ms": 2.0,
        "stop_buffer_ticks": 5,
        "target_r": 3.0,
        "risk_budget_usd": 250.0,
        "es_first": True,
        "mes_fallback": True,
        "max_es_contracts": 6,
        "max_mes_contracts": 60,
    }
    assert replay.V3_CONTRACT_SHA256 == "a0ce94eeb78dcbf865cf4464bdf97ebc3f014a8ec5e2f559f99798534dfcbcb4"


def test_native_adapter_initialization_transient_no_stale_bbo_and_reopen():
    adapter = native.NativeMBP10Adapter()
    day = date(2025, 12, 2)
    first = adapter.feed(_native_record(_ns(day, time(13, 0))))
    assert first is not None and adapter.state == "EXECUTABLE"
    crossed = adapter.feed(_native_record(_ns(day, time(14, 0)), action="M", side="A",
                                            bid=5001_000_000_000, ask=5000_750_000_000))
    assert crossed is None and adapter.state == "TEMPORARILY_NON_EXECUTABLE" and adapter.previous is None
    reopened = adapter.feed(_native_record(_ns(day, time(14, 0, 1))))
    assert reopened is not None and adapter.state == "EXECUTABLE"


def test_native_adapter_ignores_zero_effect_modify_outside_public_top_ten():
    adapter = native.NativeMBP10Adapter()
    day = date(2025, 12, 2)
    adapter.feed(_native_record(_ns(day, time(13, 0))))
    record = _native_record(_ns(day, time(13, 0, 1)), action="M", side="A")
    record.price = 686_400_000_000_000
    public = adapter.feed(record)
    assert public is not None
    assert public.update is None
    assert adapter.state == "EXECUTABLE"


def test_calendar_maintenance_and_shortened_hard_flat_are_explicit():
    normal = date(2025, 12, 2)
    assert replay._in_maintenance(_ns(normal, time(22, 0)), normal)
    assert not replay._in_maintenance(_ns(normal, time(21, 59, 59)), normal)
    session = next(item for item in quote.build_sessions(replay.FIRST_TARGET, replay.LAST_TARGET)
                   if item.session_date == date(2025, 12, 24))
    assert session.effective_end == time(18, 15)
    assert replay._effective_hard_flat(normal).time() == time(22, 0)
    assert replay._effective_hard_flat(date(2025, 12, 24)).time() == time(18, 15)


def test_merged_profile_excludes_pre_rth_and_concatenates_tail(monkeypatch: pytest.MonkeyPatch):
    day = date(2025, 12, 1)
    base, tail = Path("base"), Path("tail")
    records = {
        base: [
            SimpleNamespace(ts_event=_ns(day, time(13, 45)), price=4999_000_000_000, size=1000),
            SimpleNamespace(ts_event=_ns(day, time(14, 30)), price=5000_000_000_000, size=2),
            SimpleNamespace(ts_event=_ns(day, time(19, 59)), price=5000_250_000_000, size=1),
        ],
        tail: [SimpleNamespace(ts_event=_ns(day, time(20, 0)), price=5000_500_000_000, size=3)],
    }
    monkeypatch.setattr(replay, "_stream_trade_records", lambda path: iter(records[path]))
    levels = replay._profile_levels_from_trade_sources(
        (base, tail), required_start_ns=_ns(day, time(14, 30)), required_end_ns=_ns(day, time(21, 0)),
    )
    by_name = {level.name: level.price for level in levels}
    assert by_name["PRIOR_RTH_LOW"] == 5000.0
    assert by_name["PRIOR_RTH_HIGH"] == 5000.5


def test_degraded_nov28_decision_is_source_only_and_constructs_poc(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    path = tmp_path / "prior.dbn"
    path.write_bytes(b"fixture")
    start = _ns(date(2025, 11, 28), time(13, 30))
    end = _ns(date(2025, 11, 28), time(18, 15))
    records = [SimpleNamespace(ts_event=start + index * ((end - start - 1) // 240),
                               price=5000_000_000_000, size=1) for index in range(241)]

    class Store:
        @staticmethod
        def from_file(_path: Path): return iter(records)

    monkeypatch.setitem(sys.modules, "databento", types.SimpleNamespace(DBNStore=Store))
    preflight = {
        "base": {"session_inputs": {"2025-12-01": {"PRIOR_RTH_TRADES": {"local_path": "prior.dbn"}}}},
        "supplement": {"by_target_session": {}},
        "sufficiency_matrix": [{
            "session_date": "2025-12-01", "overall_sufficient": True,
            "required_prior_rth_utc": {"start_utc": "2025-11-28T14:30:00Z", "end_utc": "2025-11-28T18:00:00Z"},
        }],
    }
    result = replay.audit_degraded_nov28_source(tmp_path, preflight)
    assert result["usable"] is True
    assert result["outcome_or_pnl_consulted"] is False
    assert result["decision"] == "USABLE_WITH_DEGRADED_SOURCE_WARNING"


def test_read_only_adapter_audit_all_42_invokes_no_strategy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    sessions = quote.build_sessions(replay.FIRST_TARGET, replay.LAST_TARGET)
    session_inputs = {}
    for session in sessions:
        relative = f"es_mbp10/{session.session_date}.dbn"
        session_inputs[str(session.session_date)] = {"ES_MBP10": {
            "local_path": relative,
            "end_utc": replay._utc(_ns(session.session_date, session.effective_end) + 1_000_000_000).replace(
                ".000000000Z", "Z"
            ),
        }}
    verification = {
        "target_sessions": [str(item.session_date) for item in sessions],
        "session_inputs": session_inputs,
        "manifest_sha256": "a" * 64,
    }
    matrix = [{
        "session_date": str(item.session_date), "overall_sufficient": True,
        "required_prior_rth_utc": {"start_utc": replay._iso_seconds(replay._semantic_rth_window(item.session_date)[0]),
                                    "end_utc": replay._iso_seconds(replay._semantic_rth_window(item.session_date)[1])},
        "required_mes_mbp1_utc": {"start_utc": replay._iso_seconds(replay._semantic_rth_window(item.session_date)[0])},
        "effective_hard_flat_utc": replay._iso_seconds(replay._effective_hard_flat(item.session_date)),
    } for item in sessions]
    preflight = {
        "status": "ALL_42_SESSIONS_DATA_SUFFICIENT", "base": verification,
        "supplement": {"manifest_sha256": "b" * 64}, "sufficiency_matrix": matrix,
        "sufficient_session_count": 42, "excluded_sessions": [],
    }
    monkeypatch.setattr(replay, "verify_data_preflight", lambda *_args, **_kwargs: preflight)
    monkeypatch.setattr(replay, "audit_degraded_nov28_source", lambda *_args, **_kwargs: {
        "usable": True, "decision": "USABLE_WITH_DEGRADED_SOURCE_WARNING",
    })

    def stream(path: Path):
        day = date.fromisoformat(path.stem)
        cutoff = int(replay._effective_hard_flat(day).timestamp() * 1_000_000_000)
        yield _native_record(_ns(day, time(13, 0)))
        yield _native_record(cutoff - 100_000_000)
        yield _native_record(cutoff)

    monkeypatch.setattr(native, "_stream_native_mbp10_records", stream)
    monkeypatch.setattr(replay, "_in_maintenance", lambda *_args: False)
    monkeypatch.setattr(replay.historical, "HistoricalL2Runner",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("strategy invoked")))
    result = replay.audit_native_mbp10(tmp_path, verify_hashes=False)
    assert result["status"] == "ADAPTER_AUDIT_PASS"
    assert result["totals"]["adapter_exceptions"] == 0
    assert result["totals"]["unresolved_episodes"] == 0
    assert result["totals"]["hard_flat_boundaries_available"] == 42
    assert result["strategy_runner_invoked"] is False
    assert result["setups_created"] is False and result["trades_created"] is False


def test_audit_gate_rejects_unresolved_or_outcome_tainted_report(tmp_path: Path):
    path = tmp_path / "audit.json"
    path.write_text(json.dumps({
        "status": "ADAPTER_AUDIT_FAIL_CLOSED", "strategy_runner_invoked": False,
        "pnl_or_outcomes_accessed": False, "manifest_sha256": "a" * 64,
        "v3_contract_sha256": replay.V3_CONTRACT_SHA256,
        "supplement_manifest_verification": {"manifest_sha256": "b" * 64},
        "data_sufficiency_status": "ALL_42_SESSIONS_DATA_SUFFICIENT", "sufficient_session_count": 42,
        "totals": {"unresolved_episodes": 1, "adapter_exceptions": 0},
    }), encoding="utf-8")
    with pytest.raises(replay.DecJanReplayError, match="did not pass"):
        replay._load_passing_audit(path, {"base": {"manifest_sha256": "a" * 64},
                                          "supplement": {"manifest_sha256": "b" * 64}})


def test_no_network_download_or_optimizer_path_and_features_are_pre_gate():
    source = Path(replay.__file__).read_text(encoding="utf-8")
    assert "Historical(" not in source
    assert "get_range(" not in source
    assert "metadata.get_cost" not in source
    assert "optimizer" not in source.lower()
    assert "interaction-features.csv contains every completed POC interaction" in source
    assert "PRE_QUALITY_RAW_COMPONENTS" in source
