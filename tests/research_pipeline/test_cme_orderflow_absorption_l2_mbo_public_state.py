from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from research_pipeline.cme_orderflow_absorption_l2_v1 import historical_runner as historical
from research_pipeline.cme_orderflow_absorption_l2_v1 import mbo_public_state_audit as audit
from research_pipeline.cme_orderflow_absorption_l2_v1 import v3_poc_fresh_august_replay as native


BASE = 1_800_000_000_000_000_000
PX = 7_250.0


def _private(offset: int, action: str, side: str, *, price: float = PX, size: int = 1,
             order_id: int = 1, flags: int = 0) -> historical.PrivateMBORecord:
    return historical.PrivateMBORecord(BASE + offset, action, side, price, size, order_id, flags)


def _ready(*, second_ask: bool = False) -> historical.HistoricalMBOToMBP10Adapter:
    adapter = historical.HistoricalMBOToMBP10Adapter()
    adapter.feed(_private(0, "R", "B", size=0, order_id=0, flags=historical.F_SNAPSHOT))
    adapter.feed(_private(1, "A", "B", order_id=1, flags=historical.F_SNAPSHOT))
    if second_ask:
        adapter.feed(_private(2, "A", "A", price=PX + .50, order_id=3, flags=historical.F_SNAPSHOT))
    event = adapter.feed(_private(
        3, "A", "A", price=PX + .25, order_id=2,
        flags=historical.F_SNAPSHOT | historical.F_LAST,
    ))
    assert event is not None
    return adapter


class _Runner:
    def __init__(self, *, position: object | None = None) -> None:
        self.signals = SimpleNamespace(position=position)
        self.es_quote = (PX, PX + .25)
        self.mes_quote = (PX / 10, PX / 10 + .25)
        self.es_quote_timestamp_ns = BASE
        self.mes_quote_timestamp_ns = BASE
        self.diagnostic_events: list[dict[str, object]] = []
        self.public_calls = 0

    def observe_public(self, _public: object) -> None:
        self.public_calls += 1


def test_valid_mbo_derived_public_book_is_executable():
    adapter = _ready()
    assert adapter.state == "EXECUTABLE"
    assert historical._quote(adapter.current_public_snapshot(BASE + 4)) == (PX, PX + .25)


def test_temporary_public_book_loss_suspends_and_emits_no_public_event():
    adapter = _ready()
    assert adapter.feed(_private(4, "A", "B", price=PX + .25, order_id=4)) is None
    assert adapter.state == "TEMPORARILY_NON_EXECUTABLE"


def test_multiple_reconstruction_rows_remain_suspended_without_raising():
    adapter = _ready(second_ask=True)
    adapter.feed(_private(4, "A", "B", price=PX + .25, order_id=4))
    assert adapter.feed(_private(5, "A", "B", price=PX + .50, order_id=5)) is None
    assert adapter.state == "TEMPORARILY_NON_EXECUTABLE"
    assert adapter.temporary_non_executable_records == 2


def test_route_clears_stale_bbo_and_does_not_advance_strategy_during_suspension():
    adapter = _ready()
    public = adapter.feed(_private(4, "A", "B", price=PX + .25, order_id=4))
    runner = _Runner()
    assert historical.route_mbo_public_event(runner, adapter, public, BASE + 4) is False
    assert runner.es_quote is None and runner.mes_quote is None and runner.public_calls == 0


def test_valid_reopen_requires_fresh_two_sided_uncrossed_book_and_suppresses_delta():
    adapter = _ready(second_ask=True)
    adapter.feed(_private(4, "A", "B", price=PX + .25, order_id=4))
    reopened = adapter.feed(_private(5, "C", "A", price=PX + .25, order_id=2))
    assert reopened is not None and reopened.update is None
    assert historical._quote(reopened.snapshot) == (PX + .25, PX + .50)
    assert adapter.last_transient == {
        "start_timestamp_ns": BASE + 4, "reopen_timestamp_ns": BASE + 5,
        "non_executable_records": 1,
    }


def test_may4_same_group_trade_fill_add_cancel_sequence_has_one_zero_duration_transition():
    adapter = _ready(second_ask=True)
    assert adapter.feed(_private(4, "T", "B", price=PX + .25, order_id=4)) is not None
    assert adapter.feed(_private(4, "F", "A", price=PX + .25, order_id=2)) is not None
    assert adapter.feed(_private(4, "A", "B", price=PX + .25, size=3, order_id=4)) is None
    reopened = adapter.feed(_private(4, "C", "A", price=PX + .25, order_id=2))
    assert reopened is not None and historical._quote(reopened.snapshot) == (PX + .25, PX + .50)
    assert adapter.last_transient == {
        "start_timestamp_ns": BASE + 4, "reopen_timestamp_ns": BASE + 4,
        "non_executable_records": 1,
    }


def test_crossed_reconstruction_is_suspended_until_uncrossed():
    adapter = _ready(second_ask=True)
    assert adapter.feed(_private(4, "A", "B", price=PX + .50, order_id=4)) is None
    snapshot = adapter.current_public_snapshot(BASE + 4)
    assert snapshot.bids[0].price > snapshot.asks[0].price
    reopened = adapter.feed(_private(5, "C", "B", price=PX + .50, order_id=4))
    assert reopened is not None and adapter.state == "EXECUTABLE"


def test_one_sided_reconstruction_is_suspended_until_ask_returns():
    adapter = _ready()
    assert adapter.feed(_private(4, "C", "A", price=PX + .25, order_id=2)) is None
    snapshot = adapter.current_public_snapshot(BASE + 4)
    assert snapshot.bids and not snapshot.asks
    reopened = adapter.feed(_private(5, "A", "A", price=PX + .25, order_id=5))
    assert reopened is not None and adapter.state == "EXECUTABLE"


def test_impossible_ordering_and_ordinary_reset_still_fail_closed():
    adapter = _ready()
    with pytest.raises(historical.HistoricalReplayError, match="decreased"):
        adapter.feed(_private(2, "T", "A", price=PX + .25, order_id=90))
    adapter = _ready()
    with pytest.raises(historical.HistoricalReplayError, match="ordinary reset"):
        adapter.feed(_private(4, "R", "B", size=0, order_id=0))


def _offbook_ready() -> historical.HistoricalMBOToMBP10Adapter:
    adapter = _ready()
    for index in range(4, 13):
        adapter.feed(_private(index, "A", "A", price=PX + .25 * (index - 2), order_id=index))
    return adapter


def test_private_retained_anomaly_stays_strategy_invisible():
    adapter = _offbook_ready()
    raw = 752_000_000_000_000
    event = adapter.feed(historical.PrivateMBORecord(BASE + 20, "A", "A", 752000.0, 1, 77, 128, raw))
    assert event is not None and all(level.price != 752000.0 for level in event.snapshot.asks)
    assert len(adapter.source_integrity_diagnostics()) == 1


def test_private_anomaly_reaching_public_top_ten_fails_closed():
    adapter = _ready()
    with pytest.raises(historical.HistoricalReplayError, match="OFFBOOK_ANOMALY_EXPOSED"):
        adapter.feed(historical.PrivateMBORecord(
            BASE + 20, "A", "A", 752000.0, 1, 77, 128, 752_000_000_000_000,
        ))


def test_open_position_overlap_fails_without_invented_execution():
    adapter = _ready()
    public = adapter.feed(_private(4, "A", "B", price=PX + .25, order_id=4))
    runner = _Runner(position=object())
    with pytest.raises(historical.HistoricalReplayError, match="overlapped an open position"):
        historical.route_mbo_public_event(runner, adapter, public, BASE + 4)
    assert runner.public_calls == 0


def test_unresolved_non_executable_source_boundary_fails_closed():
    adapter = _ready()
    adapter.feed(_private(4, "C", "A", price=PX + .25, order_id=2))
    with pytest.raises(historical.HistoricalReplayError, match="remained TEMPORARILY_NON_EXECUTABLE"):
        adapter.finish()


def _raw(offset: int, action: str, side: str, *, price: float = PX, size: int = 1,
         order_id: int = 1, flags: int = 0, sequence: int = 1) -> SimpleNamespace:
    raw_price = historical.UNDEF_PRICE if action == "R" else int(price * historical.RAW_PRICE_SCALE)
    return SimpleNamespace(
        ts_event=BASE + offset, ts_recv=BASE + offset, action=action, side=side,
        price=raw_price, size=size, order_id=order_id, flags=flags, sequence=sequence,
    )


def _audit_ready(source_tail_complete: bool) -> audit.SessionAuditor:
    source = audit.SessionSource("TEST", "2026-07-13", Path("sealed.dbn"), source_tail_complete)
    auditor = audit.SessionAuditor(source)
    auditor.observe(_raw(0, "R", "N", size=0, order_id=0, flags=historical.F_SNAPSHOT), 0)
    auditor.observe(_raw(1, "A", "B", order_id=1, flags=historical.F_SNAPSHOT), 1)
    auditor.observe(_raw(2, "A", "A", price=PX + .50, order_id=4, flags=historical.F_SNAPSHOT), 2)
    auditor.observe(_raw(3, "A", "A", price=PX + .25, order_id=2,
                         flags=historical.F_SNAPSHOT | historical.F_LAST), 3)
    return auditor


def test_incomplete_june_july_tail_remains_explicitly_fail_closed_without_strategy():
    result = _audit_ready(False).finish()
    assert result["source_tail_classification"] == "INTENTIONALLY_INCOMPLETE_FAIL_CLOSED_SOURCE_TAIL"
    assert result["replay_completion_allowed"] is False
    assert result["setups_created"] == result["trades_created"] == 0 and result["pnl_calculated"] is False


def test_native_mbp10_temporary_state_contract_remains_unchanged():
    def record(timestamp: int, bid: int, ask: int) -> SimpleNamespace:
        level = SimpleNamespace(bid_px=bid, ask_px=ask, bid_sz=1, ask_sz=1, bid_ct=1, ask_ct=1)
        return SimpleNamespace(ts_recv=timestamp, action="A", side="B", price=bid, size=1, levels=(level,))
    adapter = native.NativeMBP10Adapter()
    scale = historical.RAW_PRICE_SCALE
    assert adapter.feed(record(BASE, int(PX * scale), int((PX + .25) * scale))) is not None
    assert adapter.feed(record(BASE + 1, int(PX * scale), int(PX * scale))) is None
    assert adapter.state == "TEMPORARILY_NON_EXECUTABLE"
    assert adapter.feed(record(BASE + 2, int(PX * scale), int((PX + .25) * scale))) is not None


def test_audit_records_transition_without_strategy_or_pnl_access():
    auditor = _audit_ready(True)
    auditor.observe(_raw(4, "A", "B", price=PX + .25, order_id=3, sequence=7), 4)
    auditor.observe(_raw(4, "C", "A", price=PX + .25, order_id=2, sequence=7), 5)
    result = auditor.finish()
    assert result["temporary_non_executable_episode_count"] == 1
    assert result["episodes"][0]["classification"] == "ATOMIC_MBO_RECONSTRUCTION_TRANSITION"
    assert result["strategy_logic_executed"] is False
    assert result["interactions_created"] == result["setups_created"] == result["trades_created"] == 0
    assert result["pnl_calculated"] is False


def test_compact_audit_distinguishes_source_start_materialization_and_maintenance(tmp_path: Path):
    source_episode = {
        "episode_index": 0, "start_ts_recv_utc": "2026-05-04T00:00:00.000000Z",
        "duration_ns_by_ts_recv": 0, "non_executable_record_count": 1,
        "classification": "ATOMIC_MBO_RECONSTRUCTION_TRANSITION",
        "initiating_event": {"source_record_index": 10}, "reopen_event": {"source_record_index": 11},
    }
    materialized_episode = {
        "episode_index": 1, "start_ts_recv_utc": "2026-05-04T13:30:00.000000Z",
        "duration_ns_by_ts_recv": 0, "non_executable_record_count": 1,
        "classification": "ATOMIC_MBO_RECONSTRUCTION_TRANSITION",
        "initiating_event": {"source_record_index": 20}, "reopen_event": {"source_record_index": 21},
    }
    maintenance_episode = {
        "episode_index": 2, "start_ts_recv_utc": "2026-05-04T21:45:00.000000Z",
        "duration_ns_by_ts_recv": 900_000_000_000, "non_executable_record_count": 3,
        "classification": "TEMPORARY_MBO_RECONSTRUCTION_TRANSITION",
        "initiating_event": {"source_record_index": 30}, "reopen_event": {"source_record_index": 33},
    }
    session = {
        "period_id": "MAY_2026", "session_date": "2026-05-04", "mbo_record_count": 34,
        "executable_book_episode_count": 2, "temporary_non_executable_episode_count": 3,
        "temporary_non_executable_record_count": 5, "anomaly_retention_count": 0,
        "unresolved_episode_count": 0, "maintenance_or_reset_transition_count": 0,
        "source_integrity_anomalies": [], "exceptions": [], "final_adapter_state": "EXECUTABLE",
        "source_tail_classification": "COMPLETE_HARD_FLAT_SOURCE",
        "episodes": [source_episode, materialized_episode, maintenance_episode],
    }
    payload = {
        "sessions": [session], "exception_count": 0, "unresolved_episode_count": 0,
        "temporary_non_executable_episode_count": 3, "strategy_logic_executed": False,
        "pnl_calculated": False,
    }
    ledger = tmp_path / "audit.json"
    ledger.write_text("{}", encoding="utf-8")
    audit.finalize_audit_payload(payload)
    summary = audit.compact_audit_summary(payload, source_ledger=ledger)
    assert summary["scheduled_maintenance_transition_count"] == 1
    assert summary["may4_first_source_episode"]["start_source_record_index"] == 10
    assert summary["may4_first_strategy_materialized_episode"]["start_source_record_index"] == 20
    assert summary["maximum_duration_episode"]["scheduled_maintenance_transition"] is True
