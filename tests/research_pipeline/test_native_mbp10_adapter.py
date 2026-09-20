from dataclasses import dataclass
from pathlib import Path

import pytest

from src.research_pipeline.cme_orderflow_absorption_l2_v1 import historical_runner as historical
from src.research_pipeline.cme_orderflow_absorption_l2_v1.public_book_adapters import (
    MBO_DERIVED_MBP10,
    NATIVE_MBP10,
    NativeDatabentoMBP10Adapter,
    source_model_adapter,
)
from src.research_pipeline.cme_orderflow_absorption_l2_v1.model import MBOEvent, MBOToMBP10View
from src.research_pipeline.cme_orderflow_absorption_l2_v1 import asia_w04_replay


@dataclass(frozen=True)
class Level:
    bid_px: int = 0
    bid_sz: int = 0
    bid_ct: int = 0
    ask_px: int = 0
    ask_sz: int = 0
    ask_ct: int = 0


@dataclass(frozen=True)
class Native:
    ts_event: int
    action: str
    side: str
    price: int
    size: int
    levels: tuple[Level, ...]
    ts_recv: int | None = None


RAW = 1_000_000_000


def book(ts: int, bid_size: int = 10, ask_size: int = 12) -> Native:
    return Native(ts, "A", "N", 0, 0, (
        Level(100 * RAW, bid_size, 2, 101 * RAW, ask_size, 3),
        Level(99 * RAW, 4, 1, 102 * RAW, 5, 1),
    ))


def test_native_snapshot_matches_mbo_derived_snapshot():
    native = NativeDatabentoMBP10Adapter()
    event = native.feed(book(1))
    assert event is not None
    view = MBOToMBP10View()
    view.apply(MBOEvent(1, "R", "B", 100, 0, 0), materialize_snapshot=False, materialize_update=False)
    view.apply(MBOEvent(1, "A", "B", 100, 10, 1), materialize_snapshot=False, materialize_update=False)
    view.apply(MBOEvent(1, "A", "A", 101, 12, 2), materialize_snapshot=False, materialize_update=False)
    assert event.snapshot.bid_px[:2] == (100.0, 99.0)
    assert event.snapshot.ask_px[:2] == (101.0, 102.0)
    assert event.snapshot.bid_sz[0] == 10
    assert event.snapshot.ask_ct[0] == 3


def test_native_aggregate_update_and_fill_emit_canonical_events():
    adapter = NativeDatabentoMBP10Adapter()
    assert adapter.feed(book(1)) is not None
    changed = Native(2, "M", "B", 100 * RAW, 3, (
        Level(100 * RAW, 13, 3, 101 * RAW, 12, 3),
        Level(99 * RAW, 4, 1, 102 * RAW, 5, 1),
    ))
    update_event = adapter.feed(changed)
    assert update_event is not None and update_event.update is not None
    assert update_event.update.kind == "MODIFY"
    traded = Native(3, "T", "B", 101 * RAW, 2, (
        Level(100 * RAW, 13, 3, 101 * RAW, 10, 2),
        Level(99 * RAW, 4, 1, 102 * RAW, 5, 1),
    ))
    fill_event = adapter.feed(traded)
    assert fill_event is not None and fill_event.execution is not None
    assert fill_event.execution.aggressor == "BUY"
    assert fill_event.update is not None
    assert fill_event.update.kind == "FILL"
    assert fill_event.update.side == "A"


def test_native_rejects_bad_model_timestamp_regression_and_malformed_book():
    with pytest.raises(historical.HistoricalReplayError):
        source_model_adapter("BAD")
    adapter = NativeDatabentoMBP10Adapter()
    adapter.feed(book(2))
    with pytest.raises(historical.HistoricalReplayError, match="timestamps decreased"):
        adapter.feed(book(1))
    bad = Native(3, "A", "N", 0, 0, (Level(100 * RAW, -1, 1, 101 * RAW, 12, 3),))
    with pytest.raises(historical.HistoricalReplayError):
        NativeDatabentoMBP10Adapter().feed(bad)


def test_native_locked_book_can_recover_but_unrecoverable_one_sided_fails():
    adapter = NativeDatabentoMBP10Adapter()
    adapter.feed(book(1))
    locked = Native(2, "M", "B", 100 * RAW, 10, (
        Level(100 * RAW, 10, 2, 100 * RAW, 12, 3),
    ))
    assert adapter.feed(locked) is None
    assert adapter.state == "TEMPORARILY_NON_EXECUTABLE"
    assert adapter.feed(book(3)) is not None
    one_sided = Native(4, "M", "B", 100 * RAW, 10, (Level(100 * RAW, 10, 2, 0, 0, 0),))
    with pytest.raises(historical.HistoricalReplayError):
        adapter.feed(one_sided)


def test_source_model_dispatch_preserves_mbo_adapter_and_accepts_native():
    assert isinstance(source_model_adapter(MBO_DERIVED_MBP10), historical.HistoricalMBOToMBP10Adapter)
    assert isinstance(source_model_adapter(NATIVE_MBP10), NativeDatabentoMBP10Adapter)


def test_sealed_asia_manifest_binds_native_dec_jan_as_two_declared_parts():
    sessions = asia_w04_replay.load_audit_sessions(
        Path("research_runs/CMEOrderflowAbsorption.ES_L2_ASIA_DATA_COVERAGE_AUDIT"), include_native=True
    )
    assert len(sessions) == 90
    assert sum(session.eligible for session in sessions) == 87
    bindings = asia_w04_replay.source_bindings(Path("."), sessions)
    dec1 = next(binding for binding in bindings if binding.days == ("2025-12-01",))
    assert dec1.extra_paths and dec1.path.name.startswith("ESZ5_2025-12-01_000000_130000_mbp-10")
    assert dec1.extra_paths[0].name.startswith("ESZ5_2025-12-01_130000_224501_mbp-10")
