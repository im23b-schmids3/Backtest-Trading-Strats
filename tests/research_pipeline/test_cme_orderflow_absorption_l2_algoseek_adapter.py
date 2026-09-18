from __future__ import annotations

import csv
from pathlib import Path

import pytest

from research_pipeline.cme_orderflow_absorption_l2_v1 import algoseek_adapter as algoseek
from research_pipeline.cme_orderflow_absorption_l2_v1 import algoseek_api as adapter_api
from research_pipeline.cme_orderflow_absorption_l2_v1 import v3_poc_fresh_august_replay as databento_native


def _depth(*, side: str, timestamp: str = "2023-03-10 09:30:00.000000000", ticker: str = "ESH3", security: str = "101") -> dict[str, str]:
    row = {"TradeDate": "2023-03-10", "EventDateTime": timestamp, "Ticker": ticker, "BaseSymbol": "ES",
           "SecurityID": security, "Side": side, "Flags": "D", "Depth": "10"}
    prices = ([4000.00, 3999.75, 3999.50] if side == "B" else [4000.25, 4000.50, 4000.75])
    for number in range(1, 11):
        if number <= len(prices):
            row |= {f"L{number}Price": str(prices[number - 1]), f"L{number}Size": str(number * 2), f"L{number}Orders": str(number)}
        else:
            row |= {f"L{number}Price": "", f"L{number}Size": "", f"L{number}Orders": ""}
    return row


def _taq(*, event_type: str, timestamp: str = "2023-03-10 09:30:00.000000000", price: str = "4000.25", quantity: str = "1", ticker: str = "ESH3", security: str = "101") -> dict[str, str]:
    return {"TradeDate": "2023-03-10", "EventDateTime": timestamp, "Ticker": ticker, "BaseSymbol": "ES",
            "SecurityID": security, "EventType": event_type, "Price": price, "Quantity": quantity,
            "Orders": "1", "Flags": "T", "TypeMask": event_type}


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def test_multiple_depth_parses_bid_and_ask_and_preserves_identity() -> None:
    bid = algoseek.parse_multiple_depth_row(_depth(side="B"), source_file="depth.csv", source_index=7)
    ask = algoseek.parse_multiple_depth_row(_depth(side="S"), source_file="depth.csv", source_index=8)
    assert bid.side == "B" and bid.levels[0].price == 4000.0
    assert ask.side == "A" and ask.levels[0].price == 4000.25
    assert ask.security_id == "101" and ask.ticker == "ESH3"


def test_multiple_depth_zero_placeholders_are_empty_levels() -> None:
    row = _depth(side="B")
    row["L3Price"], row["L3Size"], row["L3Orders"] = "0.0000", "0", "0"
    parsed = algoseek.parse_multiple_depth_row(row, source_file="depth.csv", source_index=1)
    assert len(parsed.levels) == 2


def test_latest_side_book_is_incomplete_then_executable() -> None:
    book = algoseek.LatestSideBook()
    assert book.apply(algoseek.parse_multiple_depth_row(_depth(side="B"), source_file="d", source_index=1)) is None
    assert book.state == "INCOMPLETE"
    snapshot = book.apply(algoseek.parse_multiple_depth_row(_depth(side="S"), source_file="d", source_index=2))
    assert snapshot is not None and book.state == "EXECUTABLE"
    assert snapshot.bid_px[:3] == (4000.0, 3999.75, 3999.5)


def test_taq_aggressor_and_bbo_parsing_and_es_mes_separation() -> None:
    buy = algoseek.parse_taq_row(_taq(event_type="TRADE AGRESSOR ON BUY"), instrument="ES", source_file="es", source_index=1)
    sell = algoseek.parse_taq_row(_taq(event_type="TRADE AGGRESSOR ON SELL"), instrument="ES", source_file="es", source_index=2)
    bid = algoseek.parse_taq_row(_taq(event_type="QUOTE BID", quantity="0"), instrument="MES", source_file="mes", source_index=1)
    ask = algoseek.parse_taq_row(_taq(event_type="QUOTE SELL", quantity="0"), instrument="MES", source_file="mes", source_index=2)
    assert (buy.event_kind, buy.aggressor) == ("TRADE", "BUY")
    assert (sell.event_kind, sell.aggressor) == ("TRADE", "SELL")
    assert (bid.event_kind, ask.event_kind, bid.instrument) == ("BID", "ASK", "MES")


def test_non_market_zero_price_control_row_is_ignored_but_parses() -> None:
    row = algoseek.parse_taq_row(_taq(event_type="EMPTY BOOK FINAL", price="0.0000", quantity="0"), instrument="MES", source_file="mes", source_index=1)
    assert row.event_kind == "OTHER" and row.price == 0.0


def test_non_market_non_tick_control_price_is_ignored_but_parses() -> None:
    row = algoseek.parse_taq_row(_taq(event_type="FIXING PRICE", price="3871.3800", quantity="0"), instrument="ES", source_file="es", source_index=1)
    assert row.event_kind == "OTHER" and row.price == 3871.38


def test_contracts_remain_separate_in_rows() -> None:
    march = algoseek.parse_taq_row(_taq(event_type="TRADE AGRESSOR ON BUY", ticker="ESH3", security="101"), instrument="ES", source_file="a", source_index=1)
    june = algoseek.parse_taq_row(_taq(event_type="TRADE AGRESSOR ON BUY", ticker="ESM3", security="202"), instrument="ES", source_file="b", source_index=1)
    assert (march.ticker, march.security_id) != (june.ticker, june.security_id)
    depth = algoseek.parse_multiple_depth_row(_depth(side="B", ticker="ESH3", security="101"), source_file="depth", source_index=1)
    mes = algoseek.parse_taq_row(_taq(event_type="QUOTE BID", ticker="MESH3", security="9", quantity="0"), instrument="MES", source_file="mes", source_index=1)
    with pytest.raises(algoseek.AlgoseekAdapterError, match="multiple contract"):
        list(algoseek.canonical_events(es_depth=[depth], es_taq=[march, june], mes_taq=[mes]))


def test_dataset_specific_es_security_ids_are_allowed_when_ticker_matches() -> None:
    depth = algoseek.parse_multiple_depth_row(_depth(side="B", security="805512667"), source_file="depth", source_index=1)
    taq = algoseek.parse_taq_row(_taq(event_type="QUOTE BID", security="206299", quantity="0"), instrument="ES", source_file="es", source_index=1)
    mes = algoseek.parse_taq_row(_taq(event_type="QUOTE BID", ticker="MESH3", security="2080", quantity="0"), instrument="MES", source_file="mes", source_index=1)
    identities = algoseek.validate_session_contracts(es_depth=[depth], es_taq=[taq], mes_taq=[mes])
    assert identities["ES_DEPTH"] == ("ESH3", "805512667")
    assert identities["ES_TAQ"] == ("ESH3", "206299")


def test_cst_to_utc_and_dst_fail_closed_behavior() -> None:
    before = algoseek.normalize_event_datetime("2023-03-12 01:59:59")
    after = algoseek.normalize_event_datetime("2023-03-12 03:00:00")
    assert after - before == 1_000_000_000
    with pytest.raises(algoseek.AlgoseekAdapterError, match="nonexistent"):
        algoseek.normalize_event_datetime("2023-03-12 02:30:00")
    with pytest.raises(algoseek.AlgoseekAdapterError, match="ambiguous"):
        algoseek.normalize_event_datetime("2023-11-05 01:30:00")


def test_identical_timestamp_policy_is_deterministic_and_stable() -> None:
    timestamp = "2023-03-10 09:30:00"
    depth = algoseek.parse_multiple_depth_row(_depth(side="B", timestamp=timestamp), source_file="depth", source_index=5)
    mes = algoseek.parse_taq_row(_taq(event_type="QUOTE BID", timestamp=timestamp), instrument="MES", source_file="mes", source_index=4)
    trade = algoseek.parse_taq_row(_taq(event_type="TRADE AGRESSOR ON BUY", timestamp=timestamp), instrument="ES", source_file="trade", source_index=2)
    quote_early = algoseek.parse_taq_row(_taq(event_type="QUOTE ASK", timestamp=timestamp), instrument="ES", source_file="quote", source_index=1)
    quote_late = algoseek.parse_taq_row(_taq(event_type="QUOTE BID", timestamp=timestamp), instrument="ES", source_file="quote", source_index=9)
    ordered = algoseek.ordered_source_events(es_depth=[depth], es_taq=[quote_late, trade, quote_early], mes_taq=[mes])
    assert [(algoseek._event_priority(row), row.source_index) for row in ordered] == [(0, 4), (1, 5), (2, 2), (3, 1), (3, 9)]


def test_canonical_events_follow_mes_depth_trade_es_bbo_order_and_keep_stage_contract_provider_neutral() -> None:
    timestamp = "2023-03-10 09:30:00"
    depth = [algoseek.parse_multiple_depth_row(_depth(side=side, timestamp=timestamp), source_file=f"{side}.csv", source_index=1) for side in ("B", "S")]
    es_taq = [algoseek.parse_taq_row(_taq(event_type=kind, timestamp=timestamp, price=price, quantity="1" if "TRADE" in kind else "0"), instrument="ES", source_file="es.csv", source_index=index) for index, (kind, price) in enumerate((("TRADE AGRESSOR ON BUY", "4000.25"), ("QUOTE BID", "4000.00")), 1)]
    mes_taq = [algoseek.parse_taq_row(_taq(event_type="QUOTE BID", timestamp=timestamp, price="4000.00", quantity="0", ticker="MESH3"), instrument="MES", source_file="mes.csv", source_index=1)]
    events = list(algoseek.canonical_events(es_depth=depth, es_taq=es_taq, mes_taq=mes_taq))
    assert [event.kind for event in events] == ["MES_BBO", "ES_DEPTH", "ES_TRADE", "ES_BBO"]
    assert events[1].raw_event_count == 2
    assert [item["side"] for item in events[1].raw_provenance] == ["B", "A"]
    assert events[2].execution is not None and events[2].snapshot is not None
    assert events[2].snapshot.__class__.__name__ == "MBP10Snapshot"
    assert events[2].execution.__class__.__name__ == "Execution"


def test_same_timestamp_depth_exposes_final_state_not_transient_locked_state() -> None:
    timestamp = "2023-03-10 09:30:00"
    bid = algoseek.parse_multiple_depth_row(_depth(side="B", timestamp=timestamp), source_file="z", source_index=2)
    ask = algoseek.parse_multiple_depth_row(_depth(side="S", timestamp=timestamp), source_file="a", source_index=1)
    mes = algoseek.parse_taq_row(_taq(event_type="QUOTE BID", timestamp=timestamp, ticker="MESH3", quantity="0"), instrument="MES", source_file="mes", source_index=1)
    es_trade = algoseek.parse_taq_row(_taq(event_type="TRADE AGRESSOR ON BUY", timestamp=timestamp), instrument="ES", source_file="trade", source_index=1)
    events = list(algoseek.canonical_events(es_depth=[bid, ask], es_taq=[es_trade], mes_taq=[mes]))
    depth_event = next(event for event in events if event.kind == "ES_DEPTH")
    assert depth_event.book_state == "EXECUTABLE"
    assert depth_event.snapshot is not None
    assert depth_event.raw_event_count == 2
    assert [item["source_file"] for item in depth_event.raw_provenance] == ["a", "z"]


def test_same_timestamp_duplicate_depth_rows_are_retained_not_collapsed() -> None:
    timestamp = "2023-03-10 09:30:00"
    bid = algoseek.parse_multiple_depth_row(_depth(side="B", timestamp=timestamp), source_file="depth", source_index=1)
    ask = algoseek.parse_multiple_depth_row(_depth(side="S", timestamp=timestamp), source_file="depth", source_index=2)
    duplicate = algoseek.parse_multiple_depth_row(_depth(side="S", timestamp=timestamp), source_file="depth", source_index=3)
    mes = algoseek.parse_taq_row(_taq(event_type="QUOTE BID", timestamp=timestamp, ticker="MESH3", quantity="0"), instrument="MES", source_file="mes", source_index=1)
    es_quote = algoseek.parse_taq_row(_taq(event_type="QUOTE BID", timestamp=timestamp, quantity="0"), instrument="ES", source_file="es", source_index=1)
    events = list(algoseek.canonical_events(es_depth=[bid, ask, duplicate], es_taq=[es_quote], mes_taq=[mes]))
    depth_event = next(event for event in events if event.kind == "ES_DEPTH")
    assert depth_event.raw_event_count == 3
    assert len(depth_event.raw_provenance) == 3
    assert [item["source_index"] for item in depth_event.raw_provenance] == [1, 2, 3]


def test_depth_batching_never_merges_adjacent_distinct_timestamps() -> None:
    first = algoseek.parse_multiple_depth_row(_depth(side="B", timestamp="2023-03-10 09:30:00"), source_file="depth", source_index=1)
    second = algoseek.parse_multiple_depth_row(_depth(side="S", timestamp="2023-03-10 09:30:01"), source_file="depth", source_index=2)
    groups = list(algoseek._depth_groups([first, second]))
    assert [len(group) for group in groups if isinstance(group, list)] == [1, 1]


def test_batched_depth_final_state_is_invariant_to_bid_ask_input_order() -> None:
    timestamp = "2023-03-10 09:30:00"
    bid = algoseek.parse_multiple_depth_row(_depth(side="B", timestamp=timestamp), source_file="depth", source_index=1)
    ask = algoseek.parse_multiple_depth_row(_depth(side="S", timestamp=timestamp), source_file="depth", source_index=2)
    mes = algoseek.parse_taq_row(_taq(event_type="QUOTE BID", timestamp=timestamp, ticker="MESH3", quantity="0"), instrument="MES", source_file="mes", source_index=1)
    es_quote = algoseek.parse_taq_row(_taq(event_type="QUOTE BID", timestamp=timestamp, quantity="0"), instrument="ES", source_file="es", source_index=1)
    forward = next(event for event in algoseek.canonical_events(es_depth=[bid, ask], es_taq=[es_quote], mes_taq=[mes]) if event.kind == "ES_DEPTH")
    reverse = next(event for event in algoseek.canonical_events(es_depth=[ask, bid], es_taq=[es_quote], mes_taq=[mes]) if event.kind == "ES_DEPTH")
    assert forward.snapshot == reverse.snapshot
    assert forward.book_state == reverse.book_state == "EXECUTABLE"


def test_profile_uses_executions_not_quote_volume() -> None:
    trades = [
        algoseek.parse_taq_row(_taq(event_type="TRADE AGRESSOR ON BUY", price="4000.00", quantity="10"), instrument="ES", source_file="x", source_index=1),
        algoseek.parse_taq_row(_taq(event_type="TRADE AGRESSOR ON SELL", price="4000.25", quantity="4"), instrument="ES", source_file="x", source_index=2),
        algoseek.parse_taq_row(_taq(event_type="QUOTE SELL", price="4000.50", quantity="999"), instrument="ES", source_file="x", source_index=3),
    ]
    profile = algoseek.profile_from_executions(trades)
    assert profile["POC"] == 4000.0 and profile["HIGH"] == 4000.25


def test_provenance_and_session_ownership_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"; source.write_text("x\n1\n", encoding="utf-8")
    provenance = algoseek.provider_provenance(files=[source], ticker="ESH3", security_id="101")
    assert provenance["provider_semantics"] == "ALGOSEEK_CAUSAL_VARIANT"
    assert provenance["ordering_policy_id"] == "algoseek-causal-order-v2"
    assert provenance["depth_assembly_policy_id"] == "algoseek-depth-assembly-v2-final-state-with-raw-provenance"
    assert "NO_MBO_ADD_CANCEL_FILL_MODIFY_RESET_PROVENANCE" in provenance["known_semantic_limitations"]
    assert algoseek.validate_session_provider_ownership([{"date": "2023-03-10", "provider": "ALGOSEEK"}]) == {"2023-03-10": "ALGOSEEK"}
    with pytest.raises(algoseek.AlgoseekAdapterError, match="duplicate"):
        algoseek.validate_session_provider_ownership([{"date": "2023-03-10", "provider": "ALGOSEEK"}, {"date": "2023-03-10", "provider": "DATABENTO"}])


def test_audit_reports_ties_and_does_not_run_strategy(tmp_path: Path) -> None:
    depth, es, mes = tmp_path / "depth.csv", tmp_path / "es.csv", tmp_path / "mes.csv"
    _write(depth, [_depth(side="B"), _depth(side="S")])
    _write(es, [_taq(event_type="TRADE AGRESSOR ON BUY"), _taq(event_type="QUOTE BID", quantity="0", price="4000.00")])
    _write(mes, [_taq(event_type="QUOTE BID", quantity="0", ticker="MESH3")])
    report = algoseek.audit_inputs(es_depth_paths=[depth], es_taq_paths=[es], mes_taq_paths=[mes])
    assert report["status"] == "ALGOSEEK_INPUT_AUDIT_COMPLETE"
    assert report["causal_tie_diagnostics"]["same_timestamp_es_depth_es_trade"] == 1
    assert report["locked_book_states"] == 0 and report["crossed_book_states"] == 0
    assert report["provider_provenance"]["provider"] == "ALGOSEEK"


def test_databento_native_adapter_remains_available() -> None:
    assert databento_native.NativeMBP10Adapter.__name__ == "NativeMBP10Adapter"


def test_streaming_paths_batch_depth_across_chunk_boundary_and_flush_eof(tmp_path: Path) -> None:
    first, second, es, mes = (tmp_path / name for name in ("depth-1.csv", "depth-2.csv", "es.csv", "mes.csv"))
    _write(first, [_depth(side="B")]); _write(second, [_depth(side="S")])
    _write(es, [_taq(event_type="QUOTE BID", quantity="0", price="4000.00")])
    _write(mes, [_taq(event_type="QUOTE BID", ticker="MESH3", quantity="0", price="4000.00")])
    metrics = algoseek.StreamingMetrics()
    events = list(algoseek.iter_canonical_events_from_paths(
        es_depth_paths=[first, second], es_taq_paths=[es], mes_taq_paths=[mes], metrics=metrics,
    ))
    depth = next(event for event in events if event.kind == "ES_DEPTH")
    assert depth.raw_event_count == 2 and depth.snapshot is not None
    assert metrics.max_depth_timestamp_rows == 2 and metrics.canonical_depth_states == 1


def test_streaming_taq_generic_trade_is_profile_only_and_aggressors_remain_canonical(tmp_path: Path) -> None:
    depth, es, mes = (tmp_path / name for name in ("depth.csv", "es.csv", "mes.csv"))
    _write(depth, [_depth(side="B"), _depth(side="S")])
    _write(es, [
        _taq(event_type="TRADE", price="4000.00", quantity="7"),
        _taq(event_type="TRADE AGRESSOR ON BUY", price="4000.25", quantity="3"),
    ])
    _write(mes, [_taq(event_type="QUOTE BID", ticker="MESH3", quantity="0", price="4000.00")])
    profile, metrics = algoseek.StreamingProfile(), algoseek.StreamingMetrics()
    events = list(algoseek.iter_canonical_events_from_paths(
        es_depth_paths=[depth], es_taq_paths=[es], mes_taq_paths=[mes], profile=profile, metrics=metrics,
    ))
    trades = [event for event in events if event.kind == "ES_TRADE"]
    assert len(trades) == 1 and trades[0].execution is not None and trades[0].execution.aggressor == "BUY"
    assert profile.result() == {"POC": 4000.0, "VAH": 4000.0, "VAL": 4000.0, "HIGH": 4000.25, "LOW": 4000.0}
    assert metrics.es_provider_trades == 2 and metrics.generic_trades == 1 and metrics.explicit_aggressor_trades == 1


def test_zero_quantity_generic_trade_is_non_profile_provider_record_not_an_execution() -> None:
    generic = algoseek.parse_taq_row(_taq(event_type="TRADE", quantity="0"), instrument="ES", source_file="es", source_index=1)
    profile = algoseek.StreamingProfile(); profile.observe(generic)
    assert generic.event_kind == "TRADE" and not generic.aggression_eligible
    with pytest.raises(algoseek.AlgoseekAdapterError, match="ES executions"):
        profile.result()


def test_streaming_three_way_same_timestamp_order_is_deterministic(tmp_path: Path) -> None:
    depth, es, mes = (tmp_path / name for name in ("depth.csv", "es.csv", "mes.csv"))
    _write(depth, [_depth(side="S"), _depth(side="B")])
    _write(es, [
        _taq(event_type="QUOTE BID", quantity="0", price="4000.00"),
        _taq(event_type="TRADE AGRESSOR ON SELL", price="4000.00", quantity="2"),
    ])
    _write(mes, [_taq(event_type="QUOTE BID", ticker="MESH3", quantity="0", price="4000.00")])
    inputs = dict(es_depth_paths=[depth], es_taq_paths=[es], mes_taq_paths=[mes])
    first = [(event.kind, event.source_index) for event in algoseek.iter_canonical_events_from_paths(**inputs)]
    second = [(event.kind, event.source_index) for event in algoseek.iter_canonical_events_from_paths(**inputs)]
    assert first == second == [("MES_BBO", 1), ("ES_DEPTH", 2), ("ES_TRADE", 2), ("ES_BBO", 1)]


def test_streaming_rejects_timestamp_regression_across_chunk_boundary(tmp_path: Path) -> None:
    first, second = tmp_path / "one.csv", tmp_path / "two.csv"
    _write(first, [_depth(side="B", timestamp="2023-03-10 09:30:01")])
    _write(second, [_depth(side="S", timestamp="2023-03-10 09:30:00")])
    with pytest.raises(algoseek.AlgoseekAdapterError, match="timestamps decrease"):
        list(algoseek.iter_multiple_depth_paths([first, second]))


def test_streaming_empty_and_malformed_csvs_fail_or_finish_deterministically(tmp_path: Path) -> None:
    empty, malformed = tmp_path / "empty.csv", tmp_path / "malformed.csv"
    empty.write_text(",".join(_depth(side="B")) + "\n", encoding="utf-8")
    malformed.write_text('EventDateTime,Ticker,SecurityID,Side,Flags\n"unterminated', encoding="utf-8")
    assert list(algoseek.iter_multiple_depth(empty)) == []
    with pytest.raises((algoseek.AlgoseekAdapterError, csv.Error)):
        list(algoseek.iter_multiple_depth(malformed))


def test_streaming_compatibility_wrapper_equals_streaming_iterator() -> None:
    depth = [algoseek.parse_multiple_depth_row(_depth(side=side), source_file="depth", source_index=index) for index, side in enumerate(("B", "S"), 1)]
    es = [algoseek.parse_taq_row(_taq(event_type="TRADE AGRESSOR ON BUY"), instrument="ES", source_file="es", source_index=1)]
    mes = [algoseek.parse_taq_row(_taq(event_type="QUOTE BID", ticker="MESH3", quantity="0"), instrument="MES", source_file="mes", source_index=1)]
    assert list(algoseek.canonical_events(es_depth=depth, es_taq=es, mes_taq=mes)) == list(
        algoseek.iter_canonical_events(es_depth=depth, es_taq=es, mes_taq=mes)
    )


def test_projected_and_legacy_full_schema_inputs_are_causally_equivalent(tmp_path: Path) -> None:
    timestamp = "2023-03-10 09:30:00"
    depth_rows = [_depth(side=side, timestamp=timestamp) for side in ("B", "S")]
    es_rows = [
        _taq(event_type="TRADE AGRESSOR ON SELL", timestamp=timestamp, price="4000.00", quantity="4"),
        _taq(event_type="QUOTE BID", timestamp=timestamp, price="3999.75", quantity="0"),
        _taq(event_type="QUOTE SELL", timestamp=timestamp, price="4000.25", quantity="0"),
    ]
    mes_rows = [
        _taq(event_type="QUOTE BID", timestamp=timestamp, price="3999.50", quantity="0", ticker="MESH3"),
        _taq(event_type="QUOTE SELL", timestamp=timestamp, price="4000.50", quantity="0", ticker="MESH3"),
    ]
    full_paths = (tmp_path / "full-depth.csv", tmp_path / "full-es.csv", tmp_path / "full-mes.csv")
    projected_paths = (tmp_path / "projected-depth.csv", tmp_path / "projected-es.csv", tmp_path / "projected-mes.csv")
    for path, rows in zip(full_paths, (depth_rows, es_rows, mes_rows)):
        _write(path, rows)
    for path, rows, columns in zip(
        projected_paths, (depth_rows, es_rows, mes_rows),
        (adapter_api.DEPTH_PROJECTION_COLUMNS, adapter_api.TAQ_PROJECTION_COLUMNS, adapter_api.TAQ_PROJECTION_COLUMNS),
    ):
        projected = [{column: row[column] for column in columns} for row in rows]
        _write(path, projected)

    def run(paths: tuple[Path, Path, Path]) -> tuple[list[tuple[object, ...]], dict[str, float], algoseek.StreamingMetrics]:
        profile, metrics = algoseek.StreamingProfile(), algoseek.StreamingMetrics()
        events = list(algoseek.iter_canonical_events_from_paths(
            es_depth_paths=[paths[0]], es_taq_paths=[paths[1]], mes_taq_paths=[paths[2]],
            profile=profile, metrics=metrics,
        ))

        def semantic(event: algoseek.CanonicalEvent) -> tuple[object, ...]:
            snapshot = None if event.snapshot is None else (
                tuple((level.price, level.size, level.order_count) for level in event.snapshot.bids),
                tuple((level.price, level.size, level.order_count) for level in event.snapshot.asks),
            )
            execution = None if event.execution is None else (
                event.execution.price, event.execution.size, event.execution.aggressor,
            )
            return (event.kind, event.provider_timestamp, event.provider_flags, event.provider_event_type,
                    event.provider_type_mask, snapshot, execution, event.es_taq_bbo, event.mes_bbo,
                    event.book_state, event.raw_event_count)

        return [semantic(event) for event in events], profile.result(), metrics

    full_events, full_profile, full_metrics = run(full_paths)
    projected_events, projected_profile, projected_metrics = run(projected_paths)
    assert projected_events == full_events
    assert projected_profile == full_profile
    assert projected_metrics.canonical_depth_states == full_metrics.canonical_depth_states
    assert projected_metrics.canonical_events_emitted == full_metrics.canonical_events_emitted
    assert [event[6][2] for event in projected_events if event[0] == "ES_TRADE"] == ["SELL"]
    assert any(event[0] == "ES_BBO" and event[7] == (3999.75, 4000.25) for event in projected_events)
    assert any(event[0] == "MES_BBO" and event[8] == (3999.5, 4000.5) for event in projected_events)


def test_streaming_atomic_jsonl_writer_does_not_publish_partial_artifact(tmp_path: Path) -> None:
    output = tmp_path / "session.jsonl"
    event = algoseek.CanonicalEvent(1, "ES_BBO", "es", 1, "ESH3", "1")
    algoseek.write_canonical_jsonl(events=[event], output_path=output, completion_manifest={"status": "COMPLETE"})
    assert output.read_text(encoding="utf-8").count("\n") == 1
    assert output.with_suffix(".jsonl.complete.json").exists()
