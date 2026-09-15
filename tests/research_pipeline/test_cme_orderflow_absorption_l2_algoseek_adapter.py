from __future__ import annotations

import csv
from pathlib import Path

import pytest

from research_pipeline.cme_orderflow_absorption_l2_v1 import algoseek_adapter as algoseek
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


def test_contracts_remain_separate_in_rows() -> None:
    march = algoseek.parse_taq_row(_taq(event_type="TRADE AGRESSOR ON BUY", ticker="ESH3", security="101"), instrument="ES", source_file="a", source_index=1)
    june = algoseek.parse_taq_row(_taq(event_type="TRADE AGRESSOR ON BUY", ticker="ESM3", security="202"), instrument="ES", source_file="b", source_index=1)
    assert (march.ticker, march.security_id) != (june.ticker, june.security_id)
    depth = algoseek.parse_multiple_depth_row(_depth(side="B", ticker="ESH3", security="101"), source_file="depth", source_index=1)
    mes = algoseek.parse_taq_row(_taq(event_type="QUOTE BID", ticker="MESH3", security="9", quantity="0"), instrument="MES", source_file="mes", source_index=1)
    with pytest.raises(algoseek.AlgoseekAdapterError, match="multiple contract"):
        list(algoseek.canonical_events(es_depth=[depth], es_taq=[march, june], mes_taq=[mes]))


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
    assert [event.kind for event in events] == ["MES_BBO", "ES_DEPTH", "ES_DEPTH", "ES_TRADE", "ES_BBO"]
    assert events[3].execution is not None and events[3].snapshot is not None
    assert events[3].snapshot.__class__.__name__ == "MBP10Snapshot"
    assert events[3].execution.__class__.__name__ == "Execution"


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
    assert provenance["ordering_policy_id"] == "algoseek-causal-order-v1"
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
    assert report["provider_provenance"]["provider"] == "ALGOSEEK"


def test_databento_native_adapter_remains_available() -> None:
    assert databento_native.NativeMBP10Adapter.__name__ == "NativeMBP10Adapter"
