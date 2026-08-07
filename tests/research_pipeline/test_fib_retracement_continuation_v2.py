"""Synthetic-only contract tests for the sealed Fib retracement V2 helpers."""
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json

import pytest

from research_pipeline.fib_retracement_continuation_v2.aggregation import completed_utc_bars, validate_1m_bars
from research_pipeline.fib_retracement_continuation_v2.execution import execute_order, process_position, submit_order
from research_pipeline.fib_retracement_continuation_v2.ids import exit_leg_id, fib_range_id, impulse_id, order_id, setup_id, trade_id
from research_pipeline.fib_retracement_continuation_v2.manifests import EXPECTED_SCHEMA, ManifestError, canonical, canonical_manifest_hash, verify_manifest
from research_pipeline.fib_retracement_continuation_v2.models import Bar, Candidate, ExecutionAssumptions
from research_pipeline.fib_retracement_continuation_v2.reconciliation import reconcile
from research_pipeline.fib_retracement_continuation_v2.runner import _at_or_after_cutoff, _close, materialize_synthetic, run_candidate, run_holdout
from research_pipeline.fib_retracement_continuation_v2.strategy import expire_reason, fib_price, touch
from research_pipeline.fib_retracement_continuation_v2.parity import fib09_v2_v1_parity_diagnostic
from research_pipeline.fib_retracement_continuation_v1.strategy import causal_setups as v1_causal_setups


UTC = timezone.utc
START = datetime(2024, 1, 1, tzinfo=UTC)
ASSUMPTIONS = ExecutionAssumptions(fee_rate=Decimal(".001"), slippage_rate=Decimal(".001"), quantity_step=Decimal(".001"))
CANDIDATE = Candidate("SYNTH", "ETH", "4h", Decimal(".830"), 1, Decimal(".001"), 1)


def bar(stamp, open_="100", high=None, low=None, close=None, volume="1"):
    open_ = Decimal(str(open_)); high = Decimal(str(high if high is not None else open_)); low = Decimal(str(low if low is not None else open_)); close = Decimal(str(close if close is not None else open_))
    return Bar(stamp, open_, high, low, close, Decimal(str(volume)))


def reseal(data):
    data["manifestHash"] = canonical_manifest_hash(data)
    return data


def synthetic_manifest(tmp_path):
    parts, first = [], datetime(2024, 1, 1, tzinfo=UTC)
    for number in range(61):
        stamp = first + timedelta(minutes=number)
        digest = f"{number + 1:064x}"
        parts.append({"symbol": "ETHUSDT", "exchange": "BINANCE", "instrumentType": "USD_M_PERPETUAL", "interval": "1m", "sealed": True, "missingMinuteDistribution": [], "partitionKind": "DAY", "partition": stamp.date().isoformat(), "firstUtcTimestamp": stamp.isoformat().replace("+00:00", "Z"), "finalUtcTimestamp": stamp.isoformat().replace("+00:00", "Z"), "rowCount": 1, "sha256": digest, "path": f"binance_usdm/ETHUSDT/1m/{stamp.date().isoformat()}/{digest}", "file": "source.json", "byteSize": 1})
    identity = [{"path": item["path"], "sha256": item["sha256"]} for item in parts]
    data = {"symbol": "ETHUSDT", "immutable": True, "exchange": "BINANCE", "instrumentType": "USD_M_PERPETUAL", "interval": "1m", "schema": EXPECTED_SCHEMA, "schemaHash": hashlib.sha256(canonical(EXPECTED_SCHEMA).encode()).hexdigest(), "chronology": {"development": "[2022-01-01T00:00:00Z, 2025-01-01T00:00:00Z)", "holdout": "[2025-01-01T00:00:00Z, coverage end)", "holdoutStrategyAccess": False}, "integrity": {"duplicateOrNonIncreasingCount": 0, "invalidOhlcCount": 0, "nonPositivePriceCount": 0, "nullOhlcvCount": 0, "unresolvedGapCount": 0}, "partitions": parts, "aggregateDatasetHash": hashlib.sha256(canonical(identity).encode()).hexdigest(), "rowCount": 61, "coverage": {"startInclusive": parts[0]["firstUtcTimestamp"], "endExclusive": (first + timedelta(minutes=61)).isoformat().replace("+00:00", "Z")}}
    path = tmp_path / "manifest.json"; path.write_text(json.dumps(reseal(data)))
    return path, data


def write_manifest(tmp_path, data):
    path = tmp_path / "manifest.json"; path.write_text(json.dumps(reseal(data)))
    return path


def setup(direction="LONG", stamp=START):
    anchor_price = Decimal("90") if direction == "LONG" else Decimal("110")
    extreme_price = Decimal("110") if direction == "LONG" else Decimal("90")
    sid = setup_id("synthetic", direction, stamp, anchor_price, stamp, extreme_price)
    iid = impulse_id(sid, stamp, extreme_price)
    return {"setup_id": sid, "impulse_id": iid, "fib_range_id": fib_range_id(iid, Decimal("90"), Decimal("110")), "direction": direction, "low": Decimal("90"), "high": Decimal("110"), "anchor_timestamp": stamp, "extreme_timestamp": stamp, "active_timestamp": None, "version": 0}


def filled_trade(direction="LONG"):
    order = submit_order(setup(direction), START)
    trade, rejected = execute_order(order, bar(START, "100", "111", "89"), CANDIDATE, Decimal("10000"), ASSUMPTIONS)
    assert rejected is None and trade is not None
    return trade


def test_manifest_synthetic_contract_validates_without_payloads(tmp_path):
    path, _ = synthetic_manifest(tmp_path)
    result = verify_manifest(path.resolve())
    assert result["rows_read"] is False and result["partition_archives_verified"] is False


@pytest.mark.parametrize("mutate, code", [
    (lambda data: data.update(manifestHash="0" * 64), "SELF_HASH"),
    (lambda data: data.update(aggregateDatasetHash="0" * 64), "DATASET_HASH"),
    (lambda data: data["partitions"][0].update(sha256="not-a-hash"), "PARTITION_ROW_OR_HASH"),
    (lambda data: data["partitions"][0].update(path="../escape"), "PARTITION_PATH"),
    (lambda data: data.update(schema=[]), "SCHEMA"),
])
def test_manifest_contract_failures_are_closed(tmp_path, mutate, code):
    _, data = synthetic_manifest(tmp_path); mutate(data)
    path = tmp_path / "bad.json"; path.write_text(json.dumps(data if code == "SELF_HASH" else reseal(data)))
    with pytest.raises(ManifestError, match=code): verify_manifest(path.resolve())


def test_manifest_partition_cadence_failure_is_closed(tmp_path):
    _, data = synthetic_manifest(tmp_path); data["partitions"][1]["firstUtcTimestamp"] = data["partitions"][0]["firstUtcTimestamp"]
    with pytest.raises(ManifestError, match="ORDER_OR_GAP"): verify_manifest(write_manifest(tmp_path, data).resolve())


@pytest.mark.parametrize("minutes", [240, 1440])
def test_incomplete_htf_bucket_is_never_derived(minutes):
    rows = [bar(START + timedelta(minutes=index)) for index in range(minutes - 1)]
    assert completed_utc_bars(rows, minutes) == []


def test_complete_bucket_ohlcv_is_correct_and_next_minute_only():
    rows = [bar(START + timedelta(minutes=index), 100 + index, 102 + index, 99 + index, 101 + index, 2) for index in range(241)]
    aggregate = completed_utc_bars(rows, 240)
    assert len(aggregate) == 1
    value = aggregate[0]
    assert (value.open, value.high, value.low, value.close, value.volume) == (Decimal("100"), Decimal("341"), Decimal("99"), Decimal("340"), Decimal("480"))
    assert rows[240].timestamp == value.timestamp + timedelta(minutes=240)


@pytest.mark.parametrize("stamp", [START.replace(second=1), START.replace(microsecond=1)])
def test_minute_subprecision_fails(stamp):
    with pytest.raises(ValueError, match="CADENCE"): validate_1m_bars([bar(stamp)])


def test_duplicate_and_invalid_ohlc_fail():
    with pytest.raises(ValueError, match="MISSING_OR_DUPLICATE"): validate_1m_bars([bar(START), bar(START)])
    with pytest.raises(ValueError, match="INVALID_OHLC"): validate_1m_bars([bar(START, 100, 99, 101, 100)])


@pytest.mark.parametrize("direction, high, low", [("LONG", "111", "89"), ("SHORT", "111", "89")])
def test_long_and_short_limit_fills_are_adverse(direction, high, low):
    order = submit_order(setup(direction), START)
    trade, rejected = execute_order(order, bar(START, 100, high, low), CANDIDATE, Decimal("10000"), ASSUMPTIONS)
    raw = fib_price(direction, Decimal("90"), Decimal("110"), Decimal(".900"))
    assert rejected is None and trade["entry_price"] != raw and trade["quantity"] > 0


def test_tp1_is_partial_and_moved_stop_waits_until_next_minute():
    trade = filled_trade(); legs = process_position(trade, bar(START + timedelta(minutes=1), 100, 95, 94), CANDIDATE, ASSUMPTIONS)
    assert legs[0]["reason"] == "TP1" and trade["remaining_quantity"] < trade["quantity"]
    assert trade["current_stop"] == trade["initial_stop_price"]
    process_position(trade, bar(START + timedelta(minutes=2), 100, 95, 94), CANDIDATE, ASSUMPTIONS)
    assert trade["current_stop"] == fib_price("LONG", Decimal("90"), Decimal("110"), Decimal(".830"))


@pytest.mark.parametrize("ratio", [Decimal(".830"), Decimal(".786")])
def test_post_tp1_stop_contracts_are_applied(ratio):
    candidate = Candidate("SYNTH", "ETH", "4h", ratio, 1, Decimal(".001"), 1); trade = filled_trade()
    process_position(trade, bar(START + timedelta(minutes=1), 100, 95, 94), candidate, ASSUMPTIONS)
    process_position(trade, bar(START + timedelta(minutes=2), 100, 95, 94), candidate, ASSUMPTIONS)
    assert trade["current_stop"] == fib_price("LONG", Decimal("90"), Decimal("110"), ratio)


def test_stop_has_precedence_over_targets_inside_one_bar():
    trade = filled_trade(); legs = process_position(trade, bar(START + timedelta(minutes=1), 100, 110, 89), CANDIDATE, ASSUMPTIONS)
    assert len(legs) == 1 and legs[0]["reason"] == "STOP" and trade["remaining_quantity"] == 0


def test_force_exit_is_open_at_2245_after_2244_is_active():
    trade = filled_trade(); trades, events = [], []
    closed = _close(trade, bar(START.replace(hour=22, minute=45), 100), ASSUMPTIONS, trades, events)
    assert closed["legs"][-1]["raw_price"] == Decimal("100") and events[0]["timestamp"].time().isoformat() == "22:45:00"


def test_force_exit_has_adverse_slippage_and_fee():
    trade = filled_trade(); closed = _close(trade, bar(START.replace(hour=22, minute=45), 100), ASSUMPTIONS, [], [])
    leg = closed["legs"][-1]
    assert leg["fill_price"] < leg["raw_price"] and leg["fee"] > 0 and leg["slippage_cost"] > 0


def test_force_exit_closes_tp1_remainder():
    trade = filled_trade(); process_position(trade, bar(START + timedelta(minutes=1), 100, 110, 100), CANDIDATE, ASSUMPTIONS)
    remaining = trade["remaining_quantity"]; closed = _close(trade, bar(START.replace(hour=22, minute=45), 100), ASSUMPTIONS, [], [])
    assert closed["legs"][-1]["quantity"] == remaining and closed["remaining_quantity"] == 0


def test_pending_expiry_and_cutoff_predicates():
    pending = setup("LONG", START - timedelta(days=2))
    assert expire_reason(pending, bar(START), CANDIDATE) == "ENTRY_EXPIRED"
    assert _at_or_after_cutoff(START.replace(hour=22, minute=45)) and not _at_or_after_cutoff(START.replace(hour=22, minute=44))


def test_midnight_is_eligible_again_and_no_overnight_is_imposed():
    assert not _at_or_after_cutoff(START.replace(hour=0, minute=0))
    trade = filled_trade(); closed = _close(trade, bar(START.replace(hour=22, minute=45), 100), ASSUMPTIONS, [], [])
    assert closed["legs"][-1]["timestamp"].date() == trade["entry_timestamp"].date()


def test_quantity_pnl_equity_reconcile_for_closed_trade():
    trade = filled_trade(); process_position(trade, bar(START + timedelta(minutes=1), 100, 110, 100), CANDIDATE, ASSUMPTIONS)
    closed = _close(trade, bar(START.replace(hour=22, minute=45), 100), ASSUMPTIONS, [], [])
    setups = [{"setup_id": closed["setup_id"]}]; outcomes = [{"setup_id": closed["setup_id"], "disposition": "TRADE_EXECUTED"}]; orders = [{"setup_id": closed["setup_id"], "order_id": closed["order_id"]}]
    assert reconcile(setups, outcomes, orders, [closed], ASSUMPTIONS.opening_equity, final_equity=ASSUMPTIONS.opening_equity + closed["net_pnl"])["reconciles"]


def test_deterministic_ids_and_duplicate_prevention():
    sid = setup_id("c", "LONG", START, Decimal("90"), START + timedelta(hours=4), Decimal("110"))
    fid = fib_range_id(impulse_id(sid, START + timedelta(hours=4), Decimal("110")), Decimal("90"), Decimal("110"))
    oid = order_id(fid, 1, START); tid = trade_id(oid, START)
    assert sid == setup_id("c", "LONG", START, Decimal("90"), START + timedelta(hours=4), Decimal("110"))
    assert exit_leg_id(tid, 1) != exit_leg_id(tid, 2)
    assert not reconcile([{"setup_id": sid}, {"setup_id": sid}], [{"setup_id": sid}], [], [], Decimal("1"))["reconciles"]


def test_holdout_is_fail_closed_without_access():
    with pytest.raises(ManifestError, match="LOCKED_HOLDOUT_NOT_AUTHORIZED"): run_holdout()


def test_immutable_synthetic_artifact_collision(tmp_path):
    root = tmp_path / "artifacts"; materialize_synthetic(artifact_root=root, repository_root=tmp_path)
    with pytest.raises(FileExistsError, match="IMMUTABLE_ARTIFACT_ROOT_COLLISION"): materialize_synthetic(artifact_root=root, repository_root=tmp_path)


def test_empty_synthetic_run_reconciles_without_data_reads():
    result = run_candidate([], CANDIDATE)
    assert result["reconciliation"]["reconciles"] and result["holdout_status"] == "LOCKED_NOT_OPENED"


@pytest.mark.parametrize("direction, ratio, expected", [("LONG", Decimal(".5"), Decimal("100")), ("SHORT", Decimal(".5"), Decimal("100"))])
def test_fib_prices_and_touch_contract(direction, ratio, expected):
    assert fib_price(direction, Decimal("90"), Decimal("110"), ratio) == expected
    assert touch(direction, bar(START, 100, 110, 90), expected)


def test_identical_htf_signal_setups_are_exact_v1_and_no_1m_inflation():
    candidate = Candidate("PARITY", "ETH", "4h", Decimal(".830"), 1, Decimal(".001"), 10)
    rows = [bar(START + timedelta(minutes=i), 100, 101 if i < 240 else 120, 99, 100) for i in range(721)]
    reference = completed_utc_bars(rows, 240)
    report = fib09_v2_v1_parity_diagnostic(reference_bars_by_candidate={"PARITY": reference}, derived_1m_by_candidate={"PARITY": rows}, candidates=[candidate])
    assert report["implementation_difference_count"] == 0
    assert report["candidates"][0]["derived_setup_count"] == len(v1_causal_setups(reference, candidate))


def test_v1_execution_ids_prices_partials_and_sizing_are_reused():
    order = submit_order(setup(), START)
    trade, rejected = execute_order(order, bar(START, 100, 111, 89), CANDIDATE, Decimal("10000"), ASSUMPTIONS)
    assert rejected is None and trade
    assert trade["order_id"] == order_id(order["fib_range_id"], 1, START)
    leg = process_position(trade, bar(START + timedelta(minutes=1), 100, 110, 100), CANDIDATE, ASSUMPTIONS)[0]
    assert leg["reason"] == "TP1" and leg["quantity"] == trade["quantity"] * Decimal(".30")


def test_reconciliation_fails_injected_overnight_cutoff_and_pending_defects():
    trade = filled_trade()
    closed = _close(trade, bar(START.replace(hour=22, minute=45), 100), ASSUMPTIONS, [], [])
    closed["legs"][-1]["timestamp"] = START + timedelta(days=1)
    result = reconcile([{"setup_id": closed["setup_id"]}], [{"setup_id": closed["setup_id"]}], [{"setup_id": closed["setup_id"], "order_id": closed["order_id"]}], [closed], ASSUMPTIONS.opening_equity, final_equity=ASSUMPTIONS.opening_equity + closed["net_pnl"], pending_after_cutoff=[{"order_id": "pending"}])
    assert not result["reconciles"] and result["overnight_trade_count"] == 1
