from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from research_pipeline.imbalance_vwap_ride.adapter import ImbalanceVWAPRideAdapter
from research_pipeline.imbalance_vwap_ride.alpha_proxy import (
    block_bootstrap,
    evaluation_outcome,
    is_mbt_entry_available,
    map_btc_trades_to_mbt,
    qualified_outcome,
    run_alpha_proxy,
)
from research_pipeline.imbalance_vwap_ride.artifacts import ArtifactContext, sha256_file, sha256_value, write_bytes_once
from research_pipeline.imbalance_vwap_ride.footprint import (
    aggregate_trade_batches,
    build_footprint_dataset,
    validate_footprint_dataset,
)
from research_pipeline.imbalance_vwap_ride.models import (
    ADAPTER_ID,
    BASELINE,
    DATASET_HASH,
    EVIDENCE_LABEL,
    ImbalanceVWAPRideConfig,
    development_gate,
    freeze_development_candidates,
    locked_test_gate,
    preregistered_variants,
    select_validation_candidate,
    validation_gate,
)
from research_pipeline.imbalance_vwap_ride.strategy import (
    compute_completed_bar_regimes,
    maximal_imbalance_sequences,
    run_imbalance_vwap_ride,
)

UTC = timezone.utc


def _raw_batch(rows: list[dict]) -> pa.RecordBatch:
    return pa.RecordBatch.from_pylist(rows)


def _raw(timestamp: str, aggregate_id: int, price: str, quantity: str, maker: bool) -> dict:
    return {
        "event_time_utc": timestamp,
        "aggregate_trade_id": aggregate_id,
        "price": price,
        "quantity_base": quantity,
        "notional_quote": str(Decimal(price) * Decimal(quantity)),
        "buyer_is_maker": maker,
    }


def test_batch_aggregation_is_exact_half_open_and_preserves_boundaries() -> None:
    batches = [
        _raw_batch(
            [
                _raw("2024-01-01T00:04:59.000000Z", 1, "99.9", "1.25", False),
                _raw("2024-01-01T00:04:59.500000Z", 2, "100.0", "2.50", True),
            ]
        ),
        _raw_batch(
            [
                _raw("2024-01-01T00:04:59.900000Z", 3, "109.9", "0.75", False),
                _raw("2024-01-01T00:05:00.000000Z", 4, "110.0", "3.00", True),
            ]
        ),
    ]
    footprints, bars, diagnostics = aggregate_trade_batches(
        batches, month="2024-01", expected_first_id=1, expected_last_id=4
    )
    first_bar = [row for row in footprints if row["bar_start_utc"].minute == 0]
    assert [(row["bin_floor"], row["bin_upper_exclusive"]) for row in first_bar] == [
        (Decimal("90"), Decimal("100")),
        (Decimal("100"), Decimal("110")),
    ]
    hundred = first_bar[1]
    assert hundred["buy_volume_btc"] == Decimal("0.75")
    assert hundred["sell_volume_btc"] == Decimal("2.50")
    assert hundred["total_volume_btc"] == hundred["buy_volume_btc"] + hundred["sell_volume_btc"]
    assert hundred["delta_btc"] == hundred["buy_volume_btc"] - hundred["sell_volume_btc"]
    assert len(bars) == 2
    assert bars[0]["open"] == Decimal("99.9")
    assert bars[0]["close"] == Decimal("109.9")
    assert bars[0]["trade_count"] == 3
    assert len(diagnostics) == 2
    assert diagnostics[0]["last_aggregate_trade_id"] + 1 == diagnostics[1]["first_aggregate_trade_id"]


def test_batch_gap_fails_closed() -> None:
    batches = [
        _raw_batch([_raw("2024-01-01T00:00:00.000000Z", 1, "100", "1", False)]),
        _raw_batch([_raw("2024-01-01T00:00:01.000000Z", 3, "100", "1", False)]),
    ]
    with pytest.raises(ValueError, match="batch boundary gap"):
        aggregate_trade_batches(batches, month="2024-01")


def test_content_addressed_footprint_build_and_validation(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    rows = [
        _raw("2024-01-01T00:00:00.000000Z", 1, "99.9", "1.25", False),
        _raw("2024-01-01T00:01:00.000000Z", 2, "100.0", "2.50", True),
        _raw("2024-01-01T00:05:00.000000Z", 3, "109.9", "0.75", False),
        _raw("2024-01-01T00:06:00.000000Z", 4, "110.0", "3.00", True),
    ]
    parquet = source / "2024-01.parquet"
    pq.write_table(pa.Table.from_pylist(rows), parquet)
    manifest = {
        "date_start": "2024-01-01",
        "date_end": "2024-01-31",
        "duplicate_count": 0,
        "normalized_dataset_hash": "synthetic-hash",
        "row_count": 4,
        "symbol": "BTCUSDT",
        "retrieved_at": "2024-02-01T00:00:00Z",
        "partitions": [
            {
                "month": "2024-01",
                "file_name": parquet.name,
                "row_count": 4,
                "parquet_hash": sha256_file(parquet),
                "first_aggregate_trade_id": 1,
                "last_aggregate_trade_id": 4,
                "first_timestamp": rows[0]["event_time_utc"],
                "last_timestamp": rows[-1]["event_time_utc"],
            }
        ],
    }
    manifest["manifest_hash"] = sha256_value(manifest)
    manifest_path = source / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    built = build_footprint_dataset(
        manifest_path,
        tmp_path / "cache",
        batch_size=2,
        require_pinned=False,
        verify_source_hashes=True,
    )
    assert built["streamed_trade_count"] == 4
    assert built["batch_count"] == 2
    validation = validate_footprint_dataset(built["footprint_root"])
    assert validation["valid"] is True
    assert validation["five_minute_bar_count"] == 2
    metadata = pq.read_metadata(Path(built["footprint_root"]) / "bars" / "2024-01.parquet").metadata
    assert metadata[b"dataset_hash"] == b"synthetic-hash"


def _bar(minute: int, *, close: str, low: str | None = None, high: str | None = None, vwap_price: str | None = None) -> dict:
    start = datetime(2024, 1, 2, 0, minute, tzinfo=UTC)
    price = Decimal(vwap_price or close)
    return {
        "bar_start_utc": start,
        "bar_end_utc": start + timedelta(minutes=5),
        "session_date": start.date().isoformat(),
        "month": "2024-01",
        "open": Decimal(close),
        "high": Decimal(high or close),
        "low": Decimal(low or close),
        "close": Decimal(close),
        "volume": Decimal("1"),
        "notional": price,
        "trade_count": 1,
    }


def test_daily_vwap_is_completed_bar_only_and_resets() -> None:
    initial = [_bar(0, close="100"), _bar(5, close="110", vwap_price="102")]
    before = compute_completed_bar_regimes(initial, 1)
    after = compute_completed_bar_regimes(initial + [_bar(10, close="50", vwap_price="50")], 1)
    assert before[1]["daily_vwap"] == after[1]["daily_vwap"] == Decimal("101")
    assert before[1]["long_vwap_regime"] is True
    next_day = dict(_bar(15, close="80"))
    next_day["bar_start_utc"] = datetime(2024, 1, 3, tzinfo=UTC)
    next_day["bar_end_utc"] = datetime(2024, 1, 3, 0, 5, tzinfo=UTC)
    next_day["session_date"] = "2024-01-03"
    result = compute_completed_bar_regimes(initial + [next_day], 1)
    assert result[-1]["daily_vwap"] == Decimal("80")
    assert result[-1]["long_vwap_regime"] is False


def _fp(minute: int, floor: str, buy: str, sell: str) -> dict:
    start = datetime(2024, 1, 2, 0, minute, tzinfo=UTC)
    return {
        "bar_start_utc": start,
        "bin_floor": Decimal(floor),
        "bin_upper_exclusive": Decimal(floor) + Decimal("10"),
        "buy_volume_btc": Decimal(buy),
        "sell_volume_btc": Decimal(sell),
        "trade_count": 1,
    }


def test_zone_formation_uses_maximal_non_overlapping_sequences() -> None:
    rows = [
        _fp(5, "100", "6", "1"),
        _fp(5, "110", "6", "1"),
        _fp(5, "120", "6", "1"),
        _fp(5, "130", "1", "6"),
        _fp(5, "140", "6", "1"),
        _fp(5, "150", "6", "1"),
        _fp(5, "160", "6", "1"),
    ]
    config = ImbalanceVWAPRideConfig(stacked_bins=3)
    sequences = maximal_imbalance_sequences(rows, config, "LONG")
    assert [(Decimal(item["bottom"]), Decimal(item["top"]), item["bin_count"]) for item in sequences] == [
        (Decimal("100"), Decimal("130"), 3),
        (Decimal("140"), Decimal("170"), 3),
    ]
    assert len({item["sequence_id"] for item in sequences}) == 2


def _lifecycle_fixture(*, ambiguity: bool = False, include_entry: bool = True) -> tuple[list[dict], list[dict], ImbalanceVWAPRideConfig]:
    bars = [
        _bar(0, close="100", low="99", high="101", vwap_price="100"),
        _bar(5, close="121", low="119", high="122", vwap_price="102"),
        _bar(10, close="130", low="121", high="132", vwap_price="104"),
        _bar(15, close="121", low="119", high="125", vwap_price="106"),
    ]
    if include_entry:
        bars.append(_bar(20, close="140", low="99" if ambiguity else "121", high="250", vwap_price="108"))
    footprints = [_fp(5, "100", "6", "1"), _fp(5, "110", "7", "1")]
    config = ImbalanceVWAPRideConfig(
        stacked_bins=2,
        vwap_slope_bars=1,
        move_away_bars=1,
        zone_expiry_bars=10,
        stop_buffer_bins=0,
        target_r_multiple=Decimal("1"),
    )
    return bars, footprints, config


def test_lifecycle_arms_retests_and_enters_only_next_bar() -> None:
    bars, footprints, config = _lifecycle_fixture()
    result = run_imbalance_vwap_ride(bars, footprints, config)
    assert result["metrics"]["imbalance_sequences"] == 1
    assert result["metrics"]["zones_created"] == 1
    assert result["metrics"]["vwap_qualified_zones"] == 1
    assert result["metrics"]["move_away_confirmed_zones"] == 1
    assert result["metrics"]["retest_triggers"] == 1
    assert len(result["trades"]) == 1
    trade = result["trades"][0]
    assert trade["entry_timestamp"] == bars[4]["bar_start_utc"].isoformat()
    assert datetime.fromisoformat(trade["entry_timestamp"]) > datetime.fromisoformat(trade["signal_bar_start_timestamp"])
    assert trade["entry_timestamp"] == trade["signal_timestamp"]
    assert result["funnel"]["reconciles"] is True
    repeated = run_imbalance_vwap_ride(bars, footprints, config)
    assert repeated["trades"][0]["trade_id"] == trade["trade_id"]
    assert repeated["zones"][0]["zone_id"] == result["zones"][0]["zone_id"]


def test_same_bar_stop_and_target_is_stop_first() -> None:
    bars, footprints, config = _lifecycle_fixture(ambiguity=True)
    result = run_imbalance_vwap_ride(bars, footprints, config)
    assert result["trades"][0]["exit_reason"] == "STOP_FIRST_AMBIGUITY"
    assert result["trades"][0]["same_bar_ambiguity"] is True


def test_missing_next_bar_is_non_executable_and_funnel_reconciles() -> None:
    bars, footprints, config = _lifecycle_fixture(include_entry=False)
    result = run_imbalance_vwap_ride(bars, footprints, config)
    assert result["trades"] == []
    assert result["funnel"]["non_executable_setups"] == 1
    assert result["funnel"]["reconciles"] is True


def _metrics(*, trades: int = 40, pf: str = "1.1", avg_r: str = ".1", pnl: str = "10", dd: str = "5", long: int = 20, short: int = 20) -> dict:
    return {
        "executed_trades": trades,
        "profit_factor": pf,
        "average_net_r": avg_r,
        "net_pnl": pnl,
        "maximum_drawdown": dd,
        "maximum_positive_month_contribution": ".50",
        "best_five_positive_pnl_contribution": ".60",
        "long_trades": long,
        "short_trades": short,
        "funnel_reconciliation": {"reconciles": True},
    }


def test_registry_is_exact_unique_one_factor_and_gates_are_strict() -> None:
    variants = preregistered_variants()
    assert len(variants) == 17
    assert variants[0].variant_id == "baseline"
    assert len({json.dumps(item.parameter_payload(), sort_keys=True) for item in variants}) == 17
    assert development_gate(_metrics())["passed"] is True
    assert development_gate(_metrics(trades=39))["passed"] is False
    assert development_gate(_metrics(pf="1.05"))["passed"] is False
    assert validation_gate(_metrics(trades=15, pf="1.0001"))["passed"] is True
    assert validation_gate(_metrics(trades=14))["passed"] is False
    assert locked_test_gate(_metrics(trades=8, pf="1.0001"))["passed"] is True
    assert locked_test_gate(_metrics(trades=7))["passed"] is False


def test_freeze_and_validation_selection_follow_registered_order_and_tiebreakers() -> None:
    baseline = BASELINE
    other = preregistered_variants()[1]
    frozen = freeze_development_candidates([(other, _metrics()), (baseline, _metrics())])
    assert [item[0].variant_id for item in frozen][:2] == ["baseline", other.variant_id]
    selected = select_validation_candidate(
        [
            (baseline, _metrics(trades=20, pf="1.2", dd="5")),
            (other, _metrics(trades=20, pf="1.3", dd="10")),
        ]
    )
    assert selected is not None and selected[0].variant_id == other.variant_id


def test_adapter_is_standalone_local_only() -> None:
    adapter = ImbalanceVWAPRideAdapter()
    assert adapter.adapter_id == ADAPTER_ID
    assert adapter.dataset_hash == DATASET_HASH
    assert adapter.capabilities()["live_orders"] is False
    assert adapter.capabilities()["external_raw_trade_transmission"] is False


def test_artifacts_embed_provenance_and_collisions_fail(tmp_path: Path) -> None:
    context = ArtifactContext("run", "dataset", "manifest", "spec", "params", "code", EVIDENCE_LABEL, "2024-01-01T00:00:00Z")
    path = context.write_json(tmp_path / "report.json", {"status": "ok"})
    payload = json.loads(path.read_text())
    assert payload["study_run_id"] == "run"
    assert payload["parameter_hash"] == "params"
    assert payload["evidence_label"] == EVIDENCE_LABEL
    context.write_json(path, {"status": "ok"})
    with pytest.raises(ValueError, match="collision"):
        context.write_json(path, {"status": "changed"})
    write_bytes_once(tmp_path / "same.bin", b"same")
    with pytest.raises(ValueError, match="collision"):
        write_bytes_once(tmp_path / "same.bin", b"different")


def test_alpha_availability_is_dst_correct_and_weekends_are_excluded() -> None:
    assert is_mbt_entry_available("2024-03-10T21:30:00Z") is False  # 17:30 EDT Sunday
    assert is_mbt_entry_available("2024-03-10T22:30:00Z") is True   # 18:30 EDT Sunday
    assert is_mbt_entry_available("2024-03-09T15:00:00Z") is False


def _proxy_trade() -> dict:
    return {
        "trade_id": "t1",
        "direction": "LONG",
        "entry_timestamp": "2024-07-01T14:00:00+00:00",
        "exit_timestamp": "2024-07-01T14:30:00+00:00",
        "reference_entry_price": "60001",
        "entry_price": "60001.1",
        "reference_exit_price": "62001",
        "exit_price": "62001",
    }


def test_mbt_mapping_rounds_to_five_points_and_applies_conservative_costs() -> None:
    mapped = map_btc_trades_to_mbt([_proxy_trade()], [])
    trade = mapped["trades"][0]
    assert Decimal(trade["entry_points"]) % 5 == 0
    assert Decimal(trade["exit_points"]) % 5 == 0
    assert Decimal(trade["commission_usd"]) == Decimal("2.50")
    assert Decimal(trade["slippage_usd"]) > 0


def test_monte_carlo_and_qualified_rules_are_deterministic() -> None:
    first = block_bootstrap([100.0, -50.0, 200.0], paths=8, path_days=12, block_days=2, seed=7)
    second = block_bootstrap([100.0, -50.0, 200.0], paths=8, path_days=12, block_days=2, seed=7)
    assert (first == second).all()
    assert evaluation_outcome([300.0] * 5)["outcome"] == "PASS"
    assert evaluation_outcome([-501.0])["breach_reason"] == "DAILY_LOSS_GUARD"
    qualified = qualified_outcome([250.0] * 12)
    assert qualified["second_payout_achieved"] is True
    assert len(qualified["withdrawals"]) == 2


def test_alpha_proxy_enforces_10000_paths_and_reports_sample_warning() -> None:
    with pytest.raises(ValueError, match="10,000"):
        run_alpha_proxy([_proxy_trade()], [], paths=9_999)
    result = run_alpha_proxy([_proxy_trade()], [], paths=10_000, seed=9)
    assert result["evaluation"]["paths"] == 10_000
    assert result["evaluation"]["pass_probability_insufficient_sample"] is True
    assert set(result["sensitivities"]) == {
        "one_extra_tick_each_side",
        "double_commission",
        "degradation_20_percent",
        "degradation_30_percent",
        "degradation_40_percent",
    }
