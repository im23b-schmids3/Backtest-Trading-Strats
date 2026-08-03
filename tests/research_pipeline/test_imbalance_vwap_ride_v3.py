from __future__ import annotations

import json
import zipfile
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from research_pipeline.cli import _parser
from research_pipeline.imbalance_vwap_ride.v3_data import (
    _aggregate_month,
    _vectorized_month_partition,
    authorized_archive_url,
    validate_authorized_archive,
)
from research_pipeline.value_area_trap.data import AggregateTradeImporter
from research_pipeline.imbalance_vwap_ride.v3_models import (
    ADAPTER_ID,
    AUTHORIZED_MONTHS,
    BASELINE,
    EVIDENCE_LABEL,
    PARAMETER_REGISTRY,
    PERIOD_LABEL,
    SELECTION_METHOD,
    STRATEGY_ID,
    ImbalanceVWAPRideV3Config,
    freeze_passing_candidates,
    preregistered_variants,
    promotion_gate,
)
from research_pipeline.imbalance_vwap_ride.v3_runner import (
    JAN_JUL_SOURCE_FILE_COUNT,
    JAN_JUL_SOURCE_TREE_DIGEST,
    V1_FILE_COUNT,
    V1_TREE_DIGEST,
    V2_FILE_COUNT,
    V2_TREE_DIGEST,
    V3ArtifactContext,
    _existing_final,
    preservation_snapshot,
)
from research_pipeline.imbalance_vwap_ride.v3_strategy import (
    ACTIVE_STATES,
    run_imbalance_vwap_ride_v3,
    simulate_long_trade,
)

UTC = timezone.utc


def _bar(
    minute: int,
    *,
    close: str,
    open_: str | None = None,
    low: str | None = None,
    high: str | None = None,
    month: str = "2024-08",
) -> dict:
    start = datetime(2024, 8, 2, 0, minute, tzinfo=UTC)
    price = Decimal(close)
    return {
        "bar_start_utc": start,
        "bar_end_utc": start + timedelta(minutes=5),
        "session_date": start.date().isoformat(),
        "month": month,
        "open": Decimal(open_ or close),
        "high": Decimal(high or close),
        "low": Decimal(low or close),
        "close": Decimal(close),
        "volume": Decimal("1"),
        "notional": price,
        "trade_count": 1,
    }


def _fp(minute: int, floor: str, buy: str = "40", sell: str = "1") -> dict:
    start = datetime(2024, 8, 2, 0, minute, tzinfo=UTC)
    return {
        "bar_start_utc": start,
        "bin_floor": Decimal(floor),
        "bin_upper_exclusive": Decimal(floor) + Decimal("30"),
        "buy_volume_btc": Decimal(buy),
        "sell_volume_btc": Decimal(sell),
        "trade_count": 1,
    }


def test_v3_identity_baseline_and_exact_stable_seven_oat_registry() -> None:
    assert STRATEGY_ID == "ImbalanceVWAPRide.BTC_LONG_ONLY_V3_EXPLORATORY"
    assert ADAPTER_ID == "imbalance-vwap-ride-btc-long-only-v3-1"
    assert EVIDENCE_LABEL == "POST_HOC_V3_LONG_ONLY"
    assert PERIOD_LABEL == "NEW_TEMPORAL_V3_DEVELOPMENT_PERIOD"
    assert SELECTION_METHOD == "PRE_REGISTERED_LONG_ONLY_OAT"
    assert BASELINE.parameter_payload() == {
        "bin_size_usd": "50",
        "min_bin_volume_btc": "35",
        "vwap_slope_bars": 24,
        "min_imbalance_ratio": "3",
        "stacked_bins": 3,
        "move_away_bars": 1,
        "zone_expiry_bars": 36,
        "stop_buffer_bins": 2,
        "target_r_multiple": "2.5",
        "maximum_active_zones": 3,
        "maximum_trades_per_utc_day": 1,
        "maximum_trades_per_zone": 1,
        "entry_execution": "NEXT_BAR_OPEN_AFTER_CONFIRMED_RETEST",
        "direction": "LONG_ONLY",
    }
    assert PARAMETER_REGISTRY == (
        ("bin_size_usd", (Decimal("30"), Decimal("50"), Decimal("75"))),
        ("min_bin_volume_btc", (Decimal("20"), Decimal("35"), Decimal("50"))),
        ("vwap_slope_bars", (18, 24, 36)),
    )
    variants = preregistered_variants()
    assert [item.variant_id for item in variants] == [
        "baseline",
        "bin_size_usd=30",
        "bin_size_usd=75",
        "min_bin_volume_btc=20",
        "min_bin_volume_btc=50",
        "vwap_slope_bars=18",
        "vwap_slope_bars=36",
    ]
    assert len({json.dumps(item.parameter_payload(), sort_keys=True) for item in variants}) == 7
    with pytest.raises(ValueError, match="sealed V3 invariant direction"):
        ImbalanceVWAPRideV3Config(direction="SHORT")
    with pytest.raises(ValueError, match="sealed V3 invariant target_r_multiple"):
        ImbalanceVWAPRideV3Config(target_r_multiple="2")


def test_v3_cli_is_dedicated_and_has_no_arbitrary_month_or_url_surface() -> None:
    args = _parser().parse_args(["imbalance-vwap-ride", "run-btc-long-only-v3-study", "--non-interactive"])
    assert args.command == "imbalance-vwap-ride"
    assert args.imbalance_command == "run-btc-long-only-v3-study"
    assert args.batch_size == 1_000_000
    assert not hasattr(args, "start_month")
    assert not hasattr(args, "end_month")
    assert not hasattr(args, "allow_network")


def test_v3_archive_allowlist_url_and_crc_integrity(tmp_path: Path) -> None:
    assert len(AUTHORIZED_MONTHS) == 6
    assert authorized_archive_url("2024-08") == (
        "https://data.binance.vision/data/futures/um/monthly/aggTrades/"
        "BTCUSDT/BTCUSDT-aggTrades-2024-08.zip"
    )
    with pytest.raises(ValueError, match="outside the sealed V3 download allowlist"):
        authorized_archive_url("2024-07")
    archive = tmp_path / "BTCUSDT-aggTrades-2024-08.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr(
            "BTCUSDT-aggTrades-2024-08.csv",
            "aggregate_trade_id,price,quantity_base,first_trade_id,last_trade_id,trade_time,buyer_is_maker,is_best_match\n"
            "1,60000.0,0.001,1,1,1722470400000,false,true\n",
        )
    report = validate_authorized_archive(archive, "2024-08")
    assert report["zip_integrity_valid"] is True
    assert len(report["archive_sha256"]) == 64
    partition, action = _vectorized_month_partition(
        tmp_path,
        AggregateTradeImporter(tmp_path),
        "2024-08",
        archive,
    )
    assert partition.row_count == 1
    assert partition.first_aggregate_trade_id == partition.last_aggregate_trade_id == 1
    assert partition.duplicate_count == 0
    assert action["normalizer_version"].endswith("vectorized-normalizer-1")
    normalized = pq.read_table(
        tmp_path / "normalized" / "BTCUSDT" / "monthly_partitions" / "2024-08" / partition.file_name
    )
    assert normalized.schema == pa.schema(
        [
            ("event_time_utc", pa.string()),
            ("trade_time_utc", pa.string()),
            ("aggregate_trade_id", pa.int64()),
            ("first_trade_id", pa.int64()),
            ("last_trade_id", pa.int64()),
            ("price", pa.string()),
            ("quantity_base", pa.string()),
            ("notional_quote", pa.string()),
            ("buyer_is_maker", pa.bool_()),
            ("aggressor_side", pa.string()),
            ("signed_quantity", pa.string()),
            ("source", pa.string()),
            ("source_file", pa.string()),
            ("source_hash", pa.string()),
        ]
    )


def _raw_table(ids: list[int]) -> pa.Table:
    starts = [datetime(2024, 8, 1, tzinfo=UTC) + timedelta(seconds=index) for index in range(len(ids))]
    prices = ["74.9", "75.0", "100.0"][: len(ids)]
    return pa.table(
        {
            "event_time_utc": [item.isoformat() for item in starts],
            "aggregate_trade_id": pa.array(ids, type=pa.int64()),
            "price": prices,
            "quantity_base": ["1"] * len(ids),
            "notional_quote": prices,
            "buyer_is_maker": [False, True, False][: len(ids)],
        }
    )


def test_v3_exact_half_open_75_bins_and_stream_continuity(tmp_path: Path) -> None:
    parquet = tmp_path / "month.parquet"
    pq.write_table(_raw_table([100, 101, 102]), parquet)
    footprints, bars, diagnostics = _aggregate_month(
        pq.ParquetFile(parquet),
        month="2024-08",
        expected_first_id=100,
        expected_last_id=102,
        batch_size=2,
    )
    rows = footprints[Decimal("75")]
    assert [(row["bin_floor"], row["bin_upper_exclusive"], row["trade_count"]) for row in rows] == [
        (Decimal("0"), Decimal("75"), 1),
        (Decimal("75"), Decimal("150"), 2),
    ]
    assert sum(row["trade_count"] for row in rows) == 3
    assert sum(row["trade_count"] for row in bars) == 3
    assert diagnostics[0]["first_aggregate_trade_id"] == 100
    broken = tmp_path / "broken.parquet"
    pq.write_table(_raw_table([100, 102]), broken)
    with pytest.raises(ValueError, match="non-contiguous"):
        _aggregate_month(
            pq.ParquetFile(broken),
            month="2024-08",
            expected_first_id=100,
            expected_last_id=102,
            batch_size=10,
        )


def test_v3_timestamp_validation_compares_instants_not_iso_string_precision(tmp_path: Path) -> None:
    table = _raw_table([100, 101])
    table = table.set_column(
        table.schema.get_field_index("event_time_utc"),
        "event_time_utc",
        pa.array(["2025-01-01T00:00:00.100000+00:00", "2025-01-01T00:00:00.100+00:00"]),
    )
    parquet = tmp_path / "equivalent-timestamps.parquet"
    pq.write_table(table, parquet)
    footprints, bars, diagnostics = _aggregate_month(
        pq.ParquetFile(parquet),
        month="2025-01",
        expected_first_id=100,
        expected_last_id=101,
        batch_size=1,
    )
    assert sum(row["trade_count"] for row in footprints[Decimal("50")]) == 2
    assert sum(row["trade_count"] for row in bars) == 2
    assert len(diagnostics) == 2


def test_v3_actual_entry_risk_target_quantity_notional_fees_and_no_lookahead() -> None:
    signal = _bar(0, close="200")
    entry = _bar(5, close="700", open_="160", low="150", high="900")
    zone = {
        "zone_id": "z1",
        "sequence_lineage": ["s1"],
        "direction": "LONG",
        "bottom": "100",
        "top": "150",
    }
    state, trade = simulate_long_trade(
        zone=zone,
        signal_bar=signal,
        entry_index=1,
        bars=[signal, entry],
        config=ImbalanceVWAPRideV3Config(),
    )
    assert state == "TRADE_EXECUTED" and trade is not None
    assert Decimal(trade["entry_price"]) == Decimal("160.1")
    assert Decimal(trade["initial_stop_price"]) == Decimal("0")
    assert Decimal(trade["actual_risk_distance"]) == Decimal("160.1")
    assert Decimal(trade["target_price"]) == Decimal("560.4")
    assert Decimal(trade["target_distance"]) > 0
    assert Decimal(trade["quantity_btc"]) == Decimal("0.001")
    assert Decimal(trade["entry_notional_usd"]) == Decimal("0.1601")
    assert Decimal(trade["fees"]) == Decimal(trade["entry_fee"]) + Decimal(trade["exit_fee"])
    assert datetime.fromisoformat(trade["entry_timestamp"]) == signal["bar_start_utc"] + timedelta(minutes=5)
    assert datetime.fromisoformat(trade["signal_timestamp"]) <= datetime.fromisoformat(trade["entry_timestamp"])


def test_v3_nonpositive_actual_risk_is_explicitly_non_executable() -> None:
    signal = _bar(0, close="200")
    entry = _bar(5, close="30", open_="30", low="20", high="40")
    zone = {
        "zone_id": "z1",
        "sequence_lineage": ["s1"],
        "direction": "LONG",
        "bottom": "1000",
        "top": "1050",
    }
    state, trade = simulate_long_trade(
        zone=zone,
        signal_bar=signal,
        entry_index=1,
        bars=[signal, entry],
        config=ImbalanceVWAPRideV3Config(),
    )
    assert state == "INVALID_ENTRY_GEOMETRY_OR_QUANTITY"
    assert trade is None


def test_v3_lifecycle_is_long_only_terminal_and_one_trade_zone_day() -> None:
    bars = [
        _bar(0, close="100"),
        _bar(5, close="210", low="205", high="215"),
        _bar(10, close="230", low="220", high="235"),
        _bar(15, close="205", low="170", high="235"),
        _bar(20, close="900", open_="260", low="201", high="1000"),
    ]
    footprints = [_fp(5, "90"), _fp(5, "120"), _fp(5, "150")]
    config = ImbalanceVWAPRideV3Config(
        bin_size_usd="30",
        min_bin_volume_btc="1",
        vwap_slope_bars=1,
    )
    result = run_imbalance_vwap_ride_v3(bars, footprints, config)
    assert len(result["trades"]) == 1
    assert all(item["direction"] == "LONG" for item in result["trades"])
    assert all(item["direction"] == "LONG" for item in result["zones"])
    assert all(item.get("direction") in {None, "LONG"} for item in result["events"])
    traded = [zone for zone in result["zones"] if zone["state"] == "TRADED"]
    assert len(traded) == 1 and traded[0]["trade_count"] == 1
    assert traded[0]["terminal_state"] == "TRADED"
    assert traded[0]["terminal_reason"] == "POST_EXECUTION"
    assert result["funnel"]["reconciles"] is True
    assert result["metrics"]["short_trades"] == 0
    assert result["metrics"]["long_only_reconciliation"]["reconciles"] is True
    assert all(zone["state"] in ACTIVE_STATES or zone["terminal_state"] == zone["state"] for zone in result["zones"])


def _passing_metrics(*, trades: int = 48, active_months: int = 5) -> dict:
    months = {}
    for index, month in enumerate(AUTHORIZED_MONTHS):
        count = 8 if index < active_months else 0
        months[month] = {"executed_trades": count}
    return {
        "executed_trades": trades,
        "gross_pnl": "2",
        "net_pnl": "1",
        "gross_profit_factor": "1.3",
        "net_profit_factor": "1.2",
        "average_gross_r": "0.2",
        "average_net_r": "0.1",
        "maximum_drawdown": "0.5",
        "maximum_positive_month_contribution": "0.4",
        "best_five_positive_pnl_contribution": "0.5",
        "funnel_reconciliation": {"reconciles": True},
        "long_only_reconciliation": {"reconciles": True},
        "months": months,
    }


def test_v3_promotion_monthly_activity_informative_band_and_deterministic_freeze() -> None:
    assert promotion_gate(_passing_metrics())["passed"] is True
    informative = promotion_gate(_passing_metrics(trades=47))
    assert informative["passed"] is False
    assert informative["sample_classification"] == "INFORMATIVE_36_TO_47_NOT_PROMOTABLE"
    assert promotion_gate(_passing_metrics(active_months=4))["checks"]["five_active_months"] is False
    candidates = [
        (ImbalanceVWAPRideV3Config(variant_id="z", min_bin_volume_btc="20"), _passing_metrics()),
        (ImbalanceVWAPRideV3Config(variant_id="a", min_bin_volume_btc="50"), _passing_metrics()),
        (BASELINE, _passing_metrics()),
    ]
    frozen = freeze_passing_candidates(candidates)
    assert [item[0].variant_id for item in frozen] == ["baseline", "a"]


def test_v3_preserves_v1_v2_and_jan_jul_source_fingerprints() -> None:
    snapshot = preservation_snapshot(Path(__file__).resolve().parents[2])
    assert snapshot == {
        "v1_file_count": V1_FILE_COUNT,
        "v1_tree_digest": V1_TREE_DIGEST,
        "v1_preserved": True,
        "v2_file_count": V2_FILE_COUNT,
        "v2_tree_digest": V2_TREE_DIGEST,
        "v2_preserved": True,
        "jan_jul_source_file_count": JAN_JUL_SOURCE_FILE_COUNT,
        "jan_jul_source_tree_digest": JAN_JUL_SOURCE_TREE_DIGEST,
        "jan_jul_source_preserved": True,
    }


def test_v3_existing_final_is_deterministic_and_identity_collision_fails(tmp_path: Path) -> None:
    identity = {"strategy_id": "v3", "dataset_hash": "dataset", "code_hash": "code"}
    root = tmp_path / "run"
    root.mkdir()
    (root / "study-manifest.json").write_text(json.dumps(identity), encoding="utf-8")
    final = {
        "status": "DEVELOPMENT_EDGE_NOT_FOUND",
        "summary": "deterministic",
        "tests_passed": True,
        "study_executed": True,
    }
    (root / "final_report.json").write_text(json.dumps(final), encoding="utf-8")
    assert _existing_final(root, identity) == _existing_final(root, identity) == {
        "status": "DEVELOPMENT_EDGE_NOT_FOUND",
        "summary": "deterministic",
        "finalReportPath": str(root / "final_report.json"),
        "testsPassed": True,
        "studyExecuted": True,
    }
    with pytest.raises(ValueError, match="identity collision"):
        _existing_final(root, {**identity, "code_hash": "changed"})


def test_v3_artifact_context_seals_labels_in_json_and_parquet(tmp_path: Path) -> None:
    context = V3ArtifactContext("run", "data", "source", "spec", "params", "code", EVIDENCE_LABEL, "now")
    payload = json.loads(context.write_json(tmp_path / "artifact.json", {"value": 1}).read_text(encoding="utf-8"))
    assert payload["evidence_label"] == EVIDENCE_LABEL
    assert payload["period_label"] == PERIOD_LABEL
    assert payload["confirmation_evidence"] is False
    assert payload["optimization_claimed"] is False
    assert payload["external_confirmation_required"] is True
    metadata = pq.read_metadata(context.write_parquet(tmp_path / "artifact.parquet", [])).metadata
    assert metadata[b"evidence_label"] == EVIDENCE_LABEL.encode()
    assert metadata[b"period_label"] == PERIOD_LABEL.encode()
    assert metadata[b"direction"] == b"LONG_ONLY"
