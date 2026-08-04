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
from research_pipeline.imbalance_vwap_ride.artifacts import sha256_value
from research_pipeline.value_area_trap.data import AggregateTradeImporter
from research_pipeline.imbalance_vwap_ride.v4_adapter import ImbalanceVWAPRideV4Adapter
from research_pipeline.imbalance_vwap_ride.v4_alpha import (
    MINIMUM_BOOTSTRAP_PATHS,
    run_v4_alpha_proxy,
    validate_alpha_rules_artifact,
)
from research_pipeline.imbalance_vwap_ride.v4_artifacts import (
    ImmutableV4ArtifactStore,
    validate_v4_artifact_tree,
)
from research_pipeline.imbalance_vwap_ride.v4_data import (
    _aggregate_month,
    _vectorized_month_partition,
    authorized_archive_url,
    validate_authorized_archive,
)
from research_pipeline.imbalance_vwap_ride.v4_models import (
    ADAPTER_ID,
    CANDIDATE_REGISTRY,
    LOCKED_EVIDENCE,
    PHASE_A_MONTHS,
    PHASE_B_MONTHS,
    SELECTION_EVIDENCE,
    STRATEGY_ID,
    ImbalanceVWAPRideV4Config,
    candidate_registry_hash,
    candidate_registry_payload,
    phase_a_gate,
    phase_b_gate,
    preregistered_candidates,
    rank_phase_a_candidates,
)
from research_pipeline.imbalance_vwap_ride.v4_strategy import (
    reconcile_funnel,
    run_imbalance_vwap_ride_v4,
    simulate_long_trade,
)

UTC = timezone.utc


def _bar(minute: int, *, close: str, open_: str | None = None) -> dict:
    start = datetime(2023, 1, 2, tzinfo=UTC) + timedelta(minutes=minute)
    price = Decimal(close)
    return {
        "bar_start_utc": start,
        "bar_end_utc": start + timedelta(minutes=5),
        "session_date": start.date().isoformat(),
        "month": "2023-01",
        "open": Decimal(open_ or close),
        "high": price,
        "low": price,
        "close": price,
        "volume": Decimal("1"),
        "notional": price,
        "trade_count": 1,
    }


def _metrics(months: tuple[str, ...], trades_per_month: int) -> dict:
    month_count = len(months)
    monthly = {
        month: {
            "executed_trades": trades_per_month,
            "gross_pnl": "2",
            "net_pnl": "1",
            "total_costs": "1",
        }
        for month in months
    }
    return {
        "executed_trades": trades_per_month * len(months),
        "gross_pnl": str(month_count * 2),
        "net_pnl": str(month_count),
        "gross_profit_factor": "1.4",
        "net_profit_factor": "1.2",
        "average_gross_r": "0.2",
        "average_net_r": "0.1",
        "fees": str(Decimal(month_count) / 2),
        "slippage_cost": str(Decimal(month_count) / 2),
        "total_costs": str(month_count),
        "gross_risk_usd": "10",
        "maximum_drawdown": "0.5",
        "best_five_positive_pnl_contribution": "0.5",
        "funnel_reconciliation": {
            "reconciles": True,
            "proposed_setups": trades_per_month * month_count,
            "invalid_setups": 0,
            "non_executable_setups": 0,
            "compliance_blocks": 0,
            "executed_trades": trades_per_month * month_count,
        },
        "long_only_reconciliation": {
            "reconciles": True,
            "executed_trades": trades_per_month * month_count,
            "long_trades": trades_per_month * month_count,
            "short_trades": 0,
            "short_setups": 0,
            "short_pnl": "0",
        },
        "months": monthly,
    }


def test_v4_identity_exact_registry_and_strict_long_only_config() -> None:
    assert STRATEGY_ID == "ImbalanceVWAPRide.BTC_LONG_ONLY_V4_CANDIDATE_SELECTION"
    assert ADAPTER_ID == "imbalance-vwap-ride-btc-long-only-v4-1"
    assert SELECTION_EVIDENCE == "POST_HOC_V4_RETROSPECTIVE_CANDIDATE_SELECTION"
    assert LOCKED_EVIDENCE == "STRATEGY_SPECIFIC_TEMPORAL_LOCKED_TEST"
    assert CANDIDATE_REGISTRY == (
        ("V4-A-BASELINE-2P5R", 24, Decimal("2.5")),
        ("V4-B-BASELINE-3P0R", 24, Decimal("3.0")),
        ("V4-C-BASELINE-3P5R", 24, Decimal("3.5")),
        ("V4-D-SLOW-VWAP-2P5R", 36, Decimal("2.5")),
    )
    assert [item.candidate_id for item in preregistered_candidates()] == [
        item[0] for item in CANDIDATE_REGISTRY
    ]
    assert candidate_registry_hash() == sha256_value(candidate_registry_payload())
    with pytest.raises(ValueError, match="sealed V4 invariant direction"):
        ImbalanceVWAPRideV4Config(direction="SHORT")
    with pytest.raises(ValueError, match="do not match candidate_id"):
        ImbalanceVWAPRideV4Config(target_r_multiple="3.0")


def test_v4_cli_has_dedicated_bounded_surfaces() -> None:
    args = _parser().parse_args(
        ["imbalance-vwap-ride", "run-btc-long-only-v4-study", "--non-interactive"]
    )
    assert args.imbalance_command == "run-btc-long-only-v4-study"
    assert args.batch_size == 1_000_000
    assert not hasattr(args, "start_month")
    assert not hasattr(args, "end_month")
    assert not hasattr(args, "url")
    validate = _parser().parse_args(
        [
            "imbalance-vwap-ride",
            "validate-btc-long-only-v4-source",
            "manifest.json",
            "--phase",
            "PHASE_A",
        ]
    )
    assert validate.phase == "PHASE_A"


def test_v4_archive_allowlist_crc_and_phase_isolation(tmp_path: Path) -> None:
    assert len(PHASE_A_MONTHS) == 12 and len(PHASE_B_MONTHS) == 6
    assert authorized_archive_url("2023-01", phase="PHASE_A").endswith(
        "BTCUSDT/BTCUSDT-aggTrades-2023-01.zip"
    )
    with pytest.raises(ValueError, match="outside the sealed V4 download allowlist"):
        authorized_archive_url("2025-02", phase="PHASE_A")
    archive = tmp_path / "BTCUSDT-aggTrades-2023-01.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr(
            "BTCUSDT-aggTrades-2023-01.csv",
            "aggregate_trade_id,price,quantity_base,first_trade_id,last_trade_id,trade_time,buyer_is_maker,is_best_match\n"
            "1,16500.0,0.001,1,1,1672531200000,false,true\n",
        )
    report = validate_authorized_archive(archive, "2023-01", phase="PHASE_A")
    assert report["zip_integrity_valid"] is True
    assert len(report["archive_sha256"]) == 64


def test_v4_interleaved_official_archive_order_is_normalized_before_atomic_commit(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "BTCUSDT-aggTrades-2023-01.zip"
    header = (
        "agg_trade_id,price,quantity,first_trade_id,last_trade_id,"
        "transact_time,is_buyer_maker\n"
    )
    rows = [
        "1,100,1,1,1,1672531200001,false",
        "4,103,1,4,4,1672531200004,true",
        "2,101,1,2,2,1672531200002,true",
        "5,104,1,5,5,1672531200005,false",
        "3,102,1,3,3,1672531200003,false",
    ]
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("BTCUSDT-aggTrades-2023-01.csv", header + "\n".join(rows) + "\n")
    cache = tmp_path / "cache"
    partition, action = _vectorized_month_partition(
        cache,
        AggregateTradeImporter(cache),
        "2023-01",
        archive,
        phase="PHASE_A",
    )
    target = (
        cache
        / "normalized"
        / "BTCUSDT"
        / "monthly_partitions"
        / "2023-01"
        / partition.file_name
    )
    normalized_ids = pq.read_table(target, columns=["aggregate_trade_id"])[
        "aggregate_trade_id"
    ].to_pylist()
    assert normalized_ids == [1, 2, 3, 4, 5]
    assert partition.repair_status == "OFFICIAL_ARCHIVE_ORDER_NORMALIZED"
    assert action["archive_order_lane_count"] == 2
    assert (target.parent / "partition.json").is_file()
    assert not list((cache / ".staging").glob("v4-2023-01-*.part"))


def test_v4_streaming_bars_have_fixed_bins_aggressor_volume_daily_vwap_and_session_delta(
    tmp_path: Path,
) -> None:
    table = pa.table(
        {
            "event_time_utc": [
                "2023-01-01T00:00:01+00:00",
                "2023-01-01T00:00:02+00:00",
                "2023-01-01T00:05:01+00:00",
            ],
            "aggregate_trade_id": pa.array([10, 12, 14], type=pa.int64()),
            "price": ["99.9", "100", "101"],
            "quantity_base": ["1", "2", "3"],
            "notional_quote": ["99.9", "200", "303"],
            "buyer_is_maker": [False, True, False],
        }
    )
    path = tmp_path / "month.parquet"
    pq.write_table(table, path)
    footprints, bars, diagnostics = _aggregate_month(
        pq.ParquetFile(path),
        month="2023-01",
        expected_first_id=10,
        expected_last_id=14,
        expected_row_count=3,
        batch_size=1,
    )
    assert {row["bin_size_usd"] for row in footprints} == {Decimal("50")}
    assert sum(row["trade_count"] for row in footprints) == 3
    assert sum(row["trade_count"] for row in bars) == 3
    assert bars[0]["buy_volume_btc"] == Decimal("1")
    assert bars[0]["sell_volume_btc"] == Decimal("2")
    assert bars[0]["cumulative_session_delta_btc"] == Decimal("-1")
    assert bars[1]["cumulative_session_delta_btc"] == Decimal("2")
    assert bars[0]["daily_vwap"] == Decimal("299.9") / Decimal("3")
    assert len(diagnostics) == 3


def test_v4_actual_entry_risk_target_and_no_lookahead() -> None:
    signal = _bar(0, close="200")
    entry = _bar(5, close="700", open_="160")
    entry["low"] = Decimal("150")
    entry["high"] = Decimal("900")
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
        config=ImbalanceVWAPRideV4Config(),
    )
    assert state == "TRADE_EXECUTED" and trade is not None
    assert Decimal(trade["entry_price"]) == Decimal("160.1")
    assert Decimal(trade["initial_stop_price"]) == Decimal("0")
    assert Decimal(trade["target_price"]) == Decimal("560.4")
    assert trade["direction"] == "LONG"
    assert trade["candidate_id"] == "V4-A-BASELINE-2P5R"
    assert datetime.fromisoformat(trade["entry_timestamp"]) >= datetime.fromisoformat(
        trade["signal_timestamp"]
    )


def test_v4_all_funnel_paths_and_zero_short_diagnostics() -> None:
    funnel = reconcile_funnel(
        {
            "proposed_setups": 10,
            "invalid_setups": 1,
            "non_executable_setups": 2,
            "compliance_blocks": 3,
            "executed_trades": 4,
        }
    )
    assert funnel["reconciles"] is True
    result = run_imbalance_vwap_ride_v4([_bar(0, close="100")], [], phase="FIXTURE")
    assert result["funnel"]["reconciles"] is True
    assert result["metrics"]["short_trades"] == 0
    assert result["metrics"]["short_setups"] == 0
    assert result["metrics"]["short_pnl"] == "0"


def test_v4_phase_a_gate_exact_ranking_and_no_robust_candidate() -> None:
    passing = _metrics(PHASE_A_MONTHS, 6)
    assert phase_a_gate(passing)["passed"] is True
    failing = json.loads(json.dumps(passing))
    failing["executed_trades"] = 71
    assert phase_a_gate(failing)["checks"]["minimum_72_trades"] is False
    configs = preregistered_candidates()
    better = _metrics(PHASE_A_MONTHS, 6)
    better["net_profit_factor"] = "1.3"
    ranked = rank_phase_a_candidates([(configs[0], passing), (configs[3], better)])
    assert [item["candidate_id"] for item in ranked] == [
        "V4-D-SLOW-VWAP-2P5R",
        "V4-A-BASELINE-2P5R",
    ]
    assert ranked[0]["selection_method"] == "PRE_REGISTERED_ROBUSTNESS_RANKING"
    assert rank_phase_a_candidates([(configs[0], failing)]) == []


def test_v4_phase_b_locked_gate_and_alpha_eligibility_fail_closed() -> None:
    metrics = _metrics(PHASE_B_MONTHS, 5)
    metrics["hashes_valid"] = True
    metrics["costs_valid"] = True
    locked = phase_b_gate(metrics)
    assert locked["status"] == "LOCKED_TEST_PASSED"
    assert locked["evidence_label"] == LOCKED_EVIDENCE
    with pytest.raises(ValueError, match="at least 20,000"):
        run_v4_alpha_proxy(
            [],
            [],
            locked_test_status="LOCKED_TEST_PASSED",
            phase_b_execution_count=1,
            frozen_candidate_valid=True,
            rules_artifact=None,
            paths=MINIMUM_BOOTSTRAP_PATHS - 1,
        )
    report = run_v4_alpha_proxy(
        [],
        [],
        locked_test_status="LOCKED_TEST_FAILED",
        phase_b_execution_count=0,
        frozen_candidate_valid=False,
        rules_artifact=None,
    )
    assert report["status"] == "NOT_EXECUTED"
    assert report["alpha_executed"] is False


def test_v4_rules_artifact_requires_hash_freshness_officiality_and_consistency() -> None:
    now = datetime(2026, 8, 4, 12, tzinfo=UTC)
    content = {
        "official": True,
        "official_source": "https://alpha-futures.com/official-rules",
        "product": "Alpha Futures 25K Zero",
        "retrieved_at": now.isoformat(),
        "consistent": True,
    }
    artifact = {**content, "content_hash": sha256_value(content)}
    assert validate_alpha_rules_artifact(artifact, now=now)["valid"] is True
    artifact["product"] = "changed"
    assert validate_alpha_rules_artifact(artifact, now=now)["valid"] is False


def _identity() -> dict:
    return {
        "strategy_id": STRATEGY_ID,
        "adapter_id": ADAPTER_ID,
        "specification_hash": "spec",
        "candidate_registry_hash": candidate_registry_hash(),
        "code_hash": "code",
        "phase_a_dataset_hash": "phase-a",
        "phase_a_source_manifest_hash": "phase-a-source",
        "phase_b_dataset_hash": "phase-b",
        "phase_b_source_manifest_hash": "phase-b-source",
    }


def test_v4_immutable_store_freeze_phase_b_isolation_and_no_raw_disclosure(tmp_path: Path) -> None:
    store = ImmutableV4ArtifactStore(tmp_path, _identity())
    assert ImmutableV4ArtifactStore(tmp_path, _identity()).run_id == store.run_id
    store.write_json("phase_a/report.json", {"value": 1}, phase="PHASE_A")
    store.write_json("phase_a/report.json", {"value": 1}, phase="PHASE_A")
    with pytest.raises(ValueError, match="immutable artifact collision"):
        store.write_json("phase_a/report.json", {"value": 2}, phase="PHASE_A")
    frozen = store.freeze_candidate(
        ImbalanceVWAPRideV4Config(),
        _metrics(PHASE_A_MONTHS, 6),
        {"rank": 1},
    )
    marker = store.begin_phase_b(frozen["frozen_candidate_hash"])
    assert marker["execution_count"] == 1
    with pytest.raises(ValueError, match="does not match"):
        store.begin_phase_b("changed")
    with pytest.raises(ValueError, match="forbidden raw-row"):
        store.write_json(
            "phase_a/raw.json",
            {"raw_aggregate_rows": [{"aggregate_trade_id": 1}]},
            phase="PHASE_A",
        )
    store.seal_integrity_manifest()
    health = validate_v4_artifact_tree(store.root, expected_identity=_identity())
    assert health["valid"] is True
    assert health["raw_aggregate_rows_transmitted"] is False


def test_v4_adapter_has_no_live_orders_or_external_raw_transmission() -> None:
    capabilities = ImbalanceVWAPRideV4Adapter().capabilities()
    assert capabilities["live_orders"] is False
    assert capabilities["external_raw_trade_transmission"] is False
    assert capabilities["secrets_required"] is False
    assert capabilities["direction"] == "LONG_ONLY"
