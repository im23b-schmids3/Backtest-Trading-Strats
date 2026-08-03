from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from research_pipeline.cli import _parser
from research_pipeline.imbalance_vwap_ride.v2_models import (
    BASELINE,
    EVIDENCE_LABEL,
    PARAMETER_REGISTRY,
    ImbalanceVWAPRideV2Config,
    development_gate,
    preregistered_variants,
)
from research_pipeline.imbalance_vwap_ride.v2_runner import (
    SOURCE_MANIFEST_FILE_SHA256,
    V1_FILE_COUNT,
    V1_TREE_DIGEST,
    _existing_final,
    V2ArtifactContext,
    preservation_snapshot,
)
from research_pipeline.imbalance_vwap_ride.v2_strategy import (
    ACTIVE_STATES,
    run_imbalance_vwap_ride_v2,
    simulate_trade,
    summarize_strategy_result,
)

UTC = timezone.utc


def _bar(
    minute: int,
    *,
    close: str,
    open_: str | None = None,
    low: str | None = None,
    high: str | None = None,
    vwap_price: str | None = None,
) -> dict:
    start = datetime(2024, 1, 2, 0, minute, tzinfo=UTC)
    price = Decimal(vwap_price or close)
    return {
        "bar_start_utc": start,
        "bar_end_utc": start + timedelta(minutes=5),
        "session_date": start.date().isoformat(),
        "month": "2024-01",
        "open": Decimal(open_ or close),
        "high": Decimal(high or close),
        "low": Decimal(low or close),
        "close": Decimal(close),
        "volume": Decimal("1"),
        "notional": price,
        "trade_count": 1,
    }


def _fp(minute: int, floor: str, buy: str = "40", sell: str = "1") -> dict:
    start = datetime(2024, 1, 2, 0, minute, tzinfo=UTC)
    return {
        "bar_start_utc": start,
        "bin_floor": Decimal(floor),
        "bin_upper_exclusive": Decimal(floor) + Decimal("10"),
        "buy_volume_btc": Decimal(buy),
        "sell_volume_btc": Decimal(sell),
        "trade_count": 1,
    }


def test_v2_baseline_registry_is_exact_stable_unique_oat_and_not_cartesian() -> None:
    assert EVIDENCE_LABEL == "POST_HOC_V2_DEVELOPMENT"
    assert BASELINE.parameter_payload() == {
        "bin_size_usd": "50",
        "min_imbalance_ratio": "3",
        "stacked_bins": 3,
        "min_bin_volume_btc": "35",
        "vwap_slope_bars": 24,
        "move_away_bars": 2,
        "zone_expiry_bars": 36,
        "stop_buffer_bins": 2,
        "target_r_multiple": "2",
        "maximum_active_zones_per_direction": 3,
        "maximum_trades_per_utc_day": 1,
        "entry_execution": "NEXT_BAR_OPEN_AFTER_CONFIRMED_RETEST",
    }
    assert PARAMETER_REGISTRY == (
        ("bin_size_usd", (Decimal("30"), Decimal("50"), Decimal("100"))),
        ("min_bin_volume_btc", (Decimal("20"), Decimal("35"), Decimal("50"))),
        ("vwap_slope_bars", (12, 24, 36)),
        ("min_imbalance_ratio", (Decimal("2.5"), Decimal("3"), Decimal("4"))),
        ("stacked_bins", (2, 3, 4)),
        ("move_away_bars", (1, 2)),
        ("zone_expiry_bars", (20, 36, 48)),
        ("stop_buffer_bins", (1, 2, 3)),
        ("target_r_multiple", (Decimal("1.5"), Decimal("2"), Decimal("2.5"))),
    )
    variants = preregistered_variants()
    assert [item.variant_id for item in variants] == [
        "baseline",
        "bin_size_usd=30",
        "bin_size_usd=100",
        "min_bin_volume_btc=20",
        "min_bin_volume_btc=50",
        "vwap_slope_bars=12",
        "vwap_slope_bars=36",
        "min_imbalance_ratio=2p5",
        "min_imbalance_ratio=4",
        "stacked_bins=2",
        "stacked_bins=4",
        "move_away_bars=1",
        "zone_expiry_bars=20",
        "zone_expiry_bars=48",
        "stop_buffer_bins=1",
        "stop_buffer_bins=3",
        "target_r_multiple=1p5",
        "target_r_multiple=2p5",
    ]
    assert len({json.dumps(item.parameter_payload(), sort_keys=True) for item in variants}) == 18
    baseline = BASELINE.parameter_payload()
    for variant in variants[1:]:
        assert sum(variant.parameter_payload()[name] != baseline[name] for name, _ in PARAMETER_REGISTRY) == 1


def test_v2_has_a_separate_non_network_cli_entrypoint() -> None:
    args = _parser().parse_args(["imbalance-vwap-ride", "run-btc-macro-bins-v2-study", "--non-interactive"])
    assert args.command == "imbalance-vwap-ride"
    assert args.imbalance_command == "run-btc-macro-bins-v2-study"
    assert args.batch_size == 1_000_000
    assert not hasattr(args, "allow_network")


def test_v2_actual_entry_sets_risk_target_gap_quantity_notional_fees_and_cost_risk() -> None:
    signal = _bar(0, close="200")
    entry = _bar(5, close="300", open_="160", low="150", high="400")
    config = ImbalanceVWAPRideV2Config(
        stacked_bins=2,
        stop_buffer_bins=1,
        target_r_multiple=Decimal("2"),
    )
    zone = {
        "zone_id": "z1",
        "sequence_lineage": ["s1"],
        "direction": "LONG",
        "bottom": "100",
        "top": "150",
    }
    state, trade = simulate_trade(
        zone=zone,
        signal_bar=signal,
        entry_index=1,
        bars=[signal, entry],
        config=config,
    )
    assert state == "TRADE_EXECUTED" and trade is not None
    assert Decimal(trade["entry_price"]) == Decimal("160.1")
    assert Decimal(trade["theoretical_zone_edge_risk_distance"]) == Decimal("100")
    assert Decimal(trade["actual_risk_distance"]) == Decimal("110.1")
    assert Decimal(trade["entry_gap_distance"]) == Decimal("10.1")
    assert Decimal(trade["target_price"]) == Decimal("380.3")
    assert Decimal(trade["target_distance"]) == Decimal("220.2")
    assert Decimal(trade["quantity_btc"]) == Decimal("0.001")
    assert Decimal(trade["entry_notional_usd"]) == Decimal("0.1601")
    assert Decimal(trade["exit_notional_usd"]) == Decimal("0.3803")
    assert Decimal(trade["entry_fee"]) == Decimal("0.00008005")
    assert Decimal(trade["exit_fee"]) == Decimal("0.00019015")
    assert Decimal(trade["total_costs"]) == Decimal(trade["fees"]) + Decimal(trade["slippage_cost"])
    assert Decimal(trade["cost_to_risk"]) == Decimal(trade["total_costs"]) / Decimal(trade["actual_risk_usd"])
    assert Decimal(trade["gross_r"]) == Decimal(trade["gross_pnl"]) / Decimal(trade["actual_risk_usd"])
    assert Decimal(trade["net_r"]) == Decimal(trade["net_pnl"]) / Decimal(trade["actual_risk_usd"])


def test_v2_non_positive_actual_risk_is_invalid_and_non_executable() -> None:
    signal = _bar(0, close="200")
    entry = _bar(5, close="30", open_="30", low="20", high="40")
    zone = {
        "zone_id": "z1",
        "sequence_lineage": ["s1"],
        "direction": "LONG",
        "bottom": "100",
        "top": "150",
    }
    state, trade = simulate_trade(
        zone=zone,
        signal_bar=signal,
        entry_index=1,
        bars=[signal, entry],
        config=ImbalanceVWAPRideV2Config(stop_buffer_bins=1),
    )
    assert state == "INVALID_ENTRY_GEOMETRY_OR_QUANTITY"
    assert trade is None


def test_v2_terminal_trade_is_removed_from_active_matching_and_lifecycle_is_persisted() -> None:
    bars = [
        _bar(0, close="100"),
        _bar(5, close="210", low="205", high="215", vwap_price="210"),
        _bar(10, close="230", low="220", high="235", vwap_price="230"),
        _bar(15, close="205", low="195", high="235", vwap_price="205"),
        _bar(20, close="260", open_="260", low="201", high="500", vwap_price="260"),
    ]
    footprints = [
        _fp(5, "100"),
        _fp(5, "150"),
        _fp(20, "100"),
        _fp(20, "150"),
    ]
    config = ImbalanceVWAPRideV2Config(
        stacked_bins=2,
        vwap_slope_bars=1,
        move_away_bars=1,
        zone_expiry_bars=10,
        stop_buffer_bins=0,
        target_r_multiple=Decimal("1"),
    )
    result = run_imbalance_vwap_ride_v2(bars, footprints, config)
    assert len(result["trades"]) == 1
    assert len(result["zones"]) == 2
    first, second = result["zones"]
    assert first["state"] == first["terminal_state"] == "TRADED"
    assert first["terminal_reason"] == "POST_EXECUTION"
    assert first["terminal_timestamp"] == bars[4]["bar_start_utc"].isoformat()
    assert first["lifetime_bars"] == 3
    assert first["move_away_confirmed"] is True
    assert first["move_away_confirmed_timestamp"] is not None
    assert first["retest_triggered"] is True
    assert first["retest_timestamp"] is not None
    assert second["state"] in ACTIVE_STATES
    assert second["superseded_by_zone_id"] is None
    assert result["funnel"]["reconciles"] is True
    assert result["metrics"]["long_short_reconciliation"]["reconciles"] is True


def _metric_trade(
    trade_id: str,
    direction: str,
    gross: str,
    net: str,
    gross_r: str,
    net_r: str,
    risk: str,
    costs: str,
) -> dict:
    return {
        "trade_id": trade_id,
        "direction": direction,
        "entry_timestamp": "2024-01-02T00:00:00+00:00",
        "session_date": "2024-01-02",
        "gross_pnl": gross,
        "net_pnl": net,
        "gross_r": gross_r,
        "net_r": net_r,
        "actual_risk_usd": risk,
        "fees": str(Decimal(costs) / 2),
        "slippage_cost": str(Decimal(costs) / 2),
        "total_costs": costs,
        "cost_to_risk": str(Decimal(costs) / Decimal(risk)),
        "same_bar_ambiguity": False,
        "exit_reason": "TARGET" if Decimal(net) > 0 else "STOP",
    }


def test_v2_summary_reports_gross_net_cost_concentration_and_long_short_metrics() -> None:
    trades = [
        _metric_trade("t1", "LONG", "2", "1.8", "2", "1.8", "1", "0.2"),
        _metric_trade("t2", "SHORT", "-1", "-1.6", "-1", "-1.6", "1", "0.6"),
    ]
    funnel = {
        "proposed_setups": 2,
        "invalid_setups": 0,
        "non_executable_setups": 0,
        "compliance_blocks": 0,
        "executed_trades": 2,
        "components_total": 2,
        "reconciles": True,
    }
    metrics = summarize_strategy_result(
        [_bar(0, close="100")],
        trades,
        funnel,
        {"imbalance_sequences": 2, "zones_created": 2},
    )
    assert Decimal(metrics["gross_profit_factor"]) == Decimal("2")
    assert Decimal(metrics["net_profit_factor"]) == Decimal("1.125")
    assert Decimal(metrics["average_gross_r"]) == Decimal("0.5")
    assert Decimal(metrics["average_net_r"]) == Decimal("0.1")
    assert Decimal(metrics["median_initial_risk_usd"]) == Decimal("1")
    assert Decimal(metrics["gross_risk_usd"]) == Decimal("2")
    assert Decimal(metrics["total_costs"]) == Decimal("0.8")
    assert Decimal(metrics["median_cost_to_risk"]) == Decimal("0.4")
    assert Decimal(metrics["cost_to_risk_share_over_10_percent"]) == Decimal("1")
    assert Decimal(metrics["cost_to_risk_share_over_25_percent"]) == Decimal("0.5")
    assert Decimal(metrics["cost_to_risk_share_over_50_percent"]) == Decimal("0.5")
    assert metrics["long_short_metrics"]["LONG"]["executed_trades"] == 1
    assert metrics["long_short_metrics"]["SHORT"]["executed_trades"] == 1
    assert metrics["long_short_reconciliation"]["reconciles"] is True


def _gate_metrics(*, trades: int = 40, gross_pf: str = "1.2", net_pf: str = "1.1") -> dict:
    return {
        "executed_trades": trades,
        "gross_profit_factor": gross_pf,
        "net_profit_factor": net_pf,
        "average_gross_r": "0.2",
        "average_net_r": "0.1",
        "maximum_positive_month_contribution": "0.60",
        "best_five_positive_pnl_contribution": "0.69",
        "maximum_drawdown": "5",
        "long_trades": trades // 2,
        "short_trades": trades - trades // 2,
        "funnel_reconciliation": {"reconciles": True},
        "long_short_reconciliation": {"reconciles": True},
    }


def test_v2_gate_distinguishes_insufficient_pre_cost_and_cost_destroyed_edges() -> None:
    insufficient = development_gate(_gate_metrics(trades=39))
    assert insufficient["passed"] is False
    assert insufficient["edge_diagnosis"]["classification"] == "RESTRICTIVE_THRESHOLD_SAMPLE_INSUFFICIENCY"
    pre_cost = development_gate(_gate_metrics(gross_pf="1.0", net_pf="0.9"))
    assert pre_cost["edge_diagnosis"]["classification"] == "PRE_COST_EDGE_FAILURE"
    cost_destroyed = development_gate(_gate_metrics(net_pf="1.0"))
    assert cost_destroyed["edge_diagnosis"]["classification"] == "COST_DESTROYED_EDGE"
    assert development_gate(_gate_metrics())["passed"] is True


def test_v1_run_and_source_manifest_match_fixed_preservation_fingerprints() -> None:
    snapshot = preservation_snapshot(Path(__file__).resolve().parents[2])
    assert snapshot == {
        "v1_file_count": V1_FILE_COUNT,
        "v1_tree_digest": V1_TREE_DIGEST,
        "v1_preserved": True,
        "source_manifest_sha256": SOURCE_MANIFEST_FILE_SHA256,
        "source_manifest_preserved": True,
    }


def test_v2_existing_final_is_deterministic_and_identity_collision_fails(tmp_path: Path) -> None:
    identity = {"strategy_id": "v2", "dataset_hash": "dataset", "code_hash": "code"}
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
    first = _existing_final(root, identity)
    second = _existing_final(root, identity)
    assert first == second
    assert first == {
        "status": "DEVELOPMENT_EDGE_NOT_FOUND",
        "summary": "deterministic",
        "finalReportPath": str(root / "final_report.json"),
        "testsPassed": True,
        "studyExecuted": True,
    }
    with pytest.raises(ValueError, match="identity collision"):
        _existing_final(root, {**identity, "code_hash": "changed"})


def test_v2_artifact_context_labels_json_and_even_empty_parquet(tmp_path: Path) -> None:
    context = V2ArtifactContext("run", "data", "source", "spec", "params", "code", EVIDENCE_LABEL, "now")
    json_path = context.write_json(tmp_path / "artifact.json", {"value": 1})
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["evidence_label"] == EVIDENCE_LABEL
    assert payload["confirmation_evidence"] is False
    assert payload["optimization_claimed"] is False
    assert payload["external_holdout_required"] is True
    parquet_path = context.write_parquet(tmp_path / "artifact.parquet", [])
    metadata = pq.read_metadata(parquet_path).metadata
    assert metadata[b"evidence_label"] == EVIDENCE_LABEL.encode()
    assert metadata[b"confirmation_evidence"] == b"false"
    assert metadata[b"optimization_claimed"] == b"false"
    assert metadata[b"external_holdout_required"] == b"true"
