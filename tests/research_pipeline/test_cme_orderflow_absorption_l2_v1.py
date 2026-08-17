from __future__ import annotations

import json
import csv
import sys
import types
from decimal import Decimal
from pathlib import Path

import pytest

from research_pipeline.cme_orderflow_absorption_l2_v1 import model as l2
from research_pipeline.cme_orderflow_absorption_l2_v1 import historical_runner as historical
from research_pipeline.cme_orderflow_absorption_l2_v1 import rejection_funnel as funnel
from research_pipeline.cme_orderflow_absorption_l2_v1 import v2_quality050 as v2
from research_pipeline.cme_orderflow_absorption_l2_v1 import v2_performance_report as v2_report
from research_pipeline.cme_orderflow_absorption_l2_v1 import v2_score_validity as score_validity
from research_pipeline.cme_orderflow_absorption_l2_v1 import v2_extended_existing_data as extended
from research_pipeline.cme_orderflow_absorption_l2_v1 import v2_august_seen_replay as august_seen
from research_pipeline.cme_orderflow_absorption_l2_v1 import v2_august_selectivity_integrity_audit as august_audit
from research_pipeline.cme_orderflow_absorption_l2_v1 import v3_poc_only as v3
from research_pipeline.cme_orderflow_absorption_l2_v1.native_mbp10_contract import (
    native_mbp10_adapter_contract, validate_native_mbp10_field_mapping,
)
from research_pipeline.cme_orderflow_absorption_l2_v1 import v3_poc_fresh_august_replay as v3_fresh
from research_pipeline.cme_orderflow_absorption_l2_v1 import v3_poc_april_retro_replay as v3_april_replay
import quote_august_l2_v2_completion_cost as august_completion
import acquire_august_l2_v2_completion as august_acquisition
import quote_l2_v3_poc_only_cost as v3_quote
import acquire_l2_v3_poc_only_august as v3_acquisition
import acquire_l2_v3_poc_only_april as v3_april_acquisition


BASE_NS = 1_000_000_000_000
PX = 5000.0


def execution(milliseconds: float, *, price: float = PX, size: int = 50, aggressor: str = "SELL") -> l2.Execution:
    return l2.Execution(BASE_NS + int(milliseconds * 1_000_000), price, size, aggressor)  # type: ignore[arg-type]


def snapshot(milliseconds: float, *, bid_size: int = 100, bid_count: int = 2,
             ask_size: int = 100, ask_count: int = 2,
             defended_side: str | None = None) -> l2.MBP10Snapshot:
    bid_px, ask_px = (PX - 0.25, PX) if defended_side == "A" else (PX, PX + 0.25)
    return l2.MBP10Snapshot(
        BASE_NS + int(milliseconds * 1_000_000),
        (l2.MBPLevel(bid_px, bid_size, bid_count), l2.MBPLevel(bid_px - 0.25, 80, 2), l2.MBPLevel(bid_px - 0.5, 60, 1)),
        (l2.MBPLevel(ask_px, ask_size, ask_count), l2.MBPLevel(ask_px + 0.25, 80, 2), l2.MBPLevel(ask_px + 0.5, 60, 1)),
    )


def update(milliseconds: float, side: str, delta: int, count_delta: int, kind: str) -> l2.MBP10Update:
    return l2.MBP10Update(BASE_NS + int(milliseconds * 1_000_000), side, PX, delta, count_delta, kind)  # type: ignore[arg-type]


def active_interaction(*, direction: str = "BUYER_ABSORPTION", config: l2.L2Config | None = None) -> tuple[l2.L2InteractionEngine, l2.L2Interaction]:
    engine = l2.L2InteractionEngine([l2.StructuralLevel("PRIOR_RTH_POC", PX)], config or l2.L2Config(min_relevant_aggressive_volume=1, min_relevant_execution_count=1, min_quality_score=0.0))
    side, aggressor = ("B", "SELL") if direction == "BUYER_ABSORPTION" else ("A", "BUY")
    engine.observe_snapshot(snapshot(0, defended_side=side))
    engine.observe_execution(execution(1, aggressor=aggressor))
    engine.observe_snapshot(snapshot(2, defended_side=side))
    engine.observe_execution(execution(3, aggressor=aggressor))
    interaction = next(iter(engine.active.values()))
    assert interaction.defended_side == side
    return engine, interaction


def complete_consumption_restore(*, direction: str = "BUYER_ABSORPTION") -> l2.L2Interaction:
    engine, interaction = active_interaction(direction=direction)
    side = interaction.defended_side
    engine.observe_snapshot(snapshot(4, bid_size=40 if side == "B" else 100, ask_size=40 if side == "A" else 100,
                                     bid_count=1 if side == "B" else 2, ask_count=1 if side == "A" else 2, defended_side=side), update(4, side, -60, -1, "FILL"))
    engine.observe_snapshot(snapshot(5, bid_size=100, ask_size=100, bid_count=2, ask_count=2, defended_side=side), update(5, side, 60, 1, "ADD"))
    engine.finish_rth(BASE_NS + 20_000_000)
    return engine.completed[0]


def test_mbo_converter_exports_only_aggregate_mbp10_fields_and_no_order_identity():
    view = l2.MBOToMBP10View()
    snap, _ = view.apply(l2.MBOEvent(BASE_NS, "A", "B", PX, 30, 991))
    snap, _ = view.apply(l2.MBOEvent(BASE_NS + 1, "A", "B", PX, 20, 992))
    snap, public_update = view.apply(l2.MBOEvent(BASE_NS + 2, "A", "A", PX + 0.25, 40, 993))
    assert snap.bids[0].size == 50 and snap.bids[0].order_count == 2
    assert snap.bid_px[0] == PX and snap.bid_sz[0] == 50 and snap.bid_ct[0] == 2
    assert snap.ask_px[0] == PX + 0.25 and snap.ask_sz[0] == 40 and snap.ask_ct[0] == 1
    assert public_update is not None and "order_id" not in l2.public_l2_field_names()["mbp_update"]
    assert all("order_id" not in names for names in l2.public_l2_field_names().values())


def test_passive_depth_addition_never_starts_interaction_or_counts_as_refill():
    engine = l2.L2InteractionEngine([l2.StructuralLevel("PRIOR_RTH_HIGH", PX)])
    engine.observe_snapshot(snapshot(0), update(0, "B", 100, 1, "ADD"))
    assert not engine.active and not engine.completed
    _, interaction = active_interaction()
    interaction.observe_snapshot(snapshot(4, bid_size=140), update(4, "B", 40, 1, "ADD"))
    assert interaction.depth_restoration_count == 0
    assert interaction.unexecuted_add_volume == 40


def test_execution_then_same_price_depth_restoration_is_genuine_refill_with_latency():
    interaction = complete_consumption_restore()
    features = interaction.feature_inputs()
    assert interaction.depth_restoration_count == interaction.consume_restore_cycles == 1
    assert interaction.restored_depth_volume == 60
    assert features["mean_restoration_latency_ms"] == pytest.approx(2.0)
    assert features["restoration_timestamp_ns"] == BASE_NS + 5_000_000
    assert features["restoration_supported_by_execution_ratio"] == 1.0


def test_multiple_consume_restore_cycles_and_order_count_cycles_are_counted():
    engine, interaction = active_interaction()
    engine.observe_snapshot(snapshot(4, bid_size=40, bid_count=1), update(4, "B", -60, -1, "FILL"))
    engine.observe_snapshot(snapshot(5, bid_size=100, bid_count=2), update(5, "B", 60, 1, "ADD"))
    engine.observe_execution(execution(7, aggressor="SELL"))
    engine.observe_snapshot(snapshot(8, bid_size=40, bid_count=1), update(8, "B", -60, -1, "FILL"))
    engine.observe_snapshot(snapshot(9, bid_size=100, bid_count=2), update(9, "B", 60, 1, "ADD"))
    assert interaction.consume_restore_cycles == 2
    assert interaction.order_count_restoration_cycles == 2
    assert interaction.restored_order_count == 2


def test_execution_displayed_ratios_and_buyer_seller_symmetry():
    buyer, seller = complete_consumption_restore(direction="BUYER_ABSORPTION"), complete_consumption_restore(direction="SELLER_ABSORPTION")
    for item in (buyer, seller):
        f = item.feature_inputs()
        assert f["cumulative_executed_at_price"] == 100
        assert f["executed_to_initial_displayed_ratio"] == 1.0
        assert f["executed_to_median_displayed_ratio"] == 1.0
        assert f["restoration_to_consumption_ratio"] == 1.0


def test_large_aggression_with_limited_progress_scores_higher_than_large_progress():
    good = complete_consumption_restore()
    bad = complete_consumption_restore()
    bad.min_price = PX - 2.0
    bad.end_price = PX - 1.5
    assert good.quality()["l2_absorption_quality_score"] > bad.quality()["l2_absorption_quality_score"]
    assert good.component_scores()["price_resistance_score"] > bad.component_scores()["price_resistance_score"]


def test_depth_persistence_time_weighting_and_fixed_recovery_windows_are_causal():
    engine, interaction = active_interaction()
    engine.observe_snapshot(snapshot(4, bid_size=40), update(4, "B", -60, -1, "FILL"))
    engine.observe_snapshot(snapshot(105, bid_size=80), update(105, "B", 40, 0, "ADD"))
    engine.observe_snapshot(snapshot(254, bid_size=100), update(254, "B", 20, 0, "ADD"))
    f = interaction.feature_inputs()
    assert f["defended_price_present_fraction"] == 1.0
    assert f["defended_depth_time_weighted_mean"] > 0
    assert f["depth_recovery_100ms"] == pytest.approx(0.8)
    assert f["depth_recovery_250ms"] == pytest.approx(1.0)


def test_depth_imbalance_and_multilevel_ofi_are_deterministic():
    engine, interaction = active_interaction()
    engine.observe_snapshot(snapshot(4, bid_size=160, ask_size=40), update(4, "B", 60, 0, "ADD"))
    first = interaction.feature_inputs()
    engine.observe_snapshot(snapshot(5, bid_size=160, ask_size=40), None)
    second = interaction.feature_inputs()
    assert first["depth_imbalance_1"] > 0 and first["depth_imbalance_5"] > 0
    assert first["multi_level_ofi"] == second["multi_level_ofi"]


def test_rapid_unexecuted_add_cancel_penalty_does_not_penalize_real_restoration():
    _, interaction = active_interaction()
    interaction.observe_snapshot(snapshot(4, bid_size=180), update(4, "B", 80, 1, "ADD"))
    interaction.observe_snapshot(snapshot(5, bid_size=100), update(5, "B", -80, -1, "CANCEL"))
    penalized = interaction.feature_inputs()
    genuine = complete_consumption_restore().feature_inputs()
    assert penalized["rapid_cancel_volume"] == 80 and penalized["rapid_cancel_ratio"] == 1.0
    assert genuine["rapid_cancel_volume"] == 0 and genuine["restoration_supported_by_execution_ratio"] == 1.0


def test_restoration_away_from_the_defended_price_is_a_false_refill_guard():
    _, interaction = active_interaction()
    interaction.observe_snapshot(snapshot(4), l2.MBP10Update(BASE_NS + 4_000_000, "B", PX - 0.25, 25, 1, "ADD"))
    assert interaction.feature_inputs()["restoration_away_from_defended_price_volume"] == 25


def test_interaction_requires_execution_near_level_not_passive_book_activity():
    engine = l2.L2InteractionEngine([l2.StructuralLevel("PRIOR_RTH_VAL", PX)])
    for offset in range(5): engine.observe_snapshot(snapshot(offset, bid_size=100 + offset), update(offset, "B", 1, 0, "ADD"))
    assert not engine.active
    engine.observe_execution(execution(6, aggressor="SELL"))
    assert len(engine.active) == 1


def test_setup_requires_all_l2_evidence_families_and_explains_rejection():
    config = l2.L2Config(min_relevant_aggressive_volume=1, min_relevant_execution_count=1, min_quality_score=0.0)
    signal = l2.L2SignalEngine(config)
    engine = l2.L2InteractionEngine([l2.StructuralLevel("PRIOR_RTH_POC", PX)], config)
    engine.observe_execution(execution(0, aggressor="SELL")); engine.finish_rth(BASE_NS + 1)
    assert signal.register_completed(engine.completed[0]) is None
    assert signal.rejections[0]["reasons"] == ("NO_GENUINE_CONSUME_RESTORE",)
    accepted = signal.register_completed(complete_consumption_restore())
    assert accepted is not None


def _qualified_setup(identifier: str = "one") -> tuple[l2.L2SignalEngine, l2.L2Setup]:
    signal = l2.L2SignalEngine(l2.L2Config(min_relevant_aggressive_volume=1, min_relevant_execution_count=1, min_quality_score=0.0))
    interaction = complete_consumption_restore(); interaction.interaction_id = identifier
    setup = signal.register_completed(interaction)
    assert setup is not None
    return signal, setup


def test_frozen_confirmation_boundaries_three_ticks_and_two_ms_latency():
    signal, setup = _qualified_setup()
    end = setup.interaction.end_ns or BASE_NS
    signal.observe_execution(l2.Execution(end + 4_999_000_000, PX + 0.75, 1, "BUY"))
    assert setup.state == "WAIT_MIN_CONFIRMATION_TIME"
    signal.observe_execution(l2.Execution(end + 5_000_000_000, PX + 0.50, 1, "BUY"))
    assert setup.state == "WAIT_MIN_CONFIRMATION_TIME"
    signal.observe_execution(l2.Execution(end + 5_000_000_001, PX + 0.75, 1, "BUY"))
    assert setup.state == "CONFIRMED" and setup.entry_ready_ns == end + 5_002_000_001
    signal.observe_execution(l2.Execution(end + 15_000_000_000, PX + 1.0, 1, "BUY"))
    assert setup.confirmation_price == PX + 0.75
    expired_signal, expired = _qualified_setup()
    expired_signal.observe_execution(l2.Execution((expired.interaction.end_ns or BASE_NS) + 15_000_000_001, PX + 0.75, 1, "BUY"))
    assert expired.terminal_reason == "CONFIRMATION_WINDOW_EXPIRED"


def test_zone_stop_three_r_target_es_first_mes_fallback_and_position_limit():
    long = l2.initial_prices("BUYER_ABSORPTION", PX, PX + 0.25, PX - 0.5, PX + 0.5)
    short = l2.initial_prices("SELLER_ABSORPTION", PX, PX + 0.25, PX - 0.5, PX + 0.5)
    assert long["stop"] == PX - 1.75 and long["target"] == long["entry"] + 3 * (long["entry"] - long["stop"])
    assert short["stop"] == PX + 1.75 and short["target"] == short["entry"] - 3 * (short["stop"] - short["entry"])
    signal, first = _qualified_setup("one"); _, second = _qualified_setup("two"); signal.pending[second.setup_id] = second
    end = first.interaction.end_ns or BASE_NS
    signal.observe_execution(l2.Execution(end + 5_000_000_000, PX + 0.75, 1, "BUY"))
    pending = signal.pending[first.setup_id]
    position = signal.try_enter(first.setup_id, timestamp_ns=pending.entry_ready_ns or 0, es_bid=PX + 0.5, es_ask=PX + 0.75)
    assert position is not None and position.instrument == "ES" and position.contracts <= l2.ES_CAP
    assert signal.try_enter(second.setup_id, timestamp_ns=second.entry_ready_ns or 0, es_bid=PX + 0.5, es_ask=PX + 0.75) is None
    assert second.terminal_reason == "COMPLIANCE_BLOCK_ACTIVE_POSITION"


def test_l2_module_never_imports_databento_downloads_or_l3_ground_truth():
    source = Path(l2.__file__).read_text(encoding="utf-8").lower()
    assert "import databento" not in source and "from databento" not in source
    assert "timeseries.get_range" not in source and "download" not in source
    assert "cme_orderflow_absorption_v1" not in source
    assert not hasattr(l2.Execution, "order_id") and not hasattr(l2.MBPLevel, "order_id")


def test_book_price_validation_accepts_far_normalized_depth_but_rejects_raw_fixed_point():
    assert l2.MBPLevel(100.0, 1, 1).price == 100.0
    with pytest.raises(l2.L2ValidationError, match="normalized ES point prices"):
        l2.MBPLevel(7_248_500_000_000.0, 1, 1)


def test_snapshot_rendering_can_be_deferred_until_the_completed_f_last_boundary():
    view = l2.MBOToMBP10View()
    snap, _ = view.apply(l2.MBOEvent(BASE_NS, "R", "B", PX, 0, 0), materialize_snapshot=False)
    assert snap is None
    snap, _ = view.apply(l2.MBOEvent(BASE_NS + 1, "A", "B", PX, 10, 1), materialize_snapshot=False)
    assert snap is None
    snap, _ = view.apply(l2.MBOEvent(BASE_NS + 2, "A", "A", PX + .25, 10, 2))
    assert snap is not None and snap.bids[0].size == 10 and snap.asks[0].size == 10


def test_private_pre_rth_book_reconstruction_can_defer_public_snapshot_materialization():
    adapter = historical.HistoricalMBOToMBP10Adapter()
    assert adapter.feed(_private(0, "R", "B", size=0, order_id=0, flags=historical.F_SNAPSHOT), materialize_public=False) is None
    assert adapter.feed(_private(1, "A", "B", size=10, order_id=1, flags=historical.F_SNAPSHOT), materialize_public=False) is None
    assert adapter.feed(_private(2, "A", "A", price=PX + .25, size=10, order_id=2,
                                flags=historical.F_SNAPSHOT | historical.F_LAST), materialize_public=False) is None
    public = adapter.feed(_private(3, "T", "A", price=PX + .25), materialize_public=True)
    assert adapter.book_valid and public is not None and public.snapshot.bids[0].size == 10


def _private(milliseconds: float, action: str, side: str, *, price: float = PX,
             size: int = 10, order_id: int = 1, flags: int = 0) -> historical.PrivateMBORecord:
    return historical.PrivateMBORecord(BASE_NS + int(milliseconds * 1_000_000), action, side, price, size, order_id, flags)  # type: ignore[arg-type]


def test_frozen_l2_contract_is_exact_and_has_no_development_only_values():
    contract = historical.frozen_contract()
    assert contract["strategy_id"] == "CMEOrderflowAbsorption.ES_L2_V1"
    assert contract["weights_label"] == "L2_V1_PREDECLARED_RESEARCH_WEIGHTS"
    assert contract["weights"] == {
        "aggression_score": 0.28, "restoration_score": 0.25,
        "price_resistance_score": 0.22, "persistence_score": 0.12,
        "multi_level_support_score": 0.13, "false_refill_penalty": 0.25,
    }
    assert contract["development_only_parameters"] == []
    assert contract["execution"] == {
        "confirmation_window_seconds": [5.0, 15.0], "confirmation_favorable_ticks": 3,
        "entry_latency_ms": 2.0, "stop_buffer_ticks": 5, "target_r": 3.0,
        "risk_budget_usd": 250.0, "es_first": True, "mes_fallback": True,
        "max_es_contracts": 6, "max_mes_contracts": 60,
    }


def test_historical_adapter_requires_r_a_f_last_before_public_book_is_usable():
    adapter = historical.HistoricalMBOToMBP10Adapter()
    with pytest.raises(historical.HistoricalReplayError, match="missing causal"):
        adapter.feed(_private(1, "A", "B"))
    assert adapter.feed(_private(2, "R", "B", size=0, order_id=0, flags=historical.F_SNAPSHOT)) is None
    assert adapter.feed(_private(3, "A", "B", flags=historical.F_SNAPSHOT)) is None
    event = adapter.feed(_private(4, "A", "A", price=PX + .25, order_id=2, flags=historical.F_SNAPSHOT | historical.F_LAST))
    assert event is not None and adapter.book_valid and event.execution is None
    assert adapter.feed(_private(5, "F", "B", size=1)).execution is not None
    adapter.finish()


def test_execution_fill_does_not_remove_displayed_depth_before_its_later_cancel():
    adapter = historical.HistoricalMBOToMBP10Adapter()
    adapter.feed(_private(0, "R", "B", size=0, order_id=0, flags=historical.F_SNAPSHOT))
    adapter.feed(_private(1, "A", "B", size=10, order_id=11, flags=historical.F_SNAPSHOT))
    adapter.feed(_private(2, "A", "A", price=PX + .25, size=10, order_id=12,
                          flags=historical.F_SNAPSHOT | historical.F_LAST))
    fill = adapter.feed(_private(3, "F", "B", size=5, order_id=11))
    assert fill is not None and fill.snapshot.bids[0].size == 10
    cancel = adapter.feed(_private(4, "C", "B", size=5, order_id=11))
    assert cancel is not None and cancel.snapshot.bids[0].size == 5


def test_incomplete_snapshot_and_ordinary_reset_fail_closed():
    adapter = historical.HistoricalMBOToMBP10Adapter()
    adapter.feed(_private(0, "R", "B", size=0, order_id=0, flags=historical.F_SNAPSHOT))
    with pytest.raises(historical.HistoricalReplayError, match="incomplete"):
        adapter.finish()
    adapter = historical.HistoricalMBOToMBP10Adapter()
    adapter.feed(_private(0, "R", "B", size=0, order_id=0, flags=historical.F_SNAPSHOT))
    adapter.feed(_private(1, "A", "B", flags=historical.F_SNAPSHOT | historical.F_LAST))
    with pytest.raises(historical.HistoricalReplayError, match="ordinary reset"):
        adapter.feed(_private(2, "R", "B", size=0, order_id=0))


def test_historical_adapter_aggregates_top_ten_and_strips_order_identity_before_strategy():
    adapter = historical.HistoricalMBOToMBP10Adapter()
    adapter.feed(_private(0, "R", "B", size=0, order_id=0, flags=historical.F_SNAPSHOT))
    for index in range(12):
        adapter.feed(_private(1 + index, "A", "B", price=PX - .25 * index, size=10, order_id=index + 1,
                              flags=historical.F_SNAPSHOT | (historical.F_LAST if index == 11 else 0)))
    event = adapter.feed(_private(20, "T", "A", price=PX + .25, order_id=991))
    assert event is not None and len(event.snapshot.bids) == 10
    assert event.execution is not None and not hasattr(event.execution, "order_id")
    historical.assert_no_order_identity_in_strategy_layer()


def test_incremental_completed_interactions_are_not_resummarized_without_new_completion(monkeypatch):
    runner = historical.HistoricalL2Runner(date="2026-05-04", evidence_label=historical.MAY_LABEL,
                                           levels=[l2.StructuralLevel("PRIOR_RTH_POC", PX)])
    completed = complete_consumption_restore()
    calls = {"features": 0}
    original = completed.feature_inputs

    def counted():
        calls["features"] += 1
        return original()

    monkeypatch.setattr(completed, "feature_inputs", counted)
    runner.interactions.completed.append(completed)
    runner._new_completed()
    first = calls["features"]
    runner._new_completed()
    assert calls["features"] == first and runner.completed_seen == 1
    next_completed = complete_consumption_restore()
    next_calls = {"features": 0}
    next_original = next_completed.feature_inputs
    monkeypatch.setattr(next_completed, "feature_inputs", lambda: next_calls.__setitem__("features", next_calls["features"] + 1) or next_original())
    runner.interactions.completed.append(next_completed)
    runner._new_completed()
    assert calls["features"] == first and next_calls["features"] > 0 and runner.completed_seen == 2


def test_setup_ledger_keeps_deterministic_rejection_reasons_and_immutable_output_schema(tmp_path):
    runner = historical.HistoricalL2Runner(date="2026-05-04", evidence_label=historical.MAY_LABEL,
                                           levels=[l2.StructuralLevel("PRIOR_RTH_POC", PX)])
    engine = l2.L2InteractionEngine([l2.StructuralLevel("PRIOR_RTH_POC", PX)])
    engine.observe_execution(execution(0, size=1)); engine.finish_rth(BASE_NS + 1)
    runner.interactions.completed.append(engine.completed[0]); runner._new_completed()
    assert runner.setup_ledger[0]["accepted"] is False
    assert runner.setup_ledger[0]["rejection_reasons"]
    output = tmp_path / "new-output"
    summary = historical.write_future_artifacts(output, [runner])
    assert (output / "summary.json").is_file() and (output / "setup-ledger.csv").is_file()
    assert summary["first_run_policy"] == "FIRST_BROAD_HISTORICAL_L2_V1_REPLAY"
    with pytest.raises(historical.HistoricalReplayError, match="already exists"):
        historical.write_future_artifacts(output, [runner])


def test_inventory_is_filesystem_manifest_only_and_preserves_evidence_labels(tmp_path):
    root = tmp_path
    may = root / "data/cme_orderflow_absorption_v2/may_2026_cost_proxy"
    may.mkdir(parents=True)
    (may / "acquisition-manifest.json").write_text('{"request_identity":{"target_rth_dates":["2026-05-04"]}}', encoding="utf-8")
    (may / "es_mbo").mkdir(); (may / "mes_mbp1").mkdir()
    (may / "es_mbo/ESM6_2026-05-04_000000_224501_mbo.dbn.zst").write_bytes(b"not read")
    (may / "mes_mbp1/MESM6_2026-05-04_133000_224501_mbp1.dbn.zst").write_bytes(b"not read")
    rows = historical.inventory_existing_sessions(root)
    assert len(rows) == 1 and rows[0]["evidence_period_label"] == historical.MAY_LABEL
    assert rows[0]["full_execution_replay"] is True
    assert "databento" not in historical.inventory_existing_sessions.__code__.co_names


def test_historical_runner_preserves_es_first_mes_fallback_and_one_position_blocking():
    runner = historical.HistoricalL2Runner(date="2026-05-04", evidence_label=historical.MAY_LABEL,
                                           levels=[l2.StructuralLevel("PRIOR_RTH_POC", PX)])
    _, setup = _qualified_setup("mes-fallback")
    setup.interaction.zone_low = PX - 10.0  # ES cannot fit $250; MES can.
    setup.state = "CONFIRMED"; setup.entry_ready_ns = BASE_NS
    runner.signals.pending[setup.setup_id] = setup
    runner.es_quote, runner.mes_quote = (PX, PX + .25), (PX, PX + .25)
    runner._attempt_entry(BASE_NS)
    assert runner.signals.position is not None and runner.signals.position.instrument == "MES"
    _, blocked = _qualified_setup("blocked")
    blocked.state = "CONFIRMED"; blocked.entry_ready_ns = BASE_NS
    runner.signals.pending[blocked.setup_id] = blocked
    runner._attempt_entry(BASE_NS)
    assert blocked.terminal_reason == "COMPLIANCE_BLOCK_ACTIVE_POSITION"


def test_historical_runner_has_no_network_or_databento_historical_client_path():
    source = Path(historical.__file__).read_text(encoding="utf-8")
    assert "Historical(" not in source and "timeseries.get_range" not in source
    assert "metadata.get_cost" not in source and "requests." not in source


def _coverage(day: str, *, es_end: str = "2026-05-08T22:45:01Z", mes_end: str = "2026-05-08T22:45:01Z") -> dict[str, dict[str, str]]:
    return {
        "ES_MBO_L3": {"start": f"{day}T00:00:00Z", "end": es_end},
        "MES_NATIVE_EXECUTION": {"start": f"{day}T13:30:00Z", "end": mes_end},
    }


def test_declared_coverage_not_last_market_event_controls_quiet_cutoff_validity():
    coverage = historical.validate_declared_session_coverage("2026-05-08", _coverage("2026-05-08"))
    assert coverage["ES_MBO_L3"]["declared_end_ns"] > historical._clock_ns("2026-05-08", historical.HARD_CUTOFF_SECONDS)


def test_declared_coverage_before_cutoff_or_unknown_fails_closed():
    with pytest.raises(historical.HistoricalReplayError, match="ends before frozen"):
        historical.validate_declared_session_coverage("2026-05-08", _coverage("2026-05-08", es_end="2026-05-08T22:44:00Z"))
    with pytest.raises(historical.HistoricalReplayError, match="coverage is unknown"):
        historical.validate_declared_session_coverage("2026-05-08", {"ES_MBO_L3": _coverage("2026-05-08")["ES_MBO_L3"]})
    with pytest.raises(historical.HistoricalReplayError, match="coverage is unknown"):
        historical.validate_declared_session_coverage("2026-05-08", {"ES_MBO_L3": {"start": "2026-05-08T00:00:00Z", "end": None}, "MES_NATIVE_EXECUTION": _coverage("2026-05-08")["MES_NATIVE_EXECUTION"]})
    with pytest.raises(historical.HistoricalReplayError, match="chronology mismatch"):
        historical.validate_declared_session_coverage("2026-05-08", _coverage("2026-05-08", es_end="2026-05-08T00:00:00Z"))


def test_quiet_valid_coverage_does_not_require_an_exact_cutoff_event_when_flat():
    runner = historical.HistoricalL2Runner(date="2026-05-08", evidence_label=historical.MAY_LABEL,
                                           levels=[l2.StructuralLevel("PRIOR_RTH_POC", PX)])
    historical.validate_declared_session_coverage("2026-05-08", _coverage("2026-05-08"))
    runner.force_flat_from_last_causal_cutoff_quote(historical._clock_ns("2026-05-08", historical.HARD_CUTOFF_SECONDS))
    assert runner.trade_ledger == []


def test_open_position_requires_frozen_inclusive_cutoff_bbo_and_flats_at_that_causal_timestamp():
    cutoff = historical._clock_ns("2026-05-08", historical.HARD_CUTOFF_SECONDS)
    runner = historical.HistoricalL2Runner(date="2026-05-08", evidence_label=historical.MAY_LABEL,
                                           levels=[l2.StructuralLevel("PRIOR_RTH_POC", PX)])
    _, setup = _qualified_setup("cutoff")
    setup.state, setup.entry_ready_ns = "CONFIRMED", cutoff - 900_000_000
    runner.signals.pending[setup.setup_id] = setup
    runner.es_quote, runner.es_quote_timestamp_ns = (PX, PX + .25), cutoff - 500_000_000
    runner._attempt_entry(cutoff - 500_000_000)
    assert runner.signals.position is not None
    runner.force_flat_from_last_causal_cutoff_quote(cutoff)
    assert runner.signals.position is None and runner.trade_ledger[-1]["exit_reason"] == "HARD_CUTOFF_2245"
    assert runner.trade_ledger[-1]["exit_timestamp_ns"] == cutoff - 500_000_000
    runner.signals.position = l2.L2Position(setup, "ES", 1, l2.initial_prices("BUYER_ABSORPTION", PX, PX + .25, PX - .5, PX + .5), cutoff)
    runner.es_quote_timestamp_ns = cutoff - historical.CUTOFF_QUOTE_LOOKBACK_NS - 1
    with pytest.raises(historical.HistoricalReplayError, match="inclusive causal BBO"):
        runner.force_flat_from_last_causal_cutoff_quote(cutoff)


def test_frozen_strategy_contract_hash_is_unchanged_by_coverage_repair():
    import hashlib
    import json
    encoded = json.dumps(historical.frozen_contract(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert hashlib.sha256(encoded).hexdigest() == "dd71a7b501d762a17851f7187c7faef13a47031cf886fbf0484e2dfbd38126cc"


def test_source_mbo_price_normalizes_an_ordinary_es_add_exactly_once():
    assert historical.normalize_source_mbo_price(7_503_000_000_000, "A") == 7503.0
    assert historical.normalize_source_mbo_price(100_000_000_000, "A") == 100.0
    assert historical.normalize_source_mbo_price(99_999_000_000_000, "A") == 99_999.0


def test_source_mbo_price_retains_finite_may14_value_privately_and_keeps_undefined_price_strict():
    assert historical.normalize_source_mbo_price(752_000_000_000_000, "A") == 752000.0
    with pytest.raises(historical.HistoricalReplayError, match="MBO_UNDEFINED_PRICE_NON_RESET"):
        historical.normalize_source_mbo_price(historical.UNDEF_PRICE, "A")
    assert historical.normalize_source_mbo_price(historical.UNDEF_PRICE, "R") == 5000.0


def _adapter_with_ten_normal_ask_levels() -> historical.HistoricalMBOToMBP10Adapter:
    adapter = historical.HistoricalMBOToMBP10Adapter()
    adapter.feed(_private(0, "R", "B", size=0, order_id=0, flags=historical.F_SNAPSHOT))
    adapter.feed(_private(1, "A", "B", price=PX, size=10, order_id=1, flags=historical.F_SNAPSHOT))
    adapter.feed(_private(2, "A", "A", price=PX + .25, size=10, order_id=2,
                          flags=historical.F_SNAPSHOT | historical.F_LAST))
    for index in range(3, 12):
        adapter.feed(_private(index, "A", "A", price=PX + .25 * (index - 1), size=10, order_id=index))
    return adapter


def test_offbook_finite_ask_is_private_only_and_later_cancel_is_audited():
    adapter = _adapter_with_ten_normal_ask_levels()
    before = adapter.feed(_private(20, "T", "A", price=PX + .25, order_id=900))
    assert before is not None
    outlier = historical.PrivateMBORecord(BASE_NS + 21_000_000, "A", "A", 752000.0, 1, 77, 128, 752_000_000_000_000)
    event = adapter.feed(outlier)
    assert event is not None and event.update is None and event.execution is None
    assert event.snapshot.bids == before.snapshot.bids and event.snapshot.asks == before.snapshot.asks
    assert all(level.price != 752000.0 for level in event.snapshot.asks)
    assert event.snapshot.bids[0].price == before.snapshot.bids[0].price
    adapter.feed(historical.PrivateMBORecord(BASE_NS + 22_000_000, "C", "A", 752000.0, 1, 77, 128, 752_000_000_000_000))
    assert adapter._view.order(77) is None
    audit = adapter.source_integrity_diagnostics()
    assert audit == [{"timestamp_ns": BASE_NS + 21_000_000, "action": "A", "side": "A", "raw_price": 752_000_000_000_000,
                      "normalized_price": 752000.0, "size": 1, "flags": 128, "entered_top_ten": False,
                      "affected_bbo": False, "terminal_action": "C"}]


def test_offbook_anomaly_that_would_enter_top_ten_or_bbo_fails_closed():
    adapter = historical.HistoricalMBOToMBP10Adapter()
    adapter.feed(_private(0, "R", "B", size=0, order_id=0, flags=historical.F_SNAPSHOT))
    adapter.feed(_private(1, "A", "B", price=PX, size=10, order_id=1, flags=historical.F_SNAPSHOT))
    adapter.feed(_private(2, "A", "A", price=PX + .25, size=10, order_id=2,
                          flags=historical.F_SNAPSHOT | historical.F_LAST))
    with pytest.raises(historical.HistoricalReplayError, match="OFFBOOK_ANOMALY_EXPOSED"):
        adapter.feed(historical.PrivateMBORecord(BASE_NS + 3_000_000, "A", "A", 752000.0, 1, 77, 128, 752_000_000_000_000))


def test_offbook_anomaly_audit_limit_is_deterministic():
    adapter = _adapter_with_ten_normal_ask_levels()
    for index in range(historical.MAX_TOLERATED_OFFBOOK_ANOMALIES_PER_SESSION):
        adapter.feed(historical.PrivateMBORecord(BASE_NS + 30_000_000 + index, "A", "A", 752000.0 + index, 1,
                                                 100 + index, 128, 752_000_000_000_000 + index))
    with pytest.raises(historical.HistoricalReplayError, match="ANOMALY_LIMIT_EXCEEDED"):
        adapter.feed(historical.PrivateMBORecord(BASE_NS + 50_000_000, "A", "A", 752100.0, 1, 200, 128, 752_100_000_000_000))


def test_artifact_writer_keeps_source_anomalies_in_a_separate_audit_file(tmp_path: Path):
    runner = historical.HistoricalL2Runner(
        date="2026-05-14", evidence_label=historical.MAY_LABEL,
        levels=[l2.StructuralLevel("PRIOR_RTH_POC", PX)],
    )
    runner.source_integrity_diagnostics = [{
        "timestamp_ns": BASE_NS, "action": "A", "side": "A",
        "raw_price": 752_000_000_000_000, "normalized_price": 752000.0,
        "size": 1, "flags": 128, "entered_top_ten": False,
        "affected_bbo": False, "terminal_action": "C",
    }]
    output = tmp_path / "immutable-output"
    historical.write_future_artifacts(output, [runner])
    payload = json.loads((output / "source-integrity-diagnostics.json").read_text(encoding="utf-8"))
    assert payload["policy"] == "RETAIN_PRIVATE_FAIL_CLOSED_IF_STRATEGY_VISIBLE"
    assert payload["max_tolerated_offbook_anomalies_per_session"] == 10
    assert payload["sessions"][0]["anomalies"] == runner.source_integrity_diagnostics


def _funnel_interaction(interaction_id: str, *, volume: float = 100, executions: float = 2,
                        cycles: float = 1, progress: float = 0, rejection: float = 0,
                        quality: float = .60) -> dict[str, object]:
    return {
        "interaction_id": interaction_id, "date": "2026-05-04",
        "directional_aggressive_volume": volume, "relevant_execution_count": executions,
        "consume_restore_cycles": cycles, "maximum_through_level_progress_ticks": progress,
        "interaction_rejection_ticks": rejection, "l2_absorption_quality_score": quality,
        "rejection_reasons": "", "aggression_score": .7, "restoration_score": .7,
        "price_resistance_score": .5, "persistence_score": .5,
        "multi_level_support_score": .5, "false_refill_penalty": 0,
        "depth_restoration_count": cycles, "cumulative_restored_volume": 10,
        "restoration_to_consumption_ratio": 1, "mean_restoration_latency_ms": 10,
        "defended_price_present_fraction": .5, "defended_depth_time_weighted_mean": 10,
        "rapid_cancel_ratio": 0, "unexecuted_add_volume": 0,
        "interaction_start_ns": 1, "interaction_end_ns": 2, "level": "PRIOR_RTH_POC",
        "direction": "BUYER_ABSORPTION",
    }


def _funnel_setup(row: dict[str, object], *, accepted: bool, status: str = "REJECTED") -> dict[str, object]:
    return {**row, "accepted": accepted, "confirmation_status": status,
            "terminal_reason": "ENTRY" if status == "ENTRY" else "REJECTED"}


def test_rejection_funnel_counts_independent_and_sequential_failures_with_combinations():
    interactions = [
        _funnel_interaction("accepted"),
        _funnel_interaction("volume", volume=49),
        _funnel_interaction("restore", cycles=0),
        _funnel_interaction("rejection", progress=5),
        _funnel_interaction("quality", quality=.54),
        _funnel_interaction("multi", volume=49, executions=1, quality=.54),
    ]
    setups = [_funnel_setup(row, accepted=row["interaction_id"] == "accepted", status="ENTRY" if row["interaction_id"] == "accepted" else "REJECTED") for row in interactions]
    result = funnel.analyze_rows(interactions, setups, [{"interaction_id": "accepted"}])
    assert result["total_interactions"] == 6
    assert result["independent_gate_failures"] == {
        "relevant_aggressive_volume": 2, "relevant_execution_count": 1,
        "consume_restore": 1, "rejection": 1, "quality": 2,
    }
    assert [row["count"] for row in result["funnel"]] == [6, 4, 4, 3, 2, 2, 1, 1, 1, 1]
    assert result["one_gate_near_misses"]["count"] == 4
    assert "AGGRESSIVE_VOLUME + EXECUTION_COUNT + QUALITY" in {
        row["combination"] for row in result["top_rejection_combinations"]
    }


def test_rejection_funnel_near_miss_distance_and_confirmation_population_are_ledger_only():
    interactions = [_funnel_interaction("quality", quality=.50), _funnel_interaction("failed-confirmation")]
    setups = [_funnel_setup(interactions[0], accepted=False), _funnel_setup(interactions[1], accepted=True, status="FAILED")]
    result = funnel.analyze_rows(interactions, setups, [])
    quality = result["one_gate_near_misses"]["by_gate"]["quality"]
    assert quality["count"] == 1
    assert quality["distance_distribution"]["median"] == pytest.approx(.05)
    assert result["accepted_setups"][0]["interaction_id"] == "failed-confirmation"
    assert result["confirmation"] == {
        "accepted": 1, "passed": 0, "failed": 1, "trades": 0,
        "failed_price_path_mechanics": [{
            "interaction_id": "failed-confirmation", "status": "UNAVAILABLE_FROM_PUBLISHED_ARTIFACTS",
            "reason": "setup-ledger records only terminal confirmation status; it has no +5s..+15s execution-path observations",
        }],
    }
    assert "DBNStore" not in Path(funnel.__file__).read_text(encoding="utf-8")


def test_rejection_funnel_materializes_only_synthetic_csv_ledgers(tmp_path: Path):
    row = _funnel_interaction("accepted")
    root = tmp_path / "published"
    root.mkdir()
    setup = _funnel_setup(row, accepted=True, status="ENTRY")
    for name, rows in (("interaction-features.csv", [row]), ("setup-ledger.csv", [setup]), ("trade-ledger.csv", [{"interaction_id": "accepted"}])):
        names = sorted({key for item in rows for key in item})
        with (root / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=names)
            writer.writeheader(); writer.writerows(rows)
    (root / "summary.json").write_text("{}\n", encoding="utf-8")
    result = funnel.materialize(root)
    assert result["total_interactions"] == 1
    assert (root / "rejection_funnel" / "summary.json").is_file()
    assert (root / "rejection_funnel" / "funnel.csv").is_file()
    assert (root / "rejection_funnel" / "rejection-reasons.csv").is_file()
    assert (root / "rejection_funnel" / "near-misses.csv").is_file()


def test_l2_v2_changes_only_the_frozen_quality_threshold_and_hash_is_deterministic():
    diff = v2.v1_v2_config_diff()
    assert v2.V1_CONFIG.min_quality_score == .55
    assert v2.V2_CONFIG.min_quality_score == .50
    assert diff["changed_strategy_fields"] == [{"field": "min_quality_score", "v1": .55, "v2": .50}]
    assert v2.v2_contract_sha256() == v2.v2_contract_sha256()
    assert v2.v2_contract()["strategy_configuration_diff"]["only_min_quality_score_differs"] is True


def test_l2_v2_qualification_keeps_all_gates_except_quality_threshold():
    marginal = _funnel_interaction("marginal", quality=.52)
    blocked = _funnel_interaction("blocked", quality=.52, cycles=0)
    assert all(v for name, v in funnel.gate_status(marginal, v2.V2_CONFIG).items() if name != "quality")
    assert funnel.gate_status(marginal, v2.V1_CONFIG)["quality"] is False
    assert funnel.gate_status(marginal, v2.V2_CONFIG)["quality"] is True
    assert funnel.gate_status(blocked, v2.V2_CONFIG)["consume_restore"] is False


def test_l2_v2_extended_inventory_classifies_declared_source_limitations_without_opening_dbn(tmp_path: Path):
    inventory = extended.build_inventory(tmp_path)
    may = next(row for row in inventory["sessions"] if row["period"] == "may")
    retro = next(row for row in inventory["sessions"] if row["period"] == "retro_june_july")
    august = next(row for row in inventory["sessions"] if row["period"] == "august_seen")
    assert may["full_trade_replay_status"] == "UNUSABLE"
    assert retro["full_trade_replay_status"] == "MISSING_REQUIRED_CONTEXT"
    assert august["full_trade_replay_status"] == "MISSING_REQUIRED_CONTEXT"
    assert {row["evidence_label"] for row in inventory["sessions"] if row["period"] == "retro_june_july"} == {extended.RETRO_LABEL}
    source = Path(extended.__file__).read_text(encoding="utf-8")
    assert "DBNStore" not in source and "from databento" not in source


def test_l2_v2_source_end_marks_open_position_unresolved_without_a_1600_forced_exit():
    runner = historical.HistoricalL2Runner(date="2026-06-23", evidence_label=extended.RETRO_LABEL,
                                           levels=[l2.StructuralLevel("PRIOR_RTH_POC", PX)])
    signal, setup = _qualified_setup("source-end")
    setup.state, setup.terminal_reason, setup.entry_ready_ns = "ENTRY", "ENTRY", BASE_NS
    prices = l2.initial_prices("BUYER_ABSORPTION", PX, PX + .25, PX - .5, PX + .5)
    signal.position = l2.L2Position(setup, "ES", 1, prices, BASE_NS)
    runner.signals = signal
    runner.mark_source_end_incomplete(BASE_NS + 10)
    assert runner.trade_ledger == []
    assert runner.signals.position is None
    assert setup.terminal_reason == "UNRESOLVED_SOURCE_END"
    assert runner.source_end_unresolved == [{"setup_id": setup.setup_id, "timestamp_ns": BASE_NS + 10,
                                             "reason": "UNRESOLVED_SOURCE_END", "instrument": "ES", "contracts": 1}]


def test_l2_v2_native_mes_unavailable_is_source_classification_but_es_can_still_enter():
    runner = historical.HistoricalL2Runner(date="2026-08-03", evidence_label=extended.AUGUST_LABEL,
                                           levels=[l2.StructuralLevel("PRIOR_RTH_POC", PX)],
                                           require_native_mes_for_fallback=True)
    signal, mes_needed = _qualified_setup("mes-needed")
    mes_needed.interaction.zone_low = PX - 10.0
    mes_needed.state, mes_needed.entry_ready_ns = "CONFIRMED", BASE_NS
    runner.signals = signal
    runner.es_quote = (PX, PX + .25)
    runner._attempt_entry(BASE_NS)
    assert mes_needed.terminal_reason == "MES_EXECUTION_UNAVAILABLE"
    assert runner.signals.position is None and len(runner.mes_execution_unavailable) == 1

    signal, es_fits = _qualified_setup("es-fits")
    es_fits.state, es_fits.entry_ready_ns = "CONFIRMED", BASE_NS
    runner.signals = signal
    runner._attempt_entry(BASE_NS)
    assert runner.signals.position is not None and runner.signals.position.instrument == "ES"


def test_l2_v2_poc_val_subset_is_post_run_only_and_respects_original_all_level_position_blocking():
    setups = [
        {"level": "PRIOR_RTH_POC", "accepted": True, "confirmation_timestamp_ns": 1, "terminal_reason": "ENTRY"},
        {"level": "PRIOR_RTH_VAL", "accepted": True, "confirmation_timestamp_ns": 2, "terminal_reason": "COMPLIANCE_BLOCK_ACTIVE_POSITION"},
    ]
    trades = [{"trade_id": "realized", "level": "PRIOR_RTH_POC", "exit_timestamp_ns": 3, "r_multiple": "1", "net_pnl_usd": "50", "instrument": "ES", "exit_reason": "TARGET"}]
    before = json.loads(json.dumps(setups))
    subset = extended._subset(setups, trades, levels={"PRIOR_RTH_POC", "PRIOR_RTH_VAL"})
    assert subset["accepted_setups"] == 2 and subset["confirmed_setups"] == 2
    assert subset["completed_trades"] == 1
    assert subset["methodology"].startswith("POST_RUN_SUBSET")
    assert setups == before


def test_l2_v2_extended_labels_and_contract_hash_remain_frozen():
    assert (extended.MAY_LABEL, extended.RETRO_LABEL, extended.AUGUST_LABEL) == (
        "MAY_DEVELOPMENT_V2_QUALITY_0_50_NOT_OOS_EVIDENCE",
        "RETRO_JUNE_JULY_L2_V2_INCOMPLETE_TAIL_ROBUSTNESS",
        "SEEN_AUG_L2_V2_NOT_FRESH_OOS_EVIDENCE",
    )
    assert extended.v2_contract_sha256() == "f6152ed1ca32bb7c93a62ddf672a0708ff6c92abb0beb126eec78bf7ab3ab239"


def test_august_l2_v2_completion_plan_never_repurchases_owned_es_mbo_and_maps_monday_to_friday():
    payload = august_completion.plan_payload()
    assert august_completion.prior_rth_map()["2026-08-03"] == "2026-07-31"
    assert payload["existing_es_mbo_reused"].endswith("ESU6_2026-08-03_2026-08-08_mbo.dbn")
    assert all(component["schema"] != "mbo" for component in payload["required_quote_components"])
    assert [component["session_date"] for component in payload["required_quote_components"] if component["label"] == "MES_NATIVE_EXECUTION"] == [
        "2026-08-03", "2026-08-04", "2026-08-05", "2026-08-06",
    ]


def test_august_l2_v2_mes_windows_and_friday_es_tail_requirement_are_explicit():
    payload = august_completion.plan_payload()
    mes = [component for component in payload["required_quote_components"] if component["label"] == "MES_NATIVE_EXECUTION"]
    assert all(component["start"].endswith("T13:30:00Z") and component["end"].endswith("T22:45:01Z") for component in mes)
    friday = next(row for row in payload["source_audit"] if row["date"] == "2026-08-07")
    assert friday["hard_cutoff_2245_coverage"] == "INSUFFICIENT_NO_POST_CLOSE_BBO"
    assert payload["not_quoted_component"]["status"] == "NOT_QUOTABLE_FOR_COMPLETE_REPLAY"
    assert "mbo would be required" in payload["not_quoted_component"]["cheapest_semantics_preserving_schema"]


def test_august_l2_v2_quote_path_uses_only_mocked_symbology_and_cost_metadata():
    calls: list[tuple[str, dict[str, object]]] = []

    class Metadata:
        def get_cost(self, **kwargs: object) -> float:
            calls.append(("get_cost", kwargs)); return .25

    class Symbology:
        def resolve(self, **kwargs: object) -> dict[str, object]:
            calls.append(("resolve", kwargs))
            symbol = kwargs["symbols"][0]  # type: ignore[index]
            return {"status": "OK", "raw_symbol": symbol, "instrument_id": 1, "partial": [], "not_found": []}

    class Client:
        metadata, symbology = Metadata(), Symbology()

    result = august_completion.quote_plan(client=Client())
    assert result["total_estimated_usd"] == pytest.approx(2.25)
    assert {request["stype_in"] for name, request in calls if name == "resolve"} == {"raw_symbol"}
    assert all(name in {"resolve", "get_cost"} for name, _request in calls)
    source = Path(august_completion.__file__).read_text(encoding="utf-8")
    assert "timeseries.get_range" not in source and "download" in source


def test_august_l2_v2_symbol_candidates_and_frozen_contract_are_not_strategy_changes():
    payload = august_completion.plan_payload()
    assert payload["symbol_candidates"]["ES"]["raw_symbol"] == "ESU6"
    assert payload["symbol_candidates"]["MES"]["raw_symbol"] == "MESU6"
    assert payload["v2_contract_sha256"] == "f6152ed1ca32bb7c93a62ddf672a0708ff6c92abb0beb126eec78bf7ab3ab239"


def test_august_l2_v2_metadata_client_failure_is_fail_closed_without_exposing_a_key(monkeypatch, capsys):
    def unavailable_client():
        raise ValueError("secret-api-key-must-not-appear")

    monkeypatch.setitem(sys.modules, "databento", types.SimpleNamespace(Historical=unavailable_client))
    assert august_completion.main(["--resolve-symbols"]) == 1
    output = capsys.readouterr().out
    assert "Databento Historical metadata client could not initialize" in output
    assert "secret-api-key-must-not-appear" not in output


class _AugustAcquisitionClient:
    def __init__(self) -> None:
        self.cost_calls: list[dict[str, object]] = []
        self.resolve_calls: list[dict[str, object]] = []
        self.download_calls: list[dict[str, object]] = []
        self.metadata = self
        self.symbology = self
        self.timeseries = self

    def get_cost(self, **kwargs: object) -> float:
        self.cost_calls.append(kwargs)
        if kwargs["schema"] == "mbp-1":
            return float(august_acquisition.EXPECTED_MES_USD / 4)
        return float(august_acquisition.EXPECTED_PROFILE_USD / 5)

    def resolve(self, **kwargs: object) -> dict[str, object]:
        self.resolve_calls.append(kwargs)
        symbol = kwargs["symbols"][0]  # type: ignore[index]
        return {"status": "OK", "raw_symbol": symbol, "instrument_id": 1, "partial": [], "not_found": []}

    def get_range(self, **kwargs: object) -> None:
        self.download_calls.append(kwargs)
        Path(str(kwargs["path"])).write_bytes(b"sealed synthetic databento payload")


def test_august_l2_v2_acquisition_requests_exactly_nine_files_with_no_mbo_or_aug7_mes():
    items = august_acquisition.components()
    assert len(items) == 9
    assert {item.label for item in items} == {"MES_NATIVE_EXECUTION", "ES_PRIOR_RTH_PROFILE"}
    assert all(item.schema != "mbo" for item in items)
    assert all(not (item.symbol == "MESU6" and item.session_date == "2026-08-07") for item in items)
    mes = [item for item in items if item.label == "MES_NATIVE_EXECUTION"]
    assert [(item.session_date, item.start, item.end) for item in mes] == [
        (day, f"{day}T13:30:00Z", f"{day}T22:45:01Z") for day in ("2026-08-03", "2026-08-04", "2026-08-05", "2026-08-06")
    ]


def test_august_l2_v2_acquisition_quote_tolerance_and_explicit_download_gate(tmp_path: Path):
    client = _AugustAcquisitionClient()
    result = august_acquisition.acquire(root=tmp_path / "unused", client=client, download=False)
    assert result["download_requested"] is False and result["download_api_invoked"] is False
    assert client.download_calls == [] and len(client.cost_calls) == 9
    with pytest.raises(august_acquisition.AcquisitionError, match="exceeds approved tolerance"):
        august_acquisition._validate_quote({"cost_by_component_usd": {"MES_NATIVE_EXECUTION": 0, "ES_PRIOR_RTH_PROFILE": 0}, "total_estimated_usd": 99})


def test_august_l2_v2_acquisition_writes_atomic_part_then_hash_verified_manifest_and_resumes(tmp_path: Path):
    client = _AugustAcquisitionClient()
    root, item = tmp_path / "august", august_acquisition.components()[0]
    manifest = {"files": {}}
    first = august_acquisition._download_one(client=client, root=root, component=item, manifest=manifest, cost=Decimal(".1"))
    destination = august_acquisition._destination(root, item)
    assert first["status"] == "DOWNLOADED_VERIFIED" and destination.is_file()
    assert not destination.with_suffix(destination.suffix + ".part").exists()
    assert manifest["files"][destination.relative_to(root).as_posix()]["sha256"] == august_acquisition._sha256(destination)
    second = august_acquisition._download_one(client=client, root=root, component=item, manifest=manifest, cost=Decimal(".1"))
    assert second["status"] == "SKIPPED_VERIFIED" and len(client.download_calls) == 1
    destination.write_bytes(b"mismatch")
    with pytest.raises(august_acquisition.AcquisitionError, match="hash/size"):
        august_acquisition._verified_existing(root=root, destination=destination, manifest=manifest)


def test_august_l2_v2_acquisition_identity_keeps_strategy_unexecuted_and_v2_hash_frozen():
    identity = august_acquisition.request_identity(august_acquisition.components())
    assert identity["strategy_replay_occurred"] is False
    assert identity["v2_contract_sha256"] == "f6152ed1ca32bb7c93a62ddf672a0708ff6c92abb0beb126eec78bf7ab3ab239"


def _write_august_replay_fixture(repository_root: Path, completion_root: Path) -> None:
    es_path = repository_root / august_seen.ES_MBO_RELATIVE
    es_path.parent.mkdir(parents=True)
    es_path.write_bytes(b"sealed-es-mbo")
    es_manifest = {
        "status": "OOS_DATA_ACQUIRED_AND_SEALED",
        "data_acquired": True,
        "provider": {"dataset": "GLBX.MDP3", "schema": "mbo"},
        "symbol_and_instrument": {"resolved_raw_symbol": "ESU6", "instrument_ids": [42140870]},
        "chronology": {"eligible_rth_dates": ["2026-08-03", "2026-08-04", "2026-08-05", "2026-08-06", "2026-08-07"]},
        "proposed_acquisition": {
            "file_bytes": es_path.stat().st_size,
            "file_sha256": august_seen._sha256(es_path).upper(),
            "record_count": 61_106_259,
            "integrity_pass": True,
        },
    }
    manifest_path = repository_root / august_seen.ES_MANIFEST_RELATIVE
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(es_manifest), encoding="utf-8")
    files: dict[str, dict[str, object]] = {}
    expected = august_seen._expected_completion_files()
    for index, (relative, identity) in enumerate(expected.items()):
        path = completion_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"sealed-{index}".encode())
        files[relative] = {**identity, "bytes": path.stat().st_size, "sha256": august_seen._sha256(path)}
    completion = {
        "manifest_kind": "AUGUST_2026_L2_V2_MISSING_COMPONENT_ACQUISITION",
        "data_acquired": True,
        "strategy_replay_occurred": False,
        "request_identity": {
            "strategy_id": v2.STRATEGY_ID,
            "evidence_label": august_seen.EVIDENCE_LABEL,
            "v2_contract_sha256": v2.v2_contract_sha256(),
            "components": list(expected.values()),
        },
        "files": files,
    }
    completion_root.mkdir(parents=True, exist_ok=True)
    (completion_root / "acquisition-manifest.json").write_text(json.dumps(completion), encoding="utf-8")


def test_august_seen_replay_validates_reused_es_and_exact_nine_completion_inputs(tmp_path: Path):
    repository_root, completion_root = tmp_path / "repo", tmp_path / "completion"
    _write_august_replay_fixture(repository_root, completion_root)
    verification = august_seen.verify_august_inputs(repository_root=repository_root, completion_root=completion_root)
    assert verification["strategy_id"] == v2.STRATEGY_ID
    assert verification["evidence_label"] == "SEEN_AUG_L2_V2_NOT_FRESH_OOS_EVIDENCE"
    assert verification["august_7_excluded"] is True
    assert len(verification["completion_files_verified"]) == 9
    assert verification["existing_es_mbo"]["snapshot_initialization"] == "SEALED_VALIDATED_R_A_F_SNAPSHOT_THROUGH_F_LAST"


def test_august_seen_replay_fails_closed_for_tampered_completion_file(tmp_path: Path):
    repository_root, completion_root = tmp_path / "repo", tmp_path / "completion"
    _write_august_replay_fixture(repository_root, completion_root)
    target = next(iter(august_seen._expected_completion_files()))
    (completion_root / target).write_bytes(b"tampered")
    with pytest.raises(august_seen.AugustReplayError, match="hash/size mismatch"):
        august_seen.verify_august_inputs(repository_root=repository_root, completion_root=completion_root)


def test_august_seen_replay_has_exact_prior_rth_mapping_native_mes_and_no_aug7_or_network_client():
    assert august_seen.TARGET_DATES == ("2026-08-03", "2026-08-04", "2026-08-05", "2026-08-06")
    assert august_seen.PRIOR_RTH == {
        "2026-08-03": "2026-07-31", "2026-08-04": "2026-08-03",
        "2026-08-05": "2026-08-04", "2026-08-06": "2026-08-05",
    }
    profile, mes = august_seen._paths(Path("completion"), "2026-08-03")
    assert profile.as_posix().endswith("ESU6_2026-07-31_133000_200000_trades.dbn.zst")
    assert mes.as_posix().endswith("MESU6_2026-08-03_133000_224501_mbp1.dbn.zst")
    source = Path(august_seen.__file__).read_text(encoding="utf-8")
    assert "require_native_mes_for_fallback=True" in source
    assert "HistoricalMBOToMBP10Adapter" in source
    assert "if day > TARGET_DATES[-1]:" in source
    assert "if day not in runners or day in closed:" in source
    assert all(forbidden not in source for forbidden in ("Historical(", "get_cost(", "get_range(", "requests."))


def test_august_seen_cross_period_combines_published_poc_val_only_descriptively():
    payload = {
        "metrics": {"completed_trades": 5, "total_r": 2.0, "profit_factor": 1.2},
        "poc_val_descriptive_subset": {
            "PRIOR_RTH_POC": {"completed_trades": 2, "total_r": 1.5, "profit_factor": 2.0},
            "PRIOR_RTH_VAL": {"completed_trades": 1, "total_r": -0.5, "profit_factor": 0.5},
        },
    }
    row = august_seen._comparison_row("MAY", payload)
    assert (row["poc_val_trades"], row["poc_val_total_r"], row["poc_val_pf"]) == (3, 1.0, None)


def _write_v2_audit_period(root: Path, *, evidence_label: str, dates: list[str], include_integrity: bool = False) -> None:
    interactions = [
        _funnel_interaction(f"{root.name}-{day}", quality=.6 if index == 0 else .4)
        for index, day in enumerate(dates)
    ]
    for row, day in zip(interactions, dates):
        row["date"] = day
    setups = [_funnel_setup(row, accepted=index == 0, status="ENTRY" if index == 0 else "REJECTED")
              for index, row in enumerate(interactions)]
    trades = [{"interaction_id": interactions[0]["interaction_id"]}]
    for name, rows in (("interaction-features.csv", interactions), ("setup-ledger.csv", setups), ("trade-ledger.csv", trades)):
        fields = sorted({key for row in rows for key in row})
        with (root / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    summary = {"strategy_id": v2.STRATEGY_ID, "v2_contract_sha256": v2.v2_contract_sha256(), "evidence_label": evidence_label}
    if include_integrity:
        summary.update({"august_7_excluded": True, "input_verification": {
            "existing_es_mbo": {"snapshot_initialization": "SEALED_VALIDATED_R_A_F_SNAPSHOT_THROUGH_F_LAST"},
            "completion_files_verified": [
                {"relative_path": f"mes_mbp1/MESU6_{day}_133000_224501_mbp1.dbn.zst"} for day in dates
            ] + [{"relative_path": f"es_prior_rth_trades/ESU6_{day}_133000_200000_trades.dbn.zst"}
                 for day in ("2026-07-31", "2026-08-03", "2026-08-04", "2026-08-05", "2026-08-06")],
        }})
        source = {"policy": "RETAIN_PRIVATE_FAIL_CLOSED_IF_STRATEGY_VISIBLE", "max_tolerated_offbook_anomalies_per_session": 10,
                  "sessions": [{"date": day, "anomalies": []} for day in dates]}
        (root / "source-integrity-diagnostics.json").write_text(json.dumps(source), encoding="utf-8")
        with (root / "daily-results.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["date", "unresolved_source_end"])
            writer.writeheader(); writer.writerows({"date": day, "unresolved_source_end": 0} for day in dates)
    (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_august_selectivity_audit_uses_only_published_ledgers_and_materializes_requested_reports(tmp_path: Path):
    repo = tmp_path / "repo"
    roots = {name: repo / relative for name, relative in august_audit.PERIOD_ROOTS.items()}
    for root in roots.values(): root.mkdir(parents=True)
    _write_v2_audit_period(roots["MAY"], evidence_label="MAY_DEVELOPMENT", dates=["2026-05-04"])
    _write_v2_audit_period(roots["RETRO_JUNE_JULY"], evidence_label="RETRO", dates=["2026-06-23"])
    august_dates = ["2026-08-03", "2026-08-04", "2026-08-05", "2026-08-06"]
    _write_v2_audit_period(roots["AUGUST_SEEN"], evidence_label=august_audit.EVIDENCE_LABEL, dates=august_dates, include_integrity=True)
    result = august_audit.materialize(repository_root=repo, august_root=roots["AUGUST_SEEN"])
    output = roots["AUGUST_SEEN"] / august_audit.AUDIT_NAME
    assert result["strategy_mutated"] is False and result["network_or_download_used"] is False and result["full_dbn_replay"] is False
    assert result["august"]["metrics"]["completed_interactions"] == 4
    assert result["august"]["metrics"]["session_count"] == 4
    assert result["august"]["analysis"]["independent_gate_failures"]["quality"] == 3
    assert result["august"]["quality_diagnostic_buckets"] == {">=0.40": 4, ">=0.45": 1, ">=0.48": 1, ">=0.50": 1, ">=0.52": 1, ">=0.55": 1}
    assert result["august"]["integrity"]["snapshot_initialization"] is True
    assert result["august"]["integrity"]["native_mes_files_verified"] is True
    assert result["period_comparison"]["AUGUST_SEEN"]["restoration_cycle_counts"][">=1"] == 4
    assert (output / "summary.json").is_file()
    assert (output / "rejection-funnel.csv").is_file()
    assert (output / "period-comparison.csv").is_file()
    assert (output / "accepted-setups.csv").is_file()
    assert (output / "diagnostic-report.md").is_file()


def test_august_selectivity_audit_fails_closed_on_wrong_frozen_contract_and_has_no_network_or_replay_path(tmp_path: Path):
    root = tmp_path / "period"; root.mkdir()
    _write_v2_audit_period(root, evidence_label=august_audit.EVIDENCE_LABEL, dates=["2026-08-03"], include_integrity=True)
    bad = json.loads((root / "summary.json").read_text(encoding="utf-8")); bad["v2_contract_sha256"] = "wrong"
    (root / "summary.json").write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(august_audit.AugustAuditError, match="contract hash"):
        august_audit._period(root)
    source = Path(august_audit.__file__).read_text(encoding="utf-8")
    assert all(forbidden not in source for forbidden in ("DBNStore", "Historical(", "get_range(", "get_cost(", "requests.", "_stream_private_mbo"))


def test_l2_v2_selectivity_check_uses_feature_rows_only_and_never_outcomes(tmp_path: Path):
    class OutcomeGuardRow(dict[str, object]):
        def get(self, key: str, default: object = None) -> object:
            if key in {"accepted", "confirmation_status", "terminal_reason", "trade_id", "r_multiple", "net_pnl_usd"}:
                raise AssertionError(f"outcome field read: {key}")
            return super().get(key, default)

    rows = [OutcomeGuardRow(_funnel_interaction(f"v1-{index}")) for index in range(8)]
    rows.append(OutcomeGuardRow(_funnel_interaction("v2-only", quality=.52)))
    result = v2.selectivity_check(rows)  # type: ignore[arg-type]
    assert result["pnl_or_trade_outcomes_used"] is False
    assert result["v1_recomputed_counts"]["final_accepted_v2_setups"] == 8
    assert result["v2_selectivity_counts"]["final_accepted_v2_setups"] == 9
    assert result["go_no_go_guard"]["status"] == "NO_GO_DO_NOT_REPLAY"
    source = Path(v2.__file__).read_text(encoding="utf-8")
    assert "DBNStore" not in source and "from databento" not in source


def test_l2_v2_selectivity_materializes_contract_diff_without_trade_artifacts(tmp_path: Path):
    root = tmp_path / "v1-published"; root.mkdir()
    rows = [_funnel_interaction(f"v1-{index}") for index in range(8)] + [_funnel_interaction("v2-only", quality=.52)]
    names = sorted({key for row in rows for key in row})
    with (root / "interaction-features.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names); writer.writeheader(); writer.writerows(rows)
    output = tmp_path / "v2-output"
    result = v2.materialize_selectivity_check(root, output)
    assert result["v2_selectivity_counts"]["final_accepted_v2_setups"] == 9
    assert (output / "selectivity_check" / "summary.json").is_file()
    diff = json.loads((output / "selectivity_check" / "v1-v2-contract-diff.json").read_text(encoding="utf-8"))
    assert diff["changed_strategy_fields"] == [{"field": "min_quality_score", "v1": .55, "v2": .5}]


def test_l2_v2_published_ledger_performance_calculates_drawdown_and_losing_streak():
    trades = [
        {"trade_id": "1", "interaction_id": "a", "exit_timestamp_ns": "1", "r_multiple": "1", "net_pnl_usd": "100", "instrument": "ES", "exit_reason": "TARGET", "date": "2026-05-04", "direction": "LONG", "level": "POC"},
        {"trade_id": "2", "interaction_id": "b", "exit_timestamp_ns": "2", "r_multiple": "-1", "net_pnl_usd": "-50", "instrument": "MES", "exit_reason": "STOP", "date": "2026-05-05", "direction": "SHORT", "level": "VAH"},
        {"trade_id": "3", "interaction_id": "c", "exit_timestamp_ns": "3", "r_multiple": "-1", "net_pnl_usd": "-50", "instrument": "MES", "exit_reason": "HARD_CUTOFF_2245", "date": "2026-05-05", "direction": "SHORT", "level": "VAH"},
    ]
    result = v2_report._performance(trades)
    assert result["wins"] == 1 and result["losses"] == 2
    assert result["profit_factor"] == 1.0
    assert result["max_cumulative_drawdown_r"] == -2.0
    assert result["longest_losing_streak"] == 2
    assert (result["es_trades"], result["mes_trades"], result["stop_exits"], result["target_exits"], result["cutoff_exits"]) == (1, 2, 1, 1, 1)


def test_l2_v2_reporting_repairs_only_metadata_from_published_ledgers():
    setups = [
        {"interaction_id": "a", "accepted": "True", "confirmation_status": "ENTRY", "terminal_reason": "ENTRY"},
        {"interaction_id": "b", "accepted": "True", "confirmation_status": "FAILED", "terminal_reason": "CONFIRMATION_WINDOW_EXPIRED"},
        {"interaction_id": "c", "accepted": "True", "confirmation_status": "FAILED", "terminal_reason": "COMPLIANCE_BLOCK_ACTIVE_POSITION"},
    ]
    trades = [{"trade_id": "t", "interaction_id": "a", "exit_timestamp_ns": "1", "r_multiple": "1", "net_pnl_usd": "50", "instrument": "ES", "exit_reason": "TARGET", "date": "2026-05-04", "direction": "LONG", "level": "POC"}]
    daily = [{"strategy_id": v2_report.STRATEGY_ID, "interactions_completed": "3", "accepted_setups": "3"}]
    result = v2_report.build_summary(setups=setups, trades=trades, daily=daily, existing_summary={"strategy_id": v2_report.STRATEGY_ID, "first_run_policy": "stale", "frozen_contract": {}})
    assert result["strategy_id"] == v2_report.STRATEGY_ID
    assert result["variant_label"] == "L2_V2_MAY_DEVELOPMENT_QUALITY_0_50"
    assert result["counts"] == {
        "completed_interactions": 3, "accepted_setups": 3, "confirmations_passed": 1,
        "confirmations_failed_window_expired": 1, "compliance_blocks_active_position": 1,
        "completed_trades": 1, "unresolved_trades": 0, "accepted_to_confirmed_conversion_rate": pytest.approx(1 / 3),
    }
    source = Path(v2_report.__file__).read_text(encoding="utf-8")
    assert "DBNStore" not in source and "from databento" not in source


def _score_validity_fixture() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    setups: list[dict[str, str]] = []
    trades: list[dict[str, str]] = []
    scores = [.505, .525, .545, .565, .585]
    for index in range(56):
        score = scores[index % len(scores)]
        terminal = "ENTRY" if index < 32 else ("CONFIRMATION_WINDOW_EXPIRED" if index < 53 else "COMPLIANCE_BLOCK_ACTIVE_POSITION")
        setup = {key: str(value) for key, value in _funnel_interaction(f"score-{index}", quality=score).items()}
        setup.update({"accepted": "True", "confirmation_status": "ENTRY" if terminal == "ENTRY" else "FAILED", "terminal_reason": terminal,
                      "direction": "BUYER_ABSORPTION" if index % 2 == 0 else "SELLER_ABSORPTION",
                      "level": "PRIOR_RTH_POC" if index % 3 == 0 else "PRIOR_RTH_VAL"})
        setups.append(setup)
        if terminal == "ENTRY":
            pnl = "100" if index % 4 == 0 else "-50"
            trades.append({"interaction_id": f"score-{index}", "trade_id": f"trade-{index}", "r_multiple": "1" if pnl == "100" else "-1", "net_pnl_usd": pnl})
    return setups, trades


def test_l2_v2_score_validity_fixed_buckets_quartiles_spearman_and_leave_one_out():
    assert [score_validity.bucket_for(value) for value in (.50, .519, .52, .539, .54, .559, .56, .579, .58)] == [
        "0.50_to_0.52", "0.50_to_0.52", "0.52_to_0.54", "0.52_to_0.54", "0.54_to_0.56",
        "0.54_to_0.56", "0.56_to_0.58", "0.56_to_0.58", "0.58_or_higher",
    ]
    assert score_validity.spearman([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    setups, trades = _score_validity_fixture()
    result = score_validity.analyze(setups, trades)
    assert result["confirmation_outcomes"] == {"CONFIRMED": 32, "CONFIRMATION_WINDOW_EXPIRED": 21, "COMPLIANCE_BLOCK_ACTIVE_POSITION": 3}
    assert {metrics["accepted_setups"] for metrics in result["quartiles"].values()} == {14}
    assert result["leave_one_out_by_bucket"]
    assert result["poc_val_descriptive_appendix"]["setups"] == 56
    assert all(row["direction"]["LONG"] + row["direction"]["SHORT"] == result["quality_buckets"][name]["accepted_setups"]
               for name, row in result["bucket_level_direction_context"].items())


def test_l2_v2_score_validity_is_published_ledger_only(tmp_path: Path):
    setups, trades = _score_validity_fixture()
    root = tmp_path / "published"; root.mkdir()
    for name, rows in (("setup-ledger.csv", setups), ("trade-ledger.csv", trades)):
        names = sorted({key for row in rows for key in row})
        with (root / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=names); writer.writeheader(); writer.writerows(rows)
    result = score_validity.materialize(root)
    assert result["strategy_parameters_mutated"] is False
    assert (root / "score_validity" / "quality-buckets.csv").is_file()
    assert (root / "score_validity" / "component-comparison.csv").is_file()
    assert (root / "score_validity" / "raw-feature-comparison.csv").is_file()
    source = Path(score_validity.__file__).read_text(encoding="utf-8")
    assert "DBNStore" not in source and "from databento" not in source


def test_v3_poc_only_diff_has_exactly_one_semantic_change_and_preserves_v2_contract():
    diff = v3.v2_to_v3_contract_diff()
    assert diff["only_structural_eligibility_changed"] is True
    assert diff["changed_strategy_fields"] == [{
        "field": "eligible_structural_levels",
        "v2": list(l2.LEVEL_NAMES),
        "v3": ["PRIOR_RTH_POC"],
    }]
    contract = v3.v3_contract()
    assert contract["configuration"] == v2.v2_contract()["configuration"]
    assert contract["execution"] == v2.v2_contract()["execution"]
    assert v3.v3_contract_sha256() == v3.v3_contract_sha256()
    artifact = Path("docs/research_pipeline/cme_orderflow_absorption_v1/l2-v3-poc-only-contract-diff.json")
    recorded = json.loads(artifact.read_text(encoding="utf-8"))
    assert recorded["child_contract_sha256"] == v3.v3_contract_sha256()
    assert recorded["changed_strategy_fields"] == diff["changed_strategy_fields"]


def test_v3_poc_only_allows_poc_and_rejects_all_other_structural_levels():
    levels = tuple(l2.StructuralLevel(name, PX + index) for index, name in enumerate(l2.LEVEL_NAMES))
    selected = v3.filter_eligible_levels(levels)
    assert [item.name for item in selected] == ["PRIOR_RTH_POC"]
    assert selected[0].price == PX + l2.LEVEL_NAMES.index("PRIOR_RTH_POC")


def test_native_mbp10_contract_requires_aggregate_l2_fields_and_no_order_identity():
    contract = native_mbp10_adapter_contract()
    assert contract["schema"] == "mbp-10" and contract["mbo_required"] is False
    assert contract["order_id_permitted"] is False
    validate_native_mbp10_field_mapping(contract["required_fields"])
    unsafe = {**contract["required_fields"], "private": ("order_id",)}
    with pytest.raises(ValueError, match="order identity"):
        validate_native_mbp10_field_mapping(unsafe)


class _V3QuoteClient:
    class _Symbology:
        def __init__(self, calls: list[dict[str, object]]) -> None: self.calls = calls
        def resolve(self, **request: object) -> dict[str, object]:
            self.calls.append(request)
            symbol = request["symbols"][0]  # type: ignore[index]
            return {"status": "OK", "result": {symbol: [{"instrument_id": 1}]}}

    class _Metadata:
        def __init__(self, calls: list[dict[str, object]]) -> None: self.calls = calls
        def get_cost(self, **request: object) -> float:
            self.calls.append(request)
            return {"mbp-10": 2.0, "mbp-1": 1.0, "trades": .5}[str(request["schema"])]

    def __init__(self) -> None:
        self.symbology_calls: list[dict[str, object]] = []
        self.cost_calls: list[dict[str, object]] = []
        self.symbology, self.metadata = self._Symbology(self.symbology_calls), self._Metadata(self.cost_calls)


def test_v3_august_fresh_plan_is_sealed_and_never_requests_mbo():
    payload = v3_quote.plan_payload(block="august-fresh", sessions=5)
    assert payload["rth_dates"] == ["2026-08-10", "2026-08-11", "2026-08-12", "2026-08-13", "2026-08-14"]
    assert payload["previous_completed_rth"]["2026-08-10"] == "2026-08-07"
    assert all(item["schema"] != "mbo" for item in payload["components"])
    es = [item for item in payload["components"] if item["label"] == "ES_MBP10_USD"]
    mes = [item for item in payload["components"] if item["label"] == "MES_MBP1_USD"]
    profile = [item for item in payload["components"] if item["label"] == "PRIOR_RTH_TRADES_USD"]
    assert len(es) == len(mes) == len(profile) == 5
    assert all(item["start"].endswith("T13:00:00Z") and item["end"].endswith("T22:45:01Z") for item in es)
    assert all(item["start"].endswith("T13:30:00Z") and item["end"].endswith("T22:45:01Z") for item in mes)
    assert all(item["start"].endswith("T13:30:00Z") and item["end"].endswith("T20:00:00Z") for item in profile)


def test_v3_april_calendar_selection_and_quote_only_totals_reconcile():
    plan = v3_quote.plan_payload(block="april", sessions=5)
    assert plan["rth_dates"] == ["2026-04-06", "2026-04-07", "2026-04-08", "2026-04-09", "2026-04-10"]
    assert plan["previous_completed_rth"]["2026-04-06"] == "2026-04-02"
    client = _V3QuoteClient()
    result = v3_quote.quote_payload(block="april", sessions=5, client=client)
    totals = result["quote"]["component_totals_usd"]
    assert totals == {"ES_MBP10_USD": 10.0, "MES_MBP1_USD": 5.0, "PRIOR_RTH_TRADES_USD": 2.5,
                      "TOTAL_USD": 17.5, "AVG_USD_PER_SESSION": 3.5}
    assert len(client.cost_calls) == 15
    assert all("get_range" not in repr(call) and "download" not in repr(call) for call in client.cost_calls)
    assert "PRE_REPLAY_CALENDAR_CLARIFICATION_REQUIRED" in result["friday_execution_calendar_status"]
    source = Path(v3_quote.__file__).read_text(encoding="utf-8")
    assert "get_range(" not in source and "timeseries." not in source


class _V3AcquisitionClient:
    def __init__(self) -> None:
        self.cost_calls: list[dict[str, object]] = []
        self.resolve_calls: list[dict[str, object]] = []
        self.download_calls: list[dict[str, object]] = []
        self.metadata = self
        self.symbology = self
        self.timeseries = self

    def get_cost(self, **request: object) -> float:
        self.cost_calls.append(request)
        shares = {
            "mbp-10": v3_acquisition.EXPECTED_COMPONENT_TOTALS["ES_MBP10_USD"] / 5,
            "mbp-1": v3_acquisition.EXPECTED_COMPONENT_TOTALS["MES_MBP1_USD"] / 5,
            "trades": v3_acquisition.EXPECTED_COMPONENT_TOTALS["PRIOR_RTH_TRADES_USD"] / 5,
        }
        return float(shares[str(request["schema"])])

    def resolve(self, **request: object) -> dict[str, object]:
        self.resolve_calls.append(request)
        symbol = request["symbols"][0]  # type: ignore[index]
        return {"status": "OK", "raw_symbol": symbol, "instrument_id": 1, "partial": [], "not_found": []}

    def get_range(self, **request: object) -> None:
        self.download_calls.append(request)
        Path(str(request["path"])).write_bytes(b"sealed synthetic V3 payload")


class _V3AprilAcquisitionClient(_V3AcquisitionClient):
    """Metadata/download fake only; it never opens a Databento data stream."""

    def get_cost(self, **request: object) -> float:
        self.cost_calls.append(request)
        # Three MBP-10 files at $2, three MBP-1 files at $1, and three
        # prior-RTH profile files totaling $0.82: $9.82 exactly in Decimal.
        shares = {
            "mbp-10": Decimal("2"),
            "mbp-1": Decimal("1"),
            "trades": Decimal("0.2733333333333333333333333333"),
        }
        return float(shares[str(request["schema"])])


def test_v3_acquisition_requests_exact_fifteen_files_and_preserves_frozen_contract():
    items = v3_acquisition.components()
    assert len(items) == 15
    assert {item.label for item in items} == {"ES_MBP10_USD", "MES_MBP1_USD", "PRIOR_RTH_TRADES_USD"}
    assert [item.session_date for item in items if item.label == "ES_MBP10_USD"] == list(v3_acquisition.FRESH_DATES)
    assert [item.session_date for item in items if item.label == "MES_MBP1_USD"] == list(v3_acquisition.FRESH_DATES)
    assert [item.session_date for item in items if item.label == "PRIOR_RTH_TRADES_USD"] == list(v3_acquisition.PRIOR_RTH_MAP.values())
    assert all(item.schema != "mbo" for item in items)
    assert all(item.start.endswith("T13:00:00Z") and item.end.endswith("T22:45:01Z")
               for item in items if item.label == "ES_MBP10_USD")
    assert all(item.start.endswith("T13:30:00Z") and item.end.endswith("T22:45:01Z")
               for item in items if item.label == "MES_MBP1_USD")
    assert v3_acquisition.request_identity(items)["v3_contract_sha256"] == "a0ce94eeb78dcbf865cf4464bdf97ebc3f014a8ec5e2f559f99798534dfcbcb4"


def test_v3_acquisition_requotes_without_download_and_fails_closed_outside_tolerance(tmp_path: Path):
    client = _V3AcquisitionClient()
    result = v3_acquisition.acquire(root=tmp_path / "unused", client=client, download=False)
    assert result["download_requested"] is False and result["download_api_invoked"] is False
    assert result["strategy_replay_executed"] is False and result["outcomes_inspected"] is False
    assert client.download_calls == [] and len(client.cost_calls) == 15
    with pytest.raises(v3_acquisition.AcquisitionError, match="exceeds approved tolerance"):
        v3_acquisition._validate_quote({"quote": {"component_totals_usd": {
            "ES_MBP10_USD": 1, "MES_MBP1_USD": 1, "PRIOR_RTH_TRADES_USD": 1,
            "TOTAL_USD": 99, "AVG_USD_PER_SESSION": 19.8,
        }}})


def test_v3_acquisition_atomic_manifest_hash_resume_and_stale_partial_fail_closed(tmp_path: Path):
    client, root = _V3AcquisitionClient(), tmp_path / "v3"
    item = v3_acquisition.components()[0]
    manifest = {"files": {}}
    first = v3_acquisition._download_one(client=client, root=root, component=item, manifest=manifest, cost=Decimal(".1"))
    destination = v3_acquisition._destination(root, item)
    assert first["status"] == "DOWNLOADED_VERIFIED" and destination.is_file()
    assert manifest["files"][destination.relative_to(root).as_posix()]["sha256"] == v3_acquisition._sha256(destination)
    assert v3_acquisition._download_one(client=client, root=root, component=item, manifest=manifest, cost=Decimal(".1"))["status"] == "SKIPPED_VERIFIED"
    assert len(client.download_calls) == 1
    destination.write_bytes(b"mismatch")
    with pytest.raises(v3_acquisition.AcquisitionError, match="hash/size"):
        v3_acquisition._verified_existing(root=root, destination=destination, manifest=manifest)
    destination.write_bytes(b"sealed synthetic V3 payload")
    destination.with_suffix(destination.suffix + ".part").write_bytes(b"stale")
    with pytest.raises(v3_acquisition.AcquisitionError, match="stale partial"):
        v3_acquisition._assert_no_unknown_or_stale_files(root=root, items=[item], manifest=manifest)


def test_v3_acquisition_full_mock_materialization_writes_complete_manifest_and_resumes(tmp_path: Path):
    root, client = tmp_path / "v3-full", _V3AcquisitionClient()
    first = v3_acquisition.acquire(root=root, client=client, download=True)
    manifest = json.loads((root / v3_acquisition.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert first["download_api_invoked"] is True and len(first["files"]) == 15
    assert manifest["data_acquired"] is True and len(manifest["files"]) == 15
    assert manifest["strategy_replay_executed"] is False and manifest["outcomes_inspected"] is False
    assert len(client.download_calls) == 15
    second = v3_acquisition.acquire(root=root, client=client, download=True)
    assert {item["status"] for item in second["files"]} == {"SKIPPED_VERIFIED"}
    assert len(client.download_calls) == 15
    source = Path(v3_acquisition.__file__).read_text(encoding="utf-8")
    assert "historical_runner" not in source and "DBNStore" not in source and "outcome" in source


def test_v3_april_acquisition_is_exactly_three_sessions_nine_requests_and_never_mbo():
    items = v3_april_acquisition.components()
    assert len(items) == 9
    assert [item.session_date for item in items if item.label == "ES_MBP10_USD"] == list(v3_april_acquisition.TARGET_DATES)
    assert [item.session_date for item in items if item.label == "MES_MBP1_USD"] == list(v3_april_acquisition.TARGET_DATES)
    assert [item.session_date for item in items if item.label == "PRIOR_RTH_TRADES_USD"] == ["2026-04-02", "2026-04-06", "2026-04-07"]
    assert v3_april_acquisition.PRIOR_RTH_MAP["2026-04-06"] == "2026-04-02"
    assert all(item.schema != "mbo" for item in items)
    assert all(item.start.endswith("T13:00:00Z") and item.end.endswith("T22:45:01Z") for item in items if item.label == "ES_MBP10_USD")
    assert all(item.start.endswith("T13:30:00Z") and item.end.endswith("T22:45:01Z") for item in items if item.label == "MES_MBP1_USD")
    assert all(item.start.endswith("T13:30:00Z") and item.end.endswith("T20:00:00Z") for item in items if item.label == "PRIOR_RTH_TRADES_USD")
    identity = v3_april_acquisition.request_identity(items)
    assert identity["v3_contract_sha256"] == "a0ce94eeb78dcbf865cf4464bdf97ebc3f014a8ec5e2f559f99798534dfcbcb4"
    assert identity["evidence_label"] == "APRIL_2026_RETROSPECTIVE_ROBUSTNESS_L2_V3_POC_ONLY"
    assert identity["strict_chronological_oos"] is False


def test_v3_april_acquisition_requires_explicit_download_and_rejects_quote_deviation(tmp_path: Path):
    client = _V3AprilAcquisitionClient()
    result = v3_april_acquisition.acquire(root=tmp_path / "unused", client=client, download=False)
    assert result["download_requested"] is False and result["download_api_invoked"] is False
    assert result["strategy_replay_executed"] is False and result["outcomes_inspected"] is False
    assert client.download_calls == [] and len(client.cost_calls) == 9
    with pytest.raises(v3_april_acquisition.AcquisitionError, match="exceeds approved tolerance"):
        v3_april_acquisition._validate_quote({"quote": {"component_totals_usd": {
            "ES_MBP10_USD": "10", "MES_MBP1_USD": "10", "PRIOR_RTH_TRADES_USD": "10", "TOTAL_USD": "30",
        }}})


def test_v3_april_acquisition_atomic_download_hash_resume_and_no_strategy_execution(tmp_path: Path):
    client, root = _V3AprilAcquisitionClient(), tmp_path / "v3-april"
    item, manifest = v3_april_acquisition.components()[0], {"files": {}}
    first = v3_april_acquisition._download_one(client=client, root=root, component=item, manifest=manifest, cost=Decimal("2"))
    destination = v3_april_acquisition._destination(root, item)
    assert first["status"] == "DOWNLOADED_VERIFIED" and destination.is_file()
    assert v3_april_acquisition._download_one(client=client, root=root, component=item, manifest=manifest, cost=Decimal("2"))["status"] == "SKIPPED_VERIFIED"
    assert len(client.download_calls) == 1
    destination.with_suffix(destination.suffix + ".part").write_bytes(b"stale")
    with pytest.raises(v3_april_acquisition.AcquisitionError, match="stale partial"):
        v3_april_acquisition._assert_no_unknown_or_stale_files(root=root, items=[item], manifest=manifest)
    source = Path(v3_april_acquisition.__file__).read_text(encoding="utf-8")
    assert "historical_runner" not in source and "DBNStore" not in source and "strategy_replay_executed" in source


def _write_v3_fresh_manifest(root: Path) -> None:
    files: dict[str, dict[str, object]] = {}
    for relative, identity in v3_fresh._expected_files().items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode("utf-8"))
        files[relative] = {**identity, "bytes": path.stat().st_size, "sha256": v3_fresh._sha256(path)}
    manifest = {
        "manifest_kind": "AUGUST_2026_L2_V3_POC_ONLY_FRESH_ACQUISITION", "data_acquired": True,
        "strategy_replay_executed": False, "outcomes_inspected": False, "files": files,
        "request_identity": {
            "strategy_id": v3_fresh.STRATEGY_ID, "v3_contract_sha256": v3_fresh.v3_contract_sha256(),
            "fresh_rth_dates": list(v3_fresh.TARGET_DATES), "prior_rth_mapping": v3_fresh.PRIOR_RTH,
            "mbo_purchased": False,
        },
    }
    (root / "acquisition-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _write_v3_april_manifest(root: Path) -> None:
    files: dict[str, dict[str, object]] = {}
    for relative, identity in v3_april_replay._expected_files().items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode("utf-8"))
        files[relative] = {**identity, "bytes": path.stat().st_size, "sha256": v3_april_replay._sha256(path)}
    manifest = {
        "manifest_kind": "APRIL_2026_L2_V3_POC_ONLY_RETROSPECTIVE_ACQUISITION",
        "data_acquired": True, "strategy_replay_executed": False, "outcomes_inspected": False,
        "no_mbo_purchased": True, "files": files,
        "request_identity": {
            "strategy_id": v3_april_replay.STRATEGY_ID,
            "v3_contract_sha256": v3_april_replay.V3_CONTRACT_SHA256,
            "evidence_label": v3_april_replay.EVIDENCE_LABEL,
            "target_rth_dates": list(v3_april_replay.TARGET_DATES),
            "prior_rth_mapping": v3_april_replay.PRIOR_RTH,
            "mbo_purchased": False, "strict_chronological_oos": False,
        },
    }
    (root / "acquisition-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_v3_fresh_preflight_hash_verifies_5_5_5_inputs_without_opening_dbns(tmp_path: Path):
    root = tmp_path / "fresh"; _write_v3_fresh_manifest(root)
    result = v3_fresh.verify_acquisition_manifest(root)
    assert result["preflight_only"] is True and result["files_verified"] == 15
    assert result["by_label"] == {"ES_MBP10_USD": 5, "MES_MBP1_USD": 5, "PRIOR_RTH_TRADES_USD": 5}
    assert result["strategy_or_outcomes_accessed"] is False
    target = root / next(iter(v3_fresh._expected_files()))
    target.write_bytes(b"tampered")
    with pytest.raises(v3_fresh.V3FreshReplayError, match="hash/size"):
        v3_fresh.verify_acquisition_manifest(root)


def test_v3_april_preflight_binds_exact_nine_inputs_without_opening_dbns(tmp_path: Path):
    root = tmp_path / "april"; _write_v3_april_manifest(root)
    result = v3_april_replay.verify_acquisition_manifest(root)
    assert result["preflight_only"] is True and result["files_verified"] == 9
    assert result["target_dates"] == ["2026-04-06", "2026-04-07", "2026-04-08"]
    assert result["prior_rth_mapping"]["2026-04-06"] == "2026-04-02"
    assert result["by_label"] == {"ES_MBP10_USD": 3, "MES_MBP1_USD": 3, "PRIOR_RTH_TRADES_USD": 3}
    assert result["strategy_or_outcomes_accessed"] is False
    target = root / next(iter(v3_april_replay._expected_files()))
    target.write_bytes(b"tampered")
    with pytest.raises(v3_april_replay.V3AprilReplayError, match="hash/size"):
        v3_april_replay.verify_acquisition_manifest(root)


def test_v3_april_replay_preserves_poc_only_native_mbp10_and_frozen_execution_contract():
    assert v3_april_replay.ELIGIBLE_STRUCTURAL_LEVELS == ("PRIOR_RTH_POC",)
    assert v3_april_replay.V3_CONTRACT_SHA256 == v3.v3_contract_sha256()
    assert v3_april_replay.calendar_contract()["rule"] == "effective_hard_flat = min(frozen_22_45_UTC, scheduled_market_close)"
    source = Path(v3_april_replay.__file__).read_text(encoding="utf-8")
    assert "Historical(" not in source and "get_range(" not in source and "_stream_private_mbo" not in source
    assert "NativeMBP10Adapter" in source and "require_native_mes_for_fallback=True" in source
    execution = v2.v2_contract()["execution"]
    assert execution["confirmation_window_seconds_inclusive"] == [5.0, 15.0]
    assert execution["confirmation_favorable_ticks"] == 3 and execution["entry_latency_ms"] == 2.0
    assert execution["stop_buffer_ticks"] == 5 and execution["target_r"] == 3.0 and execution["risk_budget_usd"] == 250.0


def test_v3_april_poc_builder_requires_the_shared_single_structural_level_interface(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    poc = l2.StructuralLevel("PRIOR_RTH_POC", PX)
    monkeypatch.setattr(v3_fresh, "_profile_poc", lambda _: poc)
    assert v3_april_replay._validated_prior_rth_poc(tmp_path / "profile.dbn") is poc
    monkeypatch.setattr(v3_fresh, "_profile_poc", lambda _: l2.StructuralLevel("PRIOR_RTH_HIGH", PX))
    with pytest.raises(v3_april_replay.V3AprilReplayError, match="only PRIOR_RTH_POC"):
        v3_april_replay._validated_prior_rth_poc(tmp_path / "profile.dbn")
    monkeypatch.setattr(v3_fresh, "_profile_poc", lambda _: [poc])
    with pytest.raises(v3_april_replay.V3AprilReplayError, match="one StructuralLevel"):
        v3_april_replay._validated_prior_rth_poc(tmp_path / "profile.dbn")
    assert v3_april_replay.v3_contract_sha256() == "a0ce94eeb78dcbf865cf4464bdf97ebc3f014a8ec5e2f559f99798534dfcbcb4"


class _NativeLevel:
    def __init__(self, bid: int, ask: int, *, bid_size: int = 10, ask_size: int = 12) -> None:
        self.bid_px, self.bid_sz, self.bid_ct = bid, bid_size, 2
        self.ask_px, self.ask_sz, self.ask_ct = ask, ask_size, 3


class _NativeRecord:
    def __init__(self, *, timestamp: int, action: str, side: str, price: int, size: int, levels: tuple[_NativeLevel, ...]) -> None:
        self.ts_recv, self.action, self.side, self.price, self.size, self.levels = timestamp, action, side, price, size, levels


def test_v3_native_mbp10_initializes_before_rth_normalizes_once_and_maps_es_aggressor():
    adapter = v3_fresh.NativeMBP10Adapter()
    levels = (_NativeLevel(7_748_500_000_000, 7_748_750_000_000),)
    first = adapter.feed(_NativeRecord(timestamp=100, action="R", side="N", price=0, size=0, levels=levels))
    assert first.snapshot.bids[0].price == pytest.approx(7748.50)
    adapter.assert_initialized_before(101)
    buyer = adapter.feed(_NativeRecord(timestamp=102, action="T", side="B", price=7_748_750_000_000, size=5, levels=levels))
    seller = adapter.feed(_NativeRecord(timestamp=103, action="T", side="A", price=7_748_500_000_000, size=5, levels=levels))
    assert buyer.execution is not None and buyer.execution.aggressor == "BUY"
    assert seller.execution is not None and seller.execution.aggressor == "SELL"
    late = v3_fresh.NativeMBP10Adapter()
    late.feed(_NativeRecord(timestamp=101, action="R", side="N", price=0, size=0, levels=levels))
    with pytest.raises(v3_fresh.V3FreshReplayError, match="before 13:30"):
        late.assert_initialized_before(101)


def test_v3_native_book_state_machine_handles_clear_pause_reopen_without_stale_bbo():
    adapter = v3_fresh.NativeMBP10Adapter()
    empty_ask = _NativeLevel(7_748_500_000_000, 0, ask_size=0); empty_ask.ask_ct = 0
    one_sided = (empty_ask,)
    two_sided = (_NativeLevel(7_748_500_000_000, 7_748_750_000_000),)
    assert adapter.feed(_NativeRecord(timestamp=1, action="A", side="B", price=7_748_500_000_000, size=1, levels=one_sided)) is None
    assert adapter.state == "UNINITIALIZED"
    assert adapter.feed(_NativeRecord(timestamp=2, action="R", side="N", price=0, size=0, levels=two_sided)) is not None
    assert adapter.state == "EXECUTABLE"
    assert adapter.feed(_NativeRecord(timestamp=3, action="R", side="N", price=0, size=0, levels=one_sided)) is None
    assert adapter.state == "WAITING_FOR_REOPEN_BOOK" and adapter.previous is None
    reopened = adapter.feed(_NativeRecord(timestamp=4, action="A", side="B", price=7_748_500_000_000, size=2, levels=two_sided))
    assert reopened is not None and adapter.state == "EXECUTABLE"
    assert adapter.feed(_NativeRecord(timestamp=5, action="A", side="B", price=7_748_500_000_000, size=1, levels=one_sided), expected_non_executable=True) is None
    assert adapter.state == "NON_EXECUTABLE_EXPECTED" and adapter.previous is None


def test_v3_native_book_state_machine_suspends_on_side_neutral_cross_and_reopens_without_stale_bbo():
    adapter = v3_fresh.NativeMBP10Adapter()
    valid = (_NativeLevel(6_737_750_000_000, 6_738_750_000_000),)
    crossed = (_NativeLevel(6_743_750_000_000, 6_741_250_000_000),)
    adapter.feed(_NativeRecord(timestamp=1, action="R", side="N", price=0, size=0, levels=valid))
    transient = adapter.feed(_NativeRecord(timestamp=2, action="C", side="N", price=6_741_000_000_000, size=1, levels=crossed))
    assert transient is None and adapter.state == "TEMPORARILY_NON_EXECUTABLE"
    assert adapter.previous is None
    assert adapter.feed(_NativeRecord(timestamp=3, action="A", side="B", price=6_743_750_000_000, size=1, levels=crossed)) is None
    assert adapter.feed(_NativeRecord(timestamp=4, action="A", side="A", price=6_741_250_000_000, size=1, levels=crossed)) is None
    assert adapter.feed(_NativeRecord(timestamp=5, action="M", side="A", price=6_741_000_000_000, size=1, levels=crossed)) is None
    reopened = adapter.feed(_NativeRecord(timestamp=5_000_002, action="C", side="B", price=6_743_750_000_000, size=1, levels=valid))
    assert reopened is not None and adapter.state == "EXECUTABLE"
    assert reopened.snapshot.bids[0].price == pytest.approx(6737.75)
    assert adapter.last_transient == {"start_timestamp_ns": 2, "reopen_timestamp_ns": 5_000_002, "non_executable_records": 4}
    adapter.assert_executable_at_boundary()


def test_v3_native_two_sided_cross_can_begin_under_modify_without_action_special_case():
    adapter = v3_fresh.NativeMBP10Adapter()
    valid = (_NativeLevel(6_719_750_000_000, 6_720_250_000_000),)
    crossed = (_NativeLevel(6_719_750_000_000, 6_719_000_000_000),)
    adapter.feed(_NativeRecord(timestamp=1, action="R", side="N", price=0, size=0, levels=valid))
    assert adapter.feed(_NativeRecord(timestamp=2, action="M", side="A", price=6_719_000_000_000, size=1, levels=crossed)) is None
    assert adapter.state == "TEMPORARILY_NON_EXECUTABLE" and adapter.previous is None
    assert adapter.feed(_NativeRecord(timestamp=3, action="C", side="A", price=6_719_000_000_000, size=1, levels=crossed)) is None
    assert adapter.feed(_NativeRecord(timestamp=4, action="C", side="B", price=6_719_750_000_000, size=1, levels=valid)) is not None
    assert adapter.state == "EXECUTABLE"


def test_v3_transient_book_state_suppresses_all_quotes_and_fails_closed_if_unresolved_or_position_open():
    runner = historical.HistoricalL2Runner(date="2026-04-07", evidence_label=v3_april_replay.EVIDENCE_LABEL,
                                           levels=[l2.StructuralLevel("PRIOR_RTH_POC", PX)])
    runner.es_quote, runner.mes_quote = (PX, PX + .25), (PX, PX + .25)
    v3_fresh._begin_temporary_non_executable_state(runner, BASE_NS)
    assert runner.es_quote is None and runner.mes_quote is None
    v3_fresh._resume_temporary_non_executable_state(runner, BASE_NS + 5_000_000_000)
    adapter = v3_fresh.NativeMBP10Adapter()
    valid = (_NativeLevel(6_737_750_000_000, 6_738_750_000_000),)
    crossed = (_NativeLevel(6_743_750_000_000, 6_741_250_000_000),)
    adapter.feed(_NativeRecord(timestamp=1, action="R", side="N", price=0, size=0, levels=valid))
    adapter.feed(_NativeRecord(timestamp=2, action="C", side="N", price=6_741_000_000_000, size=1, levels=crossed))
    with pytest.raises(v3_fresh.V3FreshReplayError, match="source boundary"):
        adapter.assert_executable_at_boundary()
    _, setup = _qualified_setup("transient-open")
    runner.signals.position = l2.L2Position(setup, "ES", 1, l2.initial_prices("BUYER_ABSORPTION", PX, PX + .25, PX - .5, PX + .5), BASE_NS)
    v3_fresh._begin_temporary_non_executable_state(runner, BASE_NS + 1)
    assert runner.signals.position is not None and runner.es_quote is None
    with pytest.raises(v3_fresh.V3FreshReplayError, match="overlapped an open position"):
        v3_fresh._resume_temporary_non_executable_state(runner, BASE_NS + 2)


def test_v3_transient_suspension_cannot_create_setup_confirm_enter_or_execute_position():
    runner = historical.HistoricalL2Runner(date="2026-04-07", evidence_label=v3_april_replay.EVIDENCE_LABEL,
                                           levels=[l2.StructuralLevel("PRIOR_RTH_POC", PX)])
    _, setup = _qualified_setup("suspended-decisions")
    runner.signals.pending[setup.setup_id] = setup
    runner.es_quote, runner.mes_quote = (PX, PX + .25), (PX, PX + .25)
    interaction_count = len(runner.interaction_ledger)
    setup_count = len(runner.setup_ledger)
    event_count = len(runner.signals.events)
    v3_fresh._begin_temporary_non_executable_state(runner, BASE_NS)
    assert runner.es_quote is None and runner.mes_quote is None
    assert len(runner.interaction_ledger) == interaction_count and len(runner.setup_ledger) == setup_count
    assert len(runner.signals.events) == event_count and setup.confirmation_timestamp_ns is None
    assert runner.signals.position is None and runner.trade_ledger == []
    v3_fresh._resume_temporary_non_executable_state(runner, BASE_NS + 5_000_000)
    assert runner.signals.position is None and runner.trade_ledger == []


def test_v3_april_read_only_adapter_audit_runs_all_three_sessions_without_strategy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    valid = (_NativeLevel(6_719_750_000_000, 6_720_250_000_000),)
    crossed = (_NativeLevel(6_719_750_000_000, 6_719_000_000_000),)
    monkeypatch.setattr(v3_april_replay, "verify_acquisition_manifest", lambda _: {"files_verified": 9})
    monkeypatch.setattr(v3_april_replay, "_run_session", lambda *_: pytest.fail("strategy runner must not execute"))

    def records(path: Path):
        day = next(day for day in v3_april_replay.TARGET_DATES if day in path.name)
        base = historical._clock_ns(day, historical.RTH_START_SECONDS)
        yield _NativeRecord(timestamp=base, action="R", side="N", price=0, size=0, levels=valid)
        yield _NativeRecord(timestamp=base + 1, action="M", side="A", price=6_719_000_000_000, size=1, levels=crossed)
        yield _NativeRecord(timestamp=base + 2, action="A", side="B", price=6_719_750_000_000, size=1, levels=crossed)
        yield _NativeRecord(timestamp=base + 3, action="C", side="B", price=6_719_750_000_000, size=1, levels=valid)

    monkeypatch.setattr(v3_fresh, "_stream_native_mbp10_records", records)
    result = v3_april_replay.audit_native_mbp10(tmp_path)
    assert result["all_three_sessions_completed"] is True
    assert result["all_sessions_resolved_at_source_end"] is True
    assert result["strategy_runner_invoked"] is False and result["pnl_or_outcomes_accessed"] is False
    assert all(len(session["episodes"]) == 1 for session in result["sessions"])
    assert all(session["episodes"][0]["records"] == 2 for session in result["sessions"])
    assert v3_april_replay.v3_contract_sha256() == "a0ce94eeb78dcbf865cf4464bdf97ebc3f014a8ec5e2f559f99798534dfcbcb4"


def test_v3_native_book_state_machine_rejects_active_one_sided_corruption_but_not_empty_trade_record():
    adapter = v3_fresh.NativeMBP10Adapter()
    two_sided = (_NativeLevel(7_748_500_000_000, 7_748_750_000_000),)
    empty_ask = _NativeLevel(7_748_500_000_000, 0, ask_size=0); empty_ask.ask_ct = 0
    one_sided = (empty_ask,)
    adapter.feed(_NativeRecord(timestamp=1, action="R", side="N", price=0, size=0, levels=two_sided))
    with pytest.raises(v3_fresh.V3FreshReplayError, match="unexpected non-executable"):
        adapter.feed(_NativeRecord(timestamp=2, action="A", side="B", price=7_748_500_000_000, size=1, levels=one_sided))
    assert adapter.feed(_NativeRecord(timestamp=3, action="T", side="B", price=7_748_750_000_000, size=1, levels=one_sided)) is None
    assert adapter.state == "WAITING_FOR_REOPEN_BOOK"


def test_v3_fresh_calendar_clarification_is_explicit_and_does_not_change_v3_hash():
    assert v3_fresh.effective_hard_flat_seconds("2026-08-13") == 22 * 3600 + 45 * 60
    assert v3_fresh.effective_hard_flat_seconds("2026-08-14") == 21 * 3600
    start, end = v3_fresh.liquidation_window_ns("2026-08-14")
    assert end - start == 1_000_000_000
    contract = v3_fresh.calendar_contract()
    assert contract["classification"] == "EXECUTION_CALENDAR_CLARIFICATION"
    assert contract["no_invented_post_close_bbo"] is True and contract["strategy_contract_changed"] is False
    assert v3_fresh.v3_contract_sha256() == "a0ce94eeb78dcbf865cf4464bdf97ebc3f014a8ec5e2f559f99798534dfcbcb4"
    source = Path(v3_fresh.__file__).read_text(encoding="utf-8")
    assert "Historical(" not in source and "get_range(" not in source
    assert "HistoricalMBOToMBP10Adapter" not in source and "_stream_private_mbo" not in source


def test_v3_friday_calendar_flat_uses_only_last_causal_preclose_bbo():
    cutoff = historical._clock_ns("2026-08-14", v3_fresh.effective_hard_flat_seconds("2026-08-14"))
    runner = historical.HistoricalL2Runner(date="2026-08-14", evidence_label=v3_fresh.EVIDENCE_LABEL,
                                           levels=[l2.StructuralLevel("PRIOR_RTH_POC", PX)])
    _, setup = _qualified_setup("v3-friday")
    setup.state, setup.entry_ready_ns = "CONFIRMED", cutoff - 900_000_000
    runner.signals.pending[setup.setup_id] = setup
    runner.es_quote, runner.es_quote_timestamp_ns = (PX, PX + .25), cutoff - 500_000_000
    runner._attempt_entry(cutoff - 500_000_000)
    runner.force_flat_from_last_causal_cutoff_quote(cutoff, exit_reason="HARD_FLAT_SCHEDULED_CLOSE_2100")
    assert runner.trade_ledger[-1]["exit_reason"] == "HARD_FLAT_SCHEDULED_CLOSE_2100"
    runner.signals.position = l2.L2Position(setup, "ES", 1, l2.initial_prices("BUYER_ABSORPTION", PX, PX + .25, PX - .5, PX + .5), cutoff)
    runner.es_quote_timestamp_ns = cutoff + 1
    with pytest.raises(historical.HistoricalReplayError, match="inclusive causal BBO"):
        runner.force_flat_from_last_causal_cutoff_quote(cutoff, exit_reason="HARD_FLAT_SCHEDULED_CLOSE_2100")


def test_v3_non_executable_pause_clears_quotes_and_fails_closed_with_open_position():
    runner = historical.HistoricalL2Runner(date="2026-08-10", evidence_label=v3_fresh.EVIDENCE_LABEL,
                                           levels=[l2.StructuralLevel("PRIOR_RTH_POC", PX)])
    runner.es_quote, runner.mes_quote = (PX, PX + .25), (PX, PX + .25)
    v3_fresh._begin_non_executable_state(runner, BASE_NS)
    assert runner.es_quote is None and runner.mes_quote is None
    _, setup = _qualified_setup("pause-open")
    runner.signals.position = l2.L2Position(setup, "ES", 1, l2.initial_prices("BUYER_ABSORPTION", PX, PX + .25, PX - .5, PX + .5), BASE_NS)
    with pytest.raises(v3_fresh.V3FreshReplayError, match="MAINTENANCE_HALT_WITH_OPEN_POSITION"):
        v3_fresh._begin_non_executable_state(runner, BASE_NS + 1)
