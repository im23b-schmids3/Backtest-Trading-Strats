from research_pipeline.cme_orderflow_absorption_l2_v1.model import (
    ENTRY_LATENCY_NS,
    Execution,
    L2ClassBConfig,
    L2Config,
    L2Interaction,
    L2Setup,
    L2SignalEngine,
    StructuralLevel,
)


def _pending_signal(class_b: L2ClassBConfig) -> tuple[L2SignalEngine, L2Setup]:
    config = L2Config(min_quality_score=0.0)
    interaction = L2Interaction(
        "test-interaction", StructuralLevel("PRIOR_RTH_POC", 100.0), 0, "BUYER_ABSORPTION", config,
    )
    interaction.end_ns = 0
    interaction.end_price = 100.0
    interaction.zone_low = 99.5
    interaction.zone_high = 100.0
    setup = L2Setup("L2:test-interaction", interaction)
    signal = L2SignalEngine(config, class_b)
    signal.pending[setup.setup_id] = setup
    signal._confirmation_counts[setup.setup_id] = 0
    signal._confirmation_volume[setup.setup_id] = 0
    return signal, setup


def test_raw_confirmation_uses_count_and_volume_parameters() -> None:
    signal, setup = _pending_signal(
        L2ClassBConfig(confirmation_execution_count=2, confirmation_volume_threshold=100),
    )
    signal.observe_execution(Execution(5_000_000_000, 100.75, 50, "SELL"))
    assert setup.state == "WAIT_MIN_CONFIRMATION_TIME"
    signal.observe_execution(Execution(6_000_000_000, 100.75, 50, "SELL"))
    assert setup.state == "CONFIRMED"
    assert setup.entry_ready_ns == 6_000_000_000 + ENTRY_LATENCY_NS


def test_raw_entry_uses_parameterized_stop_and_target_with_fixed_latency() -> None:
    class_b = L2ClassBConfig(stop_ticks=8, target_r=4.0)
    signal, setup = _pending_signal(class_b)
    setup.state = "CONFIRMED"
    setup.entry_ready_ns = 2_000_000
    position = signal.try_enter(
        setup.setup_id, timestamp_ns=2_000_000, es_bid=100.0, es_ask=100.25,
        execution_policy="ES_DERIVED_PRICE_PATH_WITH_ES_OR_MES_ECONOMICS",
    )
    assert position is not None
    assert position.prices["stop"] == 97.5
    assert position.prices["target"] == 112.5
    assert setup.entry_ready_ns == 2_000_000


def test_class_b_defaults_preserve_frozen_contract() -> None:
    defaults = L2ClassBConfig()
    assert defaults.min_confirmation_seconds == 5.0
    assert defaults.max_confirmation_seconds == 15.0
    assert defaults.favorable_confirmation_ticks == 3.0
    assert defaults.confirmation_execution_count == 1
    assert defaults.confirmation_volume_threshold == 0
    assert defaults.stop_ticks == 5
    assert defaults.target_r == 3.0
    assert ENTRY_LATENCY_NS == 2_000_000
