from __future__ import annotations

from research_pipeline.cme_orderflow_absorption_l2_v1 import model
from research_pipeline.cme_orderflow_absorption_l2_v1 import historical_runner
from research_pipeline.cme_orderflow_absorption_l2_v1 import weight_q_research


BASE_NS = 1_000_000_000_000
ES_BID = 5000.0
ES_ASK = 5000.25


def _confirmed_setup() -> model.L2Setup:
    config = model.L2Config()
    interaction = model.L2Interaction(
        "interaction-es-only", model.StructuralLevel("PRIOR_RTH_POC", ES_BID),
        BASE_NS, "BUYER_ABSORPTION", config,
    )
    interaction.end_ns = BASE_NS - 10_000_000_000
    interaction.end_price = ES_BID
    interaction.zone_low = ES_BID - 10.0  # ES risk is too large; MES sizing is required.
    interaction.zone_high = ES_BID + 0.5
    setup = model.L2Setup("L2:es-only", interaction)
    setup.state = "CONFIRMED"
    setup.entry_ready_ns = BASE_NS
    return setup


def test_es_only_runner_uses_mes_economics_without_mes_quote_and_keeps_es_prices():
    runner = historical_runner.HistoricalL2Runner(
        date="2025-10-06", evidence_label="ES_ONLY_TEST",
        levels=[model.StructuralLevel("PRIOR_RTH_POC", ES_BID)],
        execution_policy=model.ES_DERIVED_PRICE_PATH_WITH_ES_OR_MES_ECONOMICS,
    )
    setup = _confirmed_setup()
    runner.signals.pending[setup.setup_id] = setup
    runner.es_quote = (ES_BID, ES_ASK)

    runner._attempt_entry(BASE_NS)
    position = runner.signals.position
    assert position is not None
    assert position.instrument == "MES"
    assert runner.mes_quote is None
    assert position.prices["entry"] == ES_ASK + model.TICK
    assert position.prices["stop"] == setup.interaction.zone_low - model.STOP_BUFFER_TICKS * model.TICK

    runner.es_quote = (5036.0, 5036.25)
    runner._manage_position(BASE_NS + 1, runner.es_quote, "ES")
    trade = runner.trade_ledger[0]
    assert trade["execution_policy"] == model.ES_DERIVED_PRICE_PATH_WITH_ES_OR_MES_ECONOMICS
    assert trade["point_value_usd"] == 5.0
    assert trade["gross_points"] > 0
    assert trade["gross_pnl_usd"] == trade["gross_points"] * 5.0 * trade["contracts"]


def test_es_only_compact_tape_uses_same_es_path_for_mes_sized_trade():
    tape = weight_q_research.SessionCausalTape(
        "2025-10-06",
        [
            {"event_ordinal": 0, "timestamp_ns": BASE_NS, "event_type": "REGULAR",
             "stream": "ES", "es_bid": ES_BID, "es_ask": ES_ASK,
             "es_quote_timestamp_ns": BASE_NS, "mes_bid": None, "mes_ask": None},
            {"event_ordinal": 1, "timestamp_ns": BASE_NS + 1_000_000_000, "event_type": "HARD_FLAT",
             "hard_flat_reason": "TEST", "es_bid": 5036.0, "es_ask": 5036.25,
             "es_quote_timestamp_ns": BASE_NS + 1_000_000_000},
        ],
        execution_policy=model.ES_DERIVED_PRICE_PATH_WITH_ES_OR_MES_ECONOMICS,
    )
    interaction = {
        "interaction_id": "i1", "source_interaction_id": "i1",
        "direction": "BUYER_ABSORPTION", "zone_low": ES_BID - 10.0,
        "zone_high": ES_BID + 0.5, "level": "PRIOR_RTH_POC",
    }
    outcome = tape.entry_outcome(interaction, 0)
    assert outcome.terminal_reason == "ENTRY"
    assert outcome.trade is not None
    assert outcome.trade["instrument"] == "MES"
    assert outcome.trade["entry"] == ES_ASK + model.TICK
    assert outcome.trade["point_value_usd"] == 5.0
    assert outcome.trade["execution_policy"] == model.ES_DERIVED_PRICE_PATH_WITH_ES_OR_MES_ECONOMICS


def test_es_only_policy_rejects_unknown_policy():
    try:
        weight_q_research.SessionCausalTape("2025-10-06", [], execution_policy="MES_PROXY")
    except weight_q_research.WeightQResearchError as exc:
        assert "unsupported execution policy" in str(exc)
    else:
        raise AssertionError("unknown execution policy was accepted")
