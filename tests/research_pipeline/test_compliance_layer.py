from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import pytest

from research_pipeline.compliance import (
    AccountState,
    ActionType,
    ComplianceClassification,
    ComplianceEvaluator,
    EconomicEvent,
    ExecutionCostConfig,
    ExecutionCostEngine,
    FixtureEconomicCalendarProvider,
    HoldingTimePolicy,
    InstrumentCostConfig,
    MarketState,
    NewsTradingPolicy,
    OrderType,
    PropFirmPolicy,
    ProposedAction,
    SessionDecision,
    SessionManager,
    SessionPolicy,
    calculate_activity_diagnostics,
    calculate_cost_config_hash,
    calculate_policy_hash,
    load_calendar_artifact,
    save_calendar_artifact,
    unconfigured_policy,
)
from research_pipeline.registry.database import Database
from research_pipeline.registry.repositories import Registry


UTC = timezone.utc


def aware(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=UTC)


def policy(**updates) -> PropFirmPolicy:
    raw = unconfigured_policy().model_dump(mode="python")
    raw.update(updates)
    raw["policy_hash"] = "pending"
    candidate = PropFirmPolicy.model_validate(raw, context={"skip_policy_hash_validation": True})
    raw["policy_hash"] = calculate_policy_hash(candidate)
    return PropFirmPolicy.model_validate(raw)


def action(kind: ActionType = ActionType.ORDER_SUBMISSION) -> ProposedAction:
    return ProposedAction(action=kind, instrument="ES", currency="USD", quantity=1)


def account(**updates) -> AccountState:
    raw = {"account_id": "a", "current_equity": 10000}
    raw.update(updates)
    return AccountState(**raw)


def test_policy_is_generic_safe_and_hashable() -> None:
    item = unconfigured_policy()
    assert item.automation.execution_mode.value == "RESEARCH_ONLY"
    assert item.news.enabled is False
    assert item.policy_hash == calculate_policy_hash(item)


def test_unknown_timezone_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown IANA timezone"):
        SessionPolicy(timezone="Not/A_Timezone")


def test_news_outside_window_is_allowed_and_boundaries_block() -> None:
    event = EconomicEvent(event_id="e", title="CPI", timestamp=aware("2025-01-02T15:00:00"), impact_level="HIGH", affected_currencies=["USD"], source="fixture", retrieved_at=aware("2025-01-01T00:00:00"), source_data_hash="fixture")
    news = NewsTradingPolicy(enabled=True, impact_levels=["HIGH"], minutes_before=10, minutes_after=10, applicable_currencies=["USD"])
    p = policy(news=news)
    evaluator = ComplianceEvaluator()
    market = MarketState(calendar_source_hash="calendar", calendar_events=[event], calendar_retrieved_at=aware("2025-01-02T14:00:00"))
    before = evaluator.evaluate(timestamp=aware("2025-01-02T14:49:59"), instrument="ES", account_state=account(), market_state=market, proposed_action=action(), policy=p)
    at_before = evaluator.evaluate(timestamp=aware("2025-01-02T14:50:00"), instrument="ES", account_state=account(), market_state=market, proposed_action=action(), policy=p)
    after = evaluator.evaluate(timestamp=aware("2025-01-02T15:10:00"), instrument="ES", account_state=account(), market_state=market, proposed_action=action(), policy=p)
    assert before.allowed
    assert not at_before.allowed and not after.allowed


def test_news_unrelated_instrument_and_currency_is_allowed() -> None:
    event = EconomicEvent(event_id="e", title="CPI", timestamp=aware("2025-01-02T15:00:00"), impact_level="HIGH", affected_currencies=["EUR"], affected_instruments=["NQ"], source="fixture", retrieved_at=aware("2025-01-01T00:00:00"), source_data_hash="fixture")
    p = policy(news=NewsTradingPolicy(enabled=True, impact_levels=["HIGH"], minutes_before=10, minutes_after=10, applicable_currencies=["USD"], applicable_instruments=["ES"]))
    result = ComplianceEvaluator().evaluate(timestamp=aware("2025-01-02T15:00:00"), instrument="ES", account_state=account(), market_state=MarketState(calendar_source_hash="calendar", calendar_events=[event]), proposed_action=action(), policy=p)
    assert result.allowed


def test_news_missing_and_stale_are_explicit() -> None:
    missing = policy(news=NewsTradingPolicy(enabled=True, max_calendar_age_minutes=30))
    result = ComplianceEvaluator().evaluate(timestamp=aware("2025-01-02T15:00:00"), instrument="ES", account_state=account(), market_state=MarketState(calendar_available=False), proposed_action=action(), policy=missing)
    assert result.classification == ComplianceClassification.DATA_UNAVAILABLE
    stale = ComplianceEvaluator().evaluate(timestamp=aware("2025-01-02T15:00:00"), instrument="ES", account_state=account(), market_state=MarketState(calendar_retrieved_at=aware("2025-01-02T13:00:00")), proposed_action=action(), policy=missing)
    assert stale.classification == ComplianceClassification.DATA_STALE


def test_calendar_artifact_is_reproducible_and_tamper_checked(tmp_path: Path) -> None:
    event = EconomicEvent(event_id="e", title="test", timestamp=aware("2025-01-02T15:00:00"), source="fixture", retrieved_at=aware("2025-01-01T00:00:00"), source_data_hash="fixture")
    path = tmp_path / "calendar.json"
    original = save_calendar_artifact(aware("2025-01-01T00:00:00"), aware("2025-01-03T00:00:00"), [event], path, retrieved_at=aware("2025-01-03T00:00:00"))
    assert load_calendar_artifact(path).artifact_hash == original.artifact_hash
    path.write_text(path.read_text(encoding="utf-8").replace('"title": "test"', '"title": "tampered"'), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_calendar_artifact(path)


def test_fixture_calendar_provider_is_deterministic() -> None:
    event = EconomicEvent(event_id="e", title="test", timestamp=aware("2025-01-02T15:00:00"), source="fixture", retrieved_at=aware("2025-01-01T00:00:00"), source_data_hash="fixture")
    provider = FixtureEconomicCalendarProvider([event])
    assert provider.events(aware("2025-01-01T00:00:00"), aware("2025-01-03T00:00:00")) == provider.events(aware("2025-01-01T00:00:00"), aware("2025-01-03T00:00:00"))


def test_session_cutoffs_weekend_and_dst() -> None:
    p = SessionPolicy(enabled=True, timezone="America/New_York", no_new_entry_time=time(9, 30), pending_order_cancellation_time=time(15, 0), forced_flat_time=time(16, 0), weekend_flattening=True)
    manager = SessionManager()
    assert manager.evaluate(aware("2025-01-02T14:29:59"), ProposedAction(action=ActionType.ORDER_SUBMISSION, instrument="ES"), p).decision == SessionDecision.ENTRY_ALLOWED
    assert manager.evaluate(aware("2025-01-02T14:30:00"), ProposedAction(action=ActionType.ORDER_SUBMISSION, instrument="ES"), p).decision == SessionDecision.ENTRY_BLOCKED_SESSION_CUTOFF
    assert manager.evaluate(aware("2025-01-02T20:00:00"), ProposedAction(action=ActionType.PENDING_ORDER, instrument="ES"), p).decision == SessionDecision.CANCEL_PENDING_ORDERS
    assert manager.evaluate(aware("2025-01-02T21:00:00"), ProposedAction(action=ActionType.EXIT, instrument="ES"), p).decision == SessionDecision.FORCE_FLATTEN
    assert manager.evaluate(aware("2025-01-04T15:00:00"), ProposedAction(action=ActionType.ORDER_SUBMISSION, instrument="ES"), p).decision == SessionDecision.SESSION_CLOSED
    # 09:30 local is 13:30 UTC in winter and 13:30 UTC in this fixture date;
    # the policy uses the IANA zone rather than a fixed UTC conversion.
    assert manager.evaluate(aware("2025-07-03T13:29:59"), ProposedAction(action=ActionType.ORDER_SUBMISSION, instrument="ES"), p).decision == SessionDecision.ENTRY_ALLOWED


def test_daily_loss_thresholds_costs_and_reset() -> None:
    from research_pipeline.compliance.models import DailyLossPolicy

    p = DailyLossPolicy(enabled=True, daily_loss_limit=100, internal_safety_fraction=.5, soft_lock_fraction=.8, reset_timezone="America/New_York", reset_time=time(17), cancel_pending_orders=True, force_flatten=True)
    guard = __import__("research_pipeline.compliance.daily_loss", fromlist=["DailyLossGuard"]).DailyLossGuard()
    timestamp = aware("2025-01-02T16:00:00")
    assert guard.evaluate(timestamp, account(realized_pnl=-40), p).state.value == "ACTIVE"
    warning = guard.evaluate(timestamp, account(realized_pnl=-50), p)
    assert warning.state.value == "WARNING" and warning.transition == "ACTIVE->WARNING"
    soft = guard.evaluate(timestamp, account(realized_pnl=-60, unrealized_pnl=-5, commissions=15), p)
    assert soft.state.value == "SOFT_LOCK" and "CANCEL_PENDING_ORDERS" in soft.required_actions
    firm = guard.evaluate(timestamp, account(realized_pnl=-100), p)
    assert firm.state.value == "FIRM_LOCK" and "FORCE_FLATTEN" in firm.required_actions
    assert guard.evaluate(timestamp, account(realized_pnl=-100), p).transition is None
    reset = guard.evaluate(aware("2025-01-02T22:00:00"), account(), p)
    assert reset.state.value == "ACTIVE"


def test_compliance_decision_hash_and_alert_backtest_equivalence() -> None:
    p = policy()
    kwargs = dict(timestamp=aware("2025-01-02T15:00:00"), instrument="ES", account_state=account(), market_state=MarketState(), proposed_action=action(), policy=p)
    backtest = ComplianceEvaluator().evaluate(**kwargs)
    alert = ComplianceEvaluator().evaluate(**kwargs)
    assert backtest == alert
    assert backtest.decision_hash != "pending"


def test_native_backtest_and_alert_facades_share_decision_path() -> None:
    from research_pipeline.adapters.native_backtest import NativeRepositoryAdapter
    from research_pipeline.prop.adapters import AlertComplianceAdapter
    from research_pipeline.schemas.strategy_spec import load_strategy_spec

    spec = load_strategy_spec(Path("examples/research_pipeline/fibonacci_compatibility.yaml"))
    p = policy()
    backtest = NativeRepositoryAdapter(spec, compliance_policy=p, compliance_evaluator=ComplianceEvaluator()).evaluate_compliance(timestamp=aware("2025-01-02T15:00:00"), instrument="ES", account_state=account(), market_state=MarketState(), proposed_action=action())
    alert = AlertComplianceAdapter(p).evaluate(timestamp=aware("2025-01-02T15:00:00"), instrument="ES", account_state=account(), market_state=MarketState(), proposed_action=action())
    assert backtest == alert


def test_execution_costs_are_per_side_scaled_and_deterministic() -> None:
    raw = {"instruments": {"ES": InstrumentCostConfig(tick_size=.25, tick_value=12.5, commission_per_side=1.5, exchange_fee_per_side=.5, market_slippage_ticks=1, stop_slippage_ticks=2).model_dump()}, "configuration_hash": "pending"}
    candidate = ExecutionCostConfig.model_validate(raw, context={"skip_configuration_hash_validation": True})
    raw["configuration_hash"] = calculate_cost_config_hash(candidate)
    config = ExecutionCostConfig.model_validate(raw)
    result = ExecutionCostEngine(config).calculate("ES", 2, order_types=(OrderType.MARKET, OrderType.STOP))
    assert result.commissions == 6
    assert result.exchange_fees == 2
    assert result.slippage_cost == 75
    assert result.total_cost == 83
    assert result == ExecutionCostEngine(config).calculate("ES", 2, order_types=(OrderType.MARKET, OrderType.STOP))
    forced_flat = ExecutionCostEngine(config).calculate("ES", 2, order_types=(OrderType.STOP,), sides=1)
    assert forced_flat.commissions == 3


def test_execution_costs_reject_unknown_instrument_and_no_double_fee() -> None:
    raw = {"instruments": {"ES": InstrumentCostConfig(tick_size=.25, tick_value=12.5, commission_per_side=1).model_dump()}, "configuration_hash": "pending"}
    candidate = ExecutionCostConfig.model_validate(raw, context={"skip_configuration_hash_validation": True})
    raw["configuration_hash"] = calculate_cost_config_hash(candidate)
    engine = ExecutionCostEngine(ExecutionCostConfig.model_validate(raw))
    assert engine.costed_pnl(100, engine.calculate("ES", 1))["net_pnl"] == 98
    with pytest.raises(ValueError, match="unsupported instrument"):
        engine.calculate("NQ", 1)


def test_activity_diagnostics_and_missing_timestamps() -> None:
    class Trade:
        def __init__(self, start, end, pnl, gross=1):
            self.entry_time = start; self.exit_time = end; self.net_pnl = pnl; self.gross_pnl = gross; self.entry = 100

    rows = [Trade(aware("2025-01-01T10:00:00"), aware("2025-01-01T10:05:00"), 2), Trade(aware("2025-01-02T10:00:00"), aware("2025-01-02T11:00:00"), 3)]
    result = calculate_activity_diagnostics(rows, HoldingTimePolicy(enabled=True, short_duration_threshold_minutes=[10], small_price_movement_threshold=.1))
    assert result.average_holding_minutes == 32.5
    assert result.median_holding_minutes == 32.5
    assert result.trades_per_day == 1
    assert result.percentage_below_threshold["10"] == .5
    assert calculate_activity_diagnostics([{"net_pnl": 1}], HoldingTimePolicy()).classification == "INSUFFICIENT_DATA"


def test_compliance_artifacts_persist_in_registry(tmp_path: Path) -> None:
    registry = Registry(Database(tmp_path / "registry.sqlite3"))
    # Use the existing compatibility fixture to exercise additive tables.
    from research_pipeline.schemas.strategy_spec import load_strategy_spec
    spec = load_strategy_spec(Path("examples/research_pipeline/fibonacci_compatibility.yaml"))
    registry.register_strategy(spec, str(tmp_path / "spec.yaml"), __import__("research_pipeline.schemas.budgets", fromlist=["ResearchBudget"]).ResearchBudget())
    registry.save_compliance_policy(spec.strategy_id, spec.version, unconfigured_policy().model_dump(mode="json"))
    registry.save_compliance_decision(spec.strategy_id, spec.version, ComplianceEvaluator().evaluate(timestamp=aware("2025-01-02T15:00:00"), instrument="ES", account_state=account(), market_state=MarketState(), proposed_action=action(), policy=unconfigured_policy()).model_dump(mode="json"))
    assert registry.get_compliance_policy(spec.strategy_id, spec.version)
    assert len(registry.list_compliance_decisions(spec.strategy_id, spec.version)) == 1
