from pathlib import Path

import pytest

from research_pipeline.prop.contracts import contract_registry_hash, default_contract_registry, get_contract
from research_pipeline.prop.budgets import PropBudgetEnforcer, PropBudgetExceeded
from research_pipeline.prop.fixtures import run_prop_dry_run
from research_pipeline.prop.mappings import default_market_mappings, validate_mappings
from research_pipeline.prop.models import RiskPolicy, TradeSignal
from research_pipeline.prop.models import PropBudget, PropBudgetUsage
from research_pipeline.prop.reconcile import reconcile_trade
from research_pipeline.prop.rule_registry import default_alpha_zero_rules, rule_hash, verify_rules, verified_rule_registry
from research_pipeline.prop.services import PropResearchService
from research_pipeline.prop.session import session_mark
from research_pipeline.prop.sizing import SharedExposure, risk_per_contract, size_trade


def run_fixture(tmp_path: Path, scenario: str = "profitable") -> dict:
    return run_prop_dry_run(tmp_path / "registry.sqlite3", tmp_path, f"phase-d-{scenario}", scenario)


def test_official_zero_rules_are_hashed_and_no_100k_default():
    rules = verified_rule_registry()
    assert set(rules) == {"Alpha Futures Zero 25K", "Alpha Futures Zero 50K"}
    assert rules["Alpha Futures Zero 25K"].reset_fee == 69
    assert rules["Alpha Futures Zero 50K"].reset_fee == 109
    assert all(not verify_rules(rule) for rule in rules.values())
    assert rule_hash(rules["Alpha Futures Zero 25K"]) == rules["Alpha Futures Zero 25K"].source_hash


def test_prop_budget_is_deterministic_and_blocks_before_execution():
    limits = PropBudget(max_scenarios=1, max_accounts_per_scenario=1, max_replay_duration_days=10, max_policy_variants=1, max_concurrent_evaluations=1)
    usage = PropBudgetUsage()
    consumed = PropBudgetEnforcer.consume(limits, usage, scenarios=1, accounts=1, replay_days=10, policy_variants=1, concurrent_evaluations=1)
    assert consumed.scenarios == 1 and consumed.replay_days == 10
    with pytest.raises(PropBudgetExceeded): PropBudgetEnforcer.consume(limits, consumed, scenarios=1)


def test_stale_provider_rules_are_blocked():
    rule = next(iter(default_alpha_zero_rules().values())).model_copy(update={"source_hash": "bad"})
    assert "rule hash mismatch" in verify_rules(rule)


def test_canonical_contract_registry_is_stable():
    registry = default_contract_registry()
    assert set(registry) == {"MBT", "MET", "MGC", "SIL", "MES", "MNQ", "MCL"}
    assert get_contract("mbt", registry).tick_value == 0.5
    assert contract_registry_hash(registry) == contract_registry_hash(registry)


def test_mapping_validation_rejects_cross_asset_and_bad_proxy():
    mappings = default_market_mappings()
    assert not validate_mappings(mappings)
    assert validate_mappings([mappings[0].model_copy(update={"target_futures_contract": "MET"})])
    assert validate_mappings([mappings[0].model_copy(update={"native_or_proxy": "proxy", "confidence_level": "NATIVE_FUTURES_SUPPORTED"})])


def make_trade(direction: str = "LONG") -> TradeSignal:
    return TradeSignal(trade_id="t-1", timestamp="2020-01-01T12:00:00Z", exit_timestamp="2020-01-01T12:30:00Z", source_market="BTCUSDT", timeframe="1h", direction=direction, entry_price=10000, initial_stop_price=9000, exit_price=11000 if direction == "LONG" else 9000, source_return=.1, fees=1, slippage=.5)


def test_futures_reconciliation_uses_tick_value_and_hashes():
    trade = make_trade()
    mapping = next(item for item in default_market_mappings() if item.strategy_market == "BTCUSDT")
    result = reconcile_trade(trade, mapping, 2, default_contract_registry())
    assert result.gross_pnl == 200
    assert result.net_pnl == 197
    assert result.mapping_hash and result.contract_registry_hash


def test_sizing_rounds_down_and_shared_exposure_caps():
    rule = verified_rule_registry()["Alpha Futures Zero 25K"]
    contract = get_contract("MBT")
    policy = RiskPolicy(name="fixed", kind="FIXED_INITIAL_DOLLAR_RISK", dollar_risk=250)
    exposure = SharedExposure(2)
    first = size_trade("a", make_trade(), contract, rule, policy, exposure, 25000, 1000)
    second = size_trade("a", make_trade().model_copy(update={"trade_id": "t-2"}), contract, rule, policy, exposure, 25000, 1000)
    assert risk_per_contract(make_trade(), contract) == 100
    assert first.legal_contracts == 2 and second.legal_contracts == 0
    assert second.skipped_reason == "ACCOUNT_CONTRACT_LIMIT"


def test_prop_fixture_positive_reaches_standalone_and_persists_ledgers(tmp_path: Path):
    result = run_fixture(tmp_path)
    assert result["final_classification"] == "PROP_ACCEPTED_STANDALONE"
    service = PropResearchService(tmp_path / "registry.sqlite3", tmp_path, "profitable")
    status = service.status("phase-d-profitable")
    metrics = status["scenarios"][0]["result_json"]["metrics"]
    assert metrics["first_payouts"] == 1
    assert metrics["net_trading_pnl"] == metrics["gross_trading_pnl"] - metrics["fees"] - metrics["slippage"]
    assert status["prop_run"]["current_phase"] == "COMPLETE"
    assert status["holdout_accesses"] == 1


def test_high_pass_without_payout_is_insufficient_evidence(tmp_path: Path):
    assert run_fixture(tmp_path, "high-pass-zero-payout")["final_classification"] == "INSUFFICIENT_PROP_EVIDENCE"


def test_unsupported_mapping_is_insufficient_futures_data(tmp_path: Path):
    result = run_fixture(tmp_path, "unsupported-mapping")
    assert result["final_classification"] == "INSUFFICIENT_FUTURES_DATA"
    assert result["status"]["prop_run"]["current_phase"] == "CONTRACT_VERIFICATION"


def test_noncompliant_account_model_is_rejected(tmp_path: Path):
    assert run_fixture(tmp_path, "noncompliant")["final_classification"] == "REJECTED_PROP_INCOMPATIBLE"


def test_negative_external_economics_can_be_own_capital_only(tmp_path: Path):
    assert run_fixture(tmp_path, "negative-economics")["final_classification"] == "OWN_CAPITAL_ONLY"


def test_proxy_scenario_does_not_claim_native_futures_evidence(tmp_path: Path):
    result = run_fixture(tmp_path, "synthetic-proxy")
    assert result["final_classification"] == "INSUFFICIENT_FUTURES_DATA"


def test_mll_failure_records_cancellation_and_does_not_continue(tmp_path: Path):
    run_fixture(tmp_path, "mll-sensitive")
    service = PropResearchService(tmp_path / "registry.sqlite3", tmp_path, "mll-sensitive")
    status = service.status("phase-d-mll-sensitive")
    account = status["scenarios"][0]["result_json"]["accounts"][0]
    assert account["status"] == "FAILED"
    assert account["cancellation_timestamp"] is not None
    assert any(event["event_type"] == "EVALUATION_CANCELLED" for event in status["scenarios"][0]["result_json"]["billing_events"])


def test_prop_artifacts_and_journal_survive_new_service_instance(tmp_path: Path):
    run_fixture(tmp_path)
    reopened = PropResearchService(tmp_path / "registry.sqlite3", tmp_path, "profitable")
    assert reopened.status("phase-d-profitable")["final_review"] is not None
    assert len(reopened.journal("phase-d-profitable")) >= 6


def test_completed_fixture_is_idempotent(tmp_path: Path):
    first = run_fixture(tmp_path)
    second = run_fixture(tmp_path)
    assert first["final_classification"] == second["final_classification"]
    assert second["idempotent"] is True


def test_phase_d_does_not_change_frozen_phase_c_parameters(tmp_path: Path):
    run_fixture(tmp_path)
    service = PropResearchService(tmp_path / "registry.sqlite3", tmp_path, "profitable")
    strategy = service.registry.get_strategy("phase-d-profitable")
    assert strategy["parameters_frozen"] == 1
    assert service.registry.count_holdout_accesses("phase-d-profitable") == 1


def test_session_mark_to_market_is_explicit_and_deterministic():
    event = session_mark("account", make_trade().exit_timestamp, 25000, 100, "CME crypto", unrealized_pnl=5, forced_exit=True, mll_threshold=24000, daily_realized_pnl=100)
    assert event.event_type == "SESSION_FORCED_EXIT"
    assert event.marked_equity == 25005
    assert event.remaining_mll_buffer == 1005
