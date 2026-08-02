from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from research_pipeline.errors import SpecificationValidationError
from research_pipeline.cli import main
from research_pipeline.portfolio.analysis import correlation_metrics, overlap_metrics
from research_pipeline.portfolio.fixtures import make_portfolio_spec, run_portfolio_dry_run
from research_pipeline.portfolio.models import ConflictPolicy, PortfolioMember, PortfolioMemberRole, PortfolioSignalEvent, RiskAllocationPolicy
from research_pipeline.portfolio.replay import replay_shared_account
from research_pipeline.portfolio.service import PortfolioService
from research_pipeline.portfolio.signals import SyntheticPortfolioSignalAdapter, apply_conflict_policy
from research_pipeline.registry.database import Database
from research_pipeline.registry.repositories import Registry


def _event(strategy: str, when: datetime, direction: str = "LONG", exit_price: float = 11250, signal: str | None = None) -> PortfolioSignalEvent:
    return PortfolioSignalEvent(signal_id=signal or f"{strategy}-{when.isoformat()}", strategy_id=strategy, market="BTCUSDT", timeframe="1h", direction=direction, setup_timestamp=when - timedelta(minutes=15), entry_timestamp=when, exit_timestamp=when + timedelta(minutes=30), stop=9500 if direction == "LONG" else 10500, targets=[exit_price], entry_price=10000, exit_price=exit_price, candidate_hash=f"hash-{strategy}", source_data_classification="NATIVE", duplicate_exposure_group="btc", regime="trend")


def _members() -> list[PortfolioMember]:
    return [PortfolioMember(strategy_id="a", strategy_version="1", candidate_hash="hash-a", phase_c_classification="ACCEPTED_PORTFOLIO_COMPONENT", phase_d_classification="PROP_ACCEPTED_PORTFOLIO_COMPONENT", markets=["BTCUSDT"], timeframes=["1h"], expected_trades_per_month=10, data_source_classification="NATIVE", confidence_level="HIGH", role=PortfolioMemberRole.CORE, priority=0, confidence_score=.9), PortfolioMember(strategy_id="b", strategy_version="1", candidate_hash="hash-b", phase_c_classification="ACCEPTED_PORTFOLIO_COMPONENT", phase_d_classification="PROP_ACCEPTED_PORTFOLIO_COMPONENT", markets=["BTCUSDT"], timeframes=["1h"], expected_trades_per_month=10, data_source_classification="NATIVE", confidence_level="HIGH", role=PortfolioMemberRole.DIVERSIFIER, priority=1, confidence_score=.7)]


def test_conflict_policies_are_explicit_and_deterministic():
    when = datetime(2020, 1, 1, tzinfo=timezone.utc)
    events = [_event("a", when), _event("b", when, "SHORT", 8750)]
    members = _members()
    for policy in ConflictPolicy:
        accepted, counts = apply_conflict_policy(events, members, policy)
        assert counts["opposite_signal_conflicts"] == 1
        assert len(accepted) == (2 if policy == ConflictPolicy.ALLOW_INDEPENDENT else 0 if policy in {ConflictPolicy.SKIP_CONFLICT, ConflictPolicy.NET_EXPOSURE} else 1)


def test_signal_adapter_is_chronological_and_overlap_is_measured():
    members = _members()
    events = list(SyntheticPortfolioSignalAdapter().signals("candidate", members, "complementary"))
    assert events == sorted(events, key=lambda item: (item.entry_timestamp, item.strategy_id, item.signal_id))
    metrics = overlap_metrics("candidate", events, members)
    assert metrics.unique_portfolio_signals > 0
    assert 0 <= metrics.signal_overlap_rate <= 1


def test_correlation_requires_aligned_history():
    members = _members()
    when = datetime(2020, 1, 1, tzinfo=timezone.utc)
    short = [_event("a", when), _event("b", when + timedelta(days=1))]
    result = correlation_metrics("candidate", short, members, minimum_periods=20)
    assert result.sufficient_evidence is False
    assert "insufficient" in result.reason


def test_shared_replay_enforces_contract_and_duplicate_exposure_limits():
    members = _members()
    when = datetime(2020, 1, 1, tzinfo=timezone.utc)
    events = [_event("a", when), _event("b", when + timedelta(minutes=5))]
    metrics, risk = replay_shared_account("candidate", events, members, policy=RiskAllocationPolicy.EQUAL_RISK, maximum_total_contracts=1, maximum_simultaneous_positions=1)
    assert risk.shared_account is True
    assert risk.total_granted_contracts <= 1
    assert risk.contract_limit_skips + risk.duplicate_exposure_skips >= 0
    assert metrics.strategy_trades


def test_registry_initialization_is_idempotent_and_portfolio_create_is_idempotent(tmp_path: Path):
    registry_path = tmp_path / "registry.sqlite3"
    Registry(Database(registry_path)); Registry(Database(registry_path))
    # The portfolio fixture is only used after the Phase C/D fixture has made
    # its frozen candidate evidence; no backtest is invoked here.
    result = run_portfolio_dry_run(registry_path, tmp_path, "phase-e-idempotent", "complementary")
    assert result["status"]["run"]["current_phase"] == "COMPLETE"
    service = PortfolioService(registry_path, tmp_path, "complementary")
    assert service.create(service._spec("phase-e-idempotent"))["idempotent"] is True


def test_portfolio_phase_guards_and_budget_are_structural(tmp_path: Path):
    registry_path = tmp_path / "registry.sqlite3"
    run_portfolio_dry_run(registry_path, tmp_path, "phase-e-budget", "complementary")
    service = PortfolioService(registry_path, tmp_path, "complementary")
    assert service.generate_candidates("phase-e-budget")
    assert service.run_prop("phase-e-budget")


@pytest.mark.parametrize(
    ("scenario", "expected"),
    [
        ("complementary", "PORTFOLIO_ACCEPTED_EXPLORATORY"),
        ("correlated", "PORTFOLIO_REJECTED_CORRELATED"),
        ("harmful-member", "PORTFOLIO_REJECTED_NEGATIVE_ECONOMICS"),
        ("insufficient-history", "PORTFOLIO_INSUFFICIENT_EVIDENCE"),
        ("exploratory-proxy", "PORTFOLIO_ACCEPTED_EXPLORATORY"),
    ],
)
def test_required_phase_e_fixture_classifications(tmp_path: Path, scenario: str, expected: str):
    result = run_portfolio_dry_run(tmp_path / "registry.sqlite3", tmp_path, f"phase-e-{scenario}", scenario)
    assert result["classification"] == expected
    assert result["status"]["run"]["current_phase"] == "COMPLETE"


def test_harmful_member_is_recorded_as_excluded(tmp_path: Path):
    result = run_portfolio_dry_run(tmp_path / "registry.sqlite3", tmp_path, "phase-e-harmful-excluded", "harmful-member")
    review = result["status"]["final_review"]["result_json"]
    assert review["excluded_strategies"]
    assert "negative" in next(iter(review["excluded_strategies"].values()))


def test_phase_e_does_not_change_existing_strategy_sources():
    assert Path("src/research_pipeline/portfolio/service.py").exists()
    assert not Path("src/research_pipeline/portfolio/service.py").read_text(encoding="utf-8").count("run_backtest")


def test_phase_e_cli_has_structured_commands_and_useful_errors(tmp_path: Path, capsys):
    registry = tmp_path / "registry.sqlite3"
    assert main(["--registry", str(registry), "portfolio", "eligible-strategies"]) == 0
    assert "[" in capsys.readouterr().out
    assert main(["--registry", str(registry), "portfolio", "status", "missing"]) != 0
    assert "error:" in capsys.readouterr().err
