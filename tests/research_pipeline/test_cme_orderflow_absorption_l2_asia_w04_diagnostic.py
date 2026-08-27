from __future__ import annotations

import inspect
from collections import Counter
from decimal import Decimal
from pathlib import Path

import pytest

from research_pipeline.cme_orderflow_absorption_l2_v1 import asia_w04_diagnostic as diagnostic
from research_pipeline.cme_orderflow_absorption_l2_v1 import asia_w04_diagnostic_checkpoint_worker as checkpoint_worker
from research_pipeline.cme_orderflow_absorption_l2_v1 import asia_w04_replay as asia


def _row(
    identifier: str,
    *,
    score: float,
    accepted: bool,
    reasons: str = "",
    direction: str = "BUYER_ABSORPTION",
    g: tuple[float, float, float, float, float] = (0.5, 0.5, 0.5, 0.5, 0.5),
    penalty: float = 0.0,
    start_ns: int = 1_000_000_000,
) -> dict[str, object]:
    primitive = ";".join(reason for reason in reasons.split(";") if reason and reason != "L2_QUALITY_BELOW_THRESHOLD")
    return {
        "interaction_id": identifier,
        "session_date": "2026-01-02",
        "interaction_start_ns": start_ns,
        "interaction_end_ns": start_ns + 2_000_000_000,
        "direction": direction,
        "level": asia.ASIA_LEVEL_NAME,
        "level_price": 5_000.0,
        "interaction_end_price": 5_000.25,
        "zone_low": 4_999.75,
        "zone_high": 5_000.25,
        "termination": "VICINITY_TIMEOUT",
        "execution_count": 4,
        "aggression_score": g[0],
        "restoration_score": g[1],
        "price_resistance_score": g[2],
        "persistence_score": g[3],
        "multi_level_support_score": g[4],
        "false_refill_penalty": penalty,
        "w04_quality_score": score,
        "non_quality_rejection_reasons": primitive,
        "rejection_reasons": reasons,
        "accepted": accepted,
    }


def _session(rows: list[dict[str, object]], terminals: dict[str, str] | None = None) -> dict[str, object]:
    terminals = terminals or {}
    indexes = []
    trades = []
    for row in rows:
        identifier = str(row["interaction_id"])
        if not bool(row["accepted"]):
            continue
        passed = terminals.get(identifier) == "TRADE_EXECUTED"
        indexes.append({
            "interaction_id": identifier,
            "derived_first_confirmation_timestamp_ns": 5_000_000_000 if passed else None,
        })
        if passed:
            trades.append({"interaction_id": identifier, "trade_id": f"trade:{identifier}"})
    return {
        "eligible": True,
        "session_date": "2026-01-02",
        "interactions": rows,
        "indexes": indexes,
        "terminal_outcomes": terminals,
        "trades": trades,
    }


def test_every_interaction_receives_one_setup_and_final_terminal_disposition() -> None:
    rows = [
        _row("low", score=0.20, accepted=False, reasons="L2_QUALITY_BELOW_THRESHOLD"),
        _row("hard", score=0.40, accepted=False, reasons="NO_GENUINE_CONSUME_RESTORE;L2_QUALITY_BELOW_THRESHOLD"),
        _row("accepted", score=0.50, accepted=True),
    ]
    enriched = diagnostic.enrich_interactions([_session(rows, {"accepted": "TRADE_EXECUTED"})])
    assert Counter(row["setup_terminal_disposition"] for row in enriched) == {
        "L2_QUALITY_BELOW_THRESHOLD": 1,
        "NO_GENUINE_CONSUME_RESTORE": 1,
        "SETUP_ACCEPTED": 1,
    }
    assert Counter(row["final_pipeline_disposition"] for row in enriched) == {
        "L2_QUALITY_BELOW_THRESHOLD": 1,
        "NO_GENUINE_CONSUME_RESTORE": 1,
        "TRADE_EXECUTED": 1,
    }


def test_funnel_accepted_and_confirmation_counts_reconcile() -> None:
    rows = [
        _row("rejected", score=0.30, accepted=False, reasons="L2_QUALITY_BELOW_THRESHOLD"),
        _row("failed", score=0.50, accepted=True),
        _row("passed", score=0.50, accepted=True),
    ]
    enriched = diagnostic.enrich_interactions([_session(rows, {
        "failed": "CONFIRMATION_WINDOW_EXPIRED",
        "passed": "TRADE_EXECUTED",
    })])
    funnel = {row["stage_or_disposition"]: row["count"] for row in diagnostic._funnel_rows(enriched) if row["row_type"] == "SEQUENTIAL"}
    assert funnel == {
        "SETUP_ACCEPTED": 2,
        "CONFIRMATION_FAILED": 1,
        "CONFIRMATION_PASSED": 1,
        "TRADE_EXECUTED": 1,
    }


def test_missing_terminal_for_accepted_setup_fails_closed() -> None:
    with pytest.raises(diagnostic.AsiaW04DiagnosticError, match="lacks lifecycle audit"):
        diagnostic.enrich_interactions([_session([_row("accepted", score=0.50, accepted=True)])])


def test_weighted_contributions_use_frozen_w04_weights_and_penalty() -> None:
    row = _row(
        "weighted", score=0.35, accepted=False,
        reasons="L2_QUALITY_BELOW_THRESHOLD", g=(0.5, 0.4, 0.3, 0.2, 0.1), penalty=0.2,
    )
    result = diagnostic.weighted_contributions(row)
    assert result == pytest.approx({
        "G1": 0.10,
        "G2": 0.04,
        "G3": 0.09,
        "G4": 0.04,
        "G5": 0.02,
        "FALSE_REFILL_PENALTY": -0.05,
    })


def test_quality_distance_buckets_are_exact_and_not_parameter_search() -> None:
    assert diagnostic.quality_bucket(Decimal("0.45")) == "ACCEPTED_SCORE"
    assert diagnostic.quality_bucket(Decimal("0.449999")) == "NEAR_MISS_0P40_TO_0P45"
    assert diagnostic.quality_bucket(Decimal("0.40")) == "NEAR_MISS_0P40_TO_0P45"
    assert diagnostic.quality_bucket(Decimal("0.30")) == "MEDIUM_MISS_0P30_TO_0P40"
    assert diagnostic.quality_bucket(Decimal("0.29999")) == "LOW_BELOW_0P30"


def test_counterfactual_one_feature_lift_uses_raw_domain_and_frozen_weight() -> None:
    row = _row(
        "near", score=0.40, accepted=False,
        reasons="L2_QUALITY_BELOW_THRESHOLD", g=(0.4, 0.4, 0.4, 0.4, 0.4),
    )
    lifts = {item["component"]: item for item in diagnostic.counterfactual_one_feature_lifts([row])}
    assert lifts["G1"]["mathematically_reachable_count"] == 1
    assert lifts["G1"]["median_required_raw_g_increase"] == pytest.approx(0.25)
    assert lifts["G1"]["median_required_weighted_contribution_increase"] == pytest.approx(0.05)
    assert lifts["G3"]["median_required_raw_g_increase"] == pytest.approx(1 / 6)
    assert lifts["G3"]["median_required_weighted_contribution_increase"] == pytest.approx(0.05)


def test_rejected_accepted_and_direction_groups_are_independent() -> None:
    rows = [
        _row("long-r", score=0.30, accepted=False, reasons="L2_QUALITY_BELOW_THRESHOLD"),
        _row("long-a", score=0.50, accepted=True),
        _row("short-r", score=0.30, accepted=False, reasons="L2_QUALITY_BELOW_THRESHOLD", direction="SELLER_ABSORPTION"),
    ]
    enriched = diagnostic.enrich_interactions([_session(rows, {"long-a": "CONFIRMATION_WINDOW_EXPIRED"})])
    summary = diagnostic.component_summary_rows(enriched)
    counts = {
        (row["group"], row["component"]): row["count"]
        for row in summary if row["metric_type"] == "RAW_G"
    }
    assert counts[("ALL_VALID_FEATURE_INTERACTIONS", "G1")] == 3
    assert counts[("REJECTED_INTERACTIONS", "G1")] == 2
    assert counts[("ACCEPTED_SETUPS", "G1")] == 1
    assert counts[("LONG_SIDE_CANDIDATES", "G1")] == 2
    assert counts[("SHORT_SIDE_CANDIDATES", "G1")] == 1


def test_completion_mechanism_is_not_invented_as_rejection() -> None:
    row = _row("rth", score=0.20, accepted=False, reasons="L2_QUALITY_BELOW_THRESHOLD")
    row["termination"] = "RTH_END"
    enriched = diagnostic.enrich_interactions([_session([row])])
    reasons = diagnostic._reason_rows(enriched)
    assert any(item["scope"] == "INTERACTION_COMPLETION_MECHANISM" and item["reason"] == "RTH_END" for item in reasons)
    assert enriched[0]["setup_terminal_disposition"] == "L2_QUALITY_BELOW_THRESHOLD"


def test_frozen_rejection_order_drives_exclusive_attribution() -> None:
    row = _row(
        "multi", score=0.10, accepted=False,
        reasons="INSUFFICIENT_RELEVANT_AGGRESSION;NO_GENUINE_CONSUME_RESTORE;L2_QUALITY_BELOW_THRESHOLD",
    )
    enriched = diagnostic.enrich_interactions([_session([row])])
    assert enriched[0]["setup_terminal_disposition"] == "INSUFFICIENT_RELEVANT_AGGRESSION"


def test_diagnostic_reuses_frozen_strategy_and_has_no_network_path() -> None:
    source = inspect.getsource(diagnostic)
    assert diagnostic.Q == asia.QUALITY_THRESHOLD == Decimal("0.45")
    assert tuple(weight for _name, _field, weight in diagnostic.G_COMPONENTS) == tuple(asia.W04_WEIGHTS.values())
    assert "_process_daily_binding" in source
    assert "_process_shared_binding" in source
    assert "metadata.get_cost" not in source
    assert "timeseries.get_range" not in source
    assert "DATABENTO_API_KEY" not in source
    assert "Historical(" not in source


def test_existing_diagnostic_output_fails_before_any_source_or_baseline_read(tmp_path: Path) -> None:
    output = tmp_path / "exists"
    output.mkdir()
    with pytest.raises(FileExistsError, match="immutable Asia diagnostic output"):
        diagnostic.run_diagnostic(
            repository_root=tmp_path,
            output_root=output,
            baseline_root=tmp_path / "missing-baseline",
            audit_root=tmp_path / "missing-audit",
        )


def test_checkpoint_worker_accepts_only_exact_published_daily_result() -> None:
    result = {
        "session_date": "2026-05-04",
        "raw_interactions": 3,
        "accepted_setups": 1,
        "confirmations_passed": 1,
        "confirmation_failures": 0,
        "unresolved": 0,
        "prior_asia_poc": 5_000.25,
        "session_poc": 5_001.0,
    }
    expected = {key: str(value) for key, value in result.items()}
    checkpoint_worker._validate_result(result, expected)

    result["accepted_setups"] = 2
    with pytest.raises(diagnostic.AsiaW04DiagnosticError, match="does not reproduce baseline"):
        checkpoint_worker._validate_result(result, expected)


def test_checkpoint_worker_is_bounded_to_existing_local_sources() -> None:
    source = inspect.getsource(checkpoint_worker)
    assert "_stream_private_mbo" in source
    assert "_save_checkpoint" in source
    assert "metadata.get_cost" not in source
    assert "timeseries.get_range" not in source
    assert "DATABENTO_API_KEY" not in source
    assert "Historical(" not in source
