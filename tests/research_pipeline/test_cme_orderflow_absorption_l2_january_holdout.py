from __future__ import annotations

from dataclasses import FrozenInstanceError
from decimal import Decimal
from pathlib import Path

import pytest

from research_pipeline.cme_orderflow_absorption_l2_v1 import causal_master_tape as master
from research_pipeline.cme_orderflow_absorption_l2_v1 import historical_runner as historical
from research_pipeline.cme_orderflow_absorption_l2_v1 import january_holdout_v3_vs_robust as holdout
from research_pipeline.cme_orderflow_absorption_l2_v1 import weight_q_research as matrix


DAY = "2026-01-02"


def _interaction(identifier: str) -> dict[str, object]:
    source = f"PRIOR_RTH_POC:100.00:{identifier}"
    return {
        "interaction_id": f"{DAY}|{source}",
        "source_interaction_id": source,
        "session_date": DAY,
        "interaction_start_ns": 1,
        "interaction_end_ns": 2,
        "direction": "BUYER_ABSORPTION",
        "level": "PRIOR_RTH_POC",
        "level_price": 100.0,
        "interaction_end_price": 100.0,
        "zone_low": 99.0,
        "zone_high": 100.0,
        "aggression_score": 0.80,
        "restoration_score": 0.80,
        "price_resistance_score": 0.80,
        "persistence_score": 0.80,
        "multi_level_support_score": 0.80,
        "false_refill_penalty": 0.0,
        "non_quality_rejection_reasons": "",
    }


def _event(ordinal: int, timestamp: int, bid: float, ask: float) -> dict[str, object]:
    return {
        "session_date": DAY,
        "event_ordinal": ordinal,
        "timestamp_ns": timestamp,
        "stream": "ES",
        "stream_priority": master.STREAM_PRIORITY["ES"],
        "source_index": ordinal + 1,
        "event_type": "ES_BBO",
        "es_bid": bid,
        "es_ask": ask,
        "mes_bid": bid,
        "mes_ask": ask,
        "execution_price": None,
        "execution_size": None,
        "execution_aggressor": None,
        "book_state": "EXECUTABLE",
        "hard_flat_reason": None,
        "es_quote_timestamp_ns": timestamp,
        "mes_quote_timestamp_ns": timestamp,
    }


def _hard(ordinal: int, timestamp: int) -> dict[str, object]:
    row = _event(ordinal, timestamp, 100.0, 100.25)
    row.update({
        "stream": "CALENDAR",
        "stream_priority": master.STREAM_PRIORITY["CALENDAR"],
        "event_type": "HARD_FLAT",
        "hard_flat_reason": "HARD_CUTOFF_2245",
    })
    return row


def _bundle(interactions: list[dict[str, object]]) -> holdout.JanuaryBundle:
    indexes = {
        str(row["interaction_id"]): {
            "interaction_id": row["interaction_id"],
            "session_date": DAY,
            "derived_first_confirmation_timestamp_ns": 5,
            "entry_observation_event_ordinal": 0,
        }
        for row in interactions
    }
    return holdout.JanuaryBundle(
        days=(DAY,),
        interactions_by_day={DAY: tuple(interactions)},
        indexes=indexes,
        interaction_count=len(interactions),
        source_to_master_id={
            (DAY, str(row["source_interaction_id"])): str(row["interaction_id"])
            for row in interactions
        },
    )


def test_january_filter_is_strict_and_excludes_december_and_february():
    calendar = {
        "target_sessions": ["2025-12-31", "2026-01-02", "2026-01-05", "2026-01-30", "2026-02-02"]
    }
    assert holdout.january_session_days(calendar) == ("2026-01-02", "2026-01-05", "2026-01-30")
    with pytest.raises(holdout.JanuaryHoldoutError, match="JANUARY_SESSION_CHRONOLOGY_INVALID"):
        holdout.january_session_days({"target_sessions": ["2026-01-05", "2026-01-02"]})


def test_v3_and_challenger_contracts_are_exact_and_frozen():
    holdout._assert_frozen_contracts()
    assert holdout.V3_CONTRACT.weight_mapping() == {
        "aggression_score": Decimal("0.28"),
        "restoration_score": Decimal("0.25"),
        "price_resistance_score": Decimal("0.22"),
        "persistence_score": Decimal("0.12"),
        "multi_level_support_score": Decimal("0.13"),
    }
    assert holdout.V3_CONTRACT.quality_threshold == Decimal("0.50")
    assert holdout.CHALLENGER_CONTRACT.config_id == "W02-02-07-02-07-Q40"
    assert holdout.CHALLENGER_CONTRACT.weight_mapping() == {
        "aggression_score": Decimal("0.10"),
        "restoration_score": Decimal("0.10"),
        "price_resistance_score": Decimal("0.35"),
        "persistence_score": Decimal("0.10"),
        "multi_level_support_score": Decimal("0.35"),
    }
    assert holdout.CHALLENGER_CONTRACT.quality_threshold == Decimal("0.40")
    assert holdout.V3_CONTRACT_SHA256 == "a0ce94eeb78dcbf865cf4464bdf97ebc3f014a8ec5e2f559f99798534dfcbcb4"
    assert holdout.challenger_contract_sha256() == "5d3d72cd378d0ac986670a3c48ee14344571e1f559746e3fa1f514429e80553a"
    with pytest.raises(FrozenInstanceError):
        holdout.CHALLENGER_CONTRACT.quality_threshold = Decimal("0.45")  # type: ignore[misc]


def test_two_portfolios_have_independent_state_and_one_position_blocking(
    monkeypatch: pytest.MonkeyPatch,
):
    interactions = [_interaction("A"), _interaction("B")]
    events = [
        _event(0, 10, 100.0, 100.25),
        _event(1, 11, 100.25, 100.50),
        _event(2, 12, 109.0, 109.25),
        _hard(3, 13),
    ]
    built: list[matrix.SessionCausalTape] = []

    def fake_from_parquet(_cls, day: str, _path: Path) -> matrix.SessionCausalTape:
        tape = matrix.SessionCausalTape(day, events)
        built.append(tape)
        return tape

    def forbidden(*_args, **_kwargs):
        raise AssertionError("DBN/source acquisition path called")

    monkeypatch.setattr(matrix.SessionCausalTape, "from_parquet", classmethod(fake_from_parquet))
    monkeypatch.setattr(master.native, "_stream_native_mbp10_records", forbidden)
    monkeypatch.setattr(master.historical, "_stream_mes_quotes", forbidden)
    bundle = _bundle(interactions)
    v3 = holdout._run_portfolio(Path("unused"), bundle, holdout.V3_CONTRACT)
    candidate = holdout._run_portfolio(Path("unused"), bundle, holdout.CHALLENGER_CONTRACT)
    assert len(built) == 2 and built[0] is not built[1]
    assert v3["independent_chronological_portfolio_state"] is True
    assert candidate["independent_chronological_portfolio_state"] is True
    assert v3["completed_trades"] == candidate["completed_trades"] == 1
    assert v3["active_position_blocks"] == candidate["active_position_blocks"] == 1
    first, second = (str(row["interaction_id"]) for row in interactions)
    assert v3["terminal_outcomes"] == {
        first: "TRADE_EXECUTED",
        second: "COMPLIANCE_BLOCK_ACTIVE_POSITION",
    }
    assert candidate["terminal_outcomes"] == v3["terminal_outcomes"]


def test_exact_v3_january_reproduction_gate_passes_and_fails_closed():
    passing = {
        "sessions": 20,
        "metrics": {**holdout.EXPECTED_V3_JANUARY},
        "dbn_files_opened": 0,
        "network_calls": 0,
    }
    holdout._assert_expected_v3_gate(passing, 20)
    failing = {**passing, "metrics": {**passing["metrics"], "completed_trades": 26}}
    with pytest.raises(holdout.JanuaryHoldoutError, match="V3_JANUARY_REPRODUCTION_FAILED"):
        holdout._assert_expected_v3_gate(failing, 20)


def test_compact_v3_benchmarks_are_explicitly_period_scoped(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[str, dict[str, object]]] = []

    def capture(name: str, _actual: dict[str, object], expected: dict[str, object]) -> None:
        calls.append((name, dict(expected)))

    monkeypatch.setattr(historical, "_performance", lambda _trades: dict(holdout.EXPECTED_V3_JANUARY))
    monkeypatch.setattr(master, "_assert_metrics", capture)
    matrix._assert_compact_v3(
        [], {"trades": []}, benchmark_label="JANUARY",
        expected_metrics=holdout.EXPECTED_V3_JANUARY,
    )
    matrix._assert_compact_v3(
        [], {"trades": []}, benchmark_label="DECEMBER",
        expected_metrics=master.EXPECTED_GATES["V3_DECEMBER"],
    )
    assert calls == [
        ("COMPACT_V3_JANUARY", holdout.EXPECTED_V3_JANUARY),
        ("COMPACT_V3_DECEMBER", master.EXPECTED_GATES["V3_DECEMBER"]),
    ]
    assert holdout.EXPECTED_V3_JANUARY["completed_trades"] == 27
    assert master.EXPECTED_GATES["V3_DECEMBER"]["completed_trades"] == 38


def test_january_compact_gate_precedes_challenger_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    repository = tmp_path / "repo"
    master_root = repository / holdout.MASTER_RELATIVE
    reference_root = repository / holdout.V3_REFERENCE_RELATIVE
    output_root = repository / holdout.OUTPUT_RELATIVE
    master_root.mkdir(parents=True)
    reference_root.mkdir(parents=True)
    (master_root / "calendar.json").write_text(
        '{"target_sessions":["2026-01-02"]}', encoding="utf-8",
    )
    canonical = {
        "sessions": 1, "metrics": dict(holdout.EXPECTED_V3_JANUARY),
        "trades": [], "dbn_files_opened": 0, "network_calls": 0,
    }
    monkeypatch.setattr(master, "validate_building_root", lambda _root: {"status": "VALID"})
    monkeypatch.setattr(master, "replay_configuration", lambda *_args, **_kwargs: canonical)
    monkeypatch.setattr(holdout, "_load_published_v3_reference", lambda _root: {
        "metrics": dict(holdout.EXPECTED_V3_JANUARY), "summary_sha256": "a" * 64,
        "trade_ledger_sha256": "b" * 64, "january_trade_count": 27,
    })
    monkeypatch.setattr(holdout, "load_january_bundle", lambda *_args: _bundle([]))
    order: list[str] = []

    def portfolio(_root: Path, _bundle_value: holdout.JanuaryBundle, contract: holdout.FrozenPortfolioContract):
        order.append(contract.portfolio_id)
        if contract is holdout.CHALLENGER_CONTRACT:
            raise RuntimeError("CHALLENGER_REACHED")
        return {**holdout.EXPECTED_V3_JANUARY, "trades": []}

    def compact(_trades, _exact, *, benchmark_label, expected_metrics):
        order.append(f"COMPACT_{benchmark_label}")
        assert expected_metrics is holdout.EXPECTED_V3_JANUARY
        return dict(holdout.EXPECTED_V3_JANUARY)

    monkeypatch.setattr(holdout, "_run_portfolio", portfolio)
    monkeypatch.setattr(matrix, "_assert_compact_v3", compact)
    with pytest.raises(RuntimeError, match="CHALLENGER_REACHED"):
        holdout.run_holdout(
            repository_root=repository, master_root=master_root,
            v3_reference_root=reference_root, output_root=output_root,
        )
    assert order == ["V3_BASELINE", "COMPACT_JANUARY", "ROBUST_STRESS_CANDIDATE"]
    assert not output_root.exists()


def test_partial_staging_output_fails_closed_without_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    repository = tmp_path / "repo"
    master_root = repository / holdout.MASTER_RELATIVE
    reference_root = repository / holdout.V3_REFERENCE_RELATIVE
    output_root = repository / holdout.OUTPUT_RELATIVE
    master_root.mkdir(parents=True)
    reference_root.mkdir(parents=True)
    staging = output_root.with_name(output_root.name + ".building")
    staging.mkdir(parents=True)
    marker = staging / "partial-marker.txt"
    marker.write_text("preserve for explicit audit", encoding="utf-8")
    monkeypatch.setattr(master, "validate_building_root", lambda _root: pytest.fail("work started"))
    with pytest.raises(FileExistsError, match="staging output already exists"):
        holdout.run_holdout(
            repository_root=repository, master_root=master_root,
            v3_reference_root=reference_root, output_root=output_root,
        )
    assert marker.read_text(encoding="utf-8") == "preserve for explicit audit"


def test_v3_gate_failure_prevents_candidate_execution_and_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    repository = tmp_path / "repo"
    master_root = repository / holdout.MASTER_RELATIVE
    reference_root = repository / holdout.V3_REFERENCE_RELATIVE
    output_root = repository / holdout.OUTPUT_RELATIVE
    master_root.mkdir(parents=True)
    reference_root.mkdir(parents=True)
    (master_root / "calendar.json").write_text(
        '{"target_sessions":["2026-01-02"]}', encoding="utf-8",
    )
    monkeypatch.setattr(master, "validate_building_root", lambda _root: {"status": "VALID"})
    monkeypatch.setattr(master, "replay_configuration", lambda *_args, **_kwargs: {
        "sessions": 1,
        "metrics": {**holdout.EXPECTED_V3_JANUARY, "completed_trades": 26},
        "dbn_files_opened": 0,
        "network_calls": 0,
    })

    def forbidden(*_args, **_kwargs):
        raise AssertionError("challenger or published reference ran after failed V3 gate")

    monkeypatch.setattr(holdout, "_load_published_v3_reference", forbidden)
    monkeypatch.setattr(holdout, "_run_portfolio", forbidden)
    with pytest.raises(holdout.JanuaryHoldoutError, match="V3_JANUARY_REPRODUCTION_FAILED"):
        holdout.run_holdout(
            repository_root=repository,
            master_root=master_root,
            v3_reference_root=reference_root,
            output_root=output_root,
        )
    assert not output_root.exists()


def test_no_december_selection_reuse_network_or_automatic_reselection_path():
    source = Path(holdout.__file__).read_text(encoding="utf-8")
    assert "weight-q-results.csv" not in source
    assert "robust-stress-candidate.json" not in source
    assert "run_matrix(" not in source
    assert "configuration_registry(" not in source
    assert "import databento" not in source.lower()
    assert "timeseries.get_range" not in source
    assert "_stream_native_mbp10_records" not in source
    assert "_stream_mes_quotes" not in source
    assert holdout.FROZEN_PORTFOLIOS == (holdout.V3_CONTRACT, holdout.CHALLENGER_CONTRACT)


def test_descriptive_classification_never_mutates_or_reselects_candidate():
    before = holdout.CHALLENGER_CONTRACT
    v3 = {"total_r": 10.0, "profit_factor": 1.2, "max_cumulative_drawdown_r": -5.0}
    candidate = {"total_r": 8.0, "profit_factor": 1.1, "max_cumulative_drawdown_r": -6.0}
    assert holdout.descriptive_classification(v3, candidate) == "WEAKER_THAN_V3_BUT_STILL_POSITIVE"
    assert holdout.CHALLENGER_CONTRACT is before
    assert holdout.CHALLENGER_CONTRACT.config_id == "W02-02-07-02-07-Q40"
