from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from research_pipeline.cme_orderflow_absorption_l2_v1 import causal_master_tape as master
from research_pipeline.cme_orderflow_absorption_l2_v1 import causal_level_resolver as levels
from research_pipeline.cme_orderflow_absorption_l2_v1 import model
from research_pipeline.cme_orderflow_absorption_l2_v1 import multi_strategy_research as multi


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _strategy(identifier: str, target: str, source: str, reference: str, semantics: str):
    return SimpleNamespace(
        strategy_id=identifier, session=target, source_session=source,
        reference_level=reference, reference_semantics=semantics,
    )


def _resolver(tmp_path: Path) -> levels.CausalLevelResolver:
    relationships = [
        {"target_session": "EUROPE", "trading_date": "2026-09-08", "source_session": "NEW_YORK_RTH", "source_date": "2026-09-05", "reference_semantics": levels.PRIOR_PROFILE},
        {"target_session": "EUROPE", "trading_date": "2026-09-08", "source_session": "EUROPE", "source_date": "2026-09-05", "reference_semantics": levels.PRIOR_PROFILE},
        {"target_session": "NEW_YORK_RTH", "trading_date": "2026-09-08", "source_session": "EUROPE", "source_date": "2026-09-05", "reference_semantics": levels.PRIOR_PROFILE},
        {"target_session": "NEW_YORK_RTH", "trading_date": "2026-09-08", "source_session": "ASIA", "source_date": "2026-09-08", "reference_semantics": levels.PRIOR_PROFILE},
        {"target_session": "NEW_YORK_RTH", "trading_date": "2026-09-08", "source_session": "EUROPE", "source_date": "2026-09-08", "reference_semantics": levels.CURRENT_EXTREMUM},
        {"target_session": "NEW_YORK_RTH", "trading_date": "2026-09-08", "source_session": "EUROPE", "source_date": "2026-09-08", "reference_semantics": levels.COMPLETED_CURRENT_PROFILE},
        {"target_session": "ASIA", "trading_date": "2026-09-08", "source_session": "NEW_YORK_RTH", "source_date": "2026-09-05", "reference_semantics": levels.PRIOR_PROFILE},
        # Non-contiguous explicit relation: do not calculate a previous date.
        {"target_session": "ASIA", "trading_date": "2026-09-09", "source_session": "EUROPE", "source_date": "2026-09-05", "reference_semantics": levels.PRIOR_PROFILE},
    ]
    observations = [
        {"source_session": "NEW_YORK_RTH", "source_date": "2026-09-05", "level_family": "POC", "level_value": 100.0, "available_at_ns": 10, "mode": levels.PROFILE_MODE, "source_artifact": "ny-profile.json", "source_artifact_sha256": "ny-hash"},
        {"source_session": "EUROPE", "source_date": "2026-09-05", "level_family": "VAH", "level_value": 200.0, "available_at_ns": 10, "mode": levels.PROFILE_MODE, "source_artifact": "europe-profile.json"},
        {"source_session": "EUROPE", "source_date": "2026-09-05", "level_family": "HIGH", "level_value": 195.0, "available_at_ns": 10, "mode": levels.PROFILE_MODE, "source_artifact": "europe-profile.json"},
        {"source_session": "ASIA", "source_date": "2026-09-08", "level_family": "LOW", "level_value": 300.0, "available_at_ns": 10, "mode": levels.PROFILE_MODE, "source_artifact": "asia-profile.json"},
        {"source_session": "EUROPE", "source_date": "2026-09-08", "level_family": "HIGH", "level_value": 210.0, "available_at_ns": 100, "mode": levels.DYNAMIC_MODE, "source_artifact": "europe-events.parquet"},
        {"source_session": "EUROPE", "source_date": "2026-09-08", "level_family": "HIGH", "level_value": 220.0, "available_at_ns": 200, "mode": levels.DYNAMIC_MODE, "source_artifact": "europe-events.parquet"},
        {"source_session": "EUROPE", "source_date": "2026-09-08", "level_family": "POC", "level_value": 205.0, "available_at_ns": 250, "mode": levels.PROFILE_MODE, "source_artifact": "europe-profile-current.json"},
    ]
    path = tmp_path / "level-catalog.json"
    path.write_text(json.dumps({"schema_version": 1, "session_relationships": relationships, "level_observations": observations}), encoding="utf-8")
    return levels.CausalLevelResolver.from_path(path)


def test_prior_profiles_use_explicit_session_relationships(tmp_path: Path):
    resolver = _resolver(tmp_path)
    europe_prior_ny = _strategy("EU_PRIOR_NY_POC", "EUROPE", "NEW_YORK_RTH", "PRIOR_NY_RTH_POC", levels.PRIOR_PROFILE)
    ny_prior_europe = _strategy("NY_PRIOR_EU_VAH", "NEW_YORK_RTH", "EUROPE", "PRIOR_EUROPE_SESSION_VAH", levels.PRIOR_PROFILE)
    ny_prior_asia = _strategy("NY_PRIOR_ASIA_LOW", "NEW_YORK_RTH", "ASIA", "PRIOR_ASIA_SESSION_LOW", levels.PRIOR_PROFILE)
    asia_prior_ny = _strategy("ASIA_PRIOR_NY_POC", "ASIA", "NEW_YORK_RTH", "PRIOR_NY_RTH_POC", levels.PRIOR_PROFILE)
    non_contiguous = _strategy("ASIA_PRIOR_EU_HIGH", "ASIA", "EUROPE", "PRIOR_EUROPE_SESSION_HIGH", levels.PRIOR_PROFILE)
    assert resolver.resolve(europe_prior_ny, trading_date="2026-09-08", signal_timestamp_ns=20).level_value == 100.0
    assert resolver.resolve(ny_prior_europe, trading_date="2026-09-08", signal_timestamp_ns=20).level_value == 200.0
    assert resolver.resolve(ny_prior_asia, trading_date="2026-09-08", signal_timestamp_ns=20).level_value == 300.0
    assert resolver.resolve(asia_prior_ny, trading_date="2026-09-08", signal_timestamp_ns=20).source_date == "2026-09-05"
    assert resolver.resolve(non_contiguous, trading_date="2026-09-09", signal_timestamp_ns=20).source_date == "2026-09-05"


def test_current_extrema_are_timestamped_and_cannot_leak_future_high(tmp_path: Path):
    resolver = _resolver(tmp_path)
    current_high = _strategy("NY_CURRENT_EU_HIGH", "NEW_YORK_RTH", "EUROPE", "CURRENT_EUROPE_HIGH_SWEEP", levels.CURRENT_EXTREMUM)
    early = resolver.resolve(current_high, trading_date="2026-09-08", signal_timestamp_ns=150)
    later = resolver.resolve(current_high, trading_date="2026-09-08", signal_timestamp_ns=250)
    assert early.available and early.level_value == 210.0
    assert later.available and later.level_value == 220.0
    assert early.causal_availability_timestamp_ns == 100
    assert early.structural_level("CURRENT_EUROPE_HIGH_SWEEP").price == 210.0


def test_completed_current_profile_is_unavailable_before_profile_completion(tmp_path: Path):
    resolver = _resolver(tmp_path)
    completed_poc = _strategy("NY_COMPLETED_EU_POC", "NEW_YORK_RTH", "EUROPE", "COMPLETED_EUROPE_SESSION_POC", levels.COMPLETED_CURRENT_PROFILE)
    unavailable = resolver.resolve(completed_poc, trading_date="2026-09-08", signal_timestamp_ns=249)
    available = resolver.resolve(completed_poc, trading_date="2026-09-08", signal_timestamp_ns=250)
    assert not unavailable.available and unavailable.reason == "LEVEL_UNAVAILABLE_AT_SIGNAL_TIMESTAMP"
    assert available.available and available.level_value == 205.0


def test_missing_source_level_rejects_safely(tmp_path: Path):
    resolver = _resolver(tmp_path)
    missing = _strategy("NY_PRIOR_ASIA_VAL", "NEW_YORK_RTH", "ASIA", "PRIOR_ASIA_SESSION_VAL", levels.PRIOR_PROFILE)
    result = resolver.resolve(missing, trading_date="2026-09-08", signal_timestamp_ns=20)
    assert not result.available
    assert result.reason == "LEVEL_UNAVAILABLE_AT_SIGNAL_TIMESTAMP"


def test_multi_strategy_population_binds_resolved_provenance(tmp_path: Path):
    resolver = _resolver(tmp_path)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    interaction = {
        "interaction_id": "2026-09-08|EU-NY-POC", "session_date": "2026-09-08",
        "target_session": "EUROPE",
        "interaction_start_ns": 20, "interaction_end_ns": 25, "level": "PRIOR_NY_RTH_POC",
        "level_price": 100.0,
    }
    master._write_small_parquet(artifacts / "interaction-master.parquet", [interaction])
    master._write_small_parquet(artifacts / "interaction-index.parquet", [{"interaction_id": interaction["interaction_id"]}])
    period = multi.PeriodSpec("synthetic", artifacts / "interaction-master.parquet", artifacts / "interaction-index.parquet",
                              (("2026-09-08", artifacts / "unused-event-tape.parquet"),), resolver.catalog_path)
    strategy = multi.StrategySpec(
        "EU_PRIOR_NY_POC", "EUROPE", "NEW_YORK_RTH", "PRIOR_NY_RTH_POC", levels.PRIOR_PROFILE,
        "synthetic", "shared", True, "SUPPORTED_SHARED_CAUSAL_RESOLVER", True, True, True, {}, {}, {},
    )
    rows, indexes = multi._load_population(period, strategy)
    assert indexes[interaction["interaction_id"]]["interaction_id"] == interaction["interaction_id"]
    resolved = rows["2026-09-08"][0]["level_resolution"]
    assert resolved["level_value"] == 100.0
    assert resolved["source_artifact_sha256"] == "ny-hash"


def test_all_enabled_candidates_have_a_supported_resolver_mode():
    specs = multi.load_strategy_manifest(REPOSITORY_ROOT / "examples/research_pipeline/cme_l2_candidate_universe.example.yaml")
    audit = levels.candidate_executability_audit(specs)
    assert len(audit) == 51
    assert {row["classification"] for row in audit} == {"EXECUTABLE_WITH_SUPPORTED_INPUTS"}
    assert all(model.StructuralLevel(spec.reference_level, 100.0).name == spec.reference_level for spec in specs)
