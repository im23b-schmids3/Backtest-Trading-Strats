from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
import yaml

from research_pipeline.schemas.strategy_spec import load_strategy_spec
from research_pipeline.value_area_trap.data import AggregateTradeManifest
from research_pipeline.value_area_trap.profile import FiveMinuteBar, SessionProfile
from research_pipeline.value_area_trap.strategy import ValueAreaTrapConfig, _previous_profile, run_value_area_trap
from research_pipeline.value_area_trap.variants import (
    COMPARISON_METRICS,
    PREDECLARED_VARIANTS,
    REAL_DATASET_HASH,
    build_variant_specification,
    materialize_variants,
)


def _manifest(path: Path) -> Path:
    payload = {
        "provider": "Binance USD-M Futures public data archive",
        "product": "USD-M perpetual aggregate trades",
        "symbol": "BTCUSDT",
        "date_start": "2026-04-01",
        "date_end": "2026-04-30",
        "retrieved_at": "2026-08-01T00:00:00Z",
        "source_files": ["fixture.zip"],
        "source_file_hashes": {"fixture.zip": "fixture"},
        "normalized_dataset_hash": REAL_DATASET_HASH,
        "row_count": 1,
        "duplicate_count": 0,
        "manifest_hash": "pending",
    }
    candidate = AggregateTradeManifest.model_validate(payload)
    unsigned = candidate.model_dump(mode="json")
    unsigned.pop("manifest_hash")
    payload["manifest_hash"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _bar(index: int, *, high: str = "119", low: str = "114", close: str = "116", volume: str = "1", cvd: str = "0") -> FiveMinuteBar:
    start = datetime(2025, 1, 3, 0, 0, tzinfo=timezone.utc) + timedelta(minutes=5 * index)
    return FiveMinuteBar(
        start_utc=start,
        end_utc=start + timedelta(minutes=5),
        start_new_york=start.astimezone(ZoneInfo("America/New_York")),
        end_new_york=(start + timedelta(minutes=5)).astimezone(ZoneInfo("America/New_York")),
        open=Decimal(close), high=Decimal(high), low=Decimal(low), close=Decimal(close),
        total_volume=Decimal(volume), aggressive_buy_volume=Decimal(volume),
        aggressive_sell_volume=Decimal(), bar_delta=Decimal(volume),
        cumulative_volume_delta=Decimal(cvd), trade_count=1, vwap=Decimal(close),
        session_date=date(2025, 1, 3), session_label="UTC_24H_SESSION",
    )


def _profile(day: date = date(2025, 1, 2)) -> SessionProfile:
    return SessionProfile(
        session_date=day, poc=Decimal("112"), vah=Decimal("120"), val=Decimal("110"),
        total_session_volume=Decimal("10"), value_area_volume=Decimal("7"),
        coverage_ratio=Decimal("0.7"), bucket_size=Decimal("10"), profile_hash="fixture",
        source_dataset_hash="fixture", session_label="UTC_24H_SESSION",
    )


def _bars() -> list[FiveMinuteBar]:
    bars = [_bar(i, cvd=str(i)) for i in range(18)]
    bars[10] = _bar(10, high="130", low="115", close="118", volume="20", cvd="10")
    bars[11] = _bar(11, high="128", cvd="9")
    bars[12] = _bar(12, high="127", cvd="8")
    bars[13] = _bar(13, high="132", low="116", close="119", cvd="5")
    bars[14] = _bar(14, high="130", cvd="4")
    bars[15] = _bar(15, high="129", cvd="3")
    bars[16] = _bar(16, high="119", low="114", close="119", cvd="2")
    bars[17] = _bar(17, high="119", low="111", close="116", cvd="1")
    return bars


def test_predeclared_variants_are_distinct_nonoptimizing_specifications() -> None:
    specifications = [build_variant_specification(variant) for variant in PREDECLARED_VARIANTS]
    assert [item.strategy_id for item in specifications] == [
        "ValueAreaTrap.UTC_24H_SESSION",
        "ValueAreaTrap.UTC_24H_FAST_SWING",
        "ValueAreaTrap.UTC_24H_FAST_SWING_VOLUME_125",
    ]
    assert len({item.specification_hash for item in specifications}) == 3
    assert all(all(not family.mutable and family.maximum_rounds == 0 for family in spec.parameter_families) for spec in specifications)
    assert specifications[0].baseline_parameters["swing_right_bars"] == 2
    assert specifications[1].baseline_parameters["swing_right_bars"] == 1
    assert specifications[2].baseline_parameters["breakout_volume_multiplier"] == "1.25"


def test_materialization_is_immutable_idempotent_and_has_comparison_template(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "manifest.json")
    first = materialize_variants(repository_root=tmp_path, data_manifest_path=manifest, artifact_root="variants")
    second = materialize_variants(repository_root=tmp_path, data_manifest_path=manifest, artifact_root="variants")
    assert first == second
    comparison = json.loads(Path(first["comparison_path"]).read_text(encoding="utf-8"))
    assert comparison["status"] == "NOT_EXECUTED"
    assert comparison["metrics"] == COMPARISON_METRICS
    assert len(comparison["variants"]) == 3
    for item in first["variants"]:
        spec = load_strategy_spec(item["specification_path"])
        assert spec.specification_hash == item["specification_hash"]
        parameters = json.loads(Path(item["parameter_manifest_path"]).read_text(encoding="utf-8"))
        assert parameters["dataset_hash"] == REAL_DATASET_HASH
        assert "not CME or Alpha Futures" in parameters["evidence_label"]


def test_materialization_rejects_a_changed_real_data_manifest(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "manifest.json")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["normalized_dataset_hash"] = "not-the-immutable-dataset"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        materialize_variants(repository_root=tmp_path, data_manifest_path=manifest, artifact_root="variants")


def test_previous_completed_utc_profile_only() -> None:
    older = _profile(date(2025, 1, 1))
    previous = _profile(date(2025, 1, 2))
    current = _profile(date(2025, 1, 3))
    assert _previous_profile(date(2025, 1, 3), {older.session_date: older, previous.session_date: previous, current.session_date: current}) == previous


def test_fast_swing_confirmation_is_not_backdated_and_entry_is_next_bar() -> None:
    config = ValueAreaTrapConfig(session_definition="UTC_24H_SESSION", session_timezone="UTC", swing_right_bars=1)
    result = run_value_area_trap(_bars(), {date(2025, 1, 2): _profile()}, config)
    event = next(item for item in result.setup_events if item["state"] == "DIVERGENCE_CONFIRMED")
    assert event["right_confirmation_bars"] == 1
    assert event["confirmation_timestamp"] > event["second"]["timestamp"]
    assert event["entry_not_before"] is not None
    assert result.trades[0]["entry_timestamp"] >= event["entry_not_before"]


def test_volume_median_excludes_current_bar_and_replay_is_deterministic() -> None:
    config = ValueAreaTrapConfig(session_definition="UTC_24H_SESSION", session_timezone="UTC")
    first = run_value_area_trap(_bars(), {date(2025, 1, 2): _profile()}, config)
    second = run_value_area_trap(_bars(), {date(2025, 1, 2): _profile()}, config)
    stop_run = next(item for item in first.setup_events if item["state"] == "STOP_RUN_CONFIRMED")
    assert stop_run["median_excludes_current"] == "1"
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
