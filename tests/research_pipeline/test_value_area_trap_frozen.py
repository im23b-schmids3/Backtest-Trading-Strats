from __future__ import annotations

import json
import zipfile
from datetime import date, datetime, timezone
from pathlib import Path

import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from research_pipeline.adapters.value_area_trap import ValueAreaTrapAdapter
from research_pipeline.value_area_trap.data import AggregateTradeImporter, parse_binance_aggregate_trade
from research_pipeline.value_area_trap.frozen import FROZEN_VARIANT, FrozenRunRequest, FrozenValueAreaTrapService, _frozen_spec, _hash, validate_frozen_specification
from research_pipeline.value_area_trap.strategy import ValueAreaTrapConfig


def _archive(cache_root: Path, month: str, rows: list[str]) -> None:
    target = cache_root / "downloads" / "BTCUSDT" / f"BTCUSDT-aggTrades-{month}.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    content = "agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker\n" + "\n".join(rows) + "\n"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr(target.with_suffix(".csv").name, content)


def _monthly_manifest(tmp_path: Path):
    cache = tmp_path / "data" / "value_area_trap"
    _archive(cache, "2025-01", ["1,100,1,1,1,1738367880000,false", "2,101,1,2,2,1738367940000,true"])
    _archive(cache, "2025-02", ["3,102,1,3,3,1738368000000,false", "4,103,1,4,4,1738368060000,true"])
    return AggregateTradeImporter(cache).ingest_monthly_range(symbol="BTCUSDT", start_month="2025-01", end_month="2025-02", allow_network=False)


def _api_trade(identifier: int, timestamp_ms: int, *, price: str = "100", maker: bool = False):
    return parse_binance_aggregate_trade(
        {"a": identifier, "p": price, "q": "1", "f": identifier, "l": identifier, "T": timestamp_ms, "m": maker},
        source_file="api-fixture",
        source_hash="api-fixture-hash",
        source="binance_usdm_gap_repair_api",
    )


def _api_page(rows):
    def fetch(_symbol: str, from_aggregate_trade_id: int, *, limit: int):
        return rows, {"url": "https://fixture.invalid/aggTrades", "from_aggregate_trade_id": from_aggregate_trade_id, "limit": limit, "response_hash": "fixture", "response_row_count": len(rows)}
    return fetch


def test_monthly_range_is_resumable_and_content_addressed(tmp_path: Path) -> None:
    path, manifest = _monthly_manifest(tmp_path)
    again, same = AggregateTradeImporter(tmp_path / "data" / "value_area_trap").ingest_monthly_range(symbol="BTCUSDT", start_month="2025-01", end_month="2025-02", allow_network=False)
    assert path == again
    assert manifest.normalized_dataset_hash == same.normalized_dataset_hash
    assert [item.month for item in manifest.partitions] == ["2025-01", "2025-02"]
    assert all((path.parent / item.file_name).is_file() for item in manifest.partitions)
    assert AggregateTradeImporter(tmp_path).validate_monthly_manifest(path).manifest_hash == manifest.manifest_hash


def test_monthly_range_reuse_reports_hash_verified_month_skips(tmp_path: Path) -> None:
    _monthly_manifest(tmp_path)
    importer = AggregateTradeImporter(tmp_path / "data" / "value_area_trap")
    importer.ingest_monthly_range(symbol="BTCUSDT", start_month="2025-01", end_month="2025-02", allow_network=False)
    assert [item["action"] for item in importer.last_ingestion_diagnostics] == ["SKIPPED_HASH_VERIFIED", "SKIPPED_HASH_VERIFIED"]


def test_continuous_ids_allow_and_record_long_no_trade_timestamp_interval(tmp_path: Path) -> None:
    cache = tmp_path / "data" / "value_area_trap"
    _archive(cache, "2025-01", ["1,100,1,1,1,1735689600000,false", "2,101,1,2,2,1735689960000,true"])
    _, manifest = AggregateTradeImporter(cache).ingest_monthly_range(symbol="BTCUSDT", start_month="2025-01", end_month="2025-01", allow_network=False)
    partition = manifest.partitions[0]
    assert partition.repair_status == "ARCHIVE_CONTINUOUS_ID_TIMESTAMP_GAP"
    assert partition.continuity_diagnostics[0]["ids_continuous"] is True
    assert partition.continuity_diagnostics[0]["missing_id_count"] == 0


def test_missing_ids_can_be_repaired_with_opt_in_api_and_audit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = tmp_path / "data" / "value_area_trap"
    _archive(cache, "2025-01", ["1,100,1,1,1,1735689600000,false", "3,103,1,3,3,1735689960000,true"])
    importer = AggregateTradeImporter(cache)
    monkeypatch.setattr(importer, "_fetch_api_aggregate_trades_page", _api_page([_api_trade(2, 1735689660000)]))
    manifest_path, manifest = importer.ingest_monthly_range(symbol="BTCUSDT", start_month="2025-01", end_month="2025-01", allow_network=True, allow_gap_repair=True)
    partition = manifest.partitions[0]
    assert partition.repair_status == "API_GAP_FILLED"
    audit = json.loads(Path(partition.repair_audit_path).read_text(encoding="utf-8"))
    assert audit["repairs"][0]["missing_id_count"] == 1
    assert audit["repairs"][0]["fetched_row_count"] == 1
    assert partition.row_count == 3
    assert importer.validate_monthly_manifest(manifest_path).manifest_hash == manifest.manifest_hash


def test_identical_api_archive_overlap_is_deduplicated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = tmp_path / "data" / "value_area_trap"
    _archive(cache, "2025-01", ["1,100,1,1,1,1735689600000,false", "3,103,1,3,3,1735689960000,true"])
    importer = AggregateTradeImporter(cache)
    monkeypatch.setattr(importer, "_fetch_api_aggregate_trades_page", _api_page([_api_trade(2, 1735689660000), _api_trade(3, 1735689960000, price="103", maker=True)]))
    _, manifest = importer.ingest_monthly_range(symbol="BTCUSDT", start_month="2025-01", end_month="2025-01", allow_network=True, allow_gap_repair=True)
    assert manifest.partitions[0].row_count == 3


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([_api_trade(2, 1735689660000), _api_trade(3, 1735689960000, price="999")], "conflicting API/archive duplicate"),
        ([_api_trade(3, 1735689960000, price="103", maker=True), _api_trade(2, 1735689660000)], "out-of-order or duplicate"),
        ([_api_trade(2, 1738368000000)], "outside 2025-01"),
        ([], "returned no rows"),
    ],
)
def test_gap_repair_rejects_conflict_order_out_of_month_and_unrecoverable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rows, message: str) -> None:
    cache = tmp_path / "data" / "value_area_trap"
    _archive(cache, "2025-01", ["1,100,1,1,1,1735689600000,false", "3,103,1,3,3,1735689960000,true"])
    importer = AggregateTradeImporter(cache)
    monkeypatch.setattr(importer, "_fetch_api_aggregate_trades_page", _api_page(rows))
    with pytest.raises(ValueError, match=message):
        importer.ingest_monthly_range(symbol="BTCUSDT", start_month="2025-01", end_month="2025-01", allow_network=True, allow_gap_repair=True)


def test_existing_archive_is_reused_and_later_months_continue_after_verified_month(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = tmp_path / "data" / "value_area_trap"
    _archive(cache, "2025-07", ["1,100,1,1,1,1751328000000,false"])
    _archive(cache, "2025-08", ["2,101,1,2,2,1754006400000,false"])
    _archive(cache, "2025-09", ["3,102,1,3,3,1756684800000,false"])
    AggregateTradeImporter(cache).ingest_monthly_range(symbol="BTCUSDT", start_month="2025-07", end_month="2025-07", allow_network=False)
    importer = AggregateTradeImporter(cache)
    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: pytest.fail("existing archive must be reused without download"))
    _, manifest = importer.ingest_monthly_range(symbol="BTCUSDT", start_month="2025-07", end_month="2025-09", allow_network=True)
    assert [item["action"] for item in importer.last_ingestion_diagnostics] == ["SKIPPED_HASH_VERIFIED", "NEWLY_PROCESSED", "NEWLY_PROCESSED"]
    assert [item.month for item in manifest.partitions] == ["2025-07", "2025-08", "2025-09"]


def test_monthly_partitions_reject_overlap(tmp_path: Path) -> None:
    cache = tmp_path / "data" / "value_area_trap"
    _archive(cache, "2025-01", ["2,100,1,1,1,1738367940000,false"])
    _archive(cache, "2025-02", ["1,101,1,2,2,1738368000000,false"])
    with pytest.raises(ValueError, match="aggregate trade ID overlap"):
        AggregateTradeImporter(cache).ingest_monthly_range(symbol="BTCUSDT", start_month="2025-01", end_month="2025-02", allow_network=False)


def test_frozen_specification_rejects_any_parameter_change(tmp_path: Path) -> None:
    manifest_path, manifest = _monthly_manifest(tmp_path)
    spec = _frozen_spec(manifest, manifest_path, tmp_path)
    validate_frozen_specification(spec, manifest)
    for key, value in {
        "swing_right_bars": 1,
        "breakout_volume_multiplier": "1.25",
        "entry_execution": "same_bar",
        "price_tick": "0.20",
        "optimization_allowed": True,
    }.items():
        changed = spec.model_copy(update={"baseline_parameters": {**spec.baseline_parameters, key: value}})
        with pytest.raises(Exception, match=key):
            validate_frozen_specification(changed, manifest)


def test_monthly_range_rejects_missing_ids_without_opt_in_repair(tmp_path: Path) -> None:
    cache = tmp_path / "data" / "value_area_trap"
    _archive(cache, "2025-01", ["1,100,1,1,1,1735689600000,false", "3,101,1,3,3,1735689901000,true"])
    with pytest.raises(ValueError, match="DATA_GAP_UNREPAIRABLE"):
        AggregateTradeImporter(cache).ingest_monthly_range(symbol="BTCUSDT", start_month="2025-01", end_month="2025-01", allow_network=False)


def test_packaged_frozen_adapter_requires_no_external_executor(tmp_path: Path) -> None:
    manifest_path, manifest = _monthly_manifest(tmp_path)
    spec = _frozen_spec(manifest, manifest_path, tmp_path)
    adapter = ValueAreaTrapAdapter(spec, tmp_path, manifest_path=manifest_path)
    assert adapter.health(spec).healthy
    assert adapter.identity.adapter_version == "value-area-trap-3"
    assert FROZEN_VARIANT == "UTC_24H_SESSION"


def test_partitioned_feature_stream_matches_single_fixture_and_crosses_month_profile(tmp_path: Path) -> None:
    manifest_path, manifest = _monthly_manifest(tmp_path)
    spec = _frozen_spec(manifest, manifest_path, tmp_path)
    adapter = ValueAreaTrapAdapter(spec, tmp_path, manifest_path=manifest_path)
    paths, _, _, _ = adapter._validated_dataset()
    tables = [pq.read_table(path) for path in paths]
    single = tmp_path / "single.parquet"
    pq.write_table(pa.concat_tables(tables), single)
    config = ValueAreaTrapConfig(session_definition="UTC_24H_SESSION", session_timezone="UTC")
    partitioned_bars, partitioned_profiles, _ = adapter._stream_features(paths, manifest.normalized_dataset_hash, config, config.price_tick * 100)
    single_bars, single_profiles, _ = adapter._stream_features(single, manifest.normalized_dataset_hash, config, config.price_tick * 100)
    assert [item.model_dump(mode="json") for item in partitioned_bars] == [item.model_dump(mode="json") for item in single_bars]
    assert {key: value.model_dump(mode="json") for key, value in partitioned_profiles.items()} == {key: value.model_dump(mode="json") for key, value in single_profiles.items()}
    assert sorted(partitioned_profiles)[0].isoformat() == "2025-01-31"
    assert sorted(partitioned_profiles)[1].isoformat() == "2025-02-01"


def test_frozen_run_has_no_external_executor_or_manual_resume_boundary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest_path, manifest = _monthly_manifest(tmp_path)
    # Keep the Parquet fixture tiny while exercising the fixed production
    # period gate; adapter reads the same immutable partition set.
    monkeypatch.setattr(AggregateTradeImporter, "validate_monthly_manifest", lambda _self, _path: manifest.model_copy(update={"date_start": date(2025, 1, 1), "date_end": date(2026, 4, 30)}))
    result = FrozenValueAreaTrapService().run(FrozenRunRequest(
        variant=FROZEN_VARIANT,
        data_manifest=str(manifest_path),
        artifact_root=str(tmp_path / "research_runs"),
        registry_path=str(tmp_path / "registry.sqlite3"),
        repository_root=str(tmp_path),
        auto_approve=True,
        reuse_verified_implementation=True,
    ))
    report = json.loads(Path(result.report_path).read_text(encoding="utf-8"))
    assert result.external_executor_required is False
    assert report["implementation"] == "VERIFIED_PACKAGED_IMPLEMENTATION"
    assert report["pipeline_state"] == "INSUFFICIENT_EVIDENCE"
    assert result.run_id == f"frozen-{result.strategy_id}-{_hash({'specification_hash': result.specification_hash, 'dataset_hash': result.dataset_hash})[:16]}"
    assert set(json.loads(Path(result.comparison_path).read_text(encoding="utf-8"))) >= {
        "primary_holdout", "previously_observed_selection_month", "full_period_summary",
    }


def test_frozen_run_rejects_manifest_without_required_period(tmp_path: Path) -> None:
    manifest_path, _ = _monthly_manifest(tmp_path)
    with pytest.raises(Exception, match="requires a pinned manifest covering"):
        FrozenValueAreaTrapService().run(FrozenRunRequest(variant=FROZEN_VARIANT, data_manifest=str(manifest_path), artifact_root=str(tmp_path / "research_runs"), registry_path=str(tmp_path / "registry.sqlite3"), repository_root=str(tmp_path), auto_approve=True, reuse_verified_implementation=True))
