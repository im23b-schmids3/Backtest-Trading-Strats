from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from research_pipeline.value_area_trap.cross_market import (
    CROSS_MARKET_SYMBOLS,
    CrossMarketIngestRequest,
    FrozenCrossMarketRequest,
    FrozenCrossMarketService,
    _cross_spec,
    ingest_cross_market,
    validate_cross_market,
    validate_cross_specification,
)
from research_pipeline.value_area_trap.data import (
    AggregateTradeImporter,
    BinanceSymbolMetadataArtifact,
    _metadata_artifact_hash,
    cross_market_symbol_diagnostic,
    parse_binance_symbol_metadata,
    validate_cross_market_symbol_eligibility,
)


def _milliseconds(value: datetime) -> int:
    return int(value.timestamp() * 1000)


def _archive(cache: Path, symbol: str, month: str, rows: list[str]) -> None:
    target = cache / "downloads" / symbol / f"{symbol}-aggTrades-{month}.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    content = "agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker\n" + "\n".join(rows) + "\n"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr(target.with_suffix(".csv").name, content)


def _metadata(cache: Path) -> Path:
    raw = (Path(__file__).parent / "fixtures" / "binance_usdm_tradfi_exchange_info.json").read_bytes()
    payload = json.loads(raw)
    source_hash = hashlib.sha256(raw).hexdigest()
    unsigned = BinanceSymbolMetadataArtifact(
        retrieved_at=datetime(2026, 8, 2, tzinfo=timezone.utc),
        symbols=[parse_binance_symbol_metadata(item, source_hash=source_hash) for item in payload["symbols"]],
        artifact_hash="pending",
    )
    artifact = unsigned.model_copy(update={"artifact_hash": _metadata_artifact_hash(unsigned)})
    path = cache / "metadata" / "fixture-exchange-info.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
    return path


def _create_archives(cache: Path, *, incomplete_symbol: str | None = None) -> None:
    prices = {"XAUUSDT": "3000", "QQQUSDT": "500", "SPYUSDT": "600"}
    for symbol in CROSS_MARKET_SYMBOLS:
        for index, month in enumerate((5, 6, 7)):
            start = datetime(2026, month, 1, tzinfo=timezone.utc)
            next_start = datetime(2026, month + 1, 1, tzinfo=timezone.utc)
            if symbol == incomplete_symbol and month == 5:
                start = datetime(2026, 5, 2, tzinfo=timezone.utc)
            first, last = 2 * index + 1, 2 * index + 2
            _archive(cache, symbol, f"2026-{month:02d}", [
                f"{first},{prices[symbol]},1,{first},{first},{_milliseconds(start)},false",
                f"{last},{prices[symbol]},1,{last},{last},{_milliseconds(next_start) - 1000},true",
            ])


def _ingested(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    cache = tmp_path / "data" / "value_area_trap"
    _create_archives(cache)
    metadata = _metadata(cache)
    result = ingest_cross_market(CrossMarketIngestRequest(cache_root=str(cache), metadata_artifact=str(metadata)))
    return cache, {symbol: result["symbols"][symbol]["manifest_path"] for symbol in CROSS_MARKET_SYMBOLS}


def test_non_btc_cross_market_ingestion_uses_pinned_metadata_and_independent_manifests(tmp_path: Path) -> None:
    _, manifests = _ingested(tmp_path)
    validated = validate_cross_market(manifests)
    assert set(validated["symbols"]) == set(CROSS_MARKET_SYMBOLS)
    assert len({item["dataset_hash"] for item in validated["symbols"].values()}) == 3
    assert all(item["metadata"]["contract_type"] == "TRADIFI_PERPETUAL" for item in validated["symbols"].values())
    assert all(item["metadata"]["raw_symbol_hash"] for item in validated["symbols"].values())


@pytest.mark.parametrize(
    ("symbol", "underlying_type"),
    [("XAUUSDT", "COMMODITY"), ("QQQUSDT", "EQUITY"), ("SPYUSDT", "EQUITY")],
)
def test_actual_binance_usdm_tradfi_metadata_shape_is_pinned(tmp_path: Path, symbol: str, underlying_type: str) -> None:
    metadata_path = _metadata(tmp_path)
    artifact = BinanceSymbolMetadataArtifact.model_validate_json(metadata_path.read_text(encoding="utf-8"))
    metadata = next(item for item in artifact.symbols if item.symbol == symbol)
    assert metadata.contract_type == "TRADIFI_PERPETUAL"
    assert metadata.status == "TRADING"
    assert metadata.pair == symbol
    assert metadata.margin_asset == metadata.quote_asset == "USDT"
    assert metadata.underlying_type == underlying_type
    assert metadata.underlying_sub_type == ["TradFi"]
    assert metadata.delivery_date_epoch_ms == 4133404800000
    assert metadata.raw_symbol_metadata["symbol"] == symbol
    assert metadata.raw_symbol_hash


def test_explicit_tradfi_perpetual_eligibility_accepts_actual_binance_shape(tmp_path: Path) -> None:
    artifact = BinanceSymbolMetadataArtifact.model_validate_json(_metadata(tmp_path).read_text(encoding="utf-8"))
    for metadata in artifact.symbols:
        validate_cross_market_symbol_eligibility(metadata)


@pytest.mark.parametrize(
    ("changes", "expected_reason"),
    [
        ({"contract_type": "CURRENT_QUARTER"}, "contractType must be TRADIFI_PERPETUAL"),
        ({"contract_type": "SPOT"}, "contractType must be TRADIFI_PERPETUAL"),
        ({"symbol": "BTCUSDT", "pair": "BTCUSDT"}, "explicit frozen cross-market allowlist"),
    ],
)
def test_cross_market_eligibility_rejects_delivery_spot_and_non_allowlisted_symbols(
    tmp_path: Path, changes: dict[str, str], expected_reason: str
) -> None:
    artifact = BinanceSymbolMetadataArtifact.model_validate_json(_metadata(tmp_path).read_text(encoding="utf-8"))
    metadata = artifact.symbols[0].model_copy(update=changes)
    with pytest.raises(ValueError, match=expected_reason):
        validate_cross_market_symbol_eligibility(metadata)


def test_cross_market_eligibility_error_includes_per_symbol_diagnostic(tmp_path: Path) -> None:
    artifact = BinanceSymbolMetadataArtifact.model_validate_json(_metadata(tmp_path).read_text(encoding="utf-8"))
    metadata = artifact.symbols[0].model_copy(update={"status": "BREAK"})
    with pytest.raises(ValueError) as excinfo:
        validate_cross_market_symbol_eligibility(metadata)
    message = str(excinfo.value)
    assert "XAUUSDT" in message and "Binance status is not TRADING" in message
    assert cross_market_symbol_diagnostic(metadata)["contractType"] == "TRADIFI_PERPETUAL"


def test_symbol_specific_filters_are_pinned_not_btc_defaults(tmp_path: Path) -> None:
    _, manifests = _ingested(tmp_path)
    importer = AggregateTradeImporter(".")
    xau_manifest = importer.validate_monthly_manifest(manifests["XAUUSDT"])
    spy_manifest = importer.validate_monthly_manifest(manifests["SPYUSDT"])
    xau = _cross_spec(symbol="XAUUSDT", manifest=xau_manifest, manifest_path=Path(manifests["XAUUSDT"]), root=tmp_path)
    spy = _cross_spec(symbol="SPYUSDT", manifest=spy_manifest, manifest_path=Path(manifests["SPYUSDT"]), root=tmp_path)
    validate_cross_specification(xau, xau_manifest)
    assert xau.baseline_parameters["price_tick"] == "0.01"
    assert xau.baseline_parameters["quantity"] == "0.001"
    assert spy.baseline_parameters["price_tick"] == "0.01000"
    assert spy.baseline_parameters["quantity"] == "0.01"


def test_incomplete_first_listing_month_is_rejected(tmp_path: Path) -> None:
    cache = tmp_path / "data" / "value_area_trap"
    _create_archives(cache, incomplete_symbol="XAUUSDT")
    with pytest.raises(ValueError, match="INCOMPLETE_CALENDAR_MONTH"):
        ingest_cross_market(CrossMarketIngestRequest(cache_root=str(cache), metadata_artifact=str(_metadata(cache))))


def test_cross_market_contract_rejects_april_or_other_periods(tmp_path: Path) -> None:
    with pytest.raises(Exception, match="2026-05 through 2026-07"):
        ingest_cross_market(CrossMarketIngestRequest(cache_root=str(tmp_path), start_month="2026-04", end_month="2026-07"))


def test_cross_market_resume_skips_verified_partitions(tmp_path: Path) -> None:
    cache, _ = _ingested(tmp_path)
    result = ingest_cross_market(CrossMarketIngestRequest(cache_root=str(cache), metadata_artifact=str(cache / "metadata" / "fixture-exchange-info.json")))
    assert all(
        [item["action"] for item in result["symbols"][symbol]["months"]] == ["SKIPPED_HASH_VERIFIED"] * 3
        for symbol in CROSS_MARKET_SYMBOLS
    )


def test_frozen_cross_market_is_independent_descriptive_and_never_selects_best_symbol(tmp_path: Path) -> None:
    _, manifests = _ingested(tmp_path)
    result = FrozenCrossMarketService().run(FrozenCrossMarketRequest(manifests=manifests, artifact_root=str(tmp_path / "research_runs"), repository_root=str(tmp_path)))
    comparison = json.loads(Path(result.comparison_path).read_text(encoding="utf-8"))
    assert result.external_executor_required is False
    assert len(set(result.symbol_runs.values())) == 3
    assert comparison["selection_prohibited"] is True
    assert comparison["best_symbol"] is None and comparison["ranking"] is None and comparison["promotion"] is None
    assert set(comparison["symbols"]) == set(CROSS_MARKET_SYMBOLS)
    fixed = comparison["frozen_strategy_parameters"]
    assert fixed["swing_right_bars"] == 2 and fixed["optimization_allowed"] is False
    assert set(comparison["symbols"]["XAUUSDT"]["monthly"]) == {"2026-05", "2026-06", "2026-07"}
