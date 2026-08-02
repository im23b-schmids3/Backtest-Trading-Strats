from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import date, datetime, timezone
from pathlib import Path

from research_pipeline.value_area_trap.data import (
    AggregateTradeImporter,
    BinanceSymbolMetadataArtifact,
    _metadata_artifact_hash,
    parse_binance_symbol_metadata,
)
from research_pipeline.value_area_trap.equity_variants import (
    EQUITY_PRE_REGISTERED_VARIANTS,
    EQUITY_STUDY_LABEL,
    EQUITY_STUDY_SYMBOLS,
    EquityVariantStudyRequest,
    EquityVariantStudyService,
    validate_equity_variant_study,
)
from research_pipeline.value_area_trap.profile import US_CASH_SESSION_LABEL, is_us_cash_trading_day, session_bounds


def _millis(value: datetime) -> int:
    return int(value.timestamp() * 1000)


def _archive(cache: Path, symbol: str, month: int, first_id: int, price: str) -> None:
    target = cache / "downloads" / symbol / f"{symbol}-aggTrades-2026-{month:02d}.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    next_month = datetime(2026, month + 1, 1, tzinfo=timezone.utc)
    start = datetime(2026, month, 1, tzinfo=timezone.utc)
    content = (
        "agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker\n"
        f"{first_id},{price},1,{first_id},{first_id},{_millis(start)},false\n"
        f"{first_id + 1},{price},1,{first_id + 1},{first_id + 1},{_millis(next_month) - 1000},true\n"
    )
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr(target.with_suffix(".csv").name, content)


def _metadata(cache: Path) -> tuple[Path, BinanceSymbolMetadataArtifact]:
    raw = (Path(__file__).parent / "fixtures" / "binance_usdm_tradfi_exchange_info.json").read_bytes()
    payload = json.loads(raw)
    source_hash = hashlib.sha256(raw).hexdigest()
    unsigned = BinanceSymbolMetadataArtifact(
        retrieved_at=datetime(2026, 8, 2, tzinfo=timezone.utc),
        symbols=[parse_binance_symbol_metadata(item, source_hash=source_hash) for item in payload["symbols"] if item["symbol"] in EQUITY_STUDY_SYMBOLS],
        artifact_hash="pending",
    )
    artifact = unsigned.model_copy(update={"artifact_hash": _metadata_artifact_hash(unsigned)})
    path = cache / "metadata" / "equity-study-exchange-info.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
    return path, artifact


def _manifests(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    cache = tmp_path / "data" / "value_area_trap"
    metadata_path, artifact = _metadata(cache)
    importer = AggregateTradeImporter(cache)
    manifests: dict[str, str] = {}
    prices = {"QQQUSDT": "500", "SPYUSDT": "600"}
    for symbol in EQUITY_STUDY_SYMBOLS:
        for index, month in enumerate((5, 6, 7)):
            _archive(cache, symbol, month, index * 2 + 1, prices[symbol])
        metadata = next(item for item in artifact.symbols if item.symbol == symbol)
        manifest_path, _ = importer.ingest_monthly_range(
            symbol=symbol,
            start_month="2026-05",
            end_month="2026-07",
            symbol_metadata=metadata,
            metadata_artifact_path=metadata_path,
            metadata_artifact_hash=artifact.artifact_hash,
        )
        manifests[symbol] = str(manifest_path)
    return cache, manifests


def test_exact_six_variant_registry_and_no_additional_variants() -> None:
    assert [item.variant_id for item in EQUITY_PRE_REGISTERED_VARIANTS] == list("ABCDEF")
    assert len(EQUITY_PRE_REGISTERED_VARIANTS) == 6
    assert {(item.breakout_volume_multiplier, item.swing_right_bars) for item in EQUITY_PRE_REGISTERED_VARIANTS} == {
        ("1.50", 2), ("1.25", 2), ("1.00", 2), ("1.50", 1), ("1.25", 1),
    }


def test_equity_variant_study_reuses_verified_manifests_without_normalization(tmp_path: Path) -> None:
    _, manifests = _manifests(tmp_path)
    before = {symbol: Path(path).read_bytes() for symbol, path in manifests.items()}
    validated = validate_equity_variant_study(manifests)
    assert validated["study_label"] == EQUITY_STUDY_LABEL
    assert all(item["reused_immutable_partitions"] for item in validated["symbols"].values())
    assert before == {symbol: Path(path).read_bytes() for symbol, path in manifests.items()}


def test_us_cash_session_boundaries_are_dst_aware() -> None:
    spring_start, _ = session_bounds(date(2026, 3, 9), US_CASH_SESSION_LABEL)
    autumn_start, _ = session_bounds(date(2026, 11, 2), US_CASH_SESSION_LABEL)
    assert spring_start.astimezone(timezone.utc).hour == 13  # EDT
    assert autumn_start.astimezone(timezone.utc).hour == 14  # EST
    assert spring_start.minute == autumn_start.minute == 30
    assert is_us_cash_trading_day(date(2026, 5, 25)) is False  # Memorial Day
    assert is_us_cash_trading_day(date(2026, 5, 30)) is False  # Saturday


def test_equity_variant_runs_are_immutable_independent_and_non_promotional(tmp_path: Path) -> None:
    _, manifests = _manifests(tmp_path)
    protected_btc = tmp_path / "research_runs" / "ValueAreaTrap.UTC_24H_SESSION" / "completed-btc.json"
    protected_cross = tmp_path / "research_runs" / "ValueAreaTrap.UTC_24H_SESSION.cross_market" / "completed-cross.json"
    protected_btc.parent.mkdir(parents=True); protected_cross.parent.mkdir(parents=True)
    protected_btc.write_text("btc-unchanged", encoding="utf-8"); protected_cross.write_text("cross-unchanged", encoding="utf-8")
    result = EquityVariantStudyService().run(EquityVariantStudyRequest(manifests=manifests, artifact_root=str(tmp_path / "research_runs"), repository_root=str(tmp_path)))
    comparison = json.loads(Path(result.comparison_path).read_text(encoding="utf-8"))
    assert result.variant_count == 6 and result.result_count == 12
    assert len(comparison["results"]) == 12
    assert len({item["run_id"] for item in comparison["results"]}) == 12
    assert all(item["summary"]["zero_trade_reason"] for item in comparison["results"])
    assert all(item["summary"]["funnel_reconciliation"]["reconciles"] for item in comparison["results"])
    cash_results = [item for item in comparison["results"] if item["variant_id"] in {"E", "F"}]
    assert all(item["summary"]["us_cash_session_diagnostics"]["dst_aware"] for item in cash_results)
    assert all("2026-05-25" in item["summary"]["us_cash_session_diagnostics"]["excluded_exchange_holidays"] for item in cash_results)
    assert comparison["selection_prohibited"] is True
    assert comparison["optimization_claimed"] is False
    assert comparison["confirmation_evidence"] is False
    assert comparison["requires_future_holdout"] is True
    assert comparison["best_variant"] is comparison["recommendation"] is comparison["promotion"] is None
    for item in comparison["results"]:
        metrics = json.loads((Path(item["artifact_root"]) / "research" / "metrics.json").read_text(encoding="utf-8"))
        scenario_report = json.loads((Path(item["artifact_root"]) / "research" / "scenario_reports.json").read_text(encoding="utf-8"))
        assert metrics["study_label"] == EQUITY_STUDY_LABEL
        assert metrics["selection_prohibited"] is True and metrics["confirmation_evidence"] is False
        assert scenario_report["study_label"] == EQUITY_STUDY_LABEL
    assert protected_btc.read_text(encoding="utf-8") == "btc-unchanged"
    assert protected_cross.read_text(encoding="utf-8") == "cross-unchanged"
    rerun = EquityVariantStudyService().run(EquityVariantStudyRequest(manifests=manifests, artifact_root=str(tmp_path / "research_runs"), repository_root=str(tmp_path)))
    assert rerun.run_id == result.run_id and rerun.comparison_path == result.comparison_path
