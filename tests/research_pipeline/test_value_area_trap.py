from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo
import zipfile
import json

import pytest

from research_pipeline.adapters.registry import default_adapter_registry
from research_pipeline.adapters.value_area_trap import ValueAreaTrapAdapter
from research_pipeline.cli import main as cli_main
from research_pipeline.compliance import CalendarArtifact, EconomicEvent
from research_pipeline.phase_b.models import WorkflowInput
from research_pipeline.phase_b.services import PhaseBService
from research_pipeline.value_area_trap.alpha_zero import run_alpha_zero_scenario
from research_pipeline.value_area_trap.data import AggregateTrade, AggregateTradeImporter, parse_binance_aggregate_trade
from research_pipeline.value_area_trap.profile import FiveMinuteBar, SessionProfile, build_five_minute_bars, build_session_profile
from research_pipeline.value_area_trap.strategy import ValueAreaTrapConfig, run_value_area_trap


def trade(identifier: int, when: datetime, price: str = "100", quantity: str = "1", maker: bool = False) -> AggregateTrade:
    value = Decimal(price); size = Decimal(quantity)
    return AggregateTrade(event_time_utc=when, trade_time_utc=when, aggregate_trade_id=identifier, price=value, quantity_base=size, notional_quote=value * size, buyer_is_maker=maker, aggressor_side="SELL" if maker else "BUY", signed_quantity=-size if maker else size, source_file="fixture.csv", source_hash="fixture")


def bar(index: int, *, high: str = "119", low: str = "114", close: str = "116", volume: str = "1", cvd: str = "0") -> FiveMinuteBar:
    start = datetime(2025, 1, 3, 14, 30, tzinfo=timezone.utc) + timedelta(minutes=5 * index)
    return FiveMinuteBar(start_utc=start, end_utc=start + timedelta(minutes=5), start_new_york=start.astimezone(ZoneInfo("America/New_York")), end_new_york=(start + timedelta(minutes=5)).astimezone(ZoneInfo("America/New_York")), open=Decimal(close), high=Decimal(high), low=Decimal(low), close=Decimal(close), total_volume=Decimal(volume), aggressive_buy_volume=Decimal(volume), aggressive_sell_volume=Decimal(), bar_delta=Decimal(volume), cumulative_volume_delta=Decimal(cvd), trade_count=1, vwap=Decimal(close), session_date=date(2025, 1, 3))


def profile() -> SessionProfile:
    return SessionProfile(session_date=date(2025, 1, 2), poc=Decimal("112"), vah=Decimal("120"), val=Decimal("110"), total_session_volume=Decimal("10"), value_area_volume=Decimal("7"), coverage_ratio=Decimal("0.7"), bucket_size=Decimal("10"), profile_hash="fixture", source_dataset_hash="fixture")


def strategy_bars(*, ambiguous: bool = False) -> list[FiveMinuteBar]:
    bars = [bar(i, cvd=str(i)) for i in range(18)]
    bars[10] = bar(10, high="130", low="115", close="118", volume="20", cvd="10")
    bars[11] = bar(11, high="128", cvd="9")
    bars[12] = bar(12, high="127", cvd="8")
    bars[13] = bar(13, high="132", low="116", close="119", cvd="5")
    bars[14] = bar(14, high="130", cvd="4")
    bars[15] = bar(15, high="129", cvd="3")
    bars[16] = bar(16, high="119", low="114", close="119", cvd="2")
    bars[17] = bar(17, high="143" if ambiguous else "119", low="111", close="116", cvd="1")
    return bars


def archive(tmp_path, content: str, member: str = "BTCUSDT-aggTrades-test.csv"):
    path = tmp_path / "fixture.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr(member, content)
    return path


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(True, True), (False, False), ("true", True), ("false", False),
     ("True", True), ("False", False), (1, True), (0, False),
     ("1", True), ("0", False)],
)
def test_aggregate_trade_parser_api_aliases_and_booleans(raw, expected) -> None:
    result = parse_binance_aggregate_trade({"a": 7, "p": "100.10", "q": "0.25", "f": 1, "l": 2, "T": 1735831800000, "E": 1735831800001, "m": raw}, source_file="fixture", source_hash="hash")
    assert result.buyer_is_maker is expected
    assert result.aggressor_side == ("SELL" if expected else "BUY")
    assert result.signed_quantity == (Decimal("-0.25") if expected else Decimal("0.25"))
    assert result.notional_quote == Decimal("25.025")


def test_archive_binance_headers_are_normalized(tmp_path) -> None:
    path = archive(
        tmp_path,
        "agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker\n"
        "7,100.10,0.25,1,2,1735831800000,true\n"
        "8,100.20,0.10,3,3,1735831800001,false\n",
    )
    records = list(AggregateTradeImporter(tmp_path).records_from_archive(path))
    assert [item.aggregate_trade_id for item in records] == [7, 8]
    assert records[0].buyer_is_maker is True
    assert records[1].buyer_is_maker is False


def test_headerless_archive_uses_fixed_order_and_keeps_first_row(tmp_path) -> None:
    path = archive(
        tmp_path,
        "7,100.10,0.25,1,2,1735831800000,1\n"
        "8,100.20,0.10,3,3,1735831800001,0\n",
    )
    records = list(AggregateTradeImporter(tmp_path).records_from_archive(path))
    assert [item.aggregate_trade_id for item in records] == [7, 8]
    assert records[0].price == Decimal("100.10")
    assert records[0].buyer_is_maker is True
    assert records[1].buyer_is_maker is False


def test_malformed_archive_lists_member_columns_and_missing_id(tmp_path) -> None:
    path = archive(
        tmp_path,
        "bad_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker\n"
        "7,100.10,0.25,1,2,1735831800000,true\n",
        member="bad-member.csv",
    )
    with pytest.raises(ValueError) as raised:
        list(AggregateTradeImporter(tmp_path).records_from_archive(path))
    message = str(raised.value)
    assert "bad-member.csv" in message
    assert "detected columns=" in message
    assert "aggregate_trade_id" in message


def test_boolean_parser_rejects_ambiguous_text() -> None:
    with pytest.raises(ValueError, match="expected true/false or 1/0"):
        parse_binance_aggregate_trade({"a": 7, "p": "100", "q": "1", "T": 1735831800000, "m": "no"}, source_file="fixture", source_hash="hash")


def test_validate_data_accepts_path_without_cache_root_and_emits_concise_json(tmp_path, capsys) -> None:
    when = datetime(2025, 1, 2, 14, 30, tzinfo=timezone.utc)
    parquet, manifest = AggregateTradeImporter(tmp_path / "cache").ingest_records([trade(1, when)])

    exit_code = cli_main([
        "--registry", str(tmp_path / "registry.sqlite3"),
        "value-area-trap", "validate-data", str(parquet),
    ])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    result = json.loads(captured.out)
    assert result == {
        "path": str(parquet.resolve()),
        "valid": True,
        "row_count": manifest.row_count,
        "dataset_hash": manifest.normalized_dataset_hash,
        "schema_version": manifest.schema_version,
        "errors": [],
        "warnings": [],
    }


def test_validate_data_missing_file_is_clean_nonzero_error(tmp_path, capsys) -> None:
    missing = tmp_path / "missing.parquet"
    exit_code = cli_main([
        "--registry", str(tmp_path / "registry.sqlite3"),
        "value-area-trap", "validate-data", str(missing),
    ])

    captured = capsys.readouterr()
    assert exit_code != 0
    result = json.loads(captured.out)
    assert result["valid"] is False
    assert "does not exist" in result["errors"][0]
    assert captured.err == ""


def test_validate_data_malformed_manifest_is_clean_nonzero_error(tmp_path, capsys) -> None:
    parquet = tmp_path / "malformed.parquet"
    parquet.write_bytes(b"not parquet")
    parquet.with_name("manifest.json").write_text("{}", encoding="utf-8")

    exit_code = cli_main([
        "--registry", str(tmp_path / "registry.sqlite3"),
        "value-area-trap", "validate-data", str(parquet),
    ])

    captured = capsys.readouterr()
    assert exit_code != 0
    result = json.loads(captured.out)
    assert result["valid"] is False
    assert "validation error" in result["errors"][0].lower()
    assert captured.err == ""


def test_aggregate_trade_parser_and_aggressor_classification() -> None:
    result = parse_binance_aggregate_trade({"a": 7, "p": "100.10", "q": "0.25", "f": 1, "l": 2, "T": 1735831800000, "E": 1735831800001, "m": True}, source_file="fixture", source_hash="hash")
    assert result.aggressor_side == "SELL"
    assert result.signed_quantity == Decimal("-0.25")
    assert result.notional_quote == Decimal("25.025")


def test_importer_deduplicates_content_addresses_and_writes_manifest(tmp_path) -> None:
    when = datetime(2025, 1, 2, 14, 30, tzinfo=timezone.utc)
    importer = AggregateTradeImporter(tmp_path)
    parquet, manifest = importer.ingest_records([trade(2, when + timedelta(seconds=1)), trade(1, when), trade(1, when)])
    again, same = importer.ingest_records([trade(1, when), trade(2, when + timedelta(seconds=1))])
    assert parquet == again and manifest.normalized_dataset_hash == same.normalized_dataset_hash
    assert manifest.row_count == 2 and manifest.duplicate_count == 1
    assert parquet.with_name("manifest.json").is_file()


def test_corrupted_archive_is_rejected_without_network(tmp_path) -> None:
    archive = tmp_path / "bad.zip"; archive.write_bytes(b"not a zip")
    with pytest.raises(ValueError, match="corrupted"):
        list(AggregateTradeImporter(tmp_path).records_from_archive(archive))


def test_profile_70_percent_expansion_and_tie_rules() -> None:
    base = datetime(2025, 1, 2, 14, 30, tzinfo=timezone.utc)
    records = [trade(1, base, "100", "1"), trade(2, base, "110", "4"), trade(3, base, "120", "2")]
    result = build_session_profile(records, date(2025, 1, 2), bucket_size=Decimal("10"))
    assert result and result.poc == Decimal("110")
    assert result.val == Decimal("110") and result.vah == Decimal("130")
    assert result.coverage_ratio >= Decimal("0.70")


def test_bars_reset_cvd_each_session_and_exclude_outside_window() -> None:
    day_one = datetime(2025, 1, 2, 14, 30, tzinfo=timezone.utc)
    day_two = datetime(2025, 1, 3, 14, 30, tzinfo=timezone.utc)
    bars = build_five_minute_bars([trade(1, day_one, maker=False), trade(2, day_one + timedelta(minutes=1), maker=True), trade(3, day_two, maker=False), trade(4, datetime(2025, 1, 2, 1, tzinfo=timezone.utc))])
    assert len(bars) == 2
    assert bars[0].cumulative_volume_delta == Decimal()
    assert bars[1].cumulative_volume_delta == Decimal("1")


def test_delayed_divergence_next_bar_entry_and_costed_trade() -> None:
    result = run_value_area_trap(strategy_bars(), {date(2025, 1, 2): profile()}, ValueAreaTrapConfig())
    assert result.significant_stop_runs == 1 and result.confirmed_divergences == 1
    assert len(result.trades) == 1
    assert result.trades[0]["entry_timestamp"] != result.trades[0]["signal_timestamp"]
    assert result.trades[0]["exit_reason"] == "TARGET"


def test_stop_first_ambiguity_is_persisted() -> None:
    result = run_value_area_trap(strategy_bars(ambiguous=True), {date(2025, 1, 2): profile()}, ValueAreaTrapConfig())
    assert result.same_bar_ambiguity_count == 1
    assert result.trades[0]["exit_reason"] == "STOP_FIRST_AMBIGUITY"


def test_value_area_intake_registers_native_aggregate_adapter(tmp_path) -> None:
    request = WorkflowInput(strategy_name="ValueAreaTrap", natural_language_description="ValueAreaTrap BTCUSDT aggregate trade strategy", requested_markets=["BTCUSDT"], requested_timeframes=["5m"], repository_root=str(tmp_path), registry_path=str(tmp_path / "registry.sqlite3"), dry_run=True, implementation_enabled=False)
    generated = PhaseBService(tmp_path / "registry.sqlite3").generate_spec(request)
    from research_pipeline.schemas.strategy_spec import load_strategy_spec
    spec = load_strategy_spec(generated.specification_path)
    assert spec.strategy_family == "value_area_trap_reference"
    assert default_adapter_registry().inspect(spec, tmp_path).healthy


def test_value_area_real_manifest_is_explicit_and_streaming_artifacts_are_real(tmp_path) -> None:
    from research_pipeline.phase_b.services import PhaseBService
    from research_pipeline.schemas.strategy_spec import load_strategy_spec
    from research_pipeline.phase_f1.service import MasterPipelineService

    source = archive(tmp_path, """agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker
1,100,1,1,1,1735828200000,false
2,101,1,2,2,1735828260000,true
""")
    parquet, manifest = AggregateTradeImporter(tmp_path / "cache").ingest_records(
        AggregateTradeImporter(tmp_path / "cache").records_from_archive(source)
    )
    request = WorkflowInput(strategy_name="ValueAreaTrap", natural_language_description="ValueAreaTrap BTCUSDT aggregate trade strategy", requested_markets=["BTCUSDT"], requested_timeframes=["5m"], repository_root=str(tmp_path), registry_path=str(tmp_path / "registry.sqlite3"), dry_run=True, implementation_enabled=False)
    generated = PhaseBService(tmp_path / "registry.sqlite3").generate_spec(request)
    spec = load_strategy_spec(generated.specification_path)
    adapter = default_adapter_registry().resolve(spec, tmp_path, manifest_path=parquet.with_name("manifest.json"))
    availability = adapter.data_availability(spec)[0]
    assert availability.classification.value == "AVAILABLE_PROXY"
    assert availability.dataset_hash == manifest.normalized_dataset_hash
    assert "CME" in " ".join(availability.warnings)
    split = MasterPipelineService._real_split(adapter, spec)
    artifact = adapter.run_baseline(spec, split, tmp_path / "real-baseline")
    assert artifact.command == ["value-area-trap-real-data", "baseline"]
    assert artifact.metrics["execution_mode"] == "REAL_DATA"
    assert artifact.metrics["dataset_hash"] == manifest.normalized_dataset_hash
    assert artifact.metrics["provider"] == "Binance USD-M Futures"
    assert artifact.metrics["alpha_evaluation"]["trades"] == []
    assert Path(artifact.experiment_dir, "5m_bars.parquet").is_file()
    assert Path(artifact.experiment_dir, "session_profiles.parquet").is_file()
    assert Path(artifact.experiment_dir, "strategy_events.parquet").is_file()
    assert Path(artifact.experiment_dir, "trades.parquet").is_file()
    assert artifact.diagnostic_manifest_path


def test_value_area_real_manifest_is_required_and_does_not_discover_fallback(tmp_path) -> None:
    from research_pipeline.phase_b.services import PhaseBService
    from research_pipeline.schemas.strategy_spec import load_strategy_spec

    request = WorkflowInput(strategy_name="ValueAreaTrap", natural_language_description="ValueAreaTrap BTCUSDT aggregate trade strategy", requested_markets=["BTCUSDT"], requested_timeframes=["5m"], repository_root=str(tmp_path), registry_path=str(tmp_path / "registry.sqlite3"), dry_run=True, implementation_enabled=False)
    generated = PhaseBService(tmp_path / "registry.sqlite3").generate_spec(request)
    spec = load_strategy_spec(generated.specification_path)
    with pytest.raises(ValueError, match="explicit ValueAreaTrap manifest"):
        ValueAreaTrapAdapter(spec, tmp_path).data_availability(spec)


def test_value_area_real_run_input_persists_manifest_path(tmp_path) -> None:
    from research_pipeline.phase_f1.service import MasterPipelineService

    options = MasterPipelineService.input_model("intake.json", tmp_path, mode="real_run", data_manifest_path=tmp_path / "manifest.json")
    assert options.mode == "real_run"
    assert options.data_manifest_path == str((tmp_path / "manifest.json").resolve())


def test_alpha_evaluation_pass_mll_and_qualified_calendar_requirement() -> None:
    source = {"trade_id": "t", "entry_timestamp": "2025-01-02T15:00:00+00:00", "entry_price": "100", "initial_stop_price": "90", "quantity": "1", "gross_pnl": "2000", "fees": "0", "slippage_cost": "0"}
    passed = run_alpha_zero_scenario([source], profile="ZERO_25K_EVALUATION", risk_per_trade_usd=Decimal("100"))
    assert passed.outcome == "PASSED"
    failed = run_alpha_zero_scenario([{**source, "gross_pnl": "-1100"}], profile="ZERO_25K_EVALUATION", risk_per_trade_usd=Decimal("1000"))
    assert failed.outcome == "FAILED_MLL"
    qualified = run_alpha_zero_scenario([source], profile="ZERO_25K_QUALIFIED")
    assert qualified.outcome == "NEWS_DATA_UNAVAILABLE"
