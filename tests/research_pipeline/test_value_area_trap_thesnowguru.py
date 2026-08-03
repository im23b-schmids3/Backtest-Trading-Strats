from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from research_pipeline.cli import main as cli_main
from research_pipeline.value_area_trap.thesnowguru import (
    _is_candidate,
    audit_thesnowguru_data,
    import_thesnowguru_es,
)


def _source_root(tmp_path: Path, *, exact: bool = True) -> Path:
    root = tmp_path / "external_data" / "thesnowguru"
    root.mkdir(parents=True)
    (root / "README.md").write_text(
        "Tick Data - Added only S&P futures tick data, make sure to add zip files to the tick data folders.\n"
        "Timezone: all timestamps are in GMT.\n",
        encoding="utf-8",
    )
    archive = root / "tickdata" / "indicies" / "s&p-tick.zip"
    archive.parent.mkdir(parents=True)
    volume = "2" if exact else "0"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr(
            "SP.csv",
            "date,time,price,volume,side\n"
            f"01/05/2026,09:30:00.000,5000.25,{volume},BUY\n"
            f"01/05/2026,09:35:00.000,5000.50,{volume},SELL\n",
        )
    index_tick = root / "tickdata" / "indicies" / "USA500.IDXUSD_Ticks.csv"
    index_tick.write_text(
        "Local time,Ask,Bid,AskVolume,BidVolume\n"
        "04.09.2023 01:00:05.962 GMT+0300,4516.181,4515.664,1,1\n",
        encoding="utf-8",
    )
    index_bar = root / "indices" / "nasdaq100" / "USATECHIDXUSD_M1.csv"
    index_bar.parent.mkdir(parents=True)
    index_bar.write_text(
        "Time,Open,High,Low,Close,Volume\n"
        "2026-05-01 09:30:00,12000,12001,11999,12000,1\n",
        encoding="utf-8",
    )
    stock_export = root / "stock" / "sp500-stocks.csv"
    stock_export.parent.mkdir(parents=True)
    stock_export.write_bytes(b"Company,Symbol,Price,\xa0Change\nExample,EX,1,0\n")
    return root


def test_audit_classifies_futures_and_index_candidates_without_conflation(tmp_path: Path) -> None:
    source = _source_root(tmp_path)
    result = audit_thesnowguru_data(repository_root=tmp_path, source_root=source, staging_root="staging")
    candidates = {item.relative_path: item for item in result.candidates}
    futures = candidates["tickdata/indicies/s&p-tick.zip::SP.csv"]
    usa500 = candidates["tickdata/indicies/USA500.IDXUSD_Ticks.csv"]
    nasdaq = candidates["indices/nasdaq100/USATECHIDXUSD_M1.csv"]
    assert futures.classification == "VERIFIED_FUTURES_TICK"
    assert futures.cvd_support == "EXACT_CVD_SUPPORTED"
    assert usa500.classification == "INDEX_OR_CFD_BAR"
    assert usa500.cvd_support == "CVD_NOT_SUPPORTED"
    assert nasdaq.classification == "INDEX_OR_CFD_BAR"
    assert candidates["stock/sp500-stocks.csv"].classification == "UNSUITABLE"
    assert result.nq_verified_futures_present is False
    assert result.es_import_available is True
    assert Path(result.source_manifest_path).is_file()
    assert Path(result.schema_manifest_path).is_file()
    assert Path(result.data_quality_report_path).is_file()
    assert Path(result.monthly_coverage_report_path).is_file()
    assert futures.monthly_coverage == {"2026-01": 2}


def test_audit_is_idempotent_and_imports_only_exact_verified_es_ticks(tmp_path: Path) -> None:
    source = _source_root(tmp_path)
    audit = audit_thesnowguru_data(repository_root=tmp_path, source_root=source, staging_root="staging")
    repeated = audit_thesnowguru_data(repository_root=tmp_path, source_root=source, staging_root="staging")
    assert repeated.audit_hash == audit.audit_hash
    imported = import_thesnowguru_es(audit_path=Path(audit.audit_root) / "audit.json", cache_root=tmp_path / "data" / "value_area_trap")
    assert imported["status"] == "IMPORTED"
    assert "ES_THESNOWGURU" in imported["parquet"]
    assert Path(imported["manifest_path"]).is_file()
    assert Path(imported["provenance_path"]).is_file()


def test_import_is_blocked_when_volume_cannot_support_exact_cvd(tmp_path: Path) -> None:
    source = _source_root(tmp_path, exact=False)
    audit = audit_thesnowguru_data(repository_root=tmp_path, source_root=source, staging_root="staging")
    assert audit.es_import_available is False
    with pytest.raises(ValueError, match="exact CVD-supported"):
        import_thesnowguru_es(audit_path=Path(audit.audit_root) / "audit.json", cache_root=tmp_path / "data" / "value_area_trap")


def test_audit_reports_streaming_quality_issues_without_false_es_path_matches(tmp_path: Path) -> None:
    source = _source_root(tmp_path)
    quote = source / "tickdata" / "indicies" / "USA500.IDXUSD_Ticks.csv"
    quote.write_text(
        "Local time,Ask,Bid,AskVolume,BidVolume\n"
        "04.09.2023 01:00:05.962 GMT+0300,4516.181,4515.664,1,1\n"
        "04.09.2023 01:00:05.962 GMT+0300,4516.181,4515.664,1,1\n"
        "03.09.2023 01:00:05.962 GMT+0300,-1,4515.664,-1,1\n",
        encoding="utf-8",
    )
    result = audit_thesnowguru_data(repository_root=tmp_path, source_root=source, staging_root="staging")
    quality = next(item.quality for item in result.candidates if item.relative_path.endswith("USA500.IDXUSD_Ticks.csv"))
    assert quality["duplicate_timestamps"] == 1
    assert quality["repeated_adjacent_rows"] == 1
    assert quality["non_monotonic_timestamps"] == 1
    assert quality["impossible_prices"] == 1
    assert quality["impossible_sizes"] == 1
    assert _is_candidate(Path("other/indices/unrelated.csv")) is False
    assert _is_candidate(Path("indices/dow30/USA30IDXUSD_M1.csv")) is False


def test_audit_cli_emits_json_and_writes_only_to_requested_staging(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = _source_root(tmp_path)
    assert cli_main([
        "--registry", str(tmp_path / "registry.sqlite3"),
        "value-area-trap", "audit-thesnowguru-data",
        "--repository-root", str(tmp_path),
        "--source-root", str(source),
        "--staging-root", "staging",
    ]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "AUDIT_COMPLETE"
    assert Path(result["audit_root"]).is_relative_to(tmp_path / "staging")
