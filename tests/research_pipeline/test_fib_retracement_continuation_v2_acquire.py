from __future__ import annotations

import json
import hashlib
import io
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from research_pipeline.fib_retracement_continuation_v2.acquire import AcquisitionError, BinanceUsdmKlineAcquirer


def _rows(day: date) -> list[list[object]]:
    start = datetime.combine(day, datetime.min.time(), timezone.utc)
    return [[int((start + timedelta(minutes=n)).timestamp() * 1000), "10", "11", "9", "10.5", "2"] for n in range(1440)]


def test_rest_partition_is_content_addressed_sealed_and_reusable(tmp_path: Path) -> None:
    day = date(2026, 8, 1)
    raw = json.dumps(_rows(day)).encode()
    acquirer = BinanceUsdmKlineAcquirer(tmp_path / "data" / "fib_prospective_v2", fetch=lambda _: raw)
    one = acquirer.acquire_partition("BTCUSDT", day.replace(day=1), day)
    partition = one["dailyPartitions"][0]
    assert partition["rowCount"] == 1440
    assert (tmp_path / "data" / "fib_prospective_v2" / partition["path"] / "source.json").is_file()
    two = acquirer.acquire_partition("BTCUSDT", day.replace(day=1), day)
    assert two == one


def test_validator_fails_closed_on_missing_or_duplicate_minutes() -> None:
    start = datetime(2026, 8, 6, tzinfo=timezone.utc)
    with pytest.raises(AcquisitionError, match="missing-minute distribution"):
        BinanceUsdmKlineAcquirer._validate([[int(start.timestamp() * 1000), "1", "1", "1", "1", "0"]], start, start + timedelta(minutes=2))
    with pytest.raises(AcquisitionError, match="duplicate/non-increasing"):
        BinanceUsdmKlineAcquirer._validate([[int(start.timestamp() * 1000), "1", "1", "1", "1", "0"], [int(start.timestamp() * 1000), "1", "1", "1", "1", "0"]], start, start + timedelta(minutes=1))


def test_archive_header_and_checksum_are_verified() -> None:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("BTCUSDT-1m-2022-01.csv", "open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore\n1640995200000,1,2,1,1,0,1640995259999,0,0,0,0,0\n")
    blob = stream.getvalue()
    assert BinanceUsdmKlineAcquirer._archive_rows(blob)[0][0] == "1640995200000"
    BinanceUsdmKlineAcquirer._checksum(f"{hashlib.sha256(blob).hexdigest()} file.zip".encode(), blob)
    with pytest.raises(AcquisitionError, match="checksum mismatch"):
        BinanceUsdmKlineAcquirer._checksum(b"0" * 64, blob)
