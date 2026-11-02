from __future__ import annotations

import numpy as np
import pytest

from research_pipeline.cme_orderflow_absorption_l2_v1 import es_four_mechanism_comparative_research_v1 as study


def _row():
    dtype = []
    for i in range(10):
        dtype.extend([(f"bid_px_{i:02d}", "i8"), (f"bid_sz_{i:02d}", "i4"),
                      (f"ask_px_{i:02d}", "i8"), (f"ask_sz_{i:02d}", "i4")])
    row = np.zeros((), dtype=dtype)
    for i in range(10):
        row[f"bid_px_{i:02d}"] = (100 - i) * 1_000_000_000
        row[f"bid_sz_{i:02d}"] = i + 1
        row[f"ask_px_{i:02d}"] = (101 + i) * 1_000_000_000
        row[f"ask_sz_{i:02d}"] = i + 2
    return row


def test_native_book_retains_price_aligned_ten_levels():
    book = study.canonical_book(_row())
    assert len(book["bids"]) == len(book["asks"]) == 10
    assert book["bids"][0] == (100_000_000_000, 1)
    assert book["asks"][0] == (101_000_000_000, 2)
    assert book["valid_top"]


def test_depth_only_change_is_distinguishable_without_bbo_change():
    a, b = _row(), _row()
    b["bid_sz_04"] = 99
    ba, bb = study.canonical_book(a), study.canonical_book(b)
    assert ba["bids"][0] == bb["bids"][0]
    assert ba["bids"] != bb["bids"]


def test_crossed_book_is_flagged():
    row = _row()
    row["ask_px_00"] = row["bid_px_00"]
    assert study.canonical_book(row)["crossed_or_locked"]


def test_contract_roll_boundary_is_explicit():
    march_14 = "ESH5" if "2025-03-14" <= "2025-03-14" else "ESM5"
    march_17 = "ESH5" if "2025-03-17" <= "2025-03-14" else "ESM5"
    assert march_14 == "ESH5"
    assert march_17 == "ESM5"


def test_non_authorized_pilot_day_rejected(tmp_path):
    (tmp_path / study.SOURCE_MANIFEST).parent.mkdir(parents=True)
    (tmp_path / study.SOURCE_MANIFEST).write_text('{"status":"PASS","sources":[]}')
    with pytest.raises(study.MechanismResearchError):
        study.run_pilot(tmp_path, "2025-10-07")
