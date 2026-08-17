"""Build a compact V2 research dataset from the already-seen ES MBO periods.

Inputs
------
- DEVELOPMENT_JULY: 2026-07-20..2026-07-31 sealed development DBN
- SEEN_OOS_AUG:     2026-08-03..2026-08-07 previously evaluated V1 OOS DBN

This exporter does NOT create new OOS evidence and does NOT optimize V2.
It reconstructs all structural interactions once, scores every row with the
already-frozen July DEVELOPMENT calibration, marks V1 PLUS membership, and
writes compact CSV/JSON artifacts for fast V2 research.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from databento import DBNStore

from .analysis import Diagnostics, day_and_seconds
from .engine import CausalMBOBook
from .loader import EXPECTED_SHA256 as JULY_SHA, sha256_file, stream_mbo, validate_metadata
from .oos_backtest_runner import (
    _score,
    is_snapshot_record,
    load_development_calibration,
    load_frozen,
)

ROOT = Path(__file__).resolve().parents[3]
JULY_DBN = ROOT / "data/cme_orderflow_absorption_v1/ESU6/mbo/ESU6_2026-07-20_2026-08-01_mbo.dbn"
AUG_DBN = ROOT / "data/cme_orderflow_absorption_v1/oos_v1/ESU6/mbo/ESU6_2026-08-03_2026-08-08_mbo.dbn"
OUT_DIR = ROOT / "research_runs/CMEOrderflowAbsorption.ES_V2_RESEARCH/seen_15_rth"

JULY_DATES = {
    "2026-07-20", "2026-07-21", "2026-07-22", "2026-07-23", "2026-07-24",
    "2026-07-27", "2026-07-28", "2026-07-29", "2026-07-30", "2026-07-31",
}
AUG_DATES = {"2026-08-03", "2026-08-04", "2026-08-05", "2026-08-06", "2026-08-07"}
EXPECTED_JULY_INTERACTIONS = 3089
EXPECTED_AUG_INTERACTIONS = 1430
EXPECTED_COMBINED = EXPECTED_JULY_INTERACTIONS + EXPECTED_AUG_INTERACTIONS
EXPECTED_AUG_SHA = "BE4B56639E56DF9AACE81621E4E276463EA8AF889104F35F1744400310D53AA3"
PROGRESS_EVERY = 5_000_000


def _action(rec: Any) -> str:
    return str(getattr(rec.action, "value", rec.action))


def _reconstruct_july() -> tuple[list[dict[str, Any]], int]:
    digest, _ = sha256_file(JULY_DBN)
    if digest != JULY_SHA:
        raise RuntimeError(f"July SHA mismatch: {digest}")
    validate_metadata(JULY_DBN)

    book = CausalMBOBook()
    diag = Diagnostics()
    context: set[str] = set()
    seen = 0

    for rec in stream_mbo(JULY_DBN):
        seen += 1
        day, _ = day_and_seconds(rec.ts_recv)
        if day not in context:
            diag.finish_day_context(day)
            context.add(day)

        if _action(rec) == "N":
            if seen % PROGRESS_EVERY == 0:
                print(f"[V2 research][July] records={seen:,}", flush=True)
            continue

        applied = book.apply(
            action=rec.action,
            side=rec.side,
            price=rec.price,
            size=rec.size,
            order_id=rec.order_id,
            sequence=rec.sequence,
            ts_recv=rec.ts_recv,
            channel_id=rec.channel_id,
            validate_sequence=False,
            mutate_execution=False,
        )
        diag.observe(rec, applied, book.spread())

        if seen % PROGRESS_EVERY == 0:
            print(f"[V2 research][July] records={seen:,}", flush=True)

    diag.finalize()
    rows = [r for r in diag.interaction_rows() if r["date"] in JULY_DATES]
    if len(rows) != EXPECTED_JULY_INTERACTIONS:
        raise RuntimeError(f"July interactions={len(rows)}, expected {EXPECTED_JULY_INTERACTIONS}")
    return rows, seen


def _reconstruct_aug() -> tuple[list[dict[str, Any]], int, int]:
    digest, _ = sha256_file(AUG_DBN)
    if digest != EXPECTED_AUG_SHA:
        raise RuntimeError(f"August SHA mismatch: {digest}")

    _, manifest = load_frozen()
    eligible = set(manifest["chronology"]["eligible_rth_dates"])
    if eligible != AUG_DATES:
        raise RuntimeError(f"unexpected August eligible dates: {sorted(eligible)}")

    calibration = load_development_calibration()
    july_context = calibration["previous_rth_context"]["2026-07-31"]

    book = CausalMBOBook()
    diag = Diagnostics()
    context: set[str] = set()
    seen = 0
    snapshots = 0

    for rec in DBNStore.from_file(AUG_DBN):
        seen += 1
        if getattr(rec, "instrument_id", None) != 42140870:
            raise RuntimeError("wrong ESU6 instrument id")

        if is_snapshot_record(rec, manifest):
            snapshots += 1
            book.apply(
                action=rec.action,
                side=rec.side,
                price=rec.price,
                size=rec.size,
                order_id=rec.order_id,
                sequence=rec.sequence,
                ts_recv=rec.ts_recv,
                channel_id=rec.channel_id,
                validate_sequence=False,
                mutate_execution=False,
            )
            if seen % PROGRESS_EVERY == 0:
                print(f"[V2 research][Aug] records={seen:,}", flush=True)
            continue

        day, _ = day_and_seconds(rec.ts_recv)
        if day not in eligible:
            raise RuntimeError(f"ordinary non-OOS record: {day}")

        if day not in context:
            if day == "2026-08-03":
                diag.levels[day] = dict(july_context)
            else:
                diag.finish_day_context(day)
            context.add(day)

        if _action(rec) == "N":
            if seen % PROGRESS_EVERY == 0:
                print(f"[V2 research][Aug] records={seen:,}", flush=True)
            continue

        applied = book.apply(
            action=rec.action,
            side=rec.side,
            price=rec.price,
            size=rec.size,
            order_id=rec.order_id,
            sequence=rec.sequence,
            ts_recv=rec.ts_recv,
            channel_id=rec.channel_id,
            validate_sequence=False,
            mutate_execution=False,
        )
        diag.observe(rec, applied, book.spread())

        if seen % PROGRESS_EVERY == 0:
            print(f"[V2 research][Aug] records={seen:,}", flush=True)

    diag.finalize()
    rows = [r for r in diag.interaction_rows() if r["date"] in AUG_DATES]
    if len(rows) != EXPECTED_AUG_INTERACTIONS:
        raise RuntimeError(f"August interactions={len(rows)}, expected {EXPECTED_AUG_INTERACTIONS}")
    return rows, seen, snapshots


def _json_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def main() -> int:
    calibration = load_development_calibration()
    contract, _ = load_frozen()
    frozen = contract["frozen_selection"]

    print("[V2 research] reconstructing July development...", flush=True)
    july_raw, july_records = _reconstruct_july()
    print(f"[V2 research] July interactions={len(july_raw):,}", flush=True)

    print("[V2 research] reconstructing seen August OOS...", flush=True)
    aug_raw, aug_records, aug_snapshots = _reconstruct_aug()
    print(f"[V2 research] August interactions={len(aug_raw):,}", flush=True)

    july = _score(july_raw, calibration)
    aug = _score(aug_raw, calibration)

    rows: list[dict[str, Any]] = []
    for split, source_rows in (("DEVELOPMENT_JULY", july), ("SEEN_OOS_AUG", aug)):
        for r in source_rows:
            x = dict(r)
            x["research_split"] = split
            x["v1_high_absorption"] = x["absorption_score"] >= frozen["absorption_p95"]
            x["v1_strong_replenishment"] = x["replenishment_score"] >= frozen["replenishment_p95"]
            x["v1_plus"] = (
                x["level"] in frozen["mandatory_structural_levels"]
                and x["v1_high_absorption"]
                and x["v1_strong_replenishment"]
            )
            rows.append(x)

    if len(rows) != EXPECTED_COMBINED:
        raise RuntimeError(f"combined interactions={len(rows)}, expected {EXPECTED_COMBINED}")

    july_plus = sum(r["v1_plus"] for r in rows if r["research_split"] == "DEVELOPMENT_JULY")
    aug_plus = sum(r["v1_plus"] for r in rows if r["research_split"] == "SEEN_OOS_AUG")
    if july_plus != 33:
        raise RuntimeError(f"July V1 PLUS={july_plus}, expected 33")
    if aug_plus != 21:
        raise RuntimeError(f"August V1 PLUS={aug_plus}, expected 21")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "all-interactions.csv"
    json_path = OUT_DIR / "research-summary.json"

    fields = sorted({k for r in rows for k in r})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: _json_cell(r.get(k)) for k in fields})

    by_day = Counter(str(r["date"]) for r in rows)
    plus_by_day = Counter(str(r["date"]) for r in rows if r["v1_plus"])
    direction_all = Counter(str(r["direction"]) for r in rows)
    direction_plus = Counter(str(r["direction"]) for r in rows if r["v1_plus"])
    level_all = Counter(str(r["level"]) for r in rows)
    level_plus = Counter(str(r["level"]) for r in rows if r["v1_plus"])

    summary = {
        "status": "V2_RESEARCH_DATASET_READY",
        "interpretation": "ALL_ROWS_ARE_SEEN_DATA_NOT_NEW_OOS_EVIDENCE",
        "records_scanned": {
            "development_july": july_records,
            "seen_oos_aug": aug_records,
            "seen_oos_aug_snapshot_records": aug_snapshots,
        },
        "interaction_counts": {
            "development_july": len(july),
            "seen_oos_aug": len(aug),
            "combined": len(rows),
        },
        "v1_plus_counts": {
            "development_july": july_plus,
            "seen_oos_aug": aug_plus,
            "combined": july_plus + aug_plus,
        },
        "interactions_by_day": dict(sorted(by_day.items())),
        "v1_plus_by_day": dict(sorted(plus_by_day.items())),
        "direction_all": dict(sorted(direction_all.items())),
        "direction_v1_plus": dict(sorted(direction_plus.items())),
        "level_all": dict(sorted(level_all.items())),
        "level_v1_plus": dict(sorted(level_plus.items())),
        "score_calibration_source": "DEVELOPMENT_ONLY",
        "oos_rank_recomputation_count": 0,
        "v1_absorption_p95": frozen["absorption_p95"],
        "v1_replenishment_p95": frozen["replenishment_p95"],
        "v2_rule_selected": False,
        "pnl_optimization_performed": False,
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"[V2 research] wrote {csv_path}", flush=True)
    print(f"[V2 research] wrote {json_path}", flush=True)
    print(f"[V2 research] combined interactions={len(rows):,}; V1 PLUS={july_plus + aug_plus}", flush=True)
    print("[V2 research] V2_RESEARCH_DATASET_READY", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())