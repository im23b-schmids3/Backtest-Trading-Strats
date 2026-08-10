"""Build frozen DEVELOPMENT-only score calibration for ES MBO OOS V1."""
from __future__ import annotations

import base64
import json
import struct
import zlib
from pathlib import Path

from .analysis import Diagnostics, day_and_seconds, volume_profile
from .engine import CausalMBOBook
from .loader import EXPECTED_SHA256, sha256_file, stream_mbo, validate_metadata
from .oos_backtest_runner import _development_rank, _feature_values, load_development_calibration
from .report import _rank, select_tiers

ROOT = Path(__file__).resolve().parents[3]
DBN = ROOT / "data/cme_orderflow_absorption_v1/ESU6/mbo/ESU6_2026-07-20_2026-08-01_mbo.dbn"
OUT = ROOT / "docs/research_pipeline/cme_orderflow_absorption_v1/development-score-calibration.json"

EXPECTED_INTERACTIONS = 3089
PROGRESS_EVERY = 5_000_000
PILOT_DATES = {
    "2026-07-20", "2026-07-21", "2026-07-22", "2026-07-23", "2026-07-24",
    "2026-07-27", "2026-07-28", "2026-07-29", "2026-07-30", "2026-07-31",
}
SCORE_COMPONENTS = {
    "absorption": [
        "relevant_directional_aggressive_volume",
        "executions",
        "absolute_aggressive_imbalance",
        "inverse_through_level_progress",
        "interaction_end_rejection",
    ],
    "replenishment": [
        "replenishment_count",
        "replenished_volume",
        "replenished_execution_ratio",
        "repeated_cycles",
        "queue_persistence_proxy",
    ],
}
FROZEN_THRESHOLDS = {
    "absorption_p95": 0.7977986403366786,
    "replenishment_p95": 0.7785691162188411,
}
PREVIOUS_RTH_SOURCE_BY_OOS_DATE = {
    "2026-08-03": "2026-07-31",
    "2026-08-04": "2026-08-03",
    "2026-08-05": "2026-08-04",
    "2026-08-06": "2026-08-05",
    "2026-08-07": "2026-08-06",
}

def _encode_sorted(values: list[float]) -> str:
    if len(values) != EXPECTED_INTERACTIONS:
        raise RuntimeError(f"bad calibration sample size: {len(values)}")
    packed = struct.pack(f"<{EXPECTED_INTERACTIONS}d", *sorted(float(v) for v in values))
    return base64.b64encode(zlib.compress(packed, level=9)).decode("ascii")

def _build_rows() -> tuple[list[dict], Diagnostics, int]:
    digest, size = sha256_file(DBN)
    if digest != EXPECTED_SHA256:
        raise RuntimeError(f"July DBN SHA mismatch: {digest}")
    validate_metadata(DBN)

    book = CausalMBOBook()
    diag = Diagnostics()
    context_dates: set[str] = set()
    seen = 0

    for rec in stream_mbo(DBN):
        seen += 1
        day, _ = day_and_seconds(rec.ts_recv)
        if day not in context_dates:
            diag.finish_day_context(day)
            context_dates.add(day)
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
            print(f"[calibration] records={seen:,}", flush=True)

    diag.finalize()
    rows = [r for r in diag.interaction_rows() if r["date"] in PILOT_DATES]
    if len(rows) != EXPECTED_INTERACTIONS:
        raise RuntimeError(f"development interaction mismatch: got {len(rows)}, expected {EXPECTED_INTERACTIONS}")
    print(f"[calibration] reconstructed interactions={len(rows):,}", flush=True)
    return rows, diag, size

def main() -> int:
    rows, diag, dbn_bytes = _build_rows()

    selection = select_tiers(rows)
    for key, expected in FROZEN_THRESHOLDS.items():
        actual = float(selection[key])
        if actual != expected:
            raise RuntimeError(f"frozen threshold mismatch for {key}: {actual} != {expected}")

    features = {name: [] for group in SCORE_COMPONENTS.values() for name in group}
    for row in rows:
        _, values = _feature_values(row)
        for name in features:
            features[name].append(float(values[name]))

    mappings = {name: _encode_sorted(values) for name, values in features.items()}

    for name, values in features.items():
        original = _rank(values)
        replay = [_development_rank(mappings[name], float(v)) for v in values]
        max_diff = max(abs(a - b) for a, b in zip(original, replay))
        if max_diff != 0.0:
            raise RuntimeError(f"rank mapping mismatch for {name}: max_diff={max_diff}")

    profile = volume_profile(diag.rth_prices.get("2026-07-31", []))
    if profile is None:
        raise RuntimeError("missing 2026-07-31 RTH volume profile")

    july31 = {
        "PRIOR_RTH_HIGH": int(profile["high"]),
        "PRIOR_RTH_LOW": int(profile["low"]),
        "PRIOR_RTH_POC": int(profile["poc"]),
        "PRIOR_RTH_VAH": int(profile["vah"]),
        "PRIOR_RTH_VAL": int(profile["val"]),
    }

    payload = {
        "schema_version": 1,
        "source": "DEVELOPMENT_ONLY",
        "development_dbn_sha256": EXPECTED_SHA256,
        "development_dbn_bytes": dbn_bytes,
        "development_interaction_count": EXPECTED_INTERACTIONS,
        "score_components": SCORE_COMPONENTS,
        "feature_rank_mapping_encoding": "zlib-base64-f64le-sorted-sample",
        "feature_rank_mapping": mappings,
        "verified_thresholds": FROZEN_THRESHOLDS,
        "previous_rth_context": {"2026-07-31": july31},
        "previous_rth_source_by_oos_date": PREVIOUS_RTH_SOURCE_BY_OOS_DATE,
        "oos_rank_recomputation_count": 0,
        "oos_score_calibration_source": "DEVELOPMENT_ONLY",
        "trading_strategy_executed": False,
        "pnl_calculated": False,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True), encoding="utf-8")
    tmp.replace(OUT)

    loaded = load_development_calibration()
    if loaded["development_interaction_count"] != EXPECTED_INTERACTIONS:
        raise RuntimeError("runner calibration reload failed")

    print(f"[calibration] wrote {OUT}", flush=True)
    print("[calibration] July31 levels " + " ".join(f"{k}={v / 1_000_000_000:.2f}" for k, v in july31.items()), flush=True)
    print("[calibration] DEVELOPMENT_CALIBRATION_READY", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())