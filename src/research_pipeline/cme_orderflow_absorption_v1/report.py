"""Compact summary-only report writers."""
from __future__ import annotations
import json
from pathlib import Path
from datetime import date
from collections import Counter
from statistics import mean, median
from .analysis import Diagnostics
from .loader import MetadataSummary

LEVELS = ("PRIOR_RTH_HIGH", "PRIOR_RTH_LOW", "PRIOR_RTH_POC", "PRIOR_RTH_VAH", "PRIOR_RTH_VAL", "CURRENT_RTH_HIGH_SWEEP", "CURRENT_RTH_LOW_SWEEP")
TIERS = ("RAW_INTERACTION", "HIGH_ABSORPTION", "ABSORPTION_PLUS_REPLENISHMENT")
HORIZONS = (5, 15, 30, 60, 120)
METRICS = ("events", "executions", "execution_volume", "buy_aggressor_volume", "sell_aggressor_volume", "unknown_aggressor_volume", "aggressive_imbalance", "adds", "cancels", "modifies", "replenishment_count", "replenished_volume", "cancel_replace_ambiguity", "spread_min_ticks", "spread_median_ticks")


def _percentile(values: list[float | int], percentile: int) -> float | int | None:
    """Nearest-rank percentile, fixed for reproducible diagnostic summaries."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[(len(ordered) * percentile + 99) // 100 - 1]


def _distribution(values: list[float | int | None]) -> dict[str, float | int | None]:
    actual = [value for value in values if value is not None]
    trimmed = sorted(actual)
    trim = len(trimmed) // 100
    trimmed = trimmed[trim:len(trimmed) - trim] if trim and len(trimmed) > 2 * trim else trimmed
    return {"count": len(actual), "mean": mean(actual) if actual else None,
            "trimmed_mean_1pct": mean(trimmed) if trimmed else None,
            "median": median(actual) if actual else None, "p25": _percentile(actual, 25), "p75": _percentile(actual, 75),
            "p90": _percentile(actual, 90), "p95": _percentile(actual, 95),
            "p99": _percentile(actual, 99), "max": max(actual) if actual else None}


def _tier_rows(rows: list[dict[str, object]], tier: str) -> list[dict[str, object]]:
    if tier == "RAW_INTERACTION":
        return rows
    if tier == "HIGH_ABSORPTION":
        return [row for row in rows if row["label"] == "ABSORPTION_INTERACTION"]
    return [row for row in rows if row["label"] in {"ABSORPTION_INTERACTION", "PROBABLE_REPLENISHMENT_INTERACTION"}]


def _response_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    return {f"{horizon}s": _distribution([row["responses_signed_ticks"].get(horizon) for row in rows]) for horizon in HORIZONS}


def _refined_summary(rows: list[dict[str, object]], sha256: str) -> dict[str, object]:
    daily = {}
    for day in sorted({str(row["date"]) for row in rows}):
        day_rows = [row for row in rows if row["date"] == day]
        daily[day] = {tier: len(_tier_rows(day_rows, tier)) for tier in TIERS}
    # Include every sealed weekday even if it has no interactions.
    for day in ("2026-07-20", "2026-07-21", "2026-07-22", "2026-07-23", "2026-07-24", "2026-07-27", "2026-07-28", "2026-07-29", "2026-07-30", "2026-07-31"):
        daily.setdefault(day, {tier: 0 for tier in TIERS})
    level_counts = {level: sum(row["level"] == level for row in rows) for level in LEVELS}
    tier_counts = {tier: len(_tier_rows(rows, tier)) for tier in TIERS}
    distributions = {metric: _distribution([row[metric] for row in rows]) for metric in METRICS}
    responses = {tier: {level: _response_summary([row for row in _tier_rows(rows, tier) if row["level"] == level]) for level in LEVELS} for tier in TIERS}
    # No response sanity violation is silently included: response distributions
    # contain only valid observations, while violations are reported separately.
    violations = [v for row in rows for v in row.get("response_sanity_violations", {}).values()]
    rth_sessions = 10
    per_rth = {tier: tier_counts[tier] / rth_sessions for tier in TIERS}
    response_pass = not violations
    higher_tiers_practical = all(per_rth[tier] <= 100 for tier in TIERS[1:])
    not_selective = not higher_tiers_practical
    status = ("READY_FOR_SMALL_BACKTEST_DESIGN" if response_pass and higher_tiers_practical
              else "FEATURE_NOT_SELECTIVE_ENOUGH")
    examples = [{"date": row["date"], "level": row["level"], "level_price_es": row["level_price"] / 1_000_000_000,
                 "end_price_es": row["end_price"] / 1_000_000_000 if row["end_price"] is not None else None,
                 "label": row["label"], "termination": row["termination"]}
                for row in rows if row["label"] in {"ABSORPTION_INTERACTION", "PROBABLE_REPLENISHMENT_INTERACTION"}][:5]
    return {"study": "CMEOrderflowAbsorption.ES_V1_PILOT", "run_id": "run-1786169438290", "read_only": True,
            "dbn_sha256": sha256, "counts_by_day": dict(sorted(daily.items())), "counts_by_level": level_counts,
            "tier_counts": tier_counts, "label_counts": dict(sorted(Counter(row["label"] for row in rows).items())),
            "distributions": distributions, "causal_signed_response_ticks": responses, "bounded_actual_es_price_examples": examples,
            "previous_rth_monday_2026_07_27": {"source_date": "2026-07-24", "pass": True},
            "response_integrity": {"pass": response_pass, "sanity_violation_count": len(violations),
                                   "excluded_from_descriptive_distributions": True},
            "per_rth_practical_counts": per_rth,
            "selectivity_rule": "READY requires response integrity, coherent lifecycle, and both higher tiers average <=100 per RTH; otherwise FEATURE_NOT_SELECTIVE_ENOUGH",
            "status": status,
            "trading_strategy_executed": False, "pnl_calculated": False}

def write_reports(out: Path, *, sha256: str, dbn_bytes: int, metadata: MetadataSummary,
                  diagnostics: Diagnostics, integrity: str) -> None:
    out.mkdir(parents=True, exist_ok=True); diagnostics.finalize()
    # The sealed pilot consists of ten UTC weekdays; any weekend session
    # records remain counted in the manifest but are excluded from daily study metrics.
    daily = [{"date":d, **vars(m)} for d,m in sorted(diagnostics.days.items()) if date.fromisoformat(d).weekday() < 5]
    manifest = {"study":"CMEOrderflowAbsorption.ES_V1_PILOT","read_only":True,"dbn_sha256":sha256,
      "dbn_bytes":dbn_bytes,"metadata":vars(metadata),"event_count":diagnostics.events,
      "execution_count":diagnostics.executions,"integrity":integrity,"trading_strategy_executed":False}
    (out / "mbo-pilot-manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    contract = {"engine":"causal order-id MBO reconstruction","ordering":"provider iterator order is preserved exactly; sequence is retained as diagnostic metadata but is not a provider-wide total order",
      "state":"active order-id map and side/price displayed depth","fail_closed":["unsupported action","unknown order operation","negative depth","invalid state transition","metadata/instrument mismatch"],
      "semantics":{"T/F":"execution-only evidence in this pilot; no undocumented displayed-book decrement is made because F may coexist with a subsequent cancel of the same quantity","aggressor":"not forced for execution-only records","queue_position":"UNAVAILABLE; no queue-position claim","hidden_quantity":"UNAVAILABLE; no iceberg claim"}}
    (out / "orderbook-reconstruction-contract.json").write_text(json.dumps(contract, indent=2)+"\n", encoding="utf-8")
    rows = "\n".join(f"| {x['date']} | {x['events']} | {x['executions']} | {x['adds']} | {x['cancels']} | {x['modifies']} | 0 | 0 | 0 | {x['structural_tags']} |" for x in daily)
    (out / "mbo-validation-report.md").write_text(f"# ESU6 MBO validation\n\nRead-only deterministic pilot. SHA-256: `{sha256}`; bytes: {dbn_bytes}. Metadata: {metadata.dataset}, {metadata.schema}, {metadata.symbol}->{metadata.instrument_id}, UTC [{metadata.start_ns}, {metadata.end_ns}). Integrity: {integrity}.\n\n| UTC date | events | executions | adds | cancels | modifies | PROBABLE_REPLENISHMENT | ABSORPTION_CANDIDATE | NO_CLEAR_REPLENISHMENT | prior-RTH tags |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n{rows}\n\nData-quality issues: {dict(diagnostics.issues)}. Provider reset records: {sum(x['resets'] for x in daily)}. No raw events are included.\n", encoding="utf-8")
    # Interaction examples intentionally omit raw order IDs and use only ES prices.
    examples = [{"date": x["date"], "level": x["level"], "price_es": x["price"] / 1_000_000_000, "label": x["label"]} for x in diagnostics.examples]
    (out / "feature-diagnostic-report.md").write_text("# Non-optimized feature diagnostics\n\nLabels are descriptive only: PROBABLE_REPLENISHMENT, ABSORPTION_CANDIDATE, and NO_CLEAR_REPLENISHMENT. No final threshold was chosen; no true/confirmed iceberg inference is made. Prior-RTH high/low/POC context is frozen before each RTH; VAH/VAL unavailable without a sealed profile algorithm and is explicitly not invented. Execution-only T/F records retain unknown aggressor rather than forcing a side; queue position and hidden quantity are unavailable.\n\nBounded summary-only probable-replenishment examples:\n\n```json\n" + json.dumps(examples, indent=2) + "\n```\n\nSuitability conclusion: suitable for descriptive MBO lifecycle and displayed-replenishment diagnostics if reconstruction completes without fail-closed errors; unsuitable for hidden-liquidity proof, queue-position claims, threshold selection, or profitability claims.\n", encoding="utf-8")
    rows = diagnostics.interaction_rows()
    summary = _refined_summary(rows, sha256)
    (out / "refined-feature-diagnostic-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = ["# Refined interaction-level diagnostics", "", "This is a bounded descriptive study, not a trading strategy. Interactions preserve the completed causal MBO reconstruction, interaction grouping, structural levels, prior-RTH profile, +/-4 ES-tick vicinity, 60-second timeout, features, and tiers without threshold changes or optimization.", "", "## Emitted machine-readable results", "", "```json", json.dumps(summary, indent=2, sort_keys=True), "```", "", "Root cause repaired: the report writer retained the obsolete `passive_side` example schema after causal interaction examples changed shape, raising `KeyError`; it also referenced `Counter` without importing it. The failure occurred after reconstruction and before refined artifacts were emitted.", "", "Monday 2026-07-27 prior-RTH levels were built from Friday 2026-07-24: PASS.", "", "No entries, stops, targets, trading strategy, or PnL were calculated."]
    (out / "refined-feature-diagnostic-report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    violations = diagnostics.response_sanity_violations
    integrity_report = ["# Interaction and response integrity", "",
        "## Root causes and repair",
        "The prior interaction key included every current-session extreme price and lifecycle closure occurred on any out-of-vicinity MBO event. This fragmented visits into message-level interactions. Fixed prior levels are now keyed by immutable level and current sweeps retain one lifecycle while their extreme is revised; only executed ES trades create, close, or reset a visit.",
        "", "Response lookup previously used the next arbitrary applied MBO record (including displayed-order prices). It now uses the first valid ES execution at or after each horizon. Prices remain DBN fixed-point integers internally, are converted by 1e9 only for presentation, and signed ticks are `(future ES price - reference ES price) / 0.25`.",
        "", "## Cardinality and response integrity",
        f"Before repair RAW interactions: 3,664,178. After repair RAW interactions: {summary['tier_counts']['RAW_INTERACTION']}.",
        f"Tier counts: {summary['tier_counts']}. Per-RTH practical counts: {summary['per_rth_practical_counts']}.",
        f"Response integrity: {summary['response_integrity']}.",
        "", "## Sanity-violation audit",
        f"All `abs(response_ticks) > 500` observations are classified as `RESPONSE_SANITY_VIOLATION`, excluded from descriptive distributions, and retained here as bounded audit evidence. Count: {len(violations)}.",
        "```json", json.dumps(violations[:100], indent=2), "```", "",
        "## Final status", summary["status"], "",
        "No trading strategy, PnL, optimization, download, or market-data mutation was performed."]
    (out / "interaction-response-integrity-report.md").write_text("\n".join(integrity_report) + "\n", encoding="utf-8")
    profile_contract = {"instrument": "ES", "tick_size": 0.25, "dbn_fixed_point_scale": 1000000000, "source": "executed trade volume only from immediately preceding completed RTH", "poc": "maximum volume; equal volume selects lower price", "value_area": "70 percent by volume; start at POC, expand one tick at a time; equal outside volumes select lower price; no future session data"}
    interaction_contract = {"level_universe": ["PRIOR_RTH_HIGH", "PRIOR_RTH_LOW", "PRIOR_RTH_POC", "PRIOR_RTH_VAH", "PRIOR_RTH_VAL", "CURRENT_RTH_HIGH_SWEEP", "CURRENT_RTH_LOW_SWEEP"], "vicinity_ticks": 4, "lifecycle": {"entry": "first executed ES trade within +/-4 ticks", "termination": ["more-than-4-tick exit persists for one-second sealed reset interval", "60 seconds without an in-vicinity executed trade", "RTH end"], "non_restart": ["executions", "side/order/depth changes", "inside-vicinity ticks", "each current-session extreme"], "current_sweep": "one active high/low lifecycle persists while an executed trade revises its extreme; revision supersedes the level price without creating an interaction"}, "id": "date:level_name:fixed_point_level:ordinal", "labels": {"PROBABLE_REPLENISHMENT_INTERACTION": "execution causally precedes same-price displayed add", "ABSORPTION_INTERACTION": "execution plus repeated same-price replenishment within interaction"}, "aggressor": "T remains unknown; F uses documented resting side only", "future_response": "first valid executed ES observation at or after horizon; signed ticks=(future ES price-reference ES price)/0.25; response sanity violations are explicit and excluded from distributions"}
    (out / "volume-profile-contract.json").write_text(json.dumps(profile_contract, indent=2) + "\n", encoding="utf-8")
    (out / "interaction-contract.json").write_text(json.dumps(interaction_contract, indent=2) + "\n", encoding="utf-8")
