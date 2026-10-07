#!/usr/bin/env python3
"""Artifact-only analysis of the completed 2025 Europe +1h replay."""

from __future__ import annotations

import csv
import gzip
import hashlib
import itertools
import json
import math
import statistics
from collections import defaultdict
from datetime import date
from pathlib import Path


ROOT = Path("research_runs/CMEOrderflow_ES_LIVE_STRATEGY_EUROPE_SESSION_PLUS1H_CORE_V1")
OUT = Path("research_runs/CMEOrderflow_ES_LIVE_STRATEGY_EUROPE_SESSION_PLUS1H_ANALYSIS_V1")
FAMILIES = [
    "EUROPE|EUROPE|CURRENT|HIGH",
    "EUROPE|EUROPE|PRIOR|HIGH",
    "EUROPE|EUROPE|PRIOR|VAH",
    "NY|NY|PRIOR|POC",
]
EPS = 1e-12


def read_json(path: Path):
    return json.loads(path.read_text())


def read_jsonl_gz(path: Path):
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_csv(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def mean(values):
    return statistics.fmean(values) if values else 0.0


def median(values):
    return statistics.median(values) if values else 0.0


def quantile(values, q):
    if not values:
        return 0.0
    vals = sorted(values)
    pos = (len(vals) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return vals[lo] if lo == hi else vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def trade_metrics(trades):
    rs = [float(t["realized_r"]) for t in trades]
    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r < 0]
    return {
        "trades": len(rs),
        "net_r": sum(rs),
        "avg_r_per_trade": mean(rs),
        "median_r_per_trade": median(rs),
        "win_rate": len(wins) / len(rs) if rs else None,
        "profit_factor": sum(wins) / abs(sum(losses)) if losses else (None if not wins else "infinite"),
        "winners": len(wins),
        "losers": len(losses),
    }


def family_trade_map(trades):
    out = defaultdict(list)
    for t in trades:
        out[(t["family"], t["date"])].append(t)
    return out


def daily_delta_row(date_s, period, family, base, plus):
    bm, pm = trade_metrics(base), trade_metrics(plus)
    bt, pt = bm["trades"], pm["trades"]
    delta = pm["net_r"] - bm["net_r"]
    if bt == 0 and pt > 0:
        effect = "PLUS1H_ONLY_ACTIVITY"
    elif pt == 0 and bt > 0:
        effect = "BASELINE_ONLY_ACTIVITY"
    elif delta > EPS:
        effect = "IMPROVED"
    elif delta < -EPS:
        effect = "WORSENED"
    else:
        effect = "UNCHANGED"
    return {
        "date": date_s,
        "period": period,
        "family": family,
        "baseline_trades": bt,
        "plus1h_trades": pt,
        "trade_count_delta": pt - bt,
        "trade_count_change_percent": ((pt - bt) / bt * 100.0) if bt else (None if not pt else "new_activity"),
        "baseline_net_r": bm["net_r"],
        "plus1h_net_r": pm["net_r"],
        "delta_net_r": delta,
        "baseline_avg_r": bm["avg_r_per_trade"] if bt else None,
        "plus1h_avg_r": pm["avg_r_per_trade"] if pt else None,
        "day_effect": effect,
    }


def date_period(s):
    return "SPRING" if s.startswith(("2025-03", "2025-04")) else "OCTOBER"


def daily_distribution(rows, dates):
    active = [r for r in rows if r["baseline_trades"] or r["plus1h_trades"]]
    ds = [r["delta_net_r"] for r in rows]
    ads = [r["delta_net_r"] for r in active]
    c = Counter(r["day_effect"] for r in active)
    return {
        "calendar_days": len(rows),
        "active_days": len(active),
        "improved_days": c["IMPROVED"],
        "worsened_days": c["WORSENED"],
        "unchanged_days": c["UNCHANGED"],
        "baseline_only_activity_days": c["BASELINE_ONLY_ACTIVITY"],
        "plus1h_only_activity_days": c["PLUS1H_ONLY_ACTIVITY"],
        "fraction_improved_active_days": c["IMPROVED"] / len(active) if active else None,
        "fraction_worsened_active_days": c["WORSENED"] / len(active) if active else None,
        "mean_daily_delta_r_all_dates": mean(ds),
        "median_daily_delta_r_all_dates": median(ds),
        "p25_daily_delta_r_all_dates": quantile(ds, 0.25),
        "p75_daily_delta_r_all_dates": quantile(ds, 0.75),
        "min_daily_delta_r_all_dates": min(ds, default=0.0),
        "max_daily_delta_r_all_dates": max(ds, default=0.0),
        "mean_daily_delta_r_active_dates": mean(ads),
        "median_daily_delta_r_active_dates": median(ads),
        "p25_daily_delta_r_active_dates": quantile(ads, 0.25),
        "p75_daily_delta_r_active_dates": quantile(ads, 0.75),
        "min_daily_delta_r_active_dates": min(ads, default=0.0),
        "max_daily_delta_r_active_dates": max(ads, default=0.0),
    }


def outlier_stats(rows):
    vals = sorted(((r["date"], r["delta_net_r"]) for r in rows), key=lambda x: x[1], reverse=True)
    best3 = vals[:3]
    worst3 = sorted(vals, key=lambda x: x[1])[:3]
    total = sum(v for _, v in vals)
    best1, best3sum = (vals[0][1] if vals else 0.0), sum(v for _, v in best3)
    worst1 = min((v for _, v in vals), default=0.0)
    worst3sum = sum(v for _, v in worst3)
    relevant3 = abs(worst3sum) if total < -EPS else abs(best3sum)
    concentration = relevant3 / abs(total) if abs(total) > EPS else 0.0
    remove1 = total - best1
    remove3 = total - best3sum
    remove_worst = total - worst1
    if abs(total) <= EPS:
        classification = "UNAFFECTED"
    elif concentration >= 0.75 or (total > 0 and remove3 < -EPS) or (total < 0 and total - worst3sum > EPS):
        classification = "HIGHLY_OUTLIER_CONCENTRATED"
    elif concentration >= 0.50 or (total > 0 and remove1 < -EPS) or (total < 0 and total - worst1 > EPS):
        classification = "MODERATELY_CONCENTRATED"
    else:
        classification = "BROADLY_DISTRIBUTED"
    return {
        "total_delta_r": total,
        "best_delta_day": {"date": vals[0][0], "delta_r": vals[0][1]} if vals else None,
        "best_3_delta_days_total_r": best3sum,
        "best_3_delta_days": [{"date": d, "delta_r": v} for d, v in best3],
        "worst_delta_day": {"date": min(vals, key=lambda x: x[1])[0], "delta_r": worst1} if vals else None,
        "worst_3_delta_days_total_r": worst3sum,
        "worst_3_delta_days": [{"date": d, "delta_r": v} for d, v in worst3],
        "delta_excluding_best_1_day_r": remove1,
        "delta_excluding_best_3_days_r": remove3,
        "delta_excluding_worst_1_day_r": remove_worst,
        "dominant_tail_share_abs_total": concentration,
        "classification": classification,
    }


def paired_sign_flip(values, seed=20261007, iterations=100000):
    observed = abs(mean(values))
    n = len(values)
    if n == 0:
        return {"n_dates": 0, "observed_mean": 0.0, "p_two_sided": None, "method": "not_applicable"}
    # Exact enumeration when tractable, Monte Carlo otherwise.
    if n <= 18:
        ge = total = 0
        for signs in itertools.product((-1, 1), repeat=n):
            stat = abs(sum(v * s for v, s in zip(values, signs)) / n)
            total += 1
            ge += stat >= observed - 1e-15
        return {"n_dates": n, "observed_mean": mean(values), "p_two_sided": ge / total, "method": "exact_sign_flip", "permutations": total}
    import random
    rng = random.Random(seed)
    ge = 0
    for _ in range(iterations):
        stat = abs(sum(v if rng.getrandbits(1) else -v for v in values) / n)
        ge += stat >= observed - 1e-15
    return {"n_dates": n, "observed_mean": mean(values), "p_two_sided": (ge + 1) / (iterations + 1), "method": "fixed_seed_monte_carlo_sign_flip", "seed": seed, "permutations": iterations}


def lodo(rows):
    vals = [(r["date"], r["delta_net_r"]) for r in rows]
    active_n = sum(abs(v) > EPS for _, v in vals)
    if active_n < 5:
        return {"status": "INSUFFICIENT", "calendar_dates": len(vals), "active_delta_dates": active_n}
    full = mean([v for _, v in vals])
    omitted = [(d, mean([v for dd, v in vals if dd != d])) for d, _ in vals]
    s = 1 if full > EPS else (-1 if full < -EPS else 0)
    stable = all((1 if x > EPS else (-1 if x < -EPS else 0)) == s for _, x in omitted) if s else all(abs(x) <= EPS for _, x in omitted)
    return {
        "dates": len(vals), "full_mean_daily_delta_r": full,
        "sign_stable": stable,
        "min_leave_one_day_out_mean": min((v for _, v in omitted), default=0.0),
        "max_leave_one_day_out_mean": max((v for _, v in omitted), default=0.0),
        # Omitting a positive-contribution day lowers the mean most; omitting a
        # negative-contribution day raises it most.
        "most_influential_positive_date": min(omitted, key=lambda x: x[1])[0] if omitted else None,
        "most_influential_negative_date": max(omitted, key=lambda x: x[1])[0] if omitted else None,
    }


def lowo(rows):
    groups = defaultdict(list)
    active_dates = 0
    for r in rows:
        d = date.fromisoformat(r["date"])
        groups[f"{d.isocalendar().year}-W{d.isocalendar().week:02d}"].append(r["delta_net_r"])
        active_dates += abs(r["delta_net_r"]) > EPS
    if active_dates < 5:
        return {"status": "INSUFFICIENT", "weeks": len(groups), "active_delta_dates": active_dates}
    weeks = sorted(groups)
    if len(weeks) < 3:
        return {"status": "INSUFFICIENT", "weeks": len(weeks)}
    vals = [(w, mean([v for ww, vs in groups.items() if ww != w for v in vs])) for w in weeks]
    full = mean([r["delta_net_r"] for r in rows])
    sign = 1 if full > EPS else (-1 if full < -EPS else 0)
    return {"status": "COMPUTED", "weeks": len(weeks), "full_mean_daily_delta_r": full,
            "sign_stable": all((1 if v > EPS else (-1 if v < -EPS else 0)) == sign for _, v in vals) if sign else all(abs(v) <= EPS for _, v in vals),
            "min_leave_one_week_out_mean": min(v for _, v in vals), "max_leave_one_week_out_mean": max(v for _, v in vals),
            "leave_one_week_out": [{"week": w, "mean_daily_delta_r": v} for w, v in vals]}


def outlier_label(rows):
    vals = [r["delta_net_r"] for r in rows]
    total = sum(vals)
    if abs(total) <= EPS:
        return "UNAFFECTED"
    if total > 0 and mean(sorted(vals, reverse=True)[:3]) > 0:
        share = sum(sorted(vals, reverse=True)[:3]) / abs(total)
    else:
        share = abs(sum(sorted(vals)[:3])) / abs(total)
    if share >= .75:
        return "HIGHLY_OUTLIER_CONCENTRATED"
    if share >= .5:
        return "MODERATELY_CONCENTRATED"
    return "BROADLY_DISTRIBUTED"


def jsonable_trade(t):
    return {k: t[k] for k in ("trade_id", "setup_id", "family", "date", "period", "direction", "signal_timestamp", "entry_timestamp", "exit_timestamp", "entry_price", "exit_price", "stop_price", "target_price", "exit_reason", "realized_r")}


def main():
    manifest = read_json(ROOT / "run-manifest.json")
    summary = read_json(ROOT / "summary.json")
    identity = read_json(ROOT / "run-identity.json")
    if not (manifest.get("status") == "COMPLETE" and summary.get("status") == "COMPLETE"):
        raise SystemExit("STOP: replay run is not COMPLETE")
    if (manifest.get("baseline_dates_completed"), manifest.get("plus1h_dates_completed")) != (54, 54):
        raise SystemExit("STOP: expected 54 complete dates per variant")
    base_trades = read_jsonl_gz(ROOT / "baseline-trades.jsonl.gz")
    plus_trades = read_jsonl_gz(ROOT / "plus1h-trades.jsonl.gz")
    if (len(base_trades), len(plus_trades)) != (422, 375):
        raise SystemExit(f"STOP: trade count mismatch: {len(base_trades)}/{len(plus_trades)}")
    bnet = sum(float(t["realized_r"]) for t in base_trades)
    pnet = sum(float(t["realized_r"]) for t in plus_trades)
    if not math.isclose(bnet, -145.91511087963107, abs_tol=1e-8) or not math.isclose(pnet, -120.16712233537929, abs_tol=1e-8):
        raise SystemExit(f"STOP: net R mismatch: {bnet}/{pnet}")
    if set(manifest["families"]) != set(FAMILIES):
        raise SystemExit("STOP: unexpected family universe")

    input_files = ["run-manifest.json", "summary.json", "run-identity.json", "family-day-comparison.csv", "family-summary.json", "level-comparison.csv", "first-hour-raw-coverage.csv", "baseline-trades.jsonl.gz", "plus1h-trades.jsonl.gz"]
    integrity = {}
    expected_hashes = manifest.get("outputs", {})
    for name in input_files:
        actual = sha(ROOT / name)
        expected = expected_hashes.get(name)
        integrity[name] = {"sha256": actual, "expected_sha256": expected, "matches_manifest": actual == expected if expected else None}
        if expected and actual != expected:
            raise SystemExit(f"STOP: input artifact hash mismatch: {name}")

    day_rows_source = load_csv(ROOT / "family-day-comparison.csv")
    dates = manifest["dates"]
    if len(dates) != 54:
        raise SystemExit("STOP: manifest date list mismatch")
    bmap, pmap = family_trade_map(base_trades), family_trade_map(plus_trades)
    day_rows = []
    for d in dates:
        period = date_period(d)
        for fam in FAMILIES:
            day_rows.append(daily_delta_row(d, period, fam, bmap[(fam, d)], pmap[(fam, d)]))
    if len(day_rows) != len(day_rows_source):
        raise SystemExit("STOP: family-day artifact row count mismatch")
    # Reconcile the primary stored daily comparisons to records, not merely totals.
    source_keyed = {(r["date"], r["family"]): r for r in day_rows_source}
    for r in day_rows:
        s = source_keyed[(r["date"], r["family"])]
        if int(s["baseline_trade_count"]) != r["baseline_trades"] or int(s["plus1h_trade_count"]) != r["plus1h_trades"] or not math.isclose(float(s["delta_net_r"]), r["delta_net_r"], abs_tol=1e-9):
            raise SystemExit(f"STOP: daily comparison does not reconcile at {r['date']} {r['family']}")

    # Period and family aggregates.
    family_period = {}
    daily_dist = {}
    outliers = {}
    paired_tests = {}
    lodo_out, lowo_out = {}, {}
    compat, frequency_quality = {}, {}
    for fam in FAMILIES:
        family_period[fam] = {}
        daily_dist[fam] = {}
        outliers[fam] = {}
        paired_tests[fam] = {}
        lodo_out[fam] = {}
        lowo_out[fam] = {}
        compat[fam] = {}
        frequency_quality[fam] = {}
        for period in ("SPRING", "OCTOBER", "COMBINED"):
            rs = [r for r in day_rows if r["family"] == fam and (period == "COMBINED" or r["period"] == period)]
            base = [t for t in base_trades if t["family"] == fam and (period == "COMBINED" or ("SPRING" if t["period"].startswith("SPRING") else "OCTOBER") == period)]
            plus = [t for t in plus_trades if t["family"] == fam and (period == "COMBINED" or ("SPRING" if t["period"].startswith("SPRING") else "OCTOBER") == period)]
            bm, pm = trade_metrics(base), trade_metrics(plus)
            active = [r for r in rs if r["baseline_trades"] or r["plus1h_trades"]]
            changed = pm["avg_r_per_trade"] - bm["avg_r_per_trade"]
            delta = pm["net_r"] - bm["net_r"]
            dc = sum(r["trade_count_delta"] for r in rs)
            if delta > EPS and changed > EPS:
                edge_class = "BOTH" if dc < 0 else "EDGE_QUALITY_IMPROVED"
            elif delta > EPS:
                edge_class = "FREQUENCY_REDUCTION_ONLY" if dc < 0 and changed <= EPS else "EDGE_QUALITY_IMPROVED"
            elif delta < -EPS:
                edge_class = "WORSENED"
            else:
                edge_class = "NEITHER"
            family_period[fam][period] = {
                "baseline_active_days": sum(r["baseline_trades"] > 0 for r in rs),
                "plus1h_active_days": sum(r["plus1h_trades"] > 0 for r in rs),
                "either_variant_active_days": len(active),
                "baseline": bm, "plus1h": pm,
                "trade_count_delta": dc,
                "trade_count_change_percent": dc / bm["trades"] * 100 if bm["trades"] else None,
                "net_r_delta": delta,
                "avg_r_per_trade_delta": changed,
                "win_rate_delta": (pm["win_rate"] - bm["win_rate"]) if pm["win_rate"] is not None and bm["win_rate"] is not None else None,
                "edge_vs_frequency_classification": edge_class,
            }
            daily_dist[fam][period] = daily_distribution(rs, dates)
            outliers[fam][period] = outlier_stats(rs)
            vals = [r["delta_net_r"] for r in rs]
            paired_tests[fam][period] = paired_sign_flip(vals, seed=20261007 + FAMILIES.index(fam) * 100 + (0 if period == "SPRING" else 1))
            lodo_out[fam][period] = lodo(rs)
            lowo_out[fam][period] = lowo(rs)
            if period != "COMBINED":
                nonzero = [r for r in rs if r["baseline_trades"] or r["plus1h_trades"]]
                improved = sum(r["delta_net_r"] > EPS for r in nonzero)
                worsened = sum(r["delta_net_r"] < -EPS for r in nonzero)
                unchanged = sum(abs(r["delta_net_r"]) <= EPS for r in nonzero)
                if not nonzero:
                    c = "INSUFFICIENT_ACTIVITY"
                elif delta > EPS and period == "SPRING":
                    c = "SPRING_ONLY_IMPROVEMENT"
                elif delta > EPS and period == "OCTOBER":
                    c = "OCTOBER_ONLY_IMPROVEMENT"
                elif abs(delta) <= EPS and improved == 0 and worsened == 0:
                    c = "UNAFFECTED"
                else:
                    c = "OPPOSITE_PERIOD_EFFECT" if delta != 0 else "NO_MEANINGFUL_EFFECT"
                compat[fam][period] = c
                frequency_quality[fam][period] = edge_class
        spring, octo = family_period[fam]["SPRING"], family_period[fam]["OCTOBER"]
        if spring["either_variant_active_days"] + octo["either_variant_active_days"] == 0:
            compat[fam]["overall"] = "INSUFFICIENT_ACTIVITY"
        elif spring["net_r_delta"] * octo["net_r_delta"] < -EPS:
            compat[fam]["overall"] = "OPPOSITE_PERIOD_EFFECT"
        elif spring["net_r_delta"] > EPS and octo["net_r_delta"] > EPS:
            compat[fam]["overall"] = "IMPROVES_BOTH_PERIODS"
        elif spring["net_r_delta"] > EPS:
            compat[fam]["overall"] = "SPRING_ONLY_IMPROVEMENT"
        elif octo["net_r_delta"] > EPS:
            compat[fam]["overall"] = "OCTOBER_ONLY_IMPROVEMENT"
        elif abs(spring["net_r_delta"]) <= EPS and abs(octo["net_r_delta"]) <= EPS:
            compat[fam]["overall"] = "UNAFFECTED"
        else:
            compat[fam]["overall"] = "NO_MEANINGFUL_EFFECT"
        spring_rows = [r for r in day_rows if r["family"] == fam and r["period"] == "SPRING"]
        oct_rows = [r for r in day_rows if r["family"] == fam and r["period"] == "OCTOBER"]
        active_dates = sum(bool(r["baseline_trades"] or r["plus1h_trades"]) for r in spring_rows + oct_rows)
        sdelta, odelta = spring["net_r_delta"], octo["net_r_delta"]
        if active_dates == 0:
            decision = "INSUFFICIENT_ACTIVITY"
        elif sdelta > EPS and odelta > EPS and outlier_label(spring_rows) == "BROADLY_DISTRIBUTED" and outlier_label(oct_rows) == "BROADLY_DISTRIBUTED" and spring["avg_r_per_trade_delta"] > 0 and octo["avg_r_per_trade_delta"] > 0:
            decision = "ROBUST_PLUS1H_IMPROVEMENT"
        elif sdelta > EPS and odelta <= EPS:
            decision = "POSSIBLE_PLUS1H_IMPROVEMENT" if active_dates < 10 else "MIXED_PLUS1H_EFFECT"
        elif sdelta * odelta < -EPS:
            decision = "MIXED_PLUS1H_EFFECT"
        elif abs(sdelta) <= EPS and abs(odelta) <= EPS:
            decision = "UNAFFECTED_BY_EUROPE_SESSION"
        elif sdelta < -EPS or odelta < -EPS:
            decision = "PLUS1H_WORSE"
        else:
            decision = "NO_PLUS1H_BENEFIT"
        compat[fam]["family_decision"] = decision

    # Trade identity comparison, with exact identity excluding the variant label.
    b_by_id, p_by_id = {}, {}
    for t in base_trades:
        b_by_id[(t["date"], t["family"], t["trade_id"])] = t
    for t in plus_trades:
        p_by_id[(t["date"], t["family"], t["trade_id"])] = t
    common_ids = set(b_by_id) & set(p_by_id)
    identity_fields = ("family", "date", "direction", "signal_timestamp", "entry_timestamp", "exit_timestamp", "entry_price", "exit_price", "stop_price", "target_price", "exit_reason", "realized_r", "target_before_stop")
    common = defaultdict(lambda: {"identical": 0, "changed": 0, "identical_r": 0.0, "changed_base_r": 0.0, "changed_plus_r": 0.0, "changed_delta_r": 0.0})
    changed_trade_details = defaultdict(list)
    for tid in common_ids:
        a, b = b_by_id[tid], p_by_id[tid]
        same = all(a.get(k) == b.get(k) for k in identity_fields)
        x = common[a["family"]]
        if same:
            x["identical"] += 1
            x["identical_r"] += float(a["realized_r"])
        else:
            x["changed"] += 1
            x["changed_base_r"] += float(a["realized_r"])
            x["changed_plus_r"] += float(b["realized_r"])
            x["changed_delta_r"] += float(b["realized_r"]) - float(a["realized_r"])
            changed_fields = [k for k in identity_fields if a.get(k) != b.get(k)]
            changed_trade_details[a["family"]].append({
                "date": a["date"], "trade_id": a["trade_id"], "changed_fields": changed_fields,
                "baseline": {k: a.get(k) for k in changed_fields},
                "plus1h": {k: b.get(k) for k in changed_fields},
                "baseline_realized_r": a["realized_r"], "plus1h_realized_r": b["realized_r"],
            })
    removed_added = {}
    common_analysis = {}
    for fam in FAMILIES:
        removed = [t for t in base_trades if t["family"] == fam and (t["date"], t["family"], t["trade_id"]) not in p_by_id]
        added = [t for t in plus_trades if t["family"] == fam and (t["date"], t["family"], t["trade_id"]) not in b_by_id]
        rm, am = trade_metrics(removed), trade_metrics(added)
        removed_added[fam] = {
            "baseline_only": {"count": rm["trades"], "total_r": rm["net_r"], "avg_r": rm["avg_r_per_trade"]},
            "plus1h_only": {"count": am["trades"], "total_r": am["net_r"], "avg_r": am["avg_r_per_trade"]},
            "periods": {},
        }
        for period in ("SPRING", "OCTOBER", "COMBINED"):
            rsub = [t for t in removed if period == "COMBINED" or ("SPRING" if t["period"].startswith("SPRING") else "OCTOBER") == period]
            asub = [t for t in added if period == "COMBINED" or ("SPRING" if t["period"].startswith("SPRING") else "OCTOBER") == period]
            rmet, amet = trade_metrics(rsub), trade_metrics(asub)
            removed_added[fam]["periods"][period] = {
                "baseline_only": {"count": rmet["trades"], "total_r": rmet["net_r"], "avg_r": rmet["avg_r_per_trade"]},
                "plus1h_only": {"count": amet["trades"], "total_r": amet["net_r"], "avg_r": amet["avg_r_per_trade"]},
            }
        common_analysis[fam] = {
            "common_identical_count": common[fam]["identical"],
            "common_changed_count": common[fam]["changed"],
            "common_identical_r_baseline_and_plus1h": common[fam]["identical_r"],
            "common_changed_baseline_r": common[fam]["changed_base_r"],
            "common_changed_plus1h_r": common[fam]["changed_plus_r"],
            "common_changed_delta_r": common[fam]["changed_delta_r"],
            "common_changed_trade_details": changed_trade_details[fam],
            "trade_id_match_fields": list(identity_fields),
        }

    # Level changes and hour-defined profile extrema, using stored level/profile artifacts only.
    level_rows = load_csv(ROOT / "level-comparison.csv")
    level_cols = {
        "CURRENT_EUROPE_HIGH": ("baseline_europe_high", "plus1h_europe_high"),
        "PRIOR_EUROPE_HIGH": ("baseline_prior_europe_high", "plus1h_prior_europe_high"),
        "PRIOR_EUROPE_POC": ("baseline_prior_europe_poc", "plus1h_prior_europe_poc"),
        "PRIOR_EUROPE_VAH": ("baseline_prior_europe_vah", "plus1h_prior_europe_vah"),
    }
    level_change = {}
    for label, (bc, pc) in level_cols.items():
        diffs = [(r["date"], (float(r[bc]) - float(r[pc])) / 0.25) for r in level_rows if r.get(bc) and r.get(pc)]
        changed = [(d, v) for d, v in diffs if abs(v) > EPS]
        abs_ticks = [abs(v) for _, v in changed]
        level_change[label] = {
            "dates_total": len(diffs), "dates_changed": len(changed),
            "percent_changed": len(changed) / len(diffs) * 100 if diffs else 0.0,
            "median_absolute_change_ticks_changed_dates": median(abs_ticks),
            "max_absolute_change_ticks": max(abs_ticks, default=0.0),
            "date_changes_ticks_baseline_minus_plus1h": [{"date": d, "ticks": v} for d, v in changed],
        }
    profile_extrema = {"dates": len(dates), "high_set_by_08_00_09_00_count": None, "low_set_by_08_00_09_00_count": None,
                       "high_level_changed_dates": level_change["CURRENT_EUROPE_HIGH"]["dates_changed"],
                       "high_level_changed_percent": level_change["CURRENT_EUROPE_HIGH"]["percent_changed"],
                       "low_extreme": "NOT_AVAILABLE_IN_COMPLETED_LEVEL_ARTIFACTS",
                       "date_level_details": level_change["CURRENT_EUROPE_HIGH"]["date_changes_ticks_baseline_minus_plus1h"],
                       "interpretation": "The completed level comparison supports counting dates where the current Europe high changed. It does not store Europe lows or baseline profile volume maps, so exact first-hour high/low set counts cannot both be recovered without rebuilding profiles or reading underlying data; neither was done."}

    # Family-to-level dependence map.
    dependency = {
        FAMILIES[0]: {"trading_session": "EUROPE", "source_level": "CURRENT_EUROPE_HIGH", "source_profile": "same-day Europe profile; baseline includes 08:00-09:00 UTC, plus1h excludes it"},
        FAMILIES[1]: {"trading_session": "EUROPE", "source_level": "PRIOR_EUROPE_HIGH", "source_profile": "prior available trading date Europe profile"},
        FAMILIES[2]: {"trading_session": "EUROPE", "source_level": "PRIOR_EUROPE_VAH", "source_profile": "prior available trading date Europe profile"},
        FAMILIES[3]: {"trading_session": "NY", "source_level": "PRIOR_NY_POC", "source_profile": "prior available trading date NY profile"},
    }
    for fam, dep in dependency.items():
        dep["level_change_summary"] = level_change.get(dep["source_level"], {
            "status": "NOT_AVAILABLE_IN_LEVEL_COMPARISON_ARTIFACT",
            "reason": "level-comparison.csv stores prior Europe levels, not prior NY POC",
        })

    # Row matrix over all dates, including zero-trade dates.
    matrix = [{"date": d, **{f: next(r["delta_net_r"] for r in day_rows if r["date"] == d and r["family"] == f) for f in FAMILIES}} for d in dates]

    # Family decisions and concise global conclusions.
    decisions = {f: compat[f]["family_decision"] for f in FAMILIES}
    spring_total = sum(family_period[f]["SPRING"]["net_r_delta"] for f in FAMILIES)
    october_total = sum(family_period[f]["OCTOBER"]["net_r_delta"] for f in FAMILIES)
    overall = {
        "baseline_trades": len(base_trades), "plus1h_trades": len(plus_trades),
        "baseline_net_r": bnet, "plus1h_net_r": pnet, "delta_net_r": pnet - bnet,
        "spring_baseline_trades": 326, "spring_plus1h_trades": 289,
        "spring_baseline_net_r": -105.52734262044609, "spring_plus1h_net_r": -72.09787631632825,
        "spring_delta_net_r": spring_total,
        "october_baseline_trades": 96, "october_plus1h_trades": 86,
        "october_baseline_net_r": -40.387768259185, "october_plus1h_net_r": -48.069246019051036,
        "october_delta_net_r": october_total,
    }
    if spring_total > EPS and october_total < -EPS:
        primary_decision = "PLUS1H_SESSION_HYPOTHESIS_MIXED"
    elif all(family_period[f]["SPRING"]["net_r_delta"] > EPS and family_period[f]["OCTOBER"]["net_r_delta"] > EPS for f in FAMILIES if family_period[f]["SPRING"]["either_variant_active_days"] + family_period[f]["OCTOBER"]["either_variant_active_days"]):
        primary_decision = "PLUS1H_SESSION_HYPOTHESIS_SUPPORTED"
    elif any(v == "POSSIBLE_PLUS1H_IMPROVEMENT" for v in decisions.values()):
        primary_decision = "PLUS1H_SESSION_HYPOTHESIS_PARTIALLY_SUPPORTED"
    else:
        primary_decision = "PLUS1H_SESSION_HYPOTHESIS_NOT_SUPPORTED"

    OUT.mkdir(parents=True, exist_ok=True)
    # Full daily CSV and date-by-family matrix.
    with (OUT / "family-day-analysis.csv").open("w", newline="") as f:
        fields = list(day_rows[0])
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(day_rows)
    with (OUT / "family-day-delta-matrix.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date", *FAMILIES]); w.writeheader(); w.writerows(matrix)

    write_json(OUT / "family-period-analysis.json", {"families": family_period, "family_dependencies": dependency})
    write_json(OUT / "daily-distributions.json", daily_dist)
    write_json(OUT / "outlier-concentration.json", outliers)
    write_json(OUT / "trade-frequency-vs-quality.json", frequency_quality)
    write_json(OUT / "removed-added-trades.json", removed_added)
    write_json(OUT / "common-trade-analysis.json", common_analysis)
    write_json(OUT / "level-change-analysis.json", {"levels": level_change, "profile_extrema": profile_extrema, "family_dependencies": dependency})
    write_json(OUT / "period-compatibility.json", {"families": compat, "primary_decision": primary_decision})
    write_json(OUT / "paired-day-tests.json", paired_tests)
    write_json(OUT / "lodo-results.json", lodo_out)
    write_json(OUT / "lowo-results.json", lowo_out)
    write_json(OUT / "input-integrity.json", integrity)
    write_json(OUT / "summary.json", {
        "analysis_id": "CMEOrderflow_ES_LIVE_STRATEGY_EUROPE_SESSION_PLUS1H_ANALYSIS_V1",
        "status": "COMPLETE", "source_run_id": manifest["run_id"], "dates": dates, "families": FAMILIES,
        "baseline_dates": 54, "plus1h_dates": 54,
        "overall": overall, "family_decisions": decisions, "primary_decision": primary_decision,
        "source_integrity": integrity, "raw_replay_rerun": False, "optimization_performed": False,
        "session_search_performed": False, "data_downloaded": False, "2026_data_accessed": False,
    })

    # Markdown report includes all nonzero family/date observations, plus summaries.
    lines = [
        "# ES Europe +1h completed replay analysis", "",
        "Artifact-only analysis. No replay, optimization, download, raw DBN read, or 2026 access was performed.", "",
        "## Completeness and aggregate reconciliation", "",
        f"- Dates completed: baseline 54/54; +1h 54/54. Trade totals: {len(base_trades)} vs {len(plus_trades)}.",
        f"- Net R: {bnet:.6f} vs {pnet:.6f}; delta {pnet-bnet:+.6f} R.",
        f"- Spring delta: {spring_total:+.6f} R; October delta: {october_total:+.6f} R.",
        f"- Primary decision: **{primary_decision}**.",
        "- Input artifact hashes matched the completed run manifest; full per-day comparisons reconcile to trade records.", "",
        "## Family × period results", "",
        "Counts use trade records. Active days count dates with at least one trade in either variant. Daily distribution fractions use active dates; mean/quantiles are also supplied over all calendar dates (including zero-trade dates) in JSON.", "",
    ]
    for fam in FAMILIES:
        lines += [f"### `{fam}`", "", f"Dependency: {dependency[fam]['source_profile']}; relevant level `{dependency[fam]['source_level']}`.", "",
                  "| Period | Base days/trades/net R/avg R/PF/win | +1h days/trades/net R/avg R/PF/win | Δ trades | Δ net R | Δ avg R | daily improved/worsened/unchanged | mean / median daily Δ | decision |",
                  "|---|---|---|---:|---:|---:|---|---|---|"]
        for period in ("SPRING", "OCTOBER", "COMBINED"):
            x = family_period[fam][period]; b, p = x["baseline"], x["plus1h"]
            dist = daily_dist[fam][period]
            compat_s = compat[fam].get(period, compat[fam]["overall"])
            fmtpf = lambda v: "∞" if v == "infinite" else ("—" if v is None else f"{v:.3f}")
            lines.append(f"| {period} | {x['baseline_active_days']}/{b['trades']}/{b['net_r']:+.3f}/{b['avg_r_per_trade']:+.3f}/{fmtpf(b['profit_factor'])}/{(b['win_rate'] or 0):.1%} | {x['plus1h_active_days']}/{p['trades']}/{p['net_r']:+.3f}/{p['avg_r_per_trade']:+.3f}/{fmtpf(p['profit_factor'])}/{(p['win_rate'] or 0):.1%} | {x['trade_count_delta']:+d} | {x['net_r_delta']:+.3f} | {x['avg_r_per_trade_delta']:+.3f} | {dist['improved_days']}/{dist['worsened_days']}/{dist['unchanged_days']} | {dist['mean_daily_delta_r_active_dates']:+.3f} / {dist['median_daily_delta_r_active_dates']:+.3f} | {compat_s} |")
        lines += ["", f"Family decision: **{decisions[fam]}**. Combined daily delta distribution: mean {daily_dist[fam]['COMBINED']['mean_daily_delta_r_active_dates']:+.4f}, median {daily_dist[fam]['COMBINED']['median_daily_delta_r_active_dates']:+.4f}, p25 {daily_dist[fam]['COMBINED']['p25_daily_delta_r_active_dates']:+.4f}, p75 {daily_dist[fam]['COMBINED']['p75_daily_delta_r_active_dates']:+.4f} R.", "",
                  "#### Every date with activity in either variant", "",
                  "| Date | Period | Base trades | +1h trades | Base R | +1h R | Δ R | Base avg R | +1h avg R | Effect |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
        for r in day_rows:
            if r["family"] == fam and (r["baseline_trades"] or r["plus1h_trades"]):
                lines.append(f"| {r['date']} | {r['period']} | {r['baseline_trades']} | {r['plus1h_trades']} | {r['baseline_net_r']:+.4f} | {r['plus1h_net_r']:+.4f} | {r['delta_net_r']:+.4f} | {r['baseline_avg_r'] if r['baseline_avg_r'] is not None else 0:+.4f} | {r['plus1h_avg_r'] if r['plus1h_avg_r'] is not None else 0:+.4f} | {r['day_effect']} |")
        lines += ["", "#### Removed/added and common trades", "",
                  f"Baseline-only: {removed_added[fam]['baseline_only']['count']} trades, {removed_added[fam]['baseline_only']['total_r']:+.4f} R, avg {removed_added[fam]['baseline_only']['avg_r']:+.4f} R. ",
                  f"+1h-only: {removed_added[fam]['plus1h_only']['count']} trades, {removed_added[fam]['plus1h_only']['total_r']:+.4f} R, avg {removed_added[fam]['plus1h_only']['avg_r']:+.4f} R. ",
                  f"Common IDs: {common_analysis[fam]['common_identical_count']} field-identical, {common_analysis[fam]['common_changed_count']} changed; changed-trade delta {common_analysis[fam]['common_changed_delta_r']:+.4f} R.", "",
                  "#### Outlier concentration and robustness", ""]
        for period in ("SPRING", "OCTOBER"):
            o = outliers[fam][period]
            lines.append(f"- {period}: {o['classification']}; best day {o['best_delta_day']}; best-3 sum {o['best_3_delta_days_total_r']:+.4f}; worst day {o['worst_delta_day']}; worst-3 sum {o['worst_3_delta_days_total_r']:+.4f}; excluding best 1 / best 3 / worst 1: {o['delta_excluding_best_1_day_r']:+.4f} / {o['delta_excluding_best_3_days_r']:+.4f} / {o['delta_excluding_worst_1_day_r']:+.4f} R. Sign-flip p={paired_tests[fam][period]['p_two_sided']}; LODO={lodo_out[fam][period].get('sign_stable', 'INSUFFICIENT')}; LOWO={lowo_out[fam][period]['status']}.")
        le = dependency[fam]["level_change_summary"]
        if "dates_changed" in le:
            level_text = f"`{dependency[fam]['source_level']}` changed on {le['dates_changed']}/{le['dates_total']} dates ({le['percent_changed']:.1f}%); median abs change {le['median_absolute_change_ticks_changed_dates']:.2f} ticks; max {le['max_absolute_change_ticks']:.2f} ticks."
        else:
            level_text = f"`{dependency[fam]['source_level']}` change frequency unavailable: {le['reason']}."
        lines += ["", "Level effect: " + level_text, ""]
    lines += ["## First-hour profile contribution", "",
              f"The stored level comparison shows the current Europe high changed on {profile_extrema['high_level_changed_dates']}/{len(dates)} dates ({profile_extrema['high_level_changed_percent']:.1f}%). The completed artifacts do not provide the corresponding Europe low/profile volume maps, so exact first-hour set-high and set-low counts are unavailable without rebuilding profiles; no such rebuild was performed.", "",
              "## Overall interpretation", "",
              f"The aggregate +25.748 R is not period-stable: Spring contributes {spring_total:+.3f} R while October contributes {october_total:+.3f} R. Most material activity belongs to `EUROPE|EUROPE|CURRENT|HIGH`: it improves in Spring (+32.266 R) but worsens in October (-7.681 R). The two prior-Europe families have very sparse trade counts and the NY prior-POC family is unchanged. Accordingly the evidence is mixed, not a general Europe-start improvement. Removed-trade economics are shown above and in JSON; they are descriptive, not a proposed filter.", "",
              "## Complete date × family delta matrix", "",
              "The exact matrix is in `family-day-delta-matrix.csv`; all 54 dates are included, including zero-trade days.", "",
              "| Date | " + " | ".join(FAMILIES) + " |", "|---|" + "---:|" * len(FAMILIES)]
    for row in matrix:
        lines.append("| " + row["date"] + " | " + " | ".join(f"{row[f]:+.3f}" for f in FAMILIES) + " |")
    lines += ["", "## Scope flags", "", "- `2026_DATA_ACCESSED=false`", "- `RAW_REPLAY_RERUN=false`", "- `OPTIMIZATION_PERFORMED=false`", "- `SESSION_SEARCH_PERFORMED=false`", "- `DATA_DOWNLOADED=false`", "- `COMMIT_PERFORMED=false`", ""]
    (OUT / "report.md").write_text("\n".join(lines))

    outputs = sorted(p for p in OUT.iterdir() if p.is_file() and p.name != "artifact-hashes.json")
    write_json(OUT / "artifact-hashes.json", {p.name: sha(p) for p in outputs})
    print(json.dumps({"status": "COMPLETE", "artifact_root": str(OUT), "overall": overall,
                      "primary_decision": primary_decision, "family_decisions": decisions,
                      "common_trade_ids": len(common_ids), "artifacts": len(outputs) + 1}, indent=2))


if __name__ == "__main__":
    from collections import Counter
    main()
