"""True delayed-entry V2 replay on the already-seen Aug 3-7 ES MBO block.

Research-only V2:
- V1 PLUS base signal
- 15s confirmation, >= 1 favorable ES tick
- 2ms post-confirmation latency
- Prefer ES; if one ES exceeds the $250 risk budget, use max whole MES contracts
- 2R target, existing stop rule, 22:45 UTC hard flat

MES execution in this seen-data research replay is proxied from the ES MBO
price path. Fresh holdout testing should use native MES market data for MES fills.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from databento import DBNStore

from .oos_backtest_runner import (
    COMMISSION,
    RAW_TICK,
    RISK_BUDGET,
    TICK,
    USD_PER_POINT,
    CausalMBOBook,
    _cutoff,
    _valid,
    day_and_seconds,
    digest,
    is_snapshot_record,
    load_frozen,
    prices,
    size_trade,
)

ROOT = Path(__file__).resolve().parents[3]
DBN = ROOT / "data/cme_orderflow_absorption_v1/oos_v1/ESU6/mbo/ESU6_2026-08-03_2026-08-08_mbo.dbn"
RESEARCH_CSV = ROOT / "research_runs/CMEOrderflowAbsorption.ES_V2_RESEARCH/seen_15_rth/all-interactions.csv"
V2_CONTRACT = ROOT / "docs/research_pipeline/cme_orderflow_absorption_v1/v2-confirmation-contract.json"
OUT = ROOT / "research_runs/CMEOrderflowAbsorption.ES_V2_RESEARCH/seen_aug_true_v2_replay_mes_fallback"

EXPECTED_SOURCE_SHA = "BE4B56639E56DF9AACE81621E4E276463EA8AF889104F35F1744400310D53AA3"
EXPECTED_PLUS = 21
CONFIRM_NS = 15_000_000_000
LATENCY_NS = 2_000_000
MIN_FAVORABLE_TICKS = 1.0
PROGRESS_EVERY = 5_000_000

MES_USD_PER_POINT = 5.0
MES_COMMISSION_PER_SIDE = 1.25


def action_value(rec: Any) -> str:
    return str(getattr(rec.action, "value", rec.action))


def load_v2_contract() -> dict[str, Any]:
    c = json.loads(V2_CONTRACT.read_text(encoding="utf-8-sig"))
    required = {
        "strategy": "CMEOrderflowAbsorption.ES_V2",
        "base_signal": "V1_PLUS",
        "confirmation_seconds": 15,
        "minimum_favorable_ticks": 1,
        "confirmation_reference": "interaction_end_price",
        "stop_rule": "5_ES_TICKS_BEYOND_COMPLETED_INTERACTION_ZONE",
        "target_rule": "SINGLE_2R",
        "risk_budget_usd": 250,
        "commission_usd_per_side": 3,
        "cutoff_utc": "22:45:00",
        "score_calibration_source": "DEVELOPMENT_ONLY",
        "v2_rule_frozen": True,
    }
    for k, v in required.items():
        if c.get(k) != v:
            raise RuntimeError(f"V2 contract mismatch: {k}={c.get(k)!r}, expected {v!r}")
    return c


def load_seen_aug_plus() -> list[dict[str, Any]]:
    rows = []
    with RESEARCH_CSV.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["research_split"] != "SEEN_OOS_AUG" or r["v1_plus"] != "True":
                continue
            rows.append({
                "interaction_id": r["interaction_id"],
                "date": r["date"],
                "direction": r["direction"],
                "level": r["level"],
                "interaction_end": int(r["interaction_end"]),
                "end_price": int(r["end_price"]),
                "zone_low": int(r["zone_low"]),
                "zone_high": int(r["zone_high"]),
                "absorption_score": float(r["absorption_score"]),
                "replenishment_score": float(r["replenishment_score"]),
            })
    if len(rows) != EXPECTED_PLUS:
        raise RuntimeError(f"seen-Aug V1 PLUS count={len(rows)}, expected {EXPECTED_PLUS}")
    if len({r["interaction_id"] for r in rows}) != len(rows):
        raise RuntimeError("duplicate PLUS interaction ids")
    return sorted(rows, key=lambda r: r["interaction_end"])


def audit(audits, row, outcome, **extra):
    if any(x["interaction_id"] == row["interaction_id"] for x in audits):
        raise RuntimeError(f"duplicate audit for {row['interaction_id']}")
    audits.append({
        "interaction_id": row["interaction_id"],
        "date": row["date"],
        "outcome": outcome,
        **extra,
    })


def size_trade_with_mes_fallback(p):
    es = size_trade(p)
    if es["contracts"] >= 1:
        return {
            **es,
            "instrument": "ES",
            "execution_model": "ES_NATIVE",
            "usd_per_point": USD_PER_POINT,
            "commission_per_side": COMMISSION,
        }

    mes_raw = abs(p["entry_reference"] - p["stop"]) * MES_USD_PER_POINT
    mes_slip = (
        abs(p["entry"] - p["entry_reference"])
        + abs(p["stop_exit"] - p["stop"])
    ) * MES_USD_PER_POINT
    mes_initial_risk = (
        abs(p["entry"] - p["stop_exit"]) * MES_USD_PER_POINT
        + 2 * MES_COMMISSION_PER_SIDE
    )
    if mes_initial_risk <= 0:
        raise RuntimeError("invalid MES initial risk")

    return {
        "raw_price_risk_usd": mes_raw,
        "slippage_contribution_usd": mes_slip,
        "one_contract_price_risk_usd": mes_raw + mes_slip,
        "one_contract_initial_risk_usd": mes_initial_risk,
        "contracts": int(RISK_BUDGET // mes_initial_risk),
        "instrument": "MES",
        "execution_model": "MES_PROXY_EXECUTION_FROM_ES_MBO",
        "usd_per_point": MES_USD_PER_POINT,
        "commission_per_side": MES_COMMISSION_PER_SIDE,
    }


def close_trade(trades, audits, position, ts, exit_fill, reason):
    sign = 1 if position["direction"] == "LONG" else -1
    gross = (
        (exit_fill - position["entry"])
        * sign
        * position["usd_per_point"]
        * position["contracts"]
    )
    fees = 2 * position["commission_per_side"] * position["contracts"]
    risk = position["one_contract_initial_risk_usd"] * position["contracts"]
    trades.append({
        **position,
        "exit_timestamp": ts,
        "exit_fill": exit_fill,
        "exit_reason": reason,
        "gross_usd": gross,
        "commission_usd": fees,
        "net_usd": gross - fees,
        "r_multiple": gross / risk,
    })
    audit(audits, position, "ENTERED_" + reason)


def main() -> int:
    contract = load_v2_contract()
    _, manifest = load_frozen()

    source_sha = digest(DBN)
    if source_sha != EXPECTED_SOURCE_SHA:
        raise RuntimeError(f"source SHA mismatch: {source_sha}")

    signals = load_seen_aug_plus()
    waiting = [{**r, "confirmation_due_ns": r["interaction_end"] + CONFIRM_NS} for r in signals]
    ready = []
    audits = []
    trades = []
    book = CausalMBOBook()
    position = None
    cutoff_quote = None
    records = 0
    passed = 0
    failed = 0

    for rec in DBNStore.from_file(DBN):
        records += 1

        if is_snapshot_record(rec, manifest):
            book.apply(
                action=rec.action, side=rec.side, price=rec.price, size=rec.size,
                order_id=rec.order_id, sequence=rec.sequence, ts_recv=rec.ts_recv,
                channel_id=rec.channel_id, validate_sequence=False, mutate_execution=False,
            )
            continue

        if action_value(rec) == "N":
            if records % PROGRESS_EVERY == 0:
                print(f"[V2 replay] records={records:,}", flush=True)
            continue

        applied = book.apply(
            action=rec.action, side=rec.side, price=rec.price, size=rec.size,
            order_id=rec.order_id, sequence=rec.sequence, ts_recv=rec.ts_recv,
            channel_id=rec.channel_id, validate_sequence=False, mutate_execution=False,
        )
        day, _ = day_and_seconds(rec.ts_recv)
        quote = _valid(book)
        cutoff = _cutoff(day)

        if quote and cutoff - 1_000_000_000 <= rec.ts_recv <= cutoff:
            cutoff_quote = (rec.ts_recv, quote)

        if position is not None and rec.ts_recv > position["cutoff_ns"]:
            if cutoff_quote is None:
                raise RuntimeError("CUTOFF_EXECUTION_INTEGRITY_FAILURE")
            ts, q = cutoff_quote
            exit_ref = q[0] if position["direction"] == "LONG" else q[1]
            exit_fill = exit_ref - TICK if position["direction"] == "LONG" else exit_ref + TICK
            close_trade(trades, audits, position, ts, exit_fill, "CUTOFF_FORCED_FLAT")
            position = None
            cutoff_quote = None

        nw = []
        for row in waiting:
            if rec.ts_recv >= _cutoff(row["date"]):
                audit(audits, row, "CUTOFF_CANCELLED_BEFORE_CONFIRMATION")
            else:
                nw.append(row)
        waiting = nw

        nr = []
        for row in ready:
            if rec.ts_recv >= _cutoff(row["date"]):
                audit(audits, row, "CUTOFF_CANCELLED_AFTER_CONFIRMATION")
            else:
                nr.append(row)
        ready = nr

        if applied is not None and applied.executed:
            unresolved = []
            for row in waiting:
                if row["confirmation_due_ns"] <= rec.ts_recv:
                    fav = (
                        (applied.price - row["end_price"]) / RAW_TICK
                        if row["direction"] == "BUYER_ABSORPTION"
                        else (row["end_price"] - applied.price) / RAW_TICK
                    )
                    if fav >= MIN_FAVORABLE_TICKS:
                        passed += 1
                        ready.append({
                            **row,
                            "confirmation_timestamp": rec.ts_recv,
                            "confirmation_price": applied.price / 1e9,
                            "confirmation_favorable_ticks": fav,
                            "entry_ready_ns": rec.ts_recv + LATENCY_NS,
                        })
                    else:
                        failed += 1
                        audit(
                            audits, row, "CONFIRMATION_FAILED",
                            confirmation_timestamp=rec.ts_recv,
                            confirmation_price=applied.price / 1e9,
                            confirmation_favorable_ticks=fav,
                        )
                else:
                    unresolved.append(row)
            waiting = unresolved
            ready.sort(key=lambda r: (r["entry_ready_ns"], r["interaction_end"], r["interaction_id"]))

        remaining = []
        for row in ready:
            if row["entry_ready_ns"] > rec.ts_recv:
                remaining.append(row)
                continue
            if rec.ts_recv >= _cutoff(row["date"]):
                audit(audits, row, "CUTOFF_CANCELLED_AFTER_CONFIRMATION")
                continue
            if position is not None:
                audit(audits, row, "POSITION_ALREADY_OPEN")
                continue
            if quote is None:
                audit(audits, row, "INVALID_SPREAD")
                continue

            p = prices(
                row["direction"], quote[0], quote[1],
                row["zone_low"] / 1e9, row["zone_high"] / 1e9,
            )
            s = size_trade_with_mes_fallback(p)
            if not s["contracts"]:
                audit(audits, row, "INSUFFICIENT_RISK_BUDGET_FOR_ES_AND_MES")
                continue

            position = {
                **row, **p, **s,
                "entry_timestamp": rec.ts_recv,
                "cutoff_ns": _cutoff(row["date"]),
            }
        ready = remaining

        if position is not None and quote:
            bid, ask = quote
            hit = (
                position["direction"] == "LONG"
                and (bid <= position["stop"] or bid >= position["target"])
            ) or (
                position["direction"] == "SHORT"
                and (ask >= position["stop"] or ask <= position["target"])
            )
            if hit:
                exit_ref = bid if position["direction"] == "LONG" else ask
                exit_fill = exit_ref - TICK if position["direction"] == "LONG" else exit_ref + TICK
                reason = "STOP" if (
                    (position["direction"] == "LONG" and bid <= position["stop"])
                    or (position["direction"] == "SHORT" and ask >= position["stop"])
                ) else "TARGET"
                close_trade(trades, audits, position, rec.ts_recv, exit_fill, reason)
                position = None

        if records % PROGRESS_EVERY == 0:
            print(f"[V2 replay] records={records:,}", flush=True)

    for row in waiting:
        audit(audits, row, "CONFIRMATION_NOT_OBSERVED")
    for row in ready:
        audit(audits, row, "ENTRY_NOT_REACHED_AFTER_CONFIRMATION")
    if position is not None:
        raise RuntimeError("open position remained after source end")
    if len(audits) != EXPECTED_PLUS:
        raise RuntimeError(f"audit count={len(audits)}, expected {EXPECTED_PLUS}")

    OUT.mkdir(parents=True, exist_ok=True)

    tf = sorted({k for r in trades for k in r}) or ["interaction_id"]
    with (OUT / "trades.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, tf, lineterminator="\n")
        w.writeheader()
        w.writerows(trades)

    af = sorted({k for r in audits for k in r})
    with (OUT / "audit.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, af, lineterminator="\n")
        w.writeheader()
        w.writerows(audits)

    wins = [t for t in trades if t["net_usd"] > 0]
    losses = [t for t in trades if t["net_usd"] < 0]
    gp = sum(t["net_usd"] for t in wins)
    gl = abs(sum(t["net_usd"] for t in losses))

    summary = {
        "status": "RECONCILED_RESEARCH_REPLAY",
        "interpretation": "SEEN_AUG_DATA_NOT_FRESH_OOS_EVIDENCE",
        "source_sha256": source_sha,
        "dbn_records_seen": records,
        "v1_plus_input": EXPECTED_PLUS,
        "confirmation_seconds": 15,
        "minimum_favorable_ticks": 1,
        "confirmation_semantics": "FIRST_ES_EXECUTION_AT_OR_AFTER_HORIZON",
        "post_confirmation_latency_ms": 2,
        "confirmations_passed": passed,
        "confirmations_failed": failed,
        "trades": len(trades),
        "es_trades": sum(t["instrument"] == "ES" for t in trades),
        "mes_trades": sum(t["instrument"] == "MES" for t in trades),
        "wins": len(wins),
        "losses": len(losses),
        "winrate_pct": 100 * len(wins) / len(trades) if trades else 0,
        "net_pnl_usd": sum(t["net_usd"] for t in trades),
        "gross_profit_usd": gp,
        "gross_loss_usd": gl,
        "profit_factor": gp / gl if gl else None,
        "total_r": sum(t["r_multiple"] for t in trades),
        "audit_rows": len(audits),
        "cutoff_utc": "22:45:00",
        "score_calibration_source": "DEVELOPMENT_ONLY",
        "mes_execution_model": "MES_PROXY_EXECUTION_FROM_ES_MBO",
        "mes_commission_per_side": MES_COMMISSION_PER_SIDE,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())