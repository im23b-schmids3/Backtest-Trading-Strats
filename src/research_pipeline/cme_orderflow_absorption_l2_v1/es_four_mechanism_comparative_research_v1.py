"""Source audit and native-depth technical pilot for ES mechanism research V1.

The pilot is technical only: it does not calculate forward outcomes or trading
performance. Native MBP-10 records expose a complete aggregate top-10 state on
each row, so every row is inspected in original DBN order.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

STUDY_ID = "ES_FOUR_MECHANISM_COMPARATIVE_RESEARCH_V1"
OUT = Path("research_runs") / STUDY_ID
SOURCE_MANIFEST = Path("research_runs/ES_HVN_HEAVY_ABSORPTION_RESEARCH_V1/source-integrity-results.json")
DATA_ROOT = Path("data/databento/mac-2025-native-es-mbp10-final-reduced/TRAIN")
TICK_RAW = 250_000_000
FLAG_BAD_TS_RECV = 8
FLAG_MAYBE_BAD_BOOK = 4
FLAG_SNAPSHOT = 32


class MechanismResearchError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_book(row: Any) -> dict[str, Any]:
    """Read source price-level identities without compacting away empty levels."""
    bids = tuple((int(row[f"bid_px_{i:02d}"]), int(row[f"bid_sz_{i:02d}"])) for i in range(10))
    asks = tuple((int(row[f"ask_px_{i:02d}"]), int(row[f"ask_sz_{i:02d}"])) for i in range(10))
    valid_prices = all(p > 0 and s > 0 for p, s in (bids[0], asks[0]))
    bid_ordered = all(bids[i][0] > bids[i + 1][0] for i in range(9) if bids[i][0] and bids[i + 1][0])
    ask_ordered = all(asks[i][0] < asks[i + 1][0] for i in range(9) if asks[i][0] and asks[i + 1][0])
    crossed = bool(valid_prices and bids[0][0] >= asks[0][0])
    return {
        "bids": bids,
        "asks": asks,
        "valid_top": bool(valid_prices and bid_ordered and ask_ordered and not crossed),
        "crossed_or_locked": crossed,
        "top5_bid_size": sum(size for _, size in bids[:5]),
        "top5_ask_size": sum(size for _, size in asks[:5]),
        "top10_bid_size": sum(size for _, size in bids),
        "top10_ask_size": sum(size for _, size in asks),
    }


def source_rows(repo: Path) -> list[dict[str, Any]]:
    manifest = json.loads((repo / SOURCE_MANIFEST).read_text(encoding="utf-8"))
    rows = manifest.get("sources")
    if manifest.get("status") != "PASS" or not isinstance(rows, list) or len(rows) != 35:
        raise MechanismResearchError("sealed Spring source manifest is not a 35-date PASS")
    return rows


def verify_sources(repo: Path) -> dict[str, Any]:
    verified = []
    for item in source_rows(repo):
        raw = repo / item["raw_path"]
        expected_symbol = "ESH5" if item["date"] <= "2025-03-14" else "ESM5"
        if not raw.is_file() or raw.stat().st_size != int(item["raw_bytes"]):
            raise MechanismResearchError(f"missing/size mismatch: {item['date']}")
        actual_hash = sha256(raw)
        if actual_hash != item["raw_sha256"]:
            raise MechanismResearchError(f"SHA-256 mismatch: {item['date']}")
        if item.get("dataset") != "GLBX.MDP3" or item.get("schema") != "mbp-10" or item.get("symbol") != expected_symbol:
            raise MechanismResearchError(f"identity/roll mismatch: {item['date']}")
        verified.append({"date": item["date"], "raw_path": item["raw_path"], "size_bytes": raw.stat().st_size,
                         "sha256": actual_hash, "dataset": item["dataset"], "schema": item["schema"],
                         "symbol": item["symbol"], "manifest_event_count": item["event_count"], "status": "PASS"})
    return {"status": "PASS", "date_count": len(verified), "sources": verified}


def inspect_native_day(repo: Path, source: dict[str, Any], *, chunk_rows: int = 100_000) -> dict[str, Any]:
    """Audit all native records and all source depth states for one date."""
    from databento import DBNStore

    path = repo / source["raw_path"]
    store = DBNStore.from_file(str(path))
    metadata = store.metadata
    symbol = "ESH5" if source["date"] <= "2025-03-14" else "ESM5"
    if metadata.dataset != "GLBX.MDP3" or str(metadata.schema) != "mbp-10" or symbol not in metadata.symbols:
        raise MechanismResearchError(f"native DBN header mismatch on {source['date']}: {metadata}")
    counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    first_recv = last_recv = first_event = last_event = None
    recv_backsteps = event_backsteps = seq_nonincrease = 0
    bad_recv = bad_book = snapshots = 0
    valid_top_rows = crossed_rows = depth_changes = depth_only_changes = bbo_changes = 0
    previous_bids = previous_asks = None
    previous_recv = previous_event = previous_seq = None
    total = trades = raw_trade_actions = unknown_side = positive_price_trades = 0
    for batch in store.to_ndarray(count=chunk_rows):
        if not isinstance(batch, np.ndarray) or batch.dtype.names is None:
            raise MechanismResearchError("DBN reader yielded a non-record array")
        if np.any(batch["rtype"] != 10):
            raise MechanismResearchError("source contains non-MBP-10 records")
        n = len(batch)
        total += n
        is_trade_action = batch["action"] == b"T"
        raw_trade_actions += int(np.count_nonzero(is_trade_action))
        positive_trade = is_trade_action & (batch["size"] > 0)
        trades += int(np.count_nonzero(positive_trade))
        positive_price_trades += int(np.count_nonzero(positive_trade & (batch["price"] > 0)))
        unknown_side += int(np.count_nonzero(positive_trade & ~np.isin(batch["side"], (b"A", b"B"))))
        bad_recv += int(np.count_nonzero(batch["flags"] & FLAG_BAD_TS_RECV))
        bad_book += int(np.count_nonzero(batch["flags"] & FLAG_MAYBE_BAD_BOOK))
        snapshots += int(np.count_nonzero(batch["flags"] & FLAG_SNAPSHOT))
        for action, count in zip(*np.unique(batch["action"], return_counts=True)):
            key = bytes(action).decode("ascii", "replace")
            action_counts[key] = action_counts.get(key, 0) + int(count)
        recv_arr, event_arr, seq_arr = batch["ts_recv"], batch["ts_event"], batch["sequence"]
        if first_recv is None:
            first_recv, first_event = int(recv_arr[0]), int(event_arr[0])
        recv_backsteps += int(np.count_nonzero(recv_arr[1:] < recv_arr[:-1]))
        event_backsteps += int(np.count_nonzero(event_arr[1:] < event_arr[:-1]))
        seq_nonincrease += int(np.count_nonzero(seq_arr[1:] <= seq_arr[:-1]))
        if previous_recv is not None:
            recv_backsteps += int(recv_arr[0] < previous_recv)
            event_backsteps += int(event_arr[0] < previous_event)
            seq_nonincrease += int(seq_arr[0] <= previous_seq)
        previous_recv, previous_event, previous_seq = int(recv_arr[-1]), int(event_arr[-1]), int(seq_arr[-1])
        last_recv, last_event = previous_recv, previous_event
        bid_px = np.column_stack([batch[f"bid_px_{i:02d}"] for i in range(10)])
        ask_px = np.column_stack([batch[f"ask_px_{i:02d}"] for i in range(10)])
        bid_sz = np.column_stack([batch[f"bid_sz_{i:02d}"] for i in range(10)])
        ask_sz = np.column_stack([batch[f"ask_sz_{i:02d}"] for i in range(10)])
        bid_valid = (bid_px > 0) & (bid_sz > 0)
        ask_valid = (ask_px > 0) & (ask_sz > 0)
        crossed = (bid_px[:, 0] > 0) & (ask_px[:, 0] > 0) & (bid_px[:, 0] >= ask_px[:, 0])
        bid_sorted = np.all((bid_px[:, :-1] == 0) | (bid_px[:, 1:] == 0) | (bid_px[:, :-1] > bid_px[:, 1:]), axis=1)
        ask_sorted = np.all((ask_px[:, :-1] == 0) | (ask_px[:, 1:] == 0) | (ask_px[:, :-1] < ask_px[:, 1:]), axis=1)
        valid = bid_valid[:, 0] & ask_valid[:, 0] & bid_sorted & ask_sorted & ~crossed & ((batch["flags"] & FLAG_MAYBE_BAD_BOOK) == 0)
        valid_top_rows += int(np.count_nonzero(valid))
        crossed_rows += int(np.count_nonzero(crossed))
        changed = np.zeros(n, dtype=bool)
        same_bbo = np.zeros(n, dtype=bool)
        if previous_bids is not None and n:
            changed[0] = not (np.array_equal(bid_px[0], previous_bids[0]) and np.array_equal(bid_sz[0], previous_bids[1]) and np.array_equal(ask_px[0], previous_asks[0]) and np.array_equal(ask_sz[0], previous_asks[1]))
            same_bbo[0] = bid_px[0, 0] == previous_bids[0][0] and bid_sz[0, 0] == previous_bids[1][0] and ask_px[0, 0] == previous_asks[0][0] and ask_sz[0, 0] == previous_asks[1][0]
        if n > 1:
            changed[1:] = np.any(bid_px[1:] != bid_px[:-1], axis=1) | np.any(bid_sz[1:] != bid_sz[:-1], axis=1) | np.any(ask_px[1:] != ask_px[:-1], axis=1) | np.any(ask_sz[1:] != ask_sz[:-1], axis=1)
            same_bbo[1:] = (bid_px[1:, 0] == bid_px[:-1, 0]) & (bid_sz[1:, 0] == bid_sz[:-1, 0]) & (ask_px[1:, 0] == ask_px[:-1, 0]) & (ask_sz[1:, 0] == ask_sz[:-1, 0])
        depth_changes += int(np.count_nonzero(changed))
        depth_only_changes += int(np.count_nonzero(changed & same_bbo))
        bbo_changes += int(np.count_nonzero(changed & ~same_bbo))
        previous_bids = (bid_px[-1].copy(), bid_sz[-1].copy())
        previous_asks = (ask_px[-1].copy(), ask_sz[-1].copy())
    close = getattr(store, "close", None)
    if callable(close):
        close()
    known_trades = trades - unknown_side
    return {"audit_schema_version": 3, "date": source["date"], "source_path": source["raw_path"], "sha256": source["raw_sha256"],
            "dataset": metadata.dataset, "schema": str(metadata.schema), "symbol": symbol,
            "records": total, "legacy_manifest_event_count_field": int(source["event_count"]),
            "legacy_manifest_event_count_is_raw_record_count": False,
            "raw_trade_action_records": raw_trade_actions, "nonpositive_size_trade_actions": raw_trade_actions - trades,
            "positive_price_trade_records": positive_price_trades,
            "manifest_trade_records": int(source["trade_records"]), "trades": trades,
            "known_aggressor_trade_records": known_trades,
            "known_aggressor_trade_count_matches_manifest": known_trades == int(source["trade_records"]),
            "positive_price_trade_count_matches_manifest": positive_price_trades == int(source["trade_records"]),
            "unknown_aggressor_trades": unknown_side,
            "action_counts": action_counts, "first_ts_recv": first_recv, "last_ts_recv": last_recv,
            "first_ts_event": first_event, "last_ts_event": last_event,
            "ts_recv_backsteps": recv_backsteps, "ts_event_backsteps": event_backsteps,
            "sequence_nonincreases_retained": seq_nonincrease, "bad_ts_recv_flags": bad_recv,
            "maybe_bad_book_flags": bad_book, "snapshot_flags": snapshots,
            "valid_top_book_rows": valid_top_rows, "crossed_or_locked_rows": crossed_rows,
            "depth_state_changes": depth_changes, "depth_only_changes_no_bbo_move": depth_only_changes,
            "bbo_changes": bbo_changes, "status": "PASS"}


def run_pilot(repo: Path, day: str = "2025-03-03") -> dict[str, Any]:
    sources = source_rows(repo)
    source = next((x for x in sources if x["date"] == day), None)
    if source is None:
        raise MechanismResearchError(f"pilot day not in authorized source manifest: {day}")
    result = inspect_native_day(repo, source)
    result["purpose"] = "technical source/depth pilot only; no forward outcomes or performance"
    result["depth_preservation"] = "PASS; all native rows and all ten price-size levels inspected in DBN order"
    return result


def run_full_depth_audit(repo: Path) -> dict[str, Any]:
    """Resumable source-only full-depth scan; no outcomes or PnL are computed."""
    started = time.monotonic()
    checkpoint_dir = repo / OUT / ".checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rows = source_rows(repo)
    results = []
    for index, source in enumerate(rows, 1):
        checkpoint = checkpoint_dir / f"depth-{source['date']}.json"
        cached = None
        if checkpoint.is_file():
            try:
                candidate = json.loads(checkpoint.read_text(encoding="utf-8"))
                if candidate.get("audit_schema_version") in (2, 3) and candidate.get("sha256") == source["raw_sha256"] and candidate.get("status") == "PASS":
                    cached = candidate
            except (OSError, json.JSONDecodeError):
                cached = None
        result = cached or inspect_native_day(repo, source)
        if cached is not None and cached.get("audit_schema_version") == 2:
            result["known_aggressor_trade_records"] = result["trades"] - result["unknown_aggressor_trades"]
            result["known_aggressor_trade_count_matches_manifest"] = result["known_aggressor_trade_records"] == int(source["trade_records"])
            result["audit_schema_version"] = 3
        if cached is None or cached.get("audit_schema_version") == 2:
            checkpoint.write_text(json.dumps(result, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        results.append(result)
        print(f"DEPTH_AUDIT_DATE {index}/35 {source['date']} records={result['records']} depth_only={result['depth_only_changes_no_bbo_move']}", flush=True)
    peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    payload = {"status": "PASS", "date_count": len(results), "records": sum(x["records"] for x in results),
               "trades": sum(x["trades"] for x in results), "depth_state_changes": sum(x["depth_state_changes"] for x in results),
               "known_aggressor_trade_records": sum(x["known_aggressor_trade_records"] for x in results),
               "known_aggressor_trade_count_matches_all_dates": all(x["known_aggressor_trade_count_matches_manifest"] for x in results),
               "depth_only_changes_no_bbo_move": sum(x["depth_only_changes_no_bbo_move"] for x in results),
               "ts_recv_backsteps": sum(x["ts_recv_backsteps"] for x in results),
               "ts_event_backsteps": sum(x["ts_event_backsteps"] for x in results),
               "sequence_nonincreases_retained": sum(x["sequence_nonincreases_retained"] for x in results),
               "elapsed_seconds": time.monotonic() - started, "peak_rss_bytes": peak_rss, "dates": results}
    output = repo / OUT / "native-depth-integrity.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("command", choices=("verify-sources", "depth-pilot", "depth-audit"))
    parser.add_argument("--date", default="2025-03-03")
    args = parser.parse_args(argv)
    if args.command == "verify-sources":
        result = verify_sources(args.repo)
    elif args.command == "depth-pilot":
        result = run_pilot(args.repo, args.date)
    else:
        result = run_full_depth_audit(args.repo)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.command != "depth-audit":
        name = "source-integrity.json" if args.command == "verify-sources" else "native-depth-pilot.json"
        (args.repo / OUT / name).write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("sources", "dates")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
