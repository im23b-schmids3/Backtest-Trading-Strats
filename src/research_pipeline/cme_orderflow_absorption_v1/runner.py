"""CLI entry point for the sealed, read-only ESU6 MBO pilot."""
from __future__ import annotations
from pathlib import Path
from .analysis import Diagnostics
from .analysis import day_and_seconds
from .engine import BookStateError, CausalMBOBook
from .loader import EXPECTED_SHA256, sha256_file, stream_mbo, validate_metadata, DBNValidationError
from .report import write_reports

def run(dbn: Path, out: Path) -> dict:
    digest, size = sha256_file(dbn)
    if digest != EXPECTED_SHA256: raise DBNValidationError(f"SHA-256 mismatch: {digest}")
    metadata = validate_metadata(dbn); book = CausalMBOBook(); diag = Diagnostics(); context_dates: set[str] = set()
    try:
        for rec in stream_mbo(dbn):
            date, _ = day_and_seconds(rec.ts_recv)
            if date not in context_dates: diag.finish_day_context(date); context_dates.add(date)
            # DBN iterator order is authoritative. Equal-receive-time sequence values
            # are not a total ordering across all records, so they are retained but
            # not re-sorted or used to discard/reorder the provider stream.
            applied = book.apply(action=rec.action, side=rec.side, price=rec.price, size=rec.size, order_id=rec.order_id, sequence=rec.sequence, ts_recv=rec.ts_recv, channel_id=rec.channel_id, validate_sequence=False, mutate_execution=False)
            diag.observe(rec, applied, book.spread())
    except (BookStateError, DBNValidationError) as exc:
        diag.issues[type(exc).__name__] += 1
        write_reports(out, sha256=digest, dbn_bytes=size, metadata=metadata, diagnostics=diag, integrity=f"FAILED_CLOSED: {exc}")
        raise
    write_reports(out, sha256=digest, dbn_bytes=size, metadata=metadata, diagnostics=diag, integrity="PASS: strict causal reconstruction completed")
    return {"sha256":digest,"bytes":size,"events":diag.events,"executions":diag.executions,"integrity":"PASS: strict causal reconstruction completed"}

if __name__ == "__main__":
    import argparse, json
    p=argparse.ArgumentParser(); p.add_argument("dbn", type=Path); p.add_argument("out", type=Path)
    a=p.parse_args(); print(json.dumps(run(a.dbn, a.out)))
