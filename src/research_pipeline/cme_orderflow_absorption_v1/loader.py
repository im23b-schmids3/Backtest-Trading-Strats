"""Streaming-only DBN loader with metadata and ordering validation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
import hashlib

from databento import DBNStore

EXPECTED_SHA256 = "CBA9630FF44DEAB139A5D66B7886197435FA5388EE16B5E999E26CF4DB8B8B7C"
EXPECTED_DATASET, EXPECTED_SYMBOL, EXPECTED_INSTRUMENT = "GLBX.MDP3", "ESU6", 42140870
START_NS, END_NS = 1784505600000000000, 1785542400000000000

class DBNValidationError(ValueError): pass

@dataclass(frozen=True)
class MetadataSummary:
    dataset: str; schema: str; symbol: str; instrument_id: int; start_ns: int; end_ns: int

def sha256_file(path: Path) -> tuple[str, int]:
    h = hashlib.sha256(); total = 0
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block); total += len(block)
    return h.hexdigest().upper(), total

def validate_metadata(path: Path) -> MetadataSummary:
    md = DBNStore.from_file(path).metadata
    mappings = str(md.mappings)
    if (md.dataset != EXPECTED_DATASET or str(md.schema) != "mbo" or EXPECTED_SYMBOL not in md.symbols
        or str(EXPECTED_INSTRUMENT) not in mappings or md.start != START_NS or md.end != END_NS):
        raise DBNValidationError("embedded metadata does not match sealed ESU6 MBO pilot contract")
    return MetadataSummary(md.dataset, str(md.schema), EXPECTED_SYMBOL, EXPECTED_INSTRUMENT, md.start, md.end)

def stream_mbo(path: Path) -> Iterator[object]:
    """Yield records in provider order; never materializes the DBN stream."""
    for record in DBNStore.from_file(path):
        if getattr(record, "instrument_id", None) != EXPECTED_INSTRUMENT:
            raise DBNValidationError("record instrument differs from ESU6 mapping")
        yield record
