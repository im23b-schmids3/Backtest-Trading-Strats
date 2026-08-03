from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def code_hash(repository_root: str | Path) -> str:
    root = Path(repository_root).resolve()
    package = root / "src" / "research_pipeline" / "imbalance_vwap_ride"
    files = sorted(path for path in package.glob("*.py") if path.name != "__pycache__")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    cli = root / "src" / "research_pipeline" / "cli.py"
    if cli.is_file():
        digest.update(b"cli.py")
        digest.update(cli.read_bytes())
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_bytes_once(path: str | Path, content: bytes) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() != content:
            raise ValueError(f"immutable artifact collision: {destination}")
        return destination
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_bytes(content)
    try:
        temporary.replace(destination)
    except Exception:
        if destination.exists() and destination.read_bytes() == content:
            temporary.unlink(missing_ok=True)
        else:
            raise
    return destination


@dataclass(frozen=True)
class ArtifactContext:
    run_id: str
    dataset_hash: str
    source_manifest_hash: str
    specification_hash: str
    parameter_hash: str
    code_hash: str
    evidence_label: str
    timestamp: str

    def metadata(self) -> dict[str, str]:
        return {
            "study_run_id": self.run_id,
            "dataset_hash": self.dataset_hash,
            "source_manifest_hash": self.source_manifest_hash,
            "specification_hash": self.specification_hash,
            "parameter_hash": self.parameter_hash,
            "code_hash": self.code_hash,
            "evidence_label": self.evidence_label,
            "artifact_timestamp": self.timestamp,
        }

    def envelope(self, payload: Any) -> dict[str, Any]:
        if isinstance(payload, dict):
            return {**self.metadata(), **payload}
        return {**self.metadata(), "payload": payload}

    def write_json(self, path: str | Path, payload: Any) -> Path:
        content = json.dumps(self.envelope(payload), indent=2, sort_keys=True, default=str).encode("utf-8") + b"\n"
        return write_bytes_once(path, content)

    def write_parquet(self, path: str | Path, rows: list[dict[str, Any]]) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist(rows) if rows else pa.table({"_empty": pa.array([], type=pa.bool_())})
        metadata = dict(table.schema.metadata or {})
        metadata.update({key.encode(): value.encode() for key, value in self.metadata().items()})
        table = table.replace_schema_metadata(metadata)
        temporary = destination.with_name(f".{destination.name}.tmp")
        pq.write_table(table, temporary, compression="zstd", use_dictionary=True)
        content_hash = sha256_file(temporary)
        if destination.exists():
            if sha256_file(destination) != content_hash:
                temporary.unlink(missing_ok=True)
                raise ValueError(f"immutable artifact collision: {destination}")
            temporary.unlink(missing_ok=True)
            return destination
        temporary.replace(destination)
        return destination
