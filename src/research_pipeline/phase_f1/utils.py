from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_strategy_id(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-")
    return value or "strategy"
