from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def write_artifact(root: Path, relative: str, payload: Any) -> tuple[str, str]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, default=str).encode()
    path.write_bytes(encoded)
    return str(path), hashlib.sha256(encoded).hexdigest()
