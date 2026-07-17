from __future__ import annotations

import hashlib
from pathlib import Path

from .errors import ArtifactIntegrityError, ImplementationScopeViolation


FORBIDDEN_PREFIXES = (
    "src/fib_backtester/strategy/", "src/fib_backtester/research/", "data/", "reports/",
    "research_registry/spec_drafts/", "research_registry/verification/", "research_runs/",
)
FORBIDDEN_TOKENS = ("broker", "order_router", "live_trading", "credentials", ".env", "secret")


def verify_implementation_scope(changed_files: list[str], allowed_scopes: list[str] | None = None) -> list[str]:
    scopes = tuple(scope.replace("\\", "/").lstrip("./").rstrip("/") for scope in (allowed_scopes or ["src/", "tests/", "docs/"]))
    violations: list[str] = []
    for raw in changed_files:
        path = raw.replace("\\", "/").lstrip("./")
        in_scope = any(path == scope or path.startswith(scope + "/") for scope in scopes)
        if any(path.startswith(prefix) for prefix in FORBIDDEN_PREFIXES) or any(token in path.lower() for token in FORBIDDEN_TOKENS) or not in_scope:
            violations.append(path)
    if violations:
        raise ImplementationScopeViolation(f"{ImplementationScopeViolation.code}: {violations}")
    return []


def verify_artifact(path: str | Path, expected_hash: str) -> str:
    target = Path(path)
    if not target.is_file():
        raise ArtifactIntegrityError(f"{ArtifactIntegrityError.code}: missing artifact {target}")
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if digest != expected_hash:
        raise ArtifactIntegrityError(f"{ArtifactIntegrityError.code}: {target}")
    return digest
