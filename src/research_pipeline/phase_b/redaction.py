from __future__ import annotations

import re


SECRET_PATTERNS = (
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+"),
    re.compile(r"(?i)(api[_-]?key\s*[=:]\s*)[^\s,;]+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"(?i)(token\s*[=:]\s*)[^\s,;]+"),
)


def redact_secrets(value: str) -> str:
    result = value
    for pattern in SECRET_PATTERNS:
        if pattern.groups:
            result = pattern.sub(lambda match: f"{match.group(1)}[REDACTED]", result)
        else:
            result = pattern.sub("[REDACTED]", result)
    return result

