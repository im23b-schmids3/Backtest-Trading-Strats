from __future__ import annotations

import yaml

from pathlib import Path

from .models import PropFirmPolicy, calculate_policy_hash


def unconfigured_policy() -> PropFirmPolicy:
    """Return a safe profile with no unverified firm rules enabled."""
    data = PropFirmPolicy.model_validate({"policy_hash": "pending"}, context={"skip_policy_hash_validation": True})
    return PropFirmPolicy.model_validate({**data.model_dump(mode="python"), "policy_hash": calculate_policy_hash(data)})


def load_policy(path: str | Path) -> PropFirmPolicy:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("policy YAML must contain a mapping")
    return PropFirmPolicy.model_validate(raw)


def save_policy(policy: PropFirmPolicy, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(policy.model_dump(mode="json"), destination.open("w", encoding="utf-8"), sort_keys=False)
