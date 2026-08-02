from __future__ import annotations

from datetime import datetime, timezone

from ..enums import PipelineState
from ..errors import HoldoutAccessError


class HoldoutLock:
    def __init__(self, registry):
        self.registry = registry

    def status(self, strategy_id: str) -> dict:
        strategy = self.registry.get_strategy(strategy_id)
        return {"accesses": self.registry.count_holdout_accesses(strategy_id, strategy["version"]),
                "max_accesses": self.registry.get_budget(strategy_id, strategy["version"])["limits"]["max_holdout_accesses"],
                "locked": bool(strategy["parameters_frozen"]),
                "opened": bool(strategy["holdout_opened"])}

    def open(self, strategy_id: str, reason: str, dataset_hash: str | None = None) -> dict:
        strategy = self.registry.get_strategy(strategy_id)
        if strategy["current_phase"] != PipelineState.HOLDOUT.value:
            raise HoldoutAccessError("holdout may only be opened in HOLDOUT state")
        if strategy["holdout_opened"]:
            raise HoldoutAccessError("holdout has already been opened")
        budget = self.registry.get_budget(strategy_id, strategy["version"])
        if budget["usage"]["holdout_accesses"] >= budget["limits"]["max_holdout_accesses"]:
            raise HoldoutAccessError("holdout access budget exhausted")
        split = self.registry.get_split(strategy_id, strategy["version"])
        self.registry.record_holdout_access(strategy_id, strategy["version"], datetime.now(timezone.utc).isoformat(), strategy["current_phase"], dataset_hash or (split.source_data_hash if split else ""), reason)
        return self.status(strategy_id)
