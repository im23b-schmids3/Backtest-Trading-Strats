from __future__ import annotations

import importlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from ..schemas.strategy_spec import StrategySpec
from .errors import AdapterCompatibilityError, RealAdapterRequired
from .models import AdapterCapabilities, AdapterHealth
from .native_backtest import NativeRepositoryAdapter


AdapterFactory = Callable[[StrategySpec, str | Path], object]


class AdapterRegistry:
    """Explicit, deterministic adapter registry with no synthetic fallback."""

    schema_version = "1"

    def __init__(self):
        self._factories: dict[str, AdapterFactory] = {}

    def register_family(self, strategy_family: str, factory: AdapterFactory) -> None:
        if strategy_family in self._factories:
            raise AdapterCompatibilityError(f"duplicate adapter registration: {strategy_family}")
        self._factories[strategy_family] = factory

    def register_module(self, strategy_family: str, module: str, entry_point: str) -> None:
        def factory(spec: StrategySpec, root: str | Path):
            imported = importlib.import_module(module)
            target = imported
            for part in entry_point.split("."):
                target = getattr(target, part)
            return target(specification=spec, repository_root=root)
        self.register_family(strategy_family, factory)

    def resolve(self, specification: StrategySpec, repository_root: str | Path = "."):
        factory = self._factories.get(specification.strategy_family)
        if factory is None:
            raise RealAdapterRequired(f"{RealAdapterRequired.code}: no registered adapter for strategy family {specification.strategy_family}")
        adapter = factory(specification, repository_root)
        health = adapter.health(specification)
        if not health.importable or not health.compatible or not health.healthy:
            raise AdapterCompatibilityError("adapter health check failed: " + "; ".join(health.errors))
        return adapter

    def inspect(self, specification: StrategySpec, repository_root: str | Path = ".") -> AdapterHealth:
        try:
            adapter = self.resolve(specification, repository_root)
            return adapter.health(specification)
        except Exception as exc:
            from .models import AdapterIdentity
            identity = AdapterIdentity(strategy_id=specification.strategy_id, strategy_version=specification.version,
                                       implementation_module="unresolved", entry_point="unresolved", specification_hash=specification.specification_hash)
            return AdapterHealth(identity=identity, capabilities=AdapterCapabilities(), importable=False, compatible=False, healthy=False,
                                 errors=[str(exc)], checked_at=datetime.now(timezone.utc))

    def list(self) -> list[str]:
        return sorted(self._factories)


def default_adapter_registry() -> AdapterRegistry:
    registry = AdapterRegistry()
    registry.register_family("f2_native_demo", lambda spec, root: NativeRepositoryAdapter(spec, root, source_symbols={"SPX": "SPY"}))
    registry.register_family("f2_random_open_test", lambda spec, root: NativeRepositoryAdapter(spec, root, source_symbols={"SPY": "SPY"}))
    registry.register_family("f2_native", lambda spec, root: NativeRepositoryAdapter(spec, root))
    return registry
