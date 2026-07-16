# Phase A architecture

Phase A is a deterministic control plane for research. It has no market-data
provider, backtest runner, optimizer, AI client, or web service. A strict
Pydantic schema describes a strategy, a state machine controls the order of
work, and SQLite records the resulting audit trail.

The controller delegates persistence to `Registry` and keeps execution behind
future interfaces. Registry writes are transactional and schema creation is
idempotent. The strategy specification hash covers all material fields except
approval timestamp and the hash itself. Approval is an immutable registry
event; a material change belongs in a new version.

