# Strategy specification

Strategy specifications are YAML documents loaded into the strict
`StrategySpec` Pydantic model. Unknown fields are rejected. `strategy_id` is
filesystem-safe and `version` is always explicit.

Each parameter family declares its baseline, type, bounds or allowed values,
optimization order, round limit, mutability, and relationship to the
hypothesis. Immutable families cannot be consumed as research budget.

The canonical `specification_hash` is SHA-256 over normalized JSON. Approval
adds an audit timestamp but does not alter the material hash. Approved records
cannot be edited; a material change requires another version. See
`examples/research_pipeline/fibonacci_compatibility.yaml` for a deliberately
minimal compatibility fixture, not a reconstruction of historical Fib logic.

