# RandomOpenTest reference pipeline

`RandomOpenTest` has two intentionally distinct registered families:

- `f2_random_open_test` is the existing one-hour same-bar compatibility
  fixture. Its historical fixture semantics are unchanged.
- `f2_random_open_reference` is the new fixed-quantity reference adapter. It
  uses a deterministic SHA-256 direction, fixed tick stop/target, shared
  compliance evaluation, instrument-specific execution costs, and session
  forced flattening.

The reference family is selected by the Phase B intake translator when the
description explicitly includes fixed quantity, stop ticks, target ticks, or
shared forced-flat behavior. This keeps old specifications and generated
outputs compatible.

The standard F1 commands are:

```powershell
python -m research_pipeline run .\examples\research_pipeline\random_open_test_intake.json `
  --mode dry_run `
  --dry-run

python -m research_pipeline status RUN_ID
python -m research_pipeline approve RUN_ID --decision APPROVE --note "reference fixture approval"
python -m research_pipeline resume RUN_ID
python -m research_pipeline report RUN_ID
python -m research_pipeline artifacts RUN_ID
```

The reference adapter itself is registered under
`f2_random_open_reference` and can be resolved through
`default_adapter_registry()`. It never submits broker orders. Reports include
the deterministic seed inputs, proposal/accept/block counts, compliance
decision hashes, cost-model hash, explicit PnL cost fields, forced-flat counts,
holding diagnostics, and artifact hashes.

This is a deterministic integration fixture, not a profitability test and not
a live-trading authorization.
