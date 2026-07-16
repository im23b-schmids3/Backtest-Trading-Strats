# Futures compatibility

The canonical reference registry is `src/research_pipeline/prop/contracts.py`.
It includes MBT, MET, MGC, SIL, MES, MNQ, and MCL with exchange, unit,
minimum tick, tick value, point value, session definition, and source metadata.
The registry hash is persisted with every Phase D run.

Sizing uses `abs(entry - stop) / minimum_tick * tick_value`, then floors the
number of contracts. It never rounds risk up. A contract is legal only when it
is active, mapped, within the provider micro/mini limit, and within shared
exposure.

The entries are reference metadata. Continuous-contract construction, rollover,
fills, and intrabar marked equity remain explicit limitations until a native
futures data adapter is added.
