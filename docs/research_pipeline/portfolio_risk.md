# Portfolio risk

Risk replay uses one shared ledger, one account balance and peak, one daily loss
state, Alpha Futures contract limits, duplicate-exposure limits, simultaneous
position limits, and the configured DLG/MLL rules. Contract denial, MLL/DLG
buffer, conflict, and session skips are explicit counters. Allocation policies
are named in the schema and replay is deterministic; Phase E does not optimize
their values.
