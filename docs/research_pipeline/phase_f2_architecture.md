# Phase F2 architecture

Phase F2 adds a real-mode boundary to the Phase F1 master controller. `dry_run`
continues to use the Phase F1 synthetic services. `real_run` requires an
explicit adapter family, local data availability, normalized trade artifacts,
real B.5 diagnostics, and the existing Phase C controller.

The adapter boundary is additive. Existing Fibonacci code, historical research,
providers, mappings, and outputs are not changed. Real artifacts live below the
master run directory and are referenced by hash in SQLite.

An external implementation result is not pipeline completion. After result
ingestion, the master run durably stops at `IMPLEMENTATION_VERIFICATION` with
the implementation artifacts persisted. A subsequent resume executes the
implementation and technical verification gates; only a verified B.5 result
may advance the strategy to baseline research. `COMPLETED` is reserved for the
end of the complete master pipeline (or an explicit rejection/abort terminal
record).

The included `f2_native_demo` adapter is a deterministic local-data integration
demonstration using the repository's cached parquet data. It is not a
performance claim and does not authorize trading.
