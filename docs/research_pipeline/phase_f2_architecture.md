# Phase F2 architecture

Phase F2 adds a real-mode boundary to the Phase F1 master controller. `dry_run`
continues to use the Phase F1 synthetic services. `real_run` requires an
explicit adapter family, local data availability, normalized trade artifacts,
real B.5 diagnostics, and the existing Phase C controller.

The adapter boundary is additive. Existing Fibonacci code, historical research,
providers, mappings, and outputs are not changed. Real artifacts live below the
master run directory and are referenced by hash in SQLite.

The included `f2_native_demo` adapter is a deterministic local-data integration
demonstration using the repository's cached parquet data. It is not a
performance claim and does not authorize trading.
