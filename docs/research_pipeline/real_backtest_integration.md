# Real backtest integration

`NativeRepositoryAdapter` reads approved local parquet data, runs a deterministic
bounded implementation, and emits `BacktestRun`, `NormalizedTrade`, metrics,
dataset hashes, configuration hashes, JSONL trade artifacts, and an integrity
manifest. The F2 demo uses explicit completed-bar breakout timing; the shared
Fibonacci engine remains untouched.

Large trade streams are files, not SQLite rows. SQLite stores paths, hashes,
row counts, and compact normalized summaries.
