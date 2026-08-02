# Diagnostic file contracts

Diagnostic files may be JSON mappings or CSV tables. The JSON pack uses sections named `trades`, `exit_legs`, `scaling_samples`, `fee_reconciliation`, `trade_counts`, `causality`, `session_boundary`, `report_reconciliation`, `data_sources`, `replay_hashes`, and optional `lifecycle`. Trade, exit-leg, contract, lifecycle, and report rows are strict Pydantic records. Missing required sections stop continuation with exact section names in `blocking_issues`.
