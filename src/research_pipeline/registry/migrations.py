# Preserve the Phase A/B registry schema version for backward compatibility.
# Phase C extension tables have their own version marker below.
SCHEMA_VERSION = 2
MAX_COMPATIBLE_SCHEMA_VERSION = 3


def apply_migrations(connection) -> None:
    connection.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
    row = connection.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
    if row is None:
        connection.execute("INSERT INTO schema_version(version) VALUES (?)", (SCHEMA_VERSION,))
    elif row[0] > MAX_COMPATIBLE_SCHEMA_VERSION:
        raise RuntimeError(f"registry schema {row[0]} is newer than supported schema {MAX_COMPATIBLE_SCHEMA_VERSION}")
    elif row[0] < SCHEMA_VERSION:
        connection.execute("UPDATE schema_version SET version=?", (SCHEMA_VERSION,))
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS strategies (
            strategy_id TEXT NOT NULL,
            version TEXT NOT NULL,
            specification_path TEXT NOT NULL,
            specification_hash TEXT NOT NULL,
            specification_json TEXT NOT NULL,
            current_phase TEXT NOT NULL,
            approval_status TEXT NOT NULL,
            terminal_status TEXT,
            parameters_frozen INTEGER NOT NULL DEFAULT 0,
            holdout_opened INTEGER NOT NULL DEFAULT 0,
            frozen_parameter_families_json TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY(strategy_id, version)
        );
        CREATE TABLE IF NOT EXISTS experiments (
            experiment_id TEXT PRIMARY KEY,
            strategy_id TEXT NOT NULL,
            strategy_version TEXT NOT NULL,
            phase TEXT NOT NULL,
            parameter_family TEXT,
            parameter_values_json TEXT NOT NULL,
            dataset_hash TEXT,
            code_commit TEXT,
            start_time TEXT NOT NULL,
            end_time TEXT,
            status TEXT NOT NULL,
            report_paths_json TEXT NOT NULL,
            FOREIGN KEY(strategy_id, strategy_version) REFERENCES strategies(strategy_id, version)
        );
        CREATE TABLE IF NOT EXISTS transitions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_id TEXT NOT NULL,
            strategy_version TEXT NOT NULL,
            old_state TEXT NOT NULL,
            new_state TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            reason TEXT NOT NULL,
            FOREIGN KEY(strategy_id, strategy_version) REFERENCES strategies(strategy_id, version)
        );
        CREATE TABLE IF NOT EXISTS budgets (
            strategy_id TEXT NOT NULL,
            strategy_version TEXT NOT NULL,
            limits_json TEXT NOT NULL,
            usage_json TEXT NOT NULL,
            PRIMARY KEY(strategy_id, strategy_version),
            FOREIGN KEY(strategy_id, strategy_version) REFERENCES strategies(strategy_id, version)
        );
        CREATE TABLE IF NOT EXISTS holdout_accesses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_id TEXT NOT NULL,
            strategy_version TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            phase TEXT NOT NULL,
            dataset_hash TEXT NOT NULL,
            access_reason TEXT NOT NULL,
            FOREIGN KEY(strategy_id, strategy_version) REFERENCES strategies(strategy_id, version)
        );
        CREATE TABLE IF NOT EXISTS splits (
            strategy_id TEXT NOT NULL,
            strategy_version TEXT NOT NULL,
            split_hash TEXT NOT NULL,
            split_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            invalidated_at TEXT,
            PRIMARY KEY(strategy_id, strategy_version),
            FOREIGN KEY(strategy_id, strategy_version) REFERENCES strategies(strategy_id, version)
        );
        CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_id TEXT NOT NULL,
            strategy_version TEXT NOT NULL,
            phase TEXT NOT NULL,
            decision_json TEXT NOT NULL,
            validation_status TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY(strategy_id, strategy_version) REFERENCES strategies(strategy_id, version)
        );
        CREATE TABLE IF NOT EXISTS failures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_id TEXT NOT NULL,
            strategy_version TEXT NOT NULL,
            phase TEXT NOT NULL,
            error_type TEXT NOT NULL,
            message TEXT NOT NULL,
            retry_count INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY(strategy_id, strategy_version) REFERENCES strategies(strategy_id, version)
        );
        CREATE INDEX IF NOT EXISTS idx_transitions_strategy ON transitions(strategy_id, strategy_version, id);
        CREATE INDEX IF NOT EXISTS idx_experiments_strategy ON experiments(strategy_id, strategy_version, start_time);
        CREATE TABLE IF NOT EXISTS verification_runs (
            verification_run_id TEXT PRIMARY KEY,
            strategy_id TEXT NOT NULL,
            strategy_version TEXT NOT NULL,
            implementation_commit TEXT,
            manifest_path TEXT NOT NULL,
            manifest_hash TEXT NOT NULL,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            status TEXT NOT NULL,
            outcome TEXT,
            result_json TEXT,
            FOREIGN KEY(strategy_id, strategy_version) REFERENCES strategies(strategy_id, version)
        );
        CREATE TABLE IF NOT EXISTS verification_checks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            verification_run_id TEXT NOT NULL,
            check_name TEXT NOT NULL,
            applicability TEXT NOT NULL,
            status TEXT NOT NULL,
            severity TEXT NOT NULL,
            observed_value TEXT NOT NULL,
            expected_value TEXT NOT NULL,
            tolerance TEXT,
            evidence_path TEXT,
            repair_eligibility INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY(verification_run_id) REFERENCES verification_runs(verification_run_id)
        );
        CREATE TABLE IF NOT EXISTS diagnostic_artifacts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            verification_run_id TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_hash TEXT NOT NULL,
            schema_version TEXT NOT NULL,
            row_count INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(verification_run_id) REFERENCES verification_runs(verification_run_id)
        );
        CREATE TABLE IF NOT EXISTS verification_repairs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            verification_run_id TEXT NOT NULL,
            failed_checks_json TEXT NOT NULL,
            codex_result_json TEXT,
            resulting_commit TEXT,
            rerun_result_json TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(verification_run_id) REFERENCES verification_runs(verification_run_id)
        );
        CREATE TABLE IF NOT EXISTS research_runs (
            run_id TEXT PRIMARY KEY,
            strategy_id TEXT NOT NULL,
            strategy_version TEXT NOT NULL,
            current_phase TEXT NOT NULL,
            status TEXT NOT NULL,
            root_path TEXT NOT NULL,
            scenario TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS research_baselines (
            strategy_id TEXT NOT NULL,
            strategy_version TEXT NOT NULL,
            experiment_id TEXT PRIMARY KEY,
            artifact_json TEXT NOT NULL,
            verification_outcome TEXT NOT NULL,
            gate_outcomes_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS research_rounds (
            round_id TEXT PRIMARY KEY,
            strategy_id TEXT NOT NULL,
            strategy_version TEXT NOT NULL,
            family TEXT NOT NULL,
            round_number INTEGER NOT NULL,
            proposed_values_json TEXT NOT NULL,
            experiments_json TEXT NOT NULL DEFAULT '[]',
            review_json TEXT,
            selected_value_json TEXT,
            status TEXT NOT NULL,
            reason TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS research_candidates (
            strategy_id TEXT NOT NULL,
            strategy_version TEXT NOT NULL,
            candidate_hash TEXT PRIMARY KEY,
            manifest_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS research_metric_citations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_id TEXT NOT NULL,
            strategy_version TEXT NOT NULL,
            phase TEXT NOT NULL,
            experiment_id TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            cited_value REAL NOT NULL,
            source_file TEXT NOT NULL,
            source_path TEXT NOT NULL,
            validation_status TEXT NOT NULL,
            reason TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS research_walk_forward (
            strategy_id TEXT NOT NULL,
            strategy_version TEXT NOT NULL,
            result_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY(strategy_id, strategy_version)
        );
        CREATE TABLE IF NOT EXISTS research_holdout (
            strategy_id TEXT NOT NULL,
            strategy_version TEXT NOT NULL,
            result_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY(strategy_id, strategy_version)
        );
        CREATE TABLE IF NOT EXISTS research_stress (
            strategy_id TEXT NOT NULL,
            strategy_version TEXT NOT NULL,
            result_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY(strategy_id, strategy_version)
        );
        CREATE TABLE IF NOT EXISTS research_throughput (
            strategy_id TEXT NOT NULL,
            strategy_version TEXT NOT NULL,
            result_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY(strategy_id, strategy_version)
        );
        CREATE TABLE IF NOT EXISTS research_final_reviews (
            strategy_id TEXT NOT NULL,
            strategy_version TEXT NOT NULL,
            result_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY(strategy_id, strategy_version)
        );
        CREATE TABLE IF NOT EXISTS research_journal (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_id TEXT NOT NULL,
            strategy_version TEXT NOT NULL,
            phase TEXT NOT NULL,
            entry_json TEXT NOT NULL,
            entry_markdown TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS prop_runs (
            run_id TEXT PRIMARY KEY,
            strategy_id TEXT NOT NULL,
            strategy_version TEXT NOT NULL,
            current_phase TEXT NOT NULL,
            status TEXT NOT NULL,
            root_path TEXT NOT NULL,
            scenario TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS prop_budgets (
            strategy_id TEXT NOT NULL,
            strategy_version TEXT NOT NULL,
            limits_json TEXT NOT NULL,
            usage_json TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY(strategy_id, strategy_version)
        );
        CREATE TABLE IF NOT EXISTS prop_rules (
            record_key TEXT PRIMARY KEY,
            strategy_id TEXT NOT NULL,
            strategy_version TEXT NOT NULL,
            provider TEXT NOT NULL,
            product TEXT NOT NULL,
            rule_hash TEXT NOT NULL,
            result_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS prop_contracts (
            record_key TEXT PRIMARY KEY,
            strategy_id TEXT NOT NULL,
            strategy_version TEXT NOT NULL,
            registry_hash TEXT NOT NULL,
            result_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS prop_mappings (
            record_key TEXT PRIMARY KEY,
            strategy_id TEXT NOT NULL,
            strategy_version TEXT NOT NULL,
            mapping_hash TEXT NOT NULL,
            result_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS prop_risk_runs (
            record_key TEXT PRIMARY KEY,
            strategy_id TEXT NOT NULL,
            strategy_version TEXT NOT NULL,
            result_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS prop_scenarios (
            record_key TEXT PRIMARY KEY,
            strategy_id TEXT NOT NULL,
            strategy_version TEXT NOT NULL,
            scenario_id TEXT NOT NULL,
            result_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS prop_accounts (
            record_key TEXT PRIMARY KEY,
            strategy_id TEXT NOT NULL,
            strategy_version TEXT NOT NULL,
            account_id TEXT NOT NULL,
            result_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS prop_account_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_id TEXT NOT NULL,
            strategy_version TEXT NOT NULL,
            account_id TEXT NOT NULL,
            event_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS prop_payouts (
            record_key TEXT PRIMARY KEY,
            strategy_id TEXT NOT NULL,
            strategy_version TEXT NOT NULL,
            account_id TEXT NOT NULL,
            payout_number INTEGER NOT NULL,
            result_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS prop_billing_events (
            record_key TEXT PRIMARY KEY,
            strategy_id TEXT NOT NULL,
            strategy_version TEXT NOT NULL,
            account_id TEXT NOT NULL,
            result_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS prop_economics (
            record_key TEXT PRIMARY KEY,
            strategy_id TEXT NOT NULL,
            strategy_version TEXT NOT NULL,
            scenario_id TEXT NOT NULL,
            result_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS prop_compliance (
            record_key TEXT PRIMARY KEY,
            strategy_id TEXT NOT NULL,
            strategy_version TEXT NOT NULL,
            scenario_id TEXT NOT NULL,
            result_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS prop_final_reviews (
            record_key TEXT PRIMARY KEY,
            strategy_id TEXT NOT NULL,
            strategy_version TEXT NOT NULL,
            classification TEXT NOT NULL,
            result_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS prop_journal (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_id TEXT NOT NULL,
            strategy_version TEXT NOT NULL,
            phase TEXT NOT NULL,
            entry_json TEXT NOT NULL,
            entry_markdown TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS portfolio_runs (
            portfolio_id TEXT NOT NULL,
            portfolio_version TEXT NOT NULL,
            current_phase TEXT NOT NULL,
            status TEXT NOT NULL,
            specification_hash TEXT NOT NULL,
            root_path TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY(portfolio_id, portfolio_version)
        );
        CREATE TABLE IF NOT EXISTS portfolio_specs (
            record_key TEXT PRIMARY KEY,
            portfolio_id TEXT NOT NULL,
            portfolio_version TEXT NOT NULL,
            specification_hash TEXT NOT NULL,
            result_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS portfolio_members (
            record_key TEXT PRIMARY KEY,
            portfolio_id TEXT NOT NULL,
            portfolio_version TEXT NOT NULL,
            strategy_id TEXT NOT NULL,
            candidate_hash TEXT NOT NULL,
            result_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS portfolio_candidates (
            record_key TEXT PRIMARY KEY,
            portfolio_id TEXT NOT NULL,
            portfolio_version TEXT NOT NULL,
            candidate_id TEXT NOT NULL,
            candidate_hash TEXT NOT NULL,
            result_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS portfolio_signals (
            record_key TEXT PRIMARY KEY,
            portfolio_id TEXT NOT NULL,
            portfolio_version TEXT NOT NULL,
            candidate_id TEXT NOT NULL,
            artifact_path TEXT NOT NULL,
            artifact_hash TEXT NOT NULL,
            result_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS portfolio_overlap_metrics (
            record_key TEXT PRIMARY KEY,
            portfolio_id TEXT NOT NULL,
            portfolio_version TEXT NOT NULL,
            candidate_id TEXT NOT NULL,
            result_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS portfolio_correlation_metrics (
            record_key TEXT PRIMARY KEY,
            portfolio_id TEXT NOT NULL,
            portfolio_version TEXT NOT NULL,
            candidate_id TEXT NOT NULL,
            result_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS portfolio_conflict_results (
            record_key TEXT PRIMARY KEY,
            portfolio_id TEXT NOT NULL,
            portfolio_version TEXT NOT NULL,
            candidate_id TEXT NOT NULL,
            result_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS portfolio_risk_runs (
            record_key TEXT PRIMARY KEY,
            portfolio_id TEXT NOT NULL,
            portfolio_version TEXT NOT NULL,
            candidate_id TEXT NOT NULL,
            result_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS portfolio_prop_scenarios (
            record_key TEXT PRIMARY KEY,
            portfolio_id TEXT NOT NULL,
            portfolio_version TEXT NOT NULL,
            candidate_id TEXT NOT NULL,
            result_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS portfolio_ablation_runs (
            record_key TEXT PRIMARY KEY,
            portfolio_id TEXT NOT NULL,
            portfolio_version TEXT NOT NULL,
            candidate_id TEXT NOT NULL,
            removed_strategy_id TEXT NOT NULL,
            result_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS portfolio_marginal_contributions (
            record_key TEXT PRIMARY KEY,
            portfolio_id TEXT NOT NULL,
            portfolio_version TEXT NOT NULL,
            candidate_id TEXT NOT NULL,
            strategy_id TEXT NOT NULL,
            result_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS portfolio_stress_results (
            record_key TEXT PRIMARY KEY,
            portfolio_id TEXT NOT NULL,
            portfolio_version TEXT NOT NULL,
            candidate_id TEXT NOT NULL,
            scenario TEXT NOT NULL,
            result_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS portfolio_final_reviews (
            record_key TEXT PRIMARY KEY,
            portfolio_id TEXT NOT NULL,
            portfolio_version TEXT NOT NULL,
            classification TEXT NOT NULL,
            result_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS portfolio_budgets (
            portfolio_id TEXT NOT NULL,
            portfolio_version TEXT NOT NULL,
            limits_json TEXT NOT NULL,
            usage_json TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY(portfolio_id, portfolio_version)
        );
        CREATE TABLE IF NOT EXISTS portfolio_journal (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            portfolio_id TEXT NOT NULL,
            portfolio_version TEXT NOT NULL,
            phase TEXT NOT NULL,
            entry_json TEXT NOT NULL,
            entry_markdown TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )
    connection.execute("CREATE TABLE IF NOT EXISTS research_schema_version (version INTEGER NOT NULL)")
    if connection.execute("SELECT version FROM research_schema_version LIMIT 1").fetchone() is None:
        connection.execute("INSERT INTO research_schema_version(version) VALUES (1)")
    # Phase C baseline results are one logical record per strategy version.
    # Add the natural-key index separately so databases created by the first
    # Phase C migration receive the same idempotency guarantee.
    connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS research_baselines_strategy_version ON research_baselines(strategy_id, strategy_version)")
    connection.execute("CREATE TABLE IF NOT EXISTS prop_schema_version (version INTEGER NOT NULL)")
    if connection.execute("SELECT version FROM prop_schema_version LIMIT 1").fetchone() is None:
        connection.execute("INSERT INTO prop_schema_version(version) VALUES (1)")
    connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS prop_rules_strategy_product ON prop_rules(strategy_id, strategy_version, provider, product)")
    connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS prop_contracts_strategy ON prop_contracts(strategy_id, strategy_version)")
    connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS prop_mappings_strategy ON prop_mappings(strategy_id, strategy_version)")
    connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS prop_final_reviews_strategy ON prop_final_reviews(strategy_id, strategy_version)")
    connection.commit()
