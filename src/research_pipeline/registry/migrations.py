SCHEMA_VERSION = 1


def apply_migrations(connection) -> None:
    connection.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
    row = connection.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
    if row is None:
        connection.execute("INSERT INTO schema_version(version) VALUES (?)", (SCHEMA_VERSION,))
    elif row[0] > SCHEMA_VERSION:
        raise RuntimeError(f"registry schema {row[0]} is newer than supported schema {SCHEMA_VERSION}")
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
        """
    )

