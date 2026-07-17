from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..enums import ApprovalStatus, PipelineState, TERMINAL_STATES
from ..errors import ImmutableSpecificationError, RegistryError, SplitConflictError
from ..schemas.budgets import BudgetUsage, ResearchBudget
from ..schemas.decisions import DecisionRecord
from ..schemas.splits import SplitDefinition
from ..schemas.strategy_spec import StrategySpec
from .database import Database

logger = logging.getLogger("research_pipeline")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Registry:
    def __init__(self, database: Database):
        self.database = database
        self.database.initialize()

    def _resolve(self, connection, strategy_id: str, version: str | None = None):
        if version:
            row = connection.execute("SELECT * FROM strategies WHERE strategy_id=? AND version=?", (strategy_id, version)).fetchone()
        else:
            row = connection.execute("SELECT * FROM strategies WHERE strategy_id=? ORDER BY created_at DESC, version DESC LIMIT 1", (strategy_id,)).fetchone()
        if row is None:
            raise RegistryError(f"strategy not found: {strategy_id}{'@' + version if version else ''}")
        return row

    @staticmethod
    def _row(row) -> dict[str, Any]:
        result = dict(row)
        for key in ("specification_json", "frozen_parameter_families_json"):
            if key in result:
                result[key] = json.loads(result[key])
        return result

    def register_strategy(self, specification: StrategySpec, specification_path: str, budget: ResearchBudget) -> dict:
        timestamp = now_iso()
        payload = json.dumps(specification.model_dump(mode="json"), sort_keys=True)
        with self.database.transaction() as connection:
            existing = connection.execute("SELECT 1 FROM strategies WHERE strategy_id=? AND version=?", (specification.strategy_id, specification.version)).fetchone()
            if existing:
                raise RegistryError(f"strategy version already registered: {specification.strategy_id}@{specification.version}")
            connection.execute("""INSERT INTO strategies(strategy_id,version,specification_path,specification_hash,specification_json,current_phase,approval_status,created_at,updated_at)
                VALUES(?,?,?,?,?,?,?,?,?)""", (specification.strategy_id, specification.version, str(Path(specification_path)), specification.specification_hash,
                payload, PipelineState.STRATEGY_DRAFT.value, specification.status.value, timestamp, timestamp))
            connection.execute("INSERT INTO budgets(strategy_id,strategy_version,limits_json,usage_json) VALUES(?,?,?,?)", (specification.strategy_id, specification.version,
                json.dumps(budget.model_dump(mode="json"), sort_keys=True), json.dumps(BudgetUsage().model_dump(mode="json"), sort_keys=True)))
        logger.info("strategy_registered strategy_id=%s version=%s", specification.strategy_id, specification.version)
        return self.get_strategy(specification.strategy_id, specification.version)

    def get_strategy(self, strategy_id: str, version: str | None = None) -> dict:
        with self.database.session() as connection:
            return self._row(self._resolve(connection, strategy_id, version))

    def get_specification(self, strategy_id: str, version: str | None = None) -> StrategySpec:
        row = self.get_strategy(strategy_id, version)
        return StrategySpec.model_validate(row["specification_json"])

    def list_strategies(self) -> list[dict]:
        with self.database.session() as connection:
            return [self._row(row) for row in connection.execute("SELECT * FROM strategies ORDER BY created_at, strategy_id, version")]

    def set_approval(self, strategy_id: str, version: str, status: ApprovalStatus, approved_at: str | None = None) -> None:
        with self.database.transaction() as connection:
            row = self._resolve(connection, strategy_id, version)
            if row["approval_status"] == ApprovalStatus.APPROVED.value:
                raise ImmutableSpecificationError("approved specification cannot be changed")
            connection.execute("UPDATE strategies SET approval_status=?, specification_json=?, updated_at=? WHERE strategy_id=? AND version=?", (status.value,
                json.dumps(self.get_specification(strategy_id, version).model_dump(mode="json"), sort_keys=True), now_iso(), strategy_id, version))

    def approve_specification(self, strategy_id: str, version: str, specification: StrategySpec) -> None:
        with self.database.transaction() as connection:
            row = self._resolve(connection, strategy_id, version)
            if row["approval_status"] == ApprovalStatus.APPROVED.value:
                raise ImmutableSpecificationError("approved specification cannot be changed")
            payload = json.dumps(specification.model_dump(mode="json"), sort_keys=True)
            connection.execute("UPDATE strategies SET approval_status=?, specification_json=?, specification_hash=?, updated_at=? WHERE strategy_id=? AND version=?", (ApprovalStatus.APPROVED.value, payload, specification.specification_hash, now_iso(), strategy_id, version))

    def transition(self, strategy_id: str, version: str, old_state: str, new_state: str, reason: str) -> None:
        from ..controller.state_machine import StateMachine

        StateMachine.validate_transition(PipelineState(old_state), PipelineState(new_state))
        timestamp = now_iso()
        with self.database.transaction() as connection:
            row = self._resolve(connection, strategy_id, version)
            if row["current_phase"] != old_state:
                raise RegistryError(f"state changed concurrently; expected {old_state}, found {row['current_phase']}")
            connection.execute("UPDATE strategies SET current_phase=?, terminal_status=?, updated_at=? WHERE strategy_id=? AND version=?", (new_state, new_state if new_state in {state.value for state in TERMINAL_STATES} else None, timestamp, strategy_id, version))
            connection.execute("INSERT INTO transitions(strategy_id,strategy_version,old_state,new_state,timestamp,reason) VALUES(?,?,?,?,?,?)", (strategy_id, version, old_state, new_state, timestamp, reason))
        logger.info("state_transition strategy_id=%s old_state=%s new_state=%s", strategy_id, old_state, new_state)

    def get_budget(self, strategy_id: str, version: str | None = None) -> dict:
        strategy = self.get_strategy(strategy_id, version)
        with self.database.session() as connection:
            row = connection.execute("SELECT limits_json,usage_json FROM budgets WHERE strategy_id=? AND strategy_version=?", (strategy["strategy_id"], strategy["version"])).fetchone()
            if row is None:
                raise RegistryError("budget record missing")
            return {"limits": json.loads(row[0]), "usage": json.loads(row[1])}

    def update_budget_usage(self, strategy_id: str, version: str, usage: BudgetUsage) -> None:
        with self.database.transaction() as connection:
            connection.execute("UPDATE budgets SET usage_json=? WHERE strategy_id=? AND strategy_version=?", (json.dumps(usage.model_dump(mode="json"), sort_keys=True), strategy_id, version))
        logger.info("budget_used strategy_id=%s total_backtests=%s", strategy_id, usage.total_backtests)

    def add_experiment(self, strategy_id: str, version: str | None, **values) -> str:
        strategy = self.get_strategy(strategy_id, version)
        experiment_id = values.pop("experiment_id", str(uuid.uuid4()))
        with self.database.transaction() as connection:
            connection.execute("""INSERT INTO experiments(experiment_id,strategy_id,strategy_version,phase,parameter_family,parameter_values_json,dataset_hash,code_commit,start_time,end_time,status,report_paths_json)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""", (experiment_id, strategy["strategy_id"], strategy["version"], values.get("phase", strategy["current_phase"]), values.get("parameter_family"), json.dumps(values.get("parameter_values", {}), sort_keys=True), values.get("dataset_hash"), values.get("code_commit"), values.get("start_time", now_iso()), values.get("end_time"), values.get("status", "PLANNED"), json.dumps(values.get("report_paths", []), sort_keys=True)))
        return experiment_id

    def create_split(self, strategy_id: str, version: str | None, split: SplitDefinition) -> None:
        strategy = self.get_strategy(strategy_id, version)
        with self.database.transaction() as connection:
            row = connection.execute("SELECT split_hash FROM splits WHERE strategy_id=? AND strategy_version=? AND invalidated_at IS NULL", (strategy["strategy_id"], strategy["version"])).fetchone()
            if row:
                if row[0] == split.split_hash:
                    return
                raise SplitConflictError("split definition changed; create a new strategy version or invalidate explicitly")
            connection.execute("INSERT INTO splits(strategy_id,strategy_version,split_hash,split_json,created_at) VALUES(?,?,?,?,?)", (strategy["strategy_id"], strategy["version"], split.split_hash, json.dumps(split.model_dump(mode="json"), sort_keys=True), now_iso()))

    def invalidate_split(self, strategy_id: str, reason: str, version: str | None = None) -> None:
        strategy = self.get_strategy(strategy_id, version)
        with self.database.transaction() as connection:
            cursor = connection.execute("UPDATE splits SET invalidated_at=? WHERE strategy_id=? AND strategy_version=? AND invalidated_at IS NULL", (now_iso(), strategy["strategy_id"], strategy["version"]))
            if cursor.rowcount == 0:
                raise SplitConflictError("no active split exists to invalidate")
            connection.execute("INSERT INTO failures(strategy_id,strategy_version,phase,error_type,message,retry_count,timestamp) VALUES(?,?,?,?,?,?,?)", (strategy["strategy_id"], strategy["version"], strategy["current_phase"], "SPLIT_INVALIDATED", reason, 0, now_iso()))

    def get_split(self, strategy_id: str, version: str | None = None) -> SplitDefinition | None:
        strategy = self.get_strategy(strategy_id, version)
        with self.database.session() as connection:
            row = connection.execute("SELECT split_json FROM splits WHERE strategy_id=? AND strategy_version=? AND invalidated_at IS NULL", (strategy["strategy_id"], strategy["version"])).fetchone()
            return SplitDefinition.model_validate(json.loads(row[0])) if row else None

    def freeze_parameter_families(self, strategy_id: str, version: str | None = None) -> None:
        strategy = self.get_strategy(strategy_id, version)
        spec = self.get_specification(strategy_id, strategy["version"])
        frozen = [family.name for family in spec.parameter_families if family.mutable]
        with self.database.transaction() as connection:
            connection.execute("UPDATE strategies SET parameters_frozen=1,frozen_parameter_families_json=?,updated_at=? WHERE strategy_id=? AND version=?", (json.dumps(frozen), now_iso(), strategy["strategy_id"], strategy["version"]))

    def count_holdout_accesses(self, strategy_id: str, version: str | None = None) -> int:
        strategy = self.get_strategy(strategy_id, version)
        with self.database.session() as connection:
            return connection.execute("SELECT COUNT(*) FROM holdout_accesses WHERE strategy_id=? AND strategy_version=?", (strategy["strategy_id"], strategy["version"])).fetchone()[0]

    def record_holdout_access(self, strategy_id: str, version: str, timestamp: str, phase: str, dataset_hash: str, reason: str) -> None:
        with self.database.transaction() as connection:
            connection.execute("INSERT INTO holdout_accesses(strategy_id,strategy_version,timestamp,phase,dataset_hash,access_reason) VALUES(?,?,?,?,?,?)", (strategy_id, version, timestamp, phase, dataset_hash, reason))
            row = connection.execute("SELECT usage_json FROM budgets WHERE strategy_id=? AND strategy_version=?", (strategy_id, version)).fetchone()
            usage = json.loads(row[0]); usage["holdout_accesses"] += 1
            connection.execute("UPDATE budgets SET usage_json=? WHERE strategy_id=? AND strategy_version=?", (json.dumps(usage, sort_keys=True), strategy_id, version))
            connection.execute("UPDATE strategies SET holdout_opened=1,parameters_frozen=1,updated_at=? WHERE strategy_id=? AND version=?", (now_iso(), strategy_id, version))
        logger.info("holdout_access strategy_id=%s", strategy_id)

    def record_decision(self, strategy_id: str, version: str | None, decision: DecisionRecord) -> int:
        strategy = self.get_strategy(strategy_id, version)
        if decision.strategy_id != strategy["strategy_id"]:
            raise RegistryError("decision strategy_id does not match registry record")
        with self.database.transaction() as connection:
            cursor = connection.execute("INSERT INTO decisions(strategy_id,strategy_version,phase,decision_json,validation_status,timestamp) VALUES(?,?,?,?,?,?)", (strategy["strategy_id"], strategy["version"], decision.phase.value, json.dumps(decision.model_dump(mode="json"), sort_keys=True), "VALID", now_iso()))
            return cursor.lastrowid

    def record_failure(self, strategy_id: str, version: str | None, phase: str, error_type: str, message: str, retry_count: int = 0) -> None:
        strategy = self.get_strategy(strategy_id, version)
        with self.database.transaction() as connection:
            connection.execute("INSERT INTO failures(strategy_id,strategy_version,phase,error_type,message,retry_count,timestamp) VALUES(?,?,?,?,?,?,?)", (strategy["strategy_id"], strategy["version"], phase, error_type, message, retry_count, now_iso()))
        logger.error("registry_failure strategy_id=%s phase=%s type=%s", strategy_id, phase, error_type)

    def history(self, strategy_id: str, version: str | None = None) -> dict[str, list[dict]]:
        strategy = self.get_strategy(strategy_id, version)
        with self.database.session() as connection:
            result = {}
            for table in ("transitions", "experiments", "holdout_accesses", "decisions", "failures"):
                rows = connection.execute(f"SELECT * FROM {table} WHERE strategy_id=? AND strategy_version=? ORDER BY rowid", (strategy["strategy_id"], strategy["version"])).fetchall()
                result[table] = [dict(row) for row in rows]
            return result

    def record_verification(self, result: dict, manifest: dict, artifact_rows: list[dict] | None = None) -> dict:
        """Persist a B.5 result idempotently, including its check evidence."""
        strategy = self.get_strategy(result["strategy_id"], result["strategy_version"])
        checks = result.get("checks", [])
        with self.database.transaction() as connection:
            existing = connection.execute("SELECT result_json FROM verification_runs WHERE verification_run_id=?", (result["verification_run_id"],)).fetchone()
            if existing:
                return json.loads(existing[0])
            connection.execute("""INSERT INTO verification_runs(verification_run_id,strategy_id,strategy_version,implementation_commit,manifest_path,manifest_hash,started_at,ended_at,status,outcome,result_json)
                VALUES(?,?,?,?,?,?,?,?,?,?,?)""", (result["verification_run_id"], strategy["strategy_id"], strategy["version"], manifest.get("implementation_commit"), manifest.get("manifest_path", ""), manifest["manifest_hash"], result["timestamp"], result["timestamp"], "COMPLETED", result["outcome"], json.dumps(result, sort_keys=True)))
            for check in checks:
                connection.execute("""INSERT INTO verification_checks(verification_run_id,check_name,applicability,status,severity,observed_value,expected_value,tolerance,evidence_path,repair_eligibility)
                    VALUES(?,?,?,?,?,?,?,?,?,?)""", (result["verification_run_id"], check["check_name"], check.get("applicability", "mandatory"), check["status"], check.get("severity", "blocking"), json.dumps(check.get("observed_value"), sort_keys=True), json.dumps(check.get("expected_value"), sort_keys=True), str(check.get("tolerance")) if check.get("tolerance") is not None else None, check.get("evidence_path"), int(check.get("repair_eligible", False))))
            for artifact in artifact_rows or []:
                connection.execute("INSERT INTO diagnostic_artifacts(verification_run_id,file_path,file_hash,schema_version,row_count,created_at) VALUES(?,?,?,?,?,?)", (result["verification_run_id"], artifact["file_path"], artifact["file_hash"], artifact.get("schema_version", "1"), artifact.get("row_count", 0), artifact.get("created_at", now_iso())))
        return result

    def get_verification(self, strategy_id: str, verification_run_id: str | None = None) -> dict | None:
        strategy = self.get_strategy(strategy_id)
        with self.database.session() as connection:
            query = "SELECT result_json FROM verification_runs WHERE strategy_id=? AND strategy_version=?"
            params: list = [strategy["strategy_id"], strategy["version"]]
            if verification_run_id:
                query += " AND verification_run_id=?"; params.append(verification_run_id)
            query += " ORDER BY started_at DESC LIMIT 1"
            row = connection.execute(query, params).fetchone()
            return json.loads(row[0]) if row else None

    def has_verified_verification(self, strategy_id: str, version: str | None = None) -> bool:
        strategy = self.get_strategy(strategy_id, version)
        with self.database.session() as connection:
            return connection.execute("SELECT 1 FROM verification_runs WHERE strategy_id=? AND strategy_version=? AND outcome='VERIFIED' LIMIT 1", (strategy["strategy_id"], strategy["version"])).fetchone() is not None

    # Phase C research records. These methods are intentionally small,
    # idempotent persistence primitives; policy remains in PhaseCService.
    def _strategy_key(self, strategy_id: str, version: str | None = None) -> tuple[str, str]:
        strategy = self.get_strategy(strategy_id, version)
        return strategy["strategy_id"], strategy["version"]

    def research_run(self, run_id: str, strategy_id: str, version: str, phase: str, root_path: str, scenario: str | None = None) -> dict:
        timestamp = now_iso()
        with self.database.transaction() as connection:
            connection.execute("INSERT OR IGNORE INTO research_runs(run_id,strategy_id,strategy_version,current_phase,status,root_path,scenario,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)", (run_id, strategy_id, version, phase, "RUNNING", root_path, scenario, timestamp, timestamp))
        with self.database.session() as connection:
            row = connection.execute("SELECT * FROM research_runs WHERE run_id=?", (run_id,)).fetchone()
            return dict(row)

    def save_research_json(self, table: str, strategy_id: str, version: str, payload: dict) -> None:
        allowed = {"research_walk_forward", "research_holdout", "research_stress", "research_throughput", "research_final_reviews"}
        if table not in allowed:
            raise RegistryError(f"unsupported research result table: {table}")
        with self.database.transaction() as connection:
            connection.execute(f"INSERT INTO {table}(strategy_id,strategy_version,result_json,created_at) VALUES(?,?,?,?) ON CONFLICT(strategy_id,strategy_version) DO UPDATE SET result_json=excluded.result_json", (strategy_id, version, json.dumps(payload, sort_keys=True), now_iso()))

    def get_research_json(self, table: str, strategy_id: str, version: str | None = None) -> dict | None:
        allowed = {"research_walk_forward", "research_holdout", "research_stress", "research_throughput", "research_final_reviews"}
        if table not in allowed:
            raise RegistryError(f"unsupported research result table: {table}")
        sid, ver = self._strategy_key(strategy_id, version)
        with self.database.session() as connection:
            row = connection.execute(f"SELECT result_json FROM {table} WHERE strategy_id=? AND strategy_version=?", (sid, ver)).fetchone()
            return json.loads(row[0]) if row else None

    def save_baseline(self, strategy_id: str, version: str, experiment_id: str, artifact: dict, verification_outcome: str, gates: list[dict]) -> None:
        with self.database.transaction() as connection:
            connection.execute("INSERT INTO research_baselines(strategy_id,strategy_version,experiment_id,artifact_json,verification_outcome,gate_outcomes_json,created_at) VALUES(?,?,?,?,?,?,?) ON CONFLICT(strategy_id,strategy_version) DO UPDATE SET experiment_id=excluded.experiment_id,artifact_json=excluded.artifact_json,verification_outcome=excluded.verification_outcome,gate_outcomes_json=excluded.gate_outcomes_json", (strategy_id, version, experiment_id, json.dumps(artifact, sort_keys=True), verification_outcome, json.dumps(gates, sort_keys=True), now_iso()))

    def get_baseline(self, strategy_id: str, version: str | None = None) -> dict | None:
        sid, ver = self._strategy_key(strategy_id, version)
        with self.database.session() as connection:
            row = connection.execute("SELECT * FROM research_baselines WHERE strategy_id=? AND strategy_version=?", (sid, ver)).fetchone()
            if not row: return None
            result = dict(row); result["artifact_json"] = json.loads(result["artifact_json"]); result["gate_outcomes_json"] = json.loads(result["gate_outcomes_json"]); return result

    def save_research_round(self, round_id: str, strategy_id: str, version: str, family: str, number: int, proposed: list, status: str = "PROPOSED", reason: str = "") -> None:
        timestamp = now_iso()
        with self.database.transaction() as connection:
            connection.execute("INSERT OR IGNORE INTO research_rounds(round_id,strategy_id,strategy_version,family,round_number,proposed_values_json,status,reason,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)", (round_id, strategy_id, version, family, number, json.dumps(proposed, sort_keys=True), status, reason, timestamp, timestamp))

    def update_research_round(self, round_id: str, *, experiments: list | None = None, review: dict | None = None, selected_value: Any = None, status: str | None = None, reason: str | None = None) -> None:
        with self.database.transaction() as connection:
            row = connection.execute("SELECT * FROM research_rounds WHERE round_id=?", (round_id,)).fetchone()
            if not row: raise RegistryError(f"research round not found: {round_id}")
            connection.execute("UPDATE research_rounds SET experiments_json=?,review_json=?,selected_value_json=?,status=?,reason=?,updated_at=? WHERE round_id=?", (json.dumps(experiments if experiments is not None else json.loads(row["experiments_json"]), sort_keys=True), json.dumps(review, sort_keys=True) if review is not None else row["review_json"], json.dumps(selected_value, sort_keys=True) if selected_value is not None else row["selected_value_json"], status or row["status"], reason if reason is not None else row["reason"], now_iso(), round_id))

    def get_research_round(self, round_id: str) -> dict | None:
        with self.database.session() as connection:
            row = connection.execute("SELECT * FROM research_rounds WHERE round_id=?", (round_id,)).fetchone()
            if not row: return None
            result = dict(row)
            for key in ("proposed_values_json", "experiments_json", "review_json", "selected_value_json"):
                if result[key] is not None: result[key] = json.loads(result[key])
            return result

    def list_research_rounds(self, strategy_id: str, version: str | None = None) -> list[dict]:
        sid, ver = self._strategy_key(strategy_id, version)
        with self.database.session() as connection:
            return [dict(row) for row in connection.execute("SELECT * FROM research_rounds WHERE strategy_id=? AND strategy_version=? ORDER BY round_number,created_at", (sid, ver)).fetchall()]

    def save_candidate(self, strategy_id: str, version: str, candidate_hash: str, manifest: dict) -> None:
        with self.database.transaction() as connection:
            connection.execute("INSERT OR IGNORE INTO research_candidates(strategy_id,strategy_version,candidate_hash,manifest_json,created_at) VALUES(?,?,?,?,?)", (strategy_id, version, candidate_hash, json.dumps(manifest, sort_keys=True), now_iso()))

    def get_candidate(self, strategy_id: str, version: str | None = None) -> dict | None:
        sid, ver = self._strategy_key(strategy_id, version)
        with self.database.session() as connection:
            row = connection.execute("SELECT * FROM research_candidates WHERE strategy_id=? AND strategy_version=? ORDER BY created_at DESC LIMIT 1", (sid, ver)).fetchone()
            if not row:
                return None
            result = dict(row)
            result["manifest_json"] = json.loads(result["manifest_json"])
            return result

    def record_metric_citation(self, strategy_id: str, version: str, phase: str, citation: dict, valid: bool, reason: str) -> None:
        with self.database.transaction() as connection:
            connection.execute("INSERT INTO research_metric_citations(strategy_id,strategy_version,phase,experiment_id,metric_name,cited_value,source_file,source_path,validation_status,reason,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)", (strategy_id, version, phase, citation["experiment_id"], citation["metric_name"], citation["value"], citation["source_file"], citation["source_path"], "VALID" if valid else "INVALID", reason, now_iso()))

    def journal(self, strategy_id: str, version: str | None = None) -> list[dict]:
        sid, ver = self._strategy_key(strategy_id, version)
        with self.database.session() as connection:
            return [dict(row) for row in connection.execute("SELECT * FROM research_journal WHERE strategy_id=? AND strategy_version=? ORDER BY id", (sid, ver)).fetchall()]

    def add_journal_entry(self, strategy_id: str, version: str, phase: str, entry: dict, markdown: str) -> None:
        with self.database.transaction() as connection:
            connection.execute("INSERT INTO research_journal(strategy_id,strategy_version,phase,entry_json,entry_markdown,created_at) VALUES(?,?,?,?,?,?)", (strategy_id, version, phase, json.dumps(entry, sort_keys=True), markdown, now_iso()))

    # Phase D records use hashed JSON artifacts rather than storing trade
    # journals directly in SQLite. The table name is allow-listed so callers
    # cannot turn this helper into arbitrary SQL execution.
    def prop_run(self, run_id: str, strategy_id: str, version: str, phase: str, root_path: str, scenario: str | None = None) -> dict:
        timestamp = now_iso()
        with self.database.transaction() as connection:
            connection.execute("INSERT OR IGNORE INTO prop_runs(run_id,strategy_id,strategy_version,current_phase,status,root_path,scenario,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)", (run_id, strategy_id, version, phase, "RUNNING", root_path, scenario, timestamp, timestamp))
        with self.database.session() as connection:
            row = connection.execute("SELECT * FROM prop_runs WHERE run_id=?", (run_id,)).fetchone()
            return dict(row)

    def update_prop_run(self, run_id: str, phase: str | None = None, status: str | None = None) -> None:
        with self.database.transaction() as connection:
            row = connection.execute("SELECT * FROM prop_runs WHERE run_id=?", (run_id,)).fetchone()
            if not row: raise RegistryError(f"prop run not found: {run_id}")
            connection.execute("UPDATE prop_runs SET current_phase=?,status=?,updated_at=? WHERE run_id=?", (phase or row["current_phase"], status or row["status"], now_iso(), run_id))

    def save_prop_budget(self, strategy_id: str, version: str, limits: dict, usage: dict) -> None:
        with self.database.transaction() as connection:
            connection.execute("INSERT INTO prop_budgets(strategy_id,strategy_version,limits_json,usage_json,updated_at) VALUES(?,?,?,?,?) ON CONFLICT(strategy_id,strategy_version) DO UPDATE SET limits_json=excluded.limits_json, usage_json=excluded.usage_json, updated_at=excluded.updated_at", (strategy_id, version, json.dumps(limits, sort_keys=True), json.dumps(usage, sort_keys=True), now_iso()))

    def get_prop_budget(self, strategy_id: str, version: str | None = None) -> dict | None:
        sid, ver = self._strategy_key(strategy_id, version)
        with self.database.session() as connection:
            row = connection.execute("SELECT * FROM prop_budgets WHERE strategy_id=? AND strategy_version=?", (sid, ver)).fetchone()
            if not row: return None
            result = dict(row); result["limits_json"] = json.loads(result["limits_json"]); result["usage_json"] = json.loads(result["usage_json"]); return result

    def save_prop_record(self, table: str, record_key: str, strategy_id: str, version: str, payload: dict, **fields: Any) -> None:
        allowed = {"prop_rules", "prop_contracts", "prop_mappings", "prop_risk_runs", "prop_scenarios", "prop_accounts", "prop_payouts", "prop_billing_events", "prop_economics", "prop_compliance", "prop_final_reviews"}
        if table not in allowed: raise RegistryError(f"unsupported prop table: {table}")
        columns = ["record_key", "strategy_id", "strategy_version"] + list(fields) + ["result_json", "created_at"]
        values = [record_key, strategy_id, version] + list(fields.values()) + [json.dumps(payload, sort_keys=True), now_iso()]
        placeholders = ",".join("?" for _ in values)
        updates = ",".join(f"{column}=excluded.{column}" for column in columns[1:] if column != "created_at")
        with self.database.transaction() as connection:
            connection.execute(f"INSERT INTO {table}({','.join(columns)}) VALUES({placeholders}) ON CONFLICT(record_key) DO UPDATE SET {updates}", values)

    def get_prop_record(self, table: str, strategy_id: str, version: str | None = None, record_key: str | None = None) -> dict | None:
        allowed = {"prop_rules", "prop_contracts", "prop_mappings", "prop_risk_runs", "prop_scenarios", "prop_accounts", "prop_payouts", "prop_billing_events", "prop_economics", "prop_compliance", "prop_final_reviews"}
        if table not in allowed: raise RegistryError(f"unsupported prop table: {table}")
        sid, ver = self._strategy_key(strategy_id, version)
        with self.database.session() as connection:
            query = f"SELECT * FROM {table} WHERE strategy_id=? AND strategy_version=?"
            params: list[Any] = [sid, ver]
            if record_key: query += " AND record_key=?"; params.append(record_key)
            query += " ORDER BY created_at DESC LIMIT 1"
            row = connection.execute(query, params).fetchone()
            if not row: return None
            result = dict(row); result["result_json"] = json.loads(result["result_json"]); return result

    def list_prop_records(self, table: str, strategy_id: str, version: str | None = None) -> list[dict]:
        allowed = {"prop_scenarios", "prop_economics", "prop_accounts", "prop_payouts", "prop_billing_events"}
        if table not in allowed: raise RegistryError(f"unsupported prop table: {table}")
        sid, ver = self._strategy_key(strategy_id, version)
        with self.database.session() as connection:
            result = []
            for row in connection.execute(f"SELECT * FROM {table} WHERE strategy_id=? AND strategy_version=? ORDER BY created_at", (sid, ver)).fetchall():
                item = dict(row); item["result_json"] = json.loads(item["result_json"]); result.append(item)
            return result

    def add_prop_event(self, strategy_id: str, version: str, account_id: str, event: dict) -> None:
        with self.database.transaction() as connection:
            connection.execute("INSERT INTO prop_account_events(strategy_id,strategy_version,account_id,event_json,created_at) VALUES(?,?,?,?,?)", (strategy_id, version, account_id, json.dumps(event, sort_keys=True), now_iso()))

    def prop_journal(self, strategy_id: str, version: str | None = None) -> list[dict]:
        sid, ver = self._strategy_key(strategy_id, version)
        with self.database.session() as connection:
            return [dict(row) for row in connection.execute("SELECT * FROM prop_journal WHERE strategy_id=? AND strategy_version=? ORDER BY id", (sid, ver)).fetchall()]

    def add_prop_journal_entry(self, strategy_id: str, version: str, phase: str, entry: dict, markdown: str) -> None:
        with self.database.transaction() as connection:
            connection.execute("INSERT INTO prop_journal(strategy_id,strategy_version,phase,entry_json,entry_markdown,created_at) VALUES(?,?,?,?,?,?)", (strategy_id, version, phase, json.dumps(entry, sort_keys=True), markdown, now_iso()))

    # Phase E portfolio records. Large signal streams are referenced by path
    # and hash; SQLite stores deterministic summaries and decisions only.
    def portfolio_run(self, portfolio_id: str, version: str, phase: str, status: str, specification_hash: str, root_path: str) -> dict:
        timestamp = now_iso()
        with self.database.transaction() as connection:
            connection.execute("INSERT OR IGNORE INTO portfolio_runs(portfolio_id,portfolio_version,current_phase,status,specification_hash,root_path,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)", (portfolio_id, version, phase, status, specification_hash, root_path, timestamp, timestamp))
        with self.database.session() as connection:
            row = connection.execute("SELECT * FROM portfolio_runs WHERE portfolio_id=? AND portfolio_version=?", (portfolio_id, version)).fetchone()
            if not row: raise RegistryError(f"portfolio run not found: {portfolio_id}/{version}")
            return dict(row)

    def update_portfolio_run(self, portfolio_id: str, version: str, phase: str | None = None, status: str | None = None) -> None:
        with self.database.transaction() as connection:
            row = connection.execute("SELECT * FROM portfolio_runs WHERE portfolio_id=? AND portfolio_version=?", (portfolio_id, version)).fetchone()
            if not row: raise RegistryError(f"portfolio run not found: {portfolio_id}/{version}")
            connection.execute("UPDATE portfolio_runs SET current_phase=?,status=?,updated_at=? WHERE portfolio_id=? AND portfolio_version=?", (phase or row["current_phase"], status or row["status"], now_iso(), portfolio_id, version))

    def _portfolio_key(self, portfolio_id: str, version: str | None = None) -> tuple[str, str]:
        with self.database.session() as connection:
            if version is None:
                row = connection.execute("SELECT portfolio_version FROM portfolio_runs WHERE portfolio_id=? ORDER BY created_at DESC LIMIT 1", (portfolio_id,)).fetchone()
                if not row: raise RegistryError(f"portfolio not found: {portfolio_id}")
                return portfolio_id, row["portfolio_version"]
        return portfolio_id, version

    def save_portfolio_record(self, table: str, record_key: str, portfolio_id: str, version: str, payload: dict, **fields: Any) -> None:
        allowed = {"portfolio_specs", "portfolio_members", "portfolio_candidates", "portfolio_signals", "portfolio_overlap_metrics", "portfolio_correlation_metrics", "portfolio_conflict_results", "portfolio_risk_runs", "portfolio_prop_scenarios", "portfolio_ablation_runs", "portfolio_marginal_contributions", "portfolio_stress_results", "portfolio_final_reviews"}
        if table not in allowed: raise RegistryError(f"unsupported portfolio table: {table}")
        columns = ["record_key", "portfolio_id", "portfolio_version"] + list(fields) + ["result_json", "created_at"]
        values = [record_key, portfolio_id, version] + list(fields.values()) + [json.dumps(payload, sort_keys=True), now_iso()]
        placeholders = ",".join("?" for _ in values)
        updates = ",".join(f"{column}=excluded.{column}" for column in columns[1:] if column != "created_at")
        with self.database.transaction() as connection:
            connection.execute(f"INSERT INTO {table}({','.join(columns)}) VALUES({placeholders}) ON CONFLICT(record_key) DO UPDATE SET {updates}", values)

    def get_portfolio_record(self, table: str, portfolio_id: str, version: str | None = None, record_key: str | None = None) -> dict | None:
        allowed = {"portfolio_specs", "portfolio_members", "portfolio_candidates", "portfolio_signals", "portfolio_overlap_metrics", "portfolio_correlation_metrics", "portfolio_conflict_results", "portfolio_risk_runs", "portfolio_prop_scenarios", "portfolio_ablation_runs", "portfolio_marginal_contributions", "portfolio_stress_results", "portfolio_final_reviews"}
        if table not in allowed: raise RegistryError(f"unsupported portfolio table: {table}")
        pid, ver = self._portfolio_key(portfolio_id, version)
        with self.database.session() as connection:
            query = f"SELECT * FROM {table} WHERE portfolio_id=? AND portfolio_version=?"; params: list[Any] = [pid, ver]
            if record_key: query += " AND record_key=?"; params.append(record_key)
            query += " ORDER BY created_at DESC LIMIT 1"
            row = connection.execute(query, params).fetchone()
            if not row: return None
            result = dict(row); result["result_json"] = json.loads(result["result_json"]); return result

    def list_portfolio_records(self, table: str, portfolio_id: str, version: str | None = None) -> list[dict]:
        allowed = {"portfolio_members", "portfolio_candidates", "portfolio_signals", "portfolio_overlap_metrics", "portfolio_correlation_metrics", "portfolio_conflict_results", "portfolio_risk_runs", "portfolio_prop_scenarios", "portfolio_ablation_runs", "portfolio_marginal_contributions", "portfolio_stress_results"}
        if table not in allowed: raise RegistryError(f"unsupported portfolio table: {table}")
        pid, ver = self._portfolio_key(portfolio_id, version)
        with self.database.session() as connection:
            result = []
            for row in connection.execute(f"SELECT * FROM {table} WHERE portfolio_id=? AND portfolio_version=? ORDER BY created_at", (pid, ver)).fetchall():
                item = dict(row); item["result_json"] = json.loads(item["result_json"]); result.append(item)
            return result

    def save_portfolio_budget(self, portfolio_id: str, version: str, limits: dict, usage: dict) -> None:
        with self.database.transaction() as connection:
            connection.execute("INSERT INTO portfolio_budgets(portfolio_id,portfolio_version,limits_json,usage_json,updated_at) VALUES(?,?,?,?,?) ON CONFLICT(portfolio_id,portfolio_version) DO UPDATE SET limits_json=excluded.limits_json,usage_json=excluded.usage_json,updated_at=excluded.updated_at", (portfolio_id, version, json.dumps(limits, sort_keys=True), json.dumps(usage, sort_keys=True), now_iso()))

    def get_portfolio_budget(self, portfolio_id: str, version: str | None = None) -> dict | None:
        pid, ver = self._portfolio_key(portfolio_id, version)
        with self.database.session() as connection:
            row = connection.execute("SELECT * FROM portfolio_budgets WHERE portfolio_id=? AND portfolio_version=?", (pid, ver)).fetchone()
            if not row: return None
            result = dict(row); result["limits_json"] = json.loads(result["limits_json"]); result["usage_json"] = json.loads(result["usage_json"]); return result

    def add_portfolio_journal_entry(self, portfolio_id: str, version: str, phase: str, entry: dict, markdown: str) -> None:
        with self.database.transaction() as connection:
            connection.execute("INSERT INTO portfolio_journal(portfolio_id,portfolio_version,phase,entry_json,entry_markdown,created_at) VALUES(?,?,?,?,?,?)", (portfolio_id, version, phase, json.dumps(entry, sort_keys=True), markdown, now_iso()))

    def portfolio_journal(self, portfolio_id: str, version: str | None = None) -> list[dict]:
        pid, ver = self._portfolio_key(portfolio_id, version)
        with self.database.session() as connection:
            return [dict(row) for row in connection.execute("SELECT * FROM portfolio_journal WHERE portfolio_id=? AND portfolio_version=? ORDER BY id", (pid, ver)).fetchall()]

    # Phase F1 master-run persistence. These are orchestration records only;
    # phase-specific evidence remains in the existing Phase A-E tables.
    def save_master_run(self, run_id: str, strategy_id: str, strategy_version: str | None, input_hash: str,
                        current_step: str, outcome: str, approval_status: str, root_path: str,
                        resume_state: dict[str, Any]) -> dict:
        timestamp = now_iso()
        with self.database.transaction() as connection:
            connection.execute("""INSERT INTO master_runs(run_id,strategy_id,strategy_version,input_hash,current_step,outcome,approval_status,root_path,resume_state_json,created_at,updated_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(run_id) DO UPDATE SET strategy_id=excluded.strategy_id,
                strategy_version=excluded.strategy_version,input_hash=excluded.input_hash,current_step=excluded.current_step,
                outcome=excluded.outcome,approval_status=excluded.approval_status,root_path=excluded.root_path,
                resume_state_json=excluded.resume_state_json,updated_at=excluded.updated_at""",
                (run_id, strategy_id, strategy_version, input_hash, current_step, outcome, approval_status, root_path,
                 json.dumps(resume_state, sort_keys=True), timestamp, timestamp))
        return self.get_master_run(run_id)

    def get_master_run(self, run_id: str) -> dict:
        with self.database.session() as connection:
            row = connection.execute("SELECT * FROM master_runs WHERE run_id=?", (run_id,)).fetchone()
            if not row:
                raise RegistryError(f"master run not found: {run_id}")
            result = dict(row); result["resume_state_json"] = json.loads(result["resume_state_json"]); return result

    def update_master_run(self, run_id: str, **values: Any) -> dict:
        current = self.get_master_run(run_id)
        values["updated_at"] = now_iso()
        if "resume_state" in values:
            values["resume_state_json"] = json.dumps(values.pop("resume_state"), sort_keys=True)
        allowed = {"strategy_id", "strategy_version", "current_step", "outcome", "approval_status", "root_path", "resume_state_json", "updated_at"}
        values = {key: value for key, value in values.items() if key in allowed}
        if values:
            assignments = ",".join(f"{key}=?" for key in values)
            with self.database.transaction() as connection:
                connection.execute(f"UPDATE master_runs SET {assignments} WHERE run_id=?", (*values.values(), run_id))
        return self.get_master_run(run_id)

    def save_master_phase_result(self, run_id: str, phase: str, status: str, result: dict,
                                 artifact_paths: list[str], result_hash: str, started_at: str,
                                 ended_at: str, duration_ms: int) -> dict:
        with self.database.transaction() as connection:
            connection.execute("""INSERT INTO master_phase_results(run_id,phase,status,result_json,artifact_paths_json,result_hash,started_at,ended_at,duration_ms)
                VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(run_id,phase) DO UPDATE SET status=excluded.status,
                result_json=excluded.result_json,artifact_paths_json=excluded.artifact_paths_json,result_hash=excluded.result_hash,
                started_at=excluded.started_at,ended_at=excluded.ended_at,duration_ms=excluded.duration_ms""",
                (run_id, phase, status, json.dumps(result, sort_keys=True, default=str), json.dumps(artifact_paths, sort_keys=True), result_hash, started_at, ended_at, duration_ms))
        return self.get_master_phase_result(run_id, phase)

    def get_master_phase_result(self, run_id: str, phase: str) -> dict | None:
        with self.database.session() as connection:
            row = connection.execute("SELECT * FROM master_phase_results WHERE run_id=? AND phase=?", (run_id, phase)).fetchone()
            if not row: return None
            result = dict(row); result["result_json"] = json.loads(result["result_json"]); result["artifact_paths_json"] = json.loads(result["artifact_paths_json"]); return result

    def master_phase_results(self, run_id: str) -> list[dict]:
        with self.database.session() as connection:
            rows = connection.execute("SELECT * FROM master_phase_results WHERE run_id=? ORDER BY started_at, phase", (run_id,)).fetchall()
            result = []
            for row in rows:
                item = dict(row); item["result_json"] = json.loads(item["result_json"]); item["artifact_paths_json"] = json.loads(item["artifact_paths_json"]); result.append(item)
            return result

    def add_master_journal(self, run_id: str, phase: str, event: str, payload: dict[str, Any]) -> None:
        with self.database.transaction() as connection:
            connection.execute("INSERT INTO master_journal(run_id,phase,event,payload_json,created_at) VALUES(?,?,?,?,?)", (run_id, phase, event, json.dumps(payload, sort_keys=True, default=str), now_iso()))

    def master_journal(self, run_id: str) -> list[dict]:
        with self.database.session() as connection:
            rows = connection.execute("SELECT * FROM master_journal WHERE run_id=? ORDER BY id", (run_id,)).fetchall()
            result = []
            for row in rows:
                item = dict(row); item["payload_json"] = json.loads(item["payload_json"]); result.append(item)
            return result

    def add_master_artifact(self, run_id: str, phase: str, artifact_path: str, artifact_hash: str, artifact_type: str) -> None:
        with self.database.transaction() as connection:
            connection.execute("INSERT OR IGNORE INTO master_artifacts(run_id,phase,artifact_path,artifact_hash,artifact_type,created_at) VALUES(?,?,?,?,?,?)", (run_id, phase, artifact_path, artifact_hash, artifact_type, now_iso()))

    def master_artifacts(self, run_id: str) -> list[dict]:
        with self.database.session() as connection:
            return [dict(row) for row in connection.execute("SELECT * FROM master_artifacts WHERE run_id=? ORDER BY created_at, artifact_path", (run_id,)).fetchall()]

    def save_master_report(self, run_id: str, report_path: str, report_hash: str, report: dict) -> dict:
        with self.database.transaction() as connection:
            connection.execute("INSERT INTO master_reports(run_id,report_path,report_hash,report_json,created_at) VALUES(?,?,?,?,?) ON CONFLICT(run_id) DO UPDATE SET report_path=excluded.report_path,report_hash=excluded.report_hash,report_json=excluded.report_json", (run_id, report_path, report_hash, json.dumps(report, sort_keys=True, default=str), now_iso()))
        return self.master_report(run_id)

    def master_report(self, run_id: str) -> dict | None:
        with self.database.session() as connection:
            row = connection.execute("SELECT * FROM master_reports WHERE run_id=?", (run_id,)).fetchone()
            if not row: return None
            result = dict(row); result["report_json"] = json.loads(result["report_json"]); return result

    # Specification-intake persistence. Large Codex output is kept in files;
    # SQLite stores only paths, hashes, compact errors, and run status.
    def save_specification_attempt(self, run_id: str, strategy_id: str, attempt: int, *, status: str,
                                   draft_path: str, validation_path: str | None = None,
                                   semantic_validation_path: str | None = None, repair_prompt_path: str | None = None,
                                   codex_invocation_path: str | None = None, draft_hash: str | None = None,
                                   validation_hash: str | None = None, error_summary: str = "") -> dict:
        with self.database.transaction() as connection:
            connection.execute("""INSERT INTO specification_attempts(
                run_id,strategy_id,attempt,status,draft_path,validation_path,
                semantic_validation_path,repair_prompt_path,codex_invocation_path,
                draft_hash,validation_hash,error_summary,created_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(run_id,attempt) DO UPDATE SET
                status=excluded.status,validation_path=excluded.validation_path,
                semantic_validation_path=excluded.semantic_validation_path,
                repair_prompt_path=excluded.repair_prompt_path,
                codex_invocation_path=excluded.codex_invocation_path,
                draft_hash=excluded.draft_hash,validation_hash=excluded.validation_hash,
                error_summary=excluded.error_summary""", (run_id, strategy_id, attempt, status, draft_path,
                validation_path, semantic_validation_path, repair_prompt_path, codex_invocation_path,
                draft_hash, validation_hash, error_summary, now_iso()))
        return self.get_specification_attempt(run_id, attempt)

    def get_specification_attempt(self, run_id: str, attempt: int) -> dict:
        with self.database.session() as connection:
            row = connection.execute("SELECT * FROM specification_attempts WHERE run_id=? AND attempt=?", (run_id, attempt)).fetchone()
            if not row: raise RegistryError(f"specification attempt not found: {run_id}@{attempt}")
            return dict(row)

    def specification_attempts(self, run_id: str) -> list[dict]:
        with self.database.session() as connection:
            return [dict(row) for row in connection.execute("SELECT * FROM specification_attempts WHERE run_id=? ORDER BY attempt", (run_id,)).fetchall()]

    def save_specification_ambiguity(self, run_id: str, strategy_id: str, *, kind: str, field_path: str, message: str, blocking: bool) -> None:
        with self.database.transaction() as connection:
            connection.execute("INSERT OR IGNORE INTO specification_ambiguities(run_id,strategy_id,kind,field_path,message,blocking,created_at) VALUES(?,?,?,?,?,?,?)", (run_id, strategy_id, kind, field_path, message, int(blocking), now_iso()))

    def specification_ambiguities(self, run_id: str) -> list[dict]:
        with self.database.session() as connection:
            return [dict(row) for row in connection.execute("SELECT * FROM specification_ambiguities WHERE run_id=? ORDER BY created_at, field_path", (run_id,)).fetchall()]

    def save_specification_failure(self, run_id: str, strategy_id: str, result: dict, final_reason: str) -> dict:
        with self.database.transaction() as connection:
            connection.execute("INSERT INTO specification_failures(run_id,strategy_id,classification,final_reason,result_json,created_at) VALUES(?,?,?,?,?,?) ON CONFLICT(run_id) DO UPDATE SET classification=excluded.classification,final_reason=excluded.final_reason,result_json=excluded.result_json", (run_id, strategy_id, "SPECIFICATION_GENERATION_FAILURE", final_reason, json.dumps(result, sort_keys=True, default=str), now_iso()))
        return self.specification_failure(run_id)

    def clear_specification_failure(self, run_id: str) -> None:
        with self.database.transaction() as connection:
            connection.execute("DELETE FROM specification_failures WHERE run_id=?", (run_id,))

    def specification_failure(self, run_id: str) -> dict | None:
        with self.database.session() as connection:
            row = connection.execute("SELECT * FROM specification_failures WHERE run_id=?", (run_id,)).fetchone()
            if not row: return None
            result = dict(row); result["result_json"] = json.loads(result["result_json"]); return result

    # Phase F2 adapter and real-artifact persistence. These methods store
    # references and compact summaries; trade rows remain in hashed artifacts.
    def save_strategy_adapter(self, identity: dict, capabilities: dict, health: dict) -> None:
        timestamp = now_iso()
        with self.database.transaction() as connection:
            connection.execute("""INSERT INTO strategy_adapters(strategy_id,strategy_version,adapter_version,schema_version,implementation_module,entry_point,specification_hash,code_commit,worktree_path,capabilities_json,health_json,created_at,updated_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(strategy_id,strategy_version) DO UPDATE SET adapter_version=excluded.adapter_version,schema_version=excluded.schema_version,implementation_module=excluded.implementation_module,entry_point=excluded.entry_point,specification_hash=excluded.specification_hash,code_commit=excluded.code_commit,worktree_path=excluded.worktree_path,capabilities_json=excluded.capabilities_json,health_json=excluded.health_json,updated_at=excluded.updated_at""",
                (identity["strategy_id"], identity["strategy_version"], identity["adapter_version"], identity["schema_version"], identity["implementation_module"], identity["entry_point"], identity["specification_hash"], identity.get("code_commit"), identity.get("worktree_path"), json.dumps(capabilities, sort_keys=True), json.dumps(health, sort_keys=True), timestamp, timestamp))

    def get_strategy_adapter(self, strategy_id: str, version: str | None = None) -> dict | None:
        strategy = self.get_strategy(strategy_id, version)
        with self.database.session() as connection:
            row = connection.execute("SELECT * FROM strategy_adapters WHERE strategy_id=? AND strategy_version=?", (strategy["strategy_id"], strategy["version"])).fetchone()
            if not row: return None
            result = dict(row); result["capabilities_json"] = json.loads(result["capabilities_json"]); result["health_json"] = json.loads(result["health_json"]); return result

    def save_implementation_manifest(self, master_run_id: str, manifest_path: str, manifest_hash: str, manifest: dict) -> None:
        with self.database.transaction() as connection:
            connection.execute("INSERT INTO implementation_manifests(master_run_id,strategy_id,strategy_version,manifest_path,manifest_hash,manifest_json,created_at) VALUES(?,?,?,?,?,?,?) ON CONFLICT(master_run_id) DO UPDATE SET manifest_path=excluded.manifest_path,manifest_hash=excluded.manifest_hash,manifest_json=excluded.manifest_json", (master_run_id, manifest["strategy_id"], manifest["strategy_version"], manifest_path, manifest_hash, json.dumps(manifest, sort_keys=True), now_iso()))

    def get_implementation_manifest(self, master_run_id: str) -> dict | None:
        with self.database.session() as connection:
            row = connection.execute("SELECT * FROM implementation_manifests WHERE master_run_id=?", (master_run_id,)).fetchone()
            if not row: return None
            result = dict(row); result["manifest_json"] = json.loads(result["manifest_json"]); return result

    def save_worktree_metadata(self, master_run_id: str, values: dict) -> None:
        timestamp = now_iso()
        with self.database.transaction() as connection:
            connection.execute("INSERT INTO worktree_metadata(master_run_id,strategy_id,strategy_version,base_commit,implementation_commit,branch,worktree_path,status,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?) ON CONFLICT(master_run_id) DO UPDATE SET implementation_commit=excluded.implementation_commit,status=excluded.status,updated_at=excluded.updated_at", (master_run_id, values["strategy_id"], values["strategy_version"], values["base_commit"], values.get("implementation_commit"), values["branch"], values["worktree_path"], values["status"], values.get("created_at", timestamp), timestamp))

    def get_worktree_metadata(self, master_run_id: str) -> dict | None:
        with self.database.session() as connection:
            row = connection.execute("SELECT * FROM worktree_metadata WHERE master_run_id=?", (master_run_id,)).fetchone()
            return dict(row) if row else None

    def save_backtest_run(self, result: dict, master_run_id: str | None = None) -> None:
        with self.database.transaction() as connection:
            connection.execute("INSERT INTO normalized_backtest_runs(run_id,master_run_id,strategy_id,strategy_version,phase,candidate_hash,dataset_hashes_json,configuration_hash,artifact_paths_json,artifact_hashes_json,result_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(run_id) DO UPDATE SET result_json=excluded.result_json,artifact_paths_json=excluded.artifact_paths_json,artifact_hashes_json=excluded.artifact_hashes_json", (result["run_id"], master_run_id, result["strategy_id"], result["strategy_version"], result["phase"], result["candidate_hash"], json.dumps(result["dataset_hashes"], sort_keys=True), result["configuration_hash"], json.dumps(result["artifact_paths"], sort_keys=True), json.dumps(result["artifact_hashes"], sort_keys=True), json.dumps(result, sort_keys=True), now_iso()))

    def get_backtest_run(self, run_id: str) -> dict | None:
        with self.database.session() as connection:
            row = connection.execute("SELECT * FROM normalized_backtest_runs WHERE run_id=?", (run_id,)).fetchone()
            if not row: return None
            result = dict(row); result["result_json"] = json.loads(result["result_json"]); return result

    def save_trade_artifact(self, run_id: str, path: str, artifact_hash: str, row_count: int, schema_version: str = "1") -> None:
        with self.database.transaction() as connection:
            connection.execute("INSERT INTO trade_artifact_references(run_id,artifact_path,artifact_hash,row_count,schema_version,created_at) VALUES(?,?,?,?,?,?) ON CONFLICT(run_id) DO UPDATE SET artifact_path=excluded.artifact_path,artifact_hash=excluded.artifact_hash,row_count=excluded.row_count", (run_id, path, artifact_hash, row_count, schema_version, now_iso()))

    def save_phase_d_export(self, master_run_id: str, result: dict, path: str, artifact_hash: str) -> None:
        with self.database.transaction() as connection:
            connection.execute("INSERT INTO phase_d_export_manifests(master_run_id,strategy_id,strategy_version,candidate_hash,artifact_path,artifact_hash,result_json,created_at) VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(master_run_id) DO UPDATE SET artifact_path=excluded.artifact_path,artifact_hash=excluded.artifact_hash,result_json=excluded.result_json", (master_run_id, result["strategy_id"], result["strategy_version"], result["candidate_hash"], path, artifact_hash, json.dumps(result, sort_keys=True), now_iso()))

    def save_phase_e_eligibility(self, master_run_id: str, result: dict) -> None:
        with self.database.transaction() as connection:
            connection.execute("INSERT INTO phase_e_eligibility_records(master_run_id,strategy_id,strategy_version,candidate_hash,outcome,result_json,created_at) VALUES(?,?,?,?,?,?,?) ON CONFLICT(master_run_id) DO UPDATE SET outcome=excluded.outcome,result_json=excluded.result_json", (master_run_id, result["strategy_id"], result["strategy_version"], result["candidate_hash"], result["outcome"], json.dumps(result, sort_keys=True), now_iso()))

    def record_artifact_integrity(self, master_run_id: str, path: str, expected_hash: str, observed_hash: str | None, status: str) -> None:
        with self.database.transaction() as connection:
            connection.execute("INSERT INTO artifact_integrity_checks(master_run_id,artifact_path,expected_hash,observed_hash,status,checked_at) VALUES(?,?,?,?,?,?) ON CONFLICT(master_run_id,artifact_path) DO UPDATE SET observed_hash=excluded.observed_hash,status=excluded.status,checked_at=excluded.checked_at", (master_run_id, path, expected_hash, observed_hash, status, now_iso()))
