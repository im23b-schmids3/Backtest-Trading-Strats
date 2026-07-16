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
