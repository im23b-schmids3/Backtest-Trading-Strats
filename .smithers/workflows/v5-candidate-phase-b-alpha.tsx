/** @jsxImportSource smithers-orchestrator */
import { createSmithers, Sequence, Task } from "smithers-orchestrator";
import { z } from "zod/v4";
import { agents } from "../agents";

const root = "C:/Users/sandr/Trading-Bot-Fib";
const manifest = `${root}/data/imbalance_vwap_ride/v5/footprints/BTCUSDT/phase_a/4f8e06b06b8348d9e071983bdef6239f313cdac51c89475c0ed09181843f79e3/manifest.json`;
const datasetHash = "b52cb51befdd7da894ed3249865be22166ec7c2447420f8a7391ced3fd9f1f72";
const footprintHash = "3afba9bceda58b0cd75f9d334e30e54cb5dcb551ced374f31dff25a12bbd9d4c";

const preflight = z.object({ status:z.literal("VALID"), manifestPath:z.literal(manifest), valid:z.literal(true), monthCount:z.literal(13), datasetHash:z.literal(datasetHash), footprintDatasetHash:z.literal(footprintHash), gitClean:z.literal(true), model:z.literal("gpt-5.6-terra") });
const phaseA = z.object({ status:z.enum(["PHASE_A_SELECTED","PHASE_A_NO_ROBUST_CANDIDATE"]), candidateExecutions:z.record(z.enum(["V5-A-SCALED-BIN-1P5R","V5-B-SCALED-BIN-2P0R","V5-C-SCALED-BIN-2P5R"]),z.literal(1)), metricsPath:z.string(), gatesPath:z.string(), rankingPath:z.string(), frozenCandidateHash:z.string().nullable(), model:z.literal("gpt-5.6-terra") });
const finalResult = z.object({ status:z.string(), finalReportPath:z.string(), phaseAStatus:z.string(), phaseBStatus:z.string(), alphaStatus:z.string(), model:z.literal("gpt-5.6-terra") });
const { Workflow, outputs, smithers } = createSmithers({ preflight, phaseA, finalResult });

export default smithers(() => <Workflow name="v5-candidate-phase-b-alpha"><Sequence>
  <Task id="validate-v5-phase-a-input" output={outputs.preflight} agent={agents.midTier} retries={0} timeoutMs={30*60_000} heartbeatTimeoutMs={5*60_000}>{`Work only in ${root}. Do not modify Python, manifests, V1-V4, or Phase A data. Require git status --porcelain to be empty; otherwise fail. Validate ${manifest}: valid=true, exactly 13 ordered months, normalized dataset hash ${datasetHash}, footprint dataset hash ${footprintHash}, every referenced parquet SHA-256 and row count. Return schema-valid JSON with model gpt-5.6-terra.`}</Task>
  <Task id="execute-v5-phase-a-candidates" output={outputs.phaseA} agent={agents.midTier} retries={0} timeoutMs={12*60*60_000} heartbeatTimeoutMs={30*60_000}>{`Work only in ${root}; require the prior validated immutable Phase A manifest. Do not rebuild/reacquire data and do not modify Python. Execute exactly once each sealed V5 candidate, persist complete metrics, funnel and invalid-reason counts, monthly metrics, literal gates and ranking; freeze at most one. If none pass, set PHASE_A_NO_ROBUST_CANDIDATE. No live orders or raw-row disclosure. Return schema-valid JSON with model gpt-5.6-terra.`}</Task>
  <Task id="conditional-v5-phase-b-alpha-finalize" output={outputs.finalResult} agent={agents.midTier} retries={0} timeoutMs={12*60*60_000} heartbeatTimeoutMs={30*60_000}>{`Work only in ${root}; do not modify Python or V1-V4. If Phase A has no frozen candidate, finalize without Phase B or Alpha. Otherwise execute the frozen candidate exactly once on locked Phase B 2024-02..2024-07 using existing validated inputs, then run the configured Alpha/MBT futures proxy only if every Phase B gate passes. Never place live orders. Persist the final report and return schema-valid JSON with model gpt-5.6-terra.`}</Task>
</Sequence></Workflow>);
