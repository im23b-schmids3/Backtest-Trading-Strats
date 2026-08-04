/** @jsxImportSource smithers-orchestrator */
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { createSmithers, Sequence, Task } from "smithers-orchestrator";
import { z } from "zod/v4";
import { agents } from "../agents";

const root = "C:/Users/sandr/Trading-Bot-Fib";
const historicalRun = "b4a54c83e19943c2bd1b59fe";
const manifestPath = `${root}/data/imbalance_vwap_ride/v5/footprints/BTCUSDT/phase_a/4f8e06b06b8348d9e071983bdef6239f313cdac51c89475c0ed09181843f79e3/manifest.json`;
const datasetHash = "b52cb51befdd7da894ed3249865be22166ec7c2447420f8a7391ced3fd9f1f72";
const footprintHash = "3afba9bceda58b0cd75f9d334e30e54cb5dcb551ced374f31dff25a12bbd9d4c";
const sealedCandidates = ["V5-A-SCALED-BIN-1P5R", "V5-B-SCALED-BIN-2P0R", "V5-C-SCALED-BIN-2P5R"] as const;

const preflight = z.object({ status:z.literal("VALID"), manifestPath:z.literal(manifestPath), monthCount:z.literal(13), datasetHash:z.literal(datasetHash), footprintDatasetHash:z.literal(footprintHash), gitClean:z.literal(true), freshRunRequired:z.literal(true) });
const phaseA = z.object({ status:z.enum(["PHASE_A_SELECTED","PHASE_A_NO_ROBUST_CANDIDATE"]), candidateExecutions:z.record(z.enum(sealedCandidates),z.literal(1)), freshRunPath:z.string().refine((path) => !path.includes(historicalRun)), metricsPath:z.string(), gatesPath:z.string(), rankingPath:z.string(), frozenCandidateHash:z.string().nullable(), model:z.literal("gpt-5.6-terra") });
const finalResult = z.object({ finalReportPath:z.string(), phaseAStatus:z.string(), phaseBStatus:z.string(), alphaStatus:z.string(), model:z.literal("gpt-5.6-terra") });
const { Workflow, outputs, smithers } = createSmithers({ preflight, phaseA, finalResult });

function sha256(path: string): string { return createHash("sha256").update(readFileSync(path)).digest("hex"); }
function validate(): z.infer<typeof preflight> {
  if (execFileSync("git", ["status", "--porcelain"], { cwd: root, encoding:"utf8" }).trim()) throw new Error("DIRTY_GIT_WORKTREE");
  if (!existsSync(manifestPath)) throw new Error("MISSING_V5_PHASE_A_MANIFEST");
  const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
  const months = manifest.identity?.months;
  if (manifest.valid !== true || !Array.isArray(months) || months.length !== 13) throw new Error("INVALID_V5_PHASE_A_MANIFEST");
  if (manifest.identity?.normalized_dataset_hash !== datasetHash || manifest.footprint_dataset_hash !== footprintHash) throw new Error("V5_PHASE_A_HASH_MISMATCH");
  for (const item of manifest.parquet_files ?? []) { const target=resolve(dirname(manifestPath), item.relative_path); if (!existsSync(target) || sha256(target)!==item.sha256) throw new Error(`V5_PHASE_A_PARQUET_INVALID:${item.month}`); }
  return { status:"VALID", manifestPath, monthCount:13, datasetHash, footprintDatasetHash:footprintHash, gitClean:true, freshRunRequired:true };
}

export default smithers((ctx) => {
  const gate = ctx.outputMaybe(outputs.preflight, { nodeId:"validate-v5-phase-a-input" });
  const selected = ctx.outputMaybe(outputs.phaseA, { nodeId:"execute-v5-phase-a-candidates" });
  return <Workflow name="v5-candidate-phase-b-alpha"><Sequence>
    <Task id="validate-v5-phase-a-input" output={outputs.preflight} retries={0}>{async () => validate()}</Task>
    {gate ? <Task id="execute-v5-phase-a-candidates" output={outputs.phaseA} agent={agents.midTier} retries={0} timeoutMs={12*60*60_000} heartbeatTimeoutMs={30*60_000}>{`Work only in ${root}. The deterministic preflight passed. Do not modify Python, rebuild data, use PowerShell pipelines/heredocs, or delete files. Never write ${historicalRun}; it is sealed read-only evidence. Create a fresh immutable run identity from current code hash. Invoke exactly one supported deterministic Python CLI entry point for the three sealed candidates; do not synthesize scripts. If no such CLI entry point exists, throw MISSING_DETERMINISTIC_V5_EXECUTOR. Execute each candidate exactly once, persist reports/trades/events/funnels/invalid reasons/months/gates/ranking, freeze at most one, and return schema-valid JSON with model gpt-5.6-terra.`}</Task> : null}
    {selected ? <Task id="conditional-v5-phase-b-alpha-finalize" output={outputs.finalResult} agent={agents.midTier} retries={0} timeoutMs={12*60*60_000} heartbeatTimeoutMs={30*60_000}>{`Work only in ${root}. Do not modify Python or V1-V4, never write ${historicalRun}, and use only deterministic supported Python CLI entry points. If ${selected.status} is PHASE_A_NO_ROBUST_CANDIDATE, finalize with Phase B and Alpha NOT_OPENED. Otherwise run locked Phase B once and Alpha only on a locked Phase B pass. No live orders, deletions, PowerShell pipelines, or ad-hoc scripts. Return schema-valid JSON with model gpt-5.6-terra.`}</Task> : null}
  </Sequence></Workflow>;
});
