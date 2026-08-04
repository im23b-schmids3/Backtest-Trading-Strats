/** @jsxImportSource smithers-orchestrator */
import { execFileSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { createSmithers, Sequence, Task } from "smithers-orchestrator";
import { z } from "zod/v4";

const root = "C:/Users/sandr/Trading-Bot-Fib";
const manifestPath = `${root}/data/imbalance_vwap_ride/v5/footprints/BTCUSDT/phase_a/4f8e06b06b8348d9e071983bdef6239f313cdac51c89475c0ed09181843f79e3/manifest.json`;
const artifactRoot = `${root}/research_runs`;
const sealedCandidates = ["V5-A-SCALED-BIN-1P5R", "V5-B-SCALED-BIN-2P0R", "V5-C-SCALED-BIN-2P5R"] as const;
const phaseA = z.object({ status:z.enum(["PHASE_A_SELECTED","PHASE_A_NO_ROBUST_CANDIDATE"]), candidateExecutions:z.record(z.enum(sealedCandidates),z.literal(1)), freshRunPath:z.string(), metricsPath:z.string(), gatesPath:z.string(), rankingPath:z.string(), frozenCandidateHash:z.string().nullable(), phaseBStatus:z.string(), alphaStatus:z.string(), model:z.literal("gpt-5.6-terra") });
const finalResult = z.object({ finalReportPath:z.string(), phaseAStatus:z.string(), phaseBStatus:z.string(), alphaStatus:z.string(), model:z.literal("gpt-5.6-terra") });
const { Workflow, outputs, smithers } = createSmithers({ phaseA, finalResult });

function invokeCandidateCli(): z.infer<typeof phaseA> {
  const stdout=execFileSync(process.env.PYTHON ?? "python", ["-m","research_pipeline.cli","v5-candidate-run","--phase-a-manifest",manifestPath,"--artifact-root",artifactRoot], { cwd:root, encoding:"utf8" });
  return phaseA.parse(JSON.parse(stdout));
}
function finalize(run: z.infer<typeof phaseA>): z.infer<typeof finalResult> {
  const finalReportPath=`${run.freshRunPath}/final_report.json`;
  if (!existsSync(finalReportPath)) throw new Error("MISSING_V5_FINAL_REPORT");
  const report=JSON.parse(readFileSync(finalReportPath,"utf8"));
  if (report.status!==run.status || report.phaseBStatus!==run.phaseBStatus || report.alphaStatus!==run.alphaStatus) throw new Error("V5_FINALIZER_STATE_MISMATCH");
  if (run.status==="PHASE_A_NO_ROBUST_CANDIDATE" && (run.phaseBStatus!=="NOT_OPENED" || run.alphaStatus!=="NOT_EXECUTED")) throw new Error("V5_CONDITIONAL_FINALIZER_VIOLATION");
  return { finalReportPath, phaseAStatus:run.status, phaseBStatus:run.phaseBStatus, alphaStatus:run.alphaStatus, model:"gpt-5.6-terra" };
}
export default smithers((ctx) => {
  const run=ctx.outputMaybe(outputs.phaseA,{nodeId:"execute-v5-phase-a-candidates"});
  return <Workflow name="v5-candidate-phase-b-alpha"><Sequence>
    <Task id="execute-v5-phase-a-candidates" output={outputs.phaseA} retries={0}>{async () => invokeCandidateCli()}</Task>
    {run ? <Task id="conditional-v5-phase-b-alpha-finalize" output={outputs.finalResult} retries={0}>{async () => finalize(run)}</Task> : null}
  </Sequence></Workflow>;
});
