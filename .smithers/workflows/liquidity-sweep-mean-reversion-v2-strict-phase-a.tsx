/** @jsxImportSource smithers-orchestrator */
import { createSmithers, Task, Workflow } from "smithers-orchestrator";
import { z } from "zod/v4";

const root = "C:/Users/sandr/Trading-Bot-Fib";
const result = z.object({
  status: z.enum(["PHASE_A_SELECTED", "PHASE_A_NO_ROBUST_CANDIDATE"]),
  summary: z.string(),
  selectedCandidateId: z.string().nullable(),
  phaseBStatus: z.enum(["NOT_OPENED", "PENDING_CONDITIONAL_FINALIZER"]),
  alphaStatus: z.literal("NOT_EXECUTED"),
  studyExecuted: z.literal(true),
  realStudyExecuted: z.literal(true),
  model: z.literal("gpt-5.6-terra"),
  artifactRoot: z.string(),
  candidateExecutions: z.record(z.string(), z.literal(1)),
}).strict();
const { outputs, smithers } = createSmithers({ result });

// This is deliberately a non-agent function task. It neither discovers old runs
// nor calculates research metrics: the deterministic CLI is the sole authority.
async function executePhaseA(): Promise<z.infer<typeof result>> {
  const manifest = process.env.LSMR_V2_PHASE_A_BARS_MANIFEST;
  const artifactRoot = process.env.LSMR_V2_PHASE_A_ARTIFACT_ROOT;
  if (!manifest || !artifactRoot) throw new Error("LSMR_V2_PHASE_A_BARS_MANIFEST and LSMR_V2_PHASE_A_ARTIFACT_ROOT are required absolute paths");
  const child = Bun.spawn(["python", "-m", "research_pipeline.cli", "lsmr-v2-phase-a", "--phase-a-bars-manifest", manifest, "--artifact-root", artifactRoot, "--repository-root", root], { cwd: root, stdout: "pipe", stderr: "pipe" });
  const stdout = await new Response(child.stdout).text(); const stderr = await new Response(child.stderr).text();
  if (await child.exited !== 0) throw new Error(stderr || stdout || "lsmr-v2-phase-a failed");
  return result.parse(JSON.parse(stdout));
}

export default smithers(() => <Workflow name="liquidity-sweep-mean-reversion-v2-strict-phase-a">
  <Task id="execute-sealed-v2-phase-a" output={outputs.result} retries={0}>{executePhaseA}</Task>
</Workflow>);
