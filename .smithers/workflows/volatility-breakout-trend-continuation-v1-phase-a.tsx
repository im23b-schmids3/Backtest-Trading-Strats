/** @jsxImportSource smithers-orchestrator */
import { createSmithers, Task, Workflow } from "smithers-orchestrator";
import { z } from "zod/v4";

const output = z.object({
  status: z.enum(["PHASE_A_SELECTED", "PHASE_A_NO_ROBUST_CANDIDATE"]),
  summary: z.string(), testsPassed: z.literal(true), realStudyExecuted: z.literal(true),
  selectedCandidateId: z.string().nullable(), phaseBStatus: z.literal("NOT_OPENED"),
  alphaStatus: z.literal("NOT_OPENED"), model: z.literal("gpt-5.6-terra"), artifactRoot: z.string(),
}).strict();
const { outputs, smithers } = createSmithers({ output });

// Deliberately non-agent: only the deterministic Python boundary may run Phase A.
async function phaseA(): Promise<z.infer<typeof output>> {
  const manifest = process.env.VBTC_V1_PHASE_A_BARS_MANIFEST;
  const artifacts = process.env.VBTC_V1_PHASE_A_ARTIFACT_ROOT;
  const root = process.env.VBTC_V1_REPOSITORY_ROOT;
  if (!manifest || !artifacts || !root) throw new Error("VBTC V1 requires absolute manifest, artifact root, and repository root");
  const child = Bun.spawn(["python", "-m", "research_pipeline.cli", "vbtc-v1-phase-a", "--phase-a-bars-manifest", manifest, "--artifact-root", artifacts, "--repository-root", root], { cwd: root, stdout: "pipe", stderr: "pipe" });
  const stdout = await new Response(child.stdout).text(); const stderr = await new Response(child.stderr).text();
  if (await child.exited !== 0) throw new Error(stderr || stdout);
  return output.parse(JSON.parse(stdout));
}
export default smithers(() => <Workflow name="volatility-breakout-trend-continuation-v1-phase-a"><Task id="deterministic-vbtc-v1-phase-a" output={outputs.output} retries={0}>{phaseA}</Task></Workflow>);
