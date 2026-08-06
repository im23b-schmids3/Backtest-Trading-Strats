/** @jsxImportSource smithers-orchestrator */
import { createSmithers, Task, Workflow } from "smithers-orchestrator";
import { z } from "zod/v4";

const result = z.object({
  status: z.literal("COMPLETED"),
  artifact_root: z.string(),
  phase_a_result: z.literal("phase-a-result.json"),
  phase_b_status: z.literal("NOT_OPENED"),
  realPhaseAExecuted: z.literal(true),
  phaseBExecuted: z.literal(false),
  alphaExecuted: z.literal(false),
}).strict();
const { outputs, smithers } = createSmithers({ result });

// Non-agent by design.  This is the sole future Phase-A invocation surface.
async function execute(): Promise<z.infer<typeof result>> {
  const manifest = process.env.HTF_LFVG_V2_PHASE_A_BARS_MANIFEST;
  const artifactRoot = process.env.HTF_LFVG_V2_PHASE_A_ARTIFACT_ROOT;
  if (!manifest || !artifactRoot) throw new Error("HTF_LFVG_V2_PHASE_A_BARS_MANIFEST and HTF_LFVG_V2_PHASE_A_ARTIFACT_ROOT are required");
  const child = Bun.spawn(["python", "-m", "research_pipeline.cli", "htf-lfvg-v2-phase-a", "--phase-a-bars-manifest", manifest, "--artifact-root", artifactRoot, "--repository-root", "C:/Users/sandr/Trading-Bot-Fib"], { stdout: "pipe", stderr: "pipe" });
  const stdout = await new Response(child.stdout).text(); const stderr = await new Response(child.stderr).text();
  if (await child.exited !== 0) throw new Error(stderr || stdout);
  return result.parse(JSON.parse(stdout));
}
export default smithers(() => <Workflow name="htf-lfvg-v2-phase-a"><Task id="execute-sealed-phase-a" output={outputs.result} retries={0}>{execute}</Task></Workflow>);
