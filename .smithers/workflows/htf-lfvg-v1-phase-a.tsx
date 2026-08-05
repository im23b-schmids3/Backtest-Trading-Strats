/** @jsxImportSource smithers-orchestrator */
import { createSmithers, Task, Workflow } from "smithers-orchestrator";
import { z } from "zod/v4";

const result = z.object({ status: z.string() }).passthrough();
const { outputs, smithers } = createSmithers({ result });

// Non-agent by design: the sealed CLI is the only Phase-A executor.
async function execute(): Promise<z.infer<typeof result>> {
  const manifest = process.env.HTF_LFVG_V1_PHASE_A_BARS_MANIFEST;
  const artifactRoot = process.env.HTF_LFVG_V1_PHASE_A_ARTIFACT_ROOT;
  if (!manifest || !artifactRoot) throw new Error("HTF_LFVG_V1_PHASE_A_BARS_MANIFEST and HTF_LFVG_V1_PHASE_A_ARTIFACT_ROOT are required");
  const child = Bun.spawn(["python", "-m", "research_pipeline.cli", "htf-lfvg-v1-phase-a", "--phase-a-bars-manifest", manifest, "--artifact-root", artifactRoot, "--repository-root", "C:/Users/sandr/Trading-Bot-Fib"], { stdout:"pipe", stderr:"pipe" });
  const stdout = await new Response(child.stdout).text(); const stderr = await new Response(child.stderr).text();
  if (await child.exited !== 0) throw new Error(stderr || stdout);
  return result.parse(JSON.parse(stdout));
}
export default smithers(() => <Workflow name="htf-lfvg-v1-phase-a"><Task id="execute-sealed-phase-a" output={outputs.result} retries={0}>{execute}</Task></Workflow>);
