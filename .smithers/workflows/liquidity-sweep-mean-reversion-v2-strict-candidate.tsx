/** @jsxImportSource smithers-orchestrator */
import { createSmithers, Task, Workflow } from "smithers-orchestrator";
import { z } from "zod/v4";

const root = "C:/Users/sandr/Trading-Bot-Fib";
const result = z.object({
  status: z.literal("PHASE_A_NO_ROBUST_CANDIDATE"), summary: z.string(), testsPassed: z.boolean(),
  realStudyExecuted: z.literal(false), model: z.literal("gpt-5.6-terra"), artifactRoot: z.string(),
}).strict();
const { outputs, smithers } = createSmithers({ result });

async function materializeOnly(): Promise<z.infer<typeof result>> {
  const child = Bun.spawn(["python", "-m", "research_pipeline", "lsmr-v2-strict-materialize", "--repository-root", root, "--artifact-root", `${root}/research_runs`], { cwd: root, stdout: "pipe", stderr: "pipe" });
  const stdout = await new Response(child.stdout).text(); const stderr = await new Response(child.stderr).text();
  if (await child.exited !== 0) throw new Error(stderr || stdout || "lsmr-v2-strict-materialize failed");
  return result.parse(JSON.parse(stdout));
}

// A function task, not an agent task: it only materializes unexecuted V2 evidence.
export default smithers(() => <Workflow name="liquidity-sweep-mean-reversion-v2-strict-candidate">
  <Task id="materialize-unexecuted-v2-contract" output={outputs.result} retries={0}>{materializeOnly}</Task>
</Workflow>);
