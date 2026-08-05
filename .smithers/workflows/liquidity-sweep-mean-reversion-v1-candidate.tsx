/** @jsxImportSource smithers-orchestrator */
import { createSmithers, Task, Workflow } from "smithers-orchestrator";
import { z } from "zod/v4";

const root = "C:/Users/sandr/Trading-Bot-Fib";
const result = z.object({
  status: z.string(), selectedCandidateId: z.string().nullable(),
  phaseBStatus: z.literal("NOT_OPENED"), alphaStatus: z.literal("NOT_EXECUTED"),
  artifactRoot: z.string(), studyExecuted: z.literal(true),
}).strict();
const { outputs, smithers } = createSmithers({ result });

async function executeFixedCli(): Promise<z.infer<typeof result>> {
  const child = Bun.spawn(["python", "-m", "research_pipeline", "lsmr-v1-phase-a", "--repository-root", root, "--artifact-root", `${root}/research_runs`], { cwd: root, stdout: "pipe", stderr: "pipe" });
  const stdout = await new Response(child.stdout).text();
  const stderr = await new Response(child.stderr).text();
  if (await child.exited !== 0) throw new Error(stderr || stdout || "lsmr-v1-phase-a failed");
  return result.parse(JSON.parse(stdout));
}

// This sealed final-candidate workflow invokes the fixed deterministic Python CLI only.
export default smithers(() => <Workflow name="liquidity-sweep-mean-reversion-v1-candidate">
  <Task id="execute-three-sealed-phase-a-candidates" output={outputs.result} retries={0}>{executeFixedCli}</Task>
</Workflow>);
