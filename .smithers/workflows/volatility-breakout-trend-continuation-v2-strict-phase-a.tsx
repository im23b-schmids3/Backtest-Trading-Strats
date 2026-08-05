/** @jsxImportSource smithers-orchestrator */
import { createSmithers, Task, Workflow } from "smithers-orchestrator";
import { z } from "zod/v4";
const result = z.object({ status: z.string(), summary: z.string(), realStudyExecuted: z.literal(true), marketDataAccessed: z.literal(true), model: z.literal("gpt-5.6-terra") }).passthrough();
const { outputs, smithers } = createSmithers({ result });
async function phaseA(): Promise<z.infer<typeof result>> {
  const manifest=process.env.VBTC_V2_PHASE_A_BARS_MANIFEST, artifacts=process.env.VBTC_V2_PHASE_A_ARTIFACT_ROOT, root=process.env.VBTC_V2_REPOSITORY_ROOT;
  if (!manifest || !artifacts || !root) throw new Error("VBTC V2 requires VBTC_V2_PHASE_A_BARS_MANIFEST, VBTC_V2_PHASE_A_ARTIFACT_ROOT, and VBTC_V2_REPOSITORY_ROOT");
  const p=Bun.spawn(["python","-m","research_pipeline.cli","vbtc-v2-phase-a","--phase-a-bars-manifest",manifest,"--artifact-root",artifacts,"--repository-root",root],{cwd:root,stdout:"pipe",stderr:"pipe"});
  const stdout=await new Response(p.stdout).text(), stderr=await new Response(p.stderr).text(); if(await p.exited!==0) throw new Error(stderr||stdout); return result.parse(JSON.parse(stdout));
}
export default smithers(()=><Workflow name="volatility-breakout-trend-continuation-v2-strict-phase-a"><Task id="deterministic-vbtc-v2-phase-a" output={outputs.result} retries={0}>{phaseA}</Task></Workflow>);
