/** @jsxImportSource smithers-orchestrator */
import { createSmithers, Workflow, Task } from "smithers-orchestrator";
import { z } from "zod/v4";

const input = z.object({
  ethManifest: z.string().min(1),
  btcManifest: z.string().min(1),
  chronologyManifest: z.string().min(1),
  artifactRoot: z.string().min(1),
  repositoryRoot: z.string().min(1),
}).strict();
const output = z.object({ status: z.string() });
const { smithers, outputs } = createSmithers({ input, output });

/** Future authorized invocation only. It delegates no interpretation or metric selection. */
export default smithers((ctx) => (
  <Workflow name="fib09-v1-development">
    <Task id="deterministic-development" output={outputs.output} agent={{ kind: "shell" } as never}>
      {`python -m research_pipeline fib09-v1-development --eth-manifest ${ctx.input.ethManifest} --btc-manifest ${ctx.input.btcManifest} --chronology-manifest ${ctx.input.chronologyManifest} --artifact-root ${ctx.input.artifactRoot} --repository-root ${ctx.input.repositoryRoot}`}
    </Task>
  </Workflow>
));
