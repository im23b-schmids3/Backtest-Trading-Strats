/** @jsxImportSource smithers-orchestrator */
import { createSmithers, Workflow, Task } from "smithers-orchestrator";
import { z } from "zod/v4";
const input=z.object({ethManifest:z.string().min(1),btcManifest:z.string().min(1),artifactRoot:z.string().min(1),repositoryRoot:z.string().min(1)}).strict();
const output=z.object({status:z.string()}); const {smithers,outputs}=createSmithers({input,output});
export default smithers((ctx)=><Workflow name="fib09-v2-development"><Task id="deterministic-development" output={outputs.output} agent={{kind:"shell"} as never}>{`python -m research_pipeline fib09-v2-development --eth-manifest ${ctx.input.ethManifest} --btc-manifest ${ctx.input.btcManifest} --artifact-root ${ctx.input.artifactRoot} --repository-root ${ctx.input.repositoryRoot}`}</Task></Workflow>);
