/** @jsxImportSource smithers-orchestrator */
import { createSmithers, Task, Workflow } from "smithers-orchestrator";
import { z } from "zod/v4";
import { agents } from "../agents";

const { outputs, smithers } = createSmithers({ result: z.object({ status:z.string(), summary:z.string(), testsPassed:z.boolean(), phaseBManifest:z.string().nullable(), model:z.literal("gpt-5.6-terra") }).strict() });
export default smithers(() => <Workflow name="liquidity-sweep-mean-reversion-v1-candidate"><Task id="materialize-synthetic-contract" output={outputs.result} agent={agents.midTier}>{"Work only in C:/Users/sandr/Trading-Bot-Fib. Read and validate .smithers/specs/liquidity-sweep-mean-reversion-v1.md; fail MISSING_SEALED_LSMR_SPECIFICATION if it is missing or invalid. Run synthetic tests only. Never execute Phase A, Phase B, Alpha, or a candidate CLI; never read, acquire, or modify market data. Preserve every sealed parameter and gate. Materialize only the sealed synthetic LSMR V1 contract and return the required strict JSON schema."}</Task></Workflow>);
