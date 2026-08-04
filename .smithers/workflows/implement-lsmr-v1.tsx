/** @jsxImportSource smithers-orchestrator */
import { existsSync, readFileSync } from "node:fs";
import { createSmithers, Sequence, Task } from "smithers-orchestrator";
import { z } from "zod/v4";
import { agents } from "../agents";

const root = "C:/Users/sandr/Trading-Bot-Fib";
const specificationPath = `${root}/.smithers/specs/liquidity-sweep-mean-reversion-v1.md`;
const heading = "LiquiditySweepMeanReversion.BTC_LONG_SHORT_V1_SPECIFICATION";
const specification = z.object({ path:z.literal(specificationPath), heading:z.literal(heading), valid:z.literal(true) });
const result = z.object({ status:z.string(), summary:z.string(), testsPassed:z.boolean(), phaseBManifest:z.string().nullable(), model:z.literal("gpt-5.6-terra") });
const { Workflow, outputs, smithers } = createSmithers({ specification, result });

function requireSealedSpecification(): z.infer<typeof specification> {
  if (!existsSync(specificationPath)) throw new Error("MISSING_SEALED_LSMR_SPECIFICATION");
  const text = readFileSync(specificationPath, "utf8");
  if (!text.trim() || !new RegExp(`^#?\\s*${heading}\\s*$`, "m").test(text)) throw new Error("MISSING_SEALED_LSMR_SPECIFICATION");
  return { path: specificationPath, heading, valid: true };
}

export default smithers((ctx) => {
  const sealed = ctx.outputMaybe(outputs.specification, { nodeId:"require-sealed-lsmr-specification" });
  return <Workflow name="implement-lsmr-v1"><Sequence>
    <Task id="require-sealed-lsmr-specification" output={outputs.specification} retries={0}>{async () => requireSealedSpecification()}</Task>
    {sealed ? <Task id="implement-lsmr-v1" output={outputs.result} agent={agents.midTier} retries={1} timeoutMs={4*60*60_000} heartbeatTimeoutMs={20*60_000}>{`Read ${specificationPath} completely before making changes and implement it exactly. Work only in ${root}. This task is implementation plus synthetic validation only: never run real Phase A, Phase B, or Alpha; never invoke a real candidate CLI; never download, rebuild, duplicate, or modify market data. Add strategy code, deterministic CLI, Smithers candidate workflow, synthetic tests, immutable artifacts, setup IDs, one terminal disposition per setup, and reconciliation. Preserve existing studies and sealed gates. Run focused tests, full research_pipeline tests using a repository-local basetemp, compileall, git diff --check, and Smithers graph validation. Return schema-valid JSON with model gpt-5.6-terra.`}</Task> : null}
  </Sequence></Workflow>;
});
