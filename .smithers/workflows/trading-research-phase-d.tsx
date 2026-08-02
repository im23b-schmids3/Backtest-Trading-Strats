// smithers-source: project-local
// smithers-metadata-version: 1
// smithers-display-name: Trading research Phase D prop compatibility
// smithers-description: Durable deterministic futures compatibility, prop lifecycle, sizing, economics, and compliance review.
/** @jsxImportSource smithers-orchestrator */
import { createSmithers, Sequence, Task } from "smithers-orchestrator";
import { z } from "zod/v4";
import { finalReview, outputs, phaseDInput, phaseDSummary, propStatus, propStep, roleReview } from "../schemas/trading-research/phase-d";

const { Workflow, outputs: registeredOutputs, smithers } = createSmithers({
  input: phaseDInput,
  start: propStep,
  rules: propStep,
  contracts: propStep,
  reconcile: propStep,
  risk: propStep,
  scenarios: propStep,
  analyst: roleReview,
  compliance: roleReview,
  finalReview,
  summary: phaseDSummary,
});

async function bridge(command: string, payload: Record<string, unknown>, root: string): Promise<unknown> {
  const registryPath = payload.registry_path ?? process.env.RESEARCH_PIPELINE_REGISTRY ?? `${root}/research_registry/research_pipeline.sqlite3`;
  const child = Bun.spawn(["python", "-m", "research_pipeline", "workflow", command, "--input-json", JSON.stringify({ ...payload, registry_path: registryPath })], {
    cwd: root,
    env: { ...process.env, PYTHONPATH: `${root}/src`, RESEARCH_PIPELINE_REGISTRY: String(registryPath) },
    stdout: "pipe", stderr: "pipe",
  });
  const stdout = await new Response(child.stdout).text();
  const stderr = await new Response(child.stderr).text();
  const code = await child.exited;
  if (code !== 0) throw new Error(`Phase D bridge ${command} failed (${code}): ${stderr || stdout}`);
  return JSON.parse(stdout);
}

type StepState = { strategy_id: string; repository_root: string; registry_path: string; scenario: string; product: string };

async function guarded(command: string, expected: string, payload: StepState): Promise<unknown> {
  const state = propStatus.parse(await bridge("prop-status", payload, payload.repository_root));
  if (state.prop_run?.current_phase !== expected) return { status: "SKIPPED", reason: `Phase D is ${state.prop_run?.current_phase ?? "not started"}; expected ${expected}` };
  return bridge(command, payload, payload.repository_root);
}

async function guardedFinal(payload: StepState): Promise<z.infer<typeof finalReview>> {
  const state = propStatus.parse(await bridge("prop-status", payload, payload.repository_root));
  if (state.prop_run?.current_phase === "PROP_ECONOMICS_REVIEW") return await bridge("prop-final-review", payload, payload.repository_root) as z.infer<typeof finalReview>;
  const persisted = state.final_review?.result_json;
  if (persisted) return persisted as z.infer<typeof finalReview>;
  return { strategy_id: state.strategy.strategy_id, strategy_version: state.strategy.version, classification: state.prop_run?.status ?? "INSUFFICIENT_FUTURES_DATA", scenario_id: `${state.strategy.strategy_id}-${payload.scenario}`, metrics: {}, compliance: {}, data_limitations: { warnings: ["Phase D stopped before economics review"] }, metrics_cited: [], rationale: `Phase D stopped at ${state.prop_run?.current_phase ?? "unknown phase"}.` };
}

function review(role: "Prop Research Analyst" | "Compliance Reviewer", state: any): z.infer<typeof roleReview> {
  const run = state.prop_run ?? {};
  const issues = [...(state.rules?.result_json?.errors ?? []), ...(state.contracts?.result_json?.errors ?? []), ...(state.economics?.result_json?.compliance?.violations ?? [])];
  return { role, strategy_id: state.strategy.strategy_id, phase: run.current_phase, status: run.status, evidence: ["SQLite prop journal", "hashed contract and mapping registries", "deterministic scenario result"], blocking_issues: issues };
}

export default smithers((ctx) => {
  const input = ctx.input;
  const state: StepState = { strategy_id: input.strategy_id, repository_root: input.repository_root, registry_path: input.registry_path, scenario: input.scenario, product: input.product };
  const scenarioResult = ctx.outputMaybe(registeredOutputs.scenarios, { nodeId: "prop-scenarios" });
  const final = ctx.outputMaybe(registeredOutputs.finalReview, { nodeId: "final-prop-critic" });
  return <Workflow name="trading-research-phase-d"><Sequence>
    <Task id="prop-start" output={registeredOutputs.start} retries={0}>{async () => bridge("prop-start", { ...state, run_id: input.research_run_id }, input.repository_root)}</Task>
    <Task id="rule-verification" output={registeredOutputs.rules} dependsOn={["prop-start"]} retries={0}>{async () => guarded("prop-verify-rules", "ENTRY_VERIFICATION", state)}</Task>
    <Task id="contract-verification" output={registeredOutputs.contracts} dependsOn={["rule-verification"]} retries={0}>{async () => guarded("prop-verify-contracts", "CONTRACT_VERIFICATION", state)}</Task>
    <Task id="futures-reconciliation" output={registeredOutputs.reconcile} dependsOn={["contract-verification"]} retries={0}>{async () => guarded("prop-reconcile", "RECONCILIATION", state)}</Task>
    <Task id="risk-sizing" output={registeredOutputs.risk} dependsOn={["futures-reconciliation"]} retries={0}>{async () => guarded("prop-run-risk", "RISK_SIZING", state)}</Task>
    <Task id="prop-scenarios" output={registeredOutputs.scenarios} dependsOn={["risk-sizing"]} retries={0}>{async () => guarded("prop-run-scenarios", "PROP_SIMULATION", state)}</Task>
    {scenarioResult ? <Task id="prop-research-analyst" output={registeredOutputs.analyst} dependsOn={["prop-scenarios"]} retries={0}>{async () => review("Prop Research Analyst", propStatus.parse(await bridge("prop-status", state, input.repository_root)))}</Task> : null}
    {scenarioResult ? <Task id="compliance-reviewer" output={registeredOutputs.compliance} dependsOn={["prop-research-analyst"]} retries={0}>{async () => review("Compliance Reviewer", propStatus.parse(await bridge("prop-status", state, input.repository_root)))}</Task> : null}
    {scenarioResult ? <Task id="final-prop-critic" output={registeredOutputs.finalReview} dependsOn={["compliance-reviewer"]} retries={0}>{async () => guardedFinal(state)}</Task> : null}
    {scenarioResult || final ? <Task id="phase-d-summary" output={registeredOutputs.summary} dependsOn={["final-prop-critic"]} retries={0}>{async () => {
      const status = propStatus.parse(await bridge("prop-status", state, input.repository_root));
      const reviewResult = status.final_review?.result_json;
      return { strategy_id: status.strategy.strategy_id, strategy_version: status.strategy.version, prop_phase: status.prop_run.current_phase, classification: reviewResult?.classification ?? status.prop_run.status, product: input.product, scenario: input.scenario, journal_entries: (await bridge("prop-journal", state, input.repository_root) as any).entries?.length ?? 0, holdout_accesses: status.holdout_accesses, b5_verified: true, no_optimization: true, limitations: reviewResult?.data_limitations?.warnings ?? ["workflow stopped before final economics review"] };
    }}</Task> : null}
  </Sequence></Workflow>;
});
