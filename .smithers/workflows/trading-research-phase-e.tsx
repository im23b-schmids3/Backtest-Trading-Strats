// smithers-source: project-local
// smithers-metadata-version: 1
// smithers-display-name: Trading research Phase E multi-strategy portfolio
// smithers-description: Durable deterministic portfolio construction, shared-account replay, risk, economics, and final classification.
/** @jsxImportSource smthrs */
import { createSmithers, Sequence, Task } from "smthrs";
import { z } from "zod/v4";
import { phaseEInput, outputs, phaseESummary, portfolioStatus, portfolioStep, roleReview } from "../schemas/trading-research/phase-e";

const { Workflow, outputs: registeredOutputs, smithers } = createSmithers({
  input: phaseEInput,
  create: portfolioStep,
  candidates: portfolioStep,
  signals: portfolioStep,
  overlap: portfolioStep,
  correlation: portfolioStep,
  risk: portfolioStep,
  prop: portfolioStep,
  ablation: portfolioStep,
  stress: portfolioStep,
  analyst: roleReview,
  statistical: roleReview,
  compliance: roleReview,
  finalCritic: roleReview,
  summary: phaseESummary,
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
  if (code !== 0) throw new Error(`Phase E bridge ${command} failed (${code}): ${stderr || stdout}`);
  return JSON.parse(stdout);
}

type State = { portfolio_id: string; portfolio_config: string; repository_root: string; registry_path: string; scenario: string };

function role(roleName: z.infer<typeof roleReview>["role"], state: any): z.infer<typeof roleReview> {
  const run = state.run ?? {};
  const evidence = ["portfolio journal", "portfolio candidate artifacts", "overlap and correlation metrics", "shared-account risk and prop replay", "ablation and stress results"];
  const blocking = state.final_review?.result_json?.primary_limitations ?? [];
  return { role: roleName, portfolio_id: state.specification.portfolio_id, phase: run.current_phase, status: run.status, evidence, blocking_issues: blocking, metrics_cited: ["portfolio_overlap_metrics", "portfolio_correlation_metrics", "portfolio_risk_runs", "portfolio_prop_scenarios"] };
}

export default smithers((ctx) => {
  const input = ctx.input;
  const state: State = { portfolio_id: input.portfolio_id, portfolio_config: input.portfolio_config, repository_root: input.repository_root, registry_path: input.registry_path, scenario: input.scenario };
  return <Workflow name="trading-research-phase-e"><Sequence>
    <Task id="portfolio-create" output={registeredOutputs.create} retries={0}>{async () => bridge("portfolio-create", { ...state, spec_path: state.portfolio_config }, state.repository_root)}</Task>
    <Task id="portfolio-candidates" output={registeredOutputs.candidates} dependsOn={["portfolio-create"]} retries={0}>{async () => bridge("portfolio-generate-candidates", state, state.repository_root)}</Task>
    <Task id="portfolio-merge-signals" output={registeredOutputs.signals} dependsOn={["portfolio-candidates"]} retries={0}>{async () => bridge("portfolio-merge-signals", state, state.repository_root)}</Task>
    <Task id="portfolio-overlap" output={registeredOutputs.overlap} dependsOn={["portfolio-merge-signals"]} retries={0}>{async () => bridge("portfolio-analyze-overlap", state, state.repository_root)}</Task>
    <Task id="portfolio-correlation" output={registeredOutputs.correlation} dependsOn={["portfolio-overlap"]} retries={0}>{async () => bridge("portfolio-analyze-correlation", state, state.repository_root)}</Task>
    <Task id="portfolio-risk" output={registeredOutputs.risk} dependsOn={["portfolio-correlation"]} retries={0}>{async () => bridge("portfolio-run-risk", state, state.repository_root)}</Task>
    <Task id="portfolio-prop" output={registeredOutputs.prop} dependsOn={["portfolio-risk"]} retries={0}>{async () => bridge("portfolio-run-prop", state, state.repository_root)}</Task>
    <Task id="portfolio-ablation" output={registeredOutputs.ablation} dependsOn={["portfolio-prop"]} retries={0}>{async () => bridge("portfolio-run-ablation", state, state.repository_root)}</Task>
    <Task id="portfolio-stress" output={registeredOutputs.stress} dependsOn={["portfolio-ablation"]} retries={0}>{async () => bridge("portfolio-run-stress", state, state.repository_root)}</Task>
    <Task id="portfolio-analyst" output={registeredOutputs.analyst} dependsOn={["portfolio-stress"]} retries={0}>{async () => role("Portfolio Analyst", portfolioStatus.parse(await bridge("portfolio-status", state, state.repository_root)))}</Task>
    <Task id="portfolio-statistical-reviewer" output={registeredOutputs.statistical} dependsOn={["portfolio-analyst"]} retries={0}>{async () => role("Portfolio Statistical Reviewer", portfolioStatus.parse(await bridge("portfolio-status", state, state.repository_root)))}</Task>
    <Task id="portfolio-compliance-reviewer" output={registeredOutputs.compliance} dependsOn={["portfolio-statistical-reviewer"]} retries={0}>{async () => role("Compliance Reviewer", portfolioStatus.parse(await bridge("portfolio-status", state, state.repository_root)))}</Task>
    <Task id="portfolio-final-critic" output={registeredOutputs.finalCritic} dependsOn={["portfolio-compliance-reviewer"]} retries={0}>{async () => { await bridge("portfolio-final-review", state, state.repository_root); return role("Final Portfolio Critic", portfolioStatus.parse(await bridge("portfolio-status", state, state.repository_root))); }}</Task>
    <Task id="phase-e-summary" output={registeredOutputs.summary} dependsOn={["portfolio-final-critic"]} retries={0}>{async () => { const status = portfolioStatus.parse(await bridge("portfolio-status", state, state.repository_root)); const review = status.final_review?.result_json; const journal = await bridge("portfolio-journal", state, state.repository_root) as any; return { portfolio_id: input.portfolio_id, phase: status.run.current_phase, classification: review?.classification ?? status.run.status, selected_candidate_id: review?.selected_candidate_id ?? null, members: review?.best_portfolio ?? [], journal_entries: journal.entries?.length ?? 0, no_optimization: true, no_trading: true, limitations: review?.primary_limitations ?? ["workflow stopped before final review"] }; }}</Task>
  </Sequence></Workflow>;
});
