// smithers-source: project-local
// smithers-metadata-version: 1
// smithers-display-name: Trading research Phase C deterministic research
// smithers-description: Durable bounded baseline, parameter research, validation, holdout, stress, throughput, and final review.
/** @jsxImportSource smithers-orchestrator */
import { Branch, createSmithers, Ralph, Sequence, Task } from "smithers-orchestrator";
import { z } from "zod/v4";
import {
  analystDecision, baseline, candidate, edgeGate, finalReview, holdout,
  freezeFamily, phaseCInput, phaseCSummary, proposal, review, round, start, stress, throughput, walkForward,
} from "../schemas/trading-research/phase-c";

const { Workflow, outputs, smithers } = createSmithers({
  input: phaseCInput,
  start,
  baseline,
  edgeGate,
  analyst: analystDecision,
  proposal,
  round,
  review,
  freezeFamily,
  candidate,
  walkForward,
  holdout,
  stress,
  throughput,
  finalReview,
  summary: phaseCSummary,
});

async function bridge(command: string, payload: Record<string, unknown>, root: string): Promise<unknown> {
  const registryPath = payload.registry_path ?? process.env.RESEARCH_PIPELINE_REGISTRY ?? `${root}/research_registry/research_pipeline.sqlite3`;
  const child = Bun.spawn(
    ["python", "-m", "research_pipeline", "workflow", command, "--input-json", JSON.stringify({ ...payload, registry_path: registryPath })],
    { cwd: root, env: { ...process.env, PYTHONPATH: `${root}/src`, RESEARCH_PIPELINE_REGISTRY: String(registryPath) }, stdout: "pipe", stderr: "pipe" },
  );
  const stdout = await new Response(child.stdout).text();
  const stderr = await new Response(child.stderr).text();
  const code = await child.exited;
  if (code !== 0) throw new Error(`Phase C bridge ${command} failed (${code}): ${stderr || stdout}`);
  return JSON.parse(stdout);
}

async function phaseStep(command: string, payload: Record<string, unknown>, root: string, expected: string, skipped: Record<string, unknown>): Promise<unknown> {
  const status = await bridge("research-status", payload, root) as any;
  return status.strategy?.current_phase === expected ? bridge(command, payload, root) : skipped;
}

function summary(status: any, holdoutAccesses: number): z.infer<typeof phaseCSummary> {
  const finalReview = status.final_review;
  return {
    strategy_id: status.strategy.strategy_id,
    final_state: status.strategy.current_phase,
    classification: finalReview?.classification ?? status.strategy.current_phase,
    journal_entries: 0,
    holdout_accesses: holdoutAccesses,
    no_optimization_after_holdout: true,
    limitation: "Phase C workflow uses deterministic controller policy; no live trading or Fibonacci research is run.",
  };
}

export default smithers((ctx) => {
  const input = ctx.input;
  const started = ctx.outputMaybe(outputs.baseline, { nodeId: "baseline" });
  const edge = ctx.outputMaybe(outputs.edgeGate, { nodeId: "edge-gate" });
  const analysis = ctx.latest(outputs.analyst, "research-analyst");
  const proposalResult = ctx.latest(outputs.proposal, "proposal");
  const roundResult = ctx.latest(outputs.round, "parameter-round");
  const reviewResult = ctx.latest(outputs.review, "statistical-reviewer");
  const candidateResult = ctx.outputMaybe(outputs.candidate, { nodeId: "candidate-freeze" });
  const wf = ctx.outputMaybe(outputs.walkForward, { nodeId: "walk-forward" });
  const holdoutResult = ctx.outputMaybe(outputs.holdout, { nodeId: "holdout" });
  const stressResult = ctx.outputMaybe(outputs.stress, { nodeId: "stress-tests" });
  const throughputResult = ctx.outputMaybe(outputs.throughput, { nodeId: "throughput" });
  const critic = ctx.outputMaybe(outputs.finalReview, { nodeId: "final-research-critic" });
  const terminalEdge = edge !== undefined && edge.decision !== "CONTINUE";
  const familyResearchFinished = analysis?.decision === "FREEZE_CANDIDATE" || reviewResult?.decision === "VETO" || reviewResult?.decision === "INSUFFICIENT_EVIDENCE";

  return <Workflow name="trading-research-phase-c"><Sequence>
    <Task id="start" output={outputs.start} retries={0}>
      {async () => await bridge("research-start", { strategy_id: input.strategy_id, run_id: input.research_run_id, scenario: input.scenario, repository_root: input.repository_root, registry_path: input.registry_path }, input.repository_root)}
    </Task>
    <Task id="baseline" output={outputs.baseline} dependsOn={["start"]} retries={0}>
      {async () => await bridge("research-run-baseline", { strategy_id: input.strategy_id, scenario: input.scenario, repository_root: input.repository_root, registry_path: input.registry_path }, input.repository_root)}
    </Task>
    {started ? <Task id="edge-gate" output={outputs.edgeGate} dependsOn={["baseline"]} retries={0}>
      {async () => await bridge("research-edge-gate", { strategy_id: input.strategy_id, scenario: input.scenario, repository_root: input.repository_root, registry_path: input.registry_path }, input.repository_root)}
    </Task> : null}
    {edge?.decision === "CONTINUE" ? <Ralph id="parameter-research-loop" until={familyResearchFinished} maxIterations={6} onMaxReached="return-last">
      <Sequence>
        {/* Research Analyst: typed deterministic analysis and citation selection. */}
        <Task id="research-analyst" output={outputs.analyst} retries={0}>
          {async () => await bridge("research-analyze", { strategy_id: input.strategy_id, scenario: input.scenario, repository_root: input.repository_root, registry_path: input.registry_path }, input.repository_root)}
        </Task>
        <Branch if={analysis?.decision === "CONTINUE_PARAMETER_RESEARCH"} then={<Sequence>
          <Task id="proposal" output={outputs.proposal} dependsOn={["research-analyst"]} retries={0}>
            {async () => await bridge("research-propose-round", { strategy_id: input.strategy_id, decision: analysis, scenario: input.scenario, repository_root: input.repository_root, registry_path: input.registry_path }, input.repository_root)}
          </Task>
          <Task id="parameter-round" output={outputs.round} dependsOn={["proposal"]} retries={0}>
            {async () => await bridge("research-run-round", { strategy_id: input.strategy_id, proposal: proposalResult, scenario: input.scenario, repository_root: input.repository_root, registry_path: input.registry_path }, input.repository_root)}
          </Task>
          <Task id="statistical-reviewer" output={outputs.review} dependsOn={["parameter-round"]} retries={0}>
            {async () => await bridge("research-review-round", { strategy_id: input.strategy_id, round_id: roundResult?.round_id ?? "", scenario: input.scenario, repository_root: input.repository_root, registry_path: input.registry_path }, input.repository_root)}
          </Task>
          <Branch if={reviewResult?.decision === "SELECT"} then={<Task id="freeze-family" output={outputs.freezeFamily} dependsOn={["statistical-reviewer"]} retries={0}>
            {async () => await bridge("research-freeze-family", { strategy_id: input.strategy_id, round_id: reviewResult?.round_id ?? "", scenario: input.scenario, repository_root: input.repository_root, registry_path: input.registry_path }, input.repository_root)}
          </Task>} else={null} />
        </Sequence>} else={null} />
      </Sequence>
    </Ralph> : null}
    {edge?.decision === "CONTINUE" && analysis?.decision === "FREEZE_CANDIDATE" ? <Task id="candidate-freeze" output={outputs.candidate} retries={0}>
      {async () => await bridge("research-freeze-candidate", { strategy_id: input.strategy_id, scenario: input.scenario, repository_root: input.repository_root, registry_path: input.registry_path }, input.repository_root)}
    </Task> : null}
    {candidateResult ? <Sequence>
      <Task id="walk-forward" output={outputs.walkForward} dependsOn={["candidate-freeze"]} retries={0}>
        {async () => await phaseStep("research-walk-forward", { strategy_id: input.strategy_id, scenario: input.scenario, repository_root: input.repository_root, registry_path: input.registry_path }, input.repository_root, "CANDIDATE_FREEZE", { status: "SKIPPED", reason: "candidate freeze was not available" })}
      </Task>
      <Task id="holdout" output={outputs.holdout} dependsOn={["walk-forward"]} retries={0}>
        {async () => await phaseStep("research-holdout", { strategy_id: input.strategy_id, scenario: input.scenario, repository_root: input.repository_root, registry_path: input.registry_path }, input.repository_root, "HOLDOUT", { status: "SKIPPED", reason: "walk-forward did not pass" })}
      </Task>
      <Task id="stress-tests" output={outputs.stress} dependsOn={["holdout"]} retries={0}>
        {async () => await phaseStep("research-stress", { strategy_id: input.strategy_id, scenario: input.scenario, repository_root: input.repository_root, registry_path: input.registry_path }, input.repository_root, "STRESS_TESTS", { classification: "SKIPPED", reason: "holdout did not pass" })}
      </Task>
      <Task id="throughput" output={outputs.throughput} dependsOn={["stress-tests"]} retries={0}>
        {async () => await phaseStep("research-throughput", { strategy_id: input.strategy_id, scenario: input.scenario, repository_root: input.repository_root, registry_path: input.registry_path }, input.repository_root, "THROUGHPUT", { classification: "SKIPPED", reason: "stress tests were not reached" })}
      </Task>
      <Task id="final-research-critic" output={outputs.finalReview} dependsOn={["throughput"]} retries={0}>
        {async () => {
          const payload = { strategy_id: input.strategy_id, scenario: input.scenario, repository_root: input.repository_root, registry_path: input.registry_path };
          const status = await bridge("research-status", payload, input.repository_root) as any;
          if (status.strategy?.current_phase === "FINAL_REVIEW") return await bridge("research-final-review", payload, input.repository_root);
          return { strategy_id: input.strategy_id, strategy_version: status.strategy?.version ?? "unknown", classification: status.final_review?.classification ?? "INSUFFICIENT_EVIDENCE", current_phase: status.strategy?.current_phase ?? "UNKNOWN", evidence_strength: "insufficient", evidence: {}, metrics_cited: [], risks: ["final review was not reached"], rationale: "The deterministic controller stopped before final review." };
        }}
      </Task>
    </Sequence> : null}
    {(terminalEdge || critic || reviewResult?.decision === "VETO" || reviewResult?.decision === "INSUFFICIENT_EVIDENCE") ? <Task id="final-summary" output={outputs.summary} retries={0}>
      {async () => {
        const status = await bridge("research-status", { strategy_id: input.strategy_id, scenario: input.scenario, repository_root: input.repository_root, registry_path: input.registry_path }, input.repository_root);
        return summary(status, status.holdout_accesses ?? 0);
      }}
    </Task> : null}
  </Sequence></Workflow>;
});
