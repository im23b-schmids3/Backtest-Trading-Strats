// smithers-source: project-local
// smithers-metadata-version: 1
// smithers-display-name: Trading research master Phase F1
// smithers-description: One durable user-facing workflow from natural-language intake through final archived report.
/** @jsxImportSource smithers-orchestrator */
import { Approval, createSmithers, Sequence, Task } from "smithers-orchestrator";
import { z } from "zod/v4";
import { approvalDecision, masterStatus, masterSummary, outputs, phaseF1Input } from "../schemas/trading-research/master";

const { Workflow, outputs: registeredOutputs, smithers } = createSmithers({
  input: phaseF1Input,
  start: masterStatus,
  approval: approvalDecision,
  applied: masterStatus,
  implementation: masterStatus,
  verification: masterStatus,
  research: masterStatus,
  prop: masterStatus,
  portfolio: masterStatus,
  report: masterSummary,
});

async function bridge(command: string, payload: Record<string, unknown>, root: string, registryPath?: string): Promise<unknown> {
  const { runId: _runId, workflowId: _workflowId, ...cleanPayload } = payload;
  const child = Bun.spawn(["python", "-m", "research_pipeline", "workflow", command, "--input-json", JSON.stringify(cleanPayload)], {
    cwd: root,
    env: { ...process.env, PYTHONPATH: `${root}/src`, ...(registryPath ? { RESEARCH_PIPELINE_REGISTRY: registryPath } : {}) },
    stdout: "pipe", stderr: "pipe",
  });
  const stdout = await new Response(child.stdout).text();
  const stderr = await new Response(child.stderr).text();
  const code = await child.exited;
  if (code !== 0) throw new Error(`Phase F1 bridge ${command} failed (${code}): ${stderr || stdout}`);
  const parsed = JSON.parse(stdout) as Record<string, unknown>;
  if (typeof parsed.run_id === "string") {
    const { run_id, ...rest } = parsed;
    return { ...rest, source_run_id: run_id };
  }
  return parsed;
}

export default smithers((ctx) => {
  const input = ctx.input;
  const start = ctx.outputMaybe(registeredOutputs.start, { nodeId: "master-start" });
  const approval = ctx.outputMaybe(registeredOutputs.approval, { nodeId: "specification-approval" });
  const applied = ctx.outputMaybe(registeredOutputs.applied, { nodeId: "apply-approval" });
  return <Workflow name="trading-research-master"><Sequence>
    <Task id="master-start" output={registeredOutputs.start} retries={2} timeoutMs={15 * 60 * 1000}>{async () => bridge("master-start", input, input.repository_root, input.registry_path)}</Task>
    {start?.approval_status === "PENDING" ? <Approval id="specification-approval" output={registeredOutputs.approval} onDeny="continue" dependsOn={["master-start"]} request={{ title: `Approve generated strategy specification: ${start.strategy_id}`, summary: `Review the immutable specification artifact and hash in ${start.root_path}.` }} /> : null}
    {approval ? <Task id="apply-approval" output={registeredOutputs.applied} dependsOn={["specification-approval"]} retries={0}>{async () => bridge("master-approve", { run_id: start?.source_run_id, decision: approval.approved ? "APPROVE" : "REJECT", note: approval.note ?? null }, input.repository_root, input.registry_path)}</Task> : null}
    {applied?.approval_status === "APPROVED" ? <Task id="implementation" output={registeredOutputs.implementation} dependsOn={["apply-approval"]} retries={1} timeoutMs={30 * 60 * 1000}>{async () => bridge("master-resume", { run_id: applied.source_run_id, repository_root: input.repository_root }, input.repository_root, input.registry_path)}</Task> : null}
    {applied?.approval_status === "APPROVED" ? <Task id="verification" output={registeredOutputs.verification} dependsOn={["implementation"]} retries={1} timeoutMs={30 * 60 * 1000}>{async () => bridge("master-status", { run_id: applied.source_run_id, repository_root: input.repository_root }, input.repository_root, input.registry_path)}</Task> : null}
    {applied?.approval_status === "APPROVED" ? <Task id="research" output={registeredOutputs.research} dependsOn={["verification"]} retries={1} timeoutMs={60 * 60 * 1000}>{async () => bridge("master-status", { run_id: applied.source_run_id, repository_root: input.repository_root }, input.repository_root, input.registry_path)}</Task> : null}
    {applied?.approval_status === "APPROVED" ? <Task id="prop" output={registeredOutputs.prop} dependsOn={["research"]} retries={1} timeoutMs={60 * 60 * 1000}>{async () => bridge("master-status", { run_id: applied.source_run_id, repository_root: input.repository_root }, input.repository_root, input.registry_path)}</Task> : null}
    {applied?.approval_status === "APPROVED" ? <Task id="portfolio" output={registeredOutputs.portfolio} dependsOn={["prop"]} retries={1} timeoutMs={60 * 60 * 1000}>{async () => bridge("master-status", { run_id: applied.source_run_id, repository_root: input.repository_root }, input.repository_root, input.registry_path)}</Task> : null}
    {applied?.approval_status === "APPROVED" ? <Task id="final-report" output={registeredOutputs.report} dependsOn={["portfolio"]} retries={0}>{async () => { const status = masterStatus.parse(await bridge("master-status", { run_id: applied.source_run_id, repository_root: input.repository_root }, input.repository_root, input.registry_path)); return { source_run_id: status.source_run_id, strategy_id: status.strategy_id, current_step: status.current_step, outcome: status.outcome, report_path: status.report?.report_path ?? null, classification: status.report?.report_json?.classification ?? null, artifacts: status.artifacts.length }; }}</Task> : null}
  </Sequence></Workflow>;
});
