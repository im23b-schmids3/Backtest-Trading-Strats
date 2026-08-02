// smithers-source: project-local
// smithers-metadata-version: 1
// smithers-display-name: Trading research master Phase F2
// smithers-description: Durable dry-run or real-mode strategy intake, approval, implementation verification, and research handoff.
/** @jsxImportSource smithers-orchestrator */
import { Approval, createSmithers, Sequence, Task, WaitForEvent } from "smithers-orchestrator";
import { z } from "zod/v4";
import { approvalDecision, externalExecution as externalExecutionSchema, masterStatus, masterSummary, outputs, phaseF1Input, specificationExternalExecution, specificationStatus } from "../schemas/trading-research/master";

const { Workflow, outputs: registeredOutputs, smithers } = createSmithers({
  input: phaseF1Input,
  start: masterStatus,
  specification: specificationStatus,
  repair: masterStatus,
  postValidation: specificationStatus,
  specificationResume: masterStatus,
  approval: approvalDecision,
  applied: masterStatus,
  specificationExternalExecution,
  externalExecution: externalExecutionSchema,
  resume: masterStatus,
  implementation: masterStatus,
  verification: masterStatus,
  research: masterStatus,
  prop: masterStatus,
  portfolio: masterStatus,
  report: masterSummary,
});

async function bridge(command: string, payload: Record<string, unknown>, root: string, registryPath?: string): Promise<unknown> {
  const { runId: _runId, workflowId: _workflowId, ...cleanPayload } = payload;
  const bridgedPayload = command === "master-start" ? cleanPayload : { ...cleanPayload, smithers_run_id: process.env.SMITHERS_RUN_ID ?? null };
  const child = Bun.spawn(["python", "-m", "research_pipeline", "workflow", command, "--input-json", JSON.stringify(bridgedPayload)], {
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
  const specification = ctx.outputMaybe(registeredOutputs.specification, { nodeId: "validate-specification" });
  const repair = ctx.outputMaybe(registeredOutputs.repair, { nodeId: "repair-specification" });
  const postValidation = ctx.outputMaybe(registeredOutputs.postValidation, { nodeId: "validate-after-repair" });
  const specificationResume = ctx.outputMaybe(registeredOutputs.specificationResume, { nodeId: "resume-after-specification-external" });
  const approval = ctx.outputMaybe(registeredOutputs.approval, { nodeId: "specification-approval" });
  const applied = ctx.outputMaybe(registeredOutputs.applied, { nodeId: "apply-approval" });
  const implementation = ctx.outputMaybe(registeredOutputs.implementation, { nodeId: "implementation" });
  const externalExecution = ctx.outputMaybe(registeredOutputs.externalExecution, { nodeId: "external-codex-wait" });
  const waitingForExternal = implementation?.outcome === "WAITING_EXTERNAL_CODEX";
  const externalReady = !waitingForExternal || externalExecution !== undefined;
  const specificationNeedsExternal = postValidation?.master_status?.outcome === "WAITING_EXTERNAL_CODEX";
  const specificationReady = postValidation?.specification.approval_available || specificationResume?.approval_status === "PENDING";
  return <Workflow name="trading-research-master"><Sequence>
    <Task id="master-start" output={registeredOutputs.start} retries={2} timeoutMs={15 * 60 * 1000}>{async () => bridge("master-start", input, input.repository_root, input.registry_path)}</Task>
    {start ? <Task id="validate-specification" output={registeredOutputs.specification} dependsOn={["master-start"]} retries={0}>{async () => bridge("master-specification-status", { run_id: start.source_run_id }, input.repository_root, input.registry_path)}</Task> : null}
    {specification?.specification.latest_validation_outcome !== "VALID" && specification?.specification.latest_validation_outcome !== "BLOCKED" && !specification?.specification.failure && !specification?.master_status?.outcome?.startsWith("WAITING_EXTERNAL_SPECIFICATION") ? <Task id="repair-specification" output={registeredOutputs.repair} dependsOn={["validate-specification"]} retries={0} timeoutMs={15 * 60 * 1000}>{async () => bridge("master-specification-retry", { run_id: start?.source_run_id }, input.repository_root, input.registry_path)}</Task> : null}
    {specification ? <Task id="validate-after-repair" output={registeredOutputs.postValidation} dependsOn={repair ? ["repair-specification"] : ["validate-specification"]} retries={0}>{async () => bridge("master-specification-status", { run_id: start?.source_run_id }, input.repository_root, input.registry_path)}</Task> : null}
    {specificationNeedsExternal ? <WaitForEvent id="specification-external-wait" event="external.codex.specification.completed" correlationId={postValidation?.master_status?.source_run_id} output={registeredOutputs.specificationExternalExecution} outputSchema={specificationExternalExecution} timeoutMs={7 * 24 * 60 * 60 * 1000} onTimeout="fail" label={postValidation?.master_status?.outcome === "WAITING_EXTERNAL_SPECIFICATION_REPAIR" ? "EXTERNAL_SPECIFICATION_REPAIR_REQUIRED" : "EXTERNAL_SPECIFICATION_GENERATION_REQUIRED"} /> : null}
    {specificationNeedsExternal && specificationExternalExecution && !specificationResume ? <Task id="resume-after-specification-external" output={registeredOutputs.specificationResume} dependsOn={["specification-external-wait"]} retries={0} timeoutMs={30 * 60 * 1000}>{async () => bridge("master-resume", { run_id: postValidation?.master_status?.source_run_id, repository_root: input.repository_root, mode: input.mode }, input.repository_root, input.registry_path)}</Task> : null}
    {specificationReady && start?.approval_status === "PENDING" ? <Approval id="specification-approval" output={registeredOutputs.approval} onDeny="continue" dependsOn={[specificationNeedsExternal ? "resume-after-specification-external" : "validate-after-repair"]} request={{ title: `Approve ${input.mode} strategy specification: ${start.strategy_id}`, summary: `Review the immutable specification artifact and hash in ${start.root_path}.` }} /> : null}
    {approval ? <Task id="apply-approval" output={registeredOutputs.applied} dependsOn={["specification-approval"]} retries={0}>{async () => bridge("master-approve", { run_id: start?.source_run_id, decision: approval.approved ? "APPROVE" : "REJECT", note: approval.note ?? null }, input.repository_root, input.registry_path)}</Task> : null}
    {applied?.approval_status === "APPROVED" ? <Task id="implementation" output={registeredOutputs.implementation} dependsOn={["apply-approval"]} retries={0} timeoutMs={30 * 60 * 1000}>{async () => bridge("master-resume", { run_id: applied.source_run_id, repository_root: input.repository_root, mode: input.mode }, input.repository_root, input.registry_path)}</Task> : null}
    {waitingForExternal ? <WaitForEvent id="external-codex-wait" event="external.codex.completed" correlationId={implementation?.source_run_id} dependsOn={["implementation"]} output={registeredOutputs.externalExecution} outputSchema={externalExecutionSchema} timeoutMs={7 * 24 * 60 * 60 * 1000} onTimeout="fail" label="EXTERNAL_CODEX_EXECUTION_REQUIRED" /> : null}
    {applied?.approval_status === "APPROVED" && externalReady && waitingForExternal ? <Task id="resume-after-external" output={registeredOutputs.resume} dependsOn={["external-codex-wait"]} retries={0} timeoutMs={30 * 60 * 1000}>{async () => bridge("master-resume", { run_id: applied.source_run_id, repository_root: input.repository_root, mode: input.mode }, input.repository_root, input.registry_path)}</Task> : null}
    {applied?.approval_status === "APPROVED" && externalReady ? <Task id="verification" output={registeredOutputs.verification} dependsOn={[waitingForExternal ? "resume-after-external" : "implementation"]} retries={1} timeoutMs={30 * 60 * 1000}>{async () => bridge("master-status", { run_id: applied.source_run_id, repository_root: input.repository_root }, input.repository_root, input.registry_path)}</Task> : null}
    {applied?.approval_status === "APPROVED" && externalReady ? <Task id="research" output={registeredOutputs.research} dependsOn={["verification"]} retries={1} timeoutMs={60 * 60 * 1000}>{async () => bridge("master-status", { run_id: applied.source_run_id, repository_root: input.repository_root }, input.repository_root, input.registry_path)}</Task> : null}
    {applied?.approval_status === "APPROVED" && externalReady ? <Task id="prop" output={registeredOutputs.prop} dependsOn={["research"]} retries={1} timeoutMs={60 * 60 * 1000}>{async () => bridge("master-status", { run_id: applied.source_run_id, repository_root: input.repository_root }, input.repository_root, input.registry_path)}</Task> : null}
    {applied?.approval_status === "APPROVED" && externalReady ? <Task id="portfolio" output={registeredOutputs.portfolio} dependsOn={["prop"]} retries={1} timeoutMs={60 * 60 * 1000}>{async () => bridge("master-status", { run_id: applied.source_run_id, repository_root: input.repository_root }, input.repository_root, input.registry_path)}</Task> : null}
    {applied?.approval_status === "APPROVED" && externalReady ? <Task id="final-report" output={registeredOutputs.report} dependsOn={["portfolio"]} retries={0}>{async () => { const status = masterStatus.parse(await bridge("master-status", { run_id: applied.source_run_id, repository_root: input.repository_root }, input.repository_root, input.registry_path)); return { source_run_id: status.source_run_id, strategy_id: status.strategy_id, current_step: status.current_step, outcome: status.outcome, report_path: status.report?.report_path ?? null, classification: status.report?.report_json?.classification ?? null, artifacts: status.artifacts.length }; }}</Task> : null}
  </Sequence></Workflow>;
});
