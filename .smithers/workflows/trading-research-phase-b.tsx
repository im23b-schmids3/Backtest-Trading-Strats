// smithers-source: project-local
// smithers-metadata-version: 1
// smithers-display-name: Trading research Phase B
// smithers-description: Durable strategy specification, approval, isolated implementation, and technical verification.
/** @jsxImportSource smithers-orchestrator */
import { Approval, createSmithers, Ralph, Sequence, Task } from "smithers-orchestrator";
import { z } from "zod/v4";
import {
  approvalDecision, approvalResult, codexExecution, finalSummary, generatedSpec, implementationPlan,
  registration, specValidation, testResult, workflowInput, repairResult,
} from "../schemas/trading-research/phase-b";

const SPEC_REPAIR_ATTEMPTS = 3;
let activeRegistryPath: string | undefined;

const { Workflow, outputs, smithers } = createSmithers({
  input: workflowInput,
  generated: generatedSpec,
  validation: specValidation,
  registration,
  approval: approvalDecision,
  approvalResult,
  implementationPlan,
  implementation: codexExecution,
  initialTests: testResult,
  repair: repairResult,
  repairTests: testResult,
  final: finalSummary,
});

async function bridge(command: string, payload: unknown, repositoryRoot: string): Promise<unknown> {
  const registryPath = (payload as { registry_path?: string }).registry_path ?? activeRegistryPath ?? process.env.RESEARCH_PIPELINE_REGISTRY ?? `${repositoryRoot}/research_registry/research_pipeline.sqlite3`;
  const bridgedPayload: Record<string, unknown> = { ...(payload as Record<string, unknown>), registry_path: registryPath };
  if ("source_run_id" in bridgedPayload) {
    bridgedPayload.run_id = bridgedPayload.source_run_id;
    delete bridgedPayload.source_run_id;
  }
  const env = { ...process.env, PYTHONPATH: `${repositoryRoot}/src`, RESEARCH_PIPELINE_REGISTRY: registryPath };
  const child = Bun.spawn(["python", "-m", "research_pipeline", "workflow", command, "--input-json", JSON.stringify(bridgedPayload)], {
    cwd: repositoryRoot, env, stdout: "pipe", stderr: "pipe",
  });
  const stdout = await new Response(child.stdout).text();
  const stderr = await new Response(child.stderr).text();
  const exitCode = await child.exited;
  if (exitCode !== 0) throw new Error(`Phase B bridge ${command} failed (${exitCode}): ${stderr || stdout}`);
  return JSON.parse(stdout);
}

function mockFinal(strategyId: string, version: string, state: "MANUAL_REVIEW_REQUIRED" | "REJECTED" | "TECHNICAL_FAILURE", note: string): z.infer<typeof finalSummary> {
  return { strategy_id: strategyId, version, final_state: state, approval: state === "REJECTED" ? "REJECT" : state, manual_review_required: state === "MANUAL_REVIEW_REQUIRED", implementation_executed: false, tests_passed: false, repair_attempts: 0, registry_reconciled: state === "REJECTED", worktree_path: null, outputs: [], limitation: note };
}

export default smithers((ctx) => {
  const input = ctx.input;
  activeRegistryPath = input.registry_path ?? undefined;
  const generated = ctx.outputMaybe(outputs.generated, { nodeId: "generate-spec" });
  const validation = ctx.outputMaybe(outputs.validation, { nodeId: "validate-spec" });
  const registrationResult = ctx.outputMaybe(outputs.registration, { nodeId: "register-spec" });
  const approval = ctx.outputMaybe(outputs.approval, { nodeId: "approve-spec" });
  const appliedApproval = ctx.outputMaybe(outputs.approvalResult, { nodeId: "apply-approval" });
  const plan = ctx.outputMaybe(outputs.implementationPlan, { nodeId: "implementation-plan" });
  const implementation = ctx.outputMaybe(outputs.implementation, { nodeId: "implement-strategy" });
  const initialTests = ctx.outputMaybe(outputs.initialTests, { nodeId: "initial-tests" });
  const repair = ctx.latest(outputs.repair, "repair");
  const repairTests = ctx.latest(outputs.repairTests, "repair-tests");
  const finalTests = repairTests ?? initialTests;
  const manual = generated?.manual_review_required === true;
  const approved = appliedApproval?.approved === true;
  const rejected = appliedApproval?.approved === false;
  const validationFailed = validation !== undefined && validation.valid === false;
  const implementationFailed = implementation !== undefined && implementation.success === false;
  const repairDone = repair?.attempt !== undefined && repair.attempt >= (plan?.max_repair_attempts ?? 0);

  return (
    <Workflow name="trading-research-phase-b">
      <Sequence>
        <Task id="generate-spec" output={outputs.generated} retries={SPEC_REPAIR_ATTEMPTS} timeoutMs={15 * 60 * 1000}>
          {async () => await bridge("generate-spec", input, input.repository_root)}
        </Task>
        {generated && !manual ? (
          <Task id="validate-spec" output={outputs.validation} dependsOn={["generate-spec"]} retries={0}>
            {async () => await bridge("validate-spec", generated, input.repository_root)}
          </Task>
        ) : null}
        {validation?.valid && !validation.manual_review_required ? (
          <Task id="register-spec" output={outputs.registration} dependsOn={["validate-spec"]} retries={0}>
            {async () => await bridge("register-generated-spec", validation, input.repository_root)}
          </Task>
        ) : null}
        {registrationResult && !manual ? (
          <Approval
            id="approve-spec"
            output={outputs.approval}
            onDeny="continue"
            dependsOn={["register-spec"]}
            request={{
              title: `Approve strategy specification: ${registrationResult.strategy_id}@${registrationResult.version}`,
              summary: generated?.approval_summary ?? `Review specification ${registrationResult.specification_hash}`,
            }}
          />
        ) : null}
        {approval && registrationResult ? (
          <Task id="apply-approval" output={outputs.approvalResult} dependsOn={["approve-spec"]} retries={0}>
            {async () => await bridge("approve", { strategy_id: registrationResult.strategy_id, decision: approval.approved ? "APPROVE" : "REJECT", note: approval.note ?? null }, input.repository_root)}
          </Task>
        ) : null}
        {approved && registrationResult ? (
          <Task id="implementation-plan" output={outputs.implementationPlan} dependsOn={["apply-approval"]} retries={0}>
            {async () => await bridge("implementation-plan", { strategy_id: registrationResult.strategy_id, repository_root: input.repository_root, dry_run: input.dry_run }, input.repository_root)}
          </Task>
        ) : null}
        {approved && plan ? (
          <Task id="implement-strategy" output={outputs.implementation} dependsOn={["implementation-plan"]} retries={0} timeoutMs={30 * 60 * 1000}>
            {async () => {
              const prompt = `Implement approved strategy ${plan.strategy_id}@${plan.version}. Allowed files: ${plan.allowed_files.join(", ")}. Invariants: ${plan.invariants.join(" | ")}. Do not run backtests or optimization. Required tests: ${JSON.stringify(plan.required_tests)}.`;
              return await bridge("execute-codex", { strategy_id: plan.strategy_id, repository_root: input.repository_root, plan, prompt, task_name: "implementation", dry_run: input.dry_run || !input.implementation_enabled }, input.repository_root);
            }}
          </Task>
        ) : null}
        {implementation?.success && plan ? (
          <Task id="initial-tests" output={outputs.initialTests} dependsOn={["implement-strategy"]} retries={0} timeoutMs={30 * 60 * 1000}>
            {async () => await bridge("run-required-tests", { repository_root: input.repository_root, worktree_path: plan.worktree_path, required_tests: plan.required_tests, dry_run: input.dry_run || !input.implementation_enabled }, input.repository_root)}
          </Task>
        ) : null}
        {initialTests && !initialTests.passed && plan ? (
          <Ralph id="repair-loop" until={initialTests.passed || repairTests?.passed === true || repairDone} maxIterations={plan.max_repair_attempts} onMaxReached="return-last">
            <Sequence>
              <Task id="repair" output={outputs.repair} retries={0} timeoutMs={30 * 60 * 1000}>
                {async () => {
                  const attempt = (repair?.attempt ?? 0) + 1;
                  const prompt = `Repair only the concrete technical failures for ${plan.strategy_id}@${plan.version}. Failure output: ${repairTests?.failure_summary ?? initialTests.failure_summary}. Preserve invariants: ${plan.invariants.join(" | ")}. Do not change strategy rules, run backtests, or optimize.`;
                  const result = await bridge("execute-codex", { strategy_id: plan.strategy_id, repository_root: input.repository_root, plan, prompt, task_name: `repair-${attempt}`, dry_run: input.dry_run || !input.implementation_enabled }, input.repository_root);
                  return { attempt, budget_remaining: Math.max(0, plan.max_repair_attempts - attempt), codex_result: result, test_result: null, material_change_detected: false, stopped: false, reason: "bounded repair attempt recorded" };
                }}
              </Task>
              <Task id="repair-tests" output={outputs.repairTests} dependsOn={["repair"]} retries={0} timeoutMs={30 * 60 * 1000}>
                {async () => await bridge("run-required-tests", { repository_root: input.repository_root, worktree_path: plan.worktree_path, required_tests: plan.required_tests, dry_run: input.dry_run || !input.implementation_enabled }, input.repository_root)}
              </Task>
            </Sequence>
          </Ralph>
        ) : null}
        {(manual || rejected || validationFailed || implementationFailed || finalTests) ? <Task id="technical-verification" output={outputs.final} retries={0}>
          {async () => {
            if (manual) return mockFinal(generated?.strategy_id ?? input.strategy_name, generated?.version ?? "phase-b-1", "MANUAL_REVIEW_REQUIRED", "Material ambiguity requires human review before registration.");
            if (rejected) return mockFinal(registrationResult!.strategy_id, registrationResult!.version, "REJECTED", "Specification was rejected; implementation was not started.");
            if (validationFailed) return mockFinal(generated?.strategy_id ?? input.strategy_name, generated?.version ?? "phase-b-1", "TECHNICAL_FAILURE", `Specification validation failed: ${validation?.errors.join(" | ")}`);
            if (implementationFailed) {
              const failureTest = { passed: false, command: implementation!.command, exit_code: implementation!.exit_code, parsed_passed: 0, parsed_failed: 1, parsed_skipped: 0, duration_ms: implementation!.duration_ms, report_path: null, failure_summary: implementation!.stderr || implementation!.error_type || "Codex implementation failed", executed: implementation!.executed };
              return await bridge("technical-verification", { strategy_id: registrationResult!.strategy_id, test_result: failureTest, implementation_executed: implementation!.executed, repair_attempts: 0, worktree_path: plan?.worktree_path ?? null }, input.repository_root);
            }
            return await bridge("technical-verification", { strategy_id: registrationResult!.strategy_id, test_result: finalTests, implementation_executed: implementation?.executed ?? false, repair_attempts: repair?.attempt ?? 0, worktree_path: plan?.worktree_path ?? null }, input.repository_root);
          }}
        </Task> : null}
      </Sequence>
    </Workflow>
  );
});
