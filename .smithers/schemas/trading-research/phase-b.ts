import { z } from "zod/v4";

export const pipelineState = z.enum([
  "STRATEGY_DRAFT", "WAITING_FOR_SPEC_APPROVAL", "IMPLEMENTATION", "IMPLEMENTATION_VERIFICATION",
  "BASELINE_BACKTEST", "EDGE_GATE", "PARAMETER_RESEARCH", "CANDIDATE_FREEZE", "WALK_FORWARD",
  "HOLDOUT", "STRESS_TESTS", "THROUGHPUT", "RISK_SIZING", "PROP_SIMULATION",
  "MULTI_STRATEGY_PORTFOLIO", "FINAL_REVIEW", "ACCEPTED", "REJECTED", "INSUFFICIENT_EVIDENCE",
  "TECHNICAL_FAILURE", "MANUAL_REVIEW_REQUIRED",
]);

export const workflowInput = z.object({
  strategy_name: z.string().min(1).max(128),
  natural_language_description: z.string().min(10),
  requested_markets: z.array(z.string()).min(1),
  requested_timeframes: z.array(z.string()).min(1),
  optional_notes: z.string().nullable().optional(),
  repository_root: z.string().min(1),
  registry_path: z.string().min(1).nullable().optional(),
  dry_run: z.boolean().default(true),
  implementation_enabled: z.boolean().default(false),
  confirmed_facts: z.array(z.string()).default([]),
  assumptions: z.array(z.string()).default([]),
  missing_information: z.array(z.string()).default([]),
  ambiguities: z.array(z.string()).default([]),
  source_run_id: z.string().nullable().optional(),
  max_generation_attempts: z.number().int().min(1).max(3).default(3),
  max_repair_attempts: z.number().int().min(0).max(2).default(2),
}).strict();

export const generatedSpec = z.object({
  strategy_id: z.string(), version: z.string(), specification_path: z.string(), specification_hash: z.string(),
  assumptions: z.array(z.string()), ambiguities: z.array(z.string()), fields_requiring_confirmation: z.array(z.string()),
  manual_review_required: z.boolean(), approval_summary: z.string(),
  provenance: z.any(), validation_report_path: z.string().nullable().optional(),
  semantic_validation_report_path: z.string().nullable().optional(), attempt: z.number().int().positive(),
}).strict();

export const specValidation = z.object({
  valid: z.boolean(), strategy_id: z.string(), version: z.string(), specification_path: z.string(), specification_hash: z.string(),
  errors: z.array(z.string()), manual_review_required: z.boolean(),
  structured_errors: z.array(z.any()), semantic_report: z.any().nullable(),
  canonical_path: z.string().nullable(), approval_ready: z.boolean(), provenance: z.any(),
}).strict();

export const registration = z.object({
  registered: z.boolean(), idempotent_reuse: z.boolean(), strategy_id: z.string(), version: z.string(),
  current_phase: pipelineState, specification_hash: z.string(),
}).strict();

export const approvalDecision = z.object({
  approved: z.boolean(), note: z.string().nullable().optional(), decidedBy: z.string().nullable().optional(), decidedAt: z.string().nullable().optional(),
}).strict();

export const approvalResult = z.object({
  decision: z.enum(["APPROVE", "REJECT"]), approved: z.boolean(), note: z.string().nullable().optional(), strategy_id: z.string(), version: z.string(),
  current_phase: pipelineState, immutable_verified: z.boolean(),
}).strict();

export const implementationPlan = z.object({
  strategy_id: z.string(), version: z.string(), base_commit: z.string(), branch: z.string(), worktree_path: z.string(),
  allowed_files: z.array(z.string()), required_tests: z.array(z.array(z.string())), invariants: z.array(z.string()),
  prohibited_actions: z.array(z.string()), max_repair_attempts: z.number().int().nonnegative(),
}).strict();

export const codexExecution = z.object({
  success: z.boolean(), executed: z.boolean(), command: z.array(z.string()), cwd: z.string(), sandbox: z.string(), exit_code: z.number().int().nullable(),
  stdout: z.string(), stderr: z.string(), duration_ms: z.number().int().nonnegative(), timed_out: z.boolean(), error_type: z.string().nullable().optional(),
  session_id: z.string().nullable().optional(), files_changed: z.array(z.string()), resulting_commit: z.string().nullable().optional(),
}).strict();

export const testResult = z.object({
  passed: z.boolean(), command: z.array(z.string()), exit_code: z.number().int().nullable(), parsed_passed: z.number().int(), parsed_failed: z.number().int(),
  parsed_skipped: z.number().int(), duration_ms: z.number().int().nonnegative(), report_path: z.string().nullable(), failure_summary: z.string(), executed: z.boolean(),
}).strict();

export const repairResult = z.object({
  attempt: z.number().int().positive(), budget_remaining: z.number().int().nonnegative(), codex_result: codexExecution,
  test_result: testResult.nullable(), material_change_detected: z.boolean(), stopped: z.boolean(), reason: z.string(),
}).strict();

export const finalSummary = z.object({
  strategy_id: z.string(), version: z.string(), final_state: pipelineState, approval: z.string(), manual_review_required: z.boolean(),
  implementation_executed: z.boolean(), tests_passed: z.boolean(), repair_attempts: z.number().int().nonnegative(), registry_reconciled: z.boolean(),
  worktree_path: z.string().nullable(), outputs: z.array(z.string()), limitation: z.string(),
}).strict();
