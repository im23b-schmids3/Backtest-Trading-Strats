import { z } from "zod/v4";

export const phaseF1Input = z.object({
  intake_path: z.string().min(1),
  repository_root: z.string().min(1),
  registry_path: z.string().min(1),
  dry_run: z.boolean().default(true),
  implementation_enabled: z.boolean().default(false),
  research_scenario: z.string().default("strong-stable"),
  prop_scenario: z.string().default("profitable"),
  portfolio_scenario: z.string().default("complementary"),
  prop_product: z.enum(["Alpha Futures Zero 25K", "Alpha Futures Zero 50K"]).default("Alpha Futures Zero 25K"),
  mode: z.enum(["dry_run", "real_run"]).default("dry_run"),
  allow_proxy_data: z.boolean().default(false),
  prebuilt_spec_path: z.string().nullable().optional(),
  max_generation_attempts: z.number().int().min(1).max(3).default(3),
  max_repair_attempts: z.number().int().min(0).max(2).default(2),
}).strict();

export const masterStatus = z.object({
  source_run_id: z.string(), strategy_id: z.string(), strategy_version: z.string().nullable(),
  current_step: z.string(), outcome: z.string(), approval_status: z.string(), root_path: z.string(),
  phase_results: z.array(z.any()), journal_entries: z.number().int().nonnegative(),
  artifacts: z.array(z.any()), report: z.any().nullable(), mode: z.enum(["dry_run", "real_run"]).default("dry_run"),
  pipeline_status: z.string().optional(), implementation_job_id: z.string().nullable().optional(),
  external_executor_required: z.boolean().optional(), worktree_preflight: z.any().optional(),
  codex_execution_status: z.string().nullable().optional(), implementation_test_status: z.string().nullable().optional(),
  b5_available: z.boolean().optional(), next_command: z.string().nullable().optional(),
}).strict();

export const approvalDecision = z.object({
  approved: z.boolean(), note: z.string().nullable().optional(), decidedBy: z.string().nullable().optional(), decidedAt: z.string().nullable().optional(),
}).strict();

export const masterSummary = z.object({
  source_run_id: z.string(), strategy_id: z.string(), current_step: z.string(), outcome: z.string(),
  report_path: z.string().nullable(), classification: z.string().nullable(), artifacts: z.number().int().nonnegative(),
}).strict();

export const specificationStatus = z.object({
  source_run_id: z.string(),
  master_status: masterStatus,
  specification: z.object({
    source_run_id: z.string(),
    attempt_count: z.number().int().nonnegative(),
    candidate_attempt_count: z.number().int().nonnegative(),
    repair_attempt_count: z.number().int().nonnegative(),
    latest_attempt: z.any().nullable(),
    latest_validation_outcome: z.string(),
    latest_schema_validation_outcome: z.string(),
    latest_semantic_validation_outcome: z.string(),
    blocking_ambiguities: z.array(z.any()),
    approval_available: z.boolean(),
    failure: z.any().nullable(),
    current_specification_job: z.any().nullable(),
    specification_jobs: z.array(z.any()),
    repair_budget_remaining: z.number().int().nonnegative(),
    next_command: z.string().nullable(),
  }).strict(),
}).strict();

export const externalExecution = z.object({
  source_run_id: z.string(), job_id: z.string(), status: z.string(),
}).strict();

export const specificationExternalExecution = externalExecution;

export const outputs = { start: masterStatus, specification: specificationStatus, repair: masterStatus, postValidation: specificationStatus, specificationResume: masterStatus, approval: approvalDecision, applied: masterStatus, specificationExternalExecution, externalExecution, resume: masterStatus, report: masterSummary };
