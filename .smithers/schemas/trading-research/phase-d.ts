import { z } from "zod/v4";

export const phaseDInput = z.object({
  strategy_id: z.string().min(1),
  repository_root: z.string().min(1),
  registry_path: z.string().min(1),
  product: z.enum(["Alpha Futures Zero 25K", "Alpha Futures Zero 50K"]).default("Alpha Futures Zero 25K"),
  scenario: z.string().default("profitable"),
  dry_run: z.boolean().default(true),
  research_run_id: z.string().min(1).optional(),
}).strict();

export const propStep = z.object({
  status: z.string().optional(),
  errors: z.array(z.string()).optional(),
}).passthrough();

export const propStatus = z.object({
  strategy: z.record(z.string(), z.any()),
  prop_run: z.record(z.string(), z.any()),
  budget: z.any().nullable(),
  rules: z.any().nullable(),
  contracts: z.any().nullable(),
  mappings: z.any().nullable(),
  risk: z.any().nullable(),
  scenarios: z.array(z.any()),
  compliance: z.any().nullable(),
  economics: z.any().nullable(),
  final_review: z.any().nullable(),
  holdout_accesses: z.number().int().nonnegative(),
}).strict();

export const roleReview = z.object({
  role: z.enum(["Prop Research Analyst", "Compliance Reviewer"]),
  strategy_id: z.string(),
  phase: z.string(),
  status: z.string(),
  evidence: z.array(z.string()),
  blocking_issues: z.array(z.string()),
}).strict();

export const finalReview = z.object({
  strategy_id: z.string(),
  strategy_version: z.string(),
  classification: z.string(),
  scenario_id: z.string(),
  metrics: z.record(z.string(), z.any()),
  compliance: z.record(z.string(), z.any()),
  data_limitations: z.record(z.string(), z.any()),
  metrics_cited: z.array(z.record(z.string(), z.any())),
  rationale: z.string(),
  next_phase: z.string().nullable().optional(),
}).strict();

export const phaseDSummary = z.object({
  strategy_id: z.string(),
  strategy_version: z.string(),
  prop_phase: z.string(),
  classification: z.string(),
  product: z.string(),
  scenario: z.string(),
  journal_entries: z.number().int().nonnegative(),
  holdout_accesses: z.number().int().nonnegative(),
  b5_verified: z.boolean(),
  no_optimization: z.literal(true),
  limitations: z.array(z.string()),
}).strict();

export const outputs = { start: propStep, rules: propStep, contracts: propStep, reconcile: propStep, risk: propStep, scenarios: propStep, analyst: roleReview, compliance: roleReview, finalReview, summary: phaseDSummary };
