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
}).strict();

export const masterStatus = z.object({
  source_run_id: z.string(), strategy_id: z.string(), strategy_version: z.string().nullable(),
  current_step: z.string(), outcome: z.string(), approval_status: z.string(), root_path: z.string(),
  phase_results: z.array(z.any()), journal_entries: z.number().int().nonnegative(),
  artifacts: z.array(z.any()), report: z.any().nullable(),
}).strict();

export const approvalDecision = z.object({
  approved: z.boolean(), note: z.string().nullable().optional(), decidedBy: z.string().nullable().optional(), decidedAt: z.string().nullable().optional(),
}).strict();

export const masterSummary = z.object({
  source_run_id: z.string(), strategy_id: z.string(), current_step: z.string(), outcome: z.string(),
  report_path: z.string().nullable(), classification: z.string().nullable(), artifacts: z.number().int().nonnegative(),
}).strict();

export const outputs = { start: masterStatus, approval: approvalDecision, applied: masterStatus, resume: masterStatus, report: masterSummary };
