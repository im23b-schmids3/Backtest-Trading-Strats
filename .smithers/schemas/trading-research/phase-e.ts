import { z } from "zod/v4";

export const phaseEInput = z.object({
  portfolio_id: z.string().min(1),
  portfolio_config: z.string().min(1),
  repository_root: z.string().min(1),
  registry_path: z.string().min(1),
  scenario: z.string().default("complementary"),
  dry_run: z.boolean().default(true),
}).strict();

export const portfolioStep = z.object({
  portfolio_id: z.string().optional(),
  status: z.string().optional(),
  current_phase: z.string().optional(),
  errors: z.array(z.string()).optional(),
}).passthrough();

export const portfolioStatus = z.object({
  specification: z.record(z.string(), z.any()),
  run: z.record(z.string(), z.any()),
  budget: z.any().nullable(),
  candidates: z.array(z.any()),
  signals: z.array(z.any()),
  overlap: z.array(z.any()),
  correlation: z.array(z.any()),
  risk: z.array(z.any()),
  prop: z.array(z.any()),
  ablation: z.array(z.any()),
  marginal: z.array(z.any()),
  stress: z.array(z.any()),
  final_review: z.any().nullable(),
}).strict();

export const roleReview = z.object({
  role: z.enum(["Portfolio Analyst", "Portfolio Statistical Reviewer", "Compliance Reviewer", "Final Portfolio Critic"]),
  portfolio_id: z.string(),
  phase: z.string(),
  status: z.string(),
  evidence: z.array(z.string()),
  blocking_issues: z.array(z.string()),
  metrics_cited: z.array(z.string()),
}).strict();

export const phaseESummary = z.object({
  portfolio_id: z.string(),
  phase: z.string(),
  classification: z.string(),
  selected_candidate_id: z.string().nullable(),
  members: z.array(z.string()),
  journal_entries: z.number().int().nonnegative(),
  no_optimization: z.literal(true),
  no_trading: z.literal(true),
  limitations: z.array(z.string()),
}).strict();

export const outputs = {
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
};
