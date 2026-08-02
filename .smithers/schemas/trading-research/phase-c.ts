import { z } from "zod/v4";

export const phaseCInput = z.object({
  strategy_id: z.string().min(1), repository_root: z.string().min(1), registry_path: z.string().min(1),
  scenario: z.string().default("strong-stable"), dry_run: z.boolean().default(true), research_run_id: z.string().nullable().optional(),
}).strict();
export const start = z.object({}).passthrough();

export const phaseCArtifact = z.object({ experiment_id: z.string(), strategy_id: z.string(), strategy_version: z.string(), phase: z.string(), experiment_dir: z.string(), input_path: z.string(), metrics_path: z.string(), diagnostic_manifest_path: z.string().nullable(), report_hashes: z.record(z.string(), z.string()), dataset_hash: z.string(), split_hash: z.string(), code_commit: z.string().nullable().optional(), command: z.array(z.string()), status: z.string(), metrics: z.record(z.string(), z.any()), diagnostic_manifest: z.record(z.string(), z.any()).default({}) }).strict();
export const baseline = z.object({ artifact: phaseCArtifact, verification_outcome: z.string(), gate_outcomes: z.array(z.record(z.string(), z.any())).default([]), edge_decision: z.string().nullable().optional() }).strict();
export const edgeGate = z.object({ decision: z.enum(["CONTINUE", "REJECT", "INSUFFICIENT_EVIDENCE", "MANUAL_REVIEW_REQUIRED"]), outcomes: z.array(z.record(z.string(), z.any())), metrics: z.record(z.string(), z.any()) }).strict();
export const analystDecision = z.object({ strategy_id: z.string(), strategy_version: z.string(), current_phase: z.string(), decision: z.string(), confidence: z.number(), evidence_strength: z.string(), primary_bottleneck: z.string(), selected_parameter_family: z.string().nullable(), current_value: z.any(), proposed_values: z.array(z.any()), proposal_method: z.string(), parameter_hypothesis: z.string(), expected_behavior: z.string(), files_inspected: z.array(z.string()), metrics_cited: z.array(z.record(z.string(), z.any())), risks: z.array(z.string()), overfitting_risk: z.number(), stop_reason: z.string().nullable().optional(), next_phase: z.string().nullable().optional(), rationale: z.string() }).strict();
export const proposal = z.object({ strategy_id: z.string(), strategy_version: z.string(), family: z.string(), current_value: z.any(), proposed_values: z.array(z.any()), round_number: z.number().int(), hypothesis: z.string(), reason: z.string() }).strict();
export const review = z.object({ strategy_id: z.string(), strategy_version: z.string(), round_id: z.string(), decision: z.string(), stable_region: z.array(z.any()), selected_value: z.any(), isolated_maximum_risk: z.boolean(), evidence_strength: z.string(), metrics_cited: z.array(z.record(z.string(), z.any())), veto_reason: z.string().nullable().optional(), rationale: z.string() }).strict();
export const freezeFamily = z.object({ round_id: z.string(), family: z.string(), selected_value: z.any(), status: z.literal("FROZEN") }).strict();
export const round = z.object({ round_id: z.string(), family: z.string(), experiments: z.array(z.record(z.string(), z.any())), review, selected_value: z.any(), stable_region: z.array(z.any()), stopped: z.boolean(), stop_reason: z.string().nullable().optional() }).strict();
export const candidate = z.object({}).passthrough();
export const walkForward = z.object({}).passthrough();
export const holdout = z.object({}).passthrough();
export const stress = z.object({}).passthrough();
export const throughput = z.object({}).passthrough();
export const finalReview = z.object({ strategy_id: z.string(), strategy_version: z.string(), classification: z.string(), current_phase: z.string(), evidence_strength: z.string(), evidence: z.record(z.string(), z.any()), metrics_cited: z.array(z.record(z.string(), z.any())), risks: z.array(z.string()), rationale: z.string(), next_phase: z.string().nullable().optional() }).strict();
export const phaseCSummary = z.object({ strategy_id: z.string(), final_state: z.string(), classification: z.string(), journal_entries: z.number().int(), holdout_accesses: z.number().int(), no_optimization_after_holdout: z.boolean(), limitation: z.string() }).strict();
