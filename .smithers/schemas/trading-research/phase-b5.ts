import { z } from "zod/v4";

export const verificationInput = z.object({
  strategy_id: z.string().min(1), repository_root: z.string().min(1), registry_path: z.string().min(1).nullable().optional(),
  manifest_path: z.string().min(1), dry_run: z.boolean().default(true),
}).strict();
export const manifest = z.object({ strategy_id: z.string(), strategy_version: z.string(), verification_run_id: z.string(), diagnostic_files: z.array(z.string()), manifest_hash: z.string(), manifest_path: z.string() }).passthrough();
export const verificationResult = z.object({
  strategy_id: z.string(), strategy_version: z.string(), verification_run_id: z.string(), outcome: z.enum(["VERIFIED", "TECHNICAL_REPAIR_REQUIRED", "MANUAL_REVIEW_REQUIRED", "INSUFFICIENT_DIAGNOSTIC_DATA", "TECHNICAL_FAILURE"]),
  mandatory_checks_passed: z.array(z.string()), mandatory_checks_failed: z.array(z.string()), blocking_issues: z.array(z.string()), repair_eligibility: z.boolean(), recommended_next_state: z.string(),
}).passthrough();
