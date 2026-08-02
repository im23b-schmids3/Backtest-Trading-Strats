import { spawnSync } from "node:child_process";

const repositoryRoot = "C:\\Users\\sandr\\Trading-Bot-Fib";

const input = {
  intake_path: "configs/research_pipeline/random_open_test_intake.yaml",
  repository_root: repositoryRoot,
  registry_path: `${repositoryRoot}\\research_registry\\research_pipeline.sqlite3`,
  dry_run: false,
  implementation_enabled: true,
  research_scenario: "strong-stable",
  prop_scenario: "profitable",
  portfolio_scenario: "complementary",
  prop_product: "Alpha Futures Zero 25K",
  mode: "real_run",
  allow_proxy_data: true,
  prebuilt_spec_path: null,
  max_generation_attempts: 3,
  max_repair_attempts: 2
};

const result = spawnSync(
  process.execPath,
  [
    ".smithers/node_modules/smthrs/src/bin/smithers.js",
    "up",
    "trading-research-master",
    "--detach",
    "--input",
    JSON.stringify(input),
    "--no-post-failure"
  ],
  {
    cwd: repositoryRoot,
    stdio: "inherit"
  }
);

process.exit(result.status ?? 1);
