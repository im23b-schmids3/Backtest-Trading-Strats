"""Start the local Phase C Smithers workflow against the synthetic fixture."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from research_pipeline.research.fixtures import prepare_phase_c_fixture


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy-id", default="phase-c-smithers-dry-run")
    parser.add_argument("--scenario", default="strong-stable")
    parser.add_argument("--registry-path", default="research_registry/phase-c-smithers-dry-run.sqlite3")
    parser.add_argument("--run-id", default="phase-c-smithers-dry-run")
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    registry = (root / args.registry_path).resolve()
    prepare_phase_c_fixture(registry, root, args.strategy_id, args.scenario)
    executable = root / ".smithers" / "node_modules" / ".bin" / ("smithers.exe" if shutil.which("cmd.exe") else "smithers")
    if not executable.is_file():
        executable = Path(shutil.which("smithers") or "smithers")
    payload = {"strategy_id": args.strategy_id, "repository_root": str(root), "registry_path": str(registry), "scenario": args.scenario, "dry_run": True, "research_run_id": f"{args.run_id}-python"}
    command = [str(executable), "up", str(root / ".smithers" / "workflows" / "trading-research-phase-c.tsx"), "--run-id", args.run_id, "--input", json.dumps(payload), "--no-post-failure"]
    if args.detach:
        command.insert(4, "--detach")
    result = subprocess.run(command, cwd=root, text=True)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
