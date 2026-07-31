import json
import subprocess
import sys

payload = {
    "run_id": "f1-RandomOpenTest-0c8b403e4cf2",
    "repository_root": r"C:\Users\sandr\Trading-Bot-Fib",
}

for action in ("master-specification-status", "master-status"):
    print(f"\n=== {action} ===")
    result = subprocess.run([
        sys.executable,
        "-m",
        "research_pipeline",
        "workflow",
        action,
        "--input-json",
        json.dumps(payload),
    ])
    if result.returncode != 0:
        raise SystemExit(result.returncode)
