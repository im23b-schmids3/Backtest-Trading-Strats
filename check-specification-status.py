import json
import subprocess
import sys

payload = {
    "run_id": "f1-RandomOpenTest-0c8b403e4cf2",
    "repository_root": r"C:\Users\sandr\Trading-Bot-Fib",
}

command = [
    sys.executable,
    "-m",
    "research_pipeline",
    "workflow",
    "master-specification-status",
    "--input-json",
    json.dumps(payload),
]

raise SystemExit(subprocess.run(command).returncode)
