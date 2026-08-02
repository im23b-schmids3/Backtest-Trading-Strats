from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

from .runners.codex_runner import codex_tool_diagnostic


def _find(name: str) -> str | None:
    return shutil.which(name)


def _smithers(repository_root: Path) -> str | None:
    return _find("smithers") or next((str(repository_root / ".smithers" / "node_modules" / ".bin" / name)
                                      for name in ("smithers.exe", "smithers.cmd", "smithers")
                                      if (repository_root / ".smithers" / "node_modules" / ".bin" / name).is_file()), None)


def diagnose_tools(repository_root: str | Path = ".") -> tuple[dict[str, object], int]:
    root = Path(repository_root).resolve()
    npm_dir = Path(os.environ["APPDATA"]) / "npm" if os.environ.get("APPDATA") else None
    tools = {"python_executable": sys.executable, "git_executable": _find("git"),
             "bun_executable": _find("bun"), "smithers_executable": _smithers(root),
             **codex_tool_diagnostic(), "repository_root": str(root),
             "npm_global_binary_directory": str(npm_dir) if npm_dir else None}
    required = {"python_executable", "git_executable", "bun_executable", "smithers_executable", "codex_executable"}
    visible = [str(Path(item).parent) for item in tools.values() if isinstance(item, str) and Path(item).is_file()]
    if npm_dir:
        visible.append(str(npm_dir))
    tools["required_path_entries_visible"] = {entry: str(Path(tools[entry]).parent) in visible if tools.get(entry) else False for entry in required}
    tools["required_executables_available"] = all(tools["required_path_entries_visible"].values())
    return tools, 0 if tools["required_executables_available"] else 1


def print_diagnostics(repository_root: str | Path = ".") -> int:
    payload, code = diagnose_tools(repository_root)
    print(json.dumps(payload, sort_keys=True))
    return code
