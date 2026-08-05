from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any
from ..imbalance_vwap_ride.artifacts import sha256_value

class ImmutableLSMRArtifactStore:
    def __init__(self, root: str | Path, identity: dict[str, Any]):
        self.identity=identity; self.run_id=sha256_value(identity)[:24]; self.root=Path(root)/identity["strategy_id"]/self.run_id
        if self.root.exists(): raise FileExistsError(f"immutable LSMR output directory already exists: {self.root}")
        self.root.mkdir(parents=True)
    def write_json(self, relative: str, payload: Any) -> Path:
        if "raw_aggregate_rows" in json.dumps(payload, default=str): raise ValueError("forbidden raw market-data artifact")
        path=self.root/relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(payload, sort_keys=True, indent=2, default=str)+"\n"); return path
    def seal(self) -> Path:
        rows=[{"path":p.relative_to(self.root).as_posix(),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(self.root.rglob("*")) if p.is_file() and p.name!="integrity-manifest.json"]
        return self.write_json("integrity-manifest.json", {"identity":self.identity,"files":rows,"manifest_hash":sha256_value(rows)})

def validate_artifact_tree(root: str | Path) -> dict[str, Any]:
    root=Path(root); manifest=root/"integrity-manifest.json"
    if not manifest.exists():
        candidates=list(root.rglob("integrity-manifest.json"))
        if len(candidates)!=1: return {"valid":False,"reason":"expected exactly one integrity manifest"}
        manifest=candidates[0]; root=manifest.parent
    data=json.loads(manifest.read_text()); rows=[{"path":p.relative_to(root).as_posix(),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(root.rglob("*")) if p.is_file() and p.name!="integrity-manifest.json"]
    return {"valid":rows==data.get("files") and data.get("manifest_hash")==sha256_value(rows),"file_count":len(rows)}
