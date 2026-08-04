from __future__ import annotations
import hashlib,json
from pathlib import Path
from typing import Any
from .artifacts import sha256_value
class ImmutableV5ArtifactStore:
 def __init__(self,root:str|Path,identity:dict[str,Any]):
  self.identity=identity; self.run_id=sha256_value(identity)[:24]; self.root=Path(root)/identity["strategy_id"]/self.run_id; self.root.mkdir(parents=True,exist_ok=True)
 def write_json(self,relative:str,payload:dict[str,Any],**_:Any):
  # A boolean compliance assertion is safe; the raw aggregate-row payload itself is not.
  if '"raw_aggregate_rows":' in json.dumps(payload,default=str): raise ValueError("forbidden raw-row artifact")
  p=self.root/relative;p.parent.mkdir(parents=True,exist_ok=True); rendered=json.dumps(payload,sort_keys=True,default=str,indent=2)+"\n"
  if p.exists() and p.read_text()!=rendered: raise ValueError("immutable artifact collision")
  p.write_text(rendered); return p
 def seal_integrity_manifest(self):
  rows=[]
  for p in sorted(x for x in self.root.rglob("*") if x.is_file() and x.name!="integrity_manifest.json"): rows.append({"path":p.relative_to(self.root).as_posix(),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()})
  return self.write_json("integrity_manifest.json",{"identity":self.identity,"files":rows,"manifest_hash":sha256_value(rows)})
def validate_v5_artifact_tree(root,expected_identity=None):
 root=Path(root)
 if not (root/"integrity_manifest.json").is_file():
  candidates=list(root.rglob("integrity_manifest.json"))
  if len(candidates)!=1: return {"valid":False,"reason":"expected exactly one integrity manifest","raw_aggregate_rows_transmitted":False}
  root=candidates[0].parent
 data=json.loads((root/"integrity_manifest.json").read_text())
 actual=[]
 for item in sorted(p for p in root.rglob("*") if p.is_file() and p.name!="integrity_manifest.json"):
  actual.append({"path":item.relative_to(root).as_posix(),"sha256":hashlib.sha256(item.read_bytes()).hexdigest()})
 valid=actual==data.get("files") and data.get("manifest_hash")==sha256_value(actual) and (not expected_identity or data.get("identity")==expected_identity)
 return {"valid":valid,"file_count":len(actual),"raw_aggregate_rows_transmitted":False}
