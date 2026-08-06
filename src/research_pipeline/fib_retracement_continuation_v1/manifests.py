from __future__ import annotations
import hashlib, json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any
from .ids import canonical

class ManifestError(ValueError): pass
def sha256_file(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def canonical_manifest_hash(data: dict[str, Any]) -> str:
 payload = dict(data); payload.pop("manifestSha256", None)
 return hashlib.sha256(canonical(payload).encode()).hexdigest()
def _stamp(value: Any) -> datetime:
 stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
 if stamp.tzinfo is None or stamp.utcoffset() != timedelta(): raise ManifestError("MANIFEST_TIMESTAMP_NOT_UTC")
 return stamp.astimezone(timezone.utc)
def require_development_mode(mode: str) -> None:
 if mode == "holdout": raise ManifestError("LOCKED_HOLDOUT_NOT_AUTHORIZED")
 if mode != "development": raise ManifestError("FIB09_V1_DEVELOPMENT_MODE_REQUIRED")
def load_manifest(path_value: str | Path) -> tuple[Path, dict[str, Any]]:
 path=Path(path_value)
 if not path.is_absolute(): raise ManifestError("FIB09_V1_MANIFEST_PATH_MUST_BE_ABSOLUTE")
 if not path.is_file(): raise ManifestError("FIB09_V1_MANIFEST_MISSING")
 data=json.loads(path.read_text(encoding="utf-8"))
 if data.get("manifestSha256") != canonical_manifest_hash(data): raise ManifestError("FIB09_V1_MANIFEST_SELF_HASH_MISMATCH")
 return path, data
def verify_manifest(path_value: str | Path, *, mode: str="development", rows: list[dict[str,Any]]|None=None, schema_hash: str|None=None) -> dict[str,Any]:
 """Verify a declared manifest. Rows are caller supplied (synthetic tests or future loader)."""
 require_development_mode(mode); path, data = load_manifest(path_value)
 for key, value in (("provenanceClassification","USER_SUPPLIED_PROVENANCE_EVIDENCE"),("sourceExchange","UNKNOWN"),("instrumentType","UNKNOWN"),("historicalV6Identity","NOT_CLAIMED")):
  if data.get(key)!=value: raise ManifestError("FIB09_V1_PROVENANCE_CLASSIFICATION_MISMATCH")
 source=Path(data.get("absoluteSourcePath", ""))
 if not source.is_absolute(): raise ManifestError("FIB09_V1_SOURCE_PATH_MUST_BE_ABSOLUTE")
 if not source.is_file(): raise ManifestError("FIB09_V1_SOURCE_MISSING")
 if sha256_file(source)!=data.get("sourceFileSha256"): raise ManifestError("FIB09_V1_SOURCE_HASH_MISMATCH")
 if source.stat().st_size != int(data.get("fileSizeBytes", -1)): raise ManifestError("FIB09_V1_SOURCE_SIZE_MISMATCH")
 if schema_hash is not None and schema_hash != data.get("schemaSha256"): raise ManifestError("FIB09_V1_SCHEMA_HASH_MISMATCH")
 if rows is None: return {"manifest_path":str(path),"validated":True,"rows_read":False,"provenanceClassification":"USER_SUPPLIED_PROVENANCE_EVIDENCE","sourceExchange":"UNKNOWN","instrumentType":"UNKNOWN","historicalV6Identity":"NOT_CLAIMED"}
 if len(rows)!=int(data.get("rowCount", -1)): raise ManifestError("FIB09_V1_ROW_COUNT_MISMATCH")
 stamps=[]
 for row in rows:
  for field in ("open","high","low","close","volume","timestamp"):
   if row.get(field) is None: raise ManifestError("FIB09_V1_NULL_OHLCV")
  if row["low"]>min(row["open"],row["close"]) or row["high"]<max(row["open"],row["close"]) or row["low"]>row["high"]: raise ManifestError("FIB09_V1_INVALID_OHLC")
  stamps.append(_stamp(row["timestamp"]))
 if not stamps or stamps[0]!=_stamp(data["firstTimestamp"]) or stamps[-1]!=_stamp(data["finalTimestamp"]): raise ManifestError("FIB09_V1_TIMESTAMP_COVERAGE_MISMATCH")
 seconds = 14400 if data.get("expectedInterval")=="PT4H" else 86400 if data.get("expectedInterval")=="P1D" else None
 if seconds is None: raise ManifestError("FIB09_V1_CADENCE_INVALID")
 for a,b in zip(stamps, stamps[1:]):
  if b<=a: raise ManifestError("FIB09_V1_NON_INCREASING_OR_DUPLICATE_TIMESTAMP")
  if (b-a).total_seconds()!=seconds: raise ManifestError("FIB09_V1_MISSING_INTERVAL")
 return {"manifest_path":str(path),"validated":True,"rows_read":True,"provenanceClassification":"USER_SUPPLIED_PROVENANCE_EVIDENCE","sourceExchange":"UNKNOWN","instrumentType":"UNKNOWN","historicalV6Identity":"NOT_CLAIMED"}
