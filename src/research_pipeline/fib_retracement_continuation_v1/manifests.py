from __future__ import annotations
import hashlib, json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any
from .constants import CANDIDATES, STRATEGY_ID
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

_CHRONOLOGY = {
 "development_start_inclusive": "2022-01-01T00:00:00+00:00",
 "development_end_exclusive": "2025-01-01T00:00:00+00:00",
 "holdout_start_inclusive": "2025-01-01T00:00:00+00:00",
 "holdout_end_exclusive": None,
}

def canonical_chronology_hash(data: dict[str, Any]) -> str:
 payload = dict(data); payload.pop("chronologyManifestSha256", None)
 return hashlib.sha256(canonical(payload).encode()).hexdigest()

def _require_chronology_stamp(value: Any, expected: str) -> datetime:
 if value != expected: raise ManifestError("FIB09_V1_CHRONOLOGY_BOUNDARY_MISMATCH")
 return _stamp(value)

def load_chronology_manifest(path_value: str | Path) -> tuple[Path, dict[str, Any]]:
 path=Path(path_value)
 if not path.is_absolute(): raise ManifestError("FIB09_V1_CHRONOLOGY_MANIFEST_PATH_MUST_BE_ABSOLUTE")
 if not path.is_file(): raise ManifestError("FIB09_V1_CHRONOLOGY_MANIFEST_MISSING")
 data=json.loads(path.read_text(encoding="utf-8"))
 if data.get("chronologyManifestSha256") != canonical_chronology_hash(data): raise ManifestError("FIB09_V1_CHRONOLOGY_SELF_HASH_MISMATCH")
 return path, data

def verify_chronology_manifest(path_value: str | Path, *, eth_manifest: str | Path, btc_manifest: str | Path) -> dict[str, Any]:
 """Validate the sealed split without decoding any Parquet values."""
 path,data=load_chronology_manifest(path_value)
 if data.get("chronology_contract_id") != "FIB09_V1_ETH_BTC_UTC_2022_2025_LOCK": raise ManifestError("FIB09_V1_CHRONOLOGY_CONTRACT_ID_MISMATCH")
 if data.get("study_id") != STRATEGY_ID: raise ManifestError("FIB09_V1_CHRONOLOGY_STUDY_ID_MISMATCH")
 if data.get("timezone") != "UTC": raise ManifestError("FIB09_V1_CHRONOLOGY_TIMEZONE_MISMATCH")
 start=_require_chronology_stamp(data.get("development_start_inclusive"), _CHRONOLOGY["development_start_inclusive"])
 end=_require_chronology_stamp(data.get("development_end_exclusive"), _CHRONOLOGY["development_end_exclusive"])
 holdout=_require_chronology_stamp(data.get("holdout_start_inclusive"), _CHRONOLOGY["holdout_start_inclusive"])
 if data.get("holdout_end_exclusive") is not None or end != holdout or start >= end: raise ManifestError("FIB09_V1_CHRONOLOGY_BOUNDARY_MISMATCH")
 semantics=data.get("boundary_semantics")
 if semantics != {"development":"timestamp >= development_start_inclusive and timestamp < development_end_exclusive", "holdout":"timestamp >= holdout_start_inclusive", "no_gap_requirement":True}: raise ManifestError("FIB09_V1_CHRONOLOGY_SEMANTICS_MISMATCH")
 if data.get("candidate_ids") != [row["candidate_id"] for row in CANDIDATES]: raise ManifestError("FIB09_V1_CHRONOLOGY_CANDIDATES_MISMATCH")
 for key, expected_path in (("ETH", eth_manifest), ("BTC", btc_manifest)):
  source_path,source=load_manifest(expected_path)
  asset=data.get("assets",{}).get(key,{})
  try:
   declared_source_path = Path(str(asset.get("source_manifest_path"))).resolve()
  except (OSError, TypeError, ValueError) as error:
   raise ManifestError("FIB09_V1_CHRONOLOGY_SOURCE_MANIFEST_PATH_MISMATCH") from error
  if declared_source_path != source_path.resolve(): raise ManifestError("FIB09_V1_CHRONOLOGY_SOURCE_MANIFEST_PATH_MISMATCH")
  if asset.get("source_manifest_sha256") != source.get("manifestSha256"): raise ManifestError("FIB09_V1_CHRONOLOGY_SOURCE_MANIFEST_HASH_MISMATCH")
  if int(asset.get("total_row_count",-1)) != int(source.get("rowCount",-2)): raise ManifestError("FIB09_V1_CHRONOLOGY_TOTAL_COUNT_MISMATCH")
  if int(asset.get("development_row_count",-1)) + int(asset.get("holdout_row_count",-1)) != int(asset.get("total_row_count",-2)): raise ManifestError("FIB09_V1_CHRONOLOGY_COUNT_MISMATCH")
  if asset.get("development_first_timestamp") != _CHRONOLOGY["development_start_inclusive"]: raise ManifestError("FIB09_V1_CHRONOLOGY_TIMESTAMP_CLAIM_MISMATCH")
  if asset.get("holdout_first_timestamp") != _CHRONOLOGY["holdout_start_inclusive"]: raise ManifestError("FIB09_V1_CHRONOLOGY_TIMESTAMP_CLAIM_MISMATCH")
  if (_stamp(asset.get("development_first_timestamp")) != _stamp(source.get("firstTimestamp")) or
      _stamp(asset.get("holdout_final_timestamp")) != _stamp(source.get("finalTimestamp"))):
   raise ManifestError("FIB09_V1_CHRONOLOGY_TIMESTAMP_CLAIM_MISMATCH")
  for field in ("development_row_count","holdout_row_count","development_final_timestamp","holdout_final_timestamp"):
   if field not in asset: raise ManifestError("FIB09_V1_CHRONOLOGY_CLAIM_MISSING")
 if data.get("no_overlap") is not True or data.get("holdout_locked") is not True or data.get("holdout_strategy_accessed") is not False or data.get("chronology_selected_without_strategy_results") is not True or data.get("historical_matrix_used_for_split_selection") is not False or data.get("source_files_modified") is not False: raise ManifestError("FIB09_V1_CHRONOLOGY_LOCK_STATE_INVALID")
 return {"chronology_manifest_path":str(path),"validated":True,"rows_read":False,"development_start":start,"development_end":end,"holdout_start":holdout,"assets":data["assets"]}
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
