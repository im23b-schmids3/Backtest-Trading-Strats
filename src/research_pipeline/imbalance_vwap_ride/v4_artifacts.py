from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .artifacts import ArtifactContext, sha256_file, sha256_value, utc_now, write_bytes_once
from .v4_models import (
    ADAPTER_ID,
    LOCKED_EVIDENCE,
    SELECTION_EVIDENCE,
    SELECTION_METHOD,
    STRATEGY_ID,
    ImbalanceVWAPRideV4Config,
    candidate_registry_hash,
)


def common_claims(*, phase: str | None = None) -> dict[str, Any]:
    return {
        "strategy_id": STRATEGY_ID,
        "adapter_id": ADAPTER_ID,
        "evidence_label": LOCKED_EVIDENCE if phase in {"PHASE_B", "ALPHA"} else SELECTION_EVIDENCE,
        "confirmation_evidence": False,
        "optimization_claimed": False,
        "requires_external_live_or_contract_accurate_confirmation": True,
        "selection_method": SELECTION_METHOD,
        "direction": "LONG_ONLY",
        "raw_aggregate_rows_transmitted": False,
        "secrets_used": False,
        "live_orders_used": False,
    }


def deterministic_study_id(identity: dict[str, Any]) -> str:
    required = {
        "strategy_id",
        "adapter_id",
        "specification_hash",
        "candidate_registry_hash",
        "code_hash",
        "phase_a_dataset_hash",
        "phase_b_dataset_hash",
    }
    missing = sorted(required - set(identity))
    if missing:
        raise ValueError(f"V4 study identity is incomplete: {missing}")
    if identity["strategy_id"] != STRATEGY_ID or identity["adapter_id"] != ADAPTER_ID:
        raise ValueError("V4 study identity has the wrong strategy or adapter")
    return sha256_value(identity)[:24]


def _assert_safe_payload(value: Any, path: str = "payload") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_")
            if normalized in {
                "raw_aggregate_rows",
                "aggregate_trade_rows",
                "raw_trade_rows",
                "api_key",
                "api_secret",
                "secret_key",
                "access_token",
                "private_key",
            } and child is not None and child is not False and child != 0 and child != "" and child != "REDACTED":
                raise ValueError(f"forbidden raw-row or secret payload at {path}.{key}")
            _assert_safe_payload(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_safe_payload(child, f"{path}[{index}]")


@dataclass(frozen=True)
class V4ArtifactContext(ArtifactContext):
    phase: str = "PHASE_A"

    def metadata(self) -> dict[str, str]:
        return {
            **ArtifactContext.metadata(self),
            "strategy_id": STRATEGY_ID,
            "adapter_id": ADAPTER_ID,
            "phase": self.phase,
            "evidence_label": LOCKED_EVIDENCE if self.phase == "PHASE_B" else SELECTION_EVIDENCE,
            "confirmation_evidence": "false",
            "optimization_claimed": "false",
            "external_confirmation_required": "true",
            "selection_method": SELECTION_METHOD,
            "direction": "LONG_ONLY",
        }

    def envelope(self, payload: Any) -> dict[str, Any]:
        _assert_safe_payload(payload)
        base = {**ArtifactContext.metadata(self), **common_claims(phase=self.phase), "phase": self.phase}
        if isinstance(payload, dict):
            return {**base, **payload}
        return {**base, "payload": payload}


class ImmutableV4ArtifactStore:
    """Collision-safe local artifact store for one deterministic V4 study."""

    def __init__(self, artifact_root: str | Path, identity: dict[str, Any]):
        self.identity = dict(identity)
        self.run_id = deterministic_study_id(self.identity)
        self.root = Path(artifact_root).resolve() / STRATEGY_ID / self.run_id
        self.manifest_path = self.root / "study-manifest.json"
        self._timestamp = utc_now()
        if self.manifest_path.exists():
            existing = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            for key, expected in self.identity.items():
                if existing.get(key) != expected:
                    raise ValueError(f"immutable V4 study identity collision for {key}: {self.root}")
            self._timestamp = str(existing["artifact_timestamp"])
        else:
            context = self.context("PHASE_A")
            context.write_json(
                self.manifest_path,
                {
                    **self.identity,
                    "study_run_id": self.run_id,
                    "candidate_registry_hash": candidate_registry_hash(),
                    "immutable": True,
                    "resume_policy": "RESUME_HEALTHY_IDENTICAL_RUN_ONLY",
                },
            )

    def context(self, phase: str) -> V4ArtifactContext:
        if phase not in {"PHASE_A", "PHASE_B", "FINAL", "ALPHA"}:
            raise ValueError(f"unsupported V4 artifact phase: {phase}")
        dataset_key = "phase_b_dataset_hash" if phase in {"PHASE_B", "ALPHA"} else "phase_a_dataset_hash"
        source_key = (
            "phase_b_source_manifest_hash"
            if phase in {"PHASE_B", "ALPHA"}
            else "phase_a_source_manifest_hash"
        )
        return V4ArtifactContext(
            run_id=self.run_id,
            dataset_hash=str(self.identity[dataset_key]),
            source_manifest_hash=str(self.identity.get(source_key, self.identity[dataset_key])),
            specification_hash=str(self.identity["specification_hash"]),
            parameter_hash=str(self.identity["candidate_registry_hash"]),
            code_hash=str(self.identity["code_hash"]),
            evidence_label=LOCKED_EVIDENCE if phase in {"PHASE_B", "ALPHA"} else SELECTION_EVIDENCE,
            timestamp=self._timestamp,
            phase=phase,
        )

    def _path(self, relative_path: str | Path) -> Path:
        candidate = (self.root / relative_path).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise ValueError(f"artifact path escapes V4 study root: {relative_path}")
        return candidate

    def write_json(self, relative_path: str | Path, payload: Any, *, phase: str) -> Path:
        return self.context(phase).write_json(self._path(relative_path), payload)

    def write_parquet(
        self, relative_path: str | Path, rows: list[dict[str, Any]], *, phase: str
    ) -> Path:
        _assert_safe_payload(rows)
        return self.context(phase).write_parquet(self._path(relative_path), rows)

    def freeze_candidate(
        self,
        config: ImbalanceVWAPRideV4Config,
        phase_a_metrics: dict[str, Any],
        rank_trace: dict[str, Any],
        *,
        phase_a_result_hash: str | None = None,
    ) -> dict[str, Any]:
        frozen_config = config.frozen_payload()
        unsigned = {
            "candidate_id": config.candidate_id,
            "frozen_configuration": frozen_config,
            "frozen_configuration_hash": sha256_value(frozen_config),
            "candidate_registry_hash": candidate_registry_hash(),
            "phase_a_result_hash": phase_a_result_hash or sha256_value(phase_a_metrics),
            "phase_a_metrics_hash": sha256_value(phase_a_metrics),
            "rank_trace": rank_trace,
            "selection_method": SELECTION_METHOD,
            "phase_b_execution_count": 0,
        }
        payload = {**unsigned, "frozen_candidate_hash": sha256_value(unsigned)}
        self.write_json("phase_a/frozen_candidate.json", payload, phase="PHASE_A")
        return payload

    def load_frozen_candidate(self) -> dict[str, Any]:
        path = self._path("phase_a/frozen_candidate.json")
        raw = json.loads(path.read_text(encoding="utf-8"))
        keys = (
            "candidate_id",
            "frozen_configuration",
            "frozen_configuration_hash",
            "candidate_registry_hash",
            "phase_a_result_hash",
            "phase_a_metrics_hash",
            "rank_trace",
            "selection_method",
            "phase_b_execution_count",
        )
        unsigned = {key: raw[key] for key in keys}
        if sha256_value(raw["frozen_configuration"]) != raw["frozen_configuration_hash"]:
            raise ValueError("frozen V4 configuration hash mismatch")
        if sha256_value(unsigned) != raw.get("frozen_candidate_hash"):
            raise ValueError("frozen V4 candidate hash mismatch")
        if raw["candidate_registry_hash"] != candidate_registry_hash():
            raise ValueError("frozen V4 registry hash mismatch")
        return raw

    def begin_phase_b(self, frozen_candidate_hash: str) -> dict[str, Any]:
        frozen = self.load_frozen_candidate()
        if frozen["frozen_candidate_hash"] != frozen_candidate_hash:
            raise ValueError("Phase B candidate does not match the immutable Phase A freeze")
        marker_path = self._path("phase_b/execution.json")
        marker = {
            "status": "COMMITTED",
            "execution_count": 1,
            "frozen_candidate_hash": frozen_candidate_hash,
            "phase_a_result_hash": frozen["phase_a_result_hash"],
            "phase_b_dataset_hash": self.identity["phase_b_dataset_hash"],
        }
        if marker_path.exists():
            existing = json.loads(marker_path.read_text(encoding="utf-8"))
            if any(existing.get(key) != value for key, value in marker.items()):
                raise ValueError("immutable V4 Phase B execution collision")
            return existing
        self.write_json("phase_b/execution.json", marker, phase="PHASE_B")
        return json.loads(marker_path.read_text(encoding="utf-8"))

    def validate_health(self) -> dict[str, Any]:
        return validate_v4_artifact_tree(self.root, expected_identity=self.identity)

    def seal_integrity_manifest(self) -> Path:
        hashes = {
            path.relative_to(self.root).as_posix(): sha256_file(path)
            for path in sorted(item for item in self.root.rglob("*") if item.is_file())
            if path.name != "integrity-manifest.json"
        }
        return self.write_json(
            "integrity-manifest.json",
            {"artifact_count": len(hashes), "artifact_hashes": hashes, "sealed": True},
            phase="FINAL",
        )


def validate_v4_artifact_tree(
    root: str | Path,
    *,
    expected_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    target = Path(root).resolve()
    errors: list[str] = []
    manifest_path = target / "study-manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"valid": False, "errors": [f"invalid study manifest: {exc}"], "root": str(target)}
    if manifest.get("strategy_id") != STRATEGY_ID or manifest.get("adapter_id") != ADAPTER_ID:
        errors.append("study manifest strategy/adapter identity mismatch")
    if expected_identity:
        for key, value in expected_identity.items():
            if manifest.get(key) != value:
                errors.append(f"study identity mismatch for {key}")
    temporary = [
        item.relative_to(target).as_posix()
        for item in target.rglob("*")
        if item.is_file() and (item.name.endswith(".tmp") or item.name.endswith(".part"))
    ]
    if temporary:
        errors.append(f"temporary outputs are not admissible: {temporary}")
    for path in target.rglob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            _assert_safe_payload(payload, path.relative_to(target).as_posix())
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{path}: {exc}")
    integrity_path = target / "integrity-manifest.json"
    verified_hashes: dict[str, str] = {}
    if integrity_path.exists():
        integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
        for relative, expected in integrity.get("artifact_hashes", {}).items():
            path = target / relative
            try:
                actual = sha256_file(path)
                if actual != expected:
                    raise ValueError("SHA-256 mismatch")
                verified_hashes[relative] = actual
            except (OSError, ValueError) as exc:
                errors.append(f"{path}: {exc}")
    for path in target.rglob("*.parquet"):
        try:
            pq.ParquetFile(path).metadata
        except (OSError, ValueError) as exc:
            errors.append(f"invalid Parquet artifact {path}: {exc}")
    return {
        "valid": not errors,
        "errors": errors,
        "root": str(target),
        "study_run_id": manifest.get("study_run_id"),
        "temporary_outputs": temporary,
        "verified_hashes": verified_hashes,
        "raw_aggregate_rows_transmitted": False,
    }


def write_immutable_json(path: str | Path, payload: Any) -> Path:
    """Small standalone immutable-artifact tool used by the V4 data layer."""

    _assert_safe_payload(payload)
    content = json.dumps(payload, indent=2, sort_keys=True, default=str).encode("utf-8") + b"\n"
    return write_bytes_once(path, content)


__all__ = [
    "ImmutableV4ArtifactStore",
    "V4ArtifactContext",
    "common_claims",
    "deterministic_study_id",
    "validate_v4_artifact_tree",
    "write_immutable_json",
]
