"""Inspector-owned immutable terminal bases for automatic intake."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .errors import InspectorError
from .paths import InspectorPaths
from .records import atomic_create_json, read_json_record


BASIS_SCHEMA = "system-x.inspector-automatic-intake-basis.v1"
TERMINAL_ACTIONS = frozenset(("DISPATCH_FIRST_MODEL", "NOOP_ALREADY_PROCESSED", "REJECT_CANDIDATE"))


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _identity(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _ensure_store(path: Path) -> None:
    try:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.chmod(0o700)
    except OSError as error:
        raise InspectorError("AUTOMATIC_RESULT_STORE_INVALID", "automatic basis store cannot be created") from error
    if path.is_symlink() or not path.is_dir():
        raise InspectorError("AUTOMATIC_RESULT_STORE_INVALID", "automatic basis store is not a directory")


def _basis_value(result: dict[str, Any], basis_class: str) -> dict[str, Any]:
    candidate = result.get("candidate") or {}
    request = result.get("derived_deployment_request")
    artifact_identity = candidate.get("artifact_identity") if isinstance(candidate, dict) else None
    observation_identity = candidate.get("observation_identity") if isinstance(candidate, dict) else None
    if not isinstance(artifact_identity, str) and not isinstance(observation_identity, str):
        raise InspectorError("AUTOMATIC_RESULT_STORE_INVALID", "terminal basis lacks an artifact or observation identity")
    stable_observation_identity = observation_identity if not isinstance(artifact_identity, str) else None
    return {
        "basis_class": basis_class,
        "artifact_identity": artifact_identity,
        "observation_identity": stable_observation_identity,
        "derived_deployment_request": request,
        "registry_snapshot": result.get("registry_snapshot"),
        "source_configuration_identity": result.get("source_configuration_identity"),
        "reason_code": result.get("reason_code") if basis_class == "REJECTED" else None,
    }


def persist_automatic_terminal_basis(paths: InspectorPaths, result: dict[str, Any]) -> dict[str, Any] | None:
    action = result.get("action")
    if action not in TERMINAL_ACTIONS:
        return None
    basis_class = "PROCESSED" if action in {"DISPATCH_FIRST_MODEL", "NOOP_ALREADY_PROCESSED"} else "REJECTED"
    basis_value = _basis_value(result, basis_class)
    basis_identity = _identity(basis_value)
    root = paths.automatic_processed_results if basis_class == "PROCESSED" else paths.automatic_rejected_results
    _ensure_store(root)
    path = root / f"{basis_identity[7:]}.json"
    if path.exists() or path.is_symlink():
        existing = read_json_record(path)
        if existing.get("basis_identity") != basis_identity or existing.get("basis_class") != basis_class:
            raise InspectorError("AUTOMATIC_RESULT_STORE_INVALID", "terminal basis identity collision")
        return {
            "basis_class": basis_class,
            "basis_identity": basis_identity,
            "record_identity": _identity(existing),
            "record_path": str(path),
        }
    record = {
        "schema_version": BASIS_SCHEMA,
        "basis_class": basis_class,
        "basis_identity": basis_identity,
        "artifact_identity": basis_value["artifact_identity"],
        "observation_identity": basis_value["observation_identity"],
        "dispatch_basis": {
            "derived_deployment_request": basis_value["derived_deployment_request"],
            "registry_snapshot": basis_value["registry_snapshot"],
            "source_configuration_identity": basis_value["source_configuration_identity"],
        },
        "derived_policy_identity": _identity(basis_value["derived_deployment_request"]),
        "starting_registry_snapshot_identity": _identity(basis_value["registry_snapshot"]),
        "inspector_result_identity": result.get("result_identity"),
        "deployment_result_reference": result.get("existing_result_reference") or result.get("active_transaction_reference"),
        "terminal_classification": result.get("action"),
        "reason_code": basis_value["reason_code"],
        "created_utc": result.get("created_utc"),
        "completed_utc": result.get("created_utc"),
        "source_contract_identity": result.get("source_configuration_identity"),
    }
    record_identity = atomic_create_json(path, record, mode=0o600)
    return {
        "basis_class": basis_class,
        "basis_identity": basis_identity,
        "record_identity": record_identity,
        "record_path": str(path),
    }

