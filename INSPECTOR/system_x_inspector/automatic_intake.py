"""Zero-input, copy-stable GGUF intake reconciliation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .constants import QUALIFICATION_PROFILES, SCHEMA_IDENTITIES
from .connection_receipt import bootstrap_current_receipt
from .content_identity import ArtifactIdentity, identify_regular_file
from .deployment import (
    DEPLOYMENT_INPUT_IMPLEMENTATION_EPOCH,
    CurrentSourceDeploymentAdapter,
    _clear_exact_stale_deployment_lock,
    _converged_install_first_retry,
    _find_recoverable,
    _find_retryable_failed_clean,
    _input_identity,
    deploy_transaction,
    validate_deployment_result,
)
from .errors import InspectorError
from .intake import RESERVED_NAMES, _visible_name
from .locking import inspect_active_lock
from .paths import InspectorPaths
from .records import read_json_record
from .results import utc_now


AUTOMATIC_SOURCE_IMPLEMENTATION_EPOCH = "automatic-rejected-basis-source-epoch-v12"
SERVICE_CONTINUITY_RECOVERY_EPOCH = "service-continuity-recovery-epoch-v1"


AUTOMATIC_ACTIONS = frozenset(
    {
        "NOOP_WAITING",
        "NOOP_COPY_IN_PROGRESS",
        "NOOP_MULTIPLE_CANDIDATES",
        "NOOP_ACTIVE_TRANSACTION",
        "NOOP_ALREADY_PROCESSED",
        "NOOP_READY_MODEL_PRESENT",
        "NOOP_REGISTRY_CONTRADICTORY",
        "NOOP_OWNERSHIP_UNCERTAIN",
        "REJECT_CANDIDATE",
        "DISPATCH_FIRST_MODEL",
    }
)
AUTOMATIC_REASON_CODES = frozenset(
    {
        "AUTOMATIC_NO_VISIBLE_CANDIDATE",
        "AUTOMATIC_MULTIPLE_CANDIDATES",
        "AUTOMATIC_COPY_IN_PROGRESS",
        "AUTOMATIC_READY_MODEL_PRESENT",
        "AUTOMATIC_REGISTRY_CONTRADICTORY",
        "AUTOMATIC_ACTIVE_TRANSACTION",
        "AUTOMATIC_OWNERSHIP_UNCERTAIN",
        "AUTOMATIC_ALREADY_PROCESSED",
        "AUTOMATIC_DISPATCH_ACCEPTED",
        "AUTOMATIC_CANDIDATE_REJECTED",
        "AUTOMATIC_DIRECTORY_REJECTED",
        "AUTOMATIC_SYMLINK_REJECTED",
        "AUTOMATIC_HARDLINK_REJECTED",
        "AUTOMATIC_SPECIAL_FILE_REJECTED",
        "AUTOMATIC_UNREADABLE_CANDIDATE",
        "AUTOMATIC_PATH_OUTSIDE_ROOT",
        "AUTOMATIC_MOUNT_SUBSTITUTION",
        "AUTOMATIC_POLICY_INVALID",
        "AUTOMATIC_DEPLOYMENT_SOURCE_CHANGED",
        "AUTOMATIC_DISPATCH_FAILED_CLEAN",
        "AUTOMATIC_INTAKE_UNAVAILABLE",
        "AUTOMATIC_RESULT_STORE_INVALID",
    }
)
RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "operation",
        "reconciliation_id",
        "created_utc",
        "action",
        "reason_code",
        "intake_root_identity",
        "visible_candidate_count",
        "candidate",
        "registry_snapshot",
        "derived_deployment_request",
        "active_transaction_reference",
        "existing_result_reference",
        "source_configuration_identity",
        "result_identity",
    }
)
CANDIDATE_FIELDS = frozenset(
    {
        "relative_name",
        "observation_identity",
        "artifact_identity",
        "byte_count",
        "metadata",
    }
)
METADATA_FIELDS = frozenset(
    {
        "device",
        "inode",
        "file_type",
        "mode",
        "uid",
        "gid",
        "nlink",
        "size",
        "mtime_ns",
        "ctime_ns",
    }
)
REQUEST_FIELDS = frozenset(
    {
        "deployment_mode",
        "required_capability_profile",
        "retirement_policy",
        "retirement_action",
    }
)
REGISTRY_FIELDS = frozenset(
    {
        "desired_state",
        "model_service_state",
        "ready_model_count",
        "model_rows",
        "active_managed_locations",
        "default_alias",
        "default_target",
        "warm_model_id",
        "active_transaction_reference",
        "recovery_state",
        "capability_binding_identity",
        "operating_profile_identity",
        "registry_generation",
    }
)
SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
RECONCILIATION_PATTERN = re.compile(r"reconcile-[0-9a-f]{32}\Z")


@dataclass(frozen=True)
class AutomaticIntakePolicy:
    stability_seconds: float = 5.0

    def __post_init__(self) -> None:
        value = self.stability_seconds
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise InspectorError(
                "AUTOMATIC_POLICY_INVALID",
                "stability interval must be a finite number",
            )
        numeric = float(value)
        if not math.isfinite(numeric) or not 1.0 <= numeric <= 300.0:
            raise InspectorError(
                "AUTOMATIC_POLICY_INVALID",
                "stability interval must be between 1 and 300 seconds",
            )
        object.__setattr__(self, "stability_seconds", numeric)

    def as_dict(self) -> dict[str, float]:
        return {"stability_seconds": self.stability_seconds}

    @property
    def identity(self) -> str:
        return _identity(self.as_dict())


def _identity(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _is_identity(value: object) -> bool:
    return isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None


def _policy(value: AutomaticIntakePolicy | dict[str, Any] | None) -> AutomaticIntakePolicy:
    if value is None:
        return AutomaticIntakePolicy()
    if isinstance(value, AutomaticIntakePolicy):
        return value
    if isinstance(value, dict) and set(value) == {"stability_seconds"}:
        return AutomaticIntakePolicy(value["stability_seconds"])
    raise InspectorError(
        "AUTOMATIC_POLICY_INVALID", "automatic intake policy is not closed"
    )


def _file_type(mode: int) -> str:
    if stat.S_ISREG(mode):
        return "regular_file"
    if stat.S_ISDIR(mode):
        return "regular_directory"
    if stat.S_ISLNK(mode):
        return "symlink"
    return "special"


def _root_identity(paths: InspectorPaths) -> str:
    try:
        details = paths.intake_root.lstat()
    except FileNotFoundError as error:
        raise InspectorError(
            "AUTOMATIC_INTAKE_UNAVAILABLE", "automatic intake root is absent"
        ) from error
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
        raise InspectorError(
            "AUTOMATIC_INTAKE_UNAVAILABLE",
            "automatic intake root is not a regular directory",
        )
    resolved = paths.intake_root.resolve(strict=True)
    if resolved != paths.intake_root or resolved.parent != paths.inspector_root:
        raise InspectorError(
            "AUTOMATIC_PATH_OUTSIDE_ROOT",
            "automatic intake root is not the direct Inspector child",
        )
    return _identity(
        {
            "contract": "INSPECTOR/MODEL-TEST",
            "device": details.st_dev,
            "inode": details.st_ino,
            "mode": stat.S_IMODE(details.st_mode),
            "uid": details.st_uid,
            "gid": details.st_gid,
        }
    )


def _snapshot(path: Path, root: Path) -> dict[str, Any]:
    try:
        details = path.lstat()
    except FileNotFoundError as error:
        raise InspectorError(
            "AUTOMATIC_COPY_IN_PROGRESS",
            "candidate disappeared during stability observation",
        ) from error
    file_type = _file_type(details.st_mode)
    if file_type == "symlink":
        raise InspectorError(
            "AUTOMATIC_SYMLINK_REJECTED", "automatic symlink candidate rejected"
        )
    if not path.resolve(strict=True).is_relative_to(root.resolve(strict=True)):
        raise InspectorError(
            "AUTOMATIC_PATH_OUTSIDE_ROOT", "candidate escaped the intake root"
        )
    return {
        "device": details.st_dev,
        "inode": details.st_ino,
        "file_type": file_type,
        "mode": stat.S_IMODE(details.st_mode),
        "uid": details.st_uid,
        "gid": details.st_gid,
        "nlink": details.st_nlink,
        "size": details.st_size,
        "mtime_ns": details.st_mtime_ns,
        "ctime_ns": details.st_ctime_ns,
    }


def _candidate_path(paths: InspectorPaths, candidate: dict[str, Any]) -> Path:
    name = candidate.get("relative_name")
    if (
        not isinstance(name, str)
        or not name
        or Path(name).name != name
        or not _visible_name(name)
    ):
        raise InspectorError(
            "AUTOMATIC_PATH_OUTSIDE_ROOT", "candidate name is not a direct child"
        )
    path = paths.intake_root / name
    if path.parent != paths.intake_root:
        raise InspectorError(
            "AUTOMATIC_PATH_OUTSIDE_ROOT", "candidate path escaped the intake root"
        )
    return path


def _candidate_from_name(
    paths: InspectorPaths, root_identity: str, name: str
) -> dict[str, Any]:
    path = paths.intake_root / name
    try:
        details = path.lstat()
    except FileNotFoundError as error:
        raise InspectorError(
            "AUTOMATIC_COPY_IN_PROGRESS", "candidate disappeared during discovery"
        ) from error
    file_type = _file_type(details.st_mode)
    if file_type == "symlink":
        raise InspectorError(
            "AUTOMATIC_SYMLINK_REJECTED", "automatic symlink candidate rejected"
        )
    if file_type == "regular_directory":
        raise InspectorError(
            "AUTOMATIC_DIRECTORY_REJECTED", "automatic directory candidate rejected"
        )
    if file_type == "special":
        raise InspectorError(
            "AUTOMATIC_SPECIAL_FILE_REJECTED", "automatic special file rejected"
        )
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(paths.intake_root.resolve(strict=True)):
        raise InspectorError(
            "AUTOMATIC_PATH_OUTSIDE_ROOT", "candidate escaped the intake root"
        )
    if os.path.ismount(path):
        raise InspectorError(
            "AUTOMATIC_MOUNT_SUBSTITUTION", "candidate mount substitution rejected"
        )
    if details.st_nlink != 1:
        raise InspectorError(
            "AUTOMATIC_HARDLINK_REJECTED", "candidate must have one hard link"
        )
    if not os.access(path, os.R_OK, effective_ids=True):
        raise InspectorError(
            "AUTOMATIC_UNREADABLE_CANDIDATE", "candidate is not readable"
        )
    metadata = _snapshot(path, paths.intake_root)
    observation_identity = _identity(
        {
            "intake_root_identity": root_identity,
            "relative_name": name,
            "metadata": metadata,
        }
    )
    return {
        "relative_name": name,
        "observation_identity": observation_identity,
        "artifact_identity": None,
        "byte_count": details.st_size,
        "metadata": metadata,
    }


def discover_automatic_candidate(paths: InspectorPaths, policy: AutomaticIntakePolicy | dict[str, Any] | None = None) -> dict[str, Any]:
    del policy
    root_identity = _root_identity(paths)
    try:
        with os.scandir(paths.intake_root) as entries:
            names = sorted(
                entry.name
                for entry in entries
                if _visible_name(entry.name) and entry.name not in RESERVED_NAMES
            )
    except OSError as error:
        raise InspectorError(
            "AUTOMATIC_INTAKE_UNAVAILABLE", "automatic intake directory could not be read"
        ) from error
    if not names:
        return {
            "status": "empty",
            "reason_code": "AUTOMATIC_NO_VISIBLE_CANDIDATE",
            "visible_candidate_count": 0,
            "candidate": None,
            "intake_root_identity": root_identity,
        }
    if len(names) > 1:
        return {
            "status": "multiple",
            "reason_code": "AUTOMATIC_MULTIPLE_CANDIDATES",
            "visible_candidate_count": len(names),
            "candidate": None,
            "intake_root_identity": root_identity,
        }
    try:
        candidate = _candidate_from_name(paths, root_identity, names[0])
    except InspectorError as error:
        return {
            "status": "rejected",
            "reason_code": error.reason_code,
            "visible_candidate_count": 1,
            "candidate": {
                "relative_name": names[0],
                "observation_identity": None,
                "artifact_identity": None,
                "byte_count": None,
                "metadata": None,
            },
            "intake_root_identity": root_identity,
        }
    return {
        "status": "eligible",
        "reason_code": "OK",
        "visible_candidate_count": 1,
        "candidate": candidate,
        "intake_root_identity": root_identity,
    }


def _notify(
    observer: Callable[..., None] | None,
    stage: str,
    path: Path,
    snapshot: dict[str, Any],
) -> None:
    if observer is None:
        return
    try:
        observer(stage, snapshot)
    except TypeError:
        observer(stage, path, snapshot)


def observe_candidate_stability(
    paths: InspectorPaths,
    candidate: dict[str, Any],
    policy: AutomaticIntakePolicy | dict[str, Any] | None = None,
    observer: Callable[..., None] | None = None,
    waiter: Callable[[float], None] | None = None,
    identifier: Callable[[Path], ArtifactIdentity] = identify_regular_file,
) -> dict[str, Any]:
    resolved_policy = _policy(policy)
    path = _candidate_path(paths, candidate)
    try:
        first = _snapshot(path, paths.intake_root)
        if first["file_type"] != "regular_file":
            raise InspectorError(
                "AUTOMATIC_CANDIDATE_REJECTED", "candidate is not a regular file"
            )
        _notify(observer, "A", path, first)
        (waiter or time.sleep)(resolved_policy.stability_seconds)
        second = _snapshot(path, paths.intake_root)
        _notify(observer, "B", path, second)
        if first != second:
            raise InspectorError(
                "AUTOMATIC_COPY_IN_PROGRESS", "candidate changed between snapshots A and B"
            )
        try:
            artifact = identifier(path)
        except InspectorError as error:
            if error.reason_code in {
                "ARTIFACT_CHANGED_DURING_INSPECTION",
                "ARTIFACT_READ_FAILED",
            }:
                raise InspectorError(
                    "AUTOMATIC_COPY_IN_PROGRESS",
                    "candidate changed or could not be read at the content boundary",
                ) from error
            raise
        third = _snapshot(path, paths.intake_root)
        _notify(observer, "C", path, third)
        if second != third:
            raise InspectorError(
                "AUTOMATIC_COPY_IN_PROGRESS", "candidate changed between content boundary and snapshot C"
            )
    except FileNotFoundError as error:
        raise InspectorError(
            "AUTOMATIC_COPY_IN_PROGRESS", "candidate disappeared during stability observation"
        ) from error
    snapshots = {"A": first, "B": second, "C": third}
    observation_identity = _identity(
        {
            "intake_root_identity": _root_identity(paths),
            "relative_name": candidate["relative_name"],
            "snapshots": snapshots,
        }
    )
    return {
        "observation_identity": observation_identity,
        "snapshots": snapshots,
        "artifact": artifact.as_dict(),
        "candidate": {
            "relative_name": candidate["relative_name"],
            "observation_identity": observation_identity,
            "artifact_identity": artifact.identity,
            "byte_count": artifact.byte_count,
            "metadata": third,
        },
    }


def _count(value: object, *, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        raise InspectorError(
            "AUTOMATIC_REGISTRY_CONTRADICTORY", "registry count is invalid"
        )
    if isinstance(value, (list, tuple)):
        return len(value)
    if isinstance(value, int) and value >= 0:
        return value
    raise InspectorError(
        "AUTOMATIC_REGISTRY_CONTRADICTORY", "registry count is invalid"
    )


def _optional_text(value: object) -> str | None:
    if value is None or value == "":
        return None
    return value if isinstance(value, str) else "__invalid__"


def _registry_snapshot(prestate: dict[str, Any]) -> dict[str, Any]:
    ready_rows = _count(prestate.get("ready_rows"), default=0)
    ready_count = _count(prestate.get("ready_model_count"), default=ready_rows)
    model_rows = _count(prestate.get("model_rows"), default=ready_count)
    active_locations = _count(
        prestate.get("active_managed_locations"),
        default=1 if _optional_text(prestate.get("managed_location_identity")) else 0,
    )
    active_ref = prestate.get("active_transaction_reference")
    if active_ref is not None and not isinstance(active_ref, str):
        active_ref = None
    generation = prestate.get("registry_generation")
    if not isinstance(generation, int) or isinstance(generation, bool):
        generation = None
    return {
        "desired_state": _optional_text(prestate.get("desired_state")),
        "model_service_state": _optional_text(prestate.get("model_service_state")),
        "ready_model_count": ready_count,
        "model_rows": model_rows,
        "active_managed_locations": active_locations,
        "default_alias": _optional_text(prestate.get("default_alias")),
        "default_target": _optional_text(prestate.get("default_target")),
        "warm_model_id": _optional_text(prestate.get("warm_model_id")),
        "active_transaction_reference": active_ref,
        "recovery_state": _optional_text(prestate.get("recovery_state")),
        "capability_binding_identity": _optional_text(
            prestate.get("capability_binding_identity")
        ),
        "operating_profile_identity": _optional_text(
            prestate.get("operating_profile_identity")
        ),
        "registry_generation": generation,
    }


def derive_first_model_policy(
    prestate: dict[str, Any],
    *,
    allow_converged_install_first_retry: bool = False,
) -> dict[str, Any]:
    if not isinstance(prestate, dict):
        raise InspectorError(
            "AUTOMATIC_REGISTRY_CONTRADICTORY", "deployment prestate is not an object"
        )
    registry = _registry_snapshot(prestate)
    if registry["active_transaction_reference"]:
        raise InspectorError(
            "AUTOMATIC_ACTIVE_TRANSACTION", "a deployment transaction is active"
        )
    recovery = registry["recovery_state"]
    if recovery not in {None, "IDLE"}:
        raise InspectorError(
            "AUTOMATIC_ACTIVE_TRANSACTION", "deployment recovery is active"
        )
    if (
        not allow_converged_install_first_retry
        and (
            registry["ready_model_count"] > 0
            or registry["model_rows"] > 0
            or registry["active_managed_locations"] > 0
            or registry["default_alias"] is not None
            or registry["default_target"] is not None
            or registry["warm_model_id"] is not None
            or _optional_text(prestate.get("resolved_immutable_model_id")) is not None
            or registry["model_service_state"] == "READY"
        )
    ):
        raise InspectorError(
            "AUTOMATIC_READY_MODEL_PRESENT", "automatic replacement is fenced"
        )
    if registry["desired_state"] != "RUNNING":
        raise InspectorError(
            "AUTOMATIC_REGISTRY_CONTRADICTORY", "desired service state does not permit reconciliation"
        )
    if (
        not allow_converged_install_first_retry
        and (
            registry["ready_model_count"] != 0
            or registry["model_rows"] != 0
        )
    ):
        raise InspectorError(
            "AUTOMATIC_REGISTRY_CONTRADICTORY", "empty registry counts disagree"
        )
    binding = registry["capability_binding_identity"]
    profile = registry["operating_profile_identity"]
    if not _is_identity(binding) or not _is_identity(profile):
        raise InspectorError(
            "AUTOMATIC_REGISTRY_CONTRADICTORY", "current capability or profile identity is invalid"
        )
    if (
        not allow_converged_install_first_retry
        and registry["active_managed_locations"] != 0
    ):
        raise InspectorError(
            "AUTOMATIC_REGISTRY_CONTRADICTORY", "managed locations contradict an empty registry"
        )
    return {
        "deployment_mode": "install-first",
        "required_capability_profile": "CORE_CHAT",
        "retirement_policy": "retain-incumbent",
        "retirement_action": "none",
        "registry_snapshot": registry,
        "prestate_identity": _identity(registry),
    }


def build_automatic_dispatch_basis(
    candidate: dict[str, Any],
    artifact: ArtifactIdentity | dict[str, Any],
    policy: AutomaticIntakePolicy | dict[str, Any] | None,
    prestate: dict[str, Any],
    *,
    allow_converged_install_first_retry: bool = False,
) -> dict[str, Any]:
    resolved_policy = _policy(policy)
    derived = derive_first_model_policy(
        prestate,
        allow_converged_install_first_retry=allow_converged_install_first_retry,
    )
    artifact_value = artifact.as_dict() if isinstance(artifact, ArtifactIdentity) else artifact
    if not isinstance(artifact_value, dict) or not _is_identity(artifact_value.get("identity")):
        raise InspectorError(
            "AUTOMATIC_CANDIDATE_REJECTED", "artifact identity is invalid"
        )
    if not isinstance(candidate, dict) or not isinstance(candidate.get("relative_name"), str):
        raise InspectorError(
            "AUTOMATIC_CANDIDATE_REJECTED", "candidate projection is invalid"
        )
    request = {
        key: derived[key]
        for key in (
            "deployment_mode",
            "required_capability_profile",
            "retirement_policy",
        )
    }
    basis_value = {
        "candidate_artifact_identity": artifact_value["identity"],
        "deployment_mode": request["deployment_mode"],
        "required_capability_profile": request["required_capability_profile"],
        "retirement_policy": request["retirement_policy"],
        "current_capability_binding_identity": prestate["capability_binding_identity"],
        "current_operating_profile_identity": prestate["operating_profile_identity"],
    }
    return {
        "dispatch_basis_identity": _identity(basis_value),
        "artifact_identity": artifact_value["identity"],
        "artifact_byte_count": artifact_value.get("byte_count"),
        "policy_identity": _identity(
            {
                "deployment_mode": request["deployment_mode"],
                "required_capability_profile": request["required_capability_profile"],
                "retirement_policy": request["retirement_policy"],
                "retirement_action": derived["retirement_action"],
                "stability_seconds": resolved_policy.stability_seconds,
            }
        ),
        "registry_snapshot_identity": derived["prestate_identity"],
        "derived_deployment_request": {
            **request,
            "retirement_action": derived["retirement_action"],
        },
    }


def _validate_metadata(value: object, *, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if not isinstance(value, dict) or set(value) != METADATA_FIELDS:
        raise InspectorError(
            "AUTOMATIC_RESULT_STORE_INVALID", "automatic candidate metadata is not closed"
        )
    if not isinstance(value["file_type"], str) or value["file_type"] != "regular_file":
        raise InspectorError(
            "AUTOMATIC_RESULT_STORE_INVALID", "automatic candidate metadata is invalid"
        )
    for key in METADATA_FIELDS - {"file_type"}:
        if not isinstance(value[key], int) or isinstance(value[key], bool):
            raise InspectorError(
                "AUTOMATIC_RESULT_STORE_INVALID", "automatic candidate metadata is invalid"
            )
    if value["nlink"] < 1 or value["size"] < 0 or value["file_type"] != "regular_file":
        raise InspectorError(
            "AUTOMATIC_RESULT_STORE_INVALID", "automatic candidate metadata is unsafe"
        )


def _reject_private(value: object) -> None:
    forbidden = {
        "api_key",
        "credential",
        "credential_verifier",
        "pepper",
        "endpoint",
        "port",
        "process",
        "pid",
        "environment",
        "private_router",
        "model_child",
        "socket",
    }
    if isinstance(value, dict):
        for key, child in value.items():
            if key in forbidden:
                raise InspectorError(
                    "AUTOMATIC_RESULT_STORE_INVALID", "automatic result contains private data"
                )
            _reject_private(child)
    elif isinstance(value, list):
        for child in value:
            _reject_private(child)


def _validate_reference(value: object, *, existing: bool) -> None:
    if value is None:
        return
    allowed = (
        {"transaction_id", "deployment_id"}
        if not existing
        else {"transaction_id", "deployment_id", "result_identity", "result_class"}
    )
    if not isinstance(value, dict) or set(value) != allowed:
        raise InspectorError(
            "AUTOMATIC_RESULT_STORE_INVALID", "automatic result reference is not closed"
        )
    if not isinstance(value["transaction_id"], str) or not value["transaction_id"]:
        raise InspectorError(
            "AUTOMATIC_RESULT_STORE_INVALID", "automatic transaction reference is invalid"
        )
    if not isinstance(value["deployment_id"], str) or not value["deployment_id"]:
        raise InspectorError(
            "AUTOMATIC_RESULT_STORE_INVALID", "automatic deployment reference is invalid"
        )
    if existing:
        if not _is_identity(value["result_identity"]) or not isinstance(value["result_class"], str):
            raise InspectorError(
                "AUTOMATIC_RESULT_STORE_INVALID", "automatic existing result reference is invalid"
            )


def validate_automatic_result(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != RESULT_FIELDS:
        raise InspectorError(
            "AUTOMATIC_RESULT_STORE_INVALID", "automatic result fields are not closed"
        )
    _reject_private(value)
    if value["schema_version"] != SCHEMA_IDENTITIES["automatic_intake_result"]:
        raise InspectorError(
            "AUTOMATIC_RESULT_STORE_INVALID", "automatic result schema is invalid"
        )
    if value["operation"] != "reconcile-intake" or value["action"] not in AUTOMATIC_ACTIONS:
        raise InspectorError(
            "AUTOMATIC_RESULT_STORE_INVALID", "automatic result operation or action is invalid"
        )
    if value["reason_code"] not in AUTOMATIC_REASON_CODES:
        raise InspectorError(
            "AUTOMATIC_RESULT_STORE_INVALID", "automatic result reason code is unknown"
        )
    if not isinstance(value["reconciliation_id"], str) or RECONCILIATION_PATTERN.fullmatch(value["reconciliation_id"]) is None:
        raise InspectorError(
            "AUTOMATIC_RESULT_STORE_INVALID", "automatic reconciliation identity is invalid"
        )
    if not isinstance(value["created_utc"], str) or not value["created_utc"]:
        raise InspectorError(
            "AUTOMATIC_RESULT_STORE_INVALID", "automatic result timestamp is invalid"
        )
    if not _is_identity(value["intake_root_identity"]) or not _is_identity(value["source_configuration_identity"]):
        raise InspectorError(
            "AUTOMATIC_RESULT_STORE_INVALID", "automatic result configuration identity is invalid"
        )
    if not isinstance(value["visible_candidate_count"], int) or isinstance(value["visible_candidate_count"], bool) or value["visible_candidate_count"] < 0:
        raise InspectorError(
            "AUTOMATIC_RESULT_STORE_INVALID", "automatic visible candidate count is invalid"
        )
    candidate = value["candidate"]
    if candidate is not None:
        if not isinstance(candidate, dict) or set(candidate) != CANDIDATE_FIELDS:
            raise InspectorError(
                "AUTOMATIC_RESULT_STORE_INVALID", "automatic candidate is not closed"
            )
        if not isinstance(candidate["relative_name"], str) or Path(candidate["relative_name"]).name != candidate["relative_name"] or candidate["relative_name"].startswith("."):
            raise InspectorError(
                "AUTOMATIC_RESULT_STORE_INVALID", "automatic candidate name is invalid"
            )
        if candidate["observation_identity"] is not None and not _is_identity(candidate["observation_identity"]):
            raise InspectorError(
                "AUTOMATIC_RESULT_STORE_INVALID", "automatic observation identity is invalid"
            )
        if candidate["artifact_identity"] is not None and not _is_identity(candidate["artifact_identity"]):
            raise InspectorError(
                "AUTOMATIC_RESULT_STORE_INVALID", "automatic artifact identity is invalid"
            )
        if candidate["byte_count"] is not None and (not isinstance(candidate["byte_count"], int) or candidate["byte_count"] < 0):
            raise InspectorError(
                "AUTOMATIC_RESULT_STORE_INVALID", "automatic candidate byte count is invalid"
            )
        _validate_metadata(candidate["metadata"], nullable=True)
    registry = value["registry_snapshot"]
    if not isinstance(registry, dict) or set(registry) != REGISTRY_FIELDS:
        raise InspectorError(
            "AUTOMATIC_RESULT_STORE_INVALID", "automatic registry snapshot is not closed"
        )
    for key in ("ready_model_count", "model_rows", "active_managed_locations"):
        if not isinstance(registry[key], int) or isinstance(registry[key], bool) or registry[key] < 0:
            raise InspectorError(
                "AUTOMATIC_RESULT_STORE_INVALID", "automatic registry count is invalid"
            )
    for key in ("desired_state", "model_service_state", "default_alias", "default_target", "warm_model_id", "recovery_state", "capability_binding_identity", "operating_profile_identity"):
        if registry[key] is not None and not isinstance(registry[key], str):
            raise InspectorError(
                "AUTOMATIC_RESULT_STORE_INVALID", "automatic registry text is invalid"
            )
    if registry["capability_binding_identity"] is not None and not _is_identity(registry["capability_binding_identity"]):
        raise InspectorError(
            "AUTOMATIC_RESULT_STORE_INVALID", "automatic binding identity is invalid"
        )
    if registry["operating_profile_identity"] is not None and not _is_identity(registry["operating_profile_identity"]):
        raise InspectorError(
            "AUTOMATIC_RESULT_STORE_INVALID", "automatic profile identity is invalid"
        )
    request = value["derived_deployment_request"]
    if request is not None:
        if not isinstance(request, dict) or set(request) != REQUEST_FIELDS:
            raise InspectorError(
                "AUTOMATIC_RESULT_STORE_INVALID", "automatic request is not closed"
            )
        if request["deployment_mode"] not in {"install-first", "add", "replace-default"} or request["required_capability_profile"] not in QUALIFICATION_PROFILES or request["retirement_policy"] not in {"retain-incumbent", "retire-incumbent-after-acceptance"} or request["retirement_action"] not in {"none", "retain-incumbent", "retire-incumbent-after-acceptance"}:
            raise InspectorError(
                "AUTOMATIC_RESULT_STORE_INVALID", "automatic request enum is invalid"
            )
    _validate_reference(value["active_transaction_reference"], existing=False)
    _validate_reference(value["existing_result_reference"], existing=True)
    if not _is_identity(value["result_identity"]):
        raise InspectorError(
            "AUTOMATIC_RESULT_STORE_INVALID", "automatic result identity is invalid"
        )
    expected = _identity({key: value[key] for key in sorted(value) if key != "result_identity"})
    if value["result_identity"] != expected:
        raise InspectorError(
            "AUTOMATIC_RESULT_STORE_INVALID", "automatic result identity does not round-trip"
        )
    return value


def _empty_registry() -> dict[str, Any]:
    return {
        "desired_state": None,
        "model_service_state": None,
        "ready_model_count": 0,
        "model_rows": 0,
        "active_managed_locations": 0,
        "default_alias": None,
        "default_target": None,
        "warm_model_id": None,
        "active_transaction_reference": None,
        "recovery_state": None,
        "capability_binding_identity": None,
        "operating_profile_identity": None,
        "registry_generation": None,
    }


def _make_result(
    *,
    reconciliation_id: str,
    root_identity: str,
    visible_count: int,
    candidate: dict[str, Any] | None,
    registry: dict[str, Any],
    request: dict[str, Any] | None,
    action: str,
    reason_code: str,
    active: dict[str, str] | None,
    existing: dict[str, str] | None,
    source_configuration_identity: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": SCHEMA_IDENTITIES["automatic_intake_result"],
        "operation": "reconcile-intake",
        "reconciliation_id": reconciliation_id,
        "created_utc": utc_now(),
        "action": action,
        "reason_code": reason_code,
        "intake_root_identity": root_identity,
        "visible_candidate_count": visible_count,
        "candidate": candidate,
        "registry_snapshot": registry,
        "derived_deployment_request": request,
        "active_transaction_reference": active,
        "existing_result_reference": existing,
        "source_configuration_identity": source_configuration_identity,
        "result_identity": "",
    }
    result["result_identity"] = _identity({key: result[key] for key in sorted(result) if key != "result_identity"})
    return validate_automatic_result(result)


def _source_configuration_identity(root_identity: str, policy: AutomaticIntakePolicy) -> str:
    return _identity(
        {
            "intake_root_identity": root_identity,
            "policy": policy.as_dict(),
            "schema_version": SCHEMA_IDENTITIES["automatic_intake_result"],
            "source_implementation_epoch": AUTOMATIC_SOURCE_IMPLEMENTATION_EPOCH,
            "deployment_input_implementation_epoch": DEPLOYMENT_INPUT_IMPLEMENTATION_EPOCH,
            "service_continuity_recovery_epoch": SERVICE_CONTINUITY_RECOVERY_EPOCH,
        }
    )


def _active_reference(record: dict[str, Any]) -> dict[str, str]:
    transaction_id = record.get("transaction_id")
    deployment_id = record.get("deployment_id")
    return {
        "transaction_id": transaction_id if isinstance(transaction_id, str) and transaction_id else "unknown",
        "deployment_id": deployment_id if isinstance(deployment_id, str) and deployment_id else "unknown",
    }


def _find_completed_by_basis(
    paths: InspectorPaths,
    *,
    artifact_identity: str,
    request: dict[str, Any],
    basis_identity: str,
) -> dict[str, str] | None:
    matches: list[dict[str, str]] = []
    for path in sorted(paths.deployment_results.glob("deployment-*.json")):
        try:
            result = validate_deployment_result(read_json_record(path))
        except (OSError, InspectorError) as error:
            raise InspectorError(
                "AUTOMATIC_RESULT_STORE_INVALID", "deployment result store contains invalid evidence"
            ) from error
        if result["result_class"] != "DEPLOYMENT_COMPLETE" or result["input_identity"] != basis_identity:
            continue
        source = result["source_candidate"]
        if source["artifact_identity"] != artifact_identity or result["deployment_mode"] != request["deployment_mode"] or result["required_capability_profile"] != request["required_capability_profile"] or result["retirement_policy"] != request["retirement_policy"]:
            continue
        transaction_path = paths.transactions / f"{result['transaction_id']}.json"
        try:
            transaction = read_json_record(transaction_path)
        except (OSError, InspectorError) as error:
            raise InspectorError(
                "AUTOMATIC_RESULT_STORE_INVALID", "deployment result lacks terminal transaction evidence"
            ) from error
        if (
            transaction.get("operation") != "deploy-gguf"
            or transaction.get("state") != "COMPLETE"
            or transaction.get("deployment_id") != result["deployment_id"]
            or transaction.get("deployment_result_identity") != result["result_identity"]
            or transaction.get("deployment_result_path") != str(path)
        ):
            raise InspectorError(
                "AUTOMATIC_RESULT_STORE_INVALID", "deployment result and transaction disagree"
            )
        matches.append(
            {
                "transaction_id": result["transaction_id"],
                "deployment_id": result["deployment_id"],
                "result_identity": result["result_identity"],
                "result_class": result["result_class"],
            }
        )
    if len(matches) > 1:
        raise InspectorError(
            "AUTOMATIC_RESULT_STORE_INVALID", "multiple completed deployments match one dispatch basis"
        )
    return matches[0] if matches else None


def _find_rejected_by_basis(
    paths: InspectorPaths,
    *,
    artifact_identity: str,
    request: dict[str, Any],
    registry_snapshot: dict[str, Any],
    source_configuration_identity: str,
) -> bool:
    matches: list[dict[str, Any]] = []
    for path in sorted(paths.automatic_rejected_results.glob("*.json")):
        try:
            record = read_json_record(path)
        except (OSError, InspectorError) as error:
            raise InspectorError(
                "AUTOMATIC_RESULT_STORE_INVALID",
                "automatic rejected-basis store contains invalid evidence",
            ) from error
        dispatch_basis = record.get("dispatch_basis")
        if (
            record.get("basis_class") != "REJECTED"
            or record.get("artifact_identity") != artifact_identity
            or record.get("reason_code")
            != "AUTOMATIC_DISPATCH_FAILED_CLEAN"
            or not isinstance(dispatch_basis, dict)
            or dispatch_basis.get("derived_deployment_request") != request
            or dispatch_basis.get("registry_snapshot") != registry_snapshot
            or dispatch_basis.get("source_configuration_identity")
            != source_configuration_identity
        ):
            continue
        matches.append(record)
    if len(matches) > 1:
        raise InspectorError(
            "AUTOMATIC_RESULT_STORE_INVALID",
            "multiple rejected bases match one dispatch basis",
        )
    return bool(matches)


def _map_reconciliation_error(error: InspectorError) -> tuple[str, str] | None:
    mapping = {
        "AUTOMATIC_READY_MODEL_PRESENT": ("NOOP_READY_MODEL_PRESENT", "AUTOMATIC_READY_MODEL_PRESENT"),
        "AUTOMATIC_ACTIVE_TRANSACTION": ("NOOP_ACTIVE_TRANSACTION", "AUTOMATIC_ACTIVE_TRANSACTION"),
        "AUTOMATIC_OWNERSHIP_UNCERTAIN": ("NOOP_OWNERSHIP_UNCERTAIN", "AUTOMATIC_OWNERSHIP_UNCERTAIN"),
        "DEPLOYMENT_ACTIVE": ("NOOP_ACTIVE_TRANSACTION", "AUTOMATIC_ACTIVE_TRANSACTION"),
        "DEPLOYMENT_OWNERSHIP_UNCERTAIN": ("NOOP_OWNERSHIP_UNCERTAIN", "AUTOMATIC_OWNERSHIP_UNCERTAIN"),
        "DEPLOYMENT_SOURCE_CHANGED": ("NOOP_COPY_IN_PROGRESS", "AUTOMATIC_DEPLOYMENT_SOURCE_CHANGED"),
        "ARTIFACT_CHANGED_DURING_INSPECTION": ("NOOP_COPY_IN_PROGRESS", "AUTOMATIC_COPY_IN_PROGRESS"),
    }
    return mapping.get(error.reason_code)


def reconcile_automatic_intake(
    paths: InspectorPaths,
    policy: AutomaticIntakePolicy | dict[str, Any] | None = None,
    adapter: Any | None = None,
    dispatcher: Callable[..., tuple[str, dict[str, Any], Path, str]] = deploy_transaction,
    publishers: Any | None = None,
    *,
    waiter: Callable[[float], None] | None = None,
    observer: Callable[..., None] | None = None,
    identifier: Callable[[Path], ArtifactIdentity] = identify_regular_file,
) -> dict[str, Any]:
    del publishers
    resolved_policy = _policy(policy)
    reconciliation_id = "reconcile-" + uuid.uuid4().hex
    root_identity = _root_identity(paths)
    source_config = _source_configuration_identity(root_identity, resolved_policy)
    discovered = discover_automatic_candidate(paths, resolved_policy)
    status = discovered["status"]
    if status == "empty":
        if paths.current_connection_status.exists() or (
            paths.current_connection_status.is_symlink()
        ):
            try:
                bootstrap_current_receipt(paths)
            except InspectorError as error:
                if error.reason_code not in {
                    "CONNECTION_SERVICE_UNAVAILABLE",
                    "CONNECTION_INITIALIZATION_ACTIVE",
                    "CONNECTION_STATUS_CAS_CONFLICT",
                }:
                    raise
        return _make_result(
            reconciliation_id=reconciliation_id,
            root_identity=root_identity,
            visible_count=0,
            candidate=None,
            registry=_empty_registry(),
            request=None,
            action="NOOP_WAITING",
            reason_code="AUTOMATIC_NO_VISIBLE_CANDIDATE",
            active=None,
            existing=None,
            source_configuration_identity=source_config,
        )
    if status == "multiple":
        return _make_result(
            reconciliation_id=reconciliation_id,
            root_identity=root_identity,
            visible_count=discovered["visible_candidate_count"],
            candidate=None,
            registry=_empty_registry(),
            request=None,
            action="NOOP_MULTIPLE_CANDIDATES",
            reason_code="AUTOMATIC_MULTIPLE_CANDIDATES",
            active=None,
            existing=None,
            source_configuration_identity=source_config,
        )
    if status == "rejected":
        candidate = discovered["candidate"]
        return _make_result(
            reconciliation_id=reconciliation_id,
            root_identity=root_identity,
            visible_count=1,
            candidate=candidate,
            registry=_empty_registry(),
            request=None,
            action="REJECT_CANDIDATE",
            reason_code=discovered["reason_code"],
            active=None,
            existing=None,
            source_configuration_identity=source_config,
        )
    candidate = discovered["candidate"]
    lock_state = inspect_active_lock(paths.deployment_lock)
    stale_publication_lock = False
    if lock_state["state"] == "active":
        record = lock_state.get("record") or {}
        active = {
            "transaction_id": str(record.get("transaction_id") or "unknown"),
            "deployment_id": "unknown",
        }
        return _make_result(
            reconciliation_id=reconciliation_id,
            root_identity=root_identity,
            visible_count=1,
            candidate=candidate,
            registry=_empty_registry(),
            request=None,
            action="NOOP_ACTIVE_TRANSACTION",
            reason_code="AUTOMATIC_ACTIVE_TRANSACTION",
            active=active,
            existing=None,
            source_configuration_identity=source_config,
        )
    if lock_state["state"] == "uncertain":
        return _make_result(
            reconciliation_id=reconciliation_id,
            root_identity=root_identity,
            visible_count=1,
            candidate=candidate,
            registry=_empty_registry(),
            request=None,
            action="NOOP_OWNERSHIP_UNCERTAIN",
            reason_code="AUTOMATIC_OWNERSHIP_UNCERTAIN",
            active=None,
            existing=None,
            source_configuration_identity=source_config,
        )
    if lock_state["state"] == "stale":
        lock_record = lock_state.get("record")
        transaction_id = lock_record.get("transaction_id") if isinstance(lock_record, dict) else None
        exact = isinstance(transaction_id, str) and lock_record.get("operation") == "deploy-gguf"
        if exact:
            transaction_path = paths.transactions / f"{transaction_id}.json"
            try:
                transaction = read_json_record(transaction_path)
            except (OSError, InspectorError):
                transaction = None
            exact = isinstance(transaction, dict) and transaction.get("operation") == "deploy-gguf"
        if isinstance(lock_record, dict) and lock_record.get("operation") == "publish-service":
            stale_publication_lock = True
        elif not exact:
            return _make_result(
                reconciliation_id=reconciliation_id,
                root_identity=root_identity,
                visible_count=1,
                candidate=candidate,
                registry=_empty_registry(),
                request=None,
                action="NOOP_OWNERSHIP_UNCERTAIN",
                reason_code="AUTOMATIC_OWNERSHIP_UNCERTAIN",
                active=None,
                existing=None,
                source_configuration_identity=source_config,
            )
        try:
            _clear_exact_stale_deployment_lock(paths, transaction_id)
        except InspectorError as error:
            mapped = _map_reconciliation_error(error)
            if mapped is None:
                raise
            action, reason = mapped
            return _make_result(
                reconciliation_id=reconciliation_id,
                root_identity=root_identity,
                visible_count=1,
                candidate=candidate,
                registry=_empty_registry(),
                request=None,
                action=action,
                reason_code=reason,
                active=None,
                existing=None,
                source_configuration_identity=source_config,
            )
    try:
        stable = observe_candidate_stability(
            paths,
            candidate,
            resolved_policy,
            observer=observer,
            waiter=waiter,
            identifier=identifier,
        )
    except InspectorError as error:
        if error.reason_code == "AUTOMATIC_COPY_IN_PROGRESS":
            return _make_result(
                reconciliation_id=reconciliation_id,
                root_identity=root_identity,
                visible_count=1,
                candidate=candidate,
                registry=_empty_registry(),
                request=None,
                action="NOOP_COPY_IN_PROGRESS",
                reason_code="AUTOMATIC_COPY_IN_PROGRESS",
                active=None,
                existing=None,
                source_configuration_identity=source_config,
            )
        if error.reason_code in AUTOMATIC_REASON_CODES:
            return _make_result(
                reconciliation_id=reconciliation_id,
                root_identity=root_identity,
                visible_count=1,
                candidate=candidate,
                registry=_empty_registry(),
                request=None,
                action="REJECT_CANDIDATE",
                reason_code=error.reason_code,
                active=None,
                existing=None,
                source_configuration_identity=source_config,
            )
        raise
    candidate = stable["candidate"]
    active_adapter = adapter or CurrentSourceDeploymentAdapter()
    dispatch_request = {
        "candidate_name": candidate["relative_name"],
        "deployment_mode": "install-first",
        "required_capability_profile": "CORE_CHAT",
        "retirement_policy": "retain-incumbent",
    }
    recovery_source = {
        "candidate_name": candidate["relative_name"],
        "artifact_identity": stable["artifact"]["identity"],
        "size": stable["artifact"]["byte_count"],
        "snapshot_identity": stable["artifact"].get(
            "post_inspection_snapshot_identity"
        ),
    }
    retryable = None
    converged_retry = False
    retry_prestate = None
    try:
        # Preserve lock ownership fencing for normal intake while allowing
        # converged failed-clean recovery below to use its exact matcher.
        recoverable = (
            _find_recoverable(paths, dispatch_request, recovery_source)
            if stale_publication_lock
            else None
        )
        if recoverable is None:
            try:
                retry_prestate = active_adapter.capture_prestate(paths)
            except InspectorError:
                retry_prestate = None
            if retry_prestate is not None:
                converged_state = (
                    dispatch_request["deployment_mode"] == "install-first"
                    and _converged_install_first_retry(
                        recovery_source, retry_prestate
                    )
                )
                if converged_state:
                    retryable = _find_retryable_failed_clean(
                        paths,
                        dispatch_request,
                        recovery_source,
                        current_input_identity=_input_identity(
                            dispatch_request,
                            recovery_source,
                            retry_prestate,
                        ),
                        allow_input_identity_change=True,
                    )
                    converged_retry = retryable is not None
    except InspectorError as error:
        mapped = _map_reconciliation_error(error)
        if mapped is None:
            raise
        action, reason = mapped
        return _make_result(
            reconciliation_id=reconciliation_id,
            root_identity=root_identity,
            visible_count=1,
            candidate=candidate,
            registry=_empty_registry(),
            request=None,
            action=action,
            reason_code=reason,
            active=None,
            existing=None,
            source_configuration_identity=source_config,
        )
    if stale_publication_lock and recoverable is None:
        return _make_result(
            reconciliation_id=reconciliation_id,
            root_identity=root_identity,
            visible_count=1,
            candidate=candidate,
            registry=_empty_registry(),
            request=None,
            action="NOOP_OWNERSHIP_UNCERTAIN",
            reason_code="AUTOMATIC_OWNERSHIP_UNCERTAIN",
            active=None,
            existing=None,
            source_configuration_identity=source_config,
        )
    try:
        if recoverable is not None:
            runtime = recoverable.get("deployment_runtime")
            prestate = runtime.get("prestate") if isinstance(runtime, dict) else None
            if not isinstance(prestate, dict):
                raise InspectorError(
                    "DEPLOYMENT_OWNERSHIP_UNCERTAIN",
                    "recoverable deployment prestate is incomplete",
                )
        elif retry_prestate is not None:
            prestate = retry_prestate
        else:
            prestate = active_adapter.capture_prestate(paths)
        derived = derive_first_model_policy(
            prestate,
            allow_converged_install_first_retry=converged_retry,
        )
    except InspectorError as error:
        mapped = _map_reconciliation_error(error)
        if mapped is None:
            if error.reason_code == "AUTOMATIC_REGISTRY_CONTRADICTORY":
                mapped = ("NOOP_REGISTRY_CONTRADICTORY", error.reason_code)
            else:
                raise
        action, reason = mapped
        return _make_result(
            reconciliation_id=reconciliation_id,
            root_identity=root_identity,
            visible_count=1,
            candidate=candidate,
            registry=_registry_snapshot(prestate) if "prestate" in locals() and isinstance(prestate, dict) else _empty_registry(),
            request=None,
            action=action,
            reason_code=reason,
            active=None,
            existing=None,
            source_configuration_identity=source_config,
        )
    basis = build_automatic_dispatch_basis(
        candidate,
        stable["artifact"],
        resolved_policy,
        prestate,
        allow_converged_install_first_retry=converged_retry,
    )
    request = basis["derived_deployment_request"]
    if _find_rejected_by_basis(
        paths,
        artifact_identity=basis["artifact_identity"],
        request=request,
        registry_snapshot=derived["registry_snapshot"],
        source_configuration_identity=source_config,
    ):
        return _make_result(
            reconciliation_id=reconciliation_id,
            root_identity=root_identity,
            visible_count=1,
            candidate=candidate,
            registry=derived["registry_snapshot"],
            request=request,
            action="REJECT_CANDIDATE",
            reason_code="AUTOMATIC_DISPATCH_FAILED_CLEAN",
            active=None,
            existing=None,
            source_configuration_identity=source_config,
        )
    existing = _find_completed_by_basis(
        paths,
        artifact_identity=basis["artifact_identity"],
        request=request,
        basis_identity=basis["dispatch_basis_identity"],
    )
    if existing is not None:
        return _make_result(
            reconciliation_id=reconciliation_id,
            root_identity=root_identity,
            visible_count=1,
            candidate=candidate,
            registry=derived["registry_snapshot"],
            request=request,
            action="NOOP_ALREADY_PROCESSED",
            reason_code="AUTOMATIC_ALREADY_PROCESSED",
            active=None,
            existing=existing,
            source_configuration_identity=source_config,
        )
    dispatch_request = {
        "candidate_name": candidate["relative_name"],
        "deployment_mode": request["deployment_mode"],
        "required_capability_profile": request["required_capability_profile"],
        "retirement_policy": request["retirement_policy"],
    }
    try:
        transaction_id, record, result_path, result_identity = dispatcher(
            paths, dispatch_request, adapter=active_adapter
        )
    except InspectorError as error:
        mapped = _map_reconciliation_error(error)
        if mapped is None:
            raise
        action, reason = mapped
        return _make_result(
            reconciliation_id=reconciliation_id,
            root_identity=root_identity,
            visible_count=1,
            candidate=candidate,
            registry=derived["registry_snapshot"],
            request=request,
            action=action,
            reason_code=reason,
            active=None,
            existing=None,
            source_configuration_identity=source_config,
        )
    if not isinstance(record, dict):
        raise InspectorError(
            "AUTOMATIC_RESULT_STORE_INVALID", "deployment dispatcher returned no record"
        )
    active = {
        "transaction_id": transaction_id,
        "deployment_id": str(record.get("deployment_id") or "unknown"),
    }
    if record.get("result_class") == "DEPLOYMENT_COMPLETE":
        return _make_result(
            reconciliation_id=reconciliation_id,
            root_identity=root_identity,
            visible_count=1,
            candidate=candidate,
            registry=derived["registry_snapshot"],
            request=request,
            action="DISPATCH_FIRST_MODEL",
            reason_code="AUTOMATIC_DISPATCH_ACCEPTED",
            active=active,
            existing={
                "transaction_id": transaction_id,
                "deployment_id": str(record.get("deployment_id") or "unknown"),
                "result_identity": result_identity,
                "result_class": str(record.get("result_class")),
            },
            source_configuration_identity=source_config,
        )
    return _make_result(
        reconciliation_id=reconciliation_id,
        root_identity=root_identity,
        visible_count=1,
        candidate=candidate,
        registry=derived["registry_snapshot"],
        request=request,
        action="REJECT_CANDIDATE",
        reason_code="AUTOMATIC_DISPATCH_FAILED_CLEAN",
        active=active,
        existing=None,
        source_configuration_identity=source_config,
    )
