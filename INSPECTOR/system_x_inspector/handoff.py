"""Bounded filesystem-only GGUF branch handoff."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import ctypes
import datetime as dt
import errno
import fcntl
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .capabilities import (
    load_binding,
    load_capability_record,
    verify_installed_tuple,
)
from .constants import SCHEMA_IDENTITIES
from .decision import (
    DECISION_ID_PATTERN,
    decision_result_identity,
    load_inspection_result,
    validate_decision_record,
)
from .errors import InspectorError
from .locking import TransactionLock, inspect_active_lock
from .paths import BranchHandoffPaths, InspectorPaths
from .records import (
    atomic_create_json,
    canonical_json_bytes,
    fsync_directory,
    read_json_record,
)
from .results import utc_now
from .runtime import (
    _status_value,
    _transaction_id,
    _write_status,
    _write_transaction,
)


SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
HANDOFF_ID_PATTERN = re.compile(
    r"handoff-[0-9]{8}T[0-9]{12}Z-[0-9a-f]{16}\Z"
)
MODE_PATTERN = re.compile(r"[0-7]{4}\Z")
BASE_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "handoff_id",
        "transaction_id",
        "created_utc",
        "completed_utc",
        "status",
        "decision",
        "inspection",
        "capability",
        "source",
        "staging",
        "publication",
        "identity_match",
        "registry_observation",
        "alias_protection",
        "runtime_protection",
        "cleanup_ownership",
        "result_identity",
    }
)
TOP_LEVEL_FIELDS = BASE_TOP_LEVEL_FIELDS | {"qualification"}
NESTED_FIELDS = {
    "decision": frozenset(
        {
            "decision_id",
            "result_identity",
            "decision_basis_identity",
            "capability_result",
            "selected_branch",
            "handoff_allowed",
            "spawn_allowed",
        }
    ),
    "inspection": frozenset(
        {
            "inspection_id",
            "result_identity",
            "physical_format",
            "artifact_identity",
            "artifact_size",
        }
    ),
    "capability": frozenset(
        {
            "record_id",
            "record_identity",
            "binding_identity",
            "binding_generation",
            "installed_tuple_verified",
        }
    ),
    "source": frozenset(
        {
            "intake_root_identity",
            "relative_name",
            "device",
            "inode",
            "mode",
            "link_count",
            "size_bytes",
            "sha256",
            "pre_copy_identity",
            "post_copy_identity",
            "unchanged_during_handoff",
        }
    ),
    "staging": frozenset(
        {
            "branch_relative_path",
            "transfer_method",
            "device",
            "inode",
            "mode",
            "link_count",
            "size_bytes",
            "sha256",
            "complete_write",
            "file_fsync",
            "directory_fsync",
        }
    ),
    "publication": frozenset(
        {
            "managed_relative_path",
            "method",
            "device",
            "inode",
            "mode",
            "link_count",
            "size_bytes",
            "sha256",
            "staging_inode_preserved",
            "parent_directory_fsync",
            "collision_absent",
        }
    ),
    "identity_match": frozenset(
        {
            "source_equals_decision",
            "staged_equals_source",
            "published_equals_staged",
        }
    ),
    "registry_observation": frozenset(
        {
            "state_at_handoff_completion",
            "readiness_claimed",
            "next_mini_required",
        }
    ),
    "alias_protection": frozenset(
        {
            "default_alias",
            "default_target_before",
            "default_target_after",
            "unchanged",
        }
    ),
    "runtime_protection": frozenset(
        {
            "service_was_running",
            "service_remained_running",
            "lifecycle_operation_count",
        }
    ),
    "cleanup_ownership": frozenset(
        {"source", "staging", "published_artifact"}
    ),
}
FORBIDDEN_KEYS = frozenset(
    {
        "api_key",
        "credential_verifier",
        "credential_pepper",
        "private_router_url",
        "model_child_port",
        "process_environment",
        "absolute_source_path",
        "absolute_managed_path",
        "tensor_payload",
    }
)
REGISTRY_STATES = frozenset(
    {
        "DELEGATED_NOT_OBSERVED",
        "NOT_YET_OBSERVED",
        "REGISTERED",
        "PROBING",
        "READY",
        "REJECTED",
        "ERROR",
    }
)
SOURCE_HASH_CHUNK_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class DecisionAuthorization:
    decision: dict[str, Any]
    inspection: dict[str, Any]
    inspection_result_identity: str
    capability_record: dict[str, Any]
    binding: dict[str, Any]
    installed_tuple_verification: dict[str, Any]
    qualification: dict[str, Any] | None = None


@dataclass(frozen=True)
class SourceEvidence:
    path: Path
    intake_root_identity: str
    relative_name: str
    snapshot: dict[str, int]
    snapshot_identity: str
    artifact_identity: str


@dataclass(frozen=True)
class ManagedPolicy:
    mode: int
    owner_uid: int
    owner_gid: int
    reference_names: tuple[str, ...]


@dataclass(frozen=True)
class DestinationPlan:
    branch_paths: BranchHandoffPaths
    transaction_id: str
    managed_name: str
    managed_relative_path: str
    managed_target: Path
    staging_name: str
    staging_relative_path: str
    staging_path: Path
    policy: ManagedPolicy
    managed_root_identity: tuple[int, int, int]


@dataclass(frozen=True)
class StagedArtifact:
    path: Path
    relative_path: str
    transfer_method: str
    device: int
    inode: int
    mode: int
    link_count: int
    size_bytes: int
    sha256: str
    source_snapshot_identity: str


@dataclass(frozen=True)
class PublishedArtifact:
    path: Path
    relative_path: str
    device: int
    inode: int
    mode: int
    link_count: int
    size_bytes: int
    sha256: str


FICLONE = 0x40049409
RENAME_NOREPLACE = 1
AT_FDCWD = -100
DEFAULT_SAFETY_MARGIN_BYTES = 64 * 1024 * 1024


def _error(reason: str, message: str, *, internal: bool = False) -> InspectorError:
    return InspectorError(
        reason, message, exit_status=70 if internal else 2
    )


def _exact(value: object, fields: Iterable[str], label: str) -> dict[str, Any]:
    expected = set(fields)
    if not isinstance(value, dict) or set(value) != expected:
        raise _error(
            "HANDOFF_RESULT_COLLISION", f"{label} fields are not closed"
        )
    return value


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise _error(
            "HANDOFF_RESULT_COLLISION", f"{label} is not a SHA-256 identity"
        )
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise _error(
            "HANDOFF_RESULT_COLLISION", f"{label} is not a non-empty string"
        )
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < minimum
    ):
        raise _error(
            "HANDOFF_RESULT_COLLISION", f"{label} is not a valid integer"
        )
    return value


def _mode(value: object, label: str) -> str:
    if not isinstance(value, str) or MODE_PATTERN.fullmatch(value) is None:
        raise _error(
            "HANDOFF_RESULT_COLLISION", f"{label} is not a file mode"
        )
    return value


def _relative(value: object, label: str) -> str:
    text = _string(value, label)
    path = Path(text)
    if (
        path.is_absolute()
        or "\\" in text
        or text.startswith(".")
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise _error(
            "HANDOFF_RESULT_COLLISION", f"{label} is not a safe relative path"
        )
    return text


def _reject_forbidden_keys(value: object) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_KEYS:
                raise _error(
                    "HANDOFF_RESULT_COLLISION",
                    f"handoff record contains prohibited field: {key}",
                )
            _reject_forbidden_keys(child)
    elif isinstance(value, list):
        for child in value:
            _reject_forbidden_keys(child)


def handoff_result_identity(value: dict[str, Any]) -> str:
    fields = set(value)
    if fields not in (
        set(BASE_TOP_LEVEL_FIELDS),
        set(TOP_LEVEL_FIELDS),
    ):
        raise _error(
            "HANDOFF_RESULT_COLLISION",
            "handoff result fields are not closed",
        )
    basis = {
        key: value[key]
        for key in fields
        if key != "result_identity"
    }
    return "sha256:" + hashlib.sha256(canonical_json_bytes(basis)).hexdigest()


def validate_handoff_record(value: object) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or set(value)
        not in (set(BASE_TOP_LEVEL_FIELDS), set(TOP_LEVEL_FIELDS))
    ):
        raise _error(
            "HANDOFF_RESULT_COLLISION",
            "handoff result fields are not closed",
        )
    record = value
    _reject_forbidden_keys(record)
    if record["schema_version"] != SCHEMA_IDENTITIES["handoff_result"]:
        raise _error(
            "HANDOFF_RESULT_COLLISION", "handoff schema identity is invalid"
        )
    if (
        not isinstance(record["handoff_id"], str)
        or HANDOFF_ID_PATTERN.fullmatch(record["handoff_id"]) is None
    ):
        raise _error("HANDOFF_RESULT_COLLISION", "handoff ID is invalid")
    for key in ("transaction_id", "created_utc", "completed_utc"):
        _string(record[key], f"handoff {key}")
    if record["status"] != "PUBLISHED_TO_BRANCH":
        raise _error("HANDOFF_RESULT_COLLISION", "handoff status is invalid")

    nested = {
        key: _exact(record[key], fields, f"handoff {key}")
        for key, fields in NESTED_FIELDS.items()
    }
    decision = nested["decision"]
    _string(decision["decision_id"], "decision ID")
    _sha(decision["result_identity"], "decision result identity")
    _sha(decision["decision_basis_identity"], "decision basis identity")
    qualification = record.get("qualification")
    direct_supported = (
        qualification is None
        and decision["capability_result"] == "SUPPORTED"
        and decision["selected_branch"] == "model-api-gguf"
        and decision["handoff_allowed"] is True
        and decision["spawn_allowed"] is True
    )
    qualified_runtime = (
        isinstance(qualification, dict)
        and set(qualification)
        == {
            "qualification_id",
            "result_identity",
            "result_class",
            "requested_profile",
        }
        and isinstance(qualification["qualification_id"], str)
        and isinstance(qualification["result_identity"], str)
        and SHA256_PATTERN.fullmatch(
            qualification["result_identity"]
        )
        is not None
        and qualification["result_class"]
        == "SUPPORTED_FOR_CURRENT_TUPLE"
        and isinstance(qualification["requested_profile"], str)
        and qualification["requested_profile"]
        and decision["capability_result"]
        == "RUNTIME_SMOKE_REQUIRED"
        and decision["selected_branch"] is None
        and decision["handoff_allowed"] is False
        and decision["spawn_allowed"] is False
    )
    if not direct_supported and not qualified_runtime:
        raise _error(
            "HANDOFF_RESULT_COLLISION",
            "handoff lacks direct or qualified GGUF authorization",
        )

    inspection = nested["inspection"]
    _string(inspection["inspection_id"], "inspection ID")
    _sha(inspection["result_identity"], "inspection result identity")
    _sha(inspection["artifact_identity"], "inspection artifact identity")
    _integer(inspection["artifact_size"], "inspection artifact size", minimum=1)
    if inspection["physical_format"] != "GGUF":
        raise _error(
            "HANDOFF_RESULT_COLLISION", "handoff inspection is not GGUF"
        )

    capability = nested["capability"]
    _string(capability["record_id"], "capability record ID")
    _sha(capability["record_identity"], "capability record identity")
    _sha(capability["binding_identity"], "capability binding identity")
    _integer(
        capability["binding_generation"],
        "capability binding generation",
        minimum=1,
    )
    if capability["installed_tuple_verified"] is not True:
        raise _error(
            "HANDOFF_RESULT_COLLISION",
            "installed capability tuple was not verified",
        )

    source = nested["source"]
    _sha(source["intake_root_identity"], "intake root identity")
    relative_name = _relative(source["relative_name"], "source relative name")
    if Path(relative_name).name != relative_name:
        raise _error(
            "HANDOFF_RESULT_COLLISION", "source is not a direct-child name"
        )
    _integer(source["device"], "source device")
    _integer(source["inode"], "source inode", minimum=1)
    _mode(source["mode"], "source mode")
    _integer(source["size_bytes"], "source size", minimum=1)
    _sha(source["sha256"], "source identity")
    _sha(source["pre_copy_identity"], "source pre-copy identity")
    _sha(source["post_copy_identity"], "source post-copy identity")
    if (
        source["link_count"] != 1
        or source["unchanged_during_handoff"] is not True
        or source["pre_copy_identity"] != source["post_copy_identity"]
    ):
        raise _error(
            "HANDOFF_RESULT_COLLISION", "source physical invariants are invalid"
        )

    staging = nested["staging"]
    _relative(staging["branch_relative_path"], "staging relative path")
    if staging["transfer_method"] not in {
        "reflink_clone",
        "bounded_stream_copy",
    }:
        raise _error(
            "HANDOFF_RESULT_COLLISION", "staging transfer method is invalid"
        )
    _validate_file_surface(staging, "staging")
    if (
        staging["complete_write"] is not True
        or staging["file_fsync"] is not True
        or staging["directory_fsync"] is not True
    ):
        raise _error(
            "HANDOFF_RESULT_COLLISION", "staging durability is incomplete"
        )

    publication = nested["publication"]
    _relative(
        publication["managed_relative_path"], "managed relative path"
    )
    if (
        publication["method"]
        != "same_filesystem_atomic_no_overwrite_rename"
    ):
        raise _error(
            "HANDOFF_RESULT_COLLISION", "publication method is invalid"
        )
    _validate_file_surface(publication, "publication")
    if (
        publication["staging_inode_preserved"] is not True
        or publication["parent_directory_fsync"] is not True
        or publication["collision_absent"] is not True
    ):
        raise _error(
            "HANDOFF_RESULT_COLLISION",
            "publication physical invariants are invalid",
        )

    if any(value is not True for value in nested["identity_match"].values()):
        raise _error(
            "HANDOFF_RESULT_COLLISION", "handoff identities do not match"
        )
    registry = nested["registry_observation"]
    if (
        registry["state_at_handoff_completion"] not in REGISTRY_STATES
        or registry["readiness_claimed"] is not False
        or registry["next_mini_required"] is not True
    ):
        raise _error(
            "HANDOFF_RESULT_COLLISION", "registry delegation is invalid"
        )
    alias = nested["alias_protection"]
    for key in (
        "default_alias",
        "default_target_before",
        "default_target_after",
    ):
        _string(alias[key], f"alias protection {key}")
    if (
        alias["unchanged"] is not True
        or alias["default_target_before"] != alias["default_target_after"]
    ):
        raise _error(
            "HANDOFF_RESULT_COLLISION", "default alias protection is invalid"
        )
    runtime = nested["runtime_protection"]
    if (
        runtime["service_was_running"] is not True
        or runtime["service_remained_running"] is not True
        or runtime["lifecycle_operation_count"] != 0
    ):
        raise _error(
            "HANDOFF_RESULT_COLLISION", "runtime protection is invalid"
        )
    cleanup = nested["cleanup_ownership"]
    if cleanup != {
        "source": "caller_or_packet_owned",
        "staging": "handoff_transaction_owned",
        "published_artifact": "branch_owned_after_success",
    }:
        raise _error(
            "HANDOFF_RESULT_COLLISION", "cleanup ownership is invalid"
        )
    _sha(record["result_identity"], "handoff result identity")
    if record["result_identity"] != handoff_result_identity(record):
        raise _error(
            "HANDOFF_RESULT_COLLISION",
            "handoff result identity does not match its canonical basis",
        )
    return record


def _validate_file_surface(value: dict[str, Any], label: str) -> None:
    _integer(value["device"], f"{label} device")
    _integer(value["inode"], f"{label} inode", minimum=1)
    _mode(value["mode"], f"{label} mode")
    _integer(value["size_bytes"], f"{label} size", minimum=1)
    _sha(value["sha256"], f"{label} identity")
    if value["link_count"] != 1:
        raise _error(
            "HANDOFF_RESULT_COLLISION", f"{label} link count is invalid"
        )


def finalize_handoff_record(value: dict[str, Any]) -> dict[str, Any]:
    record = {**value, "result_identity": None}
    record["result_identity"] = handoff_result_identity(record)
    return validate_handoff_record(record)


def _private_handoff_file(path: Path) -> bytes:
    try:
        details = path.lstat()
    except FileNotFoundError as error:
        raise _error(
            "HANDOFF_RESULT_COLLISION", "handoff result is absent"
        ) from error
    if (
        stat.S_ISLNK(details.st_mode)
        or not stat.S_ISREG(details.st_mode)
        or stat.S_IMODE(details.st_mode) != 0o600
        or details.st_nlink != 1
    ):
        raise _error(
            "HANDOFF_RESULT_COLLISION",
            "handoff result has an unsafe physical type",
        )
    return path.read_bytes()


def handoff_result_path(paths: InspectorPaths, handoff_id: str) -> Path:
    if HANDOFF_ID_PATTERN.fullmatch(handoff_id) is None:
        raise _error("HANDOFF_RESULT_COLLISION", "handoff ID is unsafe")
    return paths.handoff_results / f"{handoff_id}.json"


def publish_handoff_record(
    paths: InspectorPaths, value: dict[str, Any]
) -> tuple[Path, str]:
    record = validate_handoff_record(value)
    path = handoff_result_path(paths, record["handoff_id"])
    if path.parent != paths.handoff_results:
        raise _error(
            "HANDOFF_RESULT_COLLISION", "handoff path escaped its result root"
        )
    if path.exists() or path.is_symlink():
        if _private_handoff_file(path) == canonical_json_bytes(record):
            return path, record["result_identity"]
        raise _error(
            "HANDOFF_RESULT_COLLISION",
            "different immutable handoff record already exists",
        )
    try:
        atomic_create_json(path, record, mode=0o600)
    except InspectorError as error:
        if (
            path.exists()
            and not path.is_symlink()
            and _private_handoff_file(path) == canonical_json_bytes(record)
        ):
            return path, record["result_identity"]
        raise _error(
            "HANDOFF_RESULT_COLLISION",
            "handoff result atomic publication collided",
        ) from error
    if validate_handoff_record(read_json_record(path)) != record:
        raise _error(
            "HANDOFF_RESULT_COLLISION",
            "handoff result did not round-trip",
            internal=True,
        )
    return path, record["result_identity"]


def load_handoff_record(
    paths: InspectorPaths, handoff_id: str
) -> tuple[dict[str, Any], str]:
    path = handoff_result_path(paths, handoff_id)
    raw = _private_handoff_file(path)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _error(
            "HANDOFF_RESULT_COLLISION", "handoff result JSON is invalid"
        ) from error
    record = validate_handoff_record(value)
    return record, record["result_identity"]


def _read_private_json(
    path: Path,
    *,
    missing_reason: str,
    invalid_reason: str,
    label: str,
) -> dict[str, Any]:
    try:
        descriptor = os.open(
            path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
        )
    except FileNotFoundError as error:
        raise _error(missing_reason, f"{label} is absent") from error
    except (OSError, ValueError) as error:
        raise _error(invalid_reason, f"{label} cannot be opened safely") from error
    try:
        details = os.fstat(descriptor)
        if (
            not stat.S_ISREG(details.st_mode)
            or stat.S_IMODE(details.st_mode) != 0o600
            or details.st_nlink != 1
        ):
            raise _error(
                invalid_reason, f"{label} has an unsafe physical type"
            )
        chunks = []
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > 16 * 1024 * 1024:
                raise _error(invalid_reason, f"{label} exceeds its bound")
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    try:
        value = json.loads(b"".join(chunks))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _error(invalid_reason, f"{label} JSON is invalid") from error
    if not isinstance(value, dict):
        raise _error(invalid_reason, f"{label} is not a JSON object")
    return value


def _load_authenticated_decision(
    paths: InspectorPaths, decision_id: str
) -> dict[str, Any]:
    if (
        not isinstance(decision_id, str)
        or DECISION_ID_PATTERN.fullmatch(decision_id) is None
        or Path(decision_id).name != decision_id
    ):
        raise _error(
            "HANDOFF_DECISION_INVALID", "decision ID is not canonical"
        )
    path = paths.decision_results / f"{decision_id}.json"
    value = _read_private_json(
        path,
        missing_reason="HANDOFF_DECISION_NOT_FOUND",
        invalid_reason="HANDOFF_DECISION_INVALID",
        label="decision record",
    )
    try:
        decision = validate_decision_record(value)
    except InspectorError as error:
        raise _error(
            "HANDOFF_DECISION_INVALID",
            "decision record failed closed validation",
        ) from error
    if (
        decision["decision_id"] != decision_id
        or decision["result_identity"] != decision_result_identity(decision)
    ):
        raise _error(
            "HANDOFF_DECISION_INVALID",
            "decision identity does not authenticate its record",
        )
    transaction_id = decision["transaction_id"]
    if (
        not isinstance(transaction_id, str)
        or Path(transaction_id).name != transaction_id
    ):
        raise _error(
            "HANDOFF_DECISION_INVALID",
            "decision transaction reference is unsafe",
        )
    transaction_path = paths.transactions / f"{transaction_id}.json"
    transaction = _read_private_json(
        transaction_path,
        missing_reason="HANDOFF_DECISION_INVALID",
        invalid_reason="HANDOFF_DECISION_INVALID",
        label="decision transaction",
    )
    if (
        transaction.get("schema_version")
        != SCHEMA_IDENTITIES["transaction"]
        or transaction.get("transaction_id") != transaction_id
        or transaction.get("operation") != "decide"
        or transaction.get("state") != "COMPLETED"
        or transaction.get("decision_id") != decision_id
        or transaction.get("decision_result_identity")
        != decision["result_identity"]
        or transaction.get("decision_result_path") != str(path)
        or transaction.get("inspection_id")
        != decision["inspection"]["inspection_id"]
    ):
        raise _error(
            "HANDOFF_DECISION_INVALID",
            "decision transaction does not authenticate the result",
        )
    return decision


def _load_linked_inspection(
    paths: InspectorPaths, decision: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    try:
        inspection, result_identity = load_inspection_result(
            paths, decision["inspection"]["inspection_id"]
        )
    except InspectorError as error:
        raise _error(
            "HANDOFF_DECISION_STALE",
            "linked inspection record is absent, invalid, or stale",
        ) from error
    linked = decision["inspection"]
    if (
        result_identity != linked["inspection_result_identity"]
        or inspection["artifact"]["identity"] != linked["artifact_identity"]
        or inspection["classification"]["terminal_class"]
        != linked["physical_format"]
        or inspection["source"]["candidate_name"]
        != linked["source_target_name"]
    ):
        raise _error(
            "HANDOFF_DECISION_STALE",
            "linked inspection identity no longer matches the decision",
        )
    return inspection, result_identity


def _load_linked_capability(
    paths: InspectorPaths, decision: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    surface = decision["capability"]
    branch_identity = surface["branch_identity"]
    if not surface["evaluated"]:
        raise _error(
            "HANDOFF_DECISION_NOT_SUPPORTED",
            "decision contains no evaluated serving capability",
        )
    if branch_identity not in {"model-api-gguf", "model-api-native"}:
        raise _error(
            "HANDOFF_CAPABILITY_BINDING_INVALID",
            "decision capability branch is invalid",
        )
    try:
        binding = load_binding(paths, branch_identity)
        capability = load_capability_record(
            paths, binding["capability_record_id"]
        )
    except InspectorError as error:
        raise _error(
            "HANDOFF_CAPABILITY_BINDING_INVALID",
            "current capability binding or record is invalid",
        ) from error
    if (
        binding["binding_identity"] != surface["binding_identity"]
        or binding["capability_record_id"]
        != surface["capability_record_id"]
        or binding["capability_record_identity"]
        != surface["capability_record_identity"]
        or capability["capability_record_identity"]
        != binding["capability_record_identity"]
        or capability["branch_identity"] != branch_identity
    ):
        raise _error(
            "HANDOFF_CAPABILITY_BINDING_INVALID",
            "current capability binding no longer matches the decision",
        )
    try:
        verification = verify_installed_tuple(paths, capability)
    except InspectorError as error:
        raise _error(
            "HANDOFF_INSTALLED_TUPLE_MISMATCH",
            "current installed tuple cannot be verified",
        ) from error
    if verification.get("verified") is not True:
        raise _error(
            "HANDOFF_INSTALLED_TUPLE_MISMATCH",
            "current installed tuple differs from accepted capability",
        )
    return capability, binding, verification


def authenticate_handoff_decision(
    paths: InspectorPaths, decision_id: str
) -> DecisionAuthorization:
    """Authenticate the complete decision chain before source-byte access."""

    decision = _load_authenticated_decision(paths, decision_id)
    inspection, inspection_identity = _load_linked_inspection(
        paths, decision
    )
    capability_result = decision["capability"]["capability_result"]
    if not decision["capability"]["evaluated"]:
        raise _error(
            "HANDOFF_DECISION_NOT_SUPPORTED",
            "decision physical class has no serving capability",
        )
    capability, binding, verification = _load_linked_capability(
        paths, decision
    )
    if capability_result != "SUPPORTED":
        raise _error(
            "HANDOFF_DECISION_NOT_SUPPORTED",
            "decision is a no-handoff capability result",
        )
    if decision["selected_branch"] != "model-api-gguf":
        raise _error(
            "HANDOFF_BRANCH_NOT_SELECTED",
            "decision did not select the GGUF branch",
        )
    if (
        decision["handoff_allowed"] is not True
        or decision["spawn_allowed"] is not True
    ):
        raise _error(
            "HANDOFF_NOT_AUTHORIZED",
            "decision does not authorize handoff and spawn",
        )
    if (
        inspection["classification"]["terminal_class"] != "GGUF"
        or decision["inspection"]["physical_format"] != "GGUF"
        or capability["supported_physical_format"] != "GGUF"
        or capability["branch_identity"] != "model-api-gguf"
    ):
        raise _error(
            "HANDOFF_DECISION_STALE",
            "supported decision evidence is not exact GGUF",
        )
    supported = capability["supported_evidence"].get(
        "supported_exact_artifact_identities", []
    )
    if decision["inspection"]["artifact_identity"] not in supported:
        raise _error(
            "HANDOFF_NOT_AUTHORIZED",
            "decision artifact is outside the exact supported set",
        )
    return DecisionAuthorization(
        decision=decision,
        inspection=inspection,
        inspection_result_identity=inspection_identity,
        capability_record=capability,
        binding=binding,
        installed_tuple_verification=verification,
    )


def authenticate_qualified_handoff(
    paths: InspectorPaths,
    decision_id: str,
    qualification_id: str,
) -> DecisionAuthorization:
    """Authorize a smoke-required decision with exact immutable qualification."""

    from .qualification import (
        QUALIFICATION_ID_PATTERN,
        qualification_result_path,
        validate_qualification_record,
    )

    if (
        not isinstance(qualification_id, str)
        or QUALIFICATION_ID_PATTERN.fullmatch(qualification_id) is None
    ):
        raise _error(
            "HANDOFF_QUALIFICATION_INVALID",
            "qualification ID is not canonical",
        )
    path = qualification_result_path(paths, qualification_id)
    value = _read_private_json(
        path,
        missing_reason="HANDOFF_QUALIFICATION_NOT_FOUND",
        invalid_reason="HANDOFF_QUALIFICATION_INVALID",
        label="qualification record",
    )
    try:
        qualification = validate_qualification_record(value)
    except InspectorError as error:
        raise _error(
            "HANDOFF_QUALIFICATION_INVALID",
            "qualification record failed closed validation",
        ) from error
    decision = _load_authenticated_decision(paths, decision_id)
    inspection, inspection_identity = _load_linked_inspection(
        paths, decision
    )
    capability, binding, verification = _load_linked_capability(
        paths, decision
    )
    linked = qualification.get("input_decision")
    qualified_inspection = qualification.get("inspection")
    installed = qualification.get("installed_tuple")
    validity = qualification.get("validity_predicate")
    if (
        qualification["qualification_id"] != qualification_id
        or qualification["result_class"]
        != "SUPPORTED_FOR_CURRENT_TUPLE"
        or qualification["requested_profile"]
        not in qualification["supported_profiles"]
        or not isinstance(linked, dict)
        or linked.get("decision_id") != decision_id
        or linked.get("decision_result_identity")
        != decision["result_identity"]
        or linked.get("capability_result")
        != "RUNTIME_SMOKE_REQUIRED"
        or decision["capability"]["capability_result"]
        != "RUNTIME_SMOKE_REQUIRED"
        or decision["selected_branch"] is not None
        or decision["handoff_allowed"] is not False
        or decision["spawn_allowed"] is not False
        or not isinstance(qualified_inspection, dict)
        or qualified_inspection.get("inspection_id")
        != inspection["inspection_id"]
        or qualified_inspection.get("inspection_result_identity")
        != inspection_identity
        or qualified_inspection.get("artifact_identity")
        != inspection["artifact"]["identity"]
        or not isinstance(installed, dict)
        or installed.get("branch_capability_record_identity")
        != capability["capability_record_identity"]
        or installed.get("capability_binding_identity")
        != binding["binding_identity"]
        or not isinstance(validity, dict)
        or validity.get("artifact_identity")
        != inspection["artifact"]["identity"]
        or validity.get("capability_record_identity")
        != capability["capability_record_identity"]
        or validity.get("binding_identity") != binding["binding_identity"]
        or validity.get("installed_tuple_verification_identity")
        != installed.get("installed_tuple_verification_identity")
    ):
        raise _error(
            "HANDOFF_QUALIFICATION_STALE",
            "qualification does not authenticate the current decision tuple",
        )
    return DecisionAuthorization(
        decision=decision,
        inspection=inspection,
        inspection_result_identity=inspection_identity,
        capability_record=capability,
        binding=binding,
        installed_tuple_verification=verification,
        qualification=qualification,
    )


def _stat_snapshot(details: os.stat_result) -> dict[str, int]:
    return {
        "device": details.st_dev,
        "inode": details.st_ino,
        "mode": stat.S_IMODE(details.st_mode),
        "link_count": details.st_nlink,
        "size_bytes": details.st_size,
        "mtime_ns": details.st_mtime_ns,
        "ctime_ns": details.st_ctime_ns,
    }


def _snapshot_identity(value: dict[str, int]) -> str:
    return (
        "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    )


def _intake_root_identity(path: Path) -> str:
    try:
        details = path.lstat()
    except FileNotFoundError as error:
        raise _error(
            "HANDOFF_SOURCE_INVALID", "MODEL-TEST root is absent"
        ) from error
    if (
        stat.S_ISLNK(details.st_mode)
        or not stat.S_ISDIR(details.st_mode)
        or path.resolve(strict=True) != path
    ):
        raise _error(
            "HANDOFF_SOURCE_INVALID",
            "MODEL-TEST root is not a real contained directory",
        )
    basis = {
        "device": details.st_dev,
        "inode": details.st_ino,
        "mode": stat.S_IMODE(details.st_mode),
    }
    return "sha256:" + hashlib.sha256(canonical_json_bytes(basis)).hexdigest()


def _validate_source_name(source_candidate: str) -> str:
    if (
        not isinstance(source_candidate, str)
        or not source_candidate
        or source_candidate.strip() != source_candidate
        or source_candidate.startswith(".")
        or Path(source_candidate).is_absolute()
        or Path(source_candidate).name != source_candidate
        or "/" in source_candidate
        or "\\" in source_candidate
        or any(ord(character) < 32 for character in source_candidate)
    ):
        raise _error(
            "HANDOFF_SOURCE_INVALID",
            "source candidate is not one bounded direct-child name",
        )
    return source_candidate


def _source_containment(
    paths: InspectorPaths,
    branch_paths: BranchHandoffPaths,
    candidate: Path,
) -> None:
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(paths.intake_root)
    except (FileNotFoundError, ValueError) as error:
        raise _error(
            "HANDOFF_SOURCE_INVALID",
            "source candidate escaped MODEL-TEST",
        ) from error
    protected = (
        branch_paths.branch_root,
        paths.inspector_root.parent / "model-api-native",
    )
    for root in protected:
        try:
            resolved.relative_to(root)
        except ValueError:
            continue
        raise _error(
            "HANDOFF_SOURCE_INVALID",
            "source candidate is inside a serving branch",
        )
    casefolded_parts = {part.casefold() for part in resolved.parts}
    if "openclaw" in casefolded_parts or ".openclaw" in casefolded_parts:
        raise _error(
            "HANDOFF_SOURCE_INVALID",
            "source candidate is inside the agent workspace",
        )
    if len(resolved.parts) > 1 and resolved.parts[1].casefold() == "mnt":
        raise _error(
            "HANDOFF_SOURCE_INVALID",
            "source candidate is on a mounted foreign filesystem",
        )


def _reject_managed_inode(
    branch_paths: BranchHandoffPaths, source_details: os.stat_result
) -> None:
    for child in branch_paths.managed_root.iterdir():
        try:
            details = child.lstat()
        except FileNotFoundError:
            continue
        if (
            stat.S_ISREG(details.st_mode)
            and details.st_dev == source_details.st_dev
            and details.st_ino == source_details.st_ino
        ):
            raise _error(
                "HANDOFF_SOURCE_HARDLINK_REJECTED",
                "source shares an inode with a managed GGUF",
            )


def revalidate_handoff_source(
    paths: InspectorPaths,
    branch_paths: BranchHandoffPaths,
    authorization: DecisionAuthorization,
    source_candidate: str,
    *,
    read_observer: Callable[[int, Path], None] | None = None,
) -> SourceEvidence:
    """Compute a fresh complete source identity without writing the branch."""

    name = _validate_source_name(source_candidate)
    if name != authorization.decision["inspection"]["source_target_name"]:
        raise _error(
            "HANDOFF_DECISION_STALE",
            "source name differs from the authenticated inspection",
        )
    intake_identity = _intake_root_identity(paths.intake_root)
    source = paths.intake_root / name
    try:
        details = source.lstat()
    except FileNotFoundError as error:
        raise _error(
            "HANDOFF_SOURCE_NOT_FOUND", "source candidate is absent"
        ) from error
    if stat.S_ISLNK(details.st_mode):
        raise _error(
            "HANDOFF_SOURCE_SYMLINK", "source candidate is a symlink"
        )
    if not stat.S_ISREG(details.st_mode):
        raise _error(
            "HANDOFF_SOURCE_INVALID", "source candidate is not a regular file"
        )
    if details.st_nlink != 1:
        raise _error(
            "HANDOFF_SOURCE_HARDLINK_REJECTED",
            "source candidate link count is not one",
        )
    _source_containment(paths, branch_paths, source)
    _reject_managed_inode(branch_paths, details)
    expected_size = authorization.inspection["artifact"]["byte_count"]
    if details.st_size != expected_size:
        raise _error(
            "HANDOFF_SOURCE_IDENTITY_MISMATCH",
            "source size differs from the authenticated inspection",
        )
    before = _stat_snapshot(details)
    try:
        descriptor = os.open(
            source, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
        )
    except FileNotFoundError as error:
        raise _error(
            "HANDOFF_SOURCE_NOT_FOUND", "source disappeared before hashing"
        ) from error
    except OSError as error:
        raise _error(
            "HANDOFF_SOURCE_INVALID", "source cannot be opened safely"
        ) from error
    digest = hashlib.sha256()
    byte_count = 0
    try:
        opened = os.fstat(descriptor)
        if _stat_snapshot(opened) != before or not stat.S_ISREG(
            opened.st_mode
        ):
            raise _error(
                "HANDOFF_SOURCE_CHANGED",
                "source identity changed while it was opened",
            )
        while True:
            try:
                chunk = os.read(descriptor, SOURCE_HASH_CHUNK_BYTES)
            except OSError as error:
                raise _error(
                    "HANDOFF_SOURCE_INVALID", "source read failed"
                ) from error
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
            if byte_count > expected_size:
                raise _error(
                    "HANDOFF_SOURCE_CHANGED",
                    "source exceeded its authenticated size",
                )
            if read_observer is not None:
                read_observer(byte_count, source)
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        path_after = source.lstat()
    except FileNotFoundError as error:
        raise _error(
            "HANDOFF_SOURCE_CHANGED", "source disappeared during hashing"
        ) from error
    after = _stat_snapshot(path_after)
    if (
        before != _stat_snapshot(opened_after)
        or before != after
        or byte_count != expected_size
    ):
        raise _error(
            "HANDOFF_SOURCE_CHANGED",
            "source physical identity changed during complete hashing",
        )
    identity = "sha256:" + digest.hexdigest()
    expected_identity = authorization.inspection["artifact"]["identity"]
    if (
        identity != expected_identity
        or identity
        != authorization.decision["inspection"]["artifact_identity"]
    ):
        raise _error(
            "HANDOFF_SOURCE_IDENTITY_MISMATCH",
            "source content identity differs from decision evidence",
        )
    snapshot_identity = _snapshot_identity(before)
    return SourceEvidence(
        path=source,
        intake_root_identity=intake_identity,
        relative_name=name,
        snapshot=before,
        snapshot_identity=snapshot_identity,
        artifact_identity=identity,
    )


def _managed_policy(
    branch_paths: BranchHandoffPaths,
) -> ManagedPolicy:
    observed: list[tuple[str, int, int, int]] = []
    for child in sorted(branch_paths.managed_root.iterdir()):
        if child.suffix.casefold() != ".gguf":
            continue
        try:
            details = child.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
            raise _error(
                "HANDOFF_STAGING_INVALID",
                "managed GGUF policy contains an unsafe physical entry",
            )
        if details.st_nlink != 1:
            raise _error(
                "HANDOFF_STAGING_INVALID",
                "managed GGUF policy contains a hard-linked entry",
            )
        observed.append(
            (
                child.name,
                stat.S_IMODE(details.st_mode),
                details.st_uid,
                details.st_gid,
            )
        )
    if not observed:
        raise _error(
            "HANDOFF_STAGING_INVALID",
            "managed GGUF policy has no accepted reference artifact",
        )
    policies = {(mode, uid, gid) for _, mode, uid, gid in observed}
    if len(policies) != 1:
        raise _error(
            "HANDOFF_STAGING_INVALID",
            "managed GGUF ownership or mode policy is contradictory",
        )
    mode, uid, gid = next(iter(policies))
    if mode not in {0o640, 0o644}:
        raise _error(
            "HANDOFF_STAGING_INVALID",
            "managed GGUF mode policy is not an accepted regular-file mode",
        )
    return ManagedPolicy(
        mode=mode,
        owner_uid=uid,
        owner_gid=gid,
        reference_names=tuple(item[0] for item in observed),
    )


def _validate_managed_name(
    managed_name: str, artifact_identity: str
) -> str:
    if (
        not isinstance(managed_name, str)
        or not managed_name
        or managed_name.strip() != managed_name
        or len(os.fsencode(managed_name)) > 255
        or managed_name.startswith(".")
        or Path(managed_name).is_absolute()
        or Path(managed_name).name != managed_name
        or "/" in managed_name
        or "\\" in managed_name
        or not managed_name.endswith(".gguf")
        or any(ord(character) < 32 for character in managed_name)
    ):
        raise _error(
            "HANDOFF_TARGET_NAME_INVALID",
            "managed target is not one bounded GGUF basename",
        )
    folded = managed_name.casefold()
    if any(token in folded for token in ("mini", "macro", "x.e", "openclaw")):
        raise _error(
            "HANDOFF_TARGET_NAME_INVALID",
            "managed target contains a prohibited control label",
        )
    if re.fullmatch(r"[a-z0-9][a-z0-9._-]*\.gguf", managed_name) is None:
        raise _error(
            "HANDOFF_TARGET_NAME_INVALID",
            "managed target is not a neutral lowercase basename",
        )
    suffix = re.search(r"-([0-9a-f]{12,16})\.gguf\Z", managed_name)
    digest = _sha(artifact_identity, "managed target artifact").removeprefix(
        "sha256:"
    )
    if suffix is None or not digest.startswith(suffix.group(1)):
        raise _error(
            "HANDOFF_TARGET_NAME_INVALID",
            "managed target lacks its bounded artifact-hash suffix",
        )
    return managed_name


def _entry_absent(path: Path, reason: str, label: str) -> None:
    try:
        path.lstat()
    except FileNotFoundError:
        return
    raise _error(reason, f"{label} already exists")


def prepare_handoff_destination(
    branch_paths: BranchHandoffPaths,
    *,
    transaction_id: str,
    managed_name: str,
    artifact_identity: str,
    historical_registry_locations: Iterable[str] = (),
) -> DestinationPlan:
    """Validate a collision-free destination without querying a registry."""

    name = _validate_managed_name(managed_name, artifact_identity)
    if (
        not isinstance(transaction_id, str)
        or re.fullmatch(r"[A-Za-z0-9._-]{1,160}", transaction_id) is None
    ):
        raise _error(
            "HANDOFF_STAGING_INVALID",
            "handoff transaction identity is unsafe",
        )
    managed_target = branch_paths.managed_root / name
    if managed_target.parent != branch_paths.managed_root:
        raise _error(
            "HANDOFF_TARGET_NAME_INVALID", "managed target escaped its root"
        )
    _entry_absent(
        managed_target, "HANDOFF_TARGET_COLLISION", "managed target"
    )
    managed_relative = branch_paths.relative_to_branch(managed_target)
    normalized_locations = {
        Path(value).as_posix()
        for value in historical_registry_locations
        if isinstance(value, str) and value
    }
    if name in normalized_locations or managed_relative in normalized_locations:
        raise _error(
            "HANDOFF_REGISTRY_LOCATION_COLLISION",
            "managed target collides with a historical registry location",
        )
    name_identity = hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]
    staging_name = (
        f".{transaction_id}.{name_identity}.partial-staging.gguf"
    )
    staging_path = branch_paths.branch_staging_root / staging_name
    if staging_path.parent != branch_paths.branch_staging_root:
        raise _error(
            "HANDOFF_STAGING_INVALID", "staging path escaped its root"
        )
    _entry_absent(
        staging_path, "HANDOFF_STAGING_COLLISION", "transaction staging path"
    )
    policy = _managed_policy(branch_paths)
    root_details = branch_paths.managed_root.lstat()
    return DestinationPlan(
        branch_paths=branch_paths,
        transaction_id=transaction_id,
        managed_name=name,
        managed_relative_path=managed_relative,
        managed_target=managed_target,
        staging_name=staging_name,
        staging_relative_path=branch_paths.relative_to_branch(staging_path),
        staging_path=staging_path,
        policy=policy,
        managed_root_identity=(
            root_details.st_dev,
            root_details.st_ino,
            stat.S_IMODE(root_details.st_mode),
        ),
    )


def storage_preflight(
    staging_root: Path,
    artifact_size: int,
    *,
    safety_margin_bytes: int | None = None,
    statvfs_reader: Callable[[Path], os.statvfs_result] = os.statvfs,
) -> dict[str, int]:
    if (
        not isinstance(artifact_size, int)
        or isinstance(artifact_size, bool)
        or artifact_size < 1
    ):
        raise _error(
            "HANDOFF_INSUFFICIENT_STORAGE", "artifact size is invalid"
        )
    margin = (
        safety_margin_bytes
        if safety_margin_bytes is not None
        else max(
            DEFAULT_SAFETY_MARGIN_BYTES,
            min(1024 * 1024 * 1024, artifact_size // 10),
        )
    )
    if (
        not isinstance(margin, int)
        or isinstance(margin, bool)
        or margin < 0
    ):
        raise _error(
            "HANDOFF_INSUFFICIENT_STORAGE", "storage margin is invalid"
        )
    try:
        observed = statvfs_reader(staging_root)
    except OSError as error:
        raise _error(
            "HANDOFF_INSUFFICIENT_STORAGE",
            "destination free space cannot be observed",
        ) from error
    free_bytes = observed.f_bavail * observed.f_frsize
    required = artifact_size + margin
    if free_bytes < required:
        raise _error(
            "HANDOFF_INSUFFICIENT_STORAGE",
            "destination free space is below artifact plus safety margin",
        )
    return {
        "artifact_size": artifact_size,
        "safety_margin_bytes": margin,
        "free_bytes": free_bytes,
        "required_bytes": required,
    }


def _hash_descriptor(
    descriptor: int, *, chunk_size: int = SOURCE_HASH_CHUNK_BYTES
) -> tuple[str, int]:
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    count = 0
    while True:
        chunk = os.read(descriptor, chunk_size)
        if not chunk:
            break
        digest.update(chunk)
        count += len(chunk)
    return "sha256:" + digest.hexdigest(), count


def _write_all(
    descriptor: int,
    data: bytes,
    writer: Callable[[int, bytes], int],
) -> None:
    view = memoryview(data)
    offset = 0
    while offset < len(view):
        try:
            written = writer(descriptor, view[offset:])
        except OSError as error:
            raise _error(
                "HANDOFF_COPY_FAILED", "staging write failed"
            ) from error
        if (
            not isinstance(written, int)
            or isinstance(written, bool)
            or written <= 0
            or written > len(view) - offset
        ):
            raise _error(
                "HANDOFF_COPY_FAILED",
                "staging writer reported a short or invalid write",
            )
        offset += written


def _try_reflink(source_descriptor: int, staging_descriptor: int) -> bool:
    try:
        fcntl.ioctl(staging_descriptor, FICLONE, source_descriptor)
    except OSError as error:
        if error.errno in {
            errno.EOPNOTSUPP,
            errno.ENOTTY,
            errno.EXDEV,
            errno.EINVAL,
            errno.ENOSYS,
        }:
            return False
        raise _error(
            "HANDOFF_COPY_FAILED", "destination reflink attempt failed"
        ) from error
    return True


def _unlink_owned_staging(
    path: Path, *, device: int, inode: int
) -> None:
    try:
        details = path.lstat()
    except FileNotFoundError:
        return
    if (
        stat.S_ISLNK(details.st_mode)
        or not stat.S_ISREG(details.st_mode)
        or details.st_dev != device
        or details.st_ino != inode
    ):
        raise _error(
            "HANDOFF_STAGING_INVALID",
            "staging ownership became uncertain during cleanup",
        )
    path.unlink()
    fsync_directory(path.parent)


def create_staged_artifact(
    plan: DestinationPlan,
    source: SourceEvidence,
    *,
    safety_margin_bytes: int | None = None,
    reflink_cloner: Callable[[int, int], bool] = _try_reflink,
    writer: Callable[[int, bytes], int] = os.write,
    file_fsyncer: Callable[[int], None] = os.fsync,
    directory_fsyncer: Callable[[Path], None] = fsync_directory,
    staged_hasher: Callable[[int], tuple[str, int]] = _hash_descriptor,
    copy_observer: Callable[[int, Path], None] | None = None,
) -> StagedArtifact:
    """Materialize and independently verify one owned branch staging file."""

    storage_preflight(
        plan.branch_paths.branch_staging_root,
        source.snapshot["size_bytes"],
        safety_margin_bytes=safety_margin_bytes,
    )
    _entry_absent(
        plan.managed_target,
        "HANDOFF_TARGET_COLLISION",
        "managed target",
    )
    _entry_absent(
        plan.staging_path,
        "HANDOFF_STAGING_COLLISION",
        "transaction staging path",
    )
    try:
        source_descriptor = os.open(
            source.path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
        )
    except OSError as error:
        raise _error(
            "HANDOFF_SOURCE_INVALID",
            "source cannot be reopened safely for transfer",
        ) from error
    staging_descriptor: int | None = None
    staging_device: int | None = None
    staging_inode: int | None = None
    try:
        opened_source = os.fstat(source_descriptor)
        if _stat_snapshot(opened_source) != source.snapshot:
            raise _error(
                "HANDOFF_SOURCE_CHANGED",
                "source changed after authorization and before transfer",
            )
        try:
            staging_descriptor = os.open(
                plan.staging_path,
                os.O_RDWR
                | os.O_CREAT
                | os.O_EXCL
                | os.O_CLOEXEC
                | os.O_NOFOLLOW,
                0o600,
            )
        except FileExistsError as error:
            raise _error(
                "HANDOFF_STAGING_COLLISION",
                "transaction staging path collided",
            ) from error
        except OSError as error:
            raise _error(
                "HANDOFF_STAGING_INVALID",
                "transaction staging file cannot be created",
            ) from error
        created = os.fstat(staging_descriptor)
        staging_device = created.st_dev
        staging_inode = created.st_ino
        if (
            not stat.S_ISREG(created.st_mode)
            or created.st_nlink != 1
            or created.st_ino == opened_source.st_ino
            and created.st_dev == opened_source.st_dev
        ):
            raise _error(
                "HANDOFF_STAGING_INVALID",
                "transaction staging file has an unsafe physical identity",
            )
        transfer_method = "bounded_stream_copy"
        cloned = reflink_cloner(source_descriptor, staging_descriptor)
        source_digest = hashlib.sha256()
        source_count = 0
        if cloned:
            transfer_method = "reflink_clone"
            source_identity, source_count = _hash_descriptor(
                source_descriptor
            )
        else:
            os.ftruncate(staging_descriptor, 0)
            os.lseek(source_descriptor, 0, os.SEEK_SET)
            os.lseek(staging_descriptor, 0, os.SEEK_SET)
            while True:
                try:
                    chunk = os.read(
                        source_descriptor, SOURCE_HASH_CHUNK_BYTES
                    )
                except OSError as error:
                    raise _error(
                        "HANDOFF_COPY_FAILED", "source read failed during copy"
                    ) from error
                if not chunk:
                    break
                source_digest.update(chunk)
                source_count += len(chunk)
                if source_count > source.snapshot["size_bytes"]:
                    raise _error(
                        "HANDOFF_SOURCE_CHANGED",
                        "source exceeded its authorized size during copy",
                    )
                _write_all(staging_descriptor, chunk, writer)
                if copy_observer is not None:
                    copy_observer(source_count, source.path)
            source_identity = "sha256:" + source_digest.hexdigest()
        os.fchmod(staging_descriptor, plan.policy.mode)
        if (
            os.fstat(staging_descriptor).st_uid != plan.policy.owner_uid
            or os.fstat(staging_descriptor).st_gid != plan.policy.owner_gid
        ):
            try:
                os.fchown(
                    staging_descriptor,
                    plan.policy.owner_uid,
                    plan.policy.owner_gid,
                )
            except OSError as error:
                raise _error(
                    "HANDOFF_STAGING_INVALID",
                    "staging ownership cannot match managed policy",
                ) from error
        try:
            file_fsyncer(staging_descriptor)
            directory_fsyncer(plan.branch_paths.branch_staging_root)
        except OSError as error:
            raise _error(
                "HANDOFF_COPY_FAILED", "staging durability operation failed"
            ) from error
        source_after_descriptor = os.fstat(source_descriptor)
        try:
            source_after_path = source.path.lstat()
        except FileNotFoundError as error:
            raise _error(
                "HANDOFF_SOURCE_CHANGED",
                "source disappeared during branch staging",
            ) from error
        if (
            _stat_snapshot(source_after_descriptor) != source.snapshot
            or _stat_snapshot(source_after_path) != source.snapshot
            or source_count != source.snapshot["size_bytes"]
        ):
            raise _error(
                "HANDOFF_SOURCE_CHANGED",
                "source changed during branch staging",
            )
        if source_identity != source.artifact_identity:
            raise _error(
                "HANDOFF_SOURCE_IDENTITY_MISMATCH",
                "source transfer hash differs from decision evidence",
            )
        staged_identity, staged_count = staged_hasher(staging_descriptor)
        staged_details = os.fstat(staging_descriptor)
        if (
            staged_count != source_count
            or staged_identity != source_identity
            or staged_details.st_size != source_count
        ):
            raise _error(
                "HANDOFF_STAGED_IDENTITY_MISMATCH",
                "independent staged identity differs from source",
            )
        if (
            not stat.S_ISREG(staged_details.st_mode)
            or stat.S_IMODE(staged_details.st_mode) != plan.policy.mode
            or staged_details.st_nlink != 1
        ):
            raise _error(
                "HANDOFF_STAGING_INVALID",
                "verified staging file violates managed physical policy",
            )
        return StagedArtifact(
            path=plan.staging_path,
            relative_path=plan.staging_relative_path,
            transfer_method=transfer_method,
            device=staged_details.st_dev,
            inode=staged_details.st_ino,
            mode=stat.S_IMODE(staged_details.st_mode),
            link_count=staged_details.st_nlink,
            size_bytes=staged_count,
            sha256=staged_identity,
            source_snapshot_identity=source.snapshot_identity,
        )
    except Exception:
        if staging_descriptor is not None:
            os.close(staging_descriptor)
            staging_descriptor = None
        if staging_device is not None and staging_inode is not None:
            _unlink_owned_staging(
                plan.staging_path,
                device=staging_device,
                inode=staging_inode,
            )
        raise
    finally:
        if staging_descriptor is not None:
            os.close(staging_descriptor)
        os.close(source_descriptor)


def _rename_no_replace(source: Path, target: Path) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    function = getattr(library, "renameat2", None)
    if function is None:
        raise _error(
            "HANDOFF_PUBLICATION_FAILED",
            "Linux atomic no-overwrite rename is unavailable",
        )
    function.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    function.restype = ctypes.c_int
    result = function(
        AT_FDCWD,
        os.fsencode(source),
        AT_FDCWD,
        os.fsencode(target),
        RENAME_NOREPLACE,
    )
    if result == 0:
        return
    observed_errno = ctypes.get_errno()
    if observed_errno in {errno.EEXIST, errno.ENOTEMPTY}:
        raise _error(
            "HANDOFF_PUBLICATION_CONFLICT",
            "managed target appeared during no-overwrite publication",
        )
    raise _error(
        "HANDOFF_PUBLICATION_FAILED",
        "atomic no-overwrite publication failed",
    )


def _hash_path(path: Path) -> tuple[str, int]:
    try:
        descriptor = os.open(
            path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
        )
    except OSError as error:
        raise _error(
            "HANDOFF_PUBLICATION_FAILED",
            "published artifact cannot be opened safely",
        ) from error
    try:
        return _hash_descriptor(descriptor)
    finally:
        os.close(descriptor)


def publish_staged_artifact(
    plan: DestinationPlan,
    staged: StagedArtifact,
    *,
    before_rename: Callable[[Path], None] | None = None,
    renamer: Callable[[Path, Path], None] = _rename_no_replace,
    directory_fsyncer: Callable[[Path], None] = fsync_directory,
    target_hasher: Callable[[Path], tuple[str, int]] = _hash_path,
) -> PublishedArtifact:
    """Publish verified staging with one same-filesystem no-overwrite rename."""

    try:
        root_details = plan.branch_paths.managed_root.lstat()
    except FileNotFoundError as error:
        raise _error(
            "HANDOFF_PUBLICATION_FAILED", "managed root disappeared"
        ) from error
    root_identity = (
        root_details.st_dev,
        root_details.st_ino,
        stat.S_IMODE(root_details.st_mode),
    )
    if (
        stat.S_ISLNK(root_details.st_mode)
        or not stat.S_ISDIR(root_details.st_mode)
        or root_identity != plan.managed_root_identity
    ):
        raise _error(
            "HANDOFF_PUBLICATION_FAILED",
            "managed parent identity changed before publication",
        )
    try:
        staging_details = plan.staging_path.lstat()
    except FileNotFoundError as error:
        raise _error(
            "HANDOFF_STAGING_INVALID", "verified staging file is absent"
        ) from error
    if (
        stat.S_ISLNK(staging_details.st_mode)
        or not stat.S_ISREG(staging_details.st_mode)
        or staging_details.st_dev != staged.device
        or staging_details.st_ino != staged.inode
        or staging_details.st_size != staged.size_bytes
        or stat.S_IMODE(staging_details.st_mode) != staged.mode
        or staging_details.st_nlink != 1
        or staging_details.st_dev != root_details.st_dev
    ):
        raise _error(
            "HANDOFF_STAGING_INVALID",
            "staging identity changed before publication",
        )
    _entry_absent(
        plan.managed_target,
        "HANDOFF_TARGET_COLLISION",
        "managed target",
    )
    if before_rename is not None:
        before_rename(plan.managed_target)
    renamer(plan.staging_path, plan.managed_target)
    try:
        directory_fsyncer(plan.branch_paths.managed_root)
    except OSError as error:
        raise _error(
            "HANDOFF_PUBLICATION_FAILED",
            "managed directory fsync failed after publication",
            internal=True,
        ) from error
    try:
        plan.staging_path.lstat()
    except FileNotFoundError:
        pass
    else:
        raise _error(
            "HANDOFF_PUBLICATION_FAILED",
            "staging path remained after publication",
            internal=True,
        )
    try:
        target_details = plan.managed_target.lstat()
    except FileNotFoundError as error:
        raise _error(
            "HANDOFF_PUBLICATION_FAILED",
            "managed target is absent after publication",
            internal=True,
        ) from error
    if (
        stat.S_ISLNK(target_details.st_mode)
        or not stat.S_ISREG(target_details.st_mode)
        or target_details.st_dev != staged.device
        or target_details.st_ino != staged.inode
        or target_details.st_size != staged.size_bytes
        or stat.S_IMODE(target_details.st_mode) != staged.mode
        or target_details.st_nlink != 1
    ):
        raise _error(
            "HANDOFF_PUBLICATION_FAILED",
            "published target does not preserve verified staging identity",
            internal=True,
        )
    target_identity, target_count = target_hasher(plan.managed_target)
    if (
        target_identity != staged.sha256
        or target_count != staged.size_bytes
    ):
        raise _error(
            "HANDOFF_PUBLICATION_FAILED",
            "published target content differs from verified staging",
            internal=True,
        )
    return PublishedArtifact(
        path=plan.managed_target,
        relative_path=plan.managed_relative_path,
        device=target_details.st_dev,
        inode=target_details.st_ino,
        mode=stat.S_IMODE(target_details.st_mode),
        link_count=target_details.st_nlink,
        size_bytes=target_count,
        sha256=target_identity,
    )


def _handoff_id() -> str:
    stamp = dt.datetime.now(dt.timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    return f"handoff-{stamp}-{secrets.token_hex(8)}"


def _owner_surface(owner: dict[str, Any]) -> dict[str, Any]:
    return {
        key: owner.get(key)
        for key in (
            "pid",
            "process_start_identity",
            "boot_identity",
            "inspector_root_identity",
        )
    }


def _handoff_transaction_value(
    *,
    transaction_id: str,
    handoff_id: str,
    start_utc: str,
    owner_identity: dict[str, Any],
    source_candidate: str,
    managed_name: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_IDENTITIES["transaction"],
        "transaction_id": transaction_id,
        "operation": "handoff",
        "start_utc": start_utc,
        "finish_utc": None,
        "state": "VALIDATING_HANDOFF",
        "reason_code": "OK",
        "input_target_name": source_candidate,
        "intake_snapshot_identity": None,
        "owner_identity": owner_identity,
        "status_record_identity": None,
        "artifact_identity": None,
        "terminal_class": "GGUF",
        "inspection_result_identity": None,
        "inspection_result_path": None,
        "inspection_id": None,
        "capability_record_identity": None,
        "decision_id": None,
        "decision_result_identity": None,
        "decision_result_path": None,
        "handoff_id": handoff_id,
        "handoff_result_identity": None,
        "handoff_result_path": None,
        "source_candidate": source_candidate,
        "managed_name": managed_name,
        "staging_relative_path": None,
        "managed_relative_path": None,
        "transfer_method": None,
        "commit_phase": None,
        "publication_device": None,
        "publication_inode": None,
        "publication_size": None,
        "publication_sha256": None,
        "publication_mode": None,
        "publication_link_count": None,
        "handoff_record_candidate": None,
    }


def _active_handoff_state(
    paths: InspectorPaths,
    transaction: dict[str, Any],
    *,
    state: str,
    observer: Callable[[str, dict[str, Any]], None] | None,
    **updates: Any,
) -> dict[str, Any]:
    status = _status_value(
        paths,
        state=state,
        reason_code="OK",
        active_transaction_id=transaction["transaction_id"],
        last_transaction_id=None,
    )
    status_identity = _write_status(paths, status, observer)
    changed = {
        **transaction,
        **updates,
        "state": state,
        "reason_code": "OK",
        "status_record_identity": status_identity,
    }
    _write_transaction(paths, changed, observer)
    return changed


def _durable_handoff_phase(
    paths: InspectorPaths,
    transaction: dict[str, Any],
    *,
    phase: str,
    observer: Callable[[str, dict[str, Any]], None] | None,
    **updates: Any,
) -> dict[str, Any]:
    changed = {
        **transaction,
        **updates,
        "state": phase,
        "commit_phase": phase,
        "reason_code": "OK",
    }
    _write_transaction(paths, changed, observer)
    return changed


def _handoff_records(paths: InspectorPaths) -> list[tuple[dict[str, Any], Path]]:
    records: list[tuple[dict[str, Any], Path]] = []
    for path in sorted(paths.handoff_results.iterdir()):
        match = re.fullmatch(
            r"(handoff-[0-9]{8}T[0-9]{12}Z-[0-9a-f]{16})\.json",
            path.name,
        )
        if match is None:
            continue
        record, _ = load_handoff_record(paths, match.group(1))
        records.append((record, path))
    return records


def _production_reference(
    paths: InspectorPaths, policy: ManagedPolicy
) -> str:
    published_names = {
        Path(record["publication"]["managed_relative_path"]).name
        for record, _ in _handoff_records(paths)
    }
    candidates = [
        name for name in policy.reference_names if name not in published_names
    ]
    if len(candidates) != 1:
        raise _error(
            "HANDOFF_STAGING_INVALID",
            "accepted production GGUF reference is ambiguous",
        )
    return f"MODEL/SUPERMODEL/{candidates[0]}"


def _build_handoff_record(
    paths: InspectorPaths,
    *,
    transaction: dict[str, Any],
    authorization: DecisionAuthorization,
    source: SourceEvidence,
    plan: DestinationPlan,
    staged: StagedArtifact,
    published: PublishedArtifact,
    completed_utc: str,
) -> dict[str, Any]:
    decision = authorization.decision
    inspection = authorization.inspection
    capability = authorization.capability_record
    binding = authorization.binding
    production_reference = _production_reference(paths, plan.policy)
    value = {
            "schema_version": SCHEMA_IDENTITIES["handoff_result"],
            "handoff_id": transaction["handoff_id"],
            "transaction_id": transaction["transaction_id"],
            "created_utc": transaction["start_utc"],
            "completed_utc": completed_utc,
            "status": "PUBLISHED_TO_BRANCH",
            "decision": {
                "decision_id": decision["decision_id"],
                "result_identity": decision["result_identity"],
                "decision_basis_identity": decision[
                    "decision_basis_identity"
                ],
                "capability_result": decision["capability"][
                    "capability_result"
                ],
                "selected_branch": decision["selected_branch"],
                "handoff_allowed": decision["handoff_allowed"],
                "spawn_allowed": decision["spawn_allowed"],
            },
            "inspection": {
                "inspection_id": inspection["inspection_id"],
                "result_identity": (
                    authorization.inspection_result_identity
                ),
                "physical_format": inspection["classification"][
                    "terminal_class"
                ],
                "artifact_identity": inspection["artifact"]["identity"],
                "artifact_size": inspection["artifact"]["byte_count"],
            },
            "capability": {
                "record_id": capability["capability_record_id"],
                "record_identity": capability[
                    "capability_record_identity"
                ],
                "binding_identity": binding["binding_identity"],
                "binding_generation": binding["binding_generation"],
                "installed_tuple_verified": True,
            },
            "source": {
                "intake_root_identity": source.intake_root_identity,
                "relative_name": source.relative_name,
                "device": source.snapshot["device"],
                "inode": source.snapshot["inode"],
                "mode": f"{source.snapshot['mode']:04o}",
                "link_count": source.snapshot["link_count"],
                "size_bytes": source.snapshot["size_bytes"],
                "sha256": source.artifact_identity,
                "pre_copy_identity": source.snapshot_identity,
                "post_copy_identity": staged.source_snapshot_identity,
                "unchanged_during_handoff": (
                    source.snapshot_identity
                    == staged.source_snapshot_identity
                ),
            },
            "staging": {
                "branch_relative_path": staged.relative_path,
                "transfer_method": staged.transfer_method,
                "device": staged.device,
                "inode": staged.inode,
                "mode": f"{staged.mode:04o}",
                "link_count": staged.link_count,
                "size_bytes": staged.size_bytes,
                "sha256": staged.sha256,
                "complete_write": True,
                "file_fsync": True,
                "directory_fsync": True,
            },
            "publication": {
                "managed_relative_path": published.relative_path,
                "method": (
                    "same_filesystem_atomic_no_overwrite_rename"
                ),
                "device": published.device,
                "inode": published.inode,
                "mode": f"{published.mode:04o}",
                "link_count": published.link_count,
                "size_bytes": published.size_bytes,
                "sha256": published.sha256,
                "staging_inode_preserved": (
                    published.device == staged.device
                    and published.inode == staged.inode
                ),
                "parent_directory_fsync": True,
                "collision_absent": True,
            },
            "identity_match": {
                "source_equals_decision": (
                    source.artifact_identity
                    == decision["inspection"]["artifact_identity"]
                ),
                "staged_equals_source": (
                    staged.sha256 == source.artifact_identity
                ),
                "published_equals_staged": (
                    published.sha256 == staged.sha256
                    and published.inode == staged.inode
                ),
            },
            "registry_observation": {
                "state_at_handoff_completion": (
                    "DELEGATED_NOT_OBSERVED"
                ),
                "readiness_claimed": False,
                "next_mini_required": True,
            },
            "alias_protection": {
                "default_alias": "default",
                "default_target_before": production_reference,
                "default_target_after": production_reference,
                "unchanged": True,
            },
            "runtime_protection": {
                "service_was_running": True,
                "service_remained_running": True,
                "lifecycle_operation_count": 0,
            },
            "cleanup_ownership": {
                "source": "caller_or_packet_owned",
                "staging": "handoff_transaction_owned",
                "published_artifact": "branch_owned_after_success",
            },
        }
    if authorization.qualification is not None:
        qualification = authorization.qualification
        value["qualification"] = {
            "qualification_id": qualification["qualification_id"],
            "result_identity": qualification["result_identity"],
            "result_class": qualification["result_class"],
            "requested_profile": qualification["requested_profile"],
        }
    return finalize_handoff_record(value)


def _verify_record_target(
    branch_paths: BranchHandoffPaths, record: dict[str, Any]
) -> PublishedArtifact:
    publication = record["publication"]
    expected_relative = publication["managed_relative_path"]
    candidate = branch_paths.branch_root.joinpath(
        *Path(expected_relative).parts
    )
    try:
        candidate.relative_to(branch_paths.managed_root)
    except ValueError as error:
        raise _error(
            "HANDOFF_PUBLICATION_FAILED",
            "recorded managed target escaped its authenticated root",
        ) from error
    if candidate.parent != branch_paths.managed_root:
        raise _error(
            "HANDOFF_PUBLICATION_FAILED",
            "recorded managed target is not a direct managed child",
        )
    try:
        details = candidate.lstat()
    except FileNotFoundError as error:
        raise _error(
            "HANDOFF_PUBLICATION_FAILED",
            "recorded managed target is absent",
        ) from error
    if (
        stat.S_ISLNK(details.st_mode)
        or not stat.S_ISREG(details.st_mode)
        or details.st_dev != publication["device"]
        or details.st_ino != publication["inode"]
        or stat.S_IMODE(details.st_mode) != int(publication["mode"], 8)
        or details.st_nlink != publication["link_count"]
        or details.st_size != publication["size_bytes"]
    ):
        raise _error(
            "HANDOFF_PUBLICATION_FAILED",
            "recorded managed target ownership is uncertain",
        )
    identity, count = _hash_path(candidate)
    if identity != publication["sha256"] or count != publication["size_bytes"]:
        raise _error(
            "HANDOFF_PUBLICATION_FAILED",
            "recorded managed target content identity is uncertain",
        )
    return PublishedArtifact(
        path=candidate,
        relative_path=expected_relative,
        device=details.st_dev,
        inode=details.st_ino,
        mode=stat.S_IMODE(details.st_mode),
        link_count=details.st_nlink,
        size_bytes=count,
        sha256=identity,
    )


def _detect_exact_published_target(
    plan: DestinationPlan, staged: StagedArtifact
) -> PublishedArtifact | None:
    try:
        details = plan.managed_target.lstat()
    except FileNotFoundError:
        return None
    if (
        stat.S_ISLNK(details.st_mode)
        or not stat.S_ISREG(details.st_mode)
        or details.st_dev != staged.device
        or details.st_ino != staged.inode
        or stat.S_IMODE(details.st_mode) != staged.mode
        or details.st_nlink != staged.link_count
        or details.st_size != staged.size_bytes
    ):
        return None
    identity, count = _hash_path(plan.managed_target)
    if identity != staged.sha256 or count != staged.size_bytes:
        return None
    return PublishedArtifact(
        path=plan.managed_target,
        relative_path=plan.managed_relative_path,
        device=details.st_dev,
        inode=details.st_ino,
        mode=stat.S_IMODE(details.st_mode),
        link_count=details.st_nlink,
        size_bytes=count,
        sha256=identity,
    )


def _completed_handoff(
    paths: InspectorPaths,
    branch_paths: BranchHandoffPaths,
    *,
    decision_id: str,
    source_candidate: str,
    managed_name: str,
    authenticator: Callable[
        [InspectorPaths, str], DecisionAuthorization
    ],
) -> tuple[str, dict[str, Any], Path, str] | None:
    expected_relative = f"MODEL/SUPERMODEL/{managed_name}"
    for record, path in _handoff_records(paths):
        same_target = (
            record["publication"]["managed_relative_path"]
            == expected_relative
        )
        same_call = (
            same_target
            and record["decision"]["decision_id"] == decision_id
            and record["source"]["relative_name"] == source_candidate
        )
        if same_target and not same_call:
            raise _error(
                "HANDOFF_TARGET_COLLISION",
                "managed target belongs to a different completed handoff",
            )
        if not same_call:
            continue
        authorization = authenticator(paths, decision_id)
        if (
            authorization.decision["result_identity"]
            != record["decision"]["result_identity"]
            or authorization.decision["inspection"]["artifact_identity"]
            != record["source"]["sha256"]
        ):
            raise _error(
                "HANDOFF_DECISION_STALE",
                "completed handoff no longer matches current decision evidence",
            )
        _verify_record_target(branch_paths, record)
        return (
            record["transaction_id"],
            record,
            path,
            record["result_identity"],
        )
    return None


def _recoverable_handoff(
    paths: InspectorPaths,
    *,
    decision_id: str,
    source_candidate: str,
    managed_name: str,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for path in sorted(paths.transactions.glob("*.json")):
        try:
            details = path.lstat()
        except FileNotFoundError:
            continue
        if (
            stat.S_ISLNK(details.st_mode)
            or not stat.S_ISREG(details.st_mode)
            or stat.S_IMODE(details.st_mode) != 0o600
            or details.st_nlink != 1
        ):
            continue
        try:
            transaction = read_json_record(path)
        except InspectorError:
            continue
        candidate = transaction.get("handoff_record_candidate")
        if (
            transaction.get("operation") != "handoff"
            or transaction.get("decision_id") != decision_id
            or transaction.get("source_candidate") != source_candidate
            or transaction.get("managed_name") != managed_name
            or transaction.get("commit_phase")
            not in {
                "PUBLISHED_TO_MANAGED_ROOT",
                "HANDOFF_RECORD_PUBLISHED",
            }
            or not isinstance(candidate, dict)
        ):
            continue
        try:
            validated = validate_handoff_record(candidate)
        except InspectorError:
            continue
        candidates.append((transaction, validated))
    if len(candidates) > 1:
        raise _error(
            "HANDOFF_PUBLICATION_FAILED",
            "multiple post-publication recovery claims are present",
        )
    return candidates[0] if candidates else None


def _restore_idle_after_handoff(
    paths: InspectorPaths,
    transaction: dict[str, Any],
    *,
    observer: Callable[[str, dict[str, Any]], None] | None,
    reason_code: str,
    completed: bool,
) -> dict[str, Any]:
    transaction_id = transaction["transaction_id"]
    idle = _status_value(
        paths,
        state="IDLE",
        reason_code="OK",
        active_transaction_id=None,
        last_transaction_id=transaction_id,
    )
    idle_identity = _write_status(paths, idle, observer)
    terminal = {
        **transaction,
        "finish_utc": utc_now(),
        "state": "COMPLETED" if completed else "FAILED",
        "reason_code": reason_code,
        "status_record_identity": idle_identity,
    }
    _write_transaction(paths, terminal, observer)
    return terminal


def _record_failed_handoff(
    paths: InspectorPaths,
    transaction: dict[str, Any],
    *,
    reason_code: str,
    observer: Callable[[str, dict[str, Any]], None] | None,
) -> dict[str, Any]:
    failed = _status_value(
        paths,
        state="FAILED",
        reason_code=reason_code,
        active_transaction_id=transaction["transaction_id"],
        last_transaction_id=transaction["transaction_id"],
    )
    failed_identity = _write_status(paths, failed, observer)
    failed_transaction = {
        **transaction,
        "finish_utc": utc_now(),
        "state": "FAILED",
        "reason_code": reason_code,
        "status_record_identity": failed_identity,
    }
    _write_transaction(paths, failed_transaction, observer)
    return _restore_idle_after_handoff(
        paths,
        failed_transaction,
        observer=observer,
        reason_code=reason_code,
        completed=False,
    )


def handoff_transaction(
    paths: InspectorPaths,
    decision_id: str,
    source_candidate: str,
    managed_name: str,
    *,
    qualification_id: str | None = None,
    branch_path_resolver: Callable[
        [InspectorPaths], BranchHandoffPaths
    ] = BranchHandoffPaths.discover,
    authenticator: Callable[
        [InspectorPaths, str], DecisionAuthorization
    ] = authenticate_handoff_decision,
    source_validator: Callable[
        [InspectorPaths, BranchHandoffPaths, DecisionAuthorization, str],
        SourceEvidence,
    ] = revalidate_handoff_source,
    destination_preparer: Callable[..., DestinationPlan] = (
        prepare_handoff_destination
    ),
    stager: Callable[..., StagedArtifact] = create_staged_artifact,
    artifact_publisher: Callable[
        [DestinationPlan, StagedArtifact], PublishedArtifact
    ] = publish_staged_artifact,
    result_publisher: Callable[
        [InspectorPaths, dict[str, Any]], tuple[Path, str]
    ] = publish_handoff_record,
    transaction_id_factory: Callable[[], str] = _transaction_id,
    handoff_id_factory: Callable[[], str] = _handoff_id,
    historical_registry_locations: Iterable[str] = (),
    safety_margin_bytes: int | None = None,
    transition_observer: Callable[
        [str, dict[str, Any]], None
    ] | None = None,
) -> tuple[str, dict[str, Any], Path, str]:
    """Authorize, stage, publish, record, and restore IDLE."""

    branch_paths = branch_path_resolver(paths)
    effective_authenticator = authenticator
    if qualification_id is not None:
        effective_authenticator = lambda current_paths, current_decision: (
            authenticate_qualified_handoff(
                current_paths,
                current_decision,
                qualification_id,
            )
        )
    lock_state = inspect_active_lock(paths.locks / "active.json")
    if lock_state["state"] == "absent":
        completed = _completed_handoff(
            paths,
            branch_paths,
            decision_id=decision_id,
            source_candidate=source_candidate,
            managed_name=managed_name,
            authenticator=effective_authenticator,
        )
        if completed is not None:
            return completed
    recoverable = _recoverable_handoff(
        paths,
        decision_id=decision_id,
        source_candidate=source_candidate,
        managed_name=managed_name,
    )
    transaction_id = (
        recoverable[0]["transaction_id"]
        if recoverable is not None
        else transaction_id_factory()
    )
    handoff_id = (
        recoverable[0]["handoff_id"]
        if recoverable is not None
        else handoff_id_factory()
    )
    lock = TransactionLock(
        paths, transaction_id=transaction_id, operation="handoff"
    )
    try:
        owner = lock.acquire()
    except InspectorError as error:
        error.data = {**error.data, "transaction_id": transaction_id}
        raise
    transaction: dict[str, Any]
    plan: DestinationPlan | None = None
    staged: StagedArtifact | None = None
    authorization: DecisionAuthorization | None = None
    source: SourceEvidence | None = None
    publication_committed = False
    try:
        if recoverable is not None:
            previous, candidate = recoverable
            transaction = {
                **previous,
                "finish_utc": None,
                "owner_identity": _owner_surface(owner),
                "state": "VALIDATING_HANDOFF",
                "reason_code": "OK",
            }
            transaction = _active_handoff_state(
                paths,
                transaction,
                state="VALIDATING_HANDOFF",
                observer=transition_observer,
            )
            authorization = effective_authenticator(paths, decision_id)
            if (
                authorization.decision["result_identity"]
                != candidate["decision"]["result_identity"]
                or authorization.inspection["artifact"]["identity"]
                != candidate["inspection"]["artifact_identity"]
            ):
                raise _error(
                    "HANDOFF_DECISION_STALE",
                    "recovery decision differs from committed publication",
                )
            target = _verify_record_target(branch_paths, candidate)
            publication_committed = True
            try:
                source_validator(
                    paths,
                    branch_paths,
                    authorization,
                    source_candidate,
                )
            except InspectorError as error:
                if error.reason_code != "HANDOFF_SOURCE_NOT_FOUND":
                    raise
            fsync_directory(branch_paths.managed_root)
            transaction = _active_handoff_state(
                paths,
                transaction,
                state="PUBLISHING_ARTIFACT",
                observer=transition_observer,
                publication_device=target.device,
                publication_inode=target.inode,
                publication_size=target.size_bytes,
                publication_sha256=target.sha256,
                publication_mode=f"{target.mode:04o}",
                publication_link_count=target.link_count,
            )
            result_path, result_identity = result_publisher(
                paths, candidate
            )
            transaction = _durable_handoff_phase(
                paths,
                transaction,
                phase="HANDOFF_RECORD_PUBLISHED",
                observer=transition_observer,
                handoff_result_identity=result_identity,
                handoff_result_path=str(result_path),
                handoff_record_candidate=candidate,
            )
            _restore_idle_after_handoff(
                paths,
                transaction,
                observer=transition_observer,
                reason_code="HANDOFF_COMPLETE",
                completed=True,
            )
            return transaction_id, candidate, result_path, result_identity

        start_utc = utc_now()
        transaction = _handoff_transaction_value(
            transaction_id=transaction_id,
            handoff_id=handoff_id,
            start_utc=start_utc,
            owner_identity=_owner_surface(owner),
            source_candidate=source_candidate,
            managed_name=managed_name,
        )
        transaction = _active_handoff_state(
            paths,
            transaction,
            state="VALIDATING_HANDOFF",
            observer=transition_observer,
        )
        authorization = effective_authenticator(paths, decision_id)
        source = source_validator(
            paths,
            branch_paths,
            authorization,
            source_candidate,
        )
        plan = destination_preparer(
            branch_paths,
            transaction_id=transaction_id,
            managed_name=managed_name,
            artifact_identity=source.artifact_identity,
            historical_registry_locations=historical_registry_locations,
        )
        transaction = _durable_handoff_phase(
            paths,
            transaction,
            phase="VALIDATED",
            observer=transition_observer,
            decision_id=authorization.decision["decision_id"],
            decision_result_identity=authorization.decision[
                "result_identity"
            ],
            decision_result_path=str(
                paths.decision_results / f"{decision_id}.json"
            ),
            inspection_id=authorization.inspection["inspection_id"],
            inspection_result_identity=(
                authorization.inspection_result_identity
            ),
            inspection_result_path=str(
                paths.inspection_results
                / f"{authorization.inspection['inspection_id']}.json"
            ),
            capability_record_identity=authorization.capability_record[
                "capability_record_identity"
            ],
            artifact_identity=source.artifact_identity,
            intake_snapshot_identity=source.snapshot_identity,
            staging_relative_path=plan.staging_relative_path,
            managed_relative_path=plan.managed_relative_path,
        )
        transaction = _active_handoff_state(
            paths,
            transaction,
            state="STAGING_ARTIFACT",
            observer=transition_observer,
        )
        staged = stager(
            plan,
            source,
            safety_margin_bytes=safety_margin_bytes,
        )
        transaction = _durable_handoff_phase(
            paths,
            transaction,
            phase="STAGED",
            observer=transition_observer,
            transfer_method=staged.transfer_method,
        )
        transaction = _active_handoff_state(
            paths,
            transaction,
            state="VERIFYING_STAGED_ARTIFACT",
            observer=transition_observer,
        )
        transaction = _durable_handoff_phase(
            paths,
            transaction,
            phase="STAGED_IDENTITY_VERIFIED",
            observer=transition_observer,
        )
        transaction = _active_handoff_state(
            paths,
            transaction,
            state="PUBLISHING_ARTIFACT",
            observer=transition_observer,
        )
        published = artifact_publisher(plan, staged)
        publication_committed = True
        candidate = _build_handoff_record(
            paths,
            transaction=transaction,
            authorization=authorization,
            source=source,
            plan=plan,
            staged=staged,
            published=published,
            completed_utc=utc_now(),
        )
        transaction = _durable_handoff_phase(
            paths,
            transaction,
            phase="PUBLISHED_TO_MANAGED_ROOT",
            observer=transition_observer,
            publication_device=published.device,
            publication_inode=published.inode,
            publication_size=published.size_bytes,
            publication_sha256=published.sha256,
            publication_mode=f"{published.mode:04o}",
            publication_link_count=published.link_count,
            handoff_record_candidate=candidate,
        )
        result_path, result_identity = result_publisher(paths, candidate)
        transaction = _durable_handoff_phase(
            paths,
            transaction,
            phase="HANDOFF_RECORD_PUBLISHED",
            observer=transition_observer,
            handoff_result_identity=result_identity,
            handoff_result_path=str(result_path),
        )
        _restore_idle_after_handoff(
            paths,
            transaction,
            observer=transition_observer,
            reason_code="HANDOFF_COMPLETE",
            completed=True,
        )
        return transaction_id, candidate, result_path, result_identity
    except InspectorError as error:
        if (
            not publication_committed
            and staged is not None
            and plan is not None
        ):
            detected = _detect_exact_published_target(plan, staged)
            if (
                detected is not None
                and authorization is not None
                and source is not None
            ):
                fsync_directory(plan.branch_paths.managed_root)
                candidate = _build_handoff_record(
                    paths,
                    transaction=transaction,
                    authorization=authorization,
                    source=source,
                    plan=plan,
                    staged=staged,
                    published=detected,
                    completed_utc=utc_now(),
                )
                transaction = _durable_handoff_phase(
                    paths,
                    transaction,
                    phase="PUBLISHED_TO_MANAGED_ROOT",
                    observer=transition_observer,
                    publication_device=detected.device,
                    publication_inode=detected.inode,
                    publication_size=detected.size_bytes,
                    publication_sha256=detected.sha256,
                    publication_mode=f"{detected.mode:04o}",
                    publication_link_count=detected.link_count,
                    handoff_record_candidate=candidate,
                )
                publication_committed = True
        if (
            not publication_committed
            and staged is not None
            and plan is not None
        ):
            try:
                _unlink_owned_staging(
                    plan.staging_path,
                    device=staged.device,
                    inode=staged.inode,
                )
            except InspectorError:
                error = _error(
                    "HANDOFF_STAGING_INVALID",
                    "staging cleanup ownership became uncertain",
                )
        _record_failed_handoff(
            paths,
            transaction,
            reason_code=error.reason_code,
            observer=transition_observer,
        )
        error.data = {**error.data, "transaction_id": transaction_id}
        raise error
    except Exception as error:
        if (
            not publication_committed
            and staged is not None
            and plan is not None
        ):
            detected = _detect_exact_published_target(plan, staged)
            if (
                detected is not None
                and authorization is not None
                and source is not None
            ):
                fsync_directory(plan.branch_paths.managed_root)
                candidate = _build_handoff_record(
                    paths,
                    transaction=transaction,
                    authorization=authorization,
                    source=source,
                    plan=plan,
                    staged=staged,
                    published=detected,
                    completed_utc=utc_now(),
                )
                transaction = _durable_handoff_phase(
                    paths,
                    transaction,
                    phase="PUBLISHED_TO_MANAGED_ROOT",
                    observer=transition_observer,
                    publication_device=detected.device,
                    publication_inode=detected.inode,
                    publication_size=detected.size_bytes,
                    publication_sha256=detected.sha256,
                    publication_mode=f"{detected.mode:04o}",
                    publication_link_count=detected.link_count,
                    handoff_record_candidate=candidate,
                )
                publication_committed = True
        if (
            not publication_committed
            and staged is not None
            and plan is not None
        ):
            _unlink_owned_staging(
                plan.staging_path,
                device=staged.device,
                inode=staged.inode,
            )
        reason_code = "HANDOFF_INTERNAL_ERROR"
        _record_failed_handoff(
            paths,
            transaction,
            reason_code=reason_code,
            observer=transition_observer,
        )
        raise _error(
            reason_code,
            "Unexpected Inspector handoff failure",
            internal=True,
        ) from error
    finally:
        lock.release()
