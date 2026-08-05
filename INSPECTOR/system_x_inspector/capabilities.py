"""Deterministic installed-capability records and private bindings."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any, Iterable

from .constants import SCHEMA_IDENTITIES
from .errors import InspectorError
from .paths import InspectorPaths
from .records import (
    atomic_create_json,
    atomic_write_json,
    canonical_json_bytes,
    read_json_record,
)


CAPABILITY_FIELDS = (
    "schema_version",
    "capability_record_id",
    "capability_record_identity",
    "created_utc",
    "branch_identity",
    "supported_physical_format",
    "availability",
    "runtime_engine",
    "installed_tuple",
    "accepted_evidence",
    "supported_evidence",
    "unsupported_primary_artifact_roles",
    "unproven_valid_policy",
    "reason_code",
)
CAPABILITY_IDENTITY_FIELDS = (
    "schema_version",
    "branch_identity",
    "supported_physical_format",
    "availability",
    "runtime_engine",
    "installed_tuple",
    "accepted_evidence",
    "supported_evidence",
    "unsupported_primary_artifact_roles",
    "unproven_valid_policy",
    "reason_code",
)
BINDING_FIELDS = (
    "schema_version",
    "branch_identity",
    "capability_record_id",
    "capability_record_identity",
    "binding_generation",
    "updated_utc",
    "binding_identity",
)
BINDING_IDENTITY_FIELDS = (
    "schema_version",
    "branch_identity",
    "capability_record_id",
    "capability_record_identity",
    "binding_generation",
)
BRANCH_FORMAT = {
    "model-api-gguf": "GGUF",
    "model-api-native": "NATIVE",
}
BRANCH_ENGINE = {
    "model-api-gguf": "llama-server",
    "model-api-native": "vLLM",
}
SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
RECORD_ID_PATTERN = re.compile(r"capability-[0-9a-f]{24}\Z")
RELATIVE_PATH_PATTERN = re.compile(r"(?!/)(?!.*(?:^|/)\.\.(?:/|$)).+\Z")
FORBIDDEN_CAPABILITY_KEYS = frozenset(
    {
        "pid",
        "current_pid",
        "process_start_identity",
        "listener",
        "port",
        "transaction_id",
        "registry_generation",
        "service_readiness",
        "runtime_state",
        "absolute_path",
        "launch_argv",
    }
)
GGUF_SUPPORTED_FIELDS = frozenset(
    {
        "supported_exact_artifact_identities",
        "accepted_format_versions",
        "accepted_architectures",
        "accepted_primary_model_types",
        "accepted_modalities",
        "accepted_tensor_type_evidence",
        "accepted_tokenizer_evidence",
        "accepted_chat_template_evidence",
        "accepted_runtime_capabilities",
        "public_model_id",
        "accepted_capability_manifest_identity",
    }
)
NATIVE_SUPPORTED_FIELDS = frozenset({"supported_artifact_identities"})
INSTALLED_TUPLE_FIELDS = frozenset(
    {
        "source_commit",
        "accepted_tag",
        "clean_worktree_required",
        "components",
        "manifests",
        "platform_registration",
    }
)
COMPONENT_FIELDS = frozenset(
    {"name", "root", "path", "byte_count", "sha256"}
)
MANIFEST_FIELDS = frozenset(
    {"name", "identity", "file_count", "byte_count", "files"}
)
PLATFORM_REGISTRATION_FIELDS = frozenset(
    {
        "schema_version",
        "adapter_identity",
        "registered",
        "enabled",
    }
)
EVIDENCE_FIELDS = frozenset({"basename", "sha256"})


def _fail(reason: str, message: str) -> InspectorError:
    return InspectorError(reason, message)


def _identity(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _exact_fields(
    value: object, expected: Iterable[str], reason: str, label: str
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(expected):
        raise _fail(reason, f"{label} fields are not closed")
    return value


def _sha256(value: object, reason: str, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise _fail(reason, f"{label} is not a SHA-256 identity")
    return value


def _nonempty_string(value: object, reason: str, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise _fail(reason, f"{label} must be a non-empty string")
    return value


def _bounded_string_list(
    value: object, reason: str, label: str
) -> list[str]:
    if (
        not isinstance(value, list)
        or len(value) > 256
        or any(not isinstance(item, str) or not item for item in value)
        or len(set(value)) != len(value)
    ):
        raise _fail(reason, f"{label} must be a bounded unique string list")
    return value


def _reject_forbidden_keys(value: object) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_CAPABILITY_KEYS:
                raise _fail(
                    "CAPABILITY_RECORD_INVALID",
                    f"capability record contains prohibited field: {key}",
                )
            _reject_forbidden_keys(child)
    elif isinstance(value, list):
        for child in value:
            _reject_forbidden_keys(child)


def _validate_locator(value: dict[str, Any], label: str) -> None:
    if value["root"] not in {"branch", "user_config"}:
        raise _fail(
            "CAPABILITY_RECORD_INVALID", f"{label} has an unknown root"
        )
    path = value["path"]
    if (
        not isinstance(path, str)
        or RELATIVE_PATH_PATTERN.fullmatch(path) is None
        or "\\" in path
    ):
        raise _fail(
            "CAPABILITY_RECORD_INVALID", f"{label} path is not relative"
        )
    if (
        not isinstance(value["byte_count"], int)
        or isinstance(value["byte_count"], bool)
        or value["byte_count"] < 0
    ):
        raise _fail(
            "CAPABILITY_RECORD_INVALID", f"{label} byte count is invalid"
        )
    _sha256(
        value["sha256"], "CAPABILITY_RECORD_INVALID", f"{label} SHA-256"
    )


def _validate_component(value: object, label: str) -> dict[str, Any]:
    result = _exact_fields(
        value,
        COMPONENT_FIELDS,
        "CAPABILITY_RECORD_INVALID",
        label,
    )
    _nonempty_string(
        result["name"], "CAPABILITY_RECORD_INVALID", f"{label} name"
    )
    _validate_locator(result, label)
    return result


def _manifest_basis(files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "root": item["root"],
            "path": item["path"],
            "byte_count": item["byte_count"],
            "sha256": item["sha256"],
        }
        for item in sorted(files, key=lambda row: (row["root"], row["path"]))
    ]


def _validate_manifest(value: object, label: str) -> dict[str, Any]:
    result = _exact_fields(
        value,
        MANIFEST_FIELDS,
        "CAPABILITY_RECORD_INVALID",
        label,
    )
    _nonempty_string(
        result["name"], "CAPABILITY_RECORD_INVALID", f"{label} name"
    )
    if not isinstance(result["files"], list) or not result["files"]:
        raise _fail(
            "CAPABILITY_RECORD_INVALID", f"{label} files are invalid"
        )
    files = [
        _validate_component(item, f"{label} file")
        for item in result["files"]
    ]
    locators = [(item["root"], item["path"]) for item in files]
    if len(set(locators)) != len(locators):
        raise _fail(
            "CAPABILITY_RECORD_INVALID", f"{label} has duplicate files"
        )
    if result["file_count"] != len(files):
        raise _fail(
            "CAPABILITY_RECORD_INVALID", f"{label} file count is invalid"
        )
    if result["byte_count"] != sum(item["byte_count"] for item in files):
        raise _fail(
            "CAPABILITY_RECORD_INVALID", f"{label} byte count is invalid"
        )
    expected = _identity(_manifest_basis(files))
    if result["identity"] != expected:
        raise _fail(
            "CAPABILITY_RECORD_INVALID", f"{label} identity is invalid"
        )
    return result


def _validate_installed_tuple(value: object) -> dict[str, Any]:
    result = _exact_fields(
        value,
        INSTALLED_TUPLE_FIELDS,
        "CAPABILITY_RECORD_INVALID",
        "installed tuple",
    )
    commit = _nonempty_string(
        result["source_commit"],
        "CAPABILITY_RECORD_INVALID",
        "source commit",
    )
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise _fail(
            "CAPABILITY_RECORD_INVALID", "source commit is not canonical"
        )
    if result["accepted_tag"] is not None:
        _nonempty_string(
            result["accepted_tag"],
            "CAPABILITY_RECORD_INVALID",
            "accepted tag",
        )
    if result["clean_worktree_required"] is not True:
        raise _fail(
            "CAPABILITY_RECORD_INVALID",
            "clean worktree evidence is required",
        )
    if not isinstance(result["components"], list) or not result["components"]:
        raise _fail(
            "CAPABILITY_RECORD_INVALID", "installed components are invalid"
        )
    components = [
        _validate_component(item, "installed component")
        for item in result["components"]
    ]
    names = [item["name"] for item in components]
    if len(set(names)) != len(names):
        raise _fail(
            "CAPABILITY_RECORD_INVALID", "component names are duplicated"
        )
    if not isinstance(result["manifests"], list) or not result["manifests"]:
        raise _fail(
            "CAPABILITY_RECORD_INVALID", "installed manifests are invalid"
        )
    manifests = [
        _validate_manifest(item, "installed manifest")
        for item in result["manifests"]
    ]
    manifest_names = [item["name"] for item in manifests]
    if len(set(manifest_names)) != len(manifest_names):
        raise _fail(
            "CAPABILITY_RECORD_INVALID", "manifest names are duplicated"
        )
    registration = _exact_fields(
        result["platform_registration"],
        PLATFORM_REGISTRATION_FIELDS,
        "CAPABILITY_RECORD_INVALID",
        "platform registration",
    )
    for key in (
        "schema_version",
        "adapter_identity",
    ):
        _nonempty_string(
            registration[key],
            "CAPABILITY_RECORD_INVALID",
            f"platform registration {key}",
        )
    if registration["registered"] is not True or registration["enabled"] is not True:
        raise _fail(
            "CAPABILITY_RECORD_INVALID",
            "accepted platform registration is not enabled",
        )
    return result


def capability_identity(value: dict[str, Any]) -> str:
    try:
        basis = {key: value[key] for key in CAPABILITY_IDENTITY_FIELDS}
    except KeyError as error:
        raise _fail(
            "CAPABILITY_RECORD_INVALID",
            f"capability identity field is missing: {error.args[0]}",
        ) from error
    return _identity(basis)


def capability_record_id(identity: str) -> str:
    _sha256(identity, "CAPABILITY_RECORD_INVALID", "capability identity")
    return "capability-" + identity.removeprefix("sha256:")[:24]


def build_capability_record(
    *,
    created_utc: str,
    branch_identity: str,
    supported_physical_format: str,
    availability: str,
    runtime_engine: str,
    installed_tuple: dict[str, Any] | None,
    accepted_evidence: list[dict[str, Any]],
    supported_evidence: dict[str, Any],
    unsupported_primary_artifact_roles: list[str],
    unproven_valid_policy: str | None,
    reason_code: str | None,
) -> dict[str, Any]:
    basis = {
        "schema_version": SCHEMA_IDENTITIES["branch_capability"],
        "branch_identity": branch_identity,
        "supported_physical_format": supported_physical_format,
        "availability": availability,
        "runtime_engine": runtime_engine,
        "installed_tuple": installed_tuple,
        "accepted_evidence": accepted_evidence,
        "supported_evidence": supported_evidence,
        "unsupported_primary_artifact_roles": unsupported_primary_artifact_roles,
        "unproven_valid_policy": unproven_valid_policy,
        "reason_code": reason_code,
    }
    identity = _identity(basis)
    record = {
        "schema_version": basis["schema_version"],
        "capability_record_id": capability_record_id(identity),
        "capability_record_identity": identity,
        "created_utc": created_utc,
        **{key: basis[key] for key in CAPABILITY_IDENTITY_FIELDS[1:]},
    }
    validate_capability_record(record)
    return record


def validate_capability_record(value: object) -> dict[str, Any]:
    record = _exact_fields(
        value,
        CAPABILITY_FIELDS,
        "CAPABILITY_RECORD_INVALID",
        "capability record",
    )
    _reject_forbidden_keys(record)
    if record["schema_version"] != SCHEMA_IDENTITIES["branch_capability"]:
        raise _fail(
            "CAPABILITY_RECORD_INVALID",
            "capability schema identity is invalid",
        )
    branch = record["branch_identity"]
    if branch not in BRANCH_FORMAT:
        raise _fail(
            "CAPABILITY_RECORD_INVALID", "branch identity is invalid"
        )
    if record["supported_physical_format"] != BRANCH_FORMAT[branch]:
        raise _fail(
            "CAPABILITY_RECORD_INVALID",
            "branch and physical format do not agree",
        )
    if record["runtime_engine"] != BRANCH_ENGINE[branch]:
        raise _fail(
            "CAPABILITY_RECORD_INVALID",
            "branch and runtime engine do not agree",
        )
    if record["availability"] not in {"AVAILABLE", "UNAVAILABLE"}:
        raise _fail(
            "CAPABILITY_RECORD_INVALID", "availability is invalid"
        )
    _nonempty_string(
        record["created_utc"],
        "CAPABILITY_RECORD_INVALID",
        "created time",
    )
    if (
        not isinstance(record["accepted_evidence"], list)
        or len(record["accepted_evidence"]) > 16
    ):
        raise _fail(
            "CAPABILITY_RECORD_INVALID", "accepted evidence is invalid"
        )
    for evidence in record["accepted_evidence"]:
        row = _exact_fields(
            evidence,
            EVIDENCE_FIELDS,
            "CAPABILITY_RECORD_INVALID",
            "accepted evidence",
        )
        basename = _nonempty_string(
            row["basename"],
            "CAPABILITY_RECORD_INVALID",
            "accepted evidence basename",
        )
        if Path(basename).name != basename:
            raise _fail(
                "CAPABILITY_RECORD_INVALID",
                "accepted evidence must use a basename",
            )
        _sha256(
            row["sha256"],
            "CAPABILITY_RECORD_INVALID",
            "accepted evidence identity",
        )
    roles = _bounded_string_list(
        record["unsupported_primary_artifact_roles"],
        "CAPABILITY_RECORD_INVALID",
        "unsupported roles",
    )
    if branch == "model-api-gguf":
        if record["availability"] != "AVAILABLE":
            raise _fail(
                "CAPABILITY_RECORD_INVALID",
                "GGUF capability must be AVAILABLE in the current capability contract",
            )
        _validate_installed_tuple(record["installed_tuple"])
        supported = _exact_fields(
            record["supported_evidence"],
            GGUF_SUPPORTED_FIELDS,
            "CAPABILITY_RECORD_INVALID",
            "GGUF supported evidence",
        )
        for item in supported["supported_exact_artifact_identities"]:
            _sha256(
                item,
                "CAPABILITY_RECORD_INVALID",
                "supported artifact identity",
            )
        for key in (
            "accepted_architectures",
            "accepted_primary_model_types",
            "accepted_modalities",
            "accepted_tensor_type_evidence",
            "accepted_runtime_capabilities",
        ):
            _bounded_string_list(
                supported[key], "CAPABILITY_RECORD_INVALID", key
            )
        if not isinstance(supported["accepted_format_versions"], list) or any(
            not isinstance(item, int) or isinstance(item, bool) or item < 1
            for item in supported["accepted_format_versions"]
        ):
            raise _fail(
                "CAPABILITY_RECORD_INVALID",
                "accepted GGUF versions are invalid",
            )
        for key in (
            "accepted_tokenizer_evidence",
            "accepted_chat_template_evidence",
        ):
            _sha256(
                supported[key],
                "CAPABILITY_RECORD_INVALID",
                key,
            )
        _nonempty_string(
            supported["public_model_id"],
            "CAPABILITY_RECORD_INVALID",
            "public model ID",
        )
        _sha256(
            supported["accepted_capability_manifest_identity"],
            "CAPABILITY_RECORD_INVALID",
            "accepted capability manifest",
        )
        if roles != ["adapter"]:
            raise _fail(
                "CAPABILITY_RECORD_INVALID",
                "GGUF unsupported role contract is not exact",
            )
        if record["unproven_valid_policy"] != "RUNTIME_SMOKE_REQUIRED":
            raise _fail(
                "CAPABILITY_RECORD_INVALID",
                "GGUF unproven-valid policy is invalid",
            )
        if record["reason_code"] is not None:
            raise _fail(
                "CAPABILITY_RECORD_INVALID",
                "available GGUF record has an unavailable reason",
            )
    else:
        if record["availability"] != "UNAVAILABLE":
            raise _fail(
                "CAPABILITY_RECORD_INVALID",
                "native capability must be UNAVAILABLE in the current capability contract",
            )
        if record["installed_tuple"] is not None:
            raise _fail(
                "CAPABILITY_RECORD_INVALID",
                "native unavailable record has an installed tuple",
            )
        supported = _exact_fields(
            record["supported_evidence"],
            NATIVE_SUPPORTED_FIELDS,
            "CAPABILITY_RECORD_INVALID",
            "native supported evidence",
        )
        if supported["supported_artifact_identities"] != []:
            raise _fail(
                "CAPABILITY_RECORD_INVALID",
                "native supported artifacts must be empty",
            )
        if roles:
            raise _fail(
                "CAPABILITY_RECORD_INVALID",
                "native unavailable record has unsupported-role rules",
            )
        if record["unproven_valid_policy"] is not None:
            raise _fail(
                "CAPABILITY_RECORD_INVALID",
                "native unavailable record has an unproven-valid policy",
            )
        if record["reason_code"] != "NATIVE_BRANCH_ACCEPTANCE_NOT_CLOSED":
            raise _fail(
                "CAPABILITY_RECORD_INVALID",
                "native unavailable reason is invalid",
            )
    expected_identity = capability_identity(record)
    if record["capability_record_identity"] != expected_identity:
        raise _fail(
            "CAPABILITY_RECORD_IDENTITY_MISMATCH",
            "capability record identity does not match its canonical basis",
        )
    if record["capability_record_id"] != capability_record_id(
        expected_identity
    ):
        raise _fail(
            "CAPABILITY_RECORD_INVALID", "capability record ID is invalid"
        )
    if RECORD_ID_PATTERN.fullmatch(record["capability_record_id"]) is None:
        raise _fail(
            "CAPABILITY_RECORD_INVALID",
            "capability record ID is not canonical",
        )
    return record


def binding_identity(value: dict[str, Any]) -> str:
    try:
        basis = {key: value[key] for key in BINDING_IDENTITY_FIELDS}
    except KeyError as error:
        raise _fail(
            "CAPABILITY_BINDING_INVALID",
            f"binding identity field is missing: {error.args[0]}",
        ) from error
    return _identity(basis)


def build_binding(
    record: dict[str, Any], *, binding_generation: int, updated_utc: str
) -> dict[str, Any]:
    validate_capability_record(record)
    basis = {
        "schema_version": SCHEMA_IDENTITIES["capability_binding"],
        "branch_identity": record["branch_identity"],
        "capability_record_id": record["capability_record_id"],
        "capability_record_identity": record["capability_record_identity"],
        "binding_generation": binding_generation,
    }
    value = {
        **basis,
        "updated_utc": updated_utc,
        "binding_identity": _identity(basis),
    }
    validate_binding(value)
    return value


def validate_binding(value: object) -> dict[str, Any]:
    binding = _exact_fields(
        value,
        BINDING_FIELDS,
        "CAPABILITY_BINDING_INVALID",
        "capability binding",
    )
    if binding["schema_version"] != SCHEMA_IDENTITIES["capability_binding"]:
        raise _fail(
            "CAPABILITY_BINDING_INVALID",
            "capability binding schema is invalid",
        )
    if binding["branch_identity"] not in BRANCH_FORMAT:
        raise _fail(
            "CAPABILITY_BINDING_INVALID", "binding branch is invalid"
        )
    if RECORD_ID_PATTERN.fullmatch(
        str(binding["capability_record_id"])
    ) is None:
        raise _fail(
            "CAPABILITY_BINDING_INVALID",
            "binding capability record ID is invalid",
        )
    _sha256(
        binding["capability_record_identity"],
        "CAPABILITY_BINDING_INVALID",
        "binding capability identity",
    )
    if (
        not isinstance(binding["binding_generation"], int)
        or isinstance(binding["binding_generation"], bool)
        or binding["binding_generation"] < 1
    ):
        raise _fail(
            "CAPABILITY_BINDING_INVALID",
            "binding generation is invalid",
        )
    _nonempty_string(
        binding["updated_utc"],
        "CAPABILITY_BINDING_INVALID",
        "binding update time",
    )
    expected = binding_identity(binding)
    if binding["binding_identity"] != expected:
        raise _fail(
            "CAPABILITY_BINDING_INVALID", "binding identity is invalid"
        )
    return binding


def _safe_directory(path: Path, *, mode: int = 0o700) -> None:
    details = path.lstat()
    if (
        stat.S_ISLNK(details.st_mode)
        or not stat.S_ISDIR(details.st_mode)
        or stat.S_IMODE(details.st_mode) != mode
    ):
        raise _fail(
            "CAPABILITY_RECORD_INVALID",
            f"capability path is unsafe: {path.name}",
        )


def initialize_capability_store(paths: InspectorPaths) -> None:
    root = paths.capability_root
    if root.exists() or root.is_symlink():
        _safe_directory(root)
        entries = {item.name for item in root.iterdir()}
        if entries - {"records", "bindings"}:
            raise _fail(
                "CAPABILITY_RECORD_INVALID",
                "unknown capability-root content blocks initialization",
            )
    else:
        os.mkdir(root, 0o700)
    for path in (paths.capability_records, paths.capability_bindings):
        if path.exists() or path.is_symlink():
            _safe_directory(path)
        else:
            os.mkdir(path, 0o700)


def validate_capability_store(paths: InspectorPaths) -> None:
    _safe_directory(paths.capability_root)
    entries = {item.name for item in paths.capability_root.iterdir()}
    if entries != {"records", "bindings"}:
        raise _fail(
            "CAPABILITY_RECORD_INVALID",
            "capability root does not contain exactly records and bindings",
        )
    _safe_directory(paths.capability_records)
    _safe_directory(paths.capability_bindings)


def _safe_record_path(path: Path, parent: Path, reason: str) -> None:
    if path.parent != parent or path.name != Path(path.name).name:
        raise _fail(reason, "record path is outside its private store")
    details = path.lstat()
    if (
        stat.S_ISLNK(details.st_mode)
        or not stat.S_ISREG(details.st_mode)
        or stat.S_IMODE(details.st_mode) != 0o600
        or details.st_nlink != 1
    ):
        raise _fail(reason, "record file has an unsafe physical type")


def capability_record_path(
    paths: InspectorPaths, capability_record_id_value: str
) -> Path:
    if RECORD_ID_PATTERN.fullmatch(capability_record_id_value) is None:
        raise _fail(
            "CAPABILITY_RECORD_INVALID", "capability record ID is unsafe"
        )
    return paths.capability_records / f"{capability_record_id_value}.json"


def publish_capability_record(
    paths: InspectorPaths, value: dict[str, Any]
) -> Path:
    record = validate_capability_record(value)
    initialize_capability_store(paths)
    path = capability_record_path(paths, record["capability_record_id"])
    if path.exists() or path.is_symlink():
        _safe_record_path(
            path, paths.capability_records, "CAPABILITY_RECORD_INVALID"
        )
        if path.read_bytes() == canonical_json_bytes(record):
            return path
        raise _fail(
            "CAPABILITY_RECORD_INVALID",
            "different immutable capability record already exists",
        )
    try:
        atomic_create_json(path, record, mode=0o600)
    except InspectorError as error:
        if path.exists() and not path.is_symlink():
            _safe_record_path(
                path, paths.capability_records, "CAPABILITY_RECORD_INVALID"
            )
            if path.read_bytes() == canonical_json_bytes(record):
                return path
        raise _fail(
            "CAPABILITY_RECORD_INVALID",
            "capability record atomic publication collided",
        ) from error
    _safe_record_path(
        path, paths.capability_records, "CAPABILITY_RECORD_INVALID"
    )
    if read_json_record(path) != record:
        raise _fail(
            "CAPABILITY_RECORD_INVALID",
            "capability record did not round-trip",
        )
    return path


def load_capability_record(
    paths: InspectorPaths, capability_record_id_value: str
) -> dict[str, Any]:
    path = capability_record_path(paths, capability_record_id_value)
    try:
        _safe_record_path(
            path, paths.capability_records, "CAPABILITY_RECORD_NOT_FOUND"
        )
    except FileNotFoundError as error:
        raise _fail(
            "CAPABILITY_RECORD_NOT_FOUND", "capability record is absent"
        ) from error
    return validate_capability_record(read_json_record(path))


def binding_path(paths: InspectorPaths, branch_identity: str) -> Path:
    if branch_identity not in BRANCH_FORMAT:
        raise _fail(
            "CAPABILITY_BINDING_INVALID", "binding branch is unsafe"
        )
    return paths.capability_bindings / f"{branch_identity}.json"


def publish_binding(
    paths: InspectorPaths, value: dict[str, Any]
) -> Path:
    binding = validate_binding(value)
    initialize_capability_store(paths)
    path = binding_path(paths, binding["branch_identity"])
    if path.exists() or path.is_symlink():
        _safe_record_path(
            path, paths.capability_bindings, "CAPABILITY_BINDING_INVALID"
        )
        current = validate_binding(read_json_record(path))
        expected_generation = current["binding_generation"] + 1
        if binding["binding_generation"] != expected_generation:
            raise _fail(
                "CAPABILITY_BINDING_INVALID",
                "binding generation is not the next generation",
            )
    elif binding["binding_generation"] != 1:
        raise _fail(
            "CAPABILITY_BINDING_INVALID",
            "initial binding generation must be one",
        )
    atomic_write_json(path, binding, mode=0o600)
    _safe_record_path(
        path, paths.capability_bindings, "CAPABILITY_BINDING_INVALID"
    )
    if read_json_record(path) != binding:
        raise _fail(
            "CAPABILITY_BINDING_INVALID", "binding did not round-trip"
        )
    return path


def load_binding(
    paths: InspectorPaths, branch_identity: str
) -> dict[str, Any]:
    path = binding_path(paths, branch_identity)
    try:
        _safe_record_path(
            path, paths.capability_bindings, "CAPABILITY_BINDING_NOT_FOUND"
        )
    except FileNotFoundError as error:
        raise _fail(
            "CAPABILITY_BINDING_NOT_FOUND", "capability binding is absent"
        ) from error
    return validate_binding(read_json_record(path))


def _git_commit(branch_root: Path) -> str | None:
    git = branch_root / "llama.cpp" / ".git"
    if not git.is_dir() or git.is_symlink():
        identity_path = branch_root / "LLAMA_CPP_SOURCE_IDENTITY.json"
        try:
            details = identity_path.lstat()
            if (
                stat.S_ISLNK(details.st_mode)
                or not stat.S_ISREG(details.st_mode)
                or details.st_nlink != 1
            ):
                return None
            identity = json.loads(identity_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError, OSError):
            return None
        if (
            not isinstance(identity, dict)
            or identity.get("schema")
            != "system-x.llama-cpp-source-identity.v1"
            or identity.get("origin") != "https://github.com/ggml-org/llama.cpp"
            or identity.get("build_output_excluded") is not True
            or not isinstance(identity.get("tracked_file_count"), int)
            or identity.get("tracked_file_count", 0) <= 0
            or not isinstance(identity.get("complete_vendored_manifest_sha256"), str)
            or re.fullmatch(
                r"[0-9a-f]{64}", identity["complete_vendored_manifest_sha256"]
            ) is None
        ):
            return None
        commit = identity.get("commit")
        return commit if isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit) else None
    head = git / "HEAD"
    try:
        details = head.lstat()
        if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
            return None
        value = head.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, UnicodeDecodeError, OSError):
        return None
    if value.startswith("ref: "):
        reference = value[5:]
        if (
            RELATIVE_PATH_PATTERN.fullmatch(reference) is None
            or "\\" in reference
        ):
            return None
        path = git / reference
        try:
            details = path.lstat()
            if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(
                details.st_mode
            ):
                return None
            value = path.read_text(encoding="utf-8").strip()
        except (FileNotFoundError, UnicodeDecodeError, OSError):
            return None
    return value if re.fullmatch(r"[0-9a-f]{40}", value) else None


def _resolve_locator(
    item: dict[str, Any],
    *,
    branch_root: Path,
    user_config_root: Path,
) -> Path:
    base = branch_root if item["root"] == "branch" else user_config_root
    candidate = base.joinpath(*item["path"].split("/"))
    try:
        candidate.relative_to(base)
    except ValueError as error:
        raise _fail(
            "CAPABILITY_INSTALLED_TUPLE_MISMATCH",
            "installed tuple path escaped its root",
        ) from error
    return candidate


def _observed_component(
    item: dict[str, Any],
    *,
    branch_root: Path,
    user_config_root: Path,
) -> dict[str, Any]:
    path = _resolve_locator(
        item, branch_root=branch_root, user_config_root=user_config_root
    )
    try:
        details = path.lstat()
    except FileNotFoundError as error:
        raise _fail(
            "CAPABILITY_INSTALLED_TUPLE_MISMATCH",
            f"installed tuple file is absent: {item['name']}",
        ) from error
    if (
        stat.S_ISLNK(details.st_mode)
        or not stat.S_ISREG(details.st_mode)
        or details.st_nlink != 1
    ):
        raise _fail(
            "CAPABILITY_INSTALLED_TUPLE_MISMATCH",
            f"installed tuple file is unsafe: {item['name']}",
        )
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
    return {
        "name": item["name"],
        "root": item["root"],
        "path": item["path"],
        "byte_count": byte_count,
        "sha256": "sha256:" + digest.hexdigest(),
    }


def verify_installed_tuple(
    paths: InspectorPaths,
    record: dict[str, Any],
    *,
    branch_root: Path | None = None,
    user_config_root: Path | None = None,
) -> dict[str, Any]:
    value = validate_capability_record(record)
    if value["availability"] != "AVAILABLE":
        return {"applicable": False, "verified": True, "mismatches": []}
    installed = value["installed_tuple"]
    assert isinstance(installed, dict)
    actual_branch_root = (
        branch_root
        if branch_root is not None
        else paths.inspector_root.parent / value["branch_identity"]
    )
    actual_user_config_root = (
        user_config_root
        if user_config_root is not None
        else Path.home() / ".config"
    )
    mismatches: list[dict[str, Any]] = []
    commit = _git_commit(actual_branch_root)
    if commit != installed["source_commit"]:
        mismatches.append(
            {
                "field": "source_commit",
                "expected": installed["source_commit"],
                "observed": commit,
            }
        )
    observed_by_locator: dict[tuple[str, str], dict[str, Any]] = {}
    for component in installed["components"]:
        observed = _observed_component(
            component,
            branch_root=actual_branch_root,
            user_config_root=actual_user_config_root,
        )
        observed_by_locator[(component["root"], component["path"])] = observed
        for field in ("byte_count", "sha256"):
            if observed[field] != component[field]:
                mismatches.append(
                    {
                        "field": f"component:{component['name']}:{field}",
                        "expected": component[field],
                        "observed": observed[field],
                    }
                )
    for manifest in installed["manifests"]:
        observed_files = []
        for expected in manifest["files"]:
            key = (expected["root"], expected["path"])
            observed = observed_by_locator.get(key)
            if observed is None:
                observed = _observed_component(
                    expected,
                    branch_root=actual_branch_root,
                    user_config_root=actual_user_config_root,
                )
                observed_by_locator[key] = observed
            observed_files.append(observed)
            for field in ("byte_count", "sha256"):
                if observed[field] != expected[field]:
                    mismatches.append(
                        {
                            "field": (
                                f"manifest:{manifest['name']}:"
                                f"{expected['path']}:{field}"
                            ),
                            "expected": expected[field],
                            "observed": observed[field],
                        }
                    )
        observed_identity = _identity(_manifest_basis(observed_files))
        if observed_identity != manifest["identity"]:
            mismatches.append(
                {
                    "field": f"manifest:{manifest['name']}:identity",
                    "expected": manifest["identity"],
                    "observed": observed_identity,
                }
            )
    return {
        "applicable": True,
        "verified": not mismatches,
        "mismatches": mismatches,
        "source_commit": commit,
        "component_count": len(installed["components"]),
        "manifest_count": len(installed["manifests"]),
    }


def capability_inventory(paths: InspectorPaths) -> dict[str, Any]:
    validate_capability_store(paths)
    bindings: list[dict[str, Any]] = []
    for path in sorted(paths.capability_bindings.iterdir()):
        if path.suffix != ".json":
            raise _fail(
                "CAPABILITY_BINDING_INVALID",
                "unknown capability binding content is present",
            )
        _safe_record_path(
            path, paths.capability_bindings, "CAPABILITY_BINDING_INVALID"
        )
        binding = validate_binding(read_json_record(path))
        record = load_capability_record(
            paths, binding["capability_record_id"]
        )
        if (
            record["capability_record_identity"]
            != binding["capability_record_identity"]
            or record["branch_identity"] != binding["branch_identity"]
        ):
            raise _fail(
                "CAPABILITY_RECORD_IDENTITY_MISMATCH",
                "binding and capability record do not agree",
            )
        verification = verify_installed_tuple(paths, record)
        bindings.append(
            {
                "branch_identity": binding["branch_identity"],
                "binding_identity": binding["binding_identity"],
                "binding_generation": binding["binding_generation"],
                "capability_record_id": record["capability_record_id"],
                "capability_record_identity": record[
                    "capability_record_identity"
                ],
                "availability": record["availability"],
                "supported_physical_format": record[
                    "supported_physical_format"
                ],
                "record_integrity": "VALID",
                "installed_tuple_verification": verification,
            }
        )
    return {
        "schema_version": "system-x.inspector-capability-inventory.v1",
        "bindings": bindings,
    }
