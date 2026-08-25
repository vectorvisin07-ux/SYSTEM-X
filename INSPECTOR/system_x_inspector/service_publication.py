"""Bounded read-only publication proof for one authenticated GGUF handoff."""

from __future__ import annotations

import datetime as dt
import hashlib
import http.client
import ipaddress
import json
import os
import re
import sqlite3
import stat
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .capabilities import (
    load_binding,
    load_capability_record,
    verify_installed_tuple,
)
from .constants import SCHEMA_IDENTITIES
from .decision import (
    decision_result_identity,
    load_inspection_result,
    validate_decision_record,
)
from .errors import InspectorError
from .handoff import load_handoff_record
from .locking import TransactionLock
from .paths import BranchHandoffPaths, InspectorPaths
from .records import (
    atomic_create_json,
    canonical_json_bytes,
    read_json_record,
)
from .results import utc_now
from .runtime import (
    _status_value,
    _transaction_id,
    _write_status,
    _write_transaction,
)


PUBLICATION_SCHEMA = "system-x.inspector-service-publication.v1"
REGISTRY_SCHEMA = "system-x.gguf-model-registry.v1"
OPERATION_RECORD_SCHEMA = "system-x.operation-record.v1"
PROFILE_SCHEMA = "system-x.service-operating-profile.v1"
DESIRED_SCHEMA = "system-x.service-desired-state.v1"
SUPERVISOR_SCHEMA = "system-x.service-supervisor-status.v1"
HANDOFF_BRANCH = "model-api-gguf"
REQUEST_ID_PATTERN = re.compile(r"sx_req_[0-9a-f]{32}\Z")
KEY_ID_PATTERN = re.compile(r"[0-9a-f]{32}\Z")
PUBLIC_MODEL_PATTERN = re.compile(
    r"sx-gguf-[a-z0-9](?:[a-z0-9-]{0,118}[a-z0-9])?\Z"
)
PUBLICATION_ID_PATTERN = re.compile(
    r"publication-[0-9]{8}T[0-9]{12}Z-[0-9a-f]{16}\Z"
)
SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
RAW_KEY_PATTERN = re.compile(
    r"sxk_v1_(?P<key_id>[0-9a-f]{32})_[A-Za-z0-9_-]{43}\Z"
)
BUNDLE_PATTERN = re.compile(r"bundle-[0-9a-f]{64}\Z")
MAX_JSON_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_PROFILE_BYTES = 64 * 1024
MAX_RECORD_BYTES = 2 * 1024 * 1024
MAX_LOG_SCAN_BYTES = 8 * 1024 * 1024
HTTP_TIMEOUT_SECONDS = 180.0
OPERATION_RECORD_WAIT_SECONDS = 12.0
REGISTRY_CONVERGENCE_WAIT_SECONDS = 45.0
REGISTRY_CONVERGENCE_POLL_SECONDS = 0.25
REGISTRY_CONVERGENCE_TRANSIENT_REASONS = frozenset(
    {
        "SERVICE_NOT_READY",
        "REGISTRY_UNAVAILABLE",
        "REGISTRY_SCHEMA_UNSUPPORTED",
        "REGISTRY_LOCATION_NOT_FOUND",
        "REGISTRY_LOCATION_NOT_READY",
        "REGISTRY_MODEL_VERSION_NOT_FOUND",
        "REGISTRY_MODEL_VERSION_NOT_READY",
        "CAPABILITY_MANIFEST_NOT_FOUND",
    }
)
HASH_CHUNK_BYTES = 8 * 1024 * 1024

TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "publication_id",
        "transaction_id",
        "created_utc",
        "publication_basis_identity",
        "result_identity",
        "handoff",
        "registry",
        "public_service",
        "request",
        "correlation",
        "restoration",
        "security",
    }
)
SECTION_FIELDS = {
    "handoff": frozenset(
        {
            "handoff_id",
            "handoff_result_identity",
            "decision_id",
            "decision_result_identity",
            "inspection_id",
            "inspection_result_identity",
            "artifact_identity",
            "selected_branch",
        }
    ),
    "registry": frozenset(
        {
            "schema_identity",
            "schema_version",
            "generation",
            "managed_location_present",
            "managed_location_state",
            "bundle_id",
            "model_version_id",
            "public_identity_mode",
            "capability_manifest_identity",
            "progression_evidence",
        }
    ),
    "public_service": frozenset(
        {
            "profile_identity",
            "base_url",
            "readiness_state",
            "health_http_status",
            "model_list_http_status",
            "model_detail_http_status",
            "public_model_id",
            "aliases",
        }
    ),
    "request": frozenset(
        {
            "request_id",
            "http_status",
            "operation_state",
            "finish_reason",
            "response_model_id",
            "content_nonempty",
            "final_content_bytes",
            "final_content_sha256",
            "input_tokens",
            "output_tokens",
            "operation_record_correlated",
        }
    ),
    "correlation": frozenset(
        {
            "inspector_transaction_id",
            "request_id",
            "operation_record_schema",
            "api_service_transaction_id",
            "router_transaction_id",
            "artifact_version_id",
            "non_secret_key_id",
        }
    ),
    "restoration": frozenset(
        {
            "default_alias",
            "default_target_before",
            "default_target_after",
            "restoration_required",
            "restoration_performed",
            "final_warm_target",
            "final_warm_health",
            "final_public_health",
        }
    ),
    "security": frozenset(
        {
            "outbound_provider_request",
            "raw_key_persisted",
            "private_endpoint_published",
            "physical_model_path_published",
            "prompt_persisted",
            "generated_content_persisted",
        }
    ),
}
FORBIDDEN_RECORD_KEYS = frozenset(
    {
        "api_key",
        "raw_key",
        "authorization",
        "x_api_key",
        "credential_verifier",
        "credential_pepper",
        "private_endpoint",
        "physical_path",
        "managed_relative_path",
        "prompt",
        "content",
        "reasoning",
        "pid",
        "process_start_identity",
    }
)
REGISTRY_TABLE_COLUMNS = {
    "registry_metadata": {"key", "value"},
    "artifact_locations": {
        "relative_root",
        "current_bundle_id",
        "present",
        "physical_manifest_json",
        "first_seen_utc",
        "last_seen_utc",
    },
    "model_versions": {
        "model_version_id",
        "bundle_id",
        "router_model_id",
        "router_source",
        "display_name",
        "state",
        "router_metadata_json",
        "router_metadata_sha256",
        "created_utc",
        "updated_utc",
    },
    "model_version_locations": {
        "model_version_id",
        "relative_root",
        "created_utc",
        "updated_utc",
    },
    "capability_manifests": {
        "model_version_id",
        "manifest_json",
        "manifest_sha256",
        "props_payload_sha256",
        "observed_utc",
    },
    "registry_events": {
        "event_id",
        "generation",
        "event_type",
        "subject_id",
        "detail_json",
        "created_utc",
    },
    "aliases": {
        "alias",
        "model_version_id",
        "alias_kind",
        "created_utc",
        "updated_utc",
    },
}


def _fail(reason: str, message: str, *, internal: bool = False) -> InspectorError:
    return InspectorError(
        reason,
        message,
        exit_status=70 if internal else 2,
    )


def _identity(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _publication_id() -> str:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"publication-{stamp}-{os.urandom(8).hex()}"


def _closed(
    value: object, fields: Iterable[str], reason: str, label: str
) -> dict[str, Any]:
    expected = set(fields)
    if not isinstance(value, dict) or set(value) != expected:
        raise _fail(reason, f"{label} fields are not closed")
    return value


def _private_file_bytes(
    path: Path,
    *,
    reason: str,
    maximum: int = MAX_RECORD_BYTES,
    required_mode: int | None = None,
) -> bytes:
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
    except FileNotFoundError as error:
        raise _fail(reason, "required local evidence is absent") from error
    except OSError as error:
        raise _fail(reason, "required local evidence cannot be opened safely") from error
    try:
        details = os.fstat(descriptor)
        if (
            not stat.S_ISREG(details.st_mode)
            or details.st_nlink != 1
            or (
                required_mode is not None
                and stat.S_IMODE(details.st_mode) != required_mode
            )
            or details.st_size > maximum
        ):
            raise _fail(reason, "required local evidence is physically unsafe")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum:
                raise _fail(reason, "required local evidence exceeds its bound")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _json_file(
    path: Path,
    *,
    reason: str,
    maximum: int = MAX_RECORD_BYTES,
    required_mode: int | None = None,
) -> dict[str, Any]:
    raw = _private_file_bytes(
        path,
        reason=reason,
        maximum=maximum,
        required_mode=required_mode,
    )
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _fail(reason, "required local JSON evidence is invalid") from error
    if not isinstance(value, dict):
        raise _fail(reason, "required local JSON evidence is not an object")
    return value


def _hash_regular_file(
    path: Path, expected: dict[str, Any]
) -> tuple[str, dict[str, int]]:
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
    except FileNotFoundError as error:
        raise _fail("HANDOFF_TARGET_MISSING", "managed GGUF target is absent") from error
    except OSError as error:
        raise _fail("HANDOFF_TARGET_UNSAFE", "managed GGUF cannot be opened safely") from error
    try:
        before = os.fstat(descriptor)
        snapshot = {
            "device": before.st_dev,
            "inode": before.st_ino,
            "mode": stat.S_IMODE(before.st_mode),
            "link_count": before.st_nlink,
            "size_bytes": before.st_size,
            "mtime_ns": before.st_mtime_ns,
        }
        expected_mode = int(str(expected["mode"]), 8)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or snapshot["device"] != expected["device"]
            or snapshot["inode"] != expected["inode"]
            or snapshot["mode"] != expected_mode
            or snapshot["size_bytes"] != expected["size_bytes"]
        ):
            raise _fail(
                "HANDOFF_TARGET_CHANGED",
                "managed GGUF physical identity changed after handoff",
            )
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(descriptor, HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
        after = os.fstat(descriptor)
        if (
            total != before.st_size
            or after.st_dev != before.st_dev
            or after.st_ino != before.st_ino
            or after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
        ):
            raise _fail(
                "HANDOFF_TARGET_CHANGED",
                "managed GGUF changed while it was authenticated",
            )
        identity = "sha256:" + digest.hexdigest()
        if identity != expected["sha256"]:
            raise _fail(
                "HANDOFF_TARGET_CHANGED",
                "managed GGUF content does not match the handoff",
            )
        return identity, snapshot
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class HandoffEvidence:
    record: dict[str, Any]
    result_identity: str
    inspection_result_identity: str
    decision_result_identity: str
    capability_record_identity: str
    capability_binding_identity: str
    target_identity: str
    target_name: str


def _handoff_authorization_mode(handoff: dict[str, Any]) -> str | None:
    decision = handoff.get("decision")
    qualification = handoff.get("qualification")
    if not isinstance(decision, dict):
        return None
    direct = (
        qualification is None
        and decision.get("capability_result") == "SUPPORTED"
        and decision.get("selected_branch") == HANDOFF_BRANCH
        and decision.get("handoff_allowed") is True
        and decision.get("spawn_allowed") is True
    )
    qualified = (
        isinstance(qualification, dict)
        and qualification.get("result_class")
        == "SUPPORTED_FOR_CURRENT_TUPLE"
        and isinstance(qualification.get("requested_profile"), str)
        and bool(qualification["requested_profile"])
        and decision.get("capability_result")
        == "RUNTIME_SMOKE_REQUIRED"
        and decision.get("selected_branch") is None
        and decision.get("handoff_allowed") is False
        and decision.get("spawn_allowed") is False
    )
    if direct:
        return "DIRECT_SUPPORTED"
    if qualified:
        return "QUALIFIED_RUNTIME"
    return None


def authenticate_handoff(
    paths: InspectorPaths, handoff_id: str
) -> HandoffEvidence:
    try:
        handoff, handoff_identity = load_handoff_record(paths, handoff_id)
    except InspectorError as error:
        reason = (
            "HANDOFF_RESULT_NOT_FOUND"
            if not (paths.handoff_results / f"{handoff_id}.json").exists()
            else "HANDOFF_RESULT_INVALID"
        )
        raise _fail(reason, "retained handoff could not be authenticated") from error
    if handoff["result_identity"] != handoff_identity:
        raise _fail(
            "HANDOFF_RESULT_IDENTITY_MISMATCH",
            "retained handoff identity is inconsistent",
        )
    authorization_mode = _handoff_authorization_mode(handoff)
    if (
        authorization_mode is None
        or handoff["status"] != "PUBLISHED_TO_BRANCH"
    ):
        raise _fail(
            "HANDOFF_BRANCH_INVALID",
            "retained handoff does not authorize the GGUF branch",
        )
    try:
        inspection, inspection_identity = load_inspection_result(
            paths, handoff["inspection"]["inspection_id"]
        )
    except InspectorError as error:
        raise _fail(
            "HANDOFF_RESULT_INVALID",
            "linked inspection could not be authenticated",
        ) from error
    if (
        inspection_identity != handoff["inspection"]["result_identity"]
        or inspection["artifact"]["identity"]
        != handoff["inspection"]["artifact_identity"]
    ):
        raise _fail(
            "HANDOFF_RESULT_IDENTITY_MISMATCH",
            "linked inspection does not match the retained handoff",
        )
    decision_path = (
        paths.decision_results
        / f"{handoff['decision']['decision_id']}.json"
    )
    try:
        decision_raw = _json_file(
            decision_path,
            reason="HANDOFF_RESULT_INVALID",
            required_mode=0o600,
        )
        decision = validate_decision_record(decision_raw)
    except InspectorError as error:
        raise _fail(
            "HANDOFF_RESULT_INVALID",
            "linked decision could not be authenticated",
        ) from error
    linked_decision_invalid = (
        decision_result_identity(decision)
        != handoff["decision"]["result_identity"]
        or decision["result_identity"]
        != handoff["decision"]["result_identity"]
        or decision["inspection"]["inspection_id"]
        != handoff["inspection"]["inspection_id"]
        or decision["capability"]["capability_result"]
        != handoff["decision"]["capability_result"]
    )
    if authorization_mode == "DIRECT_SUPPORTED":
        linked_decision_invalid = linked_decision_invalid or (
            decision["selected_branch"] != HANDOFF_BRANCH
            or decision["handoff_allowed"] is not True
            or decision["spawn_allowed"] is not True
            or decision["capability"]["capability_result"] != "SUPPORTED"
        )
    else:
        linked_decision_invalid = linked_decision_invalid or (
            decision["selected_branch"] is not None
            or decision["handoff_allowed"] is not False
            or decision["spawn_allowed"] is not False
            or decision["capability"]["capability_result"]
            != "RUNTIME_SMOKE_REQUIRED"
        )
    if linked_decision_invalid:
        raise _fail(
            "HANDOFF_RESULT_IDENTITY_MISMATCH",
            "linked decision does not match the retained handoff",
        )
    qualification = None
    if authorization_mode == "QUALIFIED_RUNTIME":
        from .qualification import (
            qualification_result_path,
            validate_qualification_record,
        )
        projection = handoff["qualification"]
        try:
            qualification = validate_qualification_record(
                read_json_record(
                    qualification_result_path(
                        paths, projection["qualification_id"]
                    )
                )
            )
        except (OSError, InspectorError) as error:
            raise _fail(
                "HANDOFF_RESULT_INVALID",
                "linked qualification could not be authenticated",
            ) from error
        if (
            qualification["result_identity"] != projection["result_identity"]
            or qualification["result_class"] != projection["result_class"]
            or qualification["requested_profile"]
            != projection["requested_profile"]
            or qualification["input_decision"]["decision_id"]
            != decision["decision_id"]
            or qualification["input_decision"]["decision_result_identity"]
            != decision["result_identity"]
            or qualification["inspection"]["inspection_id"]
            != inspection["inspection_id"]
            or qualification["inspection"]["artifact_identity"]
            != inspection["artifact"]["identity"]
        ):
            raise _fail(
                "HANDOFF_RESULT_IDENTITY_MISMATCH",
                "linked qualification does not match the retained handoff",
            )
    try:
        binding = load_binding(paths, HANDOFF_BRANCH)
        capability = load_capability_record(
            paths, binding["capability_record_id"]
        )
        installed = verify_installed_tuple(paths, capability)
    except InspectorError as error:
        raise _fail(
            "CAPABILITY_BINDING_INVALID",
            "current GGUF capability binding is invalid",
        ) from error
    if (
        binding["binding_identity"]
        != handoff["capability"]["binding_identity"]
        or binding["binding_generation"]
        != handoff["capability"]["binding_generation"]
        or capability["capability_record_identity"]
        != handoff["capability"]["record_identity"]
        or capability["capability_record_id"]
        != handoff["capability"]["record_id"]
    ):
        raise _fail(
            "CAPABILITY_BINDING_INVALID",
            "current GGUF capability evidence differs from the handoff",
        )
    if qualification is not None and (
        qualification["installed_tuple"].get(
            "branch_capability_record_identity"
        )
        != capability["capability_record_identity"]
        or qualification["installed_tuple"].get(
            "capability_binding_identity"
        )
        != binding["binding_identity"]
        or qualification["validity_predicate"].get(
            "capability_record_identity"
        )
        != capability["capability_record_identity"]
        or qualification["validity_predicate"].get("binding_identity")
        != binding["binding_identity"]
    ):
        raise _fail(
            "CAPABILITY_BINDING_INVALID",
            "qualified handoff differs from current capability evidence",
        )
    if installed.get("verified") is not True:
        raise _fail(
            "CAPABILITY_INSTALLED_TUPLE_MISMATCH",
            "current GGUF installed tuple differs from accepted evidence",
        )
    branch = BranchHandoffPaths.discover(paths)
    relative = Path(handoff["publication"]["managed_relative_path"])
    target = branch.branch_root / relative
    if (
        relative.parts[:2] != ("MODEL", "SUPERMODEL")
        or len(relative.parts) != 3
        or target.parent != branch.managed_root
    ):
        raise _fail(
            "HANDOFF_TARGET_UNSAFE",
            "handoff managed target is outside the branch-owned model root",
        )
    target_identity, _ = _hash_regular_file(
        target, handoff["publication"]
    )
    return HandoffEvidence(
        record=handoff,
        result_identity=handoff_identity,
        inspection_result_identity=inspection_identity,
        decision_result_identity=decision["result_identity"],
        capability_record_identity=capability[
            "capability_record_identity"
        ],
        capability_binding_identity=binding["binding_identity"],
        target_identity=target_identity,
        target_name=relative.name,
    )


@dataclass(frozen=True)
class RegistrySnapshot:
    schema_version: int
    generation: int
    bundle_id: str
    model_version_id: str
    public_identity_mode: str
    capability_manifest_identity: str
    progression_evidence: list[dict[str, Any]]
    aliases: list[str]
    default_alias: str
    default_target: str
    default_artifact_version_id: str


def _table_columns(
    connection: sqlite3.Connection, table: str
) -> set[str]:
    return {
        str(row[1])
        for row in connection.execute(f'PRAGMA table_info("{table}")')
    }


def observe_registry(
    branch_root: Path, handoff: HandoffEvidence
) -> RegistrySnapshot:
    database = (
        branch_root
        / "RUNTIME"
        / "api"
        / "database"
        / "model_registry.sqlite3"
    )
    try:
        details = database.lstat()
        if (
            stat.S_ISLNK(details.st_mode)
            or not stat.S_ISREG(details.st_mode)
            or details.st_nlink != 1
        ):
            raise OSError("unsafe registry")
        connection = sqlite3.connect(
            f"{database.as_uri()}?mode=ro",
            uri=True,
            timeout=0.25,
        )
        connection.execute("PRAGMA busy_timeout=250")
    except (OSError, sqlite3.Error) as error:
        raise _fail(
            "REGISTRY_UNAVAILABLE",
            "GGUF model registry is unavailable for read-only observation",
        ) from error
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only=ON")
        if connection.execute("PRAGMA query_only").fetchone()[0] != 1:
            raise _fail(
                "REGISTRY_UNAVAILABLE",
                "registry read-only enforcement is unavailable",
            )
        for table, required in REGISTRY_TABLE_COLUMNS.items():
            if not required <= _table_columns(connection, table):
                raise _fail(
                    "REGISTRY_SCHEMA_UNSUPPORTED",
                    "registry schema does not expose the required evidence",
                )
        metadata = {
            str(row["key"]): str(row["value"])
            for row in connection.execute(
                "SELECT key,value FROM registry_metadata"
            )
        }
        try:
            schema_version = int(metadata["schema_version"])
            generation = int(metadata["registry_generation"])
        except (KeyError, ValueError) as error:
            raise _fail(
                "REGISTRY_SCHEMA_UNSUPPORTED",
                "registry metadata is incomplete",
            ) from error
        if (
            metadata.get("schema_identity") != REGISTRY_SCHEMA
            or schema_version != 2
            or generation < 1
        ):
            raise _fail(
                "REGISTRY_SCHEMA_UNSUPPORTED",
                "registry identity or version is unsupported",
            )
        location = connection.execute(
            """
            SELECT relative_root,current_bundle_id,present
            FROM artifact_locations WHERE relative_root=?
            """,
            (handoff.target_name,),
        ).fetchone()
        if location is None:
            raise _fail(
                "REGISTRY_LOCATION_NOT_FOUND",
                "managed handoff location is absent from the registry",
            )
        if location["present"] != 1:
            raise _fail(
                "REGISTRY_LOCATION_NOT_READY",
                "managed handoff location is not physically present",
            )
        artifact_digest = handoff.target_identity.removeprefix("sha256:")
        expected_bundle = "bundle-" + artifact_digest
        if (
            location["current_bundle_id"] != expected_bundle
            or handoff.record["inspection"]["artifact_identity"]
            != handoff.target_identity
        ):
            raise _fail(
                "REGISTRY_BUNDLE_MISMATCH",
                "registry bundle does not match the handed-off artifact",
            )
        version_rows = list(
            connection.execute(
                """
                SELECT mv.model_version_id,mv.bundle_id,mv.state
                FROM model_version_locations AS mvl
                JOIN model_versions AS mv
                  ON mv.model_version_id=mvl.model_version_id
                WHERE mvl.relative_root=?
                ORDER BY mv.model_version_id
                """,
                (handoff.target_name,),
            )
        )
        if not version_rows:
            raise _fail(
                "REGISTRY_MODEL_VERSION_NOT_FOUND",
                "managed location has no public model version",
            )
        matching = [
            row
            for row in version_rows
            if row["bundle_id"] == expected_bundle
            and row["state"] == "READY"
        ]
        if not matching:
            raise _fail(
                "REGISTRY_MODEL_VERSION_NOT_READY",
                "managed model version is not READY",
            )
        if len(matching) != 1:
            raise _fail(
                "PUBLIC_MODEL_ID_AMBIGUOUS",
                "managed location resolves to multiple READY model versions",
            )
        model_version_id = str(matching[0]["model_version_id"])
        if PUBLIC_MODEL_PATTERN.fullmatch(model_version_id) is None:
            raise _fail(
                "PUBLIC_MODEL_ID_NOT_FOUND",
                "registry public model identity is invalid",
            )
        manifest = connection.execute(
            """
            SELECT manifest_sha256 FROM capability_manifests
            WHERE model_version_id=?
            """,
            (model_version_id,),
        ).fetchone()
        if manifest is None or not re.fullmatch(
            r"[0-9a-f]{64}", str(manifest["manifest_sha256"])
        ):
            raise _fail(
                "CAPABILITY_MANIFEST_NOT_FOUND",
                "READY model capability manifest is absent",
            )
        bundle_version_count = connection.execute(
            """
            SELECT COUNT(DISTINCT model_version_id)
            FROM model_versions WHERE bundle_id=?
            """,
            (expected_bundle,),
        ).fetchone()[0]
        public_identity_mode = (
            "DISTINCT_LOCATION_SCOPED_VERSION"
            if bundle_version_count > 1
            else "EXISTING_IMMUTABLE_VERSION_REUSED"
        )
        events = [
            {
                "generation": int(row["generation"]),
                "event_type": str(row["event_type"]),
            }
            for row in connection.execute(
                """
                SELECT generation,event_type FROM registry_events
                WHERE subject_id=? ORDER BY generation,event_id
                """,
                (model_version_id,),
            )
        ]
        event_types = [item["event_type"] for item in events]
        ready_event_indexes = [
            index
            for index, event_type in enumerate(event_types)
            if event_type in {"capability_ready", "replacement_ready"}
        ]
        if (
            "model_registered" not in event_types
            or not ready_event_indexes
            or event_types.index("model_registered")
            > min(ready_event_indexes)
        ):
            raise _fail(
                "REGISTRY_MODEL_VERSION_NOT_READY",
                "registry progression evidence is contradictory",
            )
        aliases = sorted(
            str(row["alias"])
            for row in connection.execute(
                "SELECT alias FROM aliases WHERE model_version_id=?",
                (model_version_id,),
            )
        )
        default = connection.execute(
            """
            SELECT a.alias,a.model_version_id,mv.bundle_id
            FROM aliases AS a JOIN model_versions AS mv
              ON mv.model_version_id=a.model_version_id
            WHERE a.alias_kind='default'
            ORDER BY a.alias
            """
        ).fetchall()
        if len(default) != 1:
            raise _fail(
                "REGISTRY_MODEL_VERSION_NOT_READY",
                "registry default alias is not singular",
            )
        if connection.total_changes != 0:
            raise _fail(
                "REGISTRY_UNAVAILABLE",
                "read-only registry observation changed the database",
                internal=True,
            )
        return RegistrySnapshot(
            schema_version=schema_version,
            generation=generation,
            bundle_id=expected_bundle,
            model_version_id=model_version_id,
            public_identity_mode=public_identity_mode,
            capability_manifest_identity=str(
                manifest["manifest_sha256"]
            ),
            progression_evidence=events,
            aliases=aliases,
            default_alias=str(default[0]["alias"]),
            default_target=str(default[0]["model_version_id"]),
            default_artifact_version_id=str(default[0]["bundle_id"]),
        )
    except sqlite3.Error as error:
        raise _fail(
            "REGISTRY_UNAVAILABLE",
            "read-only registry query failed",
        ) from error
    finally:
        connection.close()


@dataclass(frozen=True)
class ServiceSnapshot:
    profile_identity: str
    host: str
    port: int
    base_url: str
    default_alias: str
    service_transaction_id: str
    operation_log: Path
    readiness_state: str


def observe_service(
    branch_root: Path, registry: RegistrySnapshot
) -> ServiceSnapshot:
    control = branch_root / "RUNTIME" / "service_control"
    profile_path = control / "operating-profile.json"
    profile_raw = _private_file_bytes(
        profile_path,
        reason="OPERATING_PROFILE_INVALID",
        maximum=MAX_PROFILE_BYTES,
    )
    try:
        profile = json.loads(profile_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _fail(
            "OPERATING_PROFILE_INVALID",
            "operating profile is invalid JSON",
        ) from error
    if not isinstance(profile, dict) or profile.get("schema_version") != PROFILE_SCHEMA:
        raise _fail(
            "OPERATING_PROFILE_INVALID",
            "operating profile identity is invalid",
        )
    try:
        profile_canonical = json.dumps(
            profile,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise _fail(
            "OPERATING_PROFILE_INVALID",
            "operating profile cannot be canonically identified",
        ) from error
    profile_identity = (
        "sha256:" + hashlib.sha256(profile_canonical).hexdigest()
    )
    public = profile.get("public_endpoint")
    private = profile.get("private_router_endpoint")
    if not isinstance(public, dict) or not isinstance(private, dict):
        raise _fail(
            "OPERATING_PROFILE_INVALID",
            "operating profile endpoints are incomplete",
        )
    host = public.get("host")
    port = public.get("port")
    try:
        address = ipaddress.ip_address(host)
    except (TypeError, ValueError) as error:
        raise _fail(
            "PUBLIC_ENDPOINT_INVALID",
            "public endpoint is not a numeric address",
        ) from error
    if (
        not address.is_loopback
        or str(address) != host
        or type(port) is not int
        or not 1 <= port <= 65535
        or public == private
        or profile.get("startup_model_policy") != "always_warm"
        or profile.get("default_model_alias") != registry.default_alias
    ):
        raise _fail(
            "PUBLIC_ENDPOINT_INVALID",
            "operating profile public endpoint is unsafe",
        )
    desired = _json_file(
        control / "desired-state.json",
        reason="SERVICE_PROFILE_CHANGED",
    )
    if (
        desired.get("schema_version") != DESIRED_SCHEMA
        or desired.get("profile_identity") != profile_identity
    ):
        raise _fail(
            "SERVICE_PROFILE_CHANGED",
            "desired state does not bind the current operating profile",
        )
    if desired.get("desired_state") != "RUNNING":
        raise _fail(
            "SERVICE_DESIRED_STATE_STOPPED",
            "System X desired state is not RUNNING",
        )
    supervisor = _json_file(
        control / "status" / "supervisor.json",
        reason="SERVICE_NOT_READY",
    )
    recovery = supervisor.get("recovery_status")
    api = supervisor.get("observed_api_service")
    router = supervisor.get("observed_private_router")
    warm = supervisor.get("warm_model_identity")
    if (
        supervisor.get("schema_version") != SUPERVISOR_SCHEMA
        or supervisor.get("profile_identity") != profile_identity
        or supervisor.get("desired_state") != "RUNNING"
        or supervisor.get("supervisor_state") != "RUNNING"
        or supervisor.get("service_readiness_state") != "READY"
        or not isinstance(recovery, dict)
        or recovery.get("recovery_state") != "IDLE"
        or recovery.get("fail_closed_latched") is not False
        or not isinstance(api, dict)
        or api.get("active") is not True
        or api.get("consistent") is not True
        or api.get("listener_owned") is not True
        or api.get("lifecycle_state") != "STARTED"
        or api.get("endpoint") != public
        or not isinstance(router, dict)
        or router.get("active") is not True
        or router.get("consistent") is not True
        or router.get("listener_owned") is not True
        or router.get("lifecycle_state") != "STARTED"
        or not isinstance(warm, dict)
        or warm.get("health_state") != "ready"
        or warm.get("resolved_public_model_id") != registry.default_target
        or warm.get("artifact_version_id")
        != registry.default_artifact_version_id
    ):
        raise _fail(
            "SERVICE_NOT_READY",
            "manager-owned System X service is not coherently READY",
        )
    service_status = _json_file(
        branch_root / "RUNTIME" / "api" / "status" / "service.json",
        reason="SERVICE_ACTIVATION_UNAVAILABLE",
    )
    transaction_id = service_status.get("transaction_id")
    log_value = service_status.get("log_path")
    if (
        service_status.get("lifecycle_state") != "STARTED"
        or service_status.get("service_control_profile_identity")
        != profile_identity
        or service_status.get("host") != host
        or service_status.get("port") != port
        or not isinstance(transaction_id, str)
        or not isinstance(log_value, str)
    ):
        raise _fail(
            "SERVICE_ACTIVATION_UNAVAILABLE",
            "manager-owned API activation is inconsistent",
        )
    log_path = Path(log_value)
    expected_log_root = (
        branch_root / "RUNTIME" / "api" / "logs"
    ).resolve(strict=True)
    try:
        log_details = log_path.lstat()
        resolved_log = log_path.resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise _fail(
            "SERVICE_ACTIVATION_UNAVAILABLE",
            "manager-owned operation log is unavailable",
        ) from error
    if (
        stat.S_ISLNK(log_details.st_mode)
        or not stat.S_ISREG(log_details.st_mode)
        or log_details.st_nlink != 1
        or resolved_log.parent != expected_log_root
        or resolved_log.name != f"{transaction_id}.log"
    ):
        raise _fail(
            "SERVICE_ACTIVATION_UNAVAILABLE",
            "manager-owned operation log is unsafe",
        )
    return ServiceSnapshot(
        profile_identity=profile_identity,
        host=host,
        port=port,
        base_url=f"http://{host}:{port}",
        default_alias=registry.default_alias,
        service_transaction_id=transaction_id,
        operation_log=resolved_log,
        readiness_state="READY",
    )


@dataclass(frozen=True)
class SecretCredential:
    key_id: str
    raw: str = field(repr=False)


def read_local_credential(branch_root: Path) -> SecretCredential:
    path = (
        branch_root
        / "RUNTIME"
        / "api"
        / "auth"
        / "handoff"
        / "local-primary.key"
    )
    raw = _private_file_bytes(
        path,
        reason="CREDENTIAL_UNAVAILABLE",
        maximum=256,
        required_mode=0o600,
    )
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as error:
        raise _fail(
            "CREDENTIAL_UNSAFE",
            "local credential encoding is invalid",
        ) from error
    if not text.endswith("\n") or "\n" in text[:-1] or "\r" in text:
        raise _fail(
            "CREDENTIAL_UNSAFE",
            "local credential handoff is malformed",
        )
    value = text[:-1]
    match = RAW_KEY_PATTERN.fullmatch(value)
    if match is None:
        raise _fail(
            "CREDENTIAL_UNSAFE",
            "local credential handoff is malformed",
        )
    return SecretCredential(key_id=match.group("key_id"), raw=value)


@dataclass(frozen=True)
class HttpResult:
    status: int
    body: dict[str, Any]
    request_id_header: str | None


class LoopbackJsonClient:
    """Direct numeric-loopback JSON client with no redirect or proxy support."""

    def __init__(self, service: ServiceSnapshot) -> None:
        self.service = service
        address = ipaddress.ip_address(service.host)
        if not address.is_loopback or str(address) != service.host:
            raise _fail(
                "PUBLIC_ENDPOINT_INVALID",
                "public client requires one exact numeric loopback origin",
            )

    def request(
        self,
        method: str,
        path: str,
        *,
        credential: SecretCredential | None = None,
        body: dict[str, Any] | None = None,
    ) -> HttpResult:
        if (
            method not in {"GET", "POST"}
            or not path.startswith("/")
            or "://" in path
            or "\\" in path
            or "\r" in path
            or "\n" in path
        ):
            raise _fail(
                "PUBLIC_ENDPOINT_INVALID",
                "public request target is unsafe",
            )
        encoded = (
            json.dumps(
                body,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            if body is not None
            else None
        )
        headers = {
            "Accept": "application/json",
            "Connection": "close",
        }
        if encoded is not None:
            headers["Content-Type"] = "application/json"
        if credential is not None:
            headers["Authorization"] = "Bearer " + credential.raw
        connection = http.client.HTTPConnection(
            self.service.host,
            self.service.port,
            timeout=HTTP_TIMEOUT_SECONDS,
        )
        try:
            connection.request(method, path, body=encoded, headers=headers)
            response = connection.getresponse()
            status_code = response.status
            raw = response.read(MAX_JSON_RESPONSE_BYTES + 1)
            request_id_header = response.getheader(
                "X-System-X-Request-ID"
            )
        except (OSError, http.client.HTTPException) as error:
            raise _fail(
                "PUBLIC_REQUEST_FAILED",
                "bounded public System X request failed",
            ) from error
        finally:
            connection.close()
        if len(raw) > MAX_JSON_RESPONSE_BYTES:
            raise _fail(
                "PUBLIC_REQUEST_FAILED",
                "public System X response exceeded its bound",
            )
        if 300 <= status_code <= 399:
            raise _fail(
                "PUBLIC_ENDPOINT_INVALID",
                "public System X redirect was rejected",
            )
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise _fail(
                "PUBLIC_REQUEST_FAILED",
                "public System X response is not bounded JSON",
            ) from error
        if not isinstance(value, dict):
            raise _fail(
                "PUBLIC_REQUEST_FAILED",
                "public System X response is not an object",
            )
        return HttpResult(
            status=status_code,
            body=value,
            request_id_header=request_id_header,
        )


def _all_strings(value: object) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _all_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _all_strings(child)


def _validate_request_identity(result: HttpResult, reason: str) -> str:
    request_id = result.body.get("request_id")
    if (
        not isinstance(request_id, str)
        or REQUEST_ID_PATTERN.fullmatch(request_id) is None
        or (
            result.request_id_header is not None
            and result.request_id_header != request_id
        )
    ):
        raise _fail(reason, "public response request identity is invalid")
    return request_id


@dataclass(frozen=True)
class PublicEvidence:
    health_status: int
    list_status: int
    detail_status: int
    aliases: list[str]
    default_target: str
    default_artifact_version_id: str
    default_warm_target: str
    default_warm_health: str


def observe_public_surface(
    client: LoopbackJsonClient,
    credential: SecretCredential,
    registry: RegistrySnapshot,
) -> PublicEvidence:
    health = client.request("GET", "/system/v1/health")
    _validate_request_identity(health, "PUBLIC_HEALTH_FAILED")
    warm = health.body.get("warm_identity")
    if (
        health.status != 200
        or health.body.get("ready") is not True
        or health.body.get("service_readiness_state") != "READY"
        or health.body.get("registry_status") != "ready"
        or health.body.get("recovery_state") != "IDLE"
        or health.body.get("default_alias") != registry.default_alias
        or health.body.get("resolved_public_model_id")
        != registry.default_target
        or health.body.get("artifact_version_id")
        != registry.default_artifact_version_id
        or not isinstance(warm, dict)
        or warm.get("health_state") != "ready"
    ):
        raise _fail(
            "PUBLIC_HEALTH_FAILED",
            "public System X health is not READY",
        )
    catalogue = client.request(
        "GET",
        "/system/v1/models",
        credential=credential,
    )
    if catalogue.status in {401, 403}:
        raise _fail(
            "AUTHENTICATION_REJECTED",
            "local credential was rejected by System X",
        )
    _validate_request_identity(catalogue, "PUBLIC_MODEL_LIST_FAILED")
    models = catalogue.body.get("models")
    if catalogue.status != 200 or not isinstance(models, list):
        raise _fail(
            "PUBLIC_MODEL_LIST_FAILED",
            "public model catalogue is invalid",
        )
    matches = [
        item
        for item in models
        if isinstance(item, dict)
        and item.get("id") == registry.model_version_id
    ]
    if len(matches) != 1:
        raise _fail(
            "PUBLIC_MODEL_LIST_FAILED",
            "selected immutable model is absent from the public catalogue",
        )
    aliases = matches[0].get("aliases")
    if (
        not isinstance(aliases, list)
        or not all(isinstance(item, str) for item in aliases)
    ):
        raise _fail(
            "PUBLIC_MODEL_LIST_FAILED",
            "selected public model aliases are invalid",
        )
    detail = client.request(
        "GET",
        f"/system/v1/models/{registry.model_version_id}",
        credential=credential,
    )
    if detail.status in {401, 403}:
        raise _fail(
            "AUTHENTICATION_REJECTED",
            "local credential was rejected by System X",
        )
    _validate_request_identity(detail, "PUBLIC_MODEL_DETAIL_FAILED")
    model = detail.body.get("model")
    if (
        detail.status != 200
        or not isinstance(model, dict)
        or model.get("public_model_id") != registry.model_version_id
        or model.get("resolved_model_id") != registry.model_version_id
        or model.get("artifact_version_id") != registry.bundle_id
        or model.get("state") != "ready"
        or model.get("runtime_state")
        not in {"loaded", "unloaded", "loading"}
        or not isinstance(model.get("capabilities"), dict)
    ):
        raise _fail(
            "PUBLIC_MODEL_DETAIL_MISMATCH",
            "public model detail does not match registry evidence",
        )
    serialized_strings = list(_all_strings(detail.body))
    if any(
        text.startswith("/")
        or "MODEL/SUPERMODEL" in text
        or "://" in text
        or re.search(r"(?:127(?:\.[0-9]{1,3}){3}|\[?::1\]?):[0-9]+", text)
        is not None
        for text in serialized_strings
    ):
        raise _fail(
            "PUBLIC_MODEL_DETAIL_MISMATCH",
            "public model detail exposed private physical information",
        )
    return PublicEvidence(
        health_status=health.status,
        list_status=catalogue.status,
        detail_status=detail.status,
        aliases=sorted(aliases),
        default_target=registry.default_target,
        default_artifact_version_id=registry.default_artifact_version_id,
        default_warm_target=str(warm["resolved_public_model_id"]),
        default_warm_health=str(warm["health_state"]),
    )


@dataclass(frozen=True)
class PreparedPublication:
    handoff: HandoffEvidence
    registry: RegistrySnapshot
    service: ServiceSnapshot
    credential: SecretCredential = field(repr=False)
    public: PublicEvidence


def prepare_publication(
    paths: InspectorPaths, handoff_id: str
) -> PreparedPublication:
    handoff = authenticate_handoff(paths, handoff_id)
    branch = BranchHandoffPaths.discover(paths)
    registry = observe_registry(branch.branch_root, handoff)
    service = observe_service(branch.branch_root, registry)
    credential = read_local_credential(branch.branch_root)
    client = LoopbackJsonClient(service)
    public = observe_public_surface(client, credential, registry)
    if public.aliases != registry.aliases:
        raise _fail(
            "PUBLIC_MODEL_DETAIL_MISMATCH",
            "public aliases do not match read-only registry evidence",
        )
    return PreparedPublication(
        handoff=handoff,
        registry=registry,
        service=service,
        credential=credential,
        public=public,
    )


def prepare_publication_with_convergence_wait(
    paths: InspectorPaths, handoff_id: str
) -> PreparedPublication:
    """Wait for product-owned registry rows to converge before publication."""
    deadline = time.monotonic() + REGISTRY_CONVERGENCE_WAIT_SECONDS
    while True:
        try:
            return prepare_publication(paths, handoff_id)
        except InspectorError as error:
            if (
                error.reason_code
                not in REGISTRY_CONVERGENCE_TRANSIENT_REASONS
            ):
                raise
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise
            time.sleep(min(REGISTRY_CONVERGENCE_POLL_SECONDS, remaining))

def _proof_request_body(model_id: str) -> dict[str, Any]:
    instruction = (
        "Reply with exactly the "
        + "single word READY and "
        + "no other text."
    )
    return {
        "model": model_id,
        "messages": [{"role": "user", "content": instruction}],
        "max_output_tokens": 512,
        "stream": False,
        "temperature": 0.0,
    }


def issue_proof_request(
    client: LoopbackJsonClient,
    prepared: PreparedPublication,
) -> dict[str, Any]:
    response = client.request(
        "POST",
        "/system/v1/chat",
        credential=prepared.credential,
        body=_proof_request_body(prepared.registry.model_version_id),
    )
    if response.status in {401, 403}:
        raise _fail(
            "AUTHENTICATION_REJECTED",
            "local credential was rejected by System X",
        )
    request_id = _validate_request_identity(
        response, "PUBLIC_REQUEST_FAILED"
    )
    if response.status != 200:
        raise _fail(
            "PUBLIC_REQUEST_FAILED",
            "System X proof request did not return HTTP 200",
        )
    if response.body.get("model") != prepared.registry.model_version_id:
        raise _fail(
            "PUBLIC_REQUEST_MODEL_MISMATCH",
            "System X proof response model does not match the selection",
        )
    output = response.body.get("output")
    content = output.get("content") if isinstance(output, dict) else None
    if not isinstance(content, str) or not content:
        raise _fail(
            "PUBLIC_REQUEST_EMPTY_FINAL",
            "System X proof response has no final assistant content",
        )
    operation_state = response.body.get("status")
    finish_reason = response.body.get("finish_reason")
    if operation_state not in {
        "completed",
        "incomplete",
        "requires_action",
    } or not isinstance(finish_reason, str):
        raise _fail(
            "PUBLIC_REQUEST_FAILED",
            "System X proof response terminal metadata is invalid",
        )
    usage = response.body.get("usage")
    if not isinstance(usage, dict):
        raise _fail(
            "PUBLIC_REQUEST_FAILED",
            "System X proof response usage is invalid",
        )
    for name in ("input_tokens", "output_tokens"):
        value = usage.get(name)
        if value is not None and (
            type(value) is not int or value < 0
        ):
            raise _fail(
                "PUBLIC_REQUEST_FAILED",
                "System X proof response usage is invalid",
            )
    encoded_content = content.encode("utf-8")
    return {
        "request_id": request_id,
        "http_status": response.status,
        "operation_state": operation_state,
        "finish_reason": finish_reason,
        "response_model_id": response.body["model"],
        "content_nonempty": True,
        "final_content_bytes": len(encoded_content),
        "final_content_sha256": (
            "sha256:" + hashlib.sha256(encoded_content).hexdigest()
        ),
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
    }


def _operation_record_from_line(line: bytes) -> dict[str, Any] | None:
    marker = b"system_x_operation "
    position = line.find(marker)
    if position < 0:
        return None
    payload = line[position + len(marker) :].strip()
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def correlate_operation_record(
    prepared: PreparedPublication,
    proof: dict[str, Any],
    *,
    start_offset: int,
) -> tuple[dict[str, Any], str]:
    path = prepared.service.operation_log
    deadline = time.monotonic() + OPERATION_RECORD_WAIT_SECONDS
    request_id = proof["request_id"]
    while True:
        try:
            details = path.lstat()
            if (
                stat.S_ISLNK(details.st_mode)
                or not stat.S_ISREG(details.st_mode)
                or details.st_nlink != 1
            ):
                raise OSError("unsafe log")
            lower = max(start_offset, details.st_size - MAX_LOG_SCAN_BYTES)
            with path.open("rb") as handle:
                handle.seek(lower)
                raw = handle.read(MAX_LOG_SCAN_BYTES + 1)
        except OSError as error:
            raise _fail(
                "REQUEST_RECORD_NOT_FOUND",
                "operation record log became unavailable",
            ) from error
        if len(raw) > MAX_LOG_SCAN_BYTES:
            raw = raw[-MAX_LOG_SCAN_BYTES:]
        matches = []
        for line in raw.splitlines():
            if request_id.encode("ascii") not in line:
                continue
            record = _operation_record_from_line(line)
            if record is not None and record.get("request_id") == request_id:
                matches.append(record)
        if len(matches) == 1:
            record = matches[0]
            break
        if len(matches) > 1:
            raise _fail(
                "REQUEST_RECORD_MISMATCH",
                "multiple operation records exist for one request",
            )
        if time.monotonic() >= deadline:
            raise _fail(
                "REQUEST_RECORD_NOT_FOUND",
                "proof request operation record was not observed",
            )
        time.sleep(0.1)
    expected_keys = {
        "schema",
        "request_id",
        "key_id",
        "protocol_family",
        "endpoint",
        "operation",
        "streamed",
        "public_model_id",
        "artifact_version_id",
        "api_service_transaction_id",
        "router_transaction_id",
        "started_utc",
        "completed_utc",
        "latency_ms",
        "http_status",
        "error_code",
        "finish_reason",
        "operation_state",
        "input_tokens",
        "output_tokens",
    }
    if set(record) != expected_keys:
        raise _fail(
            "REQUEST_RECORD_MISMATCH",
            "operation record fields are not closed",
        )
    if (
        record["schema"] != OPERATION_RECORD_SCHEMA
        or record["request_id"] != request_id
        or record["key_id"] != prepared.credential.key_id
        or record["protocol_family"] != "system_x"
        or record["endpoint"] != "/system/v1/chat"
        or record["operation"] != "chat"
        or record["streamed"] is not False
        or record["public_model_id"]
        != prepared.registry.model_version_id
        or record["artifact_version_id"] != prepared.registry.bundle_id
        or record["api_service_transaction_id"]
        != prepared.service.service_transaction_id
        or not isinstance(record["router_transaction_id"], str)
        or record["http_status"] != proof["http_status"]
        or record["finish_reason"] != proof["finish_reason"]
        or record["operation_state"] != proof["operation_state"]
        or record["input_tokens"] != proof["input_tokens"]
        or record["output_tokens"] != proof["output_tokens"]
    ):
        raise _fail(
            "REQUEST_RECORD_MISMATCH",
            "operation record does not match proof-request evidence",
        )
    return record, _identity(record)


def restoration_requirement(
    selected_artifact_version_id: str,
    default_artifact_version_id: str,
) -> tuple[bool, bool]:
    required = selected_artifact_version_id != default_artifact_version_id
    return required, False


def verify_final_default(
    prepared: PreparedPublication,
    client: LoopbackJsonClient,
) -> tuple[dict[str, Any], HttpResult]:
    required, performed = restoration_requirement(
        prepared.registry.bundle_id,
        prepared.public.default_artifact_version_id,
    )
    if required:
        restore = client.request(
            "POST",
            "/system/v1/chat",
            credential=prepared.credential,
            body=_proof_request_body(prepared.registry.default_target),
        )
        if (
            restore.status != 200
            or restore.body.get("model")
            != prepared.registry.default_target
        ):
            raise _fail(
                "DEFAULT_RESTORATION_FAILED",
                "default target could not be restored through the public API",
            )
        performed = True
    final_health = client.request("GET", "/system/v1/health")
    _validate_request_identity(
        final_health, "DEFAULT_RESTORATION_FAILED"
    )
    warm = final_health.body.get("warm_identity")
    if (
        final_health.status != 200
        or final_health.body.get("ready") is not True
        or final_health.body.get("service_readiness_state") != "READY"
        or final_health.body.get("recovery_state") != "IDLE"
        or final_health.body.get("default_alias")
        != prepared.registry.default_alias
        or final_health.body.get("resolved_public_model_id")
        != prepared.registry.default_target
        or final_health.body.get("artifact_version_id")
        != prepared.registry.default_artifact_version_id
        or not isinstance(warm, dict)
        or warm.get("resolved_public_model_id")
        != prepared.registry.default_target
        or warm.get("artifact_version_id")
        != prepared.registry.default_artifact_version_id
        or warm.get("health_state") != "ready"
    ):
        raise _fail(
            "DEFAULT_RESTORATION_FAILED",
            "default warm identity is not READY after publication proof",
        )
    return {
        "default_alias": prepared.registry.default_alias,
        "default_target_before": prepared.public.default_target,
        "default_target_after": str(
            final_health.body["resolved_public_model_id"]
        ),
        "restoration_required": required,
        "restoration_performed": performed,
        "final_warm_target": str(warm["resolved_public_model_id"]),
        "final_warm_health": str(warm["health_state"]),
        "final_public_health": str(
            final_health.body["service_readiness_state"]
        ),
    }, final_health


def publication_result_identity(value: dict[str, Any]) -> str:
    try:
        basis = {
            key: value[key]
            for key in TOP_LEVEL_FIELDS
            if key != "result_identity"
        }
    except KeyError as error:
        raise _fail(
            "PUBLICATION_RESULT_INVALID",
            "publication result is incomplete",
        ) from error
    return _identity(basis)


def _reject_forbidden_record_content(value: object) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key.casefold() in FORBIDDEN_RECORD_KEYS:
                raise _fail(
                    "PUBLICATION_RESULT_INVALID",
                    "publication result contains a prohibited field",
                )
            _reject_forbidden_record_content(child)
    elif isinstance(value, list):
        for child in value:
            _reject_forbidden_record_content(child)
    elif isinstance(value, str):
        if (
            value.startswith("/")
            or "MODEL/SUPERMODEL" in value
            or "Bearer " in value
            or "sxk_v1_" in value
        ):
            raise _fail(
                "PUBLICATION_RESULT_INVALID",
                "publication result contains private content",
            )


def validate_publication_record(value: object) -> dict[str, Any]:
    record = _closed(
        value,
        TOP_LEVEL_FIELDS,
        "PUBLICATION_RESULT_INVALID",
        "service publication",
    )
    for name, fields in SECTION_FIELDS.items():
        _closed(
            record[name],
            fields,
            "PUBLICATION_RESULT_INVALID",
            f"service publication {name}",
        )
    _reject_forbidden_record_content(record)
    if (
        record["schema_version"] != PUBLICATION_SCHEMA
        or not isinstance(record["publication_id"], str)
        or PUBLICATION_ID_PATTERN.fullmatch(record["publication_id"]) is None
        or not isinstance(record["transaction_id"], str)
        or not isinstance(record["created_utc"], str)
        or not isinstance(record["publication_basis_identity"], str)
        or SHA256_PATTERN.fullmatch(
            record["publication_basis_identity"]
        )
        is None
        or not isinstance(record["result_identity"], str)
        or SHA256_PATTERN.fullmatch(record["result_identity"]) is None
    ):
        raise _fail(
            "PUBLICATION_RESULT_INVALID",
            "publication root identities are invalid",
        )
    handoff = record["handoff"]
    if (
        handoff["selected_branch"] != HANDOFF_BRANCH
        or not all(
            isinstance(handoff[name], str)
            for name in (
                "handoff_id",
                "decision_id",
                "inspection_id",
                "artifact_identity",
            )
        )
        or not all(
            isinstance(handoff[name], str)
            and SHA256_PATTERN.fullmatch(handoff[name]) is not None
            for name in (
                "handoff_result_identity",
                "decision_result_identity",
                "inspection_result_identity",
                "artifact_identity",
            )
        )
    ):
        raise _fail(
            "PUBLICATION_RESULT_INVALID",
            "publication handoff evidence is invalid",
        )
    registry = record["registry"]
    if (
        registry["schema_identity"] != REGISTRY_SCHEMA
        or registry["schema_version"] != 2
        or type(registry["generation"]) is not int
        or registry["generation"] < 1
        or registry["managed_location_present"] is not True
        or registry["managed_location_state"] != "READY"
        or not isinstance(registry["bundle_id"], str)
        or BUNDLE_PATTERN.fullmatch(registry["bundle_id"]) is None
        or not isinstance(registry["model_version_id"], str)
        or PUBLIC_MODEL_PATTERN.fullmatch(registry["model_version_id"])
        is None
        or registry["public_identity_mode"]
        not in {
            "DISTINCT_LOCATION_SCOPED_VERSION",
            "EXISTING_IMMUTABLE_VERSION_REUSED",
        }
        or not re.fullmatch(
            r"[0-9a-f]{64}",
            str(registry["capability_manifest_identity"]),
        )
        or not isinstance(registry["progression_evidence"], list)
    ):
        raise _fail(
            "PUBLICATION_RESULT_INVALID",
            "publication registry evidence is invalid",
        )
    public = record["public_service"]
    if (
        public["readiness_state"] != "READY"
        or public["health_http_status"] != 200
        or public["model_list_http_status"] != 200
        or public["model_detail_http_status"] != 200
        or public["public_model_id"] != registry["model_version_id"]
        or not isinstance(public["base_url"], str)
        or not public["base_url"].startswith("http://")
        or not isinstance(public["aliases"], list)
    ):
        raise _fail(
            "PUBLICATION_RESULT_INVALID",
            "publication public-service evidence is invalid",
        )
    request = record["request"]
    correlation = record["correlation"]
    if (
        request["http_status"] != 200
        or request["content_nonempty"] is not True
        or request["operation_record_correlated"] is not True
        or request["response_model_id"] != public["public_model_id"]
        or REQUEST_ID_PATTERN.fullmatch(str(request["request_id"])) is None
        or request["final_content_bytes"] < 1
        or SHA256_PATTERN.fullmatch(
            str(request["final_content_sha256"])
        )
        is None
        or correlation["request_id"] != request["request_id"]
        or correlation["inspector_transaction_id"]
        != record["transaction_id"]
        or correlation["operation_record_schema"]
        != OPERATION_RECORD_SCHEMA
        or correlation["artifact_version_id"] != registry["bundle_id"]
        or KEY_ID_PATTERN.fullmatch(
            str(correlation["non_secret_key_id"])
        )
        is None
    ):
        raise _fail(
            "PUBLICATION_RESULT_INVALID",
            "publication request correlation is invalid",
        )
    restoration = record["restoration"]
    if (
        restoration["default_alias"] == ""
        or restoration["default_target_before"]
        != restoration["default_target_after"]
        or restoration["final_warm_target"]
        != restoration["default_target_after"]
        or restoration["final_warm_health"] != "ready"
        or restoration["final_public_health"] != "READY"
        or type(restoration["restoration_required"]) is not bool
        or type(restoration["restoration_performed"]) is not bool
    ):
        raise _fail(
            "PUBLICATION_RESULT_INVALID",
            "publication restoration evidence is invalid",
        )
    if any(record["security"].values()):
        raise _fail(
            "PUBLICATION_RESULT_INVALID",
            "publication security conclusions are unsafe",
        )
    if record["result_identity"] != publication_result_identity(record):
        raise _fail(
            "PUBLICATION_RESULT_INVALID",
            "publication result identity is invalid",
        )
    return record


def build_publication_record(
    *,
    publication_id: str,
    transaction_id: str,
    prepared: PreparedPublication,
    proof: dict[str, Any],
    operation_record: dict[str, Any],
    operation_record_identity: str,
    restoration: dict[str, Any],
) -> dict[str, Any]:
    basis = {
        "handoff_result_identity": prepared.handoff.result_identity,
        "inspection_result_identity": (
            prepared.handoff.inspection_result_identity
        ),
        "decision_result_identity": (
            prepared.handoff.decision_result_identity
        ),
        "artifact_content_identity": prepared.handoff.target_identity,
        "capability_record_identity": (
            prepared.handoff.capability_record_identity
        ),
        "capability_binding_identity": (
            prepared.handoff.capability_binding_identity
        ),
        "operating_profile_identity": prepared.service.profile_identity,
        "registry_bundle_identity": prepared.registry.bundle_id,
        "public_model_id": prepared.registry.model_version_id,
        "capability_manifest_identity": (
            prepared.registry.capability_manifest_identity
        ),
        "api_request_id": proof["request_id"],
        "operation_record_identity": operation_record_identity,
    }
    value = {
        "schema_version": PUBLICATION_SCHEMA,
        "publication_id": publication_id,
        "transaction_id": transaction_id,
        "created_utc": utc_now(),
        "publication_basis_identity": _identity(basis),
        "result_identity": None,
        "handoff": {
            "handoff_id": prepared.handoff.record["handoff_id"],
            "handoff_result_identity": prepared.handoff.result_identity,
            "decision_id": prepared.handoff.record["decision"][
                "decision_id"
            ],
            "decision_result_identity": (
                prepared.handoff.decision_result_identity
            ),
            "inspection_id": prepared.handoff.record["inspection"][
                "inspection_id"
            ],
            "inspection_result_identity": (
                prepared.handoff.inspection_result_identity
            ),
            "artifact_identity": prepared.handoff.target_identity,
            "selected_branch": HANDOFF_BRANCH,
        },
        "registry": {
            "schema_identity": REGISTRY_SCHEMA,
            "schema_version": prepared.registry.schema_version,
            "generation": prepared.registry.generation,
            "managed_location_present": True,
            "managed_location_state": "READY",
            "bundle_id": prepared.registry.bundle_id,
            "model_version_id": prepared.registry.model_version_id,
            "public_identity_mode": prepared.registry.public_identity_mode,
            "capability_manifest_identity": (
                prepared.registry.capability_manifest_identity
            ),
            "progression_evidence": (
                prepared.registry.progression_evidence
            ),
        },
        "public_service": {
            "profile_identity": prepared.service.profile_identity,
            "base_url": prepared.service.base_url,
            "readiness_state": prepared.service.readiness_state,
            "health_http_status": prepared.public.health_status,
            "model_list_http_status": prepared.public.list_status,
            "model_detail_http_status": prepared.public.detail_status,
            "public_model_id": prepared.registry.model_version_id,
            "aliases": prepared.public.aliases,
        },
        "request": {
            **proof,
            "operation_record_correlated": True,
        },
        "correlation": {
            "inspector_transaction_id": transaction_id,
            "request_id": proof["request_id"],
            "operation_record_schema": operation_record["schema"],
            "api_service_transaction_id": operation_record[
                "api_service_transaction_id"
            ],
            "router_transaction_id": operation_record[
                "router_transaction_id"
            ],
            "artifact_version_id": operation_record[
                "artifact_version_id"
            ],
            "non_secret_key_id": operation_record["key_id"],
        },
        "restoration": restoration,
        "security": {
            "outbound_provider_request": False,
            "raw_key_persisted": False,
            "private_endpoint_published": False,
            "physical_model_path_published": False,
            "prompt_persisted": False,
            "generated_content_persisted": False,
        },
    }
    value["result_identity"] = publication_result_identity(value)
    return validate_publication_record(value)


def publication_result_path(
    paths: InspectorPaths, publication_id: str
) -> Path:
    if PUBLICATION_ID_PATTERN.fullmatch(publication_id) is None:
        raise _fail(
            "PUBLICATION_RESULT_INVALID",
            "publication ID is invalid",
        )
    return paths.publication_results / f"{publication_id}.json"


def publish_publication_record(
    paths: InspectorPaths, value: dict[str, Any]
) -> tuple[Path, str]:
    record = validate_publication_record(value)
    path = publication_result_path(paths, record["publication_id"])
    if path.parent != paths.publication_results:
        raise _fail(
            "PUBLICATION_RESULT_INVALID",
            "publication result path escaped its private root",
        )
    if path.exists() or path.is_symlink():
        raise _fail(
            "PUBLICATION_RESULT_COLLISION",
            "publication result target already exists",
        )
    try:
        atomic_create_json(path, record, mode=0o600)
    except InspectorError as error:
        raise _fail(
            "PUBLICATION_RESULT_COLLISION",
            "publication result atomic creation collided",
        ) from error
    details = path.lstat()
    if (
        stat.S_ISLNK(details.st_mode)
        or not stat.S_ISREG(details.st_mode)
        or stat.S_IMODE(details.st_mode) != 0o600
        or details.st_nlink != 1
        or validate_publication_record(read_json_record(path)) != record
    ):
        raise _fail(
            "PUBLICATION_RESULT_INVALID",
            "publication result failed physical verification",
            internal=True,
        )
    return path, record["result_identity"]


def _transaction_base(
    paths: InspectorPaths,
    transaction_id: str,
    publication_id: str,
    handoff_id: str,
    owner: dict[str, Any],
) -> dict[str, Any]:
    owner_identity = {
        key: owner.get(key)
        for key in (
            "pid",
            "process_start_identity",
            "boot_identity",
            "inspector_root_identity",
        )
    }
    return {
        "schema_version": SCHEMA_IDENTITIES["transaction"],
        "transaction_id": transaction_id,
        "operation": "publish-service",
        "start_utc": utc_now(),
        "finish_utc": None,
        "state": "VALIDATING_PUBLICATION",
        "reason_code": "OK",
        "input_target_name": None,
        "intake_snapshot_identity": None,
        "owner_identity": owner_identity,
        "status_record_identity": None,
        "handoff_id": handoff_id,
        "publication_id": publication_id,
        "publication_result_identity": None,
        "publication_result_path": None,
        "request_id": None,
        "request_http_status": None,
        "public_model_id": None,
        "artifact_version_id": None,
        "commit_phase": "STARTED",
        "proof_evidence": None,
    }


def _stage(
    paths: InspectorPaths,
    transaction: dict[str, Any],
    state: str,
    *,
    reason_code: str = "OK",
    **changes: Any,
) -> dict[str, Any]:
    status = _status_value(
        paths,
        state=state,
        reason_code=reason_code,
        active_transaction_id=transaction["transaction_id"],
        last_transaction_id=None,
    )
    status_identity = _write_status(paths, status, None)
    updated = {
        **transaction,
        **changes,
        "state": state,
        "reason_code": reason_code,
        "status_record_identity": status_identity,
    }
    _write_transaction(paths, updated, None)
    return updated


def _set_idle(
    paths: InspectorPaths,
    transaction_id: str,
    *,
    reason_code: str = "OK",
) -> str:
    return _write_status(
        paths,
        _status_value(
            paths,
            state="IDLE",
            reason_code=reason_code,
            active_transaction_id=None,
            last_transaction_id=transaction_id,
        ),
        None,
    )


def publish_service_transaction(
    paths: InspectorPaths,
    handoff_id: str,
) -> tuple[str, dict[str, Any], Path, str]:
    transaction_id = _transaction_id()
    publication_id = _publication_id()
    lock = TransactionLock(
        paths,
        transaction_id=transaction_id,
        operation="publish-service",
    )
    try:
        owner = lock.acquire()
    except InspectorError as error:
        error.data = {**error.data, "transaction_id": transaction_id}
        raise
    transaction = _transaction_base(
        paths,
        transaction_id,
        publication_id,
        handoff_id,
        owner,
    )
    try:
        transaction = _stage(
            paths,
            transaction,
            "VALIDATING_PUBLICATION",
        )
        prepared = prepare_publication_with_convergence_wait(
            paths,
            handoff_id,
        )
        transaction = _stage(
            paths,
            transaction,
            "OBSERVING_REGISTRY",
            public_model_id=prepared.registry.model_version_id,
            artifact_version_id=prepared.registry.bundle_id,
            commit_phase="PUBLIC_EVIDENCE_READY",
        )
        transaction = _stage(
            paths,
            transaction,
            "VERIFYING_PUBLIC_SERVICE",
        )
        client = LoopbackJsonClient(prepared.service)
        log_offset = prepared.service.operation_log.stat().st_size
        transaction = _stage(
            paths,
            transaction,
            "ISSUING_PROOF_REQUEST",
            commit_phase="PRE_REQUEST_READY",
        )
        proof = issue_proof_request(client, prepared)
        transaction = _stage(
            paths,
            transaction,
            "CORRELATING_REQUEST",
            request_id=proof["request_id"],
            request_http_status=proof["http_status"],
            proof_evidence=proof,
            commit_phase="PROOF_REQUEST_RETURNED",
        )
        operation_record, operation_identity = correlate_operation_record(
            prepared,
            proof,
            start_offset=log_offset,
        )
        transaction = _stage(
            paths,
            transaction,
            "RESTORING_DEFAULT",
            commit_phase="REQUEST_RECORD_CORRELATED",
        )
        restoration, _ = verify_final_default(
            prepared,
            client,
        )
        transaction = _stage(
            paths,
            transaction,
            "PUBLISHING_SERVICE_RESULT",
            commit_phase="FINAL_READY_VERIFIED",
        )
        record = build_publication_record(
            publication_id=publication_id,
            transaction_id=transaction_id,
            prepared=prepared,
            proof=proof,
            operation_record=operation_record,
            operation_record_identity=operation_identity,
            restoration=restoration,
        )
        result_path, result_identity = publish_publication_record(
            paths, record
        )
        idle_identity = _set_idle(paths, transaction_id)
        terminal = {
            **transaction,
            "finish_utc": utc_now(),
            "state": "COMPLETED",
            "reason_code": "SERVICE_PUBLICATION_COMPLETE",
            "status_record_identity": idle_identity,
            "publication_result_identity": result_identity,
            "publication_result_path": str(result_path),
            "commit_phase": "SERVICE_RESULT_PUBLISHED",
        }
        _write_transaction(paths, terminal, None)
        return transaction_id, record, result_path, result_identity
    except InspectorError as error:
        try:
            failed_status = _status_value(
                paths,
                state="FAILED",
                reason_code=error.reason_code,
                active_transaction_id=transaction_id,
                last_transaction_id=transaction_id,
            )
            failed_identity = _write_status(
                paths, failed_status, None
            )
            failed = {
                **transaction,
                "finish_utc": utc_now(),
                "state": "FAILED",
                "reason_code": error.reason_code,
                "status_record_identity": failed_identity,
            }
            _write_transaction(paths, failed, None)
            _set_idle(paths, transaction_id)
        except Exception:
            pass
        error.data = {
            **error.data,
            "transaction_id": transaction_id,
        }
        raise
    except Exception as error:
        try:
            failed_status = _status_value(
                paths,
                state="FAILED",
                reason_code="INTERNAL_ERROR",
                active_transaction_id=transaction_id,
                last_transaction_id=transaction_id,
            )
            failed_identity = _write_status(
                paths, failed_status, None
            )
            _write_transaction(
                paths,
                {
                    **transaction,
                    "finish_utc": utc_now(),
                    "state": "FAILED",
                    "reason_code": "INTERNAL_ERROR",
                    "status_record_identity": failed_identity,
                },
                None,
            )
            _set_idle(paths, transaction_id)
        except Exception:
            pass
        raise _fail(
            "INTERNAL_ERROR",
            "unexpected publication internal failure",
            internal=True,
        ) from error
    finally:
        lock.release()
