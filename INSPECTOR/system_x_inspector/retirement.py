"""Safe GGUF retirement, immutable evidence, and bounded owned recovery."""

from __future__ import annotations

import ctypes
import datetime as dt
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import secrets
import sqlite3
import stat
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from .constants import (
    RETIREMENT_REASON_CODES,
    RETIREMENT_RESULT_CLASSES,
    RETIREMENT_STATES,
    SCHEMA_IDENTITIES,
)
from .errors import InspectorError
from .locking import TransactionLock, inspect_active_lock
from .paths import BranchHandoffPaths, InspectorPaths
from .qualification import restore_with_accepted_platform_manager
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
from .service_publication import read_local_credential


SHA256_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")
BUNDLE_ID = re.compile(r"bundle-[0-9a-f]{64}\Z")
PUBLIC_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}\Z")
RETIREMENT_ID = re.compile(
    r"retirement-[0-9]{8}T[0-9]{12}Z-[0-9a-f]{16}\Z"
)
RETIREMENT_FILE = re.compile(
    r"retirement-[0-9]{8}T[0-9]{12}Z-[0-9a-f]{16}\.json\Z"
)
REASON_TEXT = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ._:/-]{0,255}\Z")
MAX_CONTROL_JSON_BYTES = 2 * 1024 * 1024
HASH_CHUNK_BYTES = 1024 * 1024
LAST_MODEL_POLICIES = ("REJECT", "ENTER_WAITING_FOR_MODEL")
RECOVERY_PLAN = (
    ("L1_EXACT_MODEL_CHILD_RECONCILE", 2),
    ("L2_ROUTER_DEFAULT_RELOAD", 1),
    ("L3_CONTROLLER_STACK_RECOVERY", 1),
    ("L4_PLATFORM_MANAGER_RESTART", 1),
)

RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "retirement_id",
        "transaction_id",
        "created_utc",
        "completed_utc",
        "result_class",
        "reason_codes",
        "input",
        "target",
        "replacement",
        "prestate",
        "request_activity_proof",
        "alias_transaction",
        "quarantine",
        "registry_removal",
        "catalogue_removal",
        "recovery",
        "poststate",
        "later_request",
        "waiting_for_model_health",
        "quarantine_deletion",
        "states_observed",
        "validity_predicate",
        "result_identity",
    }
)
FORBIDDEN_RESULT_KEYS = frozenset(
    {
        "raw_api_key",
        "raw_key",
        "credential_verifier",
        "pepper",
        "private_router_url",
        "model_child_port",
        "physical_gguf_path",
        "absolute_managed_path",
        "managed_path",
        "quarantine_path",
        "prompt",
        "answer",
        "reasoning",
        "process_environment",
    }
)


class RetirementFailClosed(RuntimeError):
    """A post-intent condition whose ownership cannot safely be inferred."""

    def __init__(
        self,
        reason_code: str,
        message: str,
        *,
        ownership_certain: bool = False,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.message = message
        self.ownership_certain = ownership_certain


def _identity(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _retirement_error(
    reason_code: str,
    message: str,
    *,
    data: dict[str, Any] | None = None,
    internal: bool = False,
) -> InspectorError:
    return InspectorError(
        reason_code,
        message,
        data=data,
        exit_status=70 if internal else 2,
    )


def _new_retirement_id() -> str:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"retirement-{stamp}-{secrets.token_hex(8)}"


def _reject_forbidden(value: object) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_RESULT_KEYS:
                raise _retirement_error(
                    "RETIREMENT_RESULT_INVALID",
                    f"retirement result contains prohibited field: {key}",
                )
            _reject_forbidden(child)
    elif isinstance(value, list):
        for child in value:
            _reject_forbidden(child)


def retirement_result_identity(value: dict[str, Any]) -> str:
    return _identity(
        {key: value[key] for key in sorted(value) if key != "result_identity"}
    )


def validate_retirement_record(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != RESULT_FIELDS:
        raise _retirement_error(
            "RETIREMENT_RESULT_INVALID",
            "retirement result fields are not closed",
        )
    record = value
    _reject_forbidden(record)
    if (
        record["schema_version"]
        != SCHEMA_IDENTITIES["gguf_retirement_result"]
        or not isinstance(record["retirement_id"], str)
        or RETIREMENT_ID.fullmatch(record["retirement_id"]) is None
        or not isinstance(record["transaction_id"], str)
        or not record["transaction_id"]
        or record["result_class"] not in RETIREMENT_RESULT_CLASSES
    ):
        raise _retirement_error(
            "RETIREMENT_RESULT_INVALID",
            "retirement result identity or enum is invalid",
        )
    for key in (
        "input",
        "target",
        "prestate",
        "request_activity_proof",
        "alias_transaction",
        "quarantine",
        "registry_removal",
        "catalogue_removal",
        "recovery",
        "poststate",
        "later_request",
        "waiting_for_model_health",
        "quarantine_deletion",
        "validity_predicate",
    ):
        if not isinstance(record[key], dict):
            raise _retirement_error(
                "RETIREMENT_RESULT_INVALID",
                f"retirement result section is invalid: {key}",
            )
    if record["replacement"] is not None and not isinstance(
        record["replacement"], dict
    ):
        raise _retirement_error(
            "RETIREMENT_RESULT_INVALID",
            "retirement replacement section is invalid",
        )
    states = record["states_observed"]
    reasons = record["reason_codes"]
    if (
        not isinstance(states, list)
        or not states
        or any(item not in RETIREMENT_STATES for item in states)
        or len(states) != len(list(dict.fromkeys(states)))
        or not isinstance(reasons, list)
        or not reasons
        or any(item not in RETIREMENT_REASON_CODES for item in reasons)
        or len(reasons) != len(list(dict.fromkeys(reasons)))
    ):
        raise _retirement_error(
            "RETIREMENT_RESULT_INVALID",
            "retirement state or reason history is invalid",
        )
    terminal = {
        "RETIREMENT_COMPLETE": "COMPLETE",
        "RETIREMENT_WAITING_FOR_MODEL": "WAITING_FOR_MODEL",
        "RETIREMENT_FAILED_CLEAN": "FAILED_CLEAN",
        "RETIREMENT_FAIL_CLOSED": "FAIL_CLOSED",
    }[record["result_class"]]
    if states[-1] != terminal:
        raise _retirement_error(
            "RETIREMENT_RESULT_INVALID",
            "retirement terminal state does not match result class",
        )
    expected = retirement_result_identity(record)
    if (
        not isinstance(record["result_identity"], str)
        or SHA256_ID.fullmatch(record["result_identity"]) is None
        or record["result_identity"] != expected
    ):
        raise _retirement_error(
            "RETIREMENT_RESULT_INVALID",
            "retirement result identity is invalid",
        )
    return record


def retirement_result_path(
    paths: InspectorPaths, retirement_id: str
) -> Path:
    if RETIREMENT_ID.fullmatch(retirement_id) is None:
        raise _retirement_error(
            "RETIREMENT_RESULT_INVALID",
            "retirement ID is not canonical",
        )
    return paths.retirement_results / f"{retirement_id}.json"


def _private_result_file(path: Path, parent: Path) -> None:
    details = path.lstat()
    if (
        path.parent != parent
        or stat.S_ISLNK(details.st_mode)
        or not stat.S_ISREG(details.st_mode)
        or stat.S_IMODE(details.st_mode) != 0o600
        or details.st_nlink != 1
        or details.st_size > MAX_CONTROL_JSON_BYTES
    ):
        raise _retirement_error(
            "RETIREMENT_RESULT_INVALID",
            "retirement result has an unsafe physical type",
        )


def publish_retirement_record(
    paths: InspectorPaths, value: dict[str, Any]
) -> Path:
    record = validate_retirement_record(value)
    path = retirement_result_path(paths, record["retirement_id"])
    if path.exists() or path.is_symlink():
        _private_result_file(path, paths.retirement_results)
        if validate_retirement_record(read_json_record(path)) == record:
            return path
        raise _retirement_error(
            "RETIREMENT_RESULT_COLLISION",
            "different immutable retirement result already exists",
        )
    try:
        atomic_create_json(path, record, mode=0o600)
    except InspectorError as error:
        if path.exists() and not path.is_symlink():
            _private_result_file(path, paths.retirement_results)
            if validate_retirement_record(read_json_record(path)) == record:
                return path
        raise _retirement_error(
            "RETIREMENT_RESULT_COLLISION",
            "retirement result publication collided",
        ) from error
    _private_result_file(path, paths.retirement_results)
    if validate_retirement_record(read_json_record(path)) != record:
        raise _retirement_error(
            "RETIREMENT_RESULT_INVALID",
            "retirement result did not round-trip",
        )
    return path


@dataclass(frozen=True)
class RetirementRequest:
    public_model_id: str
    artifact_identity: str
    managed_location_identity: str
    expected_registry_generation: int
    retirement_reason: str
    last_model_policy: str

    @classmethod
    def parse(
        cls,
        *,
        public_model_id: object,
        artifact_identity: object,
        managed_location_identity: object,
        expected_registry_generation: object,
        retirement_reason: object,
        last_model_policy: object,
    ) -> "RetirementRequest":
        if (
            not isinstance(public_model_id, str)
            or PUBLIC_MODEL_ID.fullmatch(public_model_id) is None
            or not isinstance(artifact_identity, str)
            or (
                BUNDLE_ID.fullmatch(artifact_identity) is None
                and SHA256_ID.fullmatch(artifact_identity) is None
            )
            or not isinstance(managed_location_identity, str)
            or SHA256_ID.fullmatch(managed_location_identity) is None
            or isinstance(expected_registry_generation, bool)
            or not isinstance(expected_registry_generation, int)
            or expected_registry_generation < 0
            or not isinstance(retirement_reason, str)
            or REASON_TEXT.fullmatch(retirement_reason) is None
            or last_model_policy not in LAST_MODEL_POLICIES
        ):
            raise _retirement_error(
                "RETIREMENT_INPUT_INVALID",
                "retire-gguf input is invalid",
            )
        return cls(
            public_model_id=public_model_id,
            artifact_identity=artifact_identity,
            managed_location_identity=managed_location_identity,
            expected_registry_generation=expected_registry_generation,
            retirement_reason=retirement_reason,
            last_model_policy=str(last_model_policy),
        )

    def result_projection(self) -> dict[str, Any]:
        return {
            "public_model_id": self.public_model_id,
            "artifact_identity": self.artifact_identity,
            "managed_location_identity": self.managed_location_identity,
            "expected_registry_generation": (
                self.expected_registry_generation
            ),
            "retirement_reason": self.retirement_reason,
            "last_model_policy": self.last_model_policy,
        }

    @property
    def identity(self) -> str:
        return _identity(self.result_projection())


@dataclass(frozen=True)
class RetirementTarget:
    public_model_id: str
    artifact_identity: str
    managed_location_identity: str
    registry_generation: int
    model_state: str
    relative_root: str
    target_path: Path
    managed_root: Path
    quarantine_root: Path
    device: int
    inode: int
    mode: int
    link_count: int
    size: int
    mtime_ns: int
    authenticated_content_sha256: str | None
    default_target: str | None
    ready_model_ids: tuple[str, ...]
    replacement: dict[str, Any] | None
    service_prestate: dict[str, Any]
    capability_manifest_identity: str | None = None
    rollback_dependency: bool = False

    @property
    def is_default(self) -> bool:
        return self.default_target == self.public_model_id

    @property
    def is_last_ready(self) -> bool:
        return self.ready_model_ids == (self.public_model_id,)

    def result_projection(self) -> dict[str, Any]:
        return {
            "public_model_id": self.public_model_id,
            "artifact_identity": self.artifact_identity,
            "managed_location_identity": self.managed_location_identity,
            "observed_registry_generation": self.registry_generation,
            "model_state": self.model_state,
            "was_default": self.is_default,
            "was_last_ready_model": self.is_last_ready,
            "physical_identity": {
                "device": self.device,
                "inode": self.inode,
                "mode": self.mode,
                "link_count": self.link_count,
                "size": self.size,
                "mtime_ns": self.mtime_ns,
            },
        }

    def private_projection(self) -> dict[str, Any]:
        return {
            **self.result_projection(),
            "relative_root": self.relative_root,
            "target_path": str(self.target_path),
            "managed_root": str(self.managed_root),
            "quarantine_root": str(self.quarantine_root),
            "authenticated_content_sha256": (
                self.authenticated_content_sha256
            ),
            "default_target": self.default_target,
            "ready_model_ids": list(self.ready_model_ids),
            "replacement": self.replacement,
            "service_prestate": self.service_prestate,
            "capability_manifest_identity": (
                self.capability_manifest_identity
            ),
            "rollback_dependency": self.rollback_dependency,
        }

    @classmethod
    def from_private(cls, value: dict[str, Any]) -> "RetirementTarget":
        physical = value["physical_identity"]
        return cls(
            public_model_id=str(value["public_model_id"]),
            artifact_identity=str(value["artifact_identity"]),
            managed_location_identity=str(
                value["managed_location_identity"]
            ),
            registry_generation=int(value["observed_registry_generation"]),
            model_state=str(value["model_state"]),
            relative_root=str(value["relative_root"]),
            target_path=Path(value["target_path"]),
            managed_root=Path(value["managed_root"]),
            quarantine_root=Path(value["quarantine_root"]),
            device=int(physical["device"]),
            inode=int(physical["inode"]),
            mode=int(physical["mode"]),
            link_count=int(physical["link_count"]),
            size=int(physical["size"]),
            mtime_ns=int(physical["mtime_ns"]),
            authenticated_content_sha256=value.get(
                "authenticated_content_sha256"
            ),
            default_target=value.get("default_target"),
            ready_model_ids=tuple(value["ready_model_ids"]),
            replacement=value.get("replacement"),
            service_prestate=dict(value["service_prestate"]),
            capability_manifest_identity=value.get(
                "capability_manifest_identity"
            ),
            rollback_dependency=bool(value.get("rollback_dependency")),
        )


def managed_location_identity(
    *,
    public_model_id: str,
    artifact_identity: str,
    relative_root: str,
    device: int,
    inode: int,
    mode: int,
    link_count: int,
    size: int,
) -> str:
    return _identity(
        {
            "public_model_id": public_model_id,
            "artifact_identity": artifact_identity,
            "relative_root": relative_root,
            "device": device,
            "inode": inode,
            "mode": mode,
            "link_count": link_count,
            "size": size,
        }
    )


class RetirementRuntimeAdapter(Protocol):
    def resolve_target(
        self, request: RetirementRequest
    ) -> RetirementTarget: ...

    def activity_snapshot(
        self, target: RetirementTarget
    ) -> dict[str, Any]: ...

    def clear_default(
        self, target: RetirementTarget, transaction_id: str
    ) -> dict[str, Any]: ...

    def on_quarantined(
        self, target: RetirementTarget, quarantine: dict[str, Any]
    ) -> None: ...

    def observe_registry_removal(
        self, target: RetirementTarget
    ) -> dict[str, Any]: ...

    def observe_service(
        self, target: RetirementTarget, *, last_model: bool
    ) -> dict[str, Any]: ...

    def recover(
        self,
        level: str,
        target: RetirementTarget,
        *,
        last_model: bool,
        attempt: int,
    ) -> dict[str, Any]: ...

    def later_request(
        self, target: RetirementTarget
    ) -> dict[str, Any]: ...

    def waiting_proof(
        self, target: RetirementTarget
    ) -> dict[str, Any]: ...

    def on_restored(self, target: RetirementTarget) -> None: ...

    def restore_default(
        self,
        target: RetirementTarget,
        alias_transaction: dict[str, Any],
        transaction_id: str,
    ) -> dict[str, Any]: ...

    def observe_restored(
        self, target: RetirementTarget
    ) -> dict[str, Any]: ...

    def checkpoint(self, state: str) -> None: ...


def _bounded_health(branch_root: Path) -> dict[str, Any]:
    profile = read_json_record(
        branch_root / "RUNTIME" / "service_control" / "operating-profile.json"
    )
    public = profile.get("public_endpoint")
    if not isinstance(public, dict):
        raise _retirement_error(
            "RETIREMENT_RECOVERY_FAILED",
            "public endpoint configuration is absent",
        )
    host = public.get("host")
    port = public.get("port")
    if host != "127.0.0.1" or type(port) is not int:
        raise _retirement_error(
            "RETIREMENT_RECOVERY_FAILED",
            "public endpoint configuration is unsafe",
        )
    connection = http.client.HTTPConnection(host, port, timeout=5.0)
    try:
        connection.request(
            "GET",
            "/system/v1/health",
            headers={"Accept": "application/json", "Connection": "close"},
        )
        response = connection.getresponse()
        raw = response.read(1024 * 1024 + 1)
        status_code = response.status
    except (OSError, http.client.HTTPException) as error:
        raise _retirement_error(
            "RETIREMENT_RECOVERY_FAILED",
            "bounded public health observation failed",
        ) from error
    finally:
        connection.close()
    if len(raw) > 1024 * 1024:
        raise _retirement_error(
            "RETIREMENT_RECOVERY_FAILED",
            "public health response exceeded its bound",
        )
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _retirement_error(
            "RETIREMENT_RECOVERY_FAILED",
            "public health response is invalid",
        ) from error
    if not isinstance(value, dict):
        raise _retirement_error(
            "RETIREMENT_RECOVERY_FAILED",
            "public health response is not an object",
        )
    return {
        "http_status": status_code,
        "service_readiness": value.get("service_readiness_state"),
        "model_service_state": value.get("model_service_state"),
        "service_available": value.get("service_available"),
        "inference_ready": value.get("inference_ready"),
        "default_target": value.get("resolved_public_model_id"),
        "artifact_version_id": value.get("artifact_version_id"),
        "recovery_state": value.get("recovery_state"),
        "warm": value.get("warm_identity"),
    }


class CurrentSourceRetirementAdapter:
    """Read current registry state and use only accepted owned control surfaces."""

    def __init__(
        self,
        paths: InspectorPaths,
        *,
        observation_attempts: int = 60,
        observation_interval_seconds: float = 0.1,
    ) -> None:
        self.paths = paths
        self.branch = BranchHandoffPaths.discover(paths)
        self.database = (
            self.branch.branch_root
            / "RUNTIME"
            / "api"
            / "database"
            / "model_registry.sqlite3"
        )
        self.quarantine_root = (
            self.branch.branch_root
            / "RUNTIME"
            / "api"
            / "retirement-staging"
        )
        if (
            type(observation_attempts) is not int
            or not 1 <= observation_attempts <= 600
            or type(observation_interval_seconds) not in {int, float}
            or not 0.0 <= float(observation_interval_seconds) <= 2.0
        ):
            raise _retirement_error(
                "RETIREMENT_INPUT_INVALID",
                "retirement observation bounds are invalid",
            )
        self.observation_attempts = observation_attempts
        self.observation_interval_seconds = float(
            observation_interval_seconds
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            "file:" + str(self.database) + "?mode=ro",
            uri=True,
            timeout=5.0,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        return connection

    def _content_sha256(
        self, connection: sqlite3.Connection, bundle_id: str
    ) -> str | None:
        columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(artifact_files)")
        }
        digest_column = next(
            (
                name
                for name in ("sha256", "file_sha256", "content_sha256")
                if name in columns
            ),
            None,
        )
        if digest_column is None or "bundle_id" not in columns:
            return None
        rows = connection.execute(
            f"SELECT {digest_column} FROM artifact_files "
            "WHERE bundle_id=? ORDER BY rowid",
            (bundle_id,),
        ).fetchall()
        if len(rows) != 1:
            return None
        value = str(rows[0][0])
        return (
            "sha256:" + value
            if re.fullmatch(r"[0-9a-f]{64}", value)
            else value
            if SHA256_ID.fullmatch(value)
            else None
        )

    def _lookup(self, public_model_id: str) -> RetirementTarget:
        connection = self._connect()
        try:
            generation_row = connection.execute(
                "SELECT value FROM registry_metadata "
                "WHERE key='registry_generation'"
            ).fetchone()
            row = connection.execute(
                """
                SELECT
                    mv.model_version_id,mv.bundle_id,mv.state,
                    mvl.relative_root,al.present,al.current_bundle_id,
                    al.physical_manifest_json,ab.bundle_sha256,
                    ab.bundle_kind,ab.file_count,ab.size_bytes,
                    cm.manifest_sha256
                FROM model_versions AS mv
                JOIN model_version_locations AS mvl
                  ON mvl.model_version_id=mv.model_version_id
                JOIN artifact_locations AS al
                  ON al.relative_root=mvl.relative_root
                JOIN artifact_bundles AS ab
                  ON ab.bundle_id=mv.bundle_id
                LEFT JOIN capability_manifests AS cm
                  ON cm.model_version_id=mv.model_version_id
                WHERE mv.model_version_id=?
                ORDER BY mvl.relative_root
                """,
                (public_model_id,),
            ).fetchall()
            if generation_row is None or len(row) != 1:
                raise _retirement_error(
                    "RETIREMENT_TARGET_NOT_FOUND",
                    "public model does not resolve to one managed location",
                )
            selected = row[0]
            default_row = connection.execute(
                "SELECT model_version_id FROM aliases WHERE alias='default'"
            ).fetchone()
            default_target = (
                str(default_row[0]) if default_row is not None else None
            )
            ready_rows = connection.execute(
                """
                SELECT DISTINCT mv.model_version_id
                FROM model_versions AS mv
                JOIN model_version_locations AS mvl
                  ON mvl.model_version_id=mv.model_version_id
                JOIN artifact_locations AS al
                  ON al.relative_root=mvl.relative_root
                WHERE mv.state='READY' AND al.present=1
                ORDER BY mv.model_version_id
                """
            ).fetchall()
            ready_ids = tuple(str(item[0]) for item in ready_rows)
            replacement = None
            if default_target is not None and default_target != public_model_id:
                replacement_row = connection.execute(
                    """
                    SELECT mv.model_version_id,mv.bundle_id,mv.state,
                           mvl.relative_root,cm.manifest_sha256,al.present
                    FROM model_versions AS mv
                    JOIN model_version_locations AS mvl
                      ON mvl.model_version_id=mv.model_version_id
                    JOIN artifact_locations AS al
                      ON al.relative_root=mvl.relative_root
                    LEFT JOIN capability_manifests AS cm
                      ON cm.model_version_id=mv.model_version_id
                    WHERE mv.model_version_id=?
                    """,
                    (default_target,),
                ).fetchone()
                if replacement_row is not None:
                    replacement = {
                        "public_model_id": str(
                            replacement_row["model_version_id"]
                        ),
                        "artifact_version_id": str(
                            replacement_row["bundle_id"]
                        ),
                        "state": str(replacement_row["state"]),
                        "relative_root": str(
                            replacement_row["relative_root"]
                        ),
                        "capability_manifest_identity": (
                            "sha256:"
                            + str(replacement_row["manifest_sha256"])
                            if replacement_row["manifest_sha256"]
                            else None
                        ),
                        "present": bool(replacement_row["present"]),
                    }
            content_sha256 = self._content_sha256(
                connection, str(selected["bundle_id"])
            )
            if connection.total_changes != 0:
                raise _retirement_error(
                    "RETIREMENT_OWNERSHIP_UNCERTAIN",
                    "read-only retirement observation changed the registry",
                    internal=True,
                )
        finally:
            connection.close()
        relative_root = str(selected["relative_root"])
        if len(Path(relative_root).parts) != 1:
            raise _retirement_error(
                "RETIREMENT_TARGET_OUTSIDE_ROOT",
                "retirement target is not a direct managed GGUF",
            )
        target_path = self.branch.managed_root / relative_root
        try:
            details = target_path.lstat()
        except FileNotFoundError as error:
            raise _retirement_error(
                "RETIREMENT_LOCATION_ALREADY_REMOVED",
                "managed retirement target is absent",
            ) from error
        artifact_identity = str(selected["bundle_id"])
        location_identity = managed_location_identity(
            public_model_id=public_model_id,
            artifact_identity=artifact_identity,
            relative_root=relative_root,
            device=details.st_dev,
            inode=details.st_ino,
            mode=stat.S_IMODE(details.st_mode),
            link_count=details.st_nlink,
            size=details.st_size,
        )
        service = _bounded_health(self.branch.branch_root)
        manifest_identity = (
            "sha256:" + str(selected["manifest_sha256"])
            if selected["manifest_sha256"]
            else None
        )
        return RetirementTarget(
            public_model_id=public_model_id,
            artifact_identity=artifact_identity,
            managed_location_identity=location_identity,
            registry_generation=int(generation_row[0]),
            model_state=str(selected["state"]),
            relative_root=relative_root,
            target_path=target_path,
            managed_root=self.branch.managed_root,
            quarantine_root=self.quarantine_root,
            device=details.st_dev,
            inode=details.st_ino,
            mode=stat.S_IMODE(details.st_mode),
            link_count=details.st_nlink,
            size=details.st_size,
            mtime_ns=details.st_mtime_ns,
            authenticated_content_sha256=content_sha256,
            default_target=default_target,
            ready_model_ids=ready_ids,
            replacement=replacement,
            service_prestate=service,
            capability_manifest_identity=manifest_identity,
        )

    def resolve_target(
        self, request: RetirementRequest
    ) -> RetirementTarget:
        return self._lookup(request.public_model_id)

    def activity_snapshot(
        self, target: RetirementTarget
    ) -> dict[str, Any]:
        # Current source owns exact in-process counters, but the accepted live
        # process predates this local retirement surface.  Fail closed for a
        # mutating production call; the required live REJECT is ordered before
        # this fence and isolated current-source fixtures prove the full path.
        return {
            "available": False,
            "active_requests": None,
            "active_streams": None,
            "nonterminal_operations": None,
            "target_public_model_id": target.public_model_id,
        }

    def _alias_transaction(
        self, request: dict[str, Any]
    ) -> dict[str, Any]:
        controller = (
            self.branch.branch_root
            / "api_service_controller"
            / "controller.py"
        )
        python = (
            self.branch.branch_root
            / "api_service"
            / ".venv"
            / "bin"
            / "python"
        )
        encoded = canonical_json_bytes(
            {
                "schema_version": "system-x.gguf-alias-transaction.v1",
                **request,
            }
        )
        completed = subprocess.run(
            [str(python), "-B", str(controller), "alias-transaction"],
            input=encoded,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.branch.branch_root,
            env={
                "PATH": "/usr/bin:/bin",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
            },
            timeout=30.0,
            check=False,
        )
        try:
            value = json.loads(completed.stdout)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise _retirement_error(
                "RETIREMENT_ALIAS_CONFLICT",
                "branch alias transaction emitted invalid JSON",
            ) from error
        if (
            completed.returncode != 0
            or not isinstance(value, dict)
            or value.get("ok") is not True
            or not isinstance(value.get("alias_transaction"), dict)
        ):
            raise _retirement_error(
                "RETIREMENT_ALIAS_CONFLICT",
                "branch alias transaction was rejected",
                data={"controller_reason_code": value.get("reason_code")},
            )
        return dict(value["alias_transaction"])

    def clear_default(
        self, target: RetirementTarget, transaction_id: str
    ) -> dict[str, Any]:
        return self._alias_transaction(
            {
                "action": "clear",
                "promotion_transaction_id": transaction_id,
                "alias": "default",
                "expected_current_target": target.public_model_id,
                "new_target": None,
                "expected_registry_generation": target.registry_generation,
                "target_artifact_version_id": None,
                "target_capability_manifest_identity": None,
                "target_relative_root": None,
                "promotion_alias_event_identity": None,
            }
        )

    def on_quarantined(
        self, target: RetirementTarget, quarantine: dict[str, Any]
    ) -> None:
        return None

    def observe_registry_removal(
        self, target: RetirementTarget
    ) -> dict[str, Any]:
        for attempt in range(1, self.observation_attempts + 1):
            connection = self._connect()
            try:
                generation = int(
                    connection.execute(
                        "SELECT value FROM registry_metadata "
                        "WHERE key='registry_generation'"
                    ).fetchone()[0]
                )
                location = connection.execute(
                    "SELECT present FROM artifact_locations "
                    "WHERE relative_root=?",
                    (target.relative_root,),
                ).fetchone()
                model = connection.execute(
                    "SELECT state FROM model_versions "
                    "WHERE model_version_id=?",
                    (target.public_model_id,),
                ).fetchone()
                event = connection.execute(
                    """
                    SELECT event_id,generation,event_type
                    FROM registry_events
                    WHERE event_type='artifact_location_removed'
                      AND subject_id=?
                    ORDER BY generation DESC LIMIT 1
                    """,
                    (target.relative_root,),
                ).fetchone()
                catalogue = connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM model_versions AS mv
                    JOIN model_version_locations AS mvl
                      ON mvl.model_version_id=mv.model_version_id
                    JOIN artifact_locations AS al
                      ON al.relative_root=mvl.relative_root
                    WHERE mv.model_version_id=?
                      AND mv.state='READY' AND al.present=1
                    """,
                    (target.public_model_id,),
                ).fetchone()[0]
            finally:
                connection.close()
            if (
                location is not None
                and int(location["present"]) == 0
                and model is not None
                and str(model["state"]) == "REMOVED"
                and event is not None
                and int(catalogue) == 0
            ):
                return {
                    "observed": True,
                    "attempts": attempt,
                    "observed_registry_generation": generation,
                    "registry_removal_event_identity": _identity(
                        {
                            "event_id": str(event["event_id"]),
                            "generation": int(event["generation"]),
                            "event_type": str(event["event_type"]),
                        }
                    ),
                    "catalogue_target_absent": True,
                    "immutable_history_present": True,
                }
            if attempt < self.observation_attempts:
                time.sleep(self.observation_interval_seconds)
        raise _retirement_error(
            "RETIREMENT_REGISTRY_REMOVAL_TIMEOUT",
            "registry watcher did not observe retirement in time",
        )

    @staticmethod
    def _service_exact(
        value: dict[str, Any],
        target: RetirementTarget,
        *,
        last_model: bool,
    ) -> bool:
        if last_model:
            return bool(
                value.get("http_status") == 200
                and value.get("service_available") is True
                and value.get("inference_ready") is False
                and value.get("model_service_state") == "WAITING_FOR_MODEL"
                and value.get("recovery_state") == "IDLE"
                and value.get("default_target") is None
            )
        replacement = target.replacement or {}
        warm = value.get("warm")
        return bool(
            value.get("http_status") == 200
            and value.get("service_readiness") == "READY"
            and value.get("model_service_state") == "READY"
            and value.get("service_available") is True
            and value.get("inference_ready") is True
            and value.get("recovery_state") == "IDLE"
            and value.get("default_target")
            == replacement.get("public_model_id")
            and isinstance(warm, dict)
            and warm.get("resolved_public_model_id")
            == replacement.get("public_model_id")
        )

    def observe_service(
        self, target: RetirementTarget, *, last_model: bool
    ) -> dict[str, Any]:
        value = _bounded_health(self.branch.branch_root)
        return {
            **value,
            "exact": self._service_exact(
                value, target, last_model=last_model
            ),
        }

    def recover(
        self,
        level: str,
        target: RetirementTarget,
        *,
        last_model: bool,
        attempt: int,
    ) -> dict[str, Any]:
        if level in {
            "L1_EXACT_MODEL_CHILD_RECONCILE",
            "L2_ROUTER_DEFAULT_RELOAD",
        }:
            return {
                "level": level,
                "attempt": attempt,
                "used": False,
                "reason_code": "OWNED_SURFACE_NOT_REQUIRED_OR_AVAILABLE",
                "ownership_certain": True,
            }
        if level == "L3_CONTROLLER_STACK_RECOVERY":
            controller = (
                self.branch.branch_root
                / "api_service_controller"
                / "controller.py"
            )
            python = (
                self.branch.branch_root
                / "api_service"
                / ".venv"
                / "bin"
                / "python"
            )
            completed = subprocess.run(
                [str(python), "-B", str(controller), "reconcile"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.branch.branch_root,
                env={
                    "PATH": "/usr/bin:/bin",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONNOUSERSITE": "1",
                },
                timeout=30.0,
                check=False,
            )
            return {
                "level": level,
                "attempt": attempt,
                "used": True,
                "exit_status": completed.returncode,
                "controller_result_identity": hashlib.sha256(
                    completed.stdout
                ).hexdigest(),
                "ownership_certain": True,
            }
        if level == "L4_PLATFORM_MANAGER_RESTART":
            result = restore_with_accepted_platform_manager(
                self.branch.branch_root
            )
            return {
                "level": level,
                "attempt": attempt,
                "used": True,
                "manager_result": result,
                "ownership_certain": True,
            }
        raise _retirement_error(
            "RETIREMENT_RECOVERY_FAILED",
            "unknown retirement recovery level",
            internal=True,
        )

    def _authenticated_request(
        self,
        target: RetirementTarget,
        *,
        expect_waiting: bool,
    ) -> dict[str, Any]:
        profile = read_json_record(
            self.branch.branch_root
            / "RUNTIME"
            / "service_control"
            / "operating-profile.json"
        )
        public = profile["public_endpoint"]
        credential = read_local_credential(self.branch.branch_root)
        body = canonical_json_bytes(
            {
                "model": "default",
                "messages": [
                    {
                        "role": "user",
                        "content": "Reply with exactly READY.",
                    }
                ],
                "max_output_tokens": 16,
                "stream": False,
                "temperature": 0.0,
            }
        )
        connection = http.client.HTTPConnection(
            public["host"], public["port"], timeout=30.0
        )
        try:
            connection.request(
                "POST",
                "/system/v1/chat",
                body=body,
                headers={
                    "Accept": "application/json",
                    "Authorization": "Bearer " + credential.raw,
                    "Connection": "close",
                    "Content-Type": "application/json",
                },
            )
            response = connection.getresponse()
            raw = response.read(1024 * 1024 + 1)
            status_code = response.status
            request_id = response.getheader("X-System-X-Request-ID")
        finally:
            connection.close()
        if len(raw) > 1024 * 1024:
            raise _retirement_error(
                "RETIREMENT_RECOVERY_FAILED",
                "continuity response exceeded its bound",
            )
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise _retirement_error(
                "RETIREMENT_RECOVERY_FAILED",
                "continuity response is invalid JSON",
            ) from error
        if not isinstance(value, dict):
            raise _retirement_error(
                "RETIREMENT_RECOVERY_FAILED",
                "continuity response is not an object",
            )
        if expect_waiting:
            passed = bool(
                status_code == 503
                and (
                    value.get("reason_code") == "NO_READY_MODEL"
                    or value.get("error", {}).get("code")
                    == "NO_READY_MODEL"
                )
            )
        else:
            replacement = target.replacement or {}
            passed = bool(
                status_code == 200
                and value.get("model")
                == replacement.get("public_model_id")
            )
        if not passed:
            raise _retirement_error(
                "RETIREMENT_RECOVERY_FAILED",
                "post-retirement public request did not match",
            )
        return {
            "passed": True,
            "request_id": request_id,
            "http_status": status_code,
            "response_model_match": (
                False if expect_waiting else True
            ),
            "bounded_content_present": bool(
                value.get("output") or value.get("error")
            ),
            "credential_key_id": credential.key_id,
            "reason_code": (
                "NO_READY_MODEL" if expect_waiting else "OK"
            ),
        }

    def later_request(
        self, target: RetirementTarget
    ) -> dict[str, Any]:
        return self._authenticated_request(target, expect_waiting=False)

    def waiting_proof(
        self, target: RetirementTarget
    ) -> dict[str, Any]:
        health = self.observe_service(target, last_model=True)
        request = self._authenticated_request(target, expect_waiting=True)
        return {
            "passed": health["exact"] is True and request["passed"] is True,
            "health_http_status": health["http_status"],
            "service_available": health["service_available"],
            "inference_ready": health["inference_ready"],
            "model_service_state": health["model_service_state"],
            "recovery_state": health["recovery_state"],
            "request_id": request["request_id"],
            "inference_http_status": request["http_status"],
            "reason_code": request["reason_code"],
        }

    def on_restored(self, target: RetirementTarget) -> None:
        return None

    def restore_default(
        self,
        target: RetirementTarget,
        alias_transaction: dict[str, Any],
        transaction_id: str,
    ) -> dict[str, Any]:
        for attempt in range(1, self.observation_attempts + 1):
            try:
                observed = self._lookup(target.public_model_id)
            except InspectorError:
                observed = None
            if observed is not None and observed.model_state == "READY":
                return self._alias_transaction(
                    {
                        "action": "promote",
                        "promotion_transaction_id": transaction_id,
                        "alias": "default",
                        "expected_current_target": None,
                        "new_target": target.public_model_id,
                        "expected_registry_generation": (
                            observed.registry_generation
                        ),
                        "target_artifact_version_id": (
                            target.artifact_identity
                        ),
                        "target_capability_manifest_identity": (
                            target.capability_manifest_identity
                        ),
                        "target_relative_root": target.relative_root,
                        "promotion_alias_event_identity": None,
                    }
                )
            if attempt < self.observation_attempts:
                time.sleep(self.observation_interval_seconds)
        raise RetirementFailClosed(
            "RETIREMENT_RESTORATION_FAILED",
            "restored last-model target did not return READY",
            ownership_certain=True,
        )

    def observe_restored(
        self, target: RetirementTarget
    ) -> dict[str, Any]:
        value = _bounded_health(self.branch.branch_root)
        warm = value.get("warm")
        proved = bool(
            value.get("service_readiness") == "READY"
            and value.get("recovery_state") == "IDLE"
            and value.get("default_target") == target.public_model_id
            and isinstance(warm, dict)
            and warm.get("resolved_public_model_id")
            == target.public_model_id
        )
        return {**value, "proved": proved}

    def checkpoint(self, state: str) -> None:
        return None


def _physical_metadata(target: RetirementTarget) -> dict[str, Any]:
    try:
        details = target.target_path.lstat()
    except FileNotFoundError as error:
        raise _retirement_error(
            "RETIREMENT_LOCATION_ALREADY_REMOVED",
            "managed retirement target is absent",
        ) from error
    if stat.S_ISLNK(details.st_mode):
        raise _retirement_error(
            "RETIREMENT_TARGET_SYMLINK",
            "managed retirement target is a symlink",
        )
    if not stat.S_ISREG(details.st_mode):
        raise _retirement_error(
            "RETIREMENT_TARGET_SPECIAL_FILE",
            "managed retirement target is not a regular file",
        )
    if details.st_nlink != 1:
        raise _retirement_error(
            "RETIREMENT_TARGET_HARDLINK",
            "managed retirement target has multiple links",
        )
    if (
        target.target_path.parent != target.managed_root
        or target.target_path.name != target.relative_root
        or target.target_path.resolve(strict=True)
        != target.target_path
        or target.managed_root.resolve(strict=True)
        != target.managed_root
    ):
        raise _retirement_error(
            "RETIREMENT_TARGET_OUTSIDE_ROOT",
            "managed retirement target escaped its direct root",
        )
    observed = {
        "device": details.st_dev,
        "inode": details.st_ino,
        "mode": stat.S_IMODE(details.st_mode),
        "link_count": details.st_nlink,
        "size": details.st_size,
        "mtime_ns": details.st_mtime_ns,
    }
    expected = {
        "device": target.device,
        "inode": target.inode,
        "mode": target.mode,
        "link_count": target.link_count,
        "size": target.size,
        "mtime_ns": target.mtime_ns,
    }
    if observed != expected:
        raise _retirement_error(
            "RETIREMENT_TARGET_CHANGED",
            "managed retirement target identity changed",
        )
    return observed


def _validate_preconditions(
    request: RetirementRequest,
    target: RetirementTarget,
    adapter: RetirementRuntimeAdapter,
) -> dict[str, Any]:
    if target.public_model_id != request.public_model_id:
        raise _retirement_error(
            "RETIREMENT_PUBLIC_MODEL_MISMATCH",
            "public model identity changed",
        )
    if target.artifact_identity != request.artifact_identity:
        raise _retirement_error(
            "RETIREMENT_ARTIFACT_IDENTITY_MISMATCH",
            "artifact identity changed",
        )
    if (
        target.managed_location_identity
        != request.managed_location_identity
    ):
        raise _retirement_error(
            "RETIREMENT_LOCATION_IDENTITY_MISMATCH",
            "managed location identity changed",
        )
    if target.registry_generation != request.expected_registry_generation:
        raise _retirement_error(
            "RETIREMENT_GENERATION_MISMATCH",
            "registry generation changed",
        )
    if target.model_state != "READY":
        reason = (
            "RETIREMENT_LOCATION_ALREADY_REMOVED"
            if target.model_state == "REMOVED"
            else "RETIREMENT_TARGET_INVALID"
        )
        raise _retirement_error(
            reason,
            "retirement target is not an active READY model",
        )
    _physical_metadata(target)
    if target.rollback_dependency:
        raise _retirement_error(
            "RETIREMENT_ROLLBACK_DEPENDENCY",
            "target is required by a nonterminal promotion rollback",
        )
    if request.last_model_policy == "REJECT":
        if target.is_default:
            raise _retirement_error(
                "RETIREMENT_TARGET_IS_DEFAULT",
                "current default retirement is rejected by policy",
            )
        if target.is_last_ready:
            raise _retirement_error(
                "RETIREMENT_LAST_MODEL_REJECTED",
                "last READY model retirement is rejected by policy",
            )
    if target.is_last_ready and (
        request.last_model_policy != "ENTER_WAITING_FOR_MODEL"
        or not target.is_default
    ):
        raise _retirement_error(
            "RETIREMENT_LAST_MODEL_REJECTED",
            "last-model retirement requires explicit default waiting policy",
        )
    if target.is_default and not target.is_last_ready:
        raise _retirement_error(
            "RETIREMENT_TARGET_IS_DEFAULT",
            "default retirement requires the explicit last-model case",
        )
    if not target.is_last_ready:
        replacement = target.replacement
        if (
            not isinstance(replacement, dict)
            or replacement.get("public_model_id") != target.default_target
            or replacement.get("state") != "READY"
            or replacement.get("present") is not True
        ):
            raise _retirement_error(
                "RETIREMENT_REPLACEMENT_NOT_READY",
                "replacement default is not exact and READY",
            )
        warm = target.service_prestate.get("warm")
        if (
            target.service_prestate.get("service_readiness") != "READY"
            or target.service_prestate.get("recovery_state") != "IDLE"
            or target.service_prestate.get("default_target")
            != replacement.get("public_model_id")
            or not isinstance(warm, dict)
            or warm.get("resolved_public_model_id")
            != replacement.get("public_model_id")
        ):
            raise _retirement_error(
                "RETIREMENT_REPLACEMENT_NOT_WARM",
                "replacement default is not warm and coherent",
            )
        if (
            isinstance(warm, dict)
            and warm.get("resolved_public_model_id")
            == target.public_model_id
        ):
            raise _retirement_error(
                "RETIREMENT_TARGET_LOADED",
                "retirement target is currently loaded",
            )
    activity = adapter.activity_snapshot(target)
    if activity.get("available") is not True:
        raise _retirement_error(
            "RETIREMENT_ACTIVITY_UNAVAILABLE",
            "exact request activity proof is unavailable",
        )
    counts = (
        activity.get("active_requests"),
        activity.get("active_streams"),
        activity.get("nonterminal_operations"),
    )
    if any(type(value) is not int or value < 0 for value in counts):
        raise _retirement_error(
            "RETIREMENT_ACTIVITY_UNAVAILABLE",
            "request activity proof is invalid",
        )
    if any(value != 0 for value in counts):
        raise _retirement_error(
            "RETIREMENT_TARGET_IN_USE",
            "retirement target has active work",
        )
    return activity


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        while True:
            chunk = os.read(descriptor, HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    finally:
        os.close(descriptor)
    return "sha256:" + digest.hexdigest()


def _file_snapshot(path: Path, *, hash_content: bool) -> dict[str, Any]:
    details = path.lstat()
    if (
        stat.S_ISLNK(details.st_mode)
        or not stat.S_ISREG(details.st_mode)
        or details.st_nlink != 1
    ):
        raise _retirement_error(
            "RETIREMENT_OWNERSHIP_UNCERTAIN",
            "retirement file identity is unsafe",
        )
    value = {
        "device": details.st_dev,
        "inode": details.st_ino,
        "mode": stat.S_IMODE(details.st_mode),
        "link_count": details.st_nlink,
        "size": details.st_size,
        "mtime_ns": details.st_mtime_ns,
    }
    if hash_content:
        value["sha256"] = _hash_file(path)
    value["identity"] = _identity(value)
    return value


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = libc.renameat2
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    if renameat2(
        -100, os.fsencode(source), -100, os.fsencode(destination), 1
    ) != 0:
        error_number = ctypes.get_errno()
        if error_number == 17:
            raise _retirement_error(
                "RETIREMENT_QUARANTINE_COLLISION",
                "retirement quarantine target already exists",
            )
        raise OSError(error_number, os.strerror(error_number))


def _validate_quarantine_root(target: RetirementTarget) -> None:
    try:
        details = target.quarantine_root.lstat()
    except FileNotFoundError as error:
        raise _retirement_error(
            "RETIREMENT_QUARANTINE_INVALID",
            "branch retirement quarantine is absent",
        ) from error
    if (
        stat.S_ISLNK(details.st_mode)
        or not stat.S_ISDIR(details.st_mode)
        or stat.S_IMODE(details.st_mode) != 0o700
        or target.quarantine_root.resolve(strict=True)
        != target.quarantine_root
        or details.st_dev != target.managed_root.stat().st_dev
    ):
        raise _retirement_error(
            "RETIREMENT_QUARANTINE_INVALID",
            "branch retirement quarantine identity is unsafe",
        )


def _quarantine_destination(
    target: RetirementTarget, retirement_id: str
) -> Path:
    return target.quarantine_root / f"{retirement_id}.retired.gguf"


def _public_file_identity(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value[key]
        for key in (
            "device",
            "inode",
            "mode",
            "link_count",
            "size",
            "mtime_ns",
            "sha256",
            "identity",
        )
        if key in value
    }


def _request_from_arguments(
    *,
    public_model_id: object,
    artifact_identity: object,
    managed_location_identity: object,
    expected_registry_generation: object,
    retirement_reason: object,
    last_model_policy: object,
) -> RetirementRequest:
    return RetirementRequest.parse(
        public_model_id=public_model_id,
        artifact_identity=artifact_identity,
        managed_location_identity=managed_location_identity,
        expected_registry_generation=expected_registry_generation,
        retirement_reason=retirement_reason,
        last_model_policy=last_model_policy,
    )


def _find_completed(
    paths: InspectorPaths, request: RetirementRequest
) -> tuple[dict[str, Any], Path] | None:
    if not paths.retirement_results.exists():
        return None
    for path in sorted(paths.retirement_results.glob("retirement-*.json")):
        if RETIREMENT_FILE.fullmatch(path.name) is None:
            continue
        _private_result_file(path, paths.retirement_results)
        record = validate_retirement_record(read_json_record(path))
        if record["input"] == request.result_projection():
            return record, path
    return None


def _find_recoverable(
    paths: InspectorPaths, request: RetirementRequest
) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for path in sorted(paths.transactions.glob("tx-*.json")):
        try:
            value = read_json_record(path)
        except (OSError, InspectorError):
            continue
        if (
            value.get("operation") == "retire-gguf"
            and value.get("retirement_request_identity")
            == request.identity
            and value.get("finish_utc") is None
            and isinstance(value.get("retirement_runtime"), dict)
        ):
            candidates.append(value)
    if len(candidates) > 1:
        raise _retirement_error(
            "RETIREMENT_OWNERSHIP_UNCERTAIN",
            "multiple recoverable retirement transactions exist",
            internal=True,
        )
    return candidates[0] if candidates else None


def _clear_exact_stale_lock(
    paths: InspectorPaths, transaction_id: str
) -> None:
    lock_path = paths.locks / "active.json"
    observed = inspect_active_lock(lock_path)
    if observed["state"] == "absent":
        return
    record = observed.get("record")
    if (
        observed["state"] != "stale"
        or not isinstance(record, dict)
        or record.get("transaction_id") != transaction_id
        or record.get("operation") != "retire-gguf"
    ):
        raise _retirement_error(
            "RETIREMENT_ACTIVE_TRANSACTION",
            "retirement lock ownership is not recoverable",
        )
    details = lock_path.lstat()
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
        raise _retirement_error(
            "RETIREMENT_OWNERSHIP_UNCERTAIN",
            "stale retirement lock identity is unsafe",
            internal=True,
        )
    lock_path.unlink()
    fsync_directory(lock_path.parent)


def _initial_transaction(
    paths: InspectorPaths,
    request: RetirementRequest,
    target: RetirementTarget,
    *,
    transaction_id: str,
    retirement_id: str,
    owner: dict[str, Any],
) -> dict[str, Any]:
    basis = {
        "request": request.result_projection(),
        "target": target.result_projection(),
    }
    return {
        "schema_version": SCHEMA_IDENTITIES["transaction"],
        "transaction_id": transaction_id,
        "operation": "retire-gguf",
        "start_utc": utc_now(),
        "finish_utc": None,
        "state": "PREPARING",
        "reason_code": "OK",
        "input_target_name": None,
        "intake_snapshot_identity": None,
        "owner_identity": {
            key: owner.get(key)
            for key in (
                "pid",
                "process_start_identity",
                "boot_identity",
                "inspector_root_identity",
            )
        },
        "status_record_identity": None,
        "retirement_id": retirement_id,
        "retirement_result_identity": None,
        "retirement_result_path": None,
        "retirement_basis_identity": _identity(basis),
        "retirement_request_identity": request.identity,
        "retirement_runtime": {
            "request": request.result_projection(),
            "target": target.private_projection(),
            "states_observed": [],
            "request_activity_proof": {},
            "alias_transaction": {},
            "quarantine": {},
            "registry_removal": {},
            "catalogue_removal": {},
            "recovery": {},
            "poststate": {},
            "later_request": {},
            "waiting_for_model_health": {},
            "quarantine_deletion": {},
        },
        "states_observed": [],
        "intended_action": None,
        "irreversible_action_begun": None,
        "irreversible_action_observed": None,
    }


def _persist(
    paths: InspectorPaths,
    transaction: dict[str, Any],
    *,
    state: str,
    runtime: dict[str, Any],
    reason_code: str = "OK",
    intended_action: str | None = None,
    irreversible_action_begun: str | None = None,
    irreversible_action_observed: str | None = None,
    observer: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    states = list(runtime.get("states_observed") or [])
    if not states or states[-1] != state:
        states.append(state)
    runtime = {**runtime, "states_observed": states}
    status = _status_value(
        paths,
        state=state,
        reason_code=reason_code,
        active_transaction_id=transaction["transaction_id"],
        last_transaction_id=None,
    )
    status_identity = _write_status(paths, status, observer)
    changed = {
        **transaction,
        "state": state,
        "reason_code": reason_code,
        "status_record_identity": status_identity,
        "retirement_runtime": runtime,
        "states_observed": states,
        "intended_action": intended_action,
        "irreversible_action_begun": irreversible_action_begun,
        "irreversible_action_observed": irreversible_action_observed,
    }
    _write_transaction(paths, changed, observer)
    return changed


def _checkpoint(
    adapter: RetirementRuntimeAdapter, transaction: dict[str, Any]
) -> None:
    adapter.checkpoint(str(transaction["state"]))


def _service_safe(
    value: dict[str, Any], *, last_model: bool
) -> bool:
    if value.get("exact") is not True:
        return False
    if last_model:
        return bool(
            value.get("http_status") == 200
            and value.get("service_available") is True
            and value.get("inference_ready") is False
            and value.get("model_service_state") == "WAITING_FOR_MODEL"
            and value.get("recovery_state") == "IDLE"
        )
    return bool(
        value.get("service_readiness") == "READY"
        and value.get("inference_ready") is True
        and value.get("recovery_state") == "IDLE"
    )


def _recover(
    paths: InspectorPaths,
    transaction: dict[str, Any],
    runtime: dict[str, Any],
    adapter: RetirementRuntimeAdapter,
    target: RetirementTarget,
    *,
    last_model: bool,
    observer: Callable[[str, dict[str, Any]], None] | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    attempts: dict[str, int] = {"L0_OBSERVE": 1}
    actions: list[dict[str, Any]] = []
    before = adapter.observe_service(target, last_model=last_model)
    if _service_safe(before, last_model=last_model):
        recovery = {
            "level_reached": "L0_OBSERVE",
            "attempt_count_by_level": attempts,
            "actions": actions,
            "before": before,
            "after": before,
            "reason_codes": ["OK"],
        }
        return transaction, recovery, before
    transaction = _persist(
        paths,
        transaction,
        state="RECOVERY_RECONCILING",
        runtime=runtime,
        intended_action="OWNED_RECOVERY",
        observer=observer,
    )
    _checkpoint(adapter, transaction)
    latest = before
    level_reached = "L0_OBSERVE"
    for level, maximum in RECOVERY_PLAN:
        for attempt in range(1, maximum + 1):
            attempts[level] = attempt
            if level == "L2_ROUTER_DEFAULT_RELOAD":
                transaction = _persist(
                    paths,
                    transaction,
                    state="DEFAULT_RELOADING",
                    runtime=runtime,
                    intended_action=level,
                    observer=observer,
                )
            elif level == "L4_PLATFORM_MANAGER_RESTART":
                transaction = _persist(
                    paths,
                    transaction,
                    state="MANAGER_RESTARTING",
                    runtime=runtime,
                    intended_action=level,
                    irreversible_action_begun="RESTART_MANAGER",
                    observer=observer,
                )
            action = adapter.recover(
                level,
                target,
                last_model=last_model,
                attempt=attempt,
            )
            if action.get("ownership_certain") is not True:
                raise RetirementFailClosed(
                    "RETIREMENT_OWNERSHIP_UNCERTAIN",
                    "recovery ownership is uncertain",
                    ownership_certain=False,
                )
            actions.append(action)
            level_reached = level
            latest = adapter.observe_service(
                target, last_model=last_model
            )
            if _service_safe(latest, last_model=last_model):
                return (
                    transaction,
                    {
                        "level_reached": level_reached,
                        "attempt_count_by_level": attempts,
                        "actions": actions,
                        "before": before,
                        "after": latest,
                        "reason_codes": ["OK"],
                    },
                    latest,
                )
    raise RetirementFailClosed(
        "RETIREMENT_RECOVERY_FAILED",
        "bounded owned recovery did not converge",
        ownership_certain=True,
    )


def _result_record(
    request: RetirementRequest,
    target: RetirementTarget,
    transaction: dict[str, Any],
    runtime: dict[str, Any],
    *,
    result_class: str,
    reason_codes: list[str],
) -> dict[str, Any]:
    terminal = {
        "RETIREMENT_COMPLETE": "COMPLETE",
        "RETIREMENT_WAITING_FOR_MODEL": "WAITING_FOR_MODEL",
        "RETIREMENT_FAILED_CLEAN": "FAILED_CLEAN",
        "RETIREMENT_FAIL_CLOSED": "FAIL_CLOSED",
    }[result_class]
    states = list(runtime.get("states_observed") or [])
    if not states or states[-1] != terminal:
        states.append(terminal)
    unique_reasons = list(dict.fromkeys(reason_codes))
    record: dict[str, Any] = {
        "schema_version": SCHEMA_IDENTITIES[
            "gguf_retirement_result"
        ],
        "retirement_id": transaction["retirement_id"],
        "transaction_id": transaction["transaction_id"],
        "created_utc": transaction["start_utc"],
        "completed_utc": utc_now(),
        "result_class": result_class,
        "reason_codes": unique_reasons,
        "input": request.result_projection(),
        "target": target.result_projection(),
        "replacement": (
            {
                key: target.replacement.get(key)
                for key in (
                    "public_model_id",
                    "artifact_version_id",
                    "state",
                    "capability_manifest_identity",
                )
            }
            if isinstance(target.replacement, dict)
            else None
        ),
        "prestate": target.service_prestate,
        "request_activity_proof": dict(
            runtime.get("request_activity_proof") or {}
        ),
        "alias_transaction": dict(
            runtime.get("alias_transaction") or {}
        ),
        "quarantine": dict(runtime.get("quarantine_public") or {}),
        "registry_removal": dict(
            runtime.get("registry_removal") or {}
        ),
        "catalogue_removal": dict(
            runtime.get("catalogue_removal") or {}
        ),
        "recovery": dict(runtime.get("recovery") or {}),
        "poststate": dict(runtime.get("poststate") or {}),
        "later_request": dict(runtime.get("later_request") or {}),
        "waiting_for_model_health": dict(
            runtime.get("waiting_for_model_health") or {}
        ),
        "quarantine_deletion": dict(
            runtime.get("quarantine_deletion") or {}
        ),
        "states_observed": states,
        "validity_predicate": {
            "target_identity_authenticated": True,
            "caller_supplied_physical_path": False,
            "registry_direct_write": False,
            "history_deleted": False,
            "unknown_process_signalled": False,
            "bounded_recovery": True,
            "production_claim": False,
        },
        "result_identity": "",
    }
    record["result_identity"] = retirement_result_identity(record)
    return validate_retirement_record(record)


def _finish(
    paths: InspectorPaths,
    transaction: dict[str, Any],
    runtime: dict[str, Any],
    request: RetirementRequest,
    target: RetirementTarget,
    *,
    result_class: str,
    reason_codes: list[str],
    publisher: Callable[
        [InspectorPaths, dict[str, Any]], Path
    ],
    observer: Callable[[str, dict[str, Any]], None] | None,
) -> tuple[str, dict[str, Any], Path, str]:
    record = _result_record(
        request,
        target,
        transaction,
        runtime,
        result_class=result_class,
        reason_codes=reason_codes,
    )
    path = publisher(paths, record)
    terminal_state = record["states_observed"][-1]
    idle = _status_value(
        paths,
        state="IDLE",
        reason_code="OK",
        active_transaction_id=None,
        last_transaction_id=transaction["transaction_id"],
    )
    idle_identity = _write_status(paths, idle, observer)
    terminal = {
        **transaction,
        "finish_utc": record["completed_utc"],
        "state": terminal_state,
        "reason_code": record["reason_codes"][0],
        "status_record_identity": idle_identity,
        "retirement_result_identity": record["result_identity"],
        "retirement_result_path": str(path),
        "retirement_runtime": {
            **runtime,
            "states_observed": record["states_observed"],
        },
        "states_observed": record["states_observed"],
    }
    _write_transaction(paths, terminal, observer)
    return (
        str(transaction["transaction_id"]),
        record,
        path,
        str(record["result_identity"]),
    )


def _restore_after_failure(
    target: RetirementTarget,
    runtime: dict[str, Any],
    adapter: RetirementRuntimeAdapter,
    transaction_id: str,
) -> dict[str, Any]:
    quarantine = runtime.get("quarantine")
    if not isinstance(quarantine, dict) or not quarantine:
        return {"proved": True, "mutation_started": False}
    destination = Path(str(quarantine["path"]))
    if not destination.exists() or target.target_path.exists():
        raise RetirementFailClosed(
            "RETIREMENT_RESTORATION_FAILED",
            "quarantine or original path identity is ambiguous",
            ownership_certain=False,
        )
    expected = dict(quarantine["file_identity"])
    observed = _file_snapshot(destination, hash_content=True)
    if _public_file_identity(observed) != _public_file_identity(expected):
        raise RetirementFailClosed(
            "RETIREMENT_OWNERSHIP_UNCERTAIN",
            "quarantine identity changed before restoration",
            ownership_certain=False,
        )
    _rename_noreplace(destination, target.target_path)
    fsync_directory(destination.parent)
    fsync_directory(target.target_path.parent)
    restored = _file_snapshot(target.target_path, hash_content=True)
    if _public_file_identity(restored) != _public_file_identity(expected):
        raise RetirementFailClosed(
            "RETIREMENT_RESTORATION_FAILED",
            "restored target identity changed",
            ownership_certain=False,
        )
    adapter.on_restored(target)
    alias_restore: dict[str, Any] = {"required": False}
    alias = runtime.get("alias_transaction")
    if isinstance(alias, dict) and alias:
        alias_restore = adapter.restore_default(
            target, alias, transaction_id
        )
    service = adapter.observe_restored(target)
    if service.get("proved") is not True:
        raise RetirementFailClosed(
            "RETIREMENT_RESTORATION_FAILED",
            "restored incumbent service state was not proved",
            ownership_certain=True,
        )
    return {
        "proved": True,
        "mutation_started": True,
        "file_restored": True,
        "alias_restoration": alias_restore,
        "service": service,
    }


def retire_transaction(
    paths: InspectorPaths,
    public_model_id: object,
    artifact_identity: object,
    managed_location_identity: object,
    expected_registry_generation: object,
    retirement_reason: object,
    last_model_policy: object = "REJECT",
    *,
    adapter: RetirementRuntimeAdapter | None = None,
    adapter_factory: Callable[
        [InspectorPaths], RetirementRuntimeAdapter
    ] = CurrentSourceRetirementAdapter,
    retirement_id_factory: Callable[[], str] = _new_retirement_id,
    transaction_id_factory: Callable[[], str] = _transaction_id,
    publisher: Callable[
        [InspectorPaths, dict[str, Any]], Path
    ] = publish_retirement_record,
    transition_observer: Callable[
        [str, dict[str, Any]], None
    ] | None = None,
) -> tuple[str, dict[str, Any], Path, str]:
    request = _request_from_arguments(
        public_model_id=public_model_id,
        artifact_identity=artifact_identity,
        managed_location_identity=managed_location_identity,
        expected_registry_generation=expected_registry_generation,
        retirement_reason=retirement_reason,
        last_model_policy=last_model_policy,
    )
    duplicate = _find_completed(paths, request)
    if duplicate is not None:
        record, path = duplicate
        return (
            str(record["transaction_id"]),
            record,
            path,
            str(record["result_identity"]),
        )
    runtime_adapter = adapter or adapter_factory(paths)
    recoverable = _find_recoverable(paths, request)
    if recoverable is None:
        target = runtime_adapter.resolve_target(request)
        activity = _validate_preconditions(
            request, target, runtime_adapter
        )
        transaction_id = transaction_id_factory()
        retirement_id = retirement_id_factory()
    else:
        runtime_value = recoverable.get("retirement_runtime")
        if not isinstance(runtime_value, dict) or not isinstance(
            runtime_value.get("target"), dict
        ):
            raise _retirement_error(
                "RETIREMENT_OWNERSHIP_UNCERTAIN",
                "recoverable retirement target state is incomplete",
                internal=True,
            )
        target = RetirementTarget.from_private(runtime_value["target"])
        activity = dict(
            runtime_value.get("request_activity_proof") or {}
        )
        transaction_id = str(recoverable["transaction_id"])
        retirement_id = str(recoverable["retirement_id"])
        _clear_exact_stale_lock(paths, transaction_id)
    lock = TransactionLock(
        paths,
        transaction_id=transaction_id,
        operation="retire-gguf",
    )
    try:
        owner = lock.acquire()
    except InspectorError as error:
        raise _retirement_error(
            (
                "RETIREMENT_CONCURRENCY_REJECTED"
                if error.reason_code == "TRANSACTION_LOCK_ACTIVE"
                else "RETIREMENT_ACTIVE_TRANSACTION"
            ),
            "retirement could not acquire exclusive ownership",
        ) from error
    if recoverable is None:
        revalidated = runtime_adapter.resolve_target(request)
        if (
            revalidated.result_projection()
            != target.result_projection()
        ):
            lock.release()
            raise _retirement_error(
                "RETIREMENT_TARGET_CHANGED",
                "retirement target changed while acquiring ownership",
            )
        transaction = _initial_transaction(
            paths,
            request,
            target,
            transaction_id=transaction_id,
            retirement_id=retirement_id,
            owner=owner,
        )
        runtime = dict(transaction["retirement_runtime"])
        runtime["request_activity_proof"] = activity
        transaction = _persist(
            paths,
            transaction,
            state="PREPARING",
            runtime=runtime,
            observer=transition_observer,
        )
    else:
        transaction = {
            **recoverable,
            "owner_identity": {
                key: owner.get(key)
                for key in (
                    "pid",
                    "process_start_identity",
                    "boot_identity",
                    "inspector_root_identity",
                )
            },
        }
        runtime = dict(transaction["retirement_runtime"])
        _write_transaction(paths, transaction, transition_observer)
    mutation_started = bool(
        runtime.get("alias_transaction") or runtime.get("quarantine")
    )
    last_model = target.is_last_ready
    try:
        if "TARGET_VERIFIED" not in runtime["states_observed"]:
            transaction = _persist(
                paths,
                transaction,
                state="TARGET_VERIFIED",
                runtime=runtime,
                observer=transition_observer,
            )
            _checkpoint(runtime_adapter, transaction)
        if not last_model and (
            "REPLACEMENT_VERIFIED"
            not in runtime["states_observed"]
        ):
            transaction = _persist(
                paths,
                transaction,
                state="REPLACEMENT_VERIFIED",
                runtime=runtime,
                observer=transition_observer,
            )
            _checkpoint(runtime_adapter, transaction)
        if (
            "REQUEST_ACTIVITY_CLEARED"
            not in runtime["states_observed"]
        ):
            transaction = _persist(
                paths,
                transaction,
                state="REQUEST_ACTIVITY_CLEARED",
                runtime=runtime,
                observer=transition_observer,
            )
            _checkpoint(runtime_adapter, transaction)
        if (
            "RETIREMENT_INTENT_RECORDED"
            not in runtime["states_observed"]
        ):
            transaction = _persist(
                paths,
                transaction,
                state="RETIREMENT_INTENT_RECORDED",
                runtime=runtime,
                intended_action="RETIRE_GGUF",
                observer=transition_observer,
            )
            _checkpoint(runtime_adapter, transaction)
        if last_model and not runtime.get("alias_transaction"):
            transaction = _persist(
                paths,
                transaction,
                state="DEFAULT_ALIAS_CLEARING",
                runtime=runtime,
                intended_action="CLEAR_DEFAULT_ALIAS",
                irreversible_action_begun="CLEAR_DEFAULT_ALIAS",
                observer=transition_observer,
            )
            runtime["alias_transaction"] = (
                runtime_adapter.clear_default(target, transaction_id)
            )
            mutation_started = True
            transaction = _persist(
                paths,
                transaction,
                state="DEFAULT_ALIAS_CLEARED",
                runtime=runtime,
                irreversible_action_observed="CLEAR_DEFAULT_ALIAS",
                observer=transition_observer,
            )
            _checkpoint(runtime_adapter, transaction)
        if not runtime.get("quarantine"):
            _validate_quarantine_root(target)
            transaction = _persist(
                paths,
                transaction,
                state="QUARANTINING",
                runtime=runtime,
                intended_action="QUARANTINE_TARGET",
                irreversible_action_begun="QUARANTINE_TARGET",
                observer=transition_observer,
            )
            before = _file_snapshot(
                target.target_path, hash_content=True
            )
            if (
                target.authenticated_content_sha256 is not None
                and before["sha256"]
                != target.authenticated_content_sha256
            ):
                raise _retirement_error(
                    "RETIREMENT_TARGET_CHANGED",
                    "target content identity changed before quarantine",
                )
            destination = _quarantine_destination(
                target, retirement_id
            )
            _rename_noreplace(target.target_path, destination)
            fsync_directory(target.target_path.parent)
            fsync_directory(destination.parent)
            after = _file_snapshot(destination, hash_content=True)
            if (
                target.target_path.exists()
                or _public_file_identity(before)
                != _public_file_identity(after)
            ):
                raise RetirementFailClosed(
                    "RETIREMENT_OWNERSHIP_UNCERTAIN",
                    "quarantine move identity is uncertain",
                    ownership_certain=False,
                )
            runtime["quarantine"] = {
                "path": str(destination),
                "file_identity": after,
            }
            runtime["quarantine_public"] = {
                "moved": True,
                "quarantine_identity": after["identity"],
                "file_identity": _public_file_identity(after),
            }
            mutation_started = True
            runtime_adapter.on_quarantined(
                target, runtime["quarantine"]
            )
            transaction = _persist(
                paths,
                transaction,
                state="QUARANTINED",
                runtime=runtime,
                irreversible_action_observed="QUARANTINE_TARGET",
                observer=transition_observer,
            )
            _checkpoint(runtime_adapter, transaction)
        elif not Path(runtime["quarantine"]["path"]).exists() and not runtime.get(
            "quarantine_deletion"
        ):
            raise RetirementFailClosed(
                "RETIREMENT_OWNERSHIP_UNCERTAIN",
                "recorded retirement quarantine is absent",
                ownership_certain=False,
            )
        if not runtime.get("registry_removal"):
            transaction = _persist(
                paths,
                transaction,
                state="REGISTRY_REMOVAL_OBSERVING",
                runtime=runtime,
                observer=transition_observer,
            )
            removal = runtime_adapter.observe_registry_removal(target)
            if (
                removal.get("observed") is not True
                or removal.get("catalogue_target_absent") is not True
                or removal.get("immutable_history_present") is not True
            ):
                raise _retirement_error(
                    "RETIREMENT_CATALOGUE_REMOVAL_FAILED",
                    "registry removal proof is incomplete",
                )
            runtime["registry_removal"] = removal
            runtime["catalogue_removal"] = {
                "target_absent": True,
                "immutable_history_present": True,
                "observed_registry_generation": removal.get(
                    "observed_registry_generation"
                ),
            }
            transaction = _persist(
                paths,
                transaction,
                state="REGISTRY_REMOVAL_OBSERVED",
                runtime=runtime,
                observer=transition_observer,
            )
            transaction = _persist(
                paths,
                transaction,
                state="CATALOGUE_REMOVAL_OBSERVED",
                runtime=runtime,
                observer=transition_observer,
            )
            _checkpoint(runtime_adapter, transaction)
        if not runtime.get("recovery"):
            transaction, recovery, final_service = _recover(
                paths,
                transaction,
                runtime,
                runtime_adapter,
                target,
                last_model=last_model,
                observer=transition_observer,
            )
            runtime["recovery"] = recovery
            runtime["poststate"] = final_service
            transaction = _persist(
                paths,
                transaction,
                state="POST_RETIREMENT_VERIFYING",
                runtime=runtime,
                observer=transition_observer,
            )
            _checkpoint(runtime_adapter, transaction)
        if last_model:
            if not runtime.get("waiting_for_model_health"):
                waiting = runtime_adapter.waiting_proof(target)
                if waiting.get("passed") is not True:
                    raise RetirementFailClosed(
                        "RETIREMENT_RECOVERY_FAILED",
                        "WAITING_FOR_MODEL proof did not pass",
                        ownership_certain=True,
                    )
                runtime["waiting_for_model_health"] = waiting
        elif not runtime.get("later_request"):
            later = runtime_adapter.later_request(target)
            if later.get("passed") is not True:
                raise RetirementFailClosed(
                    "RETIREMENT_RECOVERY_FAILED",
                    "post-retirement continuity request did not pass",
                    ownership_certain=True,
                )
            runtime["later_request"] = later
        if not runtime.get("quarantine_deletion"):
            transaction = _persist(
                paths,
                transaction,
                state="QUARANTINE_DELETING",
                runtime=runtime,
                intended_action="DELETE_EXACT_QUARANTINE",
                irreversible_action_begun="DELETE_EXACT_QUARANTINE",
                observer=transition_observer,
            )
            quarantine_path = Path(runtime["quarantine"]["path"])
            expected = runtime["quarantine"]["file_identity"]
            observed = _file_snapshot(
                quarantine_path, hash_content=True
            )
            if _public_file_identity(observed) != _public_file_identity(
                expected
            ):
                raise RetirementFailClosed(
                    "RETIREMENT_OWNERSHIP_UNCERTAIN",
                    "quarantine identity changed before deletion",
                    ownership_certain=False,
                )
            quarantine_path.unlink()
            fsync_directory(quarantine_path.parent)
            if quarantine_path.exists():
                raise RetirementFailClosed(
                    "RETIREMENT_QUARANTINE_DELETE_FAILED",
                    "exact quarantine file remained after deletion",
                    ownership_certain=True,
                )
            runtime["quarantine_deletion"] = {
                "deleted": True,
                "deleted_identity": observed["identity"],
            }
            runtime["quarantine_public"] = {
                **runtime["quarantine_public"],
                "deleted": True,
            }
            transaction = {
                **transaction,
                "retirement_runtime": runtime,
                "irreversible_action_observed": (
                    "DELETE_EXACT_QUARANTINE"
                ),
            }
            _write_transaction(
                paths, transaction, transition_observer
            )
            _checkpoint(runtime_adapter, transaction)
        return _finish(
            paths,
            transaction,
            runtime,
            request,
            target,
            result_class=(
                "RETIREMENT_WAITING_FOR_MODEL"
                if last_model
                else "RETIREMENT_COMPLETE"
            ),
            reason_codes=[
                (
                    "RETIREMENT_WAITING_FOR_MODEL"
                    if last_model
                    else "RETIREMENT_COMPLETE"
                )
            ],
            publisher=publisher,
            observer=transition_observer,
        )
    except BaseException as failure:
        if isinstance(failure, (KeyboardInterrupt, SystemExit)):
            raise
        trigger = (
            failure.reason_code
            if isinstance(failure, InspectorError)
            and failure.reason_code in RETIREMENT_REASON_CODES
            else failure.reason_code
            if isinstance(failure, RetirementFailClosed)
            else "RETIREMENT_INTERNAL_ERROR"
        )
        ownership_certain = not isinstance(
            failure, RetirementFailClosed
        ) or failure.ownership_certain
        restoration: dict[str, Any] = {}
        if mutation_started and ownership_certain:
            try:
                transaction = _persist(
                    paths,
                    transaction,
                    state="RESTORING_LOCATION",
                    runtime=runtime,
                    reason_code=trigger,
                    intended_action="RESTORE_LOCATION",
                    observer=transition_observer,
                )
                restoration = _restore_after_failure(
                    target,
                    runtime,
                    runtime_adapter,
                    transaction_id,
                )
            except BaseException as restore_failure:
                if isinstance(
                    restore_failure, (KeyboardInterrupt, SystemExit)
                ):
                    raise
                ownership_certain = False
                trigger = (
                    restore_failure.reason_code
                    if isinstance(
                        restore_failure, RetirementFailClosed
                    )
                    else "RETIREMENT_RESTORATION_FAILED"
                )
        if not mutation_started or (
            ownership_certain and restoration.get("proved") is True
        ):
            runtime["poststate"] = {
                "restoration": restoration,
                "safe": True,
            }
            return _finish(
                paths,
                transaction,
                runtime,
                request,
                target,
                result_class="RETIREMENT_FAILED_CLEAN",
                reason_codes=[
                    "RETIREMENT_FAILED_CLEAN",
                    trigger,
                ],
                publisher=publisher,
                observer=transition_observer,
            )
        runtime["poststate"] = {
            "safe": False,
            "fail_closed": True,
            "quarantine_retained": bool(runtime.get("quarantine")),
        }
        result = _finish(
            paths,
            transaction,
            runtime,
            request,
            target,
            result_class="RETIREMENT_FAIL_CLOSED",
            reason_codes=[
                "RETIREMENT_FAIL_CLOSED",
                trigger,
            ],
            publisher=publisher,
            observer=transition_observer,
        )
        raise _retirement_error(
            "RETIREMENT_FAIL_CLOSED",
            "retirement could not prove a safe terminal state",
            internal=True,
            data={
                "transaction_id": result[0],
                "retirement_id": result[1]["retirement_id"],
                "result_path": str(result[2]),
                "result_identity": result[3],
            },
        ) from failure
    finally:
        lock.release()
