"""Stable, standard-library-only platform-service adapter contract."""

from __future__ import annotations

from dataclasses import dataclass
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, NoReturn, Protocol


MANIFEST_SCHEMA = "system-x.platform-service-adapter-manifest.v1"
STATUS_SCHEMA = "system-x.platform-service-adapter-status.v1"
RESULT_SCHEMA = "system-x.platform-service-adapter-result.v1"

ADAPTER_IDENTITY = "system-x.foreground-process-host.v1"
ADAPTER_VERSION = "1.0.0"
PLATFORM_FAMILY = "portable-process-host"
ACTIVATION_METHOD = "foreground-process"
AUTOMATIC_ACTIVATION_SUPPORTED = False

OPERATIONS = (
    "identify",
    "validate",
    "register",
    "enable",
    "disable",
    "start",
    "stop",
    "restart",
    "status",
    "unregister",
    "capability",
    "configuration",
    "supervisor-entrypoint",
)

REQUIRED_HOST_CAPABILITIES = (
    "foreground_process_execution",
    "structured_argv",
    "process_identity_observation",
    "filesystem_atomic_replacement",
    "operator_controlled_signal_relay",
)

REASON_CODES = frozenset(
    (
        "OK",
        "ADAPTER_NOT_SUPPORTED",
        "ADAPTER_NOT_REGISTERED",
        "ADAPTER_ALREADY_REGISTERED",
        "ADAPTER_DISABLED",
        "ADAPTER_ALREADY_ENABLED",
        "ADAPTER_ALREADY_DISABLED",
        "ADAPTER_ALREADY_ACTIVE",
        "ADAPTER_INACTIVE",
        "ADAPTER_MANIFEST_INVALID",
        "ADAPTER_CONFIGURATION_CONFLICT",
        "HOST_CAPABILITY_MISSING",
        "SUPERVISOR_ENTRYPOINT_INVALID",
        "PROFILE_INVALID",
        "DESIRED_STATE_INVALID",
        "SUPERVISOR_IDENTITY_UNCERTAIN",
        "SUPERVISOR_START_FAILED",
        "SUPERVISOR_STOP_FAILED",
        "SUPERVISOR_RESTART_FAILED",
        "UNREGISTER_REQUIRES_DISABLED",
        "UNREGISTER_REQUIRES_INACTIVE",
        "UNREGISTER_NOT_EXPLICIT",
    )
)

CONFIGURATION_IDENTITY_FIELDS = (
    "adapter_identity",
    "adapter_version",
    "supported_platform_family",
    "activation_method",
    "supervisor_entrypoint_sha256",
    "profile_path",
    "state_path",
    "supervisor_runtime_root",
    "profile_identity",
)

MAX_MESSAGE_CHARACTERS = 4_096


class AdapterError(RuntimeError):
    """Stable domain error crossing the adapter machine boundary."""

    def __init__(
        self,
        reason_code: str,
        message: str,
        *,
        exit_code: int = 2,
        data: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = str(reason_code)
        self.message = str(message)[:MAX_MESSAGE_CHARACTERS]
        self.exit_code = int(exit_code)
        self.data = dict(data or {})


def fail(
    reason_code: str,
    message: str,
    *,
    exit_code: int = 2,
    data: Mapping[str, Any] | None = None,
) -> NoReturn:
    raise AdapterError(
        reason_code,
        message,
        exit_code=exit_code,
        data=data,
    )


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )


def canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        fail("ADAPTER_MANIFEST_INVALID", f"JSON serialization failed: {exc}")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def compute_configuration_identity(values: Mapping[str, Any]) -> str:
    """Identify exactly the immutable registration fields."""

    present = frozenset(values)
    expected = frozenset(CONFIGURATION_IDENTITY_FIELDS)
    if present != expected:
        missing = sorted(expected - present)
        unknown = sorted(present - expected)
        fail(
            "ADAPTER_CONFIGURATION_CONFLICT",
            (
                "configuration identity fields are incomplete or unknown: "
                f"missing={missing}, unknown={unknown}"
            ),
        )
    payload = {
        "schema_version": "system-x.platform-service-adapter-configuration.v1",
        **{name: values[name] for name in CONFIGURATION_IDENTITY_FIELDS},
    }
    return "sha256:" + sha256_bytes(canonical_json(payload).encode("utf-8"))


def bounded_activation_result(
    *,
    operation: str,
    ok: bool,
    reason_code: str,
    message: str,
    supervisor_exit_status: int | None = None,
) -> dict[str, Any]:
    return {
        "operation": operation,
        "ok": bool(ok),
        "reason_code": str(reason_code)[:128],
        "message": str(message)[:MAX_MESSAGE_CHARACTERS],
        "timestamp_utc": utc_now(),
        "supervisor_exit_status": supervisor_exit_status,
    }


@dataclass(frozen=True, slots=True)
class AdapterPathsView:
    manifest_path: Path | None
    status_path: Path | None


class PlatformServiceAdapter(Protocol):
    """Common interface implemented by every bounded adapter."""

    adapter_identity: str
    adapter_version: str

    def invoke(self, operation: str, **arguments: Any) -> dict[str, Any]:
        """Execute one operation and return one result envelope."""


def result_envelope(
    operation: str,
    *,
    ok: bool,
    reason_code: str,
    message: str,
    adapter_identity: str = ADAPTER_IDENTITY,
    adapter_version: str = ADAPTER_VERSION,
    automatic_activation_supported: bool = (
        AUTOMATIC_ACTIVATION_SUPPORTED
    ),
    configuration_identity: str | None = None,
    registered: bool = False,
    enabled: bool = False,
    active: bool = False,
    profile_identity: str | None = None,
    desired_state: str | None = None,
    desired_state_generation: int | None = None,
    paths: AdapterPathsView | None = None,
    data: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": RESULT_SCHEMA,
        "operation": operation,
        "ok": bool(ok),
        "reason_code": str(reason_code),
        "message": str(message)[:MAX_MESSAGE_CHARACTERS],
        "timestamp_utc": utc_now(),
        "adapter_identity": adapter_identity,
        "adapter_version": adapter_version,
        "configuration_identity": configuration_identity,
        "registered": bool(registered),
        "enabled": bool(enabled),
        "active": bool(active),
        "automatic_activation_supported": bool(
            automatic_activation_supported
        ),
        "profile_identity": profile_identity,
        "desired_state": desired_state,
        "desired_state_generation": desired_state_generation,
        "manifest_path": (
            str(paths.manifest_path)
            if paths is not None and paths.manifest_path is not None
            else None
        ),
        "status_path": (
            str(paths.status_path)
            if paths is not None and paths.status_path is not None
            else None
        ),
        "data": dict(data or {}),
    }
