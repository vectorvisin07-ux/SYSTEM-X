#!/usr/bin/env python3
"""Portable foreground supervisor for the existing System X controller graph.

The supervisor owns no public listener and launches no service executable.  It
uses the existing API-service controller for start/stop and the existing branch
controller for private-runtime observation.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import secrets
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, NoReturn, Sequence

if __package__:
    from .operating_profile import (
        DEFAULT_DESIRED_STATE_PATH,
        DEFAULT_PROFILE_PATH,
        DesiredState,
        OperatingProfile,
        ServiceControlError,
        load_desired_state,
        load_operating_profile,
        set_desired_state,
    )
    from .recovery import (
        RecoveryAttempt,
        RecoveryError,
        RecoveryPolicy,
        RecoveryStore,
    )
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from service_control.operating_profile import (  # type: ignore
        DEFAULT_DESIRED_STATE_PATH,
        DEFAULT_PROFILE_PATH,
        DesiredState,
        OperatingProfile,
        ServiceControlError,
        load_desired_state,
        load_operating_profile,
        set_desired_state,
    )
    from service_control.recovery import (  # type: ignore
        RecoveryAttempt,
        RecoveryError,
        RecoveryPolicy,
        RecoveryStore,
    )


STATUS_SCHEMA = "system-x.service-supervisor-status.v1"
TRANSACTION_SCHEMA = "system-x.service-supervisor-transaction.v1"
RESULT_SCHEMA = "system-x.service-supervisor-result.v1"
LOCK_SCHEMA = "system-x.service-supervisor-lock.v1"
PID_SCHEMA = "system-x.service-supervisor-pid.v1"
LOG_SCHEMA = "system-x.service-supervisor-log-event.v1"

API_CONTROLLER_SCHEMA = "system-x.gguf-api-service-controller.v1"
BRANCH_CONTROLLER_SCHEMA = "system-x.gguf-branch-controller.v1"
API_CONTROLLER_SHA256 = (
    "f20958d5f032101f96400ddc1283d66885b6e663cb474797e460d94a31727b06"
)
BRANCH_CONTROLLER_SHA256 = (
    "6dd54f90980d146115b901efc018dad108a7c944ecf48151c7c14503910401f7"
)

SUPERVISOR_STATES = frozenset(
    ("STARTING", "RUNNING", "STOPPING", "STOPPED", "FAULTED")
)
SERVICE_READINESS_STATES = frozenset(
    (
        "SUPERVISOR_STARTING",
        "API_STARTING",
        "BACKEND_STARTING",
        "MODEL_LOADING",
        "WAITING_FOR_MODEL",
        "MODEL_CANDIDATE_LOADING",
        "READY",
        "DEGRADED",
        "STOPPED",
        "FAIL_CLOSED",
    )
)
MODEL_SERVICE_STATES = frozenset(
    (
        "WAITING_FOR_MODEL",
        "MODEL_CANDIDATE_LOADING",
        "READY",
        "DEGRADED",
        "STOPPED",
        "FAIL_CLOSED",
    )
)
MAX_JSON_BYTES = 1_048_576
MAX_CONTROLLER_OUTPUT_BYTES = 1_048_576
MAX_HEALTH_BODY_BYTES = 1_048_576
MAX_MESSAGE_CHARACTERS = 4_096

SOURCE_DIR = Path(__file__).resolve().parent
BRANCH_ROOT = SOURCE_DIR.parent
DEFAULT_RUNTIME_ROOT = BRANCH_ROOT / "RUNTIME" / "service_control"
DEFAULT_API_CONTROLLER = BRANCH_ROOT / "api_service_controller" / "controller.py"
DEFAULT_BRANCH_CONTROLLER = BRANCH_ROOT / "branch_controller" / "controller.py"


class SupervisorError(RuntimeError):
    """Stable supervisor-domain failure."""

    def __init__(
        self,
        reason_code: str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
        exit_code: int = 2,
    ):
        super().__init__(message)
        self.reason_code = reason_code
        self.message = _bounded_text(message)
        self.details = dict(details or {})
        self.exit_code = exit_code


def _fail(
    reason_code: str,
    message: str,
    *,
    details: Mapping[str, Any] | None = None,
    exit_code: int = 2,
) -> NoReturn:
    raise SupervisorError(
        reason_code, message, details=details, exit_code=exit_code
    )


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )


def _bounded_text(value: Any) -> str:
    text = str(value)
    return text[:MAX_MESSAGE_CHARACTERS]


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        _fail("serialization_failed", f"JSON serialization failed: {exc}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(65_536)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def _reject_symlink_components(path: Path, label: str) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    for component in list(reversed(absolute.parents)) + [absolute]:
        try:
            metadata = component.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            _fail("path_inspection_failed", f"{label} cannot be inspected: {exc}")
        if stat.S_ISLNK(metadata.st_mode):
            _fail("symlink_rejected", f"{label} contains a symlink component")


def _ensure_directory(path: Path, label: str) -> None:
    _reject_symlink_components(path.parent, label)
    try:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError as exc:
        _fail("directory_creation_failed", f"{label} cannot be created: {exc}")
    _reject_symlink_components(path, label)
    if not path.is_dir():
        _fail("invalid_directory", f"{label} must be a directory")


def _read_json_file(path: Path, label: str) -> dict[str, Any]:
    path = Path(path)
    _reject_symlink_components(path, label)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        _fail("missing_record", f"{label} does not exist")
    except OSError as exc:
        _fail("record_open_failed", f"{label} cannot be opened: {exc}")
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            _fail("invalid_record", f"{label} must be a regular file")
        if metadata.st_size > MAX_JSON_BYTES:
            _fail("record_too_large", f"{label} exceeds the size bound")
        data = b""
        remaining = MAX_JSON_BYTES + 1
        while remaining:
            block = os.read(descriptor, min(65_536, remaining))
            if not block:
                break
            data += block
            remaining -= len(block)
    finally:
        os.close(descriptor)
    if len(data) > MAX_JSON_BYTES:
        _fail("record_too_large", f"{label} exceeds the size bound")
    if b"\0" in data:
        _fail("invalid_record", f"{label} contains a NUL byte")
    try:
        value = json.loads(data.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail("invalid_record", f"{label} is not valid UTF-8 JSON: {exc}")
    if not isinstance(value, dict):
        _fail("invalid_record", f"{label} must contain one JSON object")
    return value


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        _fail("directory_sync_failed", f"directory open failed: {exc}")
    try:
        os.fsync(descriptor)
    except OSError as exc:
        _fail("directory_sync_failed", f"directory sync failed: {exc}")
    finally:
        os.close(descriptor)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(os.path.abspath(os.fspath(path)))
    _ensure_directory(path.parent, f"{path.name} parent")
    try:
        existing = path.lstat()
    except FileNotFoundError:
        existing = None
    if existing is not None:
        if stat.S_ISLNK(existing.st_mode):
            _fail("symlink_rejected", f"{path.name} must not be a symlink")
        if not stat.S_ISREG(existing.st_mode):
            _fail("invalid_record", f"{path.name} must be regular or absent")
    payload = (_canonical_json(value) + "\n").encode("utf-8")
    descriptor: int | None = None
    temporary_path: str | None = None
    try:
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        os.fchmod(descriptor, 0o600)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(temporary_path, path)
        temporary_path = None
        _fsync_directory(path.parent)
    except SupervisorError:
        raise
    except OSError as exc:
        _fail("atomic_write_failed", f"atomic write failed: {exc}")
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass
            except OSError:
                pass


def _exclusive_write_json(path: Path, value: Mapping[str, Any]) -> None:
    _ensure_directory(path.parent, f"{path.name} parent")
    _reject_symlink_components(path, path.name)
    payload = (_canonical_json(value) + "\n").encode("utf-8")
    try:
        descriptor = os.open(
            path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
    except FileExistsError:
        _fail("supervisor_lock_active", f"{path.name} already exists")
    except OSError as exc:
        _fail("record_create_failed", f"{path.name} cannot be created: {exc}")
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write")
            view = view[written:]
        os.fsync(descriptor)
    except OSError as exc:
        try:
            os.unlink(path)
        except OSError:
            pass
        _fail("record_create_failed", f"{path.name} write failed: {exc}")
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _unlink_owned(path: Path, transaction_id: str) -> bool:
    try:
        value = _read_json_file(path, path.name)
    except SupervisorError as exc:
        if exc.reason_code == "missing_record":
            return False
        raise
    if value.get("supervisor_transaction_id") != transaction_id:
        _fail(
            "record_ownership_mismatch",
            f"{path.name} belongs to another supervisor transaction",
        )
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    except OSError as exc:
        _fail("record_cleanup_failed", f"{path.name} cannot be removed: {exc}")
    _fsync_directory(path.parent)
    return True


def _append_event(path: Path, event: Mapping[str, Any]) -> None:
    _ensure_directory(path.parent, "supervisor log directory")
    _reject_symlink_components(path, "supervisor log")
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        _fail("log_open_failed", f"supervisor log cannot be opened: {exc}")
    try:
        os.fchmod(descriptor, 0o600)
        payload = (_canonical_json(event) + "\n").encode("utf-8")
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short log write")
            view = view[written:]
        os.fsync(descriptor)
    except OSError as exc:
        _fail("log_write_failed", f"supervisor log write failed: {exc}")
    finally:
        os.close(descriptor)


def process_snapshot(pid: int) -> dict[str, Any]:
    """Return a fail-closed process identity on hosts exposing process metadata."""

    if type(pid) is not int or pid < 1:
        _fail("invalid_process_identity", "PID must be a positive integer")
    process_dir = Path("/proc") / str(pid)
    try:
        raw_stat = (process_dir / "stat").read_text(encoding="ascii")
        end = raw_stat.rfind(")")
        fields = raw_stat[end + 2 :].split()
        start_ticks = fields[19]
        executable = str((process_dir / "exe").resolve(strict=True))
        argv_bytes = (process_dir / "cmdline").read_bytes()
    except (FileNotFoundError, PermissionError, OSError, IndexError) as exc:
        _fail(
            "process_identity_unavailable",
            f"exact process identity is unavailable for PID {pid}: {exc}",
        )
    if b"\0" in argv_bytes:
        argv_parts = [part for part in argv_bytes.split(b"\0") if part]
    else:
        argv_parts = [argv_bytes] if argv_bytes else []
    argv_digest = hashlib.sha256(b"\0".join(argv_parts)).hexdigest()
    try:
        pgid = os.getpgid(pid)
        sid = os.getsid(pid)
    except (ProcessLookupError, PermissionError, OSError) as exc:
        _fail(
            "process_identity_unavailable",
            f"process group identity is unavailable for PID {pid}: {exc}",
        )
    return {
        "pid": pid,
        "process_start_identity": f"procfs-start-ticks:{start_ticks}",
        "pgid": pgid,
        "sid": sid,
        "executable": executable,
        "argv_sha256": argv_digest,
    }


def _same_process_identity(
    expected: Mapping[str, Any], observed: Mapping[str, Any]
) -> bool:
    return all(
        expected.get(name) == observed.get(name)
        for name in (
            "pid",
            "process_start_identity",
            "pgid",
            "sid",
            "executable",
            "argv_sha256",
        )
    )


def _matching_processes_for_record(
    record: Mapping[str, Any], *, exclude_pid: int | None = None
) -> list[dict[str, Any]]:
    """Find complete supervisor executable/argv-digest matches."""

    executable = record.get("executable")
    argv_sha256 = record.get("argv_sha256")
    if not isinstance(executable, str) or not isinstance(argv_sha256, str):
        return []
    matches: list[dict[str, Any]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == exclude_pid:
            continue
        try:
            observed = process_snapshot(pid)
        except SupervisorError:
            continue
        if (
            observed.get("executable") == executable
            and observed.get("argv_sha256") == argv_sha256
        ):
            matches.append(observed)
    return sorted(matches, key=lambda value: int(value["pid"]))


@dataclass(frozen=True)
class SupervisorPaths:
    runtime_root: Path

    @property
    def logs(self) -> Path:
        return self.runtime_root / "logs"

    @property
    def pids(self) -> Path:
        return self.runtime_root / "pids"

    @property
    def locks(self) -> Path:
        return self.runtime_root / "locks"

    @property
    def status(self) -> Path:
        return self.runtime_root / "status"

    @property
    def transactions(self) -> Path:
        return self.runtime_root / "transactions"

    @property
    def recovery(self) -> Path:
        return self.runtime_root / "recovery"

    @property
    def active_pid(self) -> Path:
        return self.pids / "supervisor.json"

    @property
    def active_lock(self) -> Path:
        return self.locks / "supervisor.lock"

    @property
    def status_record(self) -> Path:
        return self.status / "supervisor.json"

    def transaction(self, transaction_id: str) -> Path:
        return self.transactions / f"{transaction_id}.json"

    def log(self, transaction_id: str) -> Path:
        return self.logs / f"{transaction_id}.jsonl"

    def as_dict(self) -> dict[str, str]:
        return {
            "runtime_root": str(self.runtime_root),
            "active_pid": str(self.active_pid),
            "active_lock": str(self.active_lock),
            "status": str(self.status_record),
            "transactions": str(self.transactions),
            "logs": str(self.logs),
            "recovery": str(self.recovery),
        }


def _recovery_policy(profile: OperatingProfile) -> RecoveryPolicy:
    return RecoveryPolicy(
        automatic_recovery_enabled=profile.automatic_recovery_enabled,
        initial_delay_seconds=profile.recovery_delay_initial_seconds,
        maximum_delay_seconds=profile.recovery_delay_maximum_seconds,
        delay_multiplier=profile.recovery_delay_multiplier,
        maximum_attempts_in_window=(
            profile.recovery_maximum_attempts_in_window
        ),
        attempt_window_seconds=profile.recovery_attempt_window_seconds,
        stable_reset_seconds=profile.recovery_stable_reset_seconds,
    )


class ControllerAdapter:
    """Bounded structured-argv adapter for the two existing controllers."""

    def __init__(
        self,
        api_controller: Path = DEFAULT_API_CONTROLLER,
        branch_controller: Path = DEFAULT_BRANCH_CONTROLLER,
        *,
        timeout_seconds: float = 180.0,
        api_sha256: str | None = API_CONTROLLER_SHA256,
        branch_sha256: str | None = BRANCH_CONTROLLER_SHA256,
        python_executable: str = sys.executable,
    ):
        if (
            type(timeout_seconds) not in (int, float)
            or not math.isfinite(float(timeout_seconds))
            or not 0.05 <= float(timeout_seconds) <= 300.0
        ):
            _fail(
                "invalid_controller_timeout",
                "controller timeout must be finite and in 0.05..300 seconds",
            )
        self.api_controller = Path(api_controller)
        self.branch_controller = Path(branch_controller)
        self.timeout_seconds = float(timeout_seconds)
        self.api_sha256 = api_sha256
        self.branch_sha256 = branch_sha256
        self.python_executable = python_executable

    def validate(self) -> dict[str, Any]:
        dependencies = {}
        for kind, path, expected in (
            ("api", self.api_controller, self.api_sha256),
            ("branch", self.branch_controller, self.branch_sha256),
        ):
            _reject_symlink_components(path, f"{kind} controller")
            if not path.is_file():
                _fail(
                    "controller_dependency_missing",
                    f"{kind} controller is not a regular file",
                )
            identity = _sha256_file(path)
            if expected is not None and identity != expected:
                _fail(
                    "controller_identity_mismatch",
                    f"{kind} controller source identity changed",
                    details={"expected": expected, "observed": identity},
                )
            dependencies[kind] = {
                "path": str(path.resolve(strict=True)),
                "sha256": identity,
            }
        return dependencies

    def invoke(
        self, kind: str, operation: str, arguments: Sequence[str] = ()
    ) -> dict[str, Any]:
        if kind == "api":
            path = self.api_controller
            expected_schema = API_CONTROLLER_SCHEMA
        elif kind == "branch":
            path = self.branch_controller
            expected_schema = BRANCH_CONTROLLER_SCHEMA
        else:
            _fail("invalid_controller_kind", f"unknown controller kind: {kind}")
        argv = [
            self.python_executable,
            str(path),
            operation,
            *[str(value) for value in arguments],
        ]
        try:
            completed = subprocess.run(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout_seconds,
                check=False,
                shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            _fail(
                "controller_timeout",
                f"{kind} controller {operation} exceeded the timeout",
                details={"timeout_seconds": self.timeout_seconds},
            )
        if (
            len(completed.stdout) > MAX_CONTROLLER_OUTPUT_BYTES
            or len(completed.stderr) > MAX_CONTROLLER_OUTPUT_BYTES
        ):
            _fail(
                "controller_output_too_large",
                f"{kind} controller output exceeded the bound",
            )
        stdout = completed.stdout.decode("utf-8", errors="strict")
        stderr = completed.stderr.decode("utf-8", errors="replace")
        lines = [line for line in stdout.splitlines() if line.strip()]
        if len(lines) != 1:
            _fail(
                "controller_malformed_output",
                f"{kind} controller must emit exactly one JSON object",
                details={
                    "returncode": completed.returncode,
                    "stderr": _bounded_text(stderr),
                },
            )
        try:
            result = json.loads(lines[0])
        except json.JSONDecodeError as exc:
            _fail(
                "controller_malformed_output",
                f"{kind} controller emitted invalid JSON: {exc}",
                details={
                    "returncode": completed.returncode,
                    "stderr": _bounded_text(stderr),
                },
            )
        if not isinstance(result, dict):
            _fail(
                "controller_malformed_output",
                f"{kind} controller output must be a JSON object",
            )
        if result.get("schema_version") != expected_schema:
            _fail(
                "controller_schema_mismatch",
                f"{kind} controller schema identity is invalid",
                details={"controller_result": result},
            )
        if result.get("operation") != operation:
            _fail(
                "controller_operation_mismatch",
                f"{kind} controller operation identity is invalid",
                details={"controller_result": result},
            )
        if completed.returncode != 0 or result.get("ok") is not True:
            _fail(
                "controller_operation_failed",
                (
                    f"{kind} controller {operation} failed: "
                    f"{result.get('reason_code')}"
                ),
                details={
                    "returncode": completed.returncode,
                    "stderr": _bounded_text(stderr),
                    "controller_result": result,
                },
                exit_code=3,
            )
        return result


def _startup_configuration(
    profile: OperatingProfile, startup_model_policy: str | None
) -> tuple[str, bool, bool]:
    effective_policy = startup_model_policy or profile.startup_model_policy
    if effective_policy not in {"always_warm", "router_control"}:
        _fail(
            "unsupported_startup_model_policy",
            f"unsupported startup_model_policy: {effective_policy}",
        )
    registry_enabled = effective_policy != "router_control"
    automatic_recovery_enabled = (
        profile.automatic_recovery_enabled
        if effective_policy != "router_control"
        else False
    )
    return effective_policy, registry_enabled, automatic_recovery_enabled


def _api_arguments(
    operation: str,
    profile: OperatingProfile,
    state_path: Path = DEFAULT_DESIRED_STATE_PATH,
    *,
    startup_model_policy: str | None = None,
) -> list[str]:
    (
        effective_policy,
        registry_enabled,
        automatic_recovery_enabled,
    ) = _startup_configuration(profile, startup_model_policy)
    return [
        "--host",
        profile.public_endpoint.host,
        "--port",
        str(profile.public_endpoint.port),
        "--private-backend-host",
        profile.private_router_endpoint.host,
        "--private-backend-port",
        str(profile.private_router_endpoint.port),
        "--private-backend-enabled",
        "true",
        "--private-backend-models-max",
        "1",
        "--private-backend-start-timeout-seconds",
        "30",
        "--private-backend-model-timeout-seconds",
        "120",
        "--private-backend-inference-timeout-seconds",
        "900",
        "--private-backend-poll-interval-seconds",
        "0.25",
        "--registry-enabled",
        "true" if registry_enabled else "false",
        "--registry-default-alias",
        profile.default_model_alias,
        "--startup-model-policy",
        effective_policy,
        "--automatic-recovery-enabled",
        "true" if automatic_recovery_enabled else "false",
        "--recovery-delay-initial-seconds",
        str(profile.recovery_delay_initial_seconds),
        "--recovery-delay-maximum-seconds",
        str(profile.recovery_delay_maximum_seconds),
        "--recovery-delay-multiplier",
        str(profile.recovery_delay_multiplier),
        "--recovery-maximum-attempts-in-window",
        str(profile.recovery_maximum_attempts_in_window),
        "--recovery-attempt-window-seconds",
        str(profile.recovery_attempt_window_seconds),
        "--recovery-stable-reset-seconds",
        str(profile.recovery_stable_reset_seconds),
        "--service-control-profile-identity",
        profile.identity,
        "--service-control-desired-state-path",
        str(Path(state_path).resolve(strict=True)),
        "--external-static-enabled",
        "false",
        "--log-level",
        "info",
    ]


def _verify_api_plan(
    result: Mapping[str, Any],
    profile: OperatingProfile,
    *,
    startup_model_policy: str | None = None,
) -> None:
    (
        effective_policy,
        registry_enabled,
        automatic_recovery_enabled,
    ) = _startup_configuration(profile, startup_model_policy)
    values = result.get("input")
    plan = result.get("plan")
    if not isinstance(values, dict) or not isinstance(plan, dict):
        _fail("api_plan_mismatch", "API controller plan lacks required sections")
    expected = {
        "host": profile.public_endpoint.host,
        "port": profile.public_endpoint.port,
        "private_backend_host": profile.private_router_endpoint.host,
        "private_backend_port": profile.private_router_endpoint.port,
        "private_backend_enabled": True,
        "private_backend_models_max": 1,
        "registry_enabled": registry_enabled,
        "registry_default_alias": profile.default_model_alias,
        "startup_model_policy": effective_policy,
        "automatic_recovery_enabled": automatic_recovery_enabled,
        "recovery_maximum_attempts_in_window": (
            profile.recovery_maximum_attempts_in_window
        ),
        "recovery_attempt_window_seconds": (
            profile.recovery_attempt_window_seconds
        ),
        "recovery_stable_reset_seconds": (
            profile.recovery_stable_reset_seconds
        ),
        "service_control_profile_identity": profile.identity,
    }
    if any(values.get(name) != value for name, value in expected.items()):
        _fail(
            "api_plan_mismatch",
            "API controller plan does not match the operating profile",
            details={"expected": expected, "observed": values},
        )
    if plan.get("shell") is not False or plan.get("start_new_session") is not True:
        _fail(
            "api_plan_mismatch",
            "API controller plan violates process containment",
        )


def _listener_owned(listener: Any, active: bool) -> bool:
    if not active:
        return False
    if not isinstance(listener, dict):
        return False
    for name in ("ownership_matches", "owned", "process_group_matches"):
        if name in listener:
            return listener[name] is True
    return True


def _observed_api(
    result: Mapping[str, Any], profile: OperatingProfile
) -> dict[str, Any]:
    runtime = result.get("runtime")
    if not isinstance(runtime, dict):
        _fail("api_status_invalid", "API controller result lacks runtime")
    active = runtime.get("active") is True
    return {
        "lifecycle_state": runtime.get("lifecycle_state"),
        "active": active,
        "consistent": runtime.get("consistent") is True,
        "transaction_id": runtime.get("transaction_id"),
        "pid": runtime.get("pid"),
        "process_start_identity": runtime.get("process_start_identity"),
        "pgid": runtime.get("pgid"),
        "sid": runtime.get("sid"),
        "endpoint": profile.public_endpoint.as_dict(),
        "listener_owned": _listener_owned(result.get("listener"), active),
    }


def _observed_router(
    result: Mapping[str, Any], profile: OperatingProfile
) -> dict[str, Any]:
    data = result.get("data")
    if not isinstance(data, dict):
        _fail("router_status_invalid", "branch controller result lacks data")
    active = data.get("active") is True
    endpoint = data.get("endpoint")
    expected_endpoint = profile.private_router_endpoint.as_dict()
    endpoint_matches = endpoint == expected_endpoint if active else endpoint is None
    return {
        "lifecycle_state": data.get("lifecycle_state"),
        "active": active,
        "consistent": data.get("active_state_consistent") is True,
        "transaction_id": data.get("transaction_id"),
        "pid": data.get("pid"),
        "process_start_identity": data.get("process_start_identity"),
        "pgid": data.get("pgid"),
        "sid": data.get("sid"),
        "endpoint": endpoint if active else expected_endpoint,
        "listener_owned": bool(active and endpoint_matches),
    }


def _public_health(profile: OperatingProfile) -> dict[str, Any]:
    """Read the public readiness contract without authentication or mutation."""

    connection = http.client.HTTPConnection(
        profile.public_endpoint.host,
        profile.public_endpoint.port,
        timeout=5.0,
    )
    try:
        connection.request(
            "GET",
            "/system/v1/health?detail=true",
            headers={"Accept": "application/json", "Connection": "close"},
        )
        response = connection.getresponse()
        payload = response.read(MAX_HEALTH_BODY_BYTES + 1)
        status_code = response.status
    except (OSError, TimeoutError, http.client.HTTPException) as exc:
        _fail(
            "public_health_unavailable",
            f"public health observation failed: {type(exc).__name__}",
        )
    finally:
        connection.close()
    if len(payload) > MAX_HEALTH_BODY_BYTES:
        _fail(
            "public_health_invalid",
            "public health response exceeded the body bound",
        )
    try:
        value = json.loads(payload.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(
            "public_health_invalid",
            f"public health response was invalid JSON: {exc}",
        )
    if not isinstance(value, dict):
        _fail("public_health_invalid", "public health response is not an object")
    state = value.get("model_service_state")
    compatibility_state = value.get("service_readiness_state")
    service_available = value.get("service_available")
    inference_ready = value.get("inference_ready")
    ready = value.get("ready")
    if (
        state not in MODEL_SERVICE_STATES
        or compatibility_state != state
        or type(service_available) is not bool
        or type(inference_ready) is not bool
        or type(ready) is not bool
        or ready is not inference_ready
    ):
        _fail(
            "public_health_invalid",
            "public health readiness fields are invalid",
        )
    operational = state in {
        "WAITING_FOR_MODEL",
        "MODEL_CANDIDATE_LOADING",
        "READY",
    }
    if status_code not in {200, 503} or (
        (status_code == 200) != operational
        or service_available is not operational
        or inference_ready is not (state == "READY")
    ):
        _fail(
            "public_health_invalid",
            "public health HTTP status violates the readiness contract",
        )
    return {
        "http_status": status_code,
        "body": value,
        "observed_utc": _utc_now(),
    }


def _degraded_health(reason_code: str) -> dict[str, Any]:
    return {
        "http_status": None,
        "body": {
            "service_readiness_state": "DEGRADED",
            "model_service_state": "DEGRADED",
            "service_available": False,
            "inference_ready": False,
            "ready": False,
            "reason_code": reason_code,
            "warm_identity": None,
        },
        "observed_utc": _utc_now(),
    }


def _warm_observation(
    health: Mapping[str, Any],
    router: Mapping[str, Any],
) -> tuple[str, dict[str, Any] | None, dict[str, Any]]:
    body = health.get("body")
    if not isinstance(body, dict):
        _fail("public_health_invalid", "public health body is absent")
    readiness = body.get("model_service_state")
    if (
        readiness not in MODEL_SERVICE_STATES
        or body.get("service_readiness_state") != readiness
    ):
        _fail("public_health_invalid", "service readiness state is invalid")
    warm = body.get("warm_identity")
    model = {"present": False, "identity": None}
    if warm is not None:
        if not isinstance(warm, dict):
            _fail("public_health_invalid", "warm identity is invalid")
        fields = {
            "pid": warm.get("model_child_pid"),
            "process_start_identity": warm.get(
                "model_child_start_identity"
            ),
            "ppid": warm.get("model_child_parent"),
            "pgid": warm.get("model_child_process_group"),
            "sid": warm.get("model_child_session"),
        }
        if (
            type(fields["pid"]) is not int
            or fields["pid"] < 1
            or not isinstance(fields["process_start_identity"], str)
            or not fields["process_start_identity"]
            or type(fields["ppid"]) is not int
            or type(fields["pgid"]) is not int
            or type(fields["sid"]) is not int
        ):
            _fail("public_health_invalid", "model-child identity is invalid")
        model = {"present": True, "identity": fields}
    if readiness == "READY":
        if (
            body.get("ready") is not True
            or health.get("http_status") != 200
            or not isinstance(warm, dict)
            or warm.get("health_state") != "ready"
            or warm.get("router_transaction_id")
            != router.get("transaction_id")
            or model["identity"]["ppid"] != router.get("pid")
            or model["identity"]["pgid"] != router.get("pgid")
            or model["identity"]["sid"] != router.get("sid")
        ):
            _fail(
                "warm_identity_mismatch",
                "READY warm identity does not belong to the owned router",
            )
    elif readiness in {"WAITING_FOR_MODEL", "MODEL_CANDIDATE_LOADING"}:
        if (
            health.get("http_status") != 200
            or body.get("service_available") is not True
            or body.get("inference_ready") is not False
            or body.get("ready") is not False
            or warm is not None
            or model["present"]
        ):
            _fail(
                "public_health_invalid",
                "no-model operational health evidence is inconsistent",
            )
    return readiness, dict(warm) if isinstance(warm, dict) else None, model


def _identity_subset(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: value.get(name)
        for name in (
            "pid",
            "process_start_identity",
            "ppid",
            "pgid",
            "sid",
        )
        if name in value
    }


def _observed_model_child(branch_result: Mapping[str, Any]) -> dict[str, Any]:
    data = branch_result.get("data")
    if not isinstance(data, dict):
        _fail("router_status_invalid", "branch controller result lacks data")
    candidates: list[Mapping[str, Any]] = []
    for name in ("observed_model_child", "loaded_model_child", "model_child"):
        value = data.get(name)
        if isinstance(value, dict):
            if value.get("present") is False:
                return {"present": False, "identity": None}
            identity = value.get("identity")
            candidates.append(identity if isinstance(identity, dict) else value)
    children = data.get("runtime_children")
    if isinstance(children, dict):
        iterable = children.values()
    elif isinstance(children, list):
        iterable = children
    else:
        iterable = ()
    for child in iterable:
        if not isinstance(child, dict):
            continue
        role = " ".join(
            str(child.get(name, "")).lower()
            for name in ("role", "kind", "type", "name")
        )
        if "model" in role:
            candidates.append(child)
    if not candidates:
        return {"present": False, "identity": None}
    identity = _identity_subset(candidates[0])
    if not identity.get("pid") or not identity.get("process_start_identity"):
        _fail(
            "model_child_identity_invalid",
            "reported model child lacks PID/start identity",
        )
    return {"present": True, "identity": identity}


def _same_runtime_identity(
    expected: Mapping[str, Any], observed: Mapping[str, Any]
) -> bool:
    return all(
        expected.get(name) == observed.get(name)
        for name in (
            "transaction_id",
            "pid",
            "process_start_identity",
            "pgid",
        )
    ) and expected.get("endpoint") == observed.get("endpoint")


class ForegroundSupervisor:
    def __init__(
        self,
        profile_path: Path,
        state_path: Path,
        runtime_root: Path,
        adapter: ControllerAdapter,
        *,
        monitor_interval_seconds: float = 0.25,
        startup_model_policy: str | None = None,
        install_signal_handlers: bool = True,
        health_observer: (
            Callable[[OperatingProfile], dict[str, Any]] | None
        ) = None,
    ):
        if (
            type(monitor_interval_seconds) not in (int, float)
            or not math.isfinite(float(monitor_interval_seconds))
            or not 0.01 <= float(monitor_interval_seconds) <= 10.0
        ):
            _fail(
                "invalid_monitor_interval",
                "monitor interval must be finite and in 0.01..10 seconds",
            )
        self.profile_path = Path(profile_path)
        self.state_path = Path(state_path)
        self.paths = SupervisorPaths(
            Path(os.path.abspath(os.fspath(runtime_root)))
        )
        self.adapter = adapter
        self.monitor_interval_seconds = float(monitor_interval_seconds)
        self.startup_model_policy = startup_model_policy
        self.install_signal_handlers = install_signal_handlers
        self.health_observer = health_observer or _public_health
        self.shutdown_requested = threading.Event()
        self.transaction_id: str | None = None
        self.supervisor_identity: dict[str, Any] | None = None
        self.startup_reconciliation: dict[str, Any] | None = None
        self._previous_signal_handlers: dict[int, Any] = {}

    def request_graceful_shutdown(self) -> None:
        self.shutdown_requested.set()

    def _install_signals(self) -> None:
        if not self.install_signal_handlers:
            return

        def handler(_signum: int, _frame: Any) -> None:
            self.shutdown_requested.set()

        try:
            for signal_number in (signal.SIGTERM, signal.SIGINT):
                self._previous_signal_handlers[signal_number] = signal.getsignal(
                    signal_number
                )
                signal.signal(signal_number, handler)
        except ValueError:
            _fail(
                "signal_handler_unavailable",
                "signal handlers require the foreground main thread",
            )

    def _restore_signals(self) -> None:
        for signal_number, previous in self._previous_signal_handlers.items():
            signal.signal(signal_number, previous)
        self._previous_signal_handlers.clear()

    def _new_transaction_id(self) -> str:
        return (
            "sv-"
            + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            + "-"
            + secrets.token_hex(6)
        )

    def _prepare_runtime(self) -> None:
        _reject_symlink_components(self.paths.runtime_root, "runtime root")
        for path, label in (
            (self.paths.logs, "logs"),
            (self.paths.pids, "pids"),
            (self.paths.locks, "locks"),
            (self.paths.status, "status"),
            (self.paths.transactions, "transactions"),
        ):
            _ensure_directory(path, label)

    def _reconcile_supervisor_active_records(
        self, profile: OperatingProfile
    ) -> dict[str, Any]:
        """Reconcile only a provably absent prior foreground supervisor."""

        self._prepare_runtime()
        lock_present = (
            self.paths.active_lock.exists()
            or self.paths.active_lock.is_symlink()
        )
        pid_present = (
            self.paths.active_pid.exists()
            or self.paths.active_pid.is_symlink()
        )
        if not lock_present and not pid_present:
            return {
                "reconciled": False,
                "reason_code": "OK",
                "removed_records": [],
            }
        lock_record = (
            _read_json_file(self.paths.active_lock, "supervisor lock")
            if lock_present
            else None
        )
        pid_record = (
            _read_json_file(self.paths.active_pid, "supervisor PID record")
            if pid_present
            else None
        )
        for record, expected_schema in (
            (lock_record, LOCK_SCHEMA),
            (pid_record, PID_SCHEMA),
        ):
            if record is None:
                continue
            if record.get("schema_version") != expected_schema:
                _fail(
                    "OWNERSHIP_UNCERTAIN",
                    "supervisor active record schema is invalid",
                )
            recorded_profile = record.get("profile_identity")
            if (
                recorded_profile is not None
                and recorded_profile != profile.identity
            ):
                _fail(
                    "OWNERSHIP_UNCERTAIN",
                    "supervisor active record belongs to another profile",
                )
        transaction_ids = {
            str(record.get("supervisor_transaction_id"))
            for record in (lock_record, pid_record)
            if record is not None
        }
        if len(transaction_ids) != 1 or "None" in transaction_ids:
            _fail(
                "OWNERSHIP_UNCERTAIN",
                "supervisor active transaction identities are ambiguous",
            )
        previous_transaction_id = next(iter(transaction_ids))
        record = pid_record or lock_record
        assert record is not None
        pid = record.get("pid")
        if not isinstance(pid, int) or pid <= 0:
            _fail(
                "OWNERSHIP_UNCERTAIN",
                "supervisor active record lacks a positive PID",
            )
        process_path = Path("/proc") / str(pid)
        if process_path.exists():
            try:
                observed = process_snapshot(pid)
            except SupervisorError:
                _fail(
                    "OWNERSHIP_UNCERTAIN",
                    "live recorded supervisor identity cannot be inspected",
                )
            if _same_process_identity(record, observed):
                if lock_present and pid_present:
                    _fail(
                        "supervisor_lock_active",
                        "a matching foreground supervisor is already active",
                    )
                _fail(
                    "OWNERSHIP_UNCERTAIN",
                    "partial supervisor state refers to a live owned process",
                )
            _fail(
                "OWNERSHIP_UNCERTAIN",
                "recorded supervisor PID was reused or changed identity",
                details={"recorded": record, "observed": observed},
            )
        matching = _matching_processes_for_record(
            record, exclude_pid=os.getpid()
        )
        if matching:
            _fail(
                "OWNERSHIP_UNCERTAIN",
                "a complete supervisor executable/argv match remains alive",
                details={"matching_processes": matching},
            )
        removed: list[str] = []
        if pid_present and _unlink_owned(
            self.paths.active_pid, previous_transaction_id
        ):
            removed.append("active_pid")
        if lock_present and _unlink_owned(
            self.paths.active_lock, previous_transaction_id
        ):
            removed.append("active_lock")
        return {
            "reconciled": True,
            "reason_code": "SUPERVISOR_STATE_STALE",
            "previous_supervisor_transaction_id": previous_transaction_id,
            "removed_records": removed,
            "process_absent": True,
        }

    def _acquire_active_records(self, profile: OperatingProfile) -> None:
        assert self.transaction_id and self.supervisor_identity
        self._prepare_runtime()
        self.startup_reconciliation = (
            self._reconcile_supervisor_active_records(profile)
        )
        common = {
            "supervisor_transaction_id": self.transaction_id,
            "profile_identity": profile.identity,
            "profile_path": str(self.profile_path.resolve(strict=False)),
            "state_path": str(self.state_path.resolve(strict=False)),
            "created_utc": _utc_now(),
            **self.supervisor_identity,
        }
        lock = {"schema_version": LOCK_SCHEMA, **common}
        _exclusive_write_json(self.paths.active_lock, lock)
        try:
            _atomic_write_json(
                self.paths.active_pid,
                {"schema_version": PID_SCHEMA, **common},
            )
        except Exception:
            _unlink_owned(self.paths.active_lock, self.transaction_id)
            raise

    def _cleanup_active_records(self) -> dict[str, bool]:
        assert self.transaction_id
        return {
            "active_pid_removed": _unlink_owned(
                self.paths.active_pid, self.transaction_id
            ),
            "active_lock_removed": _unlink_owned(
                self.paths.active_lock, self.transaction_id
            ),
        }

    def _log(self, event: str, **values: Any) -> None:
        assert self.transaction_id
        _append_event(
            self.paths.log(self.transaction_id),
            {
                "schema_version": LOG_SCHEMA,
                "timestamp_utc": _utc_now(),
                "supervisor_transaction_id": self.transaction_id,
                "event": event,
                **values,
            },
        )

    def _status_value(
        self,
        *,
        profile: OperatingProfile,
        desired: DesiredState,
        supervisor_state: str,
        api: Mapping[str, Any] | None,
        router: Mapping[str, Any] | None,
        model_child: Mapping[str, Any] | None,
        service_readiness_state: str,
        warm_model_identity: Mapping[str, Any] | None,
        recovery_status: Mapping[str, Any] | None = None,
        stop_reason: str | None = None,
        fault_reason: str | None = None,
    ) -> dict[str, Any]:
        if supervisor_state not in SUPERVISOR_STATES:
            _fail("invalid_supervisor_state", "unknown supervisor state")
        if service_readiness_state not in SERVICE_READINESS_STATES:
            _fail(
                "invalid_service_readiness_state",
                "unknown service readiness state",
            )
        model_service_state = (
            service_readiness_state
            if service_readiness_state in MODEL_SERVICE_STATES
            else "STOPPED"
            if supervisor_state == "STOPPED"
            else "DEGRADED"
        )
        inference_ready = model_service_state == "READY"
        service_operational = bool(
            desired.desired_state == "RUNNING"
            and supervisor_state == "RUNNING"
            and model_service_state
            in {
                "WAITING_FOR_MODEL",
                "MODEL_CANDIDATE_LOADING",
                "READY",
            }
        )
        reason_code = (
            "OK"
            if model_service_state == "READY"
            else "NO_READY_MODEL"
            if model_service_state == "WAITING_FOR_MODEL"
            else "MODEL_CANDIDATE_LOADING"
            if model_service_state == "MODEL_CANDIDATE_LOADING"
            else stop_reason
            if model_service_state == "STOPPED"
            else fault_reason or model_service_state
        )
        assert self.transaction_id and self.supervisor_identity
        return {
            "schema_version": STATUS_SCHEMA,
            "profile_identity": profile.identity,
            "desired_state": desired.desired_state,
            "desired_state_generation": desired.generation,
            "supervisor_state": supervisor_state,
            "service_readiness_state": service_readiness_state,
            "model_service_state": model_service_state,
            "service_operational": service_operational,
            "inference_ready": inference_ready,
            "reason_code": reason_code,
            "supervisor_identity": self.supervisor_identity,
            "supervisor_transaction_id": self.transaction_id,
            "observed_api_service": dict(api) if api is not None else None,
            "observed_private_router": (
                dict(router) if router is not None else None
            ),
            "observed_model_child": (
                dict(model_child)
                if model_child is not None
                else {"present": False, "identity": None}
            ),
            "warm_model_identity": (
                dict(warm_model_identity)
                if warm_model_identity is not None
                else None
            ),
            "recovery_status": (
                dict(recovery_status)
                if recovery_status is not None
                else None
            ),
            "last_observed_utc": _utc_now(),
            "stop_reason": stop_reason,
            "fault_reason": fault_reason,
        }

    def _write_status(self, value: Mapping[str, Any]) -> None:
        _atomic_write_json(self.paths.status_record, value)

    def _write_transaction(self, value: Mapping[str, Any]) -> None:
        assert self.transaction_id
        _atomic_write_json(
            self.paths.transaction(self.transaction_id), value
        )

    def _initial_transaction(
        self,
        profile: OperatingProfile,
        desired: DesiredState,
        dependencies: Mapping[str, Any],
    ) -> dict[str, Any]:
        assert self.transaction_id and self.supervisor_identity
        return {
            "schema_version": TRANSACTION_SCHEMA,
            "supervisor_transaction_id": self.transaction_id,
            "profile_identity": profile.identity,
            "profile_path": str(self.profile_path.resolve(strict=False)),
            "state_path": str(self.state_path.resolve(strict=False)),
            "initial_desired_state": desired.desired_state,
            "initial_desired_state_generation": desired.generation,
            "supervisor_identity": self.supervisor_identity,
            "start_utc": _utc_now(),
            "stop_utc": None,
            "api_controller": dependencies["api"],
            "branch_controller": dependencies["branch"],
            "started_api_transaction_id": None,
            "observed_router_transaction_id": None,
            "observed_model_child": None,
            "initial_service_readiness_state": "SUPERVISOR_STARTING",
            "last_service_readiness_state": "SUPERVISOR_STARTING",
            "warm_model_identity": None,
            "final_reason": None,
            "final_state": None,
            "controller_stop_result": None,
            "cleanup_result": None,
            "startup_reconciliation": self.startup_reconciliation,
            "recovery_transaction_ids": [],
        }

    def plan(self) -> dict[str, Any]:
        profile = load_operating_profile(self.profile_path)
        desired = load_desired_state(self.state_path, profile.identity)
        dependencies = self.adapter.validate()
        plan = self.adapter.invoke(
            "api", "plan", _api_arguments(
                "plan",
                profile,
                self.state_path,
                startup_model_policy=self.startup_model_policy,
            )
        )
        _verify_api_plan(
            plan,
            profile,
            startup_model_policy=self.startup_model_policy,
        )
        return _result(
            "plan",
            True,
            "ok",
            "supervisor plan validated without runtime mutation",
            profile=profile,
            desired=desired,
            paths=self.paths,
            data={
                "dependencies": dependencies,
                "controller_relationship": (
                    "supervisor -> API controller -> API lifespan -> "
                    "branch controller"
                ),
                "api_plan": plan,
                "will_start": desired.desired_state == "RUNNING",
            },
        )

    def _observe(
        self,
        profile: OperatingProfile,
        *,
        tolerate_router_failure: bool = False,
    ) -> tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        str,
        dict[str, Any] | None,
        dict[str, Any],
    ]:
        api_result = self.adapter.invoke("api", "status")
        api = _observed_api(api_result, profile)
        branch_result: dict[str, Any] | None
        try:
            branch_result = self.adapter.invoke("branch", "status")
            router = _observed_router(branch_result, profile)
        except SupervisorError:
            if not tolerate_router_failure:
                raise
            branch_result = None
            router = {
                "lifecycle_state": "INCONSISTENT",
                "active": False,
                "consistent": False,
                "transaction_id": None,
                "pid": None,
                "process_start_identity": None,
                "pgid": None,
                "sid": None,
                "endpoint": profile.private_router_endpoint.as_dict(),
                "listener_owned": False,
            }
        try:
            health = self.health_observer(profile)
        except SupervisorError as exc:
            health = _degraded_health(exc.reason_code)
        if branch_result is None:
            health = _degraded_health("router_process_or_listener_lost")
            return (
                api,
                router,
                {"present": False, "identity": None},
                "DEGRADED",
                None,
                health,
            )
        readiness, warm, model = _warm_observation(health, router)
        return (
            api,
            router,
            model,
            readiness,
            warm,
            health,
        )

    def _verify_observation(
        self,
        expected_api: Mapping[str, Any],
        expected_router: Mapping[str, Any],
        api: Mapping[str, Any],
        router: Mapping[str, Any],
    ) -> None:
        api_structurally_ready = (
            api.get("active") is True
            and api.get("consistent") is True
            and api.get("listener_owned") is True
        )
        if not api_structurally_ready:
            _fail(
                "API_PROCESS_LOST",
                "API process or public-listener ownership was lost",
                details={"expected": expected_api, "observed": api},
                exit_code=4,
            )
        if not _same_runtime_identity(expected_api, api):
            _fail(
                "OWNERSHIP_UNCERTAIN",
                "live API identity changed without a recovery transaction",
                details={"expected": expected_api, "observed": api},
                exit_code=4,
            )

        router_structurally_ready = (
            router.get("active") is True
            and router.get("consistent") is True
            and router.get("listener_owned") is True
        )
        if not router_structurally_ready:
            reason_code = (
                "ROUTER_PROCESS_LOST"
                if router.get("active") is not True
                else "PRIVATE_LISTENER_LOST"
            )
            _fail(
                reason_code,
                "router process or private-listener ownership was lost",
                details={"expected": expected_router, "observed": router},
                exit_code=4,
            )
        if not _same_runtime_identity(expected_router, router):
            _fail(
                "OWNERSHIP_UNCERTAIN",
                "live router identity changed without a recovery transaction",
                details={"expected": expected_router, "observed": router},
                exit_code=4,
            )
    def _controller_owned_stop(
        self, profile: OperatingProfile
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        stop_result = self.adapter.invoke("api", "stop")
        api_after = _observed_api(
            self.adapter.invoke("api", "status"), profile
        )
        branch_result = self.adapter.invoke("branch", "status")
        router_after = _observed_router(branch_result, profile)
        model_after = _observed_model_child(branch_result)
        if api_after.get("active") or router_after.get("active"):
            _fail(
                "controller_stop_incomplete",
                "controller-owned API/router runtime remains active",
                details={"api": api_after, "router": router_after},
                exit_code=4,
            )
        if model_after.get("present"):
            _fail(
                "controller_stop_incomplete",
                "model child remains after controller-owned stop",
                details={"model_child": model_after},
                exit_code=4,
            )
        return stop_result, api_after, router_after

    def _controller_owned_fault_cleanup(self) -> dict[str, Any]:
        """Reconcile partial startup in dependency-safe order.

        A startup fault can occur after the API controller has acquired its
        active records but while the branch router is still inconsistent.
        The regular graceful-stop path assumes a coherent router status and
        can therefore fail before the API is unwound.  Fault cleanup first
        reconciles/stops the dependent branch, then reconciles/stops the API,
        and finishes with an idempotent reconciliation of each controller.
        Every signal and record removal remains controller-owned.
        """

        actions: list[dict[str, Any]] = []
        incomplete: list[str] = []

        def invoke(kind: str, operation: str) -> dict[str, Any] | None:
            try:
                result = self.adapter.invoke(kind, operation)
            except Exception as exc:
                actions.append(
                    {
                        "kind": kind,
                        "operation": operation,
                        "ok": False,
                        "reason_code": getattr(
                            exc, "reason_code", type(exc).__name__
                        ),
                        "message": _bounded_text(exc),
                    }
                )
                return None
            actions.append(
                {
                    "kind": kind,
                    "operation": operation,
                    "ok": True,
                    "reason_code": result.get("reason_code"),
                }
            )
            return result

        for kind in ("branch", "api"):
            first = invoke(kind, "reconcile")
            payload_key = "data" if kind == "branch" else "runtime"
            payload = (
                first.get(payload_key)
                if isinstance(first, dict)
                else None
            )
            if kind == "api" or (
                isinstance(payload, dict)
                and payload.get("active") is True
            ):
                invoke(kind, "stop")
            final = invoke(kind, "reconcile")
            final_payload = (
                final.get(payload_key)
                if isinstance(final, dict)
                else None
            )
            if (
                not isinstance(final_payload, dict)
                or final_payload.get("active") is not False
            ):
                incomplete.append(kind)

        return {
            "ok": not incomplete,
            "reason_code": (
                "FAULT_CLEANUP_COMPLETE"
                if not incomplete
                else "FAULT_CLEANUP_INCOMPLETE"
            ),
            "actions": actions,
            "incomplete_controllers": incomplete,
            "manual_record_deletion": False,
            "unrelated_process_signaled": False,
        }

    def _coherent_observation(
        self,
        *,
        profile: OperatingProfile,
        desired: DesiredState,
        api: Mapping[str, Any] | None,
        router: Mapping[str, Any] | None,
        model_child: Mapping[str, Any] | None,
        readiness: str,
        warm: Mapping[str, Any] | None,
        health: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        observed_utc = _utc_now()
        return {
            "observed_utc": observed_utc,
            "profile_identity": profile.identity,
            "desired_state": desired.desired_state,
            "desired_state_generation": desired.generation,
            "supervisor_identity": dict(self.supervisor_identity or {}),
            "api_identity": dict(api or {}),
            "public_listener": {
                "host": profile.public_endpoint.host,
                "port": profile.public_endpoint.port,
                "owned": bool(api and api.get("listener_owned")),
            },
            "router_identity": dict(router or {}),
            "private_listener": {
                "host": profile.private_router_endpoint.host,
                "port": profile.private_router_endpoint.port,
                "owned": bool(router and router.get("listener_owned")),
            },
            "model_child": dict(
                model_child or {"present": False, "identity": None}
            ),
            "requested_default_alias": profile.default_model_alias,
            "resolved_immutable_model_id": (
                warm.get("resolved_public_model_id") if warm else None
            ),
            "artifact_version_identity": (
                warm.get("artifact_version_id") if warm else None
            ),
            "registry_generation": (
                warm.get("registry_generation") if warm else None
            ),
            "warm_model_health": (
                warm.get("health_state") if warm else None
            ),
            "service_readiness_state": readiness,
            "model_service_state": (
                health.get("body", {}).get("model_service_state")
                if health and isinstance(health.get("body"), dict)
                else readiness
            ),
            "service_available": bool(
                health
                and isinstance(health.get("body"), dict)
                and health["body"].get("service_available") is True
            ),
            "inference_ready": bool(
                health
                and isinstance(health.get("body"), dict)
                and health["body"].get("inference_ready") is True
            ),
            "public_health": {
                "http_status": health.get("http_status") if health else None,
                "ready": bool(
                    health
                    and isinstance(health.get("body"), dict)
                    and health["body"].get("ready") is True
                ),
            },
            "private_health": {
                "ready": bool(
                    warm and warm.get("health_state") == "ready"
                )
            },
        }

    @staticmethod
    def _api_failure_reason(error: SupervisorError) -> str:
        if error.reason_code in {
            "ROUTER_PROCESS_LOST",
            "PRIVATE_LISTENER_LOST",
            "OWNERSHIP_UNCERTAIN",
            "ENDPOINT_CONFLICT",
        }:
            return error.reason_code
        controller = error.details.get("controller_result")
        reason = controller.get("reason_code") if isinstance(controller, dict) else None
        if reason == "LISTENER_OWNERSHIP_MISMATCH":
            return "PUBLIC_LISTENER_LOST"
        if reason == "ENDPOINT_CONFLICT":
            return "ENDPOINT_CONFLICT"
        if reason in {"OWNERSHIP_UNCERTAIN", "PROCESS_IDENTITY_MISMATCH"}:
            observed = error.details.get("controller_result", {}).get(
                "ownership", {}
            )
            if isinstance(observed, dict) and observed.get("observed") is not None:
                return "OWNERSHIP_UNCERTAIN"
        return "API_PROCESS_LOST"

    def _wait_for_reusable_stack_endpoints(
        self,
        *,
        profile: OperatingProfile,
        recovery: RecoveryStore,
        attempt: RecoveryAttempt,
        observation: Mapping[str, Any],
    ) -> DesiredState | None:
        """Wait boundedly for controller-owned listener teardown to settle."""

        deadline = time.monotonic() + min(
            180.0, max(30.0, self.adapter.timeout_seconds)
        )
        while True:
            current = load_desired_state(
                self.state_path, profile.identity
            )
            if (
                current.desired_state == "STOPPED"
                or self.shutdown_requested.is_set()
            ):
                recovery.complete(
                    attempt,
                    desired_state=current.desired_state,
                    desired_generation=current.generation,
                    outcome="CANCELLED_BY_STOPPED",
                    observation=observation,
                )
                return None
            public_reusable = _endpoint_bindable(
                profile.public_endpoint.host,
                profile.public_endpoint.port,
            )
            private_reusable = _endpoint_bindable(
                profile.private_router_endpoint.host,
                profile.private_router_endpoint.port,
            )
            if public_reusable and private_reusable:
                return current
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _fail(
                    "ENDPOINT_REUSE_TIMEOUT",
                    "owned API-stack endpoints did not become reusable",
                    details={
                        "public_endpoint_reusable": public_reusable,
                        "private_endpoint_reusable": private_reusable,
                    },
                    exit_code=4,
                )
            self.shutdown_requested.wait(
                min(self.monitor_interval_seconds, remaining)
            )

    def _recover_router_only(
        self,
        *,
        profile: OperatingProfile,
        desired: DesiredState,
        recovery: RecoveryStore,
        reason_code: str,
        expected_api: Mapping[str, Any],
        expected_router: Mapping[str, Any],
        pre_observation: Mapping[str, Any],
    ) -> (
        tuple[
            dict[str, Any],
            dict[str, Any],
            dict[str, Any],
            str,
            dict[str, Any] | None,
            dict[str, Any],
            str,
        ]
        | None
    ):
        """Wait for the API-owned router recovery without restarting API."""

        attempt = recovery.begin(
            reason_code=reason_code,
            desired_state=desired.desired_state,
            desired_generation=desired.generation,
            observation=pre_observation,
            selected_action="API_OWNED_ROUTER_RESTART",
        )
        if attempt is None:
            if desired.desired_state == "STOPPED":
                return None
            _fail(
                "FAIL_CLOSED_LATCHED",
                "automatic router recovery is not permitted",
                exit_code=4,
            )
        try:
            current = load_desired_state(self.state_path, profile.identity)
            if current.desired_state == "STOPPED":
                recovery.complete(
                    attempt,
                    desired_state=current.desired_state,
                    desired_generation=current.generation,
                    outcome="CANCELLED_BY_STOPPED",
                    observation=pre_observation,
                )
                return None
            recovery.transition(
                attempt,
                desired_state=current.desired_state,
                desired_generation=current.generation,
                recovery_state="WAITING_FOR_ROUTER_RECOVERY",
                observation=pre_observation,
            )
            deadline = time.monotonic() + min(
                300.0, max(30.0, self.adapter.timeout_seconds)
            )
            last_error: str | None = None
            while time.monotonic() < deadline:
                current = load_desired_state(
                    self.state_path, profile.identity
                )
                if (
                    current.desired_state == "STOPPED"
                    or self.shutdown_requested.is_set()
                ):
                    recovery.complete(
                        attempt,
                        desired_state=current.desired_state,
                        desired_generation=current.generation,
                        outcome="CANCELLED_BY_STOPPED",
                        observation=pre_observation,
                    )
                    return None
                try:
                    (
                        api_now,
                        router_now,
                        model_now,
                        readiness,
                        warm,
                        health,
                    ) = self._observe(
                        profile, tolerate_router_failure=True
                    )
                    api_ready = (
                        api_now.get("active") is True
                        and api_now.get("consistent") is True
                        and api_now.get("listener_owned") is True
                    )
                    router_ready = (
                        router_now.get("active") is True
                        and router_now.get("consistent") is True
                        and router_now.get("listener_owned") is True
                    )
                    if (
                        api_ready
                        and _same_runtime_identity(expected_api, api_now)
                        and router_ready
                        and not _same_runtime_identity(
                            expected_router, router_now
                        )
                        and model_now.get("present") is True
                        and readiness == "READY"
                        and warm is not None
                    ):
                        observation = self._coherent_observation(
                            profile=profile,
                            desired=current,
                            api=api_now,
                            router=router_now,
                            model_child=model_now,
                            readiness=readiness,
                            warm=warm,
                            health=health,
                        )
                        recovery.complete(
                            attempt,
                            desired_state=current.desired_state,
                            desired_generation=current.generation,
                            outcome="RECOVERED",
                            observation=observation,
                        )
                        return (
                            api_now,
                            router_now,
                            model_now,
                            readiness,
                            warm,
                            health,
                            attempt.recovery_transaction_id,
                        )
                except SupervisorError as exc:
                    last_error = exc.reason_code
                self.shutdown_requested.wait(
                    self.monitor_interval_seconds
                )
            _fail(
                "RECOVERY_ATTEMPT_FAILED",
                f"router-only recovery readiness timed out: {last_error}",
                exit_code=4,
            )
        except (SupervisorError, ServiceControlError, RecoveryError) as exc:
            current = load_desired_state(self.state_path, profile.identity)
            recovery.complete(
                attempt,
                desired_state=current.desired_state,
                desired_generation=current.generation,
                outcome=(
                    "CANCELLED_BY_STOPPED"
                    if current.desired_state == "STOPPED"
                    else "FAILED"
                ),
                observation=pre_observation,
                error_category=getattr(exc, "reason_code", type(exc).__name__),
            )
            if current.desired_state == "STOPPED":
                return None
            raise

    def _recover_api_stack(
        self,
        *,
        profile: OperatingProfile,
        desired: DesiredState,
        recovery: RecoveryStore,
        reason_code: str,
        pre_observation: Mapping[str, Any],
    ) -> (
        tuple[
            dict[str, Any],
            dict[str, Any],
            dict[str, Any],
            str,
            dict[str, Any] | None,
            dict[str, Any],
            str,
        ]
        | None
    ):
        attempt = recovery.begin(
            reason_code=reason_code,
            desired_state=desired.desired_state,
            desired_generation=desired.generation,
            observation=pre_observation,
            selected_action="CONTROLLER_OWNED_API_STACK_RESTART",
        )
        if attempt is None:
            if desired.desired_state == "STOPPED":
                return None
            _fail(
                "FAIL_CLOSED_LATCHED",
                "automatic API-stack recovery is not permitted",
                exit_code=4,
            )
        try:
            current = load_desired_state(self.state_path, profile.identity)
            if current.desired_state == "STOPPED":
                recovery.complete(
                    attempt,
                    desired_state=current.desired_state,
                    desired_generation=current.generation,
                    outcome="CANCELLED_BY_STOPPED",
                    observation=pre_observation,
                )
                return None
            recovery.transition(
                attempt,
                desired_state=current.desired_state,
                desired_generation=current.generation,
                recovery_state="RECONCILING",
                observation=pre_observation,
            )
            api_reconcile = self.adapter.invoke("api", "reconcile")
            recovery.transition(
                attempt,
                desired_state=current.desired_state,
                desired_generation=current.generation,
                recovery_state="RECONCILING",
                controller_result=api_reconcile,
            )
            api_runtime = api_reconcile.get("runtime")
            if isinstance(api_runtime, dict) and api_runtime.get("active") is True:
                api_stop = self.adapter.invoke("api", "stop")
                recovery.transition(
                    attempt,
                    desired_state=current.desired_state,
                    desired_generation=current.generation,
                    recovery_state="RECONCILING",
                    controller_result=api_stop,
                )

            branch_reconcile = self.adapter.invoke("branch", "reconcile")
            recovery.transition(
                attempt,
                desired_state=current.desired_state,
                desired_generation=current.generation,
                recovery_state="RECONCILING",
                controller_result=branch_reconcile,
            )
            branch_data = branch_reconcile.get("data")
            if isinstance(branch_data, dict) and branch_data.get("active") is True:
                branch_stop = self.adapter.invoke("branch", "stop")
                recovery.transition(
                    attempt,
                    desired_state=current.desired_state,
                    desired_generation=current.generation,
                    recovery_state="RECONCILING",
                    controller_result=branch_stop,
                )

            recovery.transition(
                attempt,
                desired_state=current.desired_state,
                desired_generation=current.generation,
                recovery_state="DELAYING",
                observation=pre_observation,
            )
            deadline = time.monotonic() + attempt.delay_seconds
            waiter = threading.Event()
            while time.monotonic() < deadline:
                current = load_desired_state(
                    self.state_path, profile.identity
                )
                if (
                    current.desired_state == "STOPPED"
                    or self.shutdown_requested.is_set()
                ):
                    recovery.complete(
                        attempt,
                        desired_state=current.desired_state,
                        desired_generation=current.generation,
                        outcome="CANCELLED_BY_STOPPED",
                        observation=pre_observation,
                    )
                    return None
                waiter.wait(min(0.05, deadline - time.monotonic()))

            current = load_desired_state(self.state_path, profile.identity)
            if current.desired_state == "STOPPED":
                recovery.complete(
                    attempt,
                    desired_state=current.desired_state,
                    desired_generation=current.generation,
                    outcome="CANCELLED_BY_STOPPED",
                    observation=pre_observation,
                )
                return None
            current = self._wait_for_reusable_stack_endpoints(
                profile=profile,
                recovery=recovery,
                attempt=attempt,
                observation=pre_observation,
            )
            if current is None:
                return None
            recovery.transition(
                attempt,
                desired_state=current.desired_state,
                desired_generation=current.generation,
                recovery_state="RESTARTING_API",
                observation=pre_observation,
            )
            api_start = self.adapter.invoke(
                "api",
                "start",
                _api_arguments(
                "start",
                profile,
                self.state_path,
                startup_model_policy=self.startup_model_policy,
            ),
            )
            expected_api = _observed_api(api_start, profile)
            recovery.transition(
                attempt,
                desired_state=current.desired_state,
                desired_generation=current.generation,
                recovery_state="VERIFYING",
                controller_result=api_start,
            )
            verify_deadline = time.monotonic() + min(
                300.0, max(30.0, self.adapter.timeout_seconds)
            )
            last_error: str | None = None
            while time.monotonic() < verify_deadline:
                current = load_desired_state(
                    self.state_path, profile.identity
                )
                if current.desired_state == "STOPPED":
                    recovery.complete(
                        attempt,
                        desired_state=current.desired_state,
                        desired_generation=current.generation,
                        outcome="CANCELLED_BY_STOPPED",
                        observation=pre_observation,
                    )
                    return None
                try:
                    (
                        api_now,
                        router_now,
                        model_now,
                        readiness,
                        warm,
                        health,
                    ) = self._observe(
                        profile, tolerate_router_failure=True
                    )
                    if not _same_runtime_identity(expected_api, api_now):
                        _fail(
                            "OWNERSHIP_UNCERTAIN",
                            "restarted API identity changed during verification",
                        )
                    complete = bool(
                        router_now.get("active")
                        and router_now.get("consistent")
                        and router_now.get("listener_owned")
                        and model_now.get("present")
                        and readiness == "READY"
                        and warm is not None
                    )
                    if complete:
                        observation = self._coherent_observation(
                            profile=profile,
                            desired=current,
                            api=api_now,
                            router=router_now,
                            model_child=model_now,
                            readiness=readiness,
                            warm=warm,
                            health=health,
                        )
                        recovery.complete(
                            attempt,
                            desired_state=current.desired_state,
                            desired_generation=current.generation,
                            outcome="RECOVERED",
                            observation=observation,
                        )
                        return (
                            api_now,
                            router_now,
                            model_now,
                            readiness,
                            warm,
                            health,
                            attempt.recovery_transaction_id,
                        )
                except SupervisorError as exc:
                    last_error = exc.reason_code
                self.shutdown_requested.wait(
                    self.monitor_interval_seconds
                )
            _fail(
                "RECOVERY_ATTEMPT_FAILED",
                f"API-stack recovery readiness timed out: {last_error}",
                exit_code=4,
            )
        except (SupervisorError, ServiceControlError, RecoveryError) as exc:
            current = load_desired_state(self.state_path, profile.identity)
            recovery.complete(
                attempt,
                desired_state=current.desired_state,
                desired_generation=current.generation,
                outcome=(
                    "CANCELLED_BY_STOPPED"
                    if current.desired_state == "STOPPED"
                    else "FAILED"
                ),
                observation=pre_observation,
                error_category=getattr(exc, "reason_code", type(exc).__name__),
            )
            if current.desired_state == "STOPPED":
                return None
            raise

    def run(self) -> dict[str, Any]:
        profile = load_operating_profile(self.profile_path)
        desired = load_desired_state(self.state_path, profile.identity)
        dependencies = self.adapter.validate()
        self.transaction_id = self._new_transaction_id()
        self.supervisor_identity = process_snapshot(os.getpid())
        self._acquire_active_records(profile)
        recovery = RecoveryStore(
            self.paths.recovery,
            profile.identity,
            _recovery_policy(profile),
        )
        recovery_status = recovery.initialize(
            desired_state=desired.desired_state,
            desired_generation=desired.generation,
        )
        transaction = self._initial_transaction(
            profile, desired, dependencies
        )
        self._write_transaction(transaction)
        self._write_status(
            self._status_value(
                profile=profile,
                desired=desired,
                supervisor_state="STARTING",
                api=None,
                router=None,
                model_child=None,
                service_readiness_state="SUPERVISOR_STARTING",
                warm_model_identity=None,
                recovery_status=recovery_status,
            )
        )
        self._log(
            "supervisor_starting",
            profile_identity=profile.identity,
            desired_state=desired.desired_state,
            desired_state_generation=desired.generation,
        )
        self._install_signals()
        api_started = False
        stop_result: dict[str, Any] | None = None
        final_status: dict[str, Any] | None = None
        final_reason: str | None = None
        cleanup_result: dict[str, Any] | None = None
        try:
            if desired.desired_state == "STOPPED":
                final_reason = "desired_state_stopped_at_start"
                final_status = self._status_value(
                    profile=profile,
                    desired=desired,
                    supervisor_state="STOPPED",
                    api=None,
                    router=None,
                    model_child=None,
                    service_readiness_state="STOPPED",
                    warm_model_identity=None,
                    recovery_status=recovery.public_status(),
                    stop_reason=final_reason,
                )
                self._write_status(final_status)
                transaction.update(
                    {
                        "stop_utc": _utc_now(),
                        "final_reason": final_reason,
                        "final_state": "STOPPED",
                    }
                )
                self._write_transaction(transaction)
                self._log("supervisor_stopped", reason=final_reason)
            else:
                api_plan = self.adapter.invoke(
                    "api",
                    "plan",
                    _api_arguments(
                "plan",
                profile,
                self.state_path,
                startup_model_policy=self.startup_model_policy,
            ),
                )
                _verify_api_plan(
                    api_plan,
                    profile,
                    startup_model_policy=self.startup_model_policy,
                )
                self._write_status(
                    self._status_value(
                        profile=profile,
                        desired=desired,
                        supervisor_state="STARTING",
                        api=None,
                        router=None,
                        model_child=None,
                        service_readiness_state="API_STARTING",
                        warm_model_identity=None,
                        recovery_status=recovery.public_status(),
                    )
                )
                api_start = self.adapter.invoke(
                    "api",
                    "start",
                    _api_arguments(
                "start",
                profile,
                self.state_path,
                startup_model_policy=self.startup_model_policy,
            ),
                )
                api_started = True
                expected_api = _observed_api(api_start, profile)
                if (
                    not expected_api["active"]
                    or not expected_api["consistent"]
                    or not expected_api["listener_owned"]
                ):
                    _fail(
                        "api_start_identity_invalid",
                        "API start did not return an owned active runtime",
                    )
                (
                    api_now,
                    expected_router,
                    expected_model,
                    current_readiness,
                    current_warm,
                    health_now,
                ) = self._observe(profile)
                if (
                    not expected_router["active"]
                    or not expected_router["consistent"]
                    or not expected_router["listener_owned"]
                ):
                    _fail(
                        "router_start_identity_invalid",
                        "branch status did not return an owned active router",
                    )
                self._verify_observation(
                    expected_api,
                    expected_router,
                    api_now,
                    expected_router,
                )
                transaction.update(
                    {
                        "started_api_transaction_id": expected_api[
                            "transaction_id"
                        ],
                        "observed_router_transaction_id": expected_router[
                            "transaction_id"
                        ],
                        "observed_model_child": expected_model,
                        "last_service_readiness_state": current_readiness,
                        "warm_model_identity": current_warm,
                    }
                )
                baseline_observation = self._coherent_observation(
                    profile=profile,
                    desired=desired,
                    api=expected_api,
                    router=expected_router,
                    model_child=expected_model,
                    readiness=current_readiness,
                    warm=current_warm,
                    health=health_now,
                )
                recovery.initialize(
                    desired_state=desired.desired_state,
                    desired_generation=desired.generation,
                    observation=baseline_observation,
                )
                self._write_transaction(transaction)
                running_status = self._status_value(
                    profile=profile,
                    desired=desired,
                    supervisor_state="RUNNING",
                    api=expected_api,
                    router=expected_router,
                    model_child=expected_model,
                    service_readiness_state=current_readiness,
                    warm_model_identity=current_warm,
                    recovery_status=recovery.public_status(),
                )
                self._write_status(running_status)
                self._log(
                    "runtime_running",
                    api_transaction_id=expected_api["transaction_id"],
                    router_transaction_id=expected_router["transaction_id"],
                    model_child_present=expected_model["present"],
                    service_readiness_state=current_readiness,
                    health_http_status=health_now.get("http_status"),
                )

                latest_api = expected_api
                latest_router = expected_router
                latest_model = expected_model
                latest_readiness = current_readiness
                latest_warm = current_warm
                while True:
                    if self.shutdown_requested.is_set():
                        final_reason = "external_graceful_shutdown"
                        break
                    current = load_desired_state(
                        self.state_path, profile.identity
                    )
                    if current.generation < desired.generation:
                        _fail(
                            "desired_state_generation_regressed",
                            "desired-state generation regressed",
                        )
                    desired = current
                    if desired.desired_state == "STOPPED":
                        final_reason = "desired_state_stopped"
                        break
                    try:
                        (
                            api_now,
                            router_now,
                            model_now,
                            current_readiness,
                            current_warm,
                            health_now,
                        ) = self._observe(
                            profile, tolerate_router_failure=True
                        )
                        self._verify_observation(
                            expected_api,
                            expected_router,
                            api_now,
                            router_now,
                        )
                    except SupervisorError as observation_error:
                        pre_observation = self._coherent_observation(
                            profile=profile,
                            desired=desired,
                            api=latest_api,
                            router=latest_router,
                            model_child=latest_model,
                            readiness="DEGRADED",
                            warm=latest_warm,
                            health=_degraded_health(
                                observation_error.reason_code
                            ),
                        )
                        reason_code = self._api_failure_reason(
                            observation_error
                        )
                        if reason_code in {
                            "OWNERSHIP_UNCERTAIN",
                            "ENDPOINT_CONFLICT",
                        }:
                            recovery.fail_closed_now(
                                reason_code=reason_code,
                                desired_state=desired.desired_state,
                                desired_generation=desired.generation,
                                observation=pre_observation,
                            )
                            raise
                        if reason_code in {
                            "ROUTER_PROCESS_LOST",
                            "PRIVATE_LISTENER_LOST",
                        }:
                            recovered = self._recover_router_only(
                                profile=profile,
                                desired=desired,
                                recovery=recovery,
                                reason_code=reason_code,
                                expected_api=expected_api,
                                expected_router=expected_router,
                                pre_observation=pre_observation,
                            )
                        else:
                            recovered = self._recover_api_stack(
                            profile=profile,
                            desired=desired,
                            recovery=recovery,
                            reason_code=reason_code,
                            pre_observation=pre_observation,
                        )
                        if recovered is None:
                            desired = load_desired_state(
                                self.state_path, profile.identity
                            )
                            final_reason = "desired_state_stopped"
                            break
                        (
                            api_now,
                            router_now,
                            model_now,
                            current_readiness,
                            current_warm,
                            health_now,
                            recovery_transaction_id,
                        ) = recovered
                        expected_api = api_now
                        expected_router = router_now
                        expected_model = model_now
                        transaction.setdefault(
                            "recovery_transaction_ids", []
                        ).append(recovery_transaction_id)
                        self._log(
                            "router_recovered"
                            if reason_code in {
                                "ROUTER_PROCESS_LOST",
                                "PRIVATE_LISTENER_LOST",
                            }
                            else "api_stack_recovered",
                            recovery_transaction_id=(
                                recovery_transaction_id
                            ),
                            reason_code=reason_code,
                            api_transaction_id=api_now.get(
                                "transaction_id"
                            ),
                            router_transaction_id=router_now.get(
                                "transaction_id"
                            ),
                        )
                    if (
                        router_now.get("active") is True
                        and router_now.get("consistent") is True
                        and router_now.get("listener_owned") is True
                        and current_readiness == "READY"
                        and current_warm is not None
                        and current_warm.get("router_transaction_id")
                        == router_now.get("transaction_id")
                    ):
                        if not _same_runtime_identity(
                            expected_router, router_now
                        ):
                            self._log(
                                "api_runtime_router_recovered",
                                prior_router_transaction_id=(
                                    expected_router.get("transaction_id")
                                ),
                                router_transaction_id=router_now.get(
                                    "transaction_id"
                                ),
                            )
                        expected_router = router_now
                        expected_model = model_now
                    latest_api = api_now
                    latest_router = router_now
                    latest_model = model_now
                    latest_readiness = current_readiness
                    latest_warm = current_warm
                    coherent = self._coherent_observation(
                        profile=profile,
                        desired=desired,
                        api=api_now,
                        router=router_now,
                        model_child=model_now,
                        readiness=current_readiness,
                        warm=current_warm,
                        health=health_now,
                    )
                    recovery.healthy_tick(
                        desired_state=desired.desired_state,
                        desired_generation=desired.generation,
                        observation=coherent,
                        ready=current_readiness
                        in {
                            "WAITING_FOR_MODEL",
                            "MODEL_CANDIDATE_LOADING",
                            "READY",
                        },
                    )
                    transaction.update(
                        {
                            "observed_model_child": model_now,
                            "last_service_readiness_state": (
                                current_readiness
                            ),
                            "warm_model_identity": current_warm,
                        }
                    )
                    self._write_transaction(transaction)
                    self._write_status(
                        self._status_value(
                            profile=profile,
                            desired=desired,
                            supervisor_state="RUNNING",
                            api=api_now,
                            router=router_now,
                            model_child=model_now,
                            service_readiness_state=current_readiness,
                            warm_model_identity=current_warm,
                            recovery_status=recovery.public_status(),
                        )
                    )
                    self.shutdown_requested.wait(
                        self.monitor_interval_seconds
                    )

                stopping_status = self._status_value(
                    profile=profile,
                    desired=desired,
                    supervisor_state="STOPPING",
                    api=latest_api,
                    router=latest_router,
                    model_child=latest_model,
                    service_readiness_state=latest_readiness,
                    warm_model_identity=latest_warm,
                    recovery_status=recovery.public_status(),
                    stop_reason=final_reason,
                )
                self._write_status(stopping_status)
                self._log("runtime_stopping", reason=final_reason)
                stop_result, api_after, router_after = (
                    self._controller_owned_stop(profile)
                )
                final_status = self._status_value(
                    profile=profile,
                    desired=desired,
                    supervisor_state="STOPPED",
                    api=api_after,
                    router=router_after,
                    model_child={"present": False, "identity": None},
                    service_readiness_state="STOPPED",
                    warm_model_identity=latest_warm,
                    recovery_status=recovery.initialize(
                        desired_state=desired.desired_state,
                        desired_generation=desired.generation,
                    ),
                    stop_reason=final_reason,
                )
                self._write_status(final_status)
                transaction.update(
                    {
                        "stop_utc": _utc_now(),
                        "final_reason": final_reason,
                        "final_state": "STOPPED",
                        "controller_stop_result": stop_result,
                        "last_service_readiness_state": "STOPPED",
                        "warm_model_identity": latest_warm,
                    }
                )
                self._write_transaction(transaction)
                self._log("supervisor_stopped", reason=final_reason)
        except (SupervisorError, ServiceControlError, RecoveryError) as exc:
            reason = (
                exc.reason_code
                if isinstance(
                    exc,
                    (SupervisorError, ServiceControlError, RecoveryError),
                )
                else "unexpected_error"
            )
            if api_started and stop_result is None:
                stop_result = self._controller_owned_fault_cleanup()
            fault_status = self._status_value(
                profile=profile,
                desired=desired,
                supervisor_state="FAULTED",
                api=None,
                router=None,
                model_child=None,
                service_readiness_state="DEGRADED",
                warm_model_identity=transaction.get(
                    "warm_model_identity"
                ),
                recovery_status=recovery.public_status(),
                fault_reason=reason,
            )
            self._write_status(fault_status)
            transaction.update(
                {
                    "stop_utc": _utc_now(),
                    "final_reason": reason,
                    "final_state": "FAULTED",
                    "controller_stop_result": stop_result,
                    "last_service_readiness_state": "DEGRADED",
                }
            )
            self._write_transaction(transaction)
            self._log("supervisor_faulted", reason=reason)
            raise
        finally:
            self._restore_signals()
            cleanup_result = self._cleanup_active_records()
            transaction["cleanup_result"] = cleanup_result
            self._write_transaction(transaction)
        assert final_status is not None
        return _result(
            "run",
            True,
            "ok",
            "foreground supervisor exited cleanly",
            profile=profile,
            desired=desired,
            paths=self.paths,
            data={
                "status": final_status,
                "controller_stop_result": stop_result,
                "cleanup_result": cleanup_result,
            },
        )


def _read_status_record(
    path: Path, profile_identity: str | None = None
) -> dict[str, Any]:
    value = _read_json_file(path, "supervisor status")
    if value.get("schema_version") != STATUS_SCHEMA:
        _fail("invalid_status_record", "supervisor status schema is invalid")
    if (
        profile_identity is not None
        and value.get("profile_identity") != profile_identity
    ):
        _fail(
            "profile_identity_mismatch",
            "supervisor status belongs to a different profile",
        )
    if value.get("supervisor_state") not in SUPERVISOR_STATES:
        _fail("invalid_status_record", "supervisor state is invalid")
    return value


def administrative_status(
    profile_path: Path,
    state_path: Path,
    runtime_root: Path,
    adapter: ControllerAdapter,
) -> dict[str, Any]:
    profile = load_operating_profile(profile_path)
    desired = load_desired_state(state_path, profile.identity)
    adapter.validate()
    paths = SupervisorPaths(
        Path(os.path.abspath(os.fspath(runtime_root)))
    )
    lock_present = paths.active_lock.exists() or paths.active_lock.is_symlink()
    pid_present = paths.active_pid.exists() or paths.active_pid.is_symlink()
    if lock_present != pid_present:
        _fail(
            "supervisor_identity_inconsistent",
            "supervisor lock/PID record presence disagrees",
        )
    status = (
        _read_status_record(paths.status_record, profile.identity)
        if paths.status_record.exists()
        else None
    )
    active_identity = None
    if lock_present:
        lock = _read_json_file(paths.active_lock, "supervisor lock")
        pid_record = _read_json_file(paths.active_pid, "supervisor PID record")
        if (
            lock.get("supervisor_transaction_id")
            != pid_record.get("supervisor_transaction_id")
        ):
            _fail(
                "supervisor_identity_inconsistent",
                "supervisor lock/PID transaction identities disagree",
            )
        observed = process_snapshot(int(pid_record.get("pid", 0)))
        if not _same_process_identity(pid_record, observed):
            _fail(
                "supervisor_identity_inconsistent",
                "supervisor process identity is stale or changed",
                details={"recorded": pid_record, "observed": observed},
            )
        active_identity = observed
    api = adapter.invoke("api", "status")
    branch = adapter.invoke("branch", "status")
    return _result(
        "status",
        True,
        "ok",
        "supervisor status inspected without lifecycle mutation",
        profile=profile,
        desired=desired,
        paths=paths,
        data={
            "active": lock_present,
            "active_identity": active_identity,
            "status": status,
            "api_controller_status": api,
            "branch_controller_status": branch,
        },
    )


def administrative_stop(
    profile_path: Path,
    state_path: Path,
    runtime_root: Path,
    *,
    expected_generation: int | None = None,
    wait_timeout_seconds: float = 60.0,
    poll_interval_seconds: float = 0.05,
) -> dict[str, Any]:
    if (
        type(wait_timeout_seconds) not in (int, float)
        or not math.isfinite(float(wait_timeout_seconds))
        or not 0.1 <= float(wait_timeout_seconds) <= 3_600.0
    ):
        _fail(
            "invalid_wait_timeout",
            "wait timeout must be finite and in 0.1..3600 seconds",
        )
    profile = load_operating_profile(profile_path)
    current = load_desired_state(state_path, profile.identity)
    if (
        expected_generation is not None
        and expected_generation != current.generation
    ):
        _fail(
            "stale_expected_generation",
            "expected generation does not match current desired state",
        )
    paths = SupervisorPaths(
        Path(os.path.abspath(os.fspath(runtime_root)))
    )
    if current.desired_state == "STOPPED":
        return _result(
            "stop",
            True,
            "already_stopped",
            "desired state is already STOPPED",
            profile=profile,
            desired=current,
            paths=paths,
            data={"state_changed": False},
        )
    updated = set_desired_state(
        profile,
        "STOPPED",
        state_path,
        expected_generation=(
            expected_generation
            if expected_generation is not None
            else current.generation
        ),
    )
    deadline = time.monotonic() + float(wait_timeout_seconds)
    final_status = None
    waiter = threading.Event()
    while time.monotonic() < deadline:
        lock_present = (
            paths.active_lock.exists() or paths.active_lock.is_symlink()
        )
        pid_present = paths.active_pid.exists() or paths.active_pid.is_symlink()
        if not lock_present and not pid_present and paths.status_record.exists():
            final_status = _read_status_record(
                paths.status_record, profile.identity
            )
            if (
                final_status.get("supervisor_state") == "STOPPED"
                and final_status.get("desired_state") == "STOPPED"
                and final_status.get("desired_state_generation")
                == updated.generation
            ):
                break
        waiter.wait(
            min(poll_interval_seconds, max(0.0, deadline - time.monotonic()))
        )
    else:
        _fail(
            "supervisor_stop_timeout",
            "foreground supervisor did not complete the requested stop",
            details={"desired_state": updated.as_dict()},
            exit_code=4,
        )
    return _result(
        "stop",
        True,
        "ok",
        "desired state changed to STOPPED and supervisor exited",
        profile=profile,
        desired=updated,
        paths=paths,
        data={"state_changed": True, "status": final_status},
    )


def _endpoint_bindable(host: str, port: int) -> bool:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        probe.close()


def administrative_reset_recovery(
    profile_path: Path,
    state_path: Path,
    runtime_root: Path,
    adapter: ControllerAdapter,
) -> dict[str, Any]:
    profile = load_operating_profile(profile_path)
    desired = load_desired_state(state_path, profile.identity)
    if desired.desired_state != "STOPPED":
        _fail(
            "DESIRED_STATE_STOPPED",
            "reset-recovery requires desired state STOPPED",
        )
    paths = SupervisorPaths(
        Path(os.path.abspath(os.fspath(runtime_root)))
    )
    if any(
        path.exists() or path.is_symlink()
        for path in (paths.active_lock, paths.active_pid)
    ):
        _fail(
            "OWNERSHIP_UNCERTAIN",
            "reset-recovery requires no active supervisor records",
        )
    adapter.validate()
    api_status = adapter.invoke("api", "status")
    branch_status = adapter.invoke("branch", "status")
    api_runtime = api_status.get("runtime")
    branch_data = branch_status.get("data")
    owned_runtime_absent = bool(
        isinstance(api_runtime, dict)
        and api_runtime.get("active") is False
        and api_runtime.get("consistent") is True
        and isinstance(branch_data, dict)
        and branch_data.get("active") is False
        and branch_data.get("active_state_consistent") is True
    )
    listeners_absent = bool(
        _endpoint_bindable(
            profile.public_endpoint.host, profile.public_endpoint.port
        )
        and _endpoint_bindable(
            profile.private_router_endpoint.host,
            profile.private_router_endpoint.port,
        )
    )
    recovery = RecoveryStore(
        paths.recovery, profile.identity, _recovery_policy(profile)
    )
    reset = recovery.reset_fail_closed(
        desired_state=desired.desired_state,
        desired_generation=desired.generation,
        owned_runtime_absent=owned_runtime_absent,
        listeners_absent=listeners_absent,
    )
    api_latch_path = paths.recovery / "fail-closed/api-runtime.json"
    api_latch_reset: dict[str, Any] = {
        "reset": False,
        "reason_code": "NO_ACTIVE_LATCH",
    }
    if api_latch_path.exists() or api_latch_path.is_symlink():
        api_latch = _read_json_file(
            api_latch_path, "API runtime fail-closed latch"
        )
        if (
            api_latch.get("schema_version")
            != "system-x.api-runtime-recovery-fail-closed.v1"
            or api_latch.get("profile_identity") != profile.identity
        ):
            _fail(
                "OWNERSHIP_UNCERTAIN",
                "API runtime fail-closed latch identity is invalid",
            )
        reset_id = "api-reset-" + secrets.token_hex(12)
        archive = paths.recovery / f"fail-closed/{reset_id}.json"
        _exclusive_write_json(
            archive,
            {
                **api_latch,
                "reset_id": reset_id,
                "reset_utc": _utc_now(),
                "reset_reason_code": "FAIL_CLOSED_RESET",
            },
        )
        if _read_json_file(
            api_latch_path, "API runtime fail-closed latch"
        ) != api_latch:
            _fail(
                "OWNERSHIP_UNCERTAIN",
                "API runtime latch changed before reset",
            )
        api_latch_path.unlink()
        _fsync_directory(api_latch_path.parent)
        api_latch_reset = {
            "reset": True,
            "reason_code": "FAIL_CLOSED_RESET",
            "reset_id": reset_id,
        }
    return _result(
        "reset-recovery",
        True,
        str(reset["reason_code"]),
        "fail-closed recovery latch reset boundary evaluated",
        profile=profile,
        desired=desired,
        paths=paths,
        data={
            "reset": reset,
            "api_runtime_reset": api_latch_reset,
            "owned_runtime_absent": owned_runtime_absent,
            "listeners_absent": listeners_absent,
            "api_controller_status": api_status,
            "branch_controller_status": branch_status,
        },
    )


def _result(
    operation: str,
    ok: bool,
    reason_code: str,
    message: str,
    *,
    profile: OperatingProfile | None,
    desired: DesiredState | None,
    paths: SupervisorPaths | None,
    data: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": RESULT_SCHEMA,
        "operation": operation,
        "ok": ok,
        "reason_code": reason_code,
        "message": _bounded_text(message),
        "timestamp_utc": _utc_now(),
        "profile_identity": profile.identity if profile is not None else None,
        "desired_state": (
            desired.desired_state if desired is not None else None
        ),
        "desired_state_generation": (
            desired.generation if desired is not None else None
        ),
        "resolved_paths": paths.as_dict() if paths is not None else None,
        "data": dict(data or {}),
    }


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        _fail("invalid_arguments", message)


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--profile",
        "--profile-path",
        dest="profile_path",
        type=Path,
        default=DEFAULT_PROFILE_PATH,
    )
    parser.add_argument(
        "--state-path",
        "--desired-state-path",
        dest="state_path",
        type=Path,
        default=DEFAULT_DESIRED_STATE_PATH,
    )
    parser.add_argument(
        "--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT
    )
    parser.add_argument(
        "--api-controller", type=Path, default=DEFAULT_API_CONTROLLER
    )
    parser.add_argument(
        "--api-controller-sha256", default=API_CONTROLLER_SHA256
    )
    parser.add_argument(
        "--branch-controller", type=Path, default=DEFAULT_BRANCH_CONTROLLER
    )
    parser.add_argument(
        "--branch-controller-sha256", default=BRANCH_CONTROLLER_SHA256
    )
    parser.add_argument(
        "--controller-timeout-seconds", type=float, default=180.0
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(prog="system-x-service-supervisor")
    operations = parser.add_subparsers(dest="operation", required=True)
    for name in ("plan", "run", "status"):
        command = operations.add_parser(name)
        _add_common(command)
        if name in ("plan", "run"):
            command.add_argument(
                "--startup-model-policy",
                choices=("always_warm", "router_control"),
                default=None,
            )
        if name == "run":
            command.add_argument(
                "--monitor-interval-seconds", type=float, default=0.25
            )
    stop = operations.add_parser("stop")
    _add_common(stop)
    stop.add_argument("--expected-generation", type=int)
    stop.add_argument("--wait-timeout-seconds", type=float, default=60.0)
    reset = operations.add_parser("reset-recovery")
    _add_common(reset)
    return parser


def _adapter_from_arguments(arguments: argparse.Namespace) -> ControllerAdapter:
    api_sha256 = getattr(
        arguments, "api_controller_sha256", API_CONTROLLER_SHA256
    )
    branch_sha256 = getattr(
        arguments, "branch_controller_sha256", BRANCH_CONTROLLER_SHA256
    )
    for label, value in (
        ("API controller", api_sha256),
        ("branch controller", branch_sha256),
    ):
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            _fail(
                "invalid_controller_identity",
                f"{label} SHA-256 identity must be 64 lowercase hexadecimal digits",
            )
    return ControllerAdapter(
        arguments.api_controller,
        arguments.branch_controller,
        timeout_seconds=arguments.controller_timeout_seconds,
        api_sha256=api_sha256,
        branch_sha256=branch_sha256,
    )


def execute(arguments: argparse.Namespace) -> dict[str, Any]:
    operation = arguments.operation
    if operation in ("plan", "run"):
        supervisor = ForegroundSupervisor(
            arguments.profile_path,
            arguments.state_path,
            arguments.runtime_root,
            _adapter_from_arguments(arguments),
            monitor_interval_seconds=getattr(
                arguments, "monitor_interval_seconds", 0.25
            ),
            startup_model_policy=getattr(
                arguments, "startup_model_policy", None
            ),
        )
        return supervisor.plan() if operation == "plan" else supervisor.run()
    if operation == "status":
        return administrative_status(
            arguments.profile_path,
            arguments.state_path,
            arguments.runtime_root,
            _adapter_from_arguments(arguments),
        )
    if operation == "stop":
        return administrative_stop(
            arguments.profile_path,
            arguments.state_path,
            arguments.runtime_root,
            expected_generation=arguments.expected_generation,
            wait_timeout_seconds=arguments.wait_timeout_seconds,
        )
    if operation == "reset-recovery":
        return administrative_reset_recovery(
            arguments.profile_path,
            arguments.state_path,
            arguments.runtime_root,
            _adapter_from_arguments(arguments),
        )
    _fail("unknown_operation", f"unsupported operation: {operation}")


def main(argv: Sequence[str] | None = None) -> int:
    values = list(argv) if argv is not None else sys.argv[1:]
    operation = values[0] if values else "argument-error"
    try:
        arguments = build_argument_parser().parse_args(values)
        operation = arguments.operation
        output = execute(arguments)
        exit_code = 0
    except (SupervisorError, ServiceControlError, RecoveryError) as exc:
        output = _result(
            operation,
            False,
            exc.reason_code,
            exc.message,
            profile=None,
            desired=None,
            paths=None,
            data=getattr(exc, "details", {}),
        )
        exit_code = getattr(exc, "exit_code", 2)
    except Exception as exc:
        output = _result(
            operation,
            False,
            "unexpected_error",
            f"{type(exc).__name__}: {exc}",
            profile=None,
            desired=None,
            paths=None,
        )
        print(
            f"system-x-service-supervisor unexpected error: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        exit_code = 70
    print(_canonical_json(output))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
