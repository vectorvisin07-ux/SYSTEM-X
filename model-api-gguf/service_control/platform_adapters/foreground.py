#!/usr/bin/env python3
"""Portable foreground process-host adapter for the System X supervisor.

The adapter owns registration and activation records only.  It starts exactly
one attached supervisor child through the existing supervisor machine
interface and never invokes API, router, model, or service-manager code.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import signal
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, NoReturn, Sequence

if __package__:
    from .contract import (
        ACTIVATION_METHOD,
        ADAPTER_IDENTITY,
        ADAPTER_VERSION,
        AUTOMATIC_ACTIVATION_SUPPORTED,
        MANIFEST_SCHEMA,
        OPERATIONS,
        PLATFORM_FAMILY,
        REQUIRED_HOST_CAPABILITIES,
        STATUS_SCHEMA,
        AdapterError,
        AdapterPathsView,
        bounded_activation_result,
        canonical_json,
        compute_configuration_identity,
        result_envelope,
        utc_now,
    )
    from .registry import create_adapter
    from ..operating_profile import (
        DesiredState,
        OperatingProfile,
        ServiceControlError,
        load_desired_state,
        load_operating_profile,
        set_desired_state,
    )
    from ..supervisor import (
        API_CONTROLLER_SHA256,
        BRANCH_CONTROLLER_SHA256,
        DEFAULT_API_CONTROLLER,
        DEFAULT_BRANCH_CONTROLLER,
        LOCK_SCHEMA as SUPERVISOR_LOCK_SCHEMA,
        PID_SCHEMA as SUPERVISOR_PID_SCHEMA,
        STATUS_SCHEMA as SUPERVISOR_STATUS_SCHEMA,
        SupervisorError,
        SupervisorPaths,
        administrative_stop,
        process_snapshot,
    )
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from service_control.platform_adapters.contract import (  # type: ignore
        ACTIVATION_METHOD,
        ADAPTER_IDENTITY,
        ADAPTER_VERSION,
        AUTOMATIC_ACTIVATION_SUPPORTED,
        MANIFEST_SCHEMA,
        OPERATIONS,
        PLATFORM_FAMILY,
        REQUIRED_HOST_CAPABILITIES,
        STATUS_SCHEMA,
        AdapterError,
        AdapterPathsView,
        bounded_activation_result,
        canonical_json,
        compute_configuration_identity,
        result_envelope,
        utc_now,
    )
    from service_control.platform_adapters.registry import (  # type: ignore
        create_adapter,
    )
    from service_control.operating_profile import (  # type: ignore
        DesiredState,
        OperatingProfile,
        ServiceControlError,
        load_desired_state,
        load_operating_profile,
        set_desired_state,
    )
    from service_control.supervisor import (  # type: ignore
        API_CONTROLLER_SHA256,
        BRANCH_CONTROLLER_SHA256,
        DEFAULT_API_CONTROLLER,
        DEFAULT_BRANCH_CONTROLLER,
        LOCK_SCHEMA as SUPERVISOR_LOCK_SCHEMA,
        PID_SCHEMA as SUPERVISOR_PID_SCHEMA,
        STATUS_SCHEMA as SUPERVISOR_STATUS_SCHEMA,
        SupervisorError,
        SupervisorPaths,
        administrative_stop,
        process_snapshot,
    )


ACTIVATION_LOCK_SCHEMA = "system-x.platform-service-adapter-lock.v1"
ACTIVATION_PID_SCHEMA = "system-x.platform-service-adapter-pid.v1"
ACTIVATION_TRANSACTION_SCHEMA = (
    "system-x.platform-service-adapter-transaction.v1"
)
MAX_RECORD_BYTES = 1_048_576
MAX_LOG_BYTES = 1_048_576

_MANIFEST_FIELDS = frozenset(
    (
        "schema_version",
        "adapter_identity",
        "adapter_version",
        "supported_platform_family",
        "required_host_capabilities",
        "activation_method",
        "automatic_activation_supported",
        "supervisor_entrypoint",
        "configuration_reference",
        "configuration_identity",
        "registered",
        "enabled",
        "active",
        "manifest_generation",
        "registered_utc",
        "updated_utc",
        "last_activation_result",
        "last_failure_reason",
    )
)
_REFERENCE_FIELDS = frozenset(
    (
        "profile_path",
        "state_path",
        "supervisor_runtime_root",
        "profile_identity",
    )
)
_FILE_IDENTITY_FIELDS = frozenset(("path", "sha256"))
_PROCESS_FIELDS = (
    "pid",
    "process_start_identity",
    "pgid",
    "sid",
    "executable",
    "argv_sha256",
)


def _fail(
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


def _absolute(path: Path | str) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _reject_symlink_components(path: Path, label: str) -> None:
    absolute = _absolute(path)
    for component in list(reversed(absolute.parents)) + [absolute]:
        try:
            metadata = component.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            _fail(
                "ADAPTER_MANIFEST_INVALID",
                f"{label} cannot be inspected: {exc}",
            )
        if stat.S_ISLNK(metadata.st_mode):
            _fail(
                "ADAPTER_MANIFEST_INVALID",
                f"{label} contains a symlink component",
            )


def _ensure_directory(path: Path, label: str) -> None:
    path = _absolute(path)
    _reject_symlink_components(path.parent, label)
    try:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError as exc:
        _fail(
            "ADAPTER_MANIFEST_INVALID",
            f"{label} cannot be created: {exc}",
        )
    _reject_symlink_components(path, label)
    try:
        metadata = path.stat()
    except OSError as exc:
        _fail(
            "ADAPTER_MANIFEST_INVALID",
            f"{label} cannot be inspected: {exc}",
        )
    if not stat.S_ISDIR(metadata.st_mode):
        _fail("ADAPTER_MANIFEST_INVALID", f"{label} must be a directory")


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        _fail(
            "ADAPTER_MANIFEST_INVALID",
            f"directory cannot be opened for synchronization: {exc}",
        )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _reject_duplicate_pairs(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(
                "ADAPTER_MANIFEST_INVALID",
                f"duplicate JSON key rejected: {key}",
            )
        result[key] = value
    return result


def _read_json(path: Path, label: str) -> dict[str, Any]:
    path = _absolute(path)
    _reject_symlink_components(path, label)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        _fail("ADAPTER_NOT_REGISTERED", f"{label} does not exist")
    except OSError as exc:
        _fail("ADAPTER_MANIFEST_INVALID", f"{label} cannot be opened: {exc}")
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size > MAX_RECORD_BYTES
        ):
            _fail(
                "ADAPTER_MANIFEST_INVALID",
                f"{label} must be a bounded direct regular file",
            )
        data = b""
        remaining = MAX_RECORD_BYTES + 1
        while remaining:
            block = os.read(descriptor, min(65_536, remaining))
            if not block:
                break
            data += block
            remaining -= len(block)
    finally:
        os.close(descriptor)
    if len(data) > MAX_RECORD_BYTES or b"\0" in data:
        _fail("ADAPTER_MANIFEST_INVALID", f"{label} content is invalid")
    try:
        value = json.loads(
            data.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: _fail(
                "ADAPTER_MANIFEST_INVALID",
                f"non-finite JSON number rejected: {token}",
            ),
        )
    except AdapterError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        _fail("ADAPTER_MANIFEST_INVALID", f"{label} is invalid JSON: {exc}")
    if not isinstance(value, dict):
        _fail("ADAPTER_MANIFEST_INVALID", f"{label} must be a JSON object")
    return value


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path = _absolute(path)
    _ensure_directory(path.parent, f"{path.name} parent")
    try:
        current = path.lstat()
    except FileNotFoundError:
        current = None
    if current is not None and (
        stat.S_ISLNK(current.st_mode) or not stat.S_ISREG(current.st_mode)
    ):
        _fail(
            "ADAPTER_MANIFEST_INVALID",
            f"{path.name} must be direct regular or absent",
        )
    payload = (canonical_json(dict(value)) + "\n").encode("utf-8")
    descriptor: int | None = None
    temporary: str | None = None
    try:
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        os.fchmod(descriptor, 0o600)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short adapter-record write")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(temporary, path)
        temporary = None
        _fsync_directory(path.parent)
    except AdapterError:
        raise
    except OSError as exc:
        _fail(
            "ADAPTER_MANIFEST_INVALID",
            f"atomic adapter-record write failed: {exc}",
        )
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass


def _exclusive_write_json(
    path: Path,
    value: Mapping[str, Any],
    *,
    conflict_reason: str,
) -> None:
    path = _absolute(path)
    _ensure_directory(path.parent, f"{path.name} parent")
    _reject_symlink_components(path, path.name)
    payload = (canonical_json(dict(value)) + "\n").encode("utf-8")
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except FileExistsError:
        _fail(conflict_reason, f"{path.name} already exists", exit_code=3)
    except OSError as exc:
        _fail(
            "ADAPTER_MANIFEST_INVALID",
            f"{path.name} cannot be created: {exc}",
        )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short adapter-record write")
            view = view[written:]
        os.fsync(descriptor)
    except OSError as exc:
        try:
            path.unlink()
        except OSError:
            pass
        _fail(
            "ADAPTER_MANIFEST_INVALID",
            f"{path.name} cannot be written: {exc}",
        )
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _direct_file_identity(
    path: Path | str,
    label: str,
    *,
    reason_code: str,
) -> dict[str, str]:
    path = _absolute(path)
    try:
        _reject_symlink_components(path, label)
    except AdapterError as exc:
        _fail(reason_code, f"{label} path is unsafe: {exc.message}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except (FileNotFoundError, OSError) as exc:
        _fail(reason_code, f"{label} cannot be opened: {exc}")
    digest = hashlib.sha256()
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            _fail(reason_code, f"{label} must be a direct regular file")
        while True:
            block = os.read(descriptor, 65_536)
            if not block:
                break
            digest.update(block)
    finally:
        os.close(descriptor)
    return {"path": str(path), "sha256": digest.hexdigest()}


def _same_process(
    expected: Mapping[str, Any], observed: Mapping[str, Any]
) -> bool:
    return all(expected.get(name) == observed.get(name) for name in _PROCESS_FIELDS)


def _process_observation(
    identity: Mapping[str, Any],
) -> dict[str, Any] | None:
    pid = identity.get("pid")
    if type(pid) is not int or pid < 1:
        _fail(
            "SUPERVISOR_IDENTITY_UNCERTAIN",
            "recorded process identity lacks a positive PID",
            exit_code=3,
        )
    if not (Path("/proc") / str(pid)).exists():
        return None
    try:
        observed = process_snapshot(pid)
    except SupervisorError as exc:
        _fail(
            "SUPERVISOR_IDENTITY_UNCERTAIN",
            f"live process identity cannot be inspected: {exc.reason_code}",
            exit_code=3,
        )
    if not _same_process(identity, observed):
        _fail(
            "SUPERVISOR_IDENTITY_UNCERTAIN",
            "recorded PID was reused or changed exact identity",
            exit_code=3,
            data={"recorded": dict(identity), "observed": observed},
        )
    return observed


def _unlink_transaction_owned(path: Path, transaction_id: str) -> bool:
    try:
        value = _read_json(path, path.name)
    except AdapterError as exc:
        if exc.reason_code == "ADAPTER_NOT_REGISTERED":
            return False
        raise
    if value.get("adapter_transaction_id") != transaction_id:
        _fail(
            "SUPERVISOR_IDENTITY_UNCERTAIN",
            f"{path.name} changed adapter transaction ownership",
            exit_code=3,
        )
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    except OSError as exc:
        _fail(
            "SUPERVISOR_IDENTITY_UNCERTAIN",
            f"{path.name} cannot be removed safely: {exc}",
            exit_code=3,
        )
    _fsync_directory(path.parent)
    return True


@dataclass(frozen=True, slots=True)
class AdapterPaths:
    runtime_root: Path

    @property
    def root(self) -> Path:
        return self.runtime_root / "foreground"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def status(self) -> Path:
        return self.root / "status.json"

    @property
    def pids(self) -> Path:
        return self.root / "pids"

    @property
    def locks(self) -> Path:
        return self.root / "locks"

    @property
    def transactions(self) -> Path:
        return self.root / "transactions"

    @property
    def active_pid(self) -> Path:
        return self.pids / "active.json"

    @property
    def active_lock(self) -> Path:
        return self.locks / "active.lock"

    def transaction(self, transaction_id: str) -> Path:
        return self.transactions / f"{transaction_id}.json"

    def supervisor_stdout(self, transaction_id: str) -> Path:
        return self.transactions / f"{transaction_id}.supervisor.stdout"

    def supervisor_stderr(self, transaction_id: str) -> Path:
        return self.transactions / f"{transaction_id}.supervisor.stderr"

    def view(self) -> AdapterPathsView:
        return AdapterPathsView(self.manifest, self.status)

    def status_paths(self) -> dict[str, str]:
        return {
            "manifest": str(self.manifest),
            "status": str(self.status),
            "active_pid": str(self.active_pid),
            "active_lock": str(self.active_lock),
        }


@dataclass(frozen=True, slots=True)
class ActiveObservation:
    active: bool
    supervisor_identity: dict[str, Any] | None
    supervisor_status: dict[str, Any] | None
    reconciliation_reason: str | None
    transaction_id: str | None


class ForegroundProcessHostAdapter:
    adapter_identity = ADAPTER_IDENTITY
    adapter_version = ADAPTER_VERSION

    def __init__(self, adapter_runtime_root: Path | str) -> None:
        self.paths = AdapterPaths(_absolute(adapter_runtime_root))

    def _prepare_runtime(self) -> None:
        _reject_symlink_components(
            self.paths.runtime_root, "adapter runtime root"
        )
        for path, label in (
            (self.paths.root, "foreground adapter root"),
            (self.paths.pids, "foreground adapter PID directory"),
            (self.paths.locks, "foreground adapter lock directory"),
            (self.paths.transactions, "foreground adapter transaction directory"),
        ):
            _ensure_directory(path, label)

    def _capability_value(
        self, requested: Sequence[str] | None = None
    ) -> dict[str, Any]:
        required = tuple(requested or REQUIRED_HOST_CAPABILITIES)
        unknown = sorted(set(required) - set(REQUIRED_HOST_CAPABILITIES))
        if unknown:
            _fail(
                "HOST_CAPABILITY_MISSING",
                f"unknown host capability requested: {unknown}",
                data={"unknown": unknown},
            )
        checks = {
            "foreground_process_execution": callable(subprocess.Popen),
            "structured_argv": True,
            "process_identity_observation": all(
                path.exists()
                for path in (
                    Path("/proc/self/stat"),
                    Path("/proc/self/exe"),
                    Path("/proc/self/cmdline"),
                )
            ),
            "filesystem_atomic_replacement": callable(os.replace),
            "operator_controlled_signal_relay": (
                callable(os.kill) and hasattr(signal, "SIGTERM")
            ),
        }
        missing = [name for name in required if not checks[name]]
        return {
            "available": not missing,
            "required": list(required),
            "missing": missing,
            "foreground_activation_supported": True,
            "registration_supported": True,
            "enable_disable_supported": True,
            "restart_supported": True,
            "unregister_supported": True,
        }

    def _validate_runtime_reference(self, path: Path | str) -> str:
        runtime = _absolute(path)
        _reject_symlink_components(runtime, "supervisor runtime root")
        if runtime.exists():
            try:
                metadata = runtime.stat()
            except OSError as exc:
                _fail(
                    "ADAPTER_CONFIGURATION_CONFLICT",
                    f"supervisor runtime root cannot be inspected: {exc}",
                )
            if not stat.S_ISDIR(metadata.st_mode):
                _fail(
                    "ADAPTER_CONFIGURATION_CONFLICT",
                    "supervisor runtime root must be a directory or absent",
                )
        else:
            parent = runtime.parent
            _reject_symlink_components(parent, "supervisor runtime parent")
            if not parent.is_dir():
                _fail(
                    "ADAPTER_CONFIGURATION_CONFLICT",
                    "supervisor runtime parent must already exist",
                )
        return str(runtime)

    def _validated_configuration(
        self,
        *,
        profile_path: Path | str,
        state_path: Path | str,
        supervisor_runtime_root: Path | str,
        supervisor_entrypoint: Path | str,
    ) -> tuple[
        dict[str, Any],
        OperatingProfile,
        DesiredState,
        dict[str, str],
    ]:
        capability = self._capability_value()
        if not capability["available"]:
            _fail(
                "HOST_CAPABILITY_MISSING",
                "required foreground host capabilities are unavailable",
                data=capability,
            )
        entrypoint = _direct_file_identity(
            supervisor_entrypoint,
            "supervisor entrypoint",
            reason_code="SUPERVISOR_ENTRYPOINT_INVALID",
        )
        profile_file = _absolute(profile_path)
        state_file = _absolute(state_path)
        try:
            profile = load_operating_profile(profile_file)
        except ServiceControlError as exc:
            _fail("PROFILE_INVALID", exc.message)
        try:
            desired = load_desired_state(state_file, profile.identity)
        except ServiceControlError as exc:
            _fail("DESIRED_STATE_INVALID", exc.message)
        runtime = self._validate_runtime_reference(supervisor_runtime_root)
        reference = {
            "profile_path": str(profile_file),
            "state_path": str(state_file),
            "supervisor_runtime_root": runtime,
            "profile_identity": profile.identity,
        }
        identity = compute_configuration_identity(
            {
                "adapter_identity": ADAPTER_IDENTITY,
                "adapter_version": ADAPTER_VERSION,
                "supported_platform_family": PLATFORM_FAMILY,
                "activation_method": ACTIVATION_METHOD,
                "supervisor_entrypoint_sha256": entrypoint["sha256"],
                "profile_path": reference["profile_path"],
                "state_path": reference["state_path"],
                "supervisor_runtime_root": reference[
                    "supervisor_runtime_root"
                ],
                "profile_identity": profile.identity,
            }
        )
        return (
            {
                "supervisor_entrypoint": entrypoint,
                "configuration_reference": reference,
                "configuration_identity": identity,
                "required_host_capability_result": capability,
            },
            profile,
            desired,
            entrypoint,
        )

    def _manifest_exists(self) -> bool:
        return self.paths.manifest.exists() or self.paths.manifest.is_symlink()

    def _load_manifest(
        self,
    ) -> tuple[dict[str, Any], OperatingProfile, DesiredState]:
        value = _read_json(self.paths.manifest, "adapter manifest")
        if frozenset(value) != _MANIFEST_FIELDS:
            _fail(
                "ADAPTER_MANIFEST_INVALID",
                "adapter manifest fields are incomplete or unknown",
            )
        constants = (
            (value.get("schema_version"), MANIFEST_SCHEMA),
            (value.get("adapter_identity"), ADAPTER_IDENTITY),
            (value.get("adapter_version"), ADAPTER_VERSION),
            (value.get("supported_platform_family"), PLATFORM_FAMILY),
            (value.get("activation_method"), ACTIVATION_METHOD),
            (
                value.get("automatic_activation_supported"),
                AUTOMATIC_ACTIVATION_SUPPORTED,
            ),
            (
                value.get("required_host_capabilities"),
                list(REQUIRED_HOST_CAPABILITIES),
            ),
            (value.get("registered"), True),
        )
        if any(observed != expected for observed, expected in constants):
            _fail(
                "ADAPTER_MANIFEST_INVALID",
                "adapter manifest contract identity is invalid",
            )
        if (
            type(value.get("enabled")) is not bool
            or type(value.get("active")) is not bool
            or type(value.get("manifest_generation")) is not int
            or value["manifest_generation"] < 1
            or not isinstance(value.get("registered_utc"), str)
            or not isinstance(value.get("updated_utc"), str)
            or (
                value.get("last_failure_reason") is not None
                and not isinstance(value["last_failure_reason"], str)
            )
            or (
                value.get("last_activation_result") is not None
                and not isinstance(value["last_activation_result"], dict)
            )
        ):
            _fail(
                "ADAPTER_MANIFEST_INVALID",
                "adapter manifest dynamic fields are invalid",
            )
        entrypoint = value.get("supervisor_entrypoint")
        reference = value.get("configuration_reference")
        if (
            not isinstance(entrypoint, dict)
            or frozenset(entrypoint) != _FILE_IDENTITY_FIELDS
            or not isinstance(reference, dict)
            or frozenset(reference) != _REFERENCE_FIELDS
        ):
            _fail(
                "ADAPTER_MANIFEST_INVALID",
                "adapter manifest configuration shape is invalid",
            )
        observed_entrypoint = _direct_file_identity(
            entrypoint.get("path", ""),
            "supervisor entrypoint",
            reason_code="SUPERVISOR_ENTRYPOINT_INVALID",
        )
        if observed_entrypoint != entrypoint:
            _fail(
                "SUPERVISOR_ENTRYPOINT_INVALID",
                "supervisor entrypoint identity changed",
            )
        try:
            profile = load_operating_profile(reference["profile_path"])
        except (KeyError, ServiceControlError) as exc:
            _fail("PROFILE_INVALID", f"registered profile is invalid: {exc}")
        if reference.get("profile_identity") != profile.identity:
            _fail(
                "ADAPTER_CONFIGURATION_CONFLICT",
                "manifest operating-profile identity changed",
            )
        try:
            desired = load_desired_state(
                reference["state_path"], profile.identity
            )
        except (KeyError, ServiceControlError) as exc:
            _fail(
                "DESIRED_STATE_INVALID",
                f"registered desired state is invalid: {exc}",
            )
        runtime = self._validate_runtime_reference(
            reference["supervisor_runtime_root"]
        )
        if runtime != reference["supervisor_runtime_root"]:
            _fail(
                "ADAPTER_CONFIGURATION_CONFLICT",
                "supervisor runtime root is not canonical",
            )
        expected_identity = compute_configuration_identity(
            {
                "adapter_identity": ADAPTER_IDENTITY,
                "adapter_version": ADAPTER_VERSION,
                "supported_platform_family": PLATFORM_FAMILY,
                "activation_method": ACTIVATION_METHOD,
                "supervisor_entrypoint_sha256": entrypoint["sha256"],
                "profile_path": reference["profile_path"],
                "state_path": reference["state_path"],
                "supervisor_runtime_root": reference[
                    "supervisor_runtime_root"
                ],
                "profile_identity": profile.identity,
            }
        )
        if value.get("configuration_identity") != expected_identity:
            _fail(
                "ADAPTER_CONFIGURATION_CONFLICT",
                "adapter configuration identity does not validate",
            )
        return value, profile, desired

    def _update_manifest(
        self,
        manifest: Mapping[str, Any],
        **updates: Any,
    ) -> dict[str, Any]:
        value = dict(manifest)
        immutable_identity = value.get("configuration_identity")
        value.update(updates)
        if value.get("configuration_identity") != immutable_identity:
            _fail(
                "ADAPTER_CONFIGURATION_CONFLICT",
                "dynamic update attempted to change configuration identity",
            )
        value["manifest_generation"] = int(
            manifest["manifest_generation"]
        ) + 1
        value["updated_utc"] = utc_now()
        _atomic_write_json(self.paths.manifest, value)
        return value

    def _validate_active_record(
        self,
        value: Mapping[str, Any],
        manifest: Mapping[str, Any],
        expected_schema: str,
    ) -> None:
        if (
            value.get("schema_version") != expected_schema
            or value.get("configuration_identity")
            != manifest.get("configuration_identity")
            or value.get("profile_identity")
            != manifest["configuration_reference"]["profile_identity"]
            or not isinstance(value.get("adapter_transaction_id"), str)
            or not isinstance(value.get("activation_identity"), dict)
            or (
                value.get("supervisor_identity") is not None
                and not isinstance(value["supervisor_identity"], dict)
            )
        ):
            _fail(
                "SUPERVISOR_IDENTITY_UNCERTAIN",
                "adapter active-record identity is invalid",
                exit_code=3,
            )

    def _supervisor_evidence(
        self,
        manifest: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        runtime = Path(
            manifest["configuration_reference"]["supervisor_runtime_root"]
        )
        paths = SupervisorPaths(runtime)
        lock_present = paths.active_lock.exists() or paths.active_lock.is_symlink()
        pid_present = paths.active_pid.exists() or paths.active_pid.is_symlink()
        if not lock_present and not pid_present:
            return None
        if lock_present != pid_present:
            _fail(
                "SUPERVISOR_IDENTITY_UNCERTAIN",
                "supervisor lock/PID presence is inconsistent",
                exit_code=3,
            )
        lock = _read_json(paths.active_lock, "supervisor active lock")
        pid_record = _read_json(paths.active_pid, "supervisor active PID")
        if (
            lock.get("schema_version") != SUPERVISOR_LOCK_SCHEMA
            or pid_record.get("schema_version") != SUPERVISOR_PID_SCHEMA
            or lock.get("supervisor_transaction_id")
            != pid_record.get("supervisor_transaction_id")
            or lock.get("profile_identity")
            != manifest["configuration_reference"]["profile_identity"]
            or pid_record.get("profile_identity")
            != manifest["configuration_reference"]["profile_identity"]
        ):
            _fail(
                "SUPERVISOR_IDENTITY_UNCERTAIN",
                "supervisor active-record identity is invalid",
                exit_code=3,
            )
        observed = _process_observation(pid_record)
        if observed is None:
            _fail(
                "SUPERVISOR_IDENTITY_UNCERTAIN",
                "supervisor active records refer to an absent process",
                exit_code=3,
            )
        status = _read_json(paths.status_record, "supervisor status")
        if (
            status.get("schema_version") != SUPERVISOR_STATUS_SCHEMA
            or status.get("profile_identity")
            != manifest["configuration_reference"]["profile_identity"]
            or status.get("supervisor_transaction_id")
            != pid_record.get("supervisor_transaction_id")
            or status.get("supervisor_identity") != observed
            or status.get("supervisor_state")
            not in {"STARTING", "RUNNING", "STOPPING"}
        ):
            _fail(
                "SUPERVISOR_IDENTITY_UNCERTAIN",
                "current supervisor status does not match the live process",
                exit_code=3,
            )
        return observed, status

    def _remove_active_records(self, transaction_id: str) -> None:
        _unlink_transaction_owned(self.paths.active_pid, transaction_id)
        _unlink_transaction_owned(self.paths.active_lock, transaction_id)

    def _reconcile_active(
        self,
        manifest: dict[str, Any],
    ) -> tuple[dict[str, Any], ActiveObservation]:
        lock_present = (
            self.paths.active_lock.exists()
            or self.paths.active_lock.is_symlink()
        )
        pid_present = (
            self.paths.active_pid.exists()
            or self.paths.active_pid.is_symlink()
        )
        supervisor = self._supervisor_evidence(manifest)
        if not lock_present and not pid_present:
            if supervisor is not None:
                _fail(
                    "SUPERVISOR_IDENTITY_UNCERTAIN",
                    "a live supervisor is not owned by this adapter activation",
                    exit_code=3,
                )
            reason = None
            if manifest["active"]:
                manifest = self._update_manifest(manifest, active=False)
                reason = "STALE_ACTIVE_BOOLEAN_RECONCILED"
            return manifest, ActiveObservation(
                False, None, None, reason, None
            )
        lock = (
            _read_json(self.paths.active_lock, "adapter active lock")
            if lock_present
            else None
        )
        pid_record = (
            _read_json(self.paths.active_pid, "adapter active PID")
            if pid_present
            else None
        )
        if lock is not None:
            self._validate_active_record(
                lock, manifest, ACTIVATION_LOCK_SCHEMA
            )
        if pid_record is not None:
            self._validate_active_record(
                pid_record, manifest, ACTIVATION_PID_SCHEMA
            )
        transaction_ids = {
            value["adapter_transaction_id"]
            for value in (lock, pid_record)
            if value is not None
        }
        if len(transaction_ids) != 1:
            _fail(
                "SUPERVISOR_IDENTITY_UNCERTAIN",
                "adapter active transaction identity is ambiguous",
                exit_code=3,
            )
        transaction_id = next(iter(transaction_ids))
        activation_record = lock or pid_record
        assert activation_record is not None
        activation = _process_observation(
            activation_record["activation_identity"]
        )
        supervisor_record = (
            pid_record.get("supervisor_identity")
            if pid_record is not None
            else lock.get("supervisor_identity")
        )
        supervisor_process = (
            _process_observation(supervisor_record)
            if isinstance(supervisor_record, dict)
            else None
        )
        if pid_present != lock_present:
            if activation is None and supervisor_process is None and supervisor is None:
                if pid_record is not None:
                    _unlink_transaction_owned(
                        self.paths.active_pid, transaction_id
                    )
                if lock is not None:
                    _unlink_transaction_owned(
                        self.paths.active_lock, transaction_id
                    )
                if manifest["active"]:
                    manifest = self._update_manifest(manifest, active=False)
                return manifest, ActiveObservation(
                    False,
                    None,
                    None,
                    "STALE_PARTIAL_ACTIVE_RECORD_RECONCILED",
                    transaction_id,
                )
            if (
                lock is not None
                and lock.get("activation_state") == "STARTING"
                and activation is not None
                and supervisor_record is None
            ):
                return manifest, ActiveObservation(
                    False,
                    None,
                    None,
                    "ACTIVATION_STARTING",
                    transaction_id,
                )
            _fail(
                "SUPERVISOR_IDENTITY_UNCERTAIN",
                "partial adapter active state refers to a live process",
                exit_code=3,
            )
        assert lock is not None and pid_record is not None
        if (
            lock["adapter_transaction_id"]
            != pid_record["adapter_transaction_id"]
            or lock.get("supervisor_identity")
            != pid_record.get("supervisor_identity")
        ):
            _fail(
                "SUPERVISOR_IDENTITY_UNCERTAIN",
                "adapter active records disagree",
                exit_code=3,
            )
        if activation is None and supervisor_process is None:
            if supervisor is not None:
                _fail(
                    "SUPERVISOR_IDENTITY_UNCERTAIN",
                    "stale adapter records conflict with a live supervisor",
                    exit_code=3,
                )
            self._remove_active_records(transaction_id)
            if manifest["active"]:
                manifest = self._update_manifest(manifest, active=False)
            return manifest, ActiveObservation(
                False,
                None,
                None,
                "STALE_ACTIVE_RECORDS_RECONCILED",
                transaction_id,
            )
        if activation is None or supervisor_process is None or supervisor is None:
            _fail(
                "SUPERVISOR_IDENTITY_UNCERTAIN",
                "attached activation and supervisor evidence is incomplete",
                exit_code=3,
            )
        live_identity, live_status = supervisor
        if not _same_process(supervisor_process, live_identity):
            _fail(
                "SUPERVISOR_IDENTITY_UNCERTAIN",
                "adapter and supervisor live identities disagree",
                exit_code=3,
            )
        if not manifest["active"]:
            manifest = self._update_manifest(manifest, active=True)
        return manifest, ActiveObservation(
            True,
            live_identity,
            live_status,
            None,
            transaction_id,
        )

    @staticmethod
    def _supervisor_summary(
        status: Mapping[str, Any] | None,
    ) -> dict[str, Any] | None:
        if status is None:
            return None
        return {
            name: status.get(name)
            for name in (
                "supervisor_state",
                "service_readiness_state",
                "model_service_state",
                "service_operational",
                "inference_ready",
                "reason_code",
                "supervisor_transaction_id",
                "desired_state",
                "desired_state_generation",
                "stop_reason",
                "fault_reason",
            )
        }

    def _status_value(
        self,
        manifest: Mapping[str, Any],
        desired: DesiredState,
        observation: ActiveObservation,
    ) -> dict[str, Any]:
        value = {
            "schema_version": STATUS_SCHEMA,
            "adapter_identity": ADAPTER_IDENTITY,
            "adapter_version": ADAPTER_VERSION,
            "configuration_identity": manifest["configuration_identity"],
            "registered": True,
            "enabled": manifest["enabled"],
            "active": observation.active,
            "automatic_activation_supported": False,
            "required_host_capability_result": self._capability_value(),
            "profile_identity": manifest["configuration_reference"][
                "profile_identity"
            ],
            "desired_state": desired.desired_state,
            "desired_state_generation": desired.generation,
            "supervisor_status_summary": self._supervisor_summary(
                observation.supervisor_status
            ),
            "supervisor_process_identity": observation.supervisor_identity,
            "last_activation_result": manifest["last_activation_result"],
            "last_failure_reason": manifest["last_failure_reason"],
            "reconciliation_reason": observation.reconciliation_reason,
            "resolved_paths": self.paths.status_paths(),
            "observed_utc": utc_now(),
        }
        _atomic_write_json(self.paths.status, value)
        return value

    def _state_result(
        self,
        operation: str,
        manifest: Mapping[str, Any],
        desired: DesiredState,
        observation: ActiveObservation,
        *,
        message: str,
        data: Mapping[str, Any] | None = None,
        reason_code: str = "OK",
    ) -> dict[str, Any]:
        return result_envelope(
            operation,
            ok=True,
            reason_code=reason_code,
            message=message,
            configuration_identity=manifest["configuration_identity"],
            registered=True,
            enabled=manifest["enabled"],
            active=observation.active,
            profile_identity=manifest["configuration_reference"][
                "profile_identity"
            ],
            desired_state=desired.desired_state,
            desired_state_generation=desired.generation,
            paths=self.paths.view(),
            data=data,
        )

    def identify(self) -> dict[str, Any]:
        return result_envelope(
            "identify",
            ok=True,
            reason_code="OK",
            message="foreground process-host adapter identified",
            paths=self.paths.view(),
            data={
                "supported_platform_family": PLATFORM_FAMILY,
                "activation_method": ACTIVATION_METHOD,
                "operation_set": list(OPERATIONS),
            },
        )

    def capability(
        self, requested: Sequence[str] | None = None
    ) -> dict[str, Any]:
        value = self._capability_value(requested)
        if not value["available"]:
            _fail(
                "HOST_CAPABILITY_MISSING",
                "one or more required host capabilities are unavailable",
                data=value,
            )
        return result_envelope(
            "capability",
            ok=True,
            reason_code="OK",
            message="foreground adapter host capabilities validated",
            paths=self.paths.view(),
            data={
                "adapter_identity": ADAPTER_IDENTITY,
                "supported_platform_family": PLATFORM_FAMILY,
                **value,
                "automatic_activation_supported": False,
            },
        )

    def validate(
        self,
        *,
        profile_path: Path | str | None = None,
        state_path: Path | str | None = None,
        supervisor_runtime_root: Path | str | None = None,
        supervisor_entrypoint: Path | str | None = None,
    ) -> dict[str, Any]:
        if self._manifest_exists():
            manifest, _profile, desired = self._load_manifest()
            manifest, observation = self._reconcile_active(manifest)
            status = self._status_value(manifest, desired, observation)
            return self._state_result(
                "validate",
                manifest,
                desired,
                observation,
                message="registered foreground adapter validates",
                data={"status": status},
            )
        if any(
            value is None
            for value in (
                profile_path,
                state_path,
                supervisor_runtime_root,
                supervisor_entrypoint,
            )
        ):
            _fail(
                "ADAPTER_NOT_REGISTERED",
                "unregistered validation requires complete configuration",
            )
        config, profile, desired, _entrypoint = (
            self._validated_configuration(
                profile_path=profile_path,
                state_path=state_path,
                supervisor_runtime_root=supervisor_runtime_root,
                supervisor_entrypoint=supervisor_entrypoint,
            )
        )
        return result_envelope(
            "validate",
            ok=True,
            reason_code="OK",
            message="foreground adapter configuration validates without mutation",
            configuration_identity=config["configuration_identity"],
            profile_identity=profile.identity,
            desired_state=desired.desired_state,
            desired_state_generation=desired.generation,
            paths=self.paths.view(),
            data=config,
        )

    def register(
        self,
        *,
        profile_path: Path | str,
        state_path: Path | str,
        supervisor_runtime_root: Path | str,
        supervisor_entrypoint: Path | str,
    ) -> dict[str, Any]:
        if self._manifest_exists():
            _fail(
                "ADAPTER_ALREADY_REGISTERED",
                "foreground adapter is already registered",
            )
        config, profile, desired, _entrypoint = (
            self._validated_configuration(
                profile_path=profile_path,
                state_path=state_path,
                supervisor_runtime_root=supervisor_runtime_root,
                supervisor_entrypoint=supervisor_entrypoint,
            )
        )
        self._prepare_runtime()
        timestamp = utc_now()
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "adapter_identity": ADAPTER_IDENTITY,
            "adapter_version": ADAPTER_VERSION,
            "supported_platform_family": PLATFORM_FAMILY,
            "required_host_capabilities": list(REQUIRED_HOST_CAPABILITIES),
            "activation_method": ACTIVATION_METHOD,
            "automatic_activation_supported": False,
            "supervisor_entrypoint": config["supervisor_entrypoint"],
            "configuration_reference": config["configuration_reference"],
            "configuration_identity": config["configuration_identity"],
            "registered": True,
            "enabled": False,
            "active": False,
            "manifest_generation": 1,
            "registered_utc": timestamp,
            "updated_utc": timestamp,
            "last_activation_result": None,
            "last_failure_reason": None,
        }
        _exclusive_write_json(
            self.paths.manifest,
            manifest,
            conflict_reason="ADAPTER_ALREADY_REGISTERED",
        )
        observation = ActiveObservation(False, None, None, None, None)
        status = self._status_value(manifest, desired, observation)
        return self._state_result(
            "register",
            manifest,
            desired,
            observation,
            message="foreground adapter registered without activation",
            data={
                "status": status,
                "desired_state_preserved": True,
                "process_started": False,
            },
        )

    def enable(self) -> dict[str, Any]:
        manifest, _profile, desired = self._load_manifest()
        manifest, observation = self._reconcile_active(manifest)
        if manifest["enabled"]:
            _fail(
                "ADAPTER_ALREADY_ENABLED",
                "foreground adapter is already enabled",
            )
        manifest = self._update_manifest(manifest, enabled=True)
        status = self._status_value(manifest, desired, observation)
        return self._state_result(
            "enable",
            manifest,
            desired,
            observation,
            message="foreground adapter enabled without boot activation",
            data={"status": status, "automatic_activation_created": False},
        )

    def disable(self) -> dict[str, Any]:
        manifest, _profile, desired = self._load_manifest()
        manifest, observation = self._reconcile_active(manifest)
        if observation.active:
            _fail(
                "ADAPTER_ALREADY_ACTIVE",
                "active foreground adapter must be stopped before disable",
                exit_code=3,
            )
        if not manifest["enabled"]:
            _fail(
                "ADAPTER_ALREADY_DISABLED",
                "foreground adapter is already disabled",
            )
        manifest = self._update_manifest(manifest, enabled=False)
        status = self._status_value(manifest, desired, observation)
        return self._state_result(
            "disable",
            manifest,
            desired,
            observation,
            message="inactive foreground adapter disabled",
            data={"status": status},
        )

    def _fail_closed_absent(self, manifest: Mapping[str, Any]) -> None:
        root = Path(
            manifest["configuration_reference"]["supervisor_runtime_root"]
        )
        candidates = (
            root / "recovery/fail-closed/active.json",
            root / "recovery/fail-closed/api-runtime.json",
        )
        present = [
            str(path)
            for path in candidates
            if path.exists() or path.is_symlink()
        ]
        if present:
            _fail(
                "SUPERVISOR_START_FAILED",
                "active fail-closed recovery latch blocks activation",
                exit_code=3,
                data={"reason_code": "FAIL_CLOSED_LATCHED", "paths": present},
            )

    def _controller_arguments(
        self,
        api_controller: Path | str | None,
        branch_controller: Path | str | None,
    ) -> tuple[dict[str, str], dict[str, str]]:
        api = _direct_file_identity(
            api_controller or DEFAULT_API_CONTROLLER,
            "API controller configuration",
            reason_code="ADAPTER_CONFIGURATION_CONFLICT",
        )
        branch = _direct_file_identity(
            branch_controller or DEFAULT_BRANCH_CONTROLLER,
            "branch controller configuration",
            reason_code="ADAPTER_CONFIGURATION_CONFLICT",
        )
        return api, branch

    def _new_transaction_id(self) -> str:
        return "pa-" + utc_now().replace("-", "").replace(":", "").replace(
            ".", ""
        ) + "-" + secrets.token_hex(6)

    def _write_transaction(
        self, transaction_id: str, value: Mapping[str, Any]
    ) -> None:
        _atomic_write_json(self.paths.transaction(transaction_id), value)

    def _acquire_reservation(
        self,
        manifest: Mapping[str, Any],
        transaction_id: str,
        activation_identity: Mapping[str, Any],
    ) -> dict[str, Any]:
        value = {
            "schema_version": ACTIVATION_LOCK_SCHEMA,
            "adapter_transaction_id": transaction_id,
            "configuration_identity": manifest["configuration_identity"],
            "profile_identity": manifest["configuration_reference"][
                "profile_identity"
            ],
            "activation_state": "STARTING",
            "activation_identity": dict(activation_identity),
            "supervisor_identity": None,
            "created_utc": utc_now(),
        }
        _exclusive_write_json(
            self.paths.active_lock,
            value,
            conflict_reason="ADAPTER_ALREADY_ACTIVE",
        )
        return value

    def _open_log(self, path: Path) -> Any:
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        return os.fdopen(descriptor, "wb", closefd=True)

    @staticmethod
    def _bounded_log(path: Path) -> dict[str, Any]:
        try:
            metadata = path.stat()
            size = metadata.st_size
            with path.open("rb") as handle:
                data = handle.read(MAX_LOG_BYTES + 1)
        except OSError as exc:
            return {"bytes": None, "sha256": None, "error": str(exc)[:256]}
        return {
            "bytes": size,
            "sha256": hashlib.sha256(data[:MAX_LOG_BYTES]).hexdigest(),
            "bounded": size > MAX_LOG_BYTES,
        }

    def _wait_for_child_evidence(
        self,
        child: subprocess.Popen[bytes],
        manifest: Mapping[str, Any],
        timeout_seconds: float,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if child.poll() is not None:
                _fail(
                    "SUPERVISOR_START_FAILED",
                    "supervisor exited before publishing exact active evidence",
                    exit_code=4,
                    data={"supervisor_exit_status": child.returncode},
                )
            try:
                evidence = self._supervisor_evidence(manifest)
            except AdapterError as exc:
                if exc.reason_code != "ADAPTER_NOT_REGISTERED":
                    raise
                evidence = None
            if evidence is not None:
                identity, status = evidence
                if identity["pid"] != child.pid:
                    _fail(
                        "SUPERVISOR_IDENTITY_UNCERTAIN",
                        "spawned child and supervisor record PID disagree",
                        exit_code=3,
                    )
                return identity, status
            time.sleep(0.01)
        _fail(
            "SUPERVISOR_START_FAILED",
            "supervisor did not publish active evidence before timeout",
            exit_code=4,
        )

    def _publish_active_records(
        self,
        manifest: Mapping[str, Any],
        reservation: Mapping[str, Any],
        supervisor_identity: Mapping[str, Any],
    ) -> None:
        lock = {
            **dict(reservation),
            "activation_state": "ACTIVE",
            "supervisor_identity": dict(supervisor_identity),
        }
        _atomic_write_json(self.paths.active_lock, lock)
        pid_record = {
            **lock,
            "schema_version": ACTIVATION_PID_SCHEMA,
        }
        try:
            _exclusive_write_json(
                self.paths.active_pid,
                pid_record,
                conflict_reason="ADAPTER_ALREADY_ACTIVE",
            )
        except Exception:
            transaction_id = str(reservation["adapter_transaction_id"])
            _unlink_transaction_owned(
                self.paths.active_lock, transaction_id
            )
            raise

    def _relay_signal(
        self,
        signum: int,
        supervisor_identity: Mapping[str, Any],
    ) -> dict[str, Any]:
        observed = _process_observation(supervisor_identity)
        if observed is None:
            return {
                "signal": signal.Signals(signum).name,
                "sent": False,
                "reason_code": "SUPERVISOR_ABSENT",
            }
        try:
            os.kill(int(observed["pid"]), signum)
        except OSError as exc:
            _fail(
                "SUPERVISOR_IDENTITY_UNCERTAIN",
                f"validated supervisor signal failed: {exc}",
                exit_code=4,
            )
        return {
            "signal": signal.Signals(signum).name,
            "sent": True,
            "target": observed,
            "identity_revalidated_immediately": True,
        }

    def _activate(
        self,
        operation: str,
        *,
        api_controller: Path | str | None,
        branch_controller: Path | str | None,
        controller_timeout_seconds: float,
        monitor_interval_seconds: float,
        activation_timeout_seconds: float,
    ) -> dict[str, Any]:
        manifest, profile, desired = self._load_manifest()
        manifest, observation = self._reconcile_active(manifest)
        if not manifest["enabled"]:
            _fail("ADAPTER_DISABLED", "foreground adapter is disabled")
        if observation.active or observation.reconciliation_reason == "ACTIVATION_STARTING":
            _fail(
                "ADAPTER_ALREADY_ACTIVE",
                "foreground adapter or supervisor is already active",
                exit_code=3,
            )
        if desired.desired_state != "RUNNING":
            _fail(
                "DESIRED_STATE_INVALID",
                "adapter start requires desired state RUNNING",
            )
        self._fail_closed_absent(manifest)
        for label, value, minimum, maximum in (
            (
                "controller timeout",
                controller_timeout_seconds,
                0.05,
                300.0,
            ),
            (
                "monitor interval",
                monitor_interval_seconds,
                0.01,
                10.0,
            ),
            (
                "activation timeout",
                activation_timeout_seconds,
                0.1,
                300.0,
            ),
        ):
            if (
                type(value) not in (int, float)
                or not math.isfinite(float(value))
                or not minimum <= float(value) <= maximum
            ):
                _fail(
                    "ADAPTER_CONFIGURATION_CONFLICT",
                    f"{label} is outside the bounded range",
                )
        api, branch = self._controller_arguments(
            api_controller, branch_controller
        )
        self._prepare_runtime()
        transaction_id = self._new_transaction_id()
        activation_identity = process_snapshot(os.getpid())
        transaction = {
            "schema_version": ACTIVATION_TRANSACTION_SCHEMA,
            "adapter_transaction_id": transaction_id,
            "operation": operation,
            "configuration_identity": manifest["configuration_identity"],
            "profile_identity": profile.identity,
            "activation_identity": activation_identity,
            "supervisor_identity": None,
            "supervisor_argv": None,
            "controller_configuration": {
                "api": api,
                "branch": branch,
                "controller_timeout_seconds": float(
                    controller_timeout_seconds
                ),
                "monitor_interval_seconds": float(
                    monitor_interval_seconds
                ),
            },
            "start_utc": utc_now(),
            "active_utc": None,
            "completion_utc": None,
            "supervisor_exit_status": None,
            "signals_relayed": [],
            "stdout_evidence": None,
            "stderr_evidence": None,
            "outcome": None,
        }
        _exclusive_write_json(
            self.paths.transaction(transaction_id),
            transaction,
            conflict_reason="ADAPTER_CONFIGURATION_CONFLICT",
        )
        reservation = self._acquire_reservation(
            manifest, transaction_id, activation_identity
        )
        reference = manifest["configuration_reference"]
        entrypoint = manifest["supervisor_entrypoint"]
        argv = [
            sys.executable,
            entrypoint["path"],
            "run",
            "--profile",
            reference["profile_path"],
            "--state-path",
            reference["state_path"],
            "--runtime-root",
            reference["supervisor_runtime_root"],
            "--api-controller",
            api["path"],
            "--api-controller-sha256",
            api["sha256"],
            "--branch-controller",
            branch["path"],
            "--branch-controller-sha256",
            branch["sha256"],
            "--controller-timeout-seconds",
            str(float(controller_timeout_seconds)),
            "--monitor-interval-seconds",
            str(float(monitor_interval_seconds)),
        ]
        transaction["supervisor_argv"] = argv
        self._write_transaction(transaction_id, transaction)
        child: subprocess.Popen[bytes] | None = None
        supervisor_identity: dict[str, Any] | None = None
        previous_handlers: dict[int, Any] = {}
        pending_signals: list[int] = []
        try:
            stdout_handle = self._open_log(
                self.paths.supervisor_stdout(transaction_id)
            )
            stderr_handle = self._open_log(
                self.paths.supervisor_stderr(transaction_id)
            )
            try:
                child = subprocess.Popen(
                    argv,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    shell=False,
                    start_new_session=False,
                    close_fds=True,
                )
            finally:
                stdout_handle.close()
                stderr_handle.close()
            supervisor_identity, supervisor_status = (
                self._wait_for_child_evidence(
                    child,
                    manifest,
                    float(activation_timeout_seconds),
                )
            )
            self._publish_active_records(
                manifest, reservation, supervisor_identity
            )
            manifest = self._update_manifest(
                manifest,
                active=True,
                last_activation_result=bounded_activation_result(
                    operation=operation,
                    ok=True,
                    reason_code="OK",
                    message="exact attached supervisor identity observed",
                ),
                last_failure_reason=None,
            )
            active_observation = ActiveObservation(
                True,
                supervisor_identity,
                supervisor_status,
                None,
                transaction_id,
            )
            self._status_value(manifest, desired, active_observation)
            transaction.update(
                {
                    "supervisor_identity": supervisor_identity,
                    "active_utc": utc_now(),
                }
            )
            self._write_transaction(transaction_id, transaction)

            def relay_handler(signum: int, _frame: Any) -> None:
                pending_signals.append(signum)

            for signum in (signal.SIGTERM, signal.SIGINT):
                previous_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, relay_handler)
            while child.poll() is None:
                while pending_signals:
                    relayed = self._relay_signal(
                        pending_signals.pop(0), supervisor_identity
                    )
                    transaction["signals_relayed"].append(relayed)
                    self._write_transaction(transaction_id, transaction)
                time.sleep(0.02)
            exit_status = int(child.returncode)
            for signum, previous in previous_handlers.items():
                signal.signal(signum, previous)
            previous_handlers.clear()
            self._remove_active_records(transaction_id)
            manifest, _profile, desired = self._load_manifest()
            supervisor_paths = SupervisorPaths(
                Path(reference["supervisor_runtime_root"])
            )
            final_supervisor_status = (
                _read_json(
                    supervisor_paths.status_record, "supervisor final status"
                )
                if supervisor_paths.status_record.exists()
                else None
            )
            clean = bool(
                exit_status == 0
                and isinstance(final_supervisor_status, dict)
                and final_supervisor_status.get("supervisor_state")
                == "STOPPED"
            )
            summary = bounded_activation_result(
                operation=operation,
                ok=clean,
                reason_code="OK" if clean else "SUPERVISOR_START_FAILED",
                message=(
                    "foreground supervisor exited cleanly"
                    if clean
                    else "foreground supervisor exited without clean STOPPED state"
                ),
                supervisor_exit_status=exit_status,
            )
            manifest = self._update_manifest(
                manifest,
                active=False,
                last_activation_result=summary,
                last_failure_reason=(
                    None if clean else "SUPERVISOR_START_FAILED"
                ),
            )
            inactive = ActiveObservation(
                False,
                None,
                None,
                None,
                transaction_id,
            )
            status = self._status_value(manifest, desired, inactive)
            transaction.update(
                {
                    "completion_utc": utc_now(),
                    "supervisor_exit_status": exit_status,
                    "stdout_evidence": self._bounded_log(
                        self.paths.supervisor_stdout(transaction_id)
                    ),
                    "stderr_evidence": self._bounded_log(
                        self.paths.supervisor_stderr(transaction_id)
                    ),
                    "outcome": "STOPPED" if clean else "FAILED",
                }
            )
            self._write_transaction(transaction_id, transaction)
            if not clean:
                _fail(
                    (
                        "SUPERVISOR_RESTART_FAILED"
                        if operation == "restart"
                        else "SUPERVISOR_START_FAILED"
                    ),
                    "attached supervisor did not exit cleanly",
                    exit_code=4,
                    data={"transaction": transaction},
                )
            return self._state_result(
                operation,
                manifest,
                desired,
                inactive,
                message=(
                    "foreground restart activation exited cleanly"
                    if operation == "restart"
                    else "foreground activation exited cleanly"
                ),
                data={
                    "adapter_transaction_id": transaction_id,
                    "supervisor_identity": supervisor_identity,
                    "status": status,
                    "shell": False,
                    "start_new_session": False,
                },
            )
        except (AdapterError, OSError, subprocess.SubprocessError) as exc:
            for signum, previous in previous_handlers.items():
                signal.signal(signum, previous)
            if child is not None and child.poll() is None:
                if supervisor_identity is not None:
                    self._relay_signal(signal.SIGTERM, supervisor_identity)
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
            try:
                self._remove_active_records(transaction_id)
            except AdapterError:
                pass
            reason = (
                exc.reason_code
                if isinstance(exc, AdapterError)
                else (
                    "SUPERVISOR_RESTART_FAILED"
                    if operation == "restart"
                    else "SUPERVISOR_START_FAILED"
                )
            )
            try:
                current_manifest, _profile, current_desired = (
                    self._load_manifest()
                )
                current_manifest = self._update_manifest(
                    current_manifest,
                    active=False,
                    last_activation_result=bounded_activation_result(
                        operation=operation,
                        ok=False,
                        reason_code=reason,
                        message=str(exc),
                        supervisor_exit_status=(
                            child.returncode
                            if child is not None
                            else None
                        ),
                    ),
                    last_failure_reason=reason,
                )
                self._status_value(
                    current_manifest,
                    current_desired,
                    ActiveObservation(
                        False, None, None, None, transaction_id
                    ),
                )
            except Exception:
                pass
            transaction.update(
                {
                    "completion_utc": utc_now(),
                    "supervisor_exit_status": (
                        child.returncode if child is not None else None
                    ),
                    "stdout_evidence": self._bounded_log(
                        self.paths.supervisor_stdout(transaction_id)
                    ),
                    "stderr_evidence": self._bounded_log(
                        self.paths.supervisor_stderr(transaction_id)
                    ),
                    "outcome": "FAILED",
                    "failure_reason": reason,
                }
            )
            self._write_transaction(transaction_id, transaction)
            if isinstance(exc, AdapterError):
                raise
            _fail(
                (
                    "SUPERVISOR_RESTART_FAILED"
                    if operation == "restart"
                    else "SUPERVISOR_START_FAILED"
                ),
                f"supervisor activation failed: {type(exc).__name__}: {exc}",
                exit_code=4,
            )

    def start(
        self,
        *,
        api_controller: Path | str | None = None,
        branch_controller: Path | str | None = None,
        controller_timeout_seconds: float = 180.0,
        monitor_interval_seconds: float = 0.25,
        activation_timeout_seconds: float = 30.0,
    ) -> dict[str, Any]:
        return self._activate(
            "start",
            api_controller=api_controller,
            branch_controller=branch_controller,
            controller_timeout_seconds=controller_timeout_seconds,
            monitor_interval_seconds=monitor_interval_seconds,
            activation_timeout_seconds=activation_timeout_seconds,
        )

    def stop(
        self,
        *,
        wait_timeout_seconds: float = 60.0,
    ) -> dict[str, Any]:
        manifest, _profile, desired = self._load_manifest()
        manifest, observation = self._reconcile_active(manifest)
        if not observation.active:
            _fail(
                "ADAPTER_INACTIVE",
                "foreground adapter has no exact active supervisor",
                exit_code=3,
            )
        reference = manifest["configuration_reference"]
        try:
            supervisor_result = administrative_stop(
                Path(reference["profile_path"]),
                Path(reference["state_path"]),
                Path(reference["supervisor_runtime_root"]),
                expected_generation=desired.generation,
                wait_timeout_seconds=wait_timeout_seconds,
            )
        except (SupervisorError, ServiceControlError) as exc:
            reason = getattr(exc, "reason_code", type(exc).__name__)
            manifest = self._update_manifest(
                manifest,
                last_activation_result=bounded_activation_result(
                    operation="stop",
                    ok=False,
                    reason_code="SUPERVISOR_STOP_FAILED",
                    message=str(exc),
                ),
                last_failure_reason="SUPERVISOR_STOP_FAILED",
            )
            _fail(
                "SUPERVISOR_STOP_FAILED",
                f"supervisor administrative stop failed: {reason}",
                exit_code=4,
            )
        deadline = time.monotonic() + float(wait_timeout_seconds)
        while time.monotonic() < deadline:
            if not (
                self.paths.active_lock.exists()
                or self.paths.active_pid.exists()
            ):
                break
            time.sleep(0.01)
        else:
            _fail(
                "SUPERVISOR_STOP_FAILED",
                "adapter activation records remained after supervisor stop",
                exit_code=4,
            )
        manifest, _profile, desired = self._load_manifest()
        manifest = self._update_manifest(
            manifest,
            active=False,
            last_activation_result=bounded_activation_result(
                operation="stop",
                ok=True,
                reason_code="OK",
                message="supervisor administrative stop completed",
                supervisor_exit_status=0,
            ),
            last_failure_reason=None,
        )
        inactive = ActiveObservation(False, None, None, None, None)
        status = self._status_value(manifest, desired, inactive)
        return self._state_result(
            "stop",
            manifest,
            desired,
            inactive,
            message="foreground supervisor stopped administratively",
            data={"supervisor_result": supervisor_result, "status": status},
        )

    def restart(
        self,
        *,
        api_controller: Path | str | None = None,
        branch_controller: Path | str | None = None,
        controller_timeout_seconds: float = 180.0,
        monitor_interval_seconds: float = 0.25,
        activation_timeout_seconds: float = 30.0,
        wait_timeout_seconds: float = 60.0,
    ) -> dict[str, Any]:
        manifest, profile, desired = self._load_manifest()
        manifest, observation = self._reconcile_active(manifest)
        if observation.active:
            self.stop(wait_timeout_seconds=wait_timeout_seconds)
            manifest, profile, desired = self._load_manifest()
        if not manifest["enabled"]:
            _fail("ADAPTER_DISABLED", "foreground adapter is disabled")
        try:
            set_desired_state(
                profile,
                "RUNNING",
                manifest["configuration_reference"]["state_path"],
                expected_generation=desired.generation,
            )
        except ServiceControlError as exc:
            _fail(
                "SUPERVISOR_RESTART_FAILED",
                f"generation-safe RUNNING update failed: {exc.reason_code}",
                exit_code=4,
            )
        return self._activate(
            "restart",
            api_controller=api_controller,
            branch_controller=branch_controller,
            controller_timeout_seconds=controller_timeout_seconds,
            monitor_interval_seconds=monitor_interval_seconds,
            activation_timeout_seconds=activation_timeout_seconds,
        )

    def status(self) -> dict[str, Any]:
        manifest, _profile, desired = self._load_manifest()
        manifest, observation = self._reconcile_active(manifest)
        status = self._status_value(manifest, desired, observation)
        return self._state_result(
            "status",
            manifest,
            desired,
            observation,
            message="foreground adapter status reconciled",
            data={"status": status},
        )

    def configuration(self) -> dict[str, Any]:
        manifest, _profile, desired = self._load_manifest()
        manifest, observation = self._reconcile_active(manifest)
        self._status_value(manifest, desired, observation)
        return self._state_result(
            "configuration",
            manifest,
            desired,
            observation,
            message="foreground adapter configuration reported",
            data={
                "configuration_reference": manifest[
                    "configuration_reference"
                ],
                "configuration_identity": manifest[
                    "configuration_identity"
                ],
            },
        )

    def supervisor_entrypoint(self) -> dict[str, Any]:
        manifest, _profile, desired = self._load_manifest()
        manifest, observation = self._reconcile_active(manifest)
        self._status_value(manifest, desired, observation)
        return self._state_result(
            "supervisor-entrypoint",
            manifest,
            desired,
            observation,
            message="validated supervisor entrypoint reported",
            data={
                "supervisor_entrypoint": manifest[
                    "supervisor_entrypoint"
                ]
            },
        )

    def unregister(self, *, explicit: bool = False) -> dict[str, Any]:
        if not explicit:
            _fail(
                "UNREGISTER_NOT_EXPLICIT",
                "unregister requires the exact explicit operation",
            )
        manifest, _profile, desired = self._load_manifest()
        manifest, observation = self._reconcile_active(manifest)
        if observation.active:
            _fail(
                "UNREGISTER_REQUIRES_INACTIVE",
                "active foreground adapter cannot be unregistered",
                exit_code=3,
            )
        if manifest["enabled"]:
            _fail(
                "UNREGISTER_REQUIRES_DISABLED",
                "foreground adapter must be disabled before unregister",
                exit_code=3,
            )
        if (
            self.paths.active_lock.exists()
            or self.paths.active_pid.exists()
        ):
            _fail(
                "UNREGISTER_REQUIRES_INACTIVE",
                "adapter active records must be absent before unregister",
                exit_code=3,
            )
        for path in (self.paths.status, self.paths.manifest):
            _reject_symlink_components(path, path.name)
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                _fail(
                    "ADAPTER_MANIFEST_INVALID",
                    f"adapter current state cannot be removed: {exc}",
                )
            _fsync_directory(path.parent)
        return result_envelope(
            "unregister",
            ok=True,
            reason_code="OK",
            message="foreground adapter current registration removed explicitly",
            registered=False,
            enabled=False,
            active=False,
            profile_identity=desired.profile_identity,
            desired_state=desired.desired_state,
            desired_state_generation=desired.generation,
            paths=self.paths.view(),
            data={
                "configuration_identity": manifest[
                    "configuration_identity"
                ],
                "historical_transactions_preserved": True,
                "supervisor_history_preserved": True,
            },
        )

    def invoke(self, operation: str, **arguments: Any) -> dict[str, Any]:
        methods = {
            "identify": self.identify,
            "validate": self.validate,
            "register": self.register,
            "enable": self.enable,
            "disable": self.disable,
            "start": self.start,
            "stop": self.stop,
            "restart": self.restart,
            "status": self.status,
            "unregister": self.unregister,
            "capability": self.capability,
            "configuration": self.configuration,
            "supervisor-entrypoint": self.supervisor_entrypoint,
        }
        method = methods.get(operation)
        if method is None:
            _fail("ADAPTER_NOT_SUPPORTED", f"unknown operation: {operation}")
        return method(**arguments)

    def error_result(
        self, operation: str, error: AdapterError
    ) -> dict[str, Any]:
        manifest: dict[str, Any] | None = None
        desired: DesiredState | None = None
        try:
            manifest, _profile, desired = self._load_manifest()
        except Exception:
            pass
        return result_envelope(
            operation,
            ok=False,
            reason_code=error.reason_code,
            message=error.message,
            configuration_identity=(
                manifest.get("configuration_identity")
                if manifest is not None
                else None
            ),
            registered=manifest is not None,
            enabled=bool(manifest and manifest.get("enabled")),
            active=bool(manifest and manifest.get("active")),
            profile_identity=(
                manifest["configuration_reference"]["profile_identity"]
                if manifest is not None
                else None
            ),
            desired_state=(
                desired.desired_state if desired is not None else None
            ),
            desired_state_generation=(
                desired.generation if desired is not None else None
            ),
            paths=self.paths.view(),
            data=error.data,
        )


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        _fail("ADAPTER_MANIFEST_INVALID", message)


def _add_identity_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--adapter-identity", default=ADAPTER_IDENTITY
    )
    parser.add_argument(
        "--adapter-runtime-root", type=Path, required=True
    )


def _add_configuration_arguments(
    parser: argparse.ArgumentParser, *, required: bool
) -> None:
    parser.add_argument("--profile", type=Path, required=required)
    parser.add_argument("--state-path", type=Path, required=required)
    parser.add_argument(
        "--supervisor-runtime-root", type=Path, required=required
    )
    parser.add_argument(
        "--supervisor-entrypoint", type=Path, required=required
    )


def _add_activation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--api-controller", type=Path)
    parser.add_argument("--branch-controller", type=Path)
    parser.add_argument(
        "--controller-timeout-seconds", type=float, default=180.0
    )
    parser.add_argument(
        "--monitor-interval-seconds", type=float, default=0.25
    )
    parser.add_argument(
        "--activation-timeout-seconds", type=float, default=30.0
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(
        prog="system-x-platform-service-adapter", add_help=False
    )
    operations = parser.add_subparsers(dest="operation", required=True)
    for name in OPERATIONS:
        command = operations.add_parser(name, add_help=False)
        _add_identity_arguments(command)
        if name == "validate":
            _add_configuration_arguments(command, required=False)
        elif name == "register":
            _add_configuration_arguments(command, required=True)
        if name in {"start", "restart"}:
            _add_activation_arguments(command)
        if name in {"stop", "restart"}:
            command.add_argument(
                "--wait-timeout-seconds", type=float, default=60.0
            )
        if name == "capability":
            command.add_argument(
                "--require", action="append", default=None
            )
    return parser


def _operation_arguments(arguments: argparse.Namespace) -> dict[str, Any]:
    operation = arguments.operation
    if operation in {"validate", "register"}:
        return {
            "profile_path": arguments.profile,
            "state_path": arguments.state_path,
            "supervisor_runtime_root": arguments.supervisor_runtime_root,
            "supervisor_entrypoint": arguments.supervisor_entrypoint,
        }
    if operation in {"start", "restart"}:
        values = {
            "api_controller": arguments.api_controller,
            "branch_controller": arguments.branch_controller,
            "controller_timeout_seconds": (
                arguments.controller_timeout_seconds
            ),
            "monitor_interval_seconds": arguments.monitor_interval_seconds,
            "activation_timeout_seconds": (
                arguments.activation_timeout_seconds
            ),
        }
        if operation == "restart":
            values["wait_timeout_seconds"] = arguments.wait_timeout_seconds
        return values
    if operation == "stop":
        return {"wait_timeout_seconds": arguments.wait_timeout_seconds}
    if operation == "unregister":
        return {"explicit": True}
    if operation == "capability":
        return {"requested": arguments.require}
    return {}


def main(argv: Sequence[str] | None = None) -> int:
    values = list(argv) if argv is not None else sys.argv[1:]
    operation = values[0] if values else "identify"
    requested_identity = ADAPTER_IDENTITY
    adapter: ForegroundProcessHostAdapter | None = None
    try:
        arguments = build_argument_parser().parse_args(values)
        operation = arguments.operation
        requested_identity = arguments.adapter_identity
        adapter = create_adapter(
            requested_identity, arguments.adapter_runtime_root
        )
        output = adapter.invoke(
            operation, **_operation_arguments(arguments)
        )
        exit_code = 0
    except AdapterError as exc:
        output = (
            adapter.error_result(operation, exc)
            if adapter is not None
            else result_envelope(
                operation,
                ok=False,
                reason_code=exc.reason_code,
                message=exc.message,
                adapter_identity=requested_identity,
                data=exc.data,
            )
        )
        exit_code = exc.exit_code
    except Exception as exc:
        output = result_envelope(
            operation,
            ok=False,
            reason_code="ADAPTER_MANIFEST_INVALID",
            message=f"{type(exc).__name__}: {exc}",
            adapter_identity=requested_identity,
        )
        print(
            f"system-x-platform-adapter unexpected error: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        exit_code = 70
    print(canonical_json(output))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
