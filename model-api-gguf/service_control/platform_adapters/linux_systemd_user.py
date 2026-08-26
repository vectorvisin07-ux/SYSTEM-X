#!/usr/bin/env python3
"""Linux systemd user-service adapter for the System X supervisor.

The native manager receives one deterministic unit whose ExecStart invokes
only the platform-neutral supervisor.  This adapter never starts the API,
router, Uvicorn, llama-server, or a model child directly.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import shlex
import socket
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, NoReturn, Protocol, Sequence
from urllib.parse import unquote, urlsplit

if __package__:
    from .contract import (
        MANIFEST_SCHEMA,
        OPERATIONS,
        STATUS_SCHEMA,
        AdapterError,
        AdapterPathsView,
        bounded_activation_result,
        canonical_json,
        result_envelope,
        utc_now,
    )
    from ..operating_profile import (
        DesiredState,
        OperatingProfile,
        ServiceControlError,
        configure_static_profile,
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
        process_snapshot,
    )
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from service_control.platform_adapters.contract import (  # type: ignore
        MANIFEST_SCHEMA,
        OPERATIONS,
        STATUS_SCHEMA,
        AdapterError,
        AdapterPathsView,
        bounded_activation_result,
        canonical_json,
        result_envelope,
        utc_now,
    )
    from service_control.operating_profile import (  # type: ignore
        DesiredState,
        OperatingProfile,
        ServiceControlError,
        configure_static_profile,
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
        process_snapshot,
    )


ADAPTER_IDENTITY = "system-x.linux-systemd-user-service.v1"
ADAPTER_VERSION = "1.0.0"
PLATFORM_FAMILY = "linux-systemd-user"
ACTIVATION_METHOD = "systemd-user-unit"
AUTOMATIC_ACTIVATION_SUPPORTED = True
SERVICE_NAME = "system-x.service"
TRIAL_SERVICE_NAME = re.compile(r"^system-x-trial-[0-9a-f]{16}\.service$")
SYSTEMCTL = Path("/usr/bin/systemctl")
SYSTEMD_ANALYZE = Path("/usr/bin/systemd-analyze")
PYTHON = Path("/usr/bin/python3").resolve()
SOURCE_DIR = Path(__file__).resolve().parent
BRANCH_ROOT = SOURCE_DIR.parents[1]
DEFAULT_ADAPTER_RUNTIME_ROOT = (
    BRANCH_ROOT / "RUNTIME/service_control/platform_adapters"
)
DEFAULT_UNIT_PATH = (
    Path.home() / ".config/systemd/user" / SERVICE_NAME
)
REQUIRED_HOST_CAPABILITIES = (
    "linux_systemd_pid1",
    "systemd_user_manager",
    "user_unit_registration",
    "manager_enable_disable",
    "manager_start_stop_status",
    "manager_main_pid_observation",
    "manager_restart_on_failure",
    "manager_owned_journal",
)
MAX_RECORD_BYTES = 1_048_576
STATIC_REFERENCE_PATTERN = re.compile(
    r'''(?:src|href)\s*=\s*["']([^"']+)["']''',
    re.IGNORECASE,
)
STATIC_MOUNT_PATTERN = re.compile(
    r"\A/[a-z0-9][a-z0-9._-]*(?:/[a-z0-9][a-z0-9._-]*)*\Z"
)
STATIC_RESERVED_PREFIXES = (
    "/system",
    "/v1",
    "/openapi.json",
    "/docs",
    "/redoc",
)
SAFE_UNIT_TOKEN = re.compile(r"^[A-Za-z0-9_./:=+@,-]+$")
SYSTEMD_PATH_TOKEN = re.compile(r"^[A-Za-z0-9_./:=+@, -]+$")
MANAGER_PROPERTIES = (
    "LoadState",
    "ActiveState",
    "SubState",
    "UnitFileState",
    "FragmentPath",
    "MainPID",
    "ExecMainPID",
    "ExecMainCode",
    "ExecMainStatus",
    "Result",
    "NRestarts",
    "InvocationID",
    "ActiveEnterTimestampMonotonic",
)
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
        "native_service",
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
_INTERPRETER_IDENTITY_FIELDS = frozenset(
    ("path", "device", "inode", "mode", "size", "mtime_ns")
)
_NATIVE_SERVICE_FIELDS = frozenset(
    (
        "manager",
        "manager_scope",
        "service_name",
        "registration_path",
        "service_definition_sha256",
        "interpreter_identity",
        "api_controller",
        "branch_controller",
        "timeout_stop_seconds",
    )
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


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _repository_owner() -> tuple[int, int]:
    home = Path.home()
    try:
        metadata = home.lstat()
    except OSError as exc:
        _fail("SERVICE_OWNER_INVALID", f"repository home cannot be inspected: {exc}")
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid == 0:
        _fail(
            "SERVICE_OWNER_INVALID",
            "repository home must be a non-root directory owner",
        )
    return metadata.st_uid, metadata.st_gid


def _adopt_user_owned_directory(path: Path, label: str) -> None:
    _reject_symlink_components(path, label)
    try:
        metadata = path.lstat()
    except OSError as exc:
        _fail("ADAPTER_MANIFEST_INVALID", f"{label} unavailable: {exc}")
    if not stat.S_ISDIR(metadata.st_mode):
        _fail("ADAPTER_MANIFEST_INVALID", f"{label} must be a directory")
    owner_uid, owner_gid = _repository_owner()
    if metadata.st_uid != owner_uid or metadata.st_gid != owner_gid:
        if os.geteuid() != 0:
            _fail("FOREIGN_SERVICE_DEFINITION", f"{label} is not owned by the repository user")
        try:
            if any(path.iterdir()):
                _fail(
                    "FOREIGN_SERVICE_DEFINITION",
                    f"{label} owner differs while it contains entries",
                )
            os.chown(path, owner_uid, owner_gid, follow_symlinks=False)
        except OSError as exc:
            _fail("FOREIGN_SERVICE_DEFINITION", f"{label} ownership handoff failed: {exc}")
    try:
        os.chmod(path, 0o700)
        _fsync_directory(path)
    except OSError as exc:
        _fail("ADAPTER_MANIFEST_INVALID", f"{label} mode or sync failed: {exc}")


def _adopt_user_owned_file(path: Path, label: str) -> None:
    _reject_symlink_components(path, label)
    try:
        metadata = path.lstat()
    except OSError as exc:
        _fail("ADAPTER_MANIFEST_INVALID", f"{label} unavailable: {exc}")
    if not stat.S_ISREG(metadata.st_mode):
        _fail("ADAPTER_MANIFEST_INVALID", f"{label} must be a regular file")
    owner_uid, owner_gid = _repository_owner()
    if metadata.st_uid != owner_uid or metadata.st_gid != owner_gid:
        if os.geteuid() != 0:
            _fail("FOREIGN_SERVICE_DEFINITION", f"{label} is not owned by the repository user")
        try:
            os.chown(path, owner_uid, owner_gid, follow_symlinks=False)
        except OSError as exc:
            _fail("FOREIGN_SERVICE_DEFINITION", f"{label} ownership handoff failed: {exc}")
    try:
        os.chmod(path, 0o600)
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        _fail("ADAPTER_MANIFEST_INVALID", f"{label} mode or sync failed: {exc}")


def _adopt_user_owned_tree(path: Path, label: str) -> None:
    _reject_symlink_components(path, label)
    try:
        metadata = path.lstat()
    except OSError as exc:
        _fail("ADAPTER_MANIFEST_INVALID", f"{label} unavailable: {exc}")
    if not stat.S_ISDIR(metadata.st_mode):
        _fail("ADAPTER_MANIFEST_INVALID", f"{label} must be a directory")
    owner_uid, owner_gid = _repository_owner()
    for current, directories, files in os.walk(path, topdown=True, followlinks=False):
        for name in (*directories, *files):
            entry = Path(current) / name
            try:
                entry_metadata = entry.lstat()
            except OSError as exc:
                _fail("ADAPTER_MANIFEST_INVALID", f"{label} entry cannot be inspected: {exc}")
            if stat.S_ISLNK(entry_metadata.st_mode) or not (
                stat.S_ISDIR(entry_metadata.st_mode)
                or stat.S_ISREG(entry_metadata.st_mode)
            ):
                _fail("ADAPTER_MANIFEST_INVALID", f"{label} contains an unsafe entry")
            if entry_metadata.st_uid != owner_uid or entry_metadata.st_gid != owner_gid:
                if os.geteuid() != 0:
                    _fail("FOREIGN_SERVICE_DEFINITION", f"{label} is not owned by the repository user")
                try:
                    os.chown(entry, owner_uid, owner_gid, follow_symlinks=False)
                except OSError as exc:
                    _fail("FOREIGN_SERVICE_DEFINITION", f"{label} ownership handoff failed: {exc}")
        try:
            os.chown(current, owner_uid, owner_gid, follow_symlinks=False)
            _fsync_directory(Path(current))
        except OSError as exc:
            _fail("FOREIGN_SERVICE_DEFINITION", f"{label} directory ownership handoff failed: {exc}")


def _reject_symlink_components(path: Path, label: str) -> None:
    absolute = _absolute(path)
    for component in list(reversed(absolute.parents)) + [absolute]:
        try:
            metadata = component.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            _fail(
                "PATH_INSPECTION_FAILED",
                f"{label} cannot be inspected: {exc}",
            )
        if stat.S_ISLNK(metadata.st_mode):
            _fail(
                "SERVICE_DEFINITION_SYMLINK",
                f"{label} contains a symlink component",
            )


def _ensure_private_directory(path: Path, label: str) -> None:
    _reject_symlink_components(path, label)
    try:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        metadata = path.lstat()
    except OSError as exc:
        _fail("ADAPTER_MANIFEST_INVALID", f"{label} unavailable: {exc}")
    if not stat.S_ISDIR(metadata.st_mode):
        _fail("ADAPTER_MANIFEST_INVALID", f"{label} must be a directory")
    try:
        os.chmod(path, 0o700)
    except OSError as exc:
        _fail("ADAPTER_MANIFEST_INVALID", f"{label} mode failed: {exc}")


def _directory_is_writable_or_creatable(path: Path, label: str) -> bool:
    """Return whether a private directory path can be created safely."""
    _reject_symlink_components(path, label)
    existing = _absolute(path)
    while not existing.exists():
        parent = existing.parent
        if parent == existing:
            return False
        existing = parent
    return (
        existing.is_dir()
        and os.access(existing, os.W_OK | os.X_OK)
    )

def _read_regular(path: Path, label: str) -> bytes:
    _reject_symlink_components(path, label)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        _fail("ADAPTER_NOT_REGISTERED", f"{label} is absent")
    except OSError as exc:
        _fail("ADAPTER_MANIFEST_INVALID", f"{label} cannot open: {exc}")
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            _fail(
                "ADAPTER_MANIFEST_INVALID",
                f"{label} must be a regular file",
            )
        if metadata.st_size > MAX_RECORD_BYTES:
            _fail(
                "ADAPTER_MANIFEST_INVALID",
                f"{label} exceeds the size limit",
            )
        data = b""
        while len(data) <= MAX_RECORD_BYTES:
            block = os.read(descriptor, 65_536)
            if not block:
                break
            data += block
        if len(data) > MAX_RECORD_BYTES:
            _fail(
                "ADAPTER_MANIFEST_INVALID",
                f"{label} exceeds the size limit",
            )
        return data
    finally:
        os.close(descriptor)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    data = _read_regular(path, label)
    if b"\x00" in data:
        _fail("ADAPTER_MANIFEST_INVALID", f"{label} contains a NUL")
    try:
        value = json.loads(data.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        _fail("ADAPTER_MANIFEST_INVALID", f"{label} invalid JSON: {exc}")
    if not isinstance(value, dict):
        _fail("ADAPTER_MANIFEST_INVALID", f"{label} must be an object")
    return value


def _validate_static_distribution(
    root_value: str,
    mount_path: str,
) -> list[str]:
    """Validate the unchanged distribution and every index asset reference."""
    if (
        not isinstance(mount_path, str)
        or mount_path != mount_path.strip()
        or "\x00" in mount_path
        or STATIC_MOUNT_PATTERN.fullmatch(mount_path) is None
        or any(
            mount_path == prefix or mount_path.startswith(prefix + "/")
            for prefix in STATIC_RESERVED_PREFIXES
        )
    ):
        _fail(
            "ADAPTER_CONFIGURATION_CONFLICT",
            "static mount path is not a normalized non-API path",
        )
    root = Path(root_value)
    if (
        not root.is_absolute()
        or str(root) != root_value
        or ".." in root.parts
    ):
        _fail(
            "ADAPTER_CONFIGURATION_CONFLICT",
            "static distribution root is not a normalized absolute path",
        )
    _reject_symlink_components(root, "static distribution root")
    try:
        root_metadata = root.lstat()
    except OSError as exc:
        _fail(
            "ADAPTER_CONFIGURATION_CONFLICT",
            f"static distribution root cannot be inspected: {exc}",
        )
    owner_uid, owner_gid = _repository_owner()
    if (
        not stat.S_ISDIR(root_metadata.st_mode)
        or root_metadata.st_uid != owner_uid
        or root_metadata.st_gid != owner_gid
        or stat.S_IMODE(root_metadata.st_mode) & 0o022
    ):
        _fail(
            "ADAPTER_CONFIGURATION_CONFLICT",
            "static distribution root is not a private user-owned directory",
        )
    index = _read_regular(root / "index.html", "static distribution index")
    try:
        index_text = index.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        _fail(
            "ADAPTER_CONFIGURATION_CONFLICT",
            f"static distribution index is not UTF-8: {exc}",
        )
    references = sorted(set(STATIC_REFERENCE_PATTERN.findall(index_text)))
    prefix = mount_path.rstrip("/") + "/"
    checked: list[str] = []
    for reference in references:
        parsed = urlsplit(reference)
        if (
            parsed.scheme
            or parsed.netloc
            or parsed.query
            or parsed.fragment
        ):
            _fail(
                "ADAPTER_CONFIGURATION_CONFLICT",
                "static index contains a non-local asset reference",
            )
        decoded_path = unquote(parsed.path)
        if not decoded_path.startswith(prefix):
            _fail(
                "ADAPTER_CONFIGURATION_CONFLICT",
                "static index reference is outside the configured mount",
            )
        relative_value = decoded_path[len(prefix):]
        if relative_value.startswith("chat/"):
            relative_value = relative_value[len("chat/"):]
        if not relative_value or relative_value == "chat":
            _fail(
                "ADAPTER_CONFIGURATION_CONFLICT",
                "static index contains an empty asset reference",
            )
        relative = PurePosixPath(relative_value)
        if (
            relative.is_absolute()
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            _fail(
                "ADAPTER_CONFIGURATION_CONFLICT",
                "static index contains traversal in an asset reference",
            )
        candidate = root.joinpath(*relative.parts)
        _read_regular(candidate, f"static asset {reference}")
        try:
            candidate_metadata = candidate.lstat()
        except OSError as exc:
            _fail(
                "ADAPTER_CONFIGURATION_CONFLICT",
                f"static asset cannot be inspected: {exc}",
            )
        if (
            candidate_metadata.st_uid != owner_uid
            or candidate_metadata.st_gid != owner_gid
        ):
            _fail(
                "ADAPTER_CONFIGURATION_CONFLICT",
                f"static asset is not owned by the service user: {reference}",
            )
        checked.append(decoded_path)
    return checked


def _atomic_write(
    path: Path,
    data: bytes,
    *,
    mode: int = 0o600,
    exclusive: bool = False,
    conflict_reason: str = "ADAPTER_MANIFEST_INVALID",
) -> None:
    _ensure_private_directory(path.parent, f"{path.name} parent")
    _reject_symlink_components(path, path.name)
    if exclusive:
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            descriptor = os.open(path, flags, mode)
        except FileExistsError:
            _fail(conflict_reason, f"{path.name} already exists")
        except OSError as exc:
            _fail(
                "ADAPTER_MANIFEST_INVALID",
                f"{path.name} create failed: {exc}",
            )
        try:
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            try:
                path.unlink()
            except OSError:
                pass
            _fail(
                "ADAPTER_MANIFEST_INVALID",
                f"{path.name} write failed: {exc}",
            )
        _fsync_directory(path.parent)
        _adopt_user_owned_file(path, path.name)
        return
    temporary_descriptor: int | None = None
    temporary_path: str | None = None
    try:
        temporary_descriptor, temporary_path = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        os.fchmod(temporary_descriptor, mode)
        with os.fdopen(
            temporary_descriptor, "wb", closefd=True
        ) as handle:
            temporary_descriptor = None
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
        _fsync_directory(path.parent)
        _adopt_user_owned_file(path, path.name)
    except OSError as exc:
        _fail(
            "ADAPTER_MANIFEST_INVALID",
            f"{path.name} atomic write failed: {exc}",
        )
    finally:
        if temporary_descriptor is not None:
            try:
                os.close(temporary_descriptor)
            except OSError:
                pass
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass


def _atomic_json(
    path: Path,
    value: Mapping[str, Any],
    *,
    exclusive: bool = False,
    conflict_reason: str = "ADAPTER_MANIFEST_INVALID",
) -> None:
    _atomic_write(
        path,
        (canonical_json(value) + "\n").encode("utf-8"),
        exclusive=exclusive,
        conflict_reason=conflict_reason,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(65_536)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def _file_identity(
    path: Path | str, label: str, reason_code: str
) -> dict[str, str]:
    candidate = _absolute(path)
    _reject_symlink_components(candidate, label)
    try:
        metadata = candidate.lstat()
    except OSError as exc:
        _fail(reason_code, f"{label} cannot be inspected: {exc}")
    if not stat.S_ISREG(metadata.st_mode):
        _fail(reason_code, f"{label} must be a regular file")
    try:
        return {"path": str(candidate), "sha256": _sha256(candidate)}
    except OSError as exc:
        _fail(reason_code, f"{label} cannot be hashed: {exc}")


def _interpreter_identity(path: Path = PYTHON) -> dict[str, Any]:
    candidate = path.resolve(strict=True)
    metadata = candidate.lstat()
    if not stat.S_ISREG(metadata.st_mode) or not os.access(
        candidate, os.X_OK
    ):
        _fail(
            "SUPERVISOR_ENTRYPOINT_INVALID",
            "supervisor interpreter must be executable and regular",
        )
    return {
        "path": str(candidate),
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "mode": stat.S_IMODE(metadata.st_mode),
        "size": metadata.st_size,
        "mtime_ns": metadata.st_mtime_ns,
    }


def _safe_token(value: str, label: str) -> str:
    if (
        not value
        or len(value) > 4_096
        or SAFE_UNIT_TOKEN.fullmatch(value) is None
        or any(character.isspace() for character in value)
    ):
        _fail(
            "COMMAND_INJECTION_REJECTED",
            f"{label} is not a safe structured unit token",
        )
    return value


def _safe_systemd_path(value: str, label: str) -> str:
    """Validate a product path and quote spaces for systemd ExecStart."""
    if (
        not value
        or len(value) > 4_096
        or not value.startswith("/")
        or value != value.strip()
        or SYSTEMD_PATH_TOKEN.fullmatch(value) is None
    ):
        _fail(
            "COMMAND_INJECTION_REJECTED",
            f"{label} is not a safe structured systemd path token",
        )
    if any(character.isspace() for character in value):
        return '"' + value + '"'
    return value


def _safe_systemd_assignment_path(value: str, label: str) -> str:
    """Validate a path and quote systemd C-style assignment escaping for spaces."""
    if (
        not value
        or len(value) > 4_096
        or not value.startswith("/")
        or value != value.strip()
        or SYSTEMD_PATH_TOKEN.fullmatch(value) is None
    ):
        _fail(
            "COMMAND_INJECTION_REJECTED",
            f"{label} is not a safe structured systemd path token",
        )
    return value


def _configuration_identity(value: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(
        canonical_json(value).encode("utf-8")
    ).hexdigest()


def _same_process(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> bool:
    return all(
        first.get(name) == second.get(name)
        for name in (
            "pid",
            "process_start_identity",
            "pgid",
            "sid",
            "executable",
            "argv_sha256",
        )
    )


@dataclass(frozen=True)
class AdapterPaths:
    runtime_root: Path

    @property
    def root(self) -> Path:
        return self.runtime_root / "linux-systemd-user"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def status(self) -> Path:
        return self.root / "status.json"

    @property
    def transactions(self) -> Path:
        return self.root / "transactions"

    @property
    def configuration_lock(self) -> Path:
        return self.root / "configuration.lock"

    def transaction(self, transaction_id: str) -> Path:
        return self.transactions / f"{transaction_id}.json"

    def view(self) -> AdapterPathsView:
        return AdapterPathsView(self.manifest, self.status)


class NativeManager(Protocol):
    unit_path: Path

    def capability(self) -> dict[str, Any]: ...
    def status(self) -> dict[str, Any]: ...
    def verify_unit(self) -> dict[str, Any]: ...
    def daemon_reload(self) -> dict[str, Any]: ...
    def enable(self) -> dict[str, Any]: ...
    def disable(self) -> dict[str, Any]: ...
    def start(self) -> dict[str, Any]: ...
    def stop(self) -> dict[str, Any]: ...
    def restart(self) -> dict[str, Any]: ...


class SystemdUserManager:
    """Fixed systemd --user command surface; no operator command text."""

    def __init__(
        self,
        unit_path: Path = DEFAULT_UNIT_PATH,
        service_name: str = SERVICE_NAME,
    ) -> None:
        if service_name != SERVICE_NAME and TRIAL_SERVICE_NAME.fullmatch(service_name) is None:
            _fail(
                "SERVICE_NAME_COLLISION",
                "service name is not an approved System X identity",
            )
        self.unit_path = _absolute(unit_path)
        self.service_name = service_name

    @staticmethod
    def _environment() -> dict[str, str]:
        value = {
            "HOME": str(Path.home()),
            "USER": os.environ.get("USER", ""),
            "LOGNAME": os.environ.get("LOGNAME", ""),
            "PATH": "/usr/bin:/bin",
            "LC_ALL": "C",
            "SYSTEMD_PAGER": "cat",
            "PAGER": "cat",
            "XDG_RUNTIME_DIR": os.environ.get("XDG_RUNTIME_DIR", ""),
        }
        bus = os.environ.get("DBUS_SESSION_BUS_ADDRESS")
        if bus:
            value["DBUS_SESSION_BUS_ADDRESS"] = bus
        return value

    def _run(
        self,
        arguments: Sequence[str],
        *,
        timeout: float = 30.0,
        accepted_exit_statuses: Sequence[int] = (0,),
    ) -> dict[str, Any]:
        argv = [str(SYSTEMCTL), "--user", *arguments]
        completed = subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
            env=self._environment(),
        )
        stdout = completed.stdout.decode("utf-8", errors="strict")
        stderr = completed.stderr.decode("utf-8", errors="strict")
        result = {
            "argv": argv,
            "exit_status": completed.returncode,
            "stdout": stdout[:65_536],
            "stderr": stderr[:65_536],
        }
        if completed.returncode not in accepted_exit_statuses:
            _fail(
                "NATIVE_MANAGER_UNAVAILABLE",
                (
                    f"systemd user operation failed ({arguments[0]}): "
                    f"{stderr.strip() or stdout.strip()}"
                ),
                data={
                    "operation": arguments[0],
                    "exit_status": completed.returncode,
                },
            )
        return result

    def capability(self) -> dict[str, Any]:
        checks = {
            "linux_systemd_pid1": (
                Path("/proc/1/comm")
                .read_text(encoding="utf-8")
                .strip()
                == "systemd"
            ),
            "systemd_user_manager": True,
            "user_unit_registration": _directory_is_writable_or_creatable(
                Path.home() / ".config", "user unit registration directory"
            ),
            "manager_enable_disable": True,
            "manager_start_stop_status": True,
            "manager_main_pid_observation": True,
            "manager_restart_on_failure": True,
            "manager_owned_journal": Path("/usr/bin/journalctl").is_file(),
        }
        probe = self._run(
            ["is-system-running"],
            timeout=10.0,
            accepted_exit_statuses=(0, 1),
        )
        if probe["stdout"].strip() not in {"running", "degraded"}:
            checks["systemd_user_manager"] = False
        missing = [
            name
            for name in REQUIRED_HOST_CAPABILITIES
            if not checks.get(name, False)
        ]
        return {
            "available": not missing,
            "required": list(REQUIRED_HOST_CAPABILITIES),
            "missing": missing,
            "foreground_activation_supported": False,
            "registration_supported": True,
            "enable_disable_supported": True,
            "restart_supported": True,
            "unregister_supported": True,
            "automatic_activation_supported": True,
            "checks": checks,
            "manager_probe": {
                "exit_status": probe["exit_status"],
                "state": probe["stdout"].strip(),
            },
        }

    def status(self) -> dict[str, Any]:
        arguments = ["show", self.service_name]
        for name in MANAGER_PROPERTIES:
            arguments.extend(("-p", name))
        arguments.extend(("--all", "--no-pager"))
        result = self._run(arguments, timeout=10.0)
        properties = {}
        for line in result["stdout"].splitlines():
            if "=" in line:
                name, value = line.split("=", 1)
                properties[name] = value
        missing = sorted(set(MANAGER_PROPERTIES) - set(properties))
        if missing:
            _fail(
                "STALE_MANAGER_STATUS",
                f"manager status omitted properties: {missing}",
            )
        try:
            main_pid = int(properties["MainPID"])
            n_restarts = int(properties["NRestarts"] or "0")
        except ValueError:
            _fail(
                "STALE_MANAGER_STATUS",
                "manager numeric status is invalid",
            )
        load_state = properties["LoadState"]
        registered = load_state == "loaded"
        fragment = properties["FragmentPath"]
        if registered and _absolute(fragment) != self.unit_path:
            _fail(
                "FOREIGN_SERVICE_DEFINITION",
                "manager fragment path differs from adapter registration",
            )
        return {
            "registered": registered,
            "enabled": properties["UnitFileState"] == "enabled",
            "active": properties["ActiveState"] == "active",
            "active_state": properties["ActiveState"],
            "sub_state": properties["SubState"],
            "unit_file_state": properties["UnitFileState"],
            "fragment_path": fragment or None,
            "main_pid": main_pid or None,
            "result": properties["Result"] or None,
            "n_restarts": n_restarts,
            "invocation_id": properties["InvocationID"] or None,
            "active_enter_monotonic": (
                int(properties["ActiveEnterTimestampMonotonic"] or "0")
            ),
            "exec_main_pid": int(properties["ExecMainPID"] or "0")
            or None,
            "exec_main_code": int(properties["ExecMainCode"] or "0"),
            "exec_main_status": int(
                properties["ExecMainStatus"] or "0"
            ),
            "raw_properties": properties,
        }

    def verify_unit(self) -> dict[str, Any]:
        completed = subprocess.run(
            [
                str(SYSTEMD_ANALYZE),
                "--user",
                "verify",
                str(self.unit_path),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=20,
            env=self._environment(),
        )
        if completed.returncode != 0:
            _fail(
                "SERVICE_DEFINITION_INVALID",
                completed.stderr.decode(
                    "utf-8", errors="replace"
                )[:4_096],
            )
        return {
            "exit_status": completed.returncode,
            "stdout": completed.stdout.decode(
                "utf-8", errors="replace"
            )[:65_536],
            "stderr": completed.stderr.decode(
                "utf-8", errors="replace"
            )[:65_536],
        }

    def daemon_reload(self) -> dict[str, Any]:
        return self._run(["daemon-reload"])

    def enable(self) -> dict[str, Any]:
        return self._run(["enable", "--no-reload", self.service_name])

    def disable(self) -> dict[str, Any]:
        return self._run(["disable", self.service_name])

    def start(self) -> dict[str, Any]:
        return self._run(["start", self.service_name], timeout=60.0)

    def stop(self) -> dict[str, Any]:
        return self._run(["stop", self.service_name], timeout=120.0)

    def restart(self) -> dict[str, Any]:
        return self._run(["restart", self.service_name], timeout=120.0)


def render_unit(
    *,
    interpreter: str,
    supervisor_entrypoint: str,
    profile_path: str,
    state_path: str,
    supervisor_runtime_root: str,
    api_controller: str,
    branch_controller: str,
    api_controller_sha256: str,
    branch_controller_sha256: str,
    timeout_stop_seconds: int,
) -> str:
    arguments = (
        interpreter,
        supervisor_entrypoint,
        "run",
        "--profile",
        profile_path,
        "--state",
        state_path,
        "--runtime-root",
        supervisor_runtime_root,
        "--api-controller",
        api_controller,
        "--branch-controller",
        branch_controller,
        "--api-controller-sha256",
        api_controller_sha256,
        "--branch-controller-sha256",
        branch_controller_sha256,
    )
    safe_arguments = [
        _safe_systemd_path(str(interpreter), "ExecStart interpreter"),
        _safe_systemd_path(
            str(supervisor_entrypoint), "ExecStart supervisor entrypoint"
        ),
        _safe_token("run", "ExecStart operation"),
        _safe_token("--profile", "ExecStart profile flag"),
        _safe_systemd_path(str(profile_path), "ExecStart profile path"),
        _safe_token("--state", "ExecStart state flag"),
        _safe_systemd_path(str(state_path), "ExecStart state path"),
        _safe_token("--runtime-root", "ExecStart runtime flag"),
        _safe_systemd_path(
            str(supervisor_runtime_root), "ExecStart runtime path"
        ),
        _safe_token("--api-controller", "ExecStart API flag"),
        _safe_systemd_path(str(api_controller), "ExecStart API path"),
        _safe_token("--branch-controller", "ExecStart branch flag"),
        _safe_systemd_path(
            str(branch_controller), "ExecStart branch path"
        ),
        _safe_token("--api-controller-sha256", "ExecStart API digest flag"),
        _safe_token(
            api_controller_sha256, "ExecStart API controller digest"
        ),
        _safe_token(
            "--branch-controller-sha256", "ExecStart branch digest flag"
        ),
        _safe_token(
            branch_controller_sha256, "ExecStart branch controller digest"
        ),
    ]
    working_directory = _safe_systemd_assignment_path(
        "/", "WorkingDirectory"
    )
    if (
        type(timeout_stop_seconds) is not int
        or not 10 <= timeout_stop_seconds <= 3_600
    ):
        _fail(
            "SERVICE_DEFINITION_INVALID",
            "TimeoutStopSec must be an integer in 10..3600",
        )
    value = (
        "[Unit]\n"
        "Description=System X automatic activation supervisor\n"
        "StartLimitIntervalSec=30\n"
        "StartLimitBurst=3\n"
        "\n"
        "[Service]\n"
        "Type=simple\n"
        f"WorkingDirectory={working_directory}\n"
        f"ExecStart={' '.join(safe_arguments)}\n"
        "Restart=on-failure\n"
        "RestartSec=20s\n"
        f"TimeoutStopSec={timeout_stop_seconds}s\n"
        "KillMode=control-group\n"
        "StandardOutput=journal\n"
        "StandardError=journal\n"
        "SyslogIdentifier=system-x\n"
        "UMask=0077\n"
        "\n"
        "[Install]\n"
        "WantedBy=default.target\n"
    )
    forbidden = (
        "uvicorn",
        "llama-server",
        "model-load",
        "EnvironmentFile=",
        "ExecStart=/bin/sh",
        "ExecStart=/usr/bin/env",
    )
    if any(item in value for item in forbidden):
        _fail(
            "SERVICE_DEFINITION_INVALID",
            "unit contains a prohibited direct runtime or shell boundary",
        )
    if re.search(r"(?:sk-|sxk_)[A-Za-z0-9_-]{20,}", value):
        _fail("RAW_CREDENTIAL_REJECTED", "unit contains a key pattern")
    return value


def _decode_systemd_assignment_path(value: str, label: str) -> str:
    """Decode the quoted path emitted by this adapter."""
    raw = value
    if len(raw) >= 2 and raw[0] == raw[-1] == '"':
        raw = raw[1:-1]
    decoded = raw.replace(r"\s", " ").replace(r"\x20", " ")
    if (
        not decoded
        or not decoded.startswith("/")
        or decoded != decoded.strip()
        or SYSTEMD_PATH_TOKEN.fullmatch(decoded) is None
        or "\\" in decoded
    ):
        _fail(
            "SERVICE_NAME_COLLISION",
            f"{label} is not a valid escaped System X path",
        )
    return decoded


def _existing_product_unit_identity(unit_path: Path) -> dict[str, str]:
    """Identify an existing System X unit before replacing its root binding."""

    if os.path.islink(unit_path):
        _fail("SERVICE_NAME_COLLISION", "existing service definition is a symlink")
    try:
        metadata = unit_path.lstat()
    except OSError as exc:
        _fail("SERVICE_NAME_COLLISION", f"existing service definition cannot be inspected: {exc}")
    owner_uid, owner_gid = _repository_owner()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != owner_uid
        or metadata.st_gid != owner_gid
        or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        _fail("SERVICE_NAME_COLLISION", "existing service definition is not private and user-owned")
    data = _read_regular(unit_path, "existing service definition")
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        _fail("SERVICE_NAME_COLLISION", f"existing service definition is not UTF-8: {exc}")
    if any(token in text for token in ("EnvironmentFile=", "ExecStart=/bin/sh", "ExecStart=/usr/bin/env")):
        _fail("SERVICE_NAME_COLLISION", "existing service definition is not a native System X unit")
    fields: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key in {"Description", "WorkingDirectory", "ExecStart"}:
            if key in fields:
                _fail("SERVICE_NAME_COLLISION", "existing service definition has duplicate identity fields")
            fields[key] = value
    if fields.get("Description") != "System X automatic activation supervisor":
        _fail("SERVICE_NAME_COLLISION", "existing service definition is not owned by System X")
    working_directory = fields.get("WorkingDirectory")
    exec_start = fields.get("ExecStart")
    if not working_directory or not exec_start:
        _fail("SERVICE_NAME_COLLISION", "existing service definition lacks the System X identity")
    try:
        arguments = shlex.split(exec_start, posix=True)
    except ValueError as exc:
        _fail("SERVICE_NAME_COLLISION", f"existing service definition has invalid argv: {exc}")
    if len(arguments) != 17 or arguments[0] != str(PYTHON) or arguments[2] != "run":
        _fail("SERVICE_NAME_COLLISION", "existing service definition has an unexpected supervisor argv")
    working_root = Path(_decode_systemd_assignment_path(working_directory, "WorkingDirectory"))
    if working_root != Path("/"):
        _fail("SERVICE_NAME_COLLISION", "existing service definition has an unexpected working directory")
    runtime_root = Path(arguments[8])
    if runtime_root.name != "service_control" or runtime_root.parent.name != "RUNTIME":
        _fail("SERVICE_NAME_COLLISION", "existing service definition has an unexpected runtime root")
    model_root = runtime_root.parent.parent
    if model_root.name != "model-api-gguf" or not model_root.is_absolute():
        _fail("SERVICE_NAME_COLLISION", "existing service definition has an unexpected working directory")
    branch_root = model_root.parent
    expected_paths = {
        1: model_root / "service_control/supervisor.py",
        4: model_root / "RUNTIME/service_control/operating-profile.json",
        6: model_root / "RUNTIME/service_control/desired-state.json",
        8: model_root / "RUNTIME/service_control",
        10: model_root / "api_service_controller/controller.py",
        12: model_root / "branch_controller/controller.py",
    }
    for index, expected_path in expected_paths.items():
        if arguments[index] != str(expected_path):
            _fail("SERVICE_NAME_COLLISION", "existing service definition is not a coherent System X registration")
    if arguments[13] != "--api-controller-sha256" or arguments[15] != "--branch-controller-sha256":
        _fail("SERVICE_NAME_COLLISION", "existing service definition lacks locked controller identities")
    if arguments[14] != API_CONTROLLER_SHA256 or arguments[16] != BRANCH_CONTROLLER_SHA256:
        _fail("SERVICE_NAME_COLLISION", "existing service definition has unexpected controller identities")
    return {
        "branch_root": str(branch_root),
        "unit_sha256": hashlib.sha256(data).hexdigest(),
    }


class LinuxSystemdUserServiceAdapter:
    adapter_identity = ADAPTER_IDENTITY
    adapter_version = ADAPTER_VERSION

    def __init__(
        self,
        adapter_runtime_root: Path | str,
        *,
        manager: NativeManager | None = None,
        unit_path: Path | None = None,
        service_name: str = SERVICE_NAME,
    ) -> None:
        self.paths = AdapterPaths(_absolute(adapter_runtime_root))
        if manager is not None and unit_path is not None:
            _fail(
                "ADAPTER_CONFIGURATION_CONFLICT",
                "manager and unit_path injection are mutually exclusive",
            )
        if manager is None:
            selected_unit_path = unit_path or (
                Path.home() / ".config/systemd/user" / service_name
            )
            manager = SystemdUserManager(
                selected_unit_path, service_name=service_name
            )
        self.manager = manager
        self.service_name = getattr(self.manager, "service_name", SERVICE_NAME)
        self.unit_path = _absolute(self.manager.unit_path)

    def _prepare_runtime(self) -> None:
        _ensure_private_directory(
            self.unit_path.parent,
            "systemd user unit directory",
        )
        _adopt_user_owned_directory(
            self.unit_path.parent,
            "systemd user unit directory",
        )
        for path, label in (
            (self.paths.root, "systemd adapter root"),
            (self.paths.transactions, "systemd adapter transactions"),
        ):
            _ensure_private_directory(path, label)

    def _capability_value(self) -> dict[str, Any]:
        value = self.manager.capability()
        if not isinstance(value, dict):
            _fail(
                "HOST_CAPABILITY_MISSING",
                "manager capability response is invalid",
            )
        return value

    def _render_configuration(
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
        bytes,
    ]:
        capability = self._capability_value()
        if not capability.get("available"):
            _fail(
                "HOST_CAPABILITY_MISSING",
                "systemd user adapter capability is unavailable",
                data=capability,
            )
        supervisor = _file_identity(
            supervisor_entrypoint,
            "supervisor entrypoint",
            "SUPERVISOR_ENTRYPOINT_INVALID",
        )
        api_controller = _file_identity(
            DEFAULT_API_CONTROLLER,
            "API controller",
            "SUPERVISOR_IDENTITY_MISMATCH",
        )
        branch_controller = _file_identity(
            DEFAULT_BRANCH_CONTROLLER,
            "branch controller",
            "SUPERVISOR_IDENTITY_MISMATCH",
        )
        if api_controller["sha256"] != API_CONTROLLER_SHA256:
            _fail(
                "SUPERVISOR_IDENTITY_MISMATCH",
                "API controller source identity changed",
            )
        if branch_controller["sha256"] != BRANCH_CONTROLLER_SHA256:
            _fail(
                "SUPERVISOR_IDENTITY_MISMATCH",
                "branch controller source identity changed",
            )
        profile_file = _absolute(profile_path)
        state_file = _absolute(state_path)
        runtime_root = _absolute(supervisor_runtime_root)
        try:
            profile = load_operating_profile(profile_file)
            desired = load_desired_state(state_file, profile.identity)
        except ServiceControlError as exc:
            reason = (
                "PROFILE_IDENTITY_MISMATCH"
                if "profile" in exc.message.lower()
                else "DESIRED_STATE_PROFILE_MISMATCH"
            )
            _fail(reason, exc.message)
        _reject_symlink_components(runtime_root, "supervisor runtime root")
        if not runtime_root.parent.is_dir():
            _fail(
                "ADAPTER_CONFIGURATION_CONFLICT",
                "supervisor runtime parent must exist",
            )
        timeout_stop = int(
            min(
                3_600,
                max(
                    10,
                    math.ceil(
                        profile.graceful_shutdown_timeout_seconds
                    )
                    + 15,
                ),
            )
        )
        interpreter = _interpreter_identity()
        unit = render_unit(
            interpreter=interpreter["path"],
            supervisor_entrypoint=supervisor["path"],
            profile_path=str(profile_file),
            state_path=str(state_file),
            supervisor_runtime_root=str(runtime_root),
            api_controller=api_controller["path"],
            branch_controller=branch_controller["path"],
            api_controller_sha256=api_controller["sha256"],
            branch_controller_sha256=branch_controller["sha256"],
            timeout_stop_seconds=timeout_stop,
        ).encode("utf-8")
        unit_sha = hashlib.sha256(unit).hexdigest()
        reference = {
            "profile_path": str(profile_file),
            "state_path": str(state_file),
            "supervisor_runtime_root": str(runtime_root),
            "profile_identity": profile.identity,
        }
        native_service = {
            "manager": "systemd",
            "manager_scope": "user",
            "service_name": self.service_name,
            "registration_path": str(self.unit_path),
            "service_definition_sha256": unit_sha,
            "interpreter_identity": interpreter,
            "api_controller": api_controller,
            "branch_controller": branch_controller,
            "timeout_stop_seconds": timeout_stop,
        }
        identity_value = {
            "schema_version": (
                "system-x.platform-service-adapter-configuration.v1"
            ),
            "adapter_identity": ADAPTER_IDENTITY,
            "adapter_version": ADAPTER_VERSION,
            "supported_platform_family": PLATFORM_FAMILY,
            "activation_method": ACTIVATION_METHOD,
            "supervisor_entrypoint_sha256": supervisor["sha256"],
            "configuration_reference": reference,
            "native_service": native_service,
        }
        return (
            {
                "supervisor_entrypoint": supervisor,
                "configuration_reference": reference,
                "native_service": native_service,
                "configuration_identity": _configuration_identity(
                    identity_value
                ),
                "required_host_capability_result": capability,
            },
            profile,
            desired,
            unit,
        )

    def _manifest_exists(self) -> bool:
        return os.path.lexists(self.paths.manifest)

    def _load_manifest(
        self,
        *,
        allow_stale_configuration_for_inactive_removal: bool = False,
        allow_missing_service_definition: bool = False,
    ) -> tuple[
        dict[str, Any],
        OperatingProfile,
        DesiredState,
        bytes,
    ]:
        value = _read_json(self.paths.manifest, "adapter manifest")
        if frozenset(value) != _MANIFEST_FIELDS:
            _fail(
                "ADAPTER_MANIFEST_INVALID",
                "adapter manifest fields are incomplete or unknown",
            )
        expected_constants = (
            (value.get("schema_version"), MANIFEST_SCHEMA),
            (value.get("adapter_identity"), ADAPTER_IDENTITY),
            (value.get("adapter_version"), ADAPTER_VERSION),
            (value.get("supported_platform_family"), PLATFORM_FAMILY),
            (value.get("activation_method"), ACTIVATION_METHOD),
            (value.get("automatic_activation_supported"), True),
            (
                value.get("required_host_capabilities"),
                list(REQUIRED_HOST_CAPABILITIES),
            ),
            (value.get("registered"), True),
        )
        if any(observed != expected for observed, expected in expected_constants):
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
        ):
            _fail(
                "ADAPTER_MANIFEST_INVALID",
                "adapter manifest dynamic fields are invalid",
            )
        if (
            not isinstance(value.get("supervisor_entrypoint"), dict)
            or frozenset(value["supervisor_entrypoint"])
            != _FILE_IDENTITY_FIELDS
            or not isinstance(value.get("configuration_reference"), dict)
            or frozenset(value["configuration_reference"])
            != _REFERENCE_FIELDS
            or not isinstance(value.get("native_service"), dict)
            or frozenset(value["native_service"])
            != _NATIVE_SERVICE_FIELDS
        ):
            _fail(
                "ADAPTER_MANIFEST_INVALID",
                "adapter immutable configuration shape is invalid",
            )
        reference = value["configuration_reference"]
        config, profile, desired, unit = self._render_configuration(
            profile_path=reference["profile_path"],
            state_path=reference["state_path"],
            supervisor_runtime_root=reference["supervisor_runtime_root"],
            supervisor_entrypoint=value["supervisor_entrypoint"]["path"],
        )
        if (
            config["supervisor_entrypoint"]
            != value["supervisor_entrypoint"]
            or config["native_service"] != value["native_service"]
            or config["configuration_identity"]
            != value["configuration_identity"]
        ) and not allow_stale_configuration_for_inactive_removal:
            _fail(
                "ADAPTER_CONFIGURATION_CONFLICT",
                "registered adapter configuration identity changed",
            )
        _reject_symlink_components(
            self.unit_path, "native service definition"
        )
        try:
            unit_data = _read_regular(
                self.unit_path, "native service definition"
            )
        except AdapterError as exc:
            if not (
                allow_missing_service_definition
                and exc.reason_code == "ADAPTER_NOT_REGISTERED"
            ):
                raise
            unit_data = unit
        if (
            (
                hashlib.sha256(unit_data).hexdigest()
                != value["native_service"]["service_definition_sha256"]
                and not allow_stale_configuration_for_inactive_removal
            )
            or (
                not allow_stale_configuration_for_inactive_removal
                and unit_data != unit
            )
        ):
            _fail(
                "FOREIGN_SERVICE_DEFINITION",
                "native service definition identity changed",
            )
        return value, profile, desired, unit

    def _update_manifest(
        self, manifest: Mapping[str, Any], **updates: Any
    ) -> dict[str, Any]:
        value = dict(manifest)
        identity = value["configuration_identity"]
        value.update(updates)
        if value.get("configuration_identity") != identity:
            _fail(
                "ADAPTER_CONFIGURATION_CONFLICT",
                "dynamic update changed configuration identity",
            )
        value["manifest_generation"] = int(
            manifest["manifest_generation"]
        ) + 1
        value["updated_utc"] = utc_now()
        _atomic_json(self.paths.manifest, value)
        return value

    def _record_transaction(
        self,
        operation: str,
        *,
        before: Mapping[str, Any] | None,
        after: Mapping[str, Any] | None,
        result: Mapping[str, Any],
    ) -> str:
        transaction_id = (
            "pa-"
            + utc_now()
            .replace("-", "")
            .replace(":", "")
            .replace(".", "")
            .replace("Z", "")
            + "-"
            + secrets.token_hex(6)
        )
        _atomic_json(
            self.paths.transaction(transaction_id),
            {
                "schema_version": (
                    "system-x.platform-service-adapter-transaction.v1"
                ),
                "adapter_identity": ADAPTER_IDENTITY,
                "adapter_transaction_id": transaction_id,
                "operation": operation,
                "manager_before": dict(before or {}),
                "manager_after": dict(after or {}),
                "result": dict(result),
                "timestamp_utc": utc_now(),
            },
            exclusive=True,
        )
        return transaction_id

    @staticmethod
    def _endpoint_free(host: str, port: int) -> bool:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.settimeout(0.3)
        try:
            connected = probe.connect_ex((host, port)) == 0
        finally:
            probe.close()
        if connected:
            return False
        binder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            binder.bind((host, port))
        except OSError:
            return False
        finally:
            binder.close()
        return True

    def _preflight_endpoints(self, profile: OperatingProfile) -> None:
        endpoints = (
            profile.public_endpoint,
            profile.private_router_endpoint,
        )
        conflicts = [
            {"host": endpoint.host, "port": endpoint.port}
            for endpoint in endpoints
            if not self._endpoint_free(endpoint.host, endpoint.port)
        ]
        if conflicts:
            _fail(
                "ENDPOINT_CONFLICT",
                "one or more retained endpoints are occupied",
                exit_code=3,
                data={"conflicts": conflicts},
            )

    def _wait_endpoints_free(
        self,
        profile: OperatingProfile,
        *,
        timeout_seconds: float,
    ) -> float:
        endpoints = (
            profile.public_endpoint,
            profile.private_router_endpoint,
        )
        started = time.monotonic()
        deadline = started + timeout_seconds
        while time.monotonic() < deadline:
            if all(
                self._endpoint_free(endpoint.host, endpoint.port)
                for endpoint in endpoints
            ):
                return time.monotonic() - started
            time.sleep(0.05)
        _fail(
            "ENDPOINT_CONFLICT",
            "controller-compatible endpoint release timed out",
            exit_code=4,
            data={
                "endpoints": [
                    {"host": endpoint.host, "port": endpoint.port}
                    for endpoint in endpoints
                ]
            },
        )

    def _configuration_transition_is_valid(
        self,
        manifest: Mapping[str, Any],
        previous_profile_identity: Any,
    ) -> bool:
        expected_profile_identity = manifest["configuration_reference"][
            "profile_identity"
        ]
        if not isinstance(previous_profile_identity, str):
            return False
        if not self.paths.transactions.is_dir():
            return False
        for candidate in sorted(
            self.paths.transactions.glob("*.json"),
            key=lambda item: item.name,
            reverse=True,
        ):
            transaction = _read_json(candidate, "adapter transaction")
            if transaction.get("operation") != "configure-static-ui":
                continue
            result = transaction.get("result")
            if not isinstance(result, dict):
                continue
            data = result.get("data")
            if not isinstance(data, dict):
                continue
            if (
                result.get("profile_identity") == expected_profile_identity
                and data.get("previous_profile_identity")
                == previous_profile_identity
                and data.get("new_profile_identity")
                == expected_profile_identity
            ):
                return True
        return False

    def _supervisor_evidence(
        self,
        manifest: Mapping[str, Any],
        *,
        require_status: bool = True,
    ) -> tuple[dict[str, Any], dict[str, Any] | None] | None:
        runtime = Path(
            manifest["configuration_reference"]["supervisor_runtime_root"]
        )
        paths = SupervisorPaths(runtime)
        lock_present = os.path.lexists(paths.active_lock)
        pid_present = os.path.lexists(paths.active_pid)
        if not lock_present and not pid_present:
            return None
        if lock_present != pid_present:
            _fail(
                "MANAGER_SUPERVISOR_IDENTITY_MISMATCH",
                "supervisor active record presence is partial",
                exit_code=3,
            )
        lock = _read_json(paths.active_lock, "supervisor lock")
        pid_record = _read_json(paths.active_pid, "supervisor PID")
        profile_identity = manifest["configuration_reference"][
            "profile_identity"
        ]
        active_profile_identity = profile_identity
        recorded_profile_identity = lock.get("profile_identity")
        if (
            recorded_profile_identity != profile_identity
            or pid_record.get("profile_identity") != profile_identity
        ):
            if (
                pid_record.get("profile_identity")
                != recorded_profile_identity
                or not self._configuration_transition_is_valid(
                    manifest, recorded_profile_identity
                )
            ):
                _fail(
                    "MANAGER_SUPERVISOR_IDENTITY_MISMATCH",
                    "supervisor active record identity is invalid",
                    exit_code=3,
                )
            active_profile_identity = recorded_profile_identity
        if (
            lock.get("schema_version") != SUPERVISOR_LOCK_SCHEMA
            or pid_record.get("schema_version") != SUPERVISOR_PID_SCHEMA
            or lock.get("supervisor_transaction_id")
            != pid_record.get("supervisor_transaction_id")
            or lock.get("profile_identity") != active_profile_identity
            or pid_record.get("profile_identity") != active_profile_identity
        ):
            _fail(
                "MANAGER_SUPERVISOR_IDENTITY_MISMATCH",
                "supervisor active record identity is invalid",
                exit_code=3,
            )
        try:
            observed = process_snapshot(int(pid_record.get("pid", 0)))
        except SupervisorError as exc:
            _fail(
                "PID_REUSE",
                f"supervisor process identity unavailable: {exc.message}",
                exit_code=3,
            )
        if not _same_process(pid_record, observed):
            _fail(
                "PID_REUSE",
                "supervisor PID record no longer matches the process",
                exit_code=3,
            )
        status = None
        if os.path.lexists(paths.status_record):
            status = _read_json(paths.status_record, "supervisor status")
            if (
                status.get("schema_version") != SUPERVISOR_STATUS_SCHEMA
                or status.get("profile_identity") != active_profile_identity
                or status.get("supervisor_transaction_id")
                != pid_record.get("supervisor_transaction_id")
                or status.get("supervisor_identity") != observed
            ):
                _fail(
                    "MANAGER_SUPERVISOR_IDENTITY_MISMATCH",
                    "supervisor status identity is invalid",
                    exit_code=3,
                )
        elif require_status:
            _fail(
                "MANAGER_SUPERVISOR_IDENTITY_MISMATCH",
                "live supervisor lacks a correlated status record",
                exit_code=3,
            )
        return observed, status

    def _correlate(
        self,
        manifest: Mapping[str, Any],
        *,
        require_status: bool = True,
    ) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any]]:
        manager = self.manager.status()
        if not manager.get("registered"):
            _fail(
                "NATIVE_MANAGER_UNAVAILABLE",
                "native manager no longer reports the service loaded",
            )
        if bool(manager.get("enabled")) != bool(manifest["enabled"]):
            _fail(
                "MANAGER_STATE_MISMATCH",
                "manifest and native-manager enabled states differ",
                exit_code=3,
            )
        supervisor = self._supervisor_evidence(
            manifest, require_status=require_status
        )
        if manager.get("active"):
            if supervisor is None:
                _fail(
                    "MANAGER_SUPERVISOR_IDENTITY_MISMATCH",
                    "manager is active without supervisor evidence",
                    exit_code=3,
                )
            identity, status = supervisor
            if manager.get("main_pid") != identity.get("pid"):
                _fail(
                    "MANAGER_SUPERVISOR_IDENTITY_MISMATCH",
                    "manager MainPID differs from supervisor PID",
                    exit_code=3,
                )
            return identity, status, manager
        if supervisor is not None:
            _fail(
                "MANAGER_SUPERVISOR_IDENTITY_MISMATCH",
                "supervisor evidence exists while manager is inactive",
                exit_code=3,
            )
        return {}, None, manager

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
                "observed_api_service",
                "observed_private_router",
                "observed_model_child",
                "warm_model_identity",
            )
        }

    def _status_value(
        self,
        manifest: Mapping[str, Any],
        profile: OperatingProfile,
        desired: DesiredState,
        identity: Mapping[str, Any],
        supervisor_status: Mapping[str, Any] | None,
        manager: Mapping[str, Any],
    ) -> dict[str, Any]:
        paths = SupervisorPaths(
            Path(
                manifest["configuration_reference"][
                    "supervisor_runtime_root"
                ]
            )
        )
        value = {
            "schema_version": STATUS_SCHEMA,
            "adapter_identity": ADAPTER_IDENTITY,
            "adapter_version": ADAPTER_VERSION,
            "configuration_identity": manifest["configuration_identity"],
            "registered": True,
            "enabled": bool(manager["enabled"]),
            "active": bool(manager["active"]),
            "automatic_activation_supported": True,
            "required_host_capability_result": self._capability_value(),
            "profile_identity": profile.identity,
            "desired_state": desired.desired_state,
            "desired_state_generation": desired.generation,
            "supervisor_status_summary": self._supervisor_summary(
                supervisor_status
            ),
            "supervisor_process_identity": dict(identity) or None,
            "last_activation_result": manifest["last_activation_result"],
            "last_failure_reason": manifest["last_failure_reason"],
            "reconciliation_reason": None,
            "resolved_paths": {
                "manifest": str(self.paths.manifest),
                "status": str(self.paths.status),
                "active_pid": str(paths.active_pid),
                "active_lock": str(paths.active_lock),
            },
            "observed_utc": utc_now(),
            "configured_public_base_url": (
                f"http://{profile.public_endpoint.host}:"
                f"{profile.public_endpoint.port}"
            ),
            "public_host": profile.public_endpoint.host,
            "public_port": profile.public_endpoint.port,
            "private_endpoint_configured": True,
            "default_model_alias": profile.default_model_alias,
            "manager_registered": bool(manager["registered"]),
            "manager_enabled": bool(manager["enabled"]),
            "manager_active": bool(manager["active"]),
            "manager_main_pid": manager["main_pid"],
            "manager_last_result": manager["result"],
            "system_x_readiness_state": (
                supervisor_status.get("model_service_state")
                if supervisor_status is not None
                else "STOPPED"
            ),
            "system_x_inference_ready": bool(
                supervisor_status is not None
                and supervisor_status.get("inference_ready") is True
            ),
        }
        _atomic_json(self.paths.status, value)
        return value

    def _result(
        self,
        operation: str,
        manifest: Mapping[str, Any] | None,
        profile: OperatingProfile | None,
        desired: DesiredState | None,
        manager: Mapping[str, Any] | None,
        *,
        message: str,
        data: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return result_envelope(
            operation,
            ok=True,
            reason_code="OK",
            message=message,
            adapter_identity=ADAPTER_IDENTITY,
            adapter_version=ADAPTER_VERSION,
            automatic_activation_supported=True,
            configuration_identity=(
                manifest.get("configuration_identity")
                if manifest is not None
                else None
            ),
            registered=manifest is not None,
            enabled=bool(manager and manager.get("enabled")),
            active=bool(manager and manager.get("active")),
            profile_identity=(
                profile.identity if profile is not None else None
            ),
            desired_state=(
                desired.desired_state if desired is not None else None
            ),
            desired_state_generation=(
                desired.generation if desired is not None else None
            ),
            paths=self.paths.view(),
            data=data,
        )

    def identify(self) -> dict[str, Any]:
        return self._result(
            "identify",
            None,
            None,
            None,
            None,
            message="Linux systemd user-service adapter identified",
            data={
                "supported_platform_family": PLATFORM_FAMILY,
                "activation_method": ACTIVATION_METHOD,
                "operation_set": list(LINUX_OPERATIONS),
                "service_name": self.service_name,
            },
        )

    def capability(
        self, requested: Sequence[str] | None = None
    ) -> dict[str, Any]:
        if requested:
            unknown = sorted(
                set(requested) - set(REQUIRED_HOST_CAPABILITIES)
            )
            if unknown:
                _fail(
                    "HOST_CAPABILITY_MISSING",
                    f"unknown host capability requested: {unknown}",
                    data={"unknown": unknown},
                )
        value = self._capability_value()
        if not value.get("available"):
            _fail(
                "HOST_CAPABILITY_MISSING",
                "required systemd user capability is missing",
                data=value,
            )
        return self._result(
            "capability",
            None,
            None,
            None,
            None,
            message="systemd user capabilities validated",
            data=value,
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
            manifest, profile, desired, _unit = self._load_manifest()
            identity, supervisor_status, manager = self._correlate(
                manifest
            )
            status = self._status_value(
                manifest,
                profile,
                desired,
                identity,
                supervisor_status,
                manager,
            )
            return self._result(
                "validate",
                manifest,
                profile,
                desired,
                manager,
                message="registered systemd user adapter validates",
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
        config, profile, desired, unit = self._render_configuration(
            profile_path=profile_path,
            state_path=state_path,
            supervisor_runtime_root=supervisor_runtime_root,
            supervisor_entrypoint=supervisor_entrypoint,
        )
        return self._result(
            "validate",
            None,
            profile,
            desired,
            None,
            message="systemd user adapter configuration validates",
            data={
                **config,
                "service_definition_sha256": hashlib.sha256(
                    unit
                ).hexdigest(),
                "mutated": False,
            },
        )

    def _reconcile_registered(
        self,
        *,
        profile_path: Path | str,
        state_path: Path | str,
        supervisor_runtime_root: Path | str,
        supervisor_entrypoint: Path | str,
    ) -> dict[str, Any]:
        old_manifest, _old_profile, _old_desired, _old_unit = (
            self._load_manifest(
                allow_stale_configuration_for_inactive_removal=True,
                allow_missing_service_definition=True,
            )
        )
        config, profile, desired, unit = self._render_configuration(
            profile_path=profile_path,
            state_path=state_path,
            supervisor_runtime_root=supervisor_runtime_root,
            supervisor_entrypoint=supervisor_entrypoint,
        )
        if desired.desired_state not in {"STOPPED", "RUNNING"}:
            _fail(
                "DESIRED_STATE_PROFILE_MISMATCH",
                "registration reconciliation requires a known desired state",
            )
        old_reference = old_manifest["configuration_reference"]
        if old_reference != config["configuration_reference"]:
            _fail(
                "ADAPTER_CONFIGURATION_CONFLICT",
                "registered adapter references a different production configuration",
            )
        old_native = old_manifest["native_service"]
        if (
            old_native.get("service_name") != self.service_name
            or old_native.get("registration_path") != str(self.unit_path)
        ):
            _fail(
                "SERVICE_NAME_COLLISION",
                "registered adapter ownership does not match the selected service",
            )
        if not self.unit_path.exists():
            before = self.manager.status()
            manager_stop_result = None
            if before.get("active") or before.get("active_state") in {
                "activating",
                "deactivating",
                "reloading",
                "auto-restart",
            }:
                manager_stop_result = self.manager.stop()
                before = self.manager.status()
            if before.get("active") or before.get("active_state") in {
                "activating",
                "deactivating",
                "reloading",
                "auto-restart",
            }:
                _fail(
                    "MANAGER_STATE_MISMATCH",
                    "owned service remained active while recovering its missing definition",
                    data={"manager_status": before},
                )
            if desired.desired_state == "RUNNING":
                try:
                    desired = set_desired_state(
                        profile,
                        "STOPPED",
                        config["configuration_reference"]["state_path"],
                        expected_generation=desired.generation,
                    )
                except ServiceControlError as exc:
                    _fail("DESIRED_STATE_PROFILE_MISMATCH", exc.message)
            self._prepare_runtime()
            try:
                _atomic_write(
                    self.unit_path,
                    unit,
                    exclusive=True,
                    conflict_reason="SERVICE_NAME_COLLISION",
                )
                verify = self.manager.verify_unit()
                reload_result = self.manager.daemon_reload()
                after = self.manager.status()
                if (
                    not after.get("registered")
                    or after.get("active")
                    or after.get("fragment_path") != str(self.unit_path)
                ):
                    _fail(
                        "MANAGER_STATE_MISMATCH",
                        "missing service definition recovery did not produce an inactive registered unit",
                        data={"manager_before": before, "manager_after": after},
                    )
                timestamp = utc_now()
                manifest = dict(old_manifest)
                manifest.update(
                    {
                        "supervisor_entrypoint": config["supervisor_entrypoint"],
                        "configuration_reference": config["configuration_reference"],
                        "configuration_identity": config["configuration_identity"],
                        "registered": True,
                        "enabled": bool(after.get("enabled")),
                        "active": False,
                        "manifest_generation": int(old_manifest["manifest_generation"]) + 1,
                        "updated_utc": timestamp,
                        "last_activation_result": None,
                        "last_failure_reason": None,
                        "native_service": config["native_service"],
                    }
                )
                _atomic_json(self.paths.manifest, manifest)
            except BaseException:
                try:
                    if self.unit_path.is_file() and self.unit_path.read_bytes() == unit:
                        self.unit_path.unlink()
                        _fsync_directory(self.unit_path.parent)
                    self.manager.daemon_reload()
                except BaseException:
                    pass
                raise
            identity, supervisor_status, manager = self._correlate(manifest)
            status = self._status_value(
                manifest,
                profile,
                desired,
                identity,
                supervisor_status,
                manager,
            )
            result = self._result(
                "register",
                manifest,
                profile,
                desired,
                manager,
                message="exact-owned missing systemd user service definition recovered without activation",
                data={
                    "status": status,
                    "unit_verify": verify,
                    "daemon_reload": reload_result,
                    "process_started": False,
                    "manager_stop_result": manager_stop_result,
                    "missing_service_definition_recovered": True,
                },
            )
            transaction_id = self._record_transaction(
                "register", before=before, after=manager, result=result
            )
            result["data"]["adapter_transaction_id"] = transaction_id
            return result
        metadata = self.unit_path.lstat()
        owner_uid, owner_gid = _repository_owner()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != owner_uid
            or metadata.st_gid != owner_gid
            or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            _fail(
                "FOREIGN_SERVICE_DEFINITION",
                "registered native service definition is not private and user-owned",
            )
        before = self.manager.status()
        manager_stop_result = None
        if not before.get("registered"):
            _fail(
                "MANAGER_STATE_MISMATCH",
                "registered service manager state is not registered",
                data={"manager_status": before},
            )
        if before.get("active") or before.get("active_state") in {"activating", "deactivating", "reloading", "auto-restart"}:
            manager_stop_result = self.manager.stop()
            before = self.manager.status()
        if before.get("active") or before.get("active_state") in {"activating", "deactivating", "reloading", "auto-restart"}:
            _fail("MANAGER_STATE_MISMATCH", "registered service remained active during reconciliation", data={"manager_status": before})
        if desired.desired_state == "RUNNING":
            try:
                desired = set_desired_state(profile, "STOPPED", config["configuration_reference"]["state_path"], expected_generation=desired.generation)
            except ServiceControlError as exc:
                _fail("DESIRED_STATE_PROFILE_MISMATCH", exc.message)
        self._prepare_runtime()
        _atomic_write(self.unit_path, unit)
        verify = self.manager.verify_unit()
        reload_result = self.manager.daemon_reload()
        after = self.manager.status()
        if (
            not after.get("registered")
            or after.get("active")
            or after.get("fragment_path") != str(self.unit_path)
            or bool(after.get("enabled")) != bool(before.get("enabled"))
        ):
            _fail(
                "MANAGER_STATE_MISMATCH",
                "registration reconciliation did not preserve inactive enablement state",
                data={"manager_before": before, "manager_after": after},
            )
        timestamp = utc_now()
        manifest = dict(old_manifest)
        manifest.update(
            {
                "supervisor_entrypoint": config["supervisor_entrypoint"],
                "configuration_reference": config[
                    "configuration_reference"
                ],
                "configuration_identity": config[
                    "configuration_identity"
                ],
                "registered": True,
                "enabled": bool(after.get("enabled")),
                "active": False,
                "manifest_generation": int(
                    old_manifest["manifest_generation"]
                )
                + 1,
                "updated_utc": timestamp,
                "last_activation_result": None,
                "last_failure_reason": None,
                "native_service": config["native_service"],
            }
        )
        _atomic_json(self.paths.manifest, manifest)
        identity, supervisor_status, manager = self._correlate(manifest)
        status = self._status_value(
            manifest,
            profile,
            desired,
            identity,
            supervisor_status,
            manager,
        )
        result = self._result(
            "register",
            manifest,
            profile,
            desired,
            manager,
            message=(
                "existing systemd user service reconciled without activation"
            ),
            data={
                "status": status,
                "unit_verify": verify,
                "daemon_reload": reload_result,
                "process_started": False,
                "manager_stop_result": manager_stop_result,
                "reconciled_existing_registration": True,
            },
        )
        transaction_id = self._record_transaction(
            "register", before=before, after=manager, result=result
        )
        result["data"]["adapter_transaction_id"] = transaction_id
        return result


    def _reconcile_unmanifested_existing(
        self,
        *,
        profile_path: Path | str,
        state_path: Path | str,
        supervisor_runtime_root: Path | str,
        supervisor_entrypoint: Path | str,
        config: Mapping[str, Any],
        profile: OperatingProfile,
        desired: DesiredState,
        unit: bytes,
        before: Mapping[str, Any],
    ) -> dict[str, Any]:
        existing = _existing_product_unit_identity(self.unit_path)
        old_unit = _read_regular(self.unit_path, "existing service definition")
        initial_before = dict(before)
        manager_stop_result = None
        if before.get("registered") and (
            before.get("active")
            or before.get("active_state")
            in {"activating", "deactivating", "reloading", "auto-restart"}
        ):
            manager_stop_result = self.manager.stop()
            before = self.manager.status()
        if before.get("active") or before.get("active_state") in {"activating", "deactivating", "reloading", "auto-restart"}:
            _fail("MANAGER_STATE_MISMATCH", "owned existing service remained active during reconciliation", data={"manager_status": before})
        self._prepare_runtime()
        try:
            _atomic_write(self.unit_path, unit)
            verify = self.manager.verify_unit()
            reload_result = self.manager.daemon_reload()
            after = self.manager.status()
            if (
                not after.get("registered")
                or after.get("active")
                or after.get("fragment_path") != str(self.unit_path)
                or bool(after.get("enabled")) != bool(before.get("enabled"))
            ):
                _fail("MANAGER_STATE_MISMATCH", "owned service reconciliation did not produce an inactive clone-bound unit", data={"manager_before": before, "manager_after": after})
            timestamp = utc_now()
            manifest = {
                "schema_version": MANIFEST_SCHEMA,
                "adapter_identity": ADAPTER_IDENTITY,
                "adapter_version": ADAPTER_VERSION,
                "supported_platform_family": PLATFORM_FAMILY,
                "required_host_capabilities": list(REQUIRED_HOST_CAPABILITIES),
                "activation_method": ACTIVATION_METHOD,
                "automatic_activation_supported": True,
                "supervisor_entrypoint": config["supervisor_entrypoint"],
                "configuration_reference": config["configuration_reference"],
                "configuration_identity": config["configuration_identity"],
                "registered": True,
                "enabled": bool(after.get("enabled")),
                "active": False,
                "manifest_generation": 1,
                "registered_utc": timestamp,
                "updated_utc": timestamp,
                "last_activation_result": None,
                "last_failure_reason": None,
                "native_service": config["native_service"],
            }
            _atomic_json(self.paths.manifest, manifest, exclusive=True, conflict_reason="ADAPTER_ALREADY_REGISTERED")
        except BaseException:
            try:
                _atomic_write(self.unit_path, old_unit)
                self.manager.daemon_reload()
            except BaseException:
                pass
            raise
        identity, supervisor_status, manager = self._correlate(manifest)
        status = self._status_value(manifest, profile, desired, identity, supervisor_status, manager)
        result = self._result(
            "register",
            manifest,
            profile,
            desired,
            manager,
            message="existing System X user service reconciled without activation",
            data={
                "status": status,
                "unit_verify": verify,
                "daemon_reload": reload_result,
                "process_started": False,
                "manager_stop_result": manager_stop_result,
                "existing_unit_root": existing["branch_root"],
                "existing_unit_sha256": existing["unit_sha256"],
                "reconciled_existing_registration": True,
            },
        )
        transaction_id = self._record_transaction("register", before=initial_before, after=manager, result=result)
        result["data"]["adapter_transaction_id"] = transaction_id
        return result

    def register(
        self,
        *,
        profile_path: Path | str,
        state_path: Path | str,
        supervisor_runtime_root: Path | str,
        supervisor_entrypoint: Path | str,
    ) -> dict[str, Any]:
        if self._manifest_exists():
            return self._reconcile_registered(
                profile_path=profile_path,
                state_path=state_path,
                supervisor_runtime_root=supervisor_runtime_root,
                supervisor_entrypoint=supervisor_entrypoint,
            )
        config, profile, desired, unit = self._render_configuration(
            profile_path=profile_path,
            state_path=state_path,
            supervisor_runtime_root=supervisor_runtime_root,
            supervisor_entrypoint=supervisor_entrypoint,
        )
        if desired.desired_state != "STOPPED":
            _fail(
                "DESIRED_STATE_PROFILE_MISMATCH",
                "registration requires desired state STOPPED",
            )
        before = self.manager.status()
        if before.get("registered") or os.path.lexists(self.unit_path):
            return self._reconcile_unmanifested_existing(
                profile_path=profile_path,
                state_path=state_path,
                supervisor_runtime_root=supervisor_runtime_root,
                supervisor_entrypoint=supervisor_entrypoint,
                config=config,
                profile=profile,
                desired=desired,
                unit=unit,
                before=before,
            )
        _adopt_user_owned_tree(
            _absolute(supervisor_runtime_root).parent,
            "System X generated runtime",
        )
        self._prepare_runtime()
        _atomic_write(
            self.unit_path,
            unit,
            exclusive=True,
            conflict_reason="SERVICE_NAME_COLLISION",
        )
        try:
            verify = self.manager.verify_unit()
            reload_result = self.manager.daemon_reload()
            after = self.manager.status()
            if (
                not after.get("registered")
                or after.get("enabled")
                or after.get("active")
                or after.get("fragment_path") != str(self.unit_path)
            ):
                _fail(
                    "MANAGER_STATE_MISMATCH",
                    "registration did not produce loaded disabled inactive state",
                )
            timestamp = utc_now()
            manifest = {
                "schema_version": MANIFEST_SCHEMA,
                "adapter_identity": ADAPTER_IDENTITY,
                "adapter_version": ADAPTER_VERSION,
                "supported_platform_family": PLATFORM_FAMILY,
                "required_host_capabilities": list(
                    REQUIRED_HOST_CAPABILITIES
                ),
                "activation_method": ACTIVATION_METHOD,
                "automatic_activation_supported": True,
                "supervisor_entrypoint": config[
                    "supervisor_entrypoint"
                ],
                "configuration_reference": config[
                    "configuration_reference"
                ],
                "configuration_identity": config[
                    "configuration_identity"
                ],
                "registered": True,
                "enabled": False,
                "active": False,
                "manifest_generation": 1,
                "registered_utc": timestamp,
                "updated_utc": timestamp,
                "last_activation_result": None,
                "last_failure_reason": None,
                "native_service": config["native_service"],
            }
            _atomic_json(
                self.paths.manifest,
                manifest,
                exclusive=True,
                conflict_reason="ADAPTER_ALREADY_REGISTERED",
            )
        except BaseException:
            try:
                if os.path.lexists(self.unit_path):
                    self.unit_path.unlink()
                    _fsync_directory(self.unit_path.parent)
                self.manager.daemon_reload()
            except BaseException:
                pass
            raise
        identity, supervisor_status, manager = self._correlate(manifest)
        status = self._status_value(
            manifest,
            profile,
            desired,
            identity,
            supervisor_status,
            manager,
        )
        result = self._result(
            "register",
            manifest,
            profile,
            desired,
            manager,
            message="systemd user service registered without activation",
            data={
                "status": status,
                "unit_verify": verify,
                "daemon_reload": reload_result,
                "process_started": False,
            },
        )
        transaction_id = self._record_transaction(
            "register", before=before, after=manager, result=result
        )
        result["data"]["adapter_transaction_id"] = transaction_id
        return result

    def enable(self) -> dict[str, Any]:
        manifest, profile, desired, _unit = self._load_manifest()
        identity, supervisor_status, before = self._correlate(manifest)
        if before["active"]:
            _fail(
                "ACTIVE_DISABLE_FORBIDDEN",
                "enable transition requires inactive service",
            )
        if manifest["enabled"]:
            _fail(
                "ADAPTER_ALREADY_ENABLED",
                "systemd user adapter is already enabled",
            )
        manager_result = self.manager.enable()
        after = self.manager.status()
        if not after["enabled"] or after["active"]:
            _fail(
                "MANAGER_STATE_MISMATCH",
                "enable did not leave service enabled and inactive",
            )
        manifest = self._update_manifest(manifest, enabled=True)
        status = self._status_value(
            manifest,
            profile,
            desired,
            identity,
            supervisor_status,
            after,
        )
        result = self._result(
            "enable",
            manifest,
            profile,
            desired,
            after,
            message="automatic systemd user activation enabled without start",
            data={
                "status": status,
                "manager_result": manager_result,
                "process_started": False,
            },
        )
        transaction_id = self._record_transaction(
            "enable", before=before, after=after, result=result
        )
        result["data"]["adapter_transaction_id"] = transaction_id
        return result

    def _wait_active(
        self,
        manifest: Mapping[str, Any],
        *,
        timeout_seconds: float,
        previous_pid: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        deadline = time.monotonic() + timeout_seconds
        last_manager: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            last_manager = self.manager.status()
            if last_manager.get("active") and last_manager.get("main_pid"):
                try:
                    supervisor = self._supervisor_evidence(
                        manifest, require_status=True
                    )
                except AdapterError:
                    supervisor = None
                if supervisor is not None:
                    identity, status = supervisor
                    if (
                        identity["pid"] == last_manager["main_pid"]
                        and (
                            previous_pid is None
                            or identity["pid"] != previous_pid
                        )
                    ):
                        assert status is not None
                        return identity, status, last_manager
            time.sleep(0.05)
        _fail(
            "SUPERVISOR_START_FAILED",
            "manager did not expose a correlated supervisor before timeout",
            exit_code=4,
            data={"last_manager_status": last_manager or {}},
        )

    def start(
        self,
        *,
        expected_generation: int | None = None,
        wait_timeout_seconds: float = 60.0,
    ) -> dict[str, Any]:
        manifest, profile, desired, _unit = self._load_manifest()
        _identity, _supervisor_status, before = self._correlate(manifest)
        if not manifest["enabled"]:
            _fail("ADAPTER_DISABLED", "adapter must be enabled before start")
        if before["active"]:
            _fail("ALREADY_ACTIVE", "native service is already active")
        self._preflight_endpoints(profile)
        if desired.desired_state == "STOPPED":
            try:
                desired = set_desired_state(
                    profile,
                    "RUNNING",
                    manifest["configuration_reference"]["state_path"],
                    expected_generation=(
                        expected_generation
                        if expected_generation is not None
                        else desired.generation
                    ),
                )
            except ServiceControlError as exc:
                _fail("DESIRED_STATE_PROFILE_MISMATCH", exc.message)
        elif (
            expected_generation is not None
            and expected_generation != desired.generation
        ):
            _fail(
                "DESIRED_STATE_PROFILE_MISMATCH",
                "expected desired-state generation is stale",
            )
        manager_result = self.manager.start()
        identity, supervisor_status, after = self._wait_active(
            manifest, timeout_seconds=float(wait_timeout_seconds)
        )
        activation = bounded_activation_result(
            operation="start",
            ok=True,
            reason_code="OK",
            message="native manager started one correlated supervisor",
        )
        manifest = self._update_manifest(
            manifest,
            active=True,
            last_activation_result=activation,
            last_failure_reason=None,
        )
        status = self._status_value(
            manifest,
            profile,
            desired,
            identity,
            supervisor_status,
            after,
        )
        result = self._result(
            "start",
            manifest,
            profile,
            desired,
            after,
            message="systemd user manager started the System X supervisor",
            data={
                "status": status,
                "manager_result": manager_result,
                "supervisor_identity": identity,
            },
        )
        transaction_id = self._record_transaction(
            "start", before=before, after=after, result=result
        )
        result["data"]["adapter_transaction_id"] = transaction_id
        return result

    def stop(
        self,
        *,
        expected_generation: int | None = None,
        wait_timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        manifest, profile, desired, _unit = self._load_manifest(
            allow_stale_configuration_for_inactive_removal=True
        )
        _identity, _supervisor_status, before = self._correlate(manifest)
        if desired.desired_state == "RUNNING":
            try:
                desired = set_desired_state(
                    profile,
                    "STOPPED",
                    manifest["configuration_reference"]["state_path"],
                    expected_generation=(
                        expected_generation
                        if expected_generation is not None
                        else desired.generation
                    ),
                )
            except ServiceControlError as exc:
                _fail("DESIRED_STATE_PROFILE_MISMATCH", exc.message)
        elif (
            expected_generation is not None
            and expected_generation != desired.generation
        ):
            _fail(
                "DESIRED_STATE_PROFILE_MISMATCH",
                "expected desired-state generation is stale",
            )
        timeout = float(
            wait_timeout_seconds
            if wait_timeout_seconds is not None
            else manifest["native_service"]["timeout_stop_seconds"]
        )
        deadline = time.monotonic() + timeout
        manager_needs_stop = bool(
            before["active"]
            or before.get("active_state")
            in {"activating", "deactivating", "reloading"}
            or before.get("sub_state") == "auto-restart"
        )
        last_manager = before
        manager_result: dict[str, Any] | None = None
        if before["active"] and isinstance(
            self.manager, SystemdUserManager
        ):
            graceful_deadline = min(
                deadline,
                time.monotonic()
                + float(profile.graceful_shutdown_timeout_seconds),
            )
            while time.monotonic() < graceful_deadline:
                last_manager = self.manager.status()
                supervisor = self._supervisor_evidence(
                    manifest, require_status=False
                )
                if not last_manager["active"] and supervisor is None:
                    manager_result = {
                        "desired_state_converged_without_manager_signal": True
                    }
                    break
                time.sleep(0.05)
        if manager_result is None:
            manager_result = (
                self.manager.stop()
                if manager_needs_stop
                else {"already_inactive": True}
            )
        while time.monotonic() < deadline:
            last_manager = self.manager.status()
            supervisor = self._supervisor_evidence(
                manifest, require_status=False
            )
            if not last_manager["active"] and supervisor is None:
                break
            time.sleep(0.05)
        else:
            _fail(
                "SUPERVISOR_STOP_FAILED",
                "manager or supervisor remained active after stop",
                exit_code=4,
                data={"manager_status": last_manager},
            )
        activation = bounded_activation_result(
            operation="stop",
            ok=True,
            reason_code="OK",
            message="desired state STOPPED and native service inactive",
        )
        manifest = self._update_manifest(
            manifest,
            active=False,
            last_activation_result=activation,
            last_failure_reason=None,
        )
        status = self._status_value(
            manifest,
            profile,
            desired,
            {},
            None,
            last_manager,
        )
        result = self._result(
            "stop",
            manifest,
            profile,
            desired,
            last_manager,
            message="platform adapter stopped System X through systemd",
            data={
                "status": status,
                "manager_result": manager_result,
                "state_changed": desired.desired_state == "STOPPED",
            },
        )
        transaction_id = self._record_transaction(
            "stop", before=before, after=last_manager, result=result
        )
        result["data"]["adapter_transaction_id"] = transaction_id
        return result

    def restart(
        self, *, wait_timeout_seconds: float = 60.0
    ) -> dict[str, Any]:
        manifest, profile, desired, _unit = self._load_manifest()
        old_identity, _status, before = self._correlate(manifest)
        if not manifest["enabled"]:
            _fail("ADAPTER_DISABLED", "adapter must be enabled")
        if not before["active"] or desired.desired_state != "RUNNING":
            _fail(
                "ADAPTER_INACTIVE",
                "restart requires active RUNNING service",
            )
        stop_result = self.stop(
            expected_generation=desired.generation,
            wait_timeout_seconds=float(wait_timeout_seconds),
        )
        stopped_generation = stop_result["desired_state_generation"]
        endpoint_release_wait_seconds = self._wait_endpoints_free(
            profile,
            timeout_seconds=float(wait_timeout_seconds),
        )
        start_result = self.start(
            expected_generation=stopped_generation,
            wait_timeout_seconds=float(wait_timeout_seconds),
        )
        manifest, profile, desired, _unit = self._load_manifest()
        new_identity, supervisor_status, after = self._correlate(manifest)
        if (
            isinstance(self.manager, SystemdUserManager)
            and new_identity.get("pid") == old_identity.get("pid")
        ):
            _fail(
                "MANAGER_SUPERVISOR_IDENTITY_MISMATCH",
                "graceful restart did not produce a new supervisor identity",
                exit_code=3,
            )
        activation = bounded_activation_result(
            operation="restart",
            ok=True,
            reason_code="OK",
            message=(
                "desired-state shutdown completed before native manager "
                "started one new supervisor identity"
            ),
        )
        manifest = self._update_manifest(
            manifest,
            active=True,
            last_activation_result=activation,
            last_failure_reason=None,
        )
        status = self._status_value(
            manifest,
            profile,
            desired,
            new_identity,
            supervisor_status,
            after,
        )
        result = self._result(
            "restart",
            manifest,
            profile,
            desired,
            after,
            message="platform adapter gracefully restarted System X",
            data={
                "status": status,
                "manager_result": {
                    "graceful_stop": stop_result["data"].get(
                        "manager_result"
                    ),
                    "start": start_result["data"].get("manager_result"),
                },
                "lifecycle": {
                    "stop": stop_result,
                    "start": start_result,
                },
                "endpoint_release": {
                    "wait_seconds": endpoint_release_wait_seconds,
                },
                "old_supervisor_identity": old_identity,
                "new_supervisor_identity": new_identity,
            },
        )
        transaction_id = self._record_transaction(
            "restart", before=before, after=after, result=result
        )
        result["data"]["adapter_transaction_id"] = transaction_id
        return result

    def status(self) -> dict[str, Any]:
        manifest, profile, desired, _unit = self._load_manifest()
        identity, supervisor_status, manager = self._correlate(manifest)
        if bool(manifest["active"]) != bool(manager["active"]):
            manifest = self._update_manifest(
                manifest, active=bool(manager["active"])
            )
        status = self._status_value(
            manifest,
            profile,
            desired,
            identity,
            supervisor_status,
            manager,
        )
        return self._result(
            "status",
            manifest,
            profile,
            desired,
            manager,
            message="native-manager and System X status correlated",
            data={"status": status, "manager_status": manager},
        )

    def disable(self) -> dict[str, Any]:
        manifest, profile, desired, _unit = self._load_manifest(
            allow_stale_configuration_for_inactive_removal=True
        )
        identity, supervisor_status, before = self._correlate(manifest)
        if before["active"]:
            _fail(
                "ACTIVE_DISABLE_FORBIDDEN",
                "active service must be stopped before disable",
                exit_code=3,
            )
        if desired.desired_state != "STOPPED":
            _fail(
                "DESIRED_STATE_PROFILE_MISMATCH",
                "disable requires desired state STOPPED",
            )
        if not manifest["enabled"]:
            _fail(
                "ADAPTER_ALREADY_DISABLED",
                "systemd user adapter is already disabled",
            )
        manager_result = self.manager.disable()
        after = self.manager.status()
        if after["enabled"] or after["active"]:
            _fail(
                "MANAGER_STATE_MISMATCH",
                "disable did not leave service disabled and inactive",
            )
        manifest = self._update_manifest(
            manifest, enabled=False, active=False
        )
        status = self._status_value(
            manifest,
            profile,
            desired,
            identity,
            supervisor_status,
            after,
        )
        result = self._result(
            "disable",
            manifest,
            profile,
            desired,
            after,
            message="automatic activation disabled without deletion",
            data={"status": status, "manager_result": manager_result},
        )
        transaction_id = self._record_transaction(
            "disable", before=before, after=after, result=result
        )
        result["data"]["adapter_transaction_id"] = transaction_id
        return result

    def configuration(self) -> dict[str, Any]:
        manifest, profile, desired, _unit = self._load_manifest()
        identity, supervisor_status, manager = self._correlate(manifest)
        self._status_value(
            manifest,
            profile,
            desired,
            identity,
            supervisor_status,
            manager,
        )
        return self._result(
            "configuration",
            manifest,
            profile,
            desired,
            manager,
            message="systemd user adapter configuration reported",
            data={
                "configuration_reference": manifest[
                    "configuration_reference"
                ],
                "configuration_identity": manifest[
                    "configuration_identity"
                ],
                "native_service": manifest["native_service"],
            },
        )

    @contextmanager
    def _configuration_lock(self):
        self._prepare_runtime()
        lock_path = self.paths.configuration_lock
        _reject_symlink_components(lock_path, "static configuration lock")
        try:
            descriptor = os.open(
                lock_path,
                os.O_RDWR
                | os.O_CREAT
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
        except OSError as exc:
            _fail(
                "ADAPTER_CONFIGURATION_CONFLICT",
                f"static configuration lock cannot open: {exc}",
            )
        try:
            metadata = os.fstat(descriptor)
            owner_uid, owner_gid = _repository_owner()
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != owner_uid
                or metadata.st_gid != owner_gid
            ):
                _fail(
                    "ADAPTER_CONFIGURATION_CONFLICT",
                    "static configuration lock is not user-owned",
                )
            os.fchmod(descriptor, 0o600)
            os.fsync(descriptor)
            _fsync_directory(lock_path.parent)
            try:
                fcntl.flock(
                    descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB
                )
            except BlockingIOError:
                _fail(
                    "ADAPTER_CONFIGURATION_CONFLICT",
                    "static configuration lock is busy",
                )
            try:
                yield lock_path
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def configure_static_ui(
        self,
        *,
        enabled: bool,
        distribution_root: Path | str | None,
        mount_path: str,
    ) -> dict[str, Any]:
        if type(enabled) is not bool:
            _fail(
                "ADAPTER_CONFIGURATION_CONFLICT",
                "enabled must be a Boolean",
            )
        if distribution_root is None:
            root_value = None
        else:
            try:
                root_value = os.fspath(distribution_root)
            except TypeError:
                _fail(
                    "ADAPTER_CONFIGURATION_CONFLICT",
                    "distribution_root must be a path or null",
                )
            if not isinstance(root_value, str):
                _fail(
                    "ADAPTER_CONFIGURATION_CONFLICT",
                    "distribution_root must be a text path",
                )
        if not enabled and root_value is not None:
            _fail(
                "ADAPTER_CONFIGURATION_CONFLICT",
                "distribution_root must be null when disabled",
            )
        references = (
            _validate_static_distribution(root_value, mount_path)
            if enabled and root_value is not None
            else []
        )
        with self._configuration_lock() as lock_path:
            old_manifest, old_profile, old_desired, old_unit = (
                self._load_manifest()
            )
            old_identity, old_supervisor_status, before = self._correlate(
                old_manifest
            )
            current_enabled = (
                old_profile.external_static_enabled
                if old_profile.external_static_fields_present
                else False
            )
            current_root = (
                old_profile.external_static_distribution_root
                if old_profile.external_static_fields_present
                else None
            )
            current_mount = old_profile.external_static_mount_path
            requested = (enabled, root_value, mount_path)
            current = (current_enabled, current_root, current_mount)
            if current == requested:
                status = self._status_value(
                    old_manifest,
                    old_profile,
                    old_desired,
                    old_identity,
                    old_supervisor_status,
                    before,
                )
                return self._result(
                    "configure-static-ui",
                    old_manifest,
                    old_profile,
                    old_desired,
                    before,
                    message="static UI configuration is already exact; no-op verified",
                    data={
                        "changed": False,
                        "no_op": True,
                        "configuration_lock": str(lock_path),
                        "configuration": {
                            "enabled": enabled,
                            "distribution_root": root_value,
                            "mount_path": mount_path,
                        },
                        "status": status,
                    },
                )
            reference = old_manifest["configuration_reference"]
            profile_file = Path(reference["profile_path"])
            state_file = Path(reference["state_path"])
            old_profile_bytes = _read_regular(
                profile_file, "operating profile"
            )
            old_state_bytes = _read_regular(state_file, "desired state")
            old_manifest_bytes = _read_regular(
                self.paths.manifest, "adapter manifest"
            )
            old_status_bytes = _read_regular(
                self.paths.status, "adapter status"
            )
            transaction_recorded = False
            try:
                (
                    previous_profile,
                    new_profile,
                    previous_desired,
                    new_desired,
                ) = configure_static_profile(
                    profile_file,
                    state_file,
                    external_static_enabled=enabled,
                    external_static_distribution_root=root_value,
                    external_static_mount_path=mount_path,
                )
                config, rendered_profile, rendered_desired, new_unit = (
                    self._render_configuration(
                        profile_path=profile_file,
                        state_path=state_file,
                        supervisor_runtime_root=reference[
                            "supervisor_runtime_root"
                        ],
                        supervisor_entrypoint=old_manifest[
                            "supervisor_entrypoint"
                        ]["path"],
                    )
                )
                if (
                    rendered_profile.identity != new_profile.identity
                    or rendered_desired.profile_identity
                    != new_desired.profile_identity
                    or new_unit != old_unit
                ):
                    _fail(
                        "ADAPTER_CONFIGURATION_CONFLICT",
                        "static configuration changed a generic service identity",
                    )
                new_manifest = dict(old_manifest)
                new_manifest.update(
                    {
                        "supervisor_entrypoint": config[
                            "supervisor_entrypoint"
                        ],
                        "configuration_reference": config[
                            "configuration_reference"
                        ],
                        "configuration_identity": config[
                            "configuration_identity"
                        ],
                        "manifest_generation": int(
                            old_manifest["manifest_generation"]
                        )
                        + 1,
                        "updated_utc": utc_now(),
                        "native_service": config["native_service"],
                    }
                )
                _atomic_json(self.paths.manifest, new_manifest)
                status = self._status_value(
                    new_manifest,
                    new_profile,
                    new_desired,
                    old_identity,
                    old_supervisor_status,
                    before,
                )
                result = self._result(
                    "configure-static-ui",
                    new_manifest,
                    new_profile,
                    new_desired,
                    before,
                    message="static UI configuration persisted without restart",
                    data={
                        "changed": True,
                        "no_op": False,
                        "configuration_lock": str(lock_path),
                        "configuration": {
                            "enabled": enabled,
                            "distribution_root": root_value,
                            "mount_path": mount_path,
                        },
                        "static_references": references,
                        "previous_profile_identity": previous_profile.identity,
                        "new_profile_identity": new_profile.identity,
                        "previous_desired_state_generation": (
                            previous_desired.generation
                        ),
                        "new_desired_state_generation": new_desired.generation,
                        "previous_configuration_identity": old_manifest[
                            "configuration_identity"
                        ],
                        "new_configuration_identity": new_manifest[
                            "configuration_identity"
                        ],
                        "service_restarted": False,
                        "status": status,
                    },
                )
                transaction_id = self._record_transaction(
                    "configure-static-ui",
                    before=before,
                    after=before,
                    result=result,
                )
                transaction_recorded = True
                result["data"]["adapter_transaction_id"] = transaction_id
                return result
            except BaseException:
                if not transaction_recorded:
                    try:
                        _atomic_write(profile_file, old_profile_bytes)
                    except BaseException:
                        pass
                    try:
                        _atomic_write(state_file, old_state_bytes)
                    except BaseException:
                        pass
                    try:
                        _atomic_write(
                            self.paths.manifest, old_manifest_bytes
                        )
                    except BaseException:
                        pass
                    try:
                        _atomic_write(self.paths.status, old_status_bytes)
                    except BaseException:
                        pass
                raise

    def supervisor_entrypoint(self) -> dict[str, Any]:
        manifest, profile, desired, _unit = self._load_manifest()
        identity, supervisor_status, manager = self._correlate(manifest)
        self._status_value(
            manifest,
            profile,
            desired,
            identity,
            supervisor_status,
            manager,
        )
        return self._result(
            "supervisor-entrypoint",
            manifest,
            profile,
            desired,
            manager,
            message="validated supervisor entrypoint reported",
            data={
                "supervisor_entrypoint": manifest[
                    "supervisor_entrypoint"
                ],
                "interpreter_identity": manifest["native_service"][
                    "interpreter_identity"
                ],
            },
        )

    def unregister(self, *, explicit: bool = False) -> dict[str, Any]:
        if not explicit:
            _fail(
                "UNREGISTER_NOT_EXPLICIT",
                "unregister requires the exact explicit operation",
            )
        manifest, profile, desired, _unit = self._load_manifest(
            allow_stale_configuration_for_inactive_removal=True
        )
        _identity, _supervisor_status, before = self._correlate(manifest)
        if before["active"]:
            _fail(
                "UNREGISTER_REQUIRES_DISABLED_INACTIVE_STOPPED",
                "unregister requires inactive native service",
                exit_code=3,
            )
        if manifest["enabled"] or before["enabled"]:
            _fail(
                "UNREGISTER_REQUIRES_DISABLED_INACTIVE_STOPPED",
                "unregister requires disabled native service",
                exit_code=3,
            )
        if desired.desired_state != "STOPPED":
            _fail(
                "UNREGISTER_REQUIRES_DISABLED_INACTIVE_STOPPED",
                "unregister requires desired state STOPPED",
                exit_code=3,
            )
        if self._supervisor_evidence(manifest, require_status=False):
            _fail(
                "UNREGISTER_REQUIRES_DISABLED_INACTIVE_STOPPED",
                "unregister requires no owned supervisor runtime",
                exit_code=3,
            )
        unit_data = _read_regular(
            self.unit_path, "native service definition"
        )
        if hashlib.sha256(unit_data).hexdigest() != manifest[
            "native_service"
        ]["service_definition_sha256"]:
            _fail(
                "FOREIGN_SERVICE_DEFINITION",
                "refusing to remove a changed native service definition",
            )
        try:
            self.unit_path.unlink()
            _fsync_directory(self.unit_path.parent)
        except OSError as exc:
            _fail(
                "FOREIGN_SERVICE_DEFINITION",
                f"native service definition removal failed: {exc}",
            )
        reload_result = self.manager.daemon_reload()
        after = self.manager.status()
        if after.get("registered"):
            _fail(
                "MANAGER_STATE_MISMATCH",
                "native manager still reports removed service",
            )
        for path in (self.paths.status, self.paths.manifest):
            try:
                path.unlink()
                _fsync_directory(path.parent)
            except FileNotFoundError:
                pass
            except OSError as exc:
                _fail(
                    "ADAPTER_MANIFEST_INVALID",
                    f"current-state removal failed: {exc}",
                )
        result = result_envelope(
            "unregister",
            ok=True,
            reason_code="OK",
            message="native registration removed explicitly",
            adapter_identity=ADAPTER_IDENTITY,
            adapter_version=ADAPTER_VERSION,
            automatic_activation_supported=True,
            registered=False,
            enabled=False,
            active=False,
            profile_identity=profile.identity,
            desired_state=desired.desired_state,
            desired_state_generation=desired.generation,
            paths=self.paths.view(),
            data={
                "configuration_identity": manifest[
                    "configuration_identity"
                ],
                "daemon_reload": reload_result,
                "history_preserved": True,
                "system_x_preserved": True,
            },
        )
        transaction_id = self._record_transaction(
            "unregister", before=before, after=after, result=result
        )
        result["data"]["adapter_transaction_id"] = transaction_id
        return result

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
            "configure-static-ui": self.configure_static_ui,
            "supervisor-entrypoint": self.supervisor_entrypoint,
        }
        method = methods.get(operation)
        if method is None:
            _fail("ADAPTER_NOT_SUPPORTED", f"unknown operation: {operation}")
        return method(**arguments)

    def error_result(
        self, operation: str, error: AdapterError
    ) -> dict[str, Any]:
        manifest = None
        profile = None
        desired = None
        manager = None
        try:
            manifest, profile, desired, _unit = self._load_manifest()
            manager = self.manager.status()
        except BaseException:
            pass
        return result_envelope(
            operation,
            ok=False,
            reason_code=error.reason_code,
            message=error.message,
            adapter_identity=ADAPTER_IDENTITY,
            adapter_version=ADAPTER_VERSION,
            automatic_activation_supported=True,
            configuration_identity=(
                manifest.get("configuration_identity")
                if manifest is not None
                else None
            ),
            registered=manifest is not None,
            enabled=bool(manager and manager.get("enabled")),
            active=bool(manager and manager.get("active")),
            profile_identity=(
                profile.identity if profile is not None else None
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


def _add_configuration_arguments(
    parser: argparse.ArgumentParser, *, required: bool
) -> None:
    parser.add_argument("--profile", type=Path, required=required)
    parser.add_argument("--state", type=Path, required=required)
    parser.add_argument(
        "--supervisor-runtime-root", type=Path, required=required
    )
    parser.add_argument(
        "--supervisor-entrypoint", type=Path, required=required
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser()
    parser.add_argument(
        "--adapter-runtime-root",
        type=Path,
        default=DEFAULT_ADAPTER_RUNTIME_ROOT,
    )
    parser.add_argument("--service-name", default=SERVICE_NAME)
    parser.add_argument("--unit-path", type=Path)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    for operation in LINUX_OPERATIONS:
        command = subparsers.add_parser(operation)
        if operation == "register":
            _add_configuration_arguments(command, required=True)
        elif operation == "validate":
            _add_configuration_arguments(command, required=False)
        elif operation in ("start", "stop"):
            command.add_argument("--expected-generation", type=int)
            command.add_argument("--wait-timeout-seconds", type=float)
        elif operation == "restart":
            command.add_argument("--wait-timeout-seconds", type=float)
        elif operation == "configure-static-ui":
            command.add_argument(
                "--enabled", choices=("true", "false"), required=True
            )
            command.add_argument("--distribution-root", type=Path)
            command.add_argument("--mount-path", required=True)
        elif operation == "unregister":
            command.add_argument("--explicit", action="store_true")
        elif operation == "capability":
            command.add_argument("--require", action="append")
    return parser


def _operation_arguments(
    arguments: argparse.Namespace,
) -> dict[str, Any]:
    operation = arguments.operation
    if operation in ("register", "validate"):
        return {
            "profile_path": arguments.profile,
            "state_path": arguments.state,
            "supervisor_runtime_root": (
                arguments.supervisor_runtime_root
            ),
            "supervisor_entrypoint": arguments.supervisor_entrypoint,
        }
    if operation in ("start", "stop"):
        values = {
            "expected_generation": arguments.expected_generation,
        }
        if arguments.wait_timeout_seconds is not None:
            values["wait_timeout_seconds"] = (
                arguments.wait_timeout_seconds
            )
        return values
    if operation == "restart":
        return (
            {
                "wait_timeout_seconds": arguments.wait_timeout_seconds
            }
            if arguments.wait_timeout_seconds is not None
            else {}
        )
    if operation == "configure-static-ui":
        return {
            "enabled": arguments.enabled == "true",
            "distribution_root": arguments.distribution_root,
            "mount_path": arguments.mount_path,
        }
    if operation == "unregister":
        return {"explicit": arguments.explicit}
    if operation == "capability":
        return {"requested": arguments.require}
    return {}


def main(argv: Sequence[str] | None = None) -> int:
    operation = "identify"
    adapter: LinuxSystemdUserServiceAdapter | None = None
    try:
        arguments = build_argument_parser().parse_args(argv)
        operation = arguments.operation
        adapter = LinuxSystemdUserServiceAdapter(
            arguments.adapter_runtime_root,
            service_name=arguments.service_name,
            unit_path=arguments.unit_path,
        )
        result = adapter.invoke(
            operation, **_operation_arguments(arguments)
        )
        exit_code = 0
    except AdapterError as exc:
        if adapter is None:
            adapter = LinuxSystemdUserServiceAdapter(
                DEFAULT_ADAPTER_RUNTIME_ROOT
            )
        result = adapter.error_result(operation, exc)
        exit_code = exc.exit_code
    print(canonical_json(result))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
