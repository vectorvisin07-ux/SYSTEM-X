#!/usr/bin/env python3
"""Thin lifecycle controller for a self-relative GGUF server branch."""

from __future__ import annotations

import argparse
import datetime as dt
import errno
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import stat
import subprocess
import sys
import time
from typing import Any, NoReturn


SCHEMA_VERSION = "system-x.gguf-branch-controller.v1"
READINESS = "NOT_CHECKED"
RUNTIME_CHILDREN = (
    "cache",
    "tmp",
    "logs",
    "status",
    "pids",
    "locks",
    "transactions",
)
LIFECYCLE_STATES = {
    "PREPARING",
    "STARTING",
    "STARTED",
    "START_FAILED",
    "STOPPING",
    "STOPPED",
    "STOP_FAILED",
    "INCONSISTENT",
    "RECONCILED",
}
GRACEFUL_WAIT_SECONDS = 10.0
FORCED_WAIT_SECONDS = 5.0
POLL_INTERVAL_SECONDS = 0.1
IMMEDIATE_EXIT_WINDOW_SECONDS = 1.0


class ControllerError(Exception):
    """A bounded domain failure with a stable reason and exit status."""

    def __init__(
        self,
        reason_code: str,
        message: str,
        exit_status: int,
        data: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.message = bounded_text(message)
        self.exit_status = exit_status
        self.data = data or {}


class JsonArgumentParser(argparse.ArgumentParser):
    """Argument parser that raises instead of printing prose."""

    def error(self, message: str) -> NoReturn:
        raise ControllerError("INVALID_INPUT", message, 2)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def transaction_id() -> str:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"tx-{stamp}-{secrets.token_hex(6)}"


def bounded_text(value: object, limit: int = 400) -> str:
    text = str(value).replace("\x00", "?").replace("\r", " ").replace("\n", " ")
    return text[:limit]


def derive_paths() -> dict[str, Path]:
    source_path = Path(__file__).resolve(strict=True)
    controller_dir = source_path.parent
    branch_root = controller_dir.parent.resolve(strict=True)
    return {
        "source_path": source_path,
        "controller_dir": controller_dir,
        "branch_root": branch_root,
        "binary_path": branch_root / "llama.cpp" / "build" / "bin" / "llama-server",
        "model_root": branch_root / "MODEL" / "SUPERMODEL",
        "runtime_root": branch_root / "RUNTIME",
        "router_cache": branch_root / "RUNTIME" / "api" / "backend" / "cache",
    }


def result_envelope(
    operation: str,
    ok: bool,
    reason_code: str,
    message: str,
    branch_root: Path,
    data: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "operation": operation,
        "ok": ok,
        "reason_code": reason_code,
        "message": bounded_text(message),
        "timestamp_utc": utc_now(),
        "branch_root": str(branch_root),
        "data": data,
    }


def emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def lstat_or_none(path: Path) -> os.stat_result | None:
    try:
        return path.lstat()
    except FileNotFoundError:
        return None


def require_real_directory(path: Path, label: str) -> Path:
    info = lstat_or_none(path)
    if info is None or not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise ControllerError(
            "RUNTIME_LAYOUT_INVALID",
            f"{label} must be an existing real directory",
            2,
            {"path": str(path), "label": label},
        )
    return path.resolve(strict=True)


def validate_layout(paths: dict[str, Path]) -> dict[str, Path]:
    controller_dir = require_real_directory(paths["controller_dir"], "controller_dir")
    branch_root = require_real_directory(paths["branch_root"], "branch_root")

    binary_candidate = paths["binary_path"]
    binary_info = lstat_or_none(binary_candidate)
    if binary_info is None or not stat.S_ISREG(binary_info.st_mode):
        raise ControllerError(
            "BINARY_MISSING",
            "the branch-local server executable is missing or not a regular file",
            2,
            {"path": str(binary_candidate)},
        )
    if not os.access(binary_candidate, os.X_OK):
        raise ControllerError(
            "BINARY_NOT_EXECUTABLE",
            "the branch-local server file is not executable",
            2,
            {"path": str(binary_candidate)},
        )
    binary_path = binary_candidate.resolve(strict=True)

    model_root = require_real_directory(paths["model_root"], "model_root")
    runtime_root = require_real_directory(paths["runtime_root"], "runtime_root")
    for child in RUNTIME_CHILDREN:
        require_real_directory(runtime_root / child, f"runtime/{child}")

    return {
        **paths,
        "controller_dir": controller_dir,
        "branch_root": branch_root,
        "binary_path": binary_path,
        "model_root": model_root,
        "runtime_root": runtime_root,
    }


def validate_router_layout(paths: dict[str, Path]) -> dict[str, Path]:
    models_dir = require_real_directory(paths["model_root"], "router models directory")
    try:
        models_dir.relative_to(paths["branch_root"])
    except ValueError as exc:
        raise ControllerError(
            "RUNTIME_LAYOUT_INVALID",
            "router models directory resolves outside the branch",
            2,
        ) from exc
    if not os.access(models_dir, os.R_OK | os.X_OK):
        raise ControllerError(
            "RUNTIME_LAYOUT_INVALID",
            "router models directory is not readable",
            2,
            {"path": str(models_dir)},
        )

    backend_root = require_real_directory(
        paths["runtime_root"] / "api" / "backend", "runtime/api/backend"
    )
    router_cache = require_real_directory(paths["router_cache"], "router cache")
    try:
        router_cache.relative_to(backend_root)
    except ValueError as exc:
        raise ControllerError(
            "RUNTIME_LAYOUT_INVALID",
            "router cache resolves outside runtime/api/backend",
            2,
        ) from exc
    cache_info = router_cache.stat()
    if cache_info.st_uid != os.geteuid():
        raise ControllerError(
            "RUNTIME_LAYOUT_INVALID",
            "router cache is not owned by the current branch user",
            2,
            {"path": str(router_cache)},
        )
    if not os.access(router_cache, os.R_OK | os.W_OK | os.X_OK):
        raise ControllerError(
            "RUNTIME_LAYOUT_INVALID",
            "router cache is not accessible to the current branch user",
            2,
            {"path": str(router_cache)},
        )
    return {**paths, "models_dir": models_dir, "router_cache": router_cache}


def runtime_paths(paths: dict[str, Path], tx_id: str | None = None) -> dict[str, Path]:
    runtime_root = paths["runtime_root"]
    values = {
        "lock_path": runtime_root / "locks" / "active.lock",
        "pid_path": runtime_root / "pids" / "active.json",
        "status_path": runtime_root / "status" / "branch.json",
        "transaction_parent": runtime_root / "transactions",
        "log_parent": runtime_root / "logs",
    }
    if tx_id is not None:
        values["transaction_path"] = values["transaction_parent"] / f"{tx_id}.json"
        values["log_path"] = values["log_parent"] / f"{tx_id}.log"
    return values


def write_all(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("short write")
        view = view[written:]


def fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | os.O_CLOEXEC
    directory_fd = os.open(path, flags)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def atomic_json_write(path: Path, value: dict[str, Any]) -> None:
    temporary = path.parent / f".{path.name}.{secrets.token_hex(12)}.tmp"
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | os.O_CLOEXEC
        | getattr(os, "O_NOFOLLOW", 0)
    )
    fd: int | None = None
    try:
        fd = os.open(temporary, flags, 0o600)
        write_all(fd, json_bytes(value))
        os.fsync(fd)
        os.close(fd)
        fd = None
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        if fd is not None:
            os.close(fd)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def exclusive_json_create(path: Path, value: dict[str, Any]) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | os.O_CLOEXEC
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        fd = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise ControllerError(
            "BRANCH_LOCK_ACTIVE",
            "the branch active lock already exists",
            3,
            {"lock_path": str(path)},
        ) from exc
    try:
        write_all(fd, json_bytes(value))
        os.fsync(fd)
    finally:
        os.close(fd)
    fsync_directory(path.parent)


def read_json_record(path: Path, label: str) -> dict[str, Any]:
    info = lstat_or_none(path)
    if info is None:
        raise ControllerError(
            "ACTIVE_STATE_INCONSISTENT",
            f"{label} is missing",
            3,
            {"path": str(path)},
        )
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ControllerError(
            "ACTIVE_STATE_INCONSISTENT",
            f"{label} is not a direct regular file",
            3,
            {"path": str(path)},
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ControllerError(
            "ACTIVE_STATE_INCONSISTENT",
            f"{label} could not be read as JSON",
            3,
            {"path": str(path), "error_type": type(exc).__name__},
        ) from exc
    if not isinstance(value, dict):
        raise ControllerError(
            "ACTIVE_STATE_INCONSISTENT",
            f"{label} must contain one JSON object",
            3,
            {"path": str(path)},
        )
    return value


def remove_owned_record(path: Path, tx_id: str) -> bool:
    if not path.exists() and not path.is_symlink():
        return True
    value = read_json_record(path, path.name)
    if value.get("transaction_id") != tx_id:
        return False
    path.unlink()
    fsync_directory(path.parent)
    return True


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=1024 * 1024) as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binary_identity(path: Path) -> dict[str, Any]:
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ControllerError(
            "BINARY_MISSING", "server executable is not a regular file", 2
        )
    return {
        "canonical_path": str(path),
        "device": info.st_dev,
        "inode": info.st_ino,
        "size": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "sha256": sha256_file(path),
    }


def validate_text(value: str, label: str) -> str:
    if not value:
        raise ControllerError("INVALID_INPUT", f"{label} must not be empty", 2)
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ControllerError(
            "INVALID_INPUT", f"{label} contains an ASCII control character", 2
        )
    return value


def positive_integer(value: str | int, label: str, maximum: int | None = None) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ControllerError("INVALID_INPUT", f"{label} must be an integer", 2) from exc
    if normalized <= 0 or (maximum is not None and normalized > maximum):
        detail = f"1 through {maximum}" if maximum is not None else "positive"
        raise ControllerError("INVALID_INPUT", f"{label} must be {detail}", 2)
    return normalized


def normalize_fit_target(value: str | None) -> str | None:
    if value is None:
        return None
    pieces = value.split(",")
    if not pieces or any(not piece for piece in pieces):
        raise ControllerError(
            "INVALID_INPUT", "fit_target must contain positive integer MiB values", 2
        )
    return ",".join(str(positive_integer(piece, "fit_target")) for piece in pieces)


def ipv4_loopback(value: str, label: str) -> str:
    text = validate_text(value, label)
    try:
        address = ipaddress.ip_address(text)
    except ValueError as exc:
        raise ControllerError(
            "INVALID_INPUT", f"{label} must be an IPv4 loopback address", 2
        ) from exc
    if not isinstance(address, ipaddress.IPv4Address) or not address.is_loopback:
        raise ControllerError(
            "INVALID_INPUT", f"{label} must be an IPv4 loopback address", 2
        )
    return str(address)


def validate_inputs(namespace: argparse.Namespace) -> dict[str, Any]:
    port = positive_integer(namespace.port, "port", 65535)
    extra_args = list(namespace.extra_server_arg or [])
    if namespace.router:
        conflicting = []
        for option, value in (
            ("--model", namespace.model),
            ("--alias", namespace.alias),
            ("--ctx-size", namespace.ctx_size),
            ("--fit-target", namespace.fit_target),
        ):
            if value is not None:
                conflicting.append(option)
        if extra_args:
            conflicting.append("--extra-server-arg")
        if conflicting:
            raise ControllerError(
                "INVALID_INPUT",
                "router mode forbids single-model launch arguments",
                2,
                {"forbidden_arguments": conflicting},
            )
        if namespace.models_max is None:
            raise ControllerError(
                "INVALID_INPUT", "models_max is required in router mode", 2
            )
        return {
            "launch_mode": "router",
            "model": None,
            "alias": None,
            "host": ipv4_loopback(namespace.host, "host"),
            "port": port,
            "models_max": positive_integer(namespace.models_max, "models_max"),
            "context_size": None,
            "fit_target": None,
            "extra_server_args": [],
        }
    if namespace.models_max is not None:
        raise ControllerError(
            "INVALID_INPUT", "models_max is available only in router mode", 2
        )
    if namespace.model is None:
        raise ControllerError("INVALID_INPUT", "the following arguments are required: --model", 2)
    if namespace.alias is None:
        raise ControllerError("INVALID_INPUT", "the following arguments are required: --alias", 2)
    alias = validate_text(namespace.alias, "alias")
    host = validate_text(namespace.host, "host")
    context_size = (
        positive_integer(namespace.ctx_size, "ctx_size")
        if namespace.ctx_size is not None
        else None
    )
    fit_target = normalize_fit_target(namespace.fit_target)
    if extra_args:
        raise ControllerError(
            "EXTRA_ARGUMENT_NOT_ALLOWED",
            "no extra server argument is allowlisted by this controller build",
            2,
            {"extra_argument_count": len(extra_args)},
        )
    return {
        "launch_mode": "single-model",
        "model": namespace.model,
        "alias": alias,
        "host": host,
        "port": port,
        "context_size": context_size,
        "fit_target": fit_target,
        "extra_server_args": [],
        "models_max": None,
    }


def validate_model(model_value: str, paths: dict[str, Path]) -> dict[str, Any]:
    if not model_value:
        raise ControllerError("MODEL_PATH_MISSING", "model path is empty", 2)
    supplied = Path(model_value)
    candidate = supplied if supplied.is_absolute() else paths["model_root"] / supplied
    initial = lstat_or_none(candidate)
    if initial is None:
        raise ControllerError(
            "MODEL_PATH_MISSING",
            "model path does not exist",
            2,
            {"supplied_path": model_value},
        )
    if stat.S_ISLNK(initial.st_mode):
        raise ControllerError(
            "MODEL_SYMLINK",
            "model path itself must not be a symbolic link",
            2,
            {"supplied_path": model_value},
        )
    try:
        canonical = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ControllerError(
            "MODEL_PATH_MISSING", "model path could not be resolved", 2
        ) from exc
    try:
        canonical.relative_to(paths["model_root"])
    except ValueError as exc:
        raise ControllerError(
            "MODEL_OUTSIDE_ROOT",
            "model path resolves outside the branch model root",
            2,
            {"supplied_path": model_value},
        ) from exc

    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(canonical, flags)
    except FileNotFoundError as exc:
        raise ControllerError("MODEL_PATH_MISSING", "model file disappeared", 2) from exc
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ControllerError("MODEL_SYMLINK", "model final path is a symlink", 2) from exc
        raise ControllerError(
            "MODEL_NOT_REGULAR",
            "model file could not be opened safely",
            2,
            {"error_type": type(exc).__name__},
        ) from exc
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode):
            raise ControllerError(
                "MODEL_NOT_REGULAR", "model object must be a regular file", 2
            )
        magic = os.read(fd, 4)
    finally:
        os.close(fd)
    if magic != b"GGUF":
        raise ControllerError(
            "MODEL_NOT_GGUF",
            "model file does not begin with GGUF container magic",
            2,
        )
    current = canonical.stat()
    if (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino):
        raise ControllerError(
            "MODEL_NOT_REGULAR", "model path identity changed during validation", 2
        )
    return {
        "canonical_path": str(canonical),
        "device": opened.st_dev,
        "inode": opened.st_ino,
        "size": opened.st_size,
        "mtime_ns": opened.st_mtime_ns,
        "magic": "GGUF",
    }


def same_model_identity(left: dict[str, Any], right: dict[str, Any]) -> bool:
    fields = ("canonical_path", "device", "inode", "size", "mtime_ns", "magic")
    return all(left.get(field) == right.get(field) for field in fields)


def endpoint_probe(host: str, port: int) -> dict[str, Any]:
    try:
        candidates = socket.getaddrinfo(
            host, port, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM
        )
    except socket.gaierror as exc:
        raise ControllerError(
            "ENDPOINT_UNAVAILABLE",
            "endpoint host could not be resolved",
            3,
            {"host": host, "port": port, "socket_error": bounded_text(exc)},
        ) from exc
    unique: list[tuple[int, int, int, tuple[Any, ...]]] = []
    seen: set[tuple[int, int, int, tuple[Any, ...]]] = set()
    for family, socktype, protocol, _canonname, sockaddr in candidates:
        key = (family, socktype, protocol, sockaddr)
        if key not in seen:
            seen.add(key)
            unique.append(key)
    if not unique:
        raise ControllerError(
            "ENDPOINT_UNAVAILABLE",
            "endpoint host resolved to no stream addresses",
            3,
            {"host": host, "port": port},
        )

    resolved: list[str] = []
    for family, socktype, protocol, sockaddr in unique:
        probe = socket.socket(family, socktype, protocol)
        try:
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            probe.bind(sockaddr)
            resolved.append(str(sockaddr))
        except OSError as exc:
            reason = "ENDPOINT_IN_USE" if exc.errno == errno.EADDRINUSE else "ENDPOINT_UNAVAILABLE"
            message = (
                "endpoint is already in use"
                if reason == "ENDPOINT_IN_USE"
                else "endpoint cannot be bound by the current process"
            )
            raise ControllerError(
                reason,
                message,
                3,
                {
                    "host": host,
                    "port": port,
                    "socket_errno": exc.errno,
                    "resolved_address": str(sockaddr),
                },
            ) from exc
        finally:
            probe.close()
    return {"host": host, "port": port, "resolved_addresses": resolved}


def tcp_listener_inodes(host: str, port: int) -> set[str]:
    """Return IPv4 listener socket inodes for one exact endpoint."""

    inodes: set[str] = set()
    for row in Path("/proc/net/tcp").read_text(encoding="ascii").splitlines()[1:]:
        fields = row.split()
        if len(fields) < 10 or fields[3] != "0A":
            continue
        address_hex, port_hex = fields[1].split(":")
        address = socket.inet_ntoa(bytes.fromhex(address_hex)[::-1])
        if address == host and int(port_hex, 16) == port:
            inodes.add(fields[9])
    return inodes


def endpoint_listener_owners(host: str, port: int) -> list[dict[str, Any]]:
    """Correlate one endpoint with exact live PID/start/PGID/SID evidence."""

    candidates = tcp_listener_inodes(host, port)
    if not candidates:
        return []
    owners: list[dict[str, Any]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        snapshot = process_snapshot(pid)
        if snapshot is None or snapshot["state"] == "Z":
            continue
        try:
            descriptors = (entry / "fd").iterdir()
        except (FileNotFoundError, PermissionError, OSError):
            continue
        for descriptor in descriptors:
            try:
                target = os.readlink(descriptor)
            except OSError:
                continue
            if not target.startswith("socket:[") or not target.endswith("]"):
                continue
            inode = target[8:-1]
            if inode in candidates:
                owners.append(
                    {
                        "pid": pid,
                        "process_start_identity": snapshot[
                            "process_start_identity"
                        ],
                        "pgid": snapshot["pgid"],
                        "sid": snapshot["session_id"],
                        "socket_inode": inode,
                        "fd": int(descriptor.name),
                    }
                )
    return sorted(owners, key=lambda value: (value["pid"], value["fd"]))


def build_argv(
    paths: dict[str, Path],
    inputs: dict[str, Any],
    model: dict[str, Any] | None,
) -> list[str]:
    if inputs["launch_mode"] == "router":
        return [
            str(paths["binary_path"]),
            "--host",
            inputs["host"],
            "--port",
            str(inputs["port"]),
            "--models-dir",
            str(paths["models_dir"]),
            "--models-max",
            str(inputs["models_max"]),
            "--no-models-autoload",
            "--jinja",
            "--n-gpu-layers",
            "auto",
            "--fit",
            "on",
            "--offline",
            "--no-webui",
            "--no-agent",
            "--no-ui-mcp-proxy",
            "--cors-origins",
            "localhost",
            "--no-cors-credentials",
        ]
    if model is None:
        raise ValueError("single-model launch requires a validated model")
    argv = [
        str(paths["binary_path"]),
        "--model",
        model["canonical_path"],
        "--host",
        inputs["host"],
        "--port",
        str(inputs["port"]),
        "--alias",
        inputs["alias"],
    ]
    if inputs["context_size"] is not None:
        argv.extend(["--ctx-size", str(inputs["context_size"])])
    argv.extend(["--n-gpu-layers", "auto", "--fit", "on"])
    if inputs["fit_target"] is not None:
        argv.extend(["--fit-target", inputs["fit_target"]])
    return argv


def check_lock_absent(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise ControllerError(
            "BRANCH_LOCK_ACTIVE",
            "the branch active lock already exists",
            3,
            {"lock_path": str(path)},
        )


def process_snapshot(pid: int) -> dict[str, Any] | None:
    proc_root = Path("/proc") / str(pid)
    try:
        raw_stat = (proc_root / "stat").read_text(encoding="utf-8")
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    close = raw_stat.rfind(")")
    if close < 0:
        return None
    fields = raw_stat[close + 2 :].split()
    if len(fields) < 20:
        return None
    try:
        argv_raw = (proc_root / "cmdline").read_bytes()
        argv = [
            token.decode("utf-8", "surrogateescape")
            for token in argv_raw.split(b"\x00")
            if token
        ]
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        argv = []
    try:
        executable = str((proc_root / "exe").resolve(strict=True))
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        executable = ""
    return {
        "pid": pid,
        "state": fields[0],
        "pgid": int(fields[2]),
        "session_id": int(fields[3]),
        "process_start_identity": fields[19],
        "argv": argv,
        "proc_executable_path": executable,
    }


def normalized_live_argv_match(
    live_argv: list[str], recorded_argv: list[str], executable_path: str
) -> tuple[bool, str]:
    if live_argv == recorded_argv:
        return True, "direct"
    if len(live_argv) == len(recorded_argv) + 1 and live_argv[1:] == recorded_argv:
        try:
            script_path = str(Path(live_argv[1]).resolve(strict=True))
        except (FileNotFoundError, OSError):
            return False, "none"
        if script_path == executable_path:
            return True, "interpreter-wrapper"
    return False, "none"


def ownership_predicates(record: dict[str, Any]) -> dict[str, Any]:
    pid = record.get("pid")
    pgid = record.get("pgid")
    if not isinstance(pid, int) or pid <= 0 or not isinstance(pgid, int) or pgid <= 0:
        return {
            "process_alive": False,
            "positive_identity": False,
            "all_match": False,
        }
    snapshot = process_snapshot(pid)
    if snapshot is None:
        return {
            "process_alive": False,
            "positive_identity": True,
            "all_match": False,
        }
    process_alive = snapshot["state"] != "Z"
    start_match = (
        snapshot["process_start_identity"] == record.get("process_start_identity")
    )
    pgid_match = snapshot["pgid"] == pgid
    recorded_argv = record.get("argv")
    argv_match = False
    argv_mode = "none"
    if isinstance(recorded_argv, list) and all(isinstance(item, str) for item in recorded_argv):
        argv_match, argv_mode = normalized_live_argv_match(
            snapshot["argv"], recorded_argv, str(record.get("executable_path", ""))
        )

    executable_path = Path(str(record.get("executable_path", "")))
    path_identity_match = False
    executable_hash_match = False
    try:
        current = executable_path.stat()
        path_identity_match = (
            current.st_dev == record.get("executable_device")
            and current.st_ino == record.get("executable_inode")
        )
        executable_hash_match = (
            sha256_file(executable_path) == record.get("executable_sha256")
        )
    except (FileNotFoundError, OSError):
        pass
    proc_executable_match = snapshot["proc_executable_path"] == str(executable_path)
    interpreter_script_match = argv_mode == "interpreter-wrapper"
    executable_match = (
        path_identity_match
        and executable_hash_match
        and (proc_executable_match or interpreter_script_match)
    )
    all_match = (
        process_alive
        and start_match
        and pgid_match
        and argv_match
        and executable_match
    )
    return {
        "process_alive": process_alive,
        "positive_identity": True,
        "process_start_identity_match": start_match,
        "process_group_match": pgid_match,
        "argv_match": argv_match,
        "argv_match_mode": argv_mode,
        "path_device_inode_match": path_identity_match,
        "executable_sha256_match": executable_hash_match,
        "proc_executable_match": proc_executable_match,
        "interpreter_script_match": interpreter_script_match,
        "live_snapshot": snapshot,
        "all_match": all_match,
    }


def matching_recorded_processes(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Find live processes matching the complete recorded executable/argv."""

    recorded_argv = record.get("argv")
    executable_path = str(record.get("executable_path", ""))
    if (
        not isinstance(recorded_argv, list)
        or not recorded_argv
        or not all(isinstance(item, str) for item in recorded_argv)
        or not executable_path
    ):
        return []
    matches: list[dict[str, Any]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        snapshot = process_snapshot(int(entry.name))
        if snapshot is None or snapshot["state"] == "Z":
            continue
        argv_match, argv_mode = normalized_live_argv_match(
            snapshot["argv"], recorded_argv, executable_path
        )
        executable_match = (
            snapshot["proc_executable_path"] == executable_path
            or argv_mode == "interpreter-wrapper"
        )
        if argv_match and executable_match:
            matches.append(
                {
                    "pid": snapshot["pid"],
                    "process_start_identity": snapshot[
                        "process_start_identity"
                    ],
                    "pgid": snapshot["pgid"],
                    "sid": snapshot["session_id"],
                }
            )
    return sorted(matches, key=lambda value: value["pid"])


def group_members(pgid: int) -> list[dict[str, Any]]:
    members: list[dict[str, Any]] = []
    proc = Path("/proc")
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        snapshot = process_snapshot(int(entry.name))
        if snapshot is not None and snapshot["pgid"] == pgid and snapshot["state"] != "Z":
            members.append(
                {
                    "pid": snapshot["pid"],
                    "pgid": snapshot["pgid"],
                    "session_id": snapshot["session_id"],
                    "process_start_identity": snapshot["process_start_identity"],
                }
            )
    return sorted(members, key=lambda item: item["pid"])


def group_absent(pgid: int) -> bool:
    return not group_members(pgid)


def wait_for_group_absence(pgid: int, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if group_absent(pgid):
            return True
        time.sleep(POLL_INTERVAL_SECONDS)
    return group_absent(pgid)


def status_record(
    lifecycle_state: str,
    tx_id: str | None,
    pid: int | None,
    pgid: int | None,
    process_alive: bool,
    reason_code: str,
    message: str,
    sid: int | None = None,
    launch_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if lifecycle_state not in LIFECYCLE_STATES:
        raise ValueError("invalid lifecycle state")
    value = {
        "schema_version": SCHEMA_VERSION,
        "lifecycle_state": lifecycle_state,
        "transaction_id": tx_id,
        "pid": pid,
        "pgid": pgid,
        "sid": sid,
        "process_alive": process_alive,
        "readiness": READINESS,
        "reason_code": reason_code,
        "message": bounded_text(message),
        "last_transition_utc": utc_now(),
    }
    if launch_details:
        value.update(launch_details)
    return value


def router_launch_details(
    paths: dict[str, Path], inputs: dict[str, Any]
) -> dict[str, Any]:
    if inputs["launch_mode"] != "router":
        return {}
    return {
        "launch_mode": "router",
        "models_dir": str(paths["models_dir"]),
        "models_max": inputs["models_max"],
        "models_autoload": False,
        "jinja": True,
        "offline": True,
        "private_endpoint": {"host": inputs["host"], "port": inputs["port"]},
        "router_cache": str(paths["router_cache"]),
    }


def recorded_launch_details(record: dict[str, Any]) -> dict[str, Any]:
    if record.get("launch_mode") != "router":
        return {}
    return {
        key: record.get(key)
        for key in (
            "launch_mode",
            "models_dir",
            "models_max",
            "models_autoload",
            "jinja",
            "offline",
            "private_endpoint",
            "router_cache",
        )
    }


def append_history(
    transaction: dict[str, Any], lifecycle_state: str, reason_code: str, message: str
) -> None:
    transaction.setdefault("operation_history", []).append(
        {
            "timestamp_utc": utc_now(),
            "lifecycle_state": lifecycle_state,
            "reason_code": reason_code,
            "message": bounded_text(message),
        }
    )
    transaction["updated_at_utc"] = utc_now()
    transaction["final_lifecycle_state"] = lifecycle_state
    transaction["final_reason_code"] = reason_code


def plan_operation(namespace: argparse.Namespace, paths: dict[str, Path]) -> dict[str, Any]:
    checked = validate_layout(paths)
    inputs = validate_inputs(namespace)
    model = None
    if inputs["launch_mode"] == "router":
        checked = validate_router_layout(checked)
    else:
        model = validate_model(inputs["model"], checked)
    state_paths = runtime_paths(checked)
    check_lock_absent(state_paths["lock_path"])
    endpoint_probe(inputs["host"], inputs["port"])
    argv = build_argv(checked, inputs, model)
    result = {
        "resolved_paths": {
            "controller_dir": str(checked["controller_dir"]),
            "branch_root": str(checked["branch_root"]),
            "binary_path": str(checked["binary_path"]),
            "model_root": str(checked["model_root"]),
            "runtime_root": str(checked["runtime_root"]),
        },
        "model_identity": model,
        "endpoint": {"host": inputs["host"], "port": inputs["port"]},
        "argv": argv,
        "runtime_paths": {
            "lock_path": str(state_paths["lock_path"]),
            "pid_path": str(state_paths["pid_path"]),
            "status_path": str(state_paths["status_path"]),
            "prospective_transaction_parent": str(state_paths["transaction_parent"]),
            "prospective_log_parent": str(state_paths["log_parent"]),
        },
        "readiness": READINESS,
    }
    if inputs["launch_mode"] == "router":
        result.update(router_launch_details(checked, inputs))
        result["environment_overrides"] = {
            "LLAMA_CACHE": str(checked["router_cache"])
        }
    return result


def start_operation(namespace: argparse.Namespace, paths: dict[str, Path]) -> dict[str, Any]:
    checked = validate_layout(paths)
    inputs = validate_inputs(namespace)
    initial_model = None
    if inputs["launch_mode"] == "router":
        checked = validate_router_layout(checked)
    else:
        initial_model = validate_model(inputs["model"], checked)
    state_paths = runtime_paths(checked)
    check_lock_absent(state_paths["lock_path"])
    endpoint_probe(inputs["host"], inputs["port"])

    tx_id = transaction_id()
    state_paths = runtime_paths(checked, tx_id)
    lock_value = {
        "schema_version": SCHEMA_VERSION,
        "transaction_id": tx_id,
        "created_at_utc": utc_now(),
        "controller_pid": os.getpid(),
        "controller_process_start_identity": (
            process_snapshot(os.getpid()) or {}
        ).get("process_start_identity"),
        "branch_root": str(checked["branch_root"]),
        "pid_record_path": str(state_paths["pid_path"]),
        "status_record_path": str(state_paths["status_path"]),
        "transaction_record_path": str(state_paths["transaction_path"]),
        "log_path": str(state_paths["log_path"]),
    }
    exclusive_json_create(state_paths["lock_path"], lock_value)
    lock_owned = True
    spawned: subprocess.Popen[bytes] | None = None
    active_record_written = False

    try:
        under_lock = validate_layout(paths)
        current_model = None
        if inputs["launch_mode"] == "router":
            under_lock = validate_router_layout(under_lock)
        else:
            current_model = validate_model(inputs["model"], under_lock)
            if initial_model is None or not same_model_identity(initial_model, current_model):
                raise ControllerError(
                    "MODEL_NOT_REGULAR", "model identity changed before process creation", 2
                )
        endpoint_probe(inputs["host"], inputs["port"])
        binary = binary_identity(under_lock["binary_path"])
        argv = build_argv(under_lock, inputs, current_model)
        launch_details = router_launch_details(under_lock, inputs)

        log_flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_CLOEXEC
            | getattr(os, "O_NOFOLLOW", 0)
        )
        log_fd = os.open(state_paths["log_path"], log_flags, 0o600)
        transaction = {
            "schema_version": SCHEMA_VERSION,
            "transaction_id": tx_id,
            "created_at_utc": utc_now(),
            "updated_at_utc": utc_now(),
            "operation_history": [],
            "launch_mode": inputs["launch_mode"],
            "validated_input": dict(inputs),
            "resolved_paths": {
                "controller_dir": str(under_lock["controller_dir"]),
                "branch_root": str(under_lock["branch_root"]),
                "binary_path": str(under_lock["binary_path"]),
                "model_root": str(under_lock["model_root"]),
                "runtime_root": str(under_lock["runtime_root"]),
            },
            "model_identity": current_model,
            "binary_identity": binary,
            "endpoint": {"host": inputs["host"], "port": inputs["port"]},
            "argv": argv,
            "process_identity": None,
            "log_path": str(state_paths["log_path"]),
            "final_lifecycle_state": "PREPARING",
            "final_reason_code": "OK",
            "cleanup_record": {},
        }
        transaction.update(launch_details)
        append_history(transaction, "PREPARING", "OK", "validated launch preparation")
        atomic_json_write(state_paths["transaction_path"], transaction)
        atomic_json_write(
            state_paths["status_path"],
            status_record(
                "PREPARING",
                tx_id,
                None,
                None,
                False,
                "OK",
                "validated launch preparation",
                launch_details=launch_details,
            ),
        )
        append_history(transaction, "STARTING", "OK", "creating owned process group")
        atomic_json_write(state_paths["transaction_path"], transaction)
        atomic_json_write(
            state_paths["status_path"],
            status_record(
                "STARTING",
                tx_id,
                None,
                None,
                False,
                "OK",
                "creating owned process group",
                launch_details=launch_details,
            ),
        )
        child_environment = None
        if inputs["launch_mode"] == "router":
            child_environment = dict(os.environ)
            child_environment["LLAMA_CACHE"] = str(under_lock["router_cache"])
        try:
            spawned = subprocess.Popen(
                argv,
                cwd=under_lock["branch_root"],
                stdin=subprocess.DEVNULL,
                stdout=log_fd,
                stderr=subprocess.STDOUT,
                close_fds=True,
                start_new_session=True,
                shell=False,
                env=child_environment,
            )
        except OSError as exc:
            os.close(log_fd)
            append_history(
                transaction, "START_FAILED", "SPAWN_FAILED", type(exc).__name__
            )
            transaction["cleanup_record"] = {"lock_removed": True}
            atomic_json_write(state_paths["transaction_path"], transaction)
            atomic_json_write(
                state_paths["status_path"],
                status_record(
                    "START_FAILED",
                    tx_id,
                    None,
                    None,
                    False,
                    "SPAWN_FAILED",
                    "child process creation failed",
                    launch_details=launch_details,
                ),
            )
            remove_owned_record(state_paths["lock_path"], tx_id)
            lock_owned = False
            raise ControllerError(
                "SPAWN_FAILED",
                "child process creation failed",
                4,
                {
                    "transaction_id": tx_id,
                    "error_type": type(exc).__name__,
                    "transaction_record_path": str(state_paths["transaction_path"]),
                    "log_path": str(state_paths["log_path"]),
                },
            ) from exc
        else:
            os.close(log_fd)

        deadline = time.monotonic() + 0.8
        snapshot = process_snapshot(spawned.pid)
        while snapshot is None and time.monotonic() < deadline:
            time.sleep(0.02)
            snapshot = process_snapshot(spawned.pid)
        if snapshot is None:
            raise ControllerError(
                "SPAWN_FAILED", "child process metadata was unavailable", 4
            )
        pid = spawned.pid
        pgid = os.getpgid(pid)
        if pgid != pid:
            raise ControllerError(
                "SPAWN_FAILED", "child did not become its process-group leader", 4
            )
        sid = snapshot["session_id"]
        if sid != pid:
            raise ControllerError(
                "SPAWN_FAILED", "child did not become its session leader", 4
            )
        active_record = {
            "schema_version": SCHEMA_VERSION,
            "transaction_id": tx_id,
            "pid": pid,
            "pgid": pgid,
            "sid": sid,
            "process_start_identity": snapshot["process_start_identity"],
            "executable_path": str(under_lock["binary_path"]),
            "executable_device": binary["device"],
            "executable_inode": binary["inode"],
            "executable_sha256": binary["sha256"],
            "argv": argv,
            "cwd": str(under_lock["branch_root"]),
            "environment_mode": "inherit",
            "environment_overrides": (
                {"LLAMA_CACHE": str(under_lock["router_cache"])}
                if inputs["launch_mode"] == "router"
                else {}
            ),
            "endpoint_host": inputs["host"],
            "endpoint_port": inputs["port"],
            "log_path": str(state_paths["log_path"]),
            "status_path": str(state_paths["status_path"]),
            "transaction_path": str(state_paths["transaction_path"]),
            "created_at_utc": utc_now(),
            "readiness": READINESS,
        }
        if current_model is not None:
            active_record.update(
                {
                    "model_path": current_model["canonical_path"],
                    "model_device": current_model["device"],
                    "model_inode": current_model["inode"],
                    "model_size": current_model["size"],
                    "model_mtime_ns": current_model["mtime_ns"],
                }
            )
        active_record.update(launch_details)
        atomic_json_write(state_paths["pid_path"], active_record)
        active_record_written = True
        transaction["process_identity"] = {
            "pid": pid,
            "pgid": pgid,
            "sid": sid,
            "process_start_identity": snapshot["process_start_identity"],
            "executable_path": str(under_lock["binary_path"]),
            "argv": argv,
        }
        atomic_json_write(state_paths["transaction_path"], transaction)
        atomic_json_write(
            state_paths["status_path"],
            status_record(
                "STARTING",
                tx_id,
                pid,
                pgid,
                True,
                "OK",
                "immediate-exit window",
                sid=sid,
                launch_details=launch_details,
            ),
        )

        try:
            exit_status = spawned.wait(timeout=IMMEDIATE_EXIT_WINDOW_SECONDS)
        except subprocess.TimeoutExpired:
            exit_status = None
        if exit_status is not None:
            remove_owned_record(state_paths["pid_path"], tx_id)
            active_record_written = False
            remove_owned_record(state_paths["lock_path"], tx_id)
            lock_owned = False
            append_history(
                transaction,
                "START_FAILED",
                "PROCESS_EXITED_EARLY",
                f"child exited with status {exit_status}",
            )
            transaction["cleanup_record"] = {
                "active_pid_record_removed": True,
                "active_lock_removed": True,
            }
            atomic_json_write(state_paths["transaction_path"], transaction)
            atomic_json_write(
                state_paths["status_path"],
                status_record(
                    "START_FAILED",
                    tx_id,
                    pid,
                    pgid,
                    False,
                    "PROCESS_EXITED_EARLY",
                    f"child exited with status {exit_status}",
                    sid=sid,
                    launch_details=launch_details,
                ),
            )
            raise ControllerError(
                "PROCESS_EXITED_EARLY",
                "child exited during the immediate-exit confirmation window",
                4,
                {
                    "transaction_id": tx_id,
                    "pid": pid,
                    "pgid": pgid,
                    "child_exit_status": exit_status,
                    "transaction_record_path": str(state_paths["transaction_path"]),
                    "log_path": str(state_paths["log_path"]),
                },
            )

        append_history(transaction, "STARTED", "OK", "owned process group started")
        atomic_json_write(state_paths["transaction_path"], transaction)
        atomic_json_write(
            state_paths["status_path"],
            status_record(
                "STARTED",
                tx_id,
                pid,
                pgid,
                True,
                "OK",
                "owned process group started",
                sid=sid,
                launch_details=launch_details,
            ),
        )
        return {
            "transaction_id": tx_id,
            "pid": pid,
            "pgid": pgid,
            "sid": sid,
            "process_start_identity": snapshot["process_start_identity"],
            "executable_path": str(under_lock["binary_path"]),
            "argv": argv,
            "log_path": str(state_paths["log_path"]),
            "pid_record_path": str(state_paths["pid_path"]),
            "status_record_path": str(state_paths["status_path"]),
            "transaction_record_path": str(state_paths["transaction_path"]),
            "endpoint": {"host": inputs["host"], "port": inputs["port"]},
            "readiness": READINESS,
            **launch_details,
        }
    except ControllerError:
        if spawned is None:
            if lock_owned:
                remove_owned_record(state_paths["lock_path"], tx_id)
        elif spawned.poll() is None and not active_record_written:
            snapshot = process_snapshot(spawned.pid)
            if snapshot is not None and snapshot["pgid"] == spawned.pid:
                try:
                    os.killpg(spawned.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        raise


def status_operation(paths: dict[str, Path]) -> dict[str, Any]:
    checked = validate_layout(paths)
    state_paths = runtime_paths(checked)
    lock_present = state_paths["lock_path"].exists() or state_paths["lock_path"].is_symlink()
    pid_present = state_paths["pid_path"].exists() or state_paths["pid_path"].is_symlink()
    status_value: dict[str, Any] | None = None
    if state_paths["status_path"].exists():
        status_value = read_json_record(state_paths["status_path"], "status record")
    if not lock_present and not pid_present:
        lifecycle = "STOPPED"
        if status_value is not None:
            lifecycle = str(status_value.get("lifecycle_state", "STOPPED"))
            if lifecycle not in LIFECYCLE_STATES:
                lifecycle = "INCONSISTENT"
        return {
            "active": False,
            "lifecycle_state": lifecycle,
            "active_state_consistent": lifecycle not in {"STARTING", "STARTED", "STOPPING"},
            "transaction_id": None,
            "pid": None,
            "pgid": None,
            "process_alive": False,
            "ownership_predicates": {},
            "endpoint": None,
            "readiness": READINESS,
            "status_record_path": str(state_paths["status_path"]),
            "pid_record_path": str(state_paths["pid_path"]),
            "lock_path": str(state_paths["lock_path"]),
        }
    if not lock_present or not pid_present:
        raise ControllerError(
            "ACTIVE_STATE_INCONSISTENT",
            "active lock and PID record presence disagree",
            3,
            {
                "lock_present": lock_present,
                "pid_record_present": pid_present,
                "lock_path": str(state_paths["lock_path"]),
                "pid_record_path": str(state_paths["pid_path"]),
            },
        )
    lock_value = read_json_record(state_paths["lock_path"], "active lock")
    pid_value = read_json_record(state_paths["pid_path"], "active PID record")
    if lock_value.get("transaction_id") != pid_value.get("transaction_id"):
        raise ControllerError(
            "ACTIVE_STATE_INCONSISTENT",
            "active lock and PID transaction identities disagree",
            3,
        )
    predicates = ownership_predicates(pid_value)
    endpoint_host = pid_value.get("endpoint_host")
    endpoint_port = pid_value.get("endpoint_port")
    listener_owners = (
        endpoint_listener_owners(str(endpoint_host), int(endpoint_port))
        if isinstance(endpoint_host, str)
        and isinstance(endpoint_port, int)
        else []
    )
    listener_owned = bool(
        predicates.get("all_match")
        and any(
            owner.get("pgid") == pid_value.get("pgid")
            for owner in listener_owners
        )
    )
    data = {
        "active": bool(predicates.get("all_match")),
        "lifecycle_state": (
            status_value.get("lifecycle_state", "INCONSISTENT")
            if status_value
            else "INCONSISTENT"
        ),
        "active_state_consistent": bool(
            predicates.get("all_match") and listener_owned
        ),
        "transaction_id": pid_value.get("transaction_id"),
        "pid": pid_value.get("pid"),
        "pgid": pid_value.get("pgid"),
        "sid": pid_value.get("sid"),
        "process_alive": bool(predicates.get("process_alive")),
        "ownership_predicates": predicates,
        "listener_owned": listener_owned,
        "listener_owners": listener_owners,
        "endpoint": {
            "host": pid_value.get("endpoint_host"),
            "port": pid_value.get("endpoint_port"),
        },
        "readiness": READINESS,
        "status_record_path": str(state_paths["status_path"]),
        "pid_record_path": str(state_paths["pid_path"]),
        "lock_path": str(state_paths["lock_path"]),
        **recorded_launch_details(pid_value),
    }
    if not predicates.get("all_match"):
        raise ControllerError(
            "ACTIVE_STATE_INCONSISTENT",
            "recorded active process identity does not match the live process",
            3,
            data,
        )
    if not listener_owned:
        foreign = [
            owner
            for owner in listener_owners
            if owner.get("pgid") != pid_value.get("pgid")
        ]
        raise ControllerError(
            "ENDPOINT_CONFLICT" if foreign else "PRIVATE_LISTENER_LOST",
            (
                "private endpoint is owned by a foreign process"
                if foreign
                else "owned router process is alive but its listener is absent"
            ),
            3,
            data,
        )
    return data


def reconcile_operation(paths: dict[str, Path]) -> dict[str, Any]:
    """Conservatively reconcile only controller-owned stale active records."""

    checked = validate_layout(paths)
    state_paths = runtime_paths(checked)
    lock_present = (
        state_paths["lock_path"].exists()
        or state_paths["lock_path"].is_symlink()
    )
    pid_present = (
        state_paths["pid_path"].exists()
        or state_paths["pid_path"].is_symlink()
    )
    if not lock_present and not pid_present:
        return {
            "active": False,
            "active_state_consistent": True,
            "reconciled": False,
            "reason_code": "OK",
            "removed_records": [],
        }

    lock_value = (
        read_json_record(state_paths["lock_path"], "active lock")
        if lock_present
        else None
    )
    pid_value = (
        read_json_record(state_paths["pid_path"], "active PID record")
        if pid_present
        else None
    )
    for record in (lock_value, pid_value):
        if record is not None and record.get("schema_version") != SCHEMA_VERSION:
            raise ControllerError(
                "OWNERSHIP_UNCERTAIN",
                "active record controller identity is invalid",
                3,
            )
    transaction_ids = {
        str(record.get("transaction_id"))
        for record in (lock_value, pid_value)
        if record is not None
    }
    if len(transaction_ids) != 1 or "None" in transaction_ids:
        raise ControllerError(
            "OWNERSHIP_UNCERTAIN",
            "active record transaction identities are ambiguous",
            3,
        )
    tx_id = next(iter(transaction_ids))
    transaction_path = state_paths["transaction_parent"] / f"{tx_id}.json"
    transaction = (
        read_json_record(transaction_path, "transaction record")
        if transaction_path.exists()
        else None
    )
    if transaction is not None and (
        transaction.get("schema_version") != SCHEMA_VERSION
        or transaction.get("transaction_id") != tx_id
    ):
        raise ControllerError(
            "OWNERSHIP_UNCERTAIN",
            "transaction history does not validate for active records",
            3,
        )

    record = pid_value
    if record is None:
        if transaction is None:
            raise ControllerError(
                "OWNERSHIP_UNCERTAIN",
                "partial active state lacks a validating transaction record",
                3,
            )
        resolved = transaction.get("resolved_paths")
        endpoint = transaction.get("endpoint")
        argv = transaction.get("argv")
        if (
            not isinstance(resolved, dict)
            or not isinstance(endpoint, dict)
            or not isinstance(argv, list)
        ):
            raise ControllerError(
                "OWNERSHIP_UNCERTAIN",
                "partial transaction lacks executable/argv/endpoint evidence",
                3,
            )
        record = {
            "executable_path": resolved.get("binary_path"),
            "argv": argv,
            "endpoint_host": endpoint.get("host"),
            "endpoint_port": endpoint.get("port"),
        }

    endpoint_host = record.get("endpoint_host")
    endpoint_port = record.get("endpoint_port")
    if not isinstance(endpoint_host, str) or not isinstance(endpoint_port, int):
        raise ControllerError(
            "OWNERSHIP_UNCERTAIN",
            "active state lacks one exact endpoint",
            3,
        )
    owners = endpoint_listener_owners(endpoint_host, endpoint_port)

    if pid_value is not None:
        predicates = ownership_predicates(pid_value)
        if predicates.get("all_match"):
            listener_owned = any(
                owner.get("pgid") == pid_value.get("pgid")
                for owner in owners
            )
            if listener_owned:
                return {
                    "active": True,
                    "active_state_consistent": True,
                    "reconciled": False,
                    "reason_code": "OK",
                    "listener_owned": True,
                }
            if owners:
                raise ControllerError(
                    "ENDPOINT_CONFLICT",
                    "private endpoint is owned by a foreign process",
                    3,
                    {
                        "foreign_endpoint_owners": owners,
                        "unrelated_process_signaled": False,
                    },
                )
            return {
                "active": True,
                "active_state_consistent": False,
                "reconciled": False,
                "reason_code": "PRIVATE_LISTENER_LOST",
                "listener_owned": False,
                "selected_action": "CONTROLLER_STOP_RESTART",
                "transaction_id": pid_value.get("transaction_id"),
                "pid": pid_value.get("pid"),
                "pgid": pid_value.get("pgid"),
                "sid": pid_value.get("sid"),
                "process_start_identity": pid_value.get(
                    "process_start_identity"
                ),
            }
        if predicates.get("process_alive"):
            raise ControllerError(
                "OWNERSHIP_UNCERTAIN",
                "recorded PID is alive but exact ownership does not match",
                3,
                {"ownership_predicates": predicates},
            )

    matches = matching_recorded_processes(record)
    if matches:
        raise ControllerError(
            "OWNERSHIP_UNCERTAIN",
            "a complete executable/argv match remains alive under another identity",
            3,
            {"matching_processes": matches},
        )
    if owners:
        raise ControllerError(
            "ENDPOINT_CONFLICT",
            "private endpoint has a foreign live owner",
            3,
            {
                "foreign_endpoint_owners": owners,
                "unrelated_process_signaled": False,
            },
        )

    removed: list[str] = []
    if pid_present and remove_owned_record(state_paths["pid_path"], tx_id):
        removed.append("active_pid")
    if lock_present and remove_owned_record(state_paths["lock_path"], tx_id):
        removed.append("active_lock")
    if transaction is not None:
        append_history(
            transaction,
            "RECONCILED",
            "ROUTER_STATE_STALE",
            "stale controller-owned active records reconciled",
        )
        transaction.setdefault("cleanup_record", {}).update(
            {
                "reconciled": True,
                "removed_records": removed,
                "process_absent": True,
                "listener_absent": True,
                "unrelated_process_signaled": False,
            }
        )
        atomic_json_write(transaction_path, transaction)
    atomic_json_write(
        state_paths["status_path"],
        status_record(
            "RECONCILED",
            tx_id,
            pid_value.get("pid") if pid_value else None,
            pid_value.get("pgid") if pid_value else None,
            False,
            "ROUTER_STATE_STALE",
            "stale controller-owned active records reconciled",
            sid=pid_value.get("sid") if pid_value else None,
            launch_details=(
                recorded_launch_details(pid_value) if pid_value else None
            ),
        ),
    )
    return {
        "active": False,
        "active_state_consistent": True,
        "reconciled": True,
        "reason_code": "ROUTER_STATE_STALE",
        "transaction_id": tx_id,
        "removed_records": removed,
        "process_absent": True,
        "listener_absent": True,
    }


def stop_operation(paths: dict[str, Path]) -> dict[str, Any]:
    checked = validate_layout(paths)
    state_paths = runtime_paths(checked)
    lock_present = state_paths["lock_path"].exists() or state_paths["lock_path"].is_symlink()
    pid_present = state_paths["pid_path"].exists() or state_paths["pid_path"].is_symlink()
    if not lock_present and not pid_present:
        raise ControllerError(
            "NO_ACTIVE_PROCESS",
            "no controller-owned active process is recorded",
            3,
            {
                "lock_path": str(state_paths["lock_path"]),
                "pid_record_path": str(state_paths["pid_path"]),
            },
        )
    if not lock_present or not pid_present:
        raise ControllerError(
            "ACTIVE_STATE_INCONSISTENT",
            "active lock and PID record presence disagree",
            3,
        )
    lock_value = read_json_record(state_paths["lock_path"], "active lock")
    pid_value = read_json_record(state_paths["pid_path"], "active PID record")
    tx_id = lock_value.get("transaction_id")
    if not isinstance(tx_id, str) or tx_id != pid_value.get("transaction_id"):
        raise ControllerError(
            "ACTIVE_STATE_INCONSISTENT",
            "active lock and PID transaction identities disagree",
            3,
        )
    predicates = ownership_predicates(pid_value)
    if not predicates.get("all_match"):
        raise ControllerError(
            "OWNERSHIP_MISMATCH",
            "recorded process ownership predicates do not all match",
            3,
            {"ownership_predicates": predicates},
        )
    pid = pid_value["pid"]
    pgid = pid_value["pgid"]
    sid = pid_value.get("sid")
    launch_details = recorded_launch_details(pid_value)
    transaction_path = Path(str(pid_value.get("transaction_path", "")))
    log_path = Path(str(pid_value.get("log_path", "")))
    if transaction_path.parent != state_paths["transaction_parent"]:
        raise ControllerError(
            "ACTIVE_STATE_INCONSISTENT", "transaction record path is outside runtime", 3
        )
    transaction = read_json_record(transaction_path, "transaction record")
    if transaction.get("transaction_id") != tx_id:
        raise ControllerError(
            "ACTIVE_STATE_INCONSISTENT", "transaction record identity disagrees", 3
        )

    append_history(transaction, "STOPPING", "OK", "sending SIGTERM to owned group")
    atomic_json_write(transaction_path, transaction)
    atomic_json_write(
        state_paths["status_path"],
        status_record(
            "STOPPING",
            tx_id,
            pid,
            pgid,
            True,
            "OK",
            "stopping owned process group",
            sid=sid,
            launch_details=launch_details,
        ),
    )
    members_before = group_members(pgid)
    graceful_sent = False
    forced_sent = False
    try:
        os.killpg(pgid, signal.SIGTERM)
        graceful_sent = True
    except ProcessLookupError:
        pass
    absent = wait_for_group_absence(pgid, GRACEFUL_WAIT_SECONDS)
    if not absent:
        members_after = group_members(pgid)
        prior_identities = {
            (item["pid"], item["process_start_identity"]) for item in members_before
        }
        later_identities = {
            (item["pid"], item["process_start_identity"]) for item in members_after
        }
        continuity = bool(later_identities) and later_identities.issubset(prior_identities)
        same_session = all(item["session_id"] == pgid for item in members_after)
        if not continuity or not same_session:
            append_history(
                transaction,
                "STOP_FAILED",
                "OWNERSHIP_MISMATCH",
                "owned group continuity could not be revalidated",
            )
            atomic_json_write(transaction_path, transaction)
            atomic_json_write(
                state_paths["status_path"],
                status_record(
                    "STOP_FAILED",
                    tx_id,
                    pid,
                    pgid,
                    True,
                    "OWNERSHIP_MISMATCH",
                    "owned group continuity could not be revalidated",
                ),
            )
            raise ControllerError(
                "OWNERSHIP_MISMATCH",
                "owned group continuity could not be revalidated before escalation",
                3,
                {"members_before": members_before, "members_after": members_after},
            )
        try:
            os.killpg(pgid, signal.SIGKILL)
            forced_sent = True
        except ProcessLookupError:
            pass
        absent = wait_for_group_absence(pgid, FORCED_WAIT_SECONDS)
    if not absent:
        append_history(
            transaction, "STOP_FAILED", "STOP_TIMEOUT", "owned group remains present"
        )
        atomic_json_write(transaction_path, transaction)
        atomic_json_write(
            state_paths["status_path"],
            status_record(
                "STOP_FAILED",
                tx_id,
                pid,
                pgid,
                True,
                "STOP_TIMEOUT",
                "owned group remains present after bounded shutdown",
            ),
        )
        raise ControllerError(
            "STOP_TIMEOUT",
            "owned process group remains after bounded shutdown",
            4,
            {"pgid": pgid, "members": group_members(pgid)},
        )

    endpoint_observation: dict[str, Any]
    try:
        endpoint_observation = {
            "available_after_stop": True,
            **endpoint_probe(pid_value["endpoint_host"], pid_value["endpoint_port"]),
        }
    except ControllerError as exc:
        endpoint_observation = {
            "available_after_stop": False,
            "reason_code": exc.reason_code,
            "details": exc.data,
            "unrelated_endpoint_was_not_signaled": True,
        }
    pid_removed = remove_owned_record(state_paths["pid_path"], tx_id)
    lock_removed = remove_owned_record(state_paths["lock_path"], tx_id)
    if not pid_removed or not lock_removed:
        append_history(
            transaction,
            "STOP_FAILED",
            "ACTIVE_STATE_INCONSISTENT",
            "active records changed ownership before cleanup",
        )
        atomic_json_write(transaction_path, transaction)
        atomic_json_write(
            state_paths["status_path"],
            status_record(
                "STOP_FAILED",
                tx_id,
                pid,
                pgid,
                False,
                "ACTIVE_STATE_INCONSISTENT",
                "active records changed ownership before cleanup",
                sid=sid,
                launch_details=launch_details,
            ),
        )
        raise ControllerError(
            "ACTIVE_STATE_INCONSISTENT",
            "active records changed ownership before cleanup",
            4,
        )
    append_history(transaction, "STOPPED", "OK", "owned process group stopped")
    transaction["cleanup_record"] = {
        "graceful_signal_sent": graceful_sent,
        "forced_signal_sent": forced_sent,
        "owned_group_absent": True,
        "active_pid_record_removed": pid_removed,
        "active_lock_removed": lock_removed,
        "endpoint_observation": endpoint_observation,
    }
    atomic_json_write(transaction_path, transaction)
    atomic_json_write(
        state_paths["status_path"],
        status_record(
            "STOPPED",
            tx_id,
            pid,
            pgid,
            False,
            "OK",
            "owned process group stopped",
            sid=sid,
            launch_details=launch_details,
        ),
    )
    return {
        "transaction_id": tx_id,
        "stopped_pid": pid,
        "stopped_pgid": pgid,
        "graceful_signal_sent": graceful_sent,
        "forced_signal_sent": forced_sent,
        "owned_group_absent": True,
        "active_pid_record_removed": pid_removed,
        "active_lock_removed": lock_removed,
        "endpoint_observation": endpoint_observation,
        "status_record_path": str(state_paths["status_path"]),
        "transaction_record_path": str(transaction_path),
        "log_path": str(log_path),
    }


def build_parser() -> JsonArgumentParser:
    parser = JsonArgumentParser(prog="controller.py", add_help=False)
    operations = parser.add_subparsers(dest="operation", required=True)
    for name in ("plan", "start"):
        operation = operations.add_parser(name, add_help=False)
        operation.add_argument("--router", action="store_true")
        operation.add_argument("--model")
        operation.add_argument("--alias")
        operation.add_argument("--host", required=True)
        operation.add_argument("--port", required=True)
        operation.add_argument("--models-max")
        operation.add_argument("--ctx-size")
        operation.add_argument("--fit-target")
        operation.add_argument("--extra-server-arg", action="append", default=[])
    operations.add_parser("stop", add_help=False)
    operations.add_parser("status", add_help=False)
    operations.add_parser("reconcile", add_help=False)
    return parser


def guessed_operation(arguments: list[str]) -> str:
    if arguments and arguments[0] in {
        "plan",
        "start",
        "stop",
        "status",
        "reconcile",
    }:
        return arguments[0]
    return "status"


def main(arguments: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if arguments is None else arguments)
    operation = guessed_operation(argv)
    try:
        paths = derive_paths()
        branch_root = paths["branch_root"]
    except Exception:
        branch_root = Path(__file__).absolute().parent.parent
    try:
        namespace = build_parser().parse_args(argv)
        operation = namespace.operation
        if operation == "plan":
            data = plan_operation(namespace, paths)
            message = "validated launch plan constructed"
        elif operation == "start":
            data = start_operation(namespace, paths)
            message = "owned process group started; readiness not checked"
        elif operation == "stop":
            data = stop_operation(paths)
            message = "owned process group stopped"
        elif operation == "reconcile":
            data = reconcile_operation(paths)
            message = "branch active state reconciled conservatively"
        else:
            data = status_operation(paths)
            message = "branch status inspected"
        emit(result_envelope(operation, True, "OK", message, branch_root, data))
        return 0
    except ControllerError as exc:
        emit(
            result_envelope(
                operation,
                False,
                exc.reason_code,
                exc.message,
                branch_root,
                exc.data,
            )
        )
        return exc.exit_status
    except Exception as exc:
        emit(
            result_envelope(
                operation,
                False,
                "INTERNAL_ERROR",
                f"internal controller error: {type(exc).__name__}",
                branch_root,
                {"error_type": type(exc).__name__},
            )
        )
        return 5


if __name__ == "__main__":
    raise SystemExit(main())
