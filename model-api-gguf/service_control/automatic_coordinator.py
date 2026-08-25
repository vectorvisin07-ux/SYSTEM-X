"""Service-owned bounded coordinator for zero-argument Inspector intake."""

from __future__ import annotations

import datetime as dt
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
from dataclasses import dataclass
from typing import Any, Callable, Mapping


STATUS_SCHEMA = "system-x.service-automatic-intake-coordinator-status.v1"
INVOCATION_SCHEMA = "system-x.service-automatic-intake-coordinator-invocation.v1"
COORDINATOR_STATES = frozenset(
    ("DISABLED", "IDLE", "DUE", "CHILD_STARTING", "CHILD_RUNNING", "REATTACHING", "BACKOFF", "STOPPING")
)
AUTOMATIC_ACTIONS = frozenset(
    (
        "NOOP_WAITING", "NOOP_COPY_IN_PROGRESS", "NOOP_MULTIPLE_CANDIDATES",
        "NOOP_ACTIVE_TRANSACTION", "NOOP_ALREADY_PROCESSED", "NOOP_READY_MODEL_PRESENT",
        "NOOP_REGISTRY_CONTRADICTORY", "NOOP_OWNERSHIP_UNCERTAIN", "REJECT_CANDIDATE",
        "DISPATCH_FIRST_MODEL",
    )
)
NOOP_ACTIONS = frozenset(item for item in AUTOMATIC_ACTIONS if item.startswith("NOOP_"))
DEFAULT_INTERVAL_SECONDS = 5.0
DEFAULT_MAXIMUM_BACKOFF_SECONDS = 60.0
DEFAULT_CAPTURE_BYTES = 1_048_576


class CoordinatorError(RuntimeError):
    def __init__(self, reason_code: str, message: str):
        super().__init__(message)
        self.reason_code = reason_code
        self.message = message


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _identity(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _reject_symlink_components(path: Path, label: str) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    for component in list(reversed(absolute.parents)) + [absolute]:
        try:
            details = component.lstat()
        except FileNotFoundError:
            continue
        except OSError as error:
            raise CoordinatorError("COORDINATOR_PATH_INVALID", f"{label} cannot be inspected") from error
        if stat.S_ISLNK(details.st_mode):
            raise CoordinatorError("COORDINATOR_PATH_INVALID", f"{label} contains a symlink")


def _ensure_directory(path: Path, label: str) -> None:
    _reject_symlink_components(path, label)
    try:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        details = path.lstat()
    except OSError as error:
        raise CoordinatorError("COORDINATOR_PATH_INVALID", f"{label} cannot be created") from error
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
        raise CoordinatorError("COORDINATOR_PATH_INVALID", f"{label} is not a directory")
    os.chmod(path, 0o700)


def _atomic_write(path: Path, value: Mapping[str, Any]) -> str:
    _ensure_directory(path.parent, "coordinator record directory")
    data = _canonical(value)
    temporary = path.parent / f".{path.name}.{secrets.token_hex(8)}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        offset = 0
        while offset < len(data):
            written = os.write(descriptor, data[offset:])
            if written <= 0:
                raise CoordinatorError("COORDINATOR_RECORD_WRITE_FAILED", "short coordinator record write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.replace(temporary, path)
        os.chmod(path, 0o600)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        details = path.lstat()
    except FileNotFoundError:
        return None
    except OSError as error:
        raise CoordinatorError("COORDINATOR_RECORD_READ_FAILED", "coordinator record cannot be inspected") from error
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
        raise CoordinatorError("COORDINATOR_RECORD_INVALID", "coordinator record is not a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CoordinatorError("COORDINATOR_RECORD_INVALID", "coordinator record is invalid") from error
    if not isinstance(value, dict):
        raise CoordinatorError("COORDINATOR_RECORD_INVALID", "coordinator record is not an object")
    return value


def _process_snapshot(pid: int) -> dict[str, Any]:
    if type(pid) is not int or pid <= 0:
        raise CoordinatorError("COORDINATOR_PROCESS_ID_INVALID", "process PID is invalid")
    process_dir = Path("/proc") / str(pid)
    try:
        raw_stat = (process_dir / "stat").read_text(encoding="ascii")
        close = raw_stat.rfind(")")
        if close < 0:
            raise ValueError("stat comm is missing")
        fields = raw_stat[close + 2 :].split()
        start_ticks = fields[19]
        executable = str((process_dir / "exe").resolve(strict=True))
        argv = (process_dir / "cmdline").read_bytes()
        pgid = os.getpgid(pid)
        sid = os.getsid(pid)
        cgroup = (process_dir / "cgroup").read_bytes()
        boot = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    except (FileNotFoundError, PermissionError, OSError, ValueError, IndexError) as error:
        raise CoordinatorError("COORDINATOR_PROCESS_IDENTITY_UNAVAILABLE", "complete child identity is unavailable") from error
    try:
        executable_sha256 = hashlib.sha256(Path(executable).read_bytes()).hexdigest()
    except OSError as error:
        raise CoordinatorError("COORDINATOR_PROCESS_IDENTITY_UNAVAILABLE", "child executable identity is unavailable") from error
    return {
        "pid": pid,
        "process_start_identity": f"procfs-start-ticks:{start_ticks}",
        "boot_identity": f"boot-id:{boot}",
        "executable": executable,
        "executable_sha256": executable_sha256,
        "argv_sha256": hashlib.sha256(argv).hexdigest(),
        "pgid": pgid,
        "sid": sid,
        "cgroup_identity": "sha256:" + hashlib.sha256(cgroup).hexdigest(),
    }


def _same_process(expected: Mapping[str, Any], observed: Mapping[str, Any]) -> bool:
    fields = (
        "pid", "process_start_identity", "boot_identity", "executable",
        "executable_sha256", "argv_sha256", "pgid", "sid", "cgroup_identity",
    )
    return all(expected.get(field) == observed.get(field) for field in fields)


def _file_reference(path: Path, limit: int) -> dict[str, Any]:
    try:
        details = path.lstat()
        size = details.st_size
        digest = hashlib.sha256(path.read_bytes()[: limit + 1]).hexdigest() if size <= limit else None
    except OSError:
        size = 0
        digest = None
    return {"path": str(path), "byte_count": size, "sha256": digest}


@dataclass(frozen=True)
class CoordinatorConfig:
    system_x_root: Path
    inspector_root: Path
    runtime_root: Path
    interpreter: Path
    enabled: bool = True
    interval_seconds: float = DEFAULT_INTERVAL_SECONDS
    maximum_backoff_seconds: float = DEFAULT_MAXIMUM_BACKOFF_SECONDS
    capture_limit_bytes: int = DEFAULT_CAPTURE_BYTES

    def __post_init__(self) -> None:
        for name in ("system_x_root", "inspector_root", "runtime_root", "interpreter"):
            value = Path(getattr(self, name))
            if not value.is_absolute():
                raise CoordinatorError("COORDINATOR_CONFIG_INVALID", f"{name} must be absolute")
        if type(self.enabled) is not bool:
            raise CoordinatorError("COORDINATOR_CONFIG_INVALID", "enabled must be boolean")
        if type(self.interval_seconds) not in (int, float) or not math.isfinite(float(self.interval_seconds)) or not 1.0 <= float(self.interval_seconds) <= 300.0:
            raise CoordinatorError("COORDINATOR_CONFIG_INVALID", "interval must be finite in 1..300 seconds")
        if type(self.maximum_backoff_seconds) not in (int, float) or not math.isfinite(float(self.maximum_backoff_seconds)) or not 1.0 <= float(self.maximum_backoff_seconds) <= 3600.0:
            raise CoordinatorError("COORDINATOR_CONFIG_INVALID", "maximum backoff must be finite in 1..3600 seconds")
        if type(self.capture_limit_bytes) is not int or self.capture_limit_bytes <= 0 or self.capture_limit_bytes > 16 * 1024 * 1024:
            raise CoordinatorError("COORDINATOR_CONFIG_INVALID", "capture limit must be finite and bounded")

    @classmethod
    def for_system_x(cls, runtime_root: Path, *, enabled: bool = True) -> "CoordinatorConfig":
        root = Path(__file__).resolve().parents[2]
        inspector = root / "INSPECTOR"
        candidates = (
            root / "model-api-gguf" / "api_service" / ".venv" / "bin" / "python",
            inspector / ".venv" / "bin" / "python",
        )
        interpreter = next((item for item in candidates if item.is_file() and os.access(item, os.X_OK)), None)
        if interpreter is None:
            raise CoordinatorError("COORDINATOR_INTERPRETER_UNAVAILABLE", "clone-local Inspector interpreter is unavailable")
        return cls(root, inspector, Path(runtime_root).resolve(), interpreter, enabled=enabled)

    @property
    def identity(self) -> str:
        return _identity(
            {
                "schema_version": STATUS_SCHEMA,
                "system_x_root": str(self.system_x_root),
                "inspector_root": str(self.inspector_root),
                "runtime_root": str(self.runtime_root),
                "interpreter": str(self.interpreter),
                "enabled": self.enabled,
                "interval_seconds": float(self.interval_seconds),
                "maximum_backoff_seconds": float(self.maximum_backoff_seconds),
                "capture_limit_bytes": self.capture_limit_bytes,
            }
        )


@dataclass
class _ActiveChild:
    invocation_id: str
    process: subprocess.Popen[bytes] | None
    identity: dict[str, Any]
    stdout_path: Path
    stderr_path: Path


class AutomaticIntakeCoordinator:
    """Owns cadence and one Inspector child; it never interprets candidates."""

    @classmethod
    def for_system_x(cls, runtime_root: Path, *, enabled: bool = True) -> "AutomaticIntakeCoordinator":
        return cls(
            CoordinatorConfig.for_system_x(runtime_root, enabled=enabled)
        )

    def __init__(
        self,
        config: CoordinatorConfig,
        *,
        clock: Callable[[], float] | None = None,
        utc_clock: Callable[[], str] | None = None,
        launcher: Callable[..., subprocess.Popen[bytes]] | None = None,
        observer: Callable[[int], dict[str, Any]] | None = None,
    ) -> None:
        self.config = config
        self.clock = clock or __import__("time").monotonic
        self.utc_clock = utc_clock or _utc_now
        self.launcher = launcher or subprocess.Popen
        self.observer = observer or _process_snapshot
        self.coordinator_root = config.runtime_root / "coordinator"
        self.invocation_root = self.coordinator_root / "invocations"
        self.log_root = self.coordinator_root / "logs"
        self.status_path = self.coordinator_root / "status.json"
        self._active: _ActiveChild | None = None
        self._blocked_uncertain = False
        self._supervisor_generation = "generation-unknown"
        self._state = "DISABLED" if not config.enabled else "IDLE"
        self._last_invocation_id: str | None = None
        self._last_action: str | None = None
        self._last_reason: str | None = None
        self._active_inspector_reference: dict[str, Any] | None = None
        self._dispatch_basis_reference: dict[str, Any] | None = None
        self._processed_basis_reference: dict[str, Any] | None = None
        self._rejected_basis_reference: dict[str, Any] | None = None
        self._failure_count = 0
        self._last_start_utc: str | None = None
        self._last_completion_utc: str | None = None
        self._next_due: float | None = 0.0 if config.enabled else None
        _ensure_directory(self.coordinator_root, "coordinator root")
        _ensure_directory(self.invocation_root, "coordinator invocation root")
        _ensure_directory(self.log_root, "coordinator log root")

    def set_supervisor_generation(self, generation: str) -> None:
        if not isinstance(generation, str) or not generation:
            raise CoordinatorError("COORDINATOR_GENERATION_INVALID", "supervisor generation is invalid")
        self._supervisor_generation = generation

    def _snapshot_value(self, now: float | None = None) -> dict[str, Any]:
        return {
            "schema_version": STATUS_SCHEMA,
            "enabled": self.config.enabled,
            "supervisor_generation": self._supervisor_generation,
            "coordinator_state": self._state,
            "configuration_identity": self.config.identity,
            "last_invocation_id": self._last_invocation_id,
            "active_child_identity": dict(self._active.identity) if self._active else None,
            "last_inspector_action": self._last_action,
            "last_reason_code": self._last_reason,
            "active_inspector_transaction_reference": self._active_inspector_reference,
            "dispatch_basis_reference": self._dispatch_basis_reference,
            "processed_basis_reference": self._processed_basis_reference,
            "rejected_basis_reference": self._rejected_basis_reference,
            "consecutive_method_failure_count": self._failure_count,
            "last_start_utc": self._last_start_utc,
            "last_completion_utc": self._last_completion_utc,
            "next_due_monotonic": self._next_due if now is None else self._next_due,
            "updated_utc": self.utc_clock(),
        }

    def snapshot(self) -> dict[str, Any]:
        return self._snapshot_value()

    def _publish(self) -> dict[str, Any]:
        value = self._snapshot_value()
        _atomic_write(self.status_path, value)
        return value

    def _new_invocation_id(self) -> str:
        return "inv-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + secrets.token_hex(6)

    def _argv(self) -> tuple[str, ...]:
        return (str(self.config.interpreter), "-B", "-s", "-S", "-m", "system_x_inspector", "reconcile-intake")

    def _environment(self) -> dict[str, str]:
        return {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        }

    def _intent(self, invocation_id: str, stdout_path: Path, stderr_path: Path) -> dict[str, Any]:
        argv = self._argv()
        environment = self._environment()
        return {
            "invocation_id": invocation_id,
            "operation": "reconcile-intake",
            "interpreter": str(self.config.interpreter),
            "module": "system_x_inspector",
            "cwd": str(self.config.inspector_root),
            "argv": list(argv),
            "argv_identity": _identity(list(argv)),
            "environment_identity": _identity(environment),
            "model_policy_argument_count": 0,
            "raw_credential_present": False,
            "private_endpoint_present": False,
            "stdout_reference": {"path": str(stdout_path)},
            "stderr_reference": {"path": str(stderr_path)},
            "start_new_session": True,
        }

    def _record_path(self, invocation_id: str) -> Path:
        return self.invocation_root / f"{invocation_id}.json"

    def _write_invocation(self, record: dict[str, Any], *, create: bool = False) -> None:
        body = dict(record)
        body.pop("record_identity", None)
        body["record_identity"] = _identity(body)
        path = self._record_path(str(body["invocation_id"]))
        if create and path.exists():
            raise CoordinatorError("COORDINATOR_INVOCATION_COLLISION", "invocation record already exists")
        _atomic_write(path, body)

    def _append_transition(self, record: dict[str, Any], state: str, **values: Any) -> None:
        record.setdefault("state_transitions", []).append({"state": state, "timestamp_utc": self.utc_clock(), **values})
        record["updated_utc"] = self.utc_clock()
        self._write_invocation(record)

    def _start(self, now: float) -> None:
        self._state = "CHILD_STARTING"
        invocation_id = self._new_invocation_id()
        stdout_path = self.log_root / f"{invocation_id}.stdout"
        stderr_path = self.log_root / f"{invocation_id}.stderr"
        intent = self._intent(invocation_id, stdout_path, stderr_path)
        record: dict[str, Any] = {
            "schema_version": INVOCATION_SCHEMA,
            "invocation_id": invocation_id,
            "supervisor_generation": self._supervisor_generation,
            "created_utc": self.utc_clock(),
            "operation": "reconcile-intake",
            "interpreter": str(self.config.interpreter),
            "module": "system_x_inspector",
            "cwd": str(self.config.inspector_root),
            "environment_identity": intent["environment_identity"],
            "prior_inspector_transaction_reference": self._active_inspector_reference,
            "prior_processed_basis_reference": self._processed_basis_reference,
            "prior_rejected_basis_reference": self._rejected_basis_reference,
            "launch_intent": intent,
            "process_identity": None,
            "state_transitions": [{"state": "LAUNCH_INTENT", "timestamp_utc": self.utc_clock()}],
            "stdout_reference": {"path": str(stdout_path), "byte_count": 0, "sha256": None},
            "stderr_reference": {"path": str(stderr_path), "byte_count": 0, "sha256": None},
            "exit_code": None,
            "timeout": False,
            "capture_overflow": False,
            "machine_result_identity": None,
            "inspector_transaction_reference": None,
            "deployment_result_reference": None,
            "terminal_basis_reference": None,
            "terminal_classification": None,
            "dispatch_accepted": False,
            "completion_utc": None,
            "updated_utc": self.utc_clock(),
            "record_identity": "",
        }
        self._write_invocation(record, create=True)
        stdout_handle = open(stdout_path, "ab", buffering=0)
        stderr_handle = open(stderr_path, "ab", buffering=0)
        try:
            process = self.launcher(
                list(self._argv()),
                cwd=str(self.config.inspector_root),
                env=self._environment(),
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
                close_fds=True,
                start_new_session=True,
                shell=False,
            )
        except Exception as error:
            stdout_handle.close()
            stderr_handle.close()
            record["exit_code"] = None
            record["terminal_classification"] = "LAUNCH_FAILED"
            self._append_transition(record, "LAUNCH_FAILED", reason_code=type(error).__name__)
            self._method_failure("COORDINATOR_LAUNCH_FAILED", now)
            return
        stdout_handle.close()
        stderr_handle.close()
        try:
            identity = self.observer(int(process.pid))
        except CoordinatorError as error:
            try:
                process.terminate()
            except OSError:
                pass
            record["terminal_classification"] = "IDENTITY_UNAVAILABLE"
            self._append_transition(record, "IDENTITY_UNAVAILABLE", reason_code=error.reason_code)
            self._method_failure(error.reason_code, now)
            return
        record["process_identity"] = identity
        record["state_transitions"].append({"state": "CHILD_RUNNING", "timestamp_utc": self.utc_clock()})
        self._write_invocation(record)
        self._active = _ActiveChild(invocation_id, process, identity, stdout_path, stderr_path)
        self._last_invocation_id = invocation_id
        self._last_start_utc = self.utc_clock()
        self._state = "CHILD_RUNNING"
        self._publish()

    def _method_failure(self, reason: str, now: float) -> None:
        self._failure_count += 1
        delay = min(float(self.config.maximum_backoff_seconds), float(self.config.interval_seconds) * (2 ** max(0, self._failure_count - 1)))
        self._last_reason = reason
        self._state = "BACKOFF"
        self._next_due = now + delay
        self._publish()

    def _domain_terminal(self, now: float) -> None:
        self._failure_count = 0
        self._state = "IDLE"
        self._next_due = now + float(self.config.interval_seconds)

    def _capture_overflow(self, active: _ActiveChild) -> bool:
        return any(path.exists() and path.stat().st_size > self.config.capture_limit_bytes for path in (active.stdout_path, active.stderr_path))

    def _terminate_owned(self, active: _ActiveChild) -> bool:
        try:
            observed = self.observer(int(active.identity["pid"]))
        except CoordinatorError:
            return False
        if not _same_process(active.identity, observed):
            self._blocked_uncertain = True
            self._last_reason = "COORDINATOR_OWNER_UNCERTAIN"
            return False
        try:
            if active.process is not None:
                active.process.terminate()
            else:
                os.kill(int(active.identity["pid"]), signal.SIGTERM)
            return True
        except OSError:
            return False

    def _read_machine_result(self, active: _ActiveChild) -> tuple[dict[str, Any] | None, str | None]:
        try:
            raw = active.stdout_path.read_bytes()
        except OSError:
            return None, "COORDINATOR_STDOUT_UNAVAILABLE"
        if len(raw) > self.config.capture_limit_bytes:
            return None, "COORDINATOR_CAPTURE_OVERFLOW"
        try:
            lines = [line for line in raw.decode("utf-8", errors="strict").splitlines() if line.strip()]
            value = json.loads(lines[-1]) if lines else None
        except (UnicodeDecodeError, json.JSONDecodeError, IndexError):
            return None, "COORDINATOR_MACHINE_RESULT_INVALID"
        if not isinstance(value, dict) or value.get("operation") != "reconcile-intake" or value.get("schema_version") != "system-x.inspector-machine-result.v1":
            return None, "COORDINATOR_MACHINE_RESULT_INVALID"
        data = value.get("data")
        automatic = data.get("automatic_result") if isinstance(data, dict) else None
        if not isinstance(automatic, dict) or automatic.get("action") not in AUTOMATIC_ACTIONS or not isinstance(automatic.get("reason_code"), str):
            return None, "COORDINATOR_AUTOMATIC_RESULT_INVALID"
        return value, None

    def _finish(self, active: _ActiveChild, now: float, exit_code: int | None, *, capture_overflow: bool = False) -> None:
        path = self._record_path(active.invocation_id)
        record = _read_json(path)
        if record is None:
            raise CoordinatorError("COORDINATOR_INVOCATION_MISSING", "active invocation record is missing")
        machine, error_reason = (None, "COORDINATOR_CAPTURE_OVERFLOW") if capture_overflow else self._read_machine_result(active)
        stdout_ref = _file_reference(active.stdout_path, self.config.capture_limit_bytes)
        stderr_ref = _file_reference(active.stderr_path, self.config.capture_limit_bytes)
        record["stdout_reference"] = stdout_ref
        record["stderr_reference"] = stderr_ref
        record["exit_code"] = exit_code
        record["capture_overflow"] = bool(capture_overflow)
        record["completion_utc"] = self.utc_clock()
        if machine is None:
            record["terminal_classification"] = "CAPTURE_OR_RESULT_FAILURE"
            self._append_transition(record, "TERMINAL_RESULT_UNAVAILABLE", reason_code=error_reason)
            self._active = None
            self._last_completion_utc = self.utc_clock()
            self._method_failure(str(error_reason), now)
            return
        automatic = machine["data"]["automatic_result"]
        basis = machine["data"].get("terminal_basis_reference")
        action = automatic["action"]
        reason = automatic["reason_code"]
        record["machine_result_identity"] = _identity(machine)
        active_reference = automatic.get("active_transaction_reference")
        record["inspector_transaction_reference"] = active_reference
        record["deployment_result_reference"] = automatic.get("existing_result_reference")
        record["terminal_basis_reference"] = basis
        record["dispatch_accepted"] = action == "DISPATCH_FIRST_MODEL" or (isinstance(basis, dict) and basis.get("basis_class") == "PROCESSED")
        record["terminal_classification"] = "PROCESSED" if record["dispatch_accepted"] else "REJECTED" if action == "REJECT_CANDIDATE" else "NOOP"
        self._append_transition(record, "CHILD_TERMINAL", exit_code=exit_code, action=action, reason_code=reason)
        self._last_action = action
        self._last_reason = reason
        self._active_inspector_reference = active_reference if isinstance(active_reference, dict) else None
        self._dispatch_basis_reference = basis if isinstance(basis, dict) else None
        if isinstance(basis, dict) and basis.get("basis_class") == "PROCESSED":
            self._processed_basis_reference = basis
        if isinstance(basis, dict) and basis.get("basis_class") == "REJECTED":
            self._rejected_basis_reference = basis
        self._active = None
        self._last_completion_utc = self.utc_clock()
        self._domain_terminal(now)
        self._publish()

    def observe_active_child(self, current_time: float | None = None) -> dict[str, Any]:
        now = self.clock() if current_time is None else float(current_time)
        active = self._active
        if active is None:
            return self.snapshot()
        try:
            observed = self.observer(int(active.identity["pid"]))
        except CoordinatorError:
            observed = None
        if observed is not None and not _same_process(active.identity, observed):
            self._blocked_uncertain = True
            self._state = "REATTACHING"
            self._last_reason = "COORDINATOR_OWNER_UNCERTAIN"
            self._publish()
            return self.snapshot()
        if self._capture_overflow(active):
            if observed is not None and _same_process(active.identity, observed):
                self._terminate_owned(active)
            self._finish(active, now, active.process.poll() if active.process is not None else None, capture_overflow=True)
            return self.snapshot()
        running = active.process.poll() is None if active.process is not None else observed is not None
        if running:
            self._state = "CHILD_RUNNING"
            self._publish()
            return self.snapshot()
        exit_code = active.process.returncode if active.process is not None else None
        self._finish(active, now, exit_code)
        return self.snapshot()

    def recover(self, current_time: float, desired_state: str, supervisor_generation: str | None = None) -> dict[str, Any]:
        if supervisor_generation is not None:
            self.set_supervisor_generation(supervisor_generation)
        if not self.config.enabled or desired_state == "STOPPED":
            self._state = "DISABLED"
            self._next_due = None
            return self._publish()
        previous = _read_json(self.status_path)
        if isinstance(previous, dict):
            self._last_invocation_id = previous.get("last_invocation_id")
            self._last_action = previous.get("last_inspector_action")
            self._last_reason = previous.get("last_reason_code")
            self._processed_basis_reference = previous.get("processed_basis_reference")
            self._rejected_basis_reference = previous.get("rejected_basis_reference")
            self._failure_count = int(previous.get("consecutive_method_failure_count", 0) or 0)
            identity = previous.get("active_child_identity")
            if isinstance(identity, dict):
                try:
                    observed = self.observer(int(identity["pid"]))
                except (CoordinatorError, KeyError, TypeError, ValueError):
                    observed = None
                if observed is not None and _same_process(identity, observed):
                    invocation_id = self._last_invocation_id
                    if isinstance(invocation_id, str):
                        record = _read_json(self._record_path(invocation_id))
                        if record is not None:
                            self._active = _ActiveChild(invocation_id, None, identity, Path(record["stdout_reference"]["path"]), Path(record["stderr_reference"]["path"]))
                            self._state = "REATTACHING"
                            self._next_due = None
                            return self._publish()
                record = _read_json(self._record_path(str(self._last_invocation_id))) if isinstance(self._last_invocation_id, str) else None
                if isinstance(record, dict) and record.get("dispatch_accepted") is True:
                    self._state = "IDLE"
                    self._next_due = float(current_time) + float(self.config.interval_seconds)
                    return self._publish()
                self._blocked_uncertain = True
                self._state = "REATTACHING"
                self._last_reason = "COORDINATOR_OWNER_UNCERTAIN"
                self._next_due = None
                return self._publish()
        self._state = "IDLE"
        self._next_due = float(current_time)
        return self._publish()

    def tick(self, current_time: float, desired_state: str, supervisor_generation: str | None = None) -> dict[str, Any]:
        now = float(current_time)
        if supervisor_generation is not None:
            self.set_supervisor_generation(supervisor_generation)
        if not self.config.enabled or desired_state == "STOPPED":
            self.request_stop(now)
            return self.snapshot()
        if self._active is not None:
            return self.observe_active_child(now)
        if self._blocked_uncertain:
            self._state = "REATTACHING"
            return self._publish()
        if self._next_due is None or now < self._next_due:
            if self._state not in ("BACKOFF", "REATTACHING"):
                self._state = "IDLE"
            return self._publish()
        self._state = "DUE"
        self._publish()
        self._start(now)
        return self.snapshot()

    def request_stop(self, current_time: float | None = None) -> dict[str, Any]:
        now = self.clock() if current_time is None else float(current_time)
        self._state = "STOPPING"
        self._next_due = None
        if self._active is not None:
            self._terminate_owned(self._active)
            if self._active.process is not None and self._active.process.poll() is not None:
                self.observe_active_child(now)
        if self._active is None:
            self._state = "DISABLED"
        return self._publish()

