"""Exclusive Inspector transaction-lock ownership."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any

from .errors import InspectorError
from .paths import InspectorPaths
from .records import canonical_json_bytes, fsync_directory
from .results import utc_now


def process_start_identity(pid: int) -> str | None:
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError):
        return None
    close = text.rfind(")")
    if close < 0:
        return None
    fields = text[close + 2 :].split()
    if len(fields) <= 19:
        return None
    return f"procfs-start-ticks:{fields[19]}"


def boot_identity() -> str | None:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="utf-8"
        ).strip()
    except (FileNotFoundError, PermissionError):
        return None


def _existing_lock_reason(path: Path) -> tuple[str, str]:
    try:
        details = path.lstat()
    except FileNotFoundError:
        return ("TRANSACTION_OWNERSHIP_UNCERTAIN", "lock disappeared during check")
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
        return (
            "TRANSACTION_OWNERSHIP_UNCERTAIN",
            "existing lock has an unsafe physical type",
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return (
            "TRANSACTION_OWNERSHIP_UNCERTAIN",
            "existing lock identity is unreadable",
        )
    if not isinstance(value, dict):
        return (
            "TRANSACTION_OWNERSHIP_UNCERTAIN",
            "existing lock identity is invalid",
        )
    pid = value.get("pid")
    expected_start = value.get("process_start_identity")
    if not isinstance(pid, int) or not isinstance(expected_start, str):
        return (
            "TRANSACTION_OWNERSHIP_UNCERTAIN",
            "existing lock owner identity is incomplete",
        )
    current_start = process_start_identity(pid)
    if current_start is None:
        return ("TRANSACTION_LOCK_STALE", "existing exact owner is not live")
    if current_start == expected_start:
        return ("TRANSACTION_LOCK_ACTIVE", "matching lock owner is active")
    return ("TRANSACTION_LOCK_STALE", "existing PID has a different start identity")


def inspect_active_lock(path: Path) -> dict[str, Any]:
    if not path.exists() and not path.is_symlink():
        return {"state": "absent", "reason_code": "OK", "record": None}
    reason_code, message = _existing_lock_reason(path)
    record = None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            record = {
                key: value.get(key)
                for key in (
                    "schema_version",
                    "transaction_id",
                    "operation",
                    "pid",
                    "process_start_identity",
                    "boot_identity",
                    "created_utc",
                    "inspector_root_identity",
                )
            }
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        record = None
    return {
        "state": (
            "active"
            if reason_code == "TRANSACTION_LOCK_ACTIVE"
            else "stale"
            if reason_code == "TRANSACTION_LOCK_STALE"
            else "uncertain"
        ),
        "reason_code": reason_code,
        "message": message,
        "record": record,
    }


class TransactionLock:
    def __init__(
        self,
        paths: InspectorPaths,
        *,
        transaction_id: str,
        operation: str,
    ) -> None:
        self.paths = paths
        self.path = paths.locks / "active.json"
        self.transaction_id = transaction_id
        self.operation = operation
        self.pid = os.getpid()
        self.start_identity = process_start_identity(self.pid)
        self.inode: int | None = None
        self.acquired = False

    def acquire(self) -> dict[str, Any]:
        root_stat = self.paths.inspector_root.stat()
        value = {
            "schema_version": "system-x.inspector-active-lock.v1",
            "transaction_id": self.transaction_id,
            "operation": self.operation,
            "pid": self.pid,
            "process_start_identity": self.start_identity,
            "boot_identity": boot_identity(),
            "created_utc": utc_now(),
            "inspector_root_identity": {
                "device": root_stat.st_dev,
                "inode": root_stat.st_ino,
            },
        }
        try:
            descriptor = os.open(
                self.path,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | os.O_CLOEXEC
                | os.O_NOFOLLOW,
                0o600,
            )
        except FileExistsError as error:
            reason, message = _existing_lock_reason(self.path)
            raise InspectorError(reason, message) from error
        try:
            os.write(descriptor, canonical_json_bytes(value))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        fsync_directory(self.path.parent)
        details = self.path.lstat()
        self.inode = details.st_ino
        self.acquired = True
        return value

    def release(self) -> None:
        if not self.acquired or self.inode is None:
            return
        details = self.path.lstat()
        if (
            stat.S_ISLNK(details.st_mode)
            or not stat.S_ISREG(details.st_mode)
            or details.st_ino != self.inode
        ):
            raise InspectorError(
                "TRANSACTION_OWNERSHIP_UNCERTAIN",
                "active lock identity changed before release",
            )
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if (
            value.get("transaction_id") != self.transaction_id
            or value.get("pid") != self.pid
            or value.get("process_start_identity") != self.start_identity
        ):
            raise InspectorError(
                "TRANSACTION_OWNERSHIP_UNCERTAIN",
                "active lock ownership changed before release",
            )
        self.path.unlink()
        fsync_directory(self.path.parent)
        self.acquired = False

    def __enter__(self) -> "TransactionLock":
        self.acquire()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.release()
