"""Exclusive, write-ahead bootstrap transactions with bounded ownership."""

from __future__ import annotations

import json
import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .config import canonical_json_bytes
from .errors import BootstrapError, ErrorCode
from .paths import RepositoryPaths, resolve_contained
from .result import utc_now


TRANSACTION_SCHEMA = "system-x.bootstrap.transaction.v1"


def _mkdir_owned(path: Path, owned: list[Path], *, mode: int = 0o700) -> None:
    missing: list[Path] = []
    cursor = path
    while not cursor.exists():
        missing.append(cursor)
        cursor = cursor.parent
    if not cursor.is_dir() or cursor.is_symlink():
        raise BootstrapError(ErrorCode.RUNTIME_COLLISION, "transaction parent is not a safe directory")
    for item in reversed(missing):
        item.mkdir(mode=mode)
        owned.append(item)


def _write_exclusive(path: Path, payload: bytes, mode: int = 0o600) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, mode)
    try:
        written = 0
        while written < len(payload):
            written += os.write(fd, payload[written:])
        os.fsync(fd)
    finally:
        os.close(fd)


def _return_ownership_to_repository_user(paths: RepositoryPaths, targets: Iterable[Path]) -> None:
    """Keep bootstrap state usable after the root-only host-package step."""

    if os.geteuid() != 0:
        return
    repository_info = paths.root.stat()
    for target in targets:
        try:
            os.chown(target, repository_info.st_uid, repository_info.st_gid, follow_symlinks=False)
        except OSError as exc:
            raise BootstrapError(ErrorCode.UNKNOWN_STATE, "bootstrap state ownership handoff failed") from exc


def _append_fsynced(path: Path, payload: Mapping[str, Any]) -> None:
    data = canonical_json_bytes(payload)
    fd = os.open(path, os.O_WRONLY | os.O_APPEND)
    try:
        written = 0
        while written < len(data):
            written += os.write(fd, data[written:])
        os.fsync(fd)
    finally:
        os.close(fd)


@dataclass(slots=True)
class BootstrapTransaction:
    paths: RepositoryPaths
    operation: str
    plan_identity: str
    prestate: Mapping[str, Any]
    authorized: bool
    state_root: Path | None = None
    token: str = field(default_factory=lambda: secrets.token_hex(24))
    owned_paths: list[Path] = field(default_factory=list)
    record_path: Path | None = None
    _entered: bool = False
    _finished: bool = False

    def __enter__(self) -> "BootstrapTransaction":
        if not self.authorized:
            raise BootstrapError(
                ErrorCode.AUTHORIZATION_REQUIRED,
                "mutation requires explicit operator authorization",
                context={"operation": self.operation},
            )
        if not self.operation or len(self.plan_identity) != 64:
            raise BootstrapError(ErrorCode.PLAN_MISMATCH, "operation or plan identity is invalid")
        try:
            int(self.plan_identity, 16)
        except ValueError as exc:
            raise BootstrapError(ErrorCode.PLAN_MISMATCH, "plan identity must be SHA-256") from exc

        transaction_dir, lock_path = self._locations()
        _mkdir_owned(transaction_dir, self.owned_paths)
        _mkdir_owned(lock_path.parent, self.owned_paths)
        _return_ownership_to_repository_user(self.paths, self.owned_paths)
        lock_payload = canonical_json_bytes(
            {
                "schema": "system-x.bootstrap.lock.v1",
                "version": 1,
                "operation": self.operation,
                "token": self.token,
                "pid": os.getpid(),
                "created_utc": utc_now(),
            }
        )
        try:
            _write_exclusive(lock_path, lock_payload)
        except FileExistsError as exc:
            raise BootstrapError(
                ErrorCode.LOCK_HELD,
                "another bootstrap transaction or unrecovered lock exists",
                context={"lock": str(lock_path)},
            ) from exc
        self.owned_paths.append(lock_path)
        _return_ownership_to_repository_user(self.paths, (lock_path,))
        try:
            record_name = f"{self.operation}-{self.token}.jsonl"
            self.record_path = transaction_dir / record_name
            _write_exclusive(
                self.record_path,
                canonical_json_bytes(
                    {
                        "schema": TRANSACTION_SCHEMA,
                        "version": 1,
                        "event": "write-ahead",
                        "operation": self.operation,
                        "token": self.token,
                        "plan_identity": self.plan_identity,
                        "prestate": dict(self.prestate),
                        "cleanup_ownership": [],
                        "recorded_utc": utc_now(),
                    }
                ),
            )
            self.owned_paths.append(self.record_path)
            _return_ownership_to_repository_user(self.paths, (self.record_path,))
            self._entered = True
            return self
        except BaseException:
            self._release_lock()
            raise

    def _locations(self) -> tuple[Path, Path]:
        if self.state_root is None:
            return self.paths.transaction_directory, self.paths.transaction_lock
        root = self.state_root.resolve(strict=True)
        transaction_dir = root / "transactions"
        lock_path = root / "locks" / "system-x-bootstrap.lock"
        return transaction_dir, lock_path

    def claim_created_path(self, relative: str) -> Path:
        if not self._entered or self._finished:
            raise BootstrapError(ErrorCode.UNKNOWN_STATE, "transaction is not active")
        path = resolve_contained(self.paths.root, relative, allow_missing=True)
        if path.exists() or path.is_symlink():
            raise BootstrapError(
                ErrorCode.RUNTIME_COLLISION,
                "bootstrap cannot claim a pre-existing path",
                context={"path": relative},
            )
        self.owned_paths.append(path)
        return path

    def record(self, event: str, details: Mapping[str, Any] | None = None) -> None:
        if not self._entered or self._finished or self.record_path is None:
            raise BootstrapError(ErrorCode.UNKNOWN_STATE, "transaction is not active")
        _append_fsynced(
            self.record_path,
            {
                "schema": TRANSACTION_SCHEMA,
                "version": 1,
                "event": event,
                "details": dict(details or {}),
                "recorded_utc": utc_now(),
            },
        )

    def complete(self, details: Mapping[str, Any] | None = None) -> None:
        self.record("complete", details)
        self._finished = True

    def __exit__(self, exc_type: object, exc: BaseException | None, traceback: object) -> bool:
        try:
            if self._entered and not self._finished and self.record_path is not None:
                _append_fsynced(
                    self.record_path,
                    {
                        "schema": TRANSACTION_SCHEMA,
                        "version": 1,
                        "event": "failed-clean" if exc is not None else "incomplete",
                        "error_type": type(exc).__name__ if exc is not None else None,
                        "recorded_utc": utc_now(),
                    },
                )
        finally:
            self._finished = True
            self._release_lock()
        return False

    def _release_lock(self) -> None:
        _, lock_path = self._locations()
        try:
            payload = json.loads(lock_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return
        if payload.get("token") == self.token:
            lock_path.unlink()


def incomplete_transactions(directory: Path) -> list[Path]:
    """Return records whose final canonical event is not complete."""

    if not directory.exists():
        return []
    if not directory.is_dir() or directory.is_symlink():
        raise BootstrapError(ErrorCode.RUNTIME_COLLISION, "transaction location is unsafe")
    incomplete: list[Path] = []
    for path in sorted(directory.glob("*.jsonl")):
        if path.is_symlink() or not path.is_file():
            raise BootstrapError(ErrorCode.TRANSACTION_RECOVERY_REQUIRED, "unknown transaction entry")
        try:
            events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BootstrapError(ErrorCode.TRANSACTION_RECOVERY_REQUIRED, "damaged transaction record") from exc
        if not events or events[-1].get("event") != "complete":
            incomplete.append(path)
    return incomplete


def recover_failed_clean_transactions(directory: Path, *, authorized: bool) -> list[str]:
    """Seal only records that demonstrably ended FAILED_CLEAN.

    FAIL_CLOSED, malformed, or merely interrupted write-ahead records require
    operator investigation and are never guessed recoverable.
    """

    if not authorized:
        raise BootstrapError(ErrorCode.AUTHORIZATION_REQUIRED, "transaction recovery requires explicit authorization")
    recovered: list[str] = []
    for path in incomplete_transactions(directory):
        try:
            events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BootstrapError(ErrorCode.TRANSACTION_RECOVERY_REQUIRED, "transaction record cannot be recovered") from exc
        if not events or events[-1].get("event") != "failed-clean":
            raise BootstrapError(
                ErrorCode.TRANSACTION_RECOVERY_REQUIRED,
                "only a physically recorded failed-clean transaction can be recovered automatically",
                context={"record": path.name},
            )
        _append_fsynced(
            path,
            {
                "schema": TRANSACTION_SCHEMA,
                "version": 1,
                "event": "recovered",
                "basis": "last event was failed-clean; cleanup ownership was bounded by the provider",
                "recorded_utc": utc_now(),
            },
        )
        _append_fsynced(
            path,
            {
                "schema": TRANSACTION_SCHEMA,
                "version": 1,
                "event": "complete",
                "recovered": True,
                "recorded_utc": utc_now(),
            },
        )
        recovered.append(path.name)
    return recovered
