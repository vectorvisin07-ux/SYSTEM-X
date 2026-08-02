"""Persistent bootstrap state and non-secret operation receipts."""

from __future__ import annotations

import json
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .config import canonical_json_bytes
from .errors import BootstrapError, ErrorCode
from .paths import RepositoryPaths
from .result import redact, utc_now


STATE_SCHEMA = "system-x.bootstrap.state.v1"
RECEIPT_SCHEMA = "system-x.bootstrap.receipt.v1"
STABLE_STATES = (
    "CLONED",
    "HOST_INSPECTED",
    "HOST_PLAN_READY",
    "HOST_READY",
    "SUBMODULES_READY",
    "PYTHON_ENVIRONMENTS_READY",
    "LLAMA_SERVER_BUILT",
    "RUNTIME_INITIALIZED",
    "CREDENTIAL_READY",
    "SERVICE_REGISTERED",
    "WAITING_FOR_MODEL",
    "READY",
)
FAILURE_STATES = ("FAILED_CLEAN", "FAIL_CLOSED")
ALL_STATES = frozenset((*STABLE_STATES, *FAILURE_STATES))


@dataclass(frozen=True, slots=True)
class StateDocument:
    state: str
    stable_state: str
    generation: int
    completed_operations: tuple[str, ...]
    last_receipt_id: str | None
    updated_utc: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": STATE_SCHEMA,
            "version": 1,
            "state": self.state,
            "stable_state": self.stable_state,
            "generation": self.generation,
            "completed_operations": list(self.completed_operations),
            "last_receipt_id": self.last_receipt_id,
            "updated_utc": self.updated_utc,
        }


def initial_state() -> StateDocument:
    return StateDocument("CLONED", "CLONED", 0, (), None, None)


def read_state(paths: RepositoryPaths) -> StateDocument:
    path = paths.transaction_status
    if not path.exists():
        return initial_state()
    if path.is_symlink() or not path.is_file():
        raise BootstrapError(ErrorCode.UNKNOWN_STATE, "bootstrap state target is unsafe")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BootstrapError(ErrorCode.UNKNOWN_STATE, "bootstrap state is unreadable") from exc
    if set(value) != {
        "schema", "version", "state", "stable_state", "generation", "completed_operations", "last_receipt_id", "updated_utc"
    }:
        raise BootstrapError(ErrorCode.UNKNOWN_STATE, "bootstrap state fields are not closed")
    if (
        value["schema"] != STATE_SCHEMA
        or value["version"] != 1
        or value["state"] not in ALL_STATES
        or value["stable_state"] not in STABLE_STATES
        or type(value["generation"]) is not int
        or value["generation"] < 0
        or not isinstance(value["completed_operations"], list)
        or any(not isinstance(item, str) or not item for item in value["completed_operations"])
        or (value["last_receipt_id"] is not None and not isinstance(value["last_receipt_id"], str))
        or (value["updated_utc"] is not None and not isinstance(value["updated_utc"], str))
    ):
        raise BootstrapError(ErrorCode.UNKNOWN_STATE, "bootstrap state values are invalid")
    return StateDocument(
        value["state"], value["stable_state"], value["generation"], tuple(value["completed_operations"]),
        value["last_receipt_id"], value["updated_utc"],
    )


def _write_atomic(path: Path, payload: bytes, paths: RepositoryPaths) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{secrets.token_hex(16)}.tmp"
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        if os.geteuid() == 0:
            owner = paths.root.stat()
            os.chown(path, owner.st_uid, owner.st_gid, follow_symlinks=False)
            os.chown(path.parent, owner.st_uid, owner.st_gid, follow_symlinks=False)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def write_success_state(paths: RepositoryPaths, previous: StateDocument, target: str, operation: str, receipt_id: str) -> StateDocument:
    if target not in STABLE_STATES:
        raise BootstrapError(ErrorCode.UNKNOWN_STATE, "target bootstrap state is invalid")
    document = StateDocument(
        state=target,
        stable_state=target,
        generation=previous.generation + 1,
        completed_operations=(*previous.completed_operations, operation),
        last_receipt_id=receipt_id,
        updated_utc=utc_now(),
    )
    _write_atomic(paths.transaction_status, canonical_json_bytes(document.as_dict()), paths)
    return document


def write_failure_state(paths: RepositoryPaths, previous: StateDocument, failure: str, operation: str) -> StateDocument:
    if failure not in FAILURE_STATES:
        raise BootstrapError(ErrorCode.UNKNOWN_STATE, "failure bootstrap state is invalid")
    document = StateDocument(
        state=failure,
        stable_state=previous.stable_state,
        generation=previous.generation + 1,
        completed_operations=previous.completed_operations,
        last_receipt_id=previous.last_receipt_id,
        updated_utc=utc_now(),
    )
    _write_atomic(paths.transaction_status, canonical_json_bytes(document.as_dict()), paths)
    return document


def write_receipt(
    paths: RepositoryPaths,
    *,
    receipt_id: str,
    operation: str,
    prestate: StateDocument,
    poststate: str,
    plan_identity: str,
    changed: bool,
    details: Mapping[str, Any],
) -> Path:
    directory = paths.bootstrap_state / "receipts"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = directory / f"{receipt_id}.json"
    payload = canonical_json_bytes(
        {
            "schema": RECEIPT_SCHEMA,
            "version": 1,
            "receipt_id": receipt_id,
            "recorded_utc": utc_now(),
            "operation": operation,
            "prestate": prestate.state,
            "prestate_generation": prestate.generation,
            "poststate": poststate,
            "plan_identity": plan_identity,
            "changed": bool(changed),
            "details": redact(dict(details)),
        }
    )
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    if os.geteuid() == 0:
        owner = paths.root.stat()
        os.chown(directory, owner.st_uid, owner.st_gid, follow_symlinks=False)
        os.chown(path, owner.st_uid, owner.st_gid, follow_symlinks=False)
    return path
