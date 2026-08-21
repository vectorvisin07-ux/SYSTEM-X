"""Generate-new credential initialization without exposing raw key material."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any, Mapping

from .command import Runner, SubprocessRunner
from .config import canonical_json_bytes
from .errors import BootstrapError, ErrorCode
from .paths import RepositoryPaths, resolve_contained, validate_relative_path
from .transaction import BootstrapTransaction


_RAW_KEY = re.compile(r"sxk_v1_[0-9a-f]{32}_[A-Za-z0-9_-]{43}")


def _verify_source_contract(paths: RepositoryPaths, records: Mapping[str, str]) -> None:
    for relative, expected in records.items():
        source = resolve_contained(paths.root, relative, allow_missing=False)
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise BootstrapError(ErrorCode.INTEGRITY_FAILURE, "credential source contract changed", context={"path": relative})


def _safe_file(path: Path, mode: int, size: int | None = None) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    return bool(
        stat.S_ISREG(info.st_mode)
        and not stat.S_ISLNK(info.st_mode)
        and stat.S_IMODE(info.st_mode) == mode
        and (size is None or info.st_size == size)
    )


def credential_status(paths: RepositoryPaths, contract: Mapping[str, Any]) -> dict[str, Any]:
    database = resolve_contained(paths.root, contract["database"], allow_missing=True)
    pepper = resolve_contained(paths.root, contract["pepper"], allow_missing=True)
    handoff = resolve_contained(paths.root, contract["handoff"], allow_missing=True)
    marker_path = resolve_contained(paths.root, contract["completion_marker"], allow_missing=True)
    physical = {
        "database": _safe_file(database, 0o600),
        "pepper": _safe_file(pepper, 0o600, contract["pepper_bytes"]),
        "handoff": _safe_file(handoff, 0o600),
        "marker": _safe_file(marker_path, 0o600),
    }
    marker: dict[str, Any] | None = None
    if physical["marker"]:
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            marker = None
    expected_marker = bool(
        marker
        and marker.get("schema") == "system-x.bootstrap.credential-marker.v1"
        and marker.get("credential_schema_identity") == contract["schema_identity"]
        and marker.get("credential_schema_version") == contract["schema_version"]
        and marker.get("label") == contract["label"]
        and re.fullmatch(r"[0-9a-f]{32}", str(marker.get("key_id", "")))
        and marker.get("raw_key_recorded") is False
    )
    if all(physical.values()) and expected_marker:
        state = "ready"
    elif not any(path.exists() for path in (database, pepper, handoff, marker_path)):
        state = "absent"
    else:
        state = "collision"
    return {"state": state, "physical": physical, "key_id": marker.get("key_id") if expected_marker else None}


def _credential_python(paths: RepositoryPaths, contract: Mapping[str, Any]) -> Path:
    relative = validate_relative_path(contract["python_environment"])
    candidate = paths.root.joinpath(*relative.parts)
    resolve_contained(paths.root, str(relative.parent), allow_missing=False)
    if not candidate.is_file():
        raise BootstrapError(
            ErrorCode.PATH_UNSAFE,
            "credential interpreter is not a regular file",
            context={"path": contract["python_environment"]},
        )
    if candidate.is_symlink():
        try:
            target = candidate.resolve(strict=True)
        except OSError as exc:
            raise BootstrapError(
                ErrorCode.PATH_UNSAFE,
                "credential interpreter symlink target is unavailable",
                context={"path": contract["python_environment"]},
            ) from exc
        if target != Path("/usr/bin/python3.14"):
            raise BootstrapError(
                ErrorCode.PATH_UNSAFE,
                "credential interpreter symlink target is not the pinned CPython 3.14 executable",
                context={"path": contract["python_environment"], "target": str(target)},
            )
    return candidate


def _credential_command(
    paths: RepositoryPaths,
    contract: Mapping[str, Any],
    operation: tuple[str, ...],
    runner: Runner,
) -> dict[str, Any]:
    python = _credential_python(paths, contract)
    source = resolve_contained(paths.root, contract["source_root"], allow_missing=False)
    script = (
        "import runpy,sys;"
        f"sys.path.insert(0,{str(source)!r});"
        f"sys.argv=['system-x-credential-admin',*{list(operation)!r}];"
        "runpy.run_module('system_x_gguf_api.credential_admin',run_name='__main__')"
    )
    result = runner((str(python), "-B", "-I", "-s", "-c", script), timeout=180)
    combined = result.stdout + result.stderr
    if _RAW_KEY.search(combined):
        raise BootstrapError(ErrorCode.SECRET_POLICY_VIOLATION, "credential command attempted to emit raw key material")
    if result.returncode != 0:
        raise BootstrapError(ErrorCode.EXTERNAL_COMMAND_FAILED, "credential administration operation failed")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise BootstrapError(ErrorCode.INTEGRITY_FAILURE, "credential command result is not JSON") from exc
    if payload.get("ok") is not True or not isinstance(payload.get("result"), dict):
        raise BootstrapError(ErrorCode.INTEGRITY_FAILURE, "credential command result is invalid")
    return payload["result"]


def initialize_credentials(
    paths: RepositoryPaths,
    contract: Mapping[str, Any],
    *,
    transaction: BootstrapTransaction,
    authorized: bool,
    mode: str = "generate-new",
    encrypted_bundle: Path | None = None,
    sir_authorized: bool = False,
    runner: Runner | None = None,
) -> dict[str, Any]:
    if not authorized:
        raise BootstrapError(ErrorCode.AUTHORIZATION_REQUIRED, "initialize-credentials requires explicit authorization")
    if mode == "import-encrypted":
        raise BootstrapError(
            ErrorCode.AUTHORIZATION_REQUIRED,
            "import-encrypted is blocked pending a separate encrypted bundle and explicit SIR authorization",
            context={"bundle_present": encrypted_bundle is not None, "sir_authorized": bool(sir_authorized)},
        )
    if mode != "generate-new":
        raise BootstrapError(ErrorCode.PRECONDITION_FAILED, "unsupported credential initialization mode")
    if not getattr(transaction, "_entered", False):
        raise BootstrapError(ErrorCode.TRANSACTION_RECOVERY_REQUIRED, "initialize-credentials requires an active transaction")
    _verify_source_contract(paths, contract["source_contract_sha256"])
    status = credential_status(paths, contract)
    if status["state"] == "ready":
        return {"changed": False, "state": "ready", "key_id": status["key_id"], "raw_key_recorded": False}
    if status["state"] != "absent":
        raise BootstrapError(ErrorCode.CREDENTIAL_COLLISION, "credential state already exists without a matching completion marker")

    command = runner or SubprocessRunner()
    database = resolve_contained(paths.root, contract["database"], allow_missing=True)
    pepper = resolve_contained(paths.root, contract["pepper"], allow_missing=True)
    handoff = resolve_contained(paths.root, contract["handoff"], allow_missing=True)
    marker_path = resolve_contained(paths.root, contract["completion_marker"], allow_missing=True)
    candidates = [database, Path(f"{database}-wal"), Path(f"{database}-shm"), pepper, handoff, marker_path]
    try:
        _credential_command(paths, contract, ("initialize",), command)
        issue = _credential_command(
            paths,
            contract,
            ("issue", "--label", contract["label"], "--output-file", str(handoff)),
            command,
        )
        key_id = issue.get("key_id")
        if not isinstance(key_id, str) or not re.fullmatch(r"[0-9a-f]{32}", key_id):
            raise BootstrapError(ErrorCode.INTEGRITY_FAILURE, "credential issue result omitted its non-secret key identity")
        if not (_safe_file(database, 0o600) and _safe_file(pepper, 0o600, contract["pepper_bytes"]) and _safe_file(handoff, 0o600)):
            raise BootstrapError(ErrorCode.INTEGRITY_FAILURE, "generated credential files failed type, mode, or size validation")
        marker = {
            "schema": "system-x.bootstrap.credential-marker.v1",
            "version": 1,
            "credential_schema_identity": contract["schema_identity"],
            "credential_schema_version": contract["schema_version"],
            "label": contract["label"],
            "key_id": key_id,
            "raw_key_recorded": False,
        }
        fd = os.open(marker_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(fd, canonical_json_bytes(marker))
            os.fsync(fd)
        finally:
            os.close(fd)
        transaction.record("credential-ready", {"label": contract["label"], "key_id": key_id, "raw_key_recorded": False})
        return {"changed": True, "state": "ready", "key_id": key_id, "raw_key_recorded": False}
    except BaseException:
        for candidate in candidates:
            try:
                if candidate.is_file() and not candidate.is_symlink():
                    candidate.unlink()
            except OSError:
                pass
        raise
