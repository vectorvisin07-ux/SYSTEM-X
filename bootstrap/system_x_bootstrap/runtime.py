"""Closed runtime-directory and empty registry-database initialization."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
from pathlib import Path
from typing import Any, Mapping

from .command import Runner, SubprocessRunner
from .config import canonical_json_bytes, canonical_sha256
from .errors import BootstrapError, ErrorCode
from .paths import RepositoryPaths, resolve_contained
from .transaction import BootstrapTransaction


def _verify_source_contract(paths: RepositoryPaths, records: Mapping[str, str]) -> None:
    if not records:
        raise BootstrapError(ErrorCode.CONFIGURATION_INVALID, "source contract is empty")
    for relative, expected in records.items():
        path = resolve_contained(paths.root, relative, allow_missing=False)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise BootstrapError(
                ErrorCode.INTEGRITY_FAILURE,
                "runtime source contract changed",
                context={"path": relative},
            )


def expand_runtime_layout(paths: RepositoryPaths, contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group in contract.get("groups", []):
        mode_text = group.get("mode")
        try:
            mode = int(mode_text, 8)
        except (TypeError, ValueError) as exc:
            raise BootstrapError(ErrorCode.CONFIGURATION_INVALID, "runtime directory mode is invalid") from exc
        if mode not in (0o700, 0o755):
            raise BootstrapError(ErrorCode.CONFIGURATION_INVALID, "runtime directory mode is outside policy")
        for relative in group.get("paths", []):
            if relative in seen:
                raise BootstrapError(ErrorCode.CONFIGURATION_INVALID, "runtime path is duplicated")
            resolve_contained(paths.root, relative, allow_missing=True)
            seen.add(relative)
            entries.append({
                "path": relative,
                "mode": mode,
                "owner_component": group["owner_component"],
                "cleanup_owner": group["cleanup_owner"],
                "secret": bool(group["secret"]),
            })
    if len(entries) != contract.get("entry_count"):
        raise BootstrapError(ErrorCode.CONFIGURATION_INVALID, "runtime entry count does not match its lock")
    return sorted(entries, key=lambda item: (len(Path(item["path"]).parts), item["path"]))


def _marker(contract: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": "system-x.bootstrap.runtime-marker.v1",
        "version": 1,
        "runtime_layout_identity": contract["identity"],
        "runtime_layout_sha256": canonical_sha256(contract),
        "entry_count": contract["entry_count"],
        "registry_schema_identity": contract["registry"]["schema_identity"],
        "registry_schema_version": contract["registry"]["schema_version"],
    }


def verify_registry_schema(paths: RepositoryPaths, contract: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the current registry schema while preserving model history."""
    database = resolve_contained(paths.root, contract["registry"]["database"], allow_missing=False)
    info = database.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
        raise BootstrapError(ErrorCode.RUNTIME_COLLISION, "registry database type or mode is invalid")
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        metadata = dict(connection.execute("SELECT key,value FROM registry_metadata").fetchall())
        tables = ("artifact_bundles", "artifact_files", "artifact_locations", "model_versions", "aliases", "model_version_locations", "alias_bindings", "capability_manifests", "artifact_rejections", "registry_events")
        counts = {table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]) for table in tables}
    except sqlite3.Error as exc:
        raise BootstrapError(ErrorCode.RUNTIME_COLLISION, "registry database schema is invalid") from exc
    finally:
        connection.close()
    expected = contract["registry"]
    if integrity != "ok" or metadata.get("schema_identity") != expected["schema_identity"] or metadata.get("schema_version") != str(expected["schema_version"]):
        raise BootstrapError(ErrorCode.RUNTIME_COLLISION, "registry is not current schema")
    return {"integrity": integrity, "metadata_rows": len(metadata), "model_or_event_rows": sum(counts.values()), "counts": counts}

def verify_empty_registry(paths: RepositoryPaths, contract: Mapping[str, Any]) -> dict[str, Any]:
    database = resolve_contained(paths.root, contract["registry"]["database"], allow_missing=False)
    info = database.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
        raise BootstrapError(ErrorCode.RUNTIME_COLLISION, "registry database type or mode is invalid")
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        metadata = dict(connection.execute("SELECT key,value FROM registry_metadata").fetchall())
        counts = {
            table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in (
                "artifact_bundles", "artifact_files", "artifact_locations", "model_versions", "aliases",
                "model_version_locations", "alias_bindings", "capability_manifests", "artifact_rejections",
                "registry_events",
            )
        }
    except sqlite3.Error as exc:
        raise BootstrapError(ErrorCode.RUNTIME_COLLISION, "registry database schema is invalid") from exc
    finally:
        connection.close()
    expected = contract["registry"]
    if (
        integrity != "ok"
        or metadata.get("schema_identity") != expected["schema_identity"]
        or metadata.get("schema_version") != str(expected["schema_version"])
        or metadata.get("registry_generation") != "0"
        or any(counts.values())
    ):
        raise BootstrapError(ErrorCode.RUNTIME_COLLISION, "registry is not current-schema empty state")
    return {"integrity": integrity, "metadata_rows": len(metadata), "model_or_event_rows": sum(counts.values())}


def runtime_status(paths: RepositoryPaths, contract: Mapping[str, Any]) -> str:
    marker_path = resolve_contained(paths.root, contract["completion_marker"], allow_missing=True)
    if not marker_path.exists():
        return "absent"
    if marker_path.is_symlink() or not marker_path.is_file():
        return "collision"
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return "collision"
    expected = _marker(contract)
    if marker != expected:
        immutable_keys = (
            "schema", "version", "runtime_layout_identity",
            "entry_count", "registry_schema_identity", "registry_schema_version",
        )
        historical_hash = marker.get("runtime_layout_sha256")
        if (
            set(marker) != set(expected)
            or any(marker.get(key) != expected[key] for key in immutable_keys)
            or not isinstance(historical_hash, str)
            or len(historical_hash) != 64
            or any(character not in "0123456789abcdef" for character in historical_hash)
        ):
            return "mismatch"
    try:
        entries = expand_runtime_layout(paths, contract)
        for entry in entries:
            target = resolve_contained(paths.root, entry["path"], allow_missing=False)
            info = target.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != entry["mode"]:
                return "mismatch"
        verify_registry_schema(paths, contract)
    except (BootstrapError, OSError):
        return "mismatch"
    return "ready"


def _write_exclusive(path: Path, payload: bytes, mode: int) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)


def initialize_runtime(
    paths: RepositoryPaths,
    contract: Mapping[str, Any],
    *,
    transaction: BootstrapTransaction,
    authorized: bool,
    runner: Runner | None = None,
) -> dict[str, Any]:
    if not authorized:
        raise BootstrapError(ErrorCode.AUTHORIZATION_REQUIRED, "initialize-runtime requires explicit authorization")
    if not getattr(transaction, "_entered", False):
        raise BootstrapError(ErrorCode.TRANSACTION_RECOVERY_REQUIRED, "initialize-runtime requires an active transaction")
    _verify_source_contract(paths, contract["source_contract_sha256"])
    entries = expand_runtime_layout(paths, contract)
    status = runtime_status(paths, contract)
    if status == "ready":
        return {"changed": False, "status": "ready", "entry_count": len(entries)}
    if status not in ("absent",):
        raise BootstrapError(ErrorCode.RUNTIME_COLLISION, "existing runtime marker is unknown or mismatched")
    existing = [entry["path"] for entry in entries if resolve_contained(paths.root, entry["path"], allow_missing=True).exists()]
    if existing:
        raise BootstrapError(
            ErrorCode.RUNTIME_COLLISION,
            "runtime paths exist without a matching completed marker",
            context={"count": len(existing), "first": existing[0]},
        )

    command = runner or SubprocessRunner()
    created_directories: list[Path] = []
    database = resolve_contained(paths.root, contract["registry"]["database"], allow_missing=True)
    database_candidates = [Path(f"{database}{suffix}") for suffix in ("", "-wal", "-shm")]
    marker_path = resolve_contained(paths.root, contract["completion_marker"], allow_missing=True)
    try:
        for entry in entries:
            target = transaction.claim_created_path(entry["path"])
            target.mkdir(mode=entry["mode"], parents=True)
            os.chmod(target, entry["mode"])
            created_directories.append(target)
        api_python = paths.root / "model-api-gguf" / "api_service" / ".venv" / "bin" / "python"
        source_root = paths.root / "model-api-gguf" / "api_service" / "src"
        if not api_python.is_file():
            raise BootstrapError(ErrorCode.PRECONDITION_FAILED, "GGUF API private environment is not ready")
        script = (
            "import asyncio,json,sys;from pathlib import Path;"
            f"sys.path.insert(0,{str(source_root)!r});"
            "from system_x_gguf_api.registry_store import RegistryStore;"
            f"result=asyncio.run(RegistryStore(Path({str(database)!r}),5000).initialize());"
            "print(json.dumps({'schema_identity':result['schema_identity'],'schema_version':result['schema_version']}))"
        )
        result = command((str(api_python), "-B", "-I", "-s", "-c", script), timeout=180)
        if result.returncode != 0:
            raise BootstrapError(ErrorCode.EXTERNAL_COMMAND_FAILED, "empty registry initialization failed")
        verification = verify_empty_registry(paths, contract)
        _write_exclusive(marker_path, canonical_json_bytes(_marker(contract)), 0o600)
        transaction.record("runtime-ready", {"entry_count": len(entries), "registry": verification})
        return {"changed": True, "status": "ready", "entry_count": len(entries), "registry": verification}
    except BaseException:
        for candidate in database_candidates:
            try:
                if candidate.is_file() and not candidate.is_symlink():
                    candidate.unlink()
            except OSError:
                pass
        try:
            if marker_path.is_file() and not marker_path.is_symlink():
                marker_path.unlink()
        except OSError:
            pass
        for directory in reversed(created_directories):
            try:
                directory.rmdir()
            except OSError:
                pass
        raise
