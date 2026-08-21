"""Exact vendored llama.cpp verification and model-free CUDA build."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .command import Runner, SubprocessRunner, require_success
from .errors import BootstrapError, ErrorCode
from .paths import RepositoryPaths, resolve_contained
from .transaction import BootstrapTransaction


MAX_SOURCE_IDENTITY_BYTES = 16 * 1024 * 1024
SHA256_HEX_LENGTH = 64


def _canonical_manifest(files: list[dict[str, Any]]) -> bytes:
    return (json.dumps(files, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _source_payload(path: Path, info: os.stat_result) -> bytes:
    if stat.S_ISLNK(info.st_mode):
        return os.readlink(path).encode("utf-8", "surrogateescape")
    if stat.S_ISREG(info.st_mode):
        return path.read_bytes()
    raise BootstrapError(ErrorCode.INTEGRITY_FAILURE, "vendored llama.cpp contains an unsupported file type")


def _source_files(root: Path, *, excluded: Path | None = None) -> dict[str, tuple[Path, os.stat_result]]:
    result: dict[str, tuple[Path, os.stat_result]] = {}
    stack = [root]
    while stack:
        directory = stack.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as exc:
            raise BootstrapError(ErrorCode.INTEGRITY_FAILURE, "vendored llama.cpp is unreadable") from exc
        for entry in entries:
            path = Path(entry.path)
            info = path.lstat()
            relative = path.relative_to(root).as_posix()
            if stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode):
                if excluded is not None and path == excluded:
                    continue
                stack.append(path)
            else:
                result[relative] = (path, info)
    return result


def _not_exact(reason: str, *, identity: Mapping[str, Any] | None = None) -> dict[str, Any]:
    value = identity or {}
    return {
        "present": False,
        "mode": "vendored",
        "state": "VENDORED_SOURCE_MISMATCH",
        "reason": reason,
        "origin": value.get("origin"),
        "tag": value.get("tag"),
        "commit": value.get("commit"),
        "tree": value.get("upstream_tree"),
        "file_count": value.get("tracked_file_count"),
        "byte_count": value.get("tracked_byte_count"),
        "manifest_sha256": value.get("complete_vendored_manifest_sha256"),
        "network_used": False,
        "exact": False,
    }


def inspect_vendored_source(
    paths: RepositoryPaths,
    lock: Mapping[str, Any],
    runner: Runner | None = None,
) -> dict[str, Any]:
    """Verify the complete ordinary source tree without invoking Git or network."""

    del runner  # Compatibility-only argument; vendored verification is filesystem-only.
    source = lock["source"]
    if source.get("mode") != "vendored":
        return _not_exact("source lock mode is not vendored")
    checkout = resolve_contained(paths.root, source["path"], allow_missing=True, reject_symlinks=True)
    identity_path = resolve_contained(paths.root, source["identity_record"], allow_missing=True, reject_symlinks=True)
    if not checkout.is_dir():
        return _not_exact("vendored source directory is absent")
    if (checkout / ".git").exists() or (checkout / ".git").is_symlink():
        return _not_exact("unexpected nested .git")
    try:
        locked_build = resolve_contained(
            paths.root, lock["build_directory"], allow_missing=True, reject_symlinks=True
        )
    except (BootstrapError, OSError):
        return _not_exact("locked build directory is unsafe")
    expected_build = checkout / "build"
    if locked_build != expected_build:
        return _not_exact("locked build directory is not the vendored build path")
    if locked_build.exists() and (locked_build.is_symlink() or not locked_build.is_dir()):
        return _not_exact("unexpected build output in vendored source")
    if not identity_path.is_file() or identity_path.is_symlink():
        return _not_exact("source identity record is absent or unsafe")
    try:
        raw_identity = identity_path.read_bytes()
    except OSError:
        return _not_exact("source identity record is unreadable")
    if not raw_identity or len(raw_identity) > MAX_SOURCE_IDENTITY_BYTES or b"\0" in raw_identity:
        return _not_exact("source identity record byte envelope is invalid")
    try:
        identity = json.loads(raw_identity.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return _not_exact("source identity record is not strict UTF-8 JSON")
    if not isinstance(identity, dict):
        return _not_exact("source identity record root is not an object")
    required_identity = {
        "schema", "version", "origin", "tag", "commit", "upstream_tree", "tracked_file_count",
        "tracked_byte_count", "complete_vendored_manifest_sha256", "license_identities",
        "source_patch_count", "build_output_excluded", "files",
    }
    if set(identity) != required_identity:
        return _not_exact("source identity record fields are not closed", identity=identity)
    expected_identity = {
        "schema": "system-x.llama-cpp-source-identity.v1",
        "version": 1,
        "origin": source["origin"],
        "tag": source["tag"],
        "commit": source["commit"],
        "upstream_tree": source["tree"],
        "source_patch_count": 0,
        "build_output_excluded": True,
    }
    if any(identity.get(key) != value for key, value in expected_identity.items()):
        return _not_exact("source identity record does not match the lock", identity=identity)
    files = identity.get("files")
    if not isinstance(files, list) or len(files) != identity.get("tracked_file_count"):
        return _not_exact("source identity file count is invalid", identity=identity)
    if hashlib.sha256(_canonical_manifest(files)).hexdigest() != identity.get("complete_vendored_manifest_sha256"):
        return _not_exact("source identity manifest hash is invalid", identity=identity)

    expected: dict[str, dict[str, Any]] = {}
    total_bytes = 0
    for record in files:
        if not isinstance(record, dict) or set(record) != {"path", "mode", "bytes", "sha256", "git_blob_oid"}:
            return _not_exact("vendored manifest record fields are invalid", identity=identity)
        relative = record.get("path")
        if not isinstance(relative, str) or not relative:
            return _not_exact("vendored manifest path is invalid", identity=identity)
        portable = PurePosixPath(relative)
        if portable.is_absolute() or ".." in portable.parts or relative != portable.as_posix() or relative in expected:
            return _not_exact("vendored manifest path is unsafe or duplicated", identity=identity)
        if portable.parts[0] in {".git", "build"}:
            return _not_exact("vendored manifest contains forbidden generated/Git state", identity=identity)
        if record.get("mode") not in {"100644", "100755", "120000"}:
            return _not_exact("vendored manifest mode is invalid", identity=identity)
        if type(record.get("bytes")) is not int or record["bytes"] < 0:
            return _not_exact("vendored manifest byte count is invalid", identity=identity)
        if not isinstance(record.get("sha256"), str) or len(record["sha256"]) != SHA256_HEX_LENGTH:
            return _not_exact("vendored manifest SHA-256 is invalid", identity=identity)
        expected[relative] = record
        total_bytes += record["bytes"]
    if total_bytes != identity.get("tracked_byte_count"):
        return _not_exact("vendored manifest total byte count is invalid", identity=identity)

    actual = _source_files(checkout, excluded=locked_build if locked_build.is_dir() else None)
    if set(actual) != set(expected):
        return _not_exact("vendored source has missing or extra files", identity=identity)
    for relative, record in expected.items():
        path, info = actual[relative]
        mode = "120000" if stat.S_ISLNK(info.st_mode) else ("100755" if stat.S_IMODE(info.st_mode) & 0o111 else "100644")
        if mode != record["mode"]:
            return _not_exact("vendored source mode mismatch", identity=identity)
        try:
            raw = _source_payload(path, info)
        except (OSError, UnicodeError):
            return _not_exact("vendored source content is unreadable", identity=identity)
        if len(raw) != record["bytes"] or hashlib.sha256(raw).hexdigest() != record["sha256"]:
            return _not_exact("vendored source content mismatch", identity=identity)
    return {
        "present": True,
        "mode": "vendored",
        "state": "VENDORED_SOURCE_VERIFIED",
        "reason": None,
        "origin": identity["origin"],
        "tag": identity["tag"],
        "commit": identity["commit"],
        "tree": identity["upstream_tree"],
        "file_count": identity["tracked_file_count"],
        "byte_count": identity["tracked_byte_count"],
        "manifest_sha256": identity["complete_vendored_manifest_sha256"],
        "identity_sha256": hashlib.sha256(raw_identity).hexdigest(),
        "network_used": False,
        "exact": True,
    }


def inspect_submodule(paths: RepositoryPaths, lock: Mapping[str, Any], runner: Runner | None = None) -> dict[str, Any]:
    """Compatibility alias for callers migrating from the former provider name."""

    return inspect_vendored_source(paths, lock, runner)


def initialize_submodules(
    paths: RepositoryPaths,
    lock: Mapping[str, Any],
    *,
    transaction: BootstrapTransaction,
    authorized: bool,
    runner: Runner | None = None,
) -> dict[str, Any]:
    """Compatibility no-op that verifies vendored source and never contacts GitHub."""

    if not authorized:
        raise BootstrapError(ErrorCode.AUTHORIZATION_REQUIRED, "initialize-submodules requires explicit authorization")
    if not getattr(transaction, "_entered", False):
        raise BootstrapError(ErrorCode.TRANSACTION_RECOVERY_REQUIRED, "initialize-submodules requires an active transaction")
    source = inspect_vendored_source(paths, lock, runner)
    if not source["exact"]:
        raise BootstrapError(ErrorCode.INTEGRITY_FAILURE, "vendored llama.cpp source verification failed", context={"reason": source["reason"]})
    transaction.record("vendored-source-verified", {"path": lock["source"]["path"], "commit": source["commit"]})
    return {"changed": False, "state": "VENDORED_SOURCE_VERIFIED", "source": source}


def cmake_arguments(paths: RepositoryPaths, lock: Mapping[str, Any]) -> tuple[str, ...]:
    source = resolve_contained(paths.root, lock["source"]["path"], allow_missing=False)
    build = resolve_contained(paths.root, lock["build_directory"], allow_missing=True)
    options = tuple(f"-D{key}={value}" for key, value in sorted(lock["cmake_options"].items()))
    return ("cmake", "-S", str(source), "-B", str(build), "-G", lock["generator"], *options)


def _cache_matches(build: Path, lock: Mapping[str, Any]) -> bool:
    cache = build / "CMakeCache.txt"
    if not cache.is_file() or cache.is_symlink():
        return False
    try:
        text = cache.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    for key, value in lock["cmake_options"].items():
        if not any(line.startswith(f"{key}:") and line.rsplit("=", 1)[-1] == value for line in text.splitlines()):
            return False
    return True


def _llama_probe_environment(binary: Path) -> dict[str, str]:
    """Resolve the build-local shared libraries for every llama probe."""

    library_path = str(binary.parent)
    existing = os.environ.get("LD_LIBRARY_PATH")
    if existing:
        library_path = os.pathsep.join((library_path, existing))
    return {"LD_LIBRARY_PATH": library_path}


def verify_llama_no_model(
    paths: RepositoryPaths,
    lock: Mapping[str, Any],
    runner: Runner | None = None,
) -> dict[str, Any]:
    command = runner or SubprocessRunner()
    binary = resolve_contained(paths.root, lock["binary"], allow_missing=False)
    if not binary.is_file() or binary.is_symlink() or not os.access(binary, os.X_OK):
        raise BootstrapError(ErrorCode.PRECONDITION_FAILED, "llama-server binary is absent or not executable")
    probe_environment = _llama_probe_environment(binary)
    version = require_success(command((str(binary), "--version"), env=probe_environment, timeout=60), purpose="llama-server version probe failed")
    devices = require_success(command((str(binary), "--list-devices"), env=probe_environment, timeout=60), purpose="llama-server CUDA device probe failed")
    if "CUDA0" not in devices.stdout + devices.stderr:
        raise BootstrapError(ErrorCode.PRECONDITION_FAILED, "llama-server does not expose CUDA0")
    libraries = require_success(command(("ldd", str(binary)), env=probe_environment, timeout=60), purpose="llama-server dynamic-library probe failed")
    if "not found" in libraries.stdout:
        raise BootstrapError(ErrorCode.PRECONDITION_FAILED, "llama-server has unresolved dynamic libraries")
    return {
        "version_probe": version.returncode,
        "list_devices_probe": devices.returncode,
        "cuda0_visible": True,
        "dynamic_libraries_resolved": True,
        "model_loaded": False,
        "api_called": False,
    }


def build_llama_server(
    paths: RepositoryPaths,
    lock: Mapping[str, Any],
    *,
    transaction: BootstrapTransaction,
    authorized: bool,
    runner: Runner | None = None,
) -> dict[str, Any]:
    if not authorized:
        raise BootstrapError(ErrorCode.AUTHORIZATION_REQUIRED, "build-llama-server requires explicit authorization")
    if not getattr(transaction, "_entered", False):
        raise BootstrapError(ErrorCode.TRANSACTION_RECOVERY_REQUIRED, "build-llama-server requires an active transaction")
    command = runner or SubprocessRunner()
    source = inspect_vendored_source(paths, lock)
    if not source["exact"]:
        raise BootstrapError(ErrorCode.INTEGRITY_FAILURE, "vendored llama.cpp source is not exact", context={"reason": source["reason"]})
    build = resolve_contained(paths.root, lock["build_directory"], allow_missing=True)
    binary = resolve_contained(paths.root, lock["binary"], allow_missing=True)
    if build.exists():
        if _cache_matches(build, lock) and binary.is_file():
            return {"changed": False, "verification": verify_llama_no_model(paths, lock, command)}
        raise BootstrapError(ErrorCode.BUILD_COLLISION, "existing llama.cpp build directory is not the locked build")
    transaction.claim_created_path(lock["build_directory"])
    created = False
    try:
        configure = command(cmake_arguments(paths, lock), timeout=600)
        created = build.exists()
        require_success(configure, purpose="locked llama.cpp CMake configure failed")
        require_success(
            command(("cmake", "--build", str(build), "--config", lock["build_type"], "--target", lock["target"], "--parallel"), timeout=3600),
            purpose="locked CUDA llama-server build failed",
        )
        if not _cache_matches(build, lock):
            raise BootstrapError(ErrorCode.INTEGRITY_FAILURE, "CMake cache does not round-trip the locked profile")
        verification = verify_llama_no_model(paths, lock, command)
        transaction.record("llama-server-built", {"commit": source["commit"], "cuda0_visible": True})
        return {"changed": True, "verification": verification}
    except BaseException:
        if (created or build.exists()) and build.is_dir() and not build.is_symlink():
            shutil.rmtree(build)
        raise
