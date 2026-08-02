"""Exact llama.cpp submodule materialization and model-free CUDA build."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping

from .command import Runner, SubprocessRunner, require_success
from .errors import BootstrapError, ErrorCode
from .paths import RepositoryPaths, resolve_contained
from .transaction import BootstrapTransaction


def inspect_submodule(paths: RepositoryPaths, lock: Mapping[str, Any], runner: Runner | None = None) -> dict[str, Any]:
    command = runner or SubprocessRunner()
    source = lock["source"]
    checkout = resolve_contained(paths.root, source["path"], allow_missing=True, reject_symlinks=True)
    if not checkout.is_dir():
        return {"present": False, "origin": None, "commit": None, "clean": None, "exact": False}
    origin = command(("git", "-C", str(checkout), "remote", "get-url", "origin"), timeout=30)
    commit = command(("git", "-C", str(checkout), "rev-parse", "HEAD"), timeout=30)
    status = command(("git", "-C", str(checkout), "status", "--porcelain=v1", "--untracked-files=all"), timeout=30)
    origin_value = origin.stdout.strip() if origin.returncode == 0 else None
    commit_value = commit.stdout.strip() if commit.returncode == 0 else None
    clean = status.returncode == 0 and not status.stdout
    return {
        "present": origin.returncode == 0 and commit.returncode == 0,
        "origin": origin_value,
        "commit": commit_value,
        "clean": clean,
        "exact": origin_value == source["origin"] and commit_value == source["commit"] and clean,
    }


def initialize_submodules(
    paths: RepositoryPaths,
    lock: Mapping[str, Any],
    *,
    transaction: BootstrapTransaction,
    authorized: bool,
    runner: Runner | None = None,
) -> dict[str, Any]:
    if not authorized:
        raise BootstrapError(ErrorCode.AUTHORIZATION_REQUIRED, "initialize-submodules requires explicit authorization")
    if not getattr(transaction, "_entered", False):
        raise BootstrapError(ErrorCode.TRANSACTION_RECOVERY_REQUIRED, "initialize-submodules requires an active transaction")
    command = runner or SubprocessRunner()
    before = inspect_submodule(paths, lock, command)
    if before["present"] and not before["clean"]:
        raise BootstrapError(ErrorCode.SUBMODULE_MISMATCH, "existing llama.cpp checkout is dirty")
    if before["present"] and before["origin"] != lock["source"]["origin"]:
        raise BootstrapError(ErrorCode.SUBMODULE_MISMATCH, "existing llama.cpp origin is not exact")
    if before["exact"]:
        return {"changed": False, "submodule": before}
    relative = lock["source"]["path"]
    require_success(
        command(("git", "-C", str(paths.root), "submodule", "update", "--init", "--recursive", "--checkout", "--", relative), timeout=1800),
        purpose="exact llama.cpp submodule materialization failed",
    )
    after = inspect_submodule(paths, lock, command)
    if not after["exact"]:
        raise BootstrapError(ErrorCode.SUBMODULE_MISMATCH, "materialized llama.cpp submodule does not match its lock")
    transaction.record("submodule-ready", {"path": relative, "commit": after["commit"]})
    return {"changed": True, "submodule": after}


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


def verify_llama_no_model(
    paths: RepositoryPaths,
    lock: Mapping[str, Any],
    runner: Runner | None = None,
) -> dict[str, Any]:
    command = runner or SubprocessRunner()
    binary = resolve_contained(paths.root, lock["binary"], allow_missing=False)
    if not binary.is_file() or binary.is_symlink() or not os.access(binary, os.X_OK):
        raise BootstrapError(ErrorCode.PRECONDITION_FAILED, "llama-server binary is absent or not executable")
    version = require_success(command((str(binary), "--version"), timeout=60), purpose="llama-server version probe failed")
    devices = require_success(command((str(binary), "--list-devices"), timeout=60), purpose="llama-server CUDA device probe failed")
    if "CUDA0" not in devices.stdout + devices.stderr:
        raise BootstrapError(ErrorCode.PRECONDITION_FAILED, "llama-server does not expose CUDA0")
    libraries = require_success(command(("ldd", str(binary)), timeout=60), purpose="llama-server dynamic-library probe failed")
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
    submodule = inspect_submodule(paths, lock, command)
    if not submodule["exact"]:
        raise BootstrapError(ErrorCode.SUBMODULE_MISMATCH, "llama.cpp submodule is not exact and clean")
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
        transaction.record("llama-server-built", {"commit": submodule["commit"], "cuda0_visible": True})
        return {"changed": True, "verification": verification}
    except BaseException:
        if (created or build.exists()) and build.is_dir() and not build.is_symlink():
            shutil.rmtree(build)
        raise
