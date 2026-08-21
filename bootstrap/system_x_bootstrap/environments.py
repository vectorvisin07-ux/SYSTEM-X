"""Exact CPython 3.14 private-environment reconstruction."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .command import Runner, SubprocessRunner, require_success
from .config import canonical_json_bytes, canonical_sha256
from .errors import BootstrapError, ErrorCode
from .paths import RepositoryPaths, resolve_contained
from .transaction import BootstrapTransaction


MARKER_NAME = ".system-x-bootstrap-environment.json"
_HASH = re.compile(r"^[0-9a-f]{64}$")


def _normalized_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def validate_environment_lock(paths: RepositoryPaths, lock: Mapping[str, Any]) -> None:
    python = lock.get("python", {})
    if python.get("implementation") != "CPython" or python.get("major_minor") != "3.14":
        raise BootstrapError(ErrorCode.BOOTSTRAP_ENVIRONMENT_LOCK_MISSING, "environment lock is not bound to CPython 3.14")
    environments = lock.get("environments")
    if not isinstance(environments, list) or not environments:
        raise BootstrapError(ErrorCode.BOOTSTRAP_ENVIRONMENT_LOCK_MISSING, "no private environments are locked")
    identities: set[str] = set()
    destinations: set[str] = set()
    for environment in environments:
        identity = environment.get("environment_identity")
        destination = environment.get("relative_destination")
        if not isinstance(identity, str) or identity in identities or not isinstance(destination, str) or destination in destinations:
            raise BootstrapError(ErrorCode.BOOTSTRAP_ENVIRONMENT_LOCK_MISSING, "environment identity or destination is invalid")
        identities.add(identity)
        destinations.add(destination)
        resolve_contained(paths.root, destination, allow_missing=True)
        if environment.get("system_site_packages") is not False or environment.get("user_site") != "disabled":
            raise BootstrapError(ErrorCode.CONFIGURATION_INVALID, "private environment isolation is not closed")
        if not environment.get("resolved_lock"):
            raise BootstrapError(ErrorCode.BOOTSTRAP_ENVIRONMENT_LOCK_MISSING, "authoritative environment lock is absent")
        artifacts = environment.get("artifacts")
        if not isinstance(artifacts, list):
            raise BootstrapError(ErrorCode.BOOTSTRAP_ENVIRONMENT_LOCK_MISSING, "environment artifacts are not closed")
        artifact_names: set[str] = set()
        for artifact in artifacts:
            name = _normalized_name(artifact.get("name", ""))
            if (
                not name
                or name in artifact_names
                or not artifact.get("version")
                or artifact.get("source_domain") != lock["package_source_policy"]["accepted_artifact_domain"]
                or not _HASH.fullmatch(artifact.get("sha256", ""))
            ):
                raise BootstrapError(ErrorCode.BOOTSTRAP_ENVIRONMENT_LOCK_MISSING, "artifact lock record is invalid")
            artifact_names.add(name)
        for declaration in environment.get("dependency_declarations", []):
            path = resolve_contained(paths.root, declaration["path"], allow_missing=False)
            if hashlib.sha256(path.read_bytes()).hexdigest() != declaration["sha256"]:
                raise BootstrapError(
                    ErrorCode.INTEGRITY_FAILURE,
                    "environment dependency declaration changed",
                    context={"path": declaration["path"]},
                )
        lock_declaration = next(
            (item for item in environment.get("dependency_declarations", []) if item["path"].endswith("requirements.lock")),
            None,
        )
        if lock_declaration is not None:
            declaration_path = resolve_contained(paths.root, lock_declaration["path"], allow_missing=False)
            declared_names: set[str] = set()
            for line in declaration_path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped and not stripped.startswith("#") and "==" in stripped:
                    declared_names.add(_normalized_name(stripped.split("==", 1)[0]))
            if declared_names != artifact_names:
                raise BootstrapError(ErrorCode.BOOTSTRAP_ENVIRONMENT_LOCK_MISSING, "resolved pins and artifact hashes differ")


def render_hashed_requirements(environment: Mapping[str, Any]) -> str:
    artifacts = environment.get("artifacts", [])
    lines = [
        f"{item['name']}=={item['version']} --hash=sha256:{item['sha256']}"
        for item in sorted(artifacts, key=lambda record: _normalized_name(record["name"]))
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def _environment_marker(lock: Mapping[str, Any], environment: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": "system-x.bootstrap.environment-marker.v1",
        "version": 1,
        "environment_identity": environment["environment_identity"],
        "environment_lock_identity": lock["identity"],
        "environment_record_sha256": canonical_sha256(environment),
        "python_major_minor": "3.14",
        "system_site_packages": False,
        "user_site": "disabled",
    }


def environment_status(paths: RepositoryPaths, lock: Mapping[str, Any], environment: Mapping[str, Any]) -> str:
    destination = resolve_contained(paths.root, environment["relative_destination"], allow_missing=True)
    if not destination.exists():
        return "absent"
    if destination.is_symlink() or not destination.is_dir():
        return "collision"
    marker_path = destination / MARKER_NAME
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError, OSError):
        return "unknown-existing"
    return "ready" if marker == _environment_marker(lock, environment) else "mismatch"


def _write_marker(destination: Path, marker: Mapping[str, Any]) -> None:
    path = destination / MARKER_NAME
    payload = canonical_json_bytes(marker)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)


def _verify_imports(
    paths: RepositoryPaths,
    destination: Path,
    environment: Mapping[str, Any],
    runner: Runner,
) -> None:
    python = destination / "bin" / "python"
    source_paths = [str(resolve_contained(paths.root, item, allow_missing=False)) for item in environment["source_paths"]]
    imports = list(environment["post_install_import_roots"])
    script = (
        "import importlib,json,site,sys;"
        f"assert sys.version_info[:2] == (3,14);"
        f"assert {source_paths!r} == [str(x) for x in {source_paths!r}];"
        f"[sys.path.insert(0,p) for p in reversed({source_paths!r})];"
        f"[importlib.import_module(n) for n in {imports!r}];"
        "assert site.ENABLE_USER_SITE is False;"
        "print(json.dumps({'imports':'passed','python':list(sys.version_info[:3])}))"
    )
    flags = ("-B", "-I", "-s")
    require_success(
        runner((str(python), *flags, "-c", script), timeout=120),
        purpose="private environment post-install import proof failed",
    )


def build_environments(
    paths: RepositoryPaths,
    lock: Mapping[str, Any],
    *,
    transaction: BootstrapTransaction,
    authorized: bool,
    runner: Runner | None = None,
) -> dict[str, Any]:
    """Build every locked environment; never replaces an unknown environment."""

    if not authorized:
        raise BootstrapError(ErrorCode.AUTHORIZATION_REQUIRED, "build-environments requires explicit authorization")
    if not getattr(transaction, "_entered", False):
        raise BootstrapError(ErrorCode.TRANSACTION_RECOVERY_REQUIRED, "build-environments requires an active transaction")
    validate_environment_lock(paths, lock)
    command = runner or SubprocessRunner()
    results: list[dict[str, Any]] = []
    for environment in lock["environments"]:
        status = environment_status(paths, lock, environment)
        if status == "ready":
            results.append({"identity": environment["environment_identity"], "changed": False, "status": "ready"})
            continue
        if status != "absent":
            raise BootstrapError(
                ErrorCode.RUNTIME_COLLISION,
                "private environment destination already contains unknown state",
                context={"environment": environment["environment_identity"], "status": status},
            )
        destination = transaction.claim_created_path(environment["relative_destination"])
        created = False
        try:
            arguments = ["python3.14", "-B", "-I", "-S", "-m", "venv"]
            if not environment["artifacts"]:
                arguments.append("--without-pip")
            arguments.append(str(destination))
            creation = command(tuple(arguments), timeout=300)
            created = destination.exists()
            require_success(creation, purpose="CPython 3.14 venv creation failed")
            os.chmod(destination, 0o700)
            if environment["artifacts"]:
                requirements = render_hashed_requirements(environment)
                with tempfile.NamedTemporaryFile("w", encoding="utf-8", prefix="system-x-requirements-", suffix=".txt") as handle:
                    handle.write(requirements)
                    handle.flush()
                    pip = destination / "bin" / "python"
                    policy = lock["package_source_policy"]
                    require_success(
                        command(
                            (
                                str(pip), "-B", "-I", "-s", "-m", "pip", "install",
                                "--require-hashes", "--only-binary=:all:", "--no-deps", "--no-cache-dir",
                                "--disable-pip-version-check", "--index-url", policy["index_url"], "-r", handle.name,
                            ),
                            env={"PIP_NO_CACHE_DIR": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1"},
                            timeout=1800,
                        ),
                        purpose="hash-locked private environment installation failed",
                    )
            _verify_imports(paths, destination, environment, command)
            _write_marker(destination, _environment_marker(lock, environment))
            transaction.record("environment-ready", {"identity": environment["environment_identity"]})
            results.append({"identity": environment["environment_identity"], "changed": True, "status": "ready"})
        except BaseException:
            if (created or destination.exists()) and destination.is_dir() and not destination.is_symlink():
                shutil.rmtree(destination)
            raise
    return {"changed": any(item["changed"] for item in results), "environments": results}
