"""Accepted Linux systemd-user adapter invocation and filesystem-only status."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from .command import Runner, SubprocessRunner
from .config import canonical_json_bytes
from .errors import BootstrapError, ErrorCode
from .paths import RepositoryPaths, resolve_contained
from .transaction import BootstrapTransaction


def verify_service_source_contract(paths: RepositoryPaths, contract: Mapping[str, Any]) -> None:
    for relative, expected in contract["source_contract_sha256"].items():
        source = resolve_contained(paths.root, relative, allow_missing=False)
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise BootstrapError(ErrorCode.INTEGRITY_FAILURE, "service source contract changed", context={"path": relative})


def render_operating_profile(contract: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(contract["operating_profile"]["template"])


def service_status(
    paths: RepositoryPaths,
    contract: Mapping[str, Any],
    *,
    home: Path | None = None,
) -> dict[str, Any]:
    selected_home = (home or Path.home()).resolve(strict=True)
    unit = selected_home / contract["future_generated_unit_relative_to_home"]
    link = selected_home / contract["future_generated_enablement_link_relative_to_home"]
    adapter_root = resolve_contained(paths.root, contract["adapter"]["runtime_root"], allow_missing=True)
    manifest = adapter_root / "linux-systemd-user" / "manifest.json"
    profile = resolve_contained(paths.root, contract["operating_profile"]["path"], allow_missing=True)
    desired = resolve_contained(paths.root, contract["desired_state"]["path"], allow_missing=True)
    return {
        "unit_present": unit.is_file() and not unit.is_symlink(),
        "enablement_link_present": link.is_symlink(),
        "adapter_manifest_present": manifest.is_file() and not manifest.is_symlink(),
        "operating_profile_present": profile.is_file() and not profile.is_symlink(),
        "desired_state_present": desired.is_file() and not desired.is_symlink(),
        "service_control_invoked": False,
        "query_method": "filesystem-only",
    }


def _write_exclusive(path: Path, payload: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)


def _run_json(runner: Runner, argv: tuple[str, ...], purpose: str) -> dict[str, Any]:
    result = runner(argv, timeout=180)
    if result.returncode != 0:
        raise BootstrapError(ErrorCode.EXTERNAL_COMMAND_FAILED, purpose)
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise BootstrapError(ErrorCode.INTEGRITY_FAILURE, f"{purpose}: result is not JSON") from exc
    if not isinstance(value, dict):
        raise BootstrapError(ErrorCode.INTEGRITY_FAILURE, f"{purpose}: result envelope is invalid")
    return value


def register_platform_service(
    paths: RepositoryPaths,
    contract: Mapping[str, Any],
    *,
    transaction: BootstrapTransaction,
    authorized: bool,
    runner: Runner | None = None,
    home: Path | None = None,
) -> dict[str, Any]:
    if not authorized:
        raise BootstrapError(ErrorCode.AUTHORIZATION_REQUIRED, "register-platform-service requires explicit authorization")
    if not getattr(transaction, "_entered", False):
        raise BootstrapError(ErrorCode.TRANSACTION_RECOVERY_REQUIRED, "service registration requires an active transaction")
    verify_service_source_contract(paths, contract)
    command = runner or SubprocessRunner()
    selected_home = (home or Path.home()).resolve(strict=True)
    before = service_status(paths, contract, home=selected_home)
    if before["unit_present"] or before["adapter_manifest_present"]:
        if before["adapter_manifest_present"]:
            if not all(before[key] for key in ("unit_present", "operating_profile_present", "desired_state_present")):
                raise BootstrapError(ErrorCode.RUNTIME_COLLISION, "partial or unknown service registration already exists")
            profile = resolve_contained(paths.root, contract["operating_profile"]["path"], allow_missing=False)
            desired = resolve_contained(paths.root, contract["desired_state"]["path"], allow_missing=False)
        else:
            profile = resolve_contained(paths.root, contract["operating_profile"]["path"], allow_missing=True)
            desired = resolve_contained(paths.root, contract["desired_state"]["path"], allow_missing=True)
            if profile.exists() != desired.exists():
                raise BootstrapError(ErrorCode.RUNTIME_COLLISION, "partial or unknown service registration already exists")
            if not profile.exists():
                _write_exclusive(profile, render_operating_profile(contract))
                python = "python3.14"
                profile_source = resolve_contained(paths.root, "model-api-gguf/service_control/operating_profile.py", allow_missing=False)
                validate = _run_json(
                    command,
                    (python, "-B", "-I", "-S", str(profile_source), "validate-profile", "--profile", str(profile)),
                    "operating profile validation failed",
                )
                if validate.get("ok") is not True:
                    raise BootstrapError(ErrorCode.INTEGRITY_FAILURE, "operating profile was rejected")
                initialize = _run_json(
                    command,
                    (
                        python, "-B", "-I", "-S", str(profile_source), "initialize-desired-state", "--profile", str(profile),
                        "--desired-state-path", str(desired), "--state", contract["desired_state"]["initial"],
                    ),
                    "initial STOPPED desired-state creation failed",
                )
                if initialize.get("ok") is not True or initialize.get("desired_state") != "STOPPED":
                    raise BootstrapError(ErrorCode.INTEGRITY_FAILURE, "initial desired state is not STOPPED")
        python = "python3.14"
        adapter = resolve_contained(paths.root, contract["adapter"]["source_entrypoint"], allow_missing=False)
        runtime_root = resolve_contained(paths.root, contract["adapter"]["runtime_root"], allow_missing=False)
        supervisor_root = resolve_contained(paths.root, contract["supervisor"]["runtime_root"], allow_missing=False)
        supervisor = resolve_contained(paths.root, contract["supervisor"]["entrypoint"], allow_missing=False)
        reconciled = _run_json(command, (python, "-B", "-I", "-S", str(adapter), "--adapter-runtime-root", str(runtime_root), "register", "--profile", str(profile), "--state", str(desired), "--supervisor-runtime-root", str(supervisor_root), "--supervisor-entrypoint", str(supervisor)), "Linux systemd-user adapter registration reconciliation failed")
        if reconciled.get("ok") is not True or reconciled.get("registered") is not True:
            raise BootstrapError(ErrorCode.INTEGRITY_FAILURE, "product adapter reconciliation returned an unexpected identity")
        after = service_status(paths, contract, home=selected_home)
        if not all(after[key] for key in ("unit_present", "adapter_manifest_present", "operating_profile_present", "desired_state_present")):
            raise BootstrapError(ErrorCode.INTEGRITY_FAILURE, "reconciled service registration is physically incomplete")
        transaction.record("service-registration-reconciled", {"adapter_identity": reconciled["adapter_identity"], "service_name": contract["adapter"]["service_name"], "product_owned": True})
        return {"changed": True, "state": "registered", "service_control_invoked": True, "reconciled_existing_registration": True, "adapter_identity": reconciled["adapter_identity"]}



    profile = resolve_contained(paths.root, contract["operating_profile"]["path"], allow_missing=True)
    desired = resolve_contained(paths.root, contract["desired_state"]["path"], allow_missing=True)
    if profile.exists() or desired.exists():
        raise BootstrapError(ErrorCode.RUNTIME_COLLISION, "operating profile or desired state already exists")
    _write_exclusive(profile, render_operating_profile(contract))

    python = "python3.14"
    profile_source = resolve_contained(paths.root, "model-api-gguf/service_control/operating_profile.py", allow_missing=False)
    validate = _run_json(
        command,
        (python, "-B", "-I", "-S", str(profile_source), "validate-profile", "--profile", str(profile)),
        "operating profile validation failed",
    )
    if validate.get("ok") is not True:
        raise BootstrapError(ErrorCode.INTEGRITY_FAILURE, "operating profile was rejected")
    initialize = _run_json(
        command,
        (
            python, "-B", "-I", "-S", str(profile_source), "initialize-desired-state", "--profile", str(profile),
            "--desired-state-path", str(desired), "--state", contract["desired_state"]["initial"],
        ),
        "initial STOPPED desired-state creation failed",
    )
    if initialize.get("ok") is not True or initialize.get("desired_state") != "STOPPED":
        raise BootstrapError(ErrorCode.INTEGRITY_FAILURE, "initial desired state is not STOPPED")

    adapter = resolve_contained(paths.root, contract["adapter"]["source_entrypoint"], allow_missing=False)
    runtime_root = resolve_contained(paths.root, contract["adapter"]["runtime_root"], allow_missing=False)
    supervisor_root = resolve_contained(paths.root, contract["supervisor"]["runtime_root"], allow_missing=False)
    supervisor = resolve_contained(paths.root, contract["supervisor"]["entrypoint"], allow_missing=False)
    result = _run_json(
        command,
        (
            python, "-B", "-I", "-S", str(adapter), "--adapter-runtime-root", str(runtime_root), "register",
            "--profile", str(profile), "--state", str(desired), "--supervisor-runtime-root", str(supervisor_root),
            "--supervisor-entrypoint", str(supervisor),
        ),
        "Linux systemd-user adapter registration failed",
    )
    if (
        result.get("ok") is not True
        or result.get("adapter_identity") != contract["adapter"]["identity"]
        or result.get("adapter_version") != contract["adapter"]["version"]
        or result.get("registered") is not True
    ):
        raise BootstrapError(ErrorCode.INTEGRITY_FAILURE, "platform adapter returned an unexpected registration identity")
    after = service_status(paths, contract, home=selected_home)
    if not all(
        after[key]
        for key in (
            "unit_present",
            "adapter_manifest_present",
            "operating_profile_present",
            "desired_state_present",
        )
    ):
        raise BootstrapError(ErrorCode.INTEGRITY_FAILURE, "platform adapter result lacks complete physical registration state")
    transaction.record(
        "service-registered",
        {"adapter_identity": result["adapter_identity"], "service_name": contract["adapter"]["service_name"], "initial_state": "STOPPED"},
    )
    return {
        "changed": True,
        "state": "registered",
        "adapter_identity": result["adapter_identity"],
        "initial_desired_state": "STOPPED",
        "enabled": bool(result.get("enabled")),
        "active": bool(result.get("active")),
    }


def activate_platform_service(
    paths: RepositoryPaths,
    contract: Mapping[str, Any],
    *,
    transaction: BootstrapTransaction,
    authorized: bool,
    runner: Runner | None = None,
    home: Path | None = None,
) -> dict[str, Any]:
    """Enable and start the registered service through its native adapter."""

    if not authorized:
        raise BootstrapError(ErrorCode.AUTHORIZATION_REQUIRED, "activate-platform-service requires explicit authorization")
    if not getattr(transaction, "_entered", False):
        raise BootstrapError(ErrorCode.TRANSACTION_RECOVERY_REQUIRED, "service activation requires an active transaction")
    verify_service_source_contract(paths, contract)
    command = runner or SubprocessRunner()
    selected_home = (home or Path.home()).resolve(strict=True)
    before = service_status(paths, contract, home=selected_home)
    required = ("unit_present", "adapter_manifest_present", "operating_profile_present", "desired_state_present")
    if not all(before[key] for key in required):
        raise BootstrapError(ErrorCode.PRECONDITION_FAILED, "service activation requires complete registration")

    python = "python3.14"
    adapter = resolve_contained(paths.root, contract["adapter"]["source_entrypoint"], allow_missing=False)
    runtime_root = resolve_contained(paths.root, contract["adapter"]["runtime_root"], allow_missing=False)
    observed = _run_json(command, (python, "-B", "-I", "-S", str(adapter), "--adapter-runtime-root", str(runtime_root), "status"), "Linux systemd-user adapter status probe failed")
    registered = observed.get("registered")
    if (
        observed.get("ok") is not True
        or registered is False
        or (
            registered is not True
            and not all(key in observed for key in ("enabled", "active"))
        )
    ):
        raise BootstrapError(ErrorCode.INTEGRITY_FAILURE, "platform adapter status did not report a registered service")
    if observed.get("active"):
        _run_json(command, (python, "-B", "-I", "-S", str(adapter), "--adapter-runtime-root", str(runtime_root), "stop"), "Linux systemd-user adapter pre-activation stop failed")
        observed = _run_json(command, (python, "-B", "-I", "-S", str(adapter), "--adapter-runtime-root", str(runtime_root), "status"), "Linux systemd-user adapter post-stop status probe failed")
        registered = observed.get("registered")
    if registered is True and observed.get("enabled"):
        enable = observed
    else:
        enable = _run_json(command, (python, "-B", "-I", "-S", str(adapter), "--adapter-runtime-root", str(runtime_root), "enable"), "Linux systemd-user adapter enablement failed")
    if enable.get("ok") is not True or enable.get("enabled") is not True or enable.get("active") is True:
        raise BootstrapError(ErrorCode.INTEGRITY_FAILURE, "platform adapter did not report enabled inactive state")
    start = _run_json(
        command,
        (python, "-B", "-I", "-S", str(adapter), "--adapter-runtime-root", str(runtime_root), "start"),
        "Linux systemd-user adapter activation failed",
    )
    if start.get("ok") is not True or start.get("enabled") is not True or start.get("active") is not True:
        raise BootstrapError(ErrorCode.INTEGRITY_FAILURE, "platform adapter did not report active state")
    transaction.record(
        "service-activated",
        {"enabled": True, "active": True, "adapter_identity": contract["adapter"]["identity"]},
    )
    return {
        "changed": True,
        "state": "active",
        "enabled": True,
        "active": True,
        "adapter_identity": contract["adapter"]["identity"],
    }
