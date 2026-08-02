"""Layered, model-safe bootstrap verification."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .command import Runner, SubprocessRunner
from .credentials import credential_status
from .environments import environment_status, validate_environment_lock
from .errors import BootstrapError, ErrorCode
from .host import HostInspector, cuda_toolkit_ready, host_blockers, python_ready
from .llama import inspect_submodule, verify_llama_no_model
from .paths import RepositoryPaths
from .runtime import expand_runtime_layout, runtime_status
from .service import service_status, verify_service_source_contract


VERIFICATION_LEVELS = (
    "source-only",
    "host-ready",
    "build-ready",
    "service-process-ready",
    "waiting-for-model",
    "full-ready",
)


def verify_source(
    paths: RepositoryPaths,
    configs: Mapping[str, Mapping[str, Any]],
    *,
    runner: Runner | None = None,
) -> dict[str, Any]:
    validate_environment_lock(paths, configs["python-environments.lock.json"])
    entries = expand_runtime_layout(paths, configs["runtime-layout.json"])
    verify_service_source_contract(paths, configs["service-registration.json"])
    submodule = inspect_submodule(paths, configs["llama-build.lock.json"], runner)
    if not submodule["exact"]:
        raise BootstrapError(ErrorCode.SUBMODULE_MISMATCH, "source-only verification requires the exact clean llama.cpp checkout")
    return {
        "configuration_count": len(configs),
        "runtime_entry_count": len(entries),
        "submodule_commit": submodule["commit"],
        "model_required": False,
    }


def verify_level(
    paths: RepositoryPaths,
    configs: Mapping[str, Mapping[str, Any]],
    level: str,
    *,
    runner: Runner | None = None,
    home: Path | None = None,
) -> dict[str, Any]:
    if level not in VERIFICATION_LEVELS:
        raise BootstrapError(ErrorCode.PRECONDITION_FAILED, "unknown verification level")
    command = runner or SubprocessRunner()
    source = verify_source(paths, configs, runner=command)
    details: dict[str, Any] = {"level": level, "source": source, "model_loaded": False, "api_called": False}
    if level == "source-only":
        return details

    package_lock = configs["ubuntu-package.lock.json"]
    cuda_lock = configs["cuda-wsl.lock.json"]
    inspection = HostInspector(paths.root, runner=command).inspect(
        [item["name"] for item in package_lock["packages"]], cuda_lock["forbidden_package_patterns"]
    )
    blockers = host_blockers(inspection, configs["ubuntu-26.04-wsl2-host.json"])
    if blockers or not python_ready(inspection) or not cuda_toolkit_ready(inspection):
        raise BootstrapError(ErrorCode.PRECONDITION_FAILED, "host-ready verification failed", context={"blockers": blockers})
    details["host_ready"] = True
    if level == "host-ready":
        return details

    environment_lock = configs["python-environments.lock.json"]
    environment_states = {
        item["environment_identity"]: environment_status(paths, environment_lock, item)
        for item in environment_lock["environments"]
    }
    if any(value != "ready" for value in environment_states.values()):
        raise BootstrapError(ErrorCode.PRECONDITION_FAILED, "private environments are not bootstrap-bound and ready")
    details["environments"] = environment_states
    details["llama"] = verify_llama_no_model(paths, configs["llama-build.lock.json"], command)
    if level == "build-ready":
        return details

    service_contract = configs["service-registration.json"]
    registered = service_status(paths, service_contract, home=home)
    if not all(registered[key] for key in ("unit_present", "adapter_manifest_present", "operating_profile_present", "desired_state_present")):
        raise BootstrapError(ErrorCode.PRECONDITION_FAILED, "platform service is not registered")
    details["service_registration"] = registered
    if level == "service-process-ready":
        adapter = paths.root / service_contract["adapter"]["source_entrypoint"]
        runtime_root = paths.root / service_contract["adapter"]["runtime_root"]
        result = command(("python3.14", "-B", "-I", "-S", str(adapter), "--adapter-runtime-root", str(runtime_root), "status"), timeout=120)
        if result.returncode != 0 or '"active":true' not in result.stdout.replace(" ", ""):
            raise BootstrapError(ErrorCode.PRECONDITION_FAILED, "registered service process is not active")
        details["service_process_ready"] = True
        return details

    runtime = runtime_status(paths, configs["runtime-layout.json"])
    credential = credential_status(paths, configs["credential-initialization.json"])
    if runtime != "ready" or credential["state"] != "ready":
        raise BootstrapError(ErrorCode.PRECONDITION_FAILED, "runtime or local credential is not ready")
    details.update(
        {
            "runtime": runtime,
            "credential": {"state": "ready", "key_id": credential["key_id"]},
            "service_readiness_state": "WAITING_FOR_MODEL",
            "model_required_for_core_service_health": False,
            "inference_ready": False,
        }
    )
    if level == "waiting-for-model":
        return details

    model_roots = (paths.root / "model-api-gguf" / "MODEL", paths.root / "model-api-native" / "MODEL")
    if not any(path.is_file() for root in model_roots for path in root.rglob("*.gguf")):
        raise BootstrapError(ErrorCode.PRECONDITION_FAILED, "full-ready requires an admitted GGUF model")
    details["service_readiness_state"] = "READY"
    details["inference_ready"] = True
    return details
