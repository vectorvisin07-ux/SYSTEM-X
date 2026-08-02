"""Bounded host-package planning and explicitly authorized future install."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping, Sequence

from .command import CommandResult, Runner, SubprocessRunner, require_success
from .config import canonical_json_bytes
from .cuda import assert_toolkit_only_packages, configure_official_cuda_source, forbidden_package
from .errors import BootstrapError, ErrorCode
from .host import cuda_toolkit_ready, host_blockers, python_ready
from .transaction import BootstrapTransaction


_APT_INSTALL = re.compile(r"^Inst\s+(\S+)(?:\s+\[([^]]+)\])?")


def build_host_plan(
    inspection: Mapping[str, Any],
    profile: Mapping[str, Any],
    package_lock: Mapping[str, Any],
    cuda_lock: Mapping[str, Any],
) -> dict[str, Any]:
    installed = inspection.get("installed_packages", {})
    patterns = cuda_lock["forbidden_package_patterns"]
    forbidden_installed = sorted(name for name in installed if forbidden_package(name, patterns))
    blockers = host_blockers(inspection, profile)
    if forbidden_installed:
        blockers.append("prohibited Linux NVIDIA display-driver or CUDA meta-package is installed")

    satisfied: list[dict[str, str]] = []
    would_install: list[dict[str, str]] = []
    for record in sorted(package_lock["packages"], key=lambda item: item["install_order"]):
        name = record["name"]
        version = installed.get(name)
        if version == record["observed_version"]:
            satisfied.append({"name": name, "version": version, "mode": "exact"})
        elif version and re.fullmatch(record["compatible_version_regex"], version):
            satisfied.append({"name": name, "version": version, "mode": "compatible-patch-observed"})
        else:
            would_install.append({
                "name": name,
                "version": record["observed_version"],
                "source_repository_identity": record["source_repository_identity"],
            })

    # Never treat a wrong Python or CUDA major/minor as an acceptable patch.
    py = inspection.get("python", {})
    if py.get("present") and py.get("version", [])[:2] != [3, 14]:
        blockers.append("installed Python major/minor is not 3.14")
    if py.get("present") and (not py.get("venv") or not py.get("ensurepip")):
        blockers.append("Python 3.14 venv and ensurepip support are required")
    nvcc = inspection.get("tools", {}).get("nvcc", {})
    if nvcc.get("present") and nvcc.get("major_minor") != "13.3":
        blockers.append("installed CUDA toolkit major/minor is not 13.3")

    plan: dict[str, Any] = {
        "schema": "system-x.bootstrap.host-plan.v1",
        "version": 1,
        "already_satisfied": satisfied,
        "would_install": would_install,
        "would_create": [
            "official NVIDIA WSL-Ubuntu apt source metadata"
        ] if any(item["name"] == "cuda-toolkit-13-3" for item in would_install) else [],
        "would_build": [],
        "would_register": [],
        "would_generate": [],
        "would_leave_external": [
            "Windows NVIDIA display driver",
            "credentials",
            "models",
            "runtime databases",
            "platform service registration"
        ],
        "blockers": sorted(set(blockers)),
        "forbidden_installed": forbidden_installed,
        "python_ready": python_ready(inspection),
        "cuda_toolkit_ready": cuda_toolkit_ready(inspection),
        "unrelated_packages_preserved": True,
        "network_used": False,
        "mutation_performed": False,
    }
    plan["plan_identity"] = hashlib.sha256(canonical_json_bytes(plan)).hexdigest()
    return plan


def _candidate_version(package: str, runner: Runner) -> tuple[str | None, set[str]]:
    result = runner(("apt-cache", "policy", package), timeout=30)
    if result.returncode != 0:
        return None, set()
    candidate: str | None = None
    available: set[str] = set()
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("Candidate:"):
            value = stripped.split(":", 1)[1].strip()
            candidate = None if value == "(none)" else value
        match = re.match(r"(?:\*\*\*\s+)?(\S+)\s+\d+$", stripped)
        if match and match.group(1) not in {"Installed:", "Candidate:"}:
            available.add(match.group(1))
    return candidate, available


def _select_install_specs(
    records: Sequence[Mapping[str, Any]],
    runner: Runner,
    *,
    allow_patch_difference: bool,
) -> tuple[list[str], list[dict[str, str]]]:
    specs: list[str] = []
    differences: list[dict[str, str]] = []
    for record in records:
        candidate, available = _candidate_version(record["name"], runner)
        exact = record["observed_version"]
        if exact in available or candidate == exact:
            selected = exact
        elif (
            allow_patch_difference
            and candidate is not None
            and re.fullmatch(record["compatible_version_regex"], candidate)
        ):
            selected = candidate
            differences.append({"name": record["name"], "locked": exact, "selected": candidate})
        else:
            raise BootstrapError(
                ErrorCode.PRECONDITION_FAILED,
                "locked package version is unavailable and no authorized patch-level substitute is valid",
                context={"package": record["name"], "locked": exact, "candidate": candidate},
            )
        specs.append(f"{record['name']}={selected}")
    return specs, differences


def validate_apt_simulation(
    result: CommandResult,
    *,
    direct_names: Sequence[str],
    cuda_lock: Mapping[str, Any],
) -> list[str]:
    require_success(result, purpose="apt simulation failed")
    if any(line.startswith(("Remv ", "Conf ")) for line in result.stdout.splitlines()):
        raise BootstrapError(ErrorCode.HOST_UNSUPPORTED, "apt simulation would remove or reconfigure packages")
    planned: list[str] = []
    direct = set(direct_names)
    for line in result.stdout.splitlines():
        match = _APT_INSTALL.match(line)
        if not match:
            continue
        name, previous = match.groups()
        name = name.split(":", 1)[0]
        planned.append(name)
        if forbidden_package(name, cuda_lock["forbidden_package_patterns"]):
            raise BootstrapError(ErrorCode.HOST_UNSUPPORTED, "apt simulation includes a prohibited CUDA/driver package")
        if previous is not None and name not in direct:
            raise BootstrapError(
                ErrorCode.HOST_UNSUPPORTED,
                "apt simulation would upgrade an unrelated installed package",
                context={"package": name},
            )
    return planned


def apply_host(
    inspection: Mapping[str, Any],
    package_lock: Mapping[str, Any],
    cuda_lock: Mapping[str, Any],
    *,
    transaction: BootstrapTransaction,
    authorized: bool,
    elevated: bool,
    allow_patch_difference: bool = False,
    runner: Runner | None = None,
) -> dict[str, Any]:
    """Future-only operation. The current packet never invokes this method."""

    if not authorized:
        raise BootstrapError(ErrorCode.AUTHORIZATION_REQUIRED, "apply-host requires explicit authorization")
    if not elevated:
        raise BootstrapError(ErrorCode.AUTHORIZATION_REQUIRED, "apply-host requires root elevation")
    if not getattr(transaction, "_entered", False):
        raise BootstrapError(ErrorCode.TRANSACTION_RECOVERY_REQUIRED, "apply-host requires an active transaction")
    command = runner or SubprocessRunner()
    installed = inspection.get("installed_packages", {})
    patterns = cuda_lock["forbidden_package_patterns"]
    assert_toolkit_only_packages(
        [name for name in installed if forbidden_package(name, patterns)],
        cuda_lock,
    )

    missing = [record for record in package_lock["packages"] if installed.get(record["name"]) != record["observed_version"]]
    ubuntu_records = [record for record in missing if record["source_repository_identity"] == "ubuntu-resolute-official"]
    cuda_records = [record for record in missing if record["source_repository_identity"] == "nvidia-cuda-wsl-ubuntu-x86_64"]
    all_differences: list[dict[str, str]] = []

    if ubuntu_records:
        require_success(command(("apt-get", "update"), timeout=300), purpose="Ubuntu package index refresh failed")
        specs, differences = _select_install_specs(ubuntu_records, command, allow_patch_difference=allow_patch_difference)
        all_differences.extend(differences)
        simulation = command(("apt-get", "--simulate", "install", "--no-install-recommends", *specs), timeout=300)
        validate_apt_simulation(simulation, direct_names=[item["name"] for item in ubuntu_records], cuda_lock=cuda_lock)
        transaction.record("ubuntu-package-plan-accepted", {"specifications": specs, "patch_differences": differences})
        require_success(
            command(("apt-get", "install", "--yes", "--no-install-recommends", *specs), timeout=1800),
            purpose="locked Ubuntu direct-package installation failed",
        )

    if cuda_records:
        configure_official_cuda_source(cuda_lock, command)
        require_success(command(("apt-get", "update"), timeout=300), purpose="CUDA repository verification/index refresh failed")
        specs, differences = _select_install_specs(cuda_records, command, allow_patch_difference=allow_patch_difference)
        all_differences.extend(differences)
        assert_toolkit_only_packages([item["name"] for item in cuda_records], cuda_lock)
        simulation = command(("apt-get", "--simulate", "install", "--no-install-recommends", *specs), timeout=300)
        validate_apt_simulation(simulation, direct_names=[item["name"] for item in cuda_records], cuda_lock=cuda_lock)
        transaction.record("cuda-toolkit-plan-accepted", {"specifications": specs, "patch_differences": differences})
        require_success(
            command(("apt-get", "install", "--yes", "--no-install-recommends", *specs), timeout=1800),
            purpose="toolkit-only CUDA 13.3 installation failed",
        )

    transaction.record("host-package-application-complete", {"patch_differences": all_differences})
    return {
        "changed": bool(missing),
        "patch_differences": all_differences,
        "removed_packages": [],
        "display_driver_changed": False,
        "model_installed": False,
    }
