"""Bounded host-package planning and explicitly authorized future install."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping, Sequence

from .command import CommandResult, Runner, SubprocessRunner, require_success
from .config import canonical_json_bytes
from .cuda import assert_toolkit_only_packages, configure_official_cuda_source, forbidden_package
from .errors import BootstrapError, ErrorCode
from .host import cuda_toolkit_ready, host_blockers, python_capability_state, python_ready
from .transaction import BootstrapTransaction


_APT_INSTALL = re.compile(r"^Inst\s+(\S+)(?:\s+\[([^]]+)\])?")
_APT_DEPENDENCY = re.compile(r"^\s*\|?(?:Pre-)?Depends:\s+(\S+)")


def _package_version_satisfies(record: Mapping[str, Any], version: str | None) -> bool:
    return bool(version) and (
        version == record["observed_version"]
        or bool(re.fullmatch(record["compatible_version_regex"], version))
    )


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
        if _package_version_satisfies(record, version):
            mode = "exact" if version == record["observed_version"] else "compatible-patch-observed"
            satisfied.append({"name": name, "version": version, "mode": mode})
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
    # Missing venv/ensurepip is installable when the locked python3.14-venv
    # package is in the plan; strict functional validation belongs after apply-host.
    nvcc = inspection.get("tools", {}).get("nvcc", {})
    if nvcc.get("present") and nvcc.get("major_minor") != "13.3":
        blockers.append("installed CUDA toolkit major/minor is not 13.3")

    python_capability = python_capability_state(inspection, {item["name"] for item in would_install})
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
        "python_capability": python_capability,
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


def _apt_dependency_closure(
    direct_names: Sequence[str],
    runner: Runner,
) -> set[str]:
    pending = list(direct_names)
    dependencies: set[str] = set()
    while pending:
        package = pending.pop()
        if package in dependencies:
            continue
        dependencies.add(package)
        result = runner(("apt-cache", "depends", package), timeout=30)
        require_success(result, purpose="apt dependency inspection failed")
        for line in result.stdout.splitlines():
            match = _APT_DEPENDENCY.match(line)
            if not match:
                continue
            dependency = match.group(1)
            if dependency.startswith("<"):
                continue
            dependency = dependency.split("|", 1)[0].split(":", 1)[0]
            if dependency and dependency not in dependencies:
                pending.append(dependency)
    return dependencies


def _apt_essential_packages(
    runner: Runner,
) -> set[str]:
    result = runner(("dpkg-query", "-W", "-f=" + chr(36) + "{Package}\t" + chr(36) + "{Essential}\n"), timeout=30)
    require_success(result, purpose="essential package inspection failed")
    essential: set[str] = set()
    for line in result.stdout.splitlines():
        name, separator, value = line.partition("\t")
        if separator and value.strip() == "yes":
            essential.add(name.split(":", 1)[0])
    return essential
def _apt_reverse_dependency_names(
    package_names: Sequence[str],
    runner: Runner,
) -> set[str]:
    planned = {name.split(":", 1)[0] for name in package_names}
    if not planned:
        return set()
    result = runner(("apt-cache", "rdepends", "--installed", *sorted(planned)), timeout=60)
    require_success(result, purpose="installed reverse dependency inspection failed")
    reverse: set[str] = set()
    for line in result.stdout.splitlines():
        if not line[:1].isspace():
            continue
        name = line.strip().split(maxsplit=1)[0] if line.strip() else ""
        if not name or name == "Reverse" or name.startswith("<"):
            continue
        name = name.split(":", 1)[0]
        if name not in planned:
            reverse.add(name)
    return reverse




def validate_apt_simulation(
    result: CommandResult,
    *,
    direct_names: Sequence[str],
    cuda_lock: Mapping[str, Any],
    dependency_names: Sequence[str] = (),
    essential_names: Sequence[str] = (),
    reverse_dependency_names: Sequence[str] = (),
) -> list[str]:
    require_success(result, purpose="apt simulation failed")
    if any(line.startswith("Remv ") for line in result.stdout.splitlines()):
        raise BootstrapError(ErrorCode.HOST_UNSUPPORTED, "apt simulation would remove or reconfigure packages")
    planned: list[str] = []
    direct = set(direct_names)
    dependencies = set(dependency_names)
    essential = set(essential_names)
    reverse_dependencies = set(reverse_dependency_names)
    for line in result.stdout.splitlines():
        match = _APT_INSTALL.match(line)
        if not match:
            continue
        name, previous = match.groups()
        name = name.split(":", 1)[0]
        planned.append(name)
        if forbidden_package(name, cuda_lock["forbidden_package_patterns"]):
            raise BootstrapError(ErrorCode.HOST_UNSUPPORTED, "apt simulation includes a prohibited CUDA/driver package")
        if previous is not None and name not in direct and name not in dependencies and name not in essential and name not in reverse_dependencies:
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
    if not getattr(transaction, "_entered", False):
        raise BootstrapError(ErrorCode.TRANSACTION_RECOVERY_REQUIRED, "apply-host requires an active transaction")
    command = runner or SubprocessRunner()
    installed = inspection.get("installed_packages", {})
    patterns = cuda_lock["forbidden_package_patterns"]
    assert_toolkit_only_packages(
        [name for name in installed if forbidden_package(name, patterns)],
        cuda_lock,
    )

    missing = [
        record
        for record in package_lock["packages"]
        if not _package_version_satisfies(record, installed.get(record["name"]))
    ]
    if missing and not elevated:
        raise BootstrapError(ErrorCode.AUTHORIZATION_REQUIRED, "apply-host requires root elevation when package mutation is required")
    ubuntu_records = [record for record in missing if record["source_repository_identity"] == "ubuntu-resolute-official"]
    cuda_records = [record for record in missing if record["source_repository_identity"] == "nvidia-cuda-wsl-ubuntu-x86_64"]
    all_differences: list[dict[str, str]] = []
    essential_names = _apt_essential_packages(command)

    if ubuntu_records:
        require_success(command(("apt-get", "update"), timeout=300), purpose="Ubuntu package index refresh failed")
        specs, differences = _select_install_specs(ubuntu_records, command, allow_patch_difference=allow_patch_difference)
        all_differences.extend(differences)
        direct_names = [item["name"] for item in ubuntu_records]
        dependency_names = _apt_dependency_closure(direct_names, command)
        reverse_dependency_names = _apt_reverse_dependency_names((*direct_names, *dependency_names, *essential_names), command)
        simulation = command(("apt-get", "--simulate", "install", "--no-install-recommends", "--no-upgrade", *specs), timeout=300)
        validate_apt_simulation(simulation, direct_names=direct_names, dependency_names=dependency_names, essential_names=essential_names, reverse_dependency_names=reverse_dependency_names, cuda_lock=cuda_lock)
        transaction.record("ubuntu-package-plan-accepted", {"specifications": specs, "patch_differences": differences})
        require_success(
            command(("apt-get", "install", "--yes", "--no-install-recommends", "--no-upgrade", *specs), timeout=1800),
            purpose="locked Ubuntu direct-package installation failed",
        )

    if cuda_records:
        configure_official_cuda_source(cuda_lock, command)
        require_success(command(("apt-get", "update"), timeout=300), purpose="CUDA repository verification/index refresh failed")
        specs, differences = _select_install_specs(cuda_records, command, allow_patch_difference=allow_patch_difference)
        all_differences.extend(differences)
        assert_toolkit_only_packages([item["name"] for item in cuda_records], cuda_lock)
        direct_names = [item["name"] for item in cuda_records]
        dependency_names = _apt_dependency_closure(direct_names, command)
        reverse_dependency_names = _apt_reverse_dependency_names((*direct_names, *dependency_names, *essential_names), command)
        simulation = command(("apt-get", "--simulate", "install", "--no-install-recommends", "--no-upgrade", *specs), timeout=300)
        validate_apt_simulation(simulation, direct_names=direct_names, dependency_names=dependency_names, essential_names=essential_names, reverse_dependency_names=reverse_dependency_names, cuda_lock=cuda_lock)
        transaction.record("cuda-toolkit-plan-accepted", {"specifications": specs, "patch_differences": differences})
        require_success(
            command(("apt-get", "install", "--yes", "--no-install-recommends", "--no-upgrade", *specs), timeout=1800),
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
