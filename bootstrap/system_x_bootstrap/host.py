"""Read-only Ubuntu/WSL host inspection and validation."""

from __future__ import annotations

import fnmatch
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .command import CommandResult, Runner, SubprocessRunner


_NVCC_RELEASE = re.compile(r"release\s+([0-9]+\.[0-9]+)")


def _parse_os_release(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value.strip().strip('"').strip("'")
    return values


def _first_line(result: CommandResult) -> str | None:
    lines = result.stdout.splitlines()
    return lines[0].strip() if lines else None


class HostInspector:
    """Collect facts without network, package installation, or service control."""

    def __init__(
        self,
        repository_root: Path,
        *,
        filesystem_root: Path = Path("/"),
        runner: Runner | None = None,
    ) -> None:
        self.repository_root = repository_root.resolve(strict=True)
        self.filesystem_root = filesystem_root.resolve(strict=True)
        self.runner = runner or SubprocessRunner()

    def mapped(self, absolute: str) -> Path:
        return self.filesystem_root / absolute.lstrip("/")

    def _read(self, absolute: str) -> str | None:
        path = self.mapped(absolute)
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(self.filesystem_root)
            if not resolved.is_file():
                return None
            return resolved.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError, ValueError):
            return None

    def _run(self, argv: Sequence[str], *, timeout: int = 30, cwd: Path | None = None) -> CommandResult:
        return self.runner(argv, timeout=timeout, cwd=cwd)

    def inspect(self, package_names: Sequence[str], forbidden_patterns: Sequence[str]) -> dict[str, Any]:
        os_release = _parse_os_release(self._read("/etc/os-release") or "")
        kernel = (self._read("/proc/sys/kernel/osrelease") or "").strip()
        proc_version = (self._read("/proc/version") or "").strip()
        pid1 = (self._read("/proc/1/comm") or "").strip()
        architecture_result = self._run(("uname", "-m"))
        architecture = _first_line(architecture_result) if architecture_result.returncode == 0 else None

        installed = self._installed_packages(package_names, forbidden_patterns)
        python = self._python_facts()
        tools = self._tool_facts()
        gpu = self._gpu_facts()
        submodule = self._submodule_facts()

        unit = self.mapped("/home")
        service_unit = Path.home() / ".config" / "systemd" / "user" / "system-x-current-test.service"
        if self.filesystem_root != Path("/"):
            service_unit = self.mapped("/home/fixture/.config/systemd/user/system-x-current-test.service")
        service_link = service_unit.parent / "default.target.wants" / service_unit.name

        return {
            "operating_system": os_release,
            "kernel": {
                "release": kernel,
                "proc_version": proc_version,
                "wsl_detected": "microsoft" in (kernel + " " + proc_version).lower(),
                "wsl_version": 2 if "wsl2" in kernel.lower() or "microsoft-standard" in kernel.lower() else None,
            },
            "systemd": {
                "pid1": pid1,
                "user_runtime_directory_present": bool(os.environ.get("XDG_RUNTIME_DIR")),
                "dbus_session_address_present": bool(os.environ.get("DBUS_SESSION_BUS_ADDRESS")),
                "service_control_invoked": False,
            },
            "architecture": architecture,
            "ownership": {
                "effective_uid": os.geteuid(),
                "effective_gid": os.getegid(),
                "repository_owner_uid": self.repository_root.stat().st_uid,
            },
            "package_manager": {
                "apt_get_present": self.mapped("/usr/bin/apt-get").is_file(),
                "apt_cache_present": self.mapped("/usr/bin/apt-cache").is_file(),
                "dpkg_query_present": self.mapped("/usr/bin/dpkg-query").is_file(),
            },
            "installed_packages": installed,
            "python": python,
            "tools": tools,
            "gpu": gpu,
            "submodule": submodule,
            "runtime": {
                "root_present": (self.repository_root / "model-api-gguf" / "RUNTIME").is_dir(),
                "credential_database_present": (
                    self.repository_root / "model-api-gguf" / "RUNTIME" / "api" / "auth" / "credentials.sqlite3"
                ).exists(),
                "registry_database_present": (
                    self.repository_root / "model-api-gguf" / "RUNTIME" / "api" / "database" / "model_registry.sqlite3"
                ).exists(),
            },
            "service": {
                "unit_present": service_unit.is_file(),
                "enablement_link_present": service_link.is_symlink(),
                "query_method": "filesystem-only",
                "service_control_invoked": False,
            },
        }

    def _installed_packages(self, package_names: Sequence[str], forbidden_patterns: Sequence[str]) -> dict[str, str]:
        result = self._run(("dpkg-query", "-W", "-f=${Package}\t${Version}\t${db:Status-Abbrev}\\n"), timeout=45)
        if result.returncode not in (0, 1):
            return {}
        wanted = set(package_names)
        values: dict[str, str] = {}
        for line in result.stdout.splitlines():
            fields = line.split("\t")
            if len(fields) != 3 or not fields[2].startswith("ii"):
                continue
            name = fields[0].split(":", 1)[0]
            if name in wanted or any(fnmatch.fnmatchcase(name, pattern) for pattern in forbidden_patterns):
                values[name] = fields[1]
        return dict(sorted(values.items()))

    def _python_facts(self) -> dict[str, Any]:
        script = (
            "import importlib.util,json,sys;"
            "print(json.dumps({'implementation':sys.implementation.name,'version':list(sys.version_info[:3]),"
            "'venv':importlib.util.find_spec('venv') is not None,'ensurepip':importlib.util.find_spec('ensurepip') is not None}))"
        )
        result = self._run(("python3.14", "-B", "-I", "-S", "-c", script))
        if result.returncode != 0:
            return {"present": False, "returncode": result.returncode}
        try:
            facts = json.loads(result.stdout)
        except json.JSONDecodeError:
            return {"present": False, "returncode": result.returncode, "invalid_probe_output": True}
        facts["present"] = True
        facts["user_site_disabled_during_probe"] = True
        return facts

    def _tool_facts(self) -> dict[str, Any]:
        commands = {
            "cmake": ("cmake", "--version"),
            "curl": ("curl", "--version"),
            "git": ("git", "--version"),
            "ninja": ("ninja", "--version"),
            "pkg-config": ("pkg-config", "--version"),
            "nvcc": ("/usr/local/cuda-13.3/bin/nvcc", "--version"),
        }
        values: dict[str, Any] = {}
        for name, argv in commands.items():
            result = self._run(argv)
            first = _first_line(result)
            values[name] = {"present": result.returncode == 0, "first_line": first}
            if name == "nvcc" and result.returncode == 0:
                match = _NVCC_RELEASE.search(result.stdout + result.stderr)
                values[name]["major_minor"] = match.group(1) if match else None
        return values

    def _gpu_facts(self) -> dict[str, Any]:
        bridge = {
            path: self.mapped(path).exists()
            for path in ("/dev/dxg", "/usr/lib/wsl/lib/libcuda.so.1", "/usr/lib/wsl/lib/nvidia-smi")
        }
        smi = self._run(
            (
                "/usr/lib/wsl/lib/nvidia-smi",
                "--query-gpu=name,driver_version,compute_cap,memory.total",
                "--format=csv,noheader,nounits",
            )
        )
        driver_script = (
            "import ctypes,json;"
            "lib=ctypes.CDLL('/usr/lib/wsl/lib/libcuda.so.1');"
            "count=ctypes.c_int();"
            "a=int(lib.cuInit(0));b=int(lib.cuDeviceGetCount(ctypes.byref(count)));"
            "print(json.dumps({'cu_init':a,'cu_device_get_count':b,'device_count':count.value}))"
        )
        driver = self._run(("python3.14", "-B", "-I", "-S", "-c", driver_script))
        try:
            driver_facts = json.loads(driver.stdout) if driver.returncode == 0 else {}
        except json.JSONDecodeError:
            driver_facts = {}
        return {
            "bridge_paths": bridge,
            "nvidia_smi": {
                "returncode": smi.returncode,
                "observation": _first_line(smi),
            },
            "driver_api": {
                "returncode": driver.returncode,
                **driver_facts,
            },
            "display_driver_owner": "Windows",
        }

    def _submodule_facts(self) -> dict[str, Any]:
        path = self.repository_root / "model-api-gguf" / "llama.cpp"
        if not path.is_dir():
            return {"present": False}
        origin = self._run(("git", "-C", str(path), "remote", "get-url", "origin"))
        head = self._run(("git", "-C", str(path), "rev-parse", "HEAD"))
        status = self._run(("git", "-C", str(path), "status", "--porcelain=v1", "--untracked-files=all"))
        return {
            "present": origin.returncode == 0 and head.returncode == 0,
            "origin": _first_line(origin),
            "commit": _first_line(head),
            "clean": status.returncode == 0 and not status.stdout,
        }


def host_blockers(inspection: Mapping[str, Any], profile: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    operating = inspection.get("operating_system", {})
    expected_os = profile["operating_system"]
    if operating.get("ID") != expected_os["id"] or operating.get("VERSION_ID") != expected_os["version_id"]:
        blockers.append("wrong Ubuntu release")
    kernel = inspection.get("kernel", {})
    if not kernel.get("wsl_detected") or kernel.get("wsl_version") != 2:
        blockers.append("WSL2 is required")
    if inspection.get("systemd", {}).get("pid1") != profile["pid1"]:
        blockers.append("systemd PID 1 is required")
    if inspection.get("architecture") != profile["architecture"]:
        blockers.append("wrong architecture")
    manager = inspection.get("package_manager", {})
    if not all(manager.get(key) for key in ("apt_get_present", "apt_cache_present", "dpkg_query_present")):
        blockers.append("required apt/dpkg tools are missing")
    bridge = inspection.get("gpu", {}).get("bridge_paths", {})
    for required in profile["gpu"]["required_bridge_paths"]:
        if not bridge.get(required):
            blockers.append(f"WSL GPU bridge missing: {required}")
    smi = inspection.get("gpu", {}).get("nvidia_smi", {})
    if smi.get("returncode") != 0 or not smi.get("observation"):
        blockers.append("Windows GPU is not visible through WSL nvidia-smi")
    driver = inspection.get("gpu", {}).get("driver_api", {})
    if driver.get("returncode") != 0 or driver.get("cu_init") != 0 or driver.get("device_count", 0) < 1:
        blockers.append("WSL CUDA driver API cannot initialize")
    return blockers


def python_ready(inspection: Mapping[str, Any]) -> bool:
    python = inspection.get("python", {})
    return bool(
        python.get("present")
        and python.get("implementation") == "cpython"
        and python.get("version", [])[:2] == [3, 14]
        and python.get("venv")
        and python.get("ensurepip")
    )


def cuda_toolkit_ready(inspection: Mapping[str, Any]) -> bool:
    return inspection.get("tools", {}).get("nvcc", {}).get("major_minor") == "13.3"
