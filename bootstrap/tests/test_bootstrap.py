"""No-mutation safety and reconstruction contract matrix."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


REPOSITORY_ROOT = Path(__file__).resolve(strict=True).parents[2]
BOOTSTRAP_ROOT = REPOSITORY_ROOT / "bootstrap"
sys.path.insert(0, str(BOOTSTRAP_ROOT))

from system_x_bootstrap.command import CommandResult, InstallationUser, SubprocessRunner, installation_user_context, resolve_installation_user
from system_x_bootstrap.config import load_registry
from system_x_bootstrap.credentials import (
    _credential_command,
    credential_status,
    initialize_credentials,
)
from system_x_bootstrap.cuda import CUDA_KEYRING_PATH, assert_toolkit_only_packages
from system_x_bootstrap.environments import (
    _environment_marker,
    _verify_imports,
    environment_status,
    render_hashed_requirements,
    validate_environment_lock,
)
from system_x_bootstrap.errors import BootstrapError, ErrorCode
from system_x_bootstrap.host import (
    HostInspector,
    cuda_toolkit_ready,
    host_blockers,
    python_ready,
)
from system_x_bootstrap.llama import (
    _cache_matches,
    _canonical_manifest,
    cmake_arguments,
    initialize_submodules,
    inspect_vendored_source,
    verify_llama_no_model,
)
from system_x_bootstrap.packages import _apt_dependency_closure, _apt_essential_packages, _apt_reverse_dependency_names, apply_host, build_host_plan, validate_apt_simulation
from system_x_bootstrap.paths import RepositoryPaths, discover_repository_root, resolve_contained
from system_x_bootstrap.result import MachineResult
from system_x_bootstrap.runtime import expand_runtime_layout, initialize_runtime
from system_x_bootstrap.orchestrator import BootstrapOrchestrator
from system_x_bootstrap.service import activate_platform_service, register_platform_service, render_operating_profile
from system_x_bootstrap.state import initial_state, write_receipt
from system_x_bootstrap.state import StateDocument
from system_x_bootstrap.transaction import (
    BootstrapTransaction,
    incomplete_transactions,
    recover_failed_clean_transactions,
)


CONFIGURATION_NAMES = (
    "ubuntu-26.04-wsl2-host.json",
    "ubuntu-package.lock.json",
    "cuda-wsl.lock.json",
    "python-environments.lock.json",
    "llama-build.lock.json",
    "runtime-layout.json",
    "credential-initialization.json",
    "service-registration.json",
)


def repository_paths(root: Path) -> RepositoryPaths:
    state = root / ".system-x-bootstrap-state"
    return RepositoryPaths(
        root,
        root / "bootstrap",
        root / "bootstrap/configuration",
        root / "bootstrap/schemas",
        state,
        root / "model-api-gguf/RUNTIME",
        state / "transactions",
        state / "locks/system-x-bootstrap.lock",
        state / "status.json",
    )


def repository_root_fixture(root: Path) -> None:
    (root / "bootstrap").mkdir(parents=True)
    (root / "bootstrap/run_bootstrap.py").write_text("# fixture\n", encoding="utf-8")
    (root / "model-api-gguf").mkdir()
    (root / "SYSTEM_X_REPOSITORY_MANIFEST.json").write_text("{}\n", encoding="utf-8")
    (root / "SYSTEM_X_NEW_UBUNTU_REQUIREMENTS.json").write_text("{}\n", encoding="utf-8")


class ActiveTransaction:
    _entered = True

    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []

    def record(self, event: str, details: object = None) -> None:
        self.events.append((event, details))

    def claim_created_path(self, relative: str) -> Path:
        raise AssertionError(f"fixture unexpectedly claimed {relative}")


def vendored_fixture(root: Path, base_lock: dict[str, object]) -> tuple[RepositoryPaths, dict[str, object], Path, Path]:
    source = root / "model-api-gguf/llama.cpp"
    identity_path = root / "model-api-gguf/LLAMA_CPP_SOURCE_IDENTITY.json"
    source.mkdir(parents=True)
    payloads = {
        "LICENSE": b"public fixture license\n",
        "src/main.cpp": b"int main() { return 0; }\n",
        "scripts/tool.sh": b"#!/bin/sh\nexit 0\n",
    }
    files = []
    for relative, raw in sorted(payloads.items()):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        mode = "100755" if relative.endswith(".sh") else "100644"
        if mode == "100755":
            path.chmod(0o755)
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()
        files.append({
            "path": relative,
            "mode": mode,
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "git_blob_oid": blob,
        })
    lock = copy.deepcopy(base_lock)
    source_lock = lock["source"]
    assert isinstance(source_lock, dict)
    source_lock.update({
        "mode": "vendored",
        "tag": "b10092",
        "tree": "f" * 40,
        "identity_record": "model-api-gguf/LLAMA_CPP_SOURCE_IDENTITY.json",
    })
    identity = {
        "schema": "system-x.llama-cpp-source-identity.v1",
        "version": 1,
        "origin": source_lock["origin"],
        "tag": source_lock["tag"],
        "commit": source_lock["commit"],
        "upstream_tree": source_lock["tree"],
        "tracked_file_count": len(files),
        "tracked_byte_count": sum(item["bytes"] for item in files),
        "complete_vendored_manifest_sha256": hashlib.sha256(_canonical_manifest(files)).hexdigest(),
        "license_identities": [{"path": "LICENSE", "sha256": files[0]["sha256"]}],
        "source_patch_count": 0,
        "build_output_excluded": True,
        "files": files,
    }
    identity_path.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return repository_paths(root), lock, source, identity_path


class GitRunner:
    def __init__(self, lock: dict[str, object], *, origin: str | None = None, commit: str | None = None, dirty: bool = False) -> None:
        source = lock["source"]
        assert isinstance(source, dict)
        self.origin = origin or str(source["origin"])
        self.commit = commit or str(source["commit"])
        self.dirty = dirty
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, argv, *, cwd=None, env=None, timeout=60):
        arguments = tuple(argv)
        self.calls.append(arguments)
        if "remote" in arguments:
            output = self.origin + "\n"
        elif "rev-parse" in arguments:
            output = self.commit + "\n"
        elif "status" in arguments:
            output = " M common/arg.cpp\n" if self.dirty else ""
        elif arguments[-1] == "--version":
            output = "llama-server fixture\n"
        elif arguments[-1] == "--list-devices":
            output = "CUDA0: NVIDIA GeForce RTX 3060\n"
        elif arguments[0] == "ldd":
            output = "libc.so.6 => /lib/libc.so.6\n"
        else:
            output = ""
        return CommandResult(arguments, 0, output, "")


class BootstrapMatrix(unittest.TestCase):
    def test_reconstruct_reuses_completed_phases_from_service_registered(self) -> None:
        orchestrator = object.__new__(BootstrapOrchestrator)
        orchestrator.paths = object()
        state = StateDocument("SERVICE_REGISTERED", "SERVICE_REGISTERED", 9, (), "receipt", "2026-08-18T00:00:00Z")
        levels = {
            "host-ready": "HOST_READY",
            "source-only": "CLONED",
            "build-ready": "LLAMA_SERVER_BUILT",
            "service-process-ready": "SERVICE_REGISTERED",
            "waiting-for-model": "WAITING_FOR_MODEL",
        }
        orchestrator.verify = mock.Mock(
            side_effect=lambda level: MachineResult("verify", "ok", levels[level], details={"level": level})
        )
        with mock.patch("system_x_bootstrap.orchestrator.read_state", return_value=state):
            results = orchestrator.reconstruct(authorized=True)
        self.assertEqual(
            [item.operation for item in results],
            [
                "apply-host", "initialize-submodules", "build-environments",
                "build-llama-server", "initialize-runtime", "initialize-credentials",
                "register-platform-service", "activate-platform-service", "verify",
            ],
        )
        self.assertTrue(all(item.status == "ok" and not item.changed for item in results))
        self.assertTrue(all(item.details.get("reused") for item in results[:-1]))
        self.assertEqual(orchestrator.verify.call_count, 9)

    def test_reconstruct_executes_activation_after_registration_in_same_flow(self) -> None:
        orchestrator = object.__new__(BootstrapOrchestrator)
        orchestrator.paths = object()
        orchestrator.configs = {}
        state = StateDocument("CLONED", "CLONED", 0, (), None, None)
        orchestrator._prepare_reconstruct_host = mock.Mock()
        operation_names = (
            "apply_host", "initialize_submodules", "build_environments",
            "build_llama_server", "initialize_runtime", "initialize_credentials",
            "register_platform_service", "activate_platform_service",
        )
        targets = {
            "apply_host": "HOST_READY",
            "initialize_submodules": "SUBMODULES_READY",
            "build_environments": "PYTHON_ENVIRONMENTS_READY",
            "build_llama_server": "LLAMA_SERVER_BUILT",
            "initialize_runtime": "RUNTIME_INITIALIZED",
            "initialize_credentials": "CREDENTIAL_READY",
            "register_platform_service": "SERVICE_REGISTERED",
            "activate_platform_service": "SERVICE_REGISTERED",
        }
        for name in operation_names:
            operation = name.replace("_", "-")
            setattr(orchestrator, name, mock.Mock(return_value=MachineResult(operation, "ok", targets[name], changed=True)))
        levels = {
            "host-ready": "HOST_READY", "source-only": "CLONED",
            "build-ready": "LLAMA_SERVER_BUILT",
            "service-process-ready": "SERVICE_REGISTERED",
            "waiting-for-model": "WAITING_FOR_MODEL",
        }
        orchestrator.verify = mock.Mock(side_effect=lambda level: MachineResult("verify", "ok", levels[level], details={"level": level}))
        with mock.patch("system_x_bootstrap.orchestrator.read_state", return_value=state):
            results = orchestrator.reconstruct(authorized=True)
        self.assertEqual(orchestrator.register_platform_service.call_count, 1)
        self.assertEqual(orchestrator.activate_platform_service.call_count, 1)
        activation = next(item for item in results if item.operation == "activate-platform-service")
        self.assertTrue(activation.changed)
        self.assertNotIn("reused", activation.details)

    @classmethod
    def setUpClass(cls) -> None:
        cls.paths = RepositoryPaths.discover(BOOTSTRAP_ROOT)
        cls.loaded = load_registry(cls.paths, CONFIGURATION_NAMES)
        cls.configs = {name: item.data for name, item in cls.loaded.items()}

    def inspection(self) -> dict[str, object]:
        package_lock = self.configs["ubuntu-package.lock.json"]
        installed = {record["name"]: record["observed_version"] for record in package_lock["packages"]}
        return {
            "operating_system": {"ID": "ubuntu", "VERSION_ID": "26.04", "VERSION_CODENAME": "resolute"},
            "kernel": {"wsl_detected": True, "wsl_version": 2},
            "systemd": {"pid1": "systemd"},
            "architecture": "x86_64",
            "package_manager": {"apt_get_present": True, "apt_cache_present": True, "dpkg_query_present": True},
            "installed_packages": installed,
            "python": {"present": True, "implementation": "cpython", "version": [3, 14, 4], "venv": True, "ensurepip": True},
            "tools": {"nvcc": {"present": True, "major_minor": "13.3"}},
            "gpu": {
                "bridge_paths": {path: True for path in self.configs["cuda-wsl.lock.json"]["required_bridge_paths"]},
                "nvidia_smi": {"returncode": 0, "observation": "NVIDIA GeForce RTX 3060, 595.97, 8.6, 12288"},
                "driver_api": {"returncode": 0, "cu_init": 0, "device_count": 1},
            },
        }

    def host_plan(self, inspection: dict[str, object]) -> dict[str, object]:
        return build_host_plan(
            inspection,
            self.configs["ubuntu-26.04-wsl2-host.json"],
            self.configs["ubuntu-package.lock.json"],
            self.configs["cuda-wsl.lock.json"],
        )

    def test_01_standard_library_import(self) -> None:
        code = (
            f"import sys;sys.path.insert(0,{str(BOOTSTRAP_ROOT)!r});"
            "import system_x_bootstrap,system_x_bootstrap.orchestrator;"
            "assert system_x_bootstrap.BOOTSTRAP_VERSION=='1.0.0'"
        )
        result = subprocess.run(
            (sys.executable, "-B", "-I", "-S", "-c", code),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))

    def test_02_self_relative_root_discovery(self) -> None:
        self.assertEqual(discover_repository_root(REPOSITORY_ROOT / "INSPECTOR"), REPOSITORY_ROOT)


    def test_02a_root_discovery_without_gitmodules(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository_root_fixture(root)
            self.assertFalse((root / ".gitmodules").exists())
            self.assertEqual(discover_repository_root(root / "bootstrap"), root)

    def test_02b_root_discovery_requires_durable_markers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository_root_fixture(root)
            (root / "SYSTEM_X_NEW_UBUNTU_REQUIREMENTS.json").unlink()
            with self.assertRaises(BootstrapError):
                discover_repository_root(root / "bootstrap")

    def test_02c_missing_identity_is_source_error_not_root_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository_root_fixture(root)
            paths, lock, _, identity_path = vendored_fixture(root, self.configs["llama-build.lock.json"])
            identity_path.unlink()
            self.assertEqual(discover_repository_root(root / "bootstrap"), root)
            result = inspect_vendored_source(paths, lock)
            self.assertFalse(result["exact"])
            self.assertEqual(result["reason"], "source identity record is absent or unsafe")

    def test_02d_live_llama_path_is_ordinary(self) -> None:
        self.assertFalse((REPOSITORY_ROOT / ".gitmodules").exists())
        stage = subprocess.check_output(
            ("git", "-C", os.environ.get("SYSTEM_X_GIT_ROOT", str(REPOSITORY_ROOT)), "ls-files", "--stage", "--", "model-api-gguf/llama.cpp/LICENSE"),
            text=True,
        )
        self.assertTrue(stage.startswith("100644 "))

    def test_03_ubuntu_2604_pass(self) -> None:
        self.assertEqual(host_blockers(self.inspection(), self.configs["ubuntu-26.04-wsl2-host.json"]), [])

    def test_04_wrong_ubuntu_release_block(self) -> None:
        value = self.inspection()
        value["operating_system"]["VERSION_ID"] = "24.04"
        self.assertIn("wrong Ubuntu release", host_blockers(value, self.configs["ubuntu-26.04-wsl2-host.json"]))

    def test_05_non_wsl_block(self) -> None:
        value = self.inspection()
        value["kernel"]["wsl_detected"] = False
        self.assertTrue(any("WSL2" in item for item in host_blockers(value, self.configs["ubuntu-26.04-wsl2-host.json"])))

    def test_06_systemd_missing_block(self) -> None:
        value = self.inspection()
        value["systemd"]["pid1"] = "init"
        self.assertTrue(any("systemd" in item for item in host_blockers(value, self.configs["ubuntu-26.04-wsl2-host.json"])))

    def test_07_wrong_architecture_block(self) -> None:
        value = self.inspection()
        value["architecture"] = "aarch64"
        self.assertIn("wrong architecture", host_blockers(value, self.configs["ubuntu-26.04-wsl2-host.json"]))

    def test_08_dev_dxg_missing_block(self) -> None:
        value = self.inspection()
        value["gpu"]["bridge_paths"]["/dev/dxg"] = False
        self.assertTrue(any("/dev/dxg" in item for item in host_blockers(value, self.configs["ubuntu-26.04-wsl2-host.json"])))

    def test_09_libcuda_missing_block(self) -> None:
        value = self.inspection()
        value["gpu"]["bridge_paths"]["/usr/lib/wsl/lib/libcuda.so.1"] = False
        self.assertTrue(any("libcuda.so.1" in item for item in host_blockers(value, self.configs["ubuntu-26.04-wsl2-host.json"])))

    def test_10_linux_driver_installed_plan_block(self) -> None:
        value = self.inspection()
        value["installed_packages"]["nvidia-driver-595"] = "595.1"
        plan = self.host_plan(value)
        self.assertIn("nvidia-driver-595", plan["forbidden_installed"])
        self.assertTrue(plan["blockers"])

    def test_11_cuda_meta_package_plan_block(self) -> None:
        with self.assertRaises(BootstrapError):
            assert_toolkit_only_packages(["cuda"], self.configs["cuda-wsl.lock.json"])

    def test_12_cuda_drivers_plan_block(self) -> None:
        with self.assertRaises(BootstrapError):
            assert_toolkit_only_packages(["cuda-drivers-595"], self.configs["cuda-wsl.lock.json"])

    def test_13_cuda_toolkit_133_plan_pass(self) -> None:
        assert_toolkit_only_packages(["cuda-toolkit-13-3"], self.configs["cuda-wsl.lock.json"])

    def test_13a_cuda_keyring_path_matches_pinned_package(self) -> None:
        self.assertEqual(str(CUDA_KEYRING_PATH), "/usr/share/keyrings/cuda-archive-keyring.gpg")
        self.assertIn("signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg", self.configs["cuda-wsl.lock.json"]["repository"]["source_line"])

    def test_14_wrong_cuda_major_minor_block(self) -> None:
        value = self.inspection()
        value["tools"]["nvcc"]["major_minor"] = "12.9"
        self.assertFalse(cuda_toolkit_ready(value))
        self.assertTrue(any("CUDA toolkit" in item for item in self.host_plan(value)["blockers"]))

    def test_15_python_314_pass(self) -> None:
        self.assertTrue(python_ready(self.inspection()))

    def test_16_wrong_python_major_minor_block(self) -> None:
        value = self.inspection()
        value["python"]["version"] = [3, 13, 9]
        self.assertFalse(python_ready(value))
        self.assertTrue(any("Python" in item for item in self.host_plan(value)["blockers"]))

    def test_17_missing_venv_support_block(self) -> None:
        value = self.inspection()
        value["python"]["venv"] = False
        self.assertFalse(python_ready(value))

    def test_18_package_plan_idempotence(self) -> None:
        first = self.host_plan(self.inspection())
        second = self.host_plan(self.inspection())
        self.assertEqual(first, second)
        self.assertEqual(first["would_install"], [])

    def test_18a_compatible_package_patch_is_satisfied_during_apply(self) -> None:
        value = self.inspection()
        record = next(item for item in self.configs["ubuntu-package.lock.json"]["packages"] if item["name"] == "systemd")
        compatible = record["observed_version"] + ".4"
        self.assertRegex(compatible, record["compatible_version_regex"])
        value["installed_packages"][record["name"]] = compatible
        self.assertEqual(self.host_plan(value)["would_install"], [])

        class Runner:
            def __call__(self, argv, **kwargs):
                arguments = tuple(argv)
                return CommandResult(arguments, 0, "", "")

        with mock.patch("system_x_bootstrap.packages._apt_essential_packages", return_value=set()):
            result = apply_host(
                value,
                self.configs["ubuntu-package.lock.json"],
                self.configs["cuda-wsl.lock.json"],
                transaction=ActiveTransaction(),
                authorized=True,
                elevated=False,
                runner=Runner(),
            )
        self.assertFalse(result["changed"])

    def test_19_unrelated_package_preservation(self) -> None:
        result = CommandResult(("apt-get",), 0, "Inst unrelated [1.0] (2.0 repo [amd64])\n", "")
        with self.assertRaises(BootstrapError):
            validate_apt_simulation(
                result,
                direct_names=("cuda-toolkit-13-3",),
                cuda_lock=self.configs["cuda-wsl.lock.json"],
            )

    def test_19a_apt_configuration_lines_are_allowed(self) -> None:
        result = CommandResult(
            ("apt-get",),
            0,
            "Inst cmake (4.2.3-2ubuntu2 Ubuntu)\nConf cmake (4.2.3-2ubuntu2 Ubuntu)\n",
            "",
        )
        self.assertEqual(
            validate_apt_simulation(
                result,
                direct_names=("cmake",),
                cuda_lock=self.configs["cuda-wsl.lock.json"],
            ),
            ["cmake"],
        )

    def test_19b_prior_version_dependency_is_allowed(self) -> None:
        result = CommandResult(
            ("apt-get",),
            0,
            "Inst libc-gconv-modules-extra [2.43-2ubuntu2] (2.43-2ubuntu2.3 Ubuntu)\n",
            "",
        )
        self.assertEqual(
            validate_apt_simulation(
                result,
                direct_names=("build-essential",),
                dependency_names=("libc-gconv-modules-extra",),
                cuda_lock=self.configs["cuda-wsl.lock.json"],
            ),
            ["libc-gconv-modules-extra"],
        )

    def test_19c_apt_dependency_closure_is_recursive(self) -> None:
        responses = {
            "build-essential": "build-essential\n  Depends: libc6-dev\n",
            "libc6-dev": "libc6-dev\n  Depends: libc6\n",
            "libc6": "libc6\n  Depends: libc-gconv-modules-extra\n",
            "libc-gconv-modules-extra": "libc-gconv-modules-extra\n",
        }

        def runner(argv, **kwargs):
            return CommandResult(tuple(argv), 0, responses[argv[-1]], "")

        self.assertEqual(
            _apt_dependency_closure(("build-essential",), runner),
            {"build-essential", "libc6-dev", "libc6", "libc-gconv-modules-extra"},
        )

    def test_19d_essential_package_compatibility_is_allowed(self) -> None:
        result = CommandResult(
            ("apt-get",),
            0,
            "Inst libc-bin [2.43-2ubuntu2] (2.43-2ubuntu2.3 Ubuntu)\n",
            "",
        )
        self.assertEqual(
            validate_apt_simulation(
                result,
                direct_names=("build-essential",),
                essential_names=("libc-bin",),
                cuda_lock=self.configs["cuda-wsl.lock.json"],
            ),
            ["libc-bin"],
        )

        class Runner:
            def __call__(self, argv, **kwargs):
                return CommandResult(tuple(argv), 0, "base-files\tyes\nlibc-bin\tyes\ncurl\tno\n", "")

        self.assertEqual(_apt_essential_packages(Runner()), {"base-files", "libc-bin"})


    def test_19e_reverse_dependency_compatibility_is_allowed(self) -> None:
        result = CommandResult(
            ("apt-get",),
            0,
            "Inst locales [2.43-2ubuntu2] (2.43-2ubuntu2.3 Ubuntu)\n",
            "",
        )
        self.assertEqual(
            validate_apt_simulation(result, direct_names=("build-essential",), dependency_names=("libc6",), reverse_dependency_names=("locales",), cuda_lock=self.configs["cuda-wsl.lock.json"]),
            ["locales"],
        )

        class Runner:
            def __call__(self, argv, **kwargs):
                return CommandResult(tuple(argv), 0, "libc6\nReverse Depends:\n  libc-bin\n  locales\n  <virtual>\n", "")

        self.assertEqual(_apt_reverse_dependency_names(("libc6",), Runner()), {"libc-bin", "locales"})

    def test_19f_cuda_application_builds_dependency_closure_before_reverse_query(self) -> None:
        cuda_record = next(
            record
            for record in self.configs["ubuntu-package.lock.json"]["packages"]
            if record["source_repository_identity"] == "nvidia-cuda-wsl-ubuntu-x86_64"
        )
        package_lock = {"packages": [cuda_record]}
        calls: list[tuple[str, ...]] = []

        class Runner:
            def __call__(self, argv, **kwargs):
                arguments = tuple(argv)
                calls.append(arguments)
                if arguments[0] == "dpkg-query":
                    return CommandResult(arguments, 0, "libc-bin" + chr(9) + "yes" + chr(10), "")
                if arguments[:2] == ("apt-cache", "policy"):
                    version = cuda_record["observed_version"]
                    return CommandResult(arguments, 0, "Candidate: " + version + chr(10) + " *** " + version + " 500" + chr(10), "")
                if arguments[:2] == ("apt-cache", "depends"):
                    return CommandResult(arguments, 0, "", "")
                if arguments[:2] == ("apt-cache", "rdepends"):
                    return CommandResult(arguments, 0, "Reverse Depends:" + chr(10), "")
                if arguments[:2] == ("apt-get", "--simulate"):
                    return CommandResult(arguments, 0, "Inst " + cuda_record["name"] + " (" + cuda_record["observed_version"] + ")" + chr(10), "")
                return CommandResult(arguments, 0, "", "")

        transaction = ActiveTransaction()
        with mock.patch("system_x_bootstrap.packages.configure_official_cuda_source"):
            result = apply_host(
                {"installed_packages": {}},
                package_lock,
                self.configs["cuda-wsl.lock.json"],
                transaction=transaction,
                authorized=True,
                elevated=True,
                runner=Runner(),
            )
        self.assertFalse(result["removed_packages"])
        dependency_index = next(i for i, call in enumerate(calls) if call[:2] == ("apt-cache", "depends"))
        reverse_index = next(i for i, call in enumerate(calls) if call[:2] == ("apt-cache", "rdepends"))
        self.assertLess(dependency_index, reverse_index)

    def test_19g_private_environment_proof_initializes_site_without_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "repo"
            source = root / "INSPECTOR"
            destination = root / "INSPECTOR/.venv"
            source.mkdir(parents=True)
            (destination / "bin").mkdir(parents=True)
            (destination / "bin/python").write_text("fixture\\n", encoding="utf-8")
            calls: list[tuple[str, ...]] = []

            class Runner:
                def __call__(self, argv, **kwargs):
                    arguments = tuple(argv)
                    calls.append(arguments)
                    return CommandResult(arguments, 0, "{}\\n", "")

            _verify_imports(
                repository_paths(root),
                destination,
                {"artifacts": [], "source_paths": ["INSPECTOR"], "post_install_import_roots": ["system_x_inspector"]},
                Runner(),
            )
            self.assertEqual(calls[0][1:4], ("-B", "-I", "-s"))
            self.assertNotIn("-S", calls[0])

    def test_20_vendored_source_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths, lock, _, _ = vendored_fixture(Path(temporary), self.configs["llama-build.lock.json"])
            result = inspect_vendored_source(paths, lock)
            self.assertTrue(result["exact"])
            self.assertEqual(result["state"], "VENDORED_SOURCE_VERIFIED")

    def test_21_missing_vendored_file_block(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths, lock, source, _ = vendored_fixture(Path(temporary), self.configs["llama-build.lock.json"])
            (source / "LICENSE").unlink()
            self.assertFalse(inspect_vendored_source(paths, lock)["exact"])

    def test_22_extra_vendored_file_block(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths, lock, source, _ = vendored_fixture(Path(temporary), self.configs["llama-build.lock.json"])
            (source / "extra.txt").write_text("extra\n", encoding="utf-8")
            self.assertFalse(inspect_vendored_source(paths, lock)["exact"])

    def test_23_content_mismatch_block(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths, lock, source, _ = vendored_fixture(Path(temporary), self.configs["llama-build.lock.json"])
            (source / "src/main.cpp").write_text("changed\n", encoding="utf-8")
            self.assertFalse(inspect_vendored_source(paths, lock)["exact"])

    def test_23a_mode_mismatch_block(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths, lock, source, _ = vendored_fixture(Path(temporary), self.configs["llama-build.lock.json"])
            (source / "LICENSE").chmod(0o755)
            self.assertFalse(inspect_vendored_source(paths, lock)["exact"])

    def test_23b_identity_record_mismatch_block(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths, lock, _, identity_path = vendored_fixture(Path(temporary), self.configs["llama-build.lock.json"])
            identity = json.loads(identity_path.read_text(encoding="utf-8"))
            identity["commit"] = "0" * 40
            identity_path.write_text(json.dumps(identity, sort_keys=True) + "\n", encoding="utf-8")
            self.assertFalse(inspect_vendored_source(paths, lock)["exact"])

    def test_23c_nested_git_block(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths, lock, source, _ = vendored_fixture(Path(temporary), self.configs["llama-build.lock.json"])
            (source / ".git").mkdir()
            self.assertFalse(inspect_vendored_source(paths, lock)["exact"])

    def test_23d_locked_build_output_is_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths, lock, source, _ = vendored_fixture(Path(temporary), self.configs["llama-build.lock.json"])
            (source / "build").mkdir()
            (source / "build" / "CMakeCache.txt").write_text("generated\\n", encoding="utf-8")
            self.assertTrue(inspect_vendored_source(paths, lock)["exact"])

    def test_23d1_unexpected_extra_build_output_block(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths, lock, source, _ = vendored_fixture(Path(temporary), self.configs["llama-build.lock.json"])
            (source / "build-extra").mkdir()
            (source / "build-extra" / "unexpected.o").write_bytes(b"generated")
            self.assertFalse(inspect_vendored_source(paths, lock)["exact"])

    def test_23e_vendored_mode_never_calls_network_runner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths, lock, _, _ = vendored_fixture(Path(temporary), self.configs["llama-build.lock.json"])
            def forbidden(*args, **kwargs):
                raise AssertionError("network attempted in vendored mode")
            self.assertTrue(inspect_vendored_source(paths, lock, forbidden)["exact"])

    def test_23f_initialize_submodules_is_verification_alias(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths, lock, _, _ = vendored_fixture(Path(temporary), self.configs["llama-build.lock.json"])
            transaction = ActiveTransaction()
            result = initialize_submodules(paths, lock, transaction=transaction, authorized=True)
            self.assertFalse(result["changed"])
            self.assertEqual(result["state"], "VENDORED_SOURCE_VERIFIED")

    def test_24_cmake_profile_round_trip(self) -> None:
        lock = self.configs["llama-build.lock.json"]
        arguments = cmake_arguments(self.paths, lock)
        for key, value in lock["cmake_options"].items():
            self.assertIn(f"-D{key}={value}", arguments)
        with tempfile.TemporaryDirectory() as temporary:
            build = Path(temporary)
            lines = [f"{key}:STRING={value}" for key, value in lock["cmake_options"].items()]
            (build / "CMakeCache.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            self.assertTrue(_cache_matches(build, lock))

    def test_25_runtime_layout_round_trip(self) -> None:
        entries = expand_runtime_layout(self.paths, self.configs["runtime-layout.json"])
        self.assertEqual(len(entries), 66)
        self.assertEqual(len({item["path"] for item in entries}), 66)

    def test_25a_runtime_source_contract_matches_current_source(self) -> None:
        contract = self.configs["runtime-layout.json"]
        for relative, expected in contract["source_contract_sha256"].items():
            with self.subTest(path=relative):
                source = REPOSITORY_ROOT / relative
                self.assertTrue(source.is_file())
                self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), expected)

    def test_25b_runtime_layout_creates_missing_parent_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "repo"
            root.mkdir()
            source = root / "source.py"
            source.write_text("fixture\n", encoding="utf-8")
            api_python = root / "model-api-gguf/api_service/.venv/bin/python"
            api_python.parent.mkdir(parents=True)
            api_python.symlink_to("/usr/bin/python3.14")
            contract = {
                "identity": "fixture.runtime.nested.v1",
                "version": 1,
                "entry_count": 1,
                "groups": [{
                    "owner_component": "fixture",
                    "cleanup_owner": "fixture",
                    "mode": "0755",
                    "secret": False,
                    "paths": ["model-api-native/MODEL"],
                }],
                "registry": {"database": "db.sqlite3", "schema_identity": "fixture", "schema_version": 1},
                "source_contract_sha256": {"source.py": hashlib.sha256(b"fixture\n").hexdigest()},
                "completion_marker": "marker.json",
            }
            transaction = mock.Mock()
            transaction._entered = True
            transaction.claim_created_path.side_effect = lambda relative: root / relative
            runner = mock.Mock(return_value=CommandResult(tuple(), 0, "", ""))
            with mock.patch("system_x_bootstrap.runtime.verify_empty_registry", return_value={"integrity": "ok"}):
                result = initialize_runtime(repository_paths(root), contract, transaction=transaction, authorized=True, runner=runner)
            self.assertTrue(result["changed"])
            self.assertTrue((root / "model-api-native/MODEL").is_dir())
            self.assertEqual(stat.S_IMODE((root / "model-api-native/MODEL").stat().st_mode), 0o755)
            self.assertTrue((root / "marker.json").is_file())
    def test_25c_service_source_contract_matches_current_source(self) -> None:
        contract = self.configs["service-registration.json"]
        for relative, expected in contract["source_contract_sha256"].items():
            with self.subTest(path=relative):
                source = REPOSITORY_ROOT / relative
                self.assertTrue(source.is_file())
                self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), expected)

    def test_26_path_traversal_block(self) -> None:
        with self.assertRaises(BootstrapError):
            resolve_contained(REPOSITORY_ROOT, "../escape")

    def test_27_symlink_escape_block(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "root"
            outside = Path(temporary) / "outside"
            root.mkdir()
            outside.mkdir()
            (root / "link").symlink_to(outside, target_is_directory=True)
            with self.assertRaises(BootstrapError):
                resolve_contained(root, "link/value")

    def test_28_existing_unknown_runtime_collision_block(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "repo"
            root.mkdir()
            source = root / "source.py"
            source.write_text("fixture\n", encoding="utf-8")
            existing = root / "model-api-gguf/RUNTIME"
            existing.mkdir(parents=True)
            contract = {
                "identity": "fixture.runtime.v1",
                "version": 1,
                "entry_count": 1,
                "groups": [{"owner_component": "fixture", "cleanup_owner": "fixture", "mode": "0755", "secret": False, "paths": ["model-api-gguf/RUNTIME"]}],
                "registry": {"database": "model-api-gguf/RUNTIME/db.sqlite3", "schema_identity": "fixture", "schema_version": 1},
                "source_contract_sha256": {"source.py": hashlib.sha256(b"fixture\n").hexdigest()},
                "completion_marker": "model-api-gguf/RUNTIME/marker.json",
            }
            with self.assertRaises(BootstrapError) as caught:
                initialize_runtime(repository_paths(root), contract, transaction=ActiveTransaction(), authorized=True)
            self.assertEqual(caught.exception.code, ErrorCode.RUNTIME_COLLISION)

    def test_29_existing_credential_collision_block(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "repo"
            root.mkdir()
            contract = copy.deepcopy(self.configs["credential-initialization.json"])
            fixture = b"credential-fixture\n"
            digest = hashlib.sha256(fixture).hexdigest()
            for relative in contract["source_contract_sha256"]:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(fixture)
                contract["source_contract_sha256"][relative] = digest
            pepper = root / contract["pepper"]
            pepper.parent.mkdir(parents=True, exist_ok=True)
            pepper.write_bytes(b"partial")
            with self.assertRaises(BootstrapError) as caught:
                initialize_credentials(repository_paths(root), contract, transaction=ActiveTransaction(), authorized=True)
            self.assertEqual(caught.exception.code, ErrorCode.CREDENTIAL_COLLISION)

    def test_30_new_key_raw_output_leak_block(self) -> None:
        class RawRunner:
            def __call__(self, argv, *, cwd=None, env=None, timeout=60):
                raw = "sxk_v1_" + "a" * 32 + "_" + "B" * 43
                return CommandResult(tuple(argv), 0, raw + "\n", "")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "repo"
            root.mkdir()
            paths = repository_paths(root)
            contract = copy.deepcopy(self.configs["credential-initialization.json"])
            python = root / contract["python_environment"]
            python.parent.mkdir(parents=True)
            python.write_text("fixture\n", encoding="utf-8")
            python.chmod(0o755)
            (root / contract["source_root"]).mkdir(parents=True)
            with self.assertRaises(BootstrapError) as caught:
                _credential_command(paths, contract, ("inspect",), RawRunner())
            self.assertEqual(caught.exception.code, ErrorCode.SECRET_POLICY_VIOLATION)

    def test_30a_standard_venv_interpreter_symlink_is_accepted(self) -> None:
        calls: list[tuple[str, ...]] = []

        class Runner:
            def __call__(self, argv, *, cwd=None, env=None, timeout=60):
                arguments = tuple(argv)
                calls.append(arguments)
                return CommandResult(arguments, 0, '{"ok":true,"result":{"accepted":true}}', "")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "repo"
            root.mkdir()
            paths = repository_paths(root)
            contract = copy.deepcopy(self.configs["credential-initialization.json"])
            python = root / contract["python_environment"]
            python.parent.mkdir(parents=True)
            python.symlink_to("/usr/bin/python3.14")
            (root / contract["source_root"]).mkdir(parents=True)
            result = _credential_command(paths, contract, ("inspect",), Runner())

        self.assertEqual(result, {"accepted": True})
        self.assertEqual(calls[0][0], str(python))

    def adapter_fixture(self) -> tuple[dict[str, object], list[tuple[str, ...]]]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name) / "repo"
        home = Path(temporary.name) / "home"
        root.mkdir()
        home.mkdir()
        contract = copy.deepcopy(self.configs["service-registration.json"])
        fixture = b"service-fixture\n"
        digest = hashlib.sha256(fixture).hexdigest()
        for relative in contract["source_contract_sha256"]:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(fixture)
            contract["source_contract_sha256"][relative] = digest
        for relative in (contract["adapter"]["runtime_root"], contract["supervisor"]["runtime_root"]):
            (root / relative).mkdir(parents=True, exist_ok=True)

        calls: list[tuple[str, ...]] = []
        class Runner:
            def __call__(self, argv, *, cwd=None, env=None, timeout=60):
                arguments = tuple(argv)
                calls.append(arguments)
                if "initialize-desired-state" in arguments:
                    desired = Path(arguments[arguments.index("--desired-state-path") + 1])
                    desired.write_text('{"fixture":true}\n', encoding="utf-8")
                    value = {"ok": True, "desired_state": "STOPPED"}
                elif "register" in arguments:
                    unit = Path(arguments[arguments.index("--unit-path") + 1])
                    unit.parent.mkdir(parents=True, exist_ok=True)
                    unit.write_text("[Unit]\nDescription=fixture\n", encoding="utf-8")
                    manifest = root / contract["adapter"]["runtime_root"] / "linux-systemd-user/manifest.json"
                    manifest.parent.mkdir(parents=True, exist_ok=True)
                    manifest.write_text('{"fixture":true}\n', encoding="utf-8")
                    value = {
                        "ok": True,
                        "adapter_identity": contract["adapter"]["identity"],
                        "adapter_version": contract["adapter"]["version"],
                        "registered": True,
                        "enabled": False,
                        "active": False,
                    }
                else:
                    value = {"ok": True}
                return CommandResult(arguments, 0, json.dumps(value) + "\n", "")
        result = register_platform_service(
            repository_paths(root), contract, transaction=ActiveTransaction(), authorized=True, runner=Runner(), home=home
        )
        return result, calls

    def test_31_adapter_fixture_render_pass(self) -> None:
        profile = json.loads(render_operating_profile(self.configs["service-registration.json"]))
        self.assertEqual(profile["schema_version"], "system-x.service-operating-profile.v1")
        result, _ = self.adapter_fixture()
        self.assertTrue(result["changed"])
        self.assertEqual(result["initial_desired_state"], "STOPPED")

    def test_32_adapter_real_registration_absent(self) -> None:
        unit = Path.home() / ".config/systemd/user/system-x.service"
        before = hashlib.sha256(unit.read_bytes()).hexdigest() if unit.is_file() else None
        _, calls = self.adapter_fixture()
        after = hashlib.sha256(unit.read_bytes()).hexdigest() if unit.is_file() else None
        self.assertEqual(before, after)
        self.assertFalse(any(argument == "/usr/bin/systemctl" for call in calls for argument in call))

    def test_33_service_control_absent(self) -> None:
        _, calls = self.adapter_fixture()
        flattened = [value for call in calls for value in call]
        self.assertNotIn("start", flattened)
        self.assertNotIn("enable", flattened)
        self.assertNotIn("restart", flattened)

    def test_34a_llama_probes_bind_build_library_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "repo"
            root.mkdir()
            lock = copy.deepcopy(self.configs["llama-build.lock.json"])
            binary = root / lock["binary"]
            binary.parent.mkdir(parents=True)
            binary.write_text("fixture" + chr(10), encoding="utf-8")
            binary.chmod(0o755)
            calls: list[tuple[tuple[str, ...], dict[str, str]]] = []

            class ProbeRunner:
                def __call__(self, argv, *, cwd=None, env=None, timeout=60):
                    arguments = tuple(argv)
                    calls.append((arguments, dict(env or {})))
                    if arguments[-1] == "--version":
                        output = "llama-server fixture" + chr(10)
                    elif arguments[-1] == "--list-devices":
                        output = "CUDA0: fixture" + chr(10)
                    else:
                        output = "libc.so.6 => /lib/libc.so.6" + chr(10)
                    return CommandResult(arguments, 0, output, "")

            result = verify_llama_no_model(repository_paths(root), lock, ProbeRunner())
            self.assertTrue(result["dynamic_libraries_resolved"])
            self.assertEqual(len(calls), 3)
            self.assertEqual(
                {env["LD_LIBRARY_PATH"] for _, env in calls},
                {str(binary.parent)},
            )

    def test_34b_service_activation_uses_native_enable_then_start(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "repo"
            home = Path(temporary) / "home"
            root.mkdir()
            home.mkdir()
            adapter = root / "adapter.py"
            adapter.write_text("fixture" + chr(10), encoding="utf-8")
            (root / "runtime").mkdir()
            contract = {
                "source_contract_sha256": {},
                "adapter": {
                    "source_entrypoint": "adapter.py",
                    "runtime_root": "runtime",
                    "identity": "fixture.adapter",
                },
            }
            calls: list[tuple[str, ...]] = []

            def fake_run_json(command, argv, purpose):
                del command, purpose
                calls.append(argv)
                return {"ok": True, "enabled": True, "active": argv[-1] == "start"}

            with (
                mock.patch(
                    "system_x_bootstrap.service.service_status",
                    return_value={
                        "unit_present": True,
                        "adapter_manifest_present": True,
                        "operating_profile_present": True,
                        "desired_state_present": True,
                    },
                ),
                mock.patch(
                    "system_x_bootstrap.service._run_json",
                    side_effect=fake_run_json,
                ),
            ):
                result = activate_platform_service(
                    repository_paths(root),
                    contract,
                    transaction=ActiveTransaction(),
                    authorized=True,
                    runner=mock.Mock(),
                    home=home,
                )
            self.assertTrue(result["active"])
            self.assertEqual([argv[-1] for argv in calls], ["status", "enable", "start"])

    def test_34_model_load_absent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "repo"
            root.mkdir()
            paths = repository_paths(root)
            lock = copy.deepcopy(self.configs["llama-build.lock.json"])
            binary = root / lock["binary"]
            binary.parent.mkdir(parents=True)
            binary.write_text("fixture\n", encoding="utf-8")
            binary.chmod(0o755)
            runner = GitRunner(lock)
            result = verify_llama_no_model(paths, lock, runner)
            self.assertFalse(result["model_loaded"])
            flattened = [value for call in runner.calls for value in call]
            self.assertNotIn("--model", flattened)
            self.assertFalse(any(value.endswith(".gguf") for value in flattened))

    def test_35_network_absent_for_inspect_and_plan(self) -> None:
        package_lock = self.configs["ubuntu-package.lock.json"]
        cuda_lock = self.configs["cuda-wsl.lock.json"]
        calls: list[tuple[str, ...]] = []
        class Runner:
            def __call__(self, argv, *, cwd=None, env=None, timeout=60):
                arguments = tuple(argv)
                calls.append(arguments)
                if arguments[:2] == ("uname", "-m"):
                    output = "x86_64\n"
                elif arguments[0] == "dpkg-query":
                    output = "".join(f"{item['name']}\t{item['observed_version']}\tii \n" for item in package_lock["packages"])
                elif arguments[0] == "python3.14" and "ctypes" in arguments[-1]:
                    output = '{"cu_init":0,"cu_device_get_count":0,"device_count":1}\n'
                elif arguments[0] == "python3.14":
                    output = '{"implementation":"cpython","version":[3,14,4],"venv":true,"ensurepip":true}\n'
                elif arguments[0] == "/usr/lib/wsl/lib/nvidia-smi":
                    output = "NVIDIA GeForce RTX 3060, 595.97, 8.6, 12288\n"
                elif arguments[0] == "/usr/local/cuda-13.3/bin/nvcc":
                    output = "Cuda compilation tools, release 13.3, V13.3\n"
                elif arguments[0] == "git" and "remote" in arguments:
                    output = "https://github.com/ggml-org/llama.cpp\n"
                elif arguments[0] == "git" and "rev-parse" in arguments:
                    output = "3ce7da2c852c538c4c5f9806da27029cf8c9cc4a\n"
                elif arguments[0] == "git" and "status" in arguments:
                    output = ""
                else:
                    output = "fixture\n"
                return CommandResult(arguments, 0, output, "")
        with tempfile.TemporaryDirectory() as temporary:
            filesystem = Path(temporary)
            for relative, content in {
                "etc/os-release": 'ID=ubuntu\nVERSION_ID="26.04"\nVERSION_CODENAME=resolute\n',
                "proc/sys/kernel/osrelease": "6.18-microsoft-standard-WSL2\n",
                "proc/version": "Linux Microsoft WSL2\n",
                "proc/1/comm": "systemd\n",
            }.items():
                path = filesystem / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            for relative in (
                "dev/dxg", "usr/lib/wsl/lib/libcuda.so.1", "usr/lib/wsl/lib/nvidia-smi",
                "usr/bin/apt-get", "usr/bin/apt-cache", "usr/bin/dpkg-query",
            ):
                path = filesystem / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            with mock.patch("socket.create_connection", side_effect=AssertionError("network attempted")), mock.patch(
                "urllib.request.urlopen", side_effect=AssertionError("network attempted")
            ):
                inspection = HostInspector(REPOSITORY_ROOT, filesystem_root=filesystem, runner=Runner()).inspect(
                    [item["name"] for item in package_lock["packages"]], cuda_lock["forbidden_package_patterns"]
                )
                plan = build_host_plan(
                    inspection, self.configs["ubuntu-26.04-wsl2-host.json"], package_lock, cuda_lock
                )
        self.assertFalse(plan["network_used"])
        self.assertFalse(any(call[0] in ("apt-get", "apt-cache", "systemctl") for call in calls))

    def test_36_transaction_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "repo"
            root.mkdir()
            paths = repository_paths(root)
            with self.assertRaises(RuntimeError):
                with BootstrapTransaction(paths, "fixture", "3" * 64, initial_state().as_dict(), True):
                    raise RuntimeError("fixture failure")
            self.assertEqual(len(incomplete_transactions(paths.transaction_directory)), 1)
            with self.assertRaises(BootstrapError):
                recover_failed_clean_transactions(paths.transaction_directory, authorized=False)
            self.assertEqual(len(recover_failed_clean_transactions(paths.transaction_directory, authorized=True)), 1)
            self.assertEqual(incomplete_transactions(paths.transaction_directory), [])

    def test_37_idempotent_completed_operation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "repo"
            destination = root / "component/.venv"
            destination.mkdir(parents=True)
            paths = repository_paths(root)
            lock = {"identity": "fixture.environment-lock.v1"}
            environment = {
                "environment_identity": "fixture",
                "relative_destination": "component/.venv",
                "system_site_packages": False,
                "user_site": "disabled",
            }
            marker = _environment_marker(lock, environment)
            (destination / ".system-x-bootstrap-environment.json").write_text(
                json.dumps(marker, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
            )
            self.assertEqual(environment_status(paths, lock, environment), "ready")
            self.assertEqual(environment_status(paths, lock, environment), "ready")

    def test_38_machine_result_schema(self) -> None:
        result = MachineResult("fixture", "ok", "CLONED").as_dict()
        schema = json.loads((BOOTSTRAP_ROOT / "schemas/machine-result.schema.json").read_text(encoding="utf-8"))
        self.assertEqual(set(result), set(schema["required"]))
        self.assertEqual(schema["additionalProperties"], False)
        self.assertEqual(result["schema"], schema["properties"]["schema"]["const"])

    def test_39_no_secret_in_results_or_receipts(self) -> None:
        result = MachineResult("fixture", "ok", "CLONED", details={"token": "sensitive", "raw_key": "sensitive"}).as_dict()
        self.assertEqual(result["details"]["token"], "[REDACTED]")
        self.assertEqual(result["details"]["raw_key"], "[REDACTED]")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "repo"
            root.mkdir()
            paths = repository_paths(root)
            receipt = write_receipt(
                paths, receipt_id="4" * 32, operation="fixture", prestate=initial_state(), poststate="CLONED",
                plan_identity="5" * 64, changed=False, details={"password": "sensitive"}
            )
            self.assertNotIn("sensitive", receipt.read_text(encoding="utf-8"))

    def test_40_no_absolute_old_host_path_in_persistent_source(self) -> None:
        forbidden = ("".join(("/", "home", "/", "user", "/")), "OPEN" + "CLAW", "UNCENSORED" + "-ENV")
        for path in BOOTSTRAP_ROOT.rglob("*"):
            if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
                continue
            if path.is_file() and not path.is_symlink():
                text = path.read_text(encoding="utf-8")
                for value in forbidden:
                    self.assertNotIn(value, text, str(path))

    def test_41_schemas_are_closed_utf8_json(self) -> None:
        schemas = sorted((BOOTSTRAP_ROOT / "schemas").glob("*.json"))
        self.assertEqual(len(schemas), 6)
        for path in schemas:
            value = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(value["type"], "object")
            self.assertFalse(value["additionalProperties"])

    def test_42_safe_in_root_os_release_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            filesystem = Path(temporary)
            target = filesystem / "usr/lib/os-release"
            target.parent.mkdir(parents=True)
            target.write_text("ID=ubuntu\nVERSION_ID=26.04\n", encoding="utf-8")
            link = filesystem / "etc/os-release"
            link.parent.mkdir(parents=True)
            link.symlink_to(Path("../usr/lib/os-release"))
            inspector = HostInspector(REPOSITORY_ROOT, filesystem_root=filesystem, runner=lambda *a, **k: None)
            self.assertEqual(inspector._read("/etc/os-release"), "ID=ubuntu\nVERSION_ID=26.04\n")


    def test_43_missing_external_probe_is_bounded(self) -> None:
        result = SubprocessRunner()(("/system-x/definitely-missing-probe",), timeout=5)
        self.assertEqual(result.returncode, 127)
        self.assertEqual(result.stdout, "")
        self.assertIn("FileNotFoundError", result.stderr)

    def test_44_missing_venv_is_installable_in_host_plan(self) -> None:
        value = self.inspection()
        value["python"]["venv"] = False
        value["python"]["ensurepip"] = False
        value["installed_packages"].pop("python3.14-venv", None)
        plan = self.host_plan(value)
        self.assertEqual(plan["python_capability"], "MISSING_INSTALLABLE")
        self.assertFalse(any("venv and ensurepip" in item for item in plan["blockers"]))
        self.assertIn("python3.14-venv", {item["name"] for item in plan["would_install"]})

    def test_45_installation_user_contract_rejects_root_and_sets_exact_identity(self) -> None:
        current = resolve_installation_user(os.environ.get("USER") or __import__("pwd").getpwuid(os.getuid()).pw_name)
        self.assertEqual(current.uid, os.getuid())
        self.assertEqual(current.gid, os.getgid())
        with self.assertRaises(BootstrapError):
            resolve_installation_user("root")
        with installation_user_context(current):
            self.assertEqual(os.geteuid(), current.uid)
            self.assertEqual(os.environ["HOME"], str(current.home))
            self.assertEqual(os.environ["XDG_RUNTIME_DIR"], str(current.xdg_runtime_dir))

    def test_46_elevated_entry_requires_explicit_installation_user(self) -> None:
        with mock.patch("system_x_bootstrap.command.os.geteuid", return_value=0):
            with self.assertRaises(BootstrapError):
                resolve_installation_user(None)

    def test_48_interrupted_transaction_reclaims_only_matching_stale_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "repo"
            root.mkdir()
            paths = repository_paths(root)
            prestate = initial_state().as_dict()
            plan = "7" * 64
            with BootstrapTransaction(paths, "apply-host", plan, prestate, True) as transaction:
                transaction.record("host-plan-accepted", {"fixture": True})
            record = next(paths.transaction_directory.glob("apply-host-*.jsonl"))
            lines = record.read_text(encoding="utf-8").splitlines()
            self.assertEqual(json.loads(lines[-1])["event"], "incomplete")
            record.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
            token = json.loads(lines[0])["token"]
            paths.transaction_lock.write_text(
                json.dumps(
                    {
                        "schema": "system-x.bootstrap.lock.v1",
                        "version": 1,
                        "operation": "apply-host",
                        "token": token,
                        "pid": 999999,
                        "created_utc": "fixture",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with BootstrapTransaction(paths, "apply-host", plan, prestate, True, resume_record=record) as transaction:
                transaction.complete({"fixture_stale_lock_resume": True})
            self.assertEqual(incomplete_transactions(paths.transaction_directory), [])
            self.assertFalse(paths.transaction_lock.exists())

    def test_47_interrupted_transaction_resumes_only_exact_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "repo"
            root.mkdir()
            paths = repository_paths(root)
            prestate = initial_state().as_dict()
            plan = "6" * 64
            try:
                with BootstrapTransaction(paths, "apply-host", plan, prestate, True) as transaction:
                    transaction.record("host-plan-accepted", {"fixture": True})
                    raise RuntimeError("simulated interruption")
            except RuntimeError:
                pass
            record = next(paths.transaction_directory.glob("apply-host-*.jsonl"))
            lines = record.read_text(encoding="utf-8").splitlines()
            self.assertEqual(json.loads(lines[-1])["event"], "failed-clean")
            record.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
            with BootstrapTransaction(paths, "apply-host", plan, prestate, True, resume_record=record) as transaction:
                transaction.complete({"fixture_resume": True})
            self.assertEqual(incomplete_transactions(paths.transaction_directory), [])


    def _privilege_user_fixture(self) -> InstallationUser:
        return InstallationUser(
            "user",
            1100,
            1100,
            Path("/home/user"),
            (1100,),
            Path("/run/user/1100"),
            "unix:path=/run/user/1100/bus",
        )

    def _host_preflight_fixture(self, would_install: list[dict[str, str]], *, euid: int = 1100):
        orchestrator = object.__new__(BootstrapOrchestrator)
        orchestrator.paths = mock.Mock()
        orchestrator.paths.root = Path("/tmp/system-x-fixture")
        orchestrator.installation_user = self._privilege_user_fixture()
        orchestrator.configs = {
            "ubuntu-26.04-wsl2-host.json": {},
            "ubuntu-package.lock.json": {},
            "cuda-wsl.lock.json": {},
        }
        orchestrator._inspection = mock.Mock(return_value={})
        return orchestrator, mock.patch(
            "system_x_bootstrap.orchestrator.build_host_plan",
            return_value={"blockers": [], "would_install": would_install},
        ), mock.patch("system_x_bootstrap.orchestrator.os.geteuid", return_value=euid)

    def test_P01_non_root_host_satisfied_does_not_elevate(self) -> None:
        orchestrator, plan, identity = self._host_preflight_fixture([])
        with plan, identity, mock.patch("system_x_bootstrap.orchestrator.exec_elevated_reconstruct") as handoff:
            orchestrator._prepare_reconstruct_host(allow_patch_difference=False)
        handoff.assert_not_called()

    def test_P02_non_root_missing_package_builds_exact_elevated_argv_without_transaction(self) -> None:
        orchestrator, plan, identity = self._host_preflight_fixture([{"name": "python3.14-venv"}])
        with plan, identity, mock.patch("system_x_bootstrap.orchestrator.exec_elevated_reconstruct", side_effect=SystemExit(0)) as handoff, mock.patch("system_x_bootstrap.orchestrator.BootstrapTransaction") as transaction:
            with self.assertRaises(SystemExit):
                orchestrator._prepare_reconstruct_host(allow_patch_difference=True)
        handoff.assert_called_once_with(Path("/tmp/system-x-fixture"), orchestrator.installation_user, allow_patch_difference=True)
        transaction.assert_not_called()

    def test_P03_elevated_continuation_does_not_elevate_again(self) -> None:
        orchestrator, plan, identity = self._host_preflight_fixture([{"name": "python3.14-venv"}], euid=0)
        with plan, identity, mock.patch("system_x_bootstrap.orchestrator.exec_elevated_reconstruct") as handoff:
            orchestrator._prepare_reconstruct_host(allow_patch_difference=False)
        handoff.assert_not_called()

    def test_P04_reconstruct_without_authorize_fails_before_preflight(self) -> None:
        orchestrator = object.__new__(BootstrapOrchestrator)
        with mock.patch("system_x_bootstrap.orchestrator.read_state") as read_state:
            with self.assertRaises(BootstrapError):
                orchestrator.reconstruct(authorized=False)
        read_state.assert_not_called()

    def test_P05_missing_elevation_executable_fails_closed(self) -> None:
        from system_x_bootstrap.command import build_elevated_reconstruct_argv
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "bootstrap").mkdir()
            (root / "bootstrap/run_bootstrap.py").write_text("# fixture\n", encoding="utf-8")
            with mock.patch("system_x_bootstrap.command._WSL_INIT", root / "missing-init"), mock.patch("system_x_bootstrap.command.ELEVATION_EXECUTABLE", root / "missing-sudo"):
                with self.assertRaises(BootstrapError):
                    build_elevated_reconstruct_argv(root, self._privilege_user_fixture())

    def test_P06_elevation_os_error_has_no_success_state(self) -> None:
        from system_x_bootstrap.command import exec_elevated_reconstruct
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "bootstrap").mkdir()
            (root / "bootstrap/run_bootstrap.py").write_text("# fixture\n", encoding="utf-8")
            with mock.patch("system_x_bootstrap.command.os.execve", side_effect=OSError("denied")):
                with self.assertRaises(BootstrapError):
                    exec_elevated_reconstruct(root, self._privilege_user_fixture())

    def test_P07_root_without_install_user_is_rejected(self) -> None:
        with mock.patch("system_x_bootstrap.command.os.geteuid", return_value=0):
            with self.assertRaises(BootstrapError):
                resolve_installation_user(None)

    def test_P08_root_and_unknown_install_users_are_rejected(self) -> None:
        with self.assertRaises(BootstrapError):
            resolve_installation_user("root")
        with self.assertRaises(BootstrapError):
            resolve_installation_user("definitely-unknown-xe-user")

    def test_P09_source_owner_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            foreign = InstallationUser("foreign", os.getuid() + 1, os.getgid(), Path(temporary), (os.getgid(),), Path("/run/user/99999"), "unix:path=/run/user/99999/bus")
            with self.assertRaises(BootstrapError):
                foreign.validate_repository(Path(temporary).resolve())

    def test_P10_elevation_environment_excludes_hostile_inherited_values(self) -> None:
        from system_x_bootstrap.command import build_elevated_reconstruct_environment
        environment = build_elevated_reconstruct_environment(self._privilege_user_fixture())
        self.assertNotIn("PYTHONPATH", environment)
        self.assertNotIn("PYTHONHOME", environment)
        self.assertNotIn("OPENAI_API_KEY", environment)
        self.assertEqual(environment["HOME"], "/home/user")

    def test_P11_hostile_install_user_text_is_rejected_without_shell_parsing(self) -> None:
        for hostile in ("-x", "../user", "user/name", "user\x00name"):
            with self.subTest(hostile=hostile):
                with self.assertRaises(BootstrapError):
                    InstallationUser.from_name(hostile)

    def test_P12_one_handoff_front_door_is_present_in_argv(self) -> None:
        from system_x_bootstrap.command import build_elevated_reconstruct_argv
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "bootstrap").mkdir()
            (root / "bootstrap/run_bootstrap.py").write_text("# fixture\n", encoding="utf-8")
            argv = build_elevated_reconstruct_argv(root, self._privilege_user_fixture())
        self.assertEqual(" ".join(argv).count("reconstruct"), 1)
        if argv[0] == "/init":
            self.assertEqual(argv[1:5], ("/mnt/c/Windows/System32/cmd.exe", "/d", "/s", "/c"))
            self.assertIn("--distribution Ubuntu-26.04", argv[-1])
            self.assertIn("--user root", argv[-1])
            self.assertIn("--install-user user", argv[-1])
            self.assertNotIn("password", " ".join(argv).lower())
        else:
            self.assertEqual(argv.count("/usr/bin/sudo"), 1)
            self.assertEqual(argv[argv.index("--install-user") + 1], "user")

    def test_P13_already_elevated_has_no_second_sudo_call(self) -> None:
        orchestrator, plan, identity = self._host_preflight_fixture([{"name": "pkg"}], euid=0)
        with plan, identity, mock.patch("system_x_bootstrap.orchestrator.exec_elevated_reconstruct") as handoff:
            orchestrator._prepare_reconstruct_host(allow_patch_difference=True)
        handoff.assert_not_called()

    def test_P14_installation_user_context_sets_owner_environment(self) -> None:
        current = resolve_installation_user(os.environ.get("USER") or __import__("pwd").getpwuid(os.getuid()).pw_name)
        with installation_user_context(current):
            self.assertEqual(os.environ["USER"], current.name)
            self.assertEqual(os.environ["LOGNAME"], current.name)
            self.assertEqual(os.environ["HOME"], str(current.home))
        with tempfile.TemporaryDirectory() as temporary:
            trial_home = Path(temporary) / "space-bearing trial home"
            trial_home.mkdir()
            with installation_user_context(current, home=trial_home):
                self.assertEqual(os.environ["HOME"], str(trial_home))
                self.assertEqual(os.environ["XDG_RUNTIME_DIR"], str(current.xdg_runtime_dir))

    def test_P15_user_manager_environment_derives_from_validated_uid(self) -> None:
        user = self._privilege_user_fixture()
        self.assertEqual(user.xdg_runtime_dir, Path("/run/user/1100"))
        self.assertEqual(user.dbus_session_bus_address, "unix:path=/run/user/1100/bus")

    def test_P16_preflight_creates_no_bootstrap_transaction(self) -> None:
        orchestrator, plan, identity = self._host_preflight_fixture([{"name": "pkg"}])
        with plan, identity, mock.patch("system_x_bootstrap.orchestrator.exec_elevated_reconstruct", side_effect=SystemExit(0)), mock.patch("system_x_bootstrap.orchestrator.BootstrapTransaction") as transaction:
            with self.assertRaises(SystemExit):
                orchestrator._prepare_reconstruct_host(allow_patch_difference=False)
        transaction.assert_not_called()

    def test_P17_exec_front_door_propagates_elevation_failure(self) -> None:
        from system_x_bootstrap.command import exec_elevated_reconstruct
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "bootstrap").mkdir()
            (root / "bootstrap/run_bootstrap.py").write_text("# fixture\n", encoding="utf-8")
            captured = {}
            def fake_exec(program, argv, environment):
                captured["program"] = program
                captured["argv"] = tuple(argv)
                captured["environment"] = dict(environment)
                raise OSError("route failed")
            with mock.patch("system_x_bootstrap.command.os.execve", side_effect=fake_exec):
                with self.assertRaises(BootstrapError):
                    exec_elevated_reconstruct(root, self._privilege_user_fixture())
        self.assertIn(captured["program"], {"/usr/bin/sudo", "/init"})
        self.assertIn("--authorize", " ".join(captured["argv"]))

    def test_P18_allow_patch_difference_is_forwarded_only_when_requested(self) -> None:
        from system_x_bootstrap.command import build_elevated_reconstruct_argv
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "bootstrap").mkdir()
            (root / "bootstrap/run_bootstrap.py").write_text("# fixture\n", encoding="utf-8")
            base = build_elevated_reconstruct_argv(root, self._privilege_user_fixture())
            forwarded = build_elevated_reconstruct_argv(root, self._privilege_user_fixture(), allow_patch_difference=True)
        self.assertNotIn("--allow-patch-difference", base)
        if forwarded[0] == "/init":
            self.assertIn("--allow-patch-difference", forwarded[-1])
        else:
            self.assertEqual(forwarded[-1], "--allow-patch-difference")


    def test_P19_wsl_route_is_bound_to_the_named_target(self) -> None:
        from system_x_bootstrap.command import build_elevated_reconstruct_argv
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "bootstrap").mkdir()
            (root / "bootstrap/run_bootstrap.py").write_text("# fixture\n", encoding="utf-8")
            with mock.patch.dict(os.environ, {"WSL_DISTRO_NAME": "Other-Distro"}, clear=False):
                with self.assertRaises(BootstrapError):
                    build_elevated_reconstruct_argv(root, self._privilege_user_fixture())

    def test_P20_wsl_route_carries_no_secret_or_recursive_user_entry(self) -> None:
        from system_x_bootstrap.command import build_elevated_reconstruct_argv
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "bootstrap").mkdir()
            (root / "bootstrap/run_bootstrap.py").write_text("# fixture\n", encoding="utf-8")
            with mock.patch.dict(os.environ, {"WSL_DISTRO_NAME": "Ubuntu-26.04"}, clear=False):
                argv = build_elevated_reconstruct_argv(root, self._privilege_user_fixture())
        if argv[0] == "/init":
            command = " ".join(argv)
            self.assertIn("--user root", command)
            self.assertIn("--install-user user", command)
            self.assertNotIn("sudo", command.lower())
            self.assertNotIn("password", command.lower())
            self.assertNotIn("OPENAI_API_KEY", command)

if __name__ == "__main__":
    unittest.main()
