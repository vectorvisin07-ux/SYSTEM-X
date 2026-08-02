"""No-mutation safety and reconstruction contract matrix."""

from __future__ import annotations

import copy
import hashlib
import json
import os
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

from system_x_bootstrap.command import CommandResult
from system_x_bootstrap.config import load_registry
from system_x_bootstrap.credentials import (
    _credential_command,
    credential_status,
    initialize_credentials,
)
from system_x_bootstrap.cuda import assert_toolkit_only_packages
from system_x_bootstrap.environments import (
    _environment_marker,
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
    cmake_arguments,
    inspect_submodule,
    verify_llama_no_model,
)
from system_x_bootstrap.packages import build_host_plan, validate_apt_simulation
from system_x_bootstrap.paths import RepositoryPaths, discover_repository_root, resolve_contained
from system_x_bootstrap.result import MachineResult
from system_x_bootstrap.runtime import expand_runtime_layout, initialize_runtime
from system_x_bootstrap.service import register_platform_service, render_operating_profile
from system_x_bootstrap.state import initial_state, write_receipt
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


class ActiveTransaction:
    _entered = True

    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []

    def record(self, event: str, details: object = None) -> None:
        self.events.append((event, details))

    def claim_created_path(self, relative: str) -> Path:
        raise AssertionError(f"fixture unexpectedly claimed {relative}")


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

    def test_19_unrelated_package_preservation(self) -> None:
        result = CommandResult(("apt-get",), 0, "Inst unrelated [1.0] (2.0 repo [amd64])\n", "")
        with self.assertRaises(BootstrapError):
            validate_apt_simulation(
                result,
                direct_names=("cuda-toolkit-13-3",),
                cuda_lock=self.configs["cuda-wsl.lock.json"],
            )

    def test_20_exact_submodule_pass(self) -> None:
        self.assertTrue(inspect_submodule(self.paths, self.configs["llama-build.lock.json"], GitRunner(self.configs["llama-build.lock.json"]))["exact"])

    def test_21_wrong_submodule_origin_block(self) -> None:
        runner = GitRunner(self.configs["llama-build.lock.json"], origin="https://example.invalid/llama.cpp")
        self.assertFalse(inspect_submodule(self.paths, self.configs["llama-build.lock.json"], runner)["exact"])

    def test_22_wrong_submodule_commit_block(self) -> None:
        runner = GitRunner(self.configs["llama-build.lock.json"], commit="0" * 40)
        self.assertFalse(inspect_submodule(self.paths, self.configs["llama-build.lock.json"], runner)["exact"])

    def test_23_dirty_submodule_block(self) -> None:
        runner = GitRunner(self.configs["llama-build.lock.json"], dirty=True)
        self.assertFalse(inspect_submodule(self.paths, self.configs["llama-build.lock.json"], runner)["exact"])

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
        self.assertEqual(len(entries), 60)
        self.assertEqual(len({item["path"] for item in entries}), 60)

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
        with self.assertRaises(BootstrapError) as caught:
            _credential_command(self.paths, self.configs["credential-initialization.json"], ("inspect",), RawRunner())
        self.assertEqual(caught.exception.code, ErrorCode.SECRET_POLICY_VIOLATION)

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
                    unit = home / contract["future_generated_unit_relative_to_home"]
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
        unit = Path.home() / ".config/systemd/user/system-x-current-test.service"
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

    def test_34_model_load_absent(self) -> None:
        runner = GitRunner(self.configs["llama-build.lock.json"])
        result = verify_llama_no_model(self.paths, self.configs["llama-build.lock.json"], runner)
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
            if path.is_file() and not path.is_symlink():
                text = path.read_text(encoding="utf-8")
                for value in forbidden:
                    self.assertNotIn(value, text, str(path))

    def test_41_schemas_are_closed_utf8_json(self) -> None:
        schemas = sorted((BOOTSTRAP_ROOT / "schemas").glob("*.json"))
        self.assertEqual(len(schemas), 4)
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


if __name__ == "__main__":
    unittest.main()
