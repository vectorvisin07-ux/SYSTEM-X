"""Isolated tests for the selected Linux systemd user-service adapter."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from service_control import operating_profile, supervisor
from service_control.platform_adapters import contract, registry
from service_control.platform_adapters import linux_systemd_user as selected


BRANCH_ROOT = Path(__file__).resolve().parents[2]


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def profile_value() -> dict:
    return {
        "schema_version": operating_profile.OPERATING_PROFILE_SCHEMA,
        "public_endpoint": {"host": "127.0.0.1", "port": 47821},
        "private_router_endpoint": {
            "host": "127.0.0.1",
            "port": 47822,
        },
        "default_model_alias": "default",
        "startup_model_policy": "always_warm",
        "automatic_recovery_enabled": True,
        "graceful_shutdown": {"enabled": True, "timeout_seconds": 5},
        "recovery_delay": {
            "initial_seconds": 0.01,
            "maximum_seconds": 1,
            "multiplier": 2,
        },
        "recovery_loop": {
            "maximum_attempts_in_window": 3,
            "attempt_window_seconds": 60,
            "stable_reset_seconds": 1,
        },
    }


class FakeManager:
    def __init__(self, unit_path: Path) -> None:
        self.unit_path = unit_path
        self.registered = False
        self.enabled = False
        self.active = False
        self.main_pid: int | None = None
        self.result: str | None = "success"
        self.n_restarts = 0
        self.start_calls = 0
        self.stop_calls = 0
        self.reload_calls = 0
        self.capability_available = True
        self.active_state_override: str | None = None
        self.sub_state_override: str | None = None

    def capability(self) -> dict:
        missing = (
            []
            if self.capability_available
            else ["systemd_user_manager"]
        )
        return {
            "available": not missing,
            "required": list(selected.REQUIRED_HOST_CAPABILITIES),
            "missing": missing,
            "foreground_activation_supported": False,
            "registration_supported": True,
            "enable_disable_supported": True,
            "restart_supported": True,
            "unregister_supported": True,
            "automatic_activation_supported": True,
        }

    def status(self) -> dict:
        return {
            "registered": self.registered,
            "enabled": self.enabled,
            "active": self.active,
            "active_state": (
                self.active_state_override
                or ("active" if self.active else "inactive")
            ),
            "sub_state": (
                self.sub_state_override
                or ("running" if self.active else "dead")
            ),
            "unit_file_state": "enabled" if self.enabled else "disabled",
            "fragment_path": (
                str(self.unit_path) if self.registered else None
            ),
            "main_pid": self.main_pid,
            "result": self.result,
            "n_restarts": self.n_restarts,
            "invocation_id": "fixture-invocation",
            "active_enter_monotonic": 1 if self.active else 0,
            "exec_main_pid": self.main_pid,
            "exec_main_code": 0,
            "exec_main_status": 0,
            "raw_properties": {},
        }

    def verify_unit(self) -> dict:
        text = self.unit_path.read_text(encoding="utf-8")
        if "ExecStart=" not in text:
            raise AssertionError("fixture unit lacks ExecStart")
        return {"exit_status": 0, "stdout": "", "stderr": ""}

    def daemon_reload(self) -> dict:
        self.reload_calls += 1
        self.registered = self.unit_path.is_file()
        if not self.registered:
            self.enabled = False
            self.active = False
            self.main_pid = None
        return {"exit_status": 0}

    def enable(self) -> dict:
        if not self.registered:
            raise AssertionError("enable without registration")
        self.enabled = True
        return {"exit_status": 0}

    def disable(self) -> dict:
        self.enabled = False
        return {"exit_status": 0}

    def start(self) -> dict:
        self.start_calls += 1
        self.active = True
        return {"exit_status": 0}

    def stop(self) -> dict:
        self.stop_calls += 1
        self.active = False
        self.main_pid = None
        self.active_state_override = None
        self.sub_state_override = None
        return {"exit_status": 0}

    def restart(self) -> dict:
        self.n_restarts += 1
        self.active = True
        return {"exit_status": 0}


class SelectedAdapterCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.profile_path = self.root / "operating-profile.json"
        self.state_path = self.root / "desired-state.json"
        self.supervisor_runtime = self.root / "supervisor-runtime"
        self.adapter_runtime = self.root / "adapter-runtime"
        self.unit_path = self.root / "user-units" / selected.SERVICE_NAME
        self.supervisor_runtime.mkdir()
        write_json(self.profile_path, profile_value())
        self.profile = operating_profile.load_operating_profile(
            self.profile_path
        )
        operating_profile.initialize_desired_state(
            self.profile, self.state_path, "STOPPED"
        )
        self.manager = FakeManager(self.unit_path)
        self.adapter = selected.LinuxSystemdUserServiceAdapter(
            self.adapter_runtime, manager=self.manager
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def configuration(self) -> dict:
        return {
            "profile_path": self.profile_path,
            "state_path": self.state_path,
            "supervisor_runtime_root": self.supervisor_runtime,
            "supervisor_entrypoint": (
                BRANCH_ROOT / "service_control/supervisor.py"
            ),
        }

    def register(self) -> dict:
        return self.adapter.register(**self.configuration())

    def manifest(self) -> dict:
        return json.loads(
            self.adapter.paths.manifest.read_text(encoding="utf-8")
        )

    def publish_supervisor_records(
        self,
        *,
        manager_pid: int | None = None,
        stale: bool = False,
        model_state: str = "READY",
    ) -> dict:
        paths = supervisor.SupervisorPaths(self.supervisor_runtime)
        paths.locks.mkdir(parents=True)
        paths.pids.mkdir(parents=True)
        paths.status.mkdir(parents=True)
        identity = supervisor.process_snapshot(os.getpid())
        recorded = dict(identity)
        if stale:
            recorded["process_start_identity"] = "procfs-start-ticks:1"
        transaction_id = "sv-fixture"
        common = {
            "supervisor_transaction_id": transaction_id,
            "profile_identity": self.profile.identity,
        }
        write_json(
            paths.active_lock,
            {
                "schema_version": supervisor.LOCK_SCHEMA,
                **common,
            },
        )
        write_json(
            paths.active_pid,
            {
                "schema_version": supervisor.PID_SCHEMA,
                **common,
                **recorded,
            },
        )
        write_json(
            paths.status_record,
            {
                "schema_version": supervisor.STATUS_SCHEMA,
                **common,
                "supervisor_identity": identity,
                "supervisor_state": "RUNNING",
                "service_readiness_state": model_state,
                "model_service_state": model_state,
                "service_operational": model_state
                in {"WAITING_FOR_MODEL", "MODEL_CANDIDATE_LOADING", "READY"},
                "inference_ready": model_state == "READY",
                "reason_code": (
                    "OK" if model_state == "READY" else "NO_READY_MODEL"
                ),
                "desired_state": "RUNNING",
                "desired_state_generation": 2,
                "observed_api_service": {"active": True},
                "observed_private_router": {"active": True},
                "observed_model_child": {
                    "present": model_state == "READY"
                },
                "warm_model_identity": (
                    {
                        "requested_alias": "default",
                        "health_state": "ready",
                    }
                    if model_state == "READY"
                    else None
                ),
            },
        )
        self.manager.active = True
        self.manager.main_pid = (
            manager_pid if manager_pid is not None else os.getpid()
        )
        return identity


class ContractRendererRegistryTests(SelectedAdapterCase):
    def test_selected_identity_is_second_and_only_registry_extension(self) -> None:
        self.assertEqual(
            registry.available_adapter_identities(),
            (
                contract.ADAPTER_IDENTITY,
                selected.ADAPTER_IDENTITY,
            ),
        )
        created = registry.create_adapter(
            selected.ADAPTER_IDENTITY, self.adapter_runtime
        )
        self.assertIsInstance(
            created, selected.LinuxSystemdUserServiceAdapter
        )
        with self.assertRaises(contract.AdapterError):
            registry.create_adapter(
                "system-x.unbounded.module.v1", self.adapter_runtime
            )

    def test_renderer_is_deterministic_manager_bounded_and_secret_free(self) -> None:
        values = self.adapter._render_configuration(
            **self.configuration()
        )
        unit = values[3].decode("utf-8")
        again = self.adapter._render_configuration(
            **self.configuration()
        )[3].decode("utf-8")
        self.assertEqual(unit, again)
        exec_lines = [
            line for line in unit.splitlines() if line.startswith("ExecStart=")
        ]
        self.assertEqual(len(exec_lines), 1)
        self.assertTrue(
            exec_lines[0].startswith(
                "ExecStart=/usr/bin/python3.14 "
                + str(BRANCH_ROOT / "service_control/supervisor.py")
                + " run --profile "
            )
        )
        for forbidden in (
            "/bin/sh",
            "/usr/bin/env",
            "uvicorn",
            "llama-server",
            "model-load",
            "EnvironmentFile=",
            "sk-",
            "sxk_",
        ):
            self.assertNotIn(forbidden, unit)
        self.assertIn("Restart=on-failure", unit)
        self.assertIn("RestartSec=20s", unit)
        self.assertIn("StartLimitBurst=3", unit)
        self.assertIn("KillMode=control-group", unit)
        self.assertIn("StandardOutput=journal", unit)

    def test_renderer_rejects_command_and_raw_key_injection(self) -> None:
        kwargs = {
            "interpreter": "/usr/bin/python3.14",
            "supervisor_entrypoint": "/tmp/supervisor.py;touch",
            "profile_path": "/tmp/profile.json",
            "state_path": "/tmp/state.json",
            "supervisor_runtime_root": "/tmp/runtime",
            "api_controller": "/tmp/api.py",
            "branch_controller": "/tmp/branch.py",
            "api_controller_sha256": "a" * 64,
            "branch_controller_sha256": "b" * 64,
            "timeout_stop_seconds": 20,
        }
        with self.assertRaises(contract.AdapterError) as caught:
            selected.render_unit(**kwargs)
        self.assertEqual(
            caught.exception.reason_code, "COMMAND_INJECTION_REJECTED"
        )
        kwargs["supervisor_entrypoint"] = "/tmp/supervisor.py"
        kwargs["api_controller_sha256"] = "sxk_" + "a" * 32
        with self.assertRaises(contract.AdapterError) as caught:
            selected.render_unit(**kwargs)
        self.assertEqual(
            caught.exception.reason_code, "RAW_CREDENTIAL_REJECTED"
        )


class RegistrationLifecycleTests(SelectedAdapterCase):
    def test_register_enable_disable_and_status_fields(self) -> None:
        result = self.register()
        self.assertTrue(result["ok"])
        self.assertTrue(result["registered"])
        self.assertFalse(result["enabled"])
        self.assertFalse(result["active"])
        self.assertTrue(result["automatic_activation_supported"])
        self.assertEqual(self.unit_path.stat().st_mode & 0o777, 0o600)
        manifest = self.manifest()
        self.assertEqual(
            manifest["adapter_identity"], selected.ADAPTER_IDENTITY
        )
        self.assertEqual(
            manifest["native_service"]["registration_path"],
            str(self.unit_path),
        )
        enabled = self.adapter.enable()
        self.assertTrue(enabled["enabled"])
        self.assertFalse(enabled["active"])
        status = self.adapter.status()["data"]["status"]
        self.assertEqual(
            status["configured_public_base_url"],
            "http://127.0.0.1:47821",
        )
        self.assertTrue(status["manager_enabled"])
        self.assertFalse(status["manager_active"])
        disabled = self.adapter.disable()
        self.assertFalse(disabled["enabled"])
        self.assertTrue(self.unit_path.is_file())
        self.assertTrue(self.adapter.paths.manifest.is_file())

    def test_registration_collision_and_symlink_fail_closed(self) -> None:
        self.unit_path.parent.mkdir(parents=True)
        self.unit_path.write_text("foreign\n", encoding="utf-8")
        before = self.unit_path.read_bytes()
        with self.assertRaises(contract.AdapterError) as caught:
            self.register()
        self.assertEqual(caught.exception.reason_code, "SERVICE_NAME_COLLISION")
        self.assertEqual(self.unit_path.read_bytes(), before)
        self.unit_path.unlink()
        target = self.root / "foreign-unit"
        target.write_text("foreign\n", encoding="utf-8")
        self.unit_path.symlink_to(target)
        with self.assertRaises(contract.AdapterError) as caught:
            self.register()
        self.assertEqual(
            caught.exception.reason_code, "SERVICE_NAME_COLLISION"
        )
        self.assertEqual(target.read_text(encoding="utf-8"), "foreign\n")

    def test_start_while_disabled_and_disable_while_active(self) -> None:
        self.register()
        with self.assertRaises(contract.AdapterError) as caught:
            self.adapter.start(wait_timeout_seconds=0.1)
        self.assertEqual(caught.exception.reason_code, "ADAPTER_DISABLED")
        self.assertEqual(self.manager.start_calls, 0)
        desired = operating_profile.load_desired_state(
            self.state_path, self.profile.identity
        )
        self.assertEqual(desired.desired_state, "STOPPED")
        self.adapter.enable()
        self.publish_supervisor_records()
        with self.assertRaises(contract.AdapterError) as caught:
            self.adapter.disable()
        self.assertEqual(
            caught.exception.reason_code, "ACTIVE_DISABLE_FORBIDDEN"
        )

    def test_stop_cancels_native_auto_restart_state(self) -> None:
        self.register()
        self.adapter.enable()
        running = operating_profile.set_desired_state(
            self.profile,
            "RUNNING",
            self.state_path,
            expected_generation=1,
        )
        self.manager.active_state_override = "activating"
        self.manager.sub_state_override = "auto-restart"
        stopped = self.adapter.stop(
            expected_generation=running.generation,
            wait_timeout_seconds=1.0,
        )
        self.assertTrue(stopped["ok"])
        self.assertEqual(stopped["desired_state"], "STOPPED")
        self.assertEqual(self.manager.stop_calls, 1)
        self.assertFalse(self.manager.active)

    def test_restart_closes_descendants_before_starting_new_supervisor(
        self,
    ) -> None:
        self.register()
        self.adapter.enable()
        operating_profile.set_desired_state(
            self.profile, "RUNNING", self.state_path, expected_generation=1
        )
        old_identity = self.publish_supervisor_records(
            model_state="WAITING_FOR_MODEL"
        )
        paths = supervisor.SupervisorPaths(self.supervisor_runtime)

        def graceful_stop() -> dict:
            self.manager.stop_calls += 1
            self.manager.active = False
            self.manager.main_pid = None
            paths.active_lock.unlink()
            paths.active_pid.unlink()
            paths.status_record.unlink()
            paths.locks.rmdir()
            paths.pids.rmdir()
            paths.status.rmdir()
            return {"exit_status": 0}

        def clean_start() -> dict:
            self.manager.start_calls += 1
            self.publish_supervisor_records(
                model_state="WAITING_FOR_MODEL"
            )
            return {"exit_status": 0}

        with mock.patch.object(
            self.manager, "stop", side_effect=graceful_stop
        ), mock.patch.object(
            self.manager, "start", side_effect=clean_start
        ):
            result = self.adapter.restart(wait_timeout_seconds=1.0)

        desired = operating_profile.load_desired_state(
            self.state_path, self.profile.identity
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["operation"], "restart")
        self.assertEqual(result["desired_state"], "RUNNING")
        self.assertEqual(desired.desired_state, "RUNNING")
        self.assertEqual(desired.generation, 4)
        self.assertEqual(self.manager.stop_calls, 1)
        self.assertEqual(self.manager.start_calls, 1)
        self.assertEqual(self.manager.n_restarts, 0)
        self.assertEqual(
            result["data"]["old_supervisor_identity"], old_identity
        )
        self.assertEqual(
            result["data"]["lifecycle"]["stop"]["desired_state"],
            "STOPPED",
        )
        self.assertEqual(
            result["data"]["lifecycle"]["start"]["desired_state"],
            "RUNNING",
        )

    def test_explicit_isolated_unregister_is_narrow_and_preserves_history(
        self,
    ) -> None:
        self.register()
        with self.assertRaises(contract.AdapterError) as caught:
            self.adapter.unregister()
        self.assertEqual(
            caught.exception.reason_code, "UNREGISTER_NOT_EXPLICIT"
        )
        result = self.adapter.unregister(explicit=True)
        self.assertTrue(result["ok"])
        self.assertFalse(self.unit_path.exists())
        self.assertFalse(self.adapter.paths.manifest.exists())
        self.assertFalse(self.adapter.paths.status.exists())
        self.assertTrue(self.adapter.paths.transactions.is_dir())
        self.assertTrue(self.profile_path.is_file())
        self.assertTrue(self.state_path.is_file())
        self.assertTrue(
            BRANCH_ROOT.joinpath("service_control/supervisor.py").is_file()
        )


class CorrelationAndNegativeTests(SelectedAdapterCase):
    def test_manager_supervisor_status_correlation(self) -> None:
        self.register()
        self.adapter.enable()
        operating_profile.set_desired_state(
            self.profile, "RUNNING", self.state_path, expected_generation=1
        )
        identity = self.publish_supervisor_records()
        result = self.adapter.status()
        status = result["data"]["status"]
        self.assertTrue(status["manager_active"])
        self.assertEqual(status["manager_main_pid"], os.getpid())
        self.assertEqual(
            status["supervisor_process_identity"], identity
        )
        self.assertEqual(status["system_x_readiness_state"], "READY")
        self.assertTrue(status["system_x_inference_ready"])

    def test_waiting_manager_remains_active_without_inference_readiness(
        self,
    ) -> None:
        self.register()
        self.adapter.enable()
        operating_profile.set_desired_state(
            self.profile, "RUNNING", self.state_path, expected_generation=1
        )
        self.publish_supervisor_records(model_state="WAITING_FOR_MODEL")
        status = self.adapter.status()["data"]["status"]
        self.assertTrue(status["manager_active"])
        self.assertTrue(status["active"])
        self.assertEqual(
            status["system_x_readiness_state"], "WAITING_FOR_MODEL"
        )
        self.assertFalse(status["system_x_inference_ready"])
        self.assertFalse(
            status["supervisor_status_summary"]["inference_ready"]
        )

    def test_active_without_supervisor_and_pid_mismatch_fail_closed(self) -> None:
        self.register()
        self.adapter.enable()
        self.manager.active = True
        self.manager.main_pid = os.getpid()
        with self.assertRaises(contract.AdapterError) as caught:
            self.adapter.status()
        self.assertEqual(
            caught.exception.reason_code,
            "MANAGER_SUPERVISOR_IDENTITY_MISMATCH",
        )
        self.publish_supervisor_records(manager_pid=os.getpid() + 1)
        with self.assertRaises(contract.AdapterError) as caught:
            self.adapter.status()
        self.assertEqual(
            caught.exception.reason_code,
            "MANAGER_SUPERVISOR_IDENTITY_MISMATCH",
        )

    def test_stale_pid_and_unit_tamper_fail_closed(self) -> None:
        self.register()
        self.adapter.enable()
        self.publish_supervisor_records(stale=True)
        with self.assertRaises(contract.AdapterError) as caught:
            self.adapter.status()
        self.assertEqual(caught.exception.reason_code, "PID_REUSE")
        paths = supervisor.SupervisorPaths(self.supervisor_runtime)
        paths.active_lock.unlink()
        paths.active_pid.unlink()
        self.manager.active = False
        self.manager.main_pid = None
        self.unit_path.write_text("foreign\n", encoding="utf-8")
        with self.assertRaises(contract.AdapterError) as caught:
            self.adapter.status()
        self.assertEqual(
            caught.exception.reason_code, "FOREIGN_SERVICE_DEFINITION"
        )

    def test_capability_and_manager_enabled_mismatch_fail_closed(self) -> None:
        self.manager.capability_available = False
        with self.assertRaises(contract.AdapterError) as caught:
            self.adapter.validate(**self.configuration())
        self.assertEqual(
            caught.exception.reason_code, "HOST_CAPABILITY_MISSING"
        )
        self.manager.capability_available = True
        self.register()
        self.manager.enabled = True
        with self.assertRaises(contract.AdapterError) as caught:
            self.adapter.status()
        self.assertEqual(
            caught.exception.reason_code, "MANAGER_STATE_MISMATCH"
        )


class StaticIsolationTests(unittest.TestCase):
    def test_platform_logic_is_isolated_and_no_direct_runtime_bypass(
        self,
    ) -> None:
        source = Path(selected.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = []
        popen_calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "Popen"
            ):
                popen_calls.append(node)
        self.assertEqual(popen_calls, [])
        self.assertFalse(
            any(
                name.startswith(("api_service", "branch_controller"))
                for name in imports
            )
        )
        self.assertNotIn("shell=True", source)
        core = (
            BRANCH_ROOT / "service_control/supervisor.py"
        ).read_text(encoding="utf-8")
        core_tree = ast.parse(core)
        core_imports = []
        for node in ast.walk(core_tree):
            if isinstance(node, ast.Import):
                core_imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                core_imports.append(node.module)
        self.assertFalse(
            any(
                "platform_adapter" in name
                or "systemd" in name
                or "linux_systemd_user" in name
                for name in core_imports
            )
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
