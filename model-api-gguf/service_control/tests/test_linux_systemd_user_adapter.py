"""Isolated tests for the selected Linux systemd user-service adapter."""

from __future__ import annotations

import ast
import hashlib
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

    def test_root_registration_handoffs_generated_user_files(self) -> None:
        if Path.home().stat().st_uid == 0:
            self.skipTest("root-owned home cannot model a non-root repository user")
        with mock.patch.object(selected.os, "geteuid", return_value=0):
            self.adapter._prepare_runtime()
            selected._atomic_write(self.unit_path, b"[Unit]\n")
        owner = Path.home().stat()
        self.assertEqual(self.unit_path.parent.stat().st_uid, owner.st_uid)
        self.assertEqual(self.unit_path.parent.stat().st_gid, owner.st_gid)
        self.assertEqual(self.unit_path.stat().st_uid, owner.st_uid)
        self.assertEqual(self.unit_path.stat().st_gid, owner.st_gid)

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

    def test_native_enable_skips_reload_to_preserve_stopped_floor(self) -> None:
        manager = selected.SystemdUserManager(self.unit_path)
        with mock.patch.object(
            manager, "_run", return_value={"exit_status": 0}
        ) as run:
            result = manager.enable()
        self.assertEqual(result["exit_status"], 0)
        run.assert_called_once_with(
            ["enable", "--no-reload", selected.SERVICE_NAME]
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
        expected_supervisor = selected._safe_systemd_path(
            str(BRANCH_ROOT / "service_control/supervisor.py"),
            "ExecStart supervisor entrypoint",
        )
        self.assertTrue(
            exec_lines[0].startswith(
                "ExecStart=/usr/bin/python3.14 "
                + expected_supervisor
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

    def test_renderer_quotes_space_bearing_product_paths(self) -> None:
        branch_root = self.root / "space-bearing source" / "model-api-gguf"
        supervisor = branch_root / "service_control/supervisor.py"
        profile = branch_root / "RUNTIME/service_control/operating-profile.json"
        kwargs = {
            "interpreter": str(selected.PYTHON),
            "supervisor_entrypoint": str(supervisor),
            "profile_path": str(profile),
            "state_path": str(
                branch_root / "RUNTIME/service_control/desired-state.json"
            ),
            "supervisor_runtime_root": str(
                branch_root / "RUNTIME/service_control"
            ),
            "api_controller": str(
                branch_root / "api_service_controller/controller.py"
            ),
            "branch_controller": str(
                branch_root / "branch_controller/controller.py"
            ),
            "api_controller_sha256": selected.API_CONTROLLER_SHA256,
            "branch_controller_sha256": selected.BRANCH_CONTROLLER_SHA256,
            "timeout_stop_seconds": 20,
        }
        with mock.patch.object(selected, "BRANCH_ROOT", branch_root):
            unit = selected.render_unit(**kwargs)
        self.assertIn("WorkingDirectory=/", unit)
        self.assertIn(
            f'ExecStart={selected.PYTHON} "{supervisor}" run --profile "{profile}"',
            unit,
        )
        self.assertNotIn(";", unit)
        unit_path = self.root / "space-bearing system-x.service"
        unit_path.write_text(unit, encoding="utf-8")
        unit_path.chmod(0o600)
        identity = selected._existing_product_unit_identity(unit_path)
        self.assertEqual(identity["branch_root"], str(branch_root.parent))

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

    def test_registered_stale_configuration_is_reconciled_without_start(
        self,
    ) -> None:
        self.register()
        stale_unit = self.unit_path.read_text(encoding="utf-8")
        stale_api_sha = "a" * 64
        stale_unit = stale_unit.replace(
            selected.API_CONTROLLER_SHA256, stale_api_sha
        )
        self.unit_path.write_text(stale_unit, encoding="utf-8")
        stale_manifest = self.manifest()
        stale_manifest["configuration_identity"] = "sha256:stale-fixture"
        stale_native = dict(stale_manifest["native_service"])
        stale_api = dict(stale_native["api_controller"])
        stale_api["sha256"] = stale_api_sha
        stale_native["api_controller"] = stale_api
        stale_native["service_definition_sha256"] = hashlib.sha256(
            stale_unit.encode("utf-8")
        ).hexdigest()
        stale_manifest["native_service"] = stale_native
        write_json(self.adapter.paths.manifest, stale_manifest)

        result = self.register()

        self.assertTrue(result["ok"])
        self.assertTrue(result["registered"])
        self.assertFalse(result["enabled"])
        self.assertFalse(result["active"])
        self.assertTrue(
            result["data"]["reconciled_existing_registration"]
        )
        self.assertFalse(result["data"]["process_started"])
        self.assertEqual(self.manager.start_calls, 0)
        self.assertGreaterEqual(self.manager.reload_calls, 2)
        current_unit = self.unit_path.read_text(encoding="utf-8")
        self.assertIn(selected.API_CONTROLLER_SHA256, current_unit)
        self.assertNotIn(stale_api_sha, current_unit)
        current_manifest = self.manifest()
        self.assertEqual(
            current_manifest["configuration_identity"],
            result["configuration_identity"],
        )
        self.assertEqual(
            current_manifest["native_service"]["api_controller"]["sha256"],
            selected.API_CONTROLLER_SHA256,
        )


    def test_existing_owned_unit_without_manifest_is_reconciled(self) -> None:
        old_root = self.root / "canonical-system-x"
        old_model = old_root / "model-api-gguf"
        old_unit = selected.render_unit(
            interpreter=str(selected.PYTHON),
            supervisor_entrypoint=str(old_model / "service_control/supervisor.py"),
            profile_path=str(old_model / "RUNTIME/service_control/operating-profile.json"),
            state_path=str(old_model / "RUNTIME/service_control/desired-state.json"),
            supervisor_runtime_root=str(old_model / "RUNTIME/service_control"),
            api_controller=str(old_model / "api_service_controller/controller.py"),
            branch_controller=str(old_model / "branch_controller/controller.py"),
            api_controller_sha256=selected.API_CONTROLLER_SHA256,
            branch_controller_sha256=selected.BRANCH_CONTROLLER_SHA256,
            timeout_stop_seconds=20,
        )
        old_unit = old_unit.replace(str(selected.BRANCH_ROOT), str(old_model))
        self.unit_path.parent.mkdir(parents=True)
        self.unit_path.write_text(old_unit, encoding="utf-8")
        self.unit_path.chmod(0o600)
        self.manager.registered = True
        self.manager.enabled = True
        self.manager.active = True
        self.manager.main_pid = 731

        result = self.register()

        self.assertTrue(result["ok"])
        self.assertTrue(result["registered"])
        self.assertTrue(result["enabled"])
        self.assertFalse(result["active"])
        self.assertTrue(result["data"]["reconciled_existing_registration"])
        self.assertEqual(self.manager.stop_calls, 1)
        self.assertEqual(self.manager.start_calls, 0)
        current_unit = self.unit_path.read_text(encoding="utf-8")
        self.assertIn(str(self.profile_path), current_unit)
        self.assertNotIn(str(old_model), current_unit)
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

    def test_stop_allows_stale_manifest_identity_before_floor(self) -> None:
        self.register()
        self.adapter.enable()
        running = operating_profile.set_desired_state(
            self.profile,
            "RUNNING",
            self.state_path,
            expected_generation=1,
        )
        self.publish_supervisor_records()
        stale_manifest = self.manifest()
        stale_manifest["configuration_identity"] = "sha256:stale-fixture"
        write_json(self.adapter.paths.manifest, stale_manifest)
        supervisor_paths = supervisor.SupervisorPaths(
            self.supervisor_runtime
        )

        def stop_and_remove_supervisor_records() -> dict:
            result = FakeManager.stop(self.manager)
            supervisor_paths.active_lock.unlink(missing_ok=True)
            supervisor_paths.active_pid.unlink(missing_ok=True)
            return result

        with mock.patch.object(
            self.manager,
            "stop",
            side_effect=stop_and_remove_supervisor_records,
        ):
            stopped = self.adapter.stop(
                expected_generation=running.generation,
                wait_timeout_seconds=1.0,
            )

        self.assertTrue(stopped["ok"])
        self.assertEqual(stopped["desired_state"], "STOPPED")
        self.assertEqual(self.manager.stop_calls, 1)
        self.assertFalse(self.manager.active)

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

        endpoint_checks: list[int] = []

        def delayed_endpoint_release(_host: str, port: int) -> bool:
            endpoint_checks.append(port)
            return len(endpoint_checks) > 2

        with mock.patch.object(
            self.manager, "stop", side_effect=graceful_stop
        ), mock.patch.object(
            self.manager, "start", side_effect=clean_start
        ), mock.patch.object(
            self.adapter,
            "_endpoint_free",
            side_effect=delayed_endpoint_release,
        ), mock.patch.object(
            selected.time, "sleep", return_value=None
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
        self.assertGreater(len(endpoint_checks), 2)
        self.assertGreaterEqual(result["data"]["endpoint_release"]["wait_seconds"], 0)
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

    def test_configure_static_ui_persists_and_exact_repeat_is_noop(
        self,
    ) -> None:
        self.register()
        distribution = self.root / "distribution"
        assets = distribution / "assets"
        assets.mkdir(parents=True)
        (distribution / "index.html").write_text(
            '<script type="module" src="/ui/assets/app-12345678.js"></script>'
            '<link rel="stylesheet" href="/ui/assets/app-12345678.css">',
            encoding="utf-8",
        )
        (assets / "app-12345678.js").write_text(
            "console.log('fixture');", encoding="utf-8"
        )
        (assets / "app-12345678.css").write_text(
            "body{color:white}", encoding="utf-8"
        )
        first = self.adapter.configure_static_ui(
            enabled=True,
            distribution_root=distribution,
            mount_path="/ui",
        )
        self.assertTrue(first["ok"])
        self.assertTrue(first["data"]["changed"])
        self.assertFalse(first["data"]["no_op"])
        profile = operating_profile.load_operating_profile(
            self.profile_path
        )
        self.assertTrue(profile.external_static_fields_present)
        self.assertTrue(profile.external_static_enabled)
        self.assertEqual(
            profile.external_static_distribution_root,
            str(distribution),
        )
        self.assertEqual(profile.external_static_mount_path, "/ui")
        profile_bytes = self.profile_path.read_bytes()
        state_bytes = self.state_path.read_bytes()
        manifest_bytes = self.adapter.paths.manifest.read_bytes()
        transactions = sorted(self.adapter.paths.transactions.glob("*.json"))
        second = self.adapter.configure_static_ui(
            enabled=True,
            distribution_root=distribution,
            mount_path="/ui",
        )
        self.assertTrue(second["ok"])
        self.assertFalse(second["data"]["changed"])
        self.assertTrue(second["data"]["no_op"])
        self.assertEqual(self.profile_path.read_bytes(), profile_bytes)
        self.assertEqual(self.state_path.read_bytes(), state_bytes)
        self.assertEqual(self.adapter.paths.manifest.read_bytes(), manifest_bytes)
        self.assertEqual(
            sorted(self.adapter.paths.transactions.glob("*.json")),
            transactions,
        )
        self.assertEqual(self.manager.start_calls, 0)
        self.assertEqual(self.manager.stop_calls, 0)

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

    def test_missing_user_config_directory_is_creatable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary) / "home"
            home.mkdir()
            unit_path = home / ".config/systemd/user" / selected.SERVICE_NAME
            manager = selected.SystemdUserManager(unit_path=unit_path)
            with mock.patch.object(selected.Path, "home", return_value=home):
                with mock.patch.object(
                    manager,
                    "_run",
                    return_value={"exit_status": 0, "stdout": "running\n"},
                ):
                    capability = manager.capability()
            self.assertTrue(capability["available"])
            self.assertTrue(
                capability["checks"]["user_unit_registration"]
            )
            self.assertFalse((home / ".config").exists())

    def test_prepare_runtime_creates_missing_user_unit_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            unit_path = root / ".config/systemd/user" / selected.SERVICE_NAME
            adapter_runtime = root / "adapter-runtime"
            adapter = selected.LinuxSystemdUserServiceAdapter(
                adapter_runtime,
                manager=FakeManager(unit_path),
            )
            self.assertFalse(unit_path.parent.exists())
            adapter._prepare_runtime()
            self.assertTrue(unit_path.parent.is_dir())
            self.assertEqual(
                unit_path.parent.stat().st_mode & 0o777,
                0o700,
            )
            self.assertFalse(unit_path.exists())

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
