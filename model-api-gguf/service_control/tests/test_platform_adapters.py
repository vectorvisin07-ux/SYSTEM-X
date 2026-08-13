"""Consolidated isolated platform-adapter unit and security matrix."""

from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


BRANCH_ROOT = Path(__file__).resolve().parents[2]
if str(BRANCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BRANCH_ROOT))

from service_control import operating_profile  # noqa: E402
from service_control import supervisor  # noqa: E402
from service_control.platform_adapters import contract  # noqa: E402
from service_control.platform_adapters import foreground  # noqa: E402
from service_control.platform_adapters import registry  # noqa: E402


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def profile_value() -> dict[str, object]:
    return {
        "schema_version": operating_profile.OPERATING_PROFILE_SCHEMA,
        "public_endpoint": {"host": "127.0.0.1", "port": 47931},
        "private_router_endpoint": {
            "host": "127.0.0.1",
            "port": 47932,
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


class AdapterCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.profile_path = self.root / "operating-profile.json"
        self.state_path = self.root / "desired-state.json"
        self.supervisor_runtime = self.root / "supervisor-runtime"
        self.adapter_runtime = self.root / "adapter-runtime"
        self.supervisor_runtime.mkdir()
        write_json(self.profile_path, profile_value())
        self.profile = operating_profile.load_operating_profile(
            self.profile_path
        )
        operating_profile.initialize_desired_state(
            self.profile, self.state_path, "STOPPED"
        )
        self.entrypoint = BRANCH_ROOT / "service_control/supervisor.py"
        self.adapter = foreground.ForegroundProcessHostAdapter(
            self.adapter_runtime
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def configuration(self) -> dict[str, Path]:
        return {
            "profile_path": self.profile_path,
            "state_path": self.state_path,
            "supervisor_runtime_root": self.supervisor_runtime,
            "supervisor_entrypoint": self.entrypoint,
        }

    def register(self) -> dict:
        return self.adapter.register(**self.configuration())

    def manifest(self) -> dict:
        return json.loads(
            self.adapter.paths.manifest.read_text(encoding="utf-8")
        )

    def active_record(
        self,
        schema: str,
        transaction_id: str,
        activation_identity: dict,
        supervisor_identity: dict | None,
    ) -> dict:
        manifest = self.manifest()
        return {
            "schema_version": schema,
            "adapter_transaction_id": transaction_id,
            "configuration_identity": manifest["configuration_identity"],
            "profile_identity": self.profile.identity,
            "activation_state": "ACTIVE",
            "activation_identity": activation_identity,
            "supervisor_identity": supervisor_identity,
            "created_utc": foreground.utc_now(),
        }


class ContractAndSchemaTests(AdapterCase):
    def test_startup_retries_transient_supervisor_identity_publication(self) -> None:
        child = mock.Mock()
        child.pid = 12345
        child.poll.side_effect = [None, None]
        identity = {
            "pid": child.pid,
            "process_start_identity": "start-1",
        }
        status = {"supervisor_state": "RUNNING"}
        uncertain = contract.AdapterError(
            "SUPERVISOR_IDENTITY_UNCERTAIN",
            "status publication is still converging",
            exit_code=3,
        )
        with mock.patch.object(
            self.adapter,
            "_supervisor_evidence",
            side_effect=[uncertain, (identity, status)],
        ):
            with mock.patch.object(foreground.time, "sleep"):
                observed = self.adapter._wait_for_child_evidence(
                    child, {}, 1.0
                )
        self.assertEqual(observed, (identity, status))

    def test_schema_documents_and_generated_records_validate_shape(self) -> None:
        schema_dir = BRANCH_ROOT / "service_control/platform_adapters"
        expected = {
            "adapter-manifest.schema.json": contract.MANIFEST_SCHEMA,
            "adapter-status.schema.json": contract.STATUS_SCHEMA,
            "adapter-result.schema.json": contract.RESULT_SCHEMA,
        }
        schemas = {}
        for name, identity in expected.items():
            value = json.loads(
                (schema_dir / name).read_text(encoding="utf-8")
            )
            self.assertEqual(value["$id"], identity)
            self.assertFalse(value["additionalProperties"])
            self.assertEqual(len(value["required"]), len(set(value["required"])))
            schemas[name] = value

        result = self.register()
        manifest = self.manifest()
        status = json.loads(
            self.adapter.paths.status.read_text(encoding="utf-8")
        )
        for value, name in (
            (result, "adapter-result.schema.json"),
            (manifest, "adapter-manifest.schema.json"),
            (status, "adapter-status.schema.json"),
        ):
            required = set(schemas[name]["required"])
            self.assertTrue(required.issubset(value))
            self.assertEqual(set(value), required)
        self.assertEqual(result["schema_version"], contract.RESULT_SCHEMA)
        self.assertEqual(manifest["schema_version"], contract.MANIFEST_SCHEMA)
        self.assertEqual(status["schema_version"], contract.STATUS_SCHEMA)

    def test_configuration_identity_is_deterministic_and_dynamic_free(self) -> None:
        first = self.register()
        identity = first["configuration_identity"]
        manifest = self.manifest()
        self.adapter.enable()
        changed = self.manifest()
        self.assertEqual(changed["configuration_identity"], identity)
        self.assertNotEqual(
            changed["manifest_generation"], manifest["manifest_generation"]
        )
        values = {
            "adapter_identity": contract.ADAPTER_IDENTITY,
            "adapter_version": contract.ADAPTER_VERSION,
            "supported_platform_family": contract.PLATFORM_FAMILY,
            "activation_method": contract.ACTIVATION_METHOD,
            "supervisor_entrypoint_sha256": manifest[
                "supervisor_entrypoint"
            ]["sha256"],
            "profile_path": str(self.profile_path),
            "state_path": str(self.state_path),
            "supervisor_runtime_root": str(self.supervisor_runtime),
            "profile_identity": self.profile.identity,
        }
        self.assertEqual(
            contract.compute_configuration_identity(values), identity
        )

    def test_operation_set_and_machine_parser_cover_exact_interface(self) -> None:
        parser = foreground.build_argument_parser()
        subparsers = next(
            action
            for action in parser._actions
            if isinstance(action, foreground.argparse._SubParsersAction)
        )
        self.assertEqual(
            tuple(subparsers.choices),
            contract.OPERATIONS,
        )
        for operation in contract.OPERATIONS:
            self.assertTrue(
                hasattr(
                    self.adapter,
                    operation.replace("-", "_"),
                )
            )


class RegistryAndCapabilityTests(AdapterCase):
    def test_registry_is_bounded_and_unknown_identity_writes_nothing(self) -> None:
        self.assertEqual(
            registry.available_adapter_identities(),
            (
                contract.ADAPTER_IDENTITY,
                registry.LINUX_SYSTEMD_USER_ADAPTER_IDENTITY,
            ),
        )
        with self.assertRaises(contract.AdapterError) as caught:
            registry.create_adapter(
                "arbitrary.module.or.command", self.adapter_runtime
            )
        self.assertEqual(
            caught.exception.reason_code, "ADAPTER_NOT_SUPPORTED"
        )
        self.assertFalse(self.adapter_runtime.exists())
        source = Path(registry.__file__).read_text(encoding="utf-8")
        self.assertNotIn("importlib", source)
        self.assertNotIn("entry_points", source)

    def test_capability_report_and_unknown_capability(self) -> None:
        result = self.adapter.capability()
        self.assertTrue(result["ok"])
        self.assertFalse(result["automatic_activation_supported"])
        self.assertTrue(result["data"]["available"])
        self.assertEqual(
            result["data"]["required"],
            list(contract.REQUIRED_HOST_CAPABILITIES),
        )
        with self.assertRaises(contract.AdapterError) as caught:
            self.adapter.capability(["unknown_host_marker"])
        self.assertEqual(
            caught.exception.reason_code, "HOST_CAPABILITY_MISSING"
        )

    def test_arbitrary_command_or_argv_injection_is_rejected(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exit_code = foreground.main(
                [
                    "start",
                    "--adapter-runtime-root",
                    str(self.adapter_runtime),
                    "--command",
                    "sh -c arbitrary",
                ]
            )
        self.assertEqual(exit_code, 2)
        result = json.loads(output.getvalue())
        self.assertFalse(result["ok"])
        self.assertFalse(self.adapter_runtime.exists())


class RegistrationLifecycleTests(AdapterCase):
    def test_validate_register_status_enable_disable_and_reports(self) -> None:
        state_before = self.state_path.read_bytes()
        validated = self.adapter.validate(**self.configuration())
        self.assertTrue(validated["ok"])
        self.assertFalse(self.adapter_runtime.exists())

        registered = self.register()
        self.assertTrue(registered["registered"])
        self.assertFalse(registered["enabled"])
        self.assertFalse(registered["active"])
        self.assertEqual(self.state_path.read_bytes(), state_before)
        self.assertFalse(self.adapter.paths.active_lock.exists())
        self.assertFalse(self.adapter.paths.active_pid.exists())

        status = self.adapter.status()
        self.assertFalse(status["active"])
        self.assertEqual(status["desired_state"], "STOPPED")
        self.assertTrue(self.adapter.validate()["ok"])
        self.assertEqual(
            self.adapter.configuration()["data"][
                "configuration_reference"
            ]["profile_path"],
            str(self.profile_path),
        )
        self.assertEqual(
            self.adapter.supervisor_entrypoint()["data"][
                "supervisor_entrypoint"
            ]["path"],
            str(self.entrypoint),
        )
        enabled = self.adapter.enable()
        self.assertTrue(enabled["enabled"])
        self.assertFalse(
            enabled["data"]["automatic_activation_created"]
        )
        self.assertFalse(self.adapter.paths.active_pid.exists())
        self.assertFalse(self.adapter.disable()["enabled"])

    def test_duplicate_and_inactive_lifecycle_failures(self) -> None:
        self.register()
        with self.assertRaises(contract.AdapterError) as caught:
            self.register()
        self.assertEqual(
            caught.exception.reason_code, "ADAPTER_ALREADY_REGISTERED"
        )
        with self.assertRaises(contract.AdapterError) as caught:
            self.adapter.start()
        self.assertEqual(caught.exception.reason_code, "ADAPTER_DISABLED")
        with self.assertRaises(contract.AdapterError) as caught:
            self.adapter.stop()
        self.assertEqual(caught.exception.reason_code, "ADAPTER_INACTIVE")
        with self.assertRaises(contract.AdapterError) as caught:
            self.adapter.restart()
        self.assertEqual(caught.exception.reason_code, "ADAPTER_DISABLED")
        self.adapter.enable()
        with self.assertRaises(contract.AdapterError) as caught:
            self.adapter.enable()
        self.assertEqual(
            caught.exception.reason_code, "ADAPTER_ALREADY_ENABLED"
        )
        self.adapter.disable()
        with self.assertRaises(contract.AdapterError) as caught:
            self.adapter.disable()
        self.assertEqual(
            caught.exception.reason_code, "ADAPTER_ALREADY_DISABLED"
        )

    def test_register_creates_one_registration_transaction(self) -> None:
        result = self.register()
        history = sorted(self.adapter.paths.transactions.glob("*.json"))
        self.assertEqual(len(history), 1)
        transaction = json.loads(
            history[0].read_text(encoding="utf-8")
        )
        self.assertEqual(
            transaction["schema_version"],
            foreground.ACTIVATION_TRANSACTION_SCHEMA,
        )
        self.assertEqual(transaction["operation"], "register")
        self.assertEqual(transaction["outcome"], "REGISTERED")
        self.assertEqual(
            transaction["configuration_identity"],
            self.manifest()["configuration_identity"],
        )
        self.assertEqual(
            result["data"]["registration_transaction_id"],
            transaction["adapter_transaction_id"],
        )

    def test_unregistered_start_and_all_operation_dispatch_boundaries(self) -> None:
        with self.assertRaises(contract.AdapterError) as caught:
            self.adapter.start()
        self.assertEqual(
            caught.exception.reason_code, "ADAPTER_NOT_REGISTERED"
        )
        self.register()
        self.adapter.enable()
        with mock.patch.object(
            self.adapter, "_activate", return_value={"operation": "start"}
        ):
            self.assertEqual(
                self.adapter.invoke("start")["operation"], "start"
            )
        with mock.patch.object(
            self.adapter, "_activate", return_value={"operation": "restart"}
        ):
            with mock.patch.object(
                operating_profile, "set_desired_state"
            ):
                self.assertEqual(
                    self.adapter.invoke("restart")["operation"], "restart"
                )

    def test_atomic_manifest_replace_failure_preserves_original(self) -> None:
        self.register()
        before = self.adapter.paths.manifest.read_bytes()
        with mock.patch.object(
            foreground.os,
            "replace",
            side_effect=OSError("injected atomic replace failure"),
        ):
            with self.assertRaises(contract.AdapterError) as caught:
                self.adapter.enable()
        self.assertEqual(
            caught.exception.reason_code, "ADAPTER_MANIFEST_INVALID"
        )
        self.assertEqual(self.adapter.paths.manifest.read_bytes(), before)
        self.assertEqual(
            list(
                self.adapter.paths.root.glob(
                    ".manifest.json.*.tmp"
                )
            ),
            [],
        )

    def test_unregister_is_explicit_disabled_inactive_and_narrow(self) -> None:
        self.register()
        with self.assertRaises(contract.AdapterError) as caught:
            self.adapter.unregister()
        self.assertEqual(
            caught.exception.reason_code, "UNREGISTER_NOT_EXPLICIT"
        )
        self.adapter.enable()
        with self.assertRaises(contract.AdapterError) as caught:
            self.adapter.unregister(explicit=True)
        self.assertEqual(
            caught.exception.reason_code,
            "UNREGISTER_REQUIRES_DISABLED",
        )
        self.adapter.disable()
        sentinel = self.adapter.paths.transactions / "history-sentinel.json"
        sentinel.write_text("{}\n", encoding="utf-8")
        result = self.adapter.unregister(explicit=True)
        self.assertFalse(result["registered"])
        self.assertFalse(self.adapter.paths.manifest.exists())
        self.assertFalse(self.adapter.paths.status.exists())
        self.assertTrue(sentinel.is_file())
        self.assertTrue(self.profile_path.is_file())
        self.assertTrue(self.state_path.is_file())

    def test_active_disable_and_unregister_are_rejected(self) -> None:
        self.register()
        self.adapter.enable()
        manifest = self.manifest()
        desired = operating_profile.load_desired_state(
            self.state_path, self.profile.identity
        )
        active = foreground.ActiveObservation(
            True,
            supervisor.process_snapshot(os.getpid()),
            {"supervisor_state": "RUNNING"},
            None,
            "fixture-active",
        )
        with mock.patch.object(
            self.adapter,
            "_reconcile_active",
            return_value=(manifest, active),
        ):
            with self.assertRaises(contract.AdapterError) as caught:
                self.adapter.disable()
            self.assertEqual(
                caught.exception.reason_code, "ADAPTER_ALREADY_ACTIVE"
            )
        write_json(
            self.adapter.paths.manifest,
            {**manifest, "enabled": False},
        )
        with mock.patch.object(
            self.adapter,
            "_reconcile_active",
            return_value=({**manifest, "enabled": False}, active),
        ):
            with self.assertRaises(contract.AdapterError) as caught:
                self.adapter.unregister(explicit=True)
            self.assertEqual(
                caught.exception.reason_code,
                "UNREGISTER_REQUIRES_INACTIVE",
            )
        self.assertTrue(self.adapter.paths.manifest.exists())


class PathAndIdentitySecurityTests(AdapterCase):
    def test_symlinked_profile_and_entrypoint_are_rejected(self) -> None:
        actual_profile = self.root / "actual-profile.json"
        actual_profile.write_bytes(self.profile_path.read_bytes())
        linked_profile = self.root / "linked-profile.json"
        linked_profile.symlink_to(actual_profile)
        with self.assertRaises(contract.AdapterError) as caught:
            self.adapter.validate(
                **{
                    **self.configuration(),
                    "profile_path": linked_profile,
                }
            )
        self.assertEqual(caught.exception.reason_code, "PROFILE_INVALID")

        linked_entrypoint = self.root / "linked-supervisor.py"
        linked_entrypoint.symlink_to(self.entrypoint)
        with self.assertRaises(contract.AdapterError) as caught:
            self.adapter.validate(
                **{
                    **self.configuration(),
                    "supervisor_entrypoint": linked_entrypoint,
                }
            )
        self.assertEqual(
            caught.exception.reason_code,
            "SUPERVISOR_ENTRYPOINT_INVALID",
        )
        self.assertFalse(self.adapter_runtime.exists())

    def test_symlinked_manifest_is_rejected_without_target_mutation(self) -> None:
        self.register()
        actual = self.root / "manifest-target.json"
        original = self.adapter.paths.manifest.read_bytes()
        self.adapter.paths.manifest.replace(actual)
        self.adapter.paths.manifest.symlink_to(actual)
        with self.assertRaises(contract.AdapterError) as caught:
            self.adapter.status()
        self.assertEqual(
            caught.exception.reason_code, "ADAPTER_MANIFEST_INVALID"
        )
        self.assertEqual(actual.read_bytes(), original)

    def test_manifest_profile_and_entrypoint_identity_mismatch(self) -> None:
        self.register()
        manifest = self.manifest()
        changed = {
            **manifest,
            "configuration_reference": {
                **manifest["configuration_reference"],
                "profile_identity": "sha256:" + ("f" * 64),
            },
        }
        write_json(self.adapter.paths.manifest, changed)
        with self.assertRaises(contract.AdapterError) as caught:
            self.adapter.status()
        self.assertEqual(
            caught.exception.reason_code,
            "ADAPTER_CONFIGURATION_CONFLICT",
        )

        other_runtime = self.root / "entrypoint-adapter"
        copied_entrypoint = self.root / "copied-supervisor.py"
        copied_entrypoint.write_bytes(self.entrypoint.read_bytes())
        other = foreground.ForegroundProcessHostAdapter(other_runtime)
        other.register(
            **{
                **self.configuration(),
                "supervisor_entrypoint": copied_entrypoint,
            }
        )
        copied_entrypoint.write_text(
            copied_entrypoint.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )
        with self.assertRaises(contract.AdapterError) as caught:
            other.status()
        self.assertEqual(
            caught.exception.reason_code,
            "SUPERVISOR_ENTRYPOINT_INVALID",
        )

    def test_stale_partial_pid_is_reconciled_but_pid_reuse_fails_closed(
        self,
    ) -> None:
        self.register()
        self.adapter._prepare_runtime()
        missing = {
            "pid": 999_999_991,
            "process_start_identity": "absent",
            "pgid": 999_999_991,
            "sid": 999_999_991,
            "executable": "/absent/supervisor",
            "argv_sha256": "a" * 64,
        }
        write_json(
            self.adapter.paths.active_pid,
            self.active_record(
                foreground.ACTIVATION_PID_SCHEMA,
                "stale-partial",
                missing,
                missing,
            ),
        )
        status = self.adapter.status()
        self.assertFalse(status["active"])
        self.assertFalse(self.adapter.paths.active_pid.exists())
        self.assertEqual(
            status["data"]["status"]["reconciliation_reason"],
            "STALE_PARTIAL_ACTIVE_RECORD_RECONCILED",
        )

        current = supervisor.process_snapshot(os.getpid())
        reused = {**current, "process_start_identity": "reused-start"}
        for path, schema in (
            (
                self.adapter.paths.active_lock,
                foreground.ACTIVATION_LOCK_SCHEMA,
            ),
            (
                self.adapter.paths.active_pid,
                foreground.ACTIVATION_PID_SCHEMA,
            ),
        ):
            write_json(
                path,
                self.active_record(
                    schema, "pid-reuse", reused, reused
                ),
            )
        with self.assertRaises(contract.AdapterError) as caught:
            self.adapter.status()
        self.assertEqual(
            caught.exception.reason_code,
            "SUPERVISOR_IDENTITY_UNCERTAIN",
        )
        self.assertTrue(self.adapter.paths.active_lock.exists())
        self.assertTrue(self.adapter.paths.active_pid.exists())


class StaticSecurityTests(unittest.TestCase):
    def test_standard_library_import_direction_and_no_runtime_bypass(self) -> None:
        adapter_dir = BRANCH_ROOT / "service_control/platform_adapters"
        internal_roots = {
            "__future__",
            "contract",
            "foreground",
            "linux_systemd_user",
            "registry",
            "operating_profile",
            "supervisor",
            "service_control",
        }
        for path in adapter_dir.glob("*.py"):
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
            imports = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.update(
                        alias.name.split(".", 1)[0]
                        for alias in node.names
                    )
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.add(node.module.split(".", 1)[0])
            third_party = sorted(
                name
                for name in imports
                if name not in sys.stdlib_module_names
                and name not in internal_roots
            )
            self.assertEqual(third_party, [], path)
            self.assertFalse(
                any(
                    name.startswith(("api_service", "branch_controller"))
                    for name in imports
                ),
                path,
            )

        foreground_source = Path(foreground.__file__).read_text(
            encoding="utf-8"
        )
        foreground_tree = ast.parse(foreground_source)
        popen_calls = [
            node
            for node in ast.walk(foreground_tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "Popen"
        ]
        self.assertEqual(len(popen_calls), 1)
        self.assertNotIn("os.killpg", foreground_source)
        for forbidden in (
            "importlib",
            "entry_points",
            "shell=True",
            "start_new_session=True",
            "uvicorn",
            "llama-server",
        ):
            self.assertNotIn(forbidden, foreground_source)

        supervisor_source = Path(supervisor.__file__).read_text(
            encoding="utf-8"
        )
        supervisor_tree = ast.parse(supervisor_source)
        imports = []
        for node in ast.walk(supervisor_tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        self.assertFalse(
            any("platform_adapter" in name for name in imports)
        )
        functional_strings = [
            node.value
            for node in ast.walk(supervisor_tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
        ]
        joined = "\n".join(functional_strings).lower()
        for forbidden in (
            "systemd",
            "launchd",
            "windows service",
            "containerd",
            "docker",
            "power-house",
            "/home/user",
            "cuda version",
        ):
            self.assertNotIn(forbidden, joined)

    def test_supervisor_controller_identity_interface_is_bounded(self) -> None:
        parser = supervisor.build_argument_parser()
        values = parser.parse_args(["status"])
        self.assertEqual(
            values.api_controller_sha256,
            supervisor.API_CONTROLLER_SHA256,
        )
        self.assertEqual(
            values.branch_controller_sha256,
            supervisor.BRANCH_CONTROLLER_SHA256,
        )
        values.api_controller_sha256 = "not-a-digest"
        with self.assertRaises(supervisor.SupervisorError) as caught:
            supervisor._adapter_from_arguments(values)
        self.assertEqual(
            caught.exception.reason_code,
            "invalid_controller_identity",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
