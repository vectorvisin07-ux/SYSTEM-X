"""Consolidated isolated tests for the supervisor core."""

from __future__ import annotations

import ast
import copy
import inspect
import json
from pathlib import Path
import socket
import sys
import tempfile
import threading
import unittest
from unittest import mock


BRANCH_ROOT = Path(__file__).resolve().parents[2]
if str(BRANCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BRANCH_ROOT))

from service_control import operating_profile as profile_api  # noqa: E402
from service_control import recovery as recovery_api  # noqa: E402
from service_control import supervisor as supervisor_api  # noqa: E402


FIXTURE_CONTROLLER = r'''#!/usr/bin/env python3
import json
from pathlib import Path
import sys
import time

kind = "api" if Path(__file__).name.startswith("api") else "branch"
root = Path(__file__).parent
state_path = root / "controller-state.json"
config_path = root / "controller-config.json"
calls_path = root / "controller-calls.jsonl"
state = json.loads(state_path.read_text(encoding="utf-8"))
config = (
    json.loads(config_path.read_text(encoding="utf-8"))
    if config_path.exists()
    else {}
)
operation = sys.argv[1] if len(sys.argv) > 1 else "missing"
with calls_path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(
        {"kind": kind, "operation": operation, "argv": sys.argv[1:]},
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n")

special = config.get("special", {})
if special == {"kind": kind, "operation": operation, "mode": "timeout"}:
    time.sleep(0.5)
if special == {"kind": kind, "operation": operation, "mode": "malformed"}:
    print("{not-json")
    raise SystemExit(0)
if special == {"kind": kind, "operation": operation, "mode": "nonzero"}:
    print(json.dumps({
        "schema_version": (
            "system-x.gguf-api-service-controller.v1"
            if kind == "api"
            else "system-x.gguf-branch-controller.v1"
        ),
        "operation": operation,
        "ok": False,
        "reason_code": "FIXTURE_FAILURE",
        "message": "bounded fixture failure",
    }, separators=(",", ":")))
    raise SystemExit(7)

def values_from_argv():
    values = {}
    arguments = sys.argv[2:]
    index = 0
    while index < len(arguments):
        name = arguments[index]
        value = arguments[index + 1]
        values[name[2:].replace("-", "_")] = value
        index += 2
    return {
        "host": values["host"],
        "port": int(values["port"]),
        "private_backend_host": values["private_backend_host"],
        "private_backend_port": int(values["private_backend_port"]),
        "private_backend_enabled": (
            values["private_backend_enabled"].lower() == "true"
        ),
        "private_backend_models_max": int(
            values["private_backend_models_max"]
        ),
        "registry_enabled": values["registry_enabled"].lower() == "true",
        "registry_default_alias": values["registry_default_alias"],
        "startup_model_policy": values["startup_model_policy"],
        "automatic_recovery_enabled": (
            values["automatic_recovery_enabled"].lower() == "true"
        ),
        "recovery_maximum_attempts_in_window": int(
            values["recovery_maximum_attempts_in_window"]
        ),
        "recovery_attempt_window_seconds": float(
            values["recovery_attempt_window_seconds"]
        ),
        "recovery_stable_reset_seconds": float(
            values["recovery_stable_reset_seconds"]
        ),
        "service_control_profile_identity": values[
            "service_control_profile_identity"
        ],
    }

if kind == "api":
    schema = "system-x.gguf-api-service-controller.v1"
    if operation == "plan":
        values = values_from_argv()
        result = {
            "schema_version": schema,
            "operation": operation,
            "ok": True,
            "reason_code": "OK",
            "message": "fixture plan",
            "input": values,
            "plan": {"shell": False, "start_new_session": True},
        }
    elif operation == "start":
        values = values_from_argv()
        state.update({
            "active": True,
            "api_start_count": state.get("api_start_count", 0) + 1,
            "api_status_count": 0,
            "public_endpoint": {
                "host": values["host"],
                "port": values["port"],
            },
            "private_endpoint": {
                "host": values["private_backend_host"],
                "port": values["private_backend_port"],
            },
        })
        state_path.write_text(
            json.dumps(state, sort_keys=True), encoding="utf-8"
        )
        result = {
            "schema_version": schema,
            "operation": operation,
            "ok": True,
            "reason_code": "OK",
            "message": "fixture start",
            "runtime": {
                "active": True,
                "consistent": True,
                "lifecycle_state": "STARTED",
                "transaction_id": "fixture-api-tx-1",
                "pid": 41001,
                "pgid": 41001,
                "sid": 41001,
                "process_start_identity": "fixture-api-start-1",
            },
            "listener": {"ownership_matches": True},
        }
    elif operation == "status":
        state["api_status_count"] = state.get("api_status_count", 0) + 1
        state_path.write_text(
            json.dumps(state, sort_keys=True), encoding="utf-8"
        )
        active = bool(state.get("active"))
        changed = (
            config.get("identity_change") is True
            and state["api_status_count"] >= 2
        )
        result = {
            "schema_version": schema,
            "operation": operation,
            "ok": True,
            "reason_code": "OK",
            "message": "fixture status",
            "runtime": {
                "active": active,
                "consistent": True,
                "lifecycle_state": "STARTED" if active else "STOPPED",
                "transaction_id": "fixture-api-tx-1" if active else None,
                "pid": (41999 if changed else 41001) if active else None,
                "pgid": 41001 if active else None,
                "sid": 41001 if active else None,
                "process_start_identity": (
                    "fixture-api-start-1" if active else None
                ),
            },
            "listener": (
                {"ownership_matches": True} if active else None
            ),
        }
    elif operation == "reconcile":
        active = bool(state.get("active"))
        result = {
            "schema_version": schema,
            "operation": operation,
            "ok": True,
            "reason_code": "OK",
            "message": "fixture reconcile",
            "runtime": {
                "active": active,
                "consistent": True,
                "reconciled": not active,
                "lifecycle_state": "STARTED" if active else "RECONCILED",
                "transaction_id": "fixture-api-tx-1" if active else None,
                "pid": 41001 if active else None,
                "pgid": 41001 if active else None,
                "sid": 41001 if active else None,
                "process_start_identity": (
                    "fixture-api-start-1" if active else None
                ),
            },
            "reconciliation": {
                "active_pid_record_removed": not active,
                "active_lock_removed": not active,
                "unrelated_process_signaled": False,
            },
        }
    elif operation == "stop":
        state["active"] = False
        state["api_stop_count"] = state.get("api_stop_count", 0) + 1
        state_path.write_text(
            json.dumps(state, sort_keys=True), encoding="utf-8"
        )
        result = {
            "schema_version": schema,
            "operation": operation,
            "ok": True,
            "reason_code": "OK",
            "message": "fixture stop",
            "runtime": {
                "active": False,
                "consistent": True,
                "lifecycle_state": "STOPPED",
                "transaction_id": "fixture-api-tx-1",
                "pid": 41001,
                "pgid": 41001,
                "sid": 41001,
                "process_start_identity": "fixture-api-start-1",
            },
            "cleanup": {
                "signal": "SIGTERM",
                "force_used": False,
                "owned_process_group_gone": True,
            },
        }
    else:
        raise SystemExit(9)
else:
    schema = "system-x.gguf-branch-controller.v1"
    if operation in {"status", "reconcile"}:
        active = bool(state.get("active"))
        data = {
            "active": active,
            "active_state_consistent": True,
            "lifecycle_state": (
                "STARTED"
                if active
                else ("RECONCILED" if operation == "reconcile" else "STOPPED")
            ),
            "transaction_id": "fixture-router-tx-1" if active else None,
            "pid": 42001 if active else None,
            "pgid": 42001 if active else None,
            "sid": 42001 if active else None,
            "process_start_identity": (
                "fixture-router-start-1" if active else None
            ),
            "process_alive": active,
            "endpoint": state.get("private_endpoint") if active else None,
        }
        if active and config.get("model_child_present") is True:
            data["model_child"] = {
                "present": True,
                "identity": {
                    "pid": 43001,
                    "process_start_identity": "fixture-model-start-1",
                    "ppid": 42001,
                    "pgid": 42001,
                    "sid": 42001,
                },
            }
        result = {
            "schema_version": schema,
            "operation": operation,
            "ok": True,
            "reason_code": "OK",
            "message": f"fixture branch {operation}",
            "data": data,
        }
    elif operation == "stop":
        state["active"] = False
        state_path.write_text(
            json.dumps(state, sort_keys=True), encoding="utf-8"
        )
        result = {
            "schema_version": schema,
            "operation": operation,
            "ok": True,
            "reason_code": "OK",
            "message": "fixture branch stop",
            "data": {
                "active": False,
                "active_state_consistent": True,
                "owned_group_absent": True,
                "active_pid_record_removed": True,
                "active_lock_removed": True,
            },
        }
    else:
        raise SystemExit(9)
print(json.dumps(result, sort_keys=True, separators=(",", ":")))
'''


def unused_ports() -> tuple[int, int]:
    sockets = [socket.socket(socket.AF_INET, socket.SOCK_STREAM) for _ in range(2)]
    try:
        for item in sockets:
            item.bind(("127.0.0.1", 0))
        return sockets[0].getsockname()[1], sockets[1].getsockname()[1]
    finally:
        for item in sockets:
            item.close()


def profile_value(public_port: int, private_port: int) -> dict[str, object]:
    return {
        "schema_version": profile_api.OPERATING_PROFILE_SCHEMA,
        "public_endpoint": {"host": "127.0.0.1", "port": public_port},
        "private_router_endpoint": {
            "host": "127.0.0.1",
            "port": private_port,
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


class SupervisorFixtureCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        public_port, private_port = unused_ports()
        self.profile_path = self.root / "operating-profile.json"
        self.state_path = self.root / "desired-state.json"
        self.runtime_root = self.root / "runtime"
        self.profile_path.write_text(
            json.dumps(profile_value(public_port, private_port)),
            encoding="utf-8",
        )
        self.profile = profile_api.load_operating_profile(self.profile_path)
        self.api_controller = self.root / "api-fixture.py"
        self.branch_controller = self.root / "branch-fixture.py"
        self.api_controller.write_text(FIXTURE_CONTROLLER, encoding="utf-8")
        self.branch_controller.write_text(FIXTURE_CONTROLLER, encoding="utf-8")
        self.controller_state = self.root / "controller-state.json"
        self.controller_config = self.root / "controller-config.json"
        self.controller_calls = self.root / "controller-calls.jsonl"
        self.write_controller_state()
        self.write_controller_config()
        self.adapter = supervisor_api.ControllerAdapter(
            self.api_controller,
            self.branch_controller,
            timeout_seconds=2,
            api_sha256=None,
            branch_sha256=None,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_controller_state(self, **updates: object) -> None:
        value = {
            "active": False,
            "api_start_count": 0,
            "api_stop_count": 0,
            "api_status_count": 0,
        }
        if self.controller_state.exists():
            value.update(
                json.loads(self.controller_state.read_text(encoding="utf-8"))
            )
        value.update(updates)
        self.controller_state.write_text(
            json.dumps(value, sort_keys=True), encoding="utf-8"
        )

    def write_controller_config(self, **values: object) -> None:
        self.controller_config.write_text(
            json.dumps(values, sort_keys=True), encoding="utf-8"
        )

    def initialize(self, desired_state: str) -> profile_api.DesiredState:
        return profile_api.initialize_desired_state(
            self.profile, self.state_path, desired_state
        )

    def supervisor(self) -> supervisor_api.ForegroundSupervisor:
        return supervisor_api.ForegroundSupervisor(
            self.profile_path,
            self.state_path,
            self.runtime_root,
            self.adapter,
            monitor_interval_seconds=0.01,
            install_signal_handlers=False,
            health_observer=self.health_observer,
        )

    def health_observer(
        self, _profile: profile_api.OperatingProfile
    ) -> dict[str, object]:
        state = json.loads(
            self.controller_state.read_text(encoding="utf-8")
        )
        config = json.loads(
            self.controller_config.read_text(encoding="utf-8")
        )
        readiness = str(config.get("service_readiness_state", "READY"))
        ready = readiness == "READY"
        service_available = readiness in {
            "WAITING_FOR_MODEL",
            "MODEL_CANDIDATE_LOADING",
            "READY",
        }
        warm = (
            {
                "requested_alias": "default",
                "resolved_public_model_id": "sx-fixture-ready",
                "artifact_version_id": "bundle-fixture",
                "registry_generation": 7,
                "capability_manifest_identity": "a" * 64,
                "router_transaction_id": "fixture-router-tx-1",
                "model_child_pid": 43001,
                "model_child_start_identity": "fixture-model-start-1",
                "model_child_parent": 42001,
                "model_child_process_group": 42001,
                "model_child_session": 42001,
                "warm_since_utc": "2026-01-02T03:04:05.000006Z",
                "last_verified_utc": "2026-01-02T03:04:05.000007Z",
                "health_state": "ready",
            }
            if state.get("active") and ready
            else None
        )
        return {
            "http_status": 200 if service_available else 503,
            "body": {
                "service_readiness_state": readiness,
                "model_service_state": readiness,
                "service_available": service_available,
                "inference_ready": ready,
                "ready": ready,
                "reason_code": None if ready else "fixture_loading",
                "warm_identity": warm,
            },
            "observed_utc": "2026-01-02T03:04:05.000008Z",
        }

    def wait_for_status(self, expected: str, timeout: float = 5) -> dict:
        deadline = supervisor_api.time.monotonic() + timeout
        waiter = threading.Event()
        path = self.runtime_root / "status/supervisor.json"
        while supervisor_api.time.monotonic() < deadline:
            if path.exists():
                value = json.loads(path.read_text(encoding="utf-8"))
                if value.get("supervisor_state") == expected:
                    return value
            waiter.wait(0.01)
        self.fail(f"supervisor status did not reach {expected}")

    def run_in_thread(
        self, supervisor: supervisor_api.ForegroundSupervisor
    ) -> tuple[threading.Thread, list[object]]:
        outcome: list[object] = []

        def target() -> None:
            try:
                outcome.append(supervisor.run())
            except BaseException as exc:
                outcome.append(exc)

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        return thread, outcome

    def calls(self) -> list[dict]:
        if not self.controller_calls.exists():
            return []
        return [
            json.loads(line)
            for line in self.controller_calls.read_text(
                encoding="utf-8"
            ).splitlines()
        ]


class ProfileStateIntegrationTests(SupervisorFixtureCase):
    def test_running_stopped_profile_binding_and_stale_stop(self) -> None:
        running = self.initialize("RUNNING")
        self.assertEqual(running.generation, 1)
        self.assertEqual(
            profile_api.load_desired_state(
                self.state_path, self.profile.identity
            ).desired_state,
            "RUNNING",
        )

        changed_value = self.profile.as_dict()
        changed_value["default_model_alias"] = "changed"
        changed = profile_api.validate_operating_profile(changed_value)
        with self.assertRaises(profile_api.ServiceControlError):
            profile_api.load_desired_state(
                self.state_path, changed.identity
            )

        before = self.state_path.read_bytes()
        with self.assertRaises(supervisor_api.SupervisorError) as caught:
            supervisor_api.administrative_stop(
                self.profile_path,
                self.state_path,
                self.runtime_root,
                expected_generation=2,
                wait_timeout_seconds=0.1,
            )
        self.assertEqual(
            caught.exception.reason_code, "stale_expected_generation"
        )
        self.assertEqual(self.state_path.read_bytes(), before)

    def test_already_stopped_is_idempotent(self) -> None:
        self.initialize("STOPPED")
        result = supervisor_api.administrative_stop(
            self.profile_path,
            self.state_path,
            self.runtime_root,
            expected_generation=1,
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["reason_code"], "already_stopped")
        self.assertEqual(result["desired_state_generation"], 1)


class ControllerAdapterTests(SupervisorFixtureCase):
    def test_structured_controller_matrix_and_model_child_shapes(self) -> None:
        self.initialize("RUNNING")
        dependencies = self.adapter.validate()
        self.assertIn("sha256", dependencies["api"])
        plan = self.adapter.invoke(
            "api",
            "plan",
            supervisor_api._api_arguments(
                "plan", self.profile, self.state_path
            ),
        )
        supervisor_api._verify_api_plan(plan, self.profile)
        self.assertTrue(plan["input"]["registry_enabled"])
        self.assertEqual(
            plan["input"]["startup_model_policy"], "always_warm"
        )
        start = self.adapter.invoke(
            "api",
            "start",
            supervisor_api._api_arguments(
                "start", self.profile, self.state_path
            ),
        )
        self.assertTrue(start["runtime"]["active"])
        branch = self.adapter.invoke("branch", "status")
        self.assertEqual(
            supervisor_api._observed_model_child(branch),
            {"present": False, "identity": None},
        )

        self.write_controller_config(model_child_present=True)
        branch = self.adapter.invoke("branch", "status")
        model = supervisor_api._observed_model_child(branch)
        self.assertTrue(model["present"])
        self.assertEqual(model["identity"]["pid"], 43001)
        self.adapter.invoke("api", "status")
        self.adapter.invoke("api", "stop")
        operations = [(call["kind"], call["operation"]) for call in self.calls()]
        self.assertIn(("api", "plan"), operations)
        self.assertIn(("api", "start"), operations)
        self.assertIn(("api", "status"), operations)
        self.assertIn(("api", "stop"), operations)
        self.assertIn(("branch", "status"), operations)
        self.assertTrue(
            all(isinstance(call["argv"], list) for call in self.calls())
        )

    def test_malformed_nonzero_and_timeout_are_bounded(self) -> None:
        for mode, reason in (
            ("malformed", "controller_malformed_output"),
            ("nonzero", "controller_operation_failed"),
        ):
            with self.subTest(mode=mode):
                self.write_controller_config(
                    special={"kind": "api", "operation": "status", "mode": mode}
                )
                with self.assertRaises(supervisor_api.SupervisorError) as caught:
                    self.adapter.invoke("api", "status")
                self.assertEqual(caught.exception.reason_code, reason)
                if mode == "nonzero":
                    self.assertEqual(
                        caught.exception.details["controller_result"][
                            "reason_code"
                        ],
                        "FIXTURE_FAILURE",
                    )

        self.write_controller_config(
            special={"kind": "api", "operation": "status", "mode": "timeout"}
        )
        bounded = supervisor_api.ControllerAdapter(
            self.api_controller,
            self.branch_controller,
            timeout_seconds=0.05,
            api_sha256=None,
            branch_sha256=None,
        )
        with self.assertRaises(supervisor_api.SupervisorError) as caught:
            bounded.invoke("api", "status")
        self.assertEqual(caught.exception.reason_code, "controller_timeout")


class ForegroundBehaviorTests(SupervisorFixtureCase):
    def test_router_observation_failure_has_router_reason(self) -> None:
        supervisor = object.__new__(supervisor_api.ForegroundSupervisor)
        expected_api = {
            "transaction_id": "api-transaction",
            "pid": 1001,
            "process_start_identity": "api-start",
            "pgid": 1001,
            "endpoint": "public",
            "active": True,
            "consistent": True,
            "listener_owned": True,
        }
        expected_router = {
            "transaction_id": "router-transaction",
            "pid": 2001,
            "process_start_identity": "router-start",
            "pgid": 2001,
            "endpoint": "private",
        }
        api = dict(expected_api)
        for mutation, reason in (
            (
                {
                    "active": False,
                    "consistent": False,
                    "listener_owned": False,
                },
                "ROUTER_PROCESS_LOST",
            ),
            (
                {
                    "active": True,
                    "consistent": True,
                    "listener_owned": False,
                },
                "PRIVATE_LISTENER_LOST",
            ),
        ):
            with self.subTest(reason=reason):
                router = dict(expected_router)
                router.update(mutation)
                with self.assertRaises(
                    supervisor_api.SupervisorError
                ) as caught:
                    supervisor._verify_observation(
                        expected_api, expected_router, api, router
                    )
                self.assertEqual(caught.exception.reason_code, reason)

    def test_router_recovery_preserves_api_identity(self) -> None:
        recovery_source = inspect.getsource(
            supervisor_api.ForegroundSupervisor._recover_router_only
        )
        dispatch_source = inspect.getsource(
            supervisor_api.ForegroundSupervisor.run
        )
        self.assertIn("API_OWNED_ROUTER_RESTART", recovery_source)
        self.assertNotIn('self.adapter.invoke("api", "stop")', recovery_source)
        self.assertNotIn('self.adapter.invoke("api", "start"', recovery_source)
        self.assertIn("self._recover_router_only", dispatch_source)

    def test_api_recovery_waits_for_both_stack_endpoints(self) -> None:
        recovery_source = inspect.getsource(
            supervisor_api.ForegroundSupervisor._recover_api_stack
        )
        self.assertLess(
            recovery_source.index('self.adapter.invoke("branch", "stop")'),
            recovery_source.index(
                "self._wait_for_reusable_stack_endpoints"
            ),
        )
        desired = self.initialize("RUNNING")
        supervisor = self.supervisor()
        recovery = recovery_api.RecoveryStore(
            self.runtime_root / "recovery",
            self.profile.identity,
            supervisor_api._recovery_policy(self.profile),
        )
        attempt = recovery.begin(
            reason_code="API_PROCESS_LOST",
            desired_state=desired.desired_state,
            desired_generation=desired.generation,
            observation={"fixture": "endpoint-reuse"},
            selected_action="CONTROLLER_OWNED_API_STACK_RESTART",
        )
        self.assertIsNotNone(attempt)
        assert attempt is not None
        with mock.patch.object(
            supervisor_api,
            "_endpoint_bindable",
            side_effect=(False, False, True, True),
        ) as bindable:
            current = supervisor._wait_for_reusable_stack_endpoints(
                profile=self.profile,
                recovery=recovery,
                attempt=attempt,
                observation={"fixture": "endpoint-reuse"},
            )
        self.assertIsNotNone(current)
        self.assertEqual(bindable.call_count, 4)
        recovery.complete(
            attempt,
            desired_state="RUNNING",
            desired_generation=1,
            outcome="RECOVERED",
            observation={"fixture": "endpoint-reuse-complete"},
        )

        stopped = profile_api.set_desired_state(
            self.profile,
            "STOPPED",
            self.state_path,
            expected_generation=1,
        )
        cancelled = recovery.begin(
            reason_code="API_PROCESS_LOST",
            desired_state="RUNNING",
            desired_generation=1,
            observation={"fixture": "stopped-precedence"},
            selected_action="CONTROLLER_OWNED_API_STACK_RESTART",
        )
        self.assertIsNotNone(cancelled)
        assert cancelled is not None
        with mock.patch.object(
            supervisor_api, "_endpoint_bindable", return_value=False
        ) as bindable:
            result = supervisor._wait_for_reusable_stack_endpoints(
                profile=self.profile,
                recovery=recovery,
                attempt=cancelled,
                observation={"fixture": "stopped-precedence"},
            )
        self.assertIsNone(result)
        bindable.assert_not_called()
        transaction = json.loads(
            recovery.paths.transaction(
                cancelled.recovery_transaction_id
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(transaction["outcome"], "CANCELLED_BY_STOPPED")
        self.assertEqual(stopped.desired_state, "STOPPED")

    def test_restricted_router_control_startup_policy(self) -> None:
        self.initialize("RUNNING")
        arguments = supervisor_api.build_argument_parser().parse_args(
            ["plan", "--startup-model-policy", "router_control"]
        )
        self.assertEqual(arguments.startup_model_policy, "router_control")
        values = supervisor_api._api_arguments(
            "plan",
            self.profile,
            self.state_path,
            startup_model_policy=arguments.startup_model_policy,
        )
        value_map = dict(zip(values[::2], values[1::2]))
        self.assertEqual(value_map["--startup-model-policy"], "router_control")
        self.assertEqual(value_map["--registry-enabled"], "false")
        self.assertEqual(value_map["--automatic-recovery-enabled"], "false")
        plan = self.adapter.invoke("api", "plan", values)
        supervisor_api._verify_api_plan(
            plan,
            self.profile,
            startup_model_policy=arguments.startup_model_policy,
        )
        self.assertFalse(plan["input"]["registry_enabled"])
        self.assertFalse(plan["input"]["automatic_recovery_enabled"])

    def test_stopped_at_start_creates_no_api_start(self) -> None:
        self.initialize("STOPPED")
        result = self.supervisor().run()
        self.assertTrue(result["ok"])
        self.assertEqual(
            result["data"]["status"]["supervisor_state"], "STOPPED"
        )
        state = json.loads(self.controller_state.read_text(encoding="utf-8"))
        self.assertEqual(state["api_start_count"], 0)
        self.assertEqual(state["api_stop_count"], 0)
        self.assertFalse(
            (self.runtime_root / "pids/supervisor.json").exists()
        )
        self.assertFalse(
            (self.runtime_root / "locks/supervisor.lock").exists()
        )

    def test_running_status_duplicate_stale_identity_and_admin_stop(self) -> None:
        self.initialize("RUNNING")
        first = self.supervisor()
        thread, outcome = self.run_in_thread(first)
        running = self.wait_for_status("RUNNING")
        self.assertEqual(running["desired_state_generation"], 1)
        self.assertEqual(running["service_readiness_state"], "READY")

        status = supervisor_api.administrative_status(
            self.profile_path,
            self.state_path,
            self.runtime_root,
            self.adapter,
        )
        self.assertTrue(status["data"]["active"])
        self.assertEqual(
            status["data"]["status"]["supervisor_state"], "RUNNING"
        )

        with self.assertRaises(supervisor_api.SupervisorError) as caught:
            self.supervisor().run()
        self.assertEqual(caught.exception.reason_code, "supervisor_lock_active")
        self.assertTrue(thread.is_alive())

        pid_path = self.runtime_root / "pids/supervisor.json"
        original = json.loads(pid_path.read_text(encoding="utf-8"))
        stale = copy.deepcopy(original)
        stale["process_start_identity"] = "synthetic-stale-start"
        pid_path.write_text(json.dumps(stale), encoding="utf-8")
        stale_bytes = pid_path.read_bytes()
        with self.assertRaises(supervisor_api.SupervisorError) as caught:
            supervisor_api.administrative_status(
                self.profile_path,
                self.state_path,
                self.runtime_root,
                self.adapter,
            )
        self.assertEqual(
            caught.exception.reason_code, "supervisor_identity_inconsistent"
        )
        self.assertEqual(pid_path.read_bytes(), stale_bytes)
        supervisor_api._atomic_write_json(pid_path, original)

        stopped = supervisor_api.administrative_stop(
            self.profile_path,
            self.state_path,
            self.runtime_root,
            expected_generation=1,
            wait_timeout_seconds=5,
        )
        self.assertTrue(stopped["ok"])
        self.assertEqual(stopped["desired_state"], "STOPPED")
        self.assertEqual(stopped["desired_state_generation"], 2)
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(outcome), 1)
        self.assertIsInstance(outcome[0], dict)
        state = json.loads(self.controller_state.read_text(encoding="utf-8"))
        self.assertEqual(state["api_start_count"], 1)
        self.assertEqual(state["api_stop_count"], 1)

    def test_external_graceful_shutdown_preserves_desired_state(self) -> None:
        self.initialize("RUNNING")
        supervisor = self.supervisor()
        thread, outcome = self.run_in_thread(supervisor)
        self.wait_for_status("RUNNING")
        supervisor.request_graceful_shutdown()
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertIsInstance(outcome[0], dict)
        desired = profile_api.load_desired_state(
            self.state_path, self.profile.identity
        )
        self.assertEqual(desired.desired_state, "RUNNING")
        self.assertEqual(desired.generation, 1)
        final = self.wait_for_status("STOPPED")
        self.assertEqual(
            final["stop_reason"], "external_graceful_shutdown"
        )
        state = json.loads(self.controller_state.read_text(encoding="utf-8"))
        self.assertEqual(state["api_stop_count"], 1)

    def test_identity_change_faults_without_restart(self) -> None:
        self.initialize("RUNNING")
        self.write_controller_config(identity_change=True)
        supervisor = self.supervisor()
        thread, outcome = self.run_in_thread(supervisor)
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(outcome), 1)
        self.assertIsInstance(outcome[0], supervisor_api.SupervisorError)
        self.assertEqual(
            outcome[0].reason_code, "OWNERSHIP_UNCERTAIN"
        )
        state = json.loads(self.controller_state.read_text(encoding="utf-8"))
        self.assertEqual(state["api_start_count"], 1)
        self.assertEqual(state["api_stop_count"], 1)
        final = self.wait_for_status("FAULTED")
        self.assertEqual(
            final["fault_reason"], "OWNERSHIP_UNCERTAIN"
        )
        self.assertFalse(
            (self.runtime_root / "pids/supervisor.json").exists()
        )
        self.assertFalse(
            (self.runtime_root / "locks/supervisor.lock").exists()
        )

    def test_partial_start_fault_reconciles_branch_before_api(self) -> None:
        self.initialize("RUNNING")
        self.write_controller_config(
            special={
                "kind": "branch",
                "operation": "status",
                "mode": "nonzero",
            }
        )
        with self.assertRaises(supervisor_api.SupervisorError) as caught:
            self.supervisor().run()
        self.assertEqual(
            caught.exception.reason_code, "controller_operation_failed"
        )

        operations = [
            (call["kind"], call["operation"]) for call in self.calls()
        ]
        self.assertIn(("branch", "reconcile"), operations)
        self.assertIn(("branch", "stop"), operations)
        self.assertIn(("api", "reconcile"), operations)
        self.assertLess(
            operations.index(("branch", "reconcile")),
            operations.index(("api", "reconcile")),
        )
        state = json.loads(
            self.controller_state.read_text(encoding="utf-8")
        )
        self.assertFalse(state["active"])
        final = self.wait_for_status("FAULTED")
        self.assertEqual(
            final["fault_reason"], "controller_operation_failed"
        )
        self.assertFalse(
            (self.runtime_root / "pids/supervisor.json").exists()
        )
        self.assertFalse(
            (self.runtime_root / "locks/supervisor.lock").exists()
        )

    def test_candidate_loading_is_stable_operational_without_restart(
        self,
    ) -> None:
        self.initialize("RUNNING")
        self.write_controller_config(
            service_readiness_state="MODEL_CANDIDATE_LOADING"
        )
        supervisor = self.supervisor()
        thread, outcome = self.run_in_thread(supervisor)
        running = self.wait_for_status("RUNNING")
        self.assertEqual(
            running["service_readiness_state"],
            "MODEL_CANDIDATE_LOADING",
        )
        self.assertEqual(
            running["model_service_state"],
            "MODEL_CANDIDATE_LOADING",
        )
        self.assertTrue(running["service_operational"])
        self.assertFalse(running["inference_ready"])
        self.assertEqual(
            running["reason_code"], "MODEL_CANDIDATE_LOADING"
        )
        state = json.loads(
            self.controller_state.read_text(encoding="utf-8")
        )
        self.assertEqual(state["api_start_count"], 1)
        self.assertEqual(state["api_stop_count"], 0)
        supervisor.request_graceful_shutdown()
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertIsInstance(outcome[0], dict)

    def test_waiting_for_model_is_stable_and_recovery_idle(self) -> None:
        self.initialize("RUNNING")
        self.write_controller_config(
            service_readiness_state="WAITING_FOR_MODEL"
        )
        supervisor = self.supervisor()
        thread, outcome = self.run_in_thread(supervisor)
        running = self.wait_for_status("RUNNING")
        self.assertEqual(
            running["model_service_state"], "WAITING_FOR_MODEL"
        )
        self.assertTrue(running["service_operational"])
        self.assertFalse(running["inference_ready"])
        self.assertEqual(running["reason_code"], "NO_READY_MODEL")
        self.assertEqual(
            running["recovery_status"]["recovery_state"], "IDLE"
        )
        self.assertEqual(
            running["recovery_status"]["current_attempt"], 0
        )
        state = json.loads(
            self.controller_state.read_text(encoding="utf-8")
        )
        self.assertEqual(state["api_start_count"], 1)
        self.assertEqual(state["api_stop_count"], 0)
        supervisor.request_graceful_shutdown()
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertIsInstance(outcome[0], dict)


class StaticContractTests(unittest.TestCase):
    def test_source_is_standard_library_and_contains_no_bypass(self) -> None:
        path = Path(supervisor_api.__file__).resolve()
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".", 1)[0])
        internal = {
            "service_control",
            "operating_profile",
            "recovery",
            "__future__",
        }
        third_party = sorted(
            name
            for name in imports
            if name not in sys.stdlib_module_names and name not in internal
        )
        self.assertEqual(third_party, [])
        lowered = source.lower()
        for forbidden in (
            "/home/user",
            "systemd",
            "launchctl",
            "service control manager",
            "power-house",
            "shell=true",
            "nvidia",
            "cuda",
        ):
            if forbidden in lowered:
                self.fail(f"forbidden runtime token present: {forbidden}")
        self.assertIn("shell=False", source)
        self.assertNotIn("os.killpg", source)

    def test_schemas_parse_and_match_source(self) -> None:
        source_dir = Path(supervisor_api.__file__).resolve().parent
        status = json.loads(
            (source_dir / "supervisor-status.schema.json").read_text(
                encoding="utf-8"
            )
        )
        transaction = json.loads(
            (source_dir / "supervisor-transaction.schema.json").read_text(
                encoding="utf-8"
            )
        )
        recovery_status = json.loads(
            (source_dir / "recovery-status.schema.json").read_text(
                encoding="utf-8"
            )
        )
        recovery_transaction = json.loads(
            (source_dir / "recovery-transaction.schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(status["$id"], supervisor_api.STATUS_SCHEMA)
        self.assertEqual(transaction["$id"], supervisor_api.TRANSACTION_SCHEMA)
        self.assertEqual(
            recovery_status["$id"],
            recovery_api.STATUS_SCHEMA,
        )
        self.assertEqual(
            recovery_transaction["$id"],
            recovery_api.TRANSACTION_SCHEMA,
        )
        self.assertFalse(status["additionalProperties"])
        self.assertFalse(transaction["additionalProperties"])
        self.assertFalse(recovery_status["additionalProperties"])
        self.assertFalse(recovery_transaction["additionalProperties"])

    def test_process_identity_has_pid_reuse_defense(self) -> None:
        identity = supervisor_api.process_snapshot(supervisor_api.os.getpid())
        self.assertEqual(identity["pid"], supervisor_api.os.getpid())
        self.assertTrue(identity["process_start_identity"])
        self.assertRegex(identity["argv_sha256"], r"\A[0-9a-f]{64}\Z")


if __name__ == "__main__":
    unittest.main(verbosity=2)
