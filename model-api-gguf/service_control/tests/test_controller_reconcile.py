"""Isolated controller reconciliation and listener-ownership fixtures."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import socket
import tempfile
import unittest
from unittest import mock


BRANCH_ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


api = load_module(
    "test_api_controller",
    BRANCH_ROOT / "api_service_controller/controller.py",
)
branch = load_module(
    "test_branch_controller",
    BRANCH_ROOT / "branch_controller/controller.py",
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True) + "\n", encoding="utf-8"
    )


def api_paths(root: Path) -> dict[str, Path]:
    values = {
        "branch_root": root,
        "api_service_root": root / "api-service",
        "runtime_api_root": root / "runtime-api",
        "active_lock": root / "locks/active.lock",
        "active_pid": root / "pids/active.json",
        "service_status": root / "status/service.json",
        "transaction_root": root / "transactions",
        "log_root": root / "logs",
        "database_root": root / "database",
        "registry_database": root / "database/registry.sqlite3",
        "auth_root": root / "auth",
        "credential_database": root / "auth/credentials.sqlite3",
        "credential_handoff_root": root / "auth/handoff",
    }
    for name in (
        "transaction_root",
        "log_root",
        "database_root",
        "auth_root",
        "credential_handoff_root",
    ):
        values[name].mkdir(mode=0o700, parents=True, exist_ok=True)
    return values


def api_records(paths: dict[str, Path], *, pid: int = 990001) -> str:
    transaction_id = "tx-api-fixture"
    common = {
        "schema_version": api.SCHEMA_VERSION,
        "transaction_id": transaction_id,
        "pid": pid,
        "process_start_identity": "fixture-start",
        "pgid": pid,
        "sid": pid,
        "executable": "/fixture/python",
        "executable_device": 1,
        "executable_inode": 2,
        "argv": ["/fixture/python", "-m", "fixture"],
        "host": "127.0.0.1",
        "port": 49801,
        "listener": None,
    }
    write_json(
        paths["active_lock"], {**common, "record_type": "active_lock"}
    )
    write_json(
        paths["active_pid"], {**common, "record_type": "active_pid"}
    )
    write_json(
        paths["transaction_root"] / f"{transaction_id}.json",
        {**common, "record_type": "transaction"},
    )
    write_json(
        paths["service_status"],
        {**common, "record_type": "service_status"},
    )
    return transaction_id


def branch_paths(root: Path) -> dict[str, Path]:
    for name in ("locks", "pids", "status", "transactions", "logs"):
        (root / name).mkdir(mode=0o700, parents=True)
    return {"runtime_root": root, "branch_root": root.parent}


def branch_records(paths: dict[str, Path]) -> dict[str, Path]:
    state = branch.runtime_paths(paths)
    transaction_id = "tx-branch-fixture"
    common = {
        "schema_version": branch.SCHEMA_VERSION,
        "transaction_id": transaction_id,
        "pid": 770001,
        "pgid": 770001,
        "sid": 770001,
        "process_start_identity": "fixture-start",
        "executable_path": "/fixture/llama-server",
        "argv": ["/fixture/llama-server", "--port", "49802"],
        "endpoint_host": "127.0.0.1",
        "endpoint_port": 49802,
        "transaction_path": str(
            state["transaction_parent"] / f"{transaction_id}.json"
        ),
    }
    write_json(state["lock_path"], common)
    write_json(state["pid_path"], common)
    write_json(
        state["transaction_parent"] / f"{transaction_id}.json",
        {
            "schema_version": branch.SCHEMA_VERSION,
            "transaction_id": transaction_id,
            "operation_history": [],
        },
    )
    return state


class ApiControllerReconcileTests(unittest.TestCase):
    def test_process_identity_accepts_pinned_venv_symlink_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            executable = Path(temporary) / "python"
            executable.symlink_to("/usr/bin/python3.14")
            argv = [str(executable), "-B", "-m", "uvicorn"]
            observed = {"executable": "/usr/bin/python3.14", "argv": argv}
            plan = {"executable": str(executable), "argv": argv}
            with mock.patch.object(api, "process_identity", return_value=observed):
                result = api.wait_for_expected_identity(123, plan, timeout=0.1)
            self.assertEqual(result, observed)
    def test_pinned_venv_interpreter_symlink_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            branch_root = Path(temporary)
            controller_dir = branch_root / "api_service_controller"
            controller_dir.mkdir()
            controller_file = controller_dir / "controller.py"
            controller_file.write_text("# fixture\n", encoding="utf-8")
            api_service = branch_root / "api_service"
            (api_service / ".venv/bin").mkdir(parents=True)
            (api_service / "src").mkdir()
            (api_service / ".venv/bin/python").symlink_to("/usr/bin/python3.14")
            paths = api.derive_paths(controller_file)
            with mock.patch.object(api, "DEPENDENCY_FILES", {}):
                result = api.validate_dependency(paths)
            self.assertEqual(result["venv_python"], str(api_service / ".venv/bin/python"))
    def test_stale_records_are_removed_but_history_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = api_paths(Path(temporary))
            transaction_id = api_records(paths)
            with (
                mock.patch.object(api, "validate_dependency", return_value={}),
                mock.patch.object(
                    api, "validate_runtime_layout", return_value=None
                ),
                mock.patch.object(
                    api, "identity_matches", return_value=(False, None)
                ),
                mock.patch.object(
                    api, "matching_recorded_processes", return_value=[]
                ),
                mock.patch.object(
                    api, "endpoint_listener_owners", return_value=[]
                ),
            ):
                result = api.operation_reconcile(paths)
            self.assertEqual(result["reason_code"], "API_STATE_STALE")
            self.assertFalse(paths["active_pid"].exists())
            self.assertFalse(paths["active_lock"].exists())
            self.assertTrue(
                (
                    paths["transaction_root"]
                    / f"{transaction_id}.json"
                ).is_file()
            )
            self.assertFalse(
                result["reconciliation"]["unrelated_process_signaled"]
            )

    def test_plan_reconciles_stale_records_before_new_activation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = api_paths(Path(temporary))
            api_records(paths)
            values = {
                "host": "127.0.0.1",
                "port": 49801,
                "private_backend_enabled": False,
            }
            with (
                mock.patch.object(api, "validate_dependency", return_value={}),
                mock.patch.object(
                    api, "validate_runtime_layout", return_value=None
                ),
                mock.patch.object(
                    api, "identity_matches", return_value=(False, None)
                ),
                mock.patch.object(
                    api, "matching_recorded_processes", return_value=[]
                ),
                mock.patch.object(
                    api, "endpoint_listener_owners", return_value=[]
                ),
                mock.patch.object(
                    api, "validated_input", return_value=values
                ),
                mock.patch.object(api, "build_plan", return_value={}),
                mock.patch.object(
                    api, "endpoint_available", return_value=True
                ),
            ):
                result = api.operation_plan(paths, mock.Mock())
            self.assertTrue(result["ok"])
            self.assertEqual(result["operation"], "plan")
            self.assertFalse(paths["active_pid"].exists())
            self.assertFalse(paths["active_lock"].exists())

    def test_stop_reconciles_records_after_owned_process_exits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = api_paths(Path(temporary))
            transaction_id = api_records(paths)
            with (
                mock.patch.object(api, "validate_dependency", return_value={}),
                mock.patch.object(
                    api, "validate_runtime_layout", return_value=None
                ),
                mock.patch.object(
                    api, "identity_matches", return_value=(False, None)
                ),
                mock.patch.object(
                    api, "matching_recorded_processes", return_value=[]
                ),
                mock.patch.object(
                    api, "endpoint_listener_owners", return_value=[]
                ),
                mock.patch.object(api.os, "killpg") as killpg,
            ):
                result = api.operation_stop(paths)
            self.assertTrue(result["ok"])
            self.assertEqual(result["operation"], "stop")
            self.assertEqual(result["reason_code"], "API_STATE_STALE")
            self.assertEqual(
                result["runtime"]["transaction_id"], transaction_id
            )
            self.assertFalse(paths["active_pid"].exists())
            self.assertFalse(paths["active_lock"].exists())
            self.assertFalse(result["cleanup"]["unrelated_process_signaled"])
            killpg.assert_not_called()

    def test_pid_reuse_and_foreign_endpoint_fail_closed(self) -> None:
        for reused, foreign, expected in (
            (
                {"pid": 990001, "process_start_identity": "reused"},
                [],
                "OWNERSHIP_UNCERTAIN",
            ),
            (
                None,
                [
                    {
                        "pid": 880001,
                        "pgid": 880001,
                        "process_start_identity": "foreign",
                    }
                ],
                "ENDPOINT_CONFLICT",
            ),
        ):
            with self.subTest(reason=expected):
                with tempfile.TemporaryDirectory() as temporary:
                    paths = api_paths(Path(temporary))
                    api_records(paths)
                    patches = (
                        mock.patch.object(
                            api, "validate_dependency", return_value={}
                        ),
                        mock.patch.object(
                            api,
                            "validate_runtime_layout",
                            return_value=None,
                        ),
                        mock.patch.object(
                            api,
                            "identity_matches",
                            return_value=(False, reused),
                        ),
                        mock.patch.object(
                            api,
                            "matching_recorded_processes",
                            return_value=[],
                        ),
                        mock.patch.object(
                            api,
                            "endpoint_listener_owners",
                            return_value=foreign,
                        ),
                    )
                    with patches[0], patches[1], patches[2], patches[3], patches[4]:
                        with self.assertRaises(api.ControllerError) as caught:
                            api.operation_reconcile(paths)
                    self.assertEqual(caught.exception.reason_code, expected)
                    self.assertTrue(paths["active_pid"].exists())
                    self.assertTrue(paths["active_lock"].exists())

    def test_live_process_without_listener_is_classified_not_signaled(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = api_paths(Path(temporary))
            api_records(paths)
            observed = {
                "pid": 990001,
                "pgid": 990001,
                "sid": 990001,
                "process_start_identity": "fixture-start",
            }
            with (
                mock.patch.object(api, "validate_dependency", return_value={}),
                mock.patch.object(
                    api, "validate_runtime_layout", return_value=None
                ),
                mock.patch.object(
                    api,
                    "identity_matches",
                    return_value=(True, observed),
                ),
                mock.patch.object(api, "find_listener", return_value=None),
                mock.patch.object(
                    api, "endpoint_listener_owners", return_value=[]
                ),
            ):
                result = api.operation_reconcile(paths)
            self.assertEqual(
                result["reason_code"], "PUBLIC_LISTENER_LOST"
            )
            self.assertTrue(paths["active_pid"].exists())
            self.assertFalse(
                result["listener"]["unrelated_process_signaled"]
            )


class BranchControllerReconcileTests(unittest.TestCase):
    def test_child_environment_resolves_build_libraries(self) -> None:
        with mock.patch.dict(
            branch.os.environ, {"LD_LIBRARY_PATH": "/existing"}, clear=True
        ):
            value = branch._child_environment(
                Path("/repo/build/bin/llama-server"),
                launch_mode="router",
                router_cache=Path("/repo/cache"),
            )
        self.assertEqual(
            value["LD_LIBRARY_PATH"], "/repo/build/bin:/existing"
        )
        self.assertEqual(value["LLAMA_CACHE"], "/repo/cache")

    def test_status_reconciles_dead_owned_record_without_signal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = branch_paths(Path(temporary))
            state = branch_records(paths)
            patches = (
                mock.patch.object(
                    branch, "validate_layout", return_value=paths
                ),
                mock.patch.object(
                    branch,
                    "ownership_predicates",
                    return_value={"all_match": False, "process_alive": False},
                ),
                mock.patch.object(
                    branch, "matching_recorded_processes", return_value=[]
                ),
                mock.patch.object(
                    branch, "endpoint_listener_owners", return_value=[]
                ),
            )
            with (
                patches[0],
                patches[1],
                patches[2],
                patches[3],
                mock.patch.object(branch.os, "killpg") as killpg,
            ):
                result = branch.status_operation(paths)
            self.assertFalse(result["active"])
            self.assertTrue(result["active_state_consistent"])
            self.assertEqual(result["lifecycle_state"], "RECONCILED")
            self.assertEqual(
                result["reconciliation"]["reason_code"], "ROUTER_STATE_STALE"
            )
            self.assertFalse(state["pid_path"].exists())
            self.assertFalse(state["lock_path"].exists())
            killpg.assert_not_called()

    def test_stale_router_records_and_pid_reuse(self) -> None:
        for alive, expected in (
            (False, "ROUTER_STATE_STALE"),
            (True, "OWNERSHIP_UNCERTAIN"),
        ):
            with self.subTest(process_alive=alive):
                with tempfile.TemporaryDirectory() as temporary:
                    paths = branch_paths(Path(temporary))
                    state = branch_records(paths)
                    patches = (
                        mock.patch.object(
                            branch, "validate_layout", return_value=paths
                        ),
                        mock.patch.object(
                            branch,
                            "ownership_predicates",
                            return_value={
                                "all_match": False,
                                "process_alive": alive,
                            },
                        ),
                        mock.patch.object(
                            branch,
                            "matching_recorded_processes",
                            return_value=[],
                        ),
                        mock.patch.object(
                            branch,
                            "endpoint_listener_owners",
                            return_value=[],
                        ),
                    )
                    with patches[0], patches[1], patches[2], patches[3]:
                        if alive:
                            with self.assertRaises(
                                branch.ControllerError
                            ) as caught:
                                branch.reconcile_operation(paths)
                            self.assertEqual(
                                caught.exception.reason_code, expected
                            )
                        else:
                            result = branch.reconcile_operation(paths)
                            self.assertEqual(
                                result["reason_code"], expected
                            )
                    self.assertEqual(
                        state["pid_path"].exists(), alive
                    )
                    self.assertEqual(
                        state["lock_path"].exists(), alive
                    )

    def test_real_foreign_listener_is_observed_without_signal(self) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            port = listener.getsockname()[1]
            api_owners = api.endpoint_listener_owners(
                "127.0.0.1", port
            )
            branch_owners = branch.endpoint_listener_owners(
                "127.0.0.1", port
            )
            self.assertIn(os.getpid(), {item["pid"] for item in api_owners})
            self.assertIn(
                os.getpid(), {item["pid"] for item in branch_owners}
            )
        finally:
            listener.close()


class EndpointReuseProbeTests(unittest.TestCase):
    def test_api_endpoint_probe_allows_owned_time_wait_reuse(self) -> None:
        probe = mock.Mock()
        with mock.patch.object(api.socket, "socket", return_value=probe):
            self.assertTrue(api.endpoint_available("127.0.0.1", 49802))
        probe.setsockopt.assert_called_once_with(
            socket.SOL_SOCKET, socket.SO_REUSEADDR, 1
        )

    def test_branch_endpoint_probe_allows_owned_time_wait_reuse(self) -> None:
        probe = mock.Mock()
        addresses = [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                ("127.0.0.1", 49803),
            )
        ]
        with (
            mock.patch.object(branch.socket, "getaddrinfo", return_value=addresses),
            mock.patch.object(branch.socket, "socket", return_value=probe),
        ):
            result = branch.endpoint_probe("127.0.0.1", 49803)
        self.assertEqual(result["port"], 49803)
        probe.setsockopt.assert_called_once_with(
            socket.SOL_SOCKET, socket.SO_REUSEADDR, 1
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
