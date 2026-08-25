from __future__ import annotations

import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from system_x_inspector.config import load_configuration
from system_x_inspector.constants import SCHEMA_IDENTITIES
from system_x_inspector.errors import InspectorError
from system_x_inspector.machine import main
from system_x_inspector.paths import InspectorPaths
from system_x_inspector.records import atomic_write_json, read_json_record
from system_x_inspector.results import utc_now
from system_x_inspector.runtime import validate_intake_transaction


class MachineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = Path(
            tempfile.mkdtemp(prefix="inspector-machine-", dir="/tmp")
        )
        self.root = self.temporary / "INSPECTOR"
        self.root.mkdir(mode=0o700)
        self.paths = InspectorPaths.discover(self.root)
        for path in (
            self.paths.schema_root,
            self.paths.intake_root,
            self.paths.runtime_root,
            self.paths.logs,
            self.paths.locks,
            self.paths.status,
            self.paths.transactions,
            self.paths.inspection_results,
            self.paths.decision_results,
            self.paths.handoff_results,
            self.paths.publication_results,
            self.paths.qualification_results,
            self.paths.promotion_results,
            self.paths.retirement_results,
            self.paths.deployment_results,
            self.paths.automatic_results,
            self.paths.automatic_processed_results,
            self.paths.automatic_rejected_results,
            self.paths.staging,
            self.paths.tmp,
            self.paths.capability_root,
            self.paths.capability_records,
            self.paths.capability_bindings,
        ):
            path.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.paths.environment_lock.write_text(
            json.dumps(
                {
                    "schema_version": SCHEMA_IDENTITIES["environment_lock"],
                    "test_fixture": True,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        self.status = {
            "schema_version": SCHEMA_IDENTITIES["status"],
            "state": "IDLE",
            "reason_code": "OK",
            "updated_utc": utc_now(),
            "inspector_root": str(self.paths.inspector_root),
            "active_transaction_id": None,
            "last_transaction_id": None,
        }
        atomic_write_json(
            self.paths.status / "current.json", self.status, mode=0o600
        )
        self.config_path = self.paths.tmp / "config.json"
        self.config_path.write_text(
            json.dumps(self.valid_configuration()), encoding="utf-8"
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.temporary)

    def valid_configuration(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_IDENTITIES["configuration"],
            "intake_root": str(self.paths.intake_root),
            "runtime_root": str(self.paths.runtime_root),
            "intake_bounds": {
                "maximum_directory_depth": 8,
                "maximum_entry_count": 128,
                "maximum_relative_path_bytes": 1024,
                "maximum_component_bytes": 128,
            },
            "record_policy": {
                "status_file_mode": "0600",
                "transaction_file_mode": "0600",
                "log_file_mode": "0600",
            },
            "result_roots": {
                "inspection": str(self.paths.inspection_results),
                "decision": str(self.paths.decision_results),
                "handoff": str(self.paths.handoff_results),
                "publication": str(self.paths.publication_results),
            },
        }

    def run_main(self, *arguments: str) -> tuple[int, dict[str, object], str]:
        output = io.StringIO()
        error = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
            exit_status = main(
                ["--inspector-root", str(self.root), *arguments]
            )
        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 1)
        return exit_status, json.loads(lines[0]), error.getvalue()

    def assert_envelope(
        self, result: dict[str, object], operation: str, ok: bool
    ) -> None:
        self.assertEqual(
            set(result),
            {
                "schema_version",
                "operation",
                "ok",
                "reason_code",
                "message",
                "timestamp_utc",
                "inspector_root",
                "transaction_id",
                "data",
                "paths",
            },
        )
        self.assertEqual(
            result["schema_version"], SCHEMA_IDENTITIES["machine_result"]
        )
        self.assertEqual(result["operation"], operation)
        self.assertIs(result["ok"], ok)

    def test_read_only_operations_succeed_without_transactions(self) -> None:
        operations = [
            ("identify",),
            ("validate-config", "--config", str(self.config_path)),
            ("layout",),
            ("status",),
            ("list-intake",),
        ]
        before = list(self.paths.transactions.iterdir())
        for arguments in operations:
            with self.subTest(operation=arguments[0]):
                exit_status, result, error = self.run_main(*arguments)
                self.assertEqual(exit_status, 0)
                self.assertEqual(error, "")
                self.assert_envelope(result, arguments[0], True)
                self.assertEqual(result["reason_code"], "OK")
        self.assertEqual(list(self.paths.transactions.iterdir()), before)

    def test_show_connection_absent_is_read_only_expected_result(self) -> None:
        before = {
            path.relative_to(self.root).as_posix(): path.stat().st_mtime_ns
            for path in self.root.rglob("*")
        }
        exit_status, result, error = self.run_main("show-connection")
        self.assertEqual(exit_status, 2)
        self.assertEqual(error, "")
        self.assert_envelope(result, "show-connection", False)
        self.assertEqual(
            result["reason_code"], "CONNECTION_NOT_INITIALIZED"
        )
        after = {
            path.relative_to(self.root).as_posix(): path.stat().st_mtime_ns
            for path in self.root.rglob("*")
        }
        self.assertEqual(after, before)

    def test_validate_intake_success_records_and_returns_idle(self) -> None:
        (self.paths.intake_root / "candidate").write_bytes(b"packet")
        exit_status, result, error = self.run_main(
            "validate-intake",
            "--config",
            str(self.config_path),
            "--target",
            "candidate",
        )
        self.assertEqual(exit_status, 0)
        self.assertEqual(error, "")
        self.assert_envelope(result, "validate-intake", True)
        self.assertEqual(
            result["data"]["candidate"]["format_classification"],
            "not_performed",
        )
        transaction_id = result["transaction_id"]
        self.assertIsInstance(transaction_id, str)
        transaction = read_json_record(
            self.paths.transactions / f"{transaction_id}.json"
        )
        self.assertEqual(transaction["state"], "COMPLETED")
        self.assertEqual(transaction["reason_code"], "OK")
        status = read_json_record(self.paths.status / "current.json")
        self.assertEqual(status["state"], "IDLE")
        self.assertEqual(status["last_transaction_id"], transaction_id)
        self.assertFalse((self.paths.locks / "active.json").exists())

    def test_expected_rejection_is_json_exit_two_and_returns_idle(self) -> None:
        exit_status, result, error = self.run_main(
            "validate-intake",
            "--config",
            str(self.config_path),
        )
        self.assertEqual(exit_status, 2)
        self.assertEqual(error, "")
        self.assert_envelope(result, "validate-intake", False)
        self.assertEqual(result["reason_code"], "INTAKE_EMPTY")
        self.assertIsInstance(result["transaction_id"], str)
        transaction = read_json_record(
            self.paths.transactions / f"{result['transaction_id']}.json"
        )
        self.assertEqual(transaction["state"], "FAILED")
        self.assertEqual(transaction["reason_code"], "INTAKE_EMPTY")
        status = read_json_record(self.paths.status / "current.json")
        self.assertEqual(status["state"], "IDLE")
        self.assertFalse((self.paths.locks / "active.json").exists())

    def test_invalid_config_is_json_exit_two_without_transaction(self) -> None:
        invalid = self.paths.tmp / "invalid.json"
        invalid.write_text("{}\n", encoding="utf-8")
        before = list(self.paths.transactions.iterdir())
        exit_status, result, error = self.run_main(
            "validate-config", "--config", str(invalid)
        )
        self.assertEqual(exit_status, 2)
        self.assertEqual(error, "")
        self.assert_envelope(result, "validate-config", False)
        self.assertEqual(result["reason_code"], "CONFIG_INVALID")
        self.assertEqual(list(self.paths.transactions.iterdir()), before)

    def test_injected_internal_failure_is_json_exit_seventy(self) -> None:
        with mock.patch(
            "system_x_inspector.machine.execute",
            side_effect=RuntimeError("test-only injected failure"),
        ):
            exit_status, result, error = self.run_main("identify")
        self.assertEqual(exit_status, 70)
        self.assertEqual(error, "")
        self.assert_envelope(result, "identify", False)
        self.assertEqual(result["reason_code"], "INTERNAL_ERROR")
        self.assertNotIn("Traceback", json.dumps(result))
        self.assertNotIn("test-only", json.dumps(result))

    def test_transition_sequence_success_and_internal_failure(self) -> None:
        (self.paths.intake_root / "candidate").write_bytes(b"packet")
        configuration = load_configuration(self.config_path, self.paths)
        observed: list[tuple[str, str]] = []

        def observer(kind: str, value: dict[str, object]) -> None:
            observed.append((kind, str(value["state"])))

        transaction_id, _ = validate_intake_transaction(
            self.paths,
            configuration,
            "candidate",
            transition_observer=observer,
        )
        self.assertEqual(
            observed,
            [
                ("status", "VALIDATING_INTAKE"),
                ("transaction", "VALIDATING_INTAKE"),
                ("status", "IDLE"),
                ("transaction", "COMPLETED"),
            ],
        )
        self.assertTrue(
            (self.paths.transactions / f"{transaction_id}.json").exists()
        )
        observed.clear()

        def fail_validator(*args: object, **kwargs: object) -> dict[str, object]:
            raise RuntimeError("test-only validator failure")

        with self.assertRaises(InspectorError) as caught:
            validate_intake_transaction(
                self.paths,
                configuration,
                "candidate",
                validator=fail_validator,
                transition_observer=observer,
            )
        self.assertEqual(caught.exception.reason_code, "INTERNAL_ERROR")
        self.assertEqual(caught.exception.exit_status, 70)
        self.assertEqual(
            observed,
            [
                ("status", "VALIDATING_INTAKE"),
                ("transaction", "VALIDATING_INTAKE"),
                ("status", "FAILED"),
                ("transaction", "FAILED"),
                ("status", "IDLE"),
            ],
        )
        self.assertEqual(
            read_json_record(self.paths.status / "current.json")["state"],
            "IDLE",
        )
        self.assertFalse((self.paths.locks / "active.json").exists())


if __name__ == "__main__":
    unittest.main()
