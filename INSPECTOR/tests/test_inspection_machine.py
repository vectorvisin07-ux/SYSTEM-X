from __future__ import annotations

import contextlib
import io
import json
import shutil
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from system_x_inspector.constants import SCHEMA_IDENTITIES
from system_x_inspector.errors import InspectorError
from system_x_inspector.machine import main
from system_x_inspector.paths import InspectorPaths
from system_x_inspector.records import (
    atomic_create_json,
    atomic_write_json,
    read_json_record,
)
from system_x_inspector.results import utc_now
from tests.test_gguf import build_gguf
from tests.test_native import write_config, write_safe


class InspectionMachineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = Path(
            tempfile.mkdtemp(prefix="inspector-operation-", dir="/tmp")
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
        atomic_write_json(
            self.paths.status / "current.json",
            {
                "schema_version": SCHEMA_IDENTITIES["status"],
                "state": "IDLE",
                "reason_code": "OK",
                "updated_utc": utc_now(),
                "inspector_root": str(self.root),
                "active_transaction_id": None,
                "last_transaction_id": None,
            },
            mode=0o600,
        )
        self.config_path = self.paths.tmp / "config.json"
        self.config_path.write_text(
            json.dumps(
                {
                    "schema_version": SCHEMA_IDENTITIES["configuration"],
                    "intake_root": str(self.paths.intake_root),
                    "runtime_root": str(self.paths.runtime_root),
                    "intake_bounds": {
                        "maximum_directory_depth": 8,
                        "maximum_entry_count": 256,
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
                        "publication": str(
                            self.paths.publication_results
                        ),
                    },
                },
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.temporary)

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

    def inspect(self, name: str) -> tuple[int, dict[str, object], str]:
        return self.run_main(
            "inspect",
            "--config",
            str(self.config_path),
            "--target",
            name,
        )

    def assert_idle(self) -> None:
        status = read_json_record(self.paths.status / "current.json")
        self.assertEqual(status["state"], "IDLE")
        self.assertIsNone(status["active_transaction_id"])
        self.assertFalse((self.paths.locks / "active.json").exists())

    def make_all_classes(self) -> dict[str, str]:
        valid = self.paths.intake_root / "valid-content.bin"
        valid.write_bytes(build_gguf())

        native = self.paths.intake_root / "native-bundle"
        native.mkdir()
        write_config(native)
        write_safe(native, "model.safetensors", [("w", "U8", [1], b"x")])

        unknown = self.paths.intake_root / "misleading.gguf"
        unknown.write_bytes(b"not a model container")

        corrupt = self.paths.intake_root / "corrupt-content"
        corrupt.write_bytes(
            build_gguf(
                metadata=[
                    ("general.architecture", 8, "one"),
                    ("general.architecture", 8, "two"),
                ]
            )
        )

        incomplete = self.paths.intake_root / "incomplete-content"
        incomplete.write_bytes(b"GGUF\x03\x00")

        contradictory = self.paths.intake_root / "contradictory-bundle"
        contradictory.mkdir()
        write_config(contradictory)
        write_safe(
            contradictory,
            "model.safetensors",
            [("w", "U8", [1], b"x")],
        )
        (contradictory / "coexisting.bin").write_bytes(build_gguf())
        return {
            valid.name: "GGUF",
            native.name: "NATIVE",
            unknown.name: "UNKNOWN",
            contradictory.name: "CONTRADICTORY",
            corrupt.name: "CORRUPT",
            incomplete.name: "INCOMPLETE",
        }

    def test_all_terminal_classes_exit_zero_and_publish_private_records(self) -> None:
        expected = self.make_all_classes()
        decision_sentinel = self.paths.decision_results / "preserve"
        handoff_sentinel = self.paths.handoff_results / "preserve"
        decision_sentinel.write_bytes(b"decision-unchanged")
        handoff_sentinel.write_bytes(b"handoff-unchanged")
        for name, terminal_class in expected.items():
            with self.subTest(terminal_class=terminal_class):
                exit_status, envelope, error = self.inspect(name)
                self.assertEqual(exit_status, 0)
                self.assertEqual(error, "")
                self.assertTrue(envelope["ok"])
                self.assertEqual(
                    envelope["reason_code"], "INSPECTION_COMPLETE"
                )
                self.assertEqual(
                    envelope["data"]["terminal_class"], terminal_class
                )
                result_path = Path(envelope["data"]["result_path"])
                details = result_path.lstat()
                self.assertTrue(stat.S_ISREG(details.st_mode))
                self.assertFalse(stat.S_ISLNK(details.st_mode))
                self.assertEqual(stat.S_IMODE(details.st_mode), 0o600)
                self.assertEqual(details.st_nlink, 1)
                record = read_json_record(result_path)
                self.assertEqual(
                    record["schema_version"],
                    SCHEMA_IDENTITIES["inspection_result"],
                )
                self.assertEqual(
                    record["classification"]["terminal_class"],
                    terminal_class,
                )
                self.assertEqual(len(record["normalized"]), 15)
                self.assertEqual(
                    record["artifact"]["identity"],
                    envelope["data"]["artifact_identity"],
                )
                transaction = read_json_record(
                    self.paths.transactions
                    / f"{envelope['transaction_id']}.json"
                )
                self.assertEqual(transaction["state"], "COMPLETED")
                self.assertEqual(
                    transaction["reason_code"], "INSPECTION_COMPLETE"
                )
                self.assert_idle()
        self.assertEqual(decision_sentinel.read_bytes(), b"decision-unchanged")
        self.assertEqual(handoff_sentinel.read_bytes(), b"handoff-unchanged")
        self.assertEqual(
            len(list(self.paths.inspection_results.glob("inspection-*.json"))),
            6,
        )

    def test_unsafe_and_mutated_inputs_exit_two_without_result(self) -> None:
        target = self.paths.intake_root / "target"
        target.write_bytes(build_gguf())
        symlink = self.paths.intake_root / "unsafe"
        symlink.symlink_to(target)
        before = list(self.paths.inspection_results.iterdir())
        exit_status, envelope, error = self.inspect(symlink.name)
        self.assertEqual(exit_status, 2)
        self.assertEqual(error, "")
        self.assertFalse(envelope["ok"])
        self.assertEqual(envelope["reason_code"], "INTAKE_TARGET_SYMLINK")
        self.assertEqual(list(self.paths.inspection_results.iterdir()), before)
        self.assert_idle()

        with mock.patch(
            "system_x_inspector.runtime.artifact_source_snapshot",
            side_effect=[
                "sha256:" + "1" * 64,
                "sha256:" + "2" * 64,
            ],
        ):
            exit_status, envelope, error = self.inspect(target.name)
        self.assertEqual(exit_status, 2)
        self.assertEqual(error, "")
        self.assertFalse(envelope["ok"])
        self.assertEqual(
            envelope["reason_code"],
            "ARTIFACT_CHANGED_DURING_INSPECTION",
        )
        self.assertEqual(list(self.paths.inspection_results.iterdir()), before)
        self.assert_idle()

    def test_injected_internal_error_exits_seventy_without_result(self) -> None:
        target = self.paths.intake_root / "target"
        target.write_bytes(build_gguf())
        with mock.patch(
            "system_x_inspector.runtime.classify_artifact",
            side_effect=RuntimeError("test-only injected inspection failure"),
        ):
            exit_status, envelope, error = self.inspect(target.name)
        self.assertEqual(exit_status, 70)
        self.assertEqual(error, "")
        self.assertFalse(envelope["ok"])
        self.assertEqual(
            envelope["reason_code"], "INSPECTION_INTERNAL_ERROR"
        )
        self.assertFalse(any(self.paths.inspection_results.iterdir()))
        self.assertNotIn("test-only", json.dumps(envelope))
        self.assert_idle()

    def test_atomic_result_publication_never_overwrites(self) -> None:
        path = self.paths.inspection_results / "collision.json"
        first = {"schema_version": "fixture", "value": 1}
        second = {"schema_version": "fixture", "value": 2}
        identity = atomic_create_json(path, first, mode=0o600)
        self.assertRegex(identity, r"^sha256:[0-9a-f]{64}$")
        with self.assertRaises(InspectorError) as raised:
            atomic_create_json(path, second, mode=0o600)
        self.assertEqual(
            raised.exception.reason_code, "INSPECTION_RECORD_COLLISION"
        )
        self.assertEqual(read_json_record(path), first)
        details = path.lstat()
        self.assertEqual(stat.S_IMODE(details.st_mode), 0o600)
        self.assertEqual(details.st_nlink, 1)


if __name__ == "__main__":
    unittest.main()
