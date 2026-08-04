from __future__ import annotations

import json
import os
import shutil
import stat
import tempfile
import unittest
from pathlib import Path

from system_x_inspector.config import (
    load_configuration,
    validate_configuration_values,
)
from system_x_inspector.errors import InspectorError
from system_x_inspector.locking import TransactionLock
from system_x_inspector.paths import InspectorPaths, physical_state
from system_x_inspector.records import atomic_write_json, read_json_record


class FoundationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.production_paths = InspectorPaths.discover()
        self.temporary = Path(
            tempfile.mkdtemp(prefix="inspector-foundation-", dir=os.environ["TMPDIR"])
        )
        self.explicit_root = self.temporary / "INSPECTOR"
        self.explicit_root.mkdir(mode=0o700)
        self.paths = InspectorPaths.discover(self.explicit_root)
        for path in (
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
            self.paths.staging,
            self.paths.tmp,
            self.paths.schema_root,
            self.paths.capability_root,
            self.paths.capability_records,
            self.paths.capability_bindings,
        ):
            path.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.paths.environment_lock.write_text("{}\n", encoding="utf-8")
        self.paths.environment_lock.chmod(0o600)

    def tearDown(self) -> None:
        shutil.rmtree(self.temporary)

    def valid_configuration(self) -> dict[str, object]:
        return {
            "schema_version": "system-x.inspector-configuration.v1",
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

    def test_persistent_layout_is_private_and_separate(self) -> None:
        mapping = self.paths.persistent_mapping()
        for name, path in mapping.items():
            state = physical_state(path, self.paths.inspector_root)
            if name == "environment_lock":
                self.assertEqual(state["state"], "regular_file")
                continue
            self.assertEqual(state["state"], "regular_directory", name)
            if name not in {"inspector_root", "schema_root"}:
                self.assertEqual(state["mode"], "0700", name)
        result_roots = {
            self.paths.inspection_results,
            self.paths.decision_results,
            self.paths.handoff_results,
            self.paths.publication_results,
            self.paths.qualification_results,
            self.paths.promotion_results,
            self.paths.retirement_results,
            self.paths.deployment_results,
        }
        self.assertEqual(len(result_roots), 8)
        for result_root in (
            self.paths.inspection_results,
            self.paths.decision_results,
            self.paths.handoff_results,
            self.paths.publication_results,
            self.paths.qualification_results,
            self.paths.promotion_results,
            self.paths.retirement_results,
            self.paths.deployment_results,
        ):
            for path in result_root.iterdir():
                self.assertTrue(path.is_file())
                self.assertFalse(path.is_symlink())
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_explicit_root_and_physical_states(self) -> None:
        self.assertEqual(self.paths.inspector_root, self.explicit_root)
        absent = physical_state(
            self.explicit_root / "missing", self.explicit_root
        )
        self.assertEqual(absent["state"], "absent")
        target = self.explicit_root / "target"
        target.write_text("sentinel", encoding="utf-8")
        link = self.explicit_root / "link"
        link.symlink_to(target)
        self.assertEqual(
            physical_state(link, self.explicit_root)["state"], "symlink"
        )

    def test_symlinked_and_wrong_type_roots_rejected(self) -> None:
        link = self.temporary / "root-link"
        link.symlink_to(self.explicit_root, target_is_directory=True)
        with self.assertRaisesRegex(InspectorError, "symlink"):
            InspectorPaths.discover(link)
        wrong = self.temporary / "root-file"
        wrong.write_text("not a directory", encoding="utf-8")
        with self.assertRaisesRegex(InspectorError, "not a directory"):
            InspectorPaths.discover(wrong)

    def test_valid_configuration_identity_is_deterministic(self) -> None:
        first = validate_configuration_values(
            self.valid_configuration(), self.paths
        )
        reordered = json.loads(
            json.dumps(self.valid_configuration(), sort_keys=True)
        )
        second = validate_configuration_values(reordered, self.paths)
        self.assertEqual(first.identity, second.identity)
        changed = self.valid_configuration()
        changed["intake_bounds"]["maximum_entry_count"] = 127
        third = validate_configuration_values(changed, self.paths)
        self.assertNotEqual(first.identity, third.identity)

    def test_invalid_configuration_matrix(self) -> None:
        cases = []
        unknown = self.valid_configuration()
        unknown["unknown"] = True
        cases.append(unknown)
        zero = self.valid_configuration()
        zero["intake_bounds"]["maximum_entry_count"] = 0
        cases.append(zero)
        too_large = self.valid_configuration()
        too_large["intake_bounds"]["maximum_directory_depth"] = 65
        cases.append(too_large)
        inconsistent = self.valid_configuration()
        inconsistent["runtime_root"] = str(self.explicit_root / "other")
        cases.append(inconsistent)
        open_mode = self.valid_configuration()
        open_mode["record_policy"]["status_file_mode"] = "0644"
        cases.append(open_mode)
        for document in cases:
            with self.subTest(document=document):
                with self.assertRaises(InspectorError) as caught:
                    validate_configuration_values(document, self.paths)
                self.assertEqual(caught.exception.reason_code, "CONFIG_INVALID")

    def test_configuration_file_and_symlink_handling(self) -> None:
        path = self.paths.tmp / "config.json"
        path.write_text(
            json.dumps(self.valid_configuration()), encoding="utf-8"
        )
        first = load_configuration(path, self.paths)
        self.assertTrue(first.identity.startswith("sha256:"))
        link = self.paths.tmp / "config-link.json"
        link.symlink_to(path)
        with self.assertRaises(InspectorError) as caught:
            load_configuration(link, self.paths)
        self.assertEqual(
            caught.exception.reason_code, "CONFIG_SYMLINK_REJECTED"
        )

    def test_atomic_records_and_unrelated_sentinel(self) -> None:
        record_root = self.paths.tmp / "records"
        record_root.mkdir(mode=0o700)
        sentinel = record_root / "sentinel.txt"
        sentinel.write_text("PRESERVE-ME", encoding="utf-8")
        target = record_root / "current.json"
        for sequence in range(5):
            value = {"sequence": sequence, "state": "IDLE"}
            identity = atomic_write_json(target, value, mode=0o600)
            self.assertEqual(read_json_record(target), value)
            self.assertTrue(identity.startswith("sha256:"))
            self.assertEqual(
                stat.S_IMODE(target.stat().st_mode), 0o600
            )
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "PRESERVE-ME")
        self.assertEqual(
            sorted(path.name for path in record_root.iterdir()),
            ["current.json", "sentinel.txt"],
        )

    def test_lock_acquire_duplicate_and_release(self) -> None:
        first = TransactionLock(
            self.paths, transaction_id="tx-foundation-1", operation="validate-intake"
        )
        value = first.acquire()
        self.assertEqual(value["transaction_id"], "tx-foundation-1")
        self.assertIsInstance(value["pid"], int)
        self.assertIsInstance(value["process_start_identity"], str)
        duplicate = TransactionLock(
            self.paths, transaction_id="tx-foundation-2", operation="validate-intake"
        )
        with self.assertRaises(InspectorError) as caught:
            duplicate.acquire()
        self.assertEqual(
            caught.exception.reason_code, "TRANSACTION_LOCK_ACTIVE"
        )
        first.release()
        self.assertFalse((self.paths.locks / "active.json").exists())

    def test_uncertain_preexisting_lock_fails_closed(self) -> None:
        active = self.paths.locks / "active.json"
        active.write_text("{}\n", encoding="utf-8")
        os.chmod(active, 0o600)
        lock = TransactionLock(
            self.paths, transaction_id="tx-foundation-3", operation="validate-intake"
        )
        with self.assertRaises(InspectorError) as caught:
            lock.acquire()
        self.assertEqual(
            caught.exception.reason_code,
            "TRANSACTION_OWNERSHIP_UNCERTAIN",
        )
        self.assertTrue(active.exists())


if __name__ == "__main__":
    unittest.main()
