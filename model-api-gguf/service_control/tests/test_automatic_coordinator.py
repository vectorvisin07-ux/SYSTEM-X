from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from service_control.automatic_coordinator import (
    AutomaticIntakeCoordinator,
    CoordinatorConfig,
    STATUS_SCHEMA,
)


CHILD = r'''import json,sys,time
mode=sys.argv[1]
if mode == "long":
    time.sleep(0.25)
if mode == "bad":
    print("not-json", flush=True)
    raise SystemExit(3)
action = "DISPATCH_FIRST_MODEL" if mode == "dispatch" else "NOOP_WAITING"
basis = {"basis_class":"PROCESSED","basis_identity":"sha256:"+"1"*64,"record_identity":"sha256:"+"2"*64,"record_path":"/trial/processed.json"} if action.startswith("DISPATCH") else None
automatic = {"action":action,"reason_code":"AUTOMATIC_DISPATCH_ACCEPTED" if action.startswith("DISPATCH") else "AUTOMATIC_NO_VISIBLE_CANDIDATE","active_transaction_reference":None,"existing_result_reference":None}
print(json.dumps({"schema_version":"system-x.inspector-machine-result.v1","operation":"reconcile-intake","ok":True,"reason_code":automatic["reason_code"],"message":"trial","timestamp_utc":"2026-01-01T00:00:00.000000Z","inspector_root":"/trial/INSPECTOR","transaction_id":None,"data":{"automatic_result":automatic,"terminal_basis_reference":basis},"paths":{}},sort_keys=True,separators=(",",":")),flush=True)
'''


class CoordinatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.inspector = self.root / "INSPECTOR"
        self.runtime = self.root / "runtime"
        self.inspector.mkdir()
        self.runtime.mkdir()
        self.child = self.root / "child.py"
        self.child.write_text(CHILD, encoding="utf-8")
        self.launch_mode = "dispatch"
        self.launched: list[subprocess.Popen[bytes]] = []

    def tearDown(self) -> None:
        for process in self.launched:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=2)
        self.temporary.cleanup()

    def launcher(self, argv, **kwargs):
        process = subprocess.Popen([sys.executable, str(self.child), self.launch_mode], **kwargs)
        self.launched.append(process)
        return process

    def coordinator(self, *, observer=None) -> AutomaticIntakeCoordinator:
        config = CoordinatorConfig(
            self.root,
            self.inspector,
            self.runtime,
            Path(sys.executable),
            interval_seconds=1.0,
            maximum_backoff_seconds=4.0,
            capture_limit_bytes=1024,
        )
        return AutomaticIntakeCoordinator(
            config,
            launcher=self.launcher,
            observer=observer,
            utc_clock=lambda: "2026-01-01T00:00:00.000000Z",
        )

    def wait_terminal(self, coordinator: AutomaticIntakeCoordinator, start: float = 0.0) -> dict:
        current = start
        for _ in range(100):
            snapshot = coordinator.tick(current, "RUNNING", "generation-test")
            if snapshot["active_child_identity"] is None and snapshot["last_completion_utc"] is not None:
                return snapshot
            time.sleep(0.01)
            current += 0.01
        self.fail("coordinator child did not reach a terminal state")

    def test_one_child_structured_zero_argument_dispatch_and_terminal_record(self) -> None:
        coordinator = self.coordinator()
        first = coordinator.tick(0.0, "RUNNING", "generation-test")
        self.assertEqual(first["coordinator_state"], "CHILD_RUNNING")
        self.assertEqual(first["active_child_identity"]["pid"] > 0, True)
        self.assertEqual(first["last_invocation_id"].startswith("inv-"), True)
        second = coordinator.tick(0.1, "RUNNING", "generation-test")
        self.assertEqual(second["active_child_identity"]["pid"], first["active_child_identity"]["pid"])
        final = self.wait_terminal(coordinator, 0.2)
        self.assertEqual(final["last_inspector_action"], "DISPATCH_FIRST_MODEL")
        self.assertEqual(final["consecutive_method_failure_count"], 0)
        self.assertEqual(len(self.launched), 1)
        self.assertTrue((self.runtime / "coordinator" / "status.json").is_file())

    def test_long_child_never_overlaps(self) -> None:
        self.launch_mode = "long"
        coordinator = self.coordinator()
        coordinator.tick(0.0, "RUNNING", "generation-test")
        for current in (0.1, 0.2, 0.3):
            snapshot = coordinator.tick(current, "RUNNING", "generation-test")
            self.assertLessEqual(len(self.launched), 1)
            self.assertIn(snapshot["coordinator_state"], {"CHILD_RUNNING", "IDLE"})
        self.wait_terminal(coordinator, 0.4)
        self.assertEqual(len(self.launched), 1)

    def test_bad_result_enters_bounded_backoff(self) -> None:
        self.launch_mode = "bad"
        coordinator = self.coordinator()
        final = self.wait_terminal(coordinator)
        self.assertEqual(final["coordinator_state"], "BACKOFF")
        self.assertEqual(final["consecutive_method_failure_count"], 1)
        self.assertEqual(final["last_reason_code"], "COORDINATOR_MACHINE_RESULT_INVALID")
        self.assertLessEqual(final["next_due_monotonic"], 4.0)

    def test_uncertain_owner_is_not_signalled_or_replaced(self) -> None:
        coordinator_observer = self.coordinator().observer
        observations = 0
        def uncertain(pid: int) -> dict:
            nonlocal observations
            observed = coordinator_observer(pid)
            if observations > 0:
                observed["pgid"] += 1
            observations += 1
            return observed
        self.launch_mode = "long"
        coordinator = self.coordinator(observer=uncertain)
        first = coordinator.tick(0.0, "RUNNING", "generation-test")
        second = coordinator.tick(0.1, "RUNNING", "generation-test")
        self.assertEqual(second["coordinator_state"], "REATTACHING")
        self.assertEqual(len(self.launched), 1)
        self.assertEqual(second["last_reason_code"], "COORDINATOR_OWNER_UNCERTAIN")

    def test_stopped_state_closes_owned_child_and_disables(self) -> None:
        self.launch_mode = "long"
        coordinator = self.coordinator()
        coordinator.tick(0.0, "RUNNING", "generation-test")
        snapshot = coordinator.tick(0.1, "STOPPED", "generation-test")
        self.assertIn(snapshot["coordinator_state"], {"STOPPING", "DISABLED", "BACKOFF"})
        for _ in range(20):
            if self.launched[0].poll() is not None:
                break
            time.sleep(0.01)
        self.assertIsNotNone(self.launched[0].poll())


if __name__ == "__main__":
    unittest.main()
