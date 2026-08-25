from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from system_x_inspector.automatic_intake import (
    AutomaticIntakePolicy,
    build_automatic_dispatch_basis,
    derive_first_model_policy,
    discover_automatic_candidate,
    identify_regular_file,
    reconcile_automatic_intake,
    validate_automatic_result,
)
from system_x_inspector.automatic_basis import persist_automatic_terminal_basis
from system_x_inspector.errors import InspectorError
from system_x_inspector.locking import TransactionLock
from system_x_inspector.machine import build_parser
from system_x_inspector.paths import InspectorPaths


class FakeAdapter:
    def __init__(self) -> None:
        self.prestate = {
            "desired_state": "RUNNING",
            "model_service_state": "WAITING_FOR_MODEL",
            "ready_model_count": 0,
            "model_rows": 0,
            "active_managed_locations": 0,
            "default_alias": None,
            "default_target": None,
            "warm_model_id": None,
            "active_transaction_reference": None,
            "recovery_state": "IDLE",
            "capability_binding_identity": "sha256:" + "1" * 64,
            "operating_profile_identity": "sha256:" + "2" * 64,
            "registry_generation": 0,
        }

    def capture_prestate(self, paths: InspectorPaths) -> dict[str, object]:
        return dict(self.prestate)


class AutomaticIntakeTest(unittest.TestCase):
    def setUp(self) -> None:
        base = Path(os.environ.get("TMPDIR", tempfile.gettempdir()))
        self.temporary = Path(tempfile.mkdtemp(prefix="automatic-intake-", dir=base))
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
            self.paths.deployment_results,
        ):
            path.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.adapter = FakeAdapter()
        self.calls: list[dict[str, object]] = []
        self.fail_dispatch = False

    def tearDown(self) -> None:
        shutil.rmtree(self.temporary)

    def dispatch(self, paths: InspectorPaths, request: dict[str, object], *, adapter: object):
        self.calls.append(dict(request))
        record = {
            "transaction_id": "tx-fake-automatic",
            "deployment_id": "deployment-fake-automatic",
            "result_class": (
                "DEPLOYMENT_FAILED_CLEAN"
                if self.fail_dispatch
                else "DEPLOYMENT_COMPLETE"
            ),
        }
        return (
            record["transaction_id"],
            record,
            paths.deployment_results / "deployment-fake-automatic.json",
            "sha256:" + "3" * 64,
        )

    def reconcile(self, *, waiter=None, observer=None, identifier=identify_regular_file):
        return reconcile_automatic_intake(
            self.paths,
            policy=AutomaticIntakePolicy(1.0),
            adapter=self.adapter,
            dispatcher=self.dispatch,
            waiter=waiter,
            observer=observer,
            identifier=identifier,
        )

    def test_empty_and_hidden_wait(self) -> None:
        result = self.reconcile(waiter=lambda seconds: None)
        self.assertEqual(result["action"], "NOOP_WAITING")
        self.assertEqual(result["visible_candidate_count"], 0)
        (self.paths.intake_root / ".copying.tmp").write_bytes(b"x")
        (self.paths.intake_root / "model.bin").write_bytes(b"stable")
        result = self.reconcile(waiter=lambda seconds: None)
        self.assertEqual(result["action"], "DISPATCH_FIRST_MODEL")
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(
            set(self.calls[0]),
            {
                "candidate_name",
                "deployment_mode",
                "required_capability_profile",
                "retirement_policy",
            },
        )

    def test_multiple_never_dispatches(self) -> None:
        (self.paths.intake_root / "one").write_bytes(b"x")
        (self.paths.intake_root / "two").write_bytes(b"x")
        result = self.reconcile(waiter=lambda seconds: None)
        self.assertEqual(result["action"], "NOOP_MULTIPLE_CANDIDATES")
        self.assertEqual(self.calls, [])

    def test_directory_symlink_hardlink_special_and_unreadable_reject(self) -> None:
        cases = [
            ("directory", lambda p: p.mkdir(), "AUTOMATIC_DIRECTORY_REJECTED"),
            ("symlink", lambda p: os.symlink("missing", p), "AUTOMATIC_SYMLINK_REJECTED"),
            ("fifo", lambda p: os.mkfifo(p), "AUTOMATIC_SPECIAL_FILE_REJECTED"),
        ]
        for name, creator, reason in cases:
            with self.subTest(name=name):
                for item in self.paths.intake_root.iterdir():
                    if item.is_dir() and not item.is_symlink():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                creator(self.paths.intake_root / name)
                result = self.reconcile(waiter=lambda seconds: None)
                self.assertEqual(result["action"], "REJECT_CANDIDATE")
                self.assertEqual(result["reason_code"], reason)
                self.assertEqual(self.calls, [])
        for item in self.paths.intake_root.iterdir():
            item.unlink()
        original = self.paths.intake_root / ".original"
        link = self.paths.intake_root / "hardlink"
        original.write_bytes(b"x")
        os.link(original, link)
        result = self.reconcile(waiter=lambda seconds: None)
        self.assertEqual(result["reason_code"], "AUTOMATIC_HARDLINK_REJECTED")
        link.unlink()
        unreadable = self.paths.intake_root / "unreadable"
        unreadable.write_bytes(b"x")
        unreadable.chmod(0)
        try:
            result = self.reconcile(waiter=lambda seconds: None)
            self.assertEqual(result["reason_code"], "AUTOMATIC_UNREADABLE_CANDIDATE")
        finally:
            unreadable.chmod(0o600)

    def test_copy_changes_between_a_b_and_content_c_do_not_dispatch(self) -> None:
        candidate = self.paths.intake_root / "model"
        candidate.write_bytes(b"a")
        result = self.reconcile(
            waiter=lambda seconds: candidate.write_bytes(b"b")
        )
        self.assertEqual(result["action"], "NOOP_COPY_IN_PROGRESS")
        self.assertEqual(self.calls, [])
        candidate.write_bytes(b"a")

        def changing_identifier(path: Path):
            path.write_bytes(b"c")
            return identify_regular_file(path)

        result = self.reconcile(
            waiter=lambda seconds: None,
            identifier=changing_identifier,
        )
        self.assertEqual(result["action"], "NOOP_COPY_IN_PROGRESS")
        self.assertEqual(self.calls, [])

    def test_policy_and_filename_independent_basis(self) -> None:
        policy = derive_first_model_policy(self.adapter.prestate)
        self.assertEqual(policy["deployment_mode"], "install-first")
        self.assertEqual(policy["required_capability_profile"], "CORE_CHAT")
        self.assertEqual(policy["retirement_policy"], "retain-incumbent")
        self.assertEqual(policy["retirement_action"], "none")
        (self.paths.intake_root / "first").write_bytes(b"same")
        artifact = identify_regular_file(self.paths.intake_root / "first")
        first = build_automatic_dispatch_basis(
            {"relative_name": "first"}, artifact, AutomaticIntakePolicy(1), self.adapter.prestate
        )
        second = build_automatic_dispatch_basis(
            {"relative_name": "different"}, artifact, AutomaticIntakePolicy(1), self.adapter.prestate
        )
        self.assertEqual(first["dispatch_basis_identity"], second["dispatch_basis_identity"])
        self.adapter.prestate["ready_model_count"] = 1
        with self.assertRaisesRegex(InspectorError, "replacement"):
            derive_first_model_policy(self.adapter.prestate)
        self.adapter.prestate["ready_model_count"] = 0
        self.adapter.prestate["capability_binding_identity"] = None
        with self.assertRaises(InspectorError):
            derive_first_model_policy(self.adapter.prestate)
    def test_failed_clean_basis_is_not_dispatched_again(self) -> None:
        self.fail_dispatch = True
        (self.paths.intake_root / "model").write_bytes(b"stable")
        first = self.reconcile(waiter=lambda seconds: None)
        self.assertEqual(first["action"], "REJECT_CANDIDATE")
        persist_automatic_terminal_basis(self.paths, first)
        self.calls.clear()
        second = self.reconcile(waiter=lambda seconds: None)
        self.assertEqual(second["action"], "REJECT_CANDIDATE")
        self.assertEqual(second["reason_code"], "AUTOMATIC_DISPATCH_FAILED_CLEAN")
        self.assertEqual(self.calls, [])

    def test_failed_clean_basis_reopens_after_source_epoch_advances(self) -> None:
        self.fail_dispatch = True
        (self.paths.intake_root / "model").write_bytes(b"stable")
        epoch_name = "system_x_inspector.automatic_intake.AUTOMATIC_SOURCE_IMPLEMENTATION_EPOCH"
        with mock.patch(epoch_name, "test-source-epoch-one"):
            first = self.reconcile(waiter=lambda seconds: None)
            self.assertEqual(first["action"], "REJECT_CANDIDATE")
            persist_automatic_terminal_basis(self.paths, first)
        self.calls.clear()
        with mock.patch(epoch_name, "test-source-epoch-two"):
            second = self.reconcile(waiter=lambda seconds: None)
        self.assertEqual(second["action"], "REJECT_CANDIDATE")
        self.assertEqual(second["reason_code"], "AUTOMATIC_DISPATCH_FAILED_CLEAN")
        self.assertEqual(len(self.calls), 1)

    def test_active_stale_and_uncertain_lock_fences(self) -> None:
        (self.paths.intake_root / "model").write_bytes(b"x")
        lock = TransactionLock(self.paths, transaction_id="tx-live", operation="deploy-gguf")
        lock.path = self.paths.deployment_lock
        lock.acquire()
        try:
            result = self.reconcile(waiter=lambda seconds: None)
            self.assertEqual(result["action"], "NOOP_ACTIVE_TRANSACTION")
            self.assertEqual(self.calls, [])
        finally:
            lock.release()
        transaction_id = "tx-stale-automatic"
        (self.paths.transactions / (transaction_id + ".json")).write_text(
            json.dumps({"operation": "deploy-gguf", "state": "PREPARING"}) + "\n",
            encoding="utf-8",
        )
        (self.paths.deployment_lock).write_text(
            json.dumps(
                {
                    "transaction_id": transaction_id,
                    "operation": "deploy-gguf",
                    "pid": 999999,
                    "process_start_identity": "procfs-start-ticks:1",
                }
            ) + "\n",
            encoding="utf-8",
        )
        result = self.reconcile(waiter=lambda seconds: None)
        self.assertEqual(result["action"], "DISPATCH_FIRST_MODEL")
        self.assertFalse(self.paths.deployment_lock.exists())
        self.paths.deployment_lock.write_text("{}\n", encoding="utf-8")
        self.calls.clear()
        result = self.reconcile(waiter=lambda seconds: None)
        self.assertEqual(result["action"], "NOOP_OWNERSHIP_UNCERTAIN")
        self.assertEqual(self.calls, [])
        self.assertTrue(self.paths.deployment_lock.exists())

    def test_result_validator_closes_fields_and_private_data(self) -> None:
        result = self.reconcile(waiter=lambda seconds: None)
        validate_automatic_result(result)
        unknown = dict(result)
        unknown["unexpected"] = True
        with self.assertRaises(InspectorError):
            validate_automatic_result(unknown)
        private = dict(result)
        private["registry_snapshot"] = dict(result["registry_snapshot"])
        private["registry_snapshot"]["pid"] = 12
        private["result_identity"] = "sha256:" + "0" * 64
        with self.assertRaises(InspectorError):
            validate_automatic_result(private)

    def test_machine_parser_adds_zero_argument_operation_and_preserves_deploy(self) -> None:
        parser = build_parser()
        self.assertEqual(parser.parse_args(["reconcile-intake"]).operation, "reconcile-intake")
        with self.assertRaises(InspectorError):
            parser.parse_args(["deploy-gguf"])
        parsed = parser.parse_args(
            [
                "deploy-gguf",
                "--candidate-name",
                "model",
                "--deployment-mode",
                "install-first",
                "--required-capability-profile",
                "CORE_CHAT",
                "--retirement-policy",
                "retain-incumbent",
            ]
        )
        self.assertEqual(parsed.candidate_name, "model")


    def test_converged_retry_is_checked_before_ready_fence_and_requires_basis(self) -> None:
        candidate = self.paths.intake_root / "model"
        candidate.write_bytes(b"stable")
        self.adapter.prestate.update(
            {
                "model_service_state": "READY",
                "ready_model_count": 1,
                "model_rows": 1,
                "active_managed_locations": 1,
                "default_alias": "default",
                "default_target": "sx-gguf-converged-0123456789abcdef",
                "warm_model_id": "sx-gguf-converged-0123456789abcdef",
                "resolved_immutable_model_id": "sx-gguf-converged-0123456789abcdef",
                "artifact_identity": "sha256:" + "4" * 64,
                "artifact_version_id": "bundle-" + "4" * 64,
                "capability_manifest_identity": "sha256:" + "5" * 64,
                "managed_location_identity": "sha256:" + "6" * 64,
            }
        )
        with (
            mock.patch(
                "system_x_inspector.automatic_intake._converged_install_first_retry",
                return_value=True,
            ),
            mock.patch(
                "system_x_inspector.automatic_intake._find_retryable_failed_clean",
                return_value={"transaction_id": "tx-authenticated"},
            ),
        ):
            result = self.reconcile(waiter=lambda seconds: None)
        self.assertEqual(result["action"], "DISPATCH_FIRST_MODEL")
        self.calls.clear()
        with (
            mock.patch(
                "system_x_inspector.automatic_intake._converged_install_first_retry",
                return_value=True,
            ),
            mock.patch(
                "system_x_inspector.automatic_intake._find_retryable_failed_clean",
                return_value=None,
            ),
        ):
            result = self.reconcile(waiter=lambda seconds: None)
        self.assertEqual(result["action"], "NOOP_READY_MODEL_PRESENT")
        self.assertEqual(self.calls, [])

    def test_missing_model_rows_derives_from_validated_ready_count(self) -> None:
        prestate = dict(self.adapter.prestate)
        prestate.pop("model_rows")
        prestate["ready_model_count"] = 2
        snapshot = __import__(
            "system_x_inspector.automatic_intake",
            fromlist=["_registry_snapshot"],
        )._registry_snapshot(prestate)
        self.assertEqual(snapshot["model_rows"], 2)

    def test_explicit_contradictory_model_rows_is_not_hidden_by_fallback(self) -> None:
        prestate = dict(self.adapter.prestate)
        prestate["ready_model_count"] = 0
        prestate["model_rows"] = 1
        with self.assertRaises(InspectorError) as caught:
            derive_first_model_policy(prestate)
        self.assertEqual(
            caught.exception.reason_code,
            "AUTOMATIC_READY_MODEL_PRESENT",
        )

if __name__ == "__main__":
    unittest.main()
