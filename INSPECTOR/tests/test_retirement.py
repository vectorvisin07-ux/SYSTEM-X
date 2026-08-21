"""Focused isolated retirement and recovery acceptance."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest import mock

from system_x_inspector.errors import InspectorError
from system_x_inspector.machine import build_parser
from system_x_inspector.paths import InspectorPaths
from system_x_inspector.retirement import (
    CurrentSourceRetirementAdapter,
    RetirementRequest,
    RetirementTarget,
    managed_location_identity,
    retire_transaction,
    validate_retirement_record,
)


INCUMBENT = "model-incumbent"
TARGET = "model-retirement-target"


class FixtureAdapter:
    """One isolated managed root, SQLite history, and bounded fake ownership."""

    def __init__(
        self,
        case: "RetirementFixture",
        *,
        last_model: bool = False,
        safe_at: str = "L0_OBSERVE",
        ownership_uncertain: bool = False,
        restoration_fails: bool = False,
        activity: tuple[int, int, int] = (0, 0, 0),
        activity_available: bool = True,
    ) -> None:
        self.case = case
        self.target = case.make_target(last_model=last_model)
        if last_model:
            connection = sqlite3.connect(self.case.database)
            try:
                connection.execute(
                    "UPDATE aliases SET model_id=? WHERE alias='default'",
                    (TARGET,),
                )
                connection.execute(
                    "UPDATE models SET state='REMOVED',present=0 "
                    "WHERE model_id=?",
                    (INCUMBENT,),
                )
                connection.commit()
            finally:
                connection.close()
        self.safe_at = safe_at
        self.ownership_uncertain = ownership_uncertain
        self.restoration_fails = restoration_fails
        self.activity = activity
        self.activity_available = activity_available
        self.level = "L0_OBSERVE"
        self.recover_calls: list[tuple[str, int]] = []
        self.clear_calls = 0
        self.quarantine_calls = 0
        self.restore_calls = 0
        self.request_calls = 0
        self.waiting_calls = 0
        self.crash_state: str | None = None
        self.crashed = False

    def resolve_target(
        self, request: RetirementRequest
    ) -> RetirementTarget:
        return self.target

    def activity_snapshot(
        self, target: RetirementTarget
    ) -> dict[str, object]:
        return {
            "available": self.activity_available,
            "active_requests": self.activity[0],
            "active_streams": self.activity[1],
            "nonterminal_operations": self.activity[2],
            "target_public_model_id": target.public_model_id,
        }

    def clear_default(
        self, target: RetirementTarget, transaction_id: str
    ) -> dict[str, object]:
        self.clear_calls += 1
        connection = sqlite3.connect(self.case.database)
        try:
            current = connection.execute(
                "SELECT model_id FROM aliases WHERE alias='default'"
            ).fetchone()
            if current is None:
                return {
                    "action": "clear",
                    "changed": False,
                    "alias_event_identity": "sha256:" + "a" * 64,
                    "new_registry_generation": self.case.generation,
                    "new_target": None,
                    "previous_target": target.public_model_id,
                }
            if current[0] != target.public_model_id:
                raise AssertionError("fixture alias changed")
            connection.execute("DELETE FROM aliases WHERE alias='default'")
            self.case.generation += 1
            connection.execute(
                "INSERT INTO events(kind,subject,generation) VALUES (?,?,?)",
                (
                    "default_alias_cleared",
                    target.public_model_id,
                    self.case.generation,
                ),
            )
            connection.commit()
        finally:
            connection.close()
        return {
            "action": "clear",
            "changed": True,
            "alias_event_identity": "sha256:" + "a" * 64,
            "new_registry_generation": self.case.generation,
            "new_target": None,
            "previous_target": target.public_model_id,
            "transaction_id": transaction_id,
        }

    def on_quarantined(
        self, target: RetirementTarget, quarantine: dict[str, object]
    ) -> None:
        self.quarantine_calls += 1
        connection = sqlite3.connect(self.case.database)
        try:
            connection.execute(
                "UPDATE models SET state='REMOVED',present=0 WHERE model_id=?",
                (target.public_model_id,),
            )
            self.case.generation += 1
            connection.execute(
                "INSERT INTO events(kind,subject,generation) VALUES (?,?,?)",
                (
                    "artifact_location_removed",
                    target.public_model_id,
                    self.case.generation,
                ),
            )
            connection.commit()
        finally:
            connection.close()

    def observe_registry_removal(
        self, target: RetirementTarget
    ) -> dict[str, object]:
        connection = sqlite3.connect(self.case.database)
        try:
            model = connection.execute(
                "SELECT state,present FROM models WHERE model_id=?",
                (target.public_model_id,),
            ).fetchone()
            event = connection.execute(
                "SELECT generation FROM events "
                "WHERE kind='artifact_location_removed' AND subject=? "
                "ORDER BY generation DESC LIMIT 1",
                (target.public_model_id,),
            ).fetchone()
            history = connection.execute(
                "SELECT COUNT(*) FROM history WHERE model_id=?",
                (target.public_model_id,),
            ).fetchone()[0]
        finally:
            connection.close()
        return {
            "observed": model == ("REMOVED", 0) and event is not None,
            "attempts": 1,
            "observed_registry_generation": self.case.generation,
            "registry_removal_event_identity": "sha256:" + "b" * 64,
            "catalogue_target_absent": model == ("REMOVED", 0),
            "immutable_history_present": history == 1,
        }

    def _safe(self, *, last_model: bool) -> dict[str, object]:
        exact = self.level == self.safe_at
        if last_model:
            return {
                "exact": exact,
                "http_status": 200,
                "service_readiness": "WAITING_FOR_MODEL",
                "model_service_state": "WAITING_FOR_MODEL",
                "service_available": True,
                "inference_ready": False,
                "default_target": None,
                "recovery_state": "IDLE" if exact else "RECOVERING",
                "warm": None,
            }
        return {
            "exact": exact,
            "http_status": 200,
            "service_readiness": "READY" if exact else "DEGRADED",
            "model_service_state": "READY" if exact else "MODEL_CHILD_LOST",
            "service_available": exact,
            "inference_ready": exact,
            "default_target": INCUMBENT,
            "recovery_state": "IDLE" if exact else "RECOVERING",
            "warm": (
                {"resolved_public_model_id": INCUMBENT}
                if exact
                else None
            ),
        }

    def observe_service(
        self, target: RetirementTarget, *, last_model: bool
    ) -> dict[str, object]:
        return self._safe(last_model=last_model)

    def recover(
        self,
        level: str,
        target: RetirementTarget,
        *,
        last_model: bool,
        attempt: int,
    ) -> dict[str, object]:
        self.recover_calls.append((level, attempt))
        if self.ownership_uncertain:
            return {
                "level": level,
                "attempt": attempt,
                "used": True,
                "ownership_certain": False,
                "reason_code": "FOREIGN_LISTENER",
            }
        self.level = level
        return {
            "level": level,
            "attempt": attempt,
            "used": True,
            "ownership_certain": True,
            "controller_transaction_id": (
                f"fixture-{level.lower()}-{attempt}"
            ),
        }

    def later_request(
        self, target: RetirementTarget
    ) -> dict[str, object]:
        self.request_calls += 1
        return {
            "passed": True,
            "request_id": "req-fixture-normal",
            "http_status": 200,
            "response_model_match": True,
            "bounded_content_present": True,
            "credential_key_id": "fixture-key-id",
            "reason_code": "OK",
        }

    def waiting_proof(
        self, target: RetirementTarget
    ) -> dict[str, object]:
        self.waiting_calls += 1
        safe = self._safe(last_model=True)
        return {
            "passed": safe["exact"] is True,
            "health_http_status": 200,
            "service_available": True,
            "inference_ready": False,
            "model_service_state": "WAITING_FOR_MODEL",
            "recovery_state": "IDLE",
            "request_id": "req-fixture-waiting",
            "inference_http_status": 503,
            "reason_code": "NO_READY_MODEL",
        }

    def on_restored(self, target: RetirementTarget) -> None:
        self.restore_calls += 1
        connection = sqlite3.connect(self.case.database)
        try:
            connection.execute(
                "UPDATE models SET state='READY',present=1 WHERE model_id=?",
                (target.public_model_id,),
            )
            connection.commit()
        finally:
            connection.close()

    def restore_default(
        self,
        target: RetirementTarget,
        alias_transaction: dict[str, object],
        transaction_id: str,
    ) -> dict[str, object]:
        if self.restoration_fails:
            raise RuntimeError("fixture alias restoration failed")
        connection = sqlite3.connect(self.case.database)
        try:
            connection.execute(
                "INSERT INTO aliases(alias,model_id) VALUES ('default',?)",
                (target.public_model_id,),
            )
            connection.commit()
        finally:
            connection.close()
        return {
            "changed": True,
            "new_target": target.public_model_id,
            "transaction_id": transaction_id,
        }

    def observe_restored(
        self, target: RetirementTarget
    ) -> dict[str, object]:
        return {
            "proved": not self.restoration_fails,
            "service_readiness": "READY",
            "recovery_state": "IDLE",
            "default_target": target.public_model_id,
        }

    def checkpoint(self, state: str) -> None:
        if (
            self.crash_state == state
            and not self.crashed
        ):
            self.crashed = True
            raise KeyboardInterrupt("fixture crash boundary")


class RetirementFixture:
    def __init__(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.system = self.root / "system-x"
        self.inspector = self.system / "INSPECTOR"
        self.branch = self.system / "model-api-gguf"
        for path in (
            self.inspector,
            self.inspector / "RUNTIME" / "locks",
            self.inspector / "RUNTIME" / "status",
            self.inspector / "RUNTIME" / "transactions",
            self.inspector / "RUNTIME" / "results" / "retirement",
            self.branch / "MODEL" / "SUPERMODEL",
            self.branch / "RUNTIME" / "api" / "retirement-staging",
        ):
            path.mkdir(parents=True, exist_ok=True)
            path.chmod(0o700)
        self.paths = InspectorPaths.discover(self.inspector)
        self.target_path = (
            self.branch / "MODEL" / "SUPERMODEL" / "candidate.gguf"
        )
        self.target_bytes = b"GGUF-fixture-retirement-candidate\n"
        self.target_path.write_bytes(self.target_bytes)
        self.target_path.chmod(0o600)
        self.artifact = (
            "bundle-" + hashlib.sha256(self.target_bytes).hexdigest()
        )
        self.content = (
            "sha256:" + hashlib.sha256(self.target_bytes).hexdigest()
        )
        self.database = self.root / "registry.sqlite3"
        self.generation = 7
        connection = sqlite3.connect(self.database)
        try:
            connection.executescript(
                """
                CREATE TABLE models(
                    model_id TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    present INTEGER NOT NULL
                );
                CREATE TABLE aliases(
                    alias TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL
                );
                CREATE TABLE events(
                    kind TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    generation INTEGER NOT NULL
                );
                CREATE TABLE history(
                    model_id TEXT PRIMARY KEY,
                    created TEXT NOT NULL
                );
                """
            )
            for model in (INCUMBENT, TARGET):
                connection.execute(
                    "INSERT INTO models VALUES (?,'READY',1)", (model,)
                )
                connection.execute(
                    "INSERT INTO history VALUES (?,'immutable')", (model,)
                )
            connection.execute(
                "INSERT INTO aliases VALUES ('default',?)", (INCUMBENT,)
            )
            connection.commit()
        finally:
            connection.close()

    def cleanup(self) -> None:
        self.temporary.cleanup()

    def make_target(self, *, last_model: bool) -> RetirementTarget:
        details = self.target_path.lstat()
        default = TARGET if last_model else INCUMBENT
        ready = (TARGET,) if last_model else (INCUMBENT, TARGET)
        replacement = (
            None
            if last_model
            else {
                "public_model_id": INCUMBENT,
                "artifact_version_id": "bundle-" + "1" * 64,
                "state": "READY",
                "relative_root": "incumbent.gguf",
                "capability_manifest_identity": "sha256:" + "2" * 64,
                "present": True,
            }
        )
        warm_model = TARGET if last_model else INCUMBENT
        location = managed_location_identity(
            public_model_id=TARGET,
            artifact_identity=self.artifact,
            relative_root=self.target_path.name,
            device=details.st_dev,
            inode=details.st_ino,
            mode=details.st_mode & 0o777,
            link_count=details.st_nlink,
            size=details.st_size,
        )
        return RetirementTarget(
            public_model_id=TARGET,
            artifact_identity=self.artifact,
            managed_location_identity=location,
            registry_generation=self.generation,
            model_state="READY",
            relative_root=self.target_path.name,
            target_path=self.target_path,
            managed_root=self.target_path.parent,
            quarantine_root=(
                self.branch / "RUNTIME" / "api" / "retirement-staging"
            ),
            device=details.st_dev,
            inode=details.st_ino,
            mode=details.st_mode & 0o777,
            link_count=details.st_nlink,
            size=details.st_size,
            mtime_ns=details.st_mtime_ns,
            authenticated_content_sha256=self.content,
            default_target=default,
            ready_model_ids=ready,
            replacement=replacement,
            service_prestate={
                "http_status": 200,
                "service_readiness": "READY",
                "model_service_state": "READY",
                "service_available": True,
                "inference_ready": True,
                "default_target": default,
                "artifact_version_id": (
                    self.artifact
                    if last_model
                    else replacement["artifact_version_id"]
                ),
                "recovery_state": "IDLE",
                "warm": {
                    "resolved_public_model_id": warm_model,
                    "artifact_version_id": (
                        self.artifact
                        if last_model
                        else replacement["artifact_version_id"]
                    ),
                },
            },
            capability_manifest_identity="sha256:" + "3" * 64,
        )


class RetirementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = RetirementFixture()

    def tearDown(self) -> None:
        self.fixture.cleanup()

    def invoke(
        self,
        adapter: FixtureAdapter,
        *,
        policy: str = "REJECT",
    ):
        target = adapter.target
        return retire_transaction(
            self.fixture.paths,
            public_model_id=target.public_model_id,
            artifact_identity=target.artifact_identity,
            managed_location_identity=target.managed_location_identity,
            expected_registry_generation=target.registry_generation,
            retirement_reason="packet isolated fixture",
            last_model_policy=policy,
            adapter=adapter,
        )

    def test_normal_nondefault_retirement_is_atomic_and_immutable(self) -> None:
        adapter = FixtureAdapter(self.fixture)
        transaction, record, path, identity = self.invoke(adapter)
        self.assertTrue(transaction.startswith("tx-"))
        self.assertEqual(record["result_class"], "RETIREMENT_COMPLETE")
        self.assertEqual(record["result_identity"], identity)
        self.assertFalse(self.fixture.target_path.exists())
        self.assertFalse(
            any(adapter.target.quarantine_root.iterdir())
        )
        self.assertEqual(adapter.request_calls, 1)
        self.assertEqual(adapter.clear_calls, 0)
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        before = path.read_bytes()
        duplicate = self.invoke(adapter)
        self.assertEqual(duplicate[3], identity)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(adapter.quarantine_calls, 1)
        connection = sqlite3.connect(self.fixture.database)
        try:
            self.assertEqual(
                connection.execute(
                    "SELECT state,present FROM models WHERE model_id=?",
                    (TARGET,),
                ).fetchone(),
                ("REMOVED", 0),
            )
            self.assertEqual(
                connection.execute(
                    "SELECT model_id FROM aliases WHERE alias='default'"
                ).fetchone()[0],
                INCUMBENT,
            )
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM history WHERE model_id=?",
                    (TARGET,),
                ).fetchone()[0],
                1,
            )
        finally:
            connection.close()

    def test_default_and_last_model_reject_before_mutation(self) -> None:
        adapter = FixtureAdapter(self.fixture, last_model=True)
        with self.assertRaises(InspectorError) as caught:
            self.invoke(adapter, policy="REJECT")
        self.assertEqual(
            caught.exception.reason_code, "RETIREMENT_TARGET_IS_DEFAULT"
        )
        self.assertTrue(self.fixture.target_path.exists())
        self.assertEqual(adapter.clear_calls, 0)
        self.assertEqual(adapter.quarantine_calls, 0)
        self.assertEqual(
            list(self.fixture.paths.retirement_results.iterdir()), []
        )
        self.assertEqual(
            list(self.fixture.paths.transactions.iterdir()), []
        )

    def test_explicit_last_model_enters_waiting(self) -> None:
        adapter = FixtureAdapter(self.fixture, last_model=True)
        _, record, _, _ = self.invoke(
            adapter, policy="ENTER_WAITING_FOR_MODEL"
        )
        self.assertEqual(
            record["result_class"], "RETIREMENT_WAITING_FOR_MODEL"
        )
        self.assertEqual(
            record["states_observed"][-1], "WAITING_FOR_MODEL"
        )
        self.assertEqual(adapter.clear_calls, 1)
        self.assertEqual(adapter.waiting_calls, 1)
        self.assertEqual(
            record["waiting_for_model_health"]["inference_http_status"],
            503,
        )
        self.assertEqual(
            record["waiting_for_model_health"]["reason_code"],
            "NO_READY_MODEL",
        )
        connection = sqlite3.connect(self.fixture.database)
        try:
            self.assertIsNone(
                connection.execute(
                    "SELECT model_id FROM aliases WHERE alias='default'"
                ).fetchone()
            )
        finally:
            connection.close()

    def test_activity_fence_rejects_every_nonzero_class(self) -> None:
        for counts in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            with self.subTest(counts=counts):
                fixture = RetirementFixture()
                try:
                    adapter = FixtureAdapter(fixture, activity=counts)
                    target = adapter.target
                    with self.assertRaises(InspectorError) as caught:
                        retire_transaction(
                            fixture.paths,
                            public_model_id=target.public_model_id,
                            artifact_identity=target.artifact_identity,
                            managed_location_identity=(
                                target.managed_location_identity
                            ),
                            expected_registry_generation=(
                                target.registry_generation
                            ),
                            retirement_reason="activity rejection",
                            last_model_policy="REJECT",
                            adapter=adapter,
                        )
                    self.assertEqual(
                        caught.exception.reason_code,
                        "RETIREMENT_TARGET_IN_USE",
                    )
                    self.assertTrue(fixture.target_path.exists())
                    self.assertEqual(adapter.quarantine_calls, 0)
                finally:
                    fixture.cleanup()

    def test_activity_unavailable_fails_before_transaction(self) -> None:
        adapter = FixtureAdapter(
            self.fixture, activity_available=False
        )
        with self.assertRaises(InspectorError) as caught:
            self.invoke(adapter)
        self.assertEqual(
            caught.exception.reason_code,
            "RETIREMENT_ACTIVITY_UNAVAILABLE",
        )
        self.assertEqual(
            list(self.fixture.paths.transactions.iterdir()), []
        )

    def test_identity_generation_and_replacement_matrices_reject(self) -> None:
        scenarios = (
            (
                "artifact",
                lambda target: replace(
                    target, artifact_identity="bundle-" + "9" * 64
                ),
                "RETIREMENT_ARTIFACT_IDENTITY_MISMATCH",
            ),
            (
                "location",
                lambda target: replace(
                    target,
                    managed_location_identity="sha256:" + "8" * 64,
                ),
                "RETIREMENT_LOCATION_IDENTITY_MISMATCH",
            ),
            (
                "generation",
                lambda target: replace(
                    target, registry_generation=target.registry_generation + 1
                ),
                "RETIREMENT_GENERATION_MISMATCH",
            ),
            (
                "replacement",
                lambda target: replace(
                    target,
                    replacement={
                        **target.replacement,
                        "state": "UNAVAILABLE",
                    },
                ),
                "RETIREMENT_REPLACEMENT_NOT_READY",
            ),
            (
                "rollback",
                lambda target: replace(
                    target, rollback_dependency=True
                ),
                "RETIREMENT_ROLLBACK_DEPENDENCY",
            ),
        )
        for name, change, expected in scenarios:
            with self.subTest(name=name):
                adapter = FixtureAdapter(self.fixture)
                request_target = adapter.target
                adapter.target = change(adapter.target)
                with self.assertRaises(InspectorError) as caught:
                    retire_transaction(
                        self.fixture.paths,
                        public_model_id=request_target.public_model_id,
                        artifact_identity=request_target.artifact_identity,
                        managed_location_identity=(
                            request_target.managed_location_identity
                        ),
                        expected_registry_generation=(
                            request_target.registry_generation
                        ),
                        retirement_reason="negative identity fixture",
                        last_model_policy="REJECT",
                        adapter=adapter,
                    )
                self.assertEqual(caught.exception.reason_code, expected)
                self.assertTrue(self.fixture.target_path.exists())
                self.assertEqual(adapter.quarantine_calls, 0)

    def test_symlink_hardlink_special_and_outside_targets_reject(self) -> None:
        adapter = FixtureAdapter(self.fixture)
        original = self.fixture.target_path
        original.unlink()
        outside = self.fixture.root / "outside.gguf"
        outside.write_bytes(self.fixture.target_bytes)
        original.symlink_to(outside)
        target = adapter.target
        details = original.lstat()
        adapter.target = replace(
            target,
            device=details.st_dev,
            inode=details.st_ino,
            mode=details.st_mode & 0o777,
            link_count=details.st_nlink,
            size=details.st_size,
            mtime_ns=details.st_mtime_ns,
        )
        with self.assertRaises(InspectorError) as caught:
            self.invoke(adapter)
        self.assertEqual(
            caught.exception.reason_code, "RETIREMENT_TARGET_SYMLINK"
        )

    def test_complete_owned_recovery_matrix(self) -> None:
        matrix = (
            ("L0_OBSERVE", False, False, False, "RETIREMENT_COMPLETE"),
            (
                "L1_EXACT_MODEL_CHILD_RECONCILE",
                False,
                False,
                False,
                "RETIREMENT_COMPLETE",
            ),
            (
                "L2_ROUTER_DEFAULT_RELOAD",
                False,
                False,
                False,
                "RETIREMENT_COMPLETE",
            ),
            (
                "L3_CONTROLLER_STACK_RECOVERY",
                False,
                False,
                False,
                "RETIREMENT_COMPLETE",
            ),
            (
                "L4_PLATFORM_MANAGER_RESTART",
                False,
                False,
                False,
                "RETIREMENT_COMPLETE",
            ),
            (
                "L4_PLATFORM_MANAGER_RESTART",
                True,
                False,
                False,
                "RETIREMENT_WAITING_FOR_MODEL",
            ),
            (
                "NEVER",
                False,
                True,
                False,
                "RETIREMENT_FAIL_CLOSED",
            ),
            (
                "NEVER",
                False,
                False,
                True,
                "RETIREMENT_FAIL_CLOSED",
            ),
        )
        for index, (
            safe_at,
            last_model,
            uncertain,
            restoration_fails,
            expected,
        ) in enumerate(matrix, 1):
            with self.subTest(case=index):
                fixture = RetirementFixture()
                try:
                    adapter = FixtureAdapter(
                        fixture,
                        last_model=last_model,
                        safe_at=safe_at,
                        ownership_uncertain=uncertain,
                        restoration_fails=restoration_fails,
                    )
                    target = adapter.target
                    arguments = {
                        "paths": fixture.paths,
                        "public_model_id": target.public_model_id,
                        "artifact_identity": target.artifact_identity,
                        "managed_location_identity": (
                            target.managed_location_identity
                        ),
                        "expected_registry_generation": (
                            target.registry_generation
                        ),
                        "retirement_reason": f"recovery matrix {index}",
                        "last_model_policy": (
                            "ENTER_WAITING_FOR_MODEL"
                            if last_model
                            else "REJECT"
                        ),
                        "adapter": adapter,
                    }
                    if expected == "RETIREMENT_FAIL_CLOSED":
                        with self.assertRaises(InspectorError):
                            retire_transaction(**arguments)
                        result_files = list(
                            fixture.paths.retirement_results.glob("*.json")
                        )
                        self.assertEqual(len(result_files), 1)
                        record = validate_retirement_record(
                            json.loads(result_files[0].read_text())
                        )
                    else:
                        record = retire_transaction(**arguments)[1]
                    self.assertEqual(record["result_class"], expected)
                    if safe_at != "L0_OBSERVE" and not uncertain:
                        reached = [
                            level for level, _ in adapter.recover_calls
                        ]
                        self.assertIn(
                            "L1_EXACT_MODEL_CHILD_RECONCILE", reached
                        )
                    self.assertLessEqual(
                        len(adapter.recover_calls), 5
                    )
                finally:
                    fixture.cleanup()

    def test_crash_reentry_and_completed_duplicate_do_not_repeat_move(self) -> None:
        adapter = FixtureAdapter(self.fixture)
        adapter.crash_state = "QUARANTINED"
        with self.assertRaises(KeyboardInterrupt):
            self.invoke(adapter)
        self.assertEqual(adapter.quarantine_calls, 1)
        self.assertFalse(self.fixture.target_path.exists())
        result = self.invoke(adapter)
        self.assertEqual(
            result[1]["result_class"], "RETIREMENT_COMPLETE"
        )
        self.assertEqual(adapter.quarantine_calls, 1)
        identity = result[3]
        duplicate = self.invoke(adapter)
        self.assertEqual(duplicate[3], identity)
        self.assertEqual(adapter.quarantine_calls, 1)
        self.assertEqual(adapter.request_calls, 1)

    def test_result_privacy_identity_and_schema(self) -> None:
        adapter = FixtureAdapter(self.fixture)
        _, record, _, _ = self.invoke(adapter)
        validate_retirement_record(record)
        schema = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "schemas"
                / "gguf-retirement-result.schema.json"
            ).read_text()
        )
        self.assertEqual(
            schema["$id"], "system-x.inspector-gguf-retirement-result.v1"
        )
        self.assertEqual(set(record), set(schema["required"]))
        serialized = json.dumps(record, sort_keys=True)
        for prohibited in (
            str(self.fixture.target_path),
            str(adapter.target.quarantine_root),
            "raw_api_key",
            "credential_verifier",
            "private_router_url",
            "process_environment",
        ):
            self.assertNotIn(prohibited, serialized)

    def test_machine_input_has_no_caller_path_surface(self) -> None:
        parser = build_parser()
        parsed = parser.parse_args(
            [
                "retire-gguf",
                "--public-model-id",
                TARGET,
                "--artifact-identity",
                "bundle-" + "1" * 64,
                "--managed-location-identity",
                "sha256:" + "2" * 64,
                "--expected-registry-generation",
                "7",
                "--retirement-reason",
                "operator retirement",
            ]
        )
        self.assertEqual(parsed.last_model_policy, "REJECT")
        with self.assertRaises(InspectorError):
            parser.parse_args(
                [
                    "retire-gguf",
                    "--public-model-id",
                    TARGET,
                    "--artifact-identity",
                    "bundle-" + "1" * 64,
                    "--managed-location-identity",
                    "sha256:" + "2" * 64,
                    "--expected-registry-generation",
                    "7",
                    "--retirement-reason",
                    "operator retirement",
                    "--absolute-path",
                    "/caller/model.gguf",
                ]
            )


class RestoredRegistryProofTests(unittest.TestCase):
    def test_restored_registry_proof_rejects_removed_and_accepts_coherent_present(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "registry.sqlite3"
            connection = sqlite3.connect(database)
            try:
                connection.executescript(
                    """
                    CREATE TABLE model_versions(
                        model_version_id TEXT PRIMARY KEY,
                        state TEXT NOT NULL,
                        bundle_id TEXT NOT NULL
                    );
                    CREATE TABLE model_version_locations(
                        model_version_id TEXT NOT NULL,
                        relative_root TEXT NOT NULL
                    );
                    CREATE TABLE artifact_locations(
                        relative_root TEXT PRIMARY KEY,
                        present INTEGER NOT NULL,
                        current_bundle_id TEXT NOT NULL
                    );
                    """
                )
                connection.execute(
                    "INSERT INTO model_versions VALUES (?,?,?)",
                    ("restored-model", "READY", "bundle-restored"),
                )
                connection.execute(
                    "INSERT INTO model_version_locations VALUES (?,?)",
                    ("restored-model", "restored.gguf"),
                )
                connection.execute(
                    "INSERT INTO artifact_locations VALUES (?,?,?)",
                    ("restored.gguf", 1, "bundle-restored"),
                )
                connection.commit()
            finally:
                connection.close()
            adapter = object.__new__(CurrentSourceRetirementAdapter)
            adapter.database = database
            target = type(
                "Target",
                (),
                {
                    "public_model_id": "restored-model",
                    "artifact_identity": "bundle-restored",
                    "relative_root": "restored.gguf",
                },
            )()
            self.assertTrue(
                adapter._restored_registry_observation(target)["proved"]
            )
            connection = sqlite3.connect(database)
            try:
                connection.execute(
                    "UPDATE model_versions SET state='REMOVED' "
                    "WHERE model_version_id='restored-model'"
                )
                connection.execute(
                    "UPDATE artifact_locations SET present=0 "
                    "WHERE relative_root='restored.gguf'"
                )
                connection.commit()
            finally:
                connection.close()
            self.assertFalse(
                adapter._restored_registry_observation(target)["proved"]
            )

    def test_restoration_reconcile_uses_manager_for_removed_registry(self) -> None:
        adapter = object.__new__(CurrentSourceRetirementAdapter)
        adapter.branch = type("Branch", (), {"branch_root": Path("/r4-branch")})()
        target = object()
        with (
            mock.patch.object(
                adapter,
                "_restored_registry_observation",
                return_value={"proved": False},
            ),
            mock.patch(
                "system_x_inspector.retirement.recover_with_accepted_platform_manager",
                return_value={"used": True},
            ) as manager,
        ):
            adapter.on_restored(target)
        manager.assert_called_once_with(Path("/r4-branch"))


if __name__ == "__main__":
    unittest.main()
