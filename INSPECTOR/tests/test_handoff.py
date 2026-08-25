from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import os
import shutil
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from system_x_inspector.constants import SCHEMA_IDENTITIES
from system_x_inspector.errors import InspectorError
from system_x_inspector.handoff import (
    DecisionAuthorization,
    SourceEvidence,
    _load_authenticated_decision,
    _load_linked_inspection,
    _handoff_transaction_value,
    authenticate_handoff_decision,
    create_staged_artifact,
    finalize_handoff_record,
    handoff_transaction,
    prepare_handoff_destination,
    publish_staged_artifact,
    publish_handoff_record,
    revalidate_handoff_source,
    storage_preflight,
    validate_handoff_record,
)
from system_x_inspector.machine import main
from system_x_inspector.paths import BranchHandoffPaths, InspectorPaths
from system_x_inspector.records import atomic_write_json
from system_x_inspector.records import canonical_json_bytes
from system_x_inspector.records import read_json_record
from system_x_inspector.results import utc_now


IDENTITY_A = "sha256:" + "a" * 64
IDENTITY_B = "sha256:" + "b" * 64
IDENTITY_C = "sha256:" + "c" * 64


class HandoffFoundationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = Path(
            tempfile.mkdtemp(prefix="inspector-handoff-", dir="/tmp")
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
        self.branch = self.temporary / "model-api-gguf"
        self.managed = self.branch / "MODEL" / "SUPERMODEL"
        self.branch_staging = (
            self.branch / "RUNTIME" / "api" / "replacement-staging"
        )
        self.managed.mkdir(mode=0o700, parents=True)
        self.branch_staging.mkdir(mode=0o700, parents=True)
        atomic_write_json(
            self.paths.status / "current.json",
            {
                "schema_version": SCHEMA_IDENTITIES["status"],
                "state": "IDLE",
                "reason_code": "OK",
                "updated_utc": utc_now(),
                "inspector_root": str(self.paths.inspector_root),
                "active_transaction_id": None,
                "last_transaction_id": None,
            },
            mode=0o600,
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.temporary)

    def valid_record(self) -> dict[str, object]:
        return finalize_handoff_record(
            {
                "schema_version": SCHEMA_IDENTITIES["handoff_result"],
                "handoff_id": (
                    "handoff-20260101T000000000000Z-"
                    "0123456789abcdef"
                ),
                "transaction_id": "tx-fixture",
                "created_utc": "2026-01-01T00:00:00Z",
                "completed_utc": "2026-01-01T00:00:01Z",
                "status": "PUBLISHED_TO_BRANCH",
                "decision": {
                    "decision_id": "decision-fixture",
                    "result_identity": IDENTITY_A,
                    "decision_basis_identity": IDENTITY_B,
                    "capability_result": "SUPPORTED",
                    "selected_branch": "model-api-gguf",
                    "handoff_allowed": True,
                    "spawn_allowed": True,
                },
                "inspection": {
                    "inspection_id": "inspection-fixture",
                    "result_identity": IDENTITY_B,
                    "physical_format": "GGUF",
                    "artifact_identity": IDENTITY_C,
                    "artifact_size": 7,
                },
                "capability": {
                    "record_id": "capability-fixture",
                    "record_identity": IDENTITY_A,
                    "binding_identity": IDENTITY_B,
                    "binding_generation": 1,
                    "installed_tuple_verified": True,
                },
                "source": {
                    "intake_root_identity": IDENTITY_A,
                    "relative_name": "accepted.gguf",
                    "device": 1,
                    "inode": 2,
                    "mode": "0644",
                    "link_count": 1,
                    "size_bytes": 7,
                    "sha256": IDENTITY_C,
                    "pre_copy_identity": IDENTITY_B,
                    "post_copy_identity": IDENTITY_B,
                    "unchanged_during_handoff": True,
                },
                "staging": {
                    "branch_relative_path": (
                        "RUNTIME/api/replacement-staging/"
                        ".handoff-fixture.staging"
                    ),
                    "transfer_method": "bounded_stream_copy",
                    "device": 1,
                    "inode": 3,
                    "mode": "0644",
                    "link_count": 1,
                    "size_bytes": 7,
                    "sha256": IDENTITY_C,
                    "complete_write": True,
                    "file_fsync": True,
                    "directory_fsync": True,
                },
                "publication": {
                    "managed_relative_path": (
                        "MODEL/SUPERMODEL/accepted-"
                        "cccccccccccc.gguf"
                    ),
                    "method": (
                        "same_filesystem_atomic_no_overwrite_rename"
                    ),
                    "device": 1,
                    "inode": 3,
                    "mode": "0644",
                    "link_count": 1,
                    "size_bytes": 7,
                    "sha256": IDENTITY_C,
                    "staging_inode_preserved": True,
                    "parent_directory_fsync": True,
                    "collision_absent": True,
                },
                "identity_match": {
                    "source_equals_decision": True,
                    "staged_equals_source": True,
                    "published_equals_staged": True,
                },
                "registry_observation": {
                    "state_at_handoff_completion": (
                        "DELEGATED_NOT_OBSERVED"
                    ),
                    "readiness_claimed": False,
                    "next_mini_required": True,
                },
                "alias_protection": {
                    "default_alias": "default",
                    "default_target_before": "production.gguf",
                    "default_target_after": "production.gguf",
                    "unchanged": True,
                },
                "runtime_protection": {
                    "service_was_running": True,
                    "service_remained_running": True,
                    "lifecycle_operation_count": 0,
                },
                "cleanup_ownership": {
                    "source": "caller_or_packet_owned",
                    "staging": "handoff_transaction_owned",
                    "published_artifact": "branch_owned_after_success",
                },
            }
        )

    def run_main(self, *arguments: str) -> tuple[int, dict[str, object]]:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = main(
                ["--inspector-root", str(self.root), *arguments]
            )
        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 1)
        return status, json.loads(lines[0])

    def test_schema_is_closed_and_record_identity_is_immutable(self) -> None:
        schema = json.loads(
            (
                Path(__file__).resolve().parent.parent
                / "schemas"
                / "handoff-result.schema.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(schema["$id"], SCHEMA_IDENTITIES["handoff_result"])
        self.assertFalse(schema["additionalProperties"])
        self.assertTrue(
            all(
                definition.get("additionalProperties") is False
                for definition in schema["$defs"].values()
                if definition.get("type") == "object"
            )
        )
        record = self.valid_record()
        self.assertEqual(validate_handoff_record(record), record)
        changed = dict(record)
        changed["completed_utc"] = "2026-01-01T00:00:02Z"
        with self.assertRaises(InspectorError) as caught:
            validate_handoff_record(changed)
        self.assertEqual(
            caught.exception.reason_code, "HANDOFF_RESULT_COLLISION"
        )
        cold = self.valid_record()
        cold["alias_protection"] = {
            **cold["alias_protection"],
            "default_target_before": None,
            "default_target_after": None,
        }
        cold.pop("result_identity")
        cold = finalize_handoff_record(cold)
        self.assertEqual(validate_handoff_record(cold), cold)

    def test_result_publication_is_private_and_no_overwrite(self) -> None:
        record = self.valid_record()
        path, identity = publish_handoff_record(self.paths, record)
        self.assertEqual(identity, record["result_identity"])
        details = path.lstat()
        self.assertTrue(stat.S_ISREG(details.st_mode))
        self.assertFalse(path.is_symlink())
        self.assertEqual(stat.S_IMODE(details.st_mode), 0o600)
        self.assertEqual(details.st_nlink, 1)
        repeated, repeated_identity = publish_handoff_record(
            self.paths, record
        )
        self.assertEqual(repeated, path)
        self.assertEqual(repeated_identity, identity)
        changed = dict(record)
        changed["completed_utc"] = "2026-01-01T00:00:03Z"
        changed = finalize_handoff_record(changed)
        with self.assertRaises(InspectorError):
            publish_handoff_record(self.paths, changed)

    def test_branch_paths_are_self_relative_and_same_filesystem(self) -> None:
        paths = BranchHandoffPaths.discover(self.paths)
        self.assertEqual(paths.branch_root, self.branch)
        self.assertEqual(paths.managed_root, self.managed)
        self.assertEqual(paths.branch_staging_root, self.branch_staging)
        self.assertEqual(
            paths.relative_to_branch(self.managed / "fixture.gguf"),
            "MODEL/SUPERMODEL/fixture.gguf",
        )
        elsewhere = self.temporary / "elsewhere"
        elsewhere.mkdir()
        with self.assertRaises(InspectorError):
            BranchHandoffPaths.discover(self.paths, elsewhere)

    def test_machine_handoff_success_rejection_and_internal_exits(self) -> None:
        record = self.valid_record()
        result_path = (
            self.paths.handoff_results / f"{record['handoff_id']}.json"
        )
        returned = (
            "tx-fixture",
            record,
            result_path,
            record["result_identity"],
        )
        arguments = (
            "handoff",
            "--decision-id",
            "decision-fixture",
            "--source-candidate",
            "accepted.gguf",
            "--managed-name",
            "accepted-cccccccccccc.gguf",
        )
        with mock.patch(
            "system_x_inspector.machine.handoff_transaction",
            return_value=returned,
        ):
            status, result = self.run_main(*arguments)
        self.assertEqual(status, 0)
        self.assertTrue(result["ok"])
        self.assertEqual(result["reason_code"], "HANDOFF_COMPLETE")

        with mock.patch(
            "system_x_inspector.machine.handoff_transaction",
            side_effect=InspectorError(
                "HANDOFF_TARGET_NAME_INVALID", "fixture rejection"
            ),
        ):
            status, result = self.run_main(*arguments)
        self.assertEqual(status, 2)
        self.assertFalse(result["ok"])
        self.assertEqual(
            result["reason_code"], "HANDOFF_TARGET_NAME_INVALID"
        )

        with mock.patch(
            "system_x_inspector.machine.handoff_transaction",
            side_effect=InspectorError(
                "HANDOFF_INTERNAL_ERROR",
                "fixture internal",
                exit_status=70,
            ),
        ):
            status, result = self.run_main(*arguments)
        self.assertEqual(status, 70)
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason_code"], "HANDOFF_INTERNAL_ERROR")

    def test_machine_parser_rejects_unbounded_inputs(self) -> None:
        status, result = self.run_main(
            "handoff",
            "--decision-id",
            "decision-fixture",
            "--source-candidate",
            "accepted.gguf",
            "--managed-name",
            "accepted-cccccccccccc.gguf",
            "--branch-root",
            str(self.branch),
        )
        self.assertEqual(status, 2)
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason_code"], "CONFIG_INVALID")

    def authorization_surfaces(
        self,
        *,
        capability_result: str | None = "SUPPORTED",
        physical_format: str = "GGUF",
        evaluated: bool = True,
    ) -> tuple[
        dict[str, object],
        dict[str, object],
        dict[str, object],
        dict[str, object],
    ]:
        artifact = "sha256:" + hashlib.sha256(
            b"accepted-artifact"
        ).hexdigest()
        branch = "model-api-gguf" if evaluated else None
        decision = {
            "decision_id": (
                "decision-20260101T000000000000Z-0123456789abcdef"
            ),
            "result_identity": IDENTITY_A,
            "decision_basis_identity": IDENTITY_B,
            "inspection": {
                "inspection_id": (
                    "inspection-20260101T000000000000Z-"
                    "0123456789abcdef"
                ),
                "inspection_result_identity": IDENTITY_B,
                "artifact_identity": artifact,
                "physical_format": physical_format,
                "source_target_name": "accepted.gguf",
            },
            "capability": {
                "branch_identity": branch,
                "binding_identity": IDENTITY_A if evaluated else None,
                "capability_record_id": (
                    "capability-fixture" if evaluated else None
                ),
                "capability_record_identity": (
                    IDENTITY_B if evaluated else None
                ),
                "capability_result": capability_result,
                "evaluated": evaluated,
            },
            "selected_branch": (
                "model-api-gguf"
                if capability_result == "SUPPORTED"
                else None
            ),
            "handoff_allowed": capability_result == "SUPPORTED",
            "spawn_allowed": capability_result == "SUPPORTED",
        }
        inspection = {
            "inspection_id": decision["inspection"]["inspection_id"],
            "artifact": {
                "identity": artifact,
                "byte_count": len(b"accepted-artifact"),
            },
            "classification": {"terminal_class": physical_format},
            "source": {"candidate_name": "accepted.gguf"},
        }
        capability = {
            "capability_record_id": "capability-fixture",
            "capability_record_identity": IDENTITY_B,
            "branch_identity": "model-api-gguf",
            "supported_physical_format": "GGUF",
            "supported_evidence": {
                "supported_exact_artifact_identities": [artifact]
            },
        }
        binding = {
            "binding_identity": IDENTITY_A,
            "binding_generation": 1,
            "capability_record_id": "capability-fixture",
            "capability_record_identity": IDENTITY_B,
            "branch_identity": "model-api-gguf",
        }
        return decision, inspection, capability, binding

    def authenticate_surfaces(
        self,
        decision: dict[str, object],
        inspection: dict[str, object],
        capability: dict[str, object],
        binding: dict[str, object],
    ) -> DecisionAuthorization:
        with (
            mock.patch(
                "system_x_inspector.handoff._load_authenticated_decision",
                return_value=decision,
            ),
            mock.patch(
                "system_x_inspector.handoff._load_linked_inspection",
                return_value=(inspection, IDENTITY_B),
            ),
            mock.patch(
                "system_x_inspector.handoff._load_linked_capability",
                return_value=(
                    capability,
                    binding,
                    {"applicable": True, "verified": True, "mismatches": []},
                ),
            ),
        ):
            return authenticate_handoff_decision(
                self.paths, str(decision["decision_id"])
            )

    def test_supported_authorization_and_all_no_spawn_classes(self) -> None:
        decision, inspection, capability, binding = (
            self.authorization_surfaces()
        )
        authorized = self.authenticate_surfaces(
            decision, inspection, capability, binding
        )
        self.assertEqual(
            authorized.decision["capability"]["capability_result"],
            "SUPPORTED",
        )
        cases = [
            ("RUNTIME_SMOKE_REQUIRED", "GGUF", True),
            ("UNSUPPORTED", "GGUF", True),
            ("UNAVAILABLE", "NATIVE", True),
            (None, "UNKNOWN", False),
            (None, "CONTRADICTORY", False),
            (None, "CORRUPT", False),
            (None, "INCOMPLETE", False),
        ]
        before_staging = list(self.branch_staging.iterdir())
        before_managed = list(self.managed.iterdir())
        before_results = list(self.paths.handoff_results.iterdir())
        for result, physical, evaluated in cases:
            with self.subTest(result=result, physical=physical):
                surfaces = self.authorization_surfaces(
                    capability_result=result,
                    physical_format=physical,
                    evaluated=evaluated,
                )
                with self.assertRaises(InspectorError) as caught:
                    self.authenticate_surfaces(*surfaces)
                self.assertEqual(
                    caught.exception.reason_code,
                    "HANDOFF_DECISION_NOT_SUPPORTED",
                )
        self.assertEqual(list(self.branch_staging.iterdir()), before_staging)
        self.assertEqual(list(self.managed.iterdir()), before_managed)
        self.assertEqual(
            list(self.paths.handoff_results.iterdir()), before_results
        )

    def test_binding_and_installed_tuple_staleness_fail_closed(self) -> None:
        decision, inspection, capability, binding = (
            self.authorization_surfaces()
        )
        with (
            mock.patch(
                "system_x_inspector.handoff._load_authenticated_decision",
                return_value=decision,
            ),
            mock.patch(
                "system_x_inspector.handoff._load_linked_inspection",
                return_value=(inspection, IDENTITY_B),
            ),
            mock.patch(
                "system_x_inspector.handoff.load_binding",
                return_value={**binding, "binding_identity": IDENTITY_C},
            ),
            mock.patch(
                "system_x_inspector.handoff.load_capability_record",
                return_value=capability,
            ),
        ):
            with self.assertRaises(InspectorError) as caught:
                authenticate_handoff_decision(
                    self.paths, str(decision["decision_id"])
                )
        self.assertEqual(
            caught.exception.reason_code,
            "HANDOFF_CAPABILITY_BINDING_INVALID",
        )

        with (
            mock.patch(
                "system_x_inspector.handoff._load_authenticated_decision",
                return_value=decision,
            ),
            mock.patch(
                "system_x_inspector.handoff._load_linked_inspection",
                return_value=(inspection, IDENTITY_B),
            ),
            mock.patch(
                "system_x_inspector.handoff.load_binding",
                return_value=binding,
            ),
            mock.patch(
                "system_x_inspector.handoff.load_capability_record",
                return_value=capability,
            ),
            mock.patch(
                "system_x_inspector.handoff.verify_installed_tuple",
                return_value={
                    "applicable": True,
                    "verified": False,
                    "mismatches": [{"field": "fixture"}],
                },
            ),
        ):
            with self.assertRaises(InspectorError) as caught:
                authenticate_handoff_decision(
                    self.paths, str(decision["decision_id"])
                )
        self.assertEqual(
            caught.exception.reason_code,
            "HANDOFF_INSTALLED_TUPLE_MISMATCH",
        )

    def test_source_revalidation_matrix_and_mutation_detection(self) -> None:
        decision, inspection, capability, binding = (
            self.authorization_surfaces()
        )
        authorization = self.authenticate_surfaces(
            decision, inspection, capability, binding
        )
        branch_paths = BranchHandoffPaths.discover(self.paths)
        source = self.paths.intake_root / "accepted.gguf"

        with self.assertRaises(InspectorError) as caught:
            revalidate_handoff_source(
                self.paths, branch_paths, authorization, source.name
            )
        self.assertEqual(
            caught.exception.reason_code, "HANDOFF_SOURCE_NOT_FOUND"
        )

        outside = self.temporary / "outside.gguf"
        outside.write_bytes(b"accepted-artifact")
        source.symlink_to(outside)
        with self.assertRaises(InspectorError) as caught:
            revalidate_handoff_source(
                self.paths, branch_paths, authorization, source.name
            )
        self.assertEqual(
            caught.exception.reason_code, "HANDOFF_SOURCE_SYMLINK"
        )
        source.unlink()

        production = self.managed / "production.gguf"
        production.write_bytes(b"accepted-artifact")
        os.link(production, source)
        with self.assertRaises(InspectorError) as caught:
            revalidate_handoff_source(
                self.paths, branch_paths, authorization, source.name
            )
        self.assertEqual(
            caught.exception.reason_code,
            "HANDOFF_SOURCE_HARDLINK_REJECTED",
        )
        source.unlink()
        production.unlink()

        source.write_bytes(b"rejected-artifact")
        with self.assertRaises(InspectorError) as caught:
            revalidate_handoff_source(
                self.paths, branch_paths, authorization, source.name
            )
        self.assertEqual(
            caught.exception.reason_code,
            "HANDOFF_SOURCE_IDENTITY_MISMATCH",
        )
        source.write_bytes(b"accepted-artifact")
        evidence = revalidate_handoff_source(
            self.paths, branch_paths, authorization, source.name
        )
        self.assertEqual(
            evidence.artifact_identity,
            decision["inspection"]["artifact_identity"],
        )

        mutated = False

        def mutate(_: int, path: Path) -> None:
            nonlocal mutated
            if not mutated:
                path.write_bytes(b"accepted-artifacT")
                mutated = True

        with self.assertRaises(InspectorError) as caught:
            revalidate_handoff_source(
                self.paths,
                branch_paths,
                authorization,
                source.name,
                read_observer=mutate,
            )
        self.assertEqual(
            caught.exception.reason_code, "HANDOFF_SOURCE_CHANGED"
        )

        with self.assertRaises(InspectorError) as caught:
            revalidate_handoff_source(
                self.paths, branch_paths, authorization, "../outside.gguf"
            )
        self.assertEqual(
            caught.exception.reason_code, "HANDOFF_SOURCE_INVALID"
        )

    def test_tampered_decision_is_rejected_before_source_access(self) -> None:
        decision_id = (
            "decision-20260101T000000000000Z-0123456789abcdef"
        )
        decision_path = self.paths.decision_results / f"{decision_id}.json"
        decision_path.write_text('{"tampered":true}\\n', encoding="utf-8")
        os.chmod(decision_path, 0o600)
        with self.assertRaises(InspectorError) as caught:
            _load_authenticated_decision(self.paths, decision_id)
        self.assertEqual(
            caught.exception.reason_code, "HANDOFF_DECISION_INVALID"
        )
        self.assertFalse(any(self.branch_staging.iterdir()))

    def transfer_fixture(
        self,
        *,
        transaction_id: str = "tx-transfer-fixture",
        prefix: str = "accepted",
    ) -> tuple[
        BranchHandoffPaths,
        DecisionAuthorization,
        SourceEvidence,
        object,
    ]:
        production = self.managed / "production.gguf"
        production.write_bytes(b"production")
        os.chmod(production, 0o644)
        decision, inspection, capability, binding = (
            self.authorization_surfaces()
        )
        authorization = self.authenticate_surfaces(
            decision, inspection, capability, binding
        )
        source = self.paths.intake_root / "accepted.gguf"
        source.write_bytes(b"accepted-artifact")
        os.chmod(source, 0o644)
        branch_paths = BranchHandoffPaths.discover(self.paths)
        evidence = revalidate_handoff_source(
            self.paths, branch_paths, authorization, source.name
        )
        digest = evidence.artifact_identity.removeprefix("sha256:")
        plan = prepare_handoff_destination(
            branch_paths,
            transaction_id=transaction_id,
            managed_name=f"{prefix}-{digest[:12]}.gguf",
            artifact_identity=evidence.artifact_identity,
        )
        return branch_paths, authorization, evidence, plan

    def test_cold_install_derives_policy_from_authenticated_root(
        self,
    ) -> None:
        branch_paths = BranchHandoffPaths.discover(self.paths)
        root_details = branch_paths.managed_root.lstat()
        digest = "a" * 64
        plan = prepare_handoff_destination(
            branch_paths,
            transaction_id="tx-cold-install",
            managed_name=f"cold-install-{digest[:12]}.gguf",
            artifact_identity=f"sha256:{digest}",
        )
        self.assertEqual(plan.policy.mode, 0o640)
        self.assertEqual(
            plan.policy.owner_uid,
            root_details.st_uid,
        )
        self.assertEqual(
            plan.policy.owner_gid,
            root_details.st_gid,
        )
        self.assertEqual(plan.policy.reference_names, ())
        self.assertFalse(plan.managed_target.exists())
        self.assertFalse(plan.staging_path.exists())

    def test_streaming_staging_and_atomic_publication(self) -> None:
        branch_paths, _, source, plan = self.transfer_fixture()
        sentinel = branch_paths.branch_staging_root / "preserve.sentinel"
        sentinel.write_bytes(b"preserve")
        staged = create_staged_artifact(
            plan,
            source,
            safety_margin_bytes=0,
            reflink_cloner=lambda _source, _staging: False,
        )
        self.assertEqual(staged.transfer_method, "bounded_stream_copy")
        self.assertEqual(staged.sha256, source.artifact_identity)
        self.assertNotEqual(staged.inode, source.snapshot["inode"])
        self.assertEqual(staged.link_count, 1)
        published = publish_staged_artifact(plan, staged)
        self.assertEqual(published.inode, staged.inode)
        self.assertEqual(published.sha256, source.artifact_identity)
        self.assertFalse(plan.staging_path.exists())
        self.assertTrue(plan.managed_target.exists())
        self.assertEqual(plan.managed_target.read_bytes(), b"accepted-artifact")
        self.assertEqual(sentinel.read_bytes(), b"preserve")

    def test_reflink_when_available_and_cross_filesystem_staging(self) -> None:
        _, _, source, plan = self.transfer_fixture(
            transaction_id="tx-reflink-fixture", prefix="reflink"
        )
        staged = create_staged_artifact(
            plan, source, safety_margin_bytes=0
        )
        self.assertIn(
            staged.transfer_method,
            {"reflink_clone", "bounded_stream_copy"},
        )
        self.assertNotEqual(staged.inode, source.snapshot["inode"])
        self.assertEqual(staged.sha256, source.artifact_identity)
        self.assertEqual(staged.link_count, 1)
        published = publish_staged_artifact(plan, staged)
        self.assertEqual(published.inode, staged.inode)

        shared_memory = Path("/dev/shm")
        if not shared_memory.is_dir():
            return
        cross_root = Path(
            tempfile.mkdtemp(prefix="handoff-cross-source-", dir=shared_memory)
        )
        try:
            cross_source = cross_root / "cross.gguf"
            cross_source.write_bytes(b"cross-filesystem")
            details = cross_source.lstat()
            snapshot = {
                "device": details.st_dev,
                "inode": details.st_ino,
                "mode": stat.S_IMODE(details.st_mode),
                "link_count": details.st_nlink,
                "size_bytes": details.st_size,
                "mtime_ns": details.st_mtime_ns,
                "ctime_ns": details.st_ctime_ns,
            }
            identity = "sha256:" + hashlib.sha256(
                cross_source.read_bytes()
            ).hexdigest()
            evidence = SourceEvidence(
                path=cross_source,
                intake_root_identity=IDENTITY_A,
                relative_name=cross_source.name,
                snapshot=snapshot,
                snapshot_identity=(
                    "sha256:"
                    + hashlib.sha256(
                        canonical_json_bytes(snapshot)
                    ).hexdigest()
                ),
                artifact_identity=identity,
            )
            digest = identity.removeprefix("sha256:")
            cross_plan = prepare_handoff_destination(
                plan.branch_paths,
                transaction_id="tx-cross-filesystem",
                managed_name=f"cross-{digest[:12]}.gguf",
                artifact_identity=identity,
            )
            cross_staged = create_staged_artifact(
                cross_plan,
                evidence,
                safety_margin_bytes=0,
                reflink_cloner=lambda _source, _staging: False,
            )
            self.assertNotEqual(
                cross_source.stat().st_dev, cross_staged.device
            )
            self.assertEqual(
                cross_staged.device,
                cross_plan.branch_paths.managed_root.stat().st_dev,
            )
            cross_published = publish_staged_artifact(
                cross_plan, cross_staged
            )
            self.assertEqual(cross_published.sha256, identity)
        finally:
            shutil.rmtree(cross_root)

    def test_storage_copy_and_staged_failure_matrix(self) -> None:
        _, _, source, plan = self.transfer_fixture(
            transaction_id="tx-storage-fixture", prefix="storage"
        )

        class NoSpace:
            f_bavail = 1
            f_frsize = 1

        with self.assertRaises(InspectorError) as caught:
            storage_preflight(
                plan.branch_paths.branch_staging_root,
                source.snapshot["size_bytes"],
                safety_margin_bytes=0,
                statvfs_reader=lambda _path: NoSpace(),
            )
        self.assertEqual(
            caught.exception.reason_code,
            "HANDOFF_INSUFFICIENT_STORAGE",
        )
        self.assertFalse(plan.staging_path.exists())

        with self.assertRaises(InspectorError) as caught:
            create_staged_artifact(
                plan,
                source,
                safety_margin_bytes=0,
                reflink_cloner=lambda _source, _staging: False,
                writer=lambda _descriptor, _data: 0,
            )
        self.assertEqual(
            caught.exception.reason_code, "HANDOFF_COPY_FAILED"
        )
        self.assertFalse(plan.staging_path.exists())

        mismatch_plan = prepare_handoff_destination(
            plan.branch_paths,
            transaction_id="tx-staged-mismatch",
            managed_name=(
                "mismatch-"
                + source.artifact_identity.removeprefix("sha256:")[:12]
                + ".gguf"
            ),
            artifact_identity=source.artifact_identity,
        )
        with self.assertRaises(InspectorError) as caught:
            create_staged_artifact(
                mismatch_plan,
                source,
                safety_margin_bytes=0,
                reflink_cloner=lambda _source, _staging: False,
                staged_hasher=lambda _descriptor: (
                    IDENTITY_A,
                    source.snapshot["size_bytes"],
                ),
            )
        self.assertEqual(
            caught.exception.reason_code,
            "HANDOFF_STAGED_IDENTITY_MISMATCH",
        )
        self.assertFalse(mismatch_plan.staging_path.exists())

        fsync_plan = prepare_handoff_destination(
            plan.branch_paths,
            transaction_id="tx-fsync-failure",
            managed_name=(
                "fsync-"
                + source.artifact_identity.removeprefix("sha256:")[:12]
                + ".gguf"
            ),
            artifact_identity=source.artifact_identity,
        )

        def fail_fsync(_: int) -> None:
            raise OSError("injected fsync failure")

        with self.assertRaises(InspectorError) as caught:
            create_staged_artifact(
                fsync_plan,
                source,
                safety_margin_bytes=0,
                reflink_cloner=lambda _source, _staging: False,
                file_fsyncer=fail_fsync,
            )
        self.assertEqual(
            caught.exception.reason_code, "HANDOFF_COPY_FAILED"
        )
        self.assertFalse(fsync_plan.staging_path.exists())

    def test_target_staging_registry_and_publication_race_matrix(self) -> None:
        _, _, source, plan = self.transfer_fixture(
            transaction_id="tx-race-fixture", prefix="race"
        )
        digest = source.artifact_identity.removeprefix("sha256:")
        with self.assertRaises(InspectorError) as caught:
            prepare_handoff_destination(
                plan.branch_paths,
                transaction_id="tx-invalid-target",
                managed_name="../escape.gguf",
                artifact_identity=source.artifact_identity,
            )
        self.assertEqual(
            caught.exception.reason_code, "HANDOFF_TARGET_NAME_INVALID"
        )
        with self.assertRaises(InspectorError) as caught:
            prepare_handoff_destination(
                plan.branch_paths,
                transaction_id="tx-registry-collision",
                managed_name=f"registry-{digest[:12]}.gguf",
                artifact_identity=source.artifact_identity,
                historical_registry_locations={
                    f"MODEL/SUPERMODEL/registry-{digest[:12]}.gguf"
                },
            )
        self.assertEqual(
            caught.exception.reason_code,
            "HANDOFF_REGISTRY_LOCATION_COLLISION",
        )

        collision_name = f"collision-{digest[:12]}.gguf"
        collision = plan.branch_paths.managed_root / collision_name
        collision.write_bytes(b"existing")
        with self.assertRaises(InspectorError) as caught:
            prepare_handoff_destination(
                plan.branch_paths,
                transaction_id="tx-target-collision",
                managed_name=collision_name,
                artifact_identity=source.artifact_identity,
            )
        self.assertEqual(
            caught.exception.reason_code, "HANDOFF_TARGET_COLLISION"
        )
        self.assertEqual(collision.read_bytes(), b"existing")
        collision.unlink()

        plan.staging_path.write_bytes(b"existing-staging")
        with self.assertRaises(InspectorError) as caught:
            create_staged_artifact(
                plan,
                source,
                safety_margin_bytes=0,
                reflink_cloner=lambda _source, _staging: False,
            )
        self.assertEqual(
            caught.exception.reason_code, "HANDOFF_STAGING_COLLISION"
        )
        self.assertEqual(plan.staging_path.read_bytes(), b"existing-staging")
        plan.staging_path.unlink()

        staged = create_staged_artifact(
            plan,
            source,
            safety_margin_bytes=0,
            reflink_cloner=lambda _source, _staging: False,
        )

        def create_racer(target: Path) -> None:
            target.write_bytes(b"racer")

        with self.assertRaises(InspectorError) as caught:
            publish_staged_artifact(
                plan, staged, before_rename=create_racer
            )
        self.assertEqual(
            caught.exception.reason_code,
            "HANDOFF_PUBLICATION_CONFLICT",
        )
        self.assertEqual(plan.managed_target.read_bytes(), b"racer")
        self.assertTrue(plan.staging_path.exists())

    def transaction_fixture(
        self,
    ) -> tuple[
        DecisionAuthorization,
        str,
        str,
        dict[str, object],
    ]:
        production = self.managed / "production.gguf"
        production.write_bytes(b"production")
        os.chmod(production, 0o644)
        decision, inspection, capability, binding = (
            self.authorization_surfaces()
        )
        authorization = self.authenticate_surfaces(
            decision, inspection, capability, binding
        )
        source_name = "accepted.gguf"
        (self.paths.intake_root / source_name).write_bytes(
            b"accepted-artifact"
        )
        digest = str(
            decision["inspection"]["artifact_identity"]
        ).removeprefix("sha256:")
        managed_name = f"transaction-{digest[:12]}.gguf"
        common = {
            "branch_path_resolver": (
                lambda paths: BranchHandoffPaths.discover(paths)
            ),
            "authenticator": (
                lambda _paths, _decision_id: authorization
            ),
            "transaction_id_factory": (
                lambda: "tx-transaction-fixture"
            ),
            "handoff_id_factory": (
                lambda: (
                    "handoff-20260101T000000000000Z-"
                    "fedcba9876543210"
                )
            ),
            "safety_margin_bytes": 0,
        }
        return authorization, source_name, managed_name, common

    def test_transaction_success_transitions_and_idempotence(self) -> None:
        authorization, source_name, managed_name, common = (
            self.transaction_fixture()
        )
        observed: list[tuple[str, str]] = []

        def observer(kind: str, value: dict[str, object]) -> None:
            observed.append((kind, str(value["state"])))

        result = handoff_transaction(
            self.paths,
            authorization.decision["decision_id"],
            source_name,
            managed_name,
            transition_observer=observer,
            **common,
        )
        transaction_id, record, result_path, result_identity = result
        self.assertEqual(transaction_id, "tx-transaction-fixture")
        self.assertEqual(record["result_identity"], result_identity)
        self.assertTrue(result_path.exists())
        self.assertEqual(
            [
                state for kind, state in observed if kind == "status"
            ],
            [
                "VALIDATING_HANDOFF",
                "STAGING_ARTIFACT",
                "VERIFYING_STAGED_ARTIFACT",
                "PUBLISHING_ARTIFACT",
                "IDLE",
            ],
        )
        transaction = read_json_record(
            self.paths.transactions / f"{transaction_id}.json"
        )
        self.assertEqual(transaction["state"], "COMPLETED")
        self.assertEqual(transaction["reason_code"], "HANDOFF_COMPLETE")
        self.assertEqual(
            transaction["commit_phase"], "HANDOFF_RECORD_PUBLISHED"
        )
        status = read_json_record(self.paths.status / "current.json")
        self.assertEqual(status["state"], "IDLE")
        self.assertFalse((self.paths.locks / "active.json").exists())

        def forbidden_stager(*args: object, **kwargs: object) -> object:
            raise AssertionError("idempotent call attempted another copy")

        before_transactions = sorted(self.paths.transactions.iterdir())
        repeated = handoff_transaction(
            self.paths,
            authorization.decision["decision_id"],
            source_name,
            managed_name,
            stager=forbidden_stager,
            **common,
        )
        self.assertEqual(repeated, result)
        self.assertEqual(
            sorted(self.paths.transactions.iterdir()), before_transactions
        )
        self.assertEqual(len(list(self.paths.handoff_results.iterdir())), 1)

    def test_stale_pre_authentication_partial_resumes_same_transaction(self) -> None:
        authorization, source_name, managed_name, common = (
            self.transaction_fixture()
        )
        transaction_id = "tx-partial-stale"
        handoff_id = (
            "handoff-20260101T000000000000Z-"
            "fedcba9876543210"
        )
        partial = _handoff_transaction_value(
            transaction_id=transaction_id,
            handoff_id=handoff_id,
            start_utc=utc_now(),
            owner_identity={
                "pid": 99999999,
                "process_start_identity": "procfs-start-ticks:0",
                "boot_identity": "dead",
                "inspector_root_identity": {"device": 1, "inode": 1},
            },
            source_candidate=source_name,
            managed_name=managed_name,
        )
        atomic_write_json(
            self.paths.transactions / f"{transaction_id}.json",
            partial,
            mode=0o600,
        )
        atomic_write_json(
            self.paths.locks / "active.json",
            {
                "schema_version": "system-x.inspector-active-lock.v1",
                "transaction_id": transaction_id,
                "operation": "handoff",
                "pid": 99999999,
                "process_start_identity": "procfs-start-ticks:0",
                "boot_identity": "dead",
                "created_utc": utc_now(),
                "inspector_root_identity": {"device": 1, "inode": 1},
            },
            mode=0o600,
        )

        def forbidden_factory() -> str:
            raise AssertionError("stale partial resume created a new identity")

        result = handoff_transaction(
            self.paths,
            authorization.decision["decision_id"],
            source_name,
            managed_name,
            transaction_id_factory=forbidden_factory,
            handoff_id_factory=forbidden_factory,
            **{key: value for key, value in common.items()
               if key not in {"transaction_id_factory", "handoff_id_factory"}},
        )
        self.assertEqual(result[0], transaction_id)
        self.assertEqual(read_json_record(
            self.paths.transactions / f"{transaction_id}.json"
        )["state"], "COMPLETED")
        self.assertFalse((self.paths.locks / "active.json").exists())

    def test_cold_install_and_unrecorded_publication_recovery(self) -> None:
        authorization, source_name, managed_name, common = (
            self.transaction_fixture()
        )
        (self.managed / "production.gguf").unlink()
        failed_once = False

        def fail_cold_reference(*_args: object) -> None:
            nonlocal failed_once
            if not failed_once:
                failed_once = True
                raise InspectorError(
                    "HANDOFF_STAGING_INVALID",
                    "injected legacy cold-install record failure",
                )
            return None

        with mock.patch(
            "system_x_inspector.handoff._production_reference",
            side_effect=fail_cold_reference,
        ):
            with self.assertRaises(InspectorError) as caught:
                handoff_transaction(
                    self.paths,
                    authorization.decision["decision_id"],
                    source_name,
                    managed_name,
                    **common,
                )
        self.assertEqual(
            caught.exception.reason_code, "HANDOFF_STAGING_INVALID"
        )
        target = self.managed / managed_name
        self.assertTrue(target.is_file())
        failed = read_json_record(
            self.paths.transactions / "tx-transaction-fixture.json"
        )
        self.assertEqual(failed["state"], "FAILED")
        self.assertEqual(
            failed["commit_phase"], "PUBLISHED_TO_MANAGED_ROOT"
        )
        self.assertIsNone(failed["handoff_record_candidate"])

        resumed = handoff_transaction(
            self.paths,
            authorization.decision["decision_id"],
            source_name,
            managed_name,
            **common,
        )
        self.assertEqual(resumed[0], "tx-transaction-fixture")
        record = resumed[1]
        self.assertIsNone(
            record["alias_protection"]["default_target_before"]
        )
        self.assertIsNone(
            record["alias_protection"]["default_target_after"]
        )
        self.assertEqual(validate_handoff_record(record), record)

    def test_transaction_prepublication_cleanup_is_exact(self) -> None:
        authorization, source_name, managed_name, common = (
            self.transaction_fixture()
        )
        staging_sentinel = self.branch_staging / "unrelated.sentinel"
        managed_sentinel = self.managed / "unrelated.sentinel"
        staging_sentinel.write_bytes(b"staging-preserve")
        managed_sentinel.write_bytes(b"managed-preserve")

        def racing_publisher(plan: object, staged: object) -> object:
            def create_racer(target: Path) -> None:
                target.write_bytes(b"racer")

            return publish_staged_artifact(
                plan, staged, before_rename=create_racer
            )

        with self.assertRaises(InspectorError) as caught:
            handoff_transaction(
                self.paths,
                authorization.decision["decision_id"],
                source_name,
                managed_name,
                artifact_publisher=racing_publisher,
                **common,
            )
        self.assertEqual(
            caught.exception.reason_code,
            "HANDOFF_PUBLICATION_CONFLICT",
        )
        self.assertEqual(staging_sentinel.read_bytes(), b"staging-preserve")
        self.assertEqual(managed_sentinel.read_bytes(), b"managed-preserve")
        transaction_staging = [
            path
            for path in self.branch_staging.iterdir()
            if "tx-transaction-fixture" in path.name
        ]
        self.assertEqual(transaction_staging, [])
        target = self.managed / managed_name
        self.assertEqual(target.read_bytes(), b"racer")
        self.assertEqual(
            read_json_record(self.paths.status / "current.json")["state"],
            "IDLE",
        )
        self.assertFalse((self.paths.locks / "active.json").exists())
        self.assertFalse(any(self.paths.handoff_results.iterdir()))

    def test_postpublication_result_resume(self) -> None:
        authorization, source_name, managed_name, common = (
            self.transaction_fixture()
        )
        calls = 0

        def fail_first_result(
            paths: InspectorPaths, record: dict[str, object]
        ) -> tuple[Path, str]:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise InspectorError(
                    "HANDOFF_RESULT_COLLISION",
                    "injected post-publication result failure",
                )
            return publish_handoff_record(paths, record)

        with self.assertRaises(InspectorError) as caught:
            handoff_transaction(
                self.paths,
                authorization.decision["decision_id"],
                source_name,
                managed_name,
                result_publisher=fail_first_result,
                **common,
            )
        self.assertEqual(
            caught.exception.reason_code, "HANDOFF_RESULT_COLLISION"
        )
        target = self.managed / managed_name
        target_identity = (
            "sha256:" + hashlib.sha256(target.read_bytes()).hexdigest()
        )
        self.assertEqual(
            target_identity,
            authorization.decision["inspection"]["artifact_identity"],
        )
        self.assertFalse(any(self.paths.handoff_results.iterdir()))
        failed = read_json_record(
            self.paths.transactions / "tx-transaction-fixture.json"
        )
        self.assertEqual(
            failed["commit_phase"], "PUBLISHED_TO_MANAGED_ROOT"
        )
        self.assertIsInstance(failed["handoff_record_candidate"], dict)

        resumed = handoff_transaction(
            self.paths,
            authorization.decision["decision_id"],
            source_name,
            managed_name,
            result_publisher=fail_first_result,
            **common,
        )
        self.assertEqual(resumed[0], "tx-transaction-fixture")
        self.assertEqual(calls, 2)
        self.assertEqual(len(list(self.paths.handoff_results.iterdir())), 1)
        final = read_json_record(
            self.paths.transactions / "tx-transaction-fixture.json"
        )
        self.assertEqual(final["state"], "COMPLETED")
        self.assertEqual(final["commit_phase"], "HANDOFF_RECORD_PUBLISHED")
        self.assertEqual(
            read_json_record(self.paths.status / "current.json")["state"],
            "IDLE",
        )

    def test_uncertain_postpublication_ownership_fails_closed(self) -> None:
        authorization, source_name, managed_name, common = (
            self.transaction_fixture()
        )

        def reject_result(
            _paths: InspectorPaths, _record: dict[str, object]
        ) -> tuple[Path, str]:
            raise InspectorError(
                "HANDOFF_RESULT_COLLISION", "injected result failure"
            )

        with self.assertRaises(InspectorError):
            handoff_transaction(
                self.paths,
                authorization.decision["decision_id"],
                source_name,
                managed_name,
                result_publisher=reject_result,
                **common,
            )
        target = self.managed / managed_name
        target.unlink()
        target.write_bytes(b"uncertain-owner")
        uncertain_inode = target.stat().st_ino
        with self.assertRaises(InspectorError) as caught:
            handoff_transaction(
                self.paths,
                authorization.decision["decision_id"],
                source_name,
                managed_name,
                **common,
            )
        self.assertEqual(
            caught.exception.reason_code,
            "HANDOFF_PUBLICATION_FAILED",
        )
        self.assertTrue(target.exists())
        self.assertEqual(target.stat().st_ino, uncertain_inode)
        self.assertEqual(target.read_bytes(), b"uncertain-owner")
        self.assertFalse(any(self.paths.handoff_results.iterdir()))
        self.assertEqual(
            read_json_record(self.paths.status / "current.json")["state"],
            "IDLE",
        )
        self.assertFalse((self.paths.locks / "active.json").exists())

    def test_postrename_fsync_failure_is_recoverable(self) -> None:
        authorization, source_name, managed_name, common = (
            self.transaction_fixture()
        )

        def fail_directory_fsync(_: Path) -> None:
            raise OSError("injected managed directory fsync failure")

        def fsync_failing_publisher(plan: object, staged: object) -> object:
            return publish_staged_artifact(
                plan,
                staged,
                directory_fsyncer=fail_directory_fsync,
            )

        with self.assertRaises(InspectorError) as caught:
            handoff_transaction(
                self.paths,
                authorization.decision["decision_id"],
                source_name,
                managed_name,
                artifact_publisher=fsync_failing_publisher,
                **common,
            )
        self.assertEqual(
            caught.exception.reason_code,
            "HANDOFF_PUBLICATION_FAILED",
        )
        self.assertTrue((self.managed / managed_name).exists())
        failed = read_json_record(
            self.paths.transactions / "tx-transaction-fixture.json"
        )
        self.assertEqual(
            failed["commit_phase"], "PUBLISHED_TO_MANAGED_ROOT"
        )
        resumed = handoff_transaction(
            self.paths,
            authorization.decision["decision_id"],
            source_name,
            managed_name,
            **common,
        )
        self.assertEqual(resumed[0], "tx-transaction-fixture")
        self.assertEqual(len(list(self.paths.handoff_results.iterdir())), 1)

    def test_real_missing_decision_and_internal_transaction_exits(self) -> None:
        decision_id = (
            "decision-20260101T000000000000Z-0000000000000001"
        )
        status, result = self.run_main(
            "handoff",
            "--decision-id",
            decision_id,
            "--source-candidate",
            "absent.gguf",
            "--managed-name",
            "absent-000000000000.gguf",
        )
        self.assertEqual(status, 2)
        self.assertEqual(
            result["reason_code"], "HANDOFF_DECISION_NOT_FOUND"
        )
        self.assertIsInstance(result["transaction_id"], str)
        self.assertEqual(
            read_json_record(self.paths.status / "current.json")["state"],
            "IDLE",
        )
        self.assertFalse((self.paths.locks / "active.json").exists())
        self.assertFalse(any(self.branch_staging.iterdir()))
        self.assertFalse(any(self.managed.iterdir()))

        authorization, source_name, managed_name, common = (
            self.transaction_fixture()
        )

        def fail_stager(*args: object, **kwargs: object) -> object:
            raise RuntimeError("injected internal staging failure")

        with self.assertRaises(InspectorError) as caught:
            handoff_transaction(
                self.paths,
                authorization.decision["decision_id"],
                source_name,
                managed_name,
                stager=fail_stager,
                **common,
            )
        self.assertEqual(
            caught.exception.reason_code, "HANDOFF_INTERNAL_ERROR"
        )
        self.assertEqual(caught.exception.exit_status, 70)
        self.assertEqual(
            read_json_record(self.paths.status / "current.json")["state"],
            "IDLE",
        )
        self.assertFalse((self.paths.locks / "active.json").exists())
        self.assertFalse((self.managed / managed_name).exists())

    def test_source_read_failure_and_target_symlink_fail_closed(self) -> None:
        _, _, source, plan = self.transfer_fixture(
            transaction_id="tx-read-failure", prefix="readfail"
        )
        with mock.patch(
            "system_x_inspector.handoff.os.read",
            side_effect=OSError("injected source read failure"),
        ):
            with self.assertRaises(InspectorError) as caught:
                create_staged_artifact(
                    plan,
                    source,
                    safety_margin_bytes=0,
                    reflink_cloner=lambda _source, _staging: False,
                )
        self.assertEqual(
            caught.exception.reason_code, "HANDOFF_COPY_FAILED"
        )
        self.assertFalse(plan.staging_path.exists())
        digest = source.artifact_identity.removeprefix("sha256:")
        symlink_name = f"symlink-{digest[:12]}.gguf"
        symlink = self.managed / symlink_name
        symlink.symlink_to(self.managed / "production.gguf")
        with self.assertRaises(InspectorError) as caught:
            prepare_handoff_destination(
                plan.branch_paths,
                transaction_id="tx-symlink-target",
                managed_name=symlink_name,
                artifact_identity=source.artifact_identity,
            )
        self.assertEqual(
            caught.exception.reason_code, "HANDOFF_TARGET_COLLISION"
        )
        self.assertTrue(symlink.is_symlink())

    def test_linked_inspection_identity_mismatch_is_stale(self) -> None:
        decision, inspection, _, _ = self.authorization_surfaces()
        with mock.patch(
            "system_x_inspector.handoff.load_inspection_result",
            return_value=(inspection, IDENTITY_C),
        ):
            with self.assertRaises(InspectorError) as caught:
                _load_linked_inspection(self.paths, decision)
        self.assertEqual(
            caught.exception.reason_code, "HANDOFF_DECISION_STALE"
        )


if __name__ == "__main__":
    unittest.main()
