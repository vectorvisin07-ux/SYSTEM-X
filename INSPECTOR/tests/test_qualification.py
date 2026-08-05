from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from system_x_inspector.capabilities import (
    build_binding,
    build_capability_record,
    publish_binding,
    publish_capability_record,
)
from system_x_inspector.config import validate_configuration_values
from system_x_inspector.constants import SAFETY_MAXIMA, SCHEMA_IDENTITIES
from system_x_inspector.errors import InspectorError
from system_x_inspector.handoff import authenticate_handoff_decision
from system_x_inspector.locking import TransactionLock
from system_x_inspector.paths import InspectorPaths
from system_x_inspector.qualification import (
    ALL_QUALIFICATION_CHECKS,
    PROFILE_MAX_OUTPUT_TOKENS,
    PublicProfileProbeAdapter,
    QUALIFICATION_MANAGED_NAME_PATTERN,
    observe_qualification_candidate,
    _accepted_registration_wait_seconds,
    _operating_profile_identity,
    authenticate_qualification,
    clear_qualification_default,
    capture_incumbent_snapshot,
    classify_qualification_result,
    cleanup_qualification_candidate,
    find_idempotent_qualification,
    parse_system_x_stream,
    profile_check_names,
    prove_incumbent_restoration,
    qualification_managed_name,
    qualification_owned_cold_default,
    qualify_transaction,
    recover_with_accepted_platform_manager,
    run_capability_profile,
    stage_qualification_candidate,
    wait_for_candidate_removal,
)
from system_x_inspector.records import (
    atomic_write_json,
    canonical_json_bytes,
)
from system_x_inspector.results import utc_now
from system_x_inspector.runtime import decide_transaction, inspect_transaction
from tests.test_gguf import build_gguf


def sha(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


class QualificationAdmissionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = Path(
            tempfile.mkdtemp(prefix="inspector-qualification-", dir="/tmp")
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
            self.paths.staging,
            self.paths.tmp,
            self.paths.capability_root,
            self.paths.capability_records,
            self.paths.capability_bindings,
        ):
            path.mkdir(mode=0o700, parents=True, exist_ok=True)
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
        self.branch = self.temporary / "model-api-gguf"
        self.managed = self.branch / "MODEL" / "SUPERMODEL"
        self.branch_staging = (
            self.branch / "RUNTIME" / "api" / "replacement-staging"
        )
        self.managed.mkdir(mode=0o700, parents=True)
        self.branch_staging.mkdir(mode=0o700, parents=True)
        reference = self.managed / "incumbent-aaaaaaaaaaaa.gguf"
        reference.write_bytes(b"incumbent")
        reference.chmod(0o644)
        self._write_installed_tuple_files()
        capability = self._capability()
        publish_capability_record(self.paths, capability)
        publish_binding(
            self.paths,
            build_binding(
                capability,
                binding_generation=1,
                updated_utc="2026-01-01T00:00:00Z",
            ),
        )
        self.candidate = self.paths.intake_root / "candidate.gguf"
        self.candidate.write_bytes(
            build_gguf(
                metadata=[
                    ("general.architecture", 8, "test"),
                    ("general.type", 8, "model"),
                    ("general.alignment", 4, 32),
                    ("general.file_type", 4, 0),
                    ("general.quantization_version", 4, 2),
                ]
            )
        )
        self.candidate.chmod(0o644)
        self.configuration = validate_configuration_values(
            {
                "schema_version": SCHEMA_IDENTITIES["configuration"],
                "intake_root": str(self.paths.intake_root),
                "runtime_root": str(self.paths.runtime_root),
                "intake_bounds": dict(SAFETY_MAXIMA),
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
            },
            self.paths,
        )
        (
            _inspection_transaction,
            self.inspection,
            _inspection_path,
            _inspection_identity,
        ) = inspect_transaction(
            self.paths, self.configuration, self.candidate.name
        )
        (
            _decision_transaction,
            self.decision,
            _decision_path,
            _decision_identity,
        ) = decide_transaction(
            self.paths, self.inspection["inspection_id"]
        )
        self.assertEqual(
            self.decision["capability"]["capability_result"],
            "RUNTIME_SMOKE_REQUIRED",
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.temporary)

    def _write_installed_tuple_files(self) -> None:
        (self.branch / "llama.cpp/.git/refs/heads").mkdir(
            parents=True, mode=0o700
        )
        (self.branch / "llama.cpp/.git/HEAD").write_text(
            "ref: refs/heads/main\n", encoding="utf-8"
        )
        (self.branch / "llama.cpp/.git/refs/heads/main").write_text(
            "1" * 40 + "\n", encoding="utf-8"
        )
        for relative, content in (
            ("llama.cpp/build/bin/llama-server", b"llama"),
            ("branch_controller/controller.py", b"branch"),
            ("api_service_controller/controller.py", b"api"),
            ("service_control/supervisor.py", b"supervisor"),
            ("api_service/src/system_x_gguf_api/application.py", b"source"),
        ):
            path = self.branch / relative
            path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
            path.write_bytes(content)

    def _component(self, name: str, relative: str) -> dict[str, object]:
        content = (self.branch / relative).read_bytes()
        return {
            "name": name,
            "root": "branch",
            "path": relative,
            "byte_count": len(content),
            "sha256": sha(content),
        }

    def _installed_tuple(self) -> dict[str, object]:
        components = [
            self._component(
                "llama_server_binary",
                "llama.cpp/build/bin/llama-server",
            ),
            self._component(
                "branch_controller", "branch_controller/controller.py"
            ),
            self._component(
                "api_service_controller",
                "api_service_controller/controller.py",
            ),
            self._component(
                "service_control_supervisor",
                "service_control/supervisor.py",
            ),
        ]
        source = self._component(
            "api_source_graph:1",
            "api_service/src/system_x_gguf_api/application.py",
        )
        basis = [
            {
                "root": source["root"],
                "path": source["path"],
                "byte_count": source["byte_count"],
                "sha256": source["sha256"],
            }
        ]
        return {
            "source_commit": "1" * 40,
            "accepted_tag": "fixture",
            "clean_worktree_required": True,
            "components": components,
            "manifests": [
                {
                    "name": "api_source_graph",
                    "identity": sha(canonical_json_bytes(basis)),
                    "file_count": 1,
                    "byte_count": source["byte_count"],
                    "files": [source],
                }
            ],
            "platform_registration": {
                "schema_version": "fixture.registration.v1",
                "adapter_identity": "fixture.adapter.v1",
                "registered": True,
                "enabled": True,
            },
        }

    def _capability(self) -> dict[str, object]:
        return build_capability_record(
            created_utc="2026-01-01T00:00:00Z",
            branch_identity="model-api-gguf",
            supported_physical_format="GGUF",
            availability="AVAILABLE",
            runtime_engine="llama-server",
            installed_tuple=self._installed_tuple(),
            accepted_evidence=[
                {"basename": "accepted.rs", "sha256": sha(b"evidence")}
            ],
            supported_evidence={
                "supported_exact_artifact_identities": [sha(b"accepted")],
                "accepted_format_versions": [3],
                "accepted_architectures": ["test"],
                "accepted_primary_model_types": ["model"],
                "accepted_modalities": ["text"],
                "accepted_tensor_type_evidence": ["F32"],
                "accepted_tokenizer_evidence": sha(b"tokens"),
                "accepted_chat_template_evidence": sha(b"template"),
                "accepted_runtime_capabilities": ["generate/chat"],
                "public_model_id": "fixture",
                "accepted_capability_manifest_identity": sha(b"manifest"),
            },
            unsupported_primary_artifact_roles=["adapter"],
            unproven_valid_policy="RUNTIME_SMOKE_REQUIRED",
            reason_code=None,
        )

    def authorize(self):
        return authenticate_qualification(
            self.paths,
            self.inspection["inspection_id"],
            self.inspection["artifact"]["identity"],
            "CORE_CHAT",
            transaction_id="tx-qualification-fixture",
        )

    def incumbent(self):
        registry = {
            "present": True,
            "generation": 9,
            "default_alias": "default",
            "public_model_id": "incumbent-public",
            "artifact_version_id": "bundle-" + "a" * 64,
            "capability_manifest_identity": sha(b"manifest"),
            "managed_location_identity": sha(b"location"),
            "historical_locations": ("incumbent-aaaaaaaaaaaa.gguf",),
        }
        service = {
            "profile_identity": sha(b"profile"),
            "service_readiness": "READY",
            "recovery_state": "IDLE",
            "warm": {
                "public_model_id": "incumbent-public",
                "artifact_version_id": "bundle-" + "a" * 64,
                "capability_manifest_identity": sha(b"manifest"),
                "health_state": "ready",
            },
            "api_service_transaction_id": "api-fixture",
            "router_transaction_id": "router-fixture",
            "model_child_identity": {"present": True},
        }
        return capture_incumbent_snapshot(
            self.paths,
            self.authorize().branch_paths,
            registry_reader=lambda _root: registry,
            service_reader=lambda _root, _registry: service,
            credential_reader=lambda _root: SimpleNamespace(
                key_id="fixture-key", raw="not-observed"
            ),
        )

    def test_operating_profile_identity_matches_service_contract(self) -> None:
        profile = {
            "schema_version": "system-x.service-operating-profile.v1",
            "public_endpoint": {"host": "127.0.0.1", "port": 8080},
        }
        compact = json.dumps(
            profile,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        observed = _operating_profile_identity(
            profile,
            reason_code="QUALIFICATION_INCUMBENT_RESTORATION_FAILED",
        )
        self.assertEqual(observed, sha(compact))
        self.assertNotEqual(observed, sha(canonical_json_bytes(profile)))

    def test_registration_wait_covers_manager_probe_budget(self) -> None:
        status = self.branch / "RUNTIME" / "api" / "status"
        status.mkdir(mode=0o700, parents=True)
        atomic_write_json(
            status / "service.json",
            {
                "lifecycle_state": "STARTED",
                "private_backend_model_timeout_seconds": 120.0,
                "service_start_timeout_seconds": 153.0,
            },
            mode=0o600,
        )
        self.assertEqual(
            _accepted_registration_wait_seconds(self.branch), 153.0
        )
        removed = wait_for_candidate_removal(
            self.branch,
            qualification_managed_name(
                self.inspection["artifact"]["identity"],
                "tx-removal-wait-fixture",
            ),
            self.inspection["artifact"]["identity"],
            observer=lambda _root, _name, _identity: {
                "present": False,
                "terminal": "REMOVED",
                "states_observed": ["UNAVAILABLE", "REMOVED"],
            },
        )
        self.assertTrue(removed["registry_location_removed"])

    def test_ready_version_at_removed_location_is_observed_as_removed(self) -> None:
        database = self.branch / "RUNTIME/api/database/model_registry.sqlite3"
        database.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        import sqlite3

        connection = sqlite3.connect(database)
        try:
            connection.executescript(
                """
                CREATE TABLE registry_metadata (key TEXT PRIMARY KEY,value TEXT);
                CREATE TABLE artifact_rejections (
                    relative_path TEXT,reason_code TEXT,detail_json TEXT,
                    first_seen_utc TEXT,last_seen_utc TEXT
                );
                CREATE TABLE artifact_locations (
                    relative_root TEXT PRIMARY KEY,current_bundle_id TEXT,
                    present INTEGER,first_seen_utc TEXT,last_seen_utc TEXT
                );
                CREATE TABLE model_versions (
                    model_version_id TEXT PRIMARY KEY,bundle_id TEXT,state TEXT,
                    created_utc TEXT
                );
                CREATE TABLE model_version_locations (
                    model_version_id TEXT,relative_root TEXT
                );
                CREATE TABLE capability_manifests (
                    model_version_id TEXT,manifest_sha256 TEXT
                );
                CREATE TABLE registry_events (
                    generation INTEGER,event_id INTEGER,event_type TEXT,
                    subject_id TEXT
                );
                CREATE TABLE aliases (
                    alias TEXT,model_version_id TEXT
                );
                """
            )
            managed_name = (
                "qualification-candidate-"
                + "a" * 16
                + "-"
                + "b" * 16
                + ".gguf"
            )
            artifact = "sha256:" + "c" * 64
            bundle = "bundle-" + "c" * 64
            model_id = "candidate-model"
            connection.execute(
                "INSERT INTO registry_metadata VALUES ('registry_generation','9')"
            )
            connection.execute(
                "INSERT INTO artifact_locations VALUES (?,?,0,'t','t')",
                (managed_name, bundle),
            )
            connection.execute(
                "INSERT INTO model_versions VALUES (?,?,?,'t')",
                (model_id, bundle, "REMOVED"),
            )
            connection.execute(
                "INSERT INTO model_version_locations VALUES (?,?)",
                (model_id, managed_name),
            )
            connection.execute(
                "INSERT INTO capability_manifests VALUES (?,?)",
                (model_id, "d" * 64),
            )
            connection.execute(
                "INSERT INTO registry_events VALUES (9,1,'artifact_location_removed',?)",
                (managed_name,),
            )
            connection.commit()
        finally:
            connection.close()
        observed = observe_qualification_candidate(
            self.branch, managed_name, artifact
        )
        self.assertFalse(observed["present"])
        self.assertEqual(observed["terminal"], "REMOVED")
        self.assertEqual(observed["public_model_id"], model_id)
        self.assertEqual(observed["artifact_version_id"], bundle)

    def test_runtime_smoke_authentication_and_atomic_admission(self) -> None:
        authorization = self.authorize()
        self.assertEqual(
            authorization.decision_authorization.decision["selected_branch"],
            None,
        )
        self.assertFalse(
            authorization.decision_authorization.decision[
                "handoff_allowed"
            ]
        )
        admission = stage_qualification_candidate(
            authorization,
            self.incumbent(),
            transaction_id="tx-qualification-fixture",
            safety_margin_bytes=0,
        )
        self.assertEqual(
            admission.published.sha256,
            self.inspection["artifact"]["identity"],
        )
        self.assertTrue(admission.published.path.is_file())
        self.assertFalse(admission.plan.staging_path.exists())
        self.assertEqual(admission.published.link_count, 1)
        self.assertEqual(
            admission.plan.managed_name,
            qualification_managed_name(
                self.inspection["artifact"]["identity"],
                "tx-qualification-fixture",
            ),
        )

    def test_managed_name_is_transaction_owned_and_artifact_suffixed(
        self,
    ) -> None:
        artifact = self.inspection["artifact"]["identity"]
        first = qualification_managed_name(artifact, "tx-one")
        second = qualification_managed_name(artifact, "tx-two")
        self.assertNotEqual(first, second)
        self.assertIsNotNone(
            QUALIFICATION_MANAGED_NAME_PATTERN.fullmatch(first)
        )
        self.assertIsNotNone(
            QUALIFICATION_MANAGED_NAME_PATTERN.fullmatch(second)
        )
        self.assertTrue(first.endswith(artifact[7:23] + ".gguf"))
        self.assertTrue(second.endswith(artifact[7:23] + ".gguf"))

    def test_ordinary_handoff_remains_blocked(self) -> None:
        with self.assertRaises(InspectorError) as caught:
            authenticate_handoff_decision(
                self.paths, self.decision["decision_id"]
            )
        self.assertEqual(
            caught.exception.reason_code,
            "HANDOFF_DECISION_NOT_SUPPORTED",
        )

    def test_identity_mismatch_rejected_before_branch_mutation(self) -> None:
        with self.assertRaises(InspectorError) as caught:
            authenticate_qualification(
                self.paths,
                self.inspection["inspection_id"],
                sha(b"different"),
                "CORE_CHAT",
                transaction_id="tx-qualification-fixture",
            )
        self.assertEqual(
            caught.exception.reason_code,
            "QUALIFICATION_ARTIFACT_IDENTITY_MISMATCH",
        )
        self.assertEqual(
            sorted(path.name for path in self.managed.iterdir()),
            ["incumbent-aaaaaaaaaaaa.gguf"],
        )
        self.assertEqual(list(self.branch_staging.iterdir()), [])

    def test_changed_source_rejected_before_branch_mutation(self) -> None:
        with self.candidate.open("ab") as handle:
            handle.write(b"x")
        with self.assertRaises(InspectorError) as caught:
            self.authorize()
        self.assertEqual(
            caught.exception.reason_code,
            "QUALIFICATION_ARTIFACT_IDENTITY_MISMATCH",
        )
        self.assertEqual(list(self.branch_staging.iterdir()), [])

    def test_symlink_source_rejected(self) -> None:
        other = self.paths.intake_root / "other.gguf"
        other.write_bytes(build_gguf())
        self.candidate.unlink()
        self.candidate.symlink_to(other.name)
        with self.assertRaises(InspectorError) as caught:
            self.authorize()
        self.assertEqual(
            caught.exception.reason_code, "QUALIFICATION_SOURCE_SYMLINK"
        )

    def test_hardlink_source_rejected(self) -> None:
        link = self.paths.intake_root / "linked.gguf"
        os.link(self.candidate, link)
        with self.assertRaises(InspectorError) as caught:
            self.authorize()
        self.assertEqual(
            caught.exception.reason_code,
            "QUALIFICATION_SOURCE_HARDLINK_REJECTED",
        )

    def ready_observation(self) -> dict[str, object]:
        return {
            "registry_generation": 10,
            "present": True,
            "terminal": "READY",
            "states_observed": ["REGISTERED", "PROBING", "READY"],
            "public_model_id": "candidate-public",
            "artifact_version_id": (
                "bundle-"
                + self.inspection["artifact"]["identity"].removeprefix(
                    "sha256:"
                )
            ),
            "capability_manifest_identity": sha(b"candidate-manifest"),
            "aliases": [],
            "default_bound": False,
            "rejection_reason_code": None,
            "rejection_detail_identity": None,
            "events": [],
        }

    def restoration_evidence(self, incumbent):
        restoration = {
            "required_state": (
                "READY" if incumbent.present else "WAITING_FOR_MODEL"
            ),
            "default_unchanged": True,
            "incumbent_ready": True if incumbent.present else None,
            "incumbent_warm": True if incumbent.present else None,
            "waiting_for_model": (
                True if not incumbent.present else None
            ),
            "recovery_idle": True,
            "proved": True,
        }
        return restoration, incumbent.warm_before

    def test_exact_cleanup_and_default_alias_protection(self) -> None:
        authorization = self.authorize()
        incumbent = self.incumbent()
        admission = stage_qualification_candidate(
            authorization,
            incumbent,
            transaction_id="tx-cleanup-fixture",
            safety_margin_bytes=0,
        )
        observation = self.ready_observation()
        cleanup, removed = cleanup_qualification_candidate(
            admission,
            observation,
            removal_waiter=lambda _root, _name, _identity: {
                **observation,
                "present": False,
                "terminal": "REMOVED",
                "states_observed": [
                    "REGISTERED",
                    "PROBING",
                    "READY",
                    "REMOVED",
                ],
                "registry_location_removed": True,
            },
        )
        self.assertTrue(cleanup["ownership_certain"])
        self.assertTrue(cleanup["managed_target_absent"])
        self.assertTrue(removed["registry_location_removed"])
        self.assertFalse(admission.plan.managed_target.exists())

        protected = stage_qualification_candidate(
            authorization,
            incumbent,
            transaction_id="tx-default-protection-fixture",
            safety_margin_bytes=0,
        )
        with self.assertRaises(InspectorError) as caught:
            cleanup_qualification_candidate(
                protected,
                {**observation, "default_bound": True},
            )
        self.assertEqual(
            caught.exception.reason_code,
            "QUALIFICATION_DEFAULT_CHANGED",
        )
        self.assertTrue(protected.plan.managed_target.exists())

    def test_cleanup_rechecks_removal_after_manager_fallback(self) -> None:
        authorization = self.authorize()
        admission = stage_qualification_candidate(
            authorization,
            self.incumbent(),
            transaction_id="tx-manager-fallback-fixture",
            safety_margin_bytes=0,
        )
        observation = self.ready_observation()
        removal_calls = []
        restore_calls = []

        def removal_waiter(_root, _name, _identity):
            removal_calls.append(len(removal_calls) + 1)
            if len(removal_calls) == 1:
                return {
                    **observation,
                    "registry_location_removed": False,
                }
            return {
                **observation,
                "present": False,
                "terminal": "REMOVED",
                "states_observed": [
                    *observation["states_observed"],
                    "REMOVED",
                ],
                "registry_location_removed": True,
            }

        def manager_restorer(root):
            self.assertEqual(root, authorization.branch_paths.branch_root)
            self.assertFalse(admission.plan.managed_target.exists())
            restore_calls.append(root)
            return {"used": True}

        cleanup, removed = cleanup_qualification_candidate(
            admission,
            observation,
            removal_waiter=removal_waiter,
            manager_restorer=manager_restorer,
        )
        self.assertEqual(removal_calls, [1, 2])
        self.assertEqual(
            restore_calls, [authorization.branch_paths.branch_root]
        )
        self.assertTrue(cleanup["ownership_certain"])
        self.assertTrue(cleanup["registry_location_removed"])
        self.assertTrue(removed["registry_location_removed"])

    def test_incumbent_and_waiting_for_model_restoration(self) -> None:
        authorization = self.authorize()
        incumbent = self.incumbent()
        registry = {
            "present": True,
            "generation": incumbent.registry_generation,
            "default_alias": incumbent.default_alias,
            "public_model_id": incumbent.public_model_id,
            "artifact_version_id": incumbent.artifact_version_id,
            "capability_manifest_identity": (
                incumbent.capability_manifest_identity
            ),
            "managed_location_identity": (
                incumbent.managed_location_identity
            ),
            "historical_locations": (
                incumbent.historical_registry_locations
            ),
        }
        service = {
            "service_readiness": "READY",
            "recovery_state": "IDLE",
            "warm": incumbent.warm_before,
        }
        restored, warm = prove_incumbent_restoration(
            authorization,
            incumbent,
            registry_reader=lambda _root: registry,
            service_reader=lambda _root, _registry: service,
        )
        self.assertTrue(restored["proved"])
        self.assertEqual(warm, incumbent.warm_before)

        no_incumbent = replace(
            incumbent,
            present=False,
            default_alias=None,
            public_model_id=None,
            artifact_version_id=None,
            capability_manifest_identity=None,
            managed_location_identity=None,
            warm_before=None,
            service_readiness="WAITING_FOR_MODEL",
            model_child_identity=None,
        )
        waiting, warm = prove_incumbent_restoration(
            authorization,
            no_incumbent,
            registry_reader=lambda _root: {
                "present": False,
                "generation": 11,
                "default_alias": None,
                "public_model_id": None,
                "artifact_version_id": None,
                "capability_manifest_identity": None,
                "managed_location_identity": None,
                "historical_locations": (),
            },
            service_reader=lambda _root, _registry: {
                "service_readiness": "WAITING_FOR_MODEL",
                "recovery_state": "IDLE",
                "warm": None,
            },
        )
        self.assertTrue(waiting["proved"])
        self.assertTrue(waiting["waiting_for_model"])
        self.assertIsNone(warm)

    def test_qualify_transaction_is_immutable_and_idempotent(self) -> None:
        incumbent = self.incumbent()
        observation = self.ready_observation()
        transaction_ids = iter(
            ("tx-qualification-one", "tx-qualification-two")
        )

        def admit(authorization, snapshot, *, transaction_id):
            return stage_qualification_candidate(
                authorization,
                snapshot,
                transaction_id=transaction_id,
                safety_margin_bytes=0,
            )

        def clean(admission, observed):
            return cleanup_qualification_candidate(
                admission,
                observed,
                removal_waiter=lambda _root, _name, _identity: {
                    **observed,
                    "present": False,
                    "terminal": "REMOVED",
                    "states_observed": [
                        *observed["states_observed"],
                        "REMOVED",
                    ],
                    "registry_location_removed": True,
                },
            )

        arguments = {
            "transaction_id_factory": lambda: next(transaction_ids),
            "qualification_id_factory": lambda: (
                "qualification-20260101T000000000000Z-"
                "0123456789abcdef"
            ),
            "incumbent_factory": lambda _paths, _branch: incumbent,
            "admission_factory": admit,
            "registry_waiter": (
                lambda _root, _name, _identity: observation
            ),
            "service_factory": (
                lambda _authorization, _incumbent: SimpleNamespace()
            ),
            "credential_reader": lambda _root: SimpleNamespace(
                key_id=incumbent.credential_key_id,
                raw="memory-only-fixture",
            ),
            "profile_adapter_factory": (
                lambda **_values: FakeProfileAdapter()
            ),
            "cleanup_factory": clean,
            "restoration_factory": (
                lambda _authorization, snapshot: (
                    self.restoration_evidence(snapshot)
                )
            ),
        }
        first_tx, first, first_path, first_identity = (
            qualify_transaction(
                self.paths,
                self.inspection["inspection_id"],
                self.inspection["artifact"]["identity"],
                "CORE_CHAT",
                **arguments,
            )
        )
        self.assertEqual(first_tx, "tx-qualification-one")
        self.assertEqual(
            first["result_class"], "SUPPORTED_FOR_CURRENT_TUPLE"
        )
        self.assertEqual(first["result_identity"], first_identity)
        self.assertEqual(first_path.stat().st_mode & 0o777, 0o600)
        self.assertFalse(
            (
                self.managed
                / qualification_managed_name(
                    self.inspection["artifact"]["identity"],
                    "tx-qualification-one",
                )
            ).exists()
        )

        arguments["admission_factory"] = (
            lambda *_args, **_kwargs: self.fail(
                "idempotent invocation repeated candidate admission"
            )
        )
        second_tx, second, second_path, second_identity = (
            qualify_transaction(
                self.paths,
                self.inspection["inspection_id"],
                self.inspection["artifact"]["identity"],
                "CORE_CHAT",
                **arguments,
            )
        )
        self.assertEqual(second_tx, "tx-qualification-two")
        self.assertEqual(second_identity, first_identity)
        self.assertEqual(second_path, first_path)
        self.assertEqual(second, first)
        self.assertEqual(
            len(list(self.paths.qualification_results.iterdir())), 1
        )
        current = self.authorize()
        self.assertRegex(
            first["validity_predicate"]["system_x_source_commit"],
            r"[0-9a-f]{40}",
        )
        self.assertRegex(
            first["validity_predicate"]["system_x_source_tree"],
            r"[0-9a-f]{40}",
        )
        repaired = replace(
            current,
            installed_tuple_evidence={
                **current.installed_tuple_evidence,
                "inspector_source_identity": sha(b"post-repair-source"),
            },
        )
        self.assertIsNone(find_idempotent_qualification(self.paths, repaired))

    def test_first_model_default_is_owned_only_for_exact_candidate(self) -> None:
        admission = SimpleNamespace(
            published=SimpleNamespace(sha256="sha256:" + "a" * 64)
        )
        incumbent = SimpleNamespace(present=False)
        observation = {
            "terminal": "READY",
            "present": True,
            "default_bound": True,
            "aliases": ["default"],
            "artifact_version_id": "bundle-" + "a" * 64,
            "public_model_id": "candidate",
            "capability_manifest_identity": "sha256:" + "b" * 64,
        }
        self.assertTrue(
            qualification_owned_cold_default(
                admission, incumbent, observation
            )
        )
        self.assertFalse(
            qualification_owned_cold_default(
                admission, SimpleNamespace(present=True), observation
            )
        )
        self.assertFalse(
            qualification_owned_cold_default(
                admission,
                incumbent,
                {**observation, "aliases": ["default", "unexpected"]},
            )
        )

    def test_cold_default_clear_uses_branch_cas_and_reobserves(self) -> None:
        public_model_id = "candidate"
        managed_name = (
            "qualification-candidate-" + "a" * 16 + "-" + "b" * 16 + ".gguf"
        )
        before = {
            "terminal": "READY",
            "present": True,
            "default_bound": True,
            "aliases": ["default"],
            "public_model_id": public_model_id,
            "artifact_version_id": "bundle-" + "c" * 64,
            "registry_generation": 7,
            "capability_manifest_identity": "sha256:" + "e" * 64,
        }
        after = {
            **before,
            "default_bound": False,
            "aliases": [],
            "registry_generation": 8,
        }
        result = {
            "ok": True,
            "alias_transaction": {
                "action": "clear",
                "alias": "default",
                "previous_target": public_model_id,
                "new_target": None,
                "new_registry_generation": 8,
            },
        }
        runner = Mock(
            return_value=SimpleNamespace(
                returncode=0,
                stdout=json.dumps(result).encode("utf-8"),
                stderr=b"",
            )
        )
        initial = {
            **before,
            "capability_manifest_identity": "sha256:" + "d" * 64,
        }
        observer = Mock(side_effect=(before, after))
        alias, observed = clear_qualification_default(
            self.branch,
            managed_name,
            "sha256:" + "c" * 64,
            initial,
            "tx-20260101T000000000000Z-aaaaaaaaaaaa",
            runner=runner,
            observer=observer,
        )
        self.assertEqual(alias["action"], "clear")
        self.assertEqual(observed, after)
        request = json.loads(runner.call_args.kwargs["input"])
        self.assertEqual(request["action"], "clear")
        self.assertEqual(request["expected_registry_generation"], 7)
        self.assertEqual(request["expected_current_target"], public_model_id)
        self.assertEqual(
            observed["capability_manifest_identity"],
            "sha256:" + "e" * 64,
        )
        self.assertEqual(observer.call_count, 2)
        observer.assert_called_with(
            self.branch, managed_name, "sha256:" + "c" * 64
        )

    def test_interrupted_recovery_requires_absent_candidate(self) -> None:
        candidate = self.managed / (
            "qualification-candidate-" + "a" * 16 + "-" + "b" * 16 + ".gguf"
        )
        candidate.write_bytes(b"owned")
        with self.assertRaises(InspectorError) as caught:
            recover_with_accepted_platform_manager(
                self.branch, sleeper=lambda _seconds: None
            )
        self.assertEqual(
            caught.exception.reason_code,
            "QUALIFICATION_INCUMBENT_RESTORATION_FAILED",
        )

    def test_active_transaction_is_rejected_concurrently(self) -> None:
        lock = TransactionLock(
            self.paths,
            transaction_id="tx-existing-owner",
            operation="inspect",
        )
        lock.acquire()
        try:
            with self.assertRaises(InspectorError) as caught:
                qualify_transaction(
                    self.paths,
                    self.inspection["inspection_id"],
                    self.inspection["artifact"]["identity"],
                    "CORE_CHAT",
                    transaction_id_factory=lambda: "tx-concurrent",
                )
            self.assertEqual(
                caught.exception.reason_code,
                "QUALIFICATION_CONCURRENCY_REJECTED",
            )
        finally:
            lock.release()


class FakeProfileAdapter:
    def __init__(
        self,
        overrides: dict[str, dict[str, object]] | None = None,
    ) -> None:
        self.overrides = overrides or {}
        self.calls: list[str] = []

    def probe(
        self,
        check_name: str,
        *,
        model_id: str,
        artifact_version_id: str,
        capability_manifest_identity: str,
    ) -> dict[str, object]:
        self.calls.append(check_name)
        if check_name in self.overrides:
            return self.overrides[check_name]
        return {
            "status": "PASSED",
            "request_id": None,
            "http_status": None,
            "finish_or_terminal_state": "fixture-complete",
            "usage": None,
            "capability_observation": None,
            "evidence": {
                "check_name": check_name,
                "model_id": model_id,
                "artifact_version_id": artifact_version_id,
                "capability_manifest_identity": (
                    capability_manifest_identity
                ),
            },
        }


class QualificationProfileRunnerTest(unittest.TestCase):
    MODEL_ID = "sx-gguf-qualification-fixture-0123456789ab"
    ARTIFACT_VERSION = "bundle-" + "1" * 64
    MANIFEST = "sha256:" + "2" * 64

    def test_live_profile_request_has_bounded_reasoning_headroom(self) -> None:
        adapter = object.__new__(PublicProfileProbeAdapter)
        adapter.public_model_id = "fixture-model"
        body = adapter._native_body("bounded fixture", stream=False)
        self.assertEqual(body["max_output_tokens"], PROFILE_MAX_OUTPUT_TOKENS)
        self.assertEqual(PROFILE_MAX_OUTPUT_TOKENS, 1024)
        self.assertLess(PROFILE_MAX_OUTPUT_TOKENS, 2048)

    def test_live_adapter_has_bounded_openai_and_heychat_fallback(self) -> None:
        adapter = object.__new__(PublicProfileProbeAdapter)
        adapter.public_model_id = self.MODEL_ID
        adapter.artifact_version_id = self.ARTIFACT_VERSION
        adapter.capability_manifest_identity = self.MANIFEST
        adapter.compatibility_probe = None
        adapter.cache = {}
        adapter.expectations = {}
        first = "sx_req_" + "1" * 32
        second = "sx_req_" + "2" * 32
        adapter._compatibility_exchange = Mock(
            side_effect=(
                {
                    "status": 200,
                    "request_id": first,
                    "compatibility_identity": (
                        "system-x.openai-compatible.v1"
                    ),
                    "streaming_identity": None,
                    "body": {
                        "object": "list",
                        "data": [
                            {"id": "default"},
                            {"id": self.MODEL_ID},
                        ],
                    },
                    "raw": None,
                },
                {
                    "status": 200,
                    "request_id": second,
                    "compatibility_identity": (
                        "system-x.openai-compatible.v1"
                    ),
                    "streaming_identity": None,
                    "body": {
                        "object": "chat.completion",
                        "model": self.MODEL_ID,
                        "choices": [
                            {
                                "message": {"content": "OK"},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 2,
                            "completion_tokens": 1,
                            "total_tokens": 3,
                        },
                    },
                    "raw": None,
                },
            )
        )
        listed = adapter._compatibility("openai_model_listing")
        chatted = adapter._compatibility("openai_nonstream_request")
        heychat = adapter._compatibility("heychat_adapter_compatibility")
        self.assertEqual(listed["status"], "PASSED")
        self.assertEqual(chatted["status"], "PASSED")
        self.assertEqual(heychat["status"], "PASSED")
        self.assertEqual(adapter._compatibility_exchange.call_count, 2)
        self.assertEqual(
            {item.protocol_family for item in adapter.expectations.values()},
            {"openai_compatible"},
        )

    def run_profile(
        self,
        profile: str,
        adapter: FakeProfileAdapter | None = None,
    ):
        selected = adapter or FakeProfileAdapter()
        result = run_capability_profile(
            selected,
            requested_profile=profile,
            model_id=self.MODEL_ID,
            artifact_version_id=self.ARTIFACT_VERSION,
            capability_manifest_identity=self.MANIFEST,
        )
        return selected, result

    def test_all_profile_compositions_are_exact_and_supported(self) -> None:
        all_names = tuple(item[0] for item in ALL_QUALIFICATION_CHECKS)
        for profile in (
            "CORE_CHAT",
            "EXTENDED_CHAT",
            "AGENT",
            "FULL_PRODUCT",
        ):
            with self.subTest(profile=profile):
                adapter, result = self.run_profile(profile)
                self.assertEqual(
                    tuple(adapter.calls), profile_check_names(profile)
                )
                self.assertEqual(
                    tuple(item["check_name"] for item in result.checks),
                    all_names,
                )
                selected = set(profile_check_names(profile))
                self.assertTrue(
                    all(
                        item["status"]
                        != (
                            "NOT_REQUESTED"
                            if item["check_name"] in selected
                            else "PASSED"
                        )
                        for item in result.checks
                    )
                )
                self.assertEqual(
                    result.result_class,
                    "SUPPORTED_FOR_CURRENT_TUPLE",
                )
                self.assertIn("CORE_CHAT", result.supported_profiles)
                if profile != "CORE_CHAT":
                    self.assertIn(profile, result.supported_profiles)

    def test_extended_and_agent_gated_capabilities_are_partial(self) -> None:
        cases = (
            ("EXTENDED_CHAT", "reasoning_output", "reasoning_output"),
            ("AGENT", "forced_registered_tool_call", "tool_calling"),
        )
        for profile, check_name, capability in cases:
            with self.subTest(profile=profile):
                adapter = FakeProfileAdapter(
                    {
                        check_name: {
                            "status": "PASSED_GATED_UNAVAILABLE",
                            "capability_observation": {
                                "capability": capability,
                                "available": False,
                                "accurately_gated": True,
                            },
                            "evidence": {
                                "capability": capability,
                                "reported_state": "unavailable",
                            },
                        }
                    }
                )
                _adapter, result = self.run_profile(profile, adapter)
                self.assertEqual(
                    result.result_class, "PARTIALLY_SUPPORTED"
                )
                self.assertEqual(
                    result.supported_profiles, ("CORE_CHAT",)
                )
                self.assertIn(
                    capability,
                    result.observed_capabilities[
                        "gated_unavailable"
                    ],
                )

    def test_full_product_accepts_accurate_capability_gating(self) -> None:
        overrides = {}
        for check_name, capability in (
            ("capability_gating_reasoning", "reasoning_output"),
            ("capability_gating_tool_calling", "tool_calling"),
            (
                "capability_gating_structured_output",
                "structured_output",
            ),
        ):
            overrides[check_name] = {
                "status": "PASSED_GATED_UNAVAILABLE",
                "capability_observation": {
                    "capability": capability,
                    "available": False,
                    "accurately_gated": True,
                },
                "evidence": {
                    "capability": capability,
                    "reported_state": "unavailable",
                },
            }
        _adapter, result = self.run_profile(
            "FULL_PRODUCT", FakeProfileAdapter(overrides)
        )
        self.assertEqual(
            result.result_class, "SUPPORTED_FOR_CURRENT_TUPLE"
        )
        self.assertEqual(
            set(result.observed_capabilities["gated_unavailable"]),
            {
                "reasoning_output",
                "tool_calling",
                "structured_output",
            },
        )

    def test_core_failure_is_clean_failure_and_content_is_excluded(self) -> None:
        adapter = FakeProfileAdapter(
            {
                "native_nonstream_chat": {
                    "status": "PASSED",
                    "evidence": {"answer": "must-not-persist"},
                }
            }
        )
        _adapter, result = self.run_profile("CORE_CHAT", adapter)
        failed = next(
            item
            for item in result.checks
            if item["check_name"] == "native_nonstream_chat"
        )
        self.assertEqual(failed["status"], "FAILED")
        self.assertEqual(
            result.result_class, "QUALIFICATION_FAILED_CLEAN"
        )
        self.assertNotIn(
            "must-not-persist",
            repr(result),
        )

    def test_terminal_result_classification_law(self) -> None:
        _adapter, run = self.run_profile("CORE_CHAT")
        self.assertEqual(
            classify_qualification_result(
                "CORE_CHAT",
                run.checks,
                runtime_outcome="UNSUPPORTED",
            )[0],
            "UNSUPPORTED",
        )
        self.assertEqual(
            classify_qualification_result(
                "CORE_CHAT",
                run.checks,
                runtime_outcome="REJECTED",
            )[0],
            "REJECTED",
        )
        self.assertEqual(
            classify_qualification_result(
                "CORE_CHAT",
                run.checks,
                runtime_outcome="UNAVAILABLE",
            )[0],
            "QUALIFICATION_FAILED_CLEAN",
        )
        self.assertEqual(
            classify_qualification_result(
                "CORE_CHAT",
                run.checks,
                ownership_certain=False,
            )[0],
            "QUALIFICATION_FAIL_CLOSED",
        )

    def test_native_stream_parser_accepts_one_legal_terminal(self) -> None:
        request_id = "sx_req_" + "3" * 32
        model_id = self.MODEL_ID
        values = [
            (
                "response.started",
                {
                    "type": "response.started",
                    "sequence": 0,
                    "request_id": request_id,
                    "model": model_id,
                    "operation": "chat",
                    "status": "in_progress",
                },
            ),
            (
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "sequence": 1,
                    "request_id": request_id,
                    "model": model_id,
                    "delta": "ready",
                },
            ),
            (
                "response.usage",
                {
                    "type": "response.usage",
                    "sequence": 2,
                    "request_id": request_id,
                    "model": model_id,
                    "usage": {
                        "input_tokens": 2,
                        "output_tokens": 1,
                        "total_tokens": 3,
                    },
                },
            ),
            (
                "response.completed",
                {
                    "type": "response.completed",
                    "sequence": 3,
                    "request_id": request_id,
                    "model": model_id,
                    "status": "completed",
                    "finish_reason": "completed",
                },
            ),
        ]
        raw = b"".join(
            (
                f"event: {event_type}\n"
                f"id: {request_id}:{index}\n"
                f"data: {json.dumps(value, separators=(',', ':'))}\n\n"
            ).encode("utf-8")
            for index, (event_type, value) in enumerate(values)
        )
        parsed = parse_system_x_stream(
            raw, expected_model_id=model_id
        )
        self.assertEqual(parsed["request_id"], request_id)
        self.assertEqual(
            parsed["terminal_state"], "completed:completed"
        )
        self.assertTrue(parsed["content_present"])
        self.assertEqual(parsed["event_count"], 4)
