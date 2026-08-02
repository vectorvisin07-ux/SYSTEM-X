from __future__ import annotations

import json
from pathlib import Path
import stat
import tempfile
import unittest

from system_x_inspector.errors import InspectorError
from system_x_inspector.paths import BranchHandoffPaths, InspectorPaths
from system_x_inspector.promotion import (
    PromotionAuthorization,
    PromotionIncumbent,
    authenticate_promotion_qualification,
    build_promotion_record,
    capture_promotion_incumbent,
    load_promotion_record,
    promote_transaction,
    publish_promotion_record,
)
from system_x_inspector.qualification import (
    IncumbentSnapshot,
    qualification_result_identity,
)


class PromotionFoundationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "INSPECTOR"
        self.root.mkdir()
        self.paths = InspectorPaths.discover(self.root)
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
            self.paths.staging,
            self.paths.tmp,
            self.paths.capability_root,
            self.paths.capability_records,
            self.paths.capability_bindings,
        ):
            path.mkdir(parents=True, exist_ok=True, mode=0o700)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def failed_clean_qualification() -> dict[str, object]:
        basis: dict[str, object] = {
            "schema_version": (
                "system-x.inspector-gguf-qualification-result.v1"
            ),
            "qualification_id": (
                "qualification-20260730T000000000000Z-0123456789abcdef"
            ),
            "transaction_id": "tx-isolated",
            "created_utc": "2026-07-30T00:00:00Z",
            "completed_utc": "2026-07-30T00:00:01Z",
            "inspection": {},
            "input_decision": {},
            "requested_profile": "CORE_CHAT",
            "installed_tuple": {},
            "incumbent": {},
            "candidate_runtime": {},
            "checks": [],
            "supported_profiles": [],
            "observed_capabilities": {},
            "result_class": "QUALIFICATION_FAILED_CLEAN",
            "reason_codes": ["QUALIFICATION_FAILED_CLEAN"],
            "restoration": {},
            "cleanup": {},
            "validity_predicate": {},
        }
        record = {**basis, "result_identity": ""}
        record["result_identity"] = qualification_result_identity(record)
        return record

    @staticmethod
    def failed_clean_promotion() -> dict[str, object]:
        return build_promotion_record(
            promotion_id=(
                "promotion-20260730T000000000000Z-fedcba9876543210"
            ),
            transaction_id="tx-isolated",
            created_utc="2026-07-30T00:00:00Z",
            completed_utc="2026-07-30T00:00:01Z",
            qualification={},
            installed_tuple={},
            candidate={},
            incumbent={},
            states_observed=["FAILED_CLEAN"],
            registry_progression=[],
            alias_promotion={},
            pre_promotion_proofs=[],
            post_promotion_proofs=[],
            stability_observation={},
            restart_verification={},
            rollback={},
            service_final={},
            result_class="PROMOTION_FAILED_CLEAN",
            reason_codes=["PROMOTION_FAILED_CLEAN"],
            validity_predicate={},
        )

    def test_non_promotable_class_rejected_before_source_access(self) -> None:
        record = self.failed_clean_qualification()
        path = (
            self.paths.qualification_results
            / f"{record['qualification_id']}.json"
        )
        path.write_text(
            json.dumps(record, sort_keys=True, separators=(",", ":"))
        )
        path.chmod(0o600)
        with self.assertRaises(InspectorError) as observed:
            authenticate_promotion_qualification(
                self.paths,
                str(record["qualification_id"]),
                "absent-candidate.gguf",
            )
        self.assertEqual(
            observed.exception.reason_code,
            "PROMOTION_QUALIFICATION_NOT_PROMOTABLE",
        )
        self.assertEqual(observed.exception.exit_status, 2)

    def test_result_publication_is_private_immutable_and_round_trips(self) -> None:
        record = self.failed_clean_promotion()
        path = publish_promotion_record(self.paths, record)
        details = path.lstat()
        self.assertTrue(stat.S_ISREG(details.st_mode))
        self.assertFalse(stat.S_ISLNK(details.st_mode))
        self.assertEqual(stat.S_IMODE(details.st_mode), 0o600)
        self.assertEqual(details.st_nlink, 1)
        self.assertEqual(
            load_promotion_record(self.paths, str(record["promotion_id"])),
            record,
        )
        self.assertEqual(publish_promotion_record(self.paths, record), path)
        changed = {**record, "completed_utc": "2026-07-30T00:00:02Z"}
        with self.assertRaises(InspectorError):
            publish_promotion_record(self.paths, changed)

    def test_result_rejects_private_fields(self) -> None:
        record = self.failed_clean_promotion()
        record["candidate"] = {"private_router_url": "http://127.0.0.1"}
        with self.assertRaises(InspectorError) as observed:
            publish_promotion_record(self.paths, record)
        self.assertEqual(
            observed.exception.reason_code, "PROMOTION_RESULT_INVALID"
        )

    def test_incumbent_snapshot_has_exact_rollback_identity(self) -> None:
        branch_root = self.root.parent / "model-api-gguf"
        managed = branch_root / "MODEL/SUPERMODEL"
        staging = branch_root / "RUNTIME/api/replacement-staging"
        managed.mkdir(parents=True)
        staging.mkdir(parents=True)
        branch_paths = BranchHandoffPaths(
            system_x_root=self.root.parent,
            branch_root=branch_root,
            managed_root=managed,
            branch_staging_root=staging,
        )
        snapshot = IncumbentSnapshot(
            present=True,
            default_alias="default",
            public_model_id="sx-gguf-incumbent-0123456789ab",
            artifact_version_id="bundle-" + "1" * 64,
            capability_manifest_identity="sha256:" + "2" * 64,
            managed_location_identity="sha256:" + "3" * 64,
            warm_before={
                "public_model_id": "sx-gguf-incumbent-0123456789ab",
                "artifact_version_id": "bundle-" + "1" * 64,
                "capability_manifest_identity": "sha256:" + "2" * 64,
                "health_state": "ready",
            },
            registry_generation=9,
            credential_key_id="a" * 32,
            profile_identity="sha256:" + "4" * 64,
            service_readiness="READY",
            recovery_state="IDLE",
            api_service_transaction_id="api-tx",
            router_transaction_id="router-tx",
            model_child_identity={"identity": "child"},
            historical_registry_locations=("incumbent.gguf",),
        )
        incumbent = capture_promotion_incumbent(
            self.paths,
            branch_paths,
            snapshotter=lambda _paths, _branch: snapshot,
        )
        self.assertEqual(incumbent.artifact_identity, "sha256:" + "1" * 64)
        self.assertRegex(
            incumbent.alias_binding_identity, r"^sha256:[0-9a-f]{64}$"
        )


class IsolatedPromotionAdapter:
    def __init__(self, *, fail_at: str | None = None) -> None:
        self.fail_at = fail_at
        self.default = "model-incumbent"
        self.generation = 10
        self.event_identity = "sha256:" + "9" * 64
        self.restart_count = 0
        self.epochs = {
            "manager": "manager-1",
            "supervisor": "supervisor-1",
            "api": "api-1",
            "router": "router-1",
            "model_child": "child-1",
        }
        self.crash_after_restart = False

    @staticmethod
    def fail(reason: str, point: str) -> None:
        raise InspectorError(reason, f"isolated failure at {point}")

    def stage_candidate(self, authorization, incumbent, transaction_id):
        if self.fail_at == "stage":
            self.fail("PROMOTION_STAGING_FAILED", "stage")
        return {
            "artifact_identity": authorization.artifact_identity,
            "managed_relative_path": "MODEL/SUPERMODEL/candidate.gguf",
            "relative_root": "candidate.gguf",
            "publication_identity": "sha256:" + "8" * 64,
            "publication_size": 32,
            "public_model_id": None,
            "artifact_version_id": None,
            "capability_manifest_identity": None,
            "registry_generation": None,
            "registry_states_observed": [],
        }

    def wait_candidate(self, candidate):
        if self.fail_at == "wait":
            self.fail("PROMOTION_REGISTRATION_FAILED", "wait")
        self.generation = 11
        return {
            **candidate,
            "public_model_id": "model-candidate",
            "artifact_version_id": "bundle-" + "b" * 64,
            "capability_manifest_identity": "sha256:" + "d" * 64,
            "registry_generation": self.generation,
            "registry_states_observed": [
                "REGISTERED",
                "PROBING",
                "READY",
            ],
            "default_bound": False,
        }

    def prove_candidate(self, candidate, requested_profile, *, use_default):
        point = "post" if use_default else "pre"
        if self.fail_at == point:
            self.fail(
                (
                    "PROMOTION_POST_REQUEST_FAILED"
                    if use_default
                    else "PROMOTION_CANDIDATE_REQUEST_FAILED"
                ),
                point,
            )
        return {
            "passed": True,
            "requested_model_reference": (
                "default" if use_default else candidate["public_model_id"]
            ),
            "resolved_public_model_id": candidate["public_model_id"],
            "requested_profile": requested_profile,
            "protocols": [
                "system-x",
                "openai",
                "messages",
                "streaming",
            ],
        }

    def alias_transaction(self, request):
        if request["action"] == "promote":
            if self.fail_at == "alias":
                self.fail("PROMOTION_ALIAS_CAS_FAILED", "alias")
            self.assert_request(
                request,
                expected="model-incumbent",
                new="model-candidate",
                generation=11,
            )
            self.default = "model-candidate"
            self.generation = 12
            return {
                "action": "promote",
                "alias": "default",
                "alias_event_identity": self.event_identity,
                "changed": True,
                "new_registry_generation": 12,
                "new_target": "model-candidate",
                "previous_target": "model-incumbent",
                "promotion_transaction_id": request[
                    "promotion_transaction_id"
                ],
            }
        if self.fail_at == "rollback_alias":
            self.fail(
                "PROMOTION_ROLLBACK_ALIAS_CONFLICT",
                "rollback_alias",
            )
        self.assert_request(
            request,
            expected="model-candidate",
            new="model-incumbent",
            generation=12,
        )
        if (
            request["promotion_alias_event_identity"]
            != self.event_identity
        ):
            self.fail(
                "PROMOTION_ROLLBACK_ALIAS_CONFLICT",
                "rollback_event",
            )
        self.default = "model-incumbent"
        self.generation = 13
        return {
            "action": "rollback",
            "alias": "default",
            "alias_event_identity": "sha256:" + "7" * 64,
            "changed": True,
            "new_registry_generation": 13,
            "new_target": "model-incumbent",
            "previous_target": "model-candidate",
            "promotion_transaction_id": request[
                "promotion_transaction_id"
            ],
        }

    @staticmethod
    def assert_request(request, *, expected, new, generation):
        assert request["alias"] == "default"
        assert request["expected_current_target"] == expected
        assert request["new_target"] == new
        assert request["expected_registry_generation"] == generation

    def observe_exact(self, candidate, *, expected_default):
        if (
            self.fail_at == "warm"
            and expected_default == "model-candidate"
        ):
            return {"exact": False}
        exact = self.default == expected_default
        return {
            "exact": exact,
            "registry_generation": self.generation,
            "default_alias": "default",
            "default_target": self.default,
            "artifact_version_id": (
                candidate.get("artifact_version_id")
                if self.default == "model-candidate"
                else "bundle-" + "a" * 64
            ),
            "service_readiness": "READY",
            "inference_ready": True,
            "recovery_state": "IDLE",
            "fail_closed_latch": False,
            "warm_public_model_id": self.default,
            "api_listener_owned": True,
            "router_listener_owned": True,
        }

    def stability_parameters(self):
        if self.fail_at == "stability":
            self.fail("PROMOTION_STABILITY_FAILED", "stability")
        return 3, 0.0

    def pause(self, seconds):
        assert seconds == 0.0

    def capture_epochs(self):
        return dict(self.epochs)

    def restart_and_verify(self, candidate, baseline, *, resume):
        changed = all(
            self.epochs[key] != baseline[key] for key in self.epochs
        )
        if not (resume and changed):
            self.restart_count += 1
            self.epochs = {
                key: f"{key}-{self.restart_count + 1}"
                for key in self.epochs
            }
        if self.crash_after_restart:
            self.crash_after_restart = False
            raise SystemExit("isolated crash after manager restart")
        if self.fail_at == "restart":
            self.fail("PROMOTION_RESTART_FAILED", "restart")
        return {
            "passed": True,
            "resumed_without_duplicate_restart": resume and changed,
            "epochs_before": baseline,
            "epochs_after": dict(self.epochs),
            "epoch_changes": {key: True for key in self.epochs},
            "later_default_request_passed": True,
        }

    def restore_incumbent(self, incumbent):
        self.default = "model-incumbent"
        return {
            "proved": True,
            "default_target": self.default,
            "service_readiness": "READY",
            "recovery_state": "IDLE",
        }

    def retain_candidate(self, candidate):
        return {
            "disposition": "RETAIN_NON_DEFAULT",
            "ownership_certain": True,
            "candidate_state": "READY",
        }


class PromotionOrchestrationTest(PromotionFoundationTest):
    def setUp(self) -> None:
        super().setUp()
        candidate = self.paths.intake_root / "candidate.gguf"
        candidate.write_bytes(b"GGUF" + b"x" * 28)
        details = candidate.stat()
        self.authorization = PromotionAuthorization(
            qualification={
                "qualification_id": (
                    "qualification-20260730T000000000000Z-"
                    "0123456789abcdef"
                ),
                "result_identity": "sha256:" + "1" * 64,
                "result_class": "SUPPORTED_FOR_CURRENT_TUPLE",
                "requested_profile": "CORE_CHAT",
                "inspection": {
                    "inspection_id": (
                        "inspection-20260730T000000000000Z-"
                        "0123456789abcdef"
                    ),
                    "artifact_identity": "sha256:" + "3" * 64,
                },
                "validity_predicate": {
                    "predicate_identity": "sha256:" + "2" * 64
                },
            },
            qualification_path=self.paths.qualification_results / "q.json",
            candidate_name="candidate.gguf",
            candidate_path=candidate,
            candidate_snapshot={
                "device": details.st_dev,
                "inode": details.st_ino,
                "mode": stat.S_IMODE(details.st_mode),
                "link_count": 1,
                "size": details.st_size,
                "mtime_ns": details.st_mtime_ns,
                "artifact_identity": "sha256:" + "3" * 64,
                "snapshot_identity": "sha256:" + "4" * 64,
            },
            inspection={},
            installed_tuple={
                "branch_capability_record_identity": "sha256:" + "5" * 64,
                "capability_binding_identity": "sha256:" + "6" * 64,
                "installed_tuple_verification_identity": "sha256:"
                + "7" * 64,
                "llama_cpp_commit": "a" * 40,
                "llama_server_sha256": "8" * 64,
                "connected_source_identity": "sha256:" + "9" * 64,
            },
        )
        snapshot = IncumbentSnapshot(
            present=True,
            default_alias="default",
            public_model_id="model-incumbent",
            artifact_version_id="bundle-" + "a" * 64,
            capability_manifest_identity="sha256:" + "c" * 64,
            managed_location_identity="sha256:" + "e" * 64,
            warm_before={
                "public_model_id": "model-incumbent",
                "artifact_version_id": "bundle-" + "a" * 64,
                "capability_manifest_identity": "sha256:" + "c" * 64,
                "health_state": "ready",
            },
            registry_generation=10,
            credential_key_id="f" * 32,
            profile_identity="sha256:" + "0" * 64,
            service_readiness="READY",
            recovery_state="IDLE",
            api_service_transaction_id="api-1",
            router_transaction_id="router-1",
            model_child_identity={"identity": "child-1"},
            historical_registry_locations=("incumbent.gguf",),
        )
        self.incumbent = PromotionIncumbent(
            snapshot=snapshot,
            artifact_identity="sha256:" + "a" * 64,
            alias_binding_identity="sha256:" + "b" * 64,
            relative_root="incumbent.gguf",
        )
        self.ids = iter(
            [
                "tx-isolated-success",
                "tx-isolated-second",
                "tx-isolated-third",
            ]
        )

    def run_promotion(self, adapter, **overrides):
        return promote_transaction(
            self.paths,
            str(self.authorization.qualification["qualification_id"]),
            "candidate.gguf",
            adapter=adapter,
            authorization_factory=lambda *_args, **_kwargs: (
                self.authorization
            ),
            installed_tuple_resolver=lambda *_args: (
                self.authorization.installed_tuple
            ),
            transaction_id_factory=lambda: next(self.ids),
            promotion_id_factory=lambda: (
                "promotion-20260730T000000000000Z-"
                "fedcba9876543210"
            ),
            incumbent_factory=lambda _paths: self.incumbent,
            **overrides,
        )

    def test_success_stability_restart_and_idempotence(self) -> None:
        adapter = IsolatedPromotionAdapter()
        transaction, record, path, identity = self.run_promotion(adapter)
        self.assertEqual(record["result_class"], "PROMOTION_COMPLETE")
        self.assertEqual(record["states_observed"][-1], "COMPLETE")
        self.assertEqual(
            record["stability_observation"]["consecutive_samples"], 3
        )
        self.assertTrue(record["restart_verification"]["passed"])
        self.assertEqual(adapter.restart_count, 1)
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        second = self.run_promotion(adapter)
        self.assertEqual(second[0], transaction)
        self.assertEqual(second[3], identity)
        self.assertEqual(adapter.restart_count, 1)

    def test_post_alias_failure_rolls_back_exact_incumbent(self) -> None:
        adapter = IsolatedPromotionAdapter(fail_at="post")
        _tx, record, _path, _identity = self.run_promotion(adapter)
        self.assertEqual(record["result_class"], "PROMOTION_ROLLED_BACK")
        self.assertEqual(record["states_observed"][-1], "ROLLED_BACK")
        self.assertEqual(adapter.default, "model-incumbent")
        self.assertEqual(
            record["rollback"]["candidate_disposition"]["disposition"],
            "RETAIN_NON_DEFAULT",
        )

    def test_restart_crash_reentry_does_not_restart_twice(self) -> None:
        adapter = IsolatedPromotionAdapter()
        adapter.crash_after_restart = True
        with self.assertRaises(SystemExit):
            self.run_promotion(adapter)
        self.assertEqual(adapter.restart_count, 1)
        _tx, record, _path, _identity = self.run_promotion(adapter)
        self.assertEqual(record["result_class"], "PROMOTION_COMPLETE")
        self.assertEqual(adapter.restart_count, 1)
        self.assertTrue(
            record["restart_verification"][
                "resumed_without_duplicate_restart"
            ]
        )

    def test_rollback_alias_conflict_fails_closed(self) -> None:
        adapter = IsolatedPromotionAdapter(fail_at="post")
        original = adapter.alias_transaction

        def alias(request):
            if request["action"] == "rollback":
                adapter.fail_at = "rollback_alias"
            return original(request)

        adapter.alias_transaction = alias
        with self.assertRaises(InspectorError) as observed:
            self.run_promotion(adapter)
        self.assertEqual(
            observed.exception.reason_code, "PROMOTION_FAIL_CLOSED"
        )
        status = json.loads(
            (self.paths.status / "current.json").read_text()
        )
        self.assertEqual(status["state"], "FAIL_CLOSED")


if __name__ == "__main__":
    unittest.main()
