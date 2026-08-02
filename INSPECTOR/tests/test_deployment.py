"""Isolated unified GGUF deployment state-machine acceptance."""

from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
import shutil
import stat
import tempfile
import unittest

from system_x_inspector.connection_receipt import (
    build_receipt,
    load_current_receipt,
    publish_current_receipt,
    show_connection,
)
from system_x_inspector.deployment import (
    DeploymentInterruption,
    deploy_transaction,
    validate_deploy_input,
    validate_deployment_result,
)
from system_x_inspector.errors import InspectorError
from system_x_inspector.paths import InspectorPaths


INCUMBENT = "sx-gguf-incumbent-0123456789abcdef"
CANDIDATE = "sx-gguf-candidate-0123456789abcdef"
PROFILE_IDENTITY = "sha256:" + "a" * 64
BINDING_IDENTITY = "sha256:" + "b" * 64
KEY_ID = "c" * 32
INCUMBENT_ARTIFACT = "sha256:" + "d" * 64
INCUMBENT_VERSION = "bundle-" + "d" * 64
INCUMBENT_MANIFEST = "sha256:" + "e" * 64
INCUMBENT_LOCATION = "sha256:" + "f" * 64
PROOF_ID = "sx_req_" + "1" * 32


def sha(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


class FixtureAdapter:
    """Tiny deterministic adapter with no model, listener, or process work."""

    def __init__(
        self,
        case: "DeploymentFixture",
        *,
        mode: str,
        decision: str = "RUNTIME_SMOKE_REQUIRED",
    ) -> None:
        self.case = case
        self.mode = mode
        self.decision_result = decision
        self.calls: Counter[str] = Counter()
        self.mutations: Counter[str] = Counter()
        self.default = None if mode == "install-first" else INCUMBENT
        self.warm = self.default
        self.ready = (
            set() if mode == "install-first" else {INCUMBENT}
        )
        self.retired: set[str] = set()
        self.generation = 10
        self.fail_at: str | None = None
        self.rollback_uncertain = False
        self.tamper_child: str | None = None
        self._candidate_artifact = "sha256:" + hashlib.sha256(
            self.case.candidate.read_bytes()
        ).hexdigest()

    def _failure(self, point: str) -> None:
        if self.fail_at == point:
            raise InspectorError(
                f"FIXTURE_{point.upper()}_FAILED",
                f"isolated failure at {point}",
            )

    @property
    def candidate_artifact(self) -> str:
        return self._candidate_artifact

    @property
    def candidate_version(self) -> str:
        return "bundle-" + self.candidate_artifact.removeprefix(
            "sha256:"
        )

    @property
    def candidate_manifest(self) -> str:
        return sha("candidate-manifest")

    def source_snapshot(
        self, paths: InspectorPaths, candidate_name: str
    ) -> dict[str, object]:
        self.calls["source_snapshot"] += 1
        path = paths.intake_root / candidate_name
        if (
            path.parent != paths.intake_root
            or not path.is_file()
            or path.is_symlink()
        ):
            raise InspectorError(
                "DEPLOYMENT_SOURCE_INVALID", "fixture source absent"
            )
        data = path.read_bytes()
        identity = "sha256:" + hashlib.sha256(data).hexdigest()
        return {
            "candidate_name": candidate_name,
            "artifact_identity": identity,
            "size": len(data),
            "snapshot_identity": sha(
                f"{identity}:{len(data)}"
            ),
        }

    def capture_prestate(
        self, paths: InspectorPaths
    ) -> dict[str, object]:
        self.calls["capture_prestate"] += 1
        waiting = self.mode == "install-first"
        return {
            "desired_state": "RUNNING",
            "model_service_state": (
                "WAITING_FOR_MODEL" if waiting else "READY"
            ),
            "ready_model_count": 0 if waiting else 1,
            "default_alias": None if waiting else "default",
            "default_target": None if waiting else INCUMBENT,
            "warm_model_id": None if waiting else INCUMBENT,
            "operating_profile_identity": PROFILE_IDENTITY,
            "capability_binding_identity": BINDING_IDENTITY,
            "non_secret_key_id": KEY_ID,
            "artifact_identity": (
                None if waiting else INCUMBENT_ARTIFACT
            ),
            "artifact_version_id": (
                None if waiting else INCUMBENT_VERSION
            ),
            "capability_manifest_identity": (
                None if waiting else INCUMBENT_MANIFEST
            ),
            "resolved_immutable_model_id": (
                None if waiting else INCUMBENT
            ),
            "managed_location_identity": (
                None if waiting else INCUMBENT_LOCATION
            ),
            "registry_generation": self.generation,
            "recovery_state": "IDLE",
        }

    def inspect(
        self, paths: InspectorPaths, candidate_name: str
    ) -> dict[str, object]:
        self.calls["inspect"] += 1
        self._failure("inspect")
        return {
            "inspection_id": "inspection-fixture",
            "identity": sha("inspection"),
            "artifact_identity": self.candidate_artifact,
            "size": self.case.candidate.stat().st_size,
            "terminal_class": "GGUF",
        }

    def decide(
        self, paths: InspectorPaths, inspection: dict[str, object]
    ) -> dict[str, object]:
        self.calls["decide"] += 1
        self._failure("decide")
        return {
            "decision_id": "decision-fixture",
            "identity": sha("decision"),
            "capability_result": self.decision_result,
        }

    def qualify(
        self,
        paths: InspectorPaths,
        inspection: dict[str, object],
        decision: dict[str, object],
        requested_profile: str,
    ) -> dict[str, object]:
        self.calls["qualify"] += 1
        if decision["capability_result"] == "RUNTIME_SMOKE_REQUIRED":
            self.calls["runtime_qualification"] += 1
        else:
            self.calls["direct_attestation"] += 1
        self._failure("qualify")
        return {
            "qualification_id": "qualification-fixture",
            "identity": sha("qualification"),
            "result_class": "SUPPORTED_FOR_CURRENT_TUPLE",
            "supported_profiles": [requested_profile],
            "artifact_identity": self.candidate_artifact,
            "capability_result": decision["capability_result"],
        }

    def handoff(
        self,
        paths: InspectorPaths,
        source_candidate: str,
        decision: dict[str, object],
        qualification: dict[str, object],
    ) -> dict[str, object]:
        self.calls["handoff"] += 1
        self._failure("handoff")
        if self.mutations["handoff"] == 0:
            self.mutations["handoff"] += 1
        return {
            "handoff_id": "handoff-fixture",
            "identity": sha("handoff"),
            "managed_relative_path": (
                "MODEL/SUPERMODEL/candidate-fixture.gguf"
            ),
        }

    def publish_candidate(
        self, paths: InspectorPaths, handoff: dict[str, object]
    ) -> dict[str, object]:
        self.calls["publication"] += 1
        self._failure("publication")
        if self.mutations["publication"] == 0:
            self.mutations["publication"] += 1
            self.ready.add(CANDIDATE)
            self.generation += 1
        return {
            "publication_id": "publication-fixture",
            "identity": sha("publication"),
            "proof_request_id": PROOF_ID,
            "artifact_identity": self.candidate_artifact,
            "registry_generation": self.generation,
            "candidate_identity": {
                "resolved_immutable_model_id": CANDIDATE,
                "artifact_version_id": self.candidate_version,
                "capability_manifest_identity": (
                    self.candidate_manifest
                ),
            },
        }

    def promote(
        self,
        paths: InspectorPaths,
        candidate_name: str,
        qualification: dict[str, object],
    ) -> dict[str, object]:
        self.calls["promotion"] += 1
        self._failure("promotion")
        if self.default != CANDIDATE:
            self.mutations["promotion"] += 1
            self.default = CANDIDATE
            self.warm = CANDIDATE
            self.generation += 1
        return {
            "promotion_id": "promotion-fixture",
            "identity": sha("promotion"),
            "result_class": "PROMOTION_COMPLETE",
        }

    def retire(
        self,
        paths: InspectorPaths,
        incumbent: dict[str, object],
    ) -> dict[str, object]:
        self.calls["retirement"] += 1
        self._failure("retirement")
        if INCUMBENT not in self.retired:
            self.mutations["retirement"] += 1
            self.ready.discard(INCUMBENT)
            self.retired.add(INCUMBENT)
            self.generation += 1
        return {
            "retirement_id": "retirement-fixture",
            "identity": sha("retirement"),
            "result_class": "RETIREMENT_COMPLETE",
        }

    def _model_values(self, model_id: str) -> tuple[str, str, str]:
        if model_id == CANDIDATE:
            return (
                self.candidate_artifact,
                self.candidate_version,
                self.candidate_manifest,
            )
        return (
            INCUMBENT_ARTIFACT,
            INCUMBENT_VERSION,
            INCUMBENT_MANIFEST,
        )

    def observe_connection(
        self,
        paths: InspectorPaths,
        reference: str = "default",
        proof_request_id: str | None = None,
    ) -> dict[str, object]:
        self.calls["observe"] += 1
        self._failure("observe")
        model_id = self.default if reference == "default" else reference
        if model_id not in self.ready:
            raise InspectorError(
                "CONNECTION_STALE", "fixture model is not READY"
            )
        artifact, version, manifest = self._model_values(model_id)
        return {
            "profile_identity": PROFILE_IDENTITY,
            "public_origin": "http://127.0.0.1:43123",
            "service_available": True,
            "inference_ready": True,
            "service_readiness": "READY",
            "model_service_state": "READY",
            "desired_state": "RUNNING",
            "always_on": True,
            "authentication_required": True,
            "default_alias": "default",
            "default_target": self.default,
            "resolved_immutable_model_id": model_id,
            "artifact_sha256": artifact,
            "artifact_version_id": version,
            "capability_manifest_identity": manifest,
            "model_state": "ready",
            "warm": model_id == self.warm or reference != "default",
            "context_window_tokens": 32768,
            "maximum_output_tokens": None,
            "non_secret_key_id": KEY_ID,
            "capabilities": {
                "protocol_families": [
                    "system_x_native",
                    "openai_compatible",
                    "messages_compatible",
                ],
                "streaming": "available",
                "token_counting": "unknown",
                "reasoning_output": "available",
                "reasoning_control": "available",
                "tool_calling": "available",
                "structured_output": "available",
                "context_window_tokens": 32768,
            },
            "token_count_proof": {
                "operation_exposed": True,
                "proof_performed": True,
                "authenticated": True,
                "http_status": 200,
                "result_valid": True,
                "authoritative_unsupported": False,
            },
            "compatibility_proof": {
                "openai_model_list_http_status": 200,
                "openai_model_list_contains_recommended_model": True,
                "openai_model_list_contains_resolved_model": True,
                "messages_model_list_http_status": 200,
                "messages_model_list_contains_recommended_model": True,
                "messages_model_list_contains_resolved_model": True,
                "messages_token_count_http_status": 200,
                "messages_token_count_result_valid": True,
            },
            "proof": {
                "health_http_status": 200,
                "model_list_http_status": 200,
                "model_detail_http_status": 200,
                "proof_request_id": proof_request_id or PROOF_ID,
                "proof_request_http_status": 200,
                "response_model_matches": True,
                "artifact_version_matches": True,
                "final_content_nonempty": True,
                "operation_record_correlated": True,
            },
            "registry_generation": self.generation,
            "recovery_state": "IDLE",
        }

    def rollback(
        self, paths: InspectorPaths, runtime: dict[str, object]
    ) -> dict[str, object]:
        self.calls["rollback"] += 1
        if self.rollback_uncertain:
            raise InspectorError(
                "DEPLOYMENT_ROLLBACK_OWNERSHIP_UNCERTAIN",
                "fixture rollback ownership uncertain",
            )
        self.mutations["rollback"] += 1
        if self.mode == "install-first":
            self.default = None
            self.warm = None
        else:
            self.default = INCUMBENT
            self.warm = INCUMBENT
            self.ready.add(INCUMBENT)
        return {
            "result_class": "PROMOTION_ROLLED_BACK",
            "reason_code": "DEPLOYMENT_ROLLED_BACK",
            "candidate_residue_absent": False,
            "ownership_certain": True,
        }

    def cleanup_failed(
        self, paths: InspectorPaths, runtime: dict[str, object]
    ) -> dict[str, object]:
        self.calls["cleanup_failed"] += 1
        return {
            "failed_residue_absent": (
                "handoff" not in runtime["child_results"]
            ),
            "ownership_certain": True,
            "source_removal_committed": False,
        }

    def remove_source(
        self, paths: InspectorPaths, source: dict[str, object]
    ) -> dict[str, object]:
        self.calls["remove_source"] += 1
        if self.case.candidate.exists():
            current = self.source_snapshot(
                paths, str(source["candidate_name"])
            )
            if (
                current["artifact_identity"]
                != source["artifact_identity"]
            ):
                raise InspectorError(
                    "DEPLOYMENT_SOURCE_CHANGED",
                    "fixture source changed before cleanup",
                )
            self.case.candidate.unlink()
            self.mutations["source_removal"] += 1
        return {
            "source_removed": True,
            "artifact_identity": source["artifact_identity"],
        }

    def authenticate_child(
        self,
        paths: InspectorPaths,
        name: str,
        projection: dict[str, str],
        data: dict[str, object],
    ) -> dict[str, object]:
        self.calls[f"authenticate_{name}"] += 1
        if self.tamper_child == name:
            return {**data, "identity": sha("tampered")}
        return dict(data)


class DeploymentFixture:
    def __init__(
        self,
        *,
        mode: str,
        decision: str = "RUNTIME_SMOKE_REQUIRED",
    ) -> None:
        self.temporary = Path(
            tempfile.mkdtemp(prefix="inspector-deployment-", dir="/tmp")
        )
        self.root = self.temporary / "INSPECTOR"
        self.root.mkdir(mode=0o700)
        self.paths = InspectorPaths.discover(self.root)
        for path in (
            self.paths.intake_root,
            self.paths.runtime_root,
            self.paths.logs,
            self.paths.locks,
            self.paths.status,
            self.paths.transactions,
            self.paths.deployment_results,
        ):
            path.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.candidate = self.paths.intake_root / "candidate.gguf"
        self.candidate.write_bytes(b"GGUF-isolated-fixture")
        self.adapter = FixtureAdapter(
            self, mode=mode, decision=decision
        )
        self.mode = mode
        self.request = {
            "candidate_name": self.candidate.name,
            "deployment_mode": mode,
            "required_capability_profile": "CORE_CHAT",
            "retirement_policy": "retain-incumbent",
        }
        self.transaction_id = (
            "tx-20260731T000000000000Z-abcdef123456"
        )
        self.deployment_id = (
            "deployment-20260731T000000000000Z-"
            "0123456789abcdef"
        )
        if mode != "install-first":
            baseline = build_receipt(
                self.adapter.observe_connection(
                    self.paths, reference="default"
                ),
                receipt_source="EXISTING_ACCEPTED_READY_BASELINE",
                deployment_id="tx-baseline",
                deployment_result_identity=None,
                recommended_reference="default",
                current_receipt_updated=True,
            )
            publish_current_receipt(
                self.paths,
                baseline,
                expected_previous_identity=None,
            )

    def deploy(self, **kwargs):
        return deploy_transaction(
            self.paths,
            self.request,
            adapter=self.adapter,
            transaction_id_factory=lambda: self.transaction_id,
            deployment_id_factory=lambda: self.deployment_id,
            **kwargs,
        )

    def close(self) -> None:
        shutil.rmtree(self.temporary)


class DeploymentTest(unittest.TestCase):
    def fixture(self, **kwargs) -> DeploymentFixture:
        value = DeploymentFixture(**kwargs)
        self.addCleanup(value.close)
        return value

    def test_closed_input_and_mode_policy(self) -> None:
        valid = {
            "candidate_name": "candidate.gguf",
            "deployment_mode": "add",
            "required_capability_profile": "CORE_CHAT",
            "retirement_policy": "retain-incumbent",
        }
        self.assertEqual(validate_deploy_input(valid), valid)
        for changed in (
            {**valid, "candidate_name": "../candidate.gguf"},
            {**valid, "unexpected": True},
            {
                **valid,
                "deployment_mode": "add",
                "retirement_policy": (
                    "retire-incumbent-after-acceptance"
                ),
            },
            {
                **valid,
                "deployment_mode": "install-first",
                "retirement_policy": (
                    "retire-incumbent-after-acceptance"
                ),
            },
        ):
            with self.subTest(changed=changed):
                with self.assertRaises(InspectorError):
                    validate_deploy_input(changed)

    def assert_complete(
        self, fixture: DeploymentFixture
    ) -> dict[str, object]:
        _tx, record, path, identity = fixture.deploy()
        self.assertEqual(
            record["result_class"], "DEPLOYMENT_COMPLETE"
        )
        self.assertEqual(validate_deployment_result(record), record)
        self.assertEqual(record["result_identity"], identity)
        details = path.lstat()
        self.assertTrue(stat.S_ISREG(details.st_mode))
        self.assertFalse(stat.S_ISLNK(details.st_mode))
        self.assertEqual(stat.S_IMODE(details.st_mode), 0o600)
        self.assertEqual(details.st_nlink, 1)
        self.assertFalse(fixture.candidate.exists())
        status = fixture.paths.status / "current.json"
        self.assertEqual(
            __import__("json").loads(status.read_text())["state"],
            "IDLE",
        )
        return record

    def assert_corrected_receipt(
        self, receipt: dict[str, object]
    ) -> None:
        openai = receipt["connections"]["openai_compatible"]
        messages = receipt["connections"]["messages_compatible"]
        self.assertEqual(
            openai["endpoint_semantics"], "base_url_relative"
        )
        self.assertEqual(
            openai["endpoints"],
            {
                "models": "/models",
                "completions": "/completions",
                "chat_completions": "/chat/completions",
                "responses": "/responses",
            },
        )
        self.assertNotIn(
            "/v1/v1",
            openai["base_url"] + openai["endpoints"]["models"],
        )
        self.assertEqual(
            messages["endpoints"],
            {
                "models": "/v1/models",
                "messages": "/v1/messages",
                "count_tokens": "/v1/messages/count_tokens",
            },
        )
        self.assertEqual(
            messages["required_headers"],
            {"anthropic-version": "2023-06-01"},
        )
        self.assertEqual(
            receipt["capabilities"]["token_counting"], "available"
        )
        self.assertEqual(
            receipt["proof"]["messages_token_count_http_status"], 200
        )
        self.assertIs(
            receipt["proof"]["messages_token_count_result_valid"], True
        )

    def test_install_first_success(self) -> None:
        fixture = self.fixture(mode="install-first")
        record = self.assert_complete(fixture)
        self.assertEqual(fixture.adapter.default, CANDIDATE)
        current = load_current_receipt(fixture.paths)
        self.assertEqual(
            current["model"]["recommended_reference"], "default"
        )
        self.assertEqual(
            current["model"]["resolved_immutable_model_id"], CANDIDATE
        )
        self.assertTrue(
            current["lifecycle"]["current_receipt_updated"]
        )
        self.assertIsNotNone(
            current["deployment_result_identity"]
        )
        self.assert_corrected_receipt(current)
        ready = show_connection(
            fixture.paths,
            observer=fixture.adapter.observe_connection,
        )
        self.assertEqual(ready["result_class"], "CONNECTION_READY")
        self.assertIsNotNone(record["child_results"]["promotion"])

    def test_add_keeps_default_and_current_receipt(self) -> None:
        fixture = self.fixture(mode="add")
        previous = load_current_receipt(fixture.paths)
        record = self.assert_complete(fixture)
        self.assertEqual(fixture.adapter.default, INCUMBENT)
        self.assertEqual(
            load_current_receipt(fixture.paths), previous
        )
        self.assertEqual(
            record["connection_receipt"]["model"][
                "recommended_reference"
            ],
            CANDIDATE,
        )
        self.assertFalse(
            record["connection_receipt"]["lifecycle"][
                "current_receipt_updated"
            ]
        )
        self.assert_corrected_receipt(record["connection_receipt"])
        self.assertIsNone(record["child_results"]["promotion"])
        self.assertIsNone(record["child_results"]["retirement"])

    def test_replace_default_retain_incumbent(self) -> None:
        fixture = self.fixture(mode="replace-default")
        record = self.assert_complete(fixture)
        self.assertEqual(fixture.adapter.default, CANDIDATE)
        self.assertIn(INCUMBENT, fixture.adapter.ready)
        self.assertIsNotNone(record["child_results"]["promotion"])
        self.assertIsNone(record["child_results"]["retirement"])
        current = load_current_receipt(fixture.paths)
        self.assertEqual(
            current["model"]["resolved_immutable_model_id"], CANDIDATE
        )
        self.assert_corrected_receipt(current)

    def test_replace_default_retire_incumbent(self) -> None:
        fixture = self.fixture(mode="replace-default")
        fixture.request["retirement_policy"] = (
            "retire-incumbent-after-acceptance"
        )
        record = self.assert_complete(fixture)
        self.assertEqual(fixture.adapter.default, CANDIDATE)
        self.assertNotIn(INCUMBENT, fixture.adapter.ready)
        self.assertIn(INCUMBENT, fixture.adapter.retired)
        self.assertIsNotNone(record["child_results"]["retirement"])
        self.assert_corrected_receipt(
            load_current_receipt(fixture.paths)
        )

    def test_runtime_qualification_and_exact_supported_attestation(self) -> None:
        smoke = self.fixture(mode="add")
        self.assert_complete(smoke)
        self.assertEqual(
            smoke.adapter.calls["runtime_qualification"], 1
        )
        self.assertEqual(
            smoke.adapter.calls["direct_attestation"], 0
        )

        direct = self.fixture(mode="add", decision="SUPPORTED")
        self.assert_complete(direct)
        self.assertEqual(
            direct.adapter.calls["runtime_qualification"], 0
        )
        self.assertEqual(
            direct.adapter.calls["direct_attestation"], 1
        )

    def test_post_promotion_failure_rolls_back(self) -> None:
        fixture = self.fixture(mode="replace-default")
        previous = load_current_receipt(fixture.paths)
        fixture.adapter.fail_at = "observe"
        _tx, record, _path, _identity = fixture.deploy()
        self.assertEqual(
            record["result_class"], "DEPLOYMENT_ROLLED_BACK"
        )
        self.assertEqual(fixture.adapter.default, INCUMBENT)
        self.assertEqual(fixture.adapter.calls["rollback"], 1)
        self.assertEqual(
            load_current_receipt(fixture.paths), previous
        )
        self.assertTrue(fixture.candidate.exists())

    def test_pre_handoff_failure_is_clean(self) -> None:
        fixture = self.fixture(mode="add")
        fixture.adapter.fail_at = "inspect"
        _tx, record, _path, _identity = fixture.deploy()
        self.assertEqual(
            record["result_class"], "DEPLOYMENT_FAILED_CLEAN"
        )
        self.assertEqual(fixture.adapter.default, INCUMBENT)
        self.assertTrue(fixture.candidate.exists())

    def test_uncertain_rollback_fails_closed(self) -> None:
        fixture = self.fixture(mode="replace-default")
        fixture.adapter.fail_at = "observe"
        fixture.adapter.rollback_uncertain = True
        _tx, record, _path, _identity = fixture.deploy()
        self.assertEqual(
            record["result_class"], "DEPLOYMENT_FAIL_CLOSED"
        )
        self.assertFalse(record["cleanup"]["ownership_certain"])
        self.assertTrue(fixture.candidate.exists())

    def test_receipt_failure_after_result_commit_fails_closed(self) -> None:
        fixture = self.fixture(mode="replace-default")

        def fail_receipt(*_args, **_kwargs):
            raise InspectorError(
                "CONNECTION_STATUS_CAS_CONFLICT",
                "isolated receipt failure",
            )

        with self.assertRaises(InspectorError) as caught:
            fixture.deploy(receipt_publisher=fail_receipt)
        self.assertEqual(
            caught.exception.reason_code, "DEPLOYMENT_FAIL_CLOSED"
        )
        transaction = __import__("json").loads(
            (
                fixture.paths.transactions
                / f"{fixture.transaction_id}.json"
            ).read_text()
        )
        self.assertEqual(transaction["state"], "FAIL_CLOSED")
        connection = show_connection(
            fixture.paths,
            observer=fixture.adapter.observe_connection,
        )
        self.assertEqual(
            connection["result_class"], "CONNECTION_STALE"
        )

    def test_crash_reentry_does_not_repeat_committed_children(self) -> None:
        boundaries = (
            ("add", "CLASSIFIED", False),
            ("add", "QUALIFIED", False),
            ("add", "REGISTERED", False),
            ("add", "CANDIDATE_REQUEST_PROVEN", False),
            ("replace-default", "DEFAULT_PROMOTED", False),
            ("replace-default", "INCUMBENT_RETIRED", True),
            (
                "replace-default",
                "CONNECTION_RECEIPT_PUBLISHED",
                False,
            ),
        )
        for mode, boundary, retire in boundaries:
            with self.subTest(boundary=boundary):
                fixture = self.fixture(mode=mode)
                if retire:
                    fixture.request["retirement_policy"] = (
                        "retire-incumbent-after-acceptance"
                    )
                crashed = False

                def interrupt(
                    state: str, _transaction: dict[str, object]
                ) -> None:
                    nonlocal crashed
                    if not crashed and state == boundary:
                        crashed = True
                        raise DeploymentInterruption(boundary)

                with self.assertRaises(DeploymentInterruption):
                    fixture.deploy(transition_observer=interrupt)
                _tx, record, _path, _identity = fixture.deploy()
                self.assertEqual(
                    record["result_class"], "DEPLOYMENT_COMPLETE"
                )
                for step in (
                    "handoff",
                    "publication",
                    "promotion",
                    "retirement",
                    "source_removal",
                ):
                    self.assertLessEqual(
                        fixture.adapter.mutations[step],
                        1,
                        step,
                    )

    def test_candidate_change_and_child_tamper_fail_closed(self) -> None:
        changed = self.fixture(mode="add")
        interrupted = False

        def stop_after_inspection(
            state: str, _transaction: dict[str, object]
        ) -> None:
            nonlocal interrupted
            if not interrupted and state == "CLASSIFIED":
                interrupted = True
                raise DeploymentInterruption(state)

        with self.assertRaises(DeploymentInterruption):
            changed.deploy(transition_observer=stop_after_inspection)
        changed.candidate.write_bytes(b"changed-isolated-fixture")
        with self.assertRaises(InspectorError) as caught:
            changed.deploy()
        self.assertEqual(
            caught.exception.reason_code, "DEPLOYMENT_SOURCE_CHANGED"
        )

        tampered = self.fixture(mode="add")
        interrupted = False

        def stop_after_qualification(
            state: str, _transaction: dict[str, object]
        ) -> None:
            nonlocal interrupted
            if not interrupted and state == "QUALIFIED":
                interrupted = True
                raise DeploymentInterruption(state)

        with self.assertRaises(DeploymentInterruption):
            tampered.deploy(
                transition_observer=stop_after_qualification
            )
        tampered.adapter.tamper_child = "qualification"
        _tx, record, _path, _identity = tampered.deploy()
        self.assertEqual(
            record["result_class"], "DEPLOYMENT_FAIL_CLOSED"
        )
        self.assertFalse(record["cleanup"]["ownership_certain"])

    def test_completed_duplicate_returns_same_result(self) -> None:
        fixture = self.fixture(mode="add")
        first = self.assert_complete(fixture)
        lifecycle = (
            "capture_prestate",
            "inspect",
            "decide",
            "qualify",
            "handoff",
            "publication",
            "promotion",
            "retirement",
            "observe",
            "rollback",
            "cleanup_failed",
            "remove_source",
        )
        before_calls = {
            name: fixture.adapter.calls[name]
            for name in lifecycle
        }
        before_mutations = dict(fixture.adapter.mutations)
        _tx, second, _path, _identity = fixture.deploy()
        self.assertEqual(second, first)
        self.assertEqual(
            {
                name: fixture.adapter.calls[name]
                for name in lifecycle
            },
            before_calls,
        )
        self.assertEqual(
            dict(fixture.adapter.mutations), before_mutations
        )


if __name__ == "__main__":
    unittest.main()
