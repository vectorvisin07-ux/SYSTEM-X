from __future__ import annotations

import copy
import hashlib
import http.server
import json
import os
import sqlite3
import stat
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from system_x_inspector.errors import InspectorError
from system_x_inspector.machine import build_parser, execute, main
from system_x_inspector.paths import InspectorPaths
from system_x_inspector.records import canonical_json_bytes
from system_x_inspector.service_publication import (
    HandoffEvidence,
    LoopbackJsonClient,
    PreparedPublication,
    PublicEvidence,
    RegistrySnapshot,
    SecretCredential,
    ServiceSnapshot,
    _handoff_authorization_mode,
    build_publication_record,
    correlate_operation_record,
    issue_proof_request,
    observe_public_surface,
    observe_registry,
    prepare_publication_with_convergence_wait,
    publication_result_identity,
    publish_publication_record,
    read_local_credential,
    restoration_requirement,
    validate_publication_record,
)


DIGEST = "a" * 64
BUNDLE = "bundle-" + DIGEST
SELECTED = "sx-gguf-managed-a"
DEFAULT = "sx-gguf-default-a"
REQUEST_ID = "sx_req_" + "b" * 32
KEY_ID = "c" * 32
RAW_KEY = "sxk_v1_" + KEY_ID + "_" + "D" * 43
API_TX = "tx-api-fixture"
ROUTER_TX = "tx-router-fixture"


def handoff_evidence() -> HandoffEvidence:
    return HandoffEvidence(
        record={
            "handoff_id": "handoff-fixture",
            "decision": {"decision_id": "decision-fixture"},
            "inspection": {
                "inspection_id": "inspection-fixture",
                "artifact_identity": "sha256:" + DIGEST,
            },
        },
        result_identity="sha256:" + "1" * 64,
        inspection_result_identity="sha256:" + "2" * 64,
        decision_result_identity="sha256:" + "3" * 64,
        capability_record_identity="sha256:" + "4" * 64,
        capability_binding_identity="sha256:" + "5" * 64,
        target_identity="sha256:" + DIGEST,
        target_name="managed.gguf",
    )


def registry_snapshot(
    *,
    selected: str = SELECTED,
    default: str = DEFAULT,
    default_bundle: str = BUNDLE,
) -> RegistrySnapshot:
    return RegistrySnapshot(
        schema_version=2,
        generation=7,
        bundle_id=BUNDLE,
        model_version_id=selected,
        public_identity_mode="DISTINCT_LOCATION_SCOPED_VERSION",
        capability_manifest_identity="6" * 64,
        progression_evidence=[
            {"generation": 4, "event_type": "model_registered"},
            {"generation": 6, "event_type": "capability_ready"},
        ],
        aliases=[],
        default_alias="default",
        default_target=default,
        default_artifact_version_id=default_bundle,
    )


def service_snapshot(log: Path, port: int) -> ServiceSnapshot:
    return ServiceSnapshot(
        profile_identity="sha256:" + "7" * 64,
        host="127.0.0.1",
        port=port,
        base_url=f"http://127.0.0.1:{port}",
        default_alias="default",
        service_transaction_id=API_TX,
        operation_log=log,
        readiness_state="READY",
    )


def public_evidence(
    *,
    default: str = DEFAULT,
    default_bundle: str = BUNDLE,
) -> PublicEvidence:
    return PublicEvidence(
        health_status=200,
        list_status=200,
        detail_status=200,
        aliases=[],
        default_target=default,
        default_artifact_version_id=default_bundle,
        default_warm_target=default,
        default_warm_health="ready",
    )


def operation_record() -> dict[str, object]:
    return {
        "schema": "system-x.operation-record.v1",
        "request_id": REQUEST_ID,
        "key_id": KEY_ID,
        "protocol_family": "system_x",
        "endpoint": "/system/v1/chat",
        "operation": "chat",
        "streamed": False,
        "public_model_id": SELECTED,
        "artifact_version_id": BUNDLE,
        "api_service_transaction_id": API_TX,
        "router_transaction_id": ROUTER_TX,
        "started_utc": "2026-01-01T00:00:00.000000Z",
        "completed_utc": "2026-01-01T00:00:01.000000Z",
        "latency_ms": 1000,
        "http_status": 200,
        "error_code": None,
        "finish_reason": "completed",
        "operation_state": "completed",
        "input_tokens": 8,
        "output_tokens": 2,
    }


def proof_evidence() -> dict[str, object]:
    return {
        "request_id": REQUEST_ID,
        "http_status": 200,
        "operation_state": "completed",
        "finish_reason": "completed",
        "response_model_id": SELECTED,
        "content_nonempty": True,
        "final_content_bytes": 5,
        "final_content_sha256": (
            "sha256:" + hashlib.sha256(b"READY").hexdigest()
        ),
        "input_tokens": 8,
        "output_tokens": 2,
    }


def restoration_evidence() -> dict[str, object]:
    return {
        "default_alias": "default",
        "default_target_before": DEFAULT,
        "default_target_after": DEFAULT,
        "restoration_required": False,
        "restoration_performed": False,
        "final_warm_target": DEFAULT,
        "final_warm_health": "ready",
        "final_public_health": "READY",
    }


class HandoffAuthorizationTest(unittest.TestCase):
    def test_direct_and_qualified_handoff_modes_are_exact(self) -> None:
        direct = {
            "decision": {
                "capability_result": "SUPPORTED",
                "selected_branch": "model-api-gguf",
                "handoff_allowed": True,
                "spawn_allowed": True,
            }
        }
        qualified = {
            "decision": {
                "capability_result": "RUNTIME_SMOKE_REQUIRED",
                "selected_branch": None,
                "handoff_allowed": False,
                "spawn_allowed": False,
            },
            "qualification": {
                "qualification_id": "qualification-fixture",
                "result_identity": "sha256:" + "1" * 64,
                "result_class": "SUPPORTED_FOR_CURRENT_TUPLE",
                "requested_profile": "FULL_PRODUCT",
            },
        }
        self.assertEqual(
            _handoff_authorization_mode(direct), "DIRECT_SUPPORTED"
        )
        self.assertEqual(
            _handoff_authorization_mode(qualified), "QUALIFIED_RUNTIME"
        )
        for changed in (
            {**qualified, "qualification": None},
            {
                **qualified,
                "decision": {
                    **qualified["decision"],
                    "selected_branch": "model-api-gguf",
                },
            },
            {
                **qualified,
                "qualification": {
                    **qualified["qualification"],
                    "result_class": "NOT_SUPPORTED_FOR_CURRENT_TUPLE",
                },
            },
        ):
            with self.subTest(changed=changed):
                self.assertIsNone(_handoff_authorization_mode(changed))


def prepared(log: Path, port: int = 1) -> PreparedPublication:
    return PreparedPublication(
        handoff=handoff_evidence(),
        registry=registry_snapshot(),
        service=service_snapshot(log, port),
        credential=SecretCredential(KEY_ID, RAW_KEY),
        public=public_evidence(),
    )


class PublicationFoundationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "RUNTIME/results/publication").mkdir(
            parents=True, mode=0o700
        )
        self.paths = InspectorPaths.discover(self.root)
        self.log = self.root / "operation.log"
        self.log.write_bytes(b"")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def record(self) -> dict[str, object]:
        item = prepared(self.log)
        operation = operation_record()
        return build_publication_record(
            publication_id="publication-20260101T000000000000Z-0123456789abcdef",
            transaction_id="tx-inspector-fixture",
            prepared=item,
            proof=proof_evidence(),
            operation_record=operation,
            operation_record_identity=(
                "sha256:"
                + hashlib.sha256(canonical_json_bytes(operation)).hexdigest()
            ),
            restoration=restoration_evidence(),
        )

    def test_schema_closure_result_identity_and_tamper(self) -> None:
        record = self.record()
        self.assertEqual(validate_publication_record(record), record)
        self.assertEqual(
            publication_result_identity(record), record["result_identity"]
        )
        tampered = copy.deepcopy(record)
        tampered["request"]["final_content_bytes"] = 6
        with self.assertRaisesRegex(
            InspectorError, "publication result identity"
        ):
            validate_publication_record(tampered)
        extra = copy.deepcopy(record)
        extra["request"]["extra"] = True
        with self.assertRaises(InspectorError):
            validate_publication_record(extra)

    def test_atomic_private_publication_and_collision(self) -> None:
        record = self.record()
        path, identity = publish_publication_record(self.paths, record)
        details = path.lstat()
        self.assertEqual(stat.S_IMODE(details.st_mode), 0o600)
        self.assertEqual(details.st_nlink, 1)
        self.assertEqual(identity, record["result_identity"])
        with self.assertRaisesRegex(InspectorError, "already exists"):
            publish_publication_record(self.paths, record)

    def test_publication_waits_for_transient_registry_convergence(self) -> None:
        prepared_value = object()
        transient = InspectorError(
            "REGISTRY_SCHEMA_UNSUPPORTED",
            "registry generation is still converging",
        )
        with (
            mock.patch(
                "system_x_inspector.service_publication.prepare_publication",
                side_effect=[transient, prepared_value],
            ) as prepare,
            mock.patch("system_x_inspector.service_publication.time.sleep")
            as sleep,
        ):
            result = prepare_publication_with_convergence_wait(
                self.paths,
                "handoff-fixture",
            )
        self.assertIs(result, prepared_value)
        self.assertEqual(prepare.call_count, 2)
        sleep.assert_called_once()

    def test_publication_waits_for_service_readiness_convergence(self) -> None:
        prepared_value = object()
        transient = InspectorError(
            "SERVICE_NOT_READY",
            "service is still warming the handed-off model",
        )
        with (
            mock.patch(
                "system_x_inspector.service_publication.prepare_publication",
                side_effect=[transient, prepared_value],
            ) as prepare,
            mock.patch("system_x_inspector.service_publication.time.sleep")
            as sleep,
        ):
            result = prepare_publication_with_convergence_wait(
                self.paths,
                "handoff-service-readiness-fixture",
            )
        self.assertIs(result, prepared_value)
        self.assertEqual(prepare.call_count, 2)
        sleep.assert_called_once()

    def test_paths_contained_and_machine_parser_input_closed(self) -> None:
        self.assertTrue(
            self.paths.publication_results.is_relative_to(self.root)
        )
        parsed = build_parser().parse_args(
            ["publish-service", "--handoff-id", "handoff-fixture"]
        )
        self.assertEqual(parsed.operation, "publish-service")
        self.assertEqual(parsed.handoff_id, "handoff-fixture")
        with self.assertRaises(InspectorError):
            build_parser().parse_args(["publish-service"])

    def test_machine_success_domain_and_internal_exit_contract(self) -> None:
        record = self.record()
        result_path = (
            self.paths.publication_results
            / f"{record['publication_id']}.json"
        )
        returned = (
            "tx-inspector-fixture",
            record,
            result_path,
            record["result_identity"],
        )
        arguments = build_parser().parse_args(
            [
                "--inspector-root",
                str(self.root),
                "publish-service",
                "--handoff-id",
                "handoff-fixture",
            ]
        )
        with mock.patch(
            "system_x_inspector.machine.publish_service_transaction",
            return_value=returned,
        ):
            status, envelope = execute(arguments)
        self.assertEqual(status, 0)
        self.assertTrue(envelope["ok"])
        with mock.patch(
            "system_x_inspector.machine.publish_service_transaction",
            side_effect=InspectorError("HANDOFF_RESULT_NOT_FOUND", "missing"),
        ):
            self.assertEqual(
                main(
                    [
                        "--inspector-root",
                        str(self.root),
                        "publish-service",
                        "--handoff-id",
                        "handoff-fixture",
                    ]
                ),
                2,
            )
        with mock.patch(
            "system_x_inspector.machine.publish_service_transaction",
            side_effect=RuntimeError("fixture"),
        ):
            self.assertEqual(
                main(
                    [
                        "--inspector-root",
                        str(self.root),
                        "publish-service",
                        "--handoff-id",
                        "handoff-fixture",
                    ]
                ),
                70,
            )


class RegistryCorrelationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.branch = Path(self.temporary.name)
        database_root = self.branch / "RUNTIME/api/database"
        database_root.mkdir(parents=True)
        self.database = database_root / "model_registry.sqlite3"
        self.connection = sqlite3.connect(self.database)
        self._schema()
        self._base_rows()
        self.connection.commit()
        self.connection.close()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE registry_metadata(key TEXT PRIMARY KEY,value TEXT);
            CREATE TABLE artifact_locations(
              relative_root TEXT PRIMARY KEY,current_bundle_id TEXT,present INTEGER,
              physical_manifest_json TEXT,first_seen_utc TEXT,last_seen_utc TEXT);
            CREATE TABLE model_versions(
              model_version_id TEXT PRIMARY KEY,bundle_id TEXT,router_model_id TEXT,
              router_source TEXT,display_name TEXT,state TEXT,
              router_metadata_json TEXT,router_metadata_sha256 TEXT,
              created_utc TEXT,updated_utc TEXT);
            CREATE TABLE model_version_locations(
              model_version_id TEXT,relative_root TEXT,
              created_utc TEXT,updated_utc TEXT);
            CREATE TABLE capability_manifests(
              model_version_id TEXT PRIMARY KEY,manifest_json TEXT,
              manifest_sha256 TEXT,props_payload_sha256 TEXT,observed_utc TEXT);
            CREATE TABLE registry_events(
              event_id TEXT PRIMARY KEY,generation INTEGER,event_type TEXT,
              subject_id TEXT,detail_json TEXT,created_utc TEXT);
            CREATE TABLE aliases(
              alias TEXT PRIMARY KEY,model_version_id TEXT,alias_kind TEXT,
              created_utc TEXT,updated_utc TEXT);
            """
        )

    def _base_rows(self) -> None:
        self.connection.executemany(
            "INSERT INTO registry_metadata VALUES(?,?)",
            [
                ("schema_identity", "system-x.gguf-model-registry.v1"),
                ("schema_version", "2"),
                ("registry_generation", "7"),
            ],
        )
        self.connection.execute(
            "INSERT INTO artifact_locations VALUES(?,?,?,?,?,?)",
            ("managed.gguf", BUNDLE, 1, "[]", "t1", "t2"),
        )
        for model, display in (
            (SELECTED, "managed"),
            (DEFAULT, "default"),
        ):
            self.connection.execute(
                "INSERT INTO model_versions VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    model,
                    BUNDLE,
                    display,
                    "models_dir",
                    display,
                    "READY",
                    "{}",
                    "0" * 64,
                    "t1",
                    "t2",
                ),
            )
        self.connection.execute(
            "INSERT INTO model_version_locations VALUES(?,?,?,?)",
            (SELECTED, "managed.gguf", "t1", "t2"),
        )
        self.connection.execute(
            "INSERT INTO model_version_locations VALUES(?,?,?,?)",
            (DEFAULT, "default.gguf", "t1", "t2"),
        )
        self.connection.execute(
            "INSERT INTO capability_manifests VALUES(?,?,?,?,?)",
            (SELECTED, "{}", "6" * 64, "7" * 64, "t2"),
        )
        self.connection.executemany(
            "INSERT INTO registry_events VALUES(?,?,?,?,?,?)",
            [
                ("e1", 4, "model_registered", SELECTED, "{}", "t1"),
                ("e2", 6, "capability_ready", SELECTED, "{}", "t2"),
            ],
        )
        self.connection.execute(
            "INSERT INTO aliases VALUES(?,?,?,?,?)",
            ("default", DEFAULT, "default", "t1", "t2"),
        )

    def mutate(self, statement: str, parameters: tuple = ()) -> None:
        connection = sqlite3.connect(self.database)
        connection.execute(statement, parameters)
        connection.commit()
        connection.close()

    def test_distinct_location_scoped_version_read_only(self) -> None:
        before = self.database.read_bytes()
        result = observe_registry(self.branch, handoff_evidence())
        self.assertEqual(
            result.public_identity_mode,
            "DISTINCT_LOCATION_SCOPED_VERSION",
        )
        self.assertEqual(result.model_version_id, SELECTED)
        self.assertEqual(result.bundle_id, BUNDLE)
        self.assertEqual(self.database.read_bytes(), before)

    def test_existing_immutable_version_reused(self) -> None:
        self.mutate(
            "DELETE FROM model_version_locations WHERE model_version_id=?",
            (DEFAULT,),
        )
        self.mutate(
            "DELETE FROM model_versions WHERE model_version_id=?",
            (DEFAULT,),
        )
        self.mutate(
            "UPDATE aliases SET model_version_id=? WHERE alias='default'",
            (SELECTED,),
        )
        result = observe_registry(self.branch, handoff_evidence())
        self.assertEqual(
            result.public_identity_mode,
            "EXISTING_IMMUTABLE_VERSION_REUSED",
        )

    def test_replacement_ready_is_valid_ready_progression(self) -> None:
        self.mutate(
            "UPDATE registry_events SET event_type='replacement_ready' "
            "WHERE event_type='capability_ready'"
        )
        result = observe_registry(self.branch, handoff_evidence())
        self.assertEqual(
            [item["event_type"] for item in result.progression_evidence],
            ["model_registered", "replacement_ready"],
        )

    def test_missing_location_bundle_state_manifest_and_ambiguity(self) -> None:
        cases = [
            (
                "DELETE FROM artifact_locations",
                (),
                "REGISTRY_LOCATION_NOT_FOUND",
            ),
            (
                "UPDATE artifact_locations SET present=0",
                (),
                "REGISTRY_LOCATION_NOT_READY",
            ),
            (
                "UPDATE artifact_locations SET current_bundle_id='bundle-"
                + "0" * 64
                + "'",
                (),
                "REGISTRY_BUNDLE_MISMATCH",
            ),
            (
                "UPDATE model_versions SET state='PROBING' "
                "WHERE model_version_id=?",
                (SELECTED,),
                "REGISTRY_MODEL_VERSION_NOT_READY",
            ),
            (
                "DELETE FROM capability_manifests",
                (),
                "CAPABILITY_MANIFEST_NOT_FOUND",
            ),
        ]
        original = self.database.read_bytes()
        for statement, parameters, reason in cases:
            with self.subTest(reason=reason):
                self.database.write_bytes(original)
                self.mutate(statement, parameters)
                with self.assertRaises(InspectorError) as caught:
                    observe_registry(self.branch, handoff_evidence())
                self.assertEqual(caught.exception.reason_code, reason)

    def test_schema_mismatch(self) -> None:
        self.mutate(
            "UPDATE registry_metadata SET value='99' "
            "WHERE key='schema_version'"
        )
        with self.assertRaises(InspectorError) as caught:
            observe_registry(self.branch, handoff_evidence())
        self.assertEqual(
            caught.exception.reason_code, "REGISTRY_SCHEMA_UNSUPPORTED"
        )


class _FixtureHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    selected = SELECTED
    default = DEFAULT
    bundle = BUNDLE
    raw_key = RAW_KEY
    mode = "success"

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send(self, status: int, value: dict[str, object]) -> None:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        request_id = value.get("request_id")
        if isinstance(request_id, str):
            header_id = (
                "sx_req_" + "e" * 32
                if self.mode == "header_mismatch"
                else request_id
            )
            self.send_header("X-System-X-Request-ID", header_id)
        if self.mode == "redirect":
            self.send_header("Location", "http://example.invalid/")
        self.end_headers()
        self.wfile.write(encoded)

    @staticmethod
    def _health() -> dict[str, object]:
        return {
            "request_id": "sx_req_" + "d" * 32,
            "ready": True,
            "service_readiness_state": "READY",
            "registry_status": "ready",
            "recovery_state": "IDLE",
            "default_alias": "default",
            "resolved_public_model_id": DEFAULT,
            "artifact_version_id": BUNDLE,
            "warm_identity": {
                "resolved_public_model_id": DEFAULT,
                "artifact_version_id": BUNDLE,
                "health_state": "ready",
            },
        }

    def do_GET(self) -> None:
        if self.mode == "redirect":
            self._send(302, {"request_id": "sx_req_" + "d" * 32})
            return
        if self.path == "/system/v1/health":
            value = self._health()
            if self.mode == "health_not_ready":
                value["ready"] = False
            self._send(200, value)
            return
        if self.headers.get("Authorization") != "Bearer " + self.raw_key:
            self._send(401, {"request_id": "sx_req_" + "d" * 32})
            return
        if self.path == "/system/v1/models":
            models: list[dict[str, object]] = [
                {
                    "id": self.selected,
                    "aliases": [],
                    "registration_state": "ready",
                    "runtime_state": "unloaded",
                }
            ]
            if self.mode == "catalogue_missing":
                models = []
            self._send(
                200,
                {
                    "request_id": "sx_req_" + "d" * 32,
                    "models": models,
                },
            )
            return
        if self.path == "/system/v1/models/" + self.selected:
            artifact = (
                "bundle-" + "0" * 64
                if self.mode == "detail_mismatch"
                else self.bundle
            )
            detail: dict[str, object] = {
                "public_model_id": self.selected,
                "resolved_model_id": self.selected,
                "artifact_version_id": artifact,
                "state": "ready",
                "runtime_state": "unloaded",
                "capabilities": {"chat": "not_tested"},
            }
            if self.mode == "detail_leak":
                detail["debug"] = "/private/model.gguf"
            self._send(
                200,
                {
                    "request_id": "sx_req_" + "d" * 32,
                    "model": detail,
                },
            )
            return
        self._send(404, {"request_id": "sx_req_" + "d" * 32})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length))
        if (
            self.headers.get("Authorization") != "Bearer " + self.raw_key
            or self.path != "/system/v1/chat"
        ):
            self._send(401, {"request_id": REQUEST_ID})
            return
        model = (
            DEFAULT
            if body.get("model") == DEFAULT
            else SELECTED
        )
        response_model = (
            DEFAULT if self.mode == "response_model_mismatch" else model
        )
        content = "" if self.mode == "empty_final" else "READY"
        self._send(
            200,
            {
                "request_id": REQUEST_ID,
                "status": "completed",
                "model": response_model,
                "output": {"content": content},
                "finish_reason": "completed",
                "usage": {"input_tokens": 8, "output_tokens": 2},
            },
        )


class PublicClientAndProofTest(unittest.TestCase):
    def setUp(self) -> None:
        _FixtureHandler.mode = "success"
        self.server = http.server.ThreadingHTTPServer(
            ("127.0.0.1", 0), _FixtureHandler
        )
        self.thread = threading.Thread(
            target=self.server.serve_forever, daemon=True
        )
        self.thread.start()
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.log = self.root / "operation.log"
        self.log.write_bytes(b"")
        self.prepared = prepared(
            self.log, self.server.server_address[1]
        )
        self.client = LoopbackJsonClient(self.prepared.service)

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temporary.cleanup()

    def test_health_catalogue_detail_and_proxy_bypass(self) -> None:
        before = os.environ.get("HTTP_PROXY")
        os.environ["HTTP_PROXY"] = "http://127.0.0.1:1"
        try:
            result = observe_public_surface(
                self.client,
                self.prepared.credential,
                self.prepared.registry,
            )
        finally:
            if before is None:
                os.environ.pop("HTTP_PROXY", None)
            else:
                os.environ["HTTP_PROXY"] = before
        self.assertEqual(result.health_status, 200)
        self.assertEqual(result.detail_status, 200)

    def test_redirect_auth_catalogue_detail_and_leak_rejections(self) -> None:
        for mode, reason in (
            ("redirect", "PUBLIC_ENDPOINT_INVALID"),
            ("health_not_ready", "PUBLIC_HEALTH_FAILED"),
            ("catalogue_missing", "PUBLIC_MODEL_LIST_FAILED"),
            ("detail_mismatch", "PUBLIC_MODEL_DETAIL_MISMATCH"),
            ("detail_leak", "PUBLIC_MODEL_DETAIL_MISMATCH"),
        ):
            with self.subTest(mode=mode):
                _FixtureHandler.mode = mode
                with self.assertRaises(InspectorError) as caught:
                    observe_public_surface(
                        self.client,
                        self.prepared.credential,
                        self.prepared.registry,
                    )
                self.assertEqual(caught.exception.reason_code, reason)
        _FixtureHandler.mode = "success"
        wrong = SecretCredential("f" * 32, "sxk_v1_" + "f" * 32 + "_" + "G" * 43)
        with self.assertRaises(InspectorError) as caught:
            observe_public_surface(
                self.client, wrong, self.prepared.registry
            )
        self.assertEqual(
            caught.exception.reason_code, "AUTHENTICATION_REJECTED"
        )

    def test_successful_proof_and_bounded_content_evidence(self) -> None:
        result = issue_proof_request(self.client, self.prepared)
        self.assertEqual(result["request_id"], REQUEST_ID)
        self.assertEqual(result["response_model_id"], SELECTED)
        self.assertEqual(result["final_content_bytes"], 5)
        self.assertNotIn("content", result)

    def test_request_identity_model_and_empty_final_rejections(self) -> None:
        for mode, reason in (
            ("header_mismatch", "PUBLIC_REQUEST_FAILED"),
            ("response_model_mismatch", "PUBLIC_REQUEST_MODEL_MISMATCH"),
            ("empty_final", "PUBLIC_REQUEST_EMPTY_FINAL"),
        ):
            with self.subTest(mode=mode):
                _FixtureHandler.mode = mode
                with self.assertRaises(InspectorError) as caught:
                    issue_proof_request(self.client, self.prepared)
                self.assertEqual(caught.exception.reason_code, reason)

    def test_operation_record_correlation_and_mismatches(self) -> None:
        record = operation_record()
        self.log.write_text(
            "INFO system_x_operation "
            + json.dumps(record, sort_keys=True, separators=(",", ":"))
            + "\n",
            encoding="utf-8",
        )
        observed, identity = correlate_operation_record(
            self.prepared, proof_evidence(), start_offset=0
        )
        self.assertEqual(observed, record)
        self.assertRegex(identity, r"^sha256:[0-9a-f]{64}$")
        for key, replacement in (
            ("public_model_id", DEFAULT),
            ("artifact_version_id", "bundle-" + "0" * 64),
            ("key_id", "0" * 32),
        ):
            with self.subTest(key=key):
                changed = operation_record()
                changed[key] = replacement
                self.log.write_text(
                    "INFO system_x_operation "
                    + json.dumps(changed, sort_keys=True, separators=(",", ":"))
                    + "\n",
                    encoding="utf-8",
                )
                with self.assertRaises(InspectorError) as caught:
                    correlate_operation_record(
                        self.prepared, proof_evidence(), start_offset=0
                    )
                self.assertEqual(
                    caught.exception.reason_code,
                    "REQUEST_RECORD_MISMATCH",
                )

    def test_restoration_decision(self) -> None:
        self.assertEqual(
            restoration_requirement(BUNDLE, BUNDLE), (False, False)
        )
        self.assertEqual(
            restoration_requirement(BUNDLE, "bundle-" + "0" * 64),
            (True, False),
        )


class CredentialSafetyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.branch = Path(self.temporary.name)
        self.root = self.branch / "RUNTIME/api/auth/handoff"
        self.root.mkdir(parents=True, mode=0o700)
        self.path = self.root / "local-primary.key"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write(self, mode: int = 0o600) -> None:
        self.path.write_text(RAW_KEY + "\n", encoding="ascii")
        self.path.chmod(mode)

    def test_private_regular_credential_is_memory_only(self) -> None:
        self.write()
        result = read_local_credential(self.branch)
        self.assertEqual(result.key_id, KEY_ID)
        self.assertNotIn(RAW_KEY, repr(result))

    def test_wrong_mode_symlink_and_malformed_rejected(self) -> None:
        self.write(0o644)
        with self.assertRaises(InspectorError):
            read_local_credential(self.branch)
        self.path.unlink()
        target = self.root / "target"
        target.write_text(RAW_KEY + "\n", encoding="ascii")
        target.chmod(0o600)
        self.path.symlink_to(target)
        with self.assertRaises(InspectorError):
            read_local_credential(self.branch)
        self.path.unlink()
        target.unlink()
        self.path.write_text("invalid\n", encoding="ascii")
        self.path.chmod(0o600)
        with self.assertRaises(InspectorError):
            read_local_credential(self.branch)


if __name__ == "__main__":
    unittest.main()
