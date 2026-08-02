from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import stat
import tempfile
import unittest
from pathlib import Path

from system_x_inspector.connection_receipt import (
    build_legacy_repair_candidate,
    build_receipt,
    complete_compatibility_proof,
    derive_token_counting,
    join_base_relative,
    load_current_receipt,
    load_legacy_current_receipt_for_repair,
    normalize_base_url,
    normalized_openai_base,
    publish_current_receipt,
    receipt_identity,
    render_connection,
    show_connection,
    validate_receipt,
)
from system_x_inspector.errors import InspectorError
from system_x_inspector.paths import InspectorPaths


class ConnectionReceiptTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = Path(
            tempfile.mkdtemp(prefix="inspector-connection-", dir="/tmp")
        )
        self.root = self.temporary / "INSPECTOR"
        self.root.mkdir(mode=0o700)
        self.paths = InspectorPaths.discover(self.root)
        for path in (
            self.paths.status,
            self.paths.deployment_results,
        ):
            path.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.observation = {
            "public_origin": "http://127.0.0.1:56259",
            "service_available": True,
            "inference_ready": True,
            "service_readiness": "READY",
            "model_service_state": "READY",
            "desired_state": "RUNNING",
            "always_on": True,
            "authentication_required": True,
            "default_alias": "default",
            "default_target": (
                "sx-gguf-current-model-0123456789abcdef"
            ),
            "resolved_immutable_model_id": (
                "sx-gguf-current-model-0123456789abcdef"
            ),
            "artifact_sha256": "sha256:" + "1" * 64,
            "artifact_version_id": "bundle-" + "1" * 64,
            "capability_manifest_identity": "sha256:" + "2" * 64,
            "model_state": "ready",
            "warm": True,
            "context_window_tokens": 131072,
            "maximum_output_tokens": None,
            "non_secret_key_id": "3" * 32,
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
                "context_window_tokens": 131072,
            },
            "token_count_proof": self.token_proof_available(),
            "compatibility_proof": self.compatibility_proof(),
            "proof": {
                "health_http_status": 200,
                "model_list_http_status": 200,
                "model_detail_http_status": 200,
                "proof_request_id": "sx_req_" + "4" * 32,
                "proof_request_http_status": 200,
                "response_model_matches": True,
                "artifact_version_matches": True,
                "final_content_nonempty": True,
                "operation_record_correlated": True,
            },
            "registry_generation": 7,
            "recovery_state": "IDLE",
        }

    def tearDown(self) -> None:
        shutil.rmtree(self.temporary)

    @staticmethod
    def token_proof_available() -> dict[str, object]:
        return {
            "operation_exposed": True,
            "proof_performed": True,
            "authenticated": True,
            "http_status": 200,
            "result_valid": True,
            "authoritative_unsupported": False,
        }

    @staticmethod
    def compatibility_proof() -> dict[str, object]:
        return {
            "openai_model_list_http_status": 200,
            "openai_model_list_contains_recommended_model": True,
            "openai_model_list_contains_resolved_model": True,
            "messages_model_list_http_status": 200,
            "messages_model_list_contains_recommended_model": True,
            "messages_model_list_contains_resolved_model": True,
            "messages_token_count_http_status": 200,
            "messages_token_count_result_valid": True,
        }

    def receipt(
        self,
        *,
        receipt_source: str = "EXISTING_ACCEPTED_READY_BASELINE",
        receipt_id: str = (
            "connection-20260730T230000000000Z-0123456789abcdef"
        ),
        observation: dict[str, object] | None = None,
    ) -> dict[str, object]:
        return build_receipt(
            observation or self.observation,
            receipt_source=receipt_source,
            deployment_id="tx-20260730T230000000000Z-123456789abc",
            deployment_result_identity=(
                "sha256:" + "9" * 64
                if receipt_source == "DEPLOY_GGUF"
                else None
            ),
            recommended_reference="default",
            current_receipt_updated=True,
            receipt_id_factory=lambda: receipt_id,
        )

    def write_current(self, receipt: dict[str, object]) -> None:
        self.paths.current_connection_status.write_bytes(
            json.dumps(
                receipt,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
            + b"\n"
        )
        self.paths.current_connection_status.chmod(0o600)

    def legacy_receipt(self) -> dict[str, object]:
        legacy = copy.deepcopy(self.receipt())
        for name in ("system_x_native", "openai_compatible"):
            item = legacy["connections"][name]
            item.pop("endpoint_semantics")
            item["compatibility_version"] = None
        messages = legacy["connections"]["messages_compatible"]
        messages.pop("endpoint_semantics")
        messages.pop("required_headers")
        legacy["connections"]["openai_compatible"]["endpoints"] = {
            "models": "/v1/models",
            "completions": "/v1/completions",
            "chat_completions": "/v1/chat/completions",
            "responses": "/v1/responses",
        }
        messages["endpoints"] = {
            "models": "/system/v1/models",
            "messages": "/v1/messages",
            "count_tokens": "/system/v1/tokens/count",
        }
        legacy["capabilities"]["token_counting"] = "not_exposed"
        for name in self.compatibility_proof():
            legacy["proof"].pop(name)
        legacy["receipt_identity"] = receipt_identity(legacy)
        return legacy

    def test_closed_routes_headers_and_url_composition(self) -> None:
        receipt = self.receipt()
        self.assertEqual(validate_receipt(receipt), receipt)
        native = receipt["connections"]["system_x_native"]
        openai = receipt["connections"]["openai_compatible"]
        messages = receipt["connections"]["messages_compatible"]
        self.assertEqual(native["endpoint_semantics"], "base_url_relative")
        self.assertEqual(openai["base_url"], "http://127.0.0.1:56259/v1")
        self.assertEqual(
            openai["endpoints"],
            {
                "models": "/models",
                "completions": "/completions",
                "chat_completions": "/chat/completions",
                "responses": "/responses",
            },
        )
        self.assertEqual(messages["base_url"], "http://127.0.0.1:56259")
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
        self.assertFalse(
            any(
                path.startswith("/system/v1/")
                for path in messages["endpoints"].values()
            )
        )
        openai_models = join_base_relative(
            openai["base_url"], openai["endpoints"]["models"]
        )
        self.assertEqual(openai_models, "http://127.0.0.1:56259/v1/models")
        self.assertNotIn("/v1/v1", openai_models)
        self.assertEqual(
            join_base_relative(
                messages["base_url"],
                messages["endpoints"]["count_tokens"],
            ),
            "http://127.0.0.1:56259/v1/messages/count_tokens",
        )

    def test_url_normalization_rejects_ambiguous_or_secret_material(self) -> None:
        self.assertEqual(
            normalize_base_url("http://127.0.0.1:56259/"),
            "http://127.0.0.1:56259",
        )
        self.assertEqual(
            normalized_openai_base("http://127.0.0.1:56259/v1/"),
            "http://127.0.0.1:56259/v1",
        )
        for value in (
            "http://user@127.0.0.1:56259",
            "http://127.0.0.1:56259?secret=value",
            "http://127.0.0.1:56259#fragment",
            "http://127.0.0.1:56259/v1/v1",
            "http://127.0.0.1:56259//",
        ):
            with self.subTest(value=value), self.assertRaises(InspectorError):
                normalized_openai_base(value)
        for endpoint in ("//models", "/v1//models", "https://host/models"):
            with self.subTest(endpoint=endpoint), self.assertRaises(InspectorError):
                join_base_relative("http://127.0.0.1:56259/v1", endpoint)
        private = copy.deepcopy(self.receipt())
        private["connections"]["messages_compatible"]["base_url"] = (
            "http://127.0.0.1:54037"
        )
        private["receipt_identity"] = receipt_identity(private)
        with self.assertRaises(InspectorError):
            validate_receipt(private)

    def test_token_count_state_is_derived_only_from_physical_proof(self) -> None:
        self.assertEqual(
            derive_token_counting(self.token_proof_available()),
            "available",
        )
        self.assertEqual(
            derive_token_counting(
                {
                    "operation_exposed": True,
                    "proof_performed": False,
                    "authenticated": None,
                    "http_status": None,
                    "result_valid": None,
                    "authoritative_unsupported": False,
                }
            ),
            "not_tested",
        )
        self.assertEqual(
            derive_token_counting(
                {
                    "operation_exposed": True,
                    "proof_performed": True,
                    "authenticated": True,
                    "http_status": 400,
                    "result_valid": False,
                    "authoritative_unsupported": True,
                }
            ),
            "unavailable",
        )
        self.assertEqual(
            derive_token_counting(
                {
                    "operation_exposed": False,
                    "proof_performed": False,
                    "authenticated": None,
                    "http_status": None,
                    "result_valid": None,
                    "authoritative_unsupported": False,
                }
            ),
            "not_exposed",
        )
        fabricated_zero = self.token_proof_available()
        fabricated_zero["result_valid"] = False
        with self.assertRaises(InspectorError):
            derive_token_counting(fabricated_zero)
        with self.assertRaises(InspectorError):
            derive_token_counting({"state": "unknown"})

    def test_renderer_follows_receipt_source(self) -> None:
        baseline = render_connection(self.receipt())
        deployed = render_connection(
            self.receipt(
                receipt_source="DEPLOY_GGUF",
                receipt_id=(
                    "connection-20260730T230000000000Z-"
                    "fedcba9876543210"
                ),
            )
        )
        self.assertTrue(baseline.startswith("SYSTEM X API CONNECTION READY"))
        self.assertTrue(
            deployed.startswith("SYSTEM X GGUF DEPLOYMENT COMPLETE")
        )
        self.assertIn("RAW API KEY:\n  not printed", baseline)
        self.assertNotIn("secret-value", baseline)

    def test_schema_is_closed_over_corrected_shapes(self) -> None:
        schema_path = (
            Path(__file__).parents[1]
            / "schemas/api-connection-receipt.schema.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(
            schema["$id"],
            "system-x.inspector-api-connection-receipt.v1",
        )
        definitions = schema["$defs"]
        self.assertEqual(
            definitions["openAICompatibleConnection"]["properties"]
            ["endpoints"]["properties"]["models"]["const"],
            "/models",
        )
        self.assertEqual(
            definitions["messagesCompatibleConnection"]["properties"]
            ["required_headers"]["properties"]["anthropic-version"]
            ["const"],
            "2023-06-01",
        )
        self.assertEqual(
            definitions["capabilities"]["properties"]["token_counting"]
            ["enum"],
            ["available", "not_tested", "unavailable", "not_exposed"],
        )

    def test_current_publication_cas_duplicate_and_show_are_read_only(self) -> None:
        receipt = self.receipt()
        identity = publish_current_receipt(
            self.paths,
            receipt,
            expected_previous_identity=None,
        )
        self.assertEqual(identity, receipt["receipt_identity"])
        details = self.paths.current_connection_status.lstat()
        self.assertTrue(stat.S_ISREG(details.st_mode))
        self.assertEqual(stat.S_IMODE(details.st_mode), 0o600)
        self.assertEqual(details.st_nlink, 1)
        before = self.paths.current_connection_status.read_bytes()
        result = show_connection(
            self.paths,
            observer=lambda *_args, **_kwargs: self.observation,
        )
        self.assertEqual(result["result_class"], "CONNECTION_READY")
        self.assertEqual(result["stored_receipt_identity"], identity)
        self.assertEqual(self.paths.current_connection_status.read_bytes(), before)
        self.assertEqual(load_current_receipt(self.paths), receipt)
        self.assertEqual(
            publish_current_receipt(
                self.paths,
                receipt,
                expected_previous_identity=None,
            ),
            identity,
        )
        alternate = self.receipt(
            receipt_id=(
                "connection-20260730T230000000000Z-"
                "1111111111111111"
            )
        )
        with self.assertRaises(InspectorError) as caught:
            publish_current_receipt(
                self.paths,
                alternate,
                expected_previous_identity=None,
            )
        self.assertEqual(
            caught.exception.reason_code,
            "CONNECTION_STATUS_CAS_CONFLICT",
        )

    def test_known_legacy_receipt_migrates_once_through_cas(self) -> None:
        legacy = self.legacy_receipt()
        self.write_current(legacy)
        loaded = load_legacy_current_receipt_for_repair(
            self.paths,
            expected_identity=legacy["receipt_identity"],
        )
        self.assertEqual(loaded, legacy)
        candidate = build_legacy_repair_candidate(
            legacy,
            self.observation,
            receipt_id_factory=lambda: (
                "connection-20260730T230000000000Z-"
                "2222222222222222"
            ),
        )
        self.assertEqual(
            candidate["capabilities"]["token_counting"],
            "not_tested",
        )
        self.assertIsNone(
            candidate["proof"]["messages_token_count_http_status"]
        )
        completed = complete_compatibility_proof(
            candidate,
            self.compatibility_proof(),
            receipt_id_factory=lambda: (
                "connection-20260730T230000000000Z-"
                "3333333333333333"
            ),
        )
        with self.assertRaises(InspectorError) as caught:
            publish_current_receipt(
                self.paths,
                completed,
                expected_previous_identity="sha256:" + "0" * 64,
            )
        self.assertEqual(
            caught.exception.reason_code,
            "CONNECTION_STATUS_CAS_CONFLICT",
        )
        identity = publish_current_receipt(
            self.paths,
            completed,
            expected_previous_identity=legacy["receipt_identity"],
        )
        self.assertEqual(identity, completed["receipt_identity"])
        self.assertEqual(load_current_receipt(self.paths), completed)
        self.assertEqual(
            publish_current_receipt(
                self.paths,
                completed,
                expected_previous_identity=legacy["receipt_identity"],
            ),
            identity,
        )

    def test_stale_invalid_secret_symlink_and_wrong_mode_fail_closed(self) -> None:
        receipt = self.receipt()
        publish_current_receipt(
            self.paths,
            receipt,
            expected_previous_identity=None,
        )
        stale = copy.deepcopy(self.observation)
        stale["artifact_version_id"] = "bundle-" + "5" * 64
        result = show_connection(
            self.paths,
            observer=lambda *_args, **_kwargs: stale,
        )
        self.assertEqual(result["result_class"], "CONNECTION_STALE")
        prohibited = copy.deepcopy(receipt)
        prohibited["authentication"]["raw_api_key"] = "secret-value"
        with self.assertRaises(InspectorError) as caught:
            validate_receipt(prohibited)
        self.assertEqual(caught.exception.reason_code, "CONNECTION_RECORD_INVALID")
        invalid_route = copy.deepcopy(receipt)
        invalid_route["connections"]["messages_compatible"]["endpoints"][
            "models"
        ] = "/system/v1/models"
        invalid_route["receipt_identity"] = receipt_identity(invalid_route)
        self.write_current(invalid_route)
        self.assertEqual(
            show_connection(self.paths)["result_class"],
            "CONNECTION_RECORD_INVALID",
        )
        self.paths.current_connection_status.unlink()
        target = self.paths.status / "target.json"
        target.write_text("{}\n", encoding="utf-8")
        target.chmod(0o600)
        os.symlink(target, self.paths.current_connection_status)
        with self.assertRaises(InspectorError):
            load_current_receipt(self.paths)
        self.paths.current_connection_status.unlink()
        self.write_current(receipt)
        self.paths.current_connection_status.chmod(0o644)
        with self.assertRaises(InspectorError):
            load_current_receipt(self.paths)


if __name__ == "__main__":
    unittest.main()
