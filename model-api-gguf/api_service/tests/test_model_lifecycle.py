from __future__ import annotations

import unittest
from types import SimpleNamespace

from pydantic import ValidationError

from system_x_gguf_api.anthropic_adapter import AnthropicCompatibilityAdapter
from system_x_gguf_api.anthropic_errors import (
    SYSTEM_ERROR_MAP as ANTHROPIC_ERROR_MAP,
)
from system_x_gguf_api.errors import SystemXError
from system_x_gguf_api.application import app
from system_x_gguf_api.model_catalogue import ModelCatalogue
from system_x_gguf_api.model_lifecycle import (
    ModelLifecycleEvidence,
    ModelServiceState,
    resolve_model_service_state,
)
from system_x_gguf_api.openai_adapter import OpenAICompatibilityAdapter
from system_x_gguf_api.openai_errors import SYSTEM_ERROR_MAP as OPENAI_ERROR_MAP
from system_x_gguf_api.schemas import HealthResponse


class ModelLifecycleResolverTests(unittest.TestCase):
    def resolve(self, **changes: object):
        return resolve_model_service_state(ModelLifecycleEvidence(**changes))

    def test_stopped_has_strongest_precedence(self) -> None:
        result = self.resolve(
            desired_state="STOPPED",
            fail_closed_latch=True,
            ownership_uncertain=True,
            registry_available=False,
        )
        self.assertEqual(result.state, ModelServiceState.STOPPED)
        self.assertFalse(result.service_available)
        self.assertFalse(result.recovery_eligible)

    def test_ownership_uncertainty_fails_closed(self) -> None:
        result = self.resolve(ownership_uncertain=True)
        self.assertEqual(result.state, ModelServiceState.FAIL_CLOSED)

    def test_registry_failure_is_not_waiting(self) -> None:
        result = self.resolve(registry_available=False)
        self.assertEqual(result.state, ModelServiceState.DEGRADED)
        self.assertEqual(result.reason_code, "REGISTRY_UNAVAILABLE")

    def test_empty_healthy_registry_waits_without_recovery(self) -> None:
        result = self.resolve()
        self.assertEqual(result.state, ModelServiceState.WAITING_FOR_MODEL)
        self.assertEqual(result.reason_code, "NO_READY_MODEL")
        self.assertTrue(result.service_available)
        self.assertFalse(result.inference_ready)
        self.assertFalse(result.model_expected)
        self.assertFalse(result.recovery_eligible)

    def test_first_candidate_loads_without_becoming_expected(self) -> None:
        result = self.resolve(candidate_model_count=1)
        self.assertEqual(
            result.state, ModelServiceState.MODEL_CANDIDATE_LOADING
        )
        self.assertFalse(result.model_expected)
        self.assertFalse(result.recovery_eligible)

    def test_ready_requires_exact_default_and_warm_health(self) -> None:
        result = self.resolve(
            resolved_default_alias="default",
            resolved_public_model_id="sx-model",
            default_target_ready=True,
            warm_identity_present=True,
            exact_target_warm_healthy=True,
            ready_public_model_count=1,
            candidate_model_count=1,
        )
        self.assertEqual(result.state, ModelServiceState.READY)
        self.assertTrue(result.inference_ready)
        self.assertTrue(result.model_expected)

    def test_expected_ready_target_loss_remains_recoverable(self) -> None:
        result = self.resolve(
            resolved_default_alias="default",
            resolved_public_model_id="sx-model",
            default_target_ready=True,
        )
        self.assertEqual(result.state, ModelServiceState.DEGRADED)
        self.assertTrue(result.model_expected)
        self.assertTrue(result.recovery_eligible)

    def test_warm_identity_loss_remains_recoverable(self) -> None:
        result = self.resolve(warm_identity_present=True)
        self.assertEqual(result.state, ModelServiceState.DEGRADED)
        self.assertTrue(result.recovery_eligible)

    def test_healthy_incumbent_wins_over_unrelated_candidate(self) -> None:
        result = self.resolve(
            resolved_default_alias="default",
            resolved_public_model_id="incumbent",
            default_target_ready=True,
            warm_identity_present=True,
            exact_target_warm_healthy=True,
            ready_public_model_count=1,
            candidate_model_count=2,
        )
        self.assertEqual(result.state, ModelServiceState.READY)

    def test_ready_without_default_is_degraded_not_arbitrarily_selected(self) -> None:
        result = self.resolve(ready_public_model_count=1)
        self.assertEqual(result.state, ModelServiceState.DEGRADED)
        self.assertEqual(result.reason_code, "READY_MODEL_WITHOUT_DEFAULT")

    def test_conflicting_alias_evidence_fails_closed(self) -> None:
        result = self.resolve(resolved_public_model_id="orphan")
        self.assertEqual(result.state, ModelServiceState.FAIL_CLOSED)


class HealthSchemaTests(unittest.TestCase):
    def test_waiting_health_has_separate_availability_and_readiness(self) -> None:
        value = HealthResponse(
            request_id="sx_req_" + "1" * 32,
            service_name="system-x-gguf-api",
            service_readiness_state="WAITING_FOR_MODEL",
            model_service_state="WAITING_FOR_MODEL",
            service_available=True,
            inference_ready=False,
            ready=False,
            service_status="ready",
            contract_version="v1",
            backend_status="router_ready",
            backend_process_running=True,
            backend_control_plane_ready=True,
            loaded_model_count=0,
            model_ready=False,
            environment_name="test",
            registry_status="ready",
            registered_model_count=0,
            ready_model_count=0,
            candidate_model_count=0,
            rejected_artifact_count=0,
            registry_generation=1,
            default_alias="default",
            configured_default_alias="default",
            resolved_default_alias=None,
            resolved_public_model_id=None,
            artifact_version_id=None,
            reason_code="NO_READY_MODEL",
            recovery_state="IDLE",
        )
        self.assertTrue(value.service_available)
        self.assertFalse(value.inference_ready)
        self.assertFalse(value.ready)

    def test_candidate_count_cannot_be_negative(self) -> None:
        with self.assertRaises(ValidationError):
            HealthResponse.model_validate(
                {
                    "request_id": "sx_req_" + "1" * 32,
                    "service_name": "system-x-gguf-api",
                    "service_readiness_state": "WAITING_FOR_MODEL",
                    "model_service_state": "WAITING_FOR_MODEL",
                    "service_available": True,
                    "inference_ready": False,
                    "ready": False,
                    "service_status": "ready",
                    "contract_version": "v1",
                    "backend_status": "router_ready",
                    "backend_process_running": True,
                    "backend_control_plane_ready": True,
                    "loaded_model_count": 0,
                    "model_ready": False,
                    "environment_name": "test",
                    "registry_status": "ready",
                    "registered_model_count": 0,
                    "ready_model_count": 0,
                    "candidate_model_count": -1,
                    "rejected_artifact_count": 0,
                    "registry_generation": 1,
                    "default_alias": "default",
                    "configured_default_alias": "default",
                }
            )


class _EmptyRegistry:
    settings = SimpleNamespace(registry_default_alias="default")

    async def resolve_public_model(self, _reference: str):
        return {"resolution": "not_found", "registry_generation": 7}

    async def public_model_rows(self):
        return {"registry_generation": 7, "models": []}


class _NoopBackend:
    async def current_router_inventory(self):
        raise RuntimeError("no model inventory required for empty list")


class _Operations:
    def __init__(self) -> None:
        self.terminal: list[str] = []

    def note_terminal(self, request_id: str, **_values: object) -> None:
        self.terminal.append(request_id)


class EmptyCatalogueAndErrorTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.operations = _Operations()
        self.catalogue = ModelCatalogue(
            _EmptyRegistry(), _NoopBackend(), self.operations
        )

    async def test_default_absence_is_retryable_no_ready_503(self) -> None:
        with self.assertRaises(SystemXError) as captured:
            await self.catalogue.resolve("default")
        self.assertEqual(captured.exception.status_code, 503)
        self.assertEqual(
            captured.exception.code, "system_x_no_ready_model"
        )
        self.assertTrue(captured.exception.retryable)

    async def test_arbitrary_unknown_model_remains_404(self) -> None:
        with self.assertRaises(SystemXError) as captured:
            await self.catalogue.resolve("sx-unknown-immutable")
        self.assertEqual(captured.exception.status_code, 404)
        self.assertEqual(
            captured.exception.code, "system_x_model_not_found"
        )

    async def test_all_three_empty_catalogue_shapes_are_valid(self) -> None:
        native = await self.catalogue.list_models("sx_req_" + "2" * 32)
        inference = SimpleNamespace(operations=self.operations)
        openai = await OpenAICompatibilityAdapter(
            self.catalogue, inference
        ).models("sx_req_" + "3" * 32)
        messages = await AnthropicCompatibilityAdapter(
            self.catalogue, inference
        ).models("sx_req_" + "4" * 32)
        self.assertEqual(native.registry_generation, 7)
        self.assertEqual(native.models, [])
        self.assertEqual(openai.object, "list")
        self.assertEqual(openai.data, [])
        self.assertEqual(messages.data, [])
        self.assertIsNone(messages.first_id)
        self.assertIsNone(messages.last_id)

    def test_no_ready_protocol_mappings_are_exact(self) -> None:
        self.assertEqual(
            OPENAI_ERROR_MAP["system_x_no_ready_model"],
            (503, "server_error", "no_ready_model", "model"),
        )
        self.assertEqual(
            ANTHROPIC_ERROR_MAP["system_x_no_ready_model"],
            (503, "api_error"),
        )


class OpenAPIModelLifecycleTests(unittest.TestCase):
    def test_openapi_exposes_health_axis_and_all_no_ready_503_surfaces(
        self,
    ) -> None:
        document = app.openapi()
        health = document["components"]["schemas"]["HealthResponse"]
        required = set(health["required"])
        lifecycle_fields = {
            "service_available",
            "inference_ready",
            "ready",
            "model_service_state",
            "configured_default_alias",
            "candidate_model_count",
        }
        self.assertTrue(lifecycle_fields <= required)
        properties = health["properties"]
        for name in (
            "resolved_default_alias",
            "resolved_public_model_id",
            "artifact_version_id",
        ):
            self.assertIn(name, properties)
        model_state_schema = properties["model_service_state"]
        self.assertEqual(
            set(model_state_schema["enum"]),
            {
                "WAITING_FOR_MODEL",
                "MODEL_CANDIDATE_LOADING",
                "READY",
                "DEGRADED",
                "STOPPED",
                "FAIL_CLOSED",
            },
        )
        for path in (
            "/system/v1/generate",
            "/system/v1/chat",
            "/system/v1/responses",
            "/system/v1/tokens/count",
            "/v1/completions",
            "/v1/chat/completions",
            "/v1/responses",
            "/v1/messages",
            "/v1/messages/count_tokens",
        ):
            self.assertIn("503", document["paths"][path]["post"]["responses"])
        self.assertEqual(
            document["paths"]["/system/v1/health"]["get"]["security"], []
        )


if __name__ == "__main__":
    unittest.main()
