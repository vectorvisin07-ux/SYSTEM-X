"""Frozen Mini 05.26 compatibility, planner and reasoning contracts."""

from __future__ import annotations

import unittest
from pydantic import ValidationError
from fastapi import FastAPI
from fastapi.testclient import TestClient

from system_x_gguf_api.compatibility_models import (
    build_compatibility_models_router,
    compatibility_model_references,
)
from system_x_gguf_api.authentication import protected_route_family
from system_x_gguf_api.anthropic_schemas import AnthropicMessageRequest
from system_x_gguf_api.anthropic_stream import (
    MessagesEventEncoder,
    MessagesStreamConfiguration,
)
from system_x_gguf_api.finalization_policy import (
    TurnIntent,
    private_chat_template_kwargs,
)
from system_x_gguf_api.model_catalogue import ModelCatalogue, ModelSnapshot
from system_x_gguf_api.openai_schemas import OpenAIChatCompletionRequest
from system_x_gguf_api.operation_records import operation_route_for
from system_x_gguf_api.request_context import new_request_id
from system_x_gguf_api.schemas import ReasoningRequest
from system_x_gguf_api.stream_types import (
    CanonicalStreamEvent,
    CanonicalStreamEventType,
)
from system_x_gguf_api.system_routes import build_system_router


class _CompatibilityStub:
    def __init__(self, family: str) -> None:
        self.family = family

    async def models(self, _request_id: str) -> dict[str, str]:
        return {"family": self.family}


class _PlannerBackend:
    async def active_model_properties(self, _model_id: str) -> dict:
        return {
            "chat_template": "{{ enable_thinking }} <think>",
            "default_generation_settings": {
                "n_ctx": 262_144,
                "n_predict": -1,
                "temperature": 0.6,
            },
        }


class Mini0526Contracts(unittest.TestCase):
    def test_model_detail_keeps_unknown_planner_fields_explicit(self) -> None:
        router = build_system_router(None, None, None)
        route = next(
            item
            for item in router.routes
            if item.path == "/system/v1/models/{model_id}"
        )
        self.assertFalse(route.response_model_exclude_none)

    def test_default_alias_precedes_immutable_without_duplicates(self) -> None:
        self.assertEqual(
            compatibility_model_references(
                [
                    {
                        "aliases": ["default"],
                        "created_utc": "2026-07-24T00:00:00Z",
                        "public_model_id": "sx-gguf-fixture",
                    }
                ]
            ),
            [
                ("default", "2026-07-24T00:00:00Z"),
                ("sx-gguf-fixture", "2026-07-24T00:00:00Z"),
            ],
        )

    def test_canonical_reasoning_request_is_strict_and_bounded(self) -> None:
        self.assertEqual(
            ReasoningRequest(mode="standard").model_dump(exclude_none=True),
            {"mode": "standard"},
        )
        self.assertEqual(
            ReasoningRequest(
                mode="pro_extended", final_answer_reserve_tokens=2048
            ).mode,
            "pro_extended",
        )
        self.assertEqual(
            ReasoningRequest(
                mode="custom",
                budget_tokens=4096,
                final_answer_reserve_tokens=512,
            ).budget_tokens,
            4096,
        )
        with self.assertRaises(ValidationError):
            ReasoningRequest.model_validate(
                {"mode": "custom", "budget_tokens": 0}
            )
        with self.assertRaises(ValidationError):
            ReasoningRequest.model_validate(
                {"mode": "standard", "unknown": True}
            )

    def test_shared_dispatch_depends_only_on_anthropic_version_presence(
        self,
    ) -> None:
        app = FastAPI()

        @app.middleware("http")
        async def identity(request, call_next):
            request.state.system_x_request_id = new_request_id()
            return await call_next(request)

        app.include_router(
            build_compatibility_models_router(
                _CompatibilityStub("openai"),
                _CompatibilityStub("messages"),
            )
        )
        with TestClient(app) as client:
            self.assertEqual(
                client.get("/v1/models").json(), {"family": "openai"}
            )
            self.assertEqual(
                client.get(
                    "/v1/models",
                    headers={"anthropic-version": "2023-06-01"},
                ).json(),
                {"family": "messages"},
            )
            self.assertEqual(
                client.get(
                    "/v1/models", headers={"x-api-key": "fixture"}
                ).json(),
                {"family": "openai"},
            )
        self.assertEqual(
            protected_route_family(
                "GET",
                "/v1/models",
                anthropic_version_present=True,
            ),
            "anthropic",
        )
        self.assertEqual(
            operation_route_for(
                "GET",
                "/v1/models",
                anthropic_version_present=True,
            ).protocol_family,
            "messages_compatible",
        )

    def test_protocol_reasoning_and_output_limit_field_laws(self) -> None:
        common = {
            "model": "default",
            "messages": [{"role": "user", "content": "hello"}],
        }
        equal = OpenAIChatCompletionRequest.model_validate(
            {
                **common,
                "max_tokens": 4096,
                "max_completion_tokens": 4096,
            }
        )
        self.assertEqual(equal.output_limit, 4096)
        with self.assertRaises(ValidationError):
            OpenAIChatCompletionRequest.model_validate(
                {
                    **common,
                    "max_tokens": 4096,
                    "max_completion_tokens": 4095,
                }
            )
        with self.assertRaises(ValidationError):
            OpenAIChatCompletionRequest.model_validate(
                {
                    **common,
                    "max_completion_tokens": 4096,
                    "reasoning_effort": "high",
                }
            )
        with self.assertRaises(ValidationError):
            AnthropicMessageRequest.model_validate(
                {
                    **common,
                    "max_tokens": 4096,
                    "thinking": {
                        "type": "enabled",
                        "budget_tokens": 1024,
                    },
                }
            )
        self.assertEqual(
            private_chat_template_kwargs(
                TurnIntent.NORMAL_TEXT, enable_thinking=True
            ),
            {"enable_thinking": True},
        )
        self.assertEqual(
            private_chat_template_kwargs(
                TurnIntent.STRUCTURED_FINALIZATION,
                enable_thinking=True,
            ),
            {"enable_thinking": False},
        )

    def test_messages_stream_keeps_reasoning_separate(self) -> None:
        request_id = "sx_req_" + ("1" * 32)
        encoder = MessagesEventEncoder(
            MessagesStreamConfiguration(
                request_id=request_id,
                model="sx-gguf-fixture",
                input_tokens=4,
            )
        )
        encoder.accept(
            CanonicalStreamEvent(
                type=CanonicalStreamEventType.STARTED,
                sequence=0,
                request_id=request_id,
                model="sx-gguf-fixture",
                payload={"operation": "chat", "status": "in_progress"},
            )
        )
        reasoning_frames = encoder.accept(
            CanonicalStreamEvent(
                type=CanonicalStreamEventType.REASONING_DELTA,
                sequence=1,
                request_id=request_id,
                model="sx-gguf-fixture",
                payload={"delta": "thought"},
            )
        )
        text_frames = encoder.accept(
            CanonicalStreamEvent(
                type=CanonicalStreamEventType.OUTPUT_TEXT_DELTA,
                sequence=2,
                request_id=request_id,
                model="sx-gguf-fixture",
                payload={"delta": "answer"},
            )
        )
        self.assertIn(b'"type":"thinking_delta"', b"".join(reasoning_frames))
        self.assertIn(b'"type":"text_delta"', b"".join(text_frames))
        self.assertNotIn(b"thought", b"".join(text_frames))


class Mini0526PlannerDetail(unittest.IsolatedAsyncioTestCase):
    async def test_active_props_precede_manifest_without_private_leak(
        self,
    ) -> None:
        catalogue = ModelCatalogue(None, _PlannerBackend(), None)
        snapshot = ModelSnapshot(
            requested_reference="default",
            registry_generation=1,
            public_model_id="sx-gguf-fixture",
            bundle_id="bundle-fixture",
            router_model_id="private-router-id",
            registration_state="READY",
            created_utc="2026-07-24T00:00:00Z",
            aliases=("default",),
            capability_manifest_identity="0" * 64,
            context_bound=131_072,
            chat_template_present=True,
            tool_calling_state="available",
            structured_output_state="available",
            parallel_tool_calling_state="available",
            streaming_state="available",
        )
        detail = await catalogue._planner_detail(
            snapshot, {"private-router-id": "loaded"}
        )
        self.assertEqual(detail.requested_alias, "default")
        self.assertEqual(detail.active_context_window_tokens, 262_144)
        self.assertEqual(detail.context_window_tokens, 262_144)
        self.assertEqual(detail.maximum_output_tokens, None)
        self.assertEqual(
            detail.capabilities.reasoning_control, "available"
        )
        serialized = detail.model_dump_json()
        self.assertNotIn("private-router-id", serialized)
        self.assertNotIn("/home/", serialized)


if __name__ == "__main__":
    unittest.main()
