"""Focused regression coverage for canonical chat finalization."""
from __future__ import annotations

from contextlib import asynccontextmanager
import asyncio
import json
from types import SimpleNamespace
import unittest

from pydantic import ValidationError

from system_x_gguf_api.anthropic_adapter import AnthropicCompatibilityAdapter
from system_x_gguf_api.anthropic_schemas import AnthropicMessageRequest
from system_x_gguf_api.errors import SystemXError
from system_x_gguf_api.finalization_policy import (
    TurnIntent,
    private_chat_template_kwargs,
)
from system_x_gguf_api.inference_service import InferenceService
from system_x_gguf_api.openai_adapter import OpenAICompatibilityAdapter
from system_x_gguf_api.openai_schemas import (
    OpenAIChatCompletionRequest,
    OpenAIResponsesRequest,
)
from system_x_gguf_api.response_normalizer import (
    ResponseNormalizationError,
    normalize_chat_turn,
)
from system_x_gguf_api.router_client import RouterObservation
from system_x_gguf_api.schemas import (
    ChatMessage,
    ChatRequest,
    ReasoningRequest,
    ResponsesRequest,
)


class _Operations:
    def __init__(self) -> None:
        self.models = []
        self.routers = []
        self.terminals = []

    def note_model(self, request_id, model, bundle) -> None:
        self.models.append((request_id, model, bundle))

    def note_router(self, request_id, router_transaction) -> None:
        self.routers.append((request_id, router_transaction))

    def note_terminal(self, request_id, **values) -> None:
        self.terminals.append((request_id, values))


class _Router:
    def __init__(self) -> None:
        self.chat_calls = []
        self.responses_calls = []

    @staticmethod
    def _observation(payload):
        return RouterObservation(200, json.dumps(payload), payload, None)

    async def chat_input_tokens(
        self, _model_id, _messages, _tools, _template_kwargs
    ):
        return self._observation({"input_tokens": 3})

    async def responses_input_tokens(
        self, _model_id, _input, _instructions
    ):
        return self._observation({"input_tokens": 3})

    async def chat_completion(
        self,
        _model_id,
        _messages,
        max_output_tokens,
        _temperature,
        _stop,
        _tools,
        _choice,
        _format,
        template_kwargs,
    ):
        self.chat_calls.append((max_output_tokens, template_kwargs))
        if len(self.chat_calls) == 1:
            payload = {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "reasoning_content": "bounded reasoning",
                    },
                    "finish_reason": "length",
                }],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": max_output_tokens,
                    "total_tokens": max_output_tokens + 3,
                },
            }
        else:
            payload = {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "final answer",
                    },
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": max_output_tokens,
                    "total_tokens": max_output_tokens + 3,
                },
            }
        return self._observation(payload)

    async def responses(
        self,
        _model_id,
        _input,
        max_output_tokens,
        _instructions,
        _temperature,
        _tools,
        _choice,
        _format,
        template_kwargs,
    ):
        self.responses_calls.append((max_output_tokens, template_kwargs))
        if len(self.responses_calls) == 1:
            payload = {
                "status": "completed",
                "output": [{
                    "id": "rs_0123456789abcdef0123456789abcdef",
                    "type": "reasoning",
                    "summary": [],
                    "content": [{
                        "type": "reasoning_text",
                        "text": "bounded reasoning",
                    }],
                    "encrypted_content": "",
                    "status": "completed",
                }],
                "usage": {
                    "input_tokens": 3,
                    "output_tokens": max_output_tokens,
                    "total_tokens": max_output_tokens + 3,
                },
            }
        else:
            payload = {
                "status": "completed",
                "output": [{
                    "id": "msg_0123456789abcdef0123456789abcdef",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{
                        "type": "output_text",
                        "text": "final response",
                    }],
                }],
                "usage": {
                    "input_tokens": 3,
                    "output_tokens": max_output_tokens,
                    "total_tokens": max_output_tokens + 3,
                },
            }
        return self._observation(payload)


class _Catalogue:
    def __init__(self) -> None:
        self.resolve_calls = 0
        self.snapshot = SimpleNamespace(
            public_model_id="sx-gguf-test",
            router_model_id="router-model",
            bundle_id="bundle-test",
            chat_template_present=True,
        )

    async def resolve(self, _model):
        self.resolve_calls += 1
        return self.snapshot

    async def reasoning_template_control_available(self, _snapshot):
        return True

    def verify(self, _snapshot):
        return None

    def mark_operation_proven(self, _model, _operation):
        return None


class _Backend:
    settings = None

    def __init__(self, router: _Router) -> None:
        self.router = router

    @asynccontextmanager
    async def inference_session(self, _model_id, _verify):
        yield SimpleNamespace(
            router=self.router,
            router_model_id="router-model",
            router_identity=SimpleNamespace(
                transaction_id="router-tx",
                pid=101,
                pgid=101,
                sid=101,
                process_start_identity="start-101",
            ),
        )


def _service():
    router = _Router()
    catalogue = _Catalogue()
    operations = _Operations()
    service = InferenceService(
        catalogue, _Backend(router), operations
    )
    return service, catalogue, router, operations


class ChatFinalizationTests(unittest.TestCase):
    def test_canonical_policy_defaults_and_preserves_explicit_intents(self):
        service, _catalogue, _router, _operations = _service()
        snapshot = SimpleNamespace()
        ordinary = asyncio.run(
            service._reasoning_template_kwargs(
                snapshot, TurnIntent.NORMAL_TEXT, None
            )
        )
        standard = asyncio.run(
            service._reasoning_template_kwargs(
                snapshot,
                TurnIntent.NORMAL_TEXT,
                ReasoningRequest(mode="standard"),
            )
        )
        pro = asyncio.run(
            service._reasoning_template_kwargs(
                snapshot,
                TurnIntent.NORMAL_TEXT,
                ReasoningRequest(mode="pro_extended"),
            )
        )
        tool_result = asyncio.run(
            service._reasoning_template_kwargs(
                snapshot, TurnIntent.TOOL_RESULT_FINALIZATION, None
            )
        )
        structured = asyncio.run(
            service._reasoning_template_kwargs(
                snapshot, TurnIntent.STRUCTURED_FINALIZATION, None
            )
        )
        self.assertEqual(ordinary, {"enable_thinking": False})
        self.assertEqual(standard, {"enable_thinking": False})
        self.assertEqual(pro, {"enable_thinking": True})
        self.assertEqual(tool_result, {"enable_thinking": False})
        self.assertEqual(structured, {"enable_thinking": False})
        self.assertEqual(
            private_chat_template_kwargs(TurnIntent.NORMAL_TEXT, None),
            None,
        )

    def test_unsupported_effort_and_budget_stay_truthful(self):
        service, _catalogue, _router, _operations = _service()
        for reasoning in (
            ReasoningRequest(mode="custom", effort="low"),
            ReasoningRequest(mode="custom", budget_tokens=8),
        ):
            with self.assertRaises(SystemXError):
                asyncio.run(
                    service._reasoning_template_kwargs(
                        SimpleNamespace(),
                        TurnIntent.NORMAL_TEXT,
                        reasoning,
                    )
                )

    def test_reserve_is_bounded_and_streaming_rejected_before_resolution(self):
        service, catalogue, _router, _operations = _service()
        request = ChatRequest(
            model="default",
            messages=[ChatMessage(role="user", content="hello")],
            max_output_tokens=16,
            stream=True,
            reasoning=ReasoningRequest(
                mode="pro_extended",
                final_answer_reserve_tokens=4,
            ),
        )
        with self.assertRaises(SystemXError):
            asyncio.run(service.prepare_chat("sx_req_" + "a" * 32, request))
        self.assertEqual(catalogue.resolve_calls, 0)
        with self.assertRaises(SystemXError):
            InferenceService._validate_reserve(
                intent=TurnIntent.NORMAL_TEXT,
                reasoning=ReasoningRequest(
                    mode="standard",
                    final_answer_reserve_tokens=4,
                ),
                max_output_tokens=16,
                stream=False,
            )
        with self.assertRaises(ValidationError):
            ReasoningRequest(
                mode="pro_extended",
                final_answer_reserve_tokens=1_048_577,
            )

    def test_chat_reserve_fallback_has_two_calls_one_public_identity(self):
        service, _catalogue, router, operations = _service()
        request_id = "sx_req_" + "b" * 32
        request = ChatRequest(
            model="default",
            messages=[ChatMessage(role="user", content="answer")],
            max_output_tokens=16,
            reasoning=ReasoningRequest(
                mode="pro_extended",
                final_answer_reserve_tokens=4,
            ),
        )
        result = asyncio.run(service.chat(request_id, request))
        self.assertEqual(result.output.content, "final answer")
        self.assertEqual(result.output.reasoning, ["bounded reasoning"])
        self.assertEqual(
            router.chat_calls,
            [
                (12, {"enable_thinking": True}),
                (4, {"enable_thinking": False}),
            ],
        )
        self.assertEqual(
            {item[0] for item in operations.models
             + operations.routers + operations.terminals},
            {request_id},
        )

    def test_responses_reserve_fallback_has_two_calls_and_separate_reasoning(self):
        service, _catalogue, router, operations = _service()
        request_id = "sx_req_" + "c" * 32
        request = ResponsesRequest(
            model="default",
            input="answer",
            max_output_tokens=16,
            reasoning=ReasoningRequest(
                mode="pro_extended",
                final_answer_reserve_tokens=4,
            ),
        )
        result = asyncio.run(service.responses(request_id, request))
        self.assertEqual(result.output.content, "final response")
        self.assertEqual(result.output.reasoning, ["bounded reasoning"])
        self.assertEqual(
            router.responses_calls,
            [
                (12, {"enable_thinking": True}),
                (4, {"enable_thinking": False}),
            ],
        )
        self.assertEqual(
            {item[0] for item in operations.models
             + operations.routers + operations.terminals},
            {request_id},
        )

    def test_compatibility_adapters_converge_on_ordinary_canonical_requests(self):
        openai = OpenAICompatibilityAdapter(None, None)
        openai_chat = openai.canonical_chat_request(
            OpenAIChatCompletionRequest.model_validate({
                "model": "default",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 32,
            })
        )
        openai_responses = openai.canonical_responses_request(
            OpenAIResponsesRequest.model_validate({
                "model": "default",
                "input": "hello",
                "max_output_tokens": 32,
            })
        )
        anthropic = AnthropicCompatibilityAdapter(None, None)
        messages = anthropic.canonical_message_request(
            AnthropicMessageRequest.model_validate({
                "model": "default",
                "messages": [{
                    "role": "user",
                    "content": "hello",
                }],
                "max_tokens": 32,
            })
        )
        for canonical in (openai_chat, openai_responses, messages):
            self.assertIsNone(canonical.reasoning)

    def test_ordinary_stream_preparation_disables_reasoning(self):
        service, _catalogue, _router, _operations = _service()
        request = ChatRequest(
            model="default",
            messages=[ChatMessage(role="user", content="stream")],
            max_output_tokens=16,
            stream=True,
        )
        prepared = asyncio.run(
            service.prepare_chat("sx_req_" + "d" * 32, request)
        )
        self.assertEqual(
            prepared.private_template_kwargs,
            {"enable_thinking": False},
        )

    def test_empty_completed_chat_output_is_never_accepted(self):
        observation = RouterObservation(
            200,
            "",
            {
                "choices": [{
                    "message": {"role": "assistant", "content": None},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
            None,
        )
        with self.assertRaises(ResponseNormalizationError) as caught:
            normalize_chat_turn(observation, [], None, None)
        self.assertEqual(
            caught.exception.kind, "empty_final_chat_output"
        )


if __name__ == "__main__":
    unittest.main()
